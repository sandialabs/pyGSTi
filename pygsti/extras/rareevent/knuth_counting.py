"""Knuth random-descent tree-size estimation of malignant cluster counts.

Implements Knuth's 1975 unbiased random-descent estimator for the size of a
search tree, applied here to count ``m(v)``: the number of *minimal*
malignant connected mechanism sets of weight ``v`` (a "minimal malignant
cluster" is a connected set of mechanisms -- connectivity in the
shared-detector graph -- that causes a decoding failure, but no proper
nonempty subset of it does). Exhaustive enumeration of all connected sets of
weight ``v`` becomes intractable at high code distance; Knuth's estimator
gives an unbiased, tractable alternative that never needs to construct or
store the whole tree.

The enumeration tree
---------------------
Connectivity is defined by ``splitting_swap.build_detector_adjacency``: two
mechanisms are adjacent iff they share at least one detector target. Over
this graph we walk a duplicate-free forest of connected sets (the classic
ESU / "reverse search" construction, also the tree Knuth's 1975 estimator was
originally designed to size):

- A virtual super-root has one child per mechanism ``r`` -- the singleton
  set ``{r}``.
- The subtree rooted at ``{r}`` enumerates, without duplication, every
  connected set whose *minimum* element is ``r``. A node is a triple
  ``(S, ext, visited)``: ``S`` is the set of mechanisms chosen so far,
  ``ext`` is the ordered list of "extension candidates" available to grow
  ``S``, and ``visited`` is the set of mechanisms already ruled out as
  duplicate-avoiding future extensions along this branch.

  - Root node for ``r``: ``S = {r}``, ``ext = [neighbors of r with index >
    r]``, ``visited = {r} | set(ext)``.
  - Children of ``(S, ext, visited)``: for each ``idx, v`` in
    ``enumerate(ext)``, the child extends ``S`` by ``v`` and appends any
    *new* neighbors of ``v`` (index ``> r``, not already in ``visited``) to
    the remaining tail of ``ext``:

        new_neighbors = [u for u in neighbors[v] if u > r and u not in visited]
        child = (S | {v}, ext[idx+1:] + new_neighbors, visited | set(new_neighbors))

Every connected set of mechanisms appears at exactly one node of this
forest (at the node whose ``S`` equals that set), so counting/summing over
nodes at a fixed depth is well defined and duplicate-free.

Knuth's estimator
------------------
To estimate ``m(v) = #{S : |S| = v, phi(S)}`` where (by default) ``phi(S) =
oracle.fails(S) and S is minimal`` (minimality requires testing *all*
``2**v - 2`` proper nonempty subsets, since failure is not weight-monotone --
checking only the ``v - 1``-subsets is not sound), one probe works as
follows: starting from a root ``r``, repeat ``v - 1`` times: if the current
node has zero children the probe returns 0; otherwise multiply a running
multiplier ``X`` by the branching factor (number of children) and descend to
a uniformly random child. At depth ``v`` (``|S| = v``), return ``X *
phi(S)``. Knuth (1975) shows ``E[probe] = (size of the subtree's depth-v
level)`` exactly, for any branching random walk of this shape. A probe costs
at most ``2**v - 1`` oracle calls (dominated by the minimality check, itself
only paid when the sampled leaf fails at all) but *usually just 1*, since
most weight-``v`` connected sets do not fail.

This module additionally applies **root stratification**: rather than
picking the initial root ``{r}`` uniformly (folding a factor of ``n`` into
``X``), it loops deterministically over every one of the ``n`` roots and
runs ``probes_per_root`` independent probes of that root's subtree alone
(*without* the initial factor of ``n``). Since the roots partition the depth
``v`` level of the forest, ``m(v) = sum_r (subtree count for root r)``, and
each root's mean probe value is an unbiased estimator of its own subtree
count; summing removes the large cross-root variance component that a
uniform first step would otherwise inject. The reported standard error uses
the delta method over independent per-root probe batches (the roots
themselves are deterministic strata, not sampled):
``SE**2 = sum_r Var_k(X_{r,k}) / probes_per_root``.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import random
import statistics
from collections.abc import Sequence
from typing import Any

import numpy as np

from .interfaces import ErrorModel, Estimator, ForwardSimulator
from .rare_event import MechanismCatalog
from .splitting_swap import build_detector_adjacency
from .weight_points import WeightPoint

PHI_MODES = ("minimal", "malignant", "all")


# ---------------------------------------------------------------------------
# The duplicate-free connected-set forest.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Node:
    """One node of the connected-set forest: (root, S, ext, visited)."""

    root: int
    members: frozenset[int]
    ext: tuple[int, ...]
    visited: frozenset[int]


def _root_node(r: int, neighbors: Sequence[tuple[int, ...]]) -> _Node:
    ext = tuple(u for u in neighbors[r] if u > r)
    return _Node(root=r, members=frozenset({r}), ext=ext, visited=frozenset({r}) | frozenset(ext))


def _child_at(node: _Node, idx: int, neighbors: Sequence[tuple[int, ...]]) -> _Node:
    v = node.ext[idx]
    new_neighbors = tuple(u for u in neighbors[v] if u > node.root and u not in node.visited)
    return _Node(
        root=node.root,
        members=node.members | {v},
        ext=node.ext[idx + 1 :] + new_neighbors,
        visited=node.visited | frozenset(new_neighbors),
    )


def _children(node: _Node, neighbors: Sequence[tuple[int, ...]]) -> list[_Node]:
    return [_child_at(node, idx, neighbors) for idx in range(len(node.ext))]


def enumerate_connected_sets(
    neighbors: Sequence[tuple[int, ...]],
    weight: int,
    *,
    max_nodes: int = 2_000_000,
) -> tuple[list[frozenset[int]], int, bool]:
    """Depth-first, budget-guarded walk collecting every connected set of size ``weight``.

    Returns ``(sets, nodes_visited, exhausted)``. ``exhausted`` is False if
    the ``max_nodes`` budget was exhausted before the walk completed (in
    which case ``sets`` is a partial, incomplete list). The walk never
    descends past depth ``weight`` (there is nothing useful to find there),
    so this only pays for the part of the forest that matters.
    """
    if weight < 1:
        raise ValueError(f"weight must be >= 1; got {weight}.")
    n = len(neighbors)
    stack: list[_Node] = [_root_node(r, neighbors) for r in range(n)]
    found: list[frozenset[int]] = []
    nodes_visited = 0
    exhausted = True
    while stack:
        if nodes_visited >= max_nodes:
            exhausted = False
            break
        node = stack.pop()
        nodes_visited += 1
        depth = len(node.members)
        if depth == weight:
            found.append(node.members)
            continue
        if depth < weight:
            stack.extend(_children(node, neighbors))
    return found, nodes_visited, exhausted


# ---------------------------------------------------------------------------
# phi(S): the malignancy / minimality predicate.
# ---------------------------------------------------------------------------


def _evaluate_phi(mode: str, oracle: ForwardSimulator, members: frozenset[int]) -> tuple[bool, int]:
    """Evaluate phi(S) for the given mode. Returns (phi(S), oracle calls used).

    - "all": phi is identically True (used only to validate the forest
      construction itself against brute-force enumeration).
    - "malignant": phi(S) = oracle.fails(S) (no minimality check).
    - "minimal": phi(S) = oracle.fails(S) and no proper nonempty subset of S
      fails. Short-circuits to a single oracle call whenever S itself does
      not fail (the common case); otherwise pays up to ``2**|S| - 2``
      additional calls to check every proper nonempty subset (failure is not
      weight-monotone, so checking only the co-dimension-1 subsets would be
      unsound).
    """
    if mode not in PHI_MODES:
        raise ValueError(f"mode must be one of {PHI_MODES}; got {mode!r}.")
    if mode == "all":
        return True, 0
    if not oracle.fails(set(members)):
        return False, 1
    if mode == "malignant":
        return True, 1
    v = len(members)
    calls = 1
    if v <= 1:
        return True, calls
    items = tuple(sorted(members))
    for k in range(1, v):
        for combo in itertools.combinations(items, k):
            calls += 1
            if oracle.fails(set(combo)):
                return False, calls
    return True, calls


# ---------------------------------------------------------------------------
# Exhaustive fallback (ground truth / cross-check at small weight and n).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ExhaustiveCountResult:
    """Result of a plain (non-random) walk of the connected-set forest."""

    count: int
    nodes_visited: int
    oracle_calls: int
    exhausted: bool


def exhaustive_count_m_v(
    catalog: MechanismCatalog,
    oracle: ForwardSimulator,
    *,
    weight: int,
    max_nodes: int = 2_000_000,
    mode: str = "minimal",
) -> ExhaustiveCountResult:
    """Exact count of ``m(weight)`` by a plain, budget-guarded walk of the forest.

    At small distances/weights this is fast enough to serve as ground truth
    for the Knuth estimator; at larger instances it is a useful sanity tool
    that reports ``exhausted=False`` (and a partial ``count``) rather than
    running forever.
    """
    neighbors = build_detector_adjacency(catalog)
    sets_found, nodes_visited, exhausted = enumerate_connected_sets(neighbors, weight, max_nodes=max_nodes)
    count = 0
    oracle_calls = 0
    for members in sets_found:
        ok, calls = _evaluate_phi(mode, oracle, members)
        oracle_calls += calls
        if ok:
            count += 1
    return ExhaustiveCountResult(count=count, nodes_visited=nodes_visited, oracle_calls=oracle_calls, exhausted=exhausted)


# ---------------------------------------------------------------------------
# Knuth's random-descent estimator.
# ---------------------------------------------------------------------------


def _one_probe(
    root: _Node,
    weight: int,
    neighbors: Sequence[tuple[int, ...]],
    rng: random.Random,
    mode: str,
    oracle: ForwardSimulator,
) -> tuple[float, int]:
    """One Knuth probe of ``root``'s subtree (the initial factor of n is NOT applied here;
    root stratification supplies it via the deterministic loop over roots instead).

    Returns ``(X * phi(S), oracle calls used)``.
    """
    node = root
    x = 1.0
    for _ in range(weight - 1):
        num_children = len(node.ext)
        if num_children == 0:
            return 0.0, 0
        x *= num_children
        idx = rng.randrange(num_children)
        node = _child_at(node, idx, neighbors)
    ok, calls = _evaluate_phi(mode, oracle, node.members)
    return (x if ok else 0.0), calls


def _single_root_probes(
    r: int,
    weight: int,
    neighbors: Sequence[tuple[int, ...]],
    rng: random.Random,
    mode: str,
    oracle: ForwardSimulator,
    probes_per_root: int,
) -> tuple[list[float], int]:
    root = _root_node(r, neighbors)
    values: list[float] = []
    oracle_calls = 0
    for _ in range(probes_per_root):
        x, calls = _one_probe(root, weight, neighbors, rng, mode, oracle)
        values.append(x)
        oracle_calls += calls
    return values, oracle_calls


def knuth_estimate_m_v(
    catalog: MechanismCatalog,
    oracle: ForwardSimulator,
    *,
    weight: int,
    probes_per_root: int = 200,
    total_probe_budget: int | None = None,
    seed: int = 1,
    minimal: bool = True,
    verbose: bool = False,
) -> WeightPoint:
    """Estimate ``m(weight)`` via root-stratified Knuth random descent.

    Args:
        catalog: The mechanism catalog defining the connectivity graph
            (shared-detector adjacency) and the ``n`` roots.
        oracle: The forward simulator/decoder; ``phi`` is evaluated by
            calling ``oracle.fails``.
        weight: The cluster weight ``v`` to estimate ``m(v)`` for.
        probes_per_root: Number of independent probes run from each of the
            ``n`` roots. Ignored if ``total_probe_budget`` is given.
        total_probe_budget: If given, spreads this total probe budget evenly
            (ceil division) over the ``n`` roots instead of using
            ``probes_per_root`` directly -- convenient when ``n *
            probes_per_root`` would be prohibitively large.
        seed: Seed for the probe RNG (a single ``random.Random`` stream is
            shared, deterministically, across all roots).
        minimal: If True (default), ``phi`` requires minimality (no proper
            nonempty subset fails); if False, ``phi`` is plain malignancy
            (``oracle.fails(S)``), which is cheaper to cross-check.
        verbose: Print periodic progress across roots.

    Returns:
        A ``WeightPoint`` with ``method="knuth_counting"``, ``kind="m_v"``,
        ``exact=False``. ``meta`` includes probe/oracle-call counts, the
        zero-probe fraction, and the top-5 highest-variance roots.
    """
    if weight < 1:
        raise ValueError(f"weight must be >= 1; got {weight}.")
    neighbors = build_detector_adjacency(catalog)
    n = len(neighbors)
    if n == 0:
        raise ValueError("catalog has no mechanisms.")

    if total_probe_budget is not None:
        if total_probe_budget < 1:
            raise ValueError(f"total_probe_budget must be >= 1; got {total_probe_budget}.")
        probes_per_root = -(-total_probe_budget // n)  # ceil division
    if probes_per_root < 1:
        raise ValueError(f"probes_per_root must be >= 1; got {probes_per_root}.")

    mode = "minimal" if minimal else "malignant"
    rng = random.Random(seed)

    root_means = np.zeros(n, dtype=np.float64)
    root_vars = np.zeros(n, dtype=np.float64)
    total_oracle_calls = 0
    total_probes = 0
    zero_probes = 0
    progress_stride = max(1, n // 10)

    for r in range(n):
        values, calls = _single_root_probes(r, weight, neighbors, rng, mode, oracle, probes_per_root)
        total_oracle_calls += calls
        total_probes += len(values)
        zero_probes += sum(1 for v in values if v == 0.0)
        root_means[r] = statistics.fmean(values)
        root_vars[r] = statistics.variance(values) if len(values) > 1 else 0.0
        if verbose and r % progress_stride == 0:
            running = float(np.sum(root_means[: r + 1]))
            print(f"[knuth v={weight}] root {r}/{n}: running m_hat={running:.6g}")

    estimate = float(np.sum(root_means))
    se2 = float(np.sum(root_vars / probes_per_root))
    se = math.sqrt(max(se2, 0.0))
    rel_err = se / estimate if estimate > 0 else 0.0

    top5_variance_roots = sorted(
        ((int(r), float(root_vars[r])) for r in range(n)), key=lambda t: t[1], reverse=True
    )[:5]

    meta: dict[str, Any] = {
        "probes_per_root": probes_per_root,
        "num_roots": n,
        "total_probes": total_probes,
        "zero_probe_fraction": zero_probes / total_probes if total_probes else 0.0,
        "oracle_calls": total_oracle_calls,
        "standard_error": se,
        "mode": mode,
        "top5_variance_roots": top5_variance_roots,
        "seed": seed,
    }

    return WeightPoint(
        method="knuth_counting",
        kind="m_v",
        weight=weight,
        estimate=estimate,
        rel_err=rel_err,
        exact=False,
        lower_bound=False,
        meta=meta,
    )


def knuth_estimate_many(
    catalog: MechanismCatalog,
    oracle: ForwardSimulator,
    *,
    weights: Sequence[int],
    probes_per_root: int = 200,
    total_probe_budget: int | None = None,
    seed: int = 1,
    minimal: bool = True,
    verbose: bool = False,
) -> list[WeightPoint]:
    """Run ``knuth_estimate_m_v`` for each weight in ``weights``.

    Each weight gets its own probe stream, seeded ``seed + i`` (``i`` its
    index in ``weights``) so the per-weight estimates are independent and
    reproducible.
    """
    return [
        knuth_estimate_m_v(
            catalog,
            oracle,
            weight=w,
            probes_per_root=probes_per_root,
            total_probe_budget=total_probe_budget,
            seed=seed + i,
            minimal=minimal,
            verbose=verbose,
        )
        for i, w in enumerate(weights)
    ]


class KnuthCountingEstimator(Estimator):
    """``Estimator``-protocol wrapper around ``knuth_estimate_many``.

    Ignores ``error_model`` (m(v) counting is purely combinatorial over the
    catalog's connectivity graph and the oracle's failure predicate; it does
    not depend on the physical rate p). Accepted kwargs: ``catalog``
    (required), ``weight`` or ``weights`` (one required), and the keyword
    arguments of ``knuth_estimate_m_v``.
    """

    def estimate(
        self,
        error_model: ErrorModel | None,
        simulator: ForwardSimulator,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del error_model
        if "catalog" not in kwargs:
            raise ValueError("catalog must be provided to KnuthCountingEstimator")
        catalog = kwargs["catalog"]

        weights = kwargs.get("weights")
        if weights is None:
            if "weight" not in kwargs:
                raise ValueError("Provide 'weight' or 'weights' to KnuthCountingEstimator")
            weights = [kwargs["weight"]]

        points = knuth_estimate_many(
            catalog,
            simulator,
            weights=weights,
            probes_per_root=kwargs.get("probes_per_root", 200),
            total_probe_budget=kwargs.get("total_probe_budget"),
            seed=kwargs.get("seed", 1),
            minimal=kwargs.get("minimal", True),
            verbose=kwargs.get("verbose", False),
        )
        return {"weight_points": points}
