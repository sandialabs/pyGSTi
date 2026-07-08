import multiprocessing as _mp
import os
import time as _time

import numpy as np
import pytest

from pygsti.tools.graphcoloring import (
    switchboard_find_edge_coloring,
    check_valid_edge_coloring,
    find_edge_coloring,
)

from ..util import BaseCase


# GitHub Actions (like most CI providers) sets the ``CI`` environment variable
# automatically. The benchmark-style scaling suite in this module is meant to be
# run locally, so we skip it in CI via this environment-variable ``skipif`` -- the
# same pattern used elsewhere in the test suite (e.g. the ``PYGSTI_NO_CUSTOMLM_SIGINT``
# check in test/unit/optimize/test_sigint.py and the ``needs_*`` helpers in
# test/unit/util.py). This keeps the slow tests local-only without any change to
# the CI workflow. To run them locally, simply invoke pytest outside of CI (or
# unset ``CI``); to also exclude them locally, use ``-m "not slow"``.
skip_in_ci = pytest.mark.skipif(
    'CI' in os.environ,
    reason="benchmark-style scaling test; run locally only (skipped in CI)")


# Curated, generally-recommended algorithms.
ALGORITHMS = ["new_bipartite", "vizing", "sinnamon", "misra_gries"]

# Every edge-coloring algorithm exposed by the switchboard. This is deliberately
# the *full* list (not the curated ALGORITHMS above) because the scaling suite's
# whole point is to characterize which algorithms are usable in which regime --
# including the ones that are known to be slow, suboptimal, or outright broken on
# certain graph families.
ALL_ALGORITHMS = ["new_bipartite", "vizing", "sinnamon", "misra_gries", "moser_tardos", "assadi"]

# Deterministic algorithms that always terminate and produce a proper, complete
# coloring with at most deg+1 colors (Vizing's theorem) on every family tested.
DETERMINISTIC_EXACT_ALGORITHMS = ["vizing", "misra_gries"]

# Randomized algorithms that were fixed to terminate and produce valid colorings.
RANDOMIZED_ALGORITHMS = ["assadi", "moser_tardos", "sinnamon"]

# "SPARSE_SAFE" = algorithms that reliably produced a proper & complete coloring
# on the low-degree (deg<=4) families in this suite, across runs and within the
# per-algorithm timeout. `vizing` and `misra_gries` qualify (both deterministic,
# fast, and near-optimal here; both are in fact reliable on *dense* graphs too).
# moser_tardos and assadi are now correct and terminate, but at the suite's
# larger low-degree sizes (e.g. the 8x8 grid) they can exceed the timeout, so
# they are recorded but not asserted on.
SPARSE_SAFE = ["vizing", "misra_gries"]

# Small enough that even the good algorithms finish comfortably, large enough to
# expose the blow-ups. Kept modest so the whole suite runs in a few seconds.
PER_ALGO_TIMEOUT = 5.0


# ---------------------------------------------------------------------------
# Graph generators
# ---------------------------------------------------------------------------
def _finalize(vertices, edges):
    """Build the (vertices, edges, neighbors, deg) 4-tuple the API expects.

    ``edges`` is a list of undirected ``(u, v)`` pairs (each listed once). The
    neighbor lists are symmetric, and ``deg`` is the true maximum degree.
    """
    neighbors = {v: [] for v in vertices}
    for u, v in edges:
        neighbors[u].append(v)
        neighbors[v].append(u)
    deg = max((len(neighbors[v]) for v in vertices), default=0)
    return list(vertices), list(edges), neighbors, deg


def make_cycle_graph(n):
    """Cycle C_n: n vertices, n edges, max degree 2."""
    return _finalize(range(n), [(i, (i + 1) % n) for i in range(n)])


def make_path_graph(n):
    """Path P_n: n vertices, n-1 edges, max degree 2."""
    return _finalize(range(n), [(i, i + 1) for i in range(n - 1)])


def make_complete_graph(n):
    """Complete graph K_n: max degree n-1, the hardest dense case."""
    return _finalize(range(n), [(i, j) for i in range(n) for j in range(i + 1, n)])


def make_grid_graph(rows, cols):
    """2D lattice (rows x cols), max degree 4 -- models a planar QPU layout."""
    def idx(r, c):
        return r * cols + c
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                edges.append((idx(r, c), idx(r + 1, c)))
    return _finalize(range(rows * cols), edges)


def make_high_degree_graph():
    """A small graph with a single high-degree hub (max degree 5)."""
    edges = [(0, i) for i in range(1, 6)] + [(1, 6), (2, 7), (3, 8), (4, 9)]
    return _finalize(range(10), edges)


def make_random_regular_graph(n, d, seed):
    """(Approximately) d-regular random graph on n vertices via a pairing model.

    Falls back to whatever simple graph the pairing produces; the returned
    ``deg`` is the realized maximum degree, which is <= d.
    """
    rng = np.random.default_rng(seed)
    stubs = []
    for v in range(n):
        stubs.extend([v] * d)
    edge_set = set()
    for _ in range(10):
        rng.shuffle(stubs)
        edge_set.clear()
        ok = True
        for i in range(0, len(stubs) - 1, 2):
            u, w = stubs[i], stubs[i + 1]
            if u == w or (min(u, w), max(u, w)) in edge_set:
                ok = False
                break
            edge_set.add((min(u, w), max(u, w)))
        if ok:
            break
    return _finalize(range(n), sorted(edge_set))


# ---------------------------------------------------------------------------
# Coloring-quality helpers (stronger than check_valid_edge_coloring, which only
# verifies each color class is a matching -- it does NOT verify completeness).
# ---------------------------------------------------------------------------
def assess_coloring(color_patches, edges):
    """Return (is_proper, is_complete, num_colors) for a coloring.

    * proper   : no edge appears twice AND no two edges sharing a vertex share a color
    * complete : every input edge received exactly one color
    """
    all_edges = {tuple(sorted(e)) for e in edges}
    seen = set()
    is_proper = True
    for _color, patch in color_patches.items():
        touched = set()
        for u, v in patch:
            e = tuple(sorted((u, v)))
            if e in seen:
                is_proper = False  # edge colored more than once
            seen.add(e)
            if u in touched or v in touched:
                is_proper = False  # adjacent edges share this color
            touched.add(u)
            touched.add(v)
    is_complete = (seen == all_edges)
    return is_proper, is_complete, len(color_patches)


def _canonical_coloring(color_patches):
    """A hashable, order-independent representation of a coloring for equality.

    Two colorings compare equal iff they assign the same set of edges to the
    same set of colors (edge order within a color and dict order are ignored).
    """
    return tuple(sorted(
        (color, tuple(sorted(tuple(sorted(e)) for e in patch)))
        for color, patch in color_patches.items() if patch
    ))


# ---------------------------------------------------------------------------
# Timeout-guarded runner. Some randomized algorithms can, on dense graphs, run
# for a very long time, so we run each attempt in a separate process and
# hard-kill it if it exceeds the budget. This keeps the scaling suite from ever
# hanging CI.
# ---------------------------------------------------------------------------
def _worker(algorithm_name, deg, vertices, edges, neighbors, seed, q):
    try:
        cp = switchboard_find_edge_coloring(algorithm_name, deg, vertices, edges, neighbors, seed=seed)
        q.put(("ok", cp))
    except Exception as ex:  # noqa: BLE001 -- we intentionally capture everything
        q.put(("error", "%s: %s" % (type(ex).__name__, ex)))


def run_with_timeout(algorithm_name, graph, seed=0, timeout=PER_ALGO_TIMEOUT):
    """Run one algorithm on one graph under a wall-clock budget.

    Returns a result dict with keys:
        status   : 'ok' | 'error' | 'timeout'
        seconds  : elapsed wall time (== timeout if it timed out)
        colors   : number of colors used (None unless status=='ok' and proper+complete)
        proper   : bool or None
        complete : bool or None
        detail   : error string or None
    """
    vertices, edges, neighbors, deg = graph
    ctx = _mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(algorithm_name, deg, vertices, edges, neighbors, seed, q))
    start = _time.perf_counter()
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return dict(status="timeout", seconds=timeout, colors=None,
                    proper=None, complete=None, detail="exceeded %.1fs" % timeout)
    elapsed = _time.perf_counter() - start
    try:
        kind, payload = q.get_nowait()
    except Exception:  # queue empty -> process died without reporting
        return dict(status="error", seconds=elapsed, colors=None,
                    proper=None, complete=None, detail="worker produced no result")
    if kind == "error":
        return dict(status="error", seconds=elapsed, colors=None,
                    proper=None, complete=None, detail=payload)
    proper, complete, ncolors = assess_coloring(payload, edges)
    return dict(status="ok", seconds=elapsed,
                colors=ncolors if (proper and complete) else None,
                proper=proper, complete=complete, detail=None)


def _print_table(title, graph, results):
    """Print a benchmark comparison table (visible with `pytest -s`)."""
    vertices, edges, neighbors, deg = graph
    print("\n" + "=" * 78)
    print(f"{title}  (|V|={len(vertices)}, |E|={len(edges)}, max_degree={deg})")
    print("-" * 78)
    print(f"{'algorithm':16s} {'status':9s} {'colors':>7s} {'time(ms)':>10s}  notes")
    for algo in ALL_ALGORITHMS:
        r = results[algo]
        colors = "-" if r["colors"] is None else str(r["colors"])
        note = ""
        if r["status"] == "ok" and not (r["proper"] and r["complete"]):
            note = f"INVALID (proper={r['proper']}, complete={r['complete']})"
        elif r["detail"]:
            note = r["detail"]
        print(f"{algo:16s} {r['status']:9s} {colors:>7s} {r['seconds']*1000:>10.1f}  {note}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Parametrization fixtures
# ---------------------------------------------------------------------------

# Sparse/mixed graphs the networkx-based ``find_edge_coloring`` handles directly.
FIND_EDGE_COLORING_GRAPHS = [
    ("cycle_C10", make_cycle_graph(10)),
    ("path_P10", make_path_graph(10)),
    ("high_degree_n10", make_high_degree_graph()),
]

# Small, sparse graphs on which the fixed randomized algorithms terminate quickly
# (so their reproducibility tests can run in-process, no timeout guard needed).
REPRODUCIBILITY_GRAPHS = [
    ("cycle_C6", make_cycle_graph(6)),
    ("path_P8", make_path_graph(8)),
    ("grid_3x3", make_grid_graph(3, 3)),
    ("random_3reg_n12", make_random_regular_graph(12, 3, seed=1234)),
]

# Dense complete graphs used as regression guards for the deterministic
# (Vizing-chain) algorithms, which previously hung or left edges uncolored here.
DENSE_REGRESSION_GRAPHS = [
    ("K6", make_complete_graph(6)),
    ("K9", make_complete_graph(9)),
    ("K12", make_complete_graph(12)),
]


# ---------------------------------------------------------------------------
# Module-level parametrized tests (data-driven).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,graph", FIND_EDGE_COLORING_GRAPHS)
def test_find_edge_coloring_is_proper_and_complete(name, graph):
    """find_edge_coloring colors every edge exactly once with no adjacent clash."""
    vertices, edges, neighbors, deg = graph
    color_patches = find_edge_coloring(deg, vertices, edges, neighbors)
    proper, complete, _ncolors = assess_coloring(color_patches, edges)
    assert proper, f"find_edge_coloring produced an improper coloring on {name}"
    assert complete, f"find_edge_coloring left an edge uncolored on {name}"
    # check_valid_edge_coloring raises (or returns False) on an improper coloring.
    assert check_valid_edge_coloring(color_patches, ret_false_on_error=True), \
        f"find_edge_coloring failed check_valid_edge_coloring on {name}"


@pytest.mark.parametrize("algorithm", RANDOMIZED_ALGORITHMS)
def test_same_seed_is_reproducible(algorithm):
    """Same integer seed => byte-for-byte identical coloring."""
    for name, graph in REPRODUCIBILITY_GRAPHS:
        vertices, edges, neighbors, deg = graph
        r1 = switchboard_find_edge_coloring(algorithm, deg, vertices, edges, neighbors, seed=42)
        r2 = switchboard_find_edge_coloring(algorithm, deg, vertices, edges, neighbors, seed=42)
        assert _canonical_coloring(r1) == _canonical_coloring(r2), \
            f"{algorithm} not reproducible with a fixed seed on {name}"


@pytest.mark.parametrize("algorithm", RANDOMIZED_ALGORITHMS)
def test_seeded_output_is_proper_and_complete(algorithm):
    """A seeded run must still be a proper, complete edge coloring."""
    for name, graph in REPRODUCIBILITY_GRAPHS:
        vertices, edges, neighbors, deg = graph
        cp = switchboard_find_edge_coloring(algorithm, deg, vertices, edges, neighbors, seed=42)
        proper, complete, _ncolors = assess_coloring(cp, edges)
        assert proper, f"{algorithm} produced an improper coloring on {name}"
        assert complete, f"{algorithm} left an edge uncolored on {name}"
        assert check_valid_edge_coloring(cp, ret_false_on_error=True), \
            f"{algorithm} failed check_valid_edge_coloring on {name}"


@pytest.mark.parametrize("algorithm", RANDOMIZED_ALGORITHMS)
def test_accepts_generator_object(algorithm):
    """A numpy.random.Generator may be passed directly as the seed, and two
    generators created from the same seed reproduce each other."""
    name, graph = REPRODUCIBILITY_GRAPHS[0]
    vertices, edges, neighbors, deg = graph
    r1 = switchboard_find_edge_coloring(
        algorithm, deg, vertices, edges, neighbors, seed=np.random.default_rng(7))
    r2 = switchboard_find_edge_coloring(
        algorithm, deg, vertices, edges, neighbors, seed=np.random.default_rng(7))
    assert _canonical_coloring(r1) == _canonical_coloring(r2), \
        f"{algorithm} not reproducible from equivalent Generator objects on {name}"


@pytest.mark.parametrize("algorithm", RANDOMIZED_ALGORITHMS)
def test_no_seed_still_valid(algorithm):
    """Omitting the seed still yields a proper, complete coloring."""
    name, graph = REPRODUCIBILITY_GRAPHS[0]
    vertices, edges, neighbors, deg = graph
    cp = switchboard_find_edge_coloring(algorithm, deg, vertices, edges, neighbors)
    proper, complete, _ncolors = assess_coloring(cp, edges)
    assert proper and complete, \
        f"{algorithm} produced an invalid coloring without a seed on {name}"


@skip_in_ci
@pytest.mark.slow
@pytest.mark.parametrize("name,graph", DENSE_REGRESSION_GRAPHS)
@pytest.mark.parametrize("algorithm", DETERMINISTIC_EXACT_ALGORITHMS)
def test_deterministic_algorithm_colors_dense_graph(algorithm, name, graph):
    """Regression guard: the deterministic (Vizing-chain) algorithms must
    terminate and produce a proper, complete, (deg+1)-color coloring on dense
    complete graphs.

    misra_gries used to raise a KeyError or hang on non-sparse graphs, and
    vizing's complex ("Vizing chain") case was an unimplemented placeholder that
    silently left edges uncolored. Both were fixed (vizing's chain now delegates
    to the canonical Misra-Gries procedure), so both are deterministic, always
    terminate, and are near-optimal even on complete graphs.
    """
    _vertices, _edges, _neighbors, deg = graph
    r = run_with_timeout(algorithm, graph, seed=0, timeout=PER_ALGO_TIMEOUT)
    assert r["status"] == "ok", \
        f"{algorithm} did not finish on {name}: {r['status']} ({r['detail']})"
    assert r["proper"] and r["complete"], \
        (f"{algorithm} produced an invalid coloring on {name} "
         f"(proper={r['proper']}, complete={r['complete']})")
    assert r["colors"] <= deg + 1, \
        f"{algorithm} used {r['colors']} colors on {name} (budget {deg + 1})"


# ---------------------------------------------------------------------------
# Class-based (example) tests.
# ---------------------------------------------------------------------------
class GraphColoringTester(BaseCase):
    """Correctness tests for the deterministic (deg+1) edge-coloring algorithms.

    These run in-process (no timeout guard) on graphs small enough that the
    algorithms finish comfortably, so they belong to the fast unit suite.
    """

    SMALL_GRAPHS = [
        ("cycle_C10", make_cycle_graph(10)),
        ("path_P10", make_path_graph(10)),
        ("grid_4x4", make_grid_graph(4, 4)),
        ("complete_K6", make_complete_graph(6)),
    ]

    def test_deterministic_algorithms_are_proper_and_complete(self):
        for algorithm in DETERMINISTIC_EXACT_ALGORITHMS:
            for name, graph in self.SMALL_GRAPHS:
                vertices, edges, neighbors, deg = graph
                cp = switchboard_find_edge_coloring(algorithm, deg, vertices, edges, neighbors)
                proper, complete, ncolors = assess_coloring(cp, edges)
                self.assertTrue(proper, f"{algorithm} produced an improper coloring on {name}")
                self.assertTrue(complete, f"{algorithm} left an edge uncolored on {name}")
                self.assertLessEqual(
                    ncolors, deg + 1,
                    f"{algorithm} used {ncolors} colors on {name} (budget {deg + 1})")

    def test_deterministic_algorithms_are_reproducible(self):
        # The deterministic algorithms must give an identical coloring on repeat.
        for algorithm in DETERMINISTIC_EXACT_ALGORITHMS:
            name, graph = self.SMALL_GRAPHS[-1]
            vertices, edges, neighbors, deg = graph
            r1 = switchboard_find_edge_coloring(algorithm, deg, vertices, edges, neighbors)
            r2 = switchboard_find_edge_coloring(algorithm, deg, vertices, edges, neighbors)
            self.assertEqual(
                _canonical_coloring(r1), _canonical_coloring(r2),
                f"{algorithm} is not deterministic on {name}")


class GraphColoringReproducibilityTester(BaseCase):
    """Seed-controlled behavior of the randomized algorithms not covered by the
    module-level parametrized reproducibility tests."""

    def test_different_seeds_can_differ(self):
        """Different seeds should generally produce different colorings.

        This is a soft check: for a sufficiently non-trivial graph at least one
        randomized algorithm must yield distinct output for two different seeds.
        (Individual algorithms may coincidentally match on small graphs, so we
        only require that *some* algorithm distinguishes the seeds.)
        """
        vertices, edges, neighbors, deg = make_grid_graph(4, 4)
        any_differs = False
        for algorithm in RANDOMIZED_ALGORITHMS:
            a = _canonical_coloring(
                switchboard_find_edge_coloring(algorithm, deg, vertices, edges, neighbors, seed=1))
            b = _canonical_coloring(
                switchboard_find_edge_coloring(algorithm, deg, vertices, edges, neighbors, seed=999))
            if a != b:
                any_differs = True
        self.assertTrue(any_differs, "no randomized algorithm distinguished two different seeds")

    def test_find_edge_coloring_is_reproducible(self):
        """The networkx-based find_edge_coloring is also seedable/reproducible."""
        vertices, edges, neighbors, deg = make_grid_graph(4, 4)
        r1 = find_edge_coloring(deg, vertices, edges, neighbors, seed=123)
        r2 = find_edge_coloring(deg, vertices, edges, neighbors, seed=123)
        self.assertEqual(
            _canonical_coloring(r1), _canonical_coloring(r2),
            "find_edge_coloring not reproducible with a fixed seed")
        proper, complete, _ncolors = assess_coloring(r1, edges)
        self.assertTrue(proper and complete)


# ---------------------------------------------------------------------------
# Scaling / algorithm-selection suite.
#
# These tests characterize *which algorithm to use in which situation*. They are
# not pure correctness tests (those live in GraphColoringTester above); instead
# they run every algorithm across graph families and sizes, recording runtime,
# color count (quality), and whether the algorithm produced a proper+complete
# coloring at all. Findings are printed as a table (visible with `pytest -s`).
#
# Because some algorithms can loop for a very long time on dense graphs, every
# run is guarded by a hard wall-clock timeout in a separate process.
#
# The suite is marked `slow` and additionally carries `@skip_in_ci`, so it is
# skipped automatically in CI (where the `CI` env var is set) and is intended to
# be run locally only:
#   * run everything locally:   pytest test/unit/tools/test_graphcoloring.py
#   * run just this suite:      pytest -m slow -s test/unit/tools/test_graphcoloring.py
#   * skip it locally too:      pytest -m "not slow" ...
#
# Per-algorithm expectations, derived from measured behavior on this codebase:
#   - vizing        : deterministic; always terminates; proper+complete and
#                     near-optimal (<= deg+1 colors) on every family tested,
#                     including dense complete graphs and grids. It uses a greedy
#                     simple-case fast path and falls back to a Vizing-chain step
#                     (the Misra-Gries procedure) for the hard case.
#   - new_bipartite : fast, but its core popularization step is unimplemented, so
#                     it leaves edges uncolored on dense/grid graphs -- only
#                     trust it after verifying completeness.
#   - sinnamon      : always terminates and is proper, but uses many extra
#                     colors (poor quality); avoid when color count matters.
#   - misra_gries   : deterministic; always terminates; proper+complete and
#                     near-optimal (<= deg+1 colors) on every family tested,
#                     including dense complete graphs and grids. Fast and the most
#                     reliable algorithm here -- a good default.
#   - moser_tardos  : randomized, seedable (numpy Generator via `seed`) and
#                     reproducible for a fixed seed; always terminates (bounded
#                     resample budget, then retries and finally raises). Proper+
#                     complete near-optimal on sparse/moderate graphs; can time
#                     out on large dense graphs (e.g. K_10).
#   - assadi        : randomized, seedable and reproducible; always terminates
#                     (bounded retry budget, then raises). Proper+complete and
#                     near-optimal on sparse/moderate graphs, but the local-search
#                     solver is slow at high degree, so it can time out as degree
#                     grows.
# ---------------------------------------------------------------------------
@skip_in_ci
@pytest.mark.slow
class GraphColoringScalingTester(BaseCase):
    """Benchmark-style tests to guide algorithm selection.

    Each test builds a graph, runs every algorithm under a timeout, prints a
    comparison table, and asserts only the invariants we are confident about
    (namely: the algorithms in SPARSE_SAFE must succeed, terminate, and be
    near-optimal on the low-degree families). The slower or incomplete
    algorithms are recorded but not asserted on, so the table stays informative
    without making the suite flaky.
    """

    def _run_all(self, graph, timeout=PER_ALGO_TIMEOUT):
        return {algo: run_with_timeout(algo, graph, seed=0, timeout=timeout)
                for algo in ALL_ALGORITHMS}

    def _assert_timeout_guard_held(self, results, graph_name):
        """No algorithm may run unbounded: each finishes or is killed on budget."""
        for algo in ALL_ALGORITHMS:
            self.assertLessEqual(
                results[algo]["seconds"], PER_ALGO_TIMEOUT + 3.0,
                f"{algo} was not bounded by the timeout on {graph_name}")

    def _assert_sparse_safe_algos_good(self, graph, results, color_slack=2):
        """The sparse-safe algorithms must succeed, terminate, and be near-optimal.

        By Vizing's theorem a simple graph needs deg or deg+1 colors; we allow a
        small slack. This is asserted only for low-degree families, where these
        algorithms were observed to be reliable across runs.
        """
        _vertices, _edges, _neighbors, deg = graph
        budget = deg + 1 + color_slack
        for algo in SPARSE_SAFE:
            r = results[algo]
            self.assertEqual(r["status"], "ok",
                             f"{algo} did not finish: {r['status']} ({r['detail']})")
            self.assertTrue(
                r["proper"] and r["complete"],
                f"{algo} produced an invalid coloring (proper={r['proper']}, complete={r['complete']})")
            self.assertLessEqual(
                r["colors"], budget,
                f"{algo} used {r['colors']} colors on a max-degree-{deg} graph (budget {budget})")

    def _assert_at_least_one_valid(self, graph, results, graph_name):
        """Sanity floor: on any graph, *some* algorithm must produce a valid coloring."""
        valid = [a for a in ALL_ALGORITHMS
                 if results[a]["status"] == "ok"
                 and results[a]["proper"] and results[a]["complete"]]
        self.assertTrue(valid, f"No algorithm produced a proper+complete coloring on {graph_name}")

    def test_scaling_cycle_graphs(self):
        """Sparse (deg=2) family across increasing size."""
        for n in [10, 50, 200]:
            graph = make_cycle_graph(n)
            results = self._run_all(graph)
            _print_table(f"cycle C_{n}", graph, results)
            self._assert_timeout_guard_held(results, f"C_{n}")
            self._assert_sparse_safe_algos_good(graph, results)

    def test_scaling_path_graphs(self):
        """Sparse path family (deg<=2)."""
        for n in [10, 50, 200]:
            graph = make_path_graph(n)
            results = self._run_all(graph)
            _print_table(f"path P_{n}", graph, results)
            self._assert_timeout_guard_held(results, f"P_{n}")
            self._assert_sparse_safe_algos_good(graph, results)

    def test_scaling_grid_graphs(self):
        """2D lattice (deg<=4) -- representative of a planar QPU connectivity."""
        for rows, cols in [(3, 3), (5, 5), (8, 8)]:
            graph = make_grid_graph(rows, cols)
            results = self._run_all(graph)
            _print_table(f"grid {rows}x{cols}", graph, results)
            self._assert_timeout_guard_held(results, f"grid {rows}x{cols}")
            self._assert_sparse_safe_algos_good(graph, results)

    def test_scaling_random_regular_graphs(self):
        """Random ~d-regular graphs at moderate degree."""
        for n, d in [(12, 3), (20, 4), (30, 5)]:
            graph = make_random_regular_graph(n, d, seed=1234)
            results = self._run_all(graph)
            _print_table(f"random ~{d}-regular on {n} verts", graph, results)
            self._assert_timeout_guard_held(results, f"random ~{d}-regular n={n}")
            self._assert_at_least_one_valid(graph, results, f"random ~{d}-regular n={n}")

    def test_scaling_complete_graphs_dense_stress(self):
        """Dense stress test (K_n, deg=n-1).

        This is where the algorithms diverge most sharply: some hang, some fail
        to converge, some explode the color count. We record everything (the
        printed table is the deliverable) and assert only that the timeout guard
        protected the run and that at least one algorithm succeeded.
        """
        for n in [6, 10, 16]:
            graph = make_complete_graph(n)
            results = self._run_all(graph)
            _print_table(f"complete K_{n}", graph, results)
            self._assert_timeout_guard_held(results, f"K_{n}")
            self._assert_at_least_one_valid(graph, results, f"K_{n}")
