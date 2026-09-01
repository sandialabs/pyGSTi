from __future__ import annotations

import contextlib
import io
import itertools
import unittest

import numpy as np
import pymatching

from pygsti.extras.rareevent.connected_enumeration import (
    ConnectedEnumerationEstimator,
    enumerate_connected_malignant,
    enumerate_connected_subgraphs,
    predicted_f_w,
    predicted_f_w_weighted,
    weight_points,
)
from pygsti.extras.rareevent.failure_spectrum import poisson_binomial_pmf
from pygsti.extras.rareevent.malignant import MalignantSetEstimator
from pygsti.extras.rareevent.noise import ExactNoiseErrorModel, SI1000NoiseModel
from pygsti.extras.rareevent.rare_event import (
    ErrorMechanism,
    FailureOracle,
    MechanismCatalog,
    make_repetition_code_memory_circuit,
)
from pygsti.extras.rareevent.splitting_swap import build_detector_adjacency

# ---------------------------------------------------------------------------
# Small synthetic catalogs (kept local to this test file, per the sibling-safe
# convention -- do not import fixtures from another test module).
# ---------------------------------------------------------------------------


def _build_ring_catalog() -> MechanismCatalog:
    # A 6-mechanism ring where consecutive mechanisms share a detector, so the
    # detector-adjacency neighborhood of any mechanism is itself plus its two
    # ring neighbors. Heterogeneous probabilities exercise the odds ratio.
    p_refs = [0.02, 0.05, 0.08, 0.12, 0.2, 0.3]
    detector_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    mechanisms = [
        ErrorMechanism(detectors=pair, observables=(), p_ref=p) for pair, p in zip(detector_pairs, p_refs)
    ]
    return MechanismCatalog(mechanisms=mechanisms, num_detectors=6, num_observables=0)


def _build_path_catalog() -> MechanismCatalog:
    # Same construction as the ring, minus the wraparound edge: a 5-mechanism
    # path 0-1-2-3-4 in the detector-adjacency graph.
    p_refs = [0.02, 0.05, 0.08, 0.12, 0.2]
    detector_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    mechanisms = [
        ErrorMechanism(detectors=pair, observables=(), p_ref=p) for pair, p in zip(detector_pairs, p_refs)
    ]
    return MechanismCatalog(mechanisms=mechanisms, num_detectors=6, num_observables=0)


def _build_isolated_vertex_catalog() -> MechanismCatalog:
    # A 3-mechanism path (0-1-2 in detector space) plus a fourth mechanism
    # that shares no detector with anything: an isolated vertex.
    mechanisms = [
        ErrorMechanism(detectors=(0, 1), observables=(), p_ref=0.05),
        ErrorMechanism(detectors=(1, 2), observables=(), p_ref=0.07),
        ErrorMechanism(detectors=(2, 3), observables=(), p_ref=0.09),
        ErrorMechanism(detectors=(), observables=(0,), p_ref=0.03),
    ]
    return MechanismCatalog(mechanisms=mechanisms, num_detectors=4, num_observables=1)


class NeverFailsOracle:
    """A trivial ForwardSimulator that never reports a failure."""

    def fails(self, active: set[int]) -> bool:
        return False


class MajorityOracle:
    """Fails iff at least half of the given "core" mechanisms are active.

    Deliberately non-monotone-friendly stand-in decoder used only to exercise
    the enumerator's malignancy/minimality bookkeeping on hand-checkable
    inputs (it is not meant to model any real decoder).
    """

    def __init__(self, core: frozenset[int], threshold: int) -> None:
        self.core = core
        self.threshold = threshold

    def fails(self, active: set[int]) -> bool:
        return len(active & self.core) >= self.threshold


# ---------------------------------------------------------------------------
# Brute-force reference: combinations + union-find/BFS connectivity check.
# ---------------------------------------------------------------------------


def _is_connected(combo: tuple[int, ...], neighbor_sets: list[set[int]]) -> bool:
    if len(combo) <= 1:
        return True
    combo_set = set(combo)
    start = combo[0]
    visited = {start}
    frontier = [start]
    while frontier:
        cur = frontier.pop()
        for nb in neighbor_sets[cur] & combo_set:
            if nb not in visited:
                visited.add(nb)
                frontier.append(nb)
    return visited == combo_set


def _brute_force_connected_sets(neighbors: list[tuple[int, ...]], n: int, max_weight: int) -> set[tuple[int, ...]]:
    neighbor_sets = [set(t) for t in neighbors]
    found: set[tuple[int, ...]] = set()
    for w in range(1, max_weight + 1):
        for combo in itertools.combinations(range(n), w):
            if _is_connected(combo, neighbor_sets):
                found.add(combo)
    return found


class TestEnumerationCorrectness(unittest.TestCase):
    def _check(self, catalog: MechanismCatalog, max_weight: int) -> None:
        neighbors = build_detector_adjacency(catalog)
        n = len(catalog.mechanisms)
        sets_by_size, nodes_visited, completed = enumerate_connected_subgraphs(neighbors, n, max_weight)
        self.assertTrue(completed)
        self.assertGreater(nodes_visited, 0)

        got: set[tuple[int, ...]] = set()
        total_emitted = 0
        for v, lst in sets_by_size.items():
            self.assertTrue(all(len(s) == v for s in lst))
            total_emitted += len(lst)
            got.update(lst)
        # No duplicates: every visited set is emitted exactly once.
        self.assertEqual(total_emitted, len(got))

        expected = _brute_force_connected_sets(neighbors, n, max_weight)
        self.assertEqual(got, expected)

    def test_ring_graph(self) -> None:
        self._check(_build_ring_catalog(), max_weight=4)

    def test_path_graph(self) -> None:
        self._check(_build_path_catalog(), max_weight=4)

    def test_isolated_vertex_graph(self) -> None:
        self._check(_build_isolated_vertex_catalog(), max_weight=4)


class TestMalignantEnumerationOnSyntheticOracle(unittest.TestCase):
    def test_minimal_and_non_minimal_counts_on_ring(self) -> None:
        # Failure iff >= 2 of the connected triple {0, 1, 2} are active.
        # Minimal malignant connected sets: {0,1} and {1,2} (weight 2; {0,2}
        # is malignant but not connected on the ring, since 0 and 2 are not
        # ring-adjacent). Every connected weight-3 set that is malignant
        # ((0,1,2), (1,2,3), (5,0,1) -- each containing >= 2 of the core)
        # contains one of these malignant pairs, so none are minimal.
        catalog = _build_ring_catalog()
        oracle = MajorityOracle(core=frozenset({0, 1, 2}), threshold=2)

        result = enumerate_connected_malignant(catalog, oracle, max_weight=3)
        self.assertTrue(result.completed)

        self.assertEqual(result.minimal_counts[1], 0)
        self.assertEqual(result.minimal_counts[2], 2)
        self.assertEqual(set(result.minimal_sets[2]), {(0, 1), (1, 2)})
        self.assertEqual(result.minimal_counts[3], 0)
        self.assertEqual(result.non_minimal_counts[3], 3)

        points = weight_points(result)
        by_weight = {p.weight: p for p in points}
        self.assertEqual(by_weight[2].estimate, 2.0)
        self.assertEqual(by_weight[2].method, "connected_enumeration")
        self.assertEqual(by_weight[2].kind, "m_v")
        self.assertTrue(by_weight[2].exact)
        self.assertEqual(by_weight[2].rel_err, 0.0)
        self.assertFalse(by_weight[2].lower_bound)


class TestBruteForceCrossCheck(unittest.TestCase):
    def test_m_v_matches_full_brute_force_repetition_code(self) -> None:
        p0 = 0.02
        circuit = make_repetition_code_memory_circuit(distance=3, rounds=2, p=0)
        noise = SI1000NoiseModel()
        error_model = ExactNoiseErrorModel(circuit, noise, p_ref=p0)
        dem = noise(circuit, p0).detector_error_model(decompose_errors=True, flatten_loops=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
        oracle = FailureOracle(error_model.catalog, matching)
        catalog = error_model.catalog

        with contextlib.redirect_stdout(io.StringIO()):
            res_mal = MalignantSetEstimator().estimate(
                error_model=error_model,
                simulator=oracle,
                p_scales=[p0],
                max_weight=3,
                num_mechanisms=error_model.num_mechanisms,
            )
        malignant_lookup = {tuple(sorted(c)) for c in res_mal["malignant_sets"]}

        neighbors = build_detector_adjacency(catalog)
        neighbor_sets = [set(t) for t in neighbors]

        expected_minimal: dict[int, set[tuple[int, ...]]] = {1: set(), 2: set(), 3: set()}
        for combo in malignant_lookup:
            v = len(combo)
            if not _is_connected(combo, neighbor_sets):
                continue
            minimal = True
            for k in range(1, v):
                for sub in itertools.combinations(combo, k):
                    if tuple(sorted(sub)) in malignant_lookup:
                        minimal = False
                        break
                if not minimal:
                    break
            if minimal:
                expected_minimal[v].add(combo)

        result = enumerate_connected_malignant(catalog, oracle, max_weight=3)
        self.assertTrue(result.completed)
        for v in (1, 2, 3):
            self.assertEqual(set(result.minimal_sets[v]), expected_minimal[v])
            self.assertEqual(result.minimal_counts[v], len(expected_minimal[v]))

        # Sanity: at least one nontrivial minimal cluster should exist for a
        # real repetition-code decoding graph, else the cross-check is vacuous.
        self.assertGreater(sum(result.minimal_counts.values()), 0)

    def test_estimator_wrapper_matches_direct_call(self) -> None:
        p0 = 0.02
        circuit = make_repetition_code_memory_circuit(distance=3, rounds=2, p=0)
        noise = SI1000NoiseModel()
        error_model = ExactNoiseErrorModel(circuit, noise, p_ref=p0)
        dem = noise(circuit, p0).detector_error_model(decompose_errors=True, flatten_loops=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
        oracle = FailureOracle(error_model.catalog, matching)

        direct = enumerate_connected_malignant(error_model.catalog, oracle, max_weight=2)
        wrapped = ConnectedEnumerationEstimator().estimate(
            error_model=error_model,
            simulator=oracle,
            catalog=error_model.catalog,
            max_weight=2,
        )
        self.assertIn("weight_points", wrapped)
        self.assertIn("minimal_malignant_sets", wrapped)
        self.assertEqual(wrapped["minimal_malignant_sets"], direct.minimal_sets)
        self.assertEqual(
            {p.weight: p.estimate for p in wrapped["weight_points"]},
            {p.weight: p.estimate for p in weight_points(direct)},
        )

        with self.assertRaises(ValueError):
            ConnectedEnumerationEstimator().estimate(error_model=error_model, simulator=oracle, max_weight=2)
        with self.assertRaises(ValueError):
            ConnectedEnumerationEstimator().estimate(
                error_model=error_model, simulator=oracle, catalog=error_model.catalog
            )


class TestBudgetGuard(unittest.TestCase):
    def test_tiny_budget_flags_incomplete_without_crashing(self) -> None:
        catalog = _build_ring_catalog()
        result = enumerate_connected_malignant(catalog, NeverFailsOracle(), max_weight=4, max_nodes=2)
        self.assertFalse(result.completed)
        self.assertLessEqual(result.nodes_visited, 3)  # stops shortly after exceeding the budget of 2

        points = weight_points(result)
        for p in points:
            self.assertTrue(p.lower_bound)
            self.assertFalse(p.exact)
            for v_meta in ("nodes_visited", "oracle_calls"):
                self.assertIn(v_meta, p.meta)
            self.assertFalse(p.meta["completed"])

    def test_enumerate_connected_subgraphs_budget_guard(self) -> None:
        catalog = _build_ring_catalog()
        neighbors = build_detector_adjacency(catalog)
        sets_by_size, nodes_visited, completed = enumerate_connected_subgraphs(
            neighbors, len(catalog.mechanisms), max_weight=4, max_nodes=1
        )
        self.assertFalse(completed)
        self.assertGreaterEqual(nodes_visited, 1)


class TestPredictedFw(unittest.TestCase):
    def test_predicted_f_w_uniform_first_order(self) -> None:
        # Two disjoint weight-1 minimal clusters among n=5 mechanisms.
        m_counts = {1: 2}
        n = 5
        w = 2
        # f(w) ~= m(1) * C(n-1, w-1) / C(n, w) = 2 * C(4,1) / C(5,2) = 2*4/10.
        expected = 2 * 4 / 10
        self.assertAlmostEqual(predicted_f_w(m_counts, n, w), expected)

    def test_predicted_f_w_zero_below_onset(self) -> None:
        self.assertEqual(predicted_f_w({3: 5}, n=10, w=1), 0.0)

    def test_predicted_f_w_weighted_matches_hand_computed_symmetric_case(self) -> None:
        # n=3 mechanisms, all q=0.5, one minimal cluster = {0}. At w=1, by
        # symmetry P(active set == {0} | W=1) = 1/3.
        probs = np.array([0.5, 0.5, 0.5])
        minimal_sets = [(0,)]
        got = predicted_f_w_weighted(minimal_sets, probs, w=1)
        self.assertAlmostEqual(got, 1.0 / 3.0)

    def test_predicted_f_w_weighted_consistent_with_poisson_binomial_pmf(self) -> None:
        probs = np.array([0.1, 0.2, 0.3, 0.4])
        minimal_sets = [(0,), (2, 3)]
        w = 2
        got = predicted_f_w_weighted(minimal_sets, probs, w)

        # Hand-expand: sum over clusters c of P(all of c) * P(exactly w-|c| of
        # the rest), divided by P(W=w).
        pmf_w = poisson_binomial_pmf(probs, max_weight=w)
        p_w = pmf_w[w]

        # cluster {0}: need 1 more of {1,2,3}
        rest0 = poisson_binomial_pmf(probs[[1, 2, 3]], max_weight=1)
        term0 = probs[0] * rest0[1]

        # cluster {2,3}: need 0 more of {0,1}
        rest1 = poisson_binomial_pmf(probs[[0, 1]], max_weight=0)
        term1 = probs[2] * probs[3] * rest1[0]

        expected = (term0 + term1) / p_w
        self.assertAlmostEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
