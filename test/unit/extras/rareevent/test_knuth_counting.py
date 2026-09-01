from __future__ import annotations

import itertools
import statistics
import unittest

import pymatching

from pygsti.extras.rareevent.knuth_counting import (
    KnuthCountingEstimator,
    enumerate_connected_sets,
    exhaustive_count_m_v,
    knuth_estimate_m_v,
    knuth_estimate_many,
)
from pygsti.extras.rareevent.noise import ExactNoiseErrorModel, SI1000NoiseModel
from pygsti.extras.rareevent.rare_event import (
    ErrorMechanism,
    FailureOracle,
    MechanismCatalog,
    make_repetition_code_memory_circuit,
)
from pygsti.extras.rareevent.splitting_swap import build_detector_adjacency


def _build_ring_catalog() -> MechanismCatalog:
    # Copied (not imported) from tests/test_splitting_local.py::_build_ring_catalog,
    # per the isolation rules for this task: a 6-mechanism ring where
    # consecutive mechanisms share a detector, so detector-adjacency of any
    # mechanism is itself plus its two ring neighbors. Heterogeneous
    # probabilities exercise the odds ratio (unused here, but kept for
    # fidelity with the original construction).
    p_refs = [0.02, 0.05, 0.08, 0.12, 0.2, 0.3]
    detector_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    mechanisms = [
        ErrorMechanism(detectors=pair, observables=(), p_ref=p) for pair, p in zip(detector_pairs, p_refs)
    ]
    return MechanismCatalog(mechanisms=mechanisms, num_detectors=6, num_observables=0)


def _build_path_catalog() -> MechanismCatalog:
    """A 6-mechanism path (no wraparound): sparser than the ring, exercising
    dead-end (zero-children) descents."""
    p_refs = [0.02, 0.05, 0.08, 0.12, 0.2, 0.3]
    detector_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    mechanisms = [
        ErrorMechanism(detectors=pair, observables=(), p_ref=p) for pair, p in zip(detector_pairs, p_refs)
    ]
    return MechanismCatalog(mechanisms=mechanisms, num_detectors=6, num_observables=0)


class CoreMajorityOracle:
    """E fails iff at least 2 of mechanisms {0, 1, 2} are active.

    Not weight-monotone in general (adding a mechanism can create OR destroy
    membership in {0, 1, 2}'s majority only by construction it can't destroy,
    but larger sets built from other non-{0,1,2} mechanisms never fail), so
    minimality of a failing set is not implied by testing only its
    (v - 1)-subsets: e.g. {0, 1, 2} fails, but so does its subset {0, 1}, so
    {0, 1, 2} is NOT minimal even though it is itself a failure -- exactly
    the case the minimality check must catch.
    """

    def fails(self, active: set[int]) -> bool:
        return len(active & {0, 1, 2}) >= 2


def _brute_force_connected_sets(
    neighbors: list[tuple[int, ...]], n: int, weight: int
) -> list[frozenset[int]]:
    found = []
    for combo in itertools.combinations(range(n), weight):
        s = set(combo)
        adj = {i: set(neighbors[i]) & s for i in s}
        start = next(iter(s))
        seen = {start}
        frontier = [start]
        while frontier:
            cur = frontier.pop()
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    frontier.append(nb)
        if seen == s:
            found.append(frozenset(s))
    return found


class TestTreeCorrectness(unittest.TestCase):
    """The duplicate-free connected-set forest must match brute-force enumeration."""

    def _check_catalog(self, catalog: MechanismCatalog) -> None:
        neighbors = build_detector_adjacency(catalog)
        n = len(neighbors)
        for w in range(1, n + 1):
            found, nodes_visited, exhausted = enumerate_connected_sets(neighbors, w, max_nodes=1_000_000)
            self.assertTrue(exhausted, f"budget exhausted at weight {w} (nodes_visited={nodes_visited})")
            self.assertEqual(len(found), len(set(found)), f"duplicate connected sets found at weight {w}")
            brute = _brute_force_connected_sets(neighbors, n, w)
            self.assertEqual(set(found), set(brute), f"mismatch at weight {w}")

    def test_ring_catalog(self) -> None:
        self._check_catalog(_build_ring_catalog())

    def test_path_catalog(self) -> None:
        self._check_catalog(_build_path_catalog())

    def test_exhaustive_count_all_mode_never_touches_oracle(self) -> None:
        catalog = _build_ring_catalog()
        neighbors = build_detector_adjacency(catalog)

        class _NeverCalledOracle:
            def fails(self, active: set[int]) -> bool:
                raise AssertionError("mode='all' must never call the oracle")

        oracle = _NeverCalledOracle()
        for w in range(1, 7):
            found, _, _ = enumerate_connected_sets(neighbors, w, max_nodes=1_000_000)
            result = exhaustive_count_m_v(catalog, oracle, weight=w, mode="all")
            self.assertEqual(result.count, len(found))
            self.assertEqual(result.oracle_calls, 0)
            self.assertTrue(result.exhausted)


class TestUnbiasedness(unittest.TestCase):
    """Knuth estimates must agree with exact counts, and reported SE with empirical spread."""

    def test_ring_matches_exact_within_5_se_at_several_weights(self) -> None:
        catalog = _build_ring_catalog()
        oracle = CoreMajorityOracle()
        for weight in (2, 3, 4):
            exact = exhaustive_count_m_v(catalog, oracle, weight=weight, mode="minimal", max_nodes=1_000_000)
            self.assertTrue(exact.exhausted)
            kn = knuth_estimate_m_v(catalog, oracle, weight=weight, probes_per_root=20_000, seed=101 + weight)
            se = kn.meta["standard_error"]
            tol = 5 * se + 1e-9  # +eps guards the exact-zero-variance case (e.g. m(v) == 0 always)
            self.assertLess(
                abs(kn.estimate - exact.count),
                tol,
                f"weight={weight}: knuth {kn.estimate} +/- {se} vs exact {exact.count}",
            )

    def test_path_matches_exact_within_5_se_at_several_weights(self) -> None:
        catalog = _build_path_catalog()
        oracle = CoreMajorityOracle()
        for weight in (2, 3, 4):
            exact = exhaustive_count_m_v(catalog, oracle, weight=weight, mode="minimal", max_nodes=1_000_000)
            self.assertTrue(exact.exhausted)
            kn = knuth_estimate_m_v(catalog, oracle, weight=weight, probes_per_root=20_000, seed=201 + weight)
            se = kn.meta["standard_error"]
            tol = 5 * se + 1e-9
            self.assertLess(
                abs(kn.estimate - exact.count),
                tol,
                f"weight={weight}: knuth {kn.estimate} +/- {se} vs exact {exact.count}",
            )

    def test_reported_se_matches_empirical_spread_of_independent_runs(self) -> None:
        # The ring at weight=2 has genuine per-probe randomness (multiple
        # roots have branching subtrees), so its estimator variance is
        # nonzero -- a meaningful case for cross-checking the SE formula.
        catalog = _build_ring_catalog()
        oracle = CoreMajorityOracle()
        weight = 2
        probes_per_root = 800
        num_replicates = 40

        estimates = []
        reported_ses = []
        for seed in range(num_replicates):
            kn = knuth_estimate_m_v(
                catalog, oracle, weight=weight, probes_per_root=probes_per_root, seed=3_000 + seed
            )
            estimates.append(kn.estimate)
            reported_ses.append(kn.meta["standard_error"])

        empirical_sd = statistics.stdev(estimates)
        mean_reported_se = statistics.fmean(reported_ses)
        self.assertGreater(empirical_sd, 0.0, "expected genuine probe-to-probe variance for this case")

        ratio = mean_reported_se / empirical_sd
        self.assertGreater(ratio, 0.5, f"reported SE {mean_reported_se} vs empirical SD {empirical_sd}")
        self.assertLess(ratio, 2.0, f"reported SE {mean_reported_se} vs empirical SD {empirical_sd}")


class TestRepetitionCodeGroundTruth(unittest.TestCase):
    """d=3 repetition code: exact m(2)/m(3) vs the Knuth estimator."""

    def _setup(self) -> tuple[MechanismCatalog, FailureOracle]:
        p0 = 0.02
        circuit = make_repetition_code_memory_circuit(distance=3, rounds=2, p=0)
        noise = SI1000NoiseModel()
        error_model = ExactNoiseErrorModel(circuit, noise, p_ref=p0)
        dem = noise(circuit, p0).detector_error_model(decompose_errors=True, flatten_loops=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
        oracle = FailureOracle(error_model.catalog, matching)
        return error_model.catalog, oracle

    def test_m2_and_m3_agree_with_exhaustive(self) -> None:
        catalog, oracle = self._setup()

        exact2 = exhaustive_count_m_v(catalog, oracle, weight=2, max_nodes=200_000)
        self.assertTrue(exact2.exhausted)
        kn2 = knuth_estimate_m_v(catalog, oracle, weight=2, probes_per_root=2_000, seed=11)
        se2 = kn2.meta["standard_error"]
        self.assertLess(abs(kn2.estimate - exact2.count), 5 * se2 + 1e-9)

        exact3 = exhaustive_count_m_v(catalog, oracle, weight=3, max_nodes=200_000)
        self.assertTrue(exact3.exhausted)
        kn3 = knuth_estimate_m_v(catalog, oracle, weight=3, probes_per_root=2_000, seed=12)
        se3 = kn3.meta["standard_error"]
        self.assertLess(abs(kn3.estimate - exact3.count), 5 * se3 + 1e-9)


class TestApiSurface(unittest.TestCase):
    """Exercise total_probe_budget, malignant mode, knuth_estimate_many, and the Estimator wrapper."""

    def test_total_probe_budget_spreads_ceil_over_roots(self) -> None:
        catalog = _build_ring_catalog()
        oracle = CoreMajorityOracle()
        n = len(catalog.mechanisms)
        kn = knuth_estimate_m_v(catalog, oracle, weight=2, total_probe_budget=6_000, seed=1)
        self.assertEqual(kn.meta["probes_per_root"], -(-6_000 // n))
        self.assertEqual(kn.meta["num_roots"], n)

    def test_malignant_mode_counts_non_minimal_failures_too(self) -> None:
        # {0, 1, 2} fails but is not minimal (its subset {0, 1} also fails),
        # so malignant-mode m(3) must exceed minimal-mode m(3) here.
        catalog = _build_ring_catalog()
        oracle = CoreMajorityOracle()
        exact_minimal = exhaustive_count_m_v(catalog, oracle, weight=3, mode="minimal")
        exact_malignant = exhaustive_count_m_v(catalog, oracle, weight=3, mode="malignant")
        self.assertGreater(exact_malignant.count, exact_minimal.count)

        kn_malignant = knuth_estimate_m_v(catalog, oracle, weight=3, probes_per_root=6_000, seed=5, minimal=False)
        se = kn_malignant.meta["standard_error"]
        self.assertLess(abs(kn_malignant.estimate - exact_malignant.count), 5 * se + 1e-9)
        self.assertEqual(kn_malignant.meta["mode"], "malignant")

    def test_knuth_estimate_many_returns_one_point_per_weight(self) -> None:
        catalog = _build_ring_catalog()
        oracle = CoreMajorityOracle()
        points = knuth_estimate_many(catalog, oracle, weights=[2, 3], probes_per_root=500, seed=1)
        self.assertEqual([p.weight for p in points], [2, 3])
        for p in points:
            self.assertEqual(p.method, "knuth_counting")
            self.assertEqual(p.kind, "m_v")
            self.assertFalse(p.exact)
            self.assertGreaterEqual(p.estimate, 0.0)

    def test_estimator_wrapper_requires_catalog_and_weight(self) -> None:
        catalog = _build_ring_catalog()
        oracle = CoreMajorityOracle()
        estimator = KnuthCountingEstimator()

        with self.assertRaises(ValueError):
            estimator.estimate(error_model=None, simulator=oracle, weight=2)
        with self.assertRaises(ValueError):
            estimator.estimate(error_model=None, simulator=oracle, catalog=catalog)

        result = estimator.estimate(
            error_model=None, simulator=oracle, catalog=catalog, weights=[2, 3], probes_per_root=500, seed=9
        )
        self.assertIn("weight_points", result)
        self.assertEqual(len(result["weight_points"]), 2)

    def test_exhaustive_count_respects_max_nodes_budget(self) -> None:
        catalog = _build_ring_catalog()
        oracle = CoreMajorityOracle()
        result = exhaustive_count_m_v(catalog, oracle, weight=4, max_nodes=1)
        self.assertFalse(result.exhausted)
        self.assertLessEqual(result.nodes_visited, 1)


if __name__ == "__main__":
    unittest.main()
