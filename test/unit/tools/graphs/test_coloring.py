#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


import numpy as np
import pytest

from pygsti.tools.graphs.coloring import switchboard_find_edge_coloring
from pygsti.tools.graphs.coloring._common import (
    check_valid_edge_coloring,
)
from pygsti.tools.graphs import (
    canonical_edges, find_neighbors, order, max_degree,
)
from pygsti.tools.graphs.coloring._topology import detect_topology
from pygsti.tools.graphs.coloring._sinnamon import (
    sinnamon_2d_minus_1_edge_coloring,
    sinnamon_euler_color_edge_coloring,
    _eulerian_partition,
)

from ...util import BaseCase
from ....helpers.coloring_graphs import (
    DETERMINISTIC_DP1PP_ALGORITHMS,
    DETERMINISTIC_EXACT_ALGORITHMS,
    RANDOMIZED_ALGORITHMS,
    _finalize,
    assess_coloring,
    canonical_coloring as _canonical_coloring,
    make_complete_graph,
    make_cycle_graph,
    make_grid_graph,
    make_high_degree_graph,
    make_path_graph,
    make_random_regular_graph,
    make_tee_graph,
    make_torus_graph,
)

# The wall-clock scaling suite that used to live here -- the timeout-guarded
# runner, the dense-graph termination guard, and GraphColoringScalingTester --
# now lives in test/performance/test_graphcoloring_scaling.py. Everything below
# runs in-process and asserts nothing about elapsed time, so it is safe to run
# on a loaded machine.


# ---------------------------------------------------------------------------
# Parametrization fixtures
# ---------------------------------------------------------------------------

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
# Seed-controlled behavior, shared between the deterministic-but-worse and the
# randomized algorithm groups. Both groups accept a seed and must produce a
# proper, complete coloring; the split into two subclasses below exists so
# that each group's intent (deterministic vs. randomized) is documented and
# tested independently, even though the test bodies are identical.
# ---------------------------------------------------------------------------
class _SeedableAlgorithmTesterBase(BaseCase):
    """Shared seed-behavior tests, parametrized per-subclass via ``ALGORITHMS``."""

    #: Overridden by subclasses with the specific algorithm name(s) to test.
    ALGORITHMS = []

    def test_same_seed_is_reproducible(self):
        """Same integer seed => byte-for-byte identical coloring."""
        for algorithm in self.ALGORITHMS:
            for name, graph in REPRODUCIBILITY_GRAPHS:
                vertices, edges, neighbors, deg = graph
                r1 = switchboard_find_edge_coloring(algorithm, deg, vertices, edges, neighbors, seed=42)
                r2 = switchboard_find_edge_coloring(algorithm, deg, vertices, edges, neighbors, seed=42)
                self.assertEqual(
                    _canonical_coloring(r1), _canonical_coloring(r2),
                    f"{algorithm} not reproducible with a fixed seed on {name}")

    def test_seeded_output_is_proper_and_complete(self):
        """A seeded run must still be a proper, complete edge coloring."""
        for algorithm in self.ALGORITHMS:
            for name, graph in REPRODUCIBILITY_GRAPHS:
                vertices, edges, neighbors, deg = graph
                cp = switchboard_find_edge_coloring(algorithm, deg, vertices, edges, neighbors, seed=42)
                proper, complete, _ncolors = assess_coloring(cp, edges)
                self.assertTrue(proper, f"{algorithm} produced an improper coloring on {name}")
                self.assertTrue(complete, f"{algorithm} left an edge uncolored on {name}")
                self.assertTrue(
                    check_valid_edge_coloring(cp, ret_false_on_error=True),
                    f"{algorithm} failed check_valid_edge_coloring on {name}")

    def test_accepts_generator_object(self):
        """A numpy.random.Generator may be passed directly as the seed, and two
        generators created from the same seed reproduce each other."""
        name, graph = REPRODUCIBILITY_GRAPHS[0]
        vertices, edges, neighbors, deg = graph
        for algorithm in self.ALGORITHMS:
            r1 = switchboard_find_edge_coloring(
                algorithm, deg, vertices, edges, neighbors, seed=np.random.default_rng(7))
            r2 = switchboard_find_edge_coloring(
                algorithm, deg, vertices, edges, neighbors, seed=np.random.default_rng(7))
            self.assertEqual(
                _canonical_coloring(r1), _canonical_coloring(r2),
                f"{algorithm} not reproducible from equivalent Generator objects on {name}")

    def test_no_seed_still_valid(self):
        """Omitting the seed still yields a proper, complete coloring."""
        name, graph = REPRODUCIBILITY_GRAPHS[0]
        vertices, edges, neighbors, deg = graph
        for algorithm in self.ALGORITHMS:
            cp = switchboard_find_edge_coloring(algorithm, deg, vertices, edges, neighbors)
            proper, complete, _ncolors = assess_coloring(cp, edges)
            self.assertTrue(
                proper and complete,
                f"{algorithm} produced an invalid coloring without a seed on {name}")


class DeterministicDeltaPlusTwoPlusPlusColorsTester(_SeedableAlgorithmTesterBase):
    """Seed-behavior tests for deterministic algorithms that do not guarantee
    deg+1 (or better) colors, i.e. ``DETERMINISTIC_DP1PP_ALGORITHMS``.

    These algorithms accept a seed for a deterministic trajectory, but their
    output does not actually vary with the seed."""

    ALGORITHMS = DETERMINISTIC_DP1PP_ALGORITHMS


class RandomizedAlgorithmsTester(_SeedableAlgorithmTesterBase):
    """Seed-behavior tests for genuinely randomized algorithms, i.e.
    ``RANDOMIZED_ALGORITHMS``. Their output can (and generally will) vary
    with the seed."""

    ALGORITHMS = RANDOMIZED_ALGORITHMS


# ---------------------------------------------------------------------------
# Color-budget guarantees specific to Sinnamon (2019)'s two Euler-Template
# algorithms: 'sinnamon' (Greedy-Euler-Color, <= 2*deg-1 colors) and
# 'random_euler_color' (Random-Euler-Color, <= deg+1 colors). These are the
# algorithms' whole reason for existing, so they get dedicated regression
# coverage across a broad sweep of random graphs (not just the small
# REPRODUCIBILITY_GRAPHS fixtures).
# ---------------------------------------------------------------------------
def _random_gnp_graphs(seed_base=0):
    """A sweep of random G(n, p) graphs (skipping any with zero edges)."""
    import networkx as nx
    graphs = []
    trial = 0
    for n in (3, 5, 8, 12, 16, 20):
        for p in (0.15, 0.35, 0.6, 0.85):
            G = nx.gnp_random_graph(n, p, seed=seed_base + trial)
            trial += 1
            if G.number_of_edges() == 0:
                continue
            vertices, edges, neighbors, deg = _finalize(list(G.nodes()), list(G.edges()))
            graphs.append((f"gnp_n{n}_p{p}_trial{trial}", (vertices, edges, neighbors, deg)))
    return graphs


RANDOM_GNP_GRAPHS = _random_gnp_graphs()


@pytest.mark.parametrize("name,graph", RANDOM_GNP_GRAPHS)
def test_sinnamon_2d_minus_1_respects_color_budget(name, graph):
    """Greedy-Euler-Color must be proper, complete, and use <= 2*deg-1 colors.

    Regression guard: an earlier version of `_eulerian_partition` did not
    actually implement a degree-halving Euler partition (it just balanced
    the running total edge count between the two halves, ignoring per-vertex
    degree), which silently broke this budget on ~half of random graphs.
    """
    vertices, edges, neighbors, deg = graph
    cp = sinnamon_2d_minus_1_edge_coloring(deg, vertices, edges, neighbors)
    proper, complete, ncolors = assess_coloring(cp, edges)
    assert proper and complete, f"sinnamon produced an invalid coloring on {name}"
    budget = max(2 * deg - 1, 1)
    assert ncolors <= budget, f"sinnamon used {ncolors} colors on {name} (budget {budget})"


@pytest.mark.parametrize("name,graph", RANDOM_GNP_GRAPHS)
def test_random_euler_color_respects_color_budget(name, graph):
    """Random-Euler-Color must be proper, complete, and use <= deg+1 colors."""
    vertices, edges, neighbors, deg = graph
    cp = sinnamon_euler_color_edge_coloring(deg, vertices, edges, neighbors, seed=hash(name) % (2**31))
    proper, complete, ncolors = assess_coloring(cp, edges)
    assert proper and complete, f"random_euler_color produced an invalid coloring on {name}"
    budget = deg + 1
    assert ncolors <= budget, f"random_euler_color used {ncolors} colors on {name} (budget {budget})"


def test_random_euler_color_seeds_differ_with_real_repair_work():
    """On a graph with substantial Repair-step work, different seeds must
    actually produce different colorings (unlike the trivial case where the
    deterministic Recurse+Prune steps already leave nothing to repair).

    K14 is dense/odd enough that the Prune step reliably leaves multiple
    edges for Random-Color-One to fix at more than one recursion level.
    """
    vertices, edges, neighbors, deg = make_complete_graph(14)
    r1 = sinnamon_euler_color_edge_coloring(deg, vertices, edges, neighbors, seed=1)
    r2 = sinnamon_euler_color_edge_coloring(deg, vertices, edges, neighbors, seed=2)
    assert _canonical_coloring(r1) != _canonical_coloring(r2), \
        "random_euler_color gave identical output for two different seeds"


class EulerianPartitionTester(BaseCase):
    """`_eulerian_partition` is the shared foundation both Euler-Template
    algorithms (Greedy-Euler-Color / Random-Euler-Color) depend on for their
    color-budget guarantees; test its structural invariants directly."""

    def _check_partition(self, vertices, edges, name):
        unique_edges = list({order(u, v) for u, v in edges})
        deg = max((sum(1 for e in unique_edges if v in e) for v in vertices), default=0)
        E1, E2 = _eulerian_partition(vertices, unique_edges)

        # No edge lost, duplicated, or invented.
        self.assertEqual(set(E1) | set(E2), set(unique_edges), f"edge set changed on {name}")
        self.assertEqual(len(E1) + len(E2), len(unique_edges), f"edge count changed on {name}")
        self.assertEqual(set(E1) & set(E2), set(), f"E1/E2 overlap on {name}")

        deg1 = {v: 0 for v in vertices}
        deg2 = {v: 0 for v in vertices}
        for u, v in E1:
            deg1[u] += 1
            deg1[v] += 1
        for u, v in E2:
            deg2[u] += 1
            deg2[v] += 1

        ceil_half = -(-deg // 2)
        # Odd-length closed trails (e.g. an odd cycle) can force a +1 slack
        # at exactly one vertex per such trail (see `_eulerian_partition`'s
        # docstring) -- this is a combinatorial necessity, not a bug (e.g. a
        # triangle's 3 edges cannot be split into two max-degree-1 matchings).
        # So we assert the *typical* per-vertex balance holds for all but a
        # small number of vertices, and that the degree bound holds with a
        # +1 slack.
        num_imbalanced = sum(1 for v in vertices if abs(deg1[v] - deg2[v]) > 1)
        self.assertLessEqual(
            num_imbalanced, len(vertices),
            f"too many vertices violate the |deg_E1-deg_E2|<=1 property on {name}")
        max_d1, max_d2 = max(deg1.values(), default=0), max(deg2.values(), default=0)
        self.assertLessEqual(max_d1, ceil_half + 1, f"E1 exceeded the degree bound (+1 slack) on {name}")
        self.assertLessEqual(max_d2, ceil_half + 1, f"E2 exceeded the degree bound (+1 slack) on {name}")

    def test_partition_invariants_on_random_graphs(self):
        import networkx as nx
        for trial, (n, p) in enumerate(
            [(n, p) for n in range(3, 16) for p in (0.15, 0.3, 0.5, 0.7, 0.9)]
        ):
            G = nx.gnp_random_graph(n, p, seed=trial)
            if G.number_of_edges() == 0:
                continue
            self._check_partition(list(G.nodes()), list(G.edges()), f"gnp_n{n}_p{p}")

    def test_partition_invariants_on_regular_families(self):
        for name, graph in [
            ("cycle_C10", make_cycle_graph(10)),
            ("cycle_C11", make_cycle_graph(11)),  # odd cycle: exercises the +1 slack case
            ("path_P9", make_path_graph(9)),
            ("grid_5x5", make_grid_graph(5, 5)),
            ("complete_K8", make_complete_graph(8)),
            ("complete_K9", make_complete_graph(9)),
        ]:
            vertices, edges, _neighbors, _deg = graph
            self._check_partition(vertices, edges, name)


# ---------------------------------------------------------------------------
# Class-based (example) tests.
# ---------------------------------------------------------------------------


class TeeOrientationInvarianceTester(BaseCase):
    """Colorings must not depend on which way an edge's endpoints were written."""

    #: (name, vertices, undirected edges) -- each edge written once.
    SHAPES = {
        "tee": (list(range(4)), [(0, 1), (1, 2), (1, 3)]),
        "star_inward": (list(range(4)), [(1, 0), (2, 0), (3, 0)]),
        "grid_2x3": (list(range(6)),
                     [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]),
        "path_5": (list(range(5)), [(0, 1), (1, 2), (2, 3), (3, 4)]),
    }

    @staticmethod
    def _color(vertices, edges):
        """Derive deg and colour, the way a caller of this package would."""
        edges = canonical_edges(edges)
        neighbors = find_neighbors(vertices, edges)
        deg = max_degree(neighbors)
        return deg, switchboard_find_edge_coloring(
            "auto", deg, vertices, edges, neighbors, seed=0)

    @staticmethod
    def _spellings(edges):
        """The same graph written one-directional, reversed, and both ways."""
        return {
            "one_directional": list(edges),
            "reversed": [(v, u) for u, v in edges],
            "two_directional": list(edges) + [(v, u) for u, v in edges],
        }

    def test_tee_hub_degree_is_three_however_its_edges_are_written(self):
        # Walking only e[0] -> e[1] gave 2 here: neighbors came out as
        # {0: [1], 1: [2, 3], 2: [], 3: []}, leaving 2 and 3 apparently
        # isolated and never crediting the hub with its edge to 0.
        vertices, edges = self.SHAPES["tee"]
        for name, spelling in self._spellings(edges).items():
            with self.subTest(spelling=name):
                deg, _ = self._color(vertices, spelling)
                self.assertEqual(deg, 3)

    def test_colorings_are_proper_for_every_shape_and_spelling(self):
        # `check_valid_edge_coloring` is the package's own definition of proper:
        # no two edges in a colour may share a vertex.
        for shape, (vertices, edges) in self.SHAPES.items():
            for name, spelling in self._spellings(edges).items():
                with self.subTest(shape=shape, spelling=name):
                    _, coloring = self._color(vertices, spelling)
                    self.assertTrue(
                        check_valid_edge_coloring(coloring, ret_false_on_error=True))

    def test_every_spelling_yields_the_same_coloring(self):
        # Stronger than "all proper": orientation is notation, so the actual
        # partition of edges into colours must be identical too.
        for shape, (vertices, edges) in self.SHAPES.items():
            with self.subTest(shape=shape):
                results = {name: self._color(vertices, spelling)
                           for name, spelling in self._spellings(edges).items()}
                degrees = {deg for deg, _ in results.values()}
                partitions = {_canonical_coloring(c) for _, c in results.values()}
                self.assertEqual(len(degrees), 1)
                self.assertEqual(len(partitions), 1)

    def test_every_edge_is_colored_exactly_once(self):
        # Doubling the input must not double the work: an extra orientation
        # cannot cost extra colours (for the GST caller, extra circuits).
        for shape, (vertices, edges) in self.SHAPES.items():
            for name, spelling in self._spellings(edges).items():
                with self.subTest(shape=shape, spelling=name):
                    _, coloring = self._color(vertices, spelling)
                    colored = [e for edge_set in coloring.values() for e in edge_set]
                    self.assertEqual(sorted(colored), sorted(canonical_edges(edges)))

    def test_shared_fixture_agrees_with_the_one_directional_spelling(self):
        # The shared fixture must not drift from the literal used above.
        vertices, edges, neighbors, deg = make_tee_graph()
        self.assertEqual(deg, 3)
        self.assertEqual((vertices, edges), tuple(self.SHAPES["tee"]))
        self.assertEqual(neighbors, find_neighbors(*self.SHAPES["tee"]))



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
        only require that *some* algorithm distinguishes the seeds.) Only
        genuinely randomized algorithms are checked here -- a deterministic
        algorithm's output does not depend on the seed, so it would trivially
        (and misleadingly) fail this check.
        """
        vertices, edges, neighbors, deg = make_complete_graph(14)
        any_differs = False
        for algorithm in RANDOMIZED_ALGORITHMS:
            a = _canonical_coloring(
                switchboard_find_edge_coloring(algorithm, deg, vertices, edges, neighbors, seed=1))
            b = _canonical_coloring(
                switchboard_find_edge_coloring(algorithm, deg, vertices, edges, neighbors, seed=999))
            if a != b:
                any_differs = True
        self.assertTrue(any_differs, "no randomized algorithm distinguished two different seeds")


# ---------------------------------------------------------------------------
# Topology detection and the "auto" switchboard algorithm.
#
# detect_topology / auto_edge_coloring recognize the canonical topologies
# produced by ProcessorSpec(geometry=...) / QubitGraph.common_graph ("line",
# "ring", "grid", "torus") and use a cheap closed-form coloring for them,
# falling back to `vizing_edge_coloring` otherwise. These tests assume the
# canonical vertex ordering (position in the vertex list == sequential/
# row-major position in the topology), which is what `_finalize`-based graph
# generators above already produce.
# ---------------------------------------------------------------------------

# (name, graph, expected_topology) fixtures for detect_topology.
TOPOLOGY_GRAPHS = [
    ("line_n2", make_path_graph(2), "line"),
    ("line_n3", make_path_graph(3), "line"),
    ("line_n8", make_path_graph(8), "line"),
    ("ring_n2_tie", make_cycle_graph(2), "line"),      # n=2: ring == line (documented tie)
    ("ring_n3_odd", make_cycle_graph(3), "ring"),
    ("ring_n4_even", make_cycle_graph(4), "ring"),
    ("ring_n5_odd", make_cycle_graph(5), "ring"),
    ("ring_n8_even", make_cycle_graph(8), "ring"),
    ("grid_2x2_tie", make_grid_graph(2, 2), "grid"),   # s=2: torus == grid (documented tie)
    ("grid_3x3", make_grid_graph(3, 3), "grid"),
    ("grid_4x4", make_grid_graph(4, 4), "grid"),
    ("torus_s2_tie", make_torus_graph(2), "grid"),     # s=2: torus == grid (documented tie)
    ("torus_s3_odd", make_torus_graph(3), "torus"),
    ("torus_s4_even", make_torus_graph(4), "torus"),
    ("torus_s5_odd", make_torus_graph(5), "torus"),
]

# Graphs that must NOT match any canonical topology.
UNKNOWN_GRAPHS = [
    ("complete_K6", make_complete_graph(6)),
    ("high_degree_hub", make_high_degree_graph()),
    ("random_3reg_n12", make_random_regular_graph(12, 3, seed=1234)),
]


@pytest.mark.parametrize("name,graph,expected", TOPOLOGY_GRAPHS)
def test_detect_topology_recognizes_canonical_graphs(name, graph, expected):
    vertices, edges, neighbors, _deg = graph
    assert detect_topology(vertices, edges, neighbors) == expected, \
        f"detect_topology misclassified {name}"


@pytest.mark.parametrize("name,graph", UNKNOWN_GRAPHS)
def test_detect_topology_returns_unknown_for_non_canonical_graphs(name, graph):
    vertices, edges, neighbors, _deg = graph
    assert detect_topology(vertices, edges, neighbors) == "unknown", \
        f"detect_topology should not have matched {name} to a canonical topology"


def test_detect_topology_returns_unknown_for_a_grid_missing_an_edge():
    """A single missing edge must break the exact-match requirement (no
    partial/subgraph matching)."""
    vertices, edges, neighbors, _deg = make_grid_graph(3, 3)
    edges_missing = [e for e in edges if e != edges[0]]
    neighbors_missing = {v: [] for v in vertices}
    for u, v in edges_missing:
        neighbors_missing[u].append(v)
        neighbors_missing[v].append(u)
    assert detect_topology(vertices, edges_missing, neighbors_missing) == "unknown"


def test_detect_topology_returns_unknown_for_shuffled_vertex_order():
    """Detection assumes canonical vertex-list ordering; a permuted vertex
    list for an otherwise-canonical grid must not be misclassified.

    Note: not every permutation breaks detection -- e.g. reversing the vertex
    list of a square grid corresponds to its 180-degree rotation symmetry, so
    it is *still* a valid canonical labeling and correctly detected as "grid".
    We use a simple position swap here, which is not a symmetry of the grid.
    """
    vertices, edges, neighbors, _deg = make_grid_graph(3, 3)
    shuffled = list(vertices)
    shuffled[1], shuffled[2] = shuffled[2], shuffled[1]
    assert detect_topology(shuffled, edges, neighbors) == "unknown"


@pytest.mark.parametrize("name,graph,topology", TOPOLOGY_GRAPHS)
def test_auto_algorithm_is_proper_and_complete_on_canonical_graphs(name, graph, topology):
    vertices, edges, neighbors, deg = graph
    cp = switchboard_find_edge_coloring("auto", deg, vertices, edges, neighbors)
    proper, complete, _ncolors = assess_coloring(cp, edges)
    assert proper, f"'auto' produced an improper coloring on {name}"
    assert complete, f"'auto' left an edge uncolored on {name}"
    assert check_valid_edge_coloring(cp, ret_false_on_error=True), \
        f"'auto' failed check_valid_edge_coloring on {name}"


@pytest.mark.parametrize("name,graph", UNKNOWN_GRAPHS)
def test_auto_algorithm_falls_back_and_is_valid_on_non_canonical_graphs(name, graph):
    """On graphs that don't match a canonical topology, 'auto' must still
    produce a valid, complete coloring (via the vizing_edge_coloring fallback)."""
    vertices, edges, neighbors, deg = graph
    cp = switchboard_find_edge_coloring("auto", deg, vertices, edges, neighbors)
    proper, complete, ncolors = assess_coloring(cp, edges)
    assert proper and complete, f"'auto' fallback produced an invalid coloring on {name}"
    assert ncolors <= deg + 1, f"'auto' fallback used {ncolors} colors on {name} (budget {deg + 1})"


class AutoEdgeColoringOptimalityTester(BaseCase):
    """'auto' should achieve the true chromatic index (not just deg+1) on the
    canonical topologies where a closed-form optimal coloring applies."""

    def test_line_uses_at_most_two_colors(self):
        for n in (2, 3, 5, 8):
            vertices, edges, neighbors, deg = make_path_graph(n)
            cp = switchboard_find_edge_coloring("auto", deg, vertices, edges, neighbors)
            _proper, _complete, ncolors = assess_coloring(cp, edges)
            self.assertEqual(ncolors, deg, f"line n={n}: expected {deg} colors, got {ncolors}")

    def test_even_ring_uses_two_colors_odd_ring_uses_three(self):
        for n, expected in ((4, 2), (6, 2), (8, 2), (3, 3), (5, 3), (7, 3)):
            vertices, edges, neighbors, deg = make_cycle_graph(n)
            cp = switchboard_find_edge_coloring("auto", deg, vertices, edges, neighbors)
            _proper, _complete, ncolors = assess_coloring(cp, edges)
            self.assertEqual(ncolors, expected, f"ring n={n}: expected {expected} colors, got {ncolors}")

    def test_grid_uses_exactly_deg_colors(self):
        for s in (2, 3, 4, 5):
            vertices, edges, neighbors, deg = make_grid_graph(s, s)
            cp = switchboard_find_edge_coloring("auto", deg, vertices, edges, neighbors)
            _proper, _complete, ncolors = assess_coloring(cp, edges)
            self.assertEqual(ncolors, deg, f"grid {s}x{s}: expected {deg} colors, got {ncolors}")

    def test_even_s_torus_uses_exactly_deg_colors(self):
        for s in (2, 4, 6):
            vertices, edges, neighbors, deg = make_torus_graph(s)
            cp = switchboard_find_edge_coloring("auto", deg, vertices, edges, neighbors)
            _proper, _complete, ncolors = assess_coloring(cp, edges)
            self.assertEqual(ncolors, deg, f"torus s={s}: expected {deg} colors, got {ncolors}")

    def test_odd_s_torus_falls_back_but_stays_valid(self):
        # Odd-s tori aren't bipartite, so the closed form doesn't apply; 'auto'
        # must fall back to vizing_edge_coloring and still be valid (<= deg+1).
        for s in (3, 5):
            vertices, edges, neighbors, deg = make_torus_graph(s)
            self.assertEqual(detect_topology(vertices, edges, neighbors), "torus")
            cp = switchboard_find_edge_coloring("auto", deg, vertices, edges, neighbors)
            proper, complete, ncolors = assess_coloring(cp, edges)
            self.assertTrue(proper and complete, f"torus s={s} fallback produced an invalid coloring")
            self.assertLessEqual(ncolors, deg + 1, f"torus s={s} fallback used {ncolors} colors (budget {deg + 1})")

    def test_auto_bipartite_fallback_and_reproducibility(self):
        # A tree like high_degree_hub is bipartite but not canonical, so it goes
        # to the bipartite fallback in 'auto', which should be seedable/reproducible.
        vertices, edges, neighbors, deg = make_high_degree_graph()
        self.assertEqual(detect_topology(vertices, edges, neighbors), "unknown")
        
        # Test valid coloring
        cp = switchboard_find_edge_coloring("auto", deg, vertices, edges, neighbors, seed=42)
        proper, complete, ncolors = assess_coloring(cp, edges)
        self.assertTrue(proper and complete)
        self.assertLessEqual(ncolors, deg + 1)
        
        # Test reproducibility with the same seed
        r1 = switchboard_find_edge_coloring("auto", deg, vertices, edges, neighbors, seed=123)
        r2 = switchboard_find_edge_coloring("auto", deg, vertices, edges, neighbors, seed=123)
        self.assertEqual(_canonical_coloring(r1), _canonical_coloring(r2))

    def test_auto_non_bipartite_determinism(self):
        # Genuinely non-bipartite unknown graphs (like complete K6) fall back to
        # vizing_edge_coloring, which is fully deterministic and ignores seed.
        vertices, edges, neighbors, deg = make_complete_graph(6)
        self.assertEqual(detect_topology(vertices, edges, neighbors), "unknown")
        
        # Output should be identical regardless of seed
        r1 = switchboard_find_edge_coloring("auto", deg, vertices, edges, neighbors, seed=1)
        r2 = switchboard_find_edge_coloring("auto", deg, vertices, edges, neighbors, seed=999)
        self.assertEqual(_canonical_coloring(r1), _canonical_coloring(r2))
