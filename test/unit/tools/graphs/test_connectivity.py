#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import unittest
import numpy as np
import networkx as nx

from pygsti.tools import graphs
from pygsti.baseobjs.qubitgraph import QubitGraph
from pygsti.processors.processorspec import QubitProcessorSpec as _ProcessorSpec
from ...util import BaseCase

try:
    import igraph
    IGRAPH_IMPORTED = True
except ImportError:
    IGRAPH_IMPORTED = False

try:
    import graph_tool
    GRAPH_TOOL_IMPORTED = True
except ImportError:
    GRAPH_TOOL_IMPORTED = False


class ConnectivityTester(BaseCase):
    # Reference data for a 4-qubit line graph 0-1-2-3, used by several tests below.
    LINE4_EDGES = [(0, 1), (1, 2), (2, 3)]
    LINE4_ADJACENCY = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])

    def test_bare_matrix_must_be_adjacency(self):
        # A bare qubit_graph matrix must be a plain adjacency matrix (entries >= 0); the
        # all-zero (edgeless) matrix is trivially valid.
        Z = np.zeros((3, 3))
        np.testing.assert_array_equal(graphs.qubit_graph_adjacency_matrix(Z), np.zeros((3, 3), int))

        A_star = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])  # star, center 0
        np.testing.assert_array_equal(graphs.qubit_graph_adjacency_matrix(A_star), A_star)

        # A graph Laplacian (L = D - A) has negative off-diagonal entries and must be rejected
        # with a clear error, rather than silently guessed at or misinterpreted as an adjacency
        # matrix.
        L = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])  # 0-1-2 line graph Laplacian
        with self.assertRaises(ValueError):
            graphs.qubit_graph_to_networkx(L)
        L_star = np.array([[3, -1, -1, -1], [-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
        with self.assertRaises(ValueError):
            graphs.qubit_graph_adjacency_matrix(L_star)

    def test_networkx_inputs(self):
        A_expected = self.LINE4_ADJACENCY[:3, :3]  # 3-qubit line sub-case (0-1-2)

        G_int = nx.Graph()
        G_int.add_nodes_from([0, 1, 2])
        G_int.add_edges_from([(0, 1), (1, 2)])
        np.testing.assert_array_equal(graphs.qubit_graph_adjacency_matrix(G_int), A_expected)

        # A DiGraph with only "forward" edges must still be symmetrized.
        G_di = nx.DiGraph()
        G_di.add_nodes_from([0, 1, 2])
        G_di.add_edges_from([(0, 1), (1, 2)])
        np.testing.assert_array_equal(graphs.qubit_graph_adjacency_matrix(G_di), A_expected)

        # Non-integer (string) qubit labels: native node order defines position.
        G_str = nx.Graph()
        G_str.add_nodes_from(['q0', 'q1', 'q2'])
        G_str.add_edges_from([('q0', 'q1'), ('q1', 'q2')])
        np.testing.assert_array_equal(graphs.qubit_graph_adjacency_matrix(G_str), A_expected)

    def test_node_ordering_regression(self):
        # An nx graph whose insertion order differs from the desired qubit_labels order must
        # still be reordered correctly (the old bare-matrix API had no such ordering check at
        # all -- a mismatched matrix would silently produce wrong results).
        G = nx.Graph()
        G.add_nodes_from(['Q2', 'Q0', 'Q1'])  # scrambled insertion order
        G.add_edge('Q0', 'Q1')
        out = graphs.qubit_graph_to_networkx(G, qubit_labels=['Q0', 'Q1', 'Q2'])
        self.assertEqual(list(out.nodes()), ['Q0', 'Q1', 'Q2'])
        A = graphs.qubit_graph_adjacency_matrix(G, qubit_labels=['Q0', 'Q1', 'Q2'])
        np.testing.assert_array_equal(A, np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]))

        # Positional fallback: a plain-int-labeled graph mapped onto non-integer qubit_labels.
        G_pos = nx.Graph()
        G_pos.add_nodes_from([0, 1, 2])
        G_pos.add_edge(0, 1)
        out_pos = graphs.qubit_graph_to_networkx(G_pos, qubit_labels=['Qa', 'Qb', 'Qc'])
        self.assertEqual(list(out_pos.nodes()), ['Qa', 'Qb', 'Qc'])
        self.assertEqual(sorted(out_pos.edges()), [('Qa', 'Qb')])

        # An unreconcilable mismatch raises a clear ValueError naming the offending node.
        with self.assertRaises(ValueError):
            graphs.qubit_graph_to_networkx(G, qubit_labels=['Q0', 'Q1'])  # 'Q2' unexplained

    @unittest.skipUnless(IGRAPH_IMPORTED, "igraph not installed")
    def test_igraph_input(self):
        g = igraph.Graph(n=4, edges=self.LINE4_EDGES)
        np.testing.assert_array_equal(graphs.qubit_graph_adjacency_matrix(g), self.LINE4_ADJACENCY)

        g_named = igraph.Graph(n=3)
        g_named.vs['name'] = ['q0', 'q1', 'q2']
        g_named.add_edges([(0, 1)])
        out = graphs.qubit_graph_to_networkx(g_named)
        self.assertEqual(list(out.nodes()), ['q0', 'q1', 'q2'])
        self.assertEqual(sorted(out.edges()), [('q0', 'q1')])

        # Directed igraph graph with only one direction present must still be symmetrized.
        g_dir = igraph.Graph(n=4, edges=self.LINE4_EDGES, directed=True)
        np.testing.assert_array_equal(graphs.qubit_graph_adjacency_matrix(g_dir), self.LINE4_ADJACENCY)

    @unittest.skipUnless(GRAPH_TOOL_IMPORTED, "graph_tool not installed")
    def test_graph_tool_input(self):
        g = graph_tool.Graph(directed=False)
        g.add_vertex(4)
        g.add_edge_list(self.LINE4_EDGES)
        np.testing.assert_array_equal(graphs.qubit_graph_adjacency_matrix(g), self.LINE4_ADJACENCY)

        g_named = graph_tool.Graph(directed=False)
        g_named.add_vertex(3)
        vprop = g_named.new_vertex_property("string")
        for i, name in enumerate(['q0', 'q1', 'q2']):
            vprop[g_named.vertex(i)] = name
        g_named.vertex_properties['name'] = vprop
        g_named.add_edge(g_named.vertex(0), g_named.vertex(1))
        out = graphs.qubit_graph_to_networkx(g_named)
        self.assertEqual(list(out.nodes()), ['q0', 'q1', 'q2'])
        self.assertEqual(sorted(out.edges()), [('q0', 'q1')])

        # Directed graph_tool graph with only one direction present must still be symmetrized.
        g_dir = graph_tool.Graph(directed=True)
        g_dir.add_vertex(4)
        g_dir.add_edge_list(self.LINE4_EDGES)
        np.testing.assert_array_equal(graphs.qubit_graph_adjacency_matrix(g_dir), self.LINE4_ADJACENCY)

    def test_qubitgraph_and_processorspec_inputs(self):
        qg = QubitGraph.common_graph(4, "line", directed=True, qubit_labels=[0, 1, 2, 3])
        np.testing.assert_array_equal(graphs.qubit_graph_adjacency_matrix(qg), self.LINE4_ADJACENCY)

        pspec = _ProcessorSpec(4, ['Gxpi2', 'Gypi2', 'Gcphase'], {}, {'Gcphase': self.LINE4_EDGES},
                                qubit_labels=[0, 1, 2, 3])
        np.testing.assert_array_equal(graphs.qubit_graph_adjacency_matrix(pspec), self.LINE4_ADJACENCY)
        # pspec.qubit_graph itself is edgeless (no explicit `geometry` given to the constructor);
        # this checks we used compute_2Q_connectivity() and not the (wrong) edgeless qubit_graph.
        self.assertEqual(pspec.qubit_graph.edges(), [])

    def test_scipy_sparse_input(self):
        import scipy.sparse as sp
        A_sparse = sp.csr_matrix(self.LINE4_ADJACENCY)
        np.testing.assert_array_equal(
            graphs.qubit_graph_adjacency_matrix(A_sparse), self.LINE4_ADJACENCY)

    def test_qubit_graph_from_edges(self):
        G = graphs.qubit_graph_from_edges([(0, 1), (1, 2)], [0, 1, 2, 3])
        self.assertEqual(list(G.nodes()), [0, 1, 2, 3])  # qubit 3 has no edges but is present
        self.assertEqual(sorted(G.edges()), [(0, 1), (1, 2)])
        with self.assertRaises(ValueError):
            graphs.qubit_graph_from_edges([(0, 5)], [0, 1, 2, 3])  # 5 isn't a qubit label


class WithinHopsTester(BaseCase):
    """
    Characterization tests for `within_hops_matrix` / `qubits_within_hops`.

    These pin down the *current* behavior of both functions on small, hand-checkable graphs
    (line, ring, star, grid, disconnected, empty), across the `hops`, `include_self` and
    `qubit_labels` arguments. They exist so that the (purely internal) BFS/post-processing
    implementations can be optimized without silently changing results: before this class, both
    functions were exercised only indirectly through the TensorFlow-gated `pygsti.extras.ml`
    tests, i.e. not at all on a machine without TensorFlow installed.
    """

    # 4-qubit line 0-1-2-3 (diameter 3), as a bare adjacency matrix.
    LINE4 = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])

    def test_line_graph_all_hop_radii(self):
        # Every hop radius on the 4-qubit line, spelled out in full. Note the diagonal is always
        # False: `within_hops_matrix` never counts a qubit as close to itself.
        expected = {
            0: np.zeros((4, 4), dtype=int),
            1: self.LINE4,
            2: np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]]),
            3: np.ones((4, 4), dtype=int) - np.eye(4, dtype=int),
        }
        for hops, expected_close in expected.items():
            close = graphs.within_hops_matrix(self.LINE4, hops)
            self.assertEqual(close.dtype, np.dtype(bool))
            np.testing.assert_array_equal(close.astype(int), expected_close,
                                          err_msg=f"hops={hops}")

        # Distance is true shortest-path distance, not "at most hops edges of any walk", so
        # `hops` beyond the graph's diameter (3 here) saturates rather than continuing to grow.
        for hops in (4, 100, 100000):
            np.testing.assert_array_equal(graphs.within_hops_matrix(self.LINE4, hops),
                                          graphs.within_hops_matrix(self.LINE4, 3))

    def test_hops_is_monotonic(self):
        # Increasing `hops` can only ever add qubits, never remove them.
        grid = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 4), ordering='sorted')
        previous = graphs.within_hops_matrix(grid, 0)
        for hops in range(1, 8):
            close = graphs.within_hops_matrix(grid, hops)
            self.assertTrue(np.all(close >= previous), f"hops={hops} lost a qubit")
            np.testing.assert_array_equal(close, close.T)  # always symmetric
            self.assertFalse(np.any(np.diag(close)))  # always a False diagonal
            previous = close

    def test_one_hop_matrix_is_the_adjacency_matrix(self):
        # hops=1 must reproduce the adjacency matrix exactly, for every input representation.
        edges = [(0, 1), (1, 2), (2, 3)]
        qubit_graphs = [
            self.LINE4,
            graphs.qubit_graph_from_edges(edges, [0, 1, 2, 3]),
            QubitGraph.common_graph(4, "line", directed=True, qubit_labels=[0, 1, 2, 3]),
            _ProcessorSpec(4, ['Gxpi2', 'Gypi2', 'Gcphase'], {}, {'Gcphase': edges},
                           qubit_labels=[0, 1, 2, 3]),
        ]
        for qubit_graph in qubit_graphs:
            close = graphs.within_hops_matrix(qubit_graph, 1)
            np.testing.assert_array_equal(
                close.astype(int), graphs.qubit_graph_adjacency_matrix(qubit_graph),
                err_msg=f"input type {type(qubit_graph).__name__}")

    def test_hops_zero_and_include_self(self):
        # hops=0 is the degenerate-but-valid case: nothing is within zero hops of anything, so
        # `include_self` is the *only* thing that populates the result.
        np.testing.assert_array_equal(graphs.within_hops_matrix(self.LINE4, 0),
                                      np.zeros((4, 4), dtype=bool))
        self.assertEqual(graphs.qubits_within_hops(self.LINE4, 0), [[0], [1], [2], [3]])
        self.assertEqual(graphs.qubits_within_hops(self.LINE4, 0, include_self=False),
                         [[], [], [], []])

        # `include_self` adds exactly qubit i to list i, and nothing else.
        self.assertEqual(graphs.qubits_within_hops(self.LINE4, 1),
                         [[0, 1], [0, 1, 2], [1, 2, 3], [2, 3]])
        self.assertEqual(graphs.qubits_within_hops(self.LINE4, 1, include_self=False),
                         [[1], [0, 2], [1, 3], [2]])

    def test_lists_agree_with_matrix(self):
        # `qubits_within_hops` is exactly the row-wise nonzero pattern of `within_hops_matrix`,
        # sorted ascending, plus the diagonal when `include_self`.
        ring = graphs.qubit_graph_from_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
                                             list(range(5)))
        for hops in range(4):
            close = graphs.within_hops_matrix(ring, hops)
            for include_self in (True, False):
                actual = graphs.qubits_within_hops(ring, hops, include_self=include_self)
                expected = []
                for i in range(5):
                    row = set(np.flatnonzero(close[i, :]).tolist())
                    if include_self:
                        row.add(i)
                    expected.append(sorted(row))
                self.assertEqual(actual, expected, f"hops={hops}, include_self={include_self}")
                self.assertTrue(all(row == sorted(row) for row in actual))  # sorted ascending

    def test_ring_graph_wraps_around(self):
        # A 5-ring: qubit 0's 1-hop neighbors include qubit 4, i.e. distance wraps around and is
        # not merely index proximity.
        ring = graphs.qubit_graph_from_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
                                             list(range(5)))
        self.assertEqual(graphs.qubits_within_hops(ring, 1, include_self=False),
                         [[1, 4], [0, 2], [1, 3], [2, 4], [0, 3]])
        # The 5-ring has diameter 2, so 2 hops already reaches everything.
        self.assertEqual(graphs.qubits_within_hops(ring, 2, include_self=False),
                         [[1, 2, 3, 4], [0, 2, 3, 4], [0, 1, 3, 4], [0, 1, 2, 4], [0, 1, 2, 3]])

    def test_star_graph_hub_asymmetry(self):
        # A star with hub 0: the hub sees every leaf at 1 hop, but leaves see only the hub --
        # a case where the per-row counts genuinely differ.
        star = graphs.qubit_graph_from_edges([(0, 1), (0, 2), (0, 3)], list(range(4)))
        self.assertEqual(graphs.qubits_within_hops(star, 1, include_self=False),
                         [[1, 2, 3], [0], [0], [0]])
        # Leaves are 2 hops apart from each other (through the hub).
        self.assertEqual(graphs.qubits_within_hops(star, 2, include_self=False),
                         [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])

    def test_grid_graph(self):
        # 3x3 grid relabeled 0..8 in row-major order -- the qubit-lattice case that motivates
        # these functions. Corners have 2 neighbors, edges 3, and the center (4) has 4.
        grid = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3), ordering='sorted')
        self.assertEqual(
            graphs.qubits_within_hops(grid, 1, include_self=False),
            [[1, 3], [0, 2, 4], [1, 5], [0, 4, 6], [1, 3, 5, 7], [2, 4, 8],
             [3, 7], [4, 6, 8], [5, 7]])
        # From the center, 2 hops reaches every other qubit (the corners are exactly 2 away);
        # from a corner it does not (the opposite corner is 4 hops away).
        two_hops = graphs.qubits_within_hops(grid, 2, include_self=False)
        self.assertEqual(two_hops[4], [0, 1, 2, 3, 5, 6, 7, 8])
        self.assertEqual(two_hops[0], [1, 2, 3, 4, 6])

    def test_disconnected_graph(self):
        # Components 0-1, 2-3, and an isolated qubit 4. No `hops`, however large, may ever leak
        # across components -- and an isolated qubit's list is empty unless `include_self`.
        disconnected = graphs.qubit_graph_from_edges([(0, 1), (2, 3)], list(range(5)))
        for hops in (1, 2, 5, 100000):
            self.assertEqual(graphs.qubits_within_hops(disconnected, hops, include_self=False),
                             [[1], [0], [3], [2], []], f"hops={hops}")
            self.assertEqual(graphs.qubits_within_hops(disconnected, hops),
                             [[0, 1], [0, 1], [2, 3], [2, 3], [4]], f"hops={hops}")

    def test_edgeless_and_empty_graphs(self):
        # An all-zero adjacency matrix: no qubit is ever close to any other.
        edgeless = np.zeros((3, 3))
        np.testing.assert_array_equal(graphs.within_hops_matrix(edgeless, 5),
                                      np.zeros((3, 3), dtype=bool))
        self.assertEqual(graphs.qubits_within_hops(edgeless, 5), [[0], [1], [2]])
        self.assertEqual(graphs.qubits_within_hops(edgeless, 5, include_self=False), [[], [], []])

        # The zero-qubit edge case must not raise.
        self.assertEqual(graphs.within_hops_matrix(np.zeros((0, 0)), 1).shape, (0, 0))
        self.assertEqual(graphs.qubits_within_hops(np.zeros((0, 0)), 1), [])

    def test_qubit_labels_fixes_positions(self):
        # Returned indices are *positional*, so the qubit_labels ordering fully determines them.
        # Here the graph's own insertion order puts the isolated qubit 'Q2' at position 0.
        G = nx.Graph()
        G.add_nodes_from(['Q2', 'Q0', 'Q1'])  # scrambled insertion order
        G.add_edge('Q0', 'Q1')
        self.assertEqual(graphs.qubits_within_hops(G, 1), [[0], [1, 2], [1, 2]])

        # ...whereas an explicit ordering re-indexes it, without changing the graph itself.
        self.assertEqual(graphs.qubits_within_hops(G, 1, qubit_labels=['Q0', 'Q1', 'Q2']),
                         [[0, 1], [0, 1], [2]])
        self.assertEqual(graphs.qubits_within_hops(G, 1, qubit_labels=['Q2', 'Q1', 'Q0']),
                         [[0], [1, 2], [1, 2]])
        np.testing.assert_array_equal(
            graphs.within_hops_matrix(G, 1, qubit_labels=['Q0', 'Q1', 'Q2']).astype(int),
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]))

        # A qubit_labels entry the graph doesn't mention is added as an isolated qubit.
        self.assertEqual(graphs.qubits_within_hops(G, 1, qubit_labels=['Q0', 'Q1', 'Q2', 'Q3']),
                         [[0, 1], [0, 1], [2], [3]])

        # A bare matrix's rows/columns are positional, so qubit_labels only renames them --
        # the resulting indices are unchanged.
        self.assertEqual(graphs.qubits_within_hops(self.LINE4, 1, qubit_labels=list('abcd')),
                         graphs.qubits_within_hops(self.LINE4, 1))

    def test_invalid_hops_rejected(self):
        # `hops` must be a non-negative integer. A float is rejected even when integral, since
        # it is far more likely to be a bug than an intentional 1.0.
        bad_hops: list = [-1, -100, 1.0, 2.5, np.float64(1.0), None, '1']
        for hops in bad_hops:
            with self.assertRaises(ValueError, msg=f"hops={hops!r} should be rejected"):
                graphs.within_hops_matrix(self.LINE4, hops)
            with self.assertRaises(ValueError, msg=f"hops={hops!r} should be rejected"):
                graphs.qubits_within_hops(self.LINE4, hops)

        # numpy integers are accepted, and agree with the corresponding Python int.
        expected = graphs.within_hops_matrix(self.LINE4, 2)
        integer_hops: list = [np.int64(2), np.int32(2), np.uint8(2)]
        for hops in integer_hops:
            np.testing.assert_array_equal(graphs.within_hops_matrix(self.LINE4, hops), expected,
                                          err_msg=f"hops={hops!r} ({type(hops).__name__})")
