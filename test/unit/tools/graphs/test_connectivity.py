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
