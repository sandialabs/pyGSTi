"""
Characterization tests for `pygsti.baseobjs.qubitgraph.QubitGraph`.

`QubitGraph` is the connectivity graph used by `QubitProcessorSpec`, the Clifford compilers
(`pygsti.algorithms.compilers`), cloud-noise model construction, stencil labels and the device
libraries -- but it had no dedicated test module. These tests pin its current behaviour so that
changes to it (in particular to the Floyd-Warshall caching in `_refresh_dists_and_predecessors`)
can be shown to preserve it.

Several tests below deliberately pin behaviour that is *wrong*. Those are marked with a
`KNOWN BUG` comment giving the offending line. They are here so that a future fix has exactly
one place to change and an explicit record of what the old answer was -- not because the
behaviour is endorsed.
"""
import numpy as np

from pygsti.baseobjs.qubitgraph import QubitGraph
from ..util import BaseCase


def line(n, directed=False):
    """The path graph 0-1-...-(n-1)."""
    return QubitGraph.common_graph(n, "line", directed=directed)


class QubitGraphConstructionTester(BaseCase):
    """Construction, node/edge accessors, and the `common_graph` topologies."""

    def test_common_graph_line(self):
        g = line(4)
        self.assertEqual(g.node_names, (0, 1, 2, 3))
        self.assertEqual(len(g), 4)
        self.assertEqual(g.nqubits, 4)
        self.assertEqual(g.edges(), [(0, 1), (1, 2), (2, 3)])
        self.assertFalse(g.directed)
        self.assertIsNone(g.directions)

    def test_common_graph_ring(self):
        self.assertEqual(QubitGraph.common_graph(4, "ring", directed=False).edges(),
                         [(0, 1), (0, 3), (1, 2), (2, 3)])
        # A 2-node "ring" is just an edge -- the wrap-around is only added for num_qubits > 2.
        self.assertEqual(QubitGraph.common_graph(2, "ring", directed=False).edges(), [(0, 1)])

    def test_common_graph_grid(self):
        # 3x3 grid, labelled row-major:  0-1-2 / 3-4-5 / 6-7-8
        self.assertEqual(QubitGraph.common_graph(9, "grid", directed=False).edges(),
                         [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4),
                          (3, 6), (4, 5), (4, 7), (5, 8), (6, 7), (7, 8)])

    def test_common_graph_torus(self):
        g = QubitGraph.common_graph(9, "torus", directed=False)
        # Every node has degree 4 on a 3x3 torus (each row and column wraps).
        degree = {node: 0 for node in g.node_names}
        for u, v in g.edges():
            degree[u] += 1
            degree[v] += 1
        self.assertEqual(set(degree.values()), {4})

    def test_common_graph_degenerate_sizes(self):
        for geometry in ("line", "ring"):
            self.assertEqual(QubitGraph.common_graph(0, geometry).edges(), [])
            self.assertEqual(QubitGraph.common_graph(1, geometry).edges(), [])

    def test_common_graph_rejects_bad_geometry_and_size(self):
        with self.assertRaises(ValueError):
            QubitGraph.common_graph(4, "dodecahedron")
        with self.assertRaises(AssertionError):
            QubitGraph.common_graph(5, "grid")          # not a perfect square
        with self.assertRaises(AssertionError):
            QubitGraph.common_graph(3, "line", qubit_labels=('a', 'b'))

    def test_custom_labels(self):
        g = QubitGraph.common_graph(3, "line", directed=False, qubit_labels=('Qa', 'Qb', 'Qc'))
        self.assertEqual(g.node_names, ('Qa', 'Qb', 'Qc'))
        self.assertEqual(g.edges(), [('Qa', 'Qb'), ('Qb', 'Qc')])

    def test_initial_edges_and_initial_connectivity_agree(self):
        by_edges = QubitGraph([0, 1, 2], initial_edges=[(0, 1), (1, 2)], directed=False)
        connectivity = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
        by_matrix = QubitGraph([0, 1, 2], initial_connectivity=connectivity, directed=False)
        self.assertEqual(by_edges.edges(), by_matrix.edges())

    def test_cannot_specify_both_initializers(self):
        with self.assertRaises(AssertionError):
            QubitGraph([0, 1], initial_connectivity=np.zeros((2, 2), dtype=bool),
                       initial_edges=[(0, 1)])

    def test_initial_connectivity_shape_is_checked(self):
        with self.assertRaises(AssertionError):
            QubitGraph([0, 1], initial_connectivity=np.zeros((3, 3), dtype=bool))

    def test_undirected_graph_ignores_edge_order(self):
        g = QubitGraph([0, 1, 2], initial_edges=[(2, 0)], directed=False)
        self.assertEqual(g.edges(), [(0, 2)])                 # normalized to i < j
        self.assertTrue(g.is_directly_connected(0, 2))
        self.assertTrue(g.is_directly_connected(2, 0))
        self.assertEqual(g.edges(double_for_undirected=True), [(0, 2), (2, 0)])

    def test_directed_graph_respects_edge_order(self):
        g = QubitGraph([0, 1], initial_edges=[(0, 1)], directed=True)
        self.assertTrue(g.is_directly_connected(0, 1))
        self.assertFalse(g.is_directly_connected(1, 0))

    def test_self_loops_rejected(self):
        with self.assertRaises(AssertionError):
            QubitGraph([0, 1], initial_edges=[(0, 0)])

    def test_directions_are_collected_from_edges(self):
        g = QubitGraph([0, 1, 2], initial_edges=[(0, 1, 'right'), (1, 0, 'left'), (1, 2, 'right')],
                       directed=True)
        self.assertEqual(g.directions, ['left', 'right'])     # sorted, collected from the edges
        self.assertEqual(g.edges(include_directions=True),
                         [(0, 1, 'right'), (1, 0, 'left'), (1, 2, 'right')])

    def test_directions_require_directed(self):
        with self.assertRaises(AssertionError):
            QubitGraph([0, 1], initial_edges=[(0, 1, 'right')], directed=False)

    def test_add_and_remove_edge(self):
        g = QubitGraph([0, 1, 2], directed=False)
        self.assertEqual(g.edges(), [])
        g.add_edge(0, 1)
        g.add_edges([(1, 2)])
        self.assertEqual(g.edges(), [(0, 1), (1, 2)])
        g.remove_edge(0, 1)
        self.assertEqual(g.edges(), [(1, 2)])

    def test_remove_nonexistent_edge_raises(self):
        g = QubitGraph([0, 1], directed=False)
        with self.assertRaises(AssertionError):
            g.remove_edge(0, 1)

    def test_setitem_and_getitem(self):
        g = QubitGraph([0, 1, 2], directed=False)
        self.assertFalse(g[0, 1])
        g[0, 1] = True
        self.assertTrue(g[0, 1])
        self.assertTrue(g[1, 0])                              # undirected
        self.assertTrue(g.has_edge((0, 1)))
        g[0, 1] = False
        self.assertFalse(g[0, 1])

    def test_str_contains_nodes_and_edges(self):
        text = str(line(3))
        self.assertIn('Undirected', text)
        self.assertIn('Edges', text)


class QubitGraphPathTester(BaseCase):
    """The Floyd-Warshall-backed query methods, on graphs with hand-computed answers."""

    def test_shortest_path_on_a_line(self):
        g = line(4)
        self.assertEqual(g.shortest_path(0, 3), [0, 1, 2, 3])
        self.assertEqual(g.shortest_path(3, 0), [3, 2, 1, 0])
        self.assertEqual(g.shortest_path(1, 2), [1, 2])
        self.assertEqual(g.shortest_path_edges(0, 3), [(0, 1), (1, 2), (2, 3)])

    def test_shortest_path_distance_on_a_line(self):
        g = line(4)
        self.assertEqual(g.shortest_path_distance(0, 3), 3.0)
        self.assertEqual(g.shortest_path_distance(0, 1), 1.0)
        self.assertEqual(g.shortest_path_distance(2, 2), 0.0)

    def test_shortest_path_distance_matrix_on_a_line(self):
        expected = np.array([[0., 1., 2., 3.], [1., 0., 1., 2.],
                             [2., 1., 0., 1.], [3., 2., 1., 0.]])
        self.assertArraysEqual(line(4).shortest_path_distance_matrix(), expected)

    def test_distance_matrix_is_symmetric_when_undirected(self):
        d = QubitGraph.common_graph(9, "grid", directed=False).shortest_path_distance_matrix()
        self.assertArraysEqual(d, d.T)

    def test_shortest_path_distance_on_a_grid(self):
        g = QubitGraph.common_graph(9, "grid", directed=False)
        self.assertEqual(g.shortest_path_distance(0, 8), 4.0)   # corner to corner, 3x3
        self.assertEqual(g.shortest_path_distance(0, 4), 2.0)   # corner to centre

    def test_shortest_path_intersect(self):
        g = line(4)
        self.assertTrue(g.shortest_path_intersect(0, 3, [2]))
        self.assertTrue(g.shortest_path_intersect(0, 3, [0]))    # endpoints count
        self.assertFalse(g.shortest_path_intersect(0, 1, [2, 3]))

    def test_disconnected_graph_distances_are_infinite(self):
        g = QubitGraph([0, 1, 2, 3], initial_edges=[(0, 1), (2, 3)], directed=False)
        self.assertTrue(np.isinf(g.shortest_path_distance(0, 2)))
        self.assertFalse(g.is_connected(0, 2))
        self.assertTrue(g.is_connected(0, 1))
        with self.assertRaises(AssertionError):
            g.shortest_path(0, 2)

    def test_is_connected_of_a_node_with_itself_is_false(self):
        # KNOWN BUG (qubitgraph.py:539). `is_connected` tests `_predecessors[i, j] >= 0`, and
        # Floyd-Warshall writes -9999 on the diagonal, so a node is reported as not connected
        # to itself even though its distance to itself is 0.
        g = line(3)
        self.assertFalse(g.is_connected(0, 0))
        self.assertEqual(g.shortest_path_distance(0, 0), 0.0)

    def test_single_node_graph(self):
        g = QubitGraph([0], directed=False)
        self.assertEqual(g.edges(), [])
        self.assertEqual(g.shortest_path_distance(0, 0), 0.0)
        self.assertEqual(g.shortest_path(0, 0), [0])
        self.assertTrue(g.is_connected_graph())

    def test_predecessor_matrix_shape_and_diagonal(self):
        p = line(4).shortest_path_predecessor_matrix()
        self.assertEqual(p.shape, (4, 4))
        self.assertArraysEqual(np.diag(p), np.array([-9999] * 4))

    def test_directed_line_is_one_way(self):
        g = QubitGraph([0, 1, 2], initial_edges=[(0, 1), (1, 2)], directed=True)
        self.assertEqual(g.shortest_path_distance(0, 2), 2.0)
        self.assertTrue(np.isinf(g.shortest_path_distance(2, 0)))

    def test_direction_indices_are_used_as_edge_weights(self):
        # KNOWN BUG (qubitgraph.py:308-310). `_connectivity` stores `direction_index + 1` when
        # direction names are in use, and Floyd-Warshall is called with `unweighted=False`, so
        # those direction indices become edge *weights*. Every string-geometry
        # QubitProcessorSpec builds its graph this way (processorspec.py:279), and
        # compilers.py:1393 routes on the result.
        g = QubitGraph.common_graph(9, "grid", directed=True, all_directions=True)
        self.assertEqual(g.directions, ['down', 'left', 'right', 'up'])
        self.assertEqual(sorted(set(g._connectivity.flatten().tolist())), [0, 1, 2, 3, 4])
        # 0->8 is 4 hops on a 3x3 grid, but is reported as 8.0 because 'right' (index 2, so
        # weight 3) costs three times as much as 'down' (index 0, so weight 1).
        self.assertEqual(g.shortest_path_distance(0, 8), 8.0)
        self.assertEqual(g.shortest_path_distance(0, 1), 3.0)   # adjacent, via 'right'
        self.assertEqual(g.shortest_path_distance(0, 3), 1.0)   # adjacent, via 'down'
        # The unweighted answer, for comparison, is 4 hops:
        undirected = QubitGraph.common_graph(9, "grid", directed=False)
        self.assertEqual(undirected.shortest_path_distance(0, 8), 4.0)


class QubitGraphCacheInvalidationTester(BaseCase):
    """
    The Floyd-Warshall results are cached on the instance and invalidated by a `_dirty` flag.
    These tests fix the contract that flag has to satisfy: every mutation must be reflected by
    the next query, and no query may hand out a reference that lets a caller corrupt the cache.
    """

    def test_add_edge_after_querying_is_reflected(self):
        g = QubitGraph([0, 1, 2], initial_edges=[(0, 1)], directed=False)
        self.assertTrue(np.isinf(g.shortest_path_distance(0, 2)))   # populates the cache
        g.add_edge(1, 2)
        self.assertEqual(g.shortest_path_distance(0, 2), 2.0)
        self.assertEqual(g.shortest_path(0, 2), [0, 1, 2])
        self.assertTrue(g.is_connected(0, 2))

    def test_remove_edge_after_querying_is_reflected(self):
        g = line(3)
        self.assertEqual(g.shortest_path_distance(0, 2), 2.0)       # populates the cache
        g.remove_edge(1, 2)
        self.assertTrue(np.isinf(g.shortest_path_distance(0, 2)))
        self.assertFalse(g.is_connected(0, 2))

    def test_setitem_after_querying_is_reflected(self):
        g = QubitGraph([0, 1, 2], initial_edges=[(0, 1)], directed=False)
        self.assertTrue(np.isinf(g.shortest_path_distance(0, 2)))   # populates the cache
        g[1, 2] = True
        self.assertEqual(g.shortest_path_distance(0, 2), 2.0)
        g[1, 2] = False
        self.assertTrue(np.isinf(g.shortest_path_distance(0, 2)))

    def test_add_edges_after_querying_is_reflected(self):
        g = QubitGraph([0, 1, 2, 3], initial_edges=[(0, 1)], directed=False)
        self.assertTrue(np.isinf(g.shortest_path_distance(0, 3)))   # populates the cache
        g.add_edges([(1, 2), (2, 3)])
        self.assertEqual(g.shortest_path_distance(0, 3), 3.0)

    def test_repeated_queries_agree(self):
        g = QubitGraph.common_graph(9, "grid", directed=False)
        first_d = g.shortest_path_distance_matrix()
        first_p = g.shortest_path_predecessor_matrix()
        for _ in range(3):
            self.assertArraysEqual(g.shortest_path_distance_matrix(), first_d)
            self.assertArraysEqual(g.shortest_path_predecessor_matrix(), first_p)
            self.assertEqual(g.shortest_path(0, 8), [0, 1, 2, 5, 8])

    def test_distance_matrix_returns_a_copy(self):
        # Callers must not be able to corrupt the cache through the returned array.
        g = line(4)
        d = g.shortest_path_distance_matrix()
        d[0, 3] = 999.0
        self.assertEqual(g.shortest_path_distance_matrix()[0, 3], 3.0)
        self.assertEqual(g.shortest_path_distance(0, 3), 3.0)

    def test_predecessor_matrix_returns_a_copy(self):
        g = line(4)
        p = g.shortest_path_predecessor_matrix()
        p[0, 3] = -1
        self.assertEqual(g.shortest_path_predecessor_matrix()[0, 3], 2)

    def test_copy_of_a_queried_graph_is_not_stale(self):
        g = line(3)
        g.shortest_path_distance(0, 2)              # populate the cache
        h = g.copy()
        h.add_edge(0, 2)
        self.assertEqual(h.shortest_path_distance(0, 2), 1.0)
        self.assertEqual(g.shortest_path_distance(0, 2), 2.0)   # original unaffected

    def test_subgraph_of_a_queried_graph_is_not_stale(self):
        g = line(4)
        g.shortest_path_distance_matrix()           # populate the cache
        sub = g.subgraph([1, 2, 3])
        self.assertEqual(sub.node_names, (1, 2, 3))
        self.assertEqual(sub.edges(), [(1, 2), (2, 3)])
        self.assertEqual(sub.shortest_path_distance(1, 3), 2.0)

    def test_map_qubit_labels_of_a_queried_graph_is_not_stale(self):
        g = line(3)
        g.shortest_path_distance_matrix()           # populate the cache
        h = g.map_qubit_labels({0: 'a', 1: 'b', 2: 'c'})
        self.assertEqual(h.node_names, ('a', 'b', 'c'))
        self.assertEqual(h.shortest_path_distance('a', 'c'), 2.0)

    def test_serialization_roundtrip_of_a_queried_graph_is_not_stale(self):
        g = line(4)
        g.shortest_path_distance_matrix()           # populate the cache
        h = QubitGraph.from_nice_serialization(g.to_nice_serialization())
        self.assertEqual(h.node_names, g.node_names)
        self.assertEqual(h.edges(), g.edges())
        self.assertArraysEqual(h.shortest_path_distance_matrix(),
                               g.shortest_path_distance_matrix())
        h.remove_edge(0, 1)
        self.assertTrue(np.isinf(h.shortest_path_distance(0, 1)))

    def test_mutation_between_two_different_query_methods(self):
        # Each public query goes through the same refresh, so a mutation seen by one must be
        # seen by all of them.
        g = QubitGraph([0, 1, 2], initial_edges=[(0, 1)], directed=False)
        g.shortest_path_distance_matrix()
        g.add_edge(1, 2)
        self.assertTrue(g.is_connected(0, 2))
        self.assertEqual(g.shortest_path(0, 2), [0, 1, 2])
        self.assertEqual(g.shortest_path_edges(0, 2), [(0, 1), (1, 2)])
        self.assertEqual(g.shortest_path_distance(0, 2), 2.0)
        self.assertEqual(g.shortest_path_distance_matrix()[0, 2], 2.0)
        self.assertGreaterEqual(g.shortest_path_predecessor_matrix()[0, 2], 0)
        self.assertTrue(g.shortest_path_intersect(0, 2, [1]))


class QubitGraphStructureTester(BaseCase):
    """Connectivity, radius, subgraph and the direction-walking helpers."""

    def test_is_connected_graph(self):
        self.assertTrue(line(4).is_connected_graph())
        self.assertFalse(QubitGraph([0, 1, 2, 3], initial_edges=[(0, 1), (2, 3)],
                                    directed=False).is_connected_graph())

    def test_is_connected_subgraph_containing_node_zero(self):
        g = line(4)
        self.assertTrue(g.is_connected_subgraph([0, 1]))
        self.assertTrue(g.is_connected_subgraph([0, 1, 2]))
        self.assertFalse(g.is_connected_subgraph([0, 2]))
        self.assertFalse(g.is_connected_subgraph([0, 3]))

    def test_is_connected_subgraph_not_containing_node_zero(self):
        # KNOWN BUG (qubitgraph.py:596). For undirected graphs the flood fill is seeded with the
        # hard-coded node *index 0* rather than a member of the requested subset, so a connected
        # subset that node 0 cannot reach through the subset is misreported as disconnected.
        g = line(4)                                  # 0-1-2-3
        self.assertFalse(g.is_connected_subgraph([2, 3]))   # WRONG: 2-3 is an edge
        self.assertTrue(g.is_connected_subgraph([1, 2]))    # right, but only because 0 reaches 1
        self.assertTrue(g.is_connected_subgraph([3]))       # right, via the len < 2 short-circuit
        # A 5-node line makes the pattern unmistakable: the subset gets further from node 0.
        self.assertFalse(line(5).is_connected_subgraph([3, 4]))     # WRONG: 3-4 is an edge

    def test_is_connected_subgraph_trivial_and_unknown_nodes(self):
        g = line(3)
        self.assertTrue(g.is_connected_subgraph([]))
        self.assertTrue(g.is_connected_subgraph([0]))
        self.assertFalse(g.is_connected_subgraph(['nope']))

    def test_connected_combos(self):
        # KNOWN BUG: inherits the `is_connected_subgraph` seeding bug above, so this undercounts.
        # On the line 0-1-2-3 the correct size-2 count is 3 (the three edges), not 2.
        g = line(4)
        self.assertEqual(g.connected_combos([0, 1, 2, 3], 2), 2)   # WRONG: should be 3
        self.assertEqual(g.connected_combos([0, 1, 2, 3], 3), 2)   # (0,1,2) and (1,2,3)
        self.assertEqual(g.connected_combos([0, 1, 2, 3], 4), 1)

    def test_radius(self):
        g = line(5)                                  # 0-1-2-3-4
        self.assertEqual(g.radius([2], 0), [2])
        self.assertEqual(g.radius([2], 1), [1, 2, 3])
        self.assertEqual(g.radius([2], 2), [0, 1, 2, 3, 4])
        self.assertEqual(g.radius([0, 4], 1), [0, 1, 3, 4])

    def test_radius_of_isolated_node_includes_itself(self):
        g = QubitGraph([0, 1, 2], initial_edges=[(0, 1)], directed=False)
        self.assertEqual(g.radius([2], 0), [2])
        self.assertEqual(g.radius([2], 3), [2])

    def test_radius_rejects_negative_hops(self):
        with self.assertRaises(AssertionError):
            line(3).radius([0], -1)

    def test_subgraph(self):
        g = line(4)
        sub = g.subgraph([1, 2, 3])
        self.assertEqual(sub.node_names, (1, 2, 3))
        self.assertEqual(sub.edges(), [(1, 2), (2, 3)])

    def test_subgraph_reset_nodes(self):
        sub = line(4).subgraph([1, 2, 3], reset_nodes=True)
        self.assertEqual(sub.node_names, (0, 1, 2))
        self.assertEqual(sub.edges(), [(0, 1), (1, 2)])

    def test_subgraph_drops_edges_to_removed_nodes(self):
        sub = QubitGraph.common_graph(9, "grid", directed=False).subgraph([0, 1, 2])
        self.assertEqual(sub.edges(), [(0, 1), (1, 2)])          # the column edges are gone

    def test_subgraph_recollects_direction_names(self):
        # KNOWN BUG (qubitgraph.py:892). `subgraph` does not pass `direction_names` through, so
        # the child re-derives the list from whichever directions survived. The child is
        # self-consistent, but its `.directions` list -- and hence the integer stored in
        # `_connectivity` for a given direction name -- differs from its parent's.
        parent = QubitGraph.common_graph(9, "grid", directed=True, all_directions=True)
        self.assertEqual(parent.directions, ['down', 'left', 'right', 'up'])
        child = parent.subgraph([0, 1, 2])
        self.assertEqual(child.directions, ['left', 'right'])
        self.assertEqual(child.move_in_direction(0, 'right'), 1)  # still self-consistent

    def test_copy_is_independent(self):
        g = line(3)
        h = g.copy()
        h.add_edge(0, 2)
        self.assertEqual(g.edges(), [(0, 1), (1, 2)])
        self.assertEqual(h.edges(), [(0, 1), (0, 2), (1, 2)])

    def test_map_qubit_labels_with_dict_and_callable(self):
        g = line(3)
        self.assertEqual(g.map_qubit_labels({0: 'a', 1: 'b', 2: 'c'}).edges(),
                         [('a', 'b'), ('b', 'c')])
        self.assertEqual(g.map_qubit_labels(lambda q: 'Q%d' % q).edges(),
                         [('Q0', 'Q1'), ('Q1', 'Q2')])

    def test_move_in_direction(self):
        g = QubitGraph.common_graph(9, "grid", directed=True, all_directions=True)
        self.assertEqual(g.move_in_direction(0, 'right'), 1)
        self.assertEqual(g.move_in_direction(0, 'down'), 3)
        self.assertIsNone(g.move_in_direction(2, 'right'))       # off the edge of the grid
        self.assertEqual(g.move_in_directions(0, ['right', 'down']), 4)
        self.assertIsNone(g.move_in_directions(0, ['right', 'right', 'right']))

    def test_move_in_direction_requires_directions(self):
        with self.assertRaises(AssertionError):
            line(3).move_in_direction(0, 'right')

    def test_resolve_relative_nodelabel(self):
        g = QubitGraph.common_graph(9, "grid", directed=True, all_directions=True)
        self.assertEqual(g.resolve_relative_nodelabel(5, [0]), 5)          # already absolute
        self.assertEqual(g.resolve_relative_nodelabel("@0", [4]), 4)
        self.assertEqual(g.resolve_relative_nodelabel("@0+right", [0]), 1)
        self.assertEqual(g.resolve_relative_nodelabel("@0+right+down", [0]), 4)
        self.assertEqual(g.resolve_relative_nodelabel("@1+up", [0, 4]), 1)
        with self.assertRaises(ValueError):
            g.resolve_relative_nodelabel("nonexistent", [0])

    def test_serialization_roundtrip_preserves_structure(self):
        for g in (line(4),
                  QubitGraph.common_graph(9, "grid", directed=False),
                  QubitGraph.common_graph(4, "line", directed=True, all_directions=True)):
            h = QubitGraph.from_nice_serialization(g.to_nice_serialization())
            self.assertEqual(h.node_names, g.node_names)
            self.assertEqual(h.directed, g.directed)
            self.assertEqual(h.directions, g.directions)
            self.assertEqual(h.edges(include_directions=True), g.edges(include_directions=True))
