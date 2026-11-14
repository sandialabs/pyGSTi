"""
Tests of `QubitGraph.edges`, whose implementation the crosstalk-free GST work changed (it is on the path of
building lane models for large processors).
"""
from pygsti.baseobjs.qubitgraph import QubitGraph


def test_edges_of_a_lattice():
    rows, cols = 6, 7
    edges = [(r * cols + c, r * cols + c + 1) for r in range(rows) for c in range(cols - 1)]
    edges += [(r * cols + c, (r + 1) * cols + c) for r in range(rows - 1) for c in range(cols)]
    for directed in (False, True):
        g = QubitGraph(list(range(rows * cols)), initial_edges=edges, directed=directed)
        assert g.edges() == sorted(edges)
        doubled = sorted(edges) if directed else sorted(edges + [(b, a) for a, b in edges])
        assert g.edges(double_for_undirected=True) == doubled


def test_edges_with_directions():
    g = QubitGraph.common_graph(36, 'grid', directed=True, all_directions=True)
    expected = [(a, a + 1, 'right') for a in range(36) if a % 6 < 5]
    expected += [(a + 1, a, 'left') for a in range(36) if a % 6 < 5]
    expected += [(a, a + 6, 'down') for a in range(30)] + [(a + 6, a, 'up') for a in range(30)]
    assert g.edges(include_directions=True) == sorted(expected)
    assert g.edges() == sorted((a, b) for a, b, _ in expected)


def test_edges_use_node_labels():
    # An undirected edge is listed from the node that comes first in the graph's node order.
    g = QubitGraph(['c', 'a', 'b'], initial_edges=[('c', 'b'), ('a', 'c')], directed=False)
    assert g.edges() == [('c', 'a'), ('c', 'b')]
    assert g.edges(double_for_undirected=True) == [('a', 'c'), ('b', 'c'), ('c', 'a'), ('c', 'b')]
    g = QubitGraph(['c', 'a', 'b'], initial_edges=[('c', 'b'), ('a', 'c')], directed=True)
    assert g.edges() == [('a', 'c'), ('c', 'b')]
    assert g.edges(double_for_undirected=True) == [('a', 'c'), ('c', 'b')]
