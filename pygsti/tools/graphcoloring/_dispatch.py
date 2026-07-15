"""Top-level dispatch: a simple networkx-based coloring, and the
name-based switchboard used to pick among all algorithms in this package."""
import numpy as np
import networkx
from typing import List, Optional, Union

from ._common import Vertex, Edge, NeighborMap, Coloring
from ._topology import auto_edge_coloring
from ._vizing import misra_gries_edge_coloring, vizing_edge_coloring, new_bipartite_edge_coloring
from ._sinnamon import sinnamon_2d_minus_1_edge_coloring, sinnamon_euler_color_edge_coloring
from ._line_graph import moser_tardos_edge_coloring, assadi_oct25_edge_coloring


def find_edge_coloring(deg: int, vertices: List[Vertex],
                       edges: List[Edge],
                       neighbors: NeighborMap,
                       seed: Optional[Union[int, np.random.Generator]] = None) -> Coloring:

    # First we need to build a Linegraph rom the edges.
    # We are assuming there is only one copy of each edge
    # or you want to have discrete colors for (u,v) and (v,u).

    rng = np.random.default_rng(seed)
    new_vertices = np.arange(len(edges))

    mygraph = networkx.Graph()

    for i in range(len(new_vertices)):
        for j in range(i + 1, len(new_vertices)):
            uv = edges[i]
            wz = edges[j]
            if uv[0] in wz or uv[1] in wz:
                mygraph.add_edge(i, j)

    # networkx's built-in "random_sequential" strategy draws from the global
    # `random` module, so it cannot be seeded via a numpy Generator. Instead we
    # supply a custom strategy that visits the nodes in an rng-shuffled order,
    # which makes the greedy coloring reproducible from `seed`.
    def _random_sequential(graph, colors):
        nodes = list(graph.nodes())
        return [nodes[i] for i in rng.permutation(len(nodes))]

    verts_in_line_graph_to_color = networkx.algorithms.coloring.greedy_color(mygraph, strategy=_random_sequential)
    # Networkx will always choose the smallest available color.
    # We just can change the order in which they visit the nodes.

    color_to_edges: Coloring = {}
    for i in range(len(new_vertices)):
        color = verts_in_line_graph_to_color[i]
        if color in color_to_edges:
            color_to_edges[color].append(edges[i])
        else:
            color_to_edges[color] = [edges[i]]
    return color_to_edges


def switchboard_find_edge_coloring(
    algorithm_name: str, deg: int, vertices: List[Vertex], edges: List[Edge],
    neighbors: NeighborMap, seed: Optional[Union[int, np.random.Generator]] = None
) -> Coloring:
    """
    Dispatches to different edge coloring algorithms based on the provided name.

    Parameters:
    algorithm_name (str): The name of the algorithm to use ('misra_gries', 'moser_tardos',
        'new_bipartite', 'assadi', 'vizing', 'sinnamon', 'random_euler_color', etc.).
        'sinnamon' is Sinnamon (2019)'s deterministic Greedy-Euler-Color, a
        (2*deg-1)-edge-coloring running in O(m log(deg)) time. 'random_euler_color' is
        Sinnamon (2019)'s randomized Random-Euler-Color, a (deg+1)-edge-coloring running
        in O(m*sqrt(n)) time with high probability -- see `sinnamon_2d_minus_1_edge_coloring`
        and `sinnamon_euler_color_edge_coloring`, respectively, for details.
        The special name 'auto' first checks whether the graph matches one of the canonical
        topologies produced by `ProcessorSpec(geometry=...)` ('line', 'ring', 'grid', 'torus')
        and, if so, uses a cheap closed-form coloring for it; otherwise it falls back to
        `vizing_edge_coloring`. See `detect_topology` and `auto_edge_coloring`.
    deg (int): The maximum degree of the graph, or the number of colors to use.
    vertices (list): A list of vertices in the graph.
    edges (list): A list of edges represented as tuples of vertices.
    neighbors (dict): A dictionary mapping each vertex to its neighboring vertices.
    seed (None, int, or numpy.random.Generator): Seed or generator controlling the
        randomization of the (randomized) algorithms. Passing the same integer seed
        yields reproducible results. Ignored by the deterministic algorithms.

    Returns:
    color_patches (dict): A dictionary mapping each color to a list of edges colored with that color.
    """
    if algorithm_name == "misra_gries":
        return misra_gries_edge_coloring(deg, vertices, edges, neighbors)
    elif algorithm_name == "moser_tardos":
        return moser_tardos_edge_coloring(deg, vertices, edges, neighbors, seed=seed)
    # Add other algorithms here as they are implemented
    elif algorithm_name == "new_bipartite":
        return new_bipartite_edge_coloring(deg, vertices, edges, neighbors, seed=seed)
    elif algorithm_name == "assadi":
        return assadi_oct25_edge_coloring(deg, vertices, edges, neighbors, seed=seed)
    elif algorithm_name == "vizing":
        return vizing_edge_coloring(deg, vertices, edges, neighbors)
    elif algorithm_name == "sinnamon":
        return sinnamon_2d_minus_1_edge_coloring(deg, vertices, edges, neighbors)
    elif algorithm_name == "random_euler_color":
        return sinnamon_euler_color_edge_coloring(deg, vertices, edges, neighbors, seed=seed)
    elif algorithm_name == "auto":
        return auto_edge_coloring(deg, vertices, edges, neighbors)
    else:
        raise ValueError(f"Unknown edge coloring algorithm: {algorithm_name}")
