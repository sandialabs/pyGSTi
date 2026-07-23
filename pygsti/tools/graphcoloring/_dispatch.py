"""Top-level dispatch switchboard used to pick among all algorithms in this package."""
import numpy as np
from typing import List, Optional, Union

from ._common import Vertex, Edge, NeighborMap, Coloring
from ._topology import auto_edge_coloring
from ._vizing import misra_gries_edge_coloring, vizing_edge_coloring
from ._sinnamon import sinnamon_2d_minus_1_edge_coloring, sinnamon_euler_color_edge_coloring


VALID_ALGORITHMS = (
    "misra_gries",
    "vizing",
    "deterministic_euler_color",
    "random_euler_color",
    "auto",
)


def switchboard_find_edge_coloring(
    algorithm_name: str, deg: int, vertices: List[Vertex], edges: List[Edge],
    neighbors: NeighborMap, seed: Optional[Union[int, np.random.Generator]] = None
) -> Coloring:
    """
    Dispatches to different edge coloring algorithms based on the provided name.

    Parameters:
    algorithm_name (str): The name of the algorithm to use ('misra_gries', 'vizing',
        'deterministic_euler_color', 'random_euler_color', 'auto').

        'deterministic_euler_color' is Sinnamon (2019)'s deterministic Greedy-Euler-Color, a
        (2*deg-1)-edge-coloring running in O(m log(deg)) time. 'random_euler_color' is
        Sinnamon (2019)'s randomized Random-Euler-Color, a (deg+1)-edge-coloring running
        in O(m*sqrt(n)) time with high probability -- see `sinnamon_2d_minus_1_edge_coloring`
        and `sinnamon_euler_color_edge_coloring`, respectively, for details.
        The special name 'auto' first checks whether the graph matches one of the canonical
        topologies produced by `ProcessorSpec(geometry=...)` ('line', 'ring', 'grid', 'torus')
        and, if so, uses a cheap closed-form coloring for it; next checks if the graph is
        bipartite, and if so, uses an internal bipartite-optimal randomized coloring (using `seed`);
        otherwise it falls back to `vizing_edge_coloring`. See `detect_topology` and `auto_edge_coloring`.
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
    elif algorithm_name == "vizing":
        return vizing_edge_coloring(deg, vertices, edges, neighbors)
    elif algorithm_name == "deterministic_euler_color":
        return sinnamon_2d_minus_1_edge_coloring(deg, vertices, edges, neighbors)
    elif algorithm_name == "random_euler_color":
        return sinnamon_euler_color_edge_coloring(deg, vertices, edges, neighbors, seed=seed)
    elif algorithm_name == "auto":
        return auto_edge_coloring(deg, vertices, edges, neighbors, seed=seed)
    else:
        raise ValueError(
            f"Unknown edge coloring algorithm: {algorithm_name!r}. "
            f"Valid options are: {VALID_ALGORITHMS}"
        )
