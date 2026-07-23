"""Shared types and small utilities used across the edge-coloring submodules."""
from typing import List, Dict, Tuple, Any

# Vertex: usually an int (qubit index) or string-like qubit label. Only needs
# to support equality/hashing (as a dict key) and, for `order`, `</>`.
Vertex = Any

# Color: an edge (or vertex) color, a non-negative integer.
Color = int

# Edge: an undirected edge as a 2-tuple of vertices. Functions in this
# package generally expect/produce *canonical* edges (v1 <= v2, see `order`);
# where a function instead wants a symmetric edge list (both (u,v) and (v,u)
# present), that's called out in its docstring.
Edge = Tuple[Vertex, Vertex]

# NeighborMap: vertex -> list of neighboring vertices.
NeighborMap = Dict[Vertex, List[Vertex]]

# Coloring: a (possibly partial) proper edge coloring: color -> canonical edges.
Coloring = Dict[Color, List[Edge]]


def order(u: Vertex, v: Vertex) -> Edge:
    """Return (u, v) sorted so the smaller vertex comes first."""
    return (min(u, v), max(u, v))


def check_valid_edge_coloring(color_patches: Coloring, ret_false_on_error: bool = False) -> bool:
    """
    color_patches (dict): A dictionary mapping each color to a list of edges
                          colored with that color. Unlike with edges, the items
                          in color_patches are NOT symmetric [i.e., it only
                          contains (v1, v2) for v1 < v2]
    """
    for c, patch in color_patches.items():
        in_patch = set()
        for pair in patch:
            in_patch.add(pair[0])
            in_patch.add(pair[1])
        if len(in_patch) != 2 * len(patch):
            if ret_false_on_error:
                return False
            raise ValueError()
    return True
