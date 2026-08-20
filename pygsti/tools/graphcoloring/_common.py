#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""Shared types and small utilities used across the edge-coloring submodules."""
from typing import Any, Dict, List, Sequence, Tuple

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


def canonical_edges(edges: Sequence[Edge]) -> List[Edge]:
    """
    Reduce `edges` to one entry per undirected edge, in :func:`order`'s orientation.

    Parameters
    ----------
    edges : sequence of tuple
        Edges as ``(u, v)`` pairs, in either orientation, possibly with repeats.

    Returns
    -------
    list of tuple
        One canonically-oriented tuple per distinct undirected edge.
    """
    # dict-as-ordered-set: dedups while preserving first-encounter order.
    return list({order(u, v): None for u, v in edges})


def find_neighbors(vertices: Sequence[Vertex], edges: Sequence[Edge]) -> NeighborMap:
    """
    Build the symmetric `NeighborMap` that every algorithm in this package takes.
    """
    # dict-as-ordered-set per vertex: dedups repeated/reversed edges while
    # preserving first-seen order.
    neighbors: Dict[Vertex, Dict[Vertex, None]] = {v: {} for v in vertices}
    for u, v in edges:
        neighbors[u][v] = None
        neighbors[v][u] = None
    return {v: list(nbrs) for v, nbrs in neighbors.items()}


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
