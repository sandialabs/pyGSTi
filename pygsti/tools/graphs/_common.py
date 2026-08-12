#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""Shared types and small utilities used across the graphs subpackage."""
from typing import Any, Dict, List, Sequence, Tuple

# Vertex: usually an int (qubit index) or string-like qubit label. Only needs
# to support equality/hashing (as a dict key) and, for `order`, `</>`.
Vertex = Any

# Edge: an undirected edge as a 2-tuple of vertices. Functions in this
# package generally expect/produce *canonical* edges (v1 <= v2, see `order`);
# where a function instead wants a symmetric edge list (both (u,v) and (v,u)
# present), that's called out in its docstring.
Edge = Tuple[Vertex, Vertex]

# NeighborMap: vertex -> list of neighboring vertices.
NeighborMap = Dict[Vertex, List[Vertex]]


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


def max_degree(neighbors: NeighborMap) -> int:
    """
    Return the maximum degree of the graph described by `neighbors`.
    """
    return max((len(nbrs) for nbrs in neighbors.values()), default=0)
