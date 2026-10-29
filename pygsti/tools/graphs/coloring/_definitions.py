#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""Coloring-specific types and utilities used across the edge-coloring submodules.

The graph types these build on (`Vertex`, `Edge`, `NeighborMap`) are defined once
in :mod:`pygsti.tools.graphs._common`; import them from there directly.
"""
from typing import Dict, List

from .._common import Edge, Vertex

# Color: an edge (or vertex) color, a non-negative integer.
Color = int

# Coloring: a (possibly partial) proper edge coloring: color -> canonical edges.
Coloring = Dict[Color, List[Edge]]


def _describe_invalid_patch(color: Color, patch: List[Edge]) -> str:
    """
    Explain why `patch` is not a matching, for `check_valid_edge_coloring`'s error message.

    Only called on the failure path, so it can afford to re-scan `patch` to pin down
    which of the three possible faults occurred and which vertices/edges are at fault.
    """
    seen: Dict[frozenset, Edge] = {}
    incident: Dict[Vertex, List[Edge]] = {}
    for pair in patch:
        # Index rather than unpack, matching the caller: a malformed longer tuple should
        # still reach a coloring-specific message instead of an unpacking ValueError.
        edge = (pair[0], pair[1])
        if pair[0] == pair[1]:
            return (f"color {color!r} contains the self-loop {edge!r}. Edge colorings are only "
                    f"defined for simple graphs, which have no self-loops.")
        endpoints = frozenset(edge)  # size 2, since self-loops are rejected above
        if endpoints in seen:
            return (f"color {color!r} contains the edge {seen[endpoints]!r} more than once "
                    f"(repeated as {edge!r}). Each edge must appear exactly once in exactly one "
                    f"color, canonically oriented as (v1, v2) with v1 < v2.")
        seen[endpoints] = edge
        incident.setdefault(edge[0], []).append(edge)
        incident.setdefault(edge[1], []).append(edge)

    clashing = [(v, edges) for v, edges in incident.items() if len(edges) > 1]
    if clashing:
        vertex, edges = clashing[0]
        n_others = len(clashing) - 1
        also = ""
        if n_others == 1:
            also = " (1 other vertex also clashes.)"
        elif n_others > 1:
            also = f" ({n_others} other vertices also clash.)"
        return (f"color {color!r} is not a matching: vertex {vertex!r} is incident to "
                f"{len(edges)} edges of this color, "
                f"{', '.join(repr(e) for e in edges)}. In a proper edge coloring, edges that "
                f"share a vertex must get different colors.{also}")

    # Unreachable: the caller's count test fails only via a self-loop, a repeat, or a
    # shared vertex. Kept so this function is guaranteed to return an explanation.
    return f"color {color!r} is not a matching: {patch!r}."


def check_valid_edge_coloring(color_patches: Coloring, ret_false_on_error: bool = False) -> bool:
    """
    Check that every color class in `color_patches` is a matching.

    Parameters:
    color_patches (dict): A dictionary mapping each color to a list of edges
                          colored with that color. Unlike with edges, the items
                          in color_patches are NOT symmetric [i.e., it only
                          contains (v1, v2) for v1 < v2]
    ret_false_on_error (bool): If True, return False for an invalid coloring instead
                          of raising.

    Returns:
    bool: True if the coloring is proper. False only if it is not proper and
          `ret_false_on_error` is set.

    Raises:
    ValueError: If the coloring is not proper and `ret_false_on_error` is False. The
        message names the offending color, vertex and edges.
    """
    for c, patch in color_patches.items():
        in_patch = set()
        for pair in patch:
            in_patch.add(pair[0])
            in_patch.add(pair[1])
        # Each of the len(patch) edges should contribute 2 distinct, not-yet-seen vertices.
        if len(in_patch) != 2 * len(patch):
            if ret_false_on_error:
                return False
            raise ValueError(f"Invalid edge coloring: {_describe_invalid_patch(c, patch)}")
    return True
