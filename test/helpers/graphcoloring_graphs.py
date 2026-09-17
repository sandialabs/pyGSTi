#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""Graph fixtures and coloring assessors shared by the graph-coloring tests.

Two suites use these: the correctness tests in
``test/unit/tools/test_graphcoloring.py`` and the wall-clock scaling suite in
``test/performance/test_graphcoloring_scaling.py``. They live here rather than
in either one so that a fix to ``assess_coloring`` -- which decides whether a
coloring is *correct* -- cannot land in one suite and not the other.
"""

import numpy as np

from pygsti.tools.graphcoloring import find_neighbors
from pygsti.tools.graphcoloring._dispatch import VALID_ALGORITHMS


# Every edge-coloring algorithm exposed by the switchboard.
ALL_ALGORITHMS = list(VALID_ALGORITHMS)

# Deterministic algorithms that always terminate and produce a proper, complete
# coloring with at most deg+1 colors (Vizing's theorem) on every family tested.
DETERMINISTIC_EXACT_ALGORITHMS = ["vizing", "misra_gries"]

# DETERMINISTIC_DP1PP_ALGORITHMS: algorithms that are already deterministic (their
# output does not vary with the seed) but do not ensure deg+1 colors or better.
# One may use them if they want a coloring scheme which is faster than one of the
# minimum coloring deterministic schemes.
DETERMINISTIC_DP1PP_ALGORITHMS = ["deterministic_euler_color"]

# RANDOMIZED_ALGORITHMS: genuinely randomized algorithms, regardless of their cap
# on the number of colors returned. Their output can (and generally will) vary
# with the seed.
RANDOMIZED_ALGORITHMS = ["random_euler_color"]

# "SPARSE_SAFE" = algorithms that reliably produced a proper & complete coloring
# on the low-degree (deg<=4) families in this suite, across runs and within the
# per-algorithm timeout. `vizing`, `misra_gries`, and `auto` qualify (the latter
# is deterministic and optimal on standard topologies, and falls back to vizing
# elsewhere; all are in fact reliable on *dense* graphs too).
SPARSE_SAFE = ["vizing", "misra_gries", "auto"]


# ---------------------------------------------------------------------------
# Graph generators
# ---------------------------------------------------------------------------
def _finalize(vertices, edges):
    """Build the (vertices, edges, neighbors, deg) 4-tuple the API expects.

    ``edges`` is a list of undirected ``(u, v)`` pairs (each listed once).
    Adjacency comes from `find_neighbors` rather than a hand-rolled copy, so
    these fixtures cannot drift from the symmetric map the API expects.
    """
    vertices = list(vertices)
    neighbors = find_neighbors(vertices, edges)
    deg = max((len(neighbors[v]) for v in vertices), default=0)
    return vertices, list(edges), neighbors, deg


def make_cycle_graph(n):
    """Cycle C_n: n vertices, n edges, max degree 2."""
    return _finalize(range(n), [(i, (i + 1) % n) for i in range(n)])


def make_path_graph(n):
    """Path P_n: n vertices, n-1 edges, max degree 2."""
    return _finalize(range(n), [(i, i + 1) for i in range(n - 1)])


def make_complete_graph(n):
    """Complete graph K_n: max degree n-1, the hardest dense case."""
    return _finalize(range(n), [(i, j) for i in range(n) for j in range(i + 1, n)])


def make_grid_graph(rows, cols):
    """2D lattice (rows x cols), max degree 4 -- models a planar QPU layout."""
    def idx(r, c):
        return r * cols + c
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                edges.append((idx(r, c), idx(r + 1, c)))
    return _finalize(range(rows * cols), edges)


def make_torus_graph(s):
    """s x s torus (grid with row/column wraparound), max degree 4 for s > 2.

    Mirrors QubitGraph.common_graph's "torus" construction: at s == 2 no
    wraparound edges are added (they would duplicate the plain grid edges),
    so a 2x2 "torus" is identical to a 2x2 grid.
    """
    def idx(r, c):
        return r * s + c
    edges = []
    for r in range(s):
        for c in range(s):
            if c + 1 < s:
                edges.append((idx(r, c), idx(r, c + 1)))
            elif s > 2:
                edges.append((idx(r, c), idx(r, 0)))
            if r + 1 < s:
                edges.append((idx(r, c), idx(r + 1, c)))
            elif s > 2:
                edges.append((idx(r, c), idx(0, c)))
    return _finalize(range(s * s), edges)


def make_tee_graph():
    """A 4-vertex "T": vertex 1 is a degree-3 hub joined to 0, 2 and 3.

    The smallest graph `detect_topology` calls "unknown", so "auto" must take a
    real coloring path rather than a closed form that ignores `deg`. See
    `TeeOrientationInvarianceTester`.
    """
    return _finalize(range(4), [(0, 1), (1, 2), (1, 3)])


def make_high_degree_graph():
    """A small graph with a single high-degree hub (max degree 5)."""
    edges = [(0, i) for i in range(1, 6)] + [(1, 6), (2, 7), (3, 8), (4, 9)]
    return _finalize(range(10), edges)


def make_random_regular_graph(n, d, seed):
    """(Approximately) d-regular random graph on n vertices via a pairing model.

    Falls back to whatever simple graph the pairing produces; the returned
    ``deg`` is the realized maximum degree, which is <= d.
    """
    rng = np.random.default_rng(seed)
    stubs = []
    for v in range(n):
        stubs.extend([v] * d)
    edge_set = set()
    for _ in range(10):
        rng.shuffle(stubs)
        edge_set.clear()
        ok = True
        for i in range(0, len(stubs) - 1, 2):
            u, w = stubs[i], stubs[i + 1]
            if u == w or (min(u, w), max(u, w)) in edge_set:
                ok = False
                break
            edge_set.add((min(u, w), max(u, w)))
        if ok:
            break
    return _finalize(range(n), sorted(edge_set))


# ---------------------------------------------------------------------------
# Coloring-quality helpers (stronger than check_valid_edge_coloring, which only
# verifies each color class is a matching -- it does NOT verify completeness).
# ---------------------------------------------------------------------------
def assess_coloring(color_patches, edges):
    """Return (is_proper, is_complete, num_colors) for a coloring.

    * proper   : no edge appears twice AND no two edges sharing a vertex share a color
    * complete : every input edge received exactly one color
    """
    all_edges = {tuple(sorted(e)) for e in edges}
    seen = set()
    is_proper = True
    for _color, patch in color_patches.items():
        touched = set()
        for u, v in patch:
            e = tuple(sorted((u, v)))
            if e in seen:
                is_proper = False  # edge colored more than once
            seen.add(e)
            if u in touched or v in touched:
                is_proper = False  # adjacent edges share this color
            touched.add(u)
            touched.add(v)
    is_complete = (seen == all_edges)
    return is_proper, is_complete, len(color_patches)


def canonical_coloring(color_patches):
    """A hashable, order-independent representation of a coloring for equality.

    Two colorings compare equal iff they assign the same set of edges to the
    same set of colors (edge order within a color and dict order are ignored).
    """
    return tuple(sorted(
        (color, tuple(sorted(tuple(sorted(e)) for e in patch)))
        for color, patch in color_patches.items() if patch
    ))
