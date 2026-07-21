"""
Topology detection and closed-form (optimal) edge colorings for the
canonical topologies produced by `ProcessorSpec(geometry=...)` /
`QubitGraph.common_graph`: "line", "ring", "grid", "torus". Falls back to
bipartite-optimal coloring for arbitrary bipartite graphs, and to
`vizing_edge_coloring` for anything else (see `auto_edge_coloring`).
"""
import numpy as np
import networkx
from typing import List, Optional, Union

from ._common import Vertex, Edge, NeighborMap, Coloring, order
from ._vizing import vizing_edge_coloring, _NewBipartiteEdgeColoring


def _is_bipartite(vertices: List[Vertex], edges: List[Edge]) -> bool:
    """Check if the graph is bipartite using networkx."""
    G = networkx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)
    return networkx.is_bipartite(G)


def detect_topology(vertices: List[Vertex], edges: List[Edge], neighbors: NeighborMap) -> str:
    """
    Detect whether a graph matches one of the canonical topologies produced by
    `QubitGraph.common_graph` / `ProcessorSpec(geometry=...)`: "line", "ring",
    "grid", or "torus".

    This is a purely structural check that assumes the position of each vertex
    in `vertices` corresponds to its position in the topology's canonical
    construction order (i.e., sequential position along a line/ring, or
    row-major position in a grid/torus) -- matching the vertex ordering
    produced by `ProcessorSpec.qubit_labels` / `compute_2Q_connectivity()`.

    Detection requires an *exact* match to the canonical edge set: if any
    edge is missing or extra (e.g. some 2-qubit gates unavailable on certain
    edges), this returns "unknown" rather than a partial/subgraph match.

    Ambiguity note: at n=2, "ring" and "line" produce identical edge sets, so
    this returns "line"; similarly at s=2, "torus" and "grid" are identical,
    so this returns "grid".

    Parameters:
    vertices (list): A list of vertices, in canonical construction order.
    edges (list): A list of edges represented as tuples of vertices (may
        include both (u,v) and (v,u); only the undirected structure matters).
    neighbors (dict): Unused; accepted for interface consistency with the
        other switchboard-family functions.

    Returns:
    str: One of "line", "ring", "grid", "torus", or "unknown".
    """
    n = len(vertices)
    unique_edges = {order(u, v) for u, v in edges}

    # --- line / ring ---
    if n >= 2:
        line_edges = {order(vertices[i], vertices[i + 1]) for i in range(n - 1)}
        if n > 2:
            ring_edges = line_edges | {order(vertices[n - 1], vertices[0])}
            if unique_edges == ring_edges:
                return "ring"
        if unique_edges == line_edges:
            return "line"

    # --- grid / torus ---
    s = int(round(np.sqrt(n)))
    if n >= 4 and s * s == n:
        grid_edges = set()
        torus_extra_edges = set()
        for irow in range(s):
            for icol in range(s):
                pos = vertices[irow * s + icol]
                if icol + 1 < s:
                    grid_edges.add(order(pos, vertices[irow * s + icol + 1]))
                elif s > 2:
                    torus_extra_edges.add(order(pos, vertices[irow * s + 0]))
                if irow + 1 < s:
                    grid_edges.add(order(pos, vertices[(irow + 1) * s + icol]))
                elif s > 2:
                    torus_extra_edges.add(order(pos, vertices[0 * s + icol]))
        if torus_extra_edges and unique_edges == (grid_edges | torus_extra_edges):
            return "torus"
        if unique_edges == grid_edges:
            return "grid"

    return "unknown"


class _ClosedFormEdgeColoring:
    """Shared helper for the closed-form (topology-specific) colorers below."""

    @staticmethod
    def _drop_empty_colors(color_patches: Coloring) -> Coloring:
        """Strip any color class that ended up unused, so the returned
        coloring reports the true (optimal) number of colors needed."""
        return {c: v for c, v in color_patches.items() if v}


class _LineRingEdgeColoring(_ClosedFormEdgeColoring):
    """
    Closed-form edge coloring for a line or ring of `len(vertices)` vertices
    laid out in canonical (sequential) order.

    - line: alternates colors 0/1 along the chain; at most 2 colors
      (optimal: max degree <= 2).
    - ring, even length: alternates 0/1 all the way around (including the
      wraparound edge); optimal (2 colors) since an even cycle is
      2-edge-colorable.
    - ring, odd length: alternates 0/1 along the first n-1 edges and colors
      the final wraparound edge 2 -- an odd cycle's true chromatic index
      (it cannot be properly 2-colored).
    """

    def __init__(self, vertices: List[Vertex], topology: str) -> None:
        self.vertices = vertices
        self.topology = topology

    def color(self) -> Coloring:
        vertices, topology = self.vertices, self.topology
        n = len(vertices)
        is_ring = (topology == "ring" and n > 2)
        odd_ring = is_ring and (n % 2 == 1)
        num_edges = n if is_ring else max(n - 1, 0)

        color_patches: Coloring = {0: [], 1: []}
        if odd_ring:
            color_patches[2] = []

        for i in range(num_edges):
            edge = order(vertices[i], vertices[(i + 1) % n])
            color = 2 if (odd_ring and i == num_edges - 1) else (i % 2)
            color_patches[color].append(edge)

        return self._drop_empty_colors(color_patches)


class _GridTorusEdgeColoring(_ClosedFormEdgeColoring):
    """
    Closed-form edge coloring for a grid (s x s mesh) or an even-`s` torus
    (s x s mesh with row/column wraparound), laid out in canonical row-major
    order.

    Horizontal edges are colored by column parity (colors 0/1); vertical
    edges by row parity (colors 2/3). Always valid and uses exactly 4 colors
    for a plain grid s>=2 (optimal: grid graphs are bipartite with max
    degree 4), and also for an even-`s` torus (each row/column wraparound is
    itself an even cycle, so the alternating pattern stays consistent
    through it).

    Not valid for an odd-`s` torus (each wraparound is then an odd cycle,
    breaking the parity alternation) -- callers must not use this for that
    case; see `auto_edge_coloring`.
    """

    def __init__(self, vertices: List[Vertex], s: int, topology: str) -> None:
        self.vertices = vertices
        self.s = s
        self.topology = topology

    def color(self) -> Coloring:
        vertices, s = self.vertices, self.s
        is_torus = (self.topology == "torus")
        color_patches: Coloring = {0: [], 1: [], 2: [], 3: []}

        def pos(irow, icol):
            return vertices[irow * s + icol]

        for irow in range(s):
            for icol in range(s):
                u = pos(irow, icol)
                if icol + 1 < s:
                    color_patches[icol % 2].append(order(u, pos(irow, icol + 1)))
                elif is_torus and s > 2:
                    color_patches[icol % 2].append(order(u, pos(irow, 0)))
                if irow + 1 < s:
                    color_patches[2 + (irow % 2)].append(order(u, pos(irow + 1, icol)))
                elif is_torus and s > 2:
                    color_patches[2 + (irow % 2)].append(order(u, pos(0, icol)))

        return self._drop_empty_colors(color_patches)


def auto_edge_coloring(deg: int, vertices: List[Vertex], edges: List[Edge],
                       neighbors: NeighborMap,
                       seed: Optional[Union[int, np.random.Generator]] = None) -> Coloring:
    """
    Detects whether the input graph matches one of the canonical topologies
    produced by `ProcessorSpec(geometry=...)` / `QubitGraph.common_graph`
    ("line", "ring", "grid", "torus") via `detect_topology`, and if so, uses a
    cheap closed-form coloring that is always valid and uses the optimal
    number of colors (the true chromatic index) for that topology.

    If not, it checks whether the graph is bipartite; if so, it uses an internal
    bipartite-optimal randomized coloring (using `seed`) that frequently achieves
    the optimal `deg` colors in practice.

    Otherwise, it falls back to `vizing_edge_coloring` -- a deterministic algorithm that is
    always valid and complete, using at most deg+1 colors -- for graphs that
    don't match one of these canonical topologies, and for odd-side-length
    tori (where the simple closed-form pattern is not valid; see
    `_GridTorusEdgeColoring` for details).

    Parameters:
    deg (int): The maximum degree of the graph (used only by the fallbacks).
    vertices (list): A list of vertices in the graph, in the same order used
        to originally construct the graph (see `detect_topology`).
    edges (list): A list of edges represented as tuples of vertices.
    neighbors (dict): A dictionary mapping each vertex to its neighboring
        vertices.
    seed (None, int, or numpy.random.Generator): Seed or generator controlling the
        randomization of the bipartite fallback. Ignored otherwise.

    Returns:
    color_patches (dict): A dictionary mapping each color to a list of edges
        colored with that color.
    """
    topology = detect_topology(vertices, edges, neighbors)

    if topology in ("line", "ring"):
        return _LineRingEdgeColoring(vertices, topology).color()

    if topology in ("grid", "torus"):
        s = int(round(np.sqrt(len(vertices))))
        if topology == "grid" or s % 2 == 0:
            return _GridTorusEdgeColoring(vertices, s, topology).color()

    if _is_bipartite(vertices, edges):
        return _NewBipartiteEdgeColoring(deg, edges, neighbors, seed=seed).color()

    return vizing_edge_coloring(deg, vertices, edges, neighbors)
