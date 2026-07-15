"""
The "Vizing family" of deg+1-edge-coloring algorithms: Misra-Gries, Vizing,
and the randomized "new_bipartite" variant. All three share one core
primitive -- a single Vizing-chain recoloring step -- via `_VizingChainState`
and its `_VizingFamilyEdgeColoring` template subclass.

References:
    V. G. Vizing, "On an estimate of the chromatic class of a p-graph,"
    Diskret. Analiz, vol. 3, pp. 25-30, 1964 (in Russian; no stable DOI/URL
    -- the original chromatic-index theorem this module's "Vizing" and
    "Misra-Gries" algorithms both constructively prove).

    J. Misra and D. Gries, "A constructive proof of Vizing's theorem,"
    Information Processing Letters, vol. 41, no. 3, pp. 131-133, 1992.
    https://doi.org/10.1016/0020-0190(92)90041-S
"""
import numpy as np
from typing import List, Dict, Set, Optional, Union

from ._common import Vertex, Color, Edge, NeighborMap, Coloring, order


class _VizingChainState:
    """
    Shared bookkeeping and primitives for the "Vizing chain" edge-recoloring
    step (fan construction, alternating c/d-path finding, free-color
    queries) used by `_VizingFamilyEdgeColoring` and its subclasses.

    Every query (`is_free`, `missing_colors`, `first_free`) is derived on
    demand from `edge_colors` -- the single source of truth -- rather than a
    separately-tracked "free colors per vertex" cache that must be kept in
    sync by hand (which is prone to state-drift bugs).
    """

    def __init__(self, deg: int, neighbors: NeighborMap,
                 edges: Optional[List[Edge]] = None) -> None:
        self.deg = deg
        self.neighbors = neighbors
        self.all_colors: List[Color] = list(range(deg + 1))
        self.color_patches: Coloring = {c: [] for c in self.all_colors}
        self.edge_colors: Dict[Edge, Color] = {}
        if edges is not None:
            for u, v in edges:
                self.edge_colors[order(u, v)] = -1

    def is_free(self, color: Color, vertex: Vertex) -> bool:
        """True iff `color` is not used by any colored edge incident to `vertex`."""
        for nb in self.neighbors.get(vertex, []):
            if self.edge_colors.get(order(vertex, nb), -1) == color:
                return False
        return True

    def missing_colors(self, vertex: Vertex) -> Set[Color]:
        """The set of colors free (missing) at `vertex`."""
        used = {self.edge_colors.get(order(vertex, nb), -1)
                for nb in self.neighbors.get(vertex, [])}
        used.discard(-1)
        return set(self.all_colors) - used

    def first_free(self, vertex: Vertex) -> Color:
        """The smallest color free on `vertex` (guaranteed to exist for a
        graph of maximum degree <= deg)."""
        for color in self.all_colors:
            if self.is_free(color, vertex):
                return color
        raise RuntimeError("no free color available; deg+1 colors should suffice")

    def set_edge_color(self, a: Vertex, b: Vertex, color: Color) -> None:
        """Set the color of edge (a, b), keeping `color_patches` in sync."""
        e = order(a, b)
        old = self.edge_colors.get(e, -1)
        if old == color:
            return
        if old != -1 and old in self.color_patches and e in self.color_patches[old]:
            self.color_patches[old].remove(e)
        self.edge_colors[e] = color
        if color != -1:
            self.color_patches.setdefault(color, [])
            if e not in self.color_patches[color]:
                self.color_patches[color].append(e)

    def build_fan(self, u: Vertex, v: Vertex) -> List[Vertex]:
        """Build a maximal fan of `u` starting at `v`.

        A fan F = [v = f0, f1, ..., fk] is a sequence of distinct neighbors of
        `u` such that (u, f0) is uncolored and, for each i >= 1, the color of
        (u, f_i) is free on f_{i-1}. This is the standard Misra-Gries fan.
        """
        fan = [v]
        in_fan = {v}
        extended = True
        while extended:
            extended = False
            last = fan[-1]
            for w in self.neighbors.get(u, []):
                if w in in_fan:
                    continue
                c = self.edge_colors.get(order(u, w), -1)
                # (u, w) must be colored, and its color must be free on `last`.
                if c != -1 and self.is_free(c, last):
                    fan.append(w)
                    in_fan.add(w)
                    extended = True
                    break
        return fan

    def cd_path(self, start: Vertex, c: Color, d: Color) -> List[Edge]:
        """Return the edges (as ordered tuples) of the maximal c/d-alternating
        path starting at `start`. The first edge taken has color `d` (in every
        algorithm below, `c` is chosen to be free on `start`)."""
        path = []
        current = start
        look_for = d
        visited = set()
        while True:
            nxt = None
            for w in self.neighbors.get(current, []):
                e = order(current, w)
                if e in visited:
                    continue
                if self.edge_colors.get(e, -1) == look_for:
                    nxt = w
                    path.append(e)
                    visited.add(e)
                    break
            if nxt is None:
                break
            current = nxt
            look_for = c if look_for == d else d
        return path

    def color_edge_vizing_chain(self, u: Vertex, v: Vertex) -> None:
        """Color the single (currently uncolored) edge (u, v) using one
        Vizing-chain step: build a maximal fan of `u` starting at `v`, invert
        a c/d-alternating (Kempe) chain through `u` so that a color is freed,
        then rotate the fan to complete the coloring. This is the Misra &
        Gries (1992) constructive proof of Vizing's theorem, and always
        succeeds using at most deg+1 colors.
        """
        # 1) Maximal fan of u starting at v.
        fan = self.build_fan(u, v)
        k = fan[-1]

        # 2) c free on u, d free on the last fan vertex k.
        c = self.first_free(u)
        d = self.first_free(k)

        # 3) If c != d, invert the cd_u path (the maximal c/d-alternating path
        #    through u) so that d becomes free on u. When c == d, d is already
        #    free on u and there is nothing to invert.
        if c != d:
            path = self.cd_path(u, c, d)
            for e in path:
                # swap each edge between c and d based on its actual current color.
                actual = self.edge_colors.get(e, -1)
                self.set_edge_color(e[0], e[1], c if actual == d else d)

        # 4) Choose w: the largest fan index such that F[0..w_index] is still
        #    a valid fan and d is free on F[w_index]. The Misra-Gries
        #    correctness proof guarantees such a w exists after the inversion
        #    (it is either k, or the index i for which (u, F[i+1]) was the
        #    unique d-colored fan edge). We validate the fan prefix explicitly
        #    against the *current* edge colors so the choice is correct even
        #    though `fan` was built before inversion.
        def prefix_is_fan(idx):
            # F[0..idx] is a fan iff for each 0 <= j < idx the color of
            # (u, F[j+1]) is free on F[j] (edge (u, F[0]) is the uncolored
            # target).
            for j in range(idx):
                col = self.edge_colors.get(order(u, fan[j + 1]), -1)
                if col == -1 or not self.is_free(col, fan[j]):
                    return False
            return True

        w_index = 0
        for i in range(len(fan)):
            if self.is_free(d, fan[i]) and prefix_is_fan(i):
                w_index = i
        w = fan[w_index]

        # 5) Rotate the subfan F[0..w_index]: shift each (u, F[i+1]) color
        #    down to (u, F[i]), then (u, F[w_index]) becomes uncolored.
        for i in range(w_index):
            nxt_color = self.edge_colors.get(order(u, fan[i + 1]), -1)
            self.set_edge_color(u, fan[i], nxt_color)
        if w_index > 0:
            self.set_edge_color(u, w, -1)

        # 6) Color (u, w) with d.
        self.set_edge_color(u, w, d)


class _VizingFamilyEdgeColoring(_VizingChainState):
    """
    Template for the "Vizing family": process every (canonicalized,
    deduplicated) edge exactly once, in some order, optionally trying a
    cheap direct-assignment fast path (`_try_direct`) before falling back to
    one full Vizing-chain step (`color_edge_vizing_chain`, inherited).
    Subclasses differ only in edge ordering (`_ordered_edges`) and in
    whether/how `_try_direct` is implemented -- both hooks default here to
    "no fast path, edges in the order given", i.e. plain Misra-Gries.
    """

    def __init__(self, deg: int, edges: List[Edge], neighbors: NeighborMap) -> None:
        self.unique_edges = list({order(u, v) for u, v in edges})
        super().__init__(deg, neighbors, self.unique_edges)

    def _try_direct(self, u: Vertex, v: Vertex) -> bool:
        """Attempt to color the (currently uncolored) edge (u, v) directly,
        without a full Vizing-chain step. Returns True iff it succeeded.
        The base implementation never takes this fast path."""
        return False

    def _ordered_edges(self) -> List[Edge]:
        """The order in which to process `self.unique_edges`. Defaults to
        the order given; override (e.g.) to randomize."""
        return self.unique_edges

    def color(self) -> Coloring:
        for u, v in self._ordered_edges():
            if not self._try_direct(u, v):
                self.color_edge_vizing_chain(u, v)
        return self.color_patches


class _MisraGriesEdgeColoring(_VizingFamilyEdgeColoring):
    """Misra & Gries' (1992) algorithm: every edge via a full Vizing-chain
    step. Uses the base class's default (no fast path, given order)."""


def misra_gries_edge_coloring(
    deg: int, vertices: List[Vertex], edges: List[Edge], neighbors: NeighborMap
) -> Coloring:
    """
    Implements Misra & Gries' edge coloring algorithm for a simple undirected graph.

    This is a deterministic algorithm that always terminates and produces a
    proper edge coloring using at most deg+1 colors (Vizing's theorem). See
    `_MisraGriesEdgeColoring` for the implementation.
    """
    return _MisraGriesEdgeColoring(deg, edges, neighbors).color()


class _VizingEdgeColoring(_VizingFamilyEdgeColoring):
    """
    Vizing's edge-coloring algorithm (constructive proof of Vizing's
    theorem): colors any simple graph with at most deg+1 colors.

    Simple case (`_try_direct`): a color free at both endpoints is used
    directly. Complex case: a Vizing-chain step (inherited), shared
    verbatim with `_MisraGriesEdgeColoring` via the common
    `_VizingFamilyEdgeColoring` base -- this class differs only in the
    greedy simple-case fast path below.
    """

    def _try_direct(self, u: Vertex, v: Vertex) -> bool:
        # Simple case: if a free color at u is also free at v.
        for color in self.missing_colors(u):
            if self.is_free(color, v):
                self.set_edge_color(u, v, color)
                return True
        return False


def vizing_edge_coloring(
    deg: int, vertices: List[Vertex], edges: List[Edge], neighbors: NeighborMap
) -> Coloring:
    """
    Vizing's edge-coloring algorithm (constructive proof of Vizing's theorem):
    colors any simple graph with at most deg+1 colors. See
    `_VizingEdgeColoring` for the implementation.
    """
    return _VizingEdgeColoring(deg, edges, neighbors).color()


class _NewBipartiteEdgeColoring(_VizingFamilyEdgeColoring):
    """
    Randomised (Delta+1)-edge coloring that always returns a complete, valid
    coloring. Edges are processed in a random order (`_ordered_edges`); for
    each edge, Case 1 (`_try_direct`) assigns a color missing at both
    endpoints directly, else Case 2 falls back to a Vizing-chain step
    (inherited).

    On bipartite graphs the random ordering lets Case 1 fire often, so the
    result frequently uses only deg colors (the bipartite optimum); every
    simple graph is still guaranteed a valid coloring.

    Case 1 here is the same fast path as `_VizingEdgeColoring._try_direct`
    (just via set-intersection instead of a scan) -- this class differs from
    `_VizingEdgeColoring` only in `_ordered_edges` (randomized vs. identity).
    """

    def __init__(self, deg: int, edges: List[Edge], neighbors: NeighborMap,
                 seed: Optional[Union[int, np.random.Generator]] = None) -> None:
        super().__init__(deg, edges, neighbors)
        self.rng = np.random.default_rng(seed)

    def _ordered_edges(self) -> List[Edge]:
        return [self.unique_edges[i] for i in self.rng.permutation(len(self.unique_edges))]

    def _try_direct(self, u: Vertex, v: Vertex) -> bool:
        common = self.missing_colors(u) & self.missing_colors(v)
        if common:
            self.set_edge_color(u, v, next(iter(common)))
            return True
        return False


def new_bipartite_edge_coloring(
    deg: int, vertices: List[Vertex], edges: List[Edge], neighbors: NeighborMap,
    seed: Optional[Union[int, np.random.Generator]] = None
) -> Coloring:
    """
    Randomised (Delta+1)-edge coloring that always returns a complete, valid
    coloring; see `_NewBipartiteEdgeColoring` for the implementation.

    Args:
        deg (int): The maximum degree of the graph.
        vertices (list): A list of vertices in the graph.
        edges (list): A list of edges represented as tuples of vertices
            (assumed to be symmetric, i.e., (u,v) and (v,u) are elements).
        neighbors (dict): A dictionary mapping each vertex to its neighboring vertices.
        seed (None, int, or numpy.random.Generator): Seed or generator controlling the
            randomization. Passing the same integer seed yields reproducible results.

    Returns:
        dict: A dictionary mapping each color to a list of edges colored with that color.
              Edges are represented as (v1, v2) where v1 < v2.  Every edge in the
              input appears in exactly one color class (complete coloring).
    """
    return _NewBipartiteEdgeColoring(deg, edges, neighbors, seed=seed).color()
