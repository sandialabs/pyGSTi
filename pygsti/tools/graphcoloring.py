import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Union, Any
import networkx

# --- Shared type aliases -----------------------------------------------
#
# Vertex: a graph vertex. In practice this is usually an int (e.g. a bare
# qubit index) or a string-like qubit label (e.g. ProcessorSpec.qubit_labels /
# QubitGraph node names). Functions in this module never rely on vertices
# supporting arithmetic -- only equality/hashing (as dict keys), and, for the
# topology-detection helpers, list position -- except for `order`, which also
# requires vertices to support `<`/`>` (true of both ints and strings, but not
# guaranteed by hashability alone). Aliased to `Any` rather than `Hashable`
# so that comparison in `order` type-checks without overclaiming a contract
# the rest of the module doesn't actually need.
Vertex = Any

# Color: an edge (or vertex) color, represented as a non-negative integer.
Color = int

# Edge: an undirected edge, represented as a 2-tuple of vertices. Many
# functions in this module expect (or produce) *canonical* edges, i.e. with
# `v1 <= v2` under the default ordering -- see `order`. Where a function
# instead accepts a symmetric edge list (both (u, v) and (v, u) present),
# that is called out explicitly in its docstring.
Edge = Tuple[Vertex, Vertex]

# NeighborMap: maps each vertex to a list of its neighboring vertices.
NeighborMap = Dict[Vertex, List[Vertex]]

# Coloring: a (possibly partial) proper edge coloring, mapping each color to
# the list of canonical edges assigned that color.
Coloring = Dict[Color, List[Edge]]


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
    algorithm_name (str): The name of the algorithm to use ('misra_gries', 'moser_tardos', etc.).
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
    elif algorithm_name == "auto":
        return auto_edge_coloring(deg, vertices, edges, neighbors)
    else:
        raise ValueError(f"Unknown edge coloring algorithm: {algorithm_name}")


def detect_topology(vertices: List[Vertex], edges: List[Edge], neighbors: NeighborMap) -> str:
    """
    Detect whether a graph matches one of the canonical topologies produced by
    `QubitGraph.common_graph` / `ProcessorSpec(geometry=...)`: "line", "ring",
    "grid", or "torus".

    This is a purely structural check that assumes the position of each vertex
    in `vertices` corresponds to its position in the topology's canonical
    construction order (i.e., sequential position along a line/ring, or
    row-major position in a grid/torus). This matches the vertex ordering
    produced by `ProcessorSpec.qubit_labels` / `compute_2Q_connectivity()`,
    which preserve the qubit-label order used at construction time.

    Detection requires an *exact* match to the canonical edge set: if any edge
    is missing or extra relative to the canonical topology (e.g. because some
    2-qubit gates are unavailable on certain edges), this returns "unknown"
    rather than attempting a partial/subgraph match.

    Note on inherent ambiguity: at n=2, "ring" and "line" produce identical
    edge sets (no distinct wraparound edge is possible there), so this
    function returns "line". Similarly, at s=2 (a 2x2 grid), "torus" and
    "grid" produce identical edge sets (no wraparound edges are added at that
    size), so this function returns "grid".

    Parameters:
    vertices (list): A list of vertices, in canonical construction order.
    edges (list): A list of edges represented as tuples of vertices (may
        include both (u,v) and (v,u); only the undirected structure matters).
    neighbors (dict): A dictionary mapping each vertex to its neighboring
        vertices. Unused by this function; accepted for interface consistency
        with the other switchboard-family functions.

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


def _line_ring_closed_form_coloring(vertices: List[Vertex], topology: str) -> Coloring:
    """
    Closed-form edge coloring for a line or ring of `len(vertices)` vertices
    laid out in canonical (sequential) order.

    - line: alternates colors 0/1 along the chain; always uses at most 2
      colors (optimal, since the max degree is at most 2).
    - ring, even length: alternates colors 0/1 all the way around the cycle
      (including the wraparound edge); optimal (2 colors), and valid because
      an even cycle is 2-edge-colorable.
    - ring, odd length: alternates 0/1 along the first n-1 edges and colors
      the final wraparound edge 2; this matches the true chromatic index of
      an odd cycle (an odd cycle cannot be properly colored with only 2
      colors).

    Parameters:
    vertices (list): The vertices, in canonical sequential order.
    topology (str): Either "line" or "ring".

    Returns:
    color_patches (dict): A dictionary mapping each color to a list of edges
        colored with that color.
    """
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

    # Drop any color slot that ended up unused, so the returned coloring
    # reports the true (optimal) number of colors actually needed.
    return {c: v for c, v in color_patches.items() if v}


def _grid_torus_closed_form_coloring(vertices: List[Vertex], s: int, topology: str) -> Coloring:
    """
    Closed-form edge coloring for a grid (s x s mesh) or an even-`s` torus
    (s x s mesh with row/column wraparound), laid out in canonical row-major
    order.

    Horizontal ("right"/"left") edges are colored by column parity (colors
    0/1); vertical ("up"/"down") edges are colored by row parity (colors
    2/3). This always produces a valid, complete edge coloring using exactly
    4 colors for a plain grid of any size s>=2 (optimal, since grid graphs
    are bipartite and an interior vertex has degree 4), and also for a torus
    when `s` is even (each row/column wraparound is itself an even cycle, so
    the alternating pattern remains consistent through the wraparound edge).

    This closed form is *not* valid for an odd-`s` torus: each row/column
    wraparound is then an odd cycle, which breaks the column/row-parity
    alternation (the same obstruction that prevents 2-coloring an odd ring).
    Callers must not use this function for that case -- see `auto_edge_coloring`.

    Parameters:
    vertices (list): The vertices, in canonical row-major order.
    s (int): The side length of the (square) mesh; len(vertices) == s*s.
    topology (str): Either "grid" or "torus".

    Returns:
    color_patches (dict): A dictionary mapping each color to a list of edges
        colored with that color.
    """
    is_torus = (topology == "torus")
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

    # Drop any color slot that ended up unused, so the returned coloring
    # reports the true (optimal) number of colors actually needed.
    return {c: v for c, v in color_patches.items() if v}


def auto_edge_coloring(deg: int, vertices: List[Vertex], edges: List[Edge],
                       neighbors: NeighborMap) -> Coloring:
    """
    Detects whether the input graph matches one of the canonical topologies
    produced by `ProcessorSpec(geometry=...)` / `QubitGraph.common_graph`
    ("line", "ring", "grid", "torus") via `detect_topology`, and if so, uses a
    cheap closed-form coloring that is always valid and uses the optimal
    number of colors (the true chromatic index) for that topology.

    Falls back to `vizing_edge_coloring` -- a deterministic algorithm that is
    always valid and complete, using at most deg+1 colors -- for graphs that
    don't match one of these canonical topologies, and for odd-side-length
    tori (where the simple closed-form pattern is not valid; see
    `_grid_torus_closed_form_coloring` for details).

    Parameters:
    deg (int): The maximum degree of the graph (used only by the fallback).
    vertices (list): A list of vertices in the graph, in the same order used
        to originally construct the graph (see `detect_topology`).
    edges (list): A list of edges represented as tuples of vertices.
    neighbors (dict): A dictionary mapping each vertex to its neighboring
        vertices.

    Returns:
    color_patches (dict): A dictionary mapping each color to a list of edges
        colored with that color.
    """
    topology = detect_topology(vertices, edges, neighbors)

    if topology in ("line", "ring"):
        return _line_ring_closed_form_coloring(vertices, topology)

    if topology in ("grid", "torus"):
        s = int(round(np.sqrt(len(vertices))))
        if topology == "grid" or s % 2 == 0:
            return _grid_torus_closed_form_coloring(vertices, s, topology)

    return vizing_edge_coloring(deg, vertices, edges, neighbors)


def order(u: Vertex, v: Vertex) -> Edge:
    """
    Return a tuple containing the two input values in sorted order.
    """
    return (min(u, v), max(u, v))


class _VizingChainState:
    """
    Shared bookkeeping and primitives for the "Vizing chain" edge-recoloring
    step (fan construction, alternating c/d-path finding, and free-color
    queries) used by several of the edge-coloring algorithms below
    (`misra_gries_edge_coloring`, `vizing_edge_coloring`,
    `new_bipartite_edge_coloring`, and `sinnamon_euler_color_edge_coloring`).

    Every query (`is_free`, `missing_colors`, `first_free`) is derived on
    demand from `edge_colors` -- the single source of truth -- rather than
    from a separately-tracked "free colors per vertex" cache that has to be
    kept in sync by hand. That design avoids the state-drift bugs (stale
    caches causing hangs/KeyErrors) that motivated the original
    `_misra_gries_color_edge` implementation this class replaces.
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


def misra_gries_edge_coloring(
    deg: int, vertices: List[Vertex], edges: List[Edge], neighbors: NeighborMap
) -> Coloring:
    """
    Implements Misra & Gries' edge coloring algorithm for a simple undirected graph.

    This is a deterministic algorithm that always terminates and produces a
    proper edge coloring using at most deg+1 colors (Vizing's theorem).
    """
    edges_canonical = list({order(u, v) for u, v in edges})
    state = _VizingChainState(deg, neighbors, edges_canonical)

    for u, v in edges_canonical:
        state.color_edge_vizing_chain(u, v)

    return state.color_patches


def sinnamon_euler_color_edge_coloring(
    deg: int, vertices: List[Vertex], edges: List[Edge], neighbors: NeighborMap,
    seed: Optional[Union[int, np.random.Generator]] = None
) -> Coloring:
    """
    Sinnamon's randomized algorithm for (d+1)-edge-coloring.

    seed (None, int, or numpy.random.Generator): Seed or generator controlling the
        randomization. Passing the same integer seed yields reproducible results.
    """
    rng = np.random.default_rng(seed)
    edges_canonical = list({order(u, v) for u, v in edges})
    state = _VizingChainState(deg, neighbors, edges_canonical)

    for u, v in edges_canonical:
        _sinnamon_color_edge(u, v, state, vertices, rng)

    return state.color_patches


def _sinnamon_color_edge(u: Vertex, v: Vertex, state: _VizingChainState,
                          vertices: List[Vertex], rng: np.random.Generator) -> None:
    """Color the single (currently uncolored) edge (u, v) using one step of
    Sinnamon's randomized algorithm: a direct assignment when a common free
    color exists, else several randomized Kempe-chain attempts, falling back
    to a deterministic Vizing-chain step (`state.color_edge_vizing_chain`) if
    those attempts don't pan out.
    """
    # Simple case: a color free at both endpoints.
    for color in sorted(state.missing_colors(u)):
        if state.is_free(color, v):
            state.set_edge_color(u, v, color)
            return

    # Randomized case.
    for _ in range(int(np.log(len(vertices))) + 1):
        fan = state.build_fan(u, v)
        k = fan[-1]

        c = int(rng.choice(sorted(state.missing_colors(u))))
        used_k_colors = sorted(set(state.all_colors) - state.missing_colors(k))
        if not used_k_colors:
            state.color_edge_vizing_chain(u, v)
            return
        d = int(rng.choice(used_k_colors))

        path = state.cd_path(u, c, d)
        if not path or path[-1][1] != v:
            # Flip the c/d-alternating path (edges alternate d, c, d, c, ...).
            for i, path_edge in enumerate(path):
                other_color = c if i % 2 == 0 else d
                state.set_edge_color(path_edge[0], path_edge[1], other_color)

            w_index = 0
            for i in range(len(fan)):
                if state.is_free(d, fan[i]):
                    w_index = i
            w, sub_fan = fan[w_index], fan[:w_index + 1]

            # Rotate the subfan: shift each (u, sub_fan[i+1]) color down to
            # (u, sub_fan[i]).
            for i in range(len(sub_fan) - 2, -1, -1):
                f_i, f_i_plus_1 = sub_fan[i], sub_fan[i + 1]
                new_color = state.edge_colors.get(order(u, f_i_plus_1), -1)
                state.set_edge_color(u, f_i, new_color)

            state.set_edge_color(u, w, d)
            return

    # Fallback to a deterministic Vizing-chain step.
    state.color_edge_vizing_chain(u, v)


def new_bipartite_edge_coloring(
    deg: int, vertices: List[Vertex], edges: List[Edge], neighbors: NeighborMap,
    seed: Optional[Union[int, np.random.Generator]] = None
) -> Coloring:
    """
    Randomised (Delta+1)-edge coloring that always returns a complete, valid coloring.

    Edges are processed in a random order.  For each edge (u, v):

    * **Case 1** – a color is missing at both endpoints: assign it directly.
    * **Case 2** – no common missing color: apply one step of the Misra-Gries
      fan/Kempe-chain algorithm, which is proven to always succeed in at most
      deg+1 colors.

    On bipartite graphs the random ordering allows Case 1 to fire often, so the
    result frequently uses only deg colors (the bipartite chromatic-index
    optimum).  Correctness is guaranteed on all simple graphs.

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
    rng = np.random.default_rng(seed)

    # Collect unique canonical edges and initialize the coloring state.
    unique_edges = list({order(u, v) for u, v in edges})
    state = _VizingChainState(deg, neighbors, unique_edges)

    if deg == 0:
        # No edges can exist in a degree-0 graph.
        return state.color_patches

    # Color every edge using a randomised order and a Kempe-chain strategy.
    #
    # For each uncolored edge (u, v):
    #
    #   Case 1 – a color c is missing at both u and v:
    #       Assign c directly.
    #
    #   Case 2 – no common missing color:
    #       Use one Vizing-chain step (state.color_edge_vizing_chain) to color
    #       the edge.  That routine always succeeds in deg+1 colors and
    #       maintains color_patches in sync.
    #
    # The random ordering of edges is the "bipartite" contribution: for graphs
    # that satisfy the conditions of Vizing's theorem for bipartite graphs (i.e.
    # χ'(G) = Δ(G)), Case 1 fires frequently enough that the result usually uses
    # exactly deg colors.  The Vizing-chain fallback guarantees correctness on
    # every graph class.

    shuffled = [unique_edges[i] for i in rng.permutation(len(unique_edges))]

    for u, v in shuffled:
        common = state.missing_colors(u) & state.missing_colors(v)
        if common:
            # Case 1: assign any common missing color directly.
            c = next(iter(common))
            state.set_edge_color(u, v, c)
        else:
            # Case 2: delegate to a single Vizing-chain step.
            state.color_edge_vizing_chain(u, v)

    return state.color_patches


class ColoringSolver:
    def __init__(self, n: int, adj: List[List[int]], K: int,
                 max_time: float = 60.0,
                 ls_iters_per_restart: Optional[int] = None,
                 max_no_improve: int = 100,
                 perturb_size_frac: float = 0.1,
                 seed: Optional[Union[int, np.random.Generator]] = None) -> None:
        self.n = n
        self.adj = adj
        self.K = K
        # parameter settings from Sec. 9
        self.max_time = max_time
        self.ls_iters = ls_iters_per_restart or 1000 * self.n
        self.max_no_improve = max_no_improve
        self.perturb_size = max(1, int(perturb_size_frac * self.n))
        self.rng = np.random.default_rng(seed)

        # state
        self.col = [int(c) for c in self.rng.integers(K, size=self.n)]
        self.best_col = list(self.col)
        self.best_conflicts = self.count_conflicts(self.col)

    def count_conflicts(self, col: List[int]) -> int:
        c = 0
        for u in range(self.n):
            cu = col[u]
            for v in self.adj[u]:
                if v > u and col[v] == cu:
                    c += 1
        return c

    def one_local_search(self) -> bool:
        """Run one phase of iterative improvement up to ls_iters."""
        col = self.col
        # maintain conflict list
        vertex_conflicts = [0] * self.n
        for u in range(self.n):
            cnt = 0
            for v in self.adj[u]:
                if col[v] == col[u]:
                    cnt += 1
            vertex_conflicts[u] = cnt

        for it in range(self.ls_iters):
            # early exit if the current coloring is conflict-free
            if not any(vertex_conflicts):
                break

            # pick a conflicting vertex at random
            conflicted = [u for u, cnt in enumerate(vertex_conflicts) if cnt > 0]
            if not conflicted:
                break
            u = int(self.rng.choice(conflicted))

            # find best color for u (minimize its local conflicts)
            best_c, best_cnt = col[u], vertex_conflicts[u]
            for c in range(self.K):
                if c == col[u]:
                    continue
                cnt = 0
                for v in self.adj[u]:
                    if col[v] == c:
                        cnt += 1
                if cnt < best_cnt:
                    best_cnt, best_c = cnt, c

            # apply move if it strictly improves, or with prob if equal
            if best_cnt < vertex_conflicts[u] or self.rng.random() < 0.01:
                col[u] = best_c
                # update conflicts for u and neighbors
                vertex_conflicts[u] = best_cnt
                for v in self.adj[u]:
                    # recalc v’s conflicts
                    cntv = 0
                    for w in self.adj[v]:
                        if col[w] == col[v]:
                            cntv += 1
                    vertex_conflicts[v] = cntv

        # update best
        curr_conf = self.count_conflicts(col)
        if curr_conf < self.best_conflicts:
            self.best_conflicts = curr_conf
            self.best_col = list(col)
            return True
        return False

    def perturb(self) -> None:
        """Randomly perturb the current best solution."""
        col = list(self.best_col)
        for _ in range(self.perturb_size):
            u = int(self.rng.integers(self.n))
            col[u] = int(self.rng.integers(self.K))
        self.col = col

    def solve(self) -> Tuple[List[int], int]:
        """Main loop (Sec. 8 pseudocode)."""
        no_improve = 0

        # initial local search
        self.one_local_search()

        loop_count = 0
        while loop_count < 100:
            loop_count += 1
            improved = self.one_local_search()
            if self.best_conflicts == 0:
                break
            if improved:
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.max_no_improve:
                # perturb around the best we've seen
                self.perturb()
                no_improve = 0
            else:
                # Resume the next phase from the best known solution rather than
                # the (possibly degraded) working coloring from this phase.
                self.col = list(self.best_col)

        return self.best_col, self.best_conflicts


def _eulerian_partition(
    vertices: List[Vertex], edges: List[Edge]
) -> Tuple[List[Edge], List[Edge]]:
    """
    Partitions the edges of a graph into two sets, E1 and E2, such that for every vertex v,
    the number of incident edges in E1 and E2 differs by at most 1.
    """
    adj: NeighborMap = {v: [] for v in vertices}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    E1: List[Edge] = []
    E2: List[Edge] = []
    visited_edges: Set[Edge] = set()

    for u, v_list in adj.items():
        for v in v_list:
            edge = order(u, v)
            if edge not in visited_edges:
                if len(E1) <= len(E2):
                    E1.append(edge)
                else:
                    E2.append(edge)
                visited_edges.add(edge)
    return E1, E2


def sinnamon_2d_minus_1_edge_coloring(
    deg: int, vertices: List[Vertex], edges: List[Edge], neighbors: NeighborMap
) -> Coloring:
    unique_edges = list(set(order(u, v) for u, v in edges))
    if not unique_edges:
        return {}
    if deg <= 1:
        return {0: unique_edges}
    E1, E2 = _eulerian_partition(vertices, unique_edges)
    neighbors1: NeighborMap = {v: [] for v in vertices}
    for u, v in E1:
        neighbors1[u].append(v)
        neighbors1[v].append(u)
    deg1 = max(len(v) for v in neighbors1.values()) if E1 else 0
    neighbors2: NeighborMap = {v: [] for v in vertices}
    for u, v in E2:
        neighbors2[u].append(v)
        neighbors2[v].append(u)
    deg2 = max(len(v) for v in neighbors2.values()) if E2 else 0
    coloring1 = sinnamon_2d_minus_1_edge_coloring(deg1, vertices, E1, neighbors1)
    coloring2 = sinnamon_2d_minus_1_edge_coloring(deg2, vertices, E2, neighbors2)
    num_colors1 = max(coloring1.keys()) + 1 if coloring1 else 0
    shifted_coloring2 = {k + num_colors1: v for k, v in coloring2.items()}
    coloring1.update(shifted_coloring2)
    return coloring1


def vizing_edge_coloring(
    deg: int, vertices: List[Vertex], edges: List[Edge], neighbors: NeighborMap
) -> Coloring:
    """
    Vizing's edge-coloring algorithm (constructive proof of Vizing's theorem):
    colors any simple graph with at most deg+1 colors.

    Each edge is colored in one of two cases:
      * Simple case: a color free at both endpoints is used directly (fast path).
      * Complex case: a Vizing-chain step -- build a fan around u, invert a
        two-colored (Kempe) chain, and rotate the fan to free a color.

    Note: the Vizing-chain step IS the Misra-Gries procedure. Misra & Gries
    (1992) is the standard formalization of Vizing's chain argument -- same
    fan/Kempe-chain/rotation primitives, same deg+1 guarantee -- differing only
    in that it pins down exactly which two colors and which alternating path to
    use. This function therefore delegates its complex case to
    `_VizingChainState.color_edge_vizing_chain` rather than duplicating that
    logic; the only thing distinguishing it from `misra_gries_edge_coloring` is
    the greedy simple-case fast path below.
    """
    state = _VizingChainState(deg, neighbors, edges)

    for u, v in edges:
        if state.edge_colors[order(u, v)] == -1:
            # Simple case: if a free color at u is also free at v.
            c = -1
            for color in state.missing_colors(u):
                if state.is_free(color, v):
                    c = color
                    break

            if c != -1:
                state.set_edge_color(u, v, c)
            else:
                # Complex case: apply one Vizing-chain step (fan + cd-path
                # inversion + fan rotation). This is exactly the Misra-Gries
                # procedure, so we reuse the tested implementation.
                state.color_edge_vizing_chain(u, v)

    return state.color_patches


def _line_graph_adjacency(vertices: List[Vertex], unique_edges: List[Edge]) -> List[Set[int]]:
    """
    Builds adjacency (by edge-index into `unique_edges`) for the line-graph
    L(G): edge-indices i and j are adjacent in L(G) iff the corresponding
    edges of G share an endpoint.
    """
    inc: Dict[Vertex, List[int]] = {v: [] for v in vertices}
    for idx, (u, v) in enumerate(unique_edges):
        inc[u].append(idx)
        inc[v].append(idx)
    L_adj: List[Set[int]] = [set() for _ in range(len(unique_edges))]
    for e_ids in inc.values():
        # all edges incident to a common vertex form a clique in the line-graph
        for i in range(len(e_ids)):
            for j in range(i + 1, len(e_ids)):
                a, b = e_ids[i], e_ids[j]
                L_adj[a].add(b)
                L_adj[b].add(a)
    return L_adj


def _line_graph_conflicts(L_adj: List[Set[int]], col: List[int]) -> List[Tuple[int, int]]:
    """Returns every (i, j), i < j, adjacent in the line-graph whose current
    colors collide."""
    return [(i, j) for i in range(len(L_adj)) for j in L_adj[i]
            if j > i and col[i] == col[j]]


def _moser_tardos_resample(
    E: int, L_adj: List[Set[int]], K: int,
    rng: np.random.Generator, max_resamples: int
) -> Optional[List[int]]:
    """
    One Las Vegas attempt: draw a uniformly random color in 0..K-1 for each of
    the E line-graph vertices, then repeatedly resample the endpoints of any
    conflicting (same-color, adjacent) pair until no conflicts remain or
    `max_resamples` is exceeded.

    Returns the conflict-free coloring, or None if the cap was hit.
    """
    col = [int(c) for c in rng.integers(K, size=E)]
    stack = _line_graph_conflicts(L_adj, col)
    resamples = 0
    while stack and resamples < max_resamples:
        resamples += 1
        i, j = stack.pop()
        if col[i] != col[j]:
            continue  # already fixed by an earlier resample
        # resample the *two* variables in this bad event, ensuring they
        # don't collide with each other again.
        col[i] = int(rng.integers(K))
        col[j] = int(rng.integers(K))
        # any neighbor edge k of i or j that now conflicts must be re-added
        for k in L_adj[i]:
            if col[k] == col[i]:
                stack.append((min(i, k), max(i, k)))
        for k in L_adj[j]:
            if col[k] == col[j]:
                stack.append((min(j, k), max(j, k)))
        # When the working stack empties, rescan to catch any conflicts that
        # were introduced but not re-queued, so we never report a false
        # "conflict-free" state.
        if not stack:
            stack = _line_graph_conflicts(L_adj, col)
    return col if not stack else None


def _edges_by_color(unique_edges: List[Edge], col_assignment: List[int]) -> Coloring:
    """Groups `unique_edges` into a Coloring dict keyed by each edge's
    assigned color (col_assignment[idx] is the color of unique_edges[idx])."""
    color_patches: Coloring = {}
    for idx, edge in enumerate(unique_edges):
        color_patches.setdefault(col_assignment[idx], []).append(edge)
    return color_patches


def moser_tardos_edge_coloring(
    deg: int, vertices: List[Vertex], edges: List[Edge], neighbors: NeighborMap,
    seed: Optional[Union[int, np.random.Generator]] = None
) -> Coloring:
    """
    Las Vegas edge-coloring via Moser-Tardos on the line-graph.

    Inputs:
      deg       : number of colors K to use
      vertices  : original vertex IDs (any hashable)
      edges     : list of pairs (u,v), original edges
      neighbors : (ignored)
      seed      : None, int, or numpy.random.Generator controlling the
                  randomization. Passing the same integer seed yields
                  reproducible results.

    Returns:
      color_patches : dict mapping each (u,v) -> color in 0..K-1
    """
    rng = np.random.default_rng(seed)
    # Canonicalize edges to unique, ordered tuples. The input `edges` list is
    # assumed symmetric; keeping both directions would create duplicate,
    # mutually-conflicting line-graph vertices and inflate the required colors.
    unique_edges = list({order(u, v) for u, v in edges})
    E = len(unique_edges)
    L_adj = _line_graph_adjacency(vertices, unique_edges)

    # A proper edge-coloring needs at most deg+1 colors (Vizing's theorem);
    # using only deg colors may make a valid coloring impossible, causing the
    # resampling to never terminate.
    K = deg + 1
    # Cap the number of resamples per attempt and retry with a fresh coloring if
    # the Las Vegas process does not converge, so we never loop forever.
    max_resamples = 1000 * (E + 1)
    max_attempts = 100

    col: Optional[List[int]] = None
    for _ in range(max_attempts):
        col = _moser_tardos_resample(E, L_adj, K, rng, max_resamples)
        if col is not None:
            break
    if col is None:
        raise RuntimeError(
            f"moser_tardos_edge_coloring failed to converge to a valid "
            f"{K}-edge-coloring after {max_attempts} attempts."
        )

    return _edges_by_color(unique_edges, col)


def assadi_oct25_edge_coloring(deg: int,
                               vertices: List[Vertex],
                               edges: List[Edge],
                               neighbors: NeighborMap,
                               seed: Optional[Union[int, np.random.Generator]] = None) -> Coloring:
    """
    Compute an edge-coloring of G=(vertices, edges) with 'deg' colors by
    reducing to a vertex-coloring on the line-graph.

    Inputs:
      deg       : number of colors (K)
      vertices  : list of original vertex IDs (can be any hashable)
      edges     : list of pairs (u,v), each an edge in the original graph
      neighbors : (not used in this implementation)
      seed      : None, int, or numpy.random.Generator controlling the
                  randomization. Passing the same integer seed yields
                  reproducible results.

    Returns:
      color_patches : dict mapping each edge (u,v) → color in {0,…,deg-1}
    """
    rng = np.random.default_rng(seed)
    # Canonicalize edges to unique, ordered tuples. The input `edges` list is
    # assumed symmetric (contains both (u,v) and (v,u)); collapsing to a single
    # representative per undirected edge is required, otherwise the two
    # directions become separate (mutually adjacent) line-graph vertices that
    # get different colors while mapping back to the same output edge.
    unique_edges = list({order(u, v) for u, v in edges})
    E = len(unique_edges)

    # Build the line-graph L(G): edge-indices i, j are adjacent iff the
    # corresponding edges of G share an endpoint.
    line_graph_adj = [list(s) for s in _line_graph_adjacency(vertices, unique_edges)]

    # Color L(G) with the local-search vertex-coloring solver.
    # A proper edge-coloring needs at most deg+1 colors (Vizing's theorem),
    # so we give the vertex-coloring solver on the line-graph deg+1 colors.
    num_colors = deg + 1
    max_attempts = 100

    for _ in range(max_attempts):
        solver = ColoringSolver(E, line_graph_adj, num_colors, seed=rng)
        best_col, conflicts = solver.solve()

        # map back: each unique edge → the color of its corresponding vertex in L(G)
        color_patches = _edges_by_color(unique_edges, best_col)

        # A conflict-free vertex coloring of L(G) is exactly a valid edge coloring.
        if conflicts == 0 and check_valid_edge_coloring(color_patches, ret_false_on_error=True):
            return color_patches

    raise RuntimeError(
        f"assadi_oct25_edge_coloring failed to find a valid {num_colors}-edge-coloring "
        f"after {max_attempts} attempts."
    )


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
