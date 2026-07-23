"""
Sinnamon (2019)'s Euler-Template edge-coloring algorithms: the deterministic
"Greedy-Euler-Color" (using ≤ 2*deg-1 colors in O(m log(deg)) worst-case time)
and the randomized "Random-Euler-Color" (using ≤ deg+1 colors in O(m*sqrt(n))
time with high probability). Both are built on the shared recursive
Partition/Recurse/Prune/Repair scaffold in `_EulerTemplateEdgeColoring`
(Sinnamon 2019, Section 2.2), differing only in their color budget and Repair step.

Reference:
    C. Sinnamon, "Fast and Simple Edge-Coloring Algorithms," arXiv:1907.03201,
    2019. https://arxiv.org/abs/1907.03201
"""
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Union

from ._common import Vertex, Color, Edge, NeighborMap, Coloring, order


def _eulerian_partition(
    vertices: List[Vertex], edges: List[Edge]
) -> Tuple[List[Edge], List[Edge]]:
    """
    Partitions the edges of a (simple) graph into two edge-disjoint sets E1,
    E2 such that, for every vertex v, `|deg_E1(v) - deg_E2(v)| <= 1` for all
    but at most one vertex per odd-length closed trail (see note below) --
    and consequently `max_degree(E1), max_degree(E2) <= ceil(max_degree(G) /
    2)`, again modulo that same rare +1 slack.

    This is the "Euler partition" of Sinnamon (2019), Section 2.3: decompose
    the edges into edge-disjoint trails such that every odd-degree vertex is
    the endpoint of exactly one trail and no even-degree vertex is a trail
    endpoint (found greedily in O(m) time by repeatedly peeling off a
    maximal trail), then alternately assign each trail's edges to E1/E2.
    This is the shared foundation of `_EulerTemplateEdgeColoring`'s
    Partition step.

    Note on odd-length closed trails: a maximal trail through an all-even-
    degree component is a closed circuit; if it has an *odd* number of
    edges (e.g. a triangle), alternating E1/E2 around it assigns its first
    and last edges -- sharing a vertex -- to the same side, giving that one
    vertex a 2-edge imbalance instead of at most 1. This is combinatorially
    unavoidable (a triangle's 3 edges can't be split into two matchings), not
    a bug, and doesn't affect correctness downstream (the recursive
    algorithms built on this function recurse on each side's *actual*
    resulting max degree, and the Prune step absorbs any accumulated slack)
    -- it only occasionally slows how fast max degree halves across
    recursion levels.

    Note: naively "balancing the running total edge count between E1/E2"
    (rather than this trail-based construction) does *not* guarantee the
    per-vertex degree-splitting property above, and silently breaks the
    degree-halving assumption the algorithms built on this function depend on.
    """
    # Per-vertex (neighbor, edge_index) incidence lists, plus a
    # monotonically-advancing per-vertex cursor so "find the next
    # not-yet-consumed incident edge" is O(1) amortized overall, giving O(m)
    # total work.
    edge_list = list(edges)
    adj: Dict[Vertex, List[Tuple[Vertex, int]]] = {v: [] for v in vertices}
    for i, (u, v) in enumerate(edge_list):
        adj[u].append((v, i))
        adj[v].append((u, i))

    used_edge = [False] * len(edge_list)
    cursor = {v: 0 for v in vertices}

    def next_unused_incidence(v: Vertex) -> Optional[Tuple[Vertex, int]]:
        lst = adj[v]
        i = cursor[v]
        while i < len(lst) and used_edge[lst[i][1]]:
            i += 1
        cursor[v] = i
        return lst[i] if i < len(lst) else None

    def walk_maximal_trail(start: Vertex) -> List[Edge]:
        trail: List[Edge] = []
        current = start
        while True:
            nxt = next_unused_incidence(current)
            if nxt is None:
                break
            neighbor, eidx = nxt
            used_edge[eidx] = True
            trail.append(edge_list[eidx])
            current = neighbor
        return trail

    def assign_trail(trail: List[Edge], E1: List[Edge], E2: List[Edge]) -> None:
        for i, (u, v) in enumerate(trail):
            (E1 if i % 2 == 0 else E2).append(order(u, v))

    E1: List[Edge] = []
    E2: List[Edge] = []

    # Peel exactly one maximal trail from each odd-degree vertex (a trail
    # starting at an odd-degree vertex is guaranteed, by parity, to end only
    # at another odd-degree vertex -- so this accounts for all of them).
    odd_vertices = [v for v in vertices if len(adj[v]) % 2 == 1]
    for v in odd_vertices:
        if next_unused_incidence(v) is not None:
            assign_trail(walk_maximal_trail(v), E1, E2)

    # Remaining edges belong to purely even-degree components (disjoint
    # circuits); peel maximal trails (which close back into a circuit) until
    # none remain.
    for v in vertices:
        while next_unused_incidence(v) is not None:
            assign_trail(walk_maximal_trail(v), E1, E2)

    return E1, E2


class _EulerTemplateEdgeColoring:
    """
    Shared recursive "Euler-Template" edge-coloring scaffold of Sinnamon
    (2019), Section 2.2: Partition, Recurse, Prune, Repair.

    Common base for `_GreedyEulerColoring` (Greedy-Euler-Color) and
    `_RandomEulerColoring` (Random-Euler-Color): they differ only in their
    color budget (`_num_colors`) and Repair step (`_repair`), which
    subclasses must override; the Partition/Recurse/Prune logic (built on
    `_eulerian_partition`) is identical and lives only here.
    """

    def _num_colors(self, deg: int) -> int:
        """Number of available colors for a subgraph of max degree `deg`."""
        raise NotImplementedError

    def _repair(self, vertices: List[Vertex], neighbors: NeighborMap, num_colors: int,
                edge_colors: Dict[Edge, Color], uncolored_edges: List[Edge]) -> None:
        """Color every edge in `uncolored_edges` using colors in
        `range(num_colors)`, mutating `edge_colors` in place."""
        raise NotImplementedError

    def _euler_template(self, deg: int, vertices: List[Vertex], edges: List[Edge]) -> Coloring:
        """The generic recursive Partition/Recurse/Prune/Repair strategy,
        parameterized by `self._num_colors` and `self._repair`."""
        unique_edges = list({order(u, v) for u, v in edges})
        if not unique_edges:
            return {}
        if deg <= 1:
            # Base case: max degree <= 1 is a matching, so 1 color suffices.
            return {0: unique_edges}

        def build_neighbors(es: List[Edge]) -> NeighborMap:
            nb: NeighborMap = {v: [] for v in vertices}
            for u, v in es:
                nb[u].append(v)
                nb[v].append(u)
            return nb

        # --- Partition ---
        E1, E2 = _eulerian_partition(vertices, unique_edges)
        neighbors1 = build_neighbors(E1)
        deg1 = max((len(nb) for nb in neighbors1.values()), default=0)
        neighbors2 = build_neighbors(E2)
        deg2 = max((len(nb) for nb in neighbors2.values()), default=0)

        # --- Recurse (using disjoint color sets, via the offset shift below) ---
        coloring1 = self._euler_template(deg1, vertices, E1)
        coloring2 = self._euler_template(deg2, vertices, E2)
        num_colors1 = (max(coloring1.keys()) + 1) if coloring1 else 0
        combined: Coloring = dict(coloring1)
        for c, es in coloring2.items():
            combined[c + num_colors1] = es

        # --- Prune ---
        target = self._num_colors(deg)
        uncolored_edges: List[Edge] = []
        if len(combined) > target:
            while len(combined) > target:
                # Choose the least-common (smallest) color class and uncolor it.
                smallest_color = min(combined, key=lambda c: len(combined[c]))
                uncolored_edges.extend(combined.pop(smallest_color))
            # Relabel surviving color classes to a contiguous 0..(K-1) range
            # (K <= target) so labels stay within [0, target) and a parent's
            # offset-shift of a sibling's colors remains collision-free.
            combined = {i: es for i, es in enumerate(combined.values())}

        # --- Repair ---
        if uncolored_edges:
            neighbors_all = build_neighbors(unique_edges)
            edge_colors: Dict[Edge, Color] = {}
            for c, es in combined.items():
                for e in es:
                    edge_colors[e] = c
            self._repair(vertices, neighbors_all, target, edge_colors, uncolored_edges)
            combined = {}
            for e, c in edge_colors.items():
                combined.setdefault(c, []).append(e)

        return combined


class _GreedyEulerColoring(_EulerTemplateEdgeColoring):
    """
    Sinnamon's (2019) "Greedy-Euler-Color": deterministic (2*deg-1)-edge-
    coloring for multigraphs, O(m log(deg)) time. `_num_colors`/`_repair`
    (Section 3) are used exclusively by this class -- Random-Euler-Color has
    its own, different, versions (see `_RandomEulerColoring`).
    """

    def _num_colors(self, deg: int) -> int:
        """Greedy-Euler-Color's color budget: 2*deg - 1."""
        return 2 * deg - 1

    def _repair(self, vertices: List[Vertex], neighbors: NeighborMap, num_colors: int,
                edge_colors: Dict[Edge, Color], uncolored_edges: List[Edge]) -> None:
        """
        For each uncolored edge, scan the `num_colors` available colors for
        one missing at both endpoints. By pigeonhole (each endpoint has at
        least `num_colors - deg <= deg` colors missing, out of
        `num_colors = 2*deg-1`), such a color always exists. O(num_colors)/edge.
        """
        used: Dict[Vertex, Set[Color]] = {v: set() for v in vertices}
        for (u, v), c in edge_colors.items():
            used[u].add(c)
            used[v].add(c)

        for u, v in uncolored_edges:
            for c in range(num_colors):
                if c not in used[u] and c not in used[v]:
                    edge_colors[order(u, v)] = c
                    used[u].add(c)
                    used[v].add(c)
                    break
            else:
                raise RuntimeError(
                    "Greedy-Color: no common free color found (should be "
                    "impossible by the pigeonhole principle)."
                )

    def color(self, deg: int, vertices: List[Vertex], edges: List[Edge],
              neighbors: NeighborMap) -> Coloring:
        return self._euler_template(deg, vertices, edges)


def sinnamon_2d_minus_1_edge_coloring(
    deg: int, vertices: List[Vertex], edges: List[Edge], neighbors: NeighborMap
) -> Coloring:
    """
    Sinnamon's (2019) "Greedy-Euler-Color": a deterministic (2*deg-1)-edge-
    coloring algorithm for multigraphs, running in O(m log(deg)) time. See
    `_GreedyEulerColoring` for the implementation.
    """
    return _GreedyEulerColoring().color(deg, vertices, edges, neighbors)


class _RandomEulerColoring(_EulerTemplateEdgeColoring):
    """
    Sinnamon's (2019) "Random-Euler-Color": randomized (deg+1)-edge-coloring
    for simple graphs, O(m*sqrt(n)) time w.h.p. `_repair` (Section 5) is
    built on the nested `_RandomColorOneState` (Sections 6-8); all three are
    used exclusively by this class (Greedy-Euler-Color has its own versions
    -- see `_GreedyEulerColoring`).

    `self.rng` and `self.D` live on this single instance and are shared
    across the *entire* recursion (every recursive `_euler_template` call
    made via `color()` reuses the same `self`) -- see `_RandomColorOneState`
    and Sinnamon (2019), Appendix A, "The Recursive Strategy and D".
    """

    class _RandomColorOneState:
        """
        Bookkeeping for Random-Euler-Color's Repair step (Sinnamon 2019,
        Sections 6-8), covering a single subgraph's Repair call.

        Uses plain Python `dict`/`set` in place of the paper's linked-list
        (mu(v)) / two-level block-array (D): a hash-based dict already gives
        O(1) *amortized average-case* ops with ~no init cost (unlike a
        preallocated array, whose whole point is avoiding an O(n*deg) init
        cost a `dict` never pays), and Appendix A explicitly sanctions
        substituting a hash-based dictionary for `D` ("...and certainly in
        the case of Random-Euler-Color, this type of dictionary can be
        substituted").

        Per vertex v (within the current subgraph):
          - `used_colors[v]`: full set of colors used on edges incident to
            v -- O(1) `is_free(v, c)` for any color c.
          - `mu[v]`: colors missing at v, restricted to `[0, deg_local(v)]`
            (mirrors the paper's mu(v), keeping init O(deg(v)) not
            O(Delta)) -- O(1) "any missing color" and an O(deg(v))
            (already within the caller's budget) random missing color.

        `D` (passed in) maps (v, color) -> the neighbor w with edge (v, w)
        colored `color`; a single dict shared across the *entire* recursion
        (built once by `_RandomEulerColoring.__init__`), populated with this
        subgraph's colored edges on construction and depopulated by
        `teardown()` once this subgraph's Repair completes (the recursion/D
        -integration scheme of Appendix A).
        """

        def __init__(self, vertices: List[Vertex], neighbors: NeighborMap,
                     edge_colors: Dict[Edge, Color], D: Dict[Tuple[Vertex, Color], Vertex]) -> None:
            self.neighbors = neighbors
            self.edge_colors = edge_colors
            self.D = D
            self.deg_local: Dict[Vertex, int] = {v: len(neighbors.get(v, [])) for v in vertices}
            self.used_colors: Dict[Vertex, Set[Color]] = {v: set() for v in vertices}
            self.mu: Dict[Vertex, Set[Color]] = {v: set(range(self.deg_local[v] + 1)) for v in vertices}
            for (u, v), c in edge_colors.items():
                self._mark_colored(u, v, c)

        def _mark_colored(self, u: Vertex, v: Vertex, c: Color) -> None:
            self.used_colors[u].add(c)
            self.used_colors[v].add(c)
            if c <= self.deg_local[u]:
                self.mu[u].discard(c)
            if c <= self.deg_local[v]:
                self.mu[v].discard(c)
            self.D[(u, c)] = v
            self.D[(v, c)] = u

        def _mark_uncolored(self, u: Vertex, v: Vertex, c: Color) -> None:
            self.used_colors[u].discard(c)
            self.used_colors[v].discard(c)
            if c <= self.deg_local[u]:
                self.mu[u].add(c)
            if c <= self.deg_local[v]:
                self.mu[v].add(c)
            del self.D[(u, c)]
            del self.D[(v, c)]

        def is_free(self, v: Vertex, c: Color) -> bool:
            return c not in self.used_colors[v]

        def any_missing(self, v: Vertex) -> Color:
            return next(iter(self.mu[v]))

        def random_missing(self, v: Vertex, rng: np.random.Generator) -> Color:
            return int(rng.choice(sorted(self.mu[v])))

        def set_color(self, u: Vertex, v: Vertex, c: Optional[Color]) -> None:
            """Set the color of (u, v) to `c` (or uncolor it, if `c is None`)."""
            e = order(u, v)
            old = self.edge_colors.get(e)
            if old is not None:
                self._mark_uncolored(u, v, old)
                del self.edge_colors[e]
            if c is not None:
                self.edge_colors[e] = c
                self._mark_colored(u, v, c)

        def find_colored_neighbor(self, v: Vertex, c: Color) -> Optional[Vertex]:
            """The neighbor w such that (v, w) is colored `c`, or None."""
            return self.D.get((v, c))

        def teardown(self) -> None:
            """Depopulate `D` of every edge colored in this subgraph, so
            a sibling subgraph's Repair step starts with a clean `D`."""
            for (u, v), c in list(self.edge_colors.items()):
                del self.D[(u, c)]
                del self.D[(v, c)]

        def make_primed_fan(self, v: Vertex, x0: Vertex, gamma: Color) -> Tuple[List[Vertex], Color]:
            """
            Sinnamon (2019), Section 7, Make-Primed-Fan: build a primed
            gamma-fan `(gamma, v, x0, x1, ..., xk)` around the uncolored
            edge (v, x0). Require: (v, x0) uncolored; gamma missing at v.

            Returns (fan, delta): `fan = [x0, ..., xk]`, `delta` primes the
            fan (missing at xk, and either missing at v or already used by
            an existing fan edge (v, x_j)).
            """
            fan = [x0]
            in_fan = {x0}
            while True:
                xk = fan[-1]
                delta = self.any_missing(xk)
                if self.is_free(v, delta):
                    return fan, delta
                w = self.find_colored_neighbor(v, delta)
                if w in in_fan:
                    return fan, delta
                fan.append(w)
                in_fan.add(w)

        def flip_alternating_path(self, v: Vertex, gamma: Color, delta: Color) -> Vertex:
            """
            Flip (interchange the colors of) the maximal gamma/delta-
            alternating path starting at `v` (first edge colored `delta`).
            Returns the path's other endpoint.
            """
            path: List[Tuple[Vertex, Vertex, Color]] = []
            visited_edges: Set[Edge] = set()
            current, look_for = v, delta
            while True:
                w = self.find_colored_neighbor(current, look_for)
                if w is None:
                    break
                e = order(current, w)
                if e in visited_edges:
                    break
                visited_edges.add(e)
                path.append((current, w, look_for))
                current = w
                look_for = gamma if look_for == delta else delta

            # Two passes -- uncolor every path edge, then recolor with the
            # swapped value -- rather than a single forward pass, which
            # would momentarily give two adjacent path edges (sharing an
            # interior vertex) the same color: harmless for per-vertex
            # color-set bookkeeping, but breaks `D`'s
            # one-edge-per-(vertex,color) invariant. Splitting avoids this
            # since no *other* edge at any path vertex uses gamma/delta.
            for a, b, _ in path:
                self.set_color(a, b, None)
            for a, b, old_color in path:
                new_color = gamma if old_color == delta else delta
                self.set_color(a, b, new_color)
            return current

        def shift_fan(self, v: Vertex, fan: List[Vertex], j: int) -> None:
            """
            Shift the fan `(v, fan[0], ..., fan[k])` from `fan[j]`: for
            i=1..j, recolor (v, fan[i-1]) with the current color of (v,
            fan[i]), then uncolor (v, fan[j]) (legal by the fan invariant:
            fan colors are always "missing on the previous leaf").

            Applied by uncoloring `fan[j]` first and cascading *backward*
            (consuming each freed color slot), rather than a naive forward
            pass -- which would momentarily give two edges at `v` the same
            color: harmless for plain color-set bookkeeping, but fatal for
            `D` (one edge per (vertex, color) key).
            """
            old_colors = [self.edge_colors[order(v, fan[i])] for i in range(1, j + 1)]
            self.set_color(v, fan[j], None)
            for i in range(j, 0, -1):
                self.set_color(v, fan[i - 1], old_colors[i - 1])

        def activate_c_fan(self, v: Vertex, fan: List[Vertex], gamma: Color, delta: Color) -> None:
            """
            Sinnamon (2019), Section 7, Activate-c-Fan: given a gamma-fan
            `(gamma, v, fan[0], ..., fan[-1])` primed by `delta`, color the
            (currently uncolored) edge `(v, fan[0])`.
            """
            xk = fan[-1]
            if self.is_free(v, delta):
                # delta missing at both v and xk: shift the whole fan from
                # xk, then color (v, xk) with delta.
                self.shift_fan(v, fan, len(fan) - 1)
                self.set_color(v, xk, delta)
                return

            # delta not missing at v, but some fan leaf x_j (j >= 1, since
            # (v, fan[0]) is uncolored) has (v, x_j) colored delta.
            j = next(i for i in range(1, len(fan)) if self.edge_colors.get(order(v, fan[i])) == delta)

            # gamma is missing at v (by construction) and vx_j is colored
            # delta, so v is an endpoint of the gamma/delta-alternating path
            # whose first edge is (v, x_j); flip it so delta becomes
            # missing at v.
            w = self.flip_alternating_path(v, gamma, delta)

            x_before_j = fan[j - 1]  # well-defined: j >= 1
            if w != x_before_j:
                # Case I: delta still missing at x_{j-1}; shift fan[0..j-1]
                # from x_{j-1}, then color (v, x_{j-1}) with delta.
                self.shift_fan(v, fan, j - 1)
                self.set_color(v, x_before_j, delta)
            else:
                # Case II: delta still missing at xk (w == x_{j-1} != xk);
                # shift the whole fan from xk, then color (v, xk) with delta.
                self.shift_fan(v, fan, len(fan) - 1)
                self.set_color(v, xk, delta)

    def __init__(self, seed: Optional[Union[int, np.random.Generator]] = None) -> None:
        self.rng = np.random.default_rng(seed)
        # A single dictionary-based `D` structure, shared across the entire
        # recursion (see `_RandomColorOneState`'s docstring).
        self.D: Dict[Tuple[Vertex, Color], Vertex] = {}

    def _num_colors(self, deg: int) -> int:
        """Random-Euler-Color's color budget: deg + 1."""
        return deg + 1

    def _repair(self, vertices: List[Vertex], neighbors: NeighborMap, num_colors: int,
                edge_colors: Dict[Edge, Color], uncolored_edges: List[Edge]) -> None:
        """
        Repair step (Sinnamon 2019, Section 5): repeatedly apply
        Random-Color-One (Section 8) -- a uniformly random uncolored edge
        and a uniformly random color missing at one endpoint, completed via
        Make-Primed-Fan + Activate-c-Fan -- until none remain.
        """
        state = self._RandomColorOneState(vertices, neighbors, edge_colors, self.D)

        # O(1) uniform-random pick/removal of an uncolored edge via a
        # swap-to-end-and-pop array, backed by a position index.
        uncolored_list = list(uncolored_edges)
        pos = {e: i for i, e in enumerate(uncolored_list)}

        def remove_uncolored(e: Edge) -> None:
            i = pos.pop(e)
            last = uncolored_list[-1]
            uncolored_list[i] = last
            pos[last] = i
            uncolored_list.pop()

        while uncolored_list:
            idx = int(self.rng.integers(len(uncolored_list)))
            v, x0 = uncolored_list[idx]
            gamma = state.random_missing(v, self.rng)
            fan, delta = state.make_primed_fan(v, x0, gamma)
            state.activate_c_fan(v, fan, gamma, delta)
            remove_uncolored(order(v, x0))

        state.teardown()

    def color(self, deg: int, vertices: List[Vertex], edges: List[Edge],
              neighbors: NeighborMap) -> Coloring:
        return self._euler_template(deg, vertices, edges)


def sinnamon_euler_color_edge_coloring(
    deg: int, vertices: List[Vertex], edges: List[Edge], neighbors: NeighborMap,
    seed: Optional[Union[int, np.random.Generator]] = None
) -> Coloring:
    """
    Sinnamon's (2019) "Random-Euler-Color": a randomized (deg+1)-edge-
    coloring algorithm for simple graphs, running in O(m*sqrt(n)) time with
    probability `1 - e^(-Omega(m))`. See `_RandomEulerColoring` for the
    implementation.

    seed (None, int, or numpy.random.Generator): Seed or generator controlling the
        randomization. Passing the same integer seed yields reproducible results.
    """
    return _RandomEulerColoring(seed=seed).color(deg, vertices, edges, neighbors)
