"""
Edge-coloring algorithms that reduce to a vertex-coloring problem on the
line-graph L(G) (edge-indices adjacent in L(G) iff the corresponding edges
of G share an endpoint): Moser-Tardos (Las Vegas resampling) and Assadi
(local search via `ColoringSolver`). Both share edge canonicalization,
line-graph construction, and a retry loop via `_LineGraphEdgeColoring`.

References:
    R. A. Moser and G. Tardos, "A constructive proof of the general Lovasz
    Local Lemma," Journal of the ACM, vol. 57, no. 2, Article 11, 2010.
    https://doi.org/10.1145/1667053.1667060

    S. Assadi, "Vizing's Theorem in Near-Linear Time," arXiv:2410.05240,
    2024. https://arxiv.org/abs/2410.05240
    Note: `_AssadiEdgeColoring` is a simplified from-scratch local-search
    vertex colorer applied to the line graph, not a port of that paper's
    algorithm/data structures.
"""
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Union

from ._common import Vertex, Edge, NeighborMap, Coloring, order, check_valid_edge_coloring


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

    def _count_matching_neighbors(self, col: List[int], x: int, c: Optional[int] = None) -> int:
        """Counts how many neighbors of vertex `x` are currently colored `c`
        (defaulting to `col[x]`, i.e. x's own current color)."""
        if c is None:
            c = col[x]
        return sum(1 for v in self.adj[x] if col[v] == c)

    def count_conflicts(self, col: List[int]) -> int:
        # Every conflicting (same-color) edge is symmetric, so it's counted
        # once from each endpoint by _count_matching_neighbors; halve to get
        # the number of conflicting edges rather than edge-endpoint pairs.
        return sum(self._count_matching_neighbors(col, u) for u in range(self.n)) // 2

    def one_local_search(self) -> bool:
        """Run one phase of iterative improvement up to ls_iters."""
        col = self.col
        # maintain conflict list
        vertex_conflicts = [self._count_matching_neighbors(col, u) for u in range(self.n)]

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
                cnt = self._count_matching_neighbors(col, u, c)
                if cnt < best_cnt:
                    best_cnt, best_c = cnt, c

            # apply move if it strictly improves, or with prob if equal
            if best_cnt < vertex_conflicts[u] or self.rng.random() < 0.01:
                col[u] = best_c
                # update conflicts for u and neighbors
                vertex_conflicts[u] = best_cnt
                for v in self.adj[u]:
                    # recalc v's conflicts
                    vertex_conflicts[v] = self._count_matching_neighbors(col, v)

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


class _LineGraphEdgeColoring:
    """
    Template for edge-coloring algorithms that reduce to a vertex-coloring
    problem on the line-graph L(G): the common base for
    `_MoserTardosEdgeColoring` and `_AssadiEdgeColoring`.

    Handles edge canonicalization, line-graph construction
    (`_line_graph_adjacency`), the shared retry loop (up to `max_attempts`
    independent attempts, since neither underlying method is guaranteed to
    succeed on a single try), and mapping a successful attempt back to
    edges (`_edges_by_color`). Subclasses provide `_num_colors(deg)` and
    `_attempt()` (one attempt at a conflict-free vertex-coloring of L(G),
    returning a `List[int]` of length `self.E`, or `None` on failure).
    """

    def __init__(self, vertices: List[Vertex], edges: List[Edge],
                 seed: Optional[Union[int, np.random.Generator]] = None) -> None:
        self.rng = np.random.default_rng(seed)
        # Canonicalize edges to unique, ordered tuples. The input `edges`
        # list is assumed symmetric; keeping both directions would create
        # duplicate, mutually-conflicting line-graph vertices and inflate
        # the required colors.
        self.unique_edges = list({order(u, v) for u, v in edges})
        self.E = len(self.unique_edges)
        self.L_adj: List[Set[int]] = _line_graph_adjacency(vertices, self.unique_edges)

    def _num_colors(self, deg: int) -> int:
        """Colors to give the line-graph vertex-coloring solver."""
        raise NotImplementedError

    def _attempt(self) -> Optional[List[int]]:
        """One attempt at a conflict-free vertex-coloring of L(G) using
        `self.num_colors` colors, or `None` if this attempt failed."""
        raise NotImplementedError

    def color(self, deg: int, max_attempts: int = 100) -> Coloring:
        self.num_colors = self._num_colors(deg)
        for _ in range(max_attempts):
            col = self._attempt()
            if col is not None:
                return _edges_by_color(self.unique_edges, col)
        raise RuntimeError(
            f"{type(self).__name__} failed to find a valid "
            f"{self.num_colors}-edge-coloring after {max_attempts} attempts."
        )


class _MoserTardosEdgeColoring(_LineGraphEdgeColoring):
    """Las Vegas edge-coloring via Moser-Tardos on the line-graph: each
    attempt draws a uniformly random color per line-graph vertex and
    repeatedly resamples conflicting pairs (`_moser_tardos_resample`)."""

    def _num_colors(self, deg: int) -> int:
        # deg+1 colors (Vizing's theorem); deg alone may make a valid
        # coloring impossible, so the resampling would never terminate.
        return deg + 1

    def _attempt(self) -> Optional[List[int]]:
        # Cap resamples per attempt so a non-converging process doesn't
        # loop forever within a single attempt.
        max_resamples = 1000 * (self.E + 1)
        return _moser_tardos_resample(self.E, self.L_adj, self.num_colors, self.rng, max_resamples)


def moser_tardos_edge_coloring(
    deg: int, vertices: List[Vertex], edges: List[Edge], neighbors: NeighborMap,
    seed: Optional[Union[int, np.random.Generator]] = None
) -> Coloring:
    """
    Las Vegas edge-coloring via Moser-Tardos on the line-graph. See
    `_MoserTardosEdgeColoring` for the implementation.

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
    return _MoserTardosEdgeColoring(vertices, edges, seed=seed).color(deg)


class _AssadiEdgeColoring(_LineGraphEdgeColoring):
    """Edge-coloring via local-search vertex-coloring (`ColoringSolver`) on
    the line-graph: each attempt runs a fresh `ColoringSolver` and accepts
    its result only if it is conflict-free."""

    def __init__(self, vertices: List[Vertex], edges: List[Edge],
                 seed: Optional[Union[int, np.random.Generator]] = None) -> None:
        super().__init__(vertices, edges, seed=seed)
        # ColoringSolver expects list-of-lists adjacency; build this once
        # (not per-attempt) since self.L_adj never changes.
        self.line_graph_adj = [list(s) for s in self.L_adj]

    def _num_colors(self, deg: int) -> int:
        return deg + 1  # deg+1 colors suffice (Vizing's theorem)

    def _attempt(self) -> Optional[List[int]]:
        solver = ColoringSolver(self.E, self.line_graph_adj, self.num_colors, seed=self.rng)
        best_col, conflicts = solver.solve()
        # A conflict-free vertex coloring of L(G) is exactly a valid edge
        # coloring; double-check via check_valid_edge_coloring rather than
        # trusting `conflicts` alone.
        color_patches = _edges_by_color(self.unique_edges, best_col)
        if conflicts == 0 and check_valid_edge_coloring(color_patches, ret_false_on_error=True):
            return best_col
        return None


def assadi_oct25_edge_coloring(deg: int,
                               vertices: List[Vertex],
                               edges: List[Edge],
                               neighbors: NeighborMap,
                               seed: Optional[Union[int, np.random.Generator]] = None) -> Coloring:
    """
    Compute an edge-coloring of G=(vertices, edges) with 'deg' colors by
    reducing to a vertex-coloring on the line-graph. See
    `_AssadiEdgeColoring` for the implementation.

    Inputs:
      deg       : number of colors (K)
      vertices  : list of original vertex IDs (can be any hashable)
      edges     : list of pairs (u,v), each an edge in the original graph
      neighbors : (not used in this implementation)
      seed      : None, int, or numpy.random.Generator controlling the
                  randomization. Passing the same integer seed yields
                  reproducible results.

    Returns:
      color_patches : dict mapping each edge (u,v) -> color in {0,...,deg-1}
    """
    return _AssadiEdgeColoring(vertices, edges, seed=seed).color(deg)
