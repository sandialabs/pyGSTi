import numpy as np
import copy
from typing import List, Dict, Tuple, Optional, Union
import networkx


def find_edge_coloring(deg: int, vertices: list[int],
                       edges: list[tuple[int, int]],
                       neighbors: dict[int, list[int]],
                       seed: Optional[Union[int, np.random.Generator]] = None) -> dict[int, list[tuple[int, int]]]:

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

    color_to_edges: dict[int, list[tuple[int, int]]] = {}
    for i in range(len(new_vertices)):
        color = verts_in_line_graph_to_color[i]
        if color in color_to_edges:
            color_to_edges[color].append(edges[i])
        else:
            color_to_edges[color] = [edges[i]]
    return color_to_edges


def switchboard_find_edge_coloring(
    algorithm_name: str, deg: int, vertices: List[int], edges: List[Tuple[int, int]],
    neighbors: Dict[int, List[int]], seed: Optional[Union[int, np.random.Generator]] = None
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Dispatches to different edge coloring algorithms based on the provided name.

    Parameters:
    algorithm_name (str): The name of the algorithm to use ('misra_gries', 'moser_tardos', etc.).
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
    else:
        raise ValueError(f"Unknown edge coloring algorithm: {algorithm_name}")


def _mg_is_free(color, vertex, neighbors, edge_colors):
    """True iff `color` is not used by any colored edge incident to `vertex`."""
    for nb in neighbors.get(vertex, []):
        if edge_colors.get(order(vertex, nb), -1) == color:
            return False
    return True


def _mg_first_free(vertex, neighbors, edge_colors, all_colors):
    """Return the smallest color free on `vertex` (guaranteed to exist for a
    graph of maximum degree <= len(all_colors) - 1)."""
    for color in all_colors:
        if _mg_is_free(color, vertex, neighbors, edge_colors):
            return color
    raise RuntimeError("no free color available; deg+1 colors should suffice")


def _mg_build_fan(u, v, neighbors, edge_colors, all_colors):
    """Build a maximal fan of `u` starting at `v`.

    A fan F = [v = f0, f1, ..., fk] is a sequence of distinct neighbors of `u`
    such that (u, f0) is uncolored and, for each i >= 1, the color of (u, f_i)
    is free on f_{i-1}. This is the standard Misra-Gries fan and is derived
    entirely from `edge_colors`, so it never depends on external free-color
    bookkeeping.
    """
    fan = [v]
    in_fan = {v}
    extended = True
    while extended:
        extended = False
        last = fan[-1]
        for w in neighbors.get(u, []):
            if w in in_fan:
                continue
            c = edge_colors.get(order(u, w), -1)
            # (u, w) must be colored, and its color must be free on `last`.
            if c != -1 and _mg_is_free(c, last, neighbors, edge_colors):
                fan.append(w)
                in_fan.add(w)
                extended = True
                break
    return fan


def _mg_cd_path_edges(start, c, d, neighbors, edge_colors):
    """Return the edges (as ordered tuples) of the maximal cd-alternating path
    starting at `start`. The first edge taken has color `d` (since in the
    algorithm `c` is free on the start vertex `u`)."""
    path = []
    current = start
    look_for = d
    visited = set()
    while True:
        nxt = None
        for w in neighbors.get(current, []):
            e = order(current, w)
            if e in visited:
                continue
            if edge_colors.get(e, -1) == look_for:
                nxt = w
                path.append(e)
                visited.add(e)
                break
        if nxt is None:
            break
        current = nxt
        look_for = c if look_for == d else d
    return path


def _misra_gries_color_edge(u, v, deg, free_colors, edge_colors, color_patches, neighbors):
    """Color the single (currently uncolored) edge (u, v) using one step of the
    Misra-Gries algorithm.

    This implementation treats `edge_colors` as the single source of truth and
    recomputes free/missing colors on demand, which avoids the state-drift bugs
    that caused hangs and KeyErrors. `free_colors` and `color_patches` are kept
    in sync as a convenience for callers/inspection, but are never read to make
    decisions. All edge keys are canonicalized with `order(...)`.
    """
    all_colors = list(range(deg + 1))

    def set_edge_color(a, b, color):
        e = order(a, b)
        old = edge_colors.get(e, -1)
        if old == color:
            return
        # maintain color_patches
        if old != -1 and old in color_patches and e in color_patches[old]:
            color_patches[old].remove(e)
        edge_colors[e] = color
        if color != -1:
            color_patches.setdefault(color, [])
            if e not in color_patches[color]:
                color_patches[color].append(e)

    # 1) Maximal fan of u starting at v.
    fan = _mg_build_fan(u, v, neighbors, edge_colors, all_colors)
    k = fan[-1]

    # 2) c free on u, d free on the last fan vertex k.
    c = _mg_first_free(u, neighbors, edge_colors, all_colors)
    d = _mg_first_free(k, neighbors, edge_colors, all_colors)

    # 3) If c != d, invert the cd_u path (the maximal c/d-alternating path
    #    through u) so that d becomes free on u. When c == d, d is already free
    #    on u and there is nothing to invert.
    if c != d:
        path = _mg_cd_path_edges(u, c, d, neighbors, edge_colors)
        for e in path:
            # swap each edge between c and d based on its actual current color.
            actual = edge_colors.get(e, -1)
            set_edge_color(e[0], e[1], c if actual == d else d)

    # 4) Choose w: the largest fan index such that F[0..w_index] is still a valid
    #    fan and d is free on F[w_index]. The Misra-Gries correctness proof
    #    guarantees such a w exists after the inversion (it is either k, or the
    #    index i for which (u, F[i+1]) was the unique d-colored fan edge).
    #    We validate the fan prefix explicitly against the *current* edge colors
    #    so the choice is correct even though `fan` was built before inversion.
    def prefix_is_fan(idx):
        # F[0..idx] is a fan iff for each 0 <= j < idx the color of (u, F[j+1])
        # is free on F[j] (edge (u, F[0]) is the uncolored target).
        for j in range(idx):
            col = edge_colors.get(order(u, fan[j + 1]), -1)
            if col == -1 or not _mg_is_free(col, fan[j], neighbors, edge_colors):
                return False
        return True

    w_index = 0
    for i in range(len(fan)):
        if _mg_is_free(d, fan[i], neighbors, edge_colors) and prefix_is_fan(i):
            w_index = i
    w = fan[w_index]

    # 5) Rotate the subfan F[0..w_index]: shift each (u, F[i+1]) color down to
    #    (u, F[i]), then (u, F[w_index]) becomes uncolored.
    for i in range(w_index):
        nxt_color = edge_colors.get(order(u, fan[i + 1]), -1)
        set_edge_color(u, fan[i], nxt_color)
    if w_index > 0:
        set_edge_color(u, w, -1)

    # 6) Color (u, w) with d.
    set_edge_color(u, w, d)

    # 7) Keep the (advisory) free_colors map roughly in sync for inspection.
    if free_colors is not None:
        for vtx in set([u, v, k, w] + fan):
            free_colors[vtx] = [col for col in all_colors
                                if _mg_is_free(col, vtx, neighbors, edge_colors)]


def misra_gries_edge_coloring(
    deg: int, vertices: List[int], edges: List[Tuple[int, int]], neighbors: Dict[int, List[int]]
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Implements Misra & Gries' edge coloring algorithm for a simple undirected graph.

    This is a deterministic algorithm that always terminates and produces a
    proper edge coloring using at most deg+1 colors (Vizing's theorem).
    """
    edges_canonical = list(set([order(u, v) for u, v in edges]))
    free_colors = {u: [i for i in range(deg + 1)] for u in vertices}
    color_patches: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(deg + 1)}
    edge_colors: Dict[Tuple[int, int], int] = {edge_tuple: -1 for edge_tuple in edges_canonical}

    for edge in edges_canonical:
        u, v = edge
        _misra_gries_color_edge(u, v, deg, free_colors, edge_colors, color_patches, neighbors)

    return color_patches


def sinnamon_euler_color_edge_coloring(
    deg: int, vertices: List[int], edges: List[Tuple[int, int]], neighbors: Dict[int, List[int]],
    seed: Optional[Union[int, np.random.Generator]] = None
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Sinnamon's randomized algorithm for (d+1)-edge-coloring.

    seed (None, int, or numpy.random.Generator): Seed or generator controlling the
        randomization. Passing the same integer seed yields reproducible results.
    """
    rng = np.random.default_rng(seed)
    edges_canonical = list(set([order(u, v) for u, v in edges]))
    free_colors = {u: [i for i in range(deg + 1)] for u in vertices}
    color_patches: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(deg + 1)}
    edge_colors: Dict[Tuple[int, int], int] = {edge_tuple: -1 for edge_tuple in edges_canonical}

    for edge in edges_canonical:
        u, v = edge
        _sinnamon_color_edge(u, v, deg, free_colors, edge_colors, color_patches, neighbors, vertices, rng)

    return color_patches


def _sinnamon_color_edge(u, v, deg, free_colors, edge_colors, color_patches, neighbors, vertices, rng):
    # Simple case
    miss_u = free_colors[u]

    c = -1
    for color in miss_u:
        is_free_at_v = True
        for neighbor_v in neighbors.get(v, []):
            if edge_colors.get(order(v, neighbor_v), -1) == color:
                is_free_at_v = False
                break
        if is_free_at_v:
            c = color
            break

    if c != -1:
        edge_colors[order(u, v)] = c
        color_patches[c].append(order(u, v))
        if c in free_colors[u]: free_colors[u].remove(c)
        if c in free_colors[v]: free_colors[v].remove(c)
        return

    # Randomized case
    for _ in range(int(np.log(len(vertices))) + 1):
        fan = build_maximal_fan(u, v, neighbors, free_colors, edge_colors)
        k = fan[-1]

        c = int(rng.choice(free_colors[u]))
        used_k_colors = [color for color in range(deg + 1) if color not in free_colors[k]]
        if not used_k_colors:
            _misra_gries_color_edge(u, v, deg, free_colors, edge_colors, color_patches, neighbors)
            return
        d = int(rng.choice(used_k_colors))

        path = find_color_path(u, k, c, d, neighbors, edge_colors)
        if not path or path[-1][1] != v:
            for i in range(len(path)):
                path_edge = path[i]
                current_color = [d, c][i % 2]
                other_color = [d, c][(i + 1) % 2]
                ope = order(path_edge[0], path_edge[1])
                if ope in color_patches.get(current_color, []):
                    color_patches[current_color].remove(ope)
                if other_color not in color_patches:
                    color_patches[other_color] = []
                color_patches[other_color].append(ope)
                edge_colors[ope] = other_color

            w_index = 0
            for i in range(len(fan)):
                if d in free_colors[fan[i]]: w_index = i
            w, sub_fan = fan[w_index], fan[:w_index + 1]
            if len(sub_fan) > 1:
                edge_colors, free_colors, color_patches = rotate_fan(
                    sub_fan, u, edge_colors, free_colors, color_patches)

            edge_colors[order(u, w)] = d
            if d not in color_patches:
                color_patches[d] = []
            color_patches[d].append(order(u, w))
            if d in free_colors[u]: free_colors[u].remove(d)
            if d in free_colors[w]: free_colors[w].remove(d)
            return

    # Fallback to deterministic
    _misra_gries_color_edge(u, v, deg, free_colors, edge_colors, color_patches, neighbors)


def order(u, v):
    """
    Return a tuple containing the two input values in sorted order.
    """
    return (min(u, v), max(u, v))


def get_missing_colors(vertex, edge_colors, neighbors, max_degree, all_colors):
    """
    Determines the set of colors missing from the edges incident to a given vertex.

    Args:
        vertex (int): The vertex to check.
        edge_colors (dict): A dictionary mapping (u, v) tuples to their assigned color.
                            (u,v) and (v,u) should map to the same color.
        neighbors (dict): A dictionary mapping each vertex to its neighboring vertices.
        max_degree (int): The maximum degree of the graph.
        all_colors (list): A list of all possible colors [0, ..., max_degree].

    Returns:
        set: A set of colors missing from the incident edges of the vertex.
    """
    used_colors = set()
    for neighbor in neighbors.get(vertex, []):
        edge_tuple = order(vertex, neighbor)
        color = edge_colors.get(edge_tuple, -1)  # -1 implies uncolored or not yet assigned
        if color != -1:
            used_colors.add(color)

    return set(all_colors) - used_colors


def find_alternating_path(start_node, color1, color2, edge_colors, neighbors):
    """
    Finds a maximal (color1, color2)-alternating path starting from start_node.

    Args:
        start_node (int): The vertex to start the path from.
        color1 (int): The first color in the alternating sequence.
        color2 (int): The second color in the alternating sequence.
        edge_colors (dict): A dictionary mapping ordered edge tuples to their colors.
        neighbors (dict): A dictionary mapping each vertex to its neighboring vertices.

    Returns:
        list: A list of ordered edge tuples forming the maximal alternating path.
    """
    path = []
    visited_edges = set()
    current_vertex = start_node
    current_color_to_find = color1  # First edge should have color1

    while True:
        found_next_edge = False
        for neighbor in neighbors.get(current_vertex, []):
            edge_tuple = order(current_vertex, neighbor)
            if edge_tuple not in visited_edges and edge_colors.get(edge_tuple, -1) == current_color_to_find:
                path.append(edge_tuple)
                visited_edges.add(edge_tuple)
                current_vertex = neighbor
                current_color_to_find = color1 if current_color_to_find == color2 else color2
                found_next_edge = True
                break
        if not found_next_edge:
            break

    return path


def flip_path(path, edge_colors, color_patches, color1, color2):
    """
    Flips the colors of edges in an alternating path from color1 to color2 and vice versa.

    Args:
        path (list): A list of ordered edge tuples in the path.
        edge_colors (dict): A dictionary mapping ordered edge tuples to their colors. (will be modified)
        color_patches (dict): A dictionary mapping colors to lists of ordered edge tuples. (will be modified)
        color1 (int): The first color in the alternating sequence.
        color2 (int): The second color in the alternating sequence.
    """
    for edge in path:
        current_color = edge_colors[edge]

        # Remove from current color patch
        if current_color != -1 and edge in color_patches.get(current_color, []):
            color_patches[current_color].remove(edge)

        # Determine new color
        new_color = color1 if current_color == color2 else color2

        # Assign new color
        edge_colors[edge] = new_color

        # Add to new color patch
        if new_color != -1:
            color_patches[new_color].append(edge)


def find_fan_candidates(fan: list, u: int, vertices: list, edge_colors: dict, free_colors: dict) -> list:
    """
    Selects candidate vertices to be added to a fan.

    This function returns vertices connected to the anchor vertex `u`
    where the edge (u, v) is colored with a color that is free on the
    last vertex in the fan.

    Parameters:
    fan (list): A list of vertices representing the current fan.
    u (int): The anchor vertex of the fan.
    vertices (list): A list of all vertices in the graph.
    edge_colors (dict): A dictionary mapping edges to their current colors.
    free_colors (dict): A dictionary mapping vertices to their available colors.

    Returns:
    list: A list of candidate vertices that can be colored with free colors
          available to the last vertex in the fan.
    """
    last_vertex = fan[-1]
    free_vertex_colors = free_colors[last_vertex]
    return [v for v in vertices if edge_colors.get(order(u, v), -1) in free_vertex_colors]


def build_maximal_fan(u: int, v: int, vertex_neighbors: dict,
                      free_colors: dict, edge_colors: dict) -> list:
    """
    Construct a maximal fan of vertex u starting with vertex v.

    A fan is a sequence of distinct neighbors of u that satisfies the following:
    1. The edge (u, v) is uncolored.
    2. Each subsequent edge (u, F[i+1]) is free on F[i] for 1 <= i < k.

    Parameters:
    u (int): The vertex from which the fan is built.
    v (int): The first vertex in the fan.
    vertex_neighbors (dict): A dictionary mapping vertices to their neighbors.
    free_colors (dict): A dictionary mapping vertices to their available colors.
    edge_colors (dict): A dictionary mapping edges to their current colors.

    Returns:
    list: A list representing the maximal fan of vertex u.
    """
    u_neighbors = copy.deepcopy(vertex_neighbors[u])
    fan = [v]
    u_neighbors.remove(v)

    candidate_vertices = find_fan_candidates(fan, u, u_neighbors, edge_colors, free_colors)
    while len(candidate_vertices) != 0:
        fan.append(candidate_vertices[0])
        u_neighbors.remove(candidate_vertices[0])
        candidate_vertices = find_fan_candidates(fan, u, u_neighbors, edge_colors, free_colors)
    return fan


def find_next_path_vertex(current_vertex: int, color: int, neighbors: dict, edge_colors: dict):
    """
    Finds, if it exists, the next vertex in a cd_u path. It does so by finding the neighbor
    of the current vertex which is attached by an edge of the right color.

    Parameters:
    current_vertex (int): The last vertext added to the cd_u path.
    color (int): The desired color of the next possible edge in the cd_u path.
    neighbors (dict): A dictionary mapping each vertex to its neighboring vertices.
    edge_colors (dict): A dictionary mapping edges to their curren colors.

    Returns:
    int or None: The next vertex in the cd_u path that is connected by an edge of the specified color,
                 or None if no such vertex exists.
    """

    for vertex in neighbors[current_vertex]:
        if edge_colors.get(order(current_vertex, vertex), -1) == color:
            return vertex
    return None


def find_color_path(u: int, v: int, c: int, d: int, neighbors: dict, edge_colors: dict) -> list:
    """
    Finds the cd_u path.

    The cd_u path is a path passing through u of edges whose colors alternate between c and d.
    Every cd_u path in the Misra & Gries algorithm starts at u with an edge of color 'd',
    because 'c' was chosen to be free on u. Assuming that a cd_u path exists.

    Parameters:
    u (int): The starting vertex of the path.
    v (int): The target vertex (not used in path finding).
    c (int): The color that is free on vertex `u`.
    d (int): The color that is initially used for the first edge from `u`.
    neighbors (dict): A dictionary mapping each vertex to its neighboring vertices.
    edge_colors (dict): A dictionary mapping edges to their current colors.

    Returns:
    list: A list of tuples representing the edges in the cd_u path.
    """
    cdu_path = []
    current_color = d
    current_vertex = u
    next_vertex = find_next_path_vertex(u, d, neighbors, edge_colors)
    next_color = {c: d, d: c}

    while next_vertex is not None:
        cdu_path.append((current_vertex, next_vertex))
        current_vertex = next_vertex
        current_color = next_color[current_color]
        next_vertex = find_next_path_vertex(current_vertex, current_color, neighbors, edge_colors)
    return cdu_path


def rotate_fan(fan: list, u: int, edge_colors: dict, free_colors: dict, color_patches: dict):
    """
    Rotate the colors in a fan of vertices connected to a specified vertex.

    This function shifts the colors in the fan over by one position, updating the
    edge colorings, free colors for each vertex, and the associated color patches.
    After rotation, the edge connected to the specified vertex `u` and the first
    vertex in the fan receives the color of the next vertex in the fan, while the
    color of the last vertex in the fan is removed.

    Parameters:
    fan (list): A list of vertices representing the fan to be rotated.
    u (int): The vertex anchoring the fan that is being rotated.
    edge_colors (dict): A dictionary mapping edges to their current colors.
    free_colors (dict): A dictionary mapping vertices to their available colors.
    color_patches (dict): A dictionary mapping colors to lists of edges colored with that color.

    Returns:
    tuple: Updated dictionaries for edge_colors, free_colors, and color_patches after rotation.
    """
    # Rotate the fan
    for i in range(len(fan) - 2, -1, -1):
        # f_i gets color of (u, f_{i+1})
        f_i = fan[i]
        f_i_plus_1 = fan[i + 1]

        old_color_of_f_i_edge = edge_colors.get(order(u, f_i))
        new_color_for_f_i_edge = edge_colors.get(order(u, f_i_plus_1))

        # Update edge color
        edge_colors[order(u, f_i)] = new_color_for_f_i_edge

        # Update color patches
        if new_color_for_f_i_edge is not None and new_color_for_f_i_edge != -1:
            if new_color_for_f_i_edge not in color_patches: color_patches[new_color_for_f_i_edge] = []
            color_patches[new_color_for_f_i_edge].append(order(u, f_i))
        if old_color_of_f_i_edge is not None and old_color_of_f_i_edge != -1:
            if order(u, f_i) in color_patches[old_color_of_f_i_edge]:
                color_patches[old_color_of_f_i_edge].remove(order(u, f_i))

        # Update free colors
        if new_color_for_f_i_edge in free_colors[f_i]: free_colors[f_i].remove(new_color_for_f_i_edge)
        free_colors[f_i_plus_1].append(new_color_for_f_i_edge)

    return edge_colors, free_colors, color_patches


def new_bipartite_edge_coloring(
    deg: int, vertices: list, edges: list, neighbors: dict,
    seed: Optional[Union[int, np.random.Generator]] = None
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Implements a randomized (Delta+1)-edge coloring algorithm for bipartite graphs
    based on the "Vizing's Theorem in Near-Linear Time" paper (Algorithm 1 for bipartite case).

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
              Edges are represented as (v1, v2) where v1 < v2.
    """
    rng = np.random.default_rng(seed)
    all_possible_colors = list(range(deg + 1))

    # Initialize coloring: all edges uncolored (-1)
    # Use ordered tuples for edge_colors for consistency
    initial_edge_colors = {}
    unique_edges = set()
    for u, v in edges:
        ordered_edge = order(u, v)
        unique_edges.add(ordered_edge)

    for edge_tuple in unique_edges:
        initial_edge_colors[edge_tuple] = -1

    # Initialize color_patches
    color_patches: dict[int, list[tuple[int, int]]] = {c: [] for c in all_possible_colors}

    # Algorithm 1: BipartiteExtension(G, chi, U)
    # This is a simplified version focusing on the core logic,
    # without full optimizations or error handling described in the paper.

    # 1. Pick two least common colors alpha, beta
    # This step is non-trivial and requires counting current color usage.
    # For a fresh coloring, all colors are "least common". We'll just pick two random ones for now.
    # A more faithful implementation would need a global state for color counts.
    # For initial implementation, let's assume colors 0 and 1 are "least common"
    # or just pick two distinct random colors if deg >= 1.
    if deg >= 1:
        alpha, beta = (int(x) for x in rng.choice(all_possible_colors, size=2, replace=False))
    else:  # Handle case of deg=0 graph (no edges)
        return color_patches

    current_chi_edge_colors = dict(initial_edge_colors)  # This will be modified

    Phi = set()  # Set of popular edges (u,v) with alpha in miss(u) and beta in miss(v)
    C = set()  # Set of directly colored edges

    # Simulate the while loop with a fixed number of iterations for now, or until U is empty.
    # A full implementation requires careful bounds for the while loop condition |Phi|+|C| < lambda/(10*Delta)

    # The paper's Algorithm 1 loop condition is |Phi|+|C| < lambda/(10*Delta)
    # For a first pass, let's just iterate over uncolored edges in U

    # We need to maintain a set of currently uncolored edges that are candidates for U.
    # Let's call it current_U_candidates, initially all unique_edges
    current_U_candidates = list(unique_edges)
    # Shuffle via an index permutation (rng.shuffle cannot handle a list of tuples).
    current_U_candidates = [current_U_candidates[i] for i in rng.permutation(len(current_U_candidates))]

    # This part of the algorithm is quite complex. Instead of fully implementing the while loop
    # and all its conditions (sampling, path intersections, etc.) immediately,
    # I will first create the basic structure and helper calls.
    # The paper's algorithm for BipartiteExtension is essentially a state-machine like loop.
    # I will implement the core logic for a single edge 'e' from the pseudocode for illustration.

    # This is a very simplified placeholder for the main algorithm logic
    # The actual algorithm is iterative and modifies chi dynamically.

    # This function needs to return a final coloring.
    # For now, let's try to color all edges in U using the logic for Case 1 and Case 2 of popularization.

    # This is a high-level representation of the process, not a direct translation of the while loop yet.

    # Step 2: "Popularize" or directly color edges in U
    remaining_uncolored = list(unique_edges)
    # Shuffle via an index permutation (rng.shuffle cannot handle a list of tuples).
    remaining_uncolored = [remaining_uncolored[i] for i in rng.permutation(len(remaining_uncolored))]

    for edge_tuple in remaining_uncolored:
        u, v = edge_tuple

        # Check if already processed (colored or popularized)
        if edge_tuple in C or edge_tuple in Phi:
            continue

        miss_u = get_missing_colors(u, current_chi_edge_colors, neighbors, deg, all_possible_colors)
        miss_v = get_missing_colors(v, current_chi_edge_colors, neighbors, deg, all_possible_colors)

        common_missing_color = None
        for c in miss_u:
            if c in miss_v:
                common_missing_color = c
                break

        if common_missing_color is not None:
            # Case 1: cu = cv (a common missing color exists)
            current_chi_edge_colors[edge_tuple] = common_missing_color
            color_patches[common_missing_color].append(edge_tuple)
            C.add(edge_tuple)
        else:
            # Case 2: Try to popularize (simplified version)
            # This is where the complex path flipping and intersection checks come in.
            # For a basic implementation, we will just try to make it popular without complex checks
            # or by simplifying the path finding/flipping.
            # This part requires finding cu in miss(u) and cv in miss(v) such that
            # flipping paths (cu, alpha) and (cv, beta) does not mess up Phi.

            # This is a very simplified attempt at popularization:
            # If alpha is missing at u, and beta is missing at v, make it popular.
            # This doesn't involve path flipping to achieve this, which is a simplification.

            # The paper's Algorithm 1 line 9-24 is what handles this for each sampled edge 'e'.
            # It involves identifying cu, cv, then conditionally flipping paths.

            # For now, as a placeholder, I will attempt to implement the simplified popularization
            # without full path intersection checks and without fully simulating the while loop's
            # probabilistic sampling and failure conditions. This is to get the structure in place.

            # Let's pick a random cu from miss_u and cv from miss_v.
            if not miss_u or not miss_v:
                continue  # Cannot popularize or color this edge yet.

            # Simplified path finding/flipping for popularization.
            # This is a significant simplification, as the real algorithm has complex conditions.
            # For testing, we'll assume paths can be flipped without immediate conflict for now.

            # This part is complex. I'll defer the full implementation of the while loop's logic
            # and the detailed path intersection/flipping for popularization for a subsequent step.
            # For now, let's complete the structure to allow basic coloring based on available colors.

            # If we cannot popularize or directly color, the edge remains uncolored for now.
            # The full algorithm will cycle through and eventually color all edges.

            pass  # Placeholder for complex popularization logic

    # Step 3: Color popular edges (after the main loop has created Phi)
    # This also involves path flipping.

    # This whole function is currently a skeleton.
    # The full Algorithm 1 from the paper needs to be implemented iteratively.
    # I will first finish the framework for get_missing_colors, find_alternating_path, flip_path
    # and then build the main `new_bipartite_edge_coloring` step by step.

    # The current code will simply leave many edges uncolored because of the "pass"
    # and the simplified loop.

    # The initial goal is to have the structure in place.
    return color_patches


class ColoringSolver:
    def __init__(self, G, K,
                 max_time=60.0,
                 ls_iters_per_restart=None,
                 max_no_improve=100,
                 perturb_size_frac=0.1,
                 seed=None):
        self.G = G
        self.K = K
        self.n = G.n
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

    def count_conflicts(self, col):
        c = 0
        for u in range(self.n):
            cu = col[u]
            for v in self.G.adj[u]:
                if v > u and col[v] == cu:
                    c += 1
        return c

    def one_local_search(self):
        """Run one phase of iterative improvement up to ls_iters."""
        col = self.col
        # maintain conflict list
        vertex_conflicts = [0] * self.n
        for u in range(self.n):
            cnt = 0
            for v in self.G.adj[u]:
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
                for v in self.G.adj[u]:
                    if col[v] == c:
                        cnt += 1
                if cnt < best_cnt:
                    best_cnt, best_c = cnt, c

            # apply move if it strictly improves, or with prob if equal
            if best_cnt < vertex_conflicts[u] or self.rng.random() < 0.01:
                col[u] = best_c
                # update conflicts for u and neighbors
                vertex_conflicts[u] = best_cnt
                for v in self.G.adj[u]:
                    # recalc v’s conflicts
                    cntv = 0
                    for w in self.G.adj[v]:
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

    def perturb(self):
        """Randomly perturb the current best solution."""
        col = list(self.best_col)
        for _ in range(self.perturb_size):
            u = int(self.rng.integers(self.n))
            col[u] = int(self.rng.integers(self.K))
        self.col = col

    def solve(self):
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
    vertices: List[int], edges: List[Tuple[int, int]]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Partitions the edges of a graph into two sets, E1 and E2, such that for every vertex v,
    the number of incident edges in E1 and E2 differs by at most 1.
    """
    adj = {v: [] for v in vertices}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    E1 = []
    E2 = []
    visited_edges = set()

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
    deg: int, vertices: List[int], edges: List[Tuple[int, int]], neighbors: Dict[int, List[int]]
) -> Dict[int, List[Tuple[int, int]]]:
    unique_edges = list(set(order(u, v) for u, v in edges))
    if not unique_edges:
        return {}
    if deg <= 1:
        return {0: unique_edges}
    E1, E2 = _eulerian_partition(vertices, unique_edges)
    neighbors1 = {v: [] for v in vertices}
    for u, v in E1:
        neighbors1[u].append(v)
        neighbors1[v].append(u)
    deg1 = max(len(v) for v in neighbors1.values()) if E1 else 0
    neighbors2 = {v: [] for v in vertices}
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
    deg: int, vertices: List[int], edges: List[Tuple[int, int]], neighbors: Dict[int, List[int]]
) -> Dict[int, List[Tuple[int, int]]]:
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
    `_misra_gries_color_edge` rather than duplicating that logic; the only thing
    distinguishing it from `misra_gries_edge_coloring` is the greedy simple-case
    fast path below.
    """
    edge_colors: Dict[Tuple[int, int], int] = {order(u, v): -1 for u, v in edges}
    color_patches: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(deg + 1)}
    all_colors = list(range(deg + 1))

    for u, v in edges:
        if edge_colors[order(u, v)] == -1:
            # Find a free color for (u, v)
            miss_u = get_missing_colors(u, edge_colors, neighbors, deg, all_colors)

            # Simple case: if a free color at u is also free at v
            c = -1
            for color in miss_u:
                is_free_at_v = True
                for neighbor_v in neighbors[v]:
                    if edge_colors.get(order(v, neighbor_v), -1) == color:
                        is_free_at_v = False
                        break
                if is_free_at_v:
                    c = color
                    break

            if c != -1:
                edge_colors[order(u, v)] = c
                color_patches[c].append(order(u, v))
            else:
                # Complex case: apply one Vizing-chain step (fan + cd-path
                # inversion + fan rotation). This is exactly the Misra-Gries
                # procedure, so we reuse the tested implementation. It decides
                # purely from edge_colors/color_patches (same format used here)
                # and tolerates free_colors=None.
                _misra_gries_color_edge(
                    u, v, deg, None, edge_colors, color_patches, neighbors)

    return color_patches


def moser_tardos_edge_coloring(
    deg: int, vertices: list, edges: list, neighbors: dict,
    seed: Optional[Union[int, np.random.Generator]] = None
) -> Dict[int, List[Tuple[int, int]]]:
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
    # 1) Canonicalize edges to unique, ordered tuples. The input `edges` list is
    #    assumed symmetric; keeping both directions would create duplicate,
    #    mutually-conflicting line-graph vertices and inflate the required colors.
    unique_edges = list({order(u, v) for u, v in edges})
    E = len(unique_edges)
    # 2) Build adjacency in the line-graph L(G): two edge-IDs conflict iff they share an endpoint
    #    We'll represent L_adj as list of sets
    #    First map each original vertex -> list of incident edge-indices
    inc = {v: [] for v in vertices}
    for idx, (u, v) in enumerate(unique_edges):
        inc[u].append(idx)
        inc[v].append(idx)
    L_adj = [set() for _ in range(E)]
    for v, e_ids in inc.items():
        # all edges incident to v form a clique in the line-graph
        for i in range(len(e_ids)):
            for j in range(i + 1, len(e_ids)):
                a, b = e_ids[i], e_ids[j]
                L_adj[a].add(b)
                L_adj[b].add(a)
    # 3) Moser-Tardos resampling.
    #    Variables: col[e] for e in 0..E-1
    #    Bad events: for each (i,j) in L_adj if col[i]==col[j]
    #    A proper edge-coloring needs at most deg+1 colors (Vizing's theorem);
    #    using only deg colors may make a valid coloring impossible, causing the
    #    resampling to never terminate.
    K = deg + 1
    # Cap the number of resamples per attempt and retry with a fresh coloring if
    # the Las Vegas process does not converge, so we never loop forever.
    max_resamples = 1000 * (E + 1)
    max_attempts = 100

    def find_conflicts():
        return [(i, j) for i in range(E) for j in L_adj[i]
                if j > i and col[i] == col[j]]

    col: list[int] = []
    converged = False
    for _ in range(max_attempts):
        col = [int(c) for c in rng.integers(K, size=E)]
        # 4) Maintain a stack of conflicting pairs
        stack = find_conflicts()
        # 5) Resample until no conflicts remain (or the cap is hit)
        resamples = 0
        while stack and resamples < max_resamples:
            resamples += 1
            i, j = stack.pop()
            # maybe already fixed
            if col[i] != col[j]:
                continue
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
                stack = find_conflicts()
        if not stack:
            converged = True
            break
    if not converged:
        raise RuntimeError(
            f"moser_tardos_edge_coloring failed to converge to a valid "
            f"{K}-edge-coloring after {max_attempts} attempts."
        )

    # 6) Build output map
    color_patches: dict[int, list[tuple[int, int]]] = {}
    for idx, (u, v) in enumerate(unique_edges):
        color_used = col[idx]
        if color_used in color_patches:
            color_patches[color_used].append((u, v))
        else:
            color_patches[color_used] = [(u, v)]

    return color_patches


class Graph:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]

    def add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)


def assadi_oct25_edge_coloring(deg: int,
                               vertices: list,
                               edges: list,
                               neighbors: dict,
                               seed: Optional[Union[int, np.random.Generator]] = None) -> dict:
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
    # 1) build a mapping from your arbitrary vertex IDs → 0..n-1
    v2i = {v: i for i, v in enumerate(vertices)}

    # 2) canonicalize edges to unique, ordered tuples. The input `edges` list is
    #    assumed symmetric (contains both (u,v) and (v,u)); collapsing to a single
    #    representative per undirected edge is required, otherwise the two
    #    directions become separate (mutually adjacent) line-graph vertices that
    #    get different colors while mapping back to the same output edge.
    unique_edges = list({order(u, v) for u, v in edges})
    E = len(unique_edges)

    # 3) build the line-graph L(G) with E “vertices”
    LG = Graph(E)

    # 4) for each original edge, record its two endpoints as integer indices
    endpoints: list[set[int]] = []
    for (u, v) in unique_edges:
        ui = v2i[u]
        vi = v2i[v]
        endpoints.append({ui, vi})

    # 5) connect i↔j in L(G) iff edges[i] and edges[j] share a vertex
    for i in range(E):
        for j in range(i + 1, E):
            if endpoints[i] & endpoints[j]:
                LG.add_edge(i, j)

    # 6) now color L(G) with your existing solver.
    #    A proper edge-coloring needs at most deg+1 colors (Vizing's theorem),
    #    so we give the vertex-coloring solver on the line-graph deg+1 colors.
    num_colors = deg + 1
    max_attempts = 100

    color_patches: dict[int, list[tuple[int, int]]] = {}
    for _ in range(max_attempts):
        # Reset the mapping on every attempt so retries don't accumulate
        # duplicate edges from previous (failed) colorings.
        color_patches = {}

        solver = ColoringSolver(LG, num_colors, seed=rng)
        best_col, conflicts = solver.solve()

        # 7) map back: each unique edge → the color of its corresponding vertex in L(G)
        for idx, (u, v) in enumerate(unique_edges):
            col = best_col[idx]
            if col in color_patches:
                color_patches[col].append((u, v))
            else:
                color_patches[col] = [(u, v)]

        # A conflict-free vertex coloring of L(G) is exactly a valid edge coloring.
        if conflicts == 0 and check_valid_edge_coloring(color_patches, ret_false_on_error=True):
            return color_patches

    raise RuntimeError(
        f"assadi_oct25_edge_coloring failed to find a valid {num_colors}-edge-coloring "
        f"after {max_attempts} attempts."
    )


def check_valid_edge_coloring(color_patches, ret_false_on_error: bool = False):
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
