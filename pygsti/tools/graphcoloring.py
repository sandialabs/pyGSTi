import numpy as np
import copy



def find_edge_coloring(deg: int, vertices: list, edges: list, neighbors: dict) -> dict:
    """
    Implements Misra & Gries' edge coloring algorithm for a simple undirected graph.

    This function colors the edges of a simple undirected graph using at most 
    d or d+1 colors, where d is the maximum degree of the graph. The algorithm 
    is optimal (or off by 1) for all simple, undirected graphs, as stated by 
    Vizing's theorem.

    Parameters:
    deg (int): The maximum degree of the graph.
    vertices (list): A list of vertices in the graph.
    edges (list): A list of edges represented as tuples of vertices [assumed to be symmetric, i.e., (u,v) and (v,u) are elements].
    neighbors (dict): A dictionary mapping each vertex to its neighboring vertices.

    Returns:
    color_patches (dict): A dictionary mapping each color to a list of edges colored with that color.
                          Unlike with edges, the items in color_patches are NOT symmetric [i.e., it only contains (v1, v2) for v1 < v2]
    """

    edges = copy.deepcopy(edges)
    free_colors = {u: [i for i in range(deg+1)] for u in vertices} # Keeps track of which colors are free on each vertex
    color_patches = {i: [] for i in range(deg+1)} # Keeps track of the edges (items) that have been assigned a color (keys)
    
    edge_colors = {edge: -1 for edge in edges} # Keeps track of which color (item) an edge has been assigned to (key)
    edges = [list(edge) for edge in edges]

    for edge in edges:
        edge.sort()
    edges = list(set([tuple(edge) for edge in edges]))

    # Loop the edges in G.
    # You will color a new edge each time.
    for edge in edges:
        # Find a maximal fan F of vertex 'u' with F[1] = 'v.'
        u, v = edge
        max_fan = build_maximal_fan(u, v, neighbors, free_colors, edge_colors)
        
        # Pick free colors c and d on u and k, the last vertex in the fan.
        # Find the cd_u path, i.e., the maximal path through u of edges whose colors alternate between c and d.
        k = max_fan[-1]
        c, d = free_colors[u][-1], free_colors[k][-1] # c is free on u, while d is free on the last entry in the fan
        cdu_path = find_color_path(u, k, c, d, neighbors, edge_colors)        
        
        # invert the cd_u path
        for i in range(len(cdu_path)):
            path_edge = cdu_path[i]
            # path should be colored as d, c, d, c, etc... because c was free on u
            current_color = [d, c][i%2]
            other_color = [d, c][(i+1)%2]
            ope = order(path_edge[0], path_edge[1])
            if ope in color_patches[current_color]:
                color_patches[current_color].remove(ope)
                #color_patches[current_color].remove(ope)
            color_patches[other_color].append(ope)
            #color_patches[other_color].append(ope)
            edge_colors[path_edge] = other_color
            edge_colors[(path_edge[1], path_edge[0])] = other_color
        if len(cdu_path) > 0: 
            free_colors[u].remove(c)
            free_colors[u].append(d)
            final_color, final_vertex = edge_colors[cdu_path[-1]], cdu_path[-1][-1]
            free_colors[final_vertex].remove(final_color)
            free_colors[final_vertex].append(list(np.setdiff1d([c, d], [final_color]))[0])        
        
        # Find a subfan of u, F' = F[1:w] for which the color d is free on w.
        w_index = 0
        for i in range(len(max_fan)):
            if d in free_colors[max_fan[i]]: w_index = i
        w, sub_fan = max_fan[w_index], max_fan[:w_index + 1]
        
        # Rotate the subfan. If it exists, then
        # you have now colored the edge (u,v) with whatever color was on (u, F[2])
        if len(sub_fan) > 1: # rotate the subfan
            edge_colors, free_colors, color_patches = rotate_fan(sub_fan, u, edge_colors, free_colors, color_patches)

        # Set the color of (u, w) to d.
        edge_colors[(u, w)] = d
        edge_colors[(w, u)] = d
        color_patches[d].append(order(u, w))
        if d in free_colors[u]:
            free_colors[u].remove(d)
        if d in free_colors[w]:
            free_colors[w].remove(d)

        check_valid_edge_coloring(color_patches)

    return color_patches


def order(u, v):
    """
    Return a tuple containing the two input values in sorted order.

    This function takes two values, `u` and `v`, and returns them as a 
    tuple in ascending order. The smaller value will be the first element 
    of the tuple.

    Parameters:
    u: The first value to be ordered.
    v: The second value to be ordered.

    Returns:
    tuple: A tuple containing the values `u` and `v` in sorted order.
    """
    return (min(u, v), max(u, v))


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
    return [v for v in vertices if edge_colors[(u, v)] in free_vertex_colors]


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
        if edge_colors[(current_vertex, vertex)] == color:
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
    for i in range(len(fan) - 1):
        curr_vertex = fan[i]
        next_vertex = fan[i+1]
        next_color = edge_colors[(u, next_vertex)]
        
        edge_colors[(u, curr_vertex)] = next_color
        edge_colors[(curr_vertex, u)] = next_color
        edge_colors[(u, next_vertex)] = -1
        edge_colors[(next_vertex, u)] = -1

        free_colors[curr_vertex].remove(next_color)
        free_colors[next_vertex].append(next_color)

        color_patches[next_color].append(order(u, curr_vertex))
        color_patches[next_color].remove(order(u, next_vertex))

    return edge_colors, free_colors, color_patches


def check_valid_edge_coloring(color_patches):
    """
    color_patches (dict): A dictionary mapping each color to a list of edges colored with that color.
                          Unlike with edges, the items in color_patches are NOT symmetric [i.e., it only contains (v1, v2) for v1 < v2]
    """
    for c, patch in color_patches.items():
        in_patch = set()
        for pair in patch:
            in_patch.add(pair[0])
            in_patch.add(pair[1])
        if len(in_patch) != 2*len(patch):
            raise ValueError()
    return

