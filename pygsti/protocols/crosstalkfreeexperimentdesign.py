import numpy as np
from pygsti.protocols import CircuitListsDesign, HasProcessorSpec
from pygsti.circuits.circuitlist import CircuitList
from pygsti.circuits.circuit import Circuit
import copy


def find_neighbors(vertices: list, edges: list) -> dict:
    """
    Find the neighbors of each vertex in a graph.

    This function takes a list of vertices and a dictionary of edges, 
    where edges are represented as tuples. It returns a dictionary 
    mapping each vertex to a list of its neighbors.

    Parameters:
    vertices (list): A list of vertices in the graph.
    edges (dict): A symmetric list of the edges in the graph [e.g., (v1, v2) and (v2, v1) are elements].

    Returns:
    dict: A dictionary where each key is a vertex and the value is a 
          list of neighboring vertices.
    """
    neighbors = {v: [] for v in vertices}
    for e in edges:
        neighbors[e[0]].append(e[1])
    return neighbors


def stitch_circuits_by_germ_power_only(color_patches: dict, vertices: list, 
                                       oneq_gstdesign, twoq_gstdesign, randstate: int) -> tuple:
    '''
    Generate crosstalk-free GST circuits by stitching together 1Q and 2Q GST circuits for 
    each color patch.

    This function combines 1Q and 2Q GST circuits based on the specified color patches.
    For each germ power L, it randomizes the order of the 2Q GST circuits and the 1Q GST 
    circuits for each edge and unused qubit. The circuits are then stitched together to 
    form the final circuit lists.

    Parameters:
    color_patches (dict): A dictionary mapping color patches to their corresponding edge sets.
                          A 'color patch' is a set of similarly colored edges in an edge coloring.
    vertices (list): A list of vertices in the graph.
    oneq_gstdesign: A GST edesign containing the 1Q GST circuits.
    twoq_gstdesign: An GST edesign containing the 2Q GST circuit.
    randstate: A random state object used for randomization.

    Returns:
    tuple: A tuple containing:
        - circuit_lists (list): A list of crosstalk-free GST circuits for each germ power.
        - aux_info (dict): Auxiliary information mapping circuits to their corresponding edges and vertices.
    '''

    circuit_lists = [[] for _ in twoq_gstdesign.circuit_lists]
    aux_info = {}

    for patch, edge_set in color_patches.items():
        # This might be broken when edge_set is empty.
        used_qubits = np.array(edge_set).ravel()
        unused_qubits = np.setdiff1d(np.array(vertices), used_qubits)
        assert len(oneq_gstdesign.circuit_lists) == len(twoq_gstdesign.circuit_lists), "Not implemented."

        for L, (oneq_circuits, twoq_circuits) in enumerate(zip(oneq_gstdesign.circuit_lists, twoq_gstdesign.circuit_lists)):   # assumes that they use the same L 
            oneq_len = len(oneq_circuits)
            twoq_len = len(twoq_circuits)

            max_len = max(oneq_len, twoq_len)
            min_len = min(oneq_len, twoq_len)
            num_batches = int(np.ceil(max_len / min_len))

            if oneq_len > twoq_len:
                # vertex_permutations = [randstate.permutation(max_len) for _ in unused_qubits]
                # edge_permutations = [[] for _ in edge_set]
                # for _ in range(num_batches):
                #     for perm in edge_permutations:
                #         perm.extend([randstate.permutation(min_len)])
                # edge_permutations = [mp[:max_len] for mp in edge_permutations]
                raise NotImplementedError()
        
            # 2Q GST circuit list is longer
            edge_permutations = [randstate.permutation(max_len) for _ in edge_set] # Randomize the order in which we place 2Q GST circuits on each edge
            vertex_permutations = [[] for _ in unused_qubits] 
            for _ in range(num_batches):
                for perm in vertex_permutations:
                    perm.extend([randstate.permutation(min_len)])
            vertex_permutations = [mp[:max_len] for mp in vertex_permutations] # Randomize the order in which we place 1Q GST circuits on each isolated qubit
            #                           ^ Before this line executes, len(vertex_permutations[i]) == num_batches for all i,
            #                             and num_batches = ceil(max_len/min_len) <= max_len, so the indexing mp[:max_len] has no effect.
        
            edge_permutations = np.array(edge_permutations)
            vertex_permutations = np.array(vertex_permutations)
            # ^ vertex_permutations.shape is either () or (len(unused_qubits), num_batches, min_len)


            """
            NOTE: I was able to infer that the twoq_gstdesign.sslabels should really be qubit labels by seeing how (oldq, newq)
            were in the iterator that zip'd (twoq_gstdesign.sslabels, other_thing).
            """
            # for j in range(max_len): # range(max(edge_permutations.shape[1], vertex_permutations.shape[1])): 
            dim1 = 0 if (edge_permutations.ndim < 2) else edge_permutations.shape[1]
            dim2 = 0 if (vertex_permutations.ndim < 2) else vertex_permutations.shape[1]
            for j in range(max(dim1, dim2)): 
                # Pick the initial subcircuit
                if len(edge_permutations):
                    c = twoq_circuits.permuted_subcircuitlist( edge_permutations[0,j] )
                    map_dict = {oldq: newq for oldq, newq in zip(twoq_gstdesign.qubit_labels, edge_set[0])}
                    c = c.map_line_labels(map_dict)
                    edge_start = 1
                    vertex_start = 0
                else:
                    # The second component of vertex_permutations should range from 
                    c = oneq_circuits.permuted_subcircuitlist(  vertex_permutations[0,j] )
                    map_dict = {oldq: newq for oldq, newq in zip(oneq_gstdesign.qubit_labels, (unused_qubits[0],))}
                    c = c.map_line_labels(map_dict)
                    edge_start = 0
                    vertex_start = 1
                        
                # Tensor together the other subcircuits
                for i in range(edge_start, edge_permutations.shape[0]):
                    c2 = twoq_circuits.permuted_subcircuitlist(  edge_permutations[i,j] )  # Fix col
                    map_dict = {oldq: newq for oldq, newq in zip(twoq_gstdesign.qubit_labels, edge_set[i])}
                    c2 = c2.map_line_labels(map_dict)
                    c = c.tensor_circuits(c2) # c is already a copy due to map_line_labels above

                for i in range(vertex_start, vertex_permutations.shape[0]):
                    c2 = oneq_circuits.permuted_subcircuitlist( vertex_permutations[i,j] ) # Fix col
                    map_dict = {oldq: newq for oldq, newq in zip(oneq_gstdesign.qubit_labels, (unused_qubits[i],))}
                    c2 = c2.map_line_labels(map_dict)
                    c = c.tensor_circuits(c2) # c is already a copy due to map_line_labels above
                    
                circuit_lists[L].append(c)

                aux_info[c] = {'edges': edge_set, 'vertices': unused_qubits} #YOLO

    return circuit_lists, aux_info


class CrosstalkFreeExperimentDesign(CircuitListsDesign):
    '''
    This class initializes a crosstalk-free GST experiment design by combining 
    1Q and 2Q GST designs based on a specified edge coloring. It assumes that 
    the GST designs share the same germ powers (Ls) and utilizes a specified 
    circuit stitcher to generate the final circuit lists.

    Attributes:
    processor_spec: Specification of the processor, including qubit labels and connectivity.
    oneq_gstdesign: The design for one-qubit GST circuits.
    twoq_gstdesign: The design for two-qubit GST circuits.
    edge_coloring (dict): A dictionary mapping color patches to their corresponding edge sets.
    circuit_stitcher (callable): A function to stitch circuits together (default: stitch_circuits_by_germ_power_only).
    seed (int, optional): Seed for random number generation.

    circuit_lists (list): The generated list of stitched circuits.
    aux_info (dict): Auxiliary information mapping circuits to their corresponding edges and vertices.
    '''
    def __init__(self, processor_spec, oneq_gstdesign, twoq_gstdesign, edge_coloring, 
                 circuit_stitcher = stitch_circuits_by_germ_power_only, seed = None):
        '''
        Assume that the GST designs have the same Ls.

        TODO: Update the init function so that it handles different circuit stitchers better (i.e., by using stitcher_kwargs, etc.)
        '''
        randstate = np.random.RandomState(seed)
        self.processor_spec = processor_spec
        self.oneq_gstdesign = oneq_gstdesign
        self.twoq_gstdesign = twoq_gstdesign
        self.vertices = self.processor_spec.qubit_labels
        self.edges = self.processor_spec.compute_2Q_connectivity().edges()
        self.neighbors = find_neighbors(self.vertices, self.edges)
        self.deg = max([len(self.neighbors[v]) for v in self.vertices])
        self.color_patches = edge_coloring
        self.circuit_stitcher = circuit_stitcher
    
        self.circuit_lists, self.aux_info = circuit_stitcher(self.color_patches, self.vertices, 
                                                            self.oneq_gstdesign, self.twoq_gstdesign, 
                                                            randstate,
        )
        
        CircuitListsDesign.__init__(self, self.circuit_lists, qubit_labels=self.vertices)


'''
Everything below is used to find an edge coloring of a graph.
'''

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
    '''
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
    '''
    last_vertex = fan[-1]
    free_vertex_colors = free_colors[last_vertex]
    return [v for v in vertices if edge_colors[(u, v)] in free_vertex_colors]


def build_maximal_fan(u: int, v: int, vertex_neighbors: dict, 
                      free_colors: dict, edge_colors: dict) -> list:
    '''
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
    '''
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
    '''
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
    '''
    
    for vertex in neighbors[current_vertex]:
        if edge_colors[(current_vertex, vertex)] == color:
            return vertex
    return None
    

def find_color_path(u: int, v: int, c: int, d: int, neighbors: dict, edge_colors: dict) -> list:
    '''
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
    '''
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
    '''
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
    '''
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
    

def find_edge_coloring(deg: int, vertices: list, edges: list, neighbors: dict) -> dict:
    '''
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
    '''

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
            if order(path_edge[0], path_edge[1]) in color_patches[current_color]:
                color_patches[current_color].remove(order(path_edge[0], path_edge[1]))
                #color_patches[current_color].remove((path_edge[1], path_edge[0]))
            color_patches[other_color].append(order(path_edge[0], path_edge[1]))
            #color_patches[other_color].append((path_edge[1], path_edge[0]))
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

    return color_patches
        