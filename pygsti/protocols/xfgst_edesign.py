import numpy as np
from pygsti.protocols import CircuitListsDesign, HasProcessorSpec
from pygsti.circuits.circuitlist import CircuitList
from pygsti.circuits.circuit import Circuit
from pygsti.circuits.split_circuits_into_lanes import batch_tensor
from pygsti.baseobjs.label import Label, LabelTup
import copy

from pygsti.tools.graphcoloring import *


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
                                       oneq_gstdesign, twoq_gstdesign, randgen) -> tuple:
    """
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
    randgen: A random number generator from numpy.

    Returns:
    tuple: A tuple containing:
        - circuit_lists (list): A list of crosstalk-free GST circuits for each germ power.
        - aux_info (dict): Auxiliary information mapping circuits to their corresponding edges and vertices.
    """

    circuit_lists = [[] for _ in twoq_gstdesign.circuit_lists]
    twoq_idle_label = Label(('Gii',) + twoq_gstdesign.qubit_labels)
    oneq_idle_label = Label(('Gi',) + oneq_gstdesign.qubit_labels)
    mapper_2q: dict[Label, Label] = {twoq_idle_label: twoq_idle_label}
    mapper_1q: dict[Label, Label] = {oneq_idle_label: oneq_idle_label}
    for cl in twoq_gstdesign.circuit_lists:
        for c in cl:
            mapper_2q.update({k:k for k in c._labels})
            mapper_2q[Label(())] = twoq_idle_label
    for cl in oneq_gstdesign.circuit_lists:
        for c in cl:
            mapper_1q.update({k:k for k in c._labels})
            mapper_1q[Label(())] = oneq_idle_label
    assert Label(()) not in mapper_2q.values()
    assert Label(()) not in mapper_1q.values()
    aux_info = {}

    m2q = mapper_2q.copy()
    for k2 in mapper_2q:
        if k2.num_qubits == 1:
            assert isinstance(k2, LabelTup)
            # So we are assuming k2 = Label("Gsingle", x) where x = 0,1.
            tgt = k2[1]
            assert tgt in [0,1]
            tmp = [None, None]
            tmp[tgt] = k2
            # This implies that the correction is only for the single noisy idle gate.
            tmp[1-tgt] = Label("Gi", 1-tgt) # Wrap around will set the tmp correctly.
            # However, we still need to ensure that Label("Gi", 1-tgt) is correct.
            m2q[k2] = tuple(tmp)
            # We are going to be replacing the 1q gate with a parallelized noisy idle and the gate.

    mapper_2q = m2q # Reset here.
    layer_mappers = {1: mapper_1q, 2: mapper_2q} # So layer mappers handles how big each lane is not the length of a circuit.

    num_lines = -1
    global_line_order = None
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
                raise NotImplementedError()
        
            # 2Q GST circuit list is longer
            n_edges = len(edge_set)
            edge_perms = np.tile(np.arange(max_len), (n_edges, 1))
            edge_perms = randgen.permuted(edge_perms, axis=1) # Note that this will permute the columns independently and not in bulk way.

            node_perms = []
            node_perms = np.tile(np.arange(min_len), (min_len, num_batches, 1))
            # truncate to largest size we will need.
            node_perms =randgen.permuted(node_perms, axis=-1).reshape(min_len, num_batches*min_len)[:, :max_len]

            assert edge_perms.shape == (len(edge_set), max_len)
            assert node_perms.shape == (min_len, max_len)
        
            # Check invariants
            edge_line_contributions = 2*edge_perms.shape[0] if edge_perms.size > 0 else 0
            node_line_contributions =   node_perms.shape[0] if node_perms.size > 0 else 0
            curr_num_lines = edge_line_contributions + node_line_contributions
            if num_lines < 0:
                num_lines = curr_num_lines
                global_line_order = tuple(range(num_lines))

            assert num_lines == curr_num_lines
            if edge_perms.size > 0 and node_perms.size > 0:
                assert edge_perms.shape[1] == node_perms.shape[1]
            
            # Form the tensor product circuits, over all qubits.
            for j in range(max_len):
                tensored_lines  = []
                circs_to_tensor = []
                if len(edge_perms):
                    edge_start = 1
                    node_start = 0
                    c = twoq_circuits[edge_perms[0,j]]
                    tensored_lines.append(edge_set[0])  #This may just pick the same value each time if edge_set does not change.
                else:
                    edge_start = 0
                    node_start = 1
                    c = oneq_circuits[node_perms[0,j]]
                    tensored_lines.append((unused_qubits[0],))
                circs_to_tensor.append(c)
                for i in range(edge_start, edge_perms.shape[0]):
                    c2 = twoq_circuits[ edge_perms[i,j] ]
                    circs_to_tensor.append( c2 )
                    tensored_lines.append(edge_set[i])
                for i in range(node_start, node_perms.shape[0]):
                    c2 = oneq_circuits[ node_perms[i,j] ]
                    circs_to_tensor.append( c2 )
                    tensored_lines.append((unused_qubits[i],))
                c_ten = batch_tensor(circs_to_tensor, layer_mappers, global_line_order, tensored_lines)
                for i in range(c_ten.num_layers): # This is just a debugging loop to ensure we have everything labeled.
                    l0 = set(c_ten.layer(i))
                    l1 = set(c_ten.layer_with_idles(i))
                    assert l0 == l1
                circuit_lists[L].append(c_ten)
                aux_info[c_ten] = {'edges': edge_set, 'vertices': unused_qubits} #YOLO

    return circuit_lists, aux_info


def generate_edge_colorings(vertices: list, edges: list) -> list:
    """
    Generate a set of edge colorings for a graph until all edges are colored.

    This function takes an edge set of a simple undirected graph and repeatedly 
    applies the Misra & Gries edge coloring algorithm until every edge is 
    contained in some edge coloring. It returns a dictionary mapping colors 
    to the edges colored with that color.

    Parameters:
    vertices (list): A list of vertices in the graph.
    edges (list): A list of edges represented as tuples (u, v) where u and v 
                are vertices in the graph.

    Returns:
    list: A list of edge colorings (dictionaries whose keys are colors and items are lists colored edges)
    """
    list_of_edge_colorings = []
    uncolored_edges = set(edges)

    while uncolored_edges:
        # Determine which vertices are neighbors in a graph with only uncolored edges
        # Could call find_neighbors here...
        updated_neighbors = {v: [] for v in vertices}
        for u, v in uncolored_edges:
            updated_neighbors[u].append(v)

        # Calculate the maximum degree of the graph
        deg = max(len(updated_neighbors[v]) for v in vertices)

        # Find an edge coloring
        new_color_patches = find_edge_coloring(deg, vertices, list(uncolored_edges), updated_neighbors)
        new_color_patches = {k: v for (k, v) in new_color_patches.items() if len(v) > 0}

        # Update color patches and remove newly colored edges from uncolored_edges
        list_of_edge_colorings.append(new_color_patches)
        for _, edge_list in new_color_patches.items():
            uncolored_edges.difference_update(edge_list)
            uncolored_edges.difference_update([(v,u) for u, v in edge_list]) # need to symmetrize

    return list_of_edge_colorings


def make_xfgst_design(nq_pspec, oneq_gstdesign, twoq_gstdesign, seed=0):
    vertices = nq_pspec.qubit_labels
    edges = nq_pspec.compute_2Q_connectivity().edges()

    # Generate the sub-experiment designs
    edge_colorings = generate_edge_colorings(vertices, edges)
    sub_designs = []
    for i,ec in enumerate(edge_colorings):
        ed = CrosstalkFreeExperimentDesign(nq_pspec, oneq_gstdesign, twoq_gstdesign, ec, seed=(seed+i))
        sub_designs.append(ed)
    
    max_circuitlists_len = max([len(sd.circuit_lists) for sd in sub_designs])
    circuitlists = [[] for _ in range(max_circuitlists_len)]
    for sd in sub_designs:
        for i,cl in enumerate(sd.circuit_lists):
            circuitlists[i].extend(cl)
    cld = CircuitListsDesign(circuit_lists=circuitlists)
    return cld


class CrosstalkFreeExperimentDesign(CircuitListsDesign):
    """
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
    """
    def __init__(self, processor_spec, oneq_gstdesign, twoq_gstdesign, edge_coloring, 
                 circuit_stitcher = stitch_circuits_by_germ_power_only, seed = None):
        """
        Assume that the GST designs have the same Ls.

        TODO: Update the init function so that it handles different circuit stitchers better (i.e., by using stitcher_kwargs, etc.)
        """
        # TODO: make sure idle gates are explicit.
        randgen = np.random.default_rng(seed)
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
                                                            randgen,
        )
        
        CircuitListsDesign.__init__(self, self.circuit_lists, qubit_labels=self.vertices)
