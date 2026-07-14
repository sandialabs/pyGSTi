import numpy as np
from collections import defaultdict
from typing import Optional, Union, TYPE_CHECKING
import tqdm as _tqdm

from pygsti.protocols import GateSetTomographyDesign
from pygsti.processors import QubitProcessorSpec
from pygsti.circuits.circuitlist import CircuitList
from pygsti.circuits.split_circuits_into_lanes import batch_tensor
from pygsti.baseobjs.label import Label, LabelTup
import copy

from pygsti.tools.graphcoloring import switchboard_find_edge_coloring


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



def build_layer_mappers(oneq_gstdesign, twoq_gstdesign) -> dict:
    """
    Build the ``layer_mappers`` used by ``batch_tensor`` when stitching 1Q and 2Q
    GST circuits together.

    The returned dict maps a lane size (1 or 2) to a per-label mapper that embeds
    the 1Q/2Q labels into the full tensored state space without introducing implicit
    idle labels. This depends only on the labels appearing in the two designs'
    circuit lists, not on any color patch.

    Parameters
    ----------
    oneq_gstdesign : GateSetTomographyDesign
        Design containing the 1Q GST circuits.
    twoq_gstdesign : GateSetTomographyDesign
        Design containing the 2Q GST circuits.

    Returns
    -------
    dict
        ``{1: mapper_1q, 2: mapper_2q}``.
    """
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
    # layer mappers handles how big each lane is not the length of a circuit.
    return {1: mapper_1q, 2: mapper_2q}


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
    aux_info = {}
    layer_mappers = build_layer_mappers(oneq_gstdesign, twoq_gstdesign)

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


                # This is just a debugging loop to ensure we have everything labeled.
                for i in range(c_ten.num_layers):
                    l0 = set(c_ten.layer(i))
                    l1 = set(c_ten.layer_with_idles(i))
                    assert l0 == l1

                circuit_lists[L].append(c_ten)
                aux_info[c_ten] = {'edges': edge_set, 'vertices': unused_qubits} #YOLO

    return circuit_lists, aux_info


def make_xfgst_design(nq_pspec: QubitProcessorSpec, oneq_gstdesign, twoq_gstdesign, seed=0):
    vertices = nq_pspec.qubit_labels
    edges = nq_pspec.compute_2Q_connectivity().edges()

    # Generate the sub-experiment designs
    
    edges = set(edges)
    neighbors = find_neighbors(vertices, edges)
    # Calculate the maximum degree of the graph
    deg = max(len(neighbors[v]) for v in vertices)
    # "auto" detects canonical topologies (line/ring/grid/torus, as produced by
    # ProcessorSpec(geometry=...)) and uses an optimal closed-form coloring for
    # them, falling back to a generic (deg+1)-color algorithm otherwise.
    edge_coloring = switchboard_find_edge_coloring("auto", deg, vertices, edges, neighbors, seed=seed)

    return CrosstalkFreeExperimentDesign(nq_pspec, oneq_gstdesign, twoq_gstdesign, edge_coloring, seed=(seed+1))


class CrosstalkFreeExperimentDesign(GateSetTomographyDesign):
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
    def __init__(self, processor_spec, oneq_gstdesign: GateSetTomographyDesign, twoq_gstdesign: GateSetTomographyDesign, edge_coloring,
                 circuit_stitcher = None, seed = None, nested: bool=False):
        """
        Assume that the GST designs have the same Ls.

        The default ``circuit_stitcher`` is ``assign_the_designs_with_mapping``,
        which expects the (oneq_circuitlists, twoq_circuitlists, vertices,
        color_patches, layer_mappers, ...) calling convention used below.

        TODO: Update the init function so that it handles different circuit stitchers better (i.e., by using stitcher_kwargs, etc.)
        """
        # TODO: make sure idle gates are explicit.
        if circuit_stitcher is None:
            circuit_stitcher = assign_the_designs_with_mapping
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

        self.circuit_lists = circuit_stitcher(self.oneq_gstdesign,
                                              self.twoq_gstdesign,
                                              self.vertices, self.color_patches,
                                              randgen=randgen, ensure_containment=nested,
                                              )
        # The default stitcher (assign_the_designs_with_mapping) does not produce
        # aux_info; keep the attribute for API compatibility.
        self.aux_info = {}

        super().__init__(processor_spec, self.circuit_lists,qubit_labels=self.vertices, nested=nested)


def ordered_edge_set(edge_set):
    """
    Canonicalize the order of edges, but preserve endpoint orientation.

    If endpoint orientation is meaningful, do NOT sort inside each edge.
    """
    return sorted([tuple(edge) for edge in edge_set])


def patch_lines(edge_set, vertices):
    """
    Return the ordered tensor lines for a patch:
      - first the 2Q edge lines
      - then the 1Q unused-qubit lines
    """
    edge_set = ordered_edge_set(edge_set)

    used_qubits = {
        q
        for edge in edge_set
        for q in edge
    }

    unused_qubits = [
        q
        for q in vertices
        if q not in used_qubits
    ]

    tensored_lines = list(edge_set) + [(q,) for q in unused_qubits]

    return edge_set, unused_qubits, tensored_lines


def make_line_mapper(source_lines, target_lines):
    """
    Construct a state-space-label mapper from source tensor lines to target
    tensor lines.

    Example:
        source_lines = [(0, 1), (4,), (5,)]
        target_lines = [(2, 3), (0,), (1,)]

        returns:
            {0: 2, 1: 3, 4: 0, 5: 1}
    """
    if len(source_lines) != len(target_lines):
        raise ValueError("Source and target line lists have different lengths.")

    mapper = {}

    for src_line, dst_line in zip(source_lines, target_lines):
        if len(src_line) != len(dst_line):
            raise ValueError(
                f"Line arity mismatch: source {src_line}, target {dst_line}"
            )

        for src_label, dst_label in zip(src_line, dst_line):
            if src_label in mapper and mapper[src_label] != dst_label:
                raise ValueError(
                    f"Inconsistent mapping for {src_label}: "
                    f"{mapper[src_label]} versus {dst_label}"
                )

            mapper[src_label] = dst_label

    if len(set(mapper.values())) != len(mapper):
        raise ValueError("Mapper is not one-to-one.")

    return mapper

def build_patch_infos(vertices, color_patches):
    vertices = list(vertices)

    patch_infos = []
    groups = defaultdict(list)

    for patch, edge_set in color_patches.items():
        edge_set, unused_qubits, tensored_lines = patch_lines(edge_set, vertices)

        info = {
            "patch": patch,
            "edge_set": edge_set,
            "unused_qubits": unused_qubits,
            "tensored_lines": tensored_lines,
            "num_edges": len(edge_set),
            "num_unused_qubits": len(unused_qubits),
        }

        key = (len(edge_set), len(unused_qubits))
        groups[key].append(info)
        patch_infos.append(info)

    return patch_infos, groups

def assign_the_designs_with_mapping(
    oneq_gstdesign,
    twoq_gstdesign,
    vertices: list[int],
    color_patches: dict[int, list[tuple[int,int]]],
    debug_check=False,
    randgen: Optional=None,
    ensure_containment: bool=False,
    _layer_mappers_override=None,
):
    """
    Construct crosstalk-free GST circuit lists for each color patch.

    For each germ-power index, this function combines 2Q GST circuits on the edges
    of each color patch with 1Q GST circuits on the vertices not used by that patch.
    Each color patch should contain mutually disjoint edges, so that the resulting
    tensored circuits do not place simultaneous 2Q operations on overlapping qubits.

    Here, a color patch is one color class from an edge coloring of the 2Q
    connectivity graph. For example, for a five-qubit line,

        0 -- 1 -- 2 -- 3 -- 4

    one valid color patch is ``[(0, 1), (2, 3)]``. For that patch, this function
    uses 2Q GST designs on edges ``(0, 1)`` and ``(2, 3)``, and a 1Q GST design on
    the unused qubit ``4``. Another valid patch is ``[(1, 2), (3, 4)]``, with qubit
    ``0`` receiving a 1Q GST design.

    Patches with the same number of 2Q edges and unused qubits share randomized
    role-based schedules. A representative tensored circuit is constructed once
    for each such group and then mapped onto equivalent patches.

    This function does not deduplicate color patches. For example, if both
    ``[(0, 1), (2, 3)]`` and ``[(1, 0), (3, 2)]`` are supplied, both designs are
    generated, even though they differ only by edge orientation.

    Parameters
    ----------
    oneq_gstdesign : GateSetTomographyDesign
        The 1Q GST experiment design.

    twoq_gstdesign : GateSetTomographyDesign
        The 2Q GST experiment design. Must have the same number of germ-power
        groups as ``oneq_gstdesign``.

    vertices : list[int]
        Vertices/qubits in the connectivity graph.

    color_patches : dict[int, list[tuple[int, int]]]
        Mapping from patch/color identifier to the list of disjoint 2Q edges in that patch.
        Each edge is represented as a pair of qubit labels.

    debug_check : bool, optional
        If True, check that the generated tensored circuits contain no implicit idle
        gates. Default is False.

    randgen : numpy.random.Generator, optional
        Random number generator used to randomize circuit assignments across edge
        and qubit slots. If None, uses ``np.random.default_rng(0)``.

    ensure_containment: bool, optional
        If True, ensure that circuitlists[L+1] contains the exact circuits
        from circuitlists[L]. Containment is enforced patch-wise, so the
    output remains patch-major. Default is False. 

    Returns
    -------
    list[list]
        ``circuit_lists[L]`` contains the generated crosstalk-free GST circuits for
        germ-power index ``L``. Within each germ-power group, circuits are ordered
        patch-major according to the input order of ``color_patches``.

    Raises
    ------
    AssertionError
        If ``oneq_gstdesign`` and ``twoq_gstdesign`` do not have the same number
        of germ-power groups.

    NotImplementedError
        If, for any germ-power group, the number of 1Q circuits exceeds the number
        of 2Q circuits.
    """
    if randgen is None:
        randgen = np.random.default_rng(0)

    oneq_gstdesign_circuitlists = oneq_gstdesign.circuit_lists
    twoq_gstdesign_circuitlists = twoq_gstdesign.circuit_lists
    if _layer_mappers_override is not None:
        layer_mappers = _layer_mappers_override
    else:
        layer_mappers = build_layer_mappers(oneq_gstdesign, twoq_gstdesign)

    assert len(oneq_gstdesign_circuitlists) == len(twoq_gstdesign_circuitlists), \
        "Not implemented."

    vertices = list(vertices)

    patch_infos, groups = build_patch_infos(vertices, color_patches)

    # Preserve user/color_patches ordering in the final output.
    patch_order = [info["patch"] for info in patch_infos]
    patch_info_by_name = {
        info["patch"]: info
        for info in patch_infos
    }

    previous_patch_buffers = {
        patch: []
        for patch in patch_order
    }

    circuit_lists = [[] for _ in twoq_gstdesign_circuitlists]

    for L, (oneq_circuits, twoq_circuits) in _tqdm.tqdm(
        enumerate(zip(oneq_gstdesign_circuitlists, twoq_gstdesign_circuitlists)),
        total=len(twoq_gstdesign_circuitlists)
    ):
        oneq_len = len(oneq_circuits)
        twoq_len = len(twoq_circuits)

        max_len = max(oneq_len, twoq_len)
        min_len = min(oneq_len, twoq_len)

        if oneq_len > twoq_len:
            raise NotImplementedError()

        # Temporary per-patch storage so output ordering remains patch-major.
        patch_buffers = {
            info["patch"]: []
            for info in patch_infos
        }

        for group_key, infos in groups.items():
            num_edges, num_unused_qubits = group_key

            representative = infos[0]
            representative_lines = representative["tensored_lines"]

            edge_perms = np.empty(
                (num_edges, max_len),
                dtype=np.int64
            )

            for edge_slot in range(num_edges):
                edge_perms[edge_slot, :] = randgen.permutation(max_len)

            oneq_base_perm = np.hstack((
                randgen.integers(0, min_len, size=max_len - min_len),
                np.arange(min_len)
            ))

            oneq_perms = np.empty(
                (num_unused_qubits, max_len),
                dtype=np.int64
            )

            for qubit_slot in range(num_unused_qubits):
                oneq_perms[qubit_slot, :] = randgen.permutation(oneq_base_perm)

            mappers = {}

            for info in infos:
                if info is representative:
                    mappers[info["patch"]] = None
                else:
                    mappers[info["patch"]] = make_line_mapper(
                        representative_lines,
                        info["tensored_lines"]
                    )

            for j in range(max_len):
                circs_to_tensor = []

                for edge_slot in range(num_edges):
                    circ_idx = edge_perms[edge_slot, j]
                    circs_to_tensor.append(twoq_circuits[circ_idx])

                for qubit_slot in range(num_unused_qubits):
                    circ_idx = oneq_perms[qubit_slot, j]
                    circs_to_tensor.append(oneq_circuits[circ_idx])

                template_circuit = batch_tensor(
                    circs_to_tensor,
                    layer_mappers,
                    None,
                    representative_lines
                )

                if debug_check:
                    for i in range(template_circuit.num_layers):
                        l0 = set(template_circuit.layer(i))
                        l1 = set(template_circuit.layer_with_idles(i))
                        assert l0 == l1

                patch_buffers[representative["patch"]].append(
                    template_circuit.copy()
                )

                for info in infos[1:]:
                    mapper = mappers[info["patch"]]

                    # mapped_circuit = template_circuit.map_state_space_labels(mapper)
                    mapped_circuit = template_circuit.copy()
                    if debug_check:
                        expected_labels = {
                            q
                            for line in info["tensored_lines"]
                            for q in line
                        }

                        actual_labels = set(mapped_circuit.line_labels)

                        assert actual_labels == expected_labels, (
                            actual_labels,
                            expected_labels
                        )

                    patch_buffers[info["patch"]].append(mapped_circuit)

        # Preserve patch-major output ordering.
        output_list = circuit_lists[L]

        if ensure_containment:
            for patch in patch_order:
                patch_buffers[patch] = (
                    previous_patch_buffers[patch] + patch_buffers[patch]
                )

        for patch in patch_order:
            output_list.extend(patch_buffers[patch])

        if ensure_containment:
            previous_patch_buffers = {
                patch: list(patch_buffers[patch])
                for patch in patch_order
            }

    return circuit_lists