"""
Utility functions for subcircuit selection
"""

#TODO: add copyright statement

from __future__ import annotations
from typing import Dict, List, Tuple, Callable, Union, Optional, Any, Protocol, Literal, Set


import warnings as _warnings
from collections import Counter as _Counter, defaultdict as _defaultdict
import itertools as _itertools
import networkx as _nx

import tqdm as _tqdm

import numpy as _np
import json as _json

from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti.baseobjs.label import Label as _Label
from pygsti.protocols.protocol import FreeformDesign as _FreeformDesign
from pygsti.tools import internalgates as _itgs

try:
    import qiskit
    if qiskit.__version__ != '2.1.1':
        _warnings.warn("The qiskit functionality of subcircuit selection is designed for qiskit 2.1.1." \
        "Your version is " + qiskit.__version__)
except:
    _warnings.warn("Qiskit does not appear to be installed." \
    "If providing a CouplingMap for `coupling_map`, qiskit must be installed.")


class HasGetMethod(Protocol):
    def get(gate_name: str, qubits: List[int]) -> float:
        pass


def sample_subcircuits(full_circs: Union[_Circuit, List[_Circuit]],
                       width_depths: Dict[int, List[int]],
                       instruction_durations: Union[qiskit.transpiler.InstructionDurations, HasGetMethod],
                       coupling_map: Union[str, List[int], List[str], qiskit.transpiler.CouplingMap],
                       num_samples_per_width_depth: int = 10,
                       strategy: Union[Literal['simple', 'greedy'], Callable[..., Any]] = 'simple',
                       strategy_args: Optional[dict] = None,
                       depth_metric: Literal['layer_count', 'falcon_depth'] = 'layer_count',
                       num_test_samples: Optional[int] = None,
                    #    subgraph_cache = None,
                       rand_state: Optional[_np.random.RandomState] = None,
                       ) -> _FreeformDesign:    
    """
    Samples subcircuits from a full circuit based on specified width and depth pairs.

    Parameters
    -------------
    full_circs : Union[pygsti.circuits.Circuit, List[pygsti.circuits.Circuit]]
        The full circuit(s) from which to sample subcircuits.

    width_depths : Dict[int, List[int]]
        A dictionary whose keys are subcircuit widths and whose values are lists of depths
        to sample for that subcircuit depth. Defines a list of (width, depth) pairs for
        subcircuits.

    instruction_durations : qiskit.transpiler.InstructionDurations
        A qiskit InstructionDurations object used to determine delay times for
        idle subcircuit layers.

    coupling_map : Union[str, qiskit.transpiler.CouplingMap]
        The coupling map defining the connectivity of qubits. Can be 'all-to-all', 'linear',
        or a qiskit CouplingMap object.

    num_samples_per_width_depth : int, optional
        The number of subcircuits to sample for each width-depth combination. Default is 10.

    strategy : Union[str, Callable[..., Any]], optional
        The subcircuit sampling strategy to use ('simple', 'greedy', or a custom function).
        Default is 'simple'.

    strategy_args : Dict, optional
        Additional arguments if a custom sampling strategy function is used. Default is None.

    depth_metric : str, optional
        The metric to use for measuring depth ('layer_count' or 'falcon_depth'). If `layer_count`,
        the depth is calculated as the number of layers in the subcircuit. If `falcon_depth`,
        the gate set must be U3-CX; U3 gates contribute 2 to the depth and CX gates contribute 1
        to the depth. The `falcon_depth` metric is based on the Falcon generation of IBMQ devices.
        Default is 'layer_count'.

    num_test_samples : int, optional
        The number of test samples to use if the 'greedy' subcircuit sampling strategy is employed.
        If `strategy` is not `greedy`, this argument is ignored.
        Default is None.

    subgraph_cache : Dict[int, List[List[int]]], optional
        Cache for subgraph information to avoid recomputation. Default is None.

    rand_state : _np.random.RandomState, optional
        A random state for reproducibility. Default is None.

    Returns
    --------
    pygsti.protocols.FreeformDesign
        A FreeformDesign object containing the sampled subcircuits and auxiliary
        information, including a circuit ID and depth.
    """
    
    if rand_state is None:
        rand_state = _np.random.RandomState()
    
    # if subgraph_cache is None:
    #     subgraph_cache = {}

    # Build unmirrored circuit
    subcircuits = _defaultdict(list)
    counter = 0

    if not isinstance(full_circs, list): # package pygsti circuit into list if list was not provided.
        full_circs = [full_circs]

    for full_circ in _tqdm.tqdm(full_circs, ascii=True, desc='sampling subcircuits'):
        # print(f'sampling circuit {name}')
        for w, ds in width_depths.items():
            print(f'Width: {w}, Depth: ', end='')
            for d in ds:
                print(d, end=' ')
                if strategy == 'simple':

                    # if arch is None:
                    #     assert placer_json is not None, "Must provide path to placer_json if no arch provided"
                    #     with placer_json.open() as f:
                    #         placer_dict = json.load(f)
                    #     arch = Architecture.from_dict(placer_dict['architecture'])

                    subcircs, drops = simple_weighted_subcirc_selection(full_circ,
                                                                        w, d,
                                                                        num_subcircs=num_samples_per_width_depth,
                                                                        depth_metric=depth_metric,
                                                                        instruction_durations=instruction_durations,
                                                                        coupling_map=coupling_map,
                                                                        # subgraph_cache=subgraph_cache,
                                                                        rand_state=rand_state,
                                                                        verbosity=0)
                                                                        
                elif strategy == 'greedy':
                    num_test_samples = 50 * num_samples_per_width_depth
                    subcircs, drops = greedy_growth_subcirc_selection(full_circ, w, d, num_subcircs=num_samples_per_width_depth,
                                                            num_test_subcircs=num_test_samples, rand_state=rand_state, verbosity=0)
                elif callable(strategy):
                    subcircs, drops = strategy(full_circ, w, d, num_subcircs=num_samples_per_width_depth, **strategy_args)
                else:
                    raise ValueError("'strategy' is not a function or known string")
                
                for subcirc, drop in zip(subcircs, drops):
                    # print(subcirc)
                    subcircuits[subcirc].append({'width': w, 'depth': d, 'dropped_gates': drop, 'id': counter})
                    counter += 1
                
            print()

    return _FreeformDesign(subcircuits)

def simple_weighted_subcirc_selection(full_circ: _Circuit,
                                      width: int,
                                      depth: int,
                                      num_subcircs: int,
                                      coupling_map: Union[str, List[int], List[str], qiskit.transpiler.CouplingMap],
                                      instruction_durations: Union[qiskit.transpiler.InstructionDurations, HasGetMethod],
                                      depth_metric: Literal['layer_count', 'falcon_depth'] = 'layer_count',
                                    #   subgraph_cache=None,
                                      rand_state: Optional[_np.random.RandomState] = None,
                                      return_depth_info: bool = False,
                                      stochastic_2q_drops: bool = False,
                                      verbosity: Literal[0, 1] = 1,
                                      ) -> List[List[_Circuit], 
                                                List[int],
                                                Optional[List[int]],
                                                Optional[List[Tuple[int, int]]],
                                                Optional[List[int]],
                                                Optional[List[List[int]]]]:
    """
    Samples subcircuits from a full circuit using a simple approach. The simple approach
    is to identify a starting layer, along with a connected subset of active qubits, and
    snip out a subcircuit with the desired width and depth.

    Parameters
    -------------
    full_circs : Union[pygsti.circuits.Circuit, List[pygsti.circuits.Circuit]]
        The full circuit(s) from which to sample subcircuits.

    width : int
        width of subcircuit to snip out.

    depth: int
        depth of subcircuit to snip out.

    num_subcircs : int
        The number of subcircuits to snip out for the given width and depth.

    coupling_map : Union[str, qiskit.transpiler.CouplingMap]
        The coupling map defining the connectivity of qubits. Can be 'all-to-all', 'linear',
        or a qiskit CouplingMap object.

    instruction_durations : qiskit.transpiler.InstructionDurations
        A qiskit InstructionDurations object used to determine delay times for
        idle subcircuit layers.

    rand_state : _np.random.RandomState, optional
        A random state for reproducibility. Default is None.

    subgraph_cache : Dict[int, List[List[int]]], optional
        Cache for subgraph information to avoid recomputation. Default is None.

    return_depth_info : bool, optional
        Whether to include compiled depths and the start and end layer for
        each subcircuit. Default is False.

    stochastic_2q_drops : bool, optional
        Whether to apply stochastic dropping of 2-qubit gates. Default is False,
        in which case all dangling 2-qubit gates are dropped. A gate is considered
        dangling if it has support on at least one qubit that is not in the
        selected subset of qubits snipped out for the subcircuit.

    verbosity : int, optional
        Level of verbosity for logging. Default is 1.

    Returns
    --------
    Tuple[List[pygsti.circuits.Circuit], List[int],
            Optional[List[int], Optional[List[Tuple[int,int]]]],
            Optional[List[int]], Optional[List[int]]]

        A tuple containing the selected subcircuits and the counts of dropped gates.
        If `return_depth_info` is set to True, then the returns are extended to include
        the compiled depth of each subcircuit and the start and end layers of each subcircuit.
        If `stochastic_2q_drops` is set to True, then the returns are extended to include
        the number of dangling gates in each subcircuit and the indices of added layers.
    """

    full_width = len(full_circ.line_labels)
    full_depth = len(full_circ)
    assert width > 1 and depth > 1, "Target width and depth must be greater than 1"
    assert width <= full_width, f"Target width has to be less than full circuit width ({full_width})"
    assert depth <= full_depth, f"Target depth has to be less than full circuit depth ({full_depth})"

    if rand_state is None:
        rand_state = _np.random.RandomState()

    subcircs = []
    failures = 0

    possible_starts = list(range(full_depth - depth + 1))

    while (len(subcircs) < num_subcircs) and (failures < 1000): #1000 is an arbitrary cutoff for the moment
        # Sample depth with cumulative layer weights
        start = rand_state.choice(possible_starts)

        # Identify layer subset to snip out.

        # Calculate physical depth
        compiled_depth = 0
        end = start - 1
        while compiled_depth < depth:
            end += 1
            if depth_metric == 'layer_count':
                layer_depth = 1

            # qubits = set(full_circ.line_labels)
            elif depth_metric == 'falcon_depth':
                layer_depth = 1
                for comp in full_circ._layer_components(end):
                    if comp.name == 'Gu3':
                        layer_depth = 2
                
                else:
                    raise RuntimeError('Invalid layer type!')
                
            else:
                raise ValueError(f"Unknown value for parameter `depth_metric`: {depth_metric}")

            compiled_depth += layer_depth
        
        if compiled_depth > depth:
            # We overshot (added a Gu3 at the end with only one space)
            # print("we overshot. whoops.")
            failures += 1
            # Skip this and try again
            continue
            
        # Now compute full weights for each allowed qubit subset (subgraph)
        #subset_weights = np.zeros(len(subgraphs))
        #for i, sg in enumerate(subgraphs):
        #    for q in sg:
        #        subset_weights[i] += cumul_line_weights[qubit_mapping[q]]

        # calculate layer durations to be used for any idle layers
        layer_durations = []
        gate_name_conversions = _itgs.standard_gatenames_qiskit_conversions() # TODO: make this gate mapping more customizable

        for layer_idx in range(start, end+1):
            max_comp_duration = 0
            for comp in full_circ._layer_components(layer_idx):
                backend_comp_name = gate_name_conversions[comp.name][1]
                backend_comp_qubits = [int(q[1:]) for q in comp.qubits]
                comp_duration = instruction_durations.get(backend_comp_name, backend_comp_qubits)
                if comp_duration > max_comp_duration:
                    max_comp_duration = comp_duration
            layer_durations.append(max_comp_duration)

        # Determine a width subset to snip out.
        if coupling_map == 'all-to-all':
            qubit_subset = set(rand_state.choice(full_circ.line_labels, size=width, replace=False))

        elif coupling_map == 'linear':
            start = rand_state.choice(full_width - width)
            end = start + width
            qubit_subset = set(full_circ.line_labels[start:end])

        elif isinstance(coupling_map, list): # this list needs to already include the 'Q' prefix on the qubits
            G = _nx.Graph()
            G.add_edges_from(coupling_map)
            qubit_subset = random_connected_subgraph(G, width, rand_state)

        else: # likely the coupling_map is a CouplingMap instance
            edges = []
            for cs in coupling_map:
                qubits = [f'Q{c}' for c in cs]
                if all([q in full_circ.line_labels for q in qubits]):
                    edges.append(qubits)

            G = _nx.Graph()
            G.add_edges_from(edges)
            qubit_subset = random_connected_subgraph(G, width, rand_state) 

        qubit_subset = set(str(qubit) for qubit in qubit_subset)

        # We have identified a width/depth to snip out, so do so now
        subcirc_layers = []
        possible_dropped_gates = []
        for layer_idx in range(start, end+1):
            new_layer = []
            for op in full_circ._layer_components(layer_idx):
                if all([q in qubit_subset for q in op.qubits]):
                    # This is inside our width/depth, so add it
                    new_layer.append(op)
                elif any([q in qubit_subset for q in op.qubits]):
                    # This is dangling, account for it and deal with it later
                    possible_dropped_gates.append((op, len(subcirc_layers)))
                # else we are totally outside, so not dropped
            
            subcirc_layers.append(new_layer)
        
        # Handle dropped gates
        total_dropped_gates = 0
        total_dangling_gates = 0
        added_layer_indices = []
        if stochastic_2q_drops:
            # Split gates into two random sets of almost equal size
            op_idxs_to_drop = rand_state.choice(list(range(len(possible_dropped_gates))), len(possible_dropped_gates)//2, replace=False)
            total_dropped_gates = len(op_idxs_to_drop)

            # Add some dangling gates to make up for drops
            ops_to_add = [op for i,op in enumerate(possible_dropped_gates) if i not in op_idxs_to_drop]
            total_dangling_gates = 2*len(ops_to_add)

            offset = 0
            last_added_layer = -1
            new_layer = []
            for op, layer_idx in ops_to_add:
                # If we're adding to a new layer, add the previously built additional layer
                if layer_idx != last_added_layer:
                    if len(new_layer):
                        print(f'Adding new layer at layer {last_added_layer} with offset {offset}')
                        subcirc_layers.insert(last_added_layer + offset + 1, new_layer)
                        added_layer_indices.append(last_added_layer + offset + 1)
                        offset += 1
                        new_layer = []
                        

                    last_added_layer = layer_idx
                
                # Add the dangling gate back to the current subcirc layer
                print(f'Adding op to layer {layer_idx} with offset {offset}')
                subcirc_layers[layer_idx + offset].append(op)
                # Add the dangling gate to an additional layer as well
                new_layer.append(op)
            
            # Ensure we add the last additional layer
            if len(new_layer):
                print(f'Adding new layer at layer {last_added_layer} with offset {offset}')
                subcirc_layers.insert(last_added_layer + offset + 1, new_layer)
                added_layer_indices.append(last_added_layer + offset + 1)
        else:
            total_dropped_gates = len(possible_dropped_gates)
        

        for i in range(len(subcirc_layers)):
            scl = subcirc_layers[i]
            if len(scl) == 0: #layer is empty. add delay of appropriate duration
                # handle the addition of a delay instruction based on the maximum duration of an instruction in this circuit layer
                delay_layer = [_Label('Gdelay', qubit, args=[layer_durations[i]]) for qubit in qubit_subset]
                subcirc_layers[i] = delay_layer

        # Build subcircuit
        subcirc = _Circuit(subcirc_layers, line_labels=sorted(list(qubit_subset)))

        # print(subcirc)
        subcircs.append((subcirc, total_dropped_gates, compiled_depth, (start, end), total_dangling_gates, added_layer_indices))
    
        if verbosity > 0:
            print(f'Found subcircuit with {total_dropped_gates} dropped gates, {compiled_depth} depth, and {total_dangling_gates} dangling gates')

    if (failures == 1000):
        raise RuntimeError("Failed to find a valid starting layer 1000 times!")
    
    # Unpacking to match greedy growth function
    selected_subcircs, dropped_counts, compiled_depths, start_ends, dangling_counts, added_layers = list(zip(*subcircs))
    
    if verbosity > 0:
        print(f'Dropped gate counts for selected circuits: {dropped_counts}')
        print(f'Compiled depths for selected circuits: {compiled_depths}')
        print(f'Dangling gate counts for selected circuits: {dangling_counts}')

    returns = [list(selected_subcircs), dropped_counts,
            #    subgraph_cache
               ]

    if return_depth_info:
        returns.extend([compiled_depths, start_ends])
    if stochastic_2q_drops:
        returns.extend([dangling_counts, added_layers])
    
    return returns

def greedy_growth_subcirc_selection(full_circ: _Circuit,
                                    width: int,
                                    depth: int,
                                    num_subcircs: int = 1,
                                    num_test_subcircs: int = 10,
                                    rand_state: Optional[_np.random.RandomState] = None,
                                    verbosity: Literal[0, 1] = 1,
                                    return_depth_info: bool = False
                                    ) -> Tuple[List[_Circuit], List[int],
                                            Optional[List[int]], Optional[List[Tuple[int,int]]]]:    
    """
    Selects subcircuits using a greedy growth strategy, starting with one gate in one layer
    and adding gates with overlapping support in subsequent layers.

    Parameters
    -------------
    full_circ : pygsti.circuits.Circuit
        The full circuit from which to select subcircuits.

    width : int
        The target width of the subcircuits to be selected.

    depth : int
        The target depth of the subcircuits to be selected.

    num_subcircs : int, optional
        The number of subcircuits to select. Default is 1.

    num_test_subcircs : int, optional
        The number of test subcircuits to generate. Default is 10.

    rand_state : Optional[_np.random.RandomState], optional
        A random state for reproducibility. Default is None.

    verbosity : int, optional
        Level of verbosity for logging. Default is 1.

    return_depth_info : bool, optional
        Whether to include compiled depths and the start and end layer for
        each subcircuit. Default is False.

    Returns
    --------
    Tuple[List[_Circuit], List[int],
            Optional[List[int]], Optional[List[Tuple[int,int]]]]
        A tuple containing the selected subcircuits and the counts of dropped gates.
        If `return_depth_info` is set to True, then the returns are extended to include
        the compiled depth of each subcircuit and the start and end layers of each subcircuit.
    """

    # TODO: sample until success wuth some upper bound on the number of samples that can be taken?

    full_width = len(full_circ.line_labels)
    full_depth = len(full_circ)
    assert width > 1 and depth > 1, "Target width and depth must be greater than 1"
    assert width <= full_width, f"Target width has to be less than full circuit width ({full_width})"
    assert depth <= full_depth, f"Target depth has to be less than full circuit depth ({full_depth})"
    assert num_subcircs <= num_test_subcircs, f"Must try at least {num_subcircs} test subcircuits"
    
    if rand_state is None:
        rand_state = _np.random.RandomState()
    
    # Get test subcircuits
    # TODO: Could be parallelized
    test_subcircs = [_greedy_growth_subcirc(full_circ, width, depth, rand_state, verbosity-1)
                     for _ in range(num_test_subcircs)]
    
    # Drop any with lower physical depth (possible error mode)
    pruned_test_subcircs = []
    seen_circuits = set()
    for sc in test_subcircs:
        # If not unique or not exact depth (possible error or trailing Gu3), skip
        if sc[0] in seen_circuits or sc[2] != depth:
            continue
        
        seen_circuits.add(sc[0])
        pruned_test_subcircs.append(sc)

    # Sort on lowest number of dropped gate counts, and then physical depth
    sorted_test_subcircs = sorted(pruned_test_subcircs, key=lambda x: (x[1], x[2]))

    print(f'greedy num_subcircs arg: {num_subcircs}')

    print(f"number of subcircuits in 'sorted_test_subcircs': {len(sorted_test_subcircs)}")

    # Select + unzip subcircs & dropped counts
    if len(sorted_test_subcircs) < num_subcircs:
        raise ValueError(f"Not enough subcircuits, only found {len(sorted_test_subcircs)}. Try increasing 'num_test_subcircs'")
    
    selected_subcircs, dropped_counts, physical_depths, start_ends = list(zip(*sorted_test_subcircs[:num_subcircs]))

    # try:
    #     selected_subcircs, dropped_counts, physical_depths, start_ends = list(zip(*sorted_test_subcircs[:num_subcircs]))
    # except ValueError as e:
    #     raise ValueError("Probably did not find enough valid subcircuits, trying increasing num_test_subcircs") from e
    
    print(f'number of subcircuits (greedy): {len(selected_subcircs)}')
    
    if verbosity > 0:
        print(f'Dropped gate counts for selected circuits: {dropped_counts}')
        print(f'Physical depths for selected circuits: {physical_depths}')
    
    if return_depth_info:
        return list(selected_subcircs), dropped_counts, physical_depths, start_ends
    
    return list(selected_subcircs), dropped_counts

# Workhorse function for greedy growth subcircuit selection
def _greedy_growth_subcirc(circ: _Circuit,
                           width: int,
                           depth: int,
                           rand_state: _np.random.RandomState,
                           verbosity: Literal[0, 1] = 0
                           ) -> Tuple[_Circuit, int, int, Tuple[int, int]]:
    """
    Workhorse function for greedy generation of candidate subcircuits.

    Parameters
    -------------
    circ : pygsti.circuits.Circuit
        The full circuit from which to grow the subcircuit.

    width : int
        The target width of the subcircuit to be grown.

    depth : int
        The target depth of the subcircuit to be grown.

    rand_state : Optional[_np.random.RandomState]
        A random state for reproducibility. Default is None.

    verbosity : int, optional
        Level of verbosity for logging. Default is 0.

    Returns
    --------
    Tuple[pygsti.circuits.Circuit, int, int, Tuple[int, int]]
        A tuple containing the grown subcircuit, the count of dropped gates,
        the physical depth, and the start-end indices of the subcircuit.
    """

    # Pick an initial layer
    start = end = rand_state.randint(circ.depth)
    
    # Pick an operation in layer
    ops = circ._layer_components(start)
    op_idx = rand_state.randint(len(ops))
    
    # Start tracking our subcircuit
    qubit_subset = set(ops[op_idx].qubits)
    physical_depth = 2 if ops[op_idx].name == 'Gu3' else 1
    
    # Helper function for adding new layers to our subcircuit
    def add_new_layer(layer_idx: int) -> Tuple[Set[_Label], int, int]:
        """
        Helper function for adding new layers to the subcircuit.

        Parameters
        ------------
        layer_idx : int
            Index of layer to be added to the subcircuit.

        Returns
        -----------
        Tuple[Set[pygsti.baseobjs.Label], int, int]
            Tuple containing labels to be added, the number of dropped gates,
            and the Falcon depth of the layer. The Falcon depth of a U3 layer
            is 2; the Falcon depth of a CX layer is 1.
        """

        labels_to_add = set()
        drops = 0
        new_depth = 0
        for op in circ._layer_components(layer_idx):
            if any([q in qubit_subset for q in op.qubits]):
                # This operation overlaps with our current subset
                # We need to see if it extends our width or must be dropped
                new_qubits = set(op.qubits) - qubit_subset
                
                if new_qubits and len(new_qubits) + len(labels_to_add) + len(qubit_subset) <= width:
                    # This operation expands our qubit subset, but we still have room the new qubit
                    labels_to_add.update(new_qubits)
                    new_depth = 2 if op.name == 'Gu3' else 1
                elif new_qubits:
                    # This operation expands our qubit subset but we don't have room, so must drop it
                    drops += 1
                else:
                    # This op is in our subset, get the physical depth
                    new_depth = 2 if op.name == 'Gu3' else 1
        
        return labels_to_add, drops, new_depth
    
    # Now try to grow the circuit
    # Technically this means physical_depth = depth or depth + 1 when we terminate
    while physical_depth < depth:
        # Look at previous layer
        prev_failed = False
        try:
            prev_labels, prev_drops, prev_depth = add_new_layer(start - 1)
        except IndexError:
            prev_failed = True
        
        # Look at next layer
        next_failed = False
        try:
            next_labels, next_drops, next_depth = add_new_layer(end + 1)
        except IndexError:
            next_failed = True

        if prev_failed and next_failed:
            # Both failed, exit early
            break
        elif prev_failed:
            labels = next_labels
            new_depth = next_depth
            end += 1
        elif next_failed:
            labels = prev_labels
            new_depth = prev_depth
            start -= 1
        else:
            # Prefer direction with fewer drops, otherwise choose randomly
            if prev_drops < next_drops or (prev_drops == next_drops and rand_state.randint(2)):
                labels = prev_labels
                new_depth = prev_depth
                start -= 1
            else:
                labels = next_labels
                new_depth = next_depth
                end += 1
        
        # Add new width/depth
        qubit_subset.update(labels)
        physical_depth += new_depth
    
    # We have the right depth, but not necessarily the right width
    if len(qubit_subset) < width:
        # We can look at same depth/complement width circuit to find qubits to add
        remaining_width_circ = _Circuit(circ[start:end+1], editable=True)
        remaining_width_circ.delete_lines(list(qubit_subset), delete_straddlers=True)

        # We can iterate through and check for 2-qubit gate counts as a heuristic for what qubits
        # are likely to not add too many dangling gates
        # Can probably do something even better here, but this may be good enough
        remaining_qubits = set(circ.line_labels) - qubit_subset
        remaining_2q_gate_counts = {q: 0 for q in remaining_qubits}
        for layer_idx in range(remaining_width_circ.depth):
            for op in remaining_width_circ._layer_components(layer_idx):
                for q in op.qubits:
                    remaining_2q_gate_counts[q] += 1

        # Sort and add qubits with lowest 2Q gate counts
        sorted_gate_counts = sorted(list(remaining_2q_gate_counts.items()), key=lambda x: x[1])
        qubits_to_add = [gc[0] for gc in sorted_gate_counts[:(width - len(qubit_subset))]]
        
        qubit_subset.update(qubits_to_add)        
        
    # We have identified a width/depth to snip out, so do so now
    # Technically this can have more dropped ops since we didn't know our full width for all layers above
    # But under the assumption that we fill out width pretty quickly, it should be close
    subcirc_layers = []
    total_dropped_gates = 0
    for layer_idx in range(start, end+1):
        new_layer = []
        
        for op in circ._layer_components(layer_idx):
            if all([q in qubit_subset for q in op.qubits]):
                # This is inside our width/depth, so add it
                new_layer.append(op)
            elif any([q in qubit_subset for q in op.qubits]):
                # This has some overlap but dangles, so drop it
                total_dropped_gates += 1
            # else we are totally outside, so not dropped
        
        if len(new_layer):
            subcirc_layers.append(new_layer)
    
    if verbosity > 0:
        print(f'Found subcircuit with {total_dropped_gates} dropped gates and {physical_depth} depth')

    # Return circuit + dropped gates/physical depth for external selection
    return _Circuit(subcirc_layers, line_labels=qubit_subset), total_dropped_gates, physical_depth, (start, end)


def random_connected_subgraph(G: _nx.Graph,
                              width: int,
                              rand_state: Optional[_np.random.RandomState] = None
                              ) -> Union[Set[int], Set[str]]:
    """
    Generates a random set of nodes that form a 
    connected subgraph of a specified width from a given graph.

    Parameters
    -------------
    G : networkx.Graph
        The graph from which to generate the connected subgraph.
    width : int
        The target width of the subgraph to be generated.
    rand_state : Optional[_np.random.RandomState]
        A random state for reproducibility. Default is None.

    Returns
    --------
    set[int or str]
        A set of nodes representing the generated connected subgraph.
    """

    if rand_state is None:
        rand_state = _np.random.RandomState()

    # pick a random starting node
    starting_node = rand_state.choice(G.nodes)

    used_nodes = set([starting_node])

    plausible_growth_nodes = set([starting_node])

    for i in range(width - 1):
        valid_node_found = False
        while valid_node_found == False and len(plausible_growth_nodes):
            growth_node = rand_state.choice(list(plausible_growth_nodes))
            neighbors = set(G.neighbors(growth_node))
            new_neighbors = neighbors.difference(used_nodes)

            if len(new_neighbors):
                new_node = rand_state.choice(list(new_neighbors))
                used_nodes.add(new_node)
                plausible_growth_nodes.add(new_node)
                valid_node_found = True

            else:
                plausible_growth_nodes.remove(growth_node)
        
        if valid_node_found == False: # we exited because all possible nodes where the graph could be extended have been exhausted
            raise RuntimeError(f'Could not generate a subgraph with {width} nodes')
            # failure should only occur if the initial node is on a connected component of the graph with less nodes than 'width'.

        
    assert len(used_nodes) == width, f'set of selected nodes has length {used_nodes} but should have length {width}'
    assert _nx.is_connected(G.subgraph(used_nodes)), f'subgraph on nodes {used_nodes} should be connected but is not'

    return used_nodes





