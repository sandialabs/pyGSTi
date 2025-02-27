"""Utility functions for subcircuit selection"""

from __future__ import annotations
from typing import Dict, List, Tuple, Callable, Union, Optional, Any


import warnings as _warnings
from collections import Counter as _Counter, defaultdict as _defaultdict
import itertools as _itertools
import networkx as _nx

import numpy as _np
import json as _json

# from pytket.architecture import Architecture
from qiskit.providers.models import BackendProperties, BackendConfiguration

from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti.baseobjs.label import Label as _Label
from pygsti.protocols.protocol import FreeformDesign as _FreeformDesign

#TODO: OOP-ify?


# Keep this for backwards compatibility for older notebooks
def sample_lcu_subcircuits(lcu_circ, width_depths, num_samples, strategy='simple',
                           num_test_samples=None, rand_state=None,
                           arch=None, subgraph_cache=None):
    return sample_subcircuits(lcu_circ, width_depths, num_samples, strategy=strategy,
                           num_test_samples=num_test_samples, rand_state=rand_state,
                           arch=arch, subgraph_cache=subgraph_cache)

def sample_subcircuits(full_circs: Union[_Circuit, Dict[str, _Circuit]],
                       width_depths: Dict[int, List[int]],
                       num_samples_per_circ: int,
                       strategy: Union[str, Callable[..., Any]] = 'simple',
                       num_test_samples: int = None,
                       rand_state: Optional[_np.random.RandomState] = None,
                       backend=None,
                       backend_is_fake_v2=False,
                       # arch=None,
                       # placer_json=None,
                       subgraph_cache=None,
                       strategy_args: Dict = None):
        
    
    if rand_state is None:
        rand_state = _np.random.RandomState()
    
    if subgraph_cache is None:
        subgraph_cache = {}

    # Build unmirrored circuit
    subcircuits = _defaultdict(list)
    counter = 0

    if not isinstance(full_circs, dict): # package pygsti circuit into dict if dict was not provided.
        full_circs = {'default name': full_circs}

    for name, full_circ in full_circs.items():
        print(f'sampling circuit {name}')
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

                    subcircs, drops = simple_weighted_subcirc_selection(full_circ, w, d, num_subcircs=num_samples_per_circ,
                                                        rand_state=rand_state, backend=backend, backend_is_fake_v2=backend_is_fake_v2, subgraph_cache=subgraph_cache, verbosity=0)
                elif strategy == 'greedy':
                    num_test_samples = 50 * num_samples_per_circ
                    subcircs, drops = greedy_growth_subcirc_selection(full_circ, w, d, num_subcircs=num_samples_per_circ,
                                                            num_test_subcircs=num_test_samples, rand_state=rand_state, verbosity=0)
                elif callable(strategy):
                    subcircs, drops = strategy(full_circ, w, d, num_subcircs=num_samples_per_circ, **strategy_args)
                else:
                    raise ValueError("'strategy' is not a function or known string")
                
                for subcirc, drop in zip(subcircs, drops):
                    print(subcirc)
                    subcircuits[subcirc].append({'width': w, 'physical_depth': d, 'dropped_gates': drop, 'id': counter})
                    counter += 1
                
            print()

    return _FreeformDesign(subcircuits)

def simple_weighted_subcirc_selection(full_circ, width, depth, num_subcircs=1, backend=None,
                                      backend_is_fake_v2=False, subgraph_cache=None,
                                      rand_state=None, verbosity=1, return_depth_info=False,
                                      stochastic_2q_drops=False):
    full_width = len(full_circ.line_labels)
    full_depth = len(full_circ)
    assert width > 1 and depth > 1, "Target width and depth must be greater than 1"
    assert width <= full_width, f"Target width has to be less than full circuit width ({full_width})"
    assert depth <= full_depth, f"Target depth has to be less than full circuit depth ({full_depth})"

    possible_starts = list(range(full_depth - depth + 1))

    

    if rand_state is None:
        rand_state = _np.random.RandomState()
    if backend_is_fake_v2:
        backend._set_props_dict_from_json()
        props_dict = backend._props_dict
        backend_props = BackendProperties.from_dict(props_dict)
        conf_dict = backend._get_conf_dict_from_json()
        backend_config = BackendConfiguration.from_dict(conf_dict)
        couplings = backend.coupling_map.get_edges()
    else:
        backend_config = backend.configuration()
        backend_props = backend.properties()
        couplings = backend_config.coupling_map

    # Check for connected graphs in cache (can be expensive)
    if subgraph_cache is None:
        subgraph_cache = {}
    
    subgraphs = []
    if subgraph_cache is not None and width in subgraph_cache:
        if verbosity > 0: print('Reusing subgraph cache for qubit connectivity')
        subgraphs = subgraph_cache[width]
    elif backend is not None:
        if verbosity > 0: print('Computing subgraphs for qubit connectivity... ', end='')

        # Build graph of only used qubits
        edges = []
        for cs in couplings:
            qubits = [f'Q{c}' for c in cs]
            if all([q in full_circ.line_labels for q in qubits]):
                edges.append(qubits)

        G = _nx.Graph()
        G.add_edges_from(edges)

        for nodes in _itertools.combinations(G.nodes, width):
            subgraph = G.subgraph(nodes)
            if _nx.is_connected(subgraph):
                subgraphs.append(list(subgraph.nodes.keys()))
        
        if subgraph_cache is not None:
            subgraph_cache[width] = subgraphs
        if verbosity > 0: print('Done!')      
    else:
        raise RuntimeError('Either subgraph_cache with proper width or arch must be provided')
    
    assert len(subgraphs), "Subgraphs provided but empty!"

    qubit_mapping = {j:i for i,j in enumerate(full_circ.line_labels)}

    subcircs = []

    failures = 0

    while (len(subcircs) < num_subcircs) and (failures < 1000): #1000 is an arbitrary cutoff for the moment
        # Sample depth with cumulative layer weights
        start = rand_state.choice(possible_starts)

        layer_names = []

        # Calculate physical depth
        compiled_depth = 0
        end = start - 1
        while compiled_depth < depth:
            end += 1

            qubits = set(full_circ.line_labels)
            layer_depth = 1
            for comp in full_circ._layer_components(end):
                if comp.name == 'Gu3':
                    layer_depth = 2
            
            if comp.name == 'Gu3':
                layer_names.append('Gu3')
            elif comp.name == 'Gcnot':
                layer_names.append('Gcnot')
            else:
                raise RuntimeError('Invalid layer type!')

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
        
        # Sample width with cumulative line weights
        subset_idx = rand_state.choice(range(len(subgraphs)))#, p=subset_weights/sum(subset_weights))[0]
        qubit_subset = subgraphs[subset_idx]

        # We have identified a width/depth to snip out, so do so now
        # But under the assumption that we fill out width pretty quickly, it should be close
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
        
        # # Drop any empty layers
        # subcirc_layers = [scl for scl in subcirc_layers if len(scl)]

        assert len(subcirc_layers) == len(layer_names), "Relationship between layers and layer names is not one to one!"

        print(len(subcirc_layers))

        for i in range(len(subcirc_layers)):
            scl = subcirc_layers[i]
            if len(scl) == 0: #layer is empty. add delay of appropriate duration
                dt = backend_config.dt
                if layer_names[i] == 'Gu3':
                    # need to compute how many dt (device-intrinsic time unit) have elapsed.
                    # assuming that sx is possible on qubit 0. u3 uses 2 sx gates, rz gates are involved but virtual and therefore have 0 duration
                    # note that in general the 'sx' time may depend on which qubit it is being executed on. For now, for simplicity, we assume it is the same for all the qubits
                    u3_time = 2 * backend_props.gate_length('sx', 0)
                    u3_time_dt = 8 * (int(u3_time / dt) // 8) #Error message from IBMQ Brisbane says that delays must be a multiple of 8 samples
                    new_layer = [_Label('Gdelay', qubit, args=[u3_time_dt]) for qubit in qubit_subset]
                    subcirc_layers[i] = new_layer
                elif layer_names[i] == 'Gcnot':
                    # look at the first coupling that exists and use that to get the cnot time. Note that, in general, the 'cx' time may vary by coupling.
                    try: # see if there is a 'cx' instruction
                        cx_time = backend_props.gate_length('cx', couplings[0])
                    except: # if not, assume there is an 'ecr' instruction
                        # cnot takes 1 'ecr' + 2 'sx' time
                        # there are also 'rz' but those are virtual and therefore do not contribute to the execution time
                        ecr_time = backend_props.gate_length('ecr', couplings[0])
                        sx_time = backend_props.gate_length('sx', 0)
                        cx_time = ecr_time + 2 * sx_time

                    cx_time_dt = 8 * (int(cx_time / dt) // 8) #Error message from IBMQ Brisbane says that delays must be a multiple of 8 samples
                    new_layer = [_Label('Gdelay', qubit, args=[cx_time_dt]) for qubit in qubit_subset]
                    subcirc_layers[i] = new_layer
                else:
                    raise RuntimeError('Invalid layer label encountered!')

        # Build subcircuit
        subcirc = _Circuit(subcirc_layers, line_labels=sorted(list(qubit_subset)))
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

    returns = [list(selected_subcircs), dropped_counts]

    if return_depth_info:
        returns.extend([compiled_depths, start_ends])
        if stochastic_2q_drops:
            returns.extend([dangling_counts, added_layers])
    
    return returns

def greedy_growth_subcirc_selection(full_circ, width, depth, num_subcircs=1, num_test_subcircs=10,
        rand_state=None, verbosity=1, return_depth_info=False):    
    # we potentially want to change this function so that it will keep sampling until it finds enough unique circuits (under some upper bound). There are some decisions to be made about how exactly that should work. For instance:
    # There is behavior that prioritizes gates with less drops and of smaller physical depth. That behavior would not work if we sampled 1 circuit at a time until we have enough. Do we sample 1 at a time and lose this behavior? Do we sample in batches? Do we do an initial large sample and then check one more circuit at a time? As I said, decisions to be made.
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
def _greedy_growth_subcirc(circ, width, depth, rand_state, verbosity=0):
    # Pick an initial layer
    start = end = rand_state.randint(circ.depth)
    
    # Pick an operation in layer
    ops = circ._layer_components(start)
    op_idx = rand_state.randint(len(ops))
    
    # Start tracking our subcircuit
    qubit_subset = set(ops[op_idx].qubits)
    physical_depth = 2 if ops[op_idx].name == 'Gu3' else 1
    
    # Helper function for adding new layers to our subcircuit
    def add_new_layer(layer_idx):
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


def useless_strategy(full_circ, width, depth, test_param, num_subcircs=1, arch=None, subgraph_cache=None,
                                      rand_state=None, verbosity=1, return_depth_info=False,
                                      stochastic_2q_drops=False):
    print(test_param)
    subcircs, drops = simple_weighted_subcirc_selection(full_circ, width, depth, num_subcircs=num_subcircs,
                                                      rand_state=rand_state, arch=arch, subgraph_cache=subgraph_cache, verbosity=0)
    
    return subcircs, drops



#REQUIRE that any user-defined sampling strategy takes in the full circuit, width, depth of the sampling?

def old_simple_weighted_subcirc_selection(full_circ, width, depth, num_subcircs=1, backend=None, subgraph_cache=None,
                                      rand_state=None, verbosity=1, return_depth_info=False,
                                      stochastic_2q_drops=False):
    full_width = len(full_circ.line_labels)
    full_depth = len(full_circ)
    assert width > 1 and depth > 1, "Target width and depth must be greater than 1"
    assert width <= full_width, f"Target width has to be less than full circuit width ({full_width})"
    assert depth <= full_depth, f"Target depth has to be less than full circuit depth ({full_depth})"

    possible_starts = list(range(full_depth  - depth + 1))

    if rand_state is None:
        rand_state = _np.random.RandomState()

    # Check for connected graphs in cache (can be expensive)
    if subgraph_cache is None:
        subgraph_cache = {}
    
    subgraphs = []
    if subgraph_cache is not None and width in subgraph_cache:
        if verbosity > 0: print('Reusing subgraph cache for qubit connectivity')
        subgraphs = subgraph_cache[width]
    elif backend is not None:
        if verbosity > 0: print('Computing subgraphs for qubit connectivity... ', end='')

        # Build graph of only used qubits
        backend_config = backend.configuration()
        backend_props = backend.properties()

        couplings = backend_config.coupling_map

        edges = []
        for cs in couplings:
            qubits = [f'Q{c}' for c in cs]
            if all([q in full_circ.line_labels for q in qubits]):
                edges.append(qubits)

        G = _nx.Graph()
        G.add_edges_from(edges)

        for nodes in _itertools.combinations(G.nodes, width):
            subgraph = G.subgraph(nodes)
            if _nx.is_connected(subgraph):
                subgraphs.append(list(subgraph.nodes.keys()))
        
        if subgraph_cache is not None:
            subgraph_cache[width] = subgraphs
        if verbosity > 0: print('Done!')      
    else:
        raise RuntimeError('Either subgraph_cache with proper width or arch must be provided')
    
    assert len(subgraphs), "Subgraphs provided but empty!"

    qubit_mapping = {j:i for i,j in enumerate(full_circ.line_labels)}

    subcircs = []
    while len(subcircs) < num_subcircs:
        # Sample depth with cumulative layer weights
        start = rand_state.choice(possible_starts)

        # Calculate physical depth
        compiled_depth = 0
        end = start - 1
        while compiled_depth < depth:
            end += 1

            qubits = set(full_circ.line_labels)
            layer_depth = 1
            for comp in full_circ._layer_components(end):
                if comp.name == 'Gu3':
                    layer_depth = 2
            
            compiled_depth += layer_depth
        
        if compiled_depth > depth:
            # We overshot (added a Gu3 at the end with only one space)
            # Skip this and try again
            continue
            
        # Now compute full weights for each allowed qubit subset (subgraph)
        #subset_weights = np.zeros(len(subgraphs))
        #for i, sg in enumerate(subgraphs):
        #    for q in sg:
        #        subset_weights[i] += cumul_line_weights[qubit_mapping[q]]
        
        # Sample width with cumulative line weights
        subset_idx = rand_state.choice(range(len(subgraphs)))#, p=subset_weights/sum(subset_weights))[0]
        qubit_subset = subgraphs[subset_idx]

        # We have identified a width/depth to snip out, so do so now
        # But under the assumption that we fill out width pretty quickly, it should be close
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
        
        # Drop any empty layers
        subcirc_layers = [scl for scl in subcirc_layers if len(scl)]

        # Build subcircuit
        subcirc = _Circuit(subcirc_layers, line_labels=sorted(list(qubit_subset)))
        subcircs.append((subcirc, total_dropped_gates, compiled_depth, (start, end), total_dangling_gates, added_layer_indices))
    
        if verbosity > 0:
            print(f'Found subcircuit with {total_dropped_gates} dropped gates, {compiled_depth} depth, and {total_dangling_gates} dangling gates')
    
    # Unpacking to match greedy growth function
    selected_subcircs, dropped_counts, compiled_depths, start_ends, dangling_counts, added_layers = list(zip(*subcircs))
    
    if verbosity > 0:
        print(f'Dropped gate counts for selected circuits: {dropped_counts}')
        print(f'Compiled depths for selected circuits: {compiled_depths}')
        print(f'Dangling gate counts for selected circuits: {dangling_counts}')

    returns = [list(selected_subcircs), dropped_counts]

    if return_depth_info:
        returns.extend([compiled_depths, start_ends])
        if stochastic_2q_drops:
            returns.extend([dangling_counts, added_layers])
    
    return returns

def simple_weighted_subcirc_selection_no_gen_all(full_circ, width, depth, num_subcircs=1, backend=None,
                                      rand_state=None, verbosity=1, return_depth_info=False,
                                      stochastic_2q_drops=False):
    # don't generate all the subgraphs at once.
    if backend is None:
        raise RuntimeError("You must provide a backend!")

    full_width = len(full_circ.line_labels)
    full_depth = len(full_circ)
    assert width > 1 and depth > 1, "Target width and depth must be greater than 1"
    assert width <= full_width, f"Target width has to be less than full circuit width ({full_width})"
    assert depth <= full_depth, f"Target depth has to be less than full circuit depth ({full_depth})"

    possible_starts = list(range(full_depth - depth + 1))

    if rand_state is None:
        rand_state = _np.random.RandomState()

    backend_config = backend.configuration()
    backend_props = backend.properties()
    couplings = backend_config.coupling_map

    # Build graph of only used qubits
    edges = []
    for cs in couplings:
        qubits = [f'Q{c}' for c in cs]
        if all([q in full_circ.line_labels for q in qubits]):
            edges.append(qubits)

    G = _nx.Graph()
    G.add_edges_from(edges)
    print(G)

    subgraph_not_found = True
    while subgraph_not_found: #PLEASE ADD ANOTHER EXIT CONDITION
        # print('hi')
        subgraph_nodes = _np.random.choice(G.nodes, width, replace=False)
        subgraph = G.subgraph(subgraph_nodes)
        # print(subgraph)
        if _nx.is_connected(subgraph):
            # qubit_subset = nodes
            qubit_subset = list(subgraph.nodes.keys())
            qubit_subset = [str(qubit) for qubit in qubit_subset] #silliness with numpy.str_
            break

        # for nodes in itertools.combinations(G.nodes, width):
        #     subgraph = G.subgraph(nodes)
        #     if nx.is_connected(subgraph):
        #         subgraphs.append(list(subgraph.nodes.keys()))
           
    # assert len(subgraphs), "Subgraphs provided but empty!"

    # qubit_mapping = {j:i for i,j in enumerate(full_circ.line_labels)}

    subcircs = []

    failures = 0

    while (len(subcircs) < num_subcircs) and (failures < 1000): #1000 is an arbitrary cutoff for the moment
        # Sample depth with cumulative layer weights
        start = rand_state.choice(possible_starts)

        layer_names = []

        # Calculate physical depth
        compiled_depth = 0
        end = start - 1
        while compiled_depth < depth:
            end += 1

            qubits = set(full_circ.line_labels)
            layer_depth = 1
            for comp in full_circ._layer_components(end):
                if comp.name == 'Gu3':
                    layer_depth = 2
            
            if comp.name == 'Gu3':
                layer_names.append('Gu3')
            elif comp.name == 'Gcnot':
                layer_names.append('Gcnot')
            else:
                raise RuntimeError('Invalid layer type!')

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
        
        # Sample width with cumulative line weights
        # subset_idx = rand_state.choice(range(len(subgraphs)))#, p=subset_weights/sum(subset_weights))[0]
        # qubit_subset = subgraphs[subset_idx]

        # We have identified a width/depth to snip out, so do so now
        # But under the assumption that we fill out width pretty quickly, it should be close
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
        
        # # Drop any empty layers
        # subcirc_layers = [scl for scl in subcirc_layers if len(scl)]

        assert len(subcirc_layers) == len(layer_names), "Relationship between layers and layer names is not one to one!"

        print(len(subcirc_layers))

        for i in range(len(subcirc_layers)):
            scl = subcirc_layers[i]
            if len(scl) == 0: #layer is empty. add delay of appropriate duration
                dt = backend_config.dt
                if layer_names[i] == 'Gu3':
                    # need to compute how many dt (device-intrinsic time unit) have elapsed.
                    # assuming that sx is possible on qubit 0. u3 uses 2 sx gates, rz gates are involved but virtual and therefore have 0 duration
                    # note that in general the 'sx' time may depend on which qubit it is being executed on. For now, for simplicity, we assume it is the same for all the qubits
                    u3_time = 2 * backend_props.gate_length('sx', 0)
                    u3_time_dt = int(u3_time / dt)
                    new_layer = [_Label('Gdelay', qubit, args=[u3_time_dt]) for qubit in qubit_subset]
                    subcirc_layers[i] = new_layer
                elif layer_names[i] == 'Gcnot':
                    # look at the first coupling that exists and use that to get the cnot time. Note that, in general, the 'cx' time may vary by coupling.
                    cx_time = backend_props.gate_length('cx', couplings[0])
                    cx_time_dt = int(cx_time / dt)
                    new_layer = [_Label('Gdelay', qubit, args=[cx_time_dt]) for qubit in qubit_subset]
                    subcirc_layers[i] = new_layer
                else:
                    raise RuntimeError('Invalid layer label encountered!')

        # Build subcircuit
        subcirc = _Circuit(subcirc_layers, line_labels=sorted(list(qubit_subset)))
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

    returns = [list(selected_subcircs), dropped_counts]

    if return_depth_info:
        returns.extend([compiled_depths, start_ends])
        if stochastic_2q_drops:
            returns.extend([dangling_counts, added_layers])
    
    return returns