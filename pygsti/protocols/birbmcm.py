import pygsti
from pygsti.circuits import circuit as _cir
from pygsti.baseobjs import Label
from pygsti.tools import symplectic as _symp
from pygsti.processors import CliffordCompilationRules as CCR

import copy as _copy

from pygsti.protocols import birb

import numpy as _np

import random

def outcome_energy(outcome, measurement, sign):
    energy = 1
    for i,j in zip(outcome, measurement):
        if int(i) == 1 and j == 'Z':
            energy = -1*energy
    return sign*energy

def ibm_no_reset_modification(outcome, reset):
    if len(reset) == 0:
        return 1
    mcm_results = [int(o) for o in outcome[:len(reset)]]
    modifier = _np.sum(_np.array(mcm_results)*_np.array(reset))
    return (-1)**modifier

def loqs_no_reset_modifier(bitstring, circuit_resets):
    if circuit_resets == None:
        return 1
    mcm_bitstring = bitstring[:len(circuit_resets)]
    modifier = _np.sum(_np.array(mcm_bitstring)*_np.array(circuit_resets))
    return (-1)**modifier

def compute_twoQ_prob(mean_two_q_gates: float, mcm_probability: float, p_2q_mcms: float, num2Qedges_mcms: int, num2Qedges: int):
    no_mcm_prob = 1 - mcm_probability
    return (mean_two_q_gates - mcm_probability*p_2q_mcms*num2Qedges_mcms) / (no_mcm_prob * num2Qedges)

def sample_rb_mcm_mixed_layer_fixed_mcm_count_and_twoQ_gate_count(pspec, qubit_labels = None, mcm_labels = None, one_q_gate_names = None, loqs = False, gate_args_lists = None, rand_state = None, twoQ_probability = .25, mcm_probability = .25, mcm_count = 1, twoQ_count = 1, debug = False):
    if rand_state is None:
        rand_state = _np.random.RandomState()
    if gate_args_lists is None: gate_args_lists = {}
    if qubit_labels is None:
        qubits = list(pspec.qubit_labels[:])  # copy this list
    else:
        assert(isinstance(qubit_labels, (list, tuple))), "SubsetQs must be a list or a tuple!"
        qubits = list(qubit_labels[:])  # copy this list
    assert(mcm_count <= len(qubits) or mcm_probability == 0) # Can't add more MCMs than you have qubits  
    assert(2*twoQ_count <= len(qubits) or twoQ_probability == 0), f'You don\'t have enough qubits to add {twoQ_count} two-qubit gates.'
    assert(mcm_labels is not None), 'Need MCM labels.'
    
    sampled_layer = []
    if loqs == True:
        pre_layer, post_layer = [], []

    num_mcms = 0
    mcm_locs = []
    edgelist = pspec.compute_2Q_connectivity().edges()
    edgelist = [e for e in edgelist if all([q in qubits for q in e])]
    selectededges = []
    
    include_mcms = (rand_state.random() <= mcm_probability) # Decide if you are going to include the MCMs
    if include_mcms is True: 
        mcm_qubits = rand_state.choice(qubits, mcm_count, replace = False)
        mcm_qubits = [str(qubit) for qubit in mcm_qubits]
        num_mcms += mcm_count
    else:
        mcm_qubits = []
    if debug:
        print(f'We included MCMs: {include_mcms}')
    
    # Pick out an independent set of edges
    while len(edgelist) > 0:
        edge = edgelist[rand_state.randint(0, len(edgelist))]
        selectededges.append(edge)
        # Delete all edges containing these qubits.
        edgelist = [e for e in edgelist if not any([q in e for q in edge])]
    if debug:
        print(selectededges)
    if len(qubits) > 1:
        mean_two_q_gates = twoQ_count * twoQ_probability
    else:
        mean_two_q_gates = 0
    if debug:
        print(f'Mean 2Q gate count: {mean_two_q_gates}')
    num2Qedges = len(selectededges)
    num2Qedges_mcms = max(0, num2Qedges - mcm_count) # this is conservative
    if debug:
        print(f'Number of 2Q edges without MCMs: {num2Qedges_mcms}')
    if num2Qedges_mcms > 0:
        p_2q_mcms = twoQ_probability
    else:
        p_2q_mcms = 0
    if debug:
        print(f'Prob 2Q with MCMs: {p_2q_mcms}')
    if mean_two_q_gates > 0:
        if include_mcms is True:
            twoQprob = p_2q_mcms
        elif include_mcms is False and p_2q_mcms == 0:
            twoQprob = 1 / (1 - mcm_probability) * twoQ_probability
        else:
            twoQprob = twoQ_probability
    else:
        twoQprob = 0
    if debug == True:   
        print(f'Adjusted 2Q gate prob: {twoQprob}')
    
    assert(0 <= twoQprob <= 1), 'Device may have insufficient connectivity. Reduce your MCM or two-qubit gate density.'

    selectededges = [edge for edge in selectededges if not any([q in mcm_qubits for q in edge])] #verify this works in the 3Q case
    unusedqubits = _copy.copy(qubits)
    for q in mcm_qubits:
        mcm_locs.append(q)
        gate_label = Label(mcm_labels['mcm'], q)
        sampled_layer.append(gate_label)
            
        if loqs == True:
            pregate_label = Label(mcm_labels['pre'], q)
            postgate_label = Label(mcm_labels['post'], q)
            pre_layer.append(pregate_label)
            post_layer.append(postgate_label)
            
        del unusedqubits[unusedqubits.index(q)]   
    
    ops_on_qubits = pspec.compute_ops_on_qubits()
    edges = []
    edge_indices = None
    # Decide if you are going to add any 2Q gates
    if bool(rand_state.binomial(1, twoQprob)): edge_indices = rand_state.choice(len(selectededges), twoQ_count, replace = False)
    if edge_indices is not None: edges = _np.array(selectededges)[edge_indices]
    for edge in edges:
        edge = tuple(edge)
        possibleops = ops_on_qubits[edge]
        argless_gate_label = possibleops[rand_state.randint(0, len(possibleops))]
        if argless_gate_label.name not in gate_args_lists.keys():
            sampled_layer.append(argless_gate_label)
        else:
            possibleargs = gate_args_lists[argless_gate_label.name]
            args = possibleargs[rand_state.randint(0, len(possibleargs))]
            sampled_layer.append(_lbl.Label(argless_gate_label.name, edge, args=args))

        for q in edge:
            del unusedqubits[unusedqubits.index(q)]
    
    if one_q_gate_names is None or len(one_q_gate_names) > 0:
        for q in unusedqubits:
            if one_q_gate_names is None:
                possibleops = ops_on_qubits[(q,)]
            else:
                print(one_q_gate_names)
                print(ops_on_qubits[(q,)])
                possibleops = [gate_lbl for gate_lbl in ops_on_qubits[(q,)] if gate_lbl.name in one_q_gate_names]
            gate_label = possibleops[rand_state.randint(0, len(possibleops))]
            sampled_layer.append(gate_label)

    if loqs == True:
        return {'pre-layers': [pre_layer], 'post-layers': [post_layer], 'mixed-layer': [sampled_layer]}, num_mcms, mcm_locs
    
    return {'mixed-layer': [sampled_layer]}, num_mcms, mcm_locs

def sample_rb_mcm_mixed_layer_fixed_mcm_count(pspec, qubit_labels = None, mcm_labels = None, one_q_gate_names = None, loqs = False, gate_args_lists = None, rand_state = None, two_q_gate_density = .25, mcm_probability = .25, mcm_count = 1, debug = False):
    if rand_state is None:
        rand_state = _np.random.RandomState()
    if gate_args_lists is None: gate_args_lists = {}
    if qubit_labels is None:
        qubits = list(pspec.qubit_labels[:])  # copy this list
    else:
        assert(isinstance(qubit_labels, (list, tuple))), "SubsetQs must be a list or a tuple!"
        qubits = list(qubit_labels[:])  # copy this list
    assert(mcm_count <= len(qubits) or mcm_probability == 0) # Can't add more MCMs than you have qubits    
    assert(mcm_labels is not None), 'Need MCM labels.'
    
    sampled_layer = []
    if loqs == True:
        pre_layer, post_layer = [], []

    num_mcms = 0
    mcm_locs = []
    edgelist = pspec.compute_2Q_connectivity().edges()
    edgelist = [e for e in edgelist if all([q in qubits for q in e])]
    selectededges = []
    
    include_mcms = (rand_state.random() <= mcm_probability) # Decide if you are going to include the MCMs
    if include_mcms is True: 
        mcm_qubits = rand_state.choice(qubits, mcm_count, replace = False)
        mcm_qubits = [str(qubit) for qubit in mcm_qubits]
        num_mcms += mcm_count
    else:
        mcm_qubits = []
    if debug:
        print(include_mcms)
    
    # Pick out an independent set of edges
    while len(edgelist) > 0:
        edge = edgelist[rand_state.randint(0, len(edgelist))]
        selectededges.append(edge)
        # Delete all edges containing these qubits.
        edgelist = [e for e in edgelist if not any([q in e for q in edge])]
    if debug:
        print(selectededges)
    if len(qubits) > 1:
        mean_two_q_gates = len(qubits) * two_q_gate_density / 2
    else:
        mean_two_q_gates = 0
    if debug:
        print(f'Mean 2Q gates: {mean_two_q_gates}')
    num2Qedges = len(selectededges)
    num2Qedges_mcms = max(0, num2Qedges - mcm_count) # this is conservative
    if debug:
        print(num2Qedges_mcms)
    if num2Qedges_mcms > 0:
        p_2q_mcms = min(1, mean_two_q_gates / num2Qedges_mcms)
    else:
        p_2q_mcms = 0
    if mean_two_q_gates > 0:
        if include_mcms is True:
            twoQprob = p_2q_mcms
        else:
            twoQprob = compute_twoQ_prob(mean_two_q_gates, mcm_probability, p_2q_mcms, num2Qedges_mcms, num2Qedges)
    else:
        twoQprob = 0
    if debug:    
        print(twoQprob)
    
    assert(0 <= twoQprob <= 1), 'Device may have insufficient connectivity. Reduce your MCM or two-qubit gate density.'

    selectededges = [edge for edge in selectededges if not any([q in mcm_qubits for q in edge])] #verify this works in the 3Q case
    unusedqubits = _copy.copy(qubits)
    for q in mcm_qubits:
        mcm_locs.append(q)
        gate_label = Label(mcm_labels['mcm'], q)
        sampled_layer.append(gate_label)
            
        if loqs == True:
            pregate_label = Label(mcm_labels['pre'], q)
            postgate_label = Label(mcm_labels['post'], q)
            pre_layer.append(pregate_label)
            post_layer.append(postgate_label)
            
        del unusedqubits[unusedqubits.index(q)]   
    
    ops_on_qubits = pspec.compute_ops_on_qubits()
    for edge in selectededges:
        if bool(rand_state.binomial(1, twoQprob)):
            # The two-qubit gates on that edge.
            possibleops = ops_on_qubits[edge]
            argless_gate_label = possibleops[rand_state.randint(0, len(possibleops))]
            if argless_gate_label.name not in gate_args_lists.keys():
                sampled_layer.append(argless_gate_label)
            else:
                possibleargs = gate_args_lists[argless_gate_label.name]
                args = possibleargs[rand_state.randint(0, len(possibleargs))]
                sampled_layer.append(_lbl.Label(argless_gate_label.name, edge, args=args))

            for q in edge:
                del unusedqubits[unusedqubits.index(q)]
    
    if one_q_gate_names is None or len(one_q_gate_names) > 0:
        for q in unusedqubits:
            if one_q_gate_names is None:
                possibleops = ops_on_qubits[(q,)]
            else:
                print(one_q_gate_names)
                print(ops_on_qubits[(q,)])
                possibleops = [gate_lbl for gate_lbl in ops_on_qubits[(q,)] if gate_lbl.name in one_q_gate_names]
            gate_label = possibleops[rand_state.randint(0, len(possibleops))]
            sampled_layer.append(gate_label)

    if loqs == True:
        return {'pre-layers': [pre_layer], 'post-layers': [post_layer], 'mixed-layer': [sampled_layer]}, num_mcms, mcm_locs
    
    return {'mixed-layer': [sampled_layer]}, num_mcms, mcm_locs

def sample_rb_mcm_mixed_layer_by_edgegrab(pspec, qubit_labels=None, mcm_labels = None, two_q_gate_density=0.25, one_q_gate_names=None, loqs = False, gate_args_lists=None, rand_state=None, mcm_only_layers = False, mcm_density = .25):
    assert(mcm_labels is not None), 'Need MCM labels.'
    if gate_args_lists is None: gate_args_lists = {}
        
    if qubit_labels is None:
        qubits = list(pspec.qubit_labels[:])  # copy this list
    else:
        assert(isinstance(qubit_labels, (list, tuple))), "SubsetQs must be a list or a tuple!"
        qubits = list(qubit_labels[:])  # copy this list
    if rand_state is None:
        rand_state = _np.random.RandomState()
        
    assert(two_q_gate_density + mcm_density <= 1), 'You cannot have more than 100\% MCMs and 2Q gates.'

    # Prep the sampling variables.
    sampled_layer = []
    if loqs == True:
        pre_layer, post_layer = [], []
    
    num_mcms = 0
    mcm_locs = []
    edgelist = pspec.compute_2Q_connectivity().edges()
    edgelist = [e for e in edgelist if all([q in qubits for q in e])]
    selectededges = []

    # Pick out an independent set of edges
    while len(edgelist) > 0:

        edge = edgelist[rand_state.randint(0, len(edgelist))]
        selectededges.append(edge)
        # Delete all edges containing these qubits.
        edgelist = [e for e in edgelist if not any([q in e for q in edge])]

    num2Qgates = len(selectededges)
    if len(qubits) > 1:
        mean_two_q_gates = len(qubits) * two_q_gate_density / 2
    else:
        mean_two_q_gates = 0
    assert(num2Qgates >= mean_two_q_gates), "Device has insufficient connectivity!"

    if mean_two_q_gates > 0:
        twoQprob = mean_two_q_gates / num2Qgates
    else:
        twoQprob = 0

    unusedqubits = _copy.copy(qubits)
    ops_on_qubits = pspec.compute_ops_on_qubits()
    for edge in selectededges:
        if bool(rand_state.binomial(1, twoQprob)):
            # The two-qubit gates on that edge.
            possibleops = ops_on_qubits[edge]
            argless_gate_label = possibleops[rand_state.randint(0, len(possibleops))]
            if argless_gate_label.name not in gate_args_lists.keys():
                sampled_layer.append(argless_gate_label)
            else:
                possibleargs = gate_args_lists[argless_gate_label.name]
                args = possibleargs[rand_state.randint(0, len(possibleargs))]
                sampled_layer.append(_lbl.Label(argless_gate_label.name, edge, args=args))

            for q in edge:
                del unusedqubits[unusedqubits.index(q)]
                
    total_remaining_qubits = len(unusedqubits)
    if total_remaining_qubits > 0:
        mcmprob = mcm_density / (1 - twoQprob)
    else:
        mcmprob = 0
    unusedqubits_copy = _copy.copy(unusedqubits)
    for q in unusedqubits:
        if bool(rand_state.binomial(1, mcmprob)):
            num_mcms += 1
            mcm_locs.append(q)
            gate_label = Label(mcm_labels['mcm'], q)
            sampled_layer.append(gate_label)
            
            if loqs == True:
                pregate_label = Label(mcm_labels['pre'], q)
                postgate_label = Label(mcm_labels['post'], q)
                pre_layer.append(pregate_label)
                post_layer.append(postgate_label)
            
            del unusedqubits_copy[unusedqubits_copy.index(q)]
        if (num_mcms == len(qubits) - 1) and (mcm_only_layers is False):
            break
    
    unusedqubits = unusedqubits_copy
                   
    if one_q_gate_names is None or len(one_q_gate_names) > 0:
        for q in unusedqubits:
            if one_q_gate_names is None:
                possibleops = ops_on_qubits[(q,)]
            else:
                print(one_q_gate_names)
                print(ops_on_qubits[(q,)])
                possibleops = [gate_lbl for gate_lbl in ops_on_qubits[(q,)] if gate_lbl.name in one_q_gate_names]
            gate_label = possibleops[rand_state.randint(0, len(possibleops))]
            sampled_layer.append(gate_label)

    if loqs == True:
        return {'pre-layers': [pre_layer], 'post-layers': [post_layer], 'mixed-layer': [sampled_layer]}, num_mcms, mcm_locs
    
    return {'mixed-layer': [sampled_layer]}, num_mcms, mcm_locs

def sample_rb_mcm_circuit_layer_with_1Q_gates(pspec, mixed_layer_sampler = sample_rb_mcm_mixed_layer_by_edgegrab, mixed_layer_sampler_kwargs = {'mcm_density': .25, 'mcm_only_layers': True, 'two_q_gate_density': .25}, qubit_labels=None, mcm_labels = None, one_q_gate_names=None, loqs = True, mcm_only_layers = True, gate_args_lists=None, rand_state=None):
    
    assert(mcm_labels is not None), 'Need MCM labels.'
    if gate_args_lists is None: gate_args_lists = {}
        
    if qubit_labels is None:
        qubits = list(pspec.qubit_labels[:])  # copy this list
    else:
        assert(isinstance(qubit_labels, (list, tuple))), "SubsetQs must be a list or a tuple!"
        qubits = list(qubit_labels[:])  # copy this list
    if rand_state is None:
        rand_state = _np.random.RandomState()
        
    num_qubits = len(qubits)
    if num_qubits == 1:
        mcm_only_layers = True
    mixed_layers, mcms, mcm_locs = mixed_layer_sampler(pspec, qubit_labels=qubit_labels, mcm_labels = mcm_labels, one_q_gate_names=one_q_gate_names, loqs = loqs, gate_args_lists=gate_args_lists, rand_state=rand_state, **mixed_layer_sampler_kwargs)
    
    unused_qubits = _np.setdiff1d(qubits, mcm_locs)
    gates = {q: 'Gc{}'.format(rand_state.choice(_np.arange(0,24))) for q in unused_qubits}
    # gates = {q: 'Gc{}'.format(rand_state.choice(_np.arange(0,24))) for q in unused_qubits}
    pre_oneq_layer = [Label(gates[q], str(q)) for q in unused_qubits]
    
    gates = {q: 'Gc{}'.format(rand_state.choice(_np.arange(0,24))) for q in unused_qubits}
    # gates = {q: 'Gc{}'.format(rand_state.choice(_np.arange(0,24))) for q in unused_qubits}
    post_oneq_layer = [Label(gates[q], str(q)) for q in unused_qubits]
    
    layers = {'mixed-layer': mixed_layers['mixed-layer']}
    layers['pre-oneq-layers'] = [pre_oneq_layer]
    layers['post-oneq-layers'] = [post_oneq_layer]
    
    if loqs == True:
        layers['pre-meas-layer'] = mixed_layers['pre-layers']
        layers['post-meas-layer'] = mixed_layers['post-layers']
    
    return layers, mcms, mcm_locs
    
def update_gate(gate, qubit_pointers: dict):
    gate_type, gate_qubits = gate[0], gate[1:]
    return Label(gate_type, tuple([qubit_pointers[qubit][-1] for qubit in gate_qubits]))

def process_rbmcm_layer(circuit_layers: dict, layer_order: list, qubit_pointers: dict, next_available_qubit: int, mcm_labels: dict):
    """
    Processes the output of any rbmcm layer sampler so that it is a circuit layer
    in the blown up circuit required for RBMCM analysis.
    
    We assume that the mixed layer only contains one layer!!!
    
    Returns: The circuit layer
    """
    new_layers = {key: [] for key in circuit_layers}
    
    mcm_circuit = _cir.Circuit('', line_labels = list(qubit_pointers.keys()), editable = True)
    no_mcm_circuit = _cir.Circuit('', line_labels = list(qubit_pointers.keys()), editable = True)
    new_next_available_qubit = next_available_qubit
    new_qubit_pointers = {qubit: _copy.deepcopy(qubit_pointers[qubit]) for qubit in qubit_pointers}
    meas_qubits, meas_qubit_og_circuit = [], []
    # mcm_layers, no_mcm_layers = [], []
        
    for key in layer_order:
        layers = circuit_layers[key]
        for i in range(len(layers)):
            layer = layers[i]
            new_layer, new_no_mcm_layer = [], []
            for gate in layer:
                new_gate = update_gate(gate, qubit_pointers)
                new_layer.append(new_gate)
                if new_gate[0] != mcm_labels['mcm'] and new_gate[0] != mcm_labels['pre'] and new_gate[0] != mcm_labels['post']:
                    new_no_mcm_layer.append(new_gate)
                if key == 'mixed-layer':
                    gate_type = gate[0]
                    if gate_type == mcm_labels['mcm']:
                        meas_qubit = gate[1]
                        meas_qubit_og_circuit.append(int(meas_qubit[1:]))
                        meas_qubits.append(int(qubit_pointers[meas_qubit][-1][1:]))
                        new_qubit_pointers[meas_qubit].append('Q{}'.format(new_next_available_qubit))
                        new_next_available_qubit += 1
            new_circuit_layer = _cir.Circuit([new_layer], line_labels = list(qubit_pointers.keys()))
            mcm_circuit.append_circuit_inplace(new_circuit_layer)
            if len(new_no_mcm_layer) > 0:
                new_no_mcm_circuit_layer = _cir.Circuit([new_no_mcm_layer], line_labels = list(qubit_pointers.keys()))
                no_mcm_circuit.append_circuit_inplace(new_no_mcm_circuit_layer)
    
    mcm_circuit.done_editing()
    no_mcm_circuit.done_editing()
    
    return mcm_circuit, no_mcm_circuit, new_qubit_pointers, new_next_available_qubit, meas_qubits, meas_qubit_og_circuit


'''
def process_loqs_layer(loqs_layers: dict, qubit_pointers: dict, next_available_qubit: int, mcm_labels: dict):
    """
    Processes the output of any rbmcm layer sampler so that it is a circuit layer
    in the blown up circuit required for RBMCM analysis.
    
    We assume that the mixed layer only contains one layer!!!
    
    Returns: The circuit layer
    """
    new_layers = {key: [] for key in loqs_layers}
    new_next_available_qubit = next_available_qubit
    new_qubit_pointers = {qubit: _copy.deepcopy(qubit_pointers[qubit]) for qubit in qubit_pointers}
    meas_qubits, meas_qubit_og_circuit = [], []
    mcm_layers, no_mcm_layers = [], []
        
    for key, layers in loqs_layers.items():
        for i in range(len(layers)):
            layer = layers[i]
            new_layer = []
            for gate in layer:
                new_gate = update_gate(gate, qubit_pointers)
                new_layer.append(new_gate)
                if key == 'mixed-layer':
                    gate_type = gate[0]
                    if gate_type == mcm_labels['mcm']:
                        meas_qubit = gate[1]
                        meas_qubit_og_circuit.append(int(meas_qubit[1:]))
                        meas_qubits.append(int(qubit_pointers[meas_qubit][-1][1:]))
                        new_qubit_pointers[meas_qubit].append('Q{}'.format(new_next_available_qubit))
                        new_next_available_qubit += 1
            new_layers[key].append(new_layer)
    
    layer_order = ['pre-oneq-layers', 'pre-meas-layer', 'mixed-layer', 'post-meas-layer', 'post-oneq-layers']
    for key in layer_order:
        layer = new_layers[key][0]
        mcm_layers.append(layer)
        if layer != []:
            no_mcm_layer = [i for i in layer if (i[0] != mcm_labels['mcm'] and i[0] != mcm_labels['pre'] and i[0] != mcm_labels['post'])]
            no_mcm_layers.append(no_mcm_layer)
        
    mcm_circuit = _cir.Circuit(mcm_layers, line_labels = list(qubit_pointers.keys()))
    no_mcm_circuit = _cir.Circuit(no_mcm_layers, line_labels = list(qubit_pointers.keys()))
    
    return mcm_circuit, no_mcm_circuit, new_qubit_pointers, new_next_available_qubit, meas_qubits, meas_qubit_og_circuit
'''

def pauli_to_z(paulis, qubits, qubit_labels, rand_state = None):
    '''
    Turns a collection of single qubit paulis into Z paulis acting on the specified qubits
    within a larger collection of qubits.
    '''
    if rand_state == None:
        rand_state = _np.random.RandomState()
    circuit = []
    for pauli, qubit in zip(paulis, qubits):
        if pauli == 'Y':
            circuit.append(Label('Gc1','Q{}'.format(qubit)))
        elif pauli == 'X':
            circuit.append(('Gc12','Q{}'.format(qubit)))
        elif pauli == 'I':
            rand_clifford = str(rand_state.choice(_np.arange(24)))
            # rand_clifford = str(_np.random.choice(_np.arange(24)))
            circuit.append(('Gc'+rand_clifford,'Q{}'.format(qubit)))
        else:
            circuit.append(('Gc0', 'Q{}'.format(qubit)))
            
    measure_circuit = _cir.Circuit(circuit, line_labels = qubit_labels).parallelize()
    
    return measure_circuit

def process_measure_layer(native_measure_layer, qubit_pointers, new_qubit_labels):
    if len(native_measure_layer) == 0:
        return _cir.Circuit([[]], line_labels = new_qubit_labels).parallelize()
    elif type(native_measure_layer[0]) == pygsti.baseobjs.label.LabelTup:
        label = native_measure_layer[0]
        return _cir.Circuit([[Label(label[0], qubit_pointers[label[1]][-1])]], line_labels = new_qubit_labels)
    else:
        labels = [Label(label[0], qubit_pointers[label[1]][-1]) for label in native_measure_layer[0]]
        return _cir.Circuit([labels], line_labels = new_qubit_labels)

def create_rb_with_mcm_circuit(pspec, length, qubit_labels, mcm_labels, loqs, debug = True, layer_sampler = sample_rb_mcm_circuit_layer_with_1Q_gates, layer_sampler_kwargs = {'mixed_layer_sampler': sample_rb_mcm_mixed_layer_by_edgegrab, 'mixed_layer_sampler_kwargs': {'mcm_density': .25, 'mcm_only_layers': True, 'two_q_gate_density': .5}}, pauli_sampler = birb.generic_pauli_sampler, pauli_sampler_kwargs = {'include_identity': True}, mcm_reset = True, seed = None):
    
    rand_state = _np.random.RandomState(seed)
    
    if qubit_labels is not None: n = len(qubit_labels)
    else: 
        n = pspec.num_qubits
        qubit_labels = pspec.qubit_labels
    
    if loqs == True:   
        layer_order = ['pre-oneq-layers', 'pre-meas-layer', 'mixed-layer', 'post-meas-layer', 'post-oneq-layers']
    else:
        layer_order = ['pre-oneq-layers', 'mixed-layer', 'post-oneq-layers']
    
    
    # Sample the "core" circuit layers. This must be done first to determine the size of the Pauli to select.
    
    layers = []
    check_circuit = _cir.Circuit([],line_labels = qubit_labels, editable = True)
    mcm_count = 0
    for i in range(length):
        layer, mcms, mcm_locs = layer_sampler(pspec, qubit_labels=None, mcm_labels = mcm_labels,
                                              loqs = loqs,
                                              rand_state = rand_state, **layer_sampler_kwargs)
 
        mcm_count += mcms
        layers.append(layer)
        for layer_type in layer_order:
            for l in layer[layer_type]:
                check_circuit.append_circuit_inplace(_cir.Circuit([l], line_labels = qubit_labels))
            
    pauli_size = n + mcm_count
    new_qubit_labels = ['Q{}'.format(i) for i in range(pauli_size)]
    
    big_pspec =  pygsti.processors.QubitProcessorSpec(num_qubits=pauli_size,
                                                      gate_names=['Gc{}'.format(i) for i in range(24)] + ['Gcnot'],
                                                      availability = {'Gcnot': 'all-permutations'},
                                                      qubit_labels=new_qubit_labels)
    
    big_compilations = {'absolute': CCR.create_standard(big_pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity = 0),
                'paulieq': CCR.create_standard(big_pspec, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity = 0)}
    
    next_available_qubit = n
    qubit_pointers = {qubit: [qubit] for qubit in new_qubit_labels}
    unmeasured_qubits = [i for i in range(n + mcm_count)]
    og_unmeasured_qubits = [i for i in range(n)]
    measurement_order = []
    if mcm_reset == False:
        reset_matters = []
    else: reset_matters = None
    
    # Sample the Pauli
    
    rand_pauli, rand_sign, pauli_circuit = birb.sample_random_pauli(n = n + mcm_count, pspec = big_pspec, absolute_compilation = big_compilations['absolute'], circuit = True, pauli_sampler = pauli_sampler, pauli_sampler_kwargs = pauli_sampler_kwargs, rand_state = rand_state)
    s_pauli_circuit, p_pauli_circuit = _symp.symplectic_rep_of_clifford_circuit(pauli_circuit, pspec = big_pspec)
    current_pauli = rand_pauli
    '''if debug: 
        print(f'The initial Pauli is {current_pauli}.')
        print(f'It has sign {rand_sign}.\n')
        print(f'The full Pauli is: {rand_pauli}')'''

    # Sample the stablizer
    
    s_inputstate, p_inputstate, s_init_layer, p_init_layer, prep_circuit = birb.sample_stabilizer(rand_pauli, rand_sign, rand_state)
    prep_circuit = birb.compose_initial_cliffords(prep_circuit)
    '''if debug:
        print(f'This is the initial state: \n{s_inputstate}')
        print(f'It has a phase matrix of: {p_inputstate}')'''
    
    # build the initial layer of the blown up circuit
    initial_circuit = _cir.Circuit([[(prep_circuit[i], new_qubit_labels[i]) for i in range(len(new_qubit_labels))]])
    expanded_circuit = initial_circuit.copy(editable = True)
    no_mcm_circuit = initial_circuit.copy(editable = True)
    s_current, p_current = s_inputstate, p_inputstate
    
    # build the initial layer of the native circuit
    initial_native_circuit = _cir.Circuit([[(prep_circuit[i], qubit_labels[i]) for i in range(len(qubit_labels))]])
    native_circuit = initial_native_circuit.copy(editable = True)
    
    # Here we blow up the circuit and create the native circuit. We do these together as we need to 
    # forward simulate the Pauli in order to build both.
    
    layer_count = 0
    
    if debug: print(f'Initial state: {[_symp.pauli_z_measurement(s_current, p_current, qubit)[0] for qubit in range(n+mcm_count)]}')

    for layer in layers:
        # Here we build an expanded circuit layer (with and without MCMs). We also determine which qubits were measured, and
        # how we ought to match up native qubits with qubits in the expanded picture.
        new_layer, new_no_mcm_layer, new_qubit_pointers, new_next_available_qubit, meas_qubits, og_meas_qubits = process_rbmcm_layer(layer, layer_order, qubit_pointers = qubit_pointers, next_available_qubit = next_available_qubit, mcm_labels = mcm_labels)
        '''print('We just processed the layer. Here are the results: \n')
        print(f'The new layer: \n{new_layer}')
        print(f'The new no mcm layer: \n{new_no_mcm_layer}')
        print(f'The new qubit_pointers: \n{new_qubit_pointers}')
        print(f'The new next available_qubit: \n{new_next_available_qubit}')
        print(f'The meas qubits: \n{meas_qubits}')
        print(f'The og meas qubits: \n{og_meas_qubits}')'''
        
        '''new_layer, new_no_mcm_layer, new_qubit_pointers, new_next_available_qubit, meas_qubits, og_meas_qubits = process_loqs_layer(layer, 
                                                                             qubit_pointers = qubit_pointers, 
                                                                             next_available_qubit = next_available_qubit, 
                                                                             mcm_labels = mcm_labels)'''
                                                                           
        
        #if debug:
        #    print(f'This is the new layer: \n{new_layer}')
        #    print(f'This is the new layer without mcms: \n{new_no_mcm_layer}')
        #    print(f'These are the new qubit pointers: {new_qubit_pointers}')
        #    print(f'This is the new next available qubit: {new_next_available_qubit}')
        #    print(f'These qubits were measured: {meas_qubits}')
        
        native_layer_circuits = {key: _cir.Circuit(layer[key], line_labels = qubit_labels) for key in layer}
        
        # if debug:
        #    print(f'These are the native layer circuits: \n{native_layer_circuits}')
        
        unmeasured_qubits = _np.setdiff1d(unmeasured_qubits, meas_qubits)
        og_unmeasured_qubits = _np.setdiff1d(og_unmeasured_qubits, og_meas_qubits)
        measurement_order = measurement_order + meas_qubits
        
        # mcm locations are stored in meas_qubits
        # we use this to determine any measurement layer that needs to be added
        
        current_pauli = [i[0] for i in _symp.find_pauli_layer(p_pauli_circuit, [j for j in range(n+mcm_count)])]
        active_paulis = [current_pauli[i] for i in meas_qubits]
        '''if debug:
            print(f'This is the pauli before the layer: {current_pauli}')'''
            
        
        # generate the measurement layer for the non-blown up circuit
        # og_meas_qubits tells me which qubits were measured
        # we can then use qubit_points to tell us which Pauli entry that qubit is currently pointing to
        # We then need to go from that entry in the current Pauli to a Z Pauli.
        
        native_measure_circuit = pauli_to_z(active_paulis, og_meas_qubits, qubit_labels, rand_state)
        full_pre_measure_circuit = native_measure_circuit.copy(editable = True)
        full_pre_measure_circuit.append_circuit_inplace(native_layer_circuits['pre-oneq-layers'])
        full_pre_measure_circuit = full_pre_measure_circuit.parallelize()
        native_circuit.append_circuit_inplace(full_pre_measure_circuit)
        
        # generate the measurement layer for the blown up circuit
        
        measure_circuit = process_measure_layer(native_measure_circuit, qubit_pointers, new_qubit_labels)
        # if debug: print(f'The enlarged measure circut: \n {measure_circuit}')
        s_measure_circuit, p_measure_circuit = _symp.symplectic_rep_of_clifford_circuit(measure_circuit, pspec = big_pspec)
        s_measure_circuit_inv, p_measure_circuit_inv = _symp.inverse_clifford(s_measure_circuit, p_measure_circuit)
        
        expanded_circuit.append_circuit_inplace(measure_circuit)
        no_mcm_circuit.append_circuit_inplace(measure_circuit)
        '''print('We just appended the measure circuit and the full_pre_measure_circuit')
        print(f'The expanded circuit: \n{expanded_circuit}')
        print(f'The no mcm circuit: \n{no_mcm_circuit}')
        print(f'The native circuit: \n{native_circuit}')
        print(f'We added this circuit to the expanded circuit: \n{measure_circuit}')
        print(f'We added this circuit to the native circuit: \n{full_pre_measure_circuit}')'''
        
        # append the new native layer to the native circuit
        
        if loqs is True:
            native_circuit.append_circuit_inplace(native_layer_circuits['pre-meas-layer'])
        native_circuit.append_circuit_inplace(native_layer_circuits['mixed-layer'])
        if loqs is True:
            native_circuit.append_circuit_inplace(native_layer_circuits['post-meas-layer'])
                
        # append the new layer to the blown up circuit
        
        expanded_circuit.append_circuit_inplace(new_layer) 
        no_mcm_circuit.append_circuit_inplace(new_no_mcm_layer)
        '''print('We just appended the mixed layer to the native and the expanded circuit.')
        print(f'This is the expanded circuit: \n{expanded_circuit}')
        print(f'The no mcm circuit: \n{no_mcm_circuit}')
        print(f'This is the native circuit: \n{native_circuit}')
        print(f'This was the new layer that was added to the expanded circuit: \n{new_layer}')
        mlayer = native_layer_circuits['mixed-layer']
        print(f'This was the native layer that was added to the native circuit: \n{mlayer}')'''
        
        # let's push the Pauli forward by this layer. We use the no_mcm_layer (and the measure layer) to do so as anywhere there was an MCM we are done with the circuit.
        
        s_new_layer, p_new_layer = _symp.symplectic_rep_of_clifford_circuit(new_no_mcm_layer, pspec = big_pspec) # U
        s_new_layer_inv, p_new_layer_inv = _symp.inverse_clifford(s_new_layer, p_new_layer) # U^(-1)
        
        s_pauli_circuit, p_pauli_circuit = _symp.compose_cliffords(s_measure_circuit_inv, p_measure_circuit_inv, s_pauli_circuit, p_pauli_circuit) # PM^(-1)
        s_pauli_circuit, p_pauli_circuit = _symp.compose_cliffords(s_new_layer_inv, p_new_layer_inv, s_pauli_circuit, p_pauli_circuit) # PM^(-1)U^(-1)
        s_pauli_circuit, p_pauli_circuit = _symp.compose_cliffords(s_pauli_circuit, p_pauli_circuit, s_measure_circuit, p_measure_circuit) # MPM^(-1)U^(-1)
        s_pauli_circuit, p_pauli_circuit = _symp.compose_cliffords(s_pauli_circuit, p_pauli_circuit, s_new_layer, p_new_layer) # UMPM^(-1)U^(-1)
        
        # let's keep track of the current state (ignoring mcms)
        
        s_current, p_current = _symp.apply_clifford_to_stabilizer_state(s_measure_circuit, p_measure_circuit, s_current, p_current)
        s_current, p_current = _symp.apply_clifford_to_stabilizer_state(s_new_layer, p_new_layer, s_current, p_current)
        
        # update our blow up pointers
        
        next_available_qubit = new_next_available_qubit
        qubit_pointers = new_qubit_pointers
        
        # add in the post-MCM prep layer for the native circuit
        
        og_meas_qubit_labels = ['Q{}'.format(i) for i in og_meas_qubits]
        next_pauli_slice = {i: int(qubit_pointers['Q{}'.format(i)][-1][1:]) for i in og_meas_qubits}
        '''
        print(f'These are the next Pauli indices: {next_pauli_slice}')
        print(f'This is the current Pauli: {current_pauli}')
        print(f'These are the actual Pauli entries: {[current_pauli[next_pauli_slice[i]] for i in next_pauli_slice]}')
        '''
        if mcm_reset == False:
            pauli_slice = [current_pauli[next_pauli_slice[i]] for i in next_pauli_slice]
            #print(f'This is the next pauli slice: {pauli_slice}.')
            reset_matters = reset_matters + [True if P != 'I' else False for P in pauli_slice] # [True if rand_pauli[int(qubit_pointers['Q{}'.format(i)][-1][1:])] != 'I' else False for i in og_meas_qubits]
            #print(f'This is the reset logic: {reset_matters}\n')
        prep_layer = _cir.Circuit([[(prep_circuit[next_pauli_slice[i]], 'Q{}'.format(i)) for i in og_meas_qubits]], line_labels = qubit_labels, editable = True)
        # if debug: print(f'The post-measurement prep layer: \n{prep_layer}')
        prep_layer.append_circuit_inplace(native_layer_circuits['post-oneq-layers'])
        prep_layer = prep_layer.parallelize()
        
        native_circuit.append_circuit_inplace(prep_layer)
        
        '''if debug: print(f'After the layer you have the following Pauli: {[i[0] for i in _symp.find_pauli_layer(p_pauli_circuit, [j for j in range(n+mcm_count)])]}')'''
        if debug: print(f'Current state after the layer: {[_symp.pauli_z_measurement(s_current, p_current, qubit)[0] for qubit in range(n+mcm_count)]}')

        
    # Need to add a measurement layer to handle any qubits that have not been hit by a MCM
    
    current_pauli = [i[0] for i in _symp.find_pauli_layer(p_pauli_circuit, [j for j in range(n+mcm_count)])]
    measurement_order = measurement_order + [int(qubit_pointers[qubit][-1][1:]) for qubit in qubit_labels]
    
    # generate the final measurement layer for the non-blown up circuit
    # for each qubit...find out what they point to...pick up that pauli entry and convert
    
    active_qubits = [int(qubit_pointers[i][-1][1:]) for i in qubit_labels]
    active_paulis = [current_pauli[active_qubit] for active_qubit in active_qubits]
    native_measure_circuit = pauli_to_z(active_paulis, _np.arange(n), qubit_labels, rand_state)
    native_circuit.append_circuit_inplace(native_measure_circuit)
    # Needed because we are using LoQs!!! Use the first line if you want noisy final measurements (that are the same as your MCMs)
    if loqs is True:
        if 'pre_final' in mcm_labels:
            final_measurements = _cir.Circuit([[Label(mcm_labels['pre_final'], qubit) for qubit in qubit_labels],
                                          [Label(mcm_labels['mcm'], qubit) for qubit in qubit_labels]], line_labels = qubit_labels)
        else:
            final_measurements = _cir.Circuit([[Label(mcm_labels['mcm'], qubit) for qubit in qubit_labels]], line_labels = qubit_labels)
    
        native_circuit.append_circuit_inplace(final_measurements)
    
    
    # generate the final measurement layer for the blown up circuit...need to decide what to do at the end....

    measure_circuit = process_measure_layer(native_measure_circuit, qubit_pointers, new_qubit_labels)
    s_measure_circuit, p_measure_circuit = _symp.symplectic_rep_of_clifford_circuit(measure_circuit, pspec = big_pspec)
    s_measure_circuit_inv, p_measure_circuit_inv = _symp.inverse_clifford(s_measure_circuit, p_measure_circuit)
    s_final_state, p_final_state = _symp.apply_clifford_to_stabilizer_state(s_measure_circuit, p_measure_circuit, s_current, p_current)
    
    # Check that we get a Z type Pauli at the end
    
    s_pauli_circuit, p_pauli_circuit = _symp.compose_cliffords(s_measure_circuit_inv, p_measure_circuit_inv, s_pauli_circuit, p_pauli_circuit) # PM^(-1)
    s_pauli_circuit, p_pauli_circuit = _symp.compose_cliffords(s_pauli_circuit, p_pauli_circuit, s_measure_circuit, p_measure_circuit) # MPM^(-1)
    final_pauli = [i[0] for i in _symp.find_pauli_layer(p_pauli_circuit, [j for j in range(n+mcm_count)])]
    
    expanded_circuit.append_circuit_inplace(measure_circuit)
    no_mcm_circuit.append_circuit_inplace(measure_circuit)
    
    # On if you want M error: 
    if 'pre_final' in mcm_labels:
        expanded_final_measurements = _cir.Circuit([[Label(mcm_labels['pre_final'], 'Q{}'.format(active_qubit)) for active_qubit in active_qubits], [Label(mcm_labels['mcm'], 'Q{}'.format(active_qubit)) for active_qubit in active_qubits]], line_labels = new_qubit_labels)
        
    # Otherwise just use this: 
    expanded_final_measurements = _cir.Circuit([[Label(mcm_labels['mcm'], 'Q{}'.format(active_qubit)) for active_qubit in active_qubits]], line_labels = new_qubit_labels)
    expanded_circuit.append_circuit_inplace(expanded_final_measurements)
    no_mcm_circuit.append_circuit_inplace(expanded_final_measurements)
    
    
    expanded_circuit.done_editing()
    no_mcm_circuit.done_editing()
    native_circuit.done_editing()
    check_circuit.done_editing()

    # need to handle the final measurement sign...the Pauli should be Z-type here because we used the measure layer earlier
    measurement = ['I' if i == 'I' else 'Z' for i in current_pauli]
    '''print('We have reached the final measurement stage.')
    print(f'This is the final state\'s s matrix: \n{s_final_state}')
    print(f'This is the final state\'s p matrix: \n{p_final_state}')
    print(f'This is the final measurement: {measurement}')
    print(f'It came from this Pauli: {current_pauli}')'''
          
    if debug: print(f'The pre-measurement pauli: {current_pauli}')
    sign = birb.determine_sign(s_final_state, p_final_state, measurement)
    if debug: print(f'Measurement Sign: {sign}')
    if debug: print(f'Final state: {[_symp.pauli_z_measurement(s_final_state, p_final_state, qubit)[0] for qubit in range(n+mcm_count)]}')

    
    # if debug:
    #    return expanded_circuit, native_circuit, rand_pauli, measurement, measurement_order, sign, no_mcm_circuit, mcm_count, check_circuit, s_final_state, p_final_state, reset_matters, qubit_pointers
    '''if mcm_reset is False:
        print(f'This is the initial pauli: {rand_pauli}')
        print(f'Here\'s the reset logic: {reset_matters}')
        print('\n')'''
    '''print('We have reached the end of the circuit generating process.')
    print(f'This is the final expanded circuit: \n{expanded_circuit}')  
    print(f'This is the final native circuit: \n{native_circuit}')
    print(f'This is the final no mcm circuit: \n{no_mcm_circuit}')'''  
    return native_circuit, measurement, measurement_order, sign, qubit_pointers, reset_matters



def quick_process_rbmcm_layer(circuit_layers: dict, layer_order: list, qubit_pointers: dict, next_available_qubit: int, mcm_labels: dict, current_pauli: list):
    """
    Processes the output of any rbmcm layer sampler. 
    
    Returns:
        - mcm_circuit: pygsti circuit of the benchmark layer,
        - no_mcm_circuit: pygsti circuit of the benchmark layer without the MCMs,
        - new_qubit_pointers: dictionary that matches each qubit with the Pauli entry it is matched up with
        - new_next_available_qubit: integer specifying the next slot in the Pauli that is available
        - meas_qubits: list of the qubits that are measured in this layer (in the expanded circuit)
        - meas_qubit_og_circuit: list of the qubits that are measured in this layer (in the non-expanded circuit)
    """
    q_labels = list(qubit_pointers.keys())
    no_mcm_circuit = _cir.Circuit('', line_labels = q_labels, editable = True)
    new_next_available_qubit = next_available_qubit
    new_qubit_pointers = {qubit: _copy.deepcopy(qubit_pointers[qubit]) for qubit in qubit_pointers}
    measured_qubits, measurement = [], ['I' for i in range(len(qubit_pointers.keys()))]
    no_mcm_mixed_layer = []
    
    layers = circuit_layers['mixed-layer']
    for i in range(len(layers)):
        layer = layers[i]
        new_no_mcm_layer = []
        for gate in layer:
            gate_type = gate[0]
            if gate_type != mcm_labels['mcm']: new_no_mcm_layer.append(gate)
            else:
                meas_qubit = int(gate[1][1:])
                measured_qubits.append(meas_qubit)
                if current_pauli[meas_qubit] != 'I': measurement[meas_qubit] = 'Z'
                new_qubit_pointers[f'Q{meas_qubit}'].append(f'Q{new_next_available_qubit}')
                new_next_available_qubit += 1
        if len(new_no_mcm_layer) > 0:
            new_no_mcm_circuit_layer = _cir.Circuit([new_no_mcm_layer], line_labels = q_labels)
            no_mcm_circuit.append_circuit_inplace(new_no_mcm_circuit_layer)
            
    no_mcm_circuit.done_editing()
    return no_mcm_circuit, new_qubit_pointers, new_next_available_qubit, measured_qubits, measurement

def quick_create_rb_with_mcm_circuit(pspec, length, qubit_labels, mcm_labels, loqs, debug = True, layer_sampler = sample_rb_mcm_circuit_layer_with_1Q_gates, layer_sampler_kwargs = {'mixed_layer_sampler': sample_rb_mcm_mixed_layer_by_edgegrab, 'mixed_layer_sampler_kwargs': {'mcm_density': .25, 'mcm_only_layers': True, 'two_q_gate_density': .5}}, include_identity = True, mcm_reset = True, seed = None):
    
    rand_state = _np.random.RandomState(seed)
    
    if qubit_labels is not None: n = len(qubit_labels)
    else: 
        n = pspec.num_qubits
        qubit_labels = pspec.qubit_labels
    
    if loqs == True:   
        layer_order = ['pre-oneq-layers', 'pre-meas-layer', 'mixed-layer', 'post-meas-layer', 'post-oneq-layers']
    else:
        layer_order = ['pre-oneq-layers', 'mixed-layer', 'post-oneq-layers']
        
        
    # Sample the "core" circuit layers. This must be done first to determine the size of the Pauli to select.
    
    layers = []
    check_circuit = _cir.Circuit([],line_labels = qubit_labels, editable = True)
    mcm_count = 0
    for i in range(length):
        layer, mcms, mcm_locs = layer_sampler(pspec, qubit_labels=None, mcm_labels = mcm_labels,
                                              loqs = loqs,
                                              rand_state = rand_state, **layer_sampler_kwargs)
        
        # if loqs == True: 
        #    mcm_count += mcms
        mcm_count += mcms
        layers.append(layer)
        for layer_type in layer_order:
            for l in layer[layer_type]:
                check_circuit.append_circuit_inplace(_cir.Circuit([l], line_labels = qubit_labels))
                
    pauli_size = n + mcm_count
    new_qubit_labels = ['Q{}'.format(i) for i in range(pauli_size)] # probably don't need these?
    
    big_pspec =  pygsti.processors.QubitProcessorSpec(num_qubits=pauli_size,
                                                      gate_names=['Gc{}'.format(i) for i in range(24)] + ['Gcnot'],
                                                      availability = {'Gcnot': 'all-permutations'},
                                                      qubit_labels=new_qubit_labels)
    
    big_compilations = {'absolute': CCR.create_standard(big_pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity = 0),
                'paulieq': CCR.create_standard(big_pspec, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity = 0)}
    
    next_available_qubit = n
    # qubit_pointers = {int(qubit[1:]): [int(qubit[1:])] for qubit in qubit_labels} # This is modified to just use the qubit labels
    qubit_pointers = {qubit: [qubit] for qubit in qubit_labels}
    unmeasured_qubits = [i for i in range(n + mcm_count)] # This is probably not needed.
    og_unmeasured_qubits = [i for i in range(n)]
    measurement_order = []
    measurement_phases = [] # This is new. It is used to keep track of the phase of the Pauli at each layer a MCM is performed.
    if mcm_reset == False:
        reset_matters = []
    else: reset_matters = None
        
    # Sample the Pauli
    rand_pauli, rand_sign, pauli_circuit = birb.sample_random_pauli(n = n + mcm_count, pspec = big_pspec, absolute_compilation = big_compilations['absolute'], circuit = True, include_identity = include_identity, rand_state = rand_state)
    final_pauli = ['I' for i in range(n+mcm_count)]
    
    # Extract the Pauli on the first n qubits
    
    current_pauli = rand_pauli[:n]
    pauli_circuit = _cir.Circuit([[(current_pauli[i], qubit_labels[i]) for i in range(n)]], line_labels = qubit_labels, editable = True)
    s_pauli_circuit, p_pauli_circuit = _symp.symplectic_rep_of_clifford_circuit(pauli_circuit, pspec = pspec)
    
    # Sample a random n+mcm-sized stablizer state
    
    s_inputstate, p_inputstate, s_init_layer, p_init_layer, prep_circuit = birb.sample_stabilizer(rand_pauli, rand_sign, rand_state)
    prep_circuit = birb.compose_initial_cliffords(prep_circuit)
    '''if debug:
        print(f'This is the initial state: \n{s_inputstate}')
        print(f'This is its phase matrix: {p_inputstate}')
        print([_symp.pauli_z_measurement(s_inputstate, p_inputstate, qubit)[0] for qubit in range(n+mcm_count)])'''

    
    # Begin to create the BiRB+MCM circuit
    native_circuit = _cir.Circuit([[(prep_circuit[i], qubit_labels[i]) for i in range(len(qubit_labels))]], line_labels = qubit_labels, editable = True)
    no_mcm_circuit = native_circuit.copy(editable = True)
    
    s_native_circuit, p_native_circuit = _symp.symplectic_rep_of_clifford_circuit(native_circuit, pspec = pspec)
    s_no_mcm_circuit, p_no_mcm_circuit = _symp.symplectic_rep_of_clifford_circuit(no_mcm_circuit, pspec = pspec)
    
    # Extract the stabilizer state on the first n qubits
    
    s_current_state, p_current_state = _symp.prep_stabilizer_state(n)
    s_current_state, p_current_state = _symp.apply_clifford_to_stabilizer_state(s_native_circuit, p_native_circuit, s_current_state, p_current_state)
    
    ''' At this point you have:
       - An n+mcm_count random Pauli
       - A random sign on the random Pauli
       - A (n+mcm)-qubit stabilizer state (stabilized by the random Pauli when accounting for the sign)
       - An n-qubit state that matches the (n+mcm)-qubit stabilizer state on the first n-qubits
       - A circuit that prepares the n-qubit state
       - A copy of the circuit that prepares the n-qubit state
       - An n-qubit Pauli
       - Symplectic representations of all of the above
    '''
    
    # Now we begin to add the benchmark layers
    '''if debug: 
        print(f'The initial Pauli is {current_pauli}.')
        print(f'It has sign {rand_sign}.\n')
        print(f'The full Pauli is: {rand_pauli}')
        print(f'The initial state is: \n{s_current_state}')
        print(f'Its phase matrix is: {p_current_state}')
        print([_symp.pauli_z_measurement(s_current_state, p_current_state, qubit)[0] for qubit in range(n)])'''
    
    if debug: print(f'Initial state: {[_symp.pauli_z_measurement(s_current_state, p_current_state, qubit)[0] for qubit in range(n)]}')

    layer_count = 0
    for layer in layers:
        '''
        Here we build mixed circuit layer without MCMs (so that we can do forward propagation of the stabilizer). We also determine which qubits were measured and which part of the large Pauli they should point to next.
        '''
        no_mcm_mixed_layer_circuit, new_qubit_pointers, new_next_available_qubit, measured_qubits, measurement = quick_process_rbmcm_layer(layer, layer_order, qubit_pointers = qubit_pointers, next_available_qubit = next_available_qubit, mcm_labels = mcm_labels, current_pauli = current_pauli)
        

        
        native_layer_circuits = {key: _cir.Circuit(layer[key], line_labels = qubit_labels) for key in layer}
        
        unmeasured_qubits = _np.setdiff1d([i for i in range(n)], measured_qubits)
        measurement_order = measurement_order + [int(qubit_pointers[f'Q{qubit}'][-1][1:]) for qubit in measured_qubits]
        # mcm locations are stored in meas_qubits
        # we use this to determine any measurement layer that needs to be added
        
        predicted_pauli = [i[0] for i in _symp.find_pauli_layer(p_pauli_circuit, [j for j in range(n)])]
        assert(all([current_pauli[i] == predicted_pauli[i] for i in range(n)]))
        measured_paulis = [current_pauli[i] for i in measured_qubits]
        
        # generate the measurement layer for the circuit
        # measured qubits tells us which qubits were measured
        
        native_measure_circuit = pauli_to_z(measured_paulis, measured_qubits, qubit_labels, rand_state)
        '''if debug: print(f'The native measure prep circut: \n {native_measure_circuit}')'''
        full_pre_measure_circuit = native_measure_circuit.copy(editable = True)
        full_pre_measure_circuit.append_circuit_inplace(native_layer_circuits['pre-oneq-layers'])
        full_pre_measure_circuit = full_pre_measure_circuit.parallelize()
        native_circuit.append_circuit_inplace(full_pre_measure_circuit)
        no_mcm_circuit.append_circuit_inplace(full_pre_measure_circuit)
        
        s_prelayer, p_prelayer = _symp.symplectic_rep_of_clifford_circuit(full_pre_measure_circuit, pspec = pspec)
        s_prelayer_inverse, p_prelayer_inverse = _symp.inverse_clifford(s_prelayer, p_prelayer)
        
        # Push the state past the pre-measurement layer
        s_current_state, p_current_state = _symp.apply_clifford_to_stabilizer_state(s_prelayer, p_prelayer, s_current_state, p_current_state)
        
        # Push the Pauli past the pre-measurement layer (at this point it should have all Zs (or identities) on the measured qubits)
        s_pauli_circuit, p_pauli_circuit = _symp.compose_cliffords(s_prelayer_inverse, p_prelayer_inverse, s_pauli_circuit, p_pauli_circuit) # PM^(-1)
        s_pauli_circuit, p_pauli_circuit = _symp.compose_cliffords(s_pauli_circuit, p_pauli_circuit, s_prelayer, p_prelayer) # MPM^(-1)
        current_pauli = [i[0] for i in _symp.find_pauli_layer(p_pauli_circuit, [f'Q{j}' for j in range(n)])]
        
        assert(all([current_pauli[i] in ['I', 'Z'] for i in measured_qubits]))
        
        if debug: 
            current_state = [_symp.pauli_z_measurement(s_current_state, p_current_state, qubit)[0] for qubit in range(n)]
            print(f'After the pre-mixed layer we have the following state: {current_state}')
            #print(f'It should be (anti-)stabilized by {current_pauli}.')
        
        # Update final Pauli with the correct Z Pauli (and identity) entries
        for qubit in measured_qubits:
            final_pauli[int(qubit_pointers[f'Q{qubit}'][-1][1:])] = current_pauli[qubit]
        
        # append the pre-measurement error gates (if necessary)
        if loqs is True:
            native_circuit.append_circuit_inplace(native_layer_circuits['pre-meas-layer'])
            
        # Now we handle the mixed layers.
        
        # Determine the sign of the measurement (i.e., does the signless measurement stabilize or antistabilize your state)?
        sign = birb.determine_sign(s_current_state, p_current_state, measurement)
        '''if debug: 
            print(f'We are peforming the following measurement: {measurement}')
            print(f'It has sign {sign}')
        
            print(f'Sign: {sign}')
            print(f'Measurement: {measurement}')
            print(f'S matrix: \n{s_current_state}')
            print(f'P matrix: {p_current_state}')
        
            s_check, p_check = _copy.deepcopy(s_current_state), _copy.deepcopy(p_current_state)
            for i in range(len(measurement)):
                if measurement[i] == 'Z':
                    s_z, p_z = _np.identity(2*n, _np.int64), _np.zeros(2*n, _np.int64)
                    p_z[i] = 2
                    s_check, p_check = _symp.apply_clifford_to_stabilizer_state(s_z, p_z, s_check, p_check)
            assert(_np.array_equal(s_check, s_current_state))
            assert(_np.array_equal(p_check, p_current_state))
            '''
        measurement_phases.append(sign)
        
        
        
        # Determine if a reset modification is necessary - You modified this and likely screwed it up (current_pauli --> rand_pauli)
        next_pauli_slice = {i: int(new_qubit_pointers[f'Q{i}'][-1][1:]) for i in measured_qubits}
        '''if debug: 
            print(f'These are the next Pauli indices: {next_pauli_slice}')
            print(f'This is the current Pauli: {current_pauli}')
            print(f'This is the original Pauli: {rand_pauli}')
            print(f'These are the next Pauli entries: {[rand_pauli[next_pauli_slice[i]] for i in next_pauli_slice]}')'''
        if mcm_reset == False:
            pauli_slice = [rand_pauli[next_pauli_slice[i]] for i in next_pauli_slice]
            reset_matters = reset_matters + [True if P != 'I' else False for P in pauli_slice] # [True if rand_pauli[int(qubit_pointers['Q{}'.format(i)][-1][1:])] != 'I' else False for i in og_meas_qubits]
            '''print(f'This is the reset logic: {reset_matters}')'''
        
        # add the mixed layer to the native circuit
        native_circuit.append_circuit_inplace(native_layer_circuits['mixed-layer'])
        
        # add the no-mcm mixed layer to the no-mcm circuit and compute its symplectic representation
        no_mcm_circuit.append_circuit_inplace(no_mcm_mixed_layer_circuit)
        s_no_mcm_mixed_layer, p_no_mcm_mixed_layer = _symp.symplectic_rep_of_clifford_circuit(no_mcm_mixed_layer_circuit, pspec = pspec)
        s_no_mcm_mixed_layer_inverse, p_no_mcm_mixed_layer_inverse = _symp.inverse_clifford(s_no_mcm_mixed_layer, p_no_mcm_mixed_layer)
        
        # push the Pauli through - so the Pauli will be correct on non-measured qubits and I or Z on measured qubits
        s_pauli_circuit, p_pauli_circuit = _symp.compose_cliffords(s_no_mcm_mixed_layer_inverse, p_no_mcm_mixed_layer_inverse, s_pauli_circuit, p_pauli_circuit)
        s_pauli_circuit, p_pauli_circuit = _symp.compose_cliffords(s_pauli_circuit, p_pauli_circuit, s_no_mcm_mixed_layer, p_no_mcm_mixed_layer)
        current_pauli = [i[0] for i in _symp.find_pauli_layer(p_pauli_circuit, [f'Q{j}' for j in range(n)])]
        assert(all([current_pauli[i] in ['I', 'Z'] for i in measured_qubits]))
        # for i in measured_qubit: # post-measurement the Pauli on the measured qubits should be Z (
            # current_pauli[i] = 'Z'
        
        # push the state through the mixed layer (w/o mcms)
        
        s_current_state, p_current_state = _symp.apply_clifford_to_stabilizer_state(s_no_mcm_mixed_layer, p_no_mcm_mixed_layer, s_current_state, p_current_state)
        
        if debug: 
            #print(f'This is the layer without MCMs: \n{no_mcm_mixed_layer_circuit}')
            current_state = [_symp.pauli_z_measurement(s_current_state, p_current_state, qubit)[0] for qubit in range(n)]
            print(f'After the mixed layer and prior to measuring we have the following state: {current_state}')
            print(f'It should be (anti-)stabilized by {current_pauli}.')
        
        # perform the "measurements" and update the state and the Pauli - This is new. You probably messed it up!!
        # Is this reseting the measured qubits into the |0> state?
        # print(f'This is the measurement: {measurement}')
        # print(f'This is the recorded sign: {sign}')
        for qubit in measured_qubits:
            #print(f'We are measuring qubit {qubit} in layer {layer_count}')
            measurement_outcome = _symp.pauli_z_measurement(s_current_state, p_current_state, qubit)
            #print(f'We measure a |0> with probability {measurement_outcome[0]}')      
            s_current_state, p_current_state = measurement_outcome[2], measurement_outcome[4] # This doesn't do what you want it to. It takes |01>+|10> to either |01> or |10>. You need it to take you to |10>+|00>.
            if measurement_outcome[0] == 0: # Then the current state is stabilized by -Z_qubit
                #print(f'We measured a 1 deterministically on qubit {qubit} on layer {layer_count}')
                s_x, p_x = _np.identity(2*n, _np.int64), _np.zeros(2*n, _np.int64)
                p_x[n+qubit] = 2
                s_current_state, p_current_state = _symp.apply_clifford_to_stabilizer_state(s_x, p_x, s_current_state, p_current_state)
            measurement_outcome = _symp.pauli_z_measurement(s_current_state, p_current_state, qubit)
            #print(f'Now we measure |0> with probability {measurement_outcome[0]}\n')
            assert(measurement_outcome[0] == 1)
            assert(measurement_outcome[1] == 0)
            if rand_pauli[next_pauli_slice[qubit]] == 'I':
                current_pauli[qubit] = 'I'
            else:
                current_pauli[qubit] = 'Z'
        if debug:        
            current_state = [_symp.pauli_z_measurement(s_current_state, p_current_state, qubit)[0] for qubit in range(n)]
            print(f'After the measurement we have the following state: {current_state}')
        #    print(f'It should be (anti-)stabilized by {current_pauli}.')
        #    print(f'It has an s of: {s_current_state}')
        #    print(f'It has a p of: {p_current_state}')
        # We may nhave a sign issue, i.e., we may have a post-measurement state that is anti-stabilized by the current pauli. Do we assume that it is stabilized?
        
        pauli_circuit = _cir.Circuit([[(current_pauli[i], qubit_labels[i]) for i in range(n)]], line_labels = qubit_labels, editable = True)
        s_pauli_circuit, p_pauli_circuit = _symp.symplectic_rep_of_clifford_circuit(pauli_circuit, pspec = pspec)
        
        # now we deal with the post-measurement layers
        
        # add in the post-measurement error layer to the native circuit if required
        if loqs is True:
            native_circuit.append_circuit_inplace(native_layer_circuits['post-meas-layer'])
            
        # create the necessary post-measurement state preparation layer and combined it with the random 1Q gates
        post_measurement_prep_circuit = _cir.Circuit([[(prep_circuit[int(new_qubit_pointers[f'Q{i}'][-1][1:])], qubit_labels[i]) for i in measured_qubits]], line_labels = qubit_labels, editable = True)
        
        full_post_measurement_circuit = post_measurement_prep_circuit.copy(editable = True)
        full_post_measurement_circuit.append_circuit_inplace(native_layer_circuits['post-oneq-layers'])
        full_post_measurement_circuit = full_post_measurement_circuit.parallelize()
        
        # add the (combined) post-measurement 1Q gate layer to the native circuit and the no-mcm circuit
        native_circuit.append_circuit_inplace(full_post_measurement_circuit)
        no_mcm_circuit.append_circuit_inplace(full_post_measurement_circuit)
        
        # get the symplectic representation of the post-measurement 1Q layer
        s_postlayer, p_postlayer = _symp.symplectic_rep_of_clifford_circuit(full_post_measurement_circuit, pspec = pspec)
        s_postlayer_inverse, p_postlayer_inverse = _symp.inverse_clifford(s_postlayer, p_postlayer)
        
        # push the Pauli through and update current pauli
        s_pauli_circuit, p_pauli_circuit = _symp.compose_cliffords(s_postlayer_inverse, p_postlayer_inverse, s_pauli_circuit, p_pauli_circuit)
        s_pauli_circuit, p_pauli_circuit = _symp.compose_cliffords(s_pauli_circuit, p_pauli_circuit, s_postlayer, p_postlayer)
        current_pauli = [i[0] for i in _symp.find_pauli_layer(p_pauli_circuit, [f'Q{j}' for j in range(n)])]
        assert(all([current_pauli[i] == rand_pauli[next_pauli_slice[i]] for i in measured_qubits])) # Make sure that we are correctly preparing the post-measurement state.
        
        # push the state through
        s_current_state, p_current_state = _symp.apply_clifford_to_stabilizer_state(s_postlayer, p_postlayer, s_current_state, p_current_state)
        
        if debug:
            current_state = [_symp.pauli_z_measurement(s_current_state, p_current_state, qubit)[0] for qubit in range(n)]
            print(f'After the post-measurement layer we have the following state: {current_state}')
        #    print(f'It should be (anti-)stabilized by {current_pauli}.\n')
        #    print(f'It has an s of: {s_current_state}')
        #    print(f'It has a p of: {p_current_state}')
        
        # update qubit_pointers
        qubit_pointers = new_qubit_pointers
        next_available_qubit = new_next_available_qubit
        
        layer_count += 1
        
        # This concludes the addition of the benchmarking layers
        
    # Now we need to handle the final measurements
    
    current_pauli = [i[0] for i in _symp.find_pauli_layer(p_pauli_circuit, [f'Q{j}' for j in range(n)])]
    
    if debug:
        current_state = [_symp.pauli_z_measurement(s_current_state, p_current_state, qubit)[0] for qubit in range(n)]
        print(f'Prior to the pre-final measurement layer, we have the following state: {current_state}')
        #print(f'It should be (anti-)stabilized by {current_pauli}.')
    
    # generate the final pre-measurement layer
    
    active_ancilla_qubits = [qubit_pointers[f'Q{i}'][-1] for i in range(n)] # probably not needed
    native_measure_circuit = pauli_to_z(current_pauli, _np.arange(n), qubit_labels, rand_state)
    s_prelayer, p_prelayer = _symp.symplectic_rep_of_clifford_circuit(native_measure_circuit, pspec = pspec)
    s_prelayer_inverse, p_prelayer_inverse = _symp.inverse_clifford(s_prelayer, p_prelayer)
    
    # append the final pre-measurement layer to the native and no mcm circuits
    native_circuit.append_circuit_inplace(native_measure_circuit)
    no_mcm_circuit.append_circuit_inplace(native_measure_circuit)
    
    # push the state through the final pre-measurement layer
    s_current_state, p_current_state = _symp.apply_clifford_to_stabilizer_state(s_prelayer, p_prelayer, s_current_state, p_current_state)
    
    # push the Pauli through the final pre-measurement layer
    s_pauli_circuit, p_pauli_circuit = _symp.compose_cliffords(s_prelayer_inverse, p_prelayer_inverse, s_pauli_circuit, p_pauli_circuit)
    s_pauli_circuit, p_pauli_circuit = _symp.compose_cliffords(s_pauli_circuit, p_pauli_circuit, s_prelayer, p_prelayer)
    current_pauli = [i[0] for i in _symp.find_pauli_layer(p_pauli_circuit, [f'Q{j}' for j in range(n)])]
    
    if debug:
        current_state = [_symp.pauli_z_measurement(s_current_state, p_current_state, qubit)[0] for qubit in range(n)]
        print(f'Prior to the final measurement we have the state: {current_state}')
        #print(f'It should be (anti-)stabilized by {current_pauli}.')
    
    # the current state should now be stabilized by some Z-type Pauli / the current Pauli should be a Z-type Pauli
    final_measurement = [i[0] for i in _symp.find_pauli_layer(p_pauli_circuit, [f'Q{j}' for j in range(n)])]
    assert(set(final_measurement) == set(['I','Z']) or set(final_measurement) == set(['Z']) or set(final_measurement) == set(['I']))
    
    for qubit in range(n):
        final_pauli[int(qubit_pointers[f'Q{qubit}'][-1][1:])] = final_measurement[qubit]
        
    measurement_order = measurement_order + [int(qubit_pointers[qubit][-1][1:]) for qubit in qubit_labels]
        
        
    # determine the sign of the final measurement
    sign = birb.determine_sign(s_current_state, p_current_state, final_measurement)
    if debug:
        print(f'The final measurement has the following sign: {sign}')
    measurement_phases.append(sign)
    
    # append the final measurement layer if you are using LoQs
    if loqs is True:
        if 'pre_final' in mcm_labels:
            final_measurement_circuit = _cir.Circuit([[Label(mcm_labels['pre_final'], qubit) for qubit in qubit_labels],
                                          [Label(mcm_labels['mcm'], qubit) for qubit in qubit_labels]], line_labels = qubit_labels)
        else:
            final_measurement_circuit = _cir.Circuit([[Label(mcm_labels['mcm'], qubit) for qubit in qubit_labels]], line_labels = qubit_labels)
        native_circuit.append_circuit_inplace(final_measurement_circuit)
        
    # compute the overall sign of the total measurement
    final_sign = _np.prod(measurement_phases)

    # return native_circuit, no_mcm_circuit, final_pauli, final_sign
    return native_circuit, final_pauli, measurement_order, final_sign, qubit_pointers, reset_matters

### Deprecated functions are kept down here ###

def old_sample_loqs_rb_mcm_circuit_layer_by_edgegrab(pspec, qubit_labels=None, mcm_labels = None, two_q_gate_density=0.25, mcm_density = .25, one_q_gate_names=None, loqs = False, mcm_only_layers = False, gate_args_lists=None, rand_state=None):
    assert(mcm_labels is not None), 'Need MCM labels.'
    if gate_args_lists is None: gate_args_lists = {}
        
    if qubit_labels is None:
        qubits = list(pspec.qubit_labels[:])  # copy this list
    else:
        assert(isinstance(qubit_labels, (list, tuple))), "SubsetQs must be a list or a tuple!"
        qubits = list(qubit_labels[:])  # copy this list
    if rand_state is None:
        print('triggered')
        rand_state = _np.random.RandomState()
        
    assert(two_q_gate_density + mcm_density <= 1), 'You cannot have more than 100\% MCMs and 2Q gates.'

    # Prep the sampling variables.
    sampled_layer, pre_layer, post_layer = [], [], []
    num_mcms = 0
    mcm_locs = []
    edgelist = pspec.compute_2Q_connectivity().edges()
    edgelist = [e for e in edgelist if all([q in qubits for q in e])]
    selectededges = []

    # Go through until all qubits have been assigned a gate.
    while len(edgelist) > 0:

        edge = edgelist[rand_state.randint(0, len(edgelist))]
        selectededges.append(edge)
        # Delete all edges containing these qubits.
        edgelist = [e for e in edgelist if not any([q in e for q in edge])]

    num2Qgates = len(selectededges)
    if len(qubits) > 1:
        mean_two_q_gates = len(qubits) * two_q_gate_density / 2
    else:
        mean_two_q_gates = 0
    assert(num2Qgates >= mean_two_q_gates), "Device has insufficient connectivity!"

    if mean_two_q_gates > 0:
        twoQprob = mean_two_q_gates / num2Qgates
    else:
        twoQprob = 0

    unusedqubits = _copy.copy(qubits)
    ops_on_qubits = pspec.compute_ops_on_qubits()
    for edge in selectededges:
        if bool(rand_state.binomial(1, twoQprob)):
            # The two-qubit gates on that edge.
            possibleops = ops_on_qubits[edge]
            argless_gate_label = possibleops[rand_state.randint(0, len(possibleops))]
            if argless_gate_label.name not in gate_args_lists.keys():
                sampled_layer.append(argless_gate_label)
            else:
                possibleargs = gate_args_lists[argless_gate_label.name]
                args = possibleargs[rand_state.randint(0, len(possibleargs))]
                sampled_layer.append(_lbl.Label(argless_gate_label.name, edge, args=args))

            for q in edge:
                del unusedqubits[unusedqubits.index(q)]
                
    total_remaining_qubits = len(unusedqubits)
    if total_remaining_qubits > 0:
        mcmprob = mcm_density / (1 - twoQprob)
    else:
        mcmprob = 0
    unusedqubits_copy = _copy.copy(unusedqubits)
    for q in unusedqubits:
        if bool(rand_state.binomial(1, mcmprob)):
            num_mcms += 1
            mcm_locs.append(q)
            gate_label = Label(mcm_labels['mcm'], q)
            pregate_label = Label(mcm_labels['pre'], q)
            postgate_label = Label(mcm_labels['post'], q)
            
            sampled_layer.append(gate_label)
            pre_layer.append(pregate_label)
            post_layer.append(postgate_label)
            
            del unusedqubits_copy[unusedqubits_copy.index(q)]
        if (num_mcms == len(qubits) - 1) and (mcm_only_layers is False):
            break
    
    unusedqubits = unusedqubits_copy
                   
    if one_q_gate_names is None or len(one_q_gate_names) > 0:
        for q in unusedqubits:
            if one_q_gate_names is None:
                possibleops = ops_on_qubits[(q,)]
            else:
                print(one_q_gate_names)
                print(ops_on_qubits[(q,)])
                possibleops = [gate_lbl for gate_lbl in ops_on_qubits[(q,)] if gate_lbl.name in one_q_gate_names]
            gate_label = possibleops[rand_state.randint(0, len(possibleops))]
            sampled_layer.append(gate_label)

    if loqs == True:
        return {'pre-layers': [pre_layer], 'post-layers': [post_layer], 'mixed-layer': [sampled_layer]}, num_mcms, mcm_locs
    
    return {'pre-layers': [], 'mixed-layer': [sampled_layer], 'post-layers': []}, num_mcms, mcm_locs

def old_sample_loqs_rb_mcm_circuit_layer_with_1Q_gates_by_edgegrab(pspec, qubit_labels=None, mcm_labels = None, two_q_gate_density=0.25, mcm_density = .25, one_q_gate_names=None, loqs = True, mcm_only_layers = True, gate_args_lists=None, rand_state=None):
    
    assert(mcm_labels is not None), 'Need MCM labels.'
    if gate_args_lists is None: gate_args_lists = {}
        
    if qubit_labels is None:
        qubits = list(pspec.qubit_labels[:])  # copy this list
    else:
        assert(isinstance(qubit_labels, (list, tuple))), "SubsetQs must be a list or a tuple!"
        qubits = list(qubit_labels[:])  # copy this list
    if rand_state is None:
        rand_state = _np.random.RandomState()
        
    assert(two_q_gate_density + mcm_density <= 1), 'You cannot have more than 100\% MCMs and 2Q gates.'
    
    
    num_qubits = len(qubits)
    if num_qubits == 1:
        mcm_only_layers = True
    mixed_layers, mcms, mcm_locs = sample_loqs_rb_mcm_circuit_layer_by_edgegrab(pspec, qubit_labels=qubit_labels, 
                                                                      mcm_labels = mcm_labels, two_q_gate_density=two_q_gate_density, mcm_density = mcm_density, 
                                                                      one_q_gate_names=one_q_gate_names, loqs = loqs,
                                                                                mcm_only_layers = mcm_only_layers,        
                                                                      gate_args_lists=gate_args_lists, rand_state=rand_state)
    
    unused_qubits = _np.setdiff1d(qubits, mcm_locs)
    gates = {q: 'Gc{}'.format(rand_state.choice(_np.arange(0,24))) for q in unused_qubits}
    pre_oneq_layer = [Label(gates[q], q) for q in unused_qubits]
    
    gates = {q: 'Gc{}'.format(rand_state.choice(_np.arange(0,24))) for q in unused_qubits}
    post_oneq_layer = [Label(gates[q], q) for q in unused_qubits]
    
    layers = {'mixed-layer': mixed_layers['mixed-layer']}
    layers['pre-oneq-layers'] = [pre_oneq_layer]
    layers['pre-meas-layer'] = mixed_layers['pre-layers']
    layers['post-meas-layer'] = mixed_layers['post-layers']
    layers['post-oneq-layers'] = [post_oneq_layer]
    
    return layers, mcms, mcm_locs

def old_process_loqs_layer(loqs_layers: dict, qubit_pointers: dict, next_available_qubit: int, mcm_labels: dict):
    """
    Processes the output of any rbmcm layer sampler so that it is a circuit layer
    in the blown up circuit required for RBMCM analysis.
    
    We assume that the mixed layer only contains one layer!!!
    
    Returns: The circuit layer
    """
    new_layers = {key: [] for key in loqs_layers}
    new_next_available_qubit = next_available_qubit
    new_qubit_pointers = {qubit: _copy.deepcopy(qubit_pointers[qubit]) for qubit in qubit_pointers}
    meas_qubits, meas_qubit_og_circuit = [], []
    mcm_layers, no_mcm_layers = [], []
        
    for key, layers in loqs_layers.items():
        for i in range(len(layers)):
            layer = layers[i]
            new_layer = []
            for gate in layer:
                new_gate = update_gate(gate, qubit_pointers)
                new_layer.append(new_gate)
                if key == 'mixed-layer':
                    gate_type = gate[0]
                    if gate_type == mcm_labels['mcm']:
                        meas_qubit = gate[1]
                        meas_qubit_og_circuit.append(int(meas_qubit[1:]))
                        meas_qubits.append(int(qubit_pointers[meas_qubit][-1][1:]))
                        new_qubit_pointers[meas_qubit].append('Q{}'.format(new_next_available_qubit))
                        new_next_available_qubit += 1
            new_layers[key].append(new_layer)
    
    layer_order = ['pre-oneq-layers', 'pre-meas-layer', 'mixed-layer', 'post-meas-layer', 'post-oneq-layers']
    for key in layer_order:
        layer = new_layers[key][0]
        mcm_layers.append(layer)
        if layer != []:
            no_mcm_layer = [i for i in layer if (i[0] != mcm_labels['mcm'] and i[0] != mcm_labels['pre'] and i[0] != mcm_labels['post'])]
            no_mcm_layers.append(no_mcm_layer)
        
    mcm_circuit = _cir.Circuit(mcm_layers, line_labels = list(qubit_pointers.keys()))
    no_mcm_circuit = _cir.Circuit(no_mcm_layers, line_labels = list(qubit_pointers.keys()))
    
    return mcm_circuit, no_mcm_circuit, new_qubit_pointers, new_next_available_qubit, meas_qubits, meas_qubit_og_circuit

