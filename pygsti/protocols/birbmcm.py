import pygsti
from pygsti.circuits import circuit as _cir
from pygsti.baseobjs import Label
from pygsti.tools import symplectic as _symp
from pygsti.processors import CliffordCompilationRules as CCR

import copy as _copy

import birb

import numpy as _np

import matplotlib.pyplot as plt



def sample_loqs_rb_mcm_circuit_layer_by_edgegrab(pspec, qubit_labels=None, mcm_labels = None, two_q_gate_density=0.25, mcm_density = .25, one_q_gate_names=None, loqs = False, mcm_only_layers = False, gate_args_lists=None, rand_state=None):
    """
    TODO: docstring
    <TODO summary>

    Parameters
    ----------
    pspec : <TODO typ>
        <TODO description>

    qubit_labels : <TODO typ>, optional
        <TODO description>

    mean_two_q_gates : <TODO typ>, optional
        <TODO description>

    modelname : <TODO typ>, optional
        <TODO description>

    rand_state: RandomState, optional
        A np.random.RandomState object for seeding RNG
        
    mcm_labels: dict
        Dictionary containing the gate labels for your mid-circuit measurement and any pre- and post-measurement gates.
        {'mcm': label, 'pre': label, 'post': label}
        The pre-measure and post-measure labels should not be in pspec. The mcm label should conform to LoQs' requirements. 

    Returns
    -------
    <TODO typ>
    """
    
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

def sample_loqs_rb_mcm_circuit_layer_with_1Q_gates_by_edgegrab(pspec, qubit_labels=None, mcm_labels = None, two_q_gate_density=0.25, mcm_density = .25, one_q_gate_names=None, loqs = True, mcm_only_layers = True, gate_args_lists=None, rand_state=None):
    
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
    gates = {q: 'Gc{}'.format(_np.random.choice(_np.arange(0,24))) for q in unused_qubits}
    pre_oneq_layer = [Label(gates[q], q) for q in unused_qubits]
    
    gates = {q: 'Gc{}'.format(_np.random.choice(_np.arange(0,24))) for q in unused_qubits}
    post_oneq_layer = [Label(gates[q], q) for q in unused_qubits]
    
    layers = {'mixed-layer': mixed_layers['mixed-layer']}
    layers['pre-oneq-layers'] = [pre_oneq_layer]
    layers['pre-meas-layer'] = mixed_layers['pre-layers']
    layers['post-meas-layer'] = mixed_layers['post-layers']
    layers['post-oneq-layers'] = [post_oneq_layer]
    
    return layers, mcms, mcm_locs
    
def update_gate(gate, qubit_pointers: dict):
    gate_type, gate_qubits = gate[0], gate[1:]
    return Label(gate_type, tuple([qubit_pointers[qubit][-1] for qubit in gate_qubits]))


def quick_process_loqs_layer(loqs_layers: dict, qubit_pointers: dict, next_available_qubit: int, mcm_labels: dict):
    """
    Processes the output of any rbmcm layer sampler. 
    
    We assume that the mixed layer only contains one layer!!!
    
    Returns:
        - mcm_circuit: pygsti circuit of the benchmark layer,
        - no_mcm_circuit: pygsti circuit of the benchmark layer without the MCMs,
        - new_qubit_pointers: dictionary that matches each qubit with the Pauli entry it is matched up with
        - new_next_available_qubit: integer specifying the next slot in the Pauli that is available
        - meas_qubits: list of the qubits that are measured in this layer (in the expanded circuit)
        - meas_qubit_og_circuit: list of the qubits that are measured in this layer (in the non-expanded circuit)
    """
    new_next_available_qubit = next_available_qubit
    new_qubit_pointers = {qubit: _copy.deepcopy(qubit_pointers[qubit]) for qubit in qubit_pointers}
    measured_qubits, measurement = [], ['I' for i in range(len(qubit_pointers.keys()))]
    no_mcm_mixed_layer = []
    
    mixed_layer = loqs_layers['mixed-layer'][0]
    for gate in mixed_layer:
        gate_type = gate[0]
        if gate_type == mcm_labels['mcm']:
            meas_qubit = int(gate[1][1:])
            measured_qubits.append(meas_qubit)
            measurement[meas_qubit] = 'Z'
            new_qubit_pointers[meas_qubit].append(new_next_available_qubit)
            new_next_available_qubit += 1
        else:
            no_mcm_mixed_layer.append(gate)

    return [no_mcm_mixed_layer], new_qubit_pointers, new_next_available_qubit, measured_qubits, measurement
            

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

def pauli_to_z(paulis, qubits, qubit_labels):
    '''
    Turns a collection of single qubit paulis into Z paulis acting on the specified qubits
    within a larger collection of qubits.
    '''
    circuit = []
    for pauli, qubit in zip(paulis, qubits):
        if pauli == 'Y':
            circuit.append(Label('Gc1','Q{}'.format(qubit)))
        elif pauli == 'X':
            circuit.append(('Gc12','Q{}'.format(qubit)))
        elif pauli == 'I':
            rand_clifford = str(_np.random.choice(_np.arange(24)))
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

def create_rb_with_mcm_circuit(pspec, length, qubit_labels, mcm_labels, two_q_gate_density, mcm_density, loqs, debug = True, layer_sampler = sample_loqs_rb_mcm_circuit_layer_with_1Q_gates_by_edgegrab, include_identity = True, mcm_only_layers = True):
    if qubit_labels is not None: n = len(qubit_labels)
    else: 
        n = pspec.num_qubits
        qubit_labels = pspec.qubit_labels
        
    layer_order = ['pre-oneq-layers', 'pre-meas-layer', 'mixed-layer', 'post-meas-layer', 'post-oneq-layers']
    
    # Sample the "core" circuit layers. This must be done first to determine the size of the Pauli to select.
    
    layers = []
    check_circuit = _cir.Circuit([],line_labels = qubit_labels, editable = True)
    mcm_count = 0
    for i in range(length):
        layer, mcms, mcm_locs = layer_sampler(pspec, qubit_labels=None, mcm_labels = mcm_labels,
                                              two_q_gate_density=two_q_gate_density, mcm_density = mcm_density, loqs = loqs,
                                              mcm_only_layers = mcm_only_layers)
        
        if loqs == True: 
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
    
    # Sample the Pauli
    
    rand_pauli, rand_sign, pauli_circuit = birb.sample_random_pauli(n = n + mcm_count, pspec = big_pspec, absolute_compilation = big_compilations['absolute'], circuit = True, include_identity = include_identity)
    s_pauli_circuit, p_pauli_circuit = _symp.symplectic_rep_of_clifford_circuit(pauli_circuit, pspec = big_pspec)
    current_pauli = rand_pauli

    # Sample the stablizer
    
    s_inputstate, p_inputstate, s_init_layer, p_init_layer, prep_circuit = birb.sample_stabilizer(rand_pauli, rand_sign)
    prep_circuit = birb.compose_initial_cliffords(prep_circuit)
    
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
    for layer in layers:
        new_layer, new_no_mcm_layer, new_qubit_pointers, new_next_available_qubit, meas_qubits, og_meas_qubits = process_loqs_layer(layer, 
                                                                             qubit_pointers = qubit_pointers, 
                                                                             next_available_qubit = next_available_qubit, 
                                                                             mcm_labels = mcm_labels)
        
        native_layer_circuits = {key: _cir.Circuit(layer[key], line_labels = qubit_labels) for key in layer}
        
        unmeasured_qubits = _np.setdiff1d(unmeasured_qubits, meas_qubits)
        og_unmeasured_qubits = _np.setdiff1d(og_unmeasured_qubits, og_meas_qubits)
        
        # mcm locations are stored in meas_qubits
        # we use this to determine any measurement layer that needs to be added
        
        current_pauli = [i[0] for i in _symp.find_pauli_layer(p_pauli_circuit, [j for j in range(n+mcm_count)])]
        active_paulis = [current_pauli[i] for i in meas_qubits]
        # print('This is the Pauli {} after layer {}'.format(current_pauli, layer_count))
        
        # generate the measurement layer for the non-blown up circuit
        # og_meas_qubits tells me which qubits were measured
        # we can then use qubit_points to tell us which Pauli entry that qubit is currently pointing to
        # We then need to go from that entry in the current Pauli to a Z Pauli.
        
        native_measure_circuit = pauli_to_z(active_paulis, og_meas_qubits, qubit_labels)
        full_pre_measure_circuit = native_measure_circuit.copy(editable = True)
        full_pre_measure_circuit.append_circuit_inplace(native_layer_circuits['pre-oneq-layers'])
        full_pre_measure_circuit = full_pre_measure_circuit.parallelize()
        native_circuit.append_circuit_inplace(full_pre_measure_circuit)
        
        # generate the measurement layer for the blown up circuit
        
        measure_circuit = process_measure_layer(native_measure_circuit, qubit_pointers, new_qubit_labels)
        s_measure_circuit, p_measure_circuit = _symp.symplectic_rep_of_clifford_circuit(measure_circuit, pspec = big_pspec)
        s_measure_circuit_inv, p_measure_circuit_inv = _symp.inverse_clifford(s_measure_circuit, p_measure_circuit)
        
        expanded_circuit.append_circuit_inplace(measure_circuit)
        no_mcm_circuit.append_circuit_inplace(measure_circuit)
        
        # append the new native layer to the native circuit
        
        native_circuit.append_circuit_inplace(native_layer_circuits['pre-meas-layer'])
        native_circuit.append_circuit_inplace(native_layer_circuits['mixed-layer'])
        native_circuit.append_circuit_inplace(native_layer_circuits['post-meas-layer'])
                
        # append the new layer to the blown up circuit
        
        expanded_circuit.append_circuit_inplace(new_layer) 
        no_mcm_circuit.append_circuit_inplace(new_no_mcm_layer)
        
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
        prep_layer = _cir.Circuit([[(prep_circuit[next_pauli_slice[i]], 'Q{}'.format(i)) for i in og_meas_qubits]], line_labels = qubit_labels, editable = True)
        prep_layer.append_circuit_inplace(native_layer_circuits['post-oneq-layers'])
        prep_layer = prep_layer.parallelize()
        native_circuit.append_circuit_inplace(prep_layer)
        
    # Need to add a measurement layer to handle any qubits that have not been hit by a MCM
    
    current_pauli = [i[0] for i in _symp.find_pauli_layer(p_pauli_circuit, [j for j in range(n+mcm_count)])]
    
    # generate the final measurement layer for the non-blown up circuit
    # for each qubit...find out what they point to...pick up that pauli entry and convert
    
    active_qubits = [int(qubit_pointers[i][-1][1:]) for i in qubit_labels]
    active_paulis = [current_pauli[active_qubit] for active_qubit in active_qubits]
    native_measure_circuit = pauli_to_z(active_paulis, _np.arange(n), qubit_labels)
    native_circuit.append_circuit_inplace(native_measure_circuit)
    # Needed because we are using LoQs!!!
    #final_measurements = _cir.Circuit([[Label(mcm_labels['pre'], qubit) for qubit in qubit_labels],
    #                                  [Label(mcm_labels['mcm'], qubit) for qubit in qubit_labels]], line_labels = qubit_labels)
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
    
    expanded_final_measurements = _cir.Circuit([[Label(mcm_labels['pre'], 'Q{}'.format(active_qubit)) for active_qubit in active_qubits],
                                                [Label(mcm_labels['mcm'], 'Q{}'.format(active_qubit)) for active_qubit in active_qubits]], line_labels = new_qubit_labels)
    expanded_circuit.append_circuit_inplace(expanded_final_measurements)
    no_mcm_circuit.append_circuit_inplace(expanded_final_measurements)
    
    
    expanded_circuit.done_editing()
    no_mcm_circuit.done_editing()
    native_circuit.done_editing()
    check_circuit.done_editing()

    # need to handle the final measurement sign...the Pauli should be Z-type here because we used the measure layer earlier
    
    measurement = ['I' if i == 'I' else 'Z' for i in current_pauli]
    sign = birb.determine_sign(s_final_state, p_final_state, measurement)
    
    if debug:
        return expanded_circuit, native_circuit, measurement, sign, no_mcm_circuit, mcm_count, check_circuit, s_final_state, p_final_state
    
    return native_circuit, measurement, sign, qubit_pointers


def quick_create_rb_with_mcm_circuit(pspec, length, qubit_labels, mcm_labels, two_q_gate_density, mcm_density, loqs, debug = True, layer_sampler = sample_loqs_rb_mcm_circuit_layer_with_1Q_gates_by_edgegrab, include_identity = True, mcm_only_layers = True):
    if qubit_labels is not None: n = len(qubit_labels)
    else: 
        n = pspec.num_qubits
        qubit_labels = pspec.qubit_labels
        
    layer_order = ['pre-oneq-layers', 'pre-meas-layer', 'mixed-layer', 'post-meas-layer', 'post-oneq-layers']
    
    # Sample the "core" circuit layers. This must be done first to determine the size of the Pauli to select.
    
    layers = []
    check_circuit = _cir.Circuit([],line_labels = qubit_labels, editable = True)
    mcm_count = 0
    for i in range(length):
        layer, mcms, mcm_locs = layer_sampler(pspec, qubit_labels=None, mcm_labels = mcm_labels,
                                              two_q_gate_density=two_q_gate_density, mcm_density = mcm_density, loqs = loqs, mcm_only_layers = mcm_only_layers)
        
        if loqs == True: 
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
                                                      qubit_labels = new_qubit_labels)
    
    big_compilations = {'absolute': CCR.create_standard(big_pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity = 0),
                'paulieq': CCR.create_standard(big_pspec, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity = 0)}
    
    next_available_qubit = n
    qubit_pointers = {qubit: [qubit] for qubit in range(n)}
    # unmeasured_qubits = [i for i in range(n + mcm_count)]
    never_measured_qubits = [i for i in range(n)]
    measurement_phases = []
    
    # Sample the Pauli
    
    rand_pauli, rand_sign, pauli_circuit = birb.sample_random_pauli(n = n + mcm_count, pspec = big_pspec, absolute_compilation = big_compilations['absolute'], circuit = True, include_identity = include_identity)
    final_pauli = ['I' for i in range(n+mcm_count)]
    
    # Sample a random n+mcm-sized stablizer state
    
    s_state, p_state, s_state_prep_layer, p_state_prep_layer, prep_circuit = birb.sample_stabilizer(rand_pauli, rand_sign)
    prep_circuit = birb.compose_initial_cliffords(prep_circuit)
    
    # Extract the Pauli on the first n qubits
    
    current_pauli = rand_pauli[:n]
    pauli_circuit = _cir.Circuit([[(current_pauli[i], qubit_labels[i]) for i in range(n)]], line_labels = qubit_labels, editable = True)
    s_pauli_circuit, p_pauli_circuit = _symp.symplectic_rep_of_clifford_circuit(pauli_circuit, pspec = pspec)
    
    # Begin to create the birb+mcm circuit
    
    native_circuit = _cir.Circuit([[(prep_circuit[i], qubit_labels[i]) for i in range(len(qubit_labels))]], line_labels = qubit_labels, editable = True)
    no_mcm_circuit = native_circuit.copy(editable = True)

    s_native_circuit, p_native_circuit = _symp.symplectic_rep_of_clifford_circuit(native_circuit, pspec = pspec)
    s_no_mcm_circuit, p_no_mcm_circuit = _symp.symplectic_rep_of_clifford_circuit(no_mcm_circuit, pspec = pspec)
    
    # Extract the stabilizer state on the first n qubits
    
    s_current_state, p_current_state = _symp.prep_stabilizer_state(n)
    s_current_state, p_current_state = _symp.apply_clifford_to_stabilizer_state(s_native_circuit, p_native_circuit, s_current_state, p_current_state)
    
    # At this point you have:
    #   - An n+mcm_count random Pauli
    #   - A random sign on the random Pauli
    #   - A (n+mcm)-qubit stabilizer state (stabilized by the random Pauli when accounting for the sign)
    #   - An n-qubit state that matches the (n+mcm)-qubit stabilizer state on the first n-qubits
    #   - A circuit that prepares the n-qubit state
    #   - A copy of the circuit that prepares the n-qubit state
    #   - An n-qubit Pauli
    #   - Symplectic representations of all of the above
    
    # Now we begin to add the benchmark layers
    
    layer_count = 0
    for layer in layers:
        no_mcm_mixed_layer, new_qubit_pointers, new_next_available_qubit, measured_qubits, measurement = quick_process_loqs_layer(layer, qubit_pointers = qubit_pointers, next_available_qubit = next_available_qubit, mcm_labels = mcm_labels)
        
        native_circuit_layers = {key: _cir.Circuit(layer[key], line_labels = qubit_labels) for key in layer}
        no_mcm_mixed_circuit = _cir.Circuit(no_mcm_mixed_layer, line_labels = qubit_labels)
        
        unmeasured_qubits = _np.setdiff1d([i for i in range(n)], measured_qubits)
        
        current_pauli = [i[0] for i in _symp.find_pauli_layer(p_pauli_circuit, [j for j in range(n)])]
        active_paulis = [current_pauli[i] for i in measured_qubits]
        
        # generate the measurement layer for the circuit
        # measured_qubits tells us which qubits were measured
        
        native_measure_circuit = pauli_to_z(active_paulis, measured_qubits, qubit_labels)
        full_pre_measure_circuit = native_measure_circuit.copy(editable = True)
        full_pre_measure_circuit.append_circuit_inplace(native_circuit_layers['pre-oneq-layers'])
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
        
        # Update final Pauli with the correct Z Pauli (and identity) entries
        for qubit in measured_qubits:
            final_pauli[qubit_pointers[qubit][-1]] = current_pauli[qubit]
        
        # Add in the pre-measurement error gates to the native circuit
        native_circuit.append_circuit_inplace(native_circuit_layers['pre-meas-layer'])
        
        # Now we handle the mixed layers
        
        # determine the sign of the measurement (i.e., does the measurement stabilize or antistabilize your state)
        sign = birb.determine_sign(s_current_state, p_current_state, measurement)
        measurement_phases.append(sign)
        
        # perform the "measurements" and update the state and the Pauli
        for qubit in measured_qubits:
            measurement_outcome = _symp.pauli_z_measurement(s_current_state, p_current_state, qubit)
            s_current_state, p_current_state = measurement_outcome[2], measurement_outcome[4]
            current_pauli[qubit] = rand_pauli[new_qubit_pointers[qubit][-1]]
        pauli_circuit = _cir.Circuit([[(current_pauli[i], qubit_labels[i]) for i in range(n)]], line_labels = qubit_labels, editable = True)
        s_pauli_circuit, p_pauli_circuit = _symp.symplectic_rep_of_clifford_circuit(pauli_circuit, pspec = pspec)
        
        # add the mixed layer to the native circuit
        native_circuit.append_circuit_inplace(native_circuit_layers['mixed-layer'])
        
        # add the no-mcm mixed layer to the no-mcm circuit
        no_mcm_mixed_layer_circuit = _cir.Circuit(no_mcm_mixed_layer, line_labels = qubit_labels)
        no_mcm_circuit.append_circuit_inplace(no_mcm_mixed_layer_circuit)
        s_no_mcm_mixed_layer, p_no_mcm_mixed_layer = _symp.symplectic_rep_of_clifford_circuit(no_mcm_mixed_layer_circuit, pspec = pspec)
        s_no_mcm_mixed_layer_inverse, p_no_mcm_mixed_layer_inverse = _symp.inverse_clifford(s_no_mcm_mixed_layer, p_no_mcm_mixed_layer)
        
        # push the Pauli through
        s_pauli_circuit, p_pauli_circuit = _symp.compose_cliffords(s_no_mcm_mixed_layer_inverse, p_no_mcm_mixed_layer_inverse, s_pauli_circuit, p_pauli_circuit)
        s_pauli_circuit, p_pauli_circuit = _symp.compose_cliffords(s_pauli_circuit, p_pauli_circuit, s_no_mcm_mixed_layer, p_no_mcm_mixed_layer)
        
        # push the state through the mixed layer (w/o mcms)
        s_current_state, p_current_state = _symp.apply_clifford_to_stabilizer_state(s_no_mcm_mixed_layer, p_no_mcm_mixed_layer, s_current_state, p_current_state)
        
        # now we deal with the post-measurement layers
        
        # add in the post-measurement error layer to the native circuit
        native_circuit.append_circuit_inplace(native_circuit_layers['post-meas-layer'])
        
        # create the necessary post-measurement state preparation layer
        post_measurement_prep_circuit = _cir.Circuit([[(prep_circuit[new_qubit_pointers[i][-1]], qubit_labels[i]) for i in measured_qubits]], line_labels = qubit_labels, editable = True)
        
        full_post_measurement_circuit = post_measurement_prep_circuit.copy(editable = True)
        full_post_measurement_circuit.append_circuit_inplace(native_circuit_layers['post-oneq-layers'])
        full_post_measurement_circuit = full_post_measurement_circuit.parallelize()
        
        # combine the post-measurement state preparation layer with the post-measurement 1Q gate layer
        
        # add the (combined) post-measurement 1Q gate layer to the native circuit and the no-mcm circuit
        native_circuit.append_circuit_inplace(full_post_measurement_circuit)
        no_mcm_circuit.append_circuit_inplace(full_post_measurement_circuit)
        
        # get the symplectic representation of the post-measurement 1Q layer
        s_postlayer, p_postlayer = _symp.symplectic_rep_of_clifford_circuit(full_post_measurement_circuit, pspec = pspec)
        s_postlayer_inverse, p_postlayer_inverse = _symp.inverse_clifford(s_postlayer, p_postlayer)
        
        # push the Pauli through
        s_pauli_circuit, p_pauli_circuit = _symp.compose_cliffords(s_postlayer_inverse, p_postlayer_inverse, s_pauli_circuit, p_pauli_circuit)
        s_pauli_circuit, p_pauli_circuit = _symp.compose_cliffords(s_pauli_circuit, p_pauli_circuit, s_postlayer, p_postlayer)
        
        # push the state through
        s_current_state, p_current_state = _symp.apply_clifford_to_stabilizer_state(s_postlayer, p_postlayer, s_current_state, p_current_state)
        
        # update qubit_pointers
        qubit_pointers = new_qubit_pointers
        next_available_qubit = new_next_available_qubit
        
        # This concludes the addition of the benchmarking layers
        
    # Now we need to handle the final measurements
    
    current_pauli = [i[0] for i in _symp.find_pauli_layer(p_pauli_circuit, [f'Q{j}' for j in range(n)])]
    
    # generate the final pre-measurement layer
    
    active_ancilla_qubits = [qubit_pointers[i][-1] for i in range(n)]
    native_measure_circuit = pauli_to_z(current_pauli, _np.arange(n), qubit_labels)
    s_prelayer, p_prelayer = _symp.symplectic_rep_of_clifford_circuit(native_measure_circuit, pspec = pspec)
    s_prelayer_inverse, p_prelayer_inverse = _symp.inverse_clifford(s_prelayer, p_prelayer)
    
    # append the final pre-measurement layer to the native circuit
    native_circuit.append_circuit_inplace(native_measure_circuit)
    no_mcm_circuit.append_circuit_inplace(native_measure_circuit)
    
    # push the state through the final pre-measurement layer
    s_current_state, p_current_state = _symp.apply_clifford_to_stabilizer_state(s_prelayer, p_prelayer, s_current_state, p_current_state)
    
    # push the Pauli through the final pre-measurement layer
    s_pauli_circuit, p_pauli_circuit = _symp.compose_cliffords(s_prelayer_inverse, p_prelayer_inverse, s_pauli_circuit, p_pauli_circuit)
    s_pauli_circuit, p_pauli_circuit = _symp.compose_cliffords(s_pauli_circuit, p_pauli_circuit, s_prelayer, p_prelayer)
    
    # the current state should now be stabilized by some Z-type Pauli / the current Pauli should be a Z-type Pauli
    final_measurement = [i[0] for i in _symp.find_pauli_layer(p_pauli_circuit, [f'Q{j}' for j in range(n)])]
    assert(set(final_measurement) == set(['I','Z']) or set(final_measurement) == set(['Z']) or set(final_measurement) == set(['I']))
    
    for qubit in range(n):
        final_pauli[qubit_pointers[qubit][-1]] = final_measurement[qubit]
    
    # determine the sign of the final measurement
    sign = birb.determine_sign(s_current_state, p_current_state, final_measurement)
    measurement_phases.append(sign)
    
    # append the final measurement layer (needed because we are using LoQs)
    final_measurement_circuit = _cir.Circuit([[Label(mcm_labels['mcm'], qubit) for qubit in qubit_labels]], line_labels = qubit_labels)
    
    native_circuit.append_circuit_inplace(final_measurement_circuit)
    
    # measurement is supposed to contain the final ''big'' Z-type Pauli. You need to build this as you go!!!
    
    final_sign = _np.prod(measurement_phases)
    
    # return native_circuit, no_mcm_circuit, final_pauli, final_sign
    return native_circuit, final_pauli, final_sign, qubit_pointers

def nothing():
        
    # Need to add a measurement layer to handle any qubits that have not been hit by a MCM
    
    current_pauli = [i[0] for i in _symp.find_pauli_layer(p_pauli_circuit, [j for j in range(n+mcm_count)])]
    
    # generate the final measurement layer for the non-blown up circuit
    # for each qubit...find out what they point to...pick up that pauli entry and convert
    
    active_qubits = [int(qubit_pointers[i][-1][1:]) for i in qubit_labels]
    active_paulis = [current_pauli[active_qubit] for active_qubit in active_qubits]
    native_measure_circuit = pauli_to_z(active_paulis, _np.arange(n), qubit_labels)
    native_circuit.append_circuit_inplace(native_measure_circuit)
    # Needed because we are using LoQs!!!
    #final_measurements = _cir.Circuit([[Label(mcm_labels['pre'], qubit) for qubit in qubit_labels],
    #                                  [Label(mcm_labels['mcm'], qubit) for qubit in qubit_labels]], line_labels = qubit_labels)
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
    
    expanded_final_measurements = _cir.Circuit([[Label(mcm_labels['pre'], 'Q{}'.format(active_qubit)) for active_qubit in active_qubits],
                                                [Label(mcm_labels['mcm'], 'Q{}'.format(active_qubit)) for active_qubit in active_qubits]], line_labels = new_qubit_labels)
    expanded_circuit.append_circuit_inplace(expanded_final_measurements)
    no_mcm_circuit.append_circuit_inplace(expanded_final_measurements)
    
    
    expanded_circuit.done_editing()
    no_mcm_circuit.done_editing()
    native_circuit.done_editing()
    check_circuit.done_editing()

    # need to handle the final measurement sign...the Pauli should be Z-type here because we used the measure layer earlier
    
    measurement = ['I' if i == 'I' else 'Z' for i in current_pauli]
    sign = birb.determine_sign(s_final_state, p_final_state, measurement)
    
    if debug:
        return expanded_circuit, native_circuit, measurement, sign, no_mcm_circuit, mcm_count, check_circuit, s_final_state, p_final_state
    
    return native_circuit, measurement, sign, qubit_pointers
