import numpy as np
import stim
import pygsti

from pygsti.baseobjs import Label, QubitSpace, BuiltinBasis

import pygsti.tools.errgenproptools as eprop
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator
from pygsti.baseobjs import Label, QubitSpace
from pygsti.models import LocalNoiseModel
from pygsti.modelmembers.operations import ComposedOp, LindbladErrorgen, ExpErrorgenOp, StaticCliffordOp, EmbeddedErrorgen, ComposedErrorgen
from pygsti.modelmembers.states import ComposedState, ComputationalBasisState
from pygsti.modelmembers.povms import ComposedPOVM, ComputationalBasisPOVM


def create_syndrome_extraction_circuit(circuit, qubit_labels, allowed_gates=['H', 'CX'], qubit_relabelling_dict=None):
    """
    Creates a pyGSTi circuit for syndrome extraction from the stim circuit 
    `circuit` for syndrome. Will not work for any other circuits.
    """    

    if qubit_relabelling_dict is None:
        qubit_relabelling_dict = {q:q for q in qubit_labels}

    def convert_hadamard_layer(stim_layer_string):
        return '[' + ''.join(['Gh:'+str(qubit_relabelling_dict[int(s)]) for s in stim_layer_string.split(' ')[5:]]) + ']'

    def convert_cnot_layer(stim_layer_string):
        cnot_qubits = stim_layer_string.split(' ')[5:]
        num_cnots = len(cnot_qubits) // 2
        pygsti_l_cnot = '['
        for i in range(num_cnots):
            q1 = str(qubit_relabelling_dict[int(cnot_qubits[2 * i])])
            q2 = str(qubit_relabelling_dict[int(cnot_qubits[2 * i + 1])])
            pygsti_l_cnot +=  'Gcnot:'+ q1 + ':' + q2
        pygsti_l_cnot += ']'
        return pygsti_l_cnot

    gate_layers = []
    cstr = str(circuit).split('{')[1] #find the repeat
    cstr = cstr.split('MR')[0] #find the measurement layer, want the gates, which are before
    #iterate through the layers, get rid of "TICK" layers, and convert the others (should only be CX and H) into pyGSTi
    for line in cstr.split('\n'):
        #check if gate layer
        if 'H' in line.split(' '):
            gate_layers.append(convert_hadamard_layer(line))
        elif 'CX' in line.split(' '):
            gate_layers.append(convert_cnot_layer(line))
        else:
            pass #WE ARE ASSUMING THAT THERE ARE ONLY H AND CX GATES. THIS SHOULD BE MADE MORE ROBUST

    pyg_cstr = ''.join(gate_layers)+'@(' + ','.join([str(q) for q in qubit_relabelling_dict.values()]) + ')'
    
    return pygsti.circuits.Circuit(pyg_cstr)

def create_multiround_syndrome_extraction_circuit(circuit, qubit_labels, ancilla_qubits, rounds):
    n_ancilla = len(ancilla_qubits)
    n_qubits = len(qubit_labels)
    pcircuits = []
    for r in range(rounds):
        qubit_relabelling_dict = {q:q for q in qubit_labels}
        # Overwrite for ancilla qubits to label by syndrome extraction round
        if r>0:
            qubit_relabelling_dict.update({q:n_qubits+n_ancilla*(r-1)+i for i,q in enumerate(ancilla_qubits)})
        round_r_circ = create_syndrome_extraction_circuit(circuit, qubit_labels, qubit_relabelling_dict=qubit_relabelling_dict)
        pcircuits.append(round_r_circ)
        
    pcirc = pcircuits[0]
    for i in range(1, rounds):
        pcirc += pcircuits[i]
        
    return pcirc

def pygsti_c_to_stim(pyg_c):
    gatenames_to_stim = {}
    gatenames_to_stim['Gh'] = 'H'
    gatenames_to_stim['Gcnot'] = 'CX'
    gatenames_to_stim['Gcphase'] = 'CZ'
    
    qs = [str(q) for q in pyg_c.line_labels]
    stim_cstr='R '+' '.join(qs)+'\n'
    for layer in pyg_c:
        for gate in layer:
            stim_name = gatenames_to_stim[gate.name]
            stim_qs = ' '.join([str(q) for q in list(gate.qubits)])
            stim_cstr += (gatenames_to_stim[gate.name]+' '+' '.join([str(q) for q in list(gate.qubits)])+'\n')
    #add measurement of all qubits
    stim_cstr += ('M '+' '.join(qs))

    #iterate through the layers
    #iterate through the gates in the layer
    #translate to stim. Probably need 1Q/2Q gate casework

    stim_c = stim.Circuit(stim_cstr)
    
    return stim_c


def create_processor_spec(pcircuit, qubit_labels, gates=['Gcnot','Gh']):
    """
    Creates a processor spec that from a pyGSTi syndrome extraction circuit, containing the 
    required gates (CNOTs between qubits that are coupled in those circuits). Will then be
    used to define the noise model we simulate.
    """
    ###Note to future self: watch out for the possibility of wanting other gates###
    pyg_cstr = pcircuit.str
    # Find all the CNOT gates used in the circuit, using brute force string comprehension
    cleaned_pcstr = pyg_cstr.replace(']', '').replace('[', '').split('@')[0]
    connections = [(int(s.split(':')[1]), int(s.split(':')[2])) for s in cleaned_pcstr.split('G') if len(s.split(':')) > 2]
    availability={'Gcnot':connections}
    pspec = pygsti.processors.QubitProcessorSpec(len(qubit_labels), gates, qubit_labels=qubit_labels,
                                             availability=availability)
    
    return pspec

def build_model(error_rates, pspec):
    """
    Creates a *spatially homogeneous* error model from
    the given error rates dictionary. Could easily be generalized to
    spatially inhomogeneous errors. Works only for a 
    gate set of only Gh and Gcnot.
    """
    ######TODO: GENERALIZE TO OTHER MODELS########
    oneQ_gate_names = ['Gh', ]
    twoQ_gate_names = ['Gcnot', ]
    gate_dictionary = dict()

    
    def create_gate_object(erates, num_qubits, gate_name):
        state_space = QubitSpace(num_qubits)
        lindblad_errorgen = LindbladErrorgen.from_elementary_errorgens(erates, 
                                                                       state_space=state_space,
                                                                       evotype='stabilizer')
        perfect_gate = StaticCliffordOp(pspec.gate_unitaries[gate_name], evotype='stabilizer')
        exp_errorgen = ExpErrorgenOp(lindblad_errorgen)
        gate_mm = ComposedOp([perfect_gate, exp_errorgen])
        return gate_mm
    
    for i, gate_name in enumerate(oneQ_gate_names):         
        for j, q in enumerate(pspec.qubit_labels):
            label = pygsti.baseobjs.label.Label(gate_name, q)
            gate_dictionary[label] = create_gate_object(error_rates[gate_name], 1, gate_name)

    for i, gate_name in enumerate(twoQ_gate_names):
        for j, qs in enumerate(pspec.availability[gate_name]):
            label = pygsti.baseobjs.label.Label(gate_name, qs)
            gate_dictionary[label] = create_gate_object(error_rates[gate_name], 2, gate_name)
   
    prep_layers = None
    povm_layers = None
    model = LocalNoiseModel(pspec, gate_dictionary, prep_layers, povm_layers, evotype='stabilizer')
    return model

import numpy as np
import stim
import pygsti
import re

import pygsti.tools.errgenproptools as eprop
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator
from pygsti.baseobjs import Label, QubitSpace
from pygsti.models import LocalNoiseModel
from pygsti.modelmembers.operations import ComposedOp, LindbladErrorgen, ExpErrorgenOp, StaticCliffordOp


def create_syndrome_extraction_circuit(circuit, qubit_labels, allowed_gates=['H', 'CX'], qubit_relabelling_dict=None):
    """
    Creates a pyGSTi circuit for syndrome extraction from the stim circuit 
    `circuit` for syndrome. Will not work for any other circuits.
    """    

    if qubit_relabelling_dict is None:
        qubit_relabelling_dict = {q:q for q in qubit_labels}

    def convert_hadamard_layer(stim_layer_string):
        return '[' + ''.join(['Gh:'+str(qubit_relabelling_dict[int(s)]) for s in stim_layer_string.split(' ')[5:]]) + ']'

    def convert_cnot_layer(stim_layer_string):
        cnot_qubits = stim_layer_string.split(' ')[5:]
        num_cnots = len(cnot_qubits) // 2
        pygsti_l_cnot = '['
        for i in range(num_cnots):
            q1 = str(qubit_relabelling_dict[int(cnot_qubits[2 * i])])
            q2 = str(qubit_relabelling_dict[int(cnot_qubits[2 * i + 1])])
            pygsti_l_cnot +=  'Gcnot:'+ q1 + ':' + q2
        pygsti_l_cnot += ']'
        return pygsti_l_cnot

    gate_layers = []
    cstr = str(circuit).split('{')[1] #find the repeat
    cstr = cstr.split('MR')[0] #find the measurement layer, want the gates, which are before
    #iterate through the layers, get rid of "TICK" layers, and convert the others (should only be CX and H) into pyGSTi
    for line in cstr.split('\n'):
        #check if gate layer
        if 'H' in line.split(' '):
            gate_layers.append(convert_hadamard_layer(line))
        elif 'CX' in line.split(' '):
            gate_layers.append(convert_cnot_layer(line))
        else:
            pass #WE ARE ASSUMING THAT THERE ARE ONLY H AND CX GATES. THIS SHOULD BE MADE MORE ROBUST

    pyg_cstr = ''.join(gate_layers)+'@(' + ','.join([str(q) for q in qubit_relabelling_dict.values()]) + ')'
    
    return pygsti.circuits.Circuit(pyg_cstr)

def create_multiround_syndrome_extraction_circuit(circuit, qubit_labels, ancilla_qubits, rounds):
    n_ancilla = len(ancilla_qubits)
    n_qubits = len(qubit_labels)
    pcircuits = []
    for r in range(rounds):
        qubit_relabelling_dict = {q:q for q in qubit_labels}
        # Overwrite for ancilla qubits to label by syndrome extraction round
        if r>0:
            qubit_relabelling_dict.update({q:n_qubits+n_ancilla*(r-1)+i for i,q in enumerate(ancilla_qubits)})
        round_r_circ = create_syndrome_extraction_circuit(circuit, qubit_labels, qubit_relabelling_dict=qubit_relabelling_dict)
        pcircuits.append(round_r_circ)
        
    pcirc = pcircuits[0]
    for i in range(1, rounds):
        pcirc += pcircuits[i]
        
    return pcirc

def pygsti_c_to_stim(pyg_c, include_measurement=True):
    gatenames_to_stim = {}
    gatenames_to_stim['Gh'] = 'H'
    gatenames_to_stim['Gzpi'] = 'Z'
    gatenames_to_stim['Gypi'] = 'Y'
    gatenames_to_stim['Gxpi'] = 'X'
    gatenames_to_stim['Gypi2'] = 'SQRT_Y'
    gatenames_to_stim['Gympi2'] = 'SQRT_Y_DAG'
    gatenames_to_stim['Gcnot'] = 'CX'
    gatenames_to_stim['Gh'] = 'H'
    gatenames_to_stim['Gcphase'] = 'CZ'
    gatenames_to_stim['Gxpi2'] = 'SQRT_X'
    gatenames_to_stim['Gxmpi2'] = 'SQRT_X_DAG'
    gatenames_to_stim['Gi'] = 'I'
    
    qs = [str(q) for q in pyg_c.line_labels]
    stim_cstr= 'I '+' '.join(q for q in qs)+'\n'
    for gate in pyg_c:
        #assume serialized
        stim_name = gatenames_to_stim[gate.name]
        stim_qs = ' '.join([str(q) for q in list(gate.qubits)])
        stim_cstr += (gatenames_to_stim[gate.name]+' '+' '.join([str(q) for q in list(gate.qubits)])+'\n')
    #add measurement of all qubits
    if include_measurement:
        stim_cstr += ('M '+' '.join(qs))


    stim_c = stim.Circuit(stim_cstr)
    
    return stim_c


def create_processor_spec(pcircuit, qubit_labels, gates=['Gcnot','Gh']):
    """
    Creates a processor spec that from a pyGSTi syndrome extraction circuit, containing the 
    required gates (CNOTs between qubits that are coupled in those circuits). Will then be
    used to define the noise model we simulate.
    """
    ###Note to future self: watch out for the possibility of wanting other gates###
    pyg_cstr = pcircuit.str
    # Find all the CNOT gates used in the circuit, using brute force string comprehension
    cleaned_pcstr = pyg_cstr.replace(']', '').replace('[', '').split('@')[0]
    connections = [(int(s.split(':')[1]), int(s.split(':')[2])) for s in cleaned_pcstr.split('G') if len(s.split(':')) > 2]
    availability={'Gcnot':connections}
    pspec = pygsti.processors.QubitProcessorSpec(len(qubit_labels), gates, qubit_labels=qubit_labels,
                                             availability=availability)
    
    return pspec

def local_depolarizing_errorgen(num_qubits, depolarization_rate):
    state_space = QubitSpace(1)
    basis_size = state_space.dim  # e.g. 4 for a single qubit
    basis = BuiltinBasis('PP', state_space)
    rate_per_pauli = depolarization_rate / (basis_size - 1)
    errdict = {('S', bl): rate_per_pauli for bl in basis.labels[1:]}
    errgen = LindbladErrorgen.from_elementary_errorgens(
        errdict, "D", basis, mx_basis='pp',
        truncate=False, evotype='stabilizer', state_space=state_space)
    #calling this twice is intentional. Second setting adds adjustment to get correct rate adjustment for depol probability.
    errgen.set_error_rates(errdict)

    big_qubitspace = QubitSpace(num_qubits)
    embedded_errorgens = []
    for i in range(num_qubits):
        embedded_errorgens.append(EmbeddedErrorgen(big_qubitspace, [i], errgen))
    composed_errorgen = ComposedErrorgen(embedded_errorgens, evotype='stabilizer', state_space=big_qubitspace)    
        
    return composed_errorgen
    

def build_model(error_rates, pspec, oneQ_gate_names, twoQ_gate_names):
    gate_dictionary = dict()

    
    def create_gate_object(erates, num_qubits, gate_name):
        state_space = QubitSpace(num_qubits)
        lindblad_errorgen = LindbladErrorgen.from_elementary_errorgens(erates,
                                                                       state_space=state_space,
                                                                       evotype='stabilizer')
        perfect_gate = StaticCliffordOp(pspec.gate_unitaries[gate_name], evotype='stabilizer')
        exp_errorgen = ExpErrorgenOp(lindblad_errorgen)
        gate_mm = ComposedOp([perfect_gate, exp_errorgen])
        return gate_mm
    
    for i, gate_name in enumerate(oneQ_gate_names):         
        for j, q in enumerate(pspec.qubit_labels):
            label = pygsti.baseobjs.label.Label(gate_name, q)
            gate_dictionary[label] = create_gate_object(error_rates[gate_name], 1, gate_name)

    for i, gate_name in enumerate(twoQ_gate_names):
        for j, qs in enumerate(pspec.availability[gate_name]):
            label = pygsti.baseobjs.label.Label(gate_name, qs)
            gate_dictionary[label] = create_gate_object(error_rates[gate_name], 2, gate_name)

    num_qubits = pspec.num_qubits
    
    prep_errorgen = local_depolarizing_errorgen(num_qubits, 0)
    prep_layers = {Label('rho0'): ComposedState(ComputationalBasisState(zvals = ['0']*num_qubits, evotype='stabilizer'), ExpErrorgenOp(prep_errorgen))}
    #setting the number of qubits for the base POVM is a hack...
    povm_layers = {Label('Mdefault'): ComposedPOVM(ExpErrorgenOp(prep_errorgen), ComputationalBasisPOVM(1, evotype='stabilizer', state_space=QubitSpace(1)))}


    #prep_layers = None
    #povm_layers = None
    model = LocalNoiseModel(pspec, gate_dictionary, prep_layers, povm_layers, evotype='stabilizer')
    return model

def parse_pauli_product(prod, current_qubit_mapping):
    #determine paulis and qubits
    if prod[0]=="!":
        prod = prod[1:] #TODO record that this measurement result should be flipped
    terms = prod.split('*')
    pauli_string = ''.join([t[0] for t in terms])
    qubits = tuple(current_qubit_mapping[int(t[1:])] for t in terms)
    return (pauli_string, qubits)

    
def stim_to_pygsti_circuit(circuit, qubit_labels, qubit_relabelling_dict=None, show_qubit_mappings=False, include_observables=False, include_idles=False):
    
    if qubit_relabelling_dict is None:
        qubit_relabelling_dict = {q:q for q in qubit_labels}

    qubit_mapping_dict = {}

    measurements = []
    detectors = []

    def convert_1q_layer(gatename, stim_layer_string, current_qubit_mapping):
        gate_dict= {'H':'Gh', 
                    'SQRT_Y': 'Gypi2',
                    'SQRT_Y_DAG': 'Gympi2',
                    'SQRT_X_DAG': 'Gxmpi2',
                    'Z': 'Gzpi',
                    'Y': 'Gypi', 
                    'X':'Gxpi',
                   'I':'Gi'}
        pygsti_name = gate_dict[gatename]
        return '[' + ''.join([f'{pygsti_name}:'+str(current_qubit_mapping[int(s)]) for s in stim_layer_string.split(' ')[1:]]) + ']'

    def convert_2q_layer(gatename, stim_layer_string, current_qubit_mapping):
        gate_dict = {'CZ': 'Gcphase', 'CX': 'Gcnot'}
        cnot_qubits = stim_layer_string.split(' ')[1:] #what????
        num_cnots = len(cnot_qubits) // 2
        pygsti_l_cnot = '['
        for i in range(num_cnots):
            q1 = str(current_qubit_mapping[int(cnot_qubits[2 * i])])
            q2 = str(current_qubit_mapping[int(cnot_qubits[2 * i + 1])])
            pygsti_l_cnot +=  f'{gate_dict[gatename]}:'+ q1 + ':' + q2
        pygsti_l_cnot += ']'
        return pygsti_l_cnot

    def update_qubit_mapping(line, current_qubit_mapping, qubit_labels):
        new_mapping = current_qubit_mapping.copy()
        if line.split(' ')[0]=='MPP':
            qs = []
            for prod in line.split(' ')[1:]:
                _, qs_prod = parse_pauli_product(prod, {q:q for q in qubit_labels})
                qs.extend(qs_prod)
            qs = list(set(qs))
            print(line, qs)
        
        else:
            qs = line.split(' ')[1:]
        nqs = np.max(list(current_qubit_mapping.values()))+1
        for i, measured_physical_q in enumerate(qs):
            new_mapping[int(measured_physical_q)] = nqs+i
        return new_mapping
    
    gate_layers = []
    cstr = str(circuit)
    #need to keep track of current virtual qubits and map
    current_qubit_mapping = {q:q for q in qubit_labels}
    gates_happened = False
    for line in cstr.split('\n'):
        #check if gate layer
        if 'H' in line.split(' ') or 'SQRT_Y' in line.split(' ') or 'SQRT_Y_DAG' in line.split(' ') or 'Z' in line.split(' ') or 'X' in line.split(' ') or 'Y' in line.split(' ')  or 'SQRT_X_DAG' in line.split(' ') or 'I' in line.split(' '):
            gatename = line.split(' ')[0]
            gate_layers.append(convert_1q_layer(gatename, line, current_qubit_mapping))
            gates_happened = True
            unused_qs = [current_qubit_mapping[k] for k in qubit_labels if str(k) not in line.split(' ')[1:]]
            if include_idles:
                gate_layers.append('[' + ''.join([f'Gi:'+str(s) for s in unused_qs]) + ']')
        elif 'CX' in line.split(' '):
            gate_layers.append(convert_2q_layer('CX', line, current_qubit_mapping))
            gates_happened = True
            unused_qs = [current_qubit_mapping[k] for k in qubit_labels if str(k) not in line.split(' ')[1:]]
            if include_idles:
                gate_layers.append('[' + ''.join([f'Gi:'+str(s) for s in unused_qs]) + ']')
        elif 'CZ' in line.split(' '):
            gate_layers.append(convert_2q_layer('CZ', line, current_qubit_mapping))
            gates_happened = True
            unused_qs = [current_qubit_mapping[k] for k in qubit_labels if str(k) not in line.split(' ')[1:]]
            if include_idles:
                gate_layers.append('[' + ''.join([f'Gi:'+str(s) for s in unused_qs]) + ']')
        elif line.split(' ')[0] in ['M','R','MR','MPP']:
            if line.split(' ')[0] in ['M','MR']:
                #gate_layers.append(convert_1q_layer('I', line, current_qubit_mapping))
                measurements.extend([('Z',(current_qubit_mapping[int(q)],)) for q in line.split(' ')[1:]])
            elif line.split(' ')[0]=='MPP':
                measurements.extend([parse_pauli_product(prod, current_qubit_mapping) for prod in line.split(' ')[1:]]) #TODO 
                
            if gates_happened and line.split(' ')[0] !='MPP': ###MIGHT NEED TO CHANGE MPP case
                #assuming there aren't initial measurements in the circuit
                current_qubit_mapping = update_qubit_mapping(line, current_qubit_mapping, qubit_labels) #in a measurement layer, we just need to update our mapping to expanded circuit qubits
                if show_qubit_mappings:
                    print(current_qubit_mapping)
        elif 'DETECTOR' in line.split(' ')[0]: #or 'OBSERVABLE_INCLUDE' in line.split('(')[0]:
            #print(line)
            det = []
            for arg in line.split(' ')[1:]:
                #get the ref number, it's negative, index into measurements
                if 'rec' in arg:
                    rel_meas = int(re.findall(r'\[-(\d+)\]', arg)[0])
                    targ = len(measurements)-1*rel_meas
                    det.append(targ)
            detectors.append(det)
        elif 'OBSERVABLE_INCLUDE' in line.split('(')[0]:
            if not include_observables:
                pass
            else:
                #print(line)
                det = []
                for arg in line.split(' ')[1:]:
                    #get the ref number, it's negative, index into measurements
                    if 'rec' in arg:
                        rel_meas = int(re.findall(r'\[-(\d+)\]', arg)[0])
                        targ = len(measurements)-1*rel_meas
                        det.append(targ)
                detectors.append(det)                
        elif 'TICK' in line.split(' ') or 'QUBIT_COORDS' or 'SHIFT_COORDS' in line.split(' '):
            pass
        else:
            print(f'instruction not found for {line}') #IF THIS COMES UP, MAYBE BE CONCERNED

    pyg_c = pygsti.circuits.Circuit(''.join(gate_layers)+'@(' + ','.join([str(q) for q in range(1+max(current_qubit_mapping.values()))]) + ')', editable=True)
    pyg_c.delete_idle_layers_inplace()
    pyg_c.done_editing()
    
    return pyg_c, current_qubit_mapping, measurements, detectors
