
import numpy as np
import stim
import pygsti
import re

from pygsti.baseobjs import Label, QubitSpace, BuiltinBasis

import pygsti.tools.errgenproptools as eprop
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator
from pygsti.baseobjs import Label, QubitSpace
from pygsti.models import LocalNoiseModel
from pygsti.modelmembers.operations import ComposedOp, LindbladErrorgen, ExpErrorgenOp, StaticCliffordOp, EmbeddedErrorgen, ComposedErrorgen
from pygsti.modelmembers.states import ComposedState, ComputationalBasisState
from pygsti.modelmembers.povms import ComposedPOVM, ComputationalBasisPOVM


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
    availability={'Gcnot':connections, 'Gcphase':connections}
    pspec = pygsti.processors.QubitProcessorSpec(len(qubit_labels), gates, qubit_labels=qubit_labels,
                                             availability=availability)
    
    return pspec


def local_x_errorgen(num_qubits, rate):
    state_space = QubitSpace(1)
    basis_size = state_space.dim  # e.g. 4 for a single qubit
    basis = BuiltinBasis('PP', state_space)
    errdict = {('S', bl): rate for bl in basis.labels[1:] if 'Y' not in bl and 'Z' not in bl}
    errgen = LindbladErrorgen.from_elementary_errorgens(
        errdict, "D", basis, mx_basis='pp',
        truncate=False, evotype='stabilizer', state_space=state_space)
    #calling this twice is intentional. Second setting adds adjustment to get correct rate adjustment for depol probability.
    #errgen.set_error_rates(errdict)
    
    embedded_errorgens = []
    for i in range(num_qubits):
        embedded_errorgens.append(EmbeddedErrorgen(QubitSpace(num_qubits), [i], errgen))
    composed_errorgen = ComposedErrorgen(embedded_errorgens, evotype='stabilizer', state_space=QubitSpace(num_qubits))    
        
    return composed_errorgen
    

def build_model(error_rates, pspec, oneQ_gate_names, twoQ_gate_names, meas_err=0, prep_err=0):
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

    nqs = pspec.num_qubits

    povm_errorgen = LindbladErrorgen.from_elementary_errorgens({('S', 'X'): meas_err}, state_space=QubitSpace(1), evotype='stabilizer')

    povm_layers = {Label('Mdefault'): ComposedPOVM(ExpErrorgenOp(povm_errorgen), ComputationalBasisPOVM(1, evotype='stabilizer', state_space=QubitSpace(1)))}
    
    prep_errorgen = local_x_errorgen(nqs, prep_err) 

    prep_layers = {Label('rho0'): ComposedState(ComputationalBasisState(zvals = ['0']*nqs, evotype='stabilizer'), ExpErrorgenOp(prep_errorgen))}

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

    
def stim_to_pygsti_circuit(circuit, qubit_labels, qubit_relabelling_dict=None, show_qubit_mappings=False, include_observables=False, separate_observables=False, include_meas_idles=False, include_idles=False, ignore_measurement=False):
    
    if qubit_relabelling_dict is None:
        qubit_relabelling_dict = {q:q for q in qubit_labels}

    qubit_mapping_dict = {}

    measurements = []
    detectors = []
    observable_detectors = []

    def convert_1q_layer(gatename, stim_layer_string, current_qubit_mapping):
        gate_dict= {'H':'Gh', 
                    'SQRT_Y': 'Gypi2',
                    'SQRT_Y_DAG': 'Gympi2',
                    'SQRT_X_DAG': 'Gxmpi2',
                    'T': 'Gt',
                    'T_DAG': 'Gtdag',
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
    if type(circuit)==stim.Circuit:
        cstr = str(circuit)
    else: cstr = circuit
    #need to keep track of current virtual qubits and map
    current_qubit_mapping = {q:q for q in qubit_labels}
    gates_happened = False
    for line in cstr.split('\n'):
        #check if gate layer
        if 'T' in line.split(' ') or 'T_DAG' in line.split(' ') or 'H' in line.split(' ') or 'SQRT_Y' in line.split(' ') or 'SQRT_Y_DAG' in line.split(' ') or 'Z' in line.split(' ') or 'X' in line.split(' ') or 'Y' in line.split(' ')  or 'SQRT_X_DAG' in line.split(' ') or 'I' in line.split(' '):
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
        elif line.split(' ')[0] in ['M','R','MR','MPP'] and not ignore_measurement:
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
            if include_meas_idles:
                unused_qs = [current_qubit_mapping[k] for k in qubit_labels if str(k) not in line.split(' ')[1:]]
                gate_layers.append('[' + ''.join([f'Gi:'+str(s) for s in unused_qs]) + ']')
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
            if include_observables:
                # Extract the integer j from OBSERVABLE_INCLUDE(j)
                match = re.search(r'OBSERVABLE_INCLUDE\((\d+)\)', line)
                if match:
                    j = int(match.group(1))
                    if len(observable_detectors) <= j:
                        observable_detectors.append([])
                    # Extract measurement references from the line
                    for arg in line.split(' ')[1:]:
                        if 'rec' in arg:
                            rel_meas = int(re.findall(r'\[-(\d+)\]', arg)[0])
                            targ = len(measurements) - 1 * rel_meas
                            observable_detectors[j].append(targ)          
        
        elif 'TICK' in line.split(' ') or 'QUBIT_COORDS' or 'SHIFT_COORDS' in line.split(' '):
            pass
        else:
            print(f'instruction not found for {line}') #IF THIS COMES UP, MAYBE BE CONCERNED

    pyg_c = pygsti.circuits.Circuit(''.join(gate_layers)+'@(' + ','.join([str(q) for q in range(1+max(current_qubit_mapping.values()))]) + ')', editable=True)
    pyg_c.delete_idle_layers_inplace()
    pyg_c.done_editing()

    if not separate_observables:
        detectors.extend(observable_detectors)
    
        return pyg_c, current_qubit_mapping, measurements, detectors
    
    else:
        return pyg_c, current_qubit_mapping, measurements, detectors, observable_detectors