
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


# ---------------------------------------------------------------------------
# Unified pyGSTi <-> stim translation tables.
# pyGSTi names follow pygsti.tools.internalgates.standard_gatename_unitaries;
# stim names follow https://github.com/quantumlib/Stim/blob/main/doc/gates.md.
# Equivalences are up to global phase (consistent with
# internalgates.standard_gatenames_stim_conversions).
# ---------------------------------------------------------------------------

#: Single-qubit Clifford gates, pyGSTi name -> canonical stim name.
PYGSTI_TO_STIM_1Q = {
    'Gi': 'I',
    'Gxpi': 'X',
    'Gypi': 'Y',
    'Gzpi': 'Z',
    'Gh': 'H',
    'Gxpi2': 'SQRT_X',
    'Gxmpi2': 'SQRT_X_DAG',
    'Gypi2': 'SQRT_Y',
    'Gympi2': 'SQRT_Y_DAG',
    'Gzpi2': 'S',
    'Gzmpi2': 'S_DAG',
    'Gp': 'S',
    'Gpdag': 'S_DAG',
    # legacy pyGSTi aliases
    'Gx': 'SQRT_X',
    'Gy': 'SQRT_Y',
    'Gz': 'S',
}

#: Two-qubit Clifford gates, pyGSTi name -> canonical stim name.
PYGSTI_TO_STIM_2Q = {
    'Gcnot': 'CX',
    'Gcphase': 'CZ',
    'Gswap': 'SWAP',
    'Giswap': 'ISWAP',
    'Gxx': 'SQRT_XX',
    'Gzz': 'SQRT_ZZ',
}

#: All supported unitary gates, pyGSTi name -> stim name.
PYGSTI_TO_STIM = {**PYGSTI_TO_STIM_1Q, **PYGSTI_TO_STIM_2Q}

#: Single-qubit stim gate name (including stim aliases) -> preferred pyGSTi name.
STIM_TO_PYGSTI_1Q = {
    'I': 'Gi',
    'X': 'Gxpi',
    'Y': 'Gypi',
    'Z': 'Gzpi',
    'H': 'Gh',
    'H_XZ': 'Gh',
    'SQRT_X': 'Gxpi2',
    'SQRT_X_DAG': 'Gxmpi2',
    'SQRT_Y': 'Gypi2',
    'SQRT_Y_DAG': 'Gympi2',
    'S': 'Gzpi2',
    'SQRT_Z': 'Gzpi2',
    'S_DAG': 'Gzmpi2',
    'SQRT_Z_DAG': 'Gzmpi2',
}

#: Two-qubit stim gate name (including stim aliases) -> preferred pyGSTi name.
STIM_TO_PYGSTI_2Q = {
    'CX': 'Gcnot',
    'CNOT': 'Gcnot',
    'ZCX': 'Gcnot',
    'CZ': 'Gcphase',
    'ZCZ': 'Gcphase',
    'SWAP': 'Gswap',
    'ISWAP': 'Giswap',
    'SQRT_XX': 'Gxx',
    'SQRT_ZZ': 'Gzz',
}

#: All supported unitary gates, stim name -> pyGSTi name.
STIM_TO_PYGSTI = {**STIM_TO_PYGSTI_1Q, **STIM_TO_PYGSTI_2Q}

#: Supported stim measurement / reset instructions
STIM_Z_MEASUREMENTS = {'M', 'MR'}
STIM_PAULI_PRODUCT_MEASUREMENTS = {'MPP'}
STIM_RESETS = {'R'}
STIM_MEASURE_RESET = STIM_Z_MEASUREMENTS | STIM_PAULI_PRODUCT_MEASUREMENTS | STIM_RESETS

#: stim annotations that carry no gate/measurement content for this conversion.
#Note for future development: Using TICK information is relevant for time-dependent error
STIM_IGNORED_ANNOTATIONS = {'TICK', 'QUBIT_COORDS', 'SHIFT_COORDS'}


def _stim_instruction_name(line):
    """Return the bare stim instruction name of a line, stripping any '(...)' arguments."""
    return line.strip().split(' ')[0].split('(')[0]


def pygsti_c_to_stim(pyg_c, include_measurement=True):
    qs = [str(q) for q in pyg_c.line_labels]
    stim_cstr= PYGSTI_TO_STIM['Gi']+' '+' '.join(q for q in qs)+'\n'
    for gate in pyg_c:
        #assume serialized
        try:
            stim_name = PYGSTI_TO_STIM[gate.name]
        except KeyError:
            raise ValueError(f'No stim translation available for pyGSTi gate {gate.name}.')
        stim_qs = ' '.join([str(q) for q in list(gate.qubits)])
        stim_cstr += (stim_name+' '+stim_qs+'\n')
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

    measurements = []
    detectors = []
    observable_detectors = []

    def convert_1q_layer(gatename, stim_layer_string, current_qubit_mapping):
        pygsti_name = STIM_TO_PYGSTI_1Q[gatename]
        return '[' + ''.join([f'{pygsti_name}:'+str(current_qubit_mapping[int(s)]) for s in stim_layer_string.split(' ')[1:]]) + ']'

    def convert_2q_layer(gatename, stim_layer_string, current_qubit_mapping):
        pygsti_name = STIM_TO_PYGSTI_2Q[gatename]
        targets = stim_layer_string.split(' ')[1:]
        num_pairs = len(targets) // 2
        pygsti_layer = '['
        for i in range(num_pairs):
            q1 = str(current_qubit_mapping[int(targets[2 * i])])
            q2 = str(current_qubit_mapping[int(targets[2 * i + 1])])
            pygsti_layer +=  f'{pygsti_name}:'+ q1 + ':' + q2
        pygsti_layer += ']'
        return pygsti_layer

    def update_qubit_mapping(line, current_qubit_mapping, qubit_labels):
        new_mapping = current_qubit_mapping.copy()
        if _stim_instruction_name(line) in STIM_PAULI_PRODUCT_MEASUREMENTS:
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
        instr = _stim_instruction_name(line)
        #check if gate layer
        if instr in STIM_TO_PYGSTI_1Q:
            gate_layers.append(convert_1q_layer(instr, line, current_qubit_mapping))
            gates_happened = True
            unused_qs = [current_qubit_mapping[k] for k in qubit_labels if str(k) not in line.split(' ')[1:]]
            if include_idles:
                gate_layers.append('[' + ''.join([f'Gi:'+str(s) for s in unused_qs]) + ']')
        elif instr in STIM_TO_PYGSTI_2Q:
            gate_layers.append(convert_2q_layer(instr, line, current_qubit_mapping))
            gates_happened = True
            unused_qs = [current_qubit_mapping[k] for k in qubit_labels if str(k) not in line.split(' ')[1:]]
            if include_idles:
                gate_layers.append('[' + ''.join([f'Gi:'+str(s) for s in unused_qs]) + ']')
        elif instr in STIM_MEASURE_RESET and not ignore_measurement:
            if instr in STIM_Z_MEASUREMENTS:
                #gate_layers.append(convert_1q_layer('I', line, current_qubit_mapping))
                measurements.extend([('Z',(current_qubit_mapping[int(q)],)) for q in line.split(' ')[1:]])
            elif instr in STIM_PAULI_PRODUCT_MEASUREMENTS:
                measurements.extend([parse_pauli_product(prod, current_qubit_mapping) for prod in line.split(' ')[1:]]) #TODO 
                
            if gates_happened and instr not in STIM_PAULI_PRODUCT_MEASUREMENTS: ###MIGHT NEED TO CHANGE MPP case
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
        
        elif instr in STIM_IGNORED_ANNOTATIONS or instr in STIM_MEASURE_RESET or instr == '':
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