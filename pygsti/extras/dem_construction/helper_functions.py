import numpy as np
import stim

from itertools import combinations, chain

def powerset(iterable):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

def is_stabilizer(stim_pauli, inverse_tableau):
    p2 = inverse_tableau(stim_pauli)
    return "X" not in str(p2) and "Y" not in str(p2) and p2.sign == +1

def is_stabilizer_or_antistabilizer_from_generators(P, stabilizer_generators):
    return np.all([P.commutes(Q) for Q in stabilizer_generators])

def is_stabilizer_or_antistabilizer(stim_pauli, inverse_tableau):
    """
    Computes whether the the `stim_pauli`, a stim.PauliString, is a stabilizer
    or antistabilizer, or neither, of the stabilizer state with the
    given inverse tableau.
    
    Returns +1 if `stim_pauli` is a stabilizer of the state, -1 if it is 
    an anti-stabilizer, and 0 if it is neither
    """
    p2 = inverse_tableau(stim_pauli)
    if  "X" not in str(p2) and "Y" not in str(p2):
        if p2.sign == +1:
            return 1
        else:
            assert(p2.sign == -1)
            return -1
    else:
        return 0
    
def z_pauli_on_qubit(q, qubit_labels):
    ps = ['_'] * len(qubit_labels)
    ps[qubit_labels.index(q)] = 'Z'
    ps = ''.join(ps)
    return stim.PauliString(ps)

def z_pauli_on_qubits(qs, qubit_labels):
    ps = ['_'] * len(qubit_labels)
    for q in qs:
        ps[qubit_labels.index(q)] = 'Z'
    ps = ''.join(ps)
    return stim.PauliString(ps)

def get_impacted_qubits(eeg, qubit_labels, included_paulis=['X','Y','Z']):
    """
    Finds all qubits on which the error `eeg` acts, where  `eeg` is a
    pyGSTi object.
    """
    pauli_string = str(eeg.basis_element_labels[0])[1:]
    error_acts_on_qubit = [i != '_' for i in pauli_string]
    q = [qubit_labels[i] for i, b in enumerate(error_acts_on_qubit) if b is True]
    #print(eeg, q)
    return q

def compute_errors_by_qubit(sensitivity_vectors, qubit_labels, included_paulis=['X','Y','Z']):
    
    # All the errors in the circuit's Linbladian that impact each qubit.
    errors_by_qubit = {q:[] for q in qubit_labels}
    eoc_eegs = list(sensitivity_vectors.keys())
    for eeg in eoc_eegs:
        qs = get_impacted_qubits(eeg, qubit_labels, included_paulis)
        for q in qs:
            errors_by_qubit[q].append(eeg)
            
    return errors_by_qubit

def compute_errors_qubit_subset(qubit_subset, sensitivity_vectors, qubit_labels, included_paulis=['X','Y','Z']):
    
    # All the errors in the circuit's Linbladian that impact each qubit.
    errors_by_qubit = compute_errors_by_qubit(sensitivity_vectors, qubit_labels, included_paulis=['X','Y','Z'])
    errors = []
    for q in qubit_subset:
        errors.extend(errors_by_qubit[q])
    errors = list(set(errors))
            
    return errors

def make_h_param_vector():
    dummy_rate = 0.0
    parameters = {}
    
    # All 3 possible local coherent errors on an H gate
    parameters['Gh'] = {}
    parameters['Gh'][('H', 'X')] = dummy_rate
    parameters['Gh'][('H', 'Z')] = dummy_rate   
    parameters['Gh'][('H', 'Y')] = dummy_rate 
    
    # All 15 possible local coherent errors on a CNOT gate
    parameters['Gcnot'] = {}
    parameters['Gcnot'][('H', ('IX',), (0,1))] = dummy_rate
    parameters['Gcnot'][('H', ('IY',), (0,1))] = dummy_rate
    parameters['Gcnot'][('H', ('IZ',), (0,1))] = dummy_rate
    
    parameters['Gcnot'][('H', ('XI',), (0,1))] = dummy_rate
    parameters['Gcnot'][('H', ('XX',), (0,1))] = dummy_rate
    parameters['Gcnot'][('H', ('XY',), (0,1))] = dummy_rate
    parameters['Gcnot'][('H', ('XZ',), (0,1))] = dummy_rate
    
    parameters['Gcnot'][('H', ('YI',), (0,1))] = dummy_rate
    parameters['Gcnot'][('H', ('YX',), (0,1))] = dummy_rate
    parameters['Gcnot'][('H', ('YY',), (0,1))] = dummy_rate
    parameters['Gcnot'][('H', ('YZ',), (0,1))] = dummy_rate
    
    parameters['Gcnot'][('H', ('ZI',), (0,1))] = dummy_rate
    parameters['Gcnot'][('H', ('ZX',), (0,1))] = dummy_rate
    parameters['Gcnot'][('H', ('ZY',), (0,1))] = dummy_rate
    parameters['Gcnot'][('H', ('ZZ',), (0,1))] = dummy_rate
    
    # Parameter indexing, used throughout the analysis. It defines our "theta" vector
    parameter_indexing = []
    parameter_indexing += [('Gh','X'),('Gh','Y'),('Gh','Z')]
    parameter_indexing += [                 ('Gcnot', 'IX'), ('Gcnot', 'IY'), ('Gcnot', 'IZ')]
    parameter_indexing += [('Gcnot', 'XI'), ('Gcnot', 'XX'), ('Gcnot', 'XY'), ('Gcnot', 'XZ')]
    parameter_indexing += [('Gcnot', 'YI'), ('Gcnot', 'YX'), ('Gcnot', 'YY'), ('Gcnot', 'YZ')]
    parameter_indexing += [('Gcnot', 'ZI'), ('Gcnot', 'ZX'), ('Gcnot', 'ZY'), ('Gcnot', 'ZZ')]

    return parameters, parameter_indexing

#Define a model with some parameters
#Make a vector of the nonzero parameters
def build_parameter_vector(error_dict):
    '''
    takes as input a dictionary of error rates
    builds a parameter vector with the relevant parameters, and ideally no others
    '''
    paramvec = []
    parameter_indexing = []
    for gatename, gatedict in error_dict.items():
        param_ordering = list((gatename, k) for k in gatedict.keys() if gatedict[k] != 0) 
        paramvec.extend([gatedict[k[1]] for k in param_ordering])
        parameter_indexing.extend(param_ordering)
    return paramvec, parameter_indexing

def error_generator_approximate_log(errorgen_dict, order = 1, truncation_threshold = 1e-14):
    """
    Compute the nth-order approximate logarithm of a small error described by the input
    error generator dictionary. (Excluding the zeroth-order identity).
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `LocalStimErrorgenLabel` and whose values are corresponding
        rates.
    
    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.
    
    truncation_threshold : float, optional (default 1e-14)
        Optional threshold used to truncate corrections whose corresponding rates
        are below this value.

    Returns
    -------
    list of dictionaries
        List of dictionaries whose keys are error generator labels and whose values are rates (including
        whatever scaling comes from order of taylor expansion). Each list corresponds to an order
        of the taylor expansion.
    """
       
 
    taylor_order_terms = [dict() for _ in range(order)]

    for lbl, rate in errorgen_dict.items():
        if abs(rate) > truncation_threshold:
            taylor_order_terms[0][lbl] = rate

    if order > 1:
        # The order of the approximation determines the combinations of error generators
        # which need to be composed. (given by cartesian products of labels in errorgen_dict).
        labels_by_order = [list(product(errorgen_dict.keys(), repeat = i+1)) for i in range(1,order)]
        # Get a similar structure for the corresponding rates
        rates_by_order = [list(product(errorgen_dict.values(), repeat = i+1)) for i in range(1,order)]
        for current_order, (current_order_labels, current_order_rates) in enumerate(zip(labels_by_order, rates_by_order), start=2):
            order_scale = (-1)**(current_order+1)/current_order
            composition_results = []
            for label_tup, rate_tup in zip(current_order_labels, current_order_rates):
                composition_results.extend(iterative_error_generator_composition(label_tup, rate_tup))
            # aggregate together any overlapping terms in composition_results
            composition_results_dict = dict()
            for lbl, rate in composition_results:
                if composition_results_dict.get(lbl,None) is None:
                    composition_results_dict[lbl] = rate
                else:
                    composition_results_dict[lbl] += rate
            for lbl, rate in composition_results_dict.items():
                if order_scale*abs(rate) > truncation_threshold:
                    taylor_order_terms[current_order-1][lbl] = order_scale*rate

    return taylor_order_terms

def unroll_repeat_blocks(circuit, add_2q_idles=False, add_measurement_idles=False):
    '''
    Unravels the repeat blocks in a stim circuit
    circuit: stim.Circuit
    add_q2_idles: bool
    Whether or not to add explicit idles on unused qubits for each two-qubit gate layer
    add_measurement_idles: bool
    Whether or not to add explicit idles on unused qubits for each measurement layer
    returns: stim.Circuit 
    A stim circuit with explicit repreated gates in place of repeat blocks
    '''
    new_circuit = stim.Circuit()
    
    for operation in circuit:
        if operation.name == "REPEAT":
            n_repeats = operation.repeat_count  # Assuming the first arg is the repeat count
            for _ in range(n_repeats):
                # Append the operations that follow the repeat block
                for op in operation.body_copy():
                    if op.name == "REPEAT":
                        break  # Stop if we hit another repeat block
                    new_circuit.append_operation(op.name, op.targets_copy(), op.gate_args_copy())
                    if (op.name=='MR' or op.name=='R') and add_measurement_idles:
                        #add idles on unmeasured qubits
                        unmeasured_qubits = set([q for q in range(circuit.num_qubits)])-set(t.qubit_value for t in op.targets_copy())
                        #print(op, unmeasured_qubits)
                        new_circuit.append_operation('I', unmeasured_qubits, op.gate_args_copy())
                    elif (op.name=='CX' or op.name=='CZ') and add_2q_idles:
                        unused_qubits = set([q for q in range(circuit.num_qubits)])-set(t.qubit_value for t in op.targets_copy())
                        #print(op, unused_qubits)
                        new_circuit.append_operation('I', unused_qubits, op.gate_args_copy())
        else:
            # Append operations that are not part of a repeat block
            new_circuit.append_operation(operation.name, operation.targets_copy())
            if (operation.name=='MR' or operation.name=='R') and add_measurement_idles:
                unmeasured_qubits = set([q for q in range(circuit.num_qubits)])-set(t.qubit_value for t in operation.targets_copy())
                #print(operation, unmeasured_qubits)
                new_circuit.append_operation('I', unmeasured_qubits, operation.gate_args_copy())
            elif (operation.name=='CX' or operation.name=='CZ') and add_2q_idles:
                unused_qubits = set([q for q in range(circuit.num_qubits)])-set(t.qubit_value for t in operation.targets_copy())
                #print(operation, unused_qubits)
                new_circuit.append_operation('I', unused_qubits, operation.gate_args_copy())
    
    return new_circuit