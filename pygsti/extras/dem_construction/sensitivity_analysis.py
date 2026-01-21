import numpy as np
import stim
import pygsti
import itertools

import pygsti.tools.errgenproptools as eprop
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator
from pygsti.baseobjs import Label, QubitSpace
from pygsti.models import LocalNoiseModel
from pygsti.modelmembers.operations import ComposedOp, LindbladErrorgen, ExpErrorgenOp, StaticCliffordOp

#from helper_functions import *

def compute_sensitivity_vectors_old(mapping_to_eoc_eegs, egp, pcircuit, parameter_indexing,
                                return_s_equivalent=False):
    """
    This function turns the output of the `errorgen_transform_map` method
    into  "sensitivity vectors" that encode how each error in the circuit
    error map's Linbladian is a function of the model parameters. Because
    we're doing a 1st order BCH expansion, this can be represented by
    a vector.
    """
    num_parameters = len(parameter_indexing)
    # The different elementary error generators in the circuit's Linbladian.
    unique_eoc_eegs = set([i[0] for i in list(mapping_to_eoc_eegs.values())])
    # This'll store how each end-of-circuit error depends on the model parameters
    sensitivity_vectors = {eeg : np.zeros((num_parameters,), float) for eeg in unique_eoc_eegs}
    if return_s_equivalent:
        # This'll store how each end-of-circuit error depends on the model parameters
        s_sensitivity_vectors = {eeg : np.zeros((num_parameters,), float) for eeg in unique_eoc_eegs}
    
    for circuit_error_key, eoc_egg in mapping_to_eoc_eegs.items():            
        # The layer the error occured in
        layer_index = circuit_error_key[1]
        # The elementary error generator
        eeg = circuit_error_key[0]
        # This tells us which gate(s) could have caused this error in this layer.
        gates_causing_error = egp.errorgen_gate_contributors(eeg, pcircuit, layer_index, include_spam=False)
        # In our circuits + error model, there is a unique gate that could have caused this error...
        assert(len(gates_causing_error) == 1)
        # ... and it is this gate:
        gate_causing_error = gates_causing_error[0]
        # The qubits that gate acts on.
        qubits = gate_causing_error.qubits
        # Then in-circuit error's Pauli string
        ps = circuit_error_key[0].bel_to_strings()[0]
        # The label for the parameter in the model associated with this gate error
        parameter = (gates_causing_error[0].name, ''.join([ps[q] for q in qubits]))
        # The parameter index
        parameter_index = parameter_indexing.index(parameter)
        # Whether the gate's error contributes positively or negatively to this end-of-circuit error.
        sign = eoc_egg[1]
        # Add this the end-of-circuits sensitivity vector
        sensitivity_vectors[eoc_egg[0]][parameter_index] += sign
        if return_s_equivalent:
            # Add this the end-of-circuits sensitivity vector. Stochastic error always contributes
            # positively.
            s_sensitivity_vectors[eoc_egg[0]][parameter_index] += 1
    if return_s_equivalent:
        return sensitivity_vectors, s_sensitivity_vectors
    else:
        return sensitivity_vectors
        
def compute_sensitivity_vectors(mapping_to_eoc_eegs, egp, pcircuit, parameter_indexing,
                                return_s_equivalent=False):
    """
    This function turns the output of the `errorgen_transform_map` method
    into  "sensitivity vectors" that encode how each error in the circuit
    error map's Linbladian is a function of the model parameters. Because
    we're doing a 1st order BCH expansion, this can be represented by
    a vector.
    """
    num_parameters = len(parameter_indexing)
    # The different elementary error generators in the circuit's Linbladian.
    unique_eoc_eegs = set([i[0] for i in list(mapping_to_eoc_eegs.values())])
    # This'll store how each end-of-circuit error depends on the model parameters
    sensitivity_vectors = {eeg : np.zeros((num_parameters,), float) for eeg in unique_eoc_eegs}
    if return_s_equivalent:
        # This'll store how each end-of-circuit error depends on the model parameters
        s_sensitivity_vectors = {eeg : np.zeros((num_parameters,), float) for eeg in unique_eoc_eegs}
    
    for circuit_error_key, eoc_egg in mapping_to_eoc_eegs.items():            
        # The layer the error occured in
        layer_index = circuit_error_key[1]
        # The elementary error generator
        eeg = circuit_error_key[0]
        # This tells us which gate(s) could have caused this error in this layer.
        gates_causing_error = egp.errorgen_gate_contributors(eeg, pcircuit, layer_index, include_spam=False)
        # In our circuits + error model, there is a unique gate that could have caused this error...
        assert(len(gates_causing_error) == 1)
        # ... and it is this gate:
        gate_causing_error = gates_causing_error[0]
        # The qubits that gate acts on.
        qubits = gate_causing_error.qubits
        # Then in-circuit error's Pauli string
        ps = circuit_error_key[0].bel_to_strings()[0]
        # The label for the parameter in the model associated with this gate error
        errgen_type = eeg.errorgen_type
        parameter = (gates_causing_error[0].name, (errgen_type,''.join([ps[q] for q in qubits]))) #TODO need to determine errorgen type
        # The parameter index
        parameter_index = parameter_indexing.index(parameter)
        # Whether the gate's error contributes positively or negatively to this end-of-circuit error.
        sign = eoc_egg[1]
        # Add this the end-of-circuits sensitivity vector
        sensitivity_vectors[eoc_egg[0]][parameter_index] += sign
        if return_s_equivalent:
            # Add this the end-of-circuits sensitivity vector. Stochastic error always contributes
            # positively.
            s_sensitivity_vectors[eoc_egg[0]][parameter_index] += 1
    if return_s_equivalent:
        return sensitivity_vectors, s_sensitivity_vectors
    else:
        return sensitivity_vectors


def create_sensitivity_matrix(svec1, svec2, num_parameters):
    svec1_col = np.reshape(svec1, (num_parameters, 1))
    svec2_row = np.reshape(svec2, (1, num_parameters))
    return np.dot(svec1_col, svec2_row)   

def compute_z_expectation_s_matrix(qubit_subset, sensitivity_vectors, qubit_labels,
                                     inverse_tableau, ancilla_qubits):
    '''
    R: qubit_subset: subset of ancilla_qubits
    '''
    num_parameters = len(list(sensitivity_vectors.values())[0])
    sensitivity_matrix = np.zeros((num_parameters, num_parameters),float) 
    z_measurement_quadratic_H_terms = {}

    # The measurement Pauli operator
    R = z_pauli_on_qubits(qubit_subset, qubit_labels) #MAKE INTO Z-PAULI ON MULTIPLE QUBITS
    assert(is_stabilizer(R, inverse_tableau)), "Z should be a stabilizer of the ancilla qubits!"

    #compute errors on qubit subset
    errors_qubit_subset = compute_errors_qubit_subset(qubit_subset, sensitivity_vectors, qubit_labels, included_paulis=['X','Y','Z'])

    #GET A LIST OF ERRORS FOR A QUBIT SUBSET
    for eeg1 in errors_qubit_subset:
        P = eeg1.basis_element_labels[0]
        # If [P,R] = 0 then this term does not contribute.
        if not P.commutes(R):
            for eeg2 in errors_qubit_subset:
                Q = eeg2.basis_element_labels[0]
                # If [Q,R] = 0 then this term does not contribute.
                if not Q.commutes(R):
                    # Only terms where [P,Q] = 0 contribute
                    if P.commutes(Q):
                        contribution = -2 * is_stabilizer_or_antistabilizer(Q * P, inverse_tableau) 
                        if contribution != 0:
                            z_measurement_quadratic_H_terms[(eeg1, eeg2)] = contribution
                                
    for (eeg1, eeg2), contribution in z_measurement_quadratic_H_terms.items():      
        svec_1 = sensitivity_vectors[eeg1]
        svec_2 = sensitivity_vectors[eeg2]
        sensitivity_matrix += contribution * create_sensitivity_matrix(svec_1, svec_2, num_parameters=num_parameters)                   
            
    return sensitivity_matrix