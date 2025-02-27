"""
Linearized GST algorithms
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np

from pygsti.errorgenpropagation import errorpropagator as _ep #import ErrorGeneratorPropagator
from pygsti.errorgenpropagation import localstimerrorgen as _lseg # import LocalStimErrorgenLabel
from pygsti.tools import optools as _optools 
import stim as _stim
from pygsti.protocols import ModelFreeformSimulator as _MFFS
from pygsti import baseobjs as _baseobjs


def build_model_parameter_indexing(lindblad_coeff, num_qubits):
	"""
	THIS FUNCTION ASSUMES A SPECIFIC ERROR MODEL. NEEDS TO BE GENERALIZED,
	IT ALSO PROBABLY SHOULDN"T RETURN IDEAL_ERROR_RATES
	"""
	error_index_list=[]
	ideal_error_rates=[]
	for key in lindblad_coeff.keys():
	    for error in lindblad_coeff[key].keys():
	        pauli=num_qubits*'I'
	        qbt1=key[1]
	        if key[0] == 'Gcphase':
	            qbt2=key[2]
	        if len(error[1]) != 2:
	            sub_pauli=error[1].split(':')[0]
	            qbt1=int(error[1].split(':')[1].split(',')[0])
	            if len(error[1])==6:
	                qbt2=int(error[1].split(':')[1].split(',')[1])
	            else:
	                qbt2=None
	        else:
	            sub_pauli=error[1]
	        pauli=pauli[:qbt1]+sub_pauli[0]+pauli[(qbt1+1):]
	        if qbt2 is not None:
	            pauli=pauli[:qbt2]+sub_pauli[1]+pauli[(qbt2+1):]
	        err_gen = _lseg.LocalStimErrorgenLabel(error[0],[_stim.PauliString(pauli),])
	        error_index_list.append(tuple([err_gen,key]))
	        ideal_error_rates.append(lindblad_coeff[key][error])
	                            
	return error_index_list, ideal_error_rates


def create_circuit_design_matrix(circuit, model, model_parameter_indexing, error_free_model):
	"""
	circuit: Circuit

	model: Noise model we're creating the design matrix for

	model_parameter_indexing : a list where the elements are 2-element tuples. Each element consists of a error generator tuple (in Stim format)
								and a gate in the model. The index where this element appears defines the indexing of the created design matrix.

	error_free_model : A model with which to compute error-free circuit outcomes.
	"""
	num_qubits = len(circuit.line_labels)

	permutation_matrix, eoc_index_guide = build_permutation_matrix(circuit, model, model_parameter_indexing, include_spam=False)
	ideal_state = generate_ideal_state(error_free_model, circuit)
	# Related to the `alpha` factors?
	sensitivity_matrix = calculate_sensitivity_matrix(ideal_state, eoc_index_guide, num_qubits)
	
	design_matrix = _np.dot(sensitivity_matrix, permutation_matrix[:][:])
	return design_matrix

def create_design_matrix(circuit_list, model, model_parameter_indexing, error_free_model):
	design_matrices = []
	for i, circuit in enumerate(circuit_list):
		design_matrices.append(create_circuit_design_matrix(circuit, model, model_parameter_indexing, error_free_model))
	return _np.vstack(design_matrices)

def build_permutation_matrix(circuit, model, model_parameter_indexing, include_spam=True):
	"""
	todo
	"""
	errorgen_prop = _ep.ErrorGeneratorPropagator(model)
	model_error_dict = errorgen_prop.errorgen_transform_map(circuit, include_spam=include_spam)
	model_error_dict_gate_lbld = dict()	
	for error_gen in model_error_dict.keys():
	    gate_contributions = errorgen_prop.errorgen_gate_contributors(error_gen[0], circuit,error_gen[1], include_spam=include_spam)
	    for gate in gate_contributions:
	        temp_key = tuple([error_gen[0], gate])
	        if temp_key in model_error_dict_gate_lbld.keys():
	            model_error_dict_gate_lbld[temp_key].append(model_error_dict[error_gen])
	        else:
	            model_error_dict_gate_lbld[temp_key] = [model_error_dict[error_gen]]	
	model_idx_list = model_error_dict_gate_lbld.keys()
	eoc_idx_list = []
	for tup in model_error_dict.values():
	    if not tup[0] in eoc_idx_list:
	        eoc_idx_list.append(tup[0]) 	
	permutation_matrix = _np.zeros([len(eoc_idx_list), len(model_parameter_indexing)])
	for error in model_idx_list:
	    idx2 = model_parameter_indexing.index(error)
	    for eoc_error, phase in model_error_dict_gate_lbld[error]:
	        idx1 = eoc_idx_list.index(eoc_error)
	        permutation_matrix[idx1][idx2] += 1 / phase	
	return permutation_matrix, eoc_idx_list	

def generate_ideal_state(error_free_model, circ):
	"""
	A function that generates the ideal output state of a circuit.

	circ: A pygsti circuit object representing the circuit in question
	num_qubits: the number of qubits in the circuit
	gates: list of gates strings representing the gates in the circuit
	"""
	#qubit_labels =range(num_qubits)
	#availability = {'Gcphase':'all-permutations' }
	#pspec = pygsti.processors.QubitProcessorSpec(num_qubits, gates,    geometry=connectivity, qubit_labels=qubit_labels)
	#mdl_noise=pygsti.models.create_crosstalk_free_model(
	#            pspec,
	#            lindblad_parameterization='GLND',
	#            basis='pp'
	#        )
	sim = _MFFS(error_free_model)
	_, final_state = sim.compute_process_matrix(error_free_model, error_free_model.complete_circuit(circ), include_final_state=True)
	return final_state
 
def calculate_sensitivity_vector(ideal_state, error_gen_list, bit_string, num_qubits):
	"""
	A functions that calculates a vector q that transforms a circuit matrix,
	i.e a matrix that determins how the errors in a layer or gate based errormodel transform
	to an end of circuit error generat	
	ideal state: The input state after it has evolved through the circuit
	error_gen_list: a list of localstimerrorgens
	bit_string:  The bit string the measurement is being projected onto
	"""
	translate_vec = _np.zeros([len(error_gen_list), 1], dtype=_np.complex128)
	for idx,error in enumerate(error_gen_list):
	    error_proc_mat = _optools.create_elementary_errorgen_nqudit(error.errorgen_type,error.bel_to_strings(), basis_1q=_baseobjs.BuiltinBasis('PP', 4))	
	    translate_vec[idx] = _np.dot(make_bit_string_vector(bit_string), _np.dot(error_proc_mat, ideal_state))
	return translate_vec

def calculate_sensitivity_matrix(ideal_state, error_gen_list, num_qubits):
	trans_mat = _np.zeros([2**num_qubits,len(error_gen_list)])
	for idx in range(2**num_qubits):
	    trans_mat[idx][:] = calculate_sensitivity_vector(ideal_state,error_gen_list,int_to_bin(idx,num_qubits),num_qubits).T
	return trans_mat

def make_bit_string_vector(bitstring :str):
	"""
	A function that given a bitstring returns the
	vector representation of the associated superket
	bitstring the bitstring you want the vector of
	"""
	vector_dict = {'0' : _np.sqrt(1/2) * _np.array([1,0,0,1]),
	             '1' : _np.sqrt(1/2) *  _np.array([1,0,0,-1])}
	vector = vector_dict[bitstring[0]]
	for char in bitstring[1:]:
	    vector = _np.kron(vector,vector_dict[char])
	return vector

def int_to_bin(integer : int, qbts :int) ->str:
    bitstr=bin(integer)[2:]
    if len(bitstr) < qbts:
        bitstr='0'*(qbts-len(bitstr))+bitstr
    return bitstr