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
from pygsti.tools import errgenproptools as _egpt #import errgenproptools
from pygsti.tools import optools as _optools 
import stim as _stim
from pygsti.protocols import ModelFreeformSimulator as _MFFS
from pygsti import baseobjs as _baseobjs
import pygsti
from scipy import optimize



def build_model_parameter_indexing(lindblad_coeff, num_qubits):
	"""
	This function can now handle generic H+S error models with up to weight 2 errors
	"""
	error_index_list=[]
	ideal_error_rates=[]
	for key in lindblad_coeff.keys():
		for error in lindblad_coeff[key].keys():
			pauli=num_qubits*'I'
			qbt1=key[1]
			#check if 2 qubit gate
			if len(key) == 3: # 2 qubit gates have three parameters
				qbt2=key[2]
			if len(error[1]) > 2: #This indicates that the gate has some crosstalk associated with it and thus the paulis contain qbt lbls
				sub_pauli=error[1].split(':')[0]
				qbts_acted_on_list=error[1].split(':')[1].split(',')
				qbt_support=[int(x) for x in qbts_acted_on_list]
			else:
				sub_pauli=error[1]
				qbt_support=[qbt1]
				if len(key)==3:
					qbt_support.append(qbt2)
			for sub_idx,pauli_idx in enumerate(qbt_support):
				pauli=pauli[:pauli_idx]+sub_pauli[sub_idx]+pauli[(pauli_idx+1):]
			err_gen = _lseg.LocalStimErrorgenLabel(error[0],[_stim.PauliString(pauli),])
			error_index_list.append(tuple([err_gen,key]))
			ideal_error_rates.append(lindblad_coeff[key][error])
	                            
	return error_index_list, ideal_error_rates


def create_circuit_design_matrix(circuit, model, model_parameter_indexing, usePaulis=False,pauli_measurements=None,include_spam=False):
	"""
	circuit: Circuit

	model: Noise model we're creating the design matrix for

	model_parameter_indexing : a list where the elements are 2-element tuples. Each element consists of a error generator tuple (in Stim format)
								and a gate in the model. The index where this element appears defines the indexing of the created design matrix.

	error_free_model : A model with which to compute error-free circuit outcomes.
	"""
	num_qubits = len(circuit.line_labels)

	permutation_matrix, eoc_index_guide = build_permutation_matrix(circuit, model, model_parameter_indexing, include_spam=include_spam)
	ideal_state = generate_ideal_state(circuit)
	# Related to the `alpha` factors?

	if not usePaulis:
		sensitivity_matrix = calculate_sensitivity_matrix_probs(ideal_state, eoc_index_guide, num_qubits)
	else:
		sensitivity_matrix =calculate_sensitivity_matrix_paulis(ideal_state,eoc_index_guide,num_qubits,pauli_subset=pauli_measurements)
	
	design_matrix = _np.dot(sensitivity_matrix, permutation_matrix[:][:])
	return design_matrix

def create_design_matrix(circuit_list, model, model_parameter_indexing,usePaulis=False,include_spam=False):
	design_matrices = []
	for i, circuit in enumerate(circuit_list):
		design_matrices.append(create_circuit_design_matrix(circuit, model, model_parameter_indexing,usePaulis=usePaulis,include_spam=include_spam))
	return _np.vstack(design_matrices)

def create_design_matrix_list(circuit_list, model, model_parameter_indexing,usePaulis=False,pauli_measurements=None,include_spam=False):
	design_matrices = []
	for i, circuit in enumerate(circuit_list):
		dm =create_circuit_design_matrix(circuit, model, model_parameter_indexing,usePaulis=usePaulis,pauli_measurements=pauli_measurements,include_spam=include_spam)
		design_matrices.append(dm)
	return design_matrices

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
			permutation_matrix[idx1][idx2] +=  phase	
	return permutation_matrix, eoc_idx_list	

def generate_ideal_state(circ):
	"""
	A function that generates the ideal output state of a circuit.

	circ: A pygsti circuit object representing the circuit in question
	num_qubits: the number of qubits in the circuit
	gates: list of gates strings representing the gates in the circuit
	"""
	final_state=circ.convert_to_stim_tableau()
	#final_state_vector=circ_tableau.to_state_vector()
	#final_state=_stim.Tableau.from_state_vector(final_state_vector)
	#qubit_labels =range(num_qubits)
	#availability = {'Gcphase':'all-permutations' }
	#pspec = pygsti.processors.QubitProcessorSpec(num_qubits, gates,    geometry=connectivity, qubit_labels=qubit_labels)
	#mdl_noise=pygsti.models.create_crosstalk_free_model(
	#            pspec,
	#            lindblad_parameterization='GLND',
	#            basis='pp'
	#        )
	#sim = _MFFS(error_free_model)
	#_, final_state = sim.compute_process_matrix(error_free_model, error_free_model.complete_circuit(circ), include_final_state=True)
	return final_state
 
def calculate_sensitivity_vector_probs(ideal_state, error_gen_list, bit_string):
	"""
	A functions that calculates a vector q that transforms a circuit matrix,
	i.e a matrix that determins how the errors in a layer or gate based errormodel transform
	to an end of circuit error generat	
	ideal state: The input state after it has evolved through the circuit as a stim tabluea
	error_gen_list: a list of localstimerrorgens
	bit_string:  The bit string the measurement is being projected onto
	"""
	translate_vec = _np.zeros([len(error_gen_list), 1], dtype=_np.complex128)
	for idx,error in enumerate(error_gen_list):
		translate_vec[idx] = _egpt.stabilizer_probability_correction({error:1},ideal_state,bit_string)
	return translate_vec

def calculate_sensitivity_matrix_probs(ideal_state, error_gen_list, num_qubits):
	trans_mat = _np.zeros([2**num_qubits,len(error_gen_list)])
	for idx in range(2**num_qubits):
		trans_mat[idx][:] = calculate_sensitivity_vector_probs(ideal_state,error_gen_list,int_to_bin(idx,num_qubits)).T
	return trans_mat

def calculate_sensitivity_vector_paulis(ideal_state, error_gen_list, pauli):
	translate_vec = _np.zeros([len(error_gen_list), 1], dtype=_np.complex128)
	for idx,error in enumerate(error_gen_list):
		translate_vec[idx] = _egpt.alpha_pauli(error,ideal_state,pauli)
	return translate_vec

def calculate_sensitivity_matrix_paulis(ideal_state, errorgen_list, num_qubits, pauli_subset=None):
	if pauli_subset == None:
		trans_mat =_np.zeros([4**num_qubits,len(errorgen_list)])
		for idx in range(4**num_qubits):
			trans_mat[idx][:]=calculate_sensitivity_vector_paulis(ideal_state,errorgen_list,int_to_pauli(idx,num_qubits)).T
	else:
		trans_mat =_np.zeros([len(pauli_subset),len(errorgen_list)])
		for idx,pauli in enumerate(pauli_subset):
			trans_mat[idx][:]=calculate_sensitivity_vector_paulis(ideal_state,errorgen_list,pauli).T
	return trans_mat

def estimate_error_rates(design_matrices:list, probability_dict_list : list, measurement_list: list, error_indexing_list : list, error_bar_params=[], coherent_solver='inversion', stochastic_solver='nnls', error_bars='bootstrap')->_np.array:
	'''
	Inputs:

	design_matrices(list): a list of design matrices as output by create_design_matrix list where each index stands for a different circuit

	probability_dict_list (list): a 2d list of probabilities, the first index marks whether the set of indices are experimental (0) or ideal (1).
	The second element indexes which circuit the probability dictionary is for.  Indexes of the second list should follow the same order as design_matrices.
	All dictionaries should follow the form bitstirng->probability i.e '000' : .025

	measurement_list (list): A list of (currently) Z type pauli measurements that should be computed from the associated probabilities

	error_indexing_list (list): The list generated by build_model_parameter_indexing that stores the data of which error vector paramter is associated with
	which error

	error_bar_params (list): A list of parameters for calculating the error bars. If the error bar propagation method is set to 'bootstrap' then the list of values is
	[bootstrap_trials,shots] where bootstrap trials is the number of times the probability distribution is resampled and shots is the number of shots per circuit

	coherent_solver (str): Option for solving for the coherent error rates, currently limited to 'inversion' which solves for the coherent error rates by
	taking the pseudoinversion of the coherent specific design matrix and dot-producting it with the measured pauli outcomes

	stochastic_solver (str): Method for solving for the stochastic error rates. If set to nnls will solve by using non-linear least squares.  If it is
	set to inversion, the code will skip the method of dividing the error sectors and perform linear inversion in the full design matrix

	error_bars (str): Determine how to find the error bars for the error rate. If the method is set to bootstrap it will perform a non-paramtric bootstrap
	to find the error bars
	_____________________________________
	Returns:

	error_rates (np.array): array of estimated error rates for the data

	error_uncertainty (np.array): array of uncertainties for each error rate
	_____________________________________
	Description: Estimates the error rates for a set of design matrices and output probabilites.
	
	'''
	expectation_value_list=[]
	for idx,_ in enumerate(probability_dict_list[0]):
		expectation_value_list.append(_bulk_Ztype_pauli_exp(measurement_list,probability_dict_list[0][idx])-_bulk_Ztype_pauli_exp(measurement_list,probability_dict_list[1][idx]))
	observed_values=_np.hstack(expectation_value_list)
	design_matrix=_np.vstack(design_matrices)
	if coherent_solver=='inversion' and stochastic_solver=='inversion':
		error_rates= _np.linalg.dot(_np.linalg.pinv(design_matrix),observed_values)
	else:
		H_columns=[]
		S_columns=[]
		for col_no,error in enumerate(error_indexing_list):
			if error[0].errorgen_type=='H':
				H_columns.append(design_matrix.T[col_no])
			elif error[0].errorgen_type=='S':
				S_columns.append(design_matrix.T[col_no])
		H_submatrix=_np.zeros([len(H_columns),len(observed_values)])
		S_submatrix=_np.zeros([len(S_columns),len(observed_values)])
		for idx,column in enumerate(H_columns):
			H_submatrix[idx][:]=column
		for idx,column in enumerate(S_columns):
			S_submatrix[idx][:]=column
		
		#coherent error solver block
		if coherent_solver=='inversion':
			H_error_rates=_np.dot(_np.linalg.pinv(H_submatrix.T),observed_values)
		
		#stochastic error solver block
		if stochastic_solver=='nnls':
			S_error_rates,_=optimize.nnls(S_submatrix.T,observed_values)
			#print(S_error_rates)
		elif stochastic_solver=='inversion':
			S_error_rates=_np.linalg.dot(_np.linalg.pinv(S_submatrix.T),observed_values)

		error_rates=_np.zeros([len(error_indexing_list),])
		h_errors=0
		s_errors=0
		for idx,error in enumerate(error_indexing_list):
			if error[0].errorgen_type=='H':
				error_rates[idx]=H_error_rates[h_errors]
				h_errors+=1
			if error[0].errorgen_type=='S':
				error_rates[idx]=S_error_rates[s_errors]
				s_errors+=1
			#print((h_errors,s_errors))

	if error_bars==None:
		error_uncertainty=None

	elif error_bars=='bootstrap':
		bootstrap_trials=error_bar_params[0]
		shots=error_bar_params[1]
		bootstrapped_estimates=_np.zeros([bootstrap_trials,len(error_indexing_list)])
		for idx in range(bootstrap_trials):
			resampled_probs=[]
			for prob_dict in probability_dict_list[0]:
				resampled_probs.append(_resample_probability_distribution(prob_dict,shots))
			full_prob_list=[resampled_probs,probability_dict_list[1]]
			bootstrapped_estimates[idx][:],_=estimate_error_rates(design_matrices,full_prob_list,measurement_list,error_indexing_list,error_bar_params=None,coherent_solver=coherent_solver,stochastic_solver=stochastic_solver,error_bars=None)
		error_uncertainty=_np.std(bootstrapped_estimates,axis=0)

	return error_rates, error_uncertainty

def _resample_probability_distribution(prob_distribution : dict,shots : int)->dict:
	'''
    Inputs:
    outcomes_dict: bistring -> probabilities dictionary
    shots: number of samples to take
    Returns:
    freq_dict: bistring -> probabilities dictionary
    '''
	noiseless_outcomes=list(prob_distribution.values())
	for idx,value in enumerate(noiseless_outcomes):
		if value<0.:
			noiseless_outcomes[idx]=0.
	frequencies=_np.zeros(_np.shape(noiseless_outcomes))       
	frequencies=_np.random.multinomial(shots,noiseless_outcomes)/shots
	freq_dict=dict()
	for idx,key in enumerate(prob_distribution.keys()):
		freq_dict[key]=frequencies[idx]
	return freq_dict



def _calculate_Ztype_pauli_exp(pauli : str,probs : dict)->float:
    '''
    Inputs:
    pauli: str denoting the pauli to take the expectation value of
    probs: dict with bitsting keys and probability values
    Returns
    exp_value: Expectation value of the passed pauli
    '''
    exp_value=0.0
    indices=pauli.pauli_indices(included_paulis='Z')
    
    for bitstring in probs.keys():
        
        weight=1
        for idx in indices:
            if bitstring[0][idx]=='1':
                weight=weight*-1
        exp_value=exp_value+weight*probs[bitstring]
    return exp_value

def _bulk_Ztype_pauli_exp(paulis: list,probs : dict)->list:
    '''
    Inputs:
    paulis: list denoting the paulis to take the expectation value of
    probs: dict with bitsting keys and probability values
    Returns
    exp_value: Expectation values of the passed paulis
    '''
    expectations=_np.zeros([len(paulis),])
    for idx,pauli in enumerate(paulis):
        expectations[idx]=_calculate_Ztype_pauli_exp(pauli,probs)
    return expectations


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

def int_to_pauli(integer : int, qbts: int)->_stim.PauliString:
	pauli_dict={0 : 'I', 1:'X', 2:'Y',3:'Z'}
	pauli=''
	temp_int=integer
	while temp_int >0:
		pauli=pauli_dict[temp_int%4]+pauli
		temp_int=temp_int // 4
	if len(pauli) < qbts:
		pauli='I'*(qbts-len(pauli))+pauli
	return _stim.PauliString(pauli)

def rate_to_model(pspec,lindblad_coeff,rate_ests_dict):
	'''
	This works beaucse of how rate_ests_dict is calculated from linblad_coeff, 
	after any changes to build_model_parameter indexing this functions should be extensively retested.
	'''
	i = 0
	for key in lindblad_coeff.keys():
		for key2 in lindblad_coeff[key].keys():
			#print(key2, key)
			lindblad_coeff[key][key2] = rate_ests_dict[i]
			i += 1

	mdl_cloudnoise = pygsti.models.create_cloud_crosstalk_model(pspec,lindblad_error_coeffs=lindblad_coeff, errcomp_type="errorgens")
	return mdl_cloudnoise
