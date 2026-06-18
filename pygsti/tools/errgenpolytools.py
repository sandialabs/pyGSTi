"""
Tools for the construction of polynomials from the propagation of error generators through circuits.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations
import warnings
try:
    import stim
except ImportError:
    msg = "Stim is required for use of the error generator polynomial tools module, " \
          "and it does not appear to be installed. If you intend to use this module please update" \
          " your environment."
    warnings.warn(msg)

import numpy as _np
from itertools import chain, product
from math import factorial
from pygsti.baseobjs.polynomial import Polynomial as _Polynomial
import pygsti.tools.errgenproptools as _eprop
import pygsti.tools.slicetools as _slct
from pygsti.models.model import OpModel as _OpModel
from pygsti.models import (ExplicitOpModel as _ExplicitOpModel, ImplicitOpModel as _ImplicitOpModel, 
                          LocalNoiseModel as _LocalNoiseModel)
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as _LEEL 
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel as _GEEL
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE

from typing import Literal, Optional, Union, Callable, Iterable

# ----------- Error Generator to Polynomial Variable Utilities ---------- #

def error_generator_to_polynomial_variable_maps(errorgen_transform_map, return_reverse=False):
    """
    Helper function which returns a map from error generator label + layer index tuples to integers for use
    as variable indices for Polynomials.

    Parameters
    ----------
    errorgen_transform_map : dict
        Dictionary mapping tuples of LocalStimErrorgenLabels and circuit layer indices to 
        tuples of final error generators and phases. 
    
    return_reverse : bool, optional (default, False)
        Optional flag that returns the reverse variable to errorgen label mapping.

    Returns
    -------
        errorgen_to_var_map : dict
            Dictionary with keys that are tuples with pairs of LocalStimErrorgenLabels and layer indices,
            and values that are integers corresponding to a Polynomial variable index.

        var_to_errorgen_map : dict
            Returned if return_reverse == True, this is a dictionary with the keys and values of errorgen_to_var_map
            reversed.
    """
    errorgen_to_var_map = {key:i for i,key in enumerate(errorgen_transform_map.keys())}
    if return_reverse:
        var_to_errorgen_map = {i:key for i,key in enumerate(errorgen_transform_map.keys())}
        return errorgen_to_var_map, var_to_errorgen_map
    else:
        return errorgen_to_var_map

def error_generator_to_polynomial_variable_maps_by_gate(model, errorgen_var_map, circuit, include_spam=True, 
                                                        aggregate_shared_parameter_gates=False, return_reverse=False):
    """
    Function which takes an existing mapping from error generator circuit layer index
    pairs to integer variable indices and aggregates error generators that
    can be associated with the same gate into a single polynomial parameter.
    
    Parameters
    ----------
    errorgen_to_var_map : dict
        A dictionary whose keys are tuples of LocalStimErrorgenLabels and integer circuit layer indices
        and whose value is an integer corresponding to the corresponding variable index to use in constructed
        Polynomials.
        
    circuit : Circuit
        Circuit associated with input errorgen_var_map argument.
    
    include_spam : bool, optional (default True)
        If True include the spam circuit layers at the beginning and 
        end of the circuit.
    
    aggregate_shared_parameter_gates : bool, optional (default False)
        If True then an attempt then gates with matching model parameter indices (gpindices)
        are treated as the same and their error generators are aggregated into a single
        variable index.

    return_reverse : bool, optional (default False)
        If True return a map from variable indices to corresponding error generators.
    
    Returns
    -------
    errorgen_var_gate_aggregated_map : dict
        Dictionary with keys that are tuples with pairs of LocalStimErrorgenLabels and layer indices,
            and values that are integers corresponding to a Polynomial variable index.

        var_gate_aggregated_errorgen_map : dict
            Returned if return_reverse == True, this is a dictionary with keys that are Polynomial
            variable indices, and values that are lists of tuples with pairs of LocalStimErrorgenLabels and layer indices
            where these lists correspond to all those error generator parameters associated with the same gate's parameters.
    """
    
    if not isinstance(model, _OpModel):
            raise ValueError('This method does not work for non-OpModel models.')
    if isinstance(model, _ImplicitOpModel) and not isinstance(model, _LocalNoiseModel):
        raise ValueError('This method does not work for ImplicitModels that are not LocalNoiseModels.')

    if include_spam:
        circuit = model.complete_circuit(circuit)
    
    if not aggregate_shared_parameter_gates:
        #keys of errorgen_to_var map are tuples of LSE and integer layer indices.
        aggregated_error_generator_indices_by_gate = dict()
        for errorgen, layer_idx in errorgen_var_map:
            gate_contributors = errorgen_gate_contributors(model, errorgen, circuit, layer_idx, include_spam=include_spam)
            if len(gate_contributors) > 1:
                msg = f'Encountered more than 1 gate contributing to this error generator {errorgen}. ' \
                      'Support for aggregating variables by gate when there is more than 1 contributor is not currently supported.'
                raise RuntimeError(msg)
            gate = gate_contributors[0]
            if gate not in aggregated_error_generator_indices_by_gate:
                aggregated_error_generator_indices_by_gate[gate] = dict()
            if errorgen not in aggregated_error_generator_indices_by_gate[gate]:
                aggregated_error_generator_indices_by_gate[gate][errorgen] = []
            aggregated_error_generator_indices_by_gate[gate][errorgen].append(layer_idx)
    
        #loop through aggregated_error_generator_indices_by_gate, assign an index to each error generator equivalence class.
        #output map will still consist of maps from error generator layer index tuples, but the output index will be this joint index.
        errorgen_var_gate_aggregated_map = dict()
        var_idx = 0
        for errorgen_classes_by_gate in aggregated_error_generator_indices_by_gate.values():
            for class_lbl, layer_indices in errorgen_classes_by_gate.items():
                for layer_idx in layer_indices:
                    errorgen_var_gate_aggregated_map[(class_lbl, layer_idx)] = var_idx
                var_idx+=1

    #try to aggregate variable indices accross gates that share a set of parameters by checking whether two gates
    #share the same set of gpindices.
    else:
        aggregated_error_generator_indices_by_gate = dict()
        for errorgen, layer_idx in errorgen_var_map:
            gate_contributors, gate_contributor_operators = errorgen_gate_contributors(model, errorgen, circuit, layer_idx, return_operators=True, include_spam=include_spam)
            if len(gate_contributors) > 1:
                msg = f'Encountered more than 1 gate contributing to this error generator {errorgen}. ' \
                      'Support for aggregating variables by gate when there is more than 1 contributor is not currently supported.'
                raise RuntimeError(msg)
            gate = gate_contributors[0]
            gate_name = gate.name
            gate_sslbls = gate.sslbls
            gate_operator = gate_contributor_operators[0]
            gate_gpindices = gate_operator.gpindices
            #use the pair of gate_name and gpindices as the aggregator
            if isinstance(gate_gpindices, slice):
                gate_gpindices = _slct.indices(gate_gpindices)
            if isinstance(gate_gpindices, (list, _np.ndarray)):
                gate_gpindices = tuple(gate_gpindices)
            
            gate_name_model_indices_tup = (gate_name, gate_gpindices)
            
            if gate_name_model_indices_tup not in aggregated_error_generator_indices_by_gate:
                aggregated_error_generator_indices_by_gate[gate_name_model_indices_tup] = dict()
            
            #break the error generators into equivalence classes based on truncations, enforcing locality.
            try:
                truncated_errorgen = _truncate_lse_support(errorgen, gate_sslbls, validate_locality=True)
            except RuntimeError:
                msg = "An error generator with nonlocal support relative to the gate with which it is associated has been detected."
                msg += " Aggregation of error generators by gates and shared model parameters together is only supported models with local errors "
                msg += "(those with support restricted to the target qubits of the gate in question) at present."
                raise RuntimeError(msg)
            if truncated_errorgen not in aggregated_error_generator_indices_by_gate[gate_name_model_indices_tup]:
                aggregated_error_generator_indices_by_gate[gate_name_model_indices_tup][truncated_errorgen] = []
            #keep track of the original errorgen layer pair
            aggregated_error_generator_indices_by_gate[gate_name_model_indices_tup][truncated_errorgen].append((errorgen, layer_idx))
        
        #loop through aggregated_error_generator_indices_by_gate, assign an index to each error generator equivalence class.
        #output map will still consist of maps from error generator layer index tuples, but the output index will be this joint index.
        errorgen_var_gate_aggregated_map = dict()
        var_idx = 0
        for errorgen_classes_by_gate in aggregated_error_generator_indices_by_gate.values():
            for errorgen_layer_idx_pairs in errorgen_classes_by_gate.values():
                for pair in errorgen_layer_idx_pairs:
                    errorgen_var_gate_aggregated_map[pair] = var_idx
                var_idx+=1
        
    if return_reverse:
        var_gate_aggregated_errorgen_map = dict()
        for errorgen_layer_idx_pair, var_idx in errorgen_var_gate_aggregated_map.items():
            if var_idx not in var_gate_aggregated_errorgen_map:
                var_gate_aggregated_errorgen_map[var_idx] = []
            var_gate_aggregated_errorgen_map[var_idx].append(errorgen_layer_idx_pair)
        return errorgen_var_gate_aggregated_map, var_gate_aggregated_errorgen_map
    else:
        return errorgen_var_gate_aggregated_map

def _truncate_lse_support(errorgen, qubit_indices, validate_locality=True):
    """
    Helper function for truncating the support of a LocalStimErrorgenLabel to a subset of qubit indices
    with the option of checking for locality of the input error generator on these indices, raising
    an exception if that check fails.
    
    Parameters
    ----------
    errorgen : LocalStimErrorgenLabel
        Error generator label to truncate.
    
    qubit_indices : iterable of int
        Indices to truncate support to.
    
    validate_locality : bool, optional (default True)
        If True then the remaining Paulis outside the truncation
        are checked to confirm that they are all identities. If not
        then a RuntimeError exception is raised.
        
    Returns
    -------
    truncated_errorgen : LocalStimErrorgenLabel
        Error generator label with support truncated to the specified qubit indices.        
    """
    
    bels = errorgen.basis_element_labels
    assert len(bels[0]) >= len(qubit_indices), "More qubit indices specified than there are qubits for this LocalStimErrorgenLabel." 
    
    #casting a stim.PauliString to a list gives a list of integers in the range 0 to 3, corresponding to I, X, Y and Z respectively.
    #sign information is not part of this list (which is fine for basis element labels). stim.PauliStrings can be initialized
    #from a list of integers of this form.
    new_bels = []
    for bel in bels:
        pauli_index_list = list(bel)
        new_bel = stim.PauliString([pauli_index_list[qubit_idx] for qubit_idx in qubit_indices])
        new_bels.append(new_bel)
        if validate_locality:
            if not all([pauli_index==0 for qubit_idx, pauli_index in enumerate(pauli_index_list) if qubit_idx not in qubit_indices]):
                msg = 'Some paulis outside of specified qubit indices are not identities, violating requested locality constraint.'
                raise RuntimeError(msg)
    new_errorgen = _LSE(errorgen.errorgen_type, new_bels)
    return new_errorgen

def errorgen_gate_contributors(model, errorgen, circuit, layer_idx, include_spam=True, return_operators=False):
    """
    Walks through the gates in the specified circuit layer and query the parent 
    model to figure out which gates could have given rise to a particular error generator
    in a layer.
    
    Parameters
    ----------
    errorgen : `ElementaryErrorgenLabel`
        Error generator layer to find instance of.
        
    circuit : `Circuit`
        Circuit to identify potential gates in.
    
    layer_idx : int
        Index of circuit layer.
    
    include_spam : bool, optional (default True)
        If True include the spam circuit layers at the beginning and 
        end of the circuit.
    
    return_operators : bool, optional (default False)
        If True return the circuit layer operators for the gates identified.
    
    Returns
    -------
    label_list_for_errorgen : list of `Label`
        A list of gate labels contained within this circuit layer that could have
        contributed this error generator.   

    circuit_layer_operators : list of `ModelMembers` (returned when return_operators==True)
        Optional list of ModelMembers corresponding to the gate contributors identified.
    """
    
    if not isinstance(model, _OpModel):
        raise ValueError('This method does not work for non-OpModel models.')
    
    if include_spam:
        circuit = model.complete_circuit(circuit)
        
    assert layer_idx < len(circuit), f'layer_idx {layer_idx} is out of range for circuit with length {len(circuit)}'
    
    if isinstance(errorgen, _GEEL):
        errorgen = _LEEL.cast(errorgen, sslbls = model.state_space.qubit_labels)
    elif isinstance(errorgen, _LSE):
        errorgen = errorgen.to_local_eel()
    else:
        assert isinstance(errorgen, _LEEL), f'Unsupported `errorgen` type {type(errorgen)}.'
    
    circuit_layer = circuit.layer(layer_idx)

    if isinstance(model, _ExplicitOpModel):
        #check if this error generator is in the error generator coefficient dictionary for this layer, and if not return the empty dictionary.
        circuit_layer_operator = model.circuit_layer_operator(circuit_layer)
        layer_errorgen_coeff_dict = circuit_layer_operator.errorgen_coefficients(label_type='local')
        
        if errorgen in layer_errorgen_coeff_dict:
            label_list_for_errorgen = [circuit_layer]
            if return_operators:
                circuit_layer_operators = [circuit_layer_operator]
        else:
            label_list_for_errorgen = []
        
    elif isinstance(model, _ImplicitOpModel):
        #Loop through each label in this layer and ask for the circuit layer operator
        #for each. Then query this for the error generator coefficients associated
        #with that layer.
        #Note: This may not be 100% robust, I'm assuming there aren't any exotic layer rules
        #that would, e.g., add in totally new error generators when certain pairs of gates appear in a layer.
        label_list_for_errorgen = []
        circuit_layer_operators = []
        for lbl in circuit_layer:
            circuit_layer_operator = model.circuit_layer_operator(lbl)
            label_errorgen_coeff_dict = circuit_layer_operator.errorgen_coefficients(label_type='local')
            if errorgen in label_errorgen_coeff_dict:
                label_list_for_errorgen.append(lbl)
                circuit_layer_operators.append(circuit_layer_operator)    
    else:
        raise ValueError(f'Type of model {type(model)=} is not supported with this method.')
    
    if return_operators:
        return label_list_for_errorgen, circuit_layer_operators
    else:
        return label_list_for_errorgen

def construct_polynomial_parameter_vector_from_propagator(error_propagator, var_to_errorgen_map, circuit, include_spam=True):
    """
    Constructs a vector of polynomial variable parameters for use in evaluation of error generator Polynomial
    objects corresponding to the error generator rate dictionary associated with the input ErrorGeneratorPropagator's 
    model and the input circuit.
    
    Parameters
    ----------
    error_propagator : `ErrorGeneratorPropagator`
        `ErrorGeneratorPropagator` object to use for the construction of the error generator
        rates to map to Polynomial variable parameters.
        
    var_to_errorgen_map : dict
        A dictionary whose keys are Polynomial variable indices and whose values are tuples (or lists of tuples) 
        of LocalStimErrorgenLabels and integer circuit layer indices whose values corresponds these variable indices. 
        
    circuit: Circuit
        Circuit associated with input errorgen_var_map argument.
    
    include_spam : bool, optional (default True)
        If True include the spam circuit layers at the beginning and 
        end of the circuit.
        
    Returns
    -------
    poly_paramvec : np.ndarray
        A vector of polynomial parameter values to use in evaluation of Polynomial objects.
    """
    
    errorgen_layers = error_propagator.construct_errorgen_layers(circuit, len(circuit.line_labels), include_spam=include_spam)
    indexed_errorgen_layers = dict()
    for i, layer in enumerate(errorgen_layers):
        for errorgen, rate in layer.items():
            indexed_errorgen_layers[(errorgen,i)] = rate 
    
    num_poly_vars = len(var_to_errorgen_map) 
    poly_paramvec = _np.zeros(num_poly_vars)
    
    many_to_one = isinstance(next(iter(var_to_errorgen_map.values())), list)

    if many_to_one:
        for i in range(num_poly_vars):
            poly_paramvec[i] = indexed_errorgen_layers[var_to_errorgen_map[i][0]]
    else:
        for i in range(num_poly_vars):
            poly_paramvec[i] = indexed_errorgen_layers[var_to_errorgen_map[i]]
    
    return poly_paramvec

#---------------- Error Generator Polynomial Construction -------------------#

def magnus_symbolic_polynomial(errorgen_transform_maps, errorgen_to_var_map, magnus_order=1):
    """
    Function for computing the symbolic magnus approximation for the effective end-or-circuit error generator.
    
    Parameters
    ----------
    errorgen_transform_maps : list of dicts
        List of dictionaries mapping tuples of LocalStimErrorgenLabels and circuit layer indices to 
        tuples of final error generators and phases. 
        
    errorgen_to_var_map : dict
        A dictionary whose keys are tuples of LocalStimErrorgenLabels and integer circuit layer indices
        and whose value is an integer corresponding to the corresponding variable index to use in constructed
        Polynomials.
        
    magnus_order : int, optional (default 1)
        Order of the magnus approximation to apply.
    
    Returns
    -------
    combined_magnus_order_dict : dict
        A dictionary whose keys are LocalStimErrorgenLabels and whose values are Polynomial
        objects corresponding to the rates.
    """
    assert magnus_order == 1 or magnus_order == 2, "Magnus expansions up to second order are currently supported for symbolic computations."
    
    magnus_terms_by_order = []
    #start off with the first-order magnus case.
    #initialize a dictionary with all of the keys that may appear in the first order magnus expansion
    #values initialized with empty lists to accumulate contributions.
      
    output_errgen_iterables = [transform_map.values() for transform_map in errorgen_transform_maps]
    first_order_magnus_var_dict = {output_errgen_tup[0]: [] for output_errgen_tup in chain(*output_errgen_iterables)}
    first_order_magnus_coeff_dict = {key: [] for key in first_order_magnus_var_dict}

    #loop through each key of errorgen_transform_map, use the value to index into current_combined_coeff_lbls
    #and append the key's label and the phase value from the transformed value.
    for transform_map in errorgen_transform_maps:
        for key, val in transform_map.items():
            first_order_magnus_var_dict[val[0]].append((errorgen_to_var_map[key],))
            first_order_magnus_coeff_dict[val[0]].append(val[1])

    #for each output error generator construct the corresponding polynomials.
    first_order_magnus_dict = dict()
    max_num_vars = len(errorgen_to_var_map)
    for (key, variables), coefficients in zip(first_order_magnus_var_dict.items(), first_order_magnus_coeff_dict.values()):
        first_order_magnus_dict[key] = _Polynomial.from_variable_and_coefficient_lists(variables, coefficients, max_num_vars)
        
    if magnus_order == 1:
        return first_order_magnus_dict
    
    magnus_terms_by_order.append(first_order_magnus_dict)
    
    if magnus_order == 2:
        second_order_magnus_dict = _second_order_magnus_term_symbolic_polynomial(errorgen_transform_maps, errorgen_to_var_map)
        magnus_terms_by_order.append(second_order_magnus_dict)
        
    #loop through the magnus terms by order and initialize a dictionary to accumulate results.
    combined_magnus_order_dict_keys = {key: None for key in chain(*magnus_terms_by_order)}
    combined_magnus_order_dict = {errorgen: _Polynomial({}, max_num_vars=len(errorgen_to_var_map)) for errorgen in combined_magnus_order_dict_keys}

    # Accumulate the coefficients contributing to each term.    
    for order_dict in magnus_terms_by_order:
        for errorgen, poly in order_dict.items():
            combined_magnus_order_dict[errorgen] += poly
    
    return combined_magnus_order_dict    

def _second_order_magnus_term_symbolic_polynomial(errorgen_transform_maps, errorgen_to_var_map, identity=None):
    """
    Helper function for computing the second-order correction term in the
    magnus expansion.

    (1/2)\sum_{t1=1}^n \sum_{t2=1}^{t1-1} [A(t1), A(t2)]

    Parameters:
    ----------
    errorgen_transform_maps : list of dicts
        List of dictionaries mapping tuples of LocalStimErrorgenLabels and circuit layer indices to 
        tuples of final error generators and phases. 
        
    errorgen_to_var_map : dict
        A dictionary whose keys are tuples of LocalStimErrorgenLabels and integer circuit layer indices
        and whose value is an integer corresponding to the corresponding variable index to use in constructed
        Polynomials.

    identity : stim.PauliString, optional (default None)
        An optional stim.PauliString to use for comparisons to the identity.
        Passing in this kwarg isn't necessary, but can allow for reduced 
        stim.PauliString creation when calling this function many times for
        improved efficiency.

    Returns
    -------
    TBD
    """
        
    errorgen_pairs = []
    for i in range(len(errorgen_transform_maps)):
        for j in range(i):
            errorgen_pairs.append((errorgen_transform_maps[i], errorgen_transform_maps[j]))
    
    # precompute an identity string for comparisons in commutator calculations if one is not provided.
    if identity is None and errorgen_transform_maps:
        for layer in errorgen_transform_maps:
            if layer:
                identity = stim.PauliString('I'*len(next(iter(layer))[0].basis_element_labels[0]))
                break
    
    # compute second-order BCH correction for each pair of error generators in the
    # errorgen_pairs list.
    commuted_errgen_poly_dicts = []
    for errorgen_pair in errorgen_pairs:
        pairwise_comm_errgen_poly_dict = _error_generator_layer_pairwise_commutator_symbolic_polynomial(errorgen_pair[0], errorgen_pair[1], errorgen_to_var_map, 
                                                                                                        addl_weight=0.5, identity=identity)
        commuted_errgen_poly_dicts.append(pairwise_comm_errgen_poly_dict)
    
    # loop through all of the elements of commuted_errgen_poly_dicts and instantiate a dictionary with the requisite keys.
    second_order_comm_dict_keys = {errorgen: None for poly_dict in commuted_errgen_poly_dicts for errorgen in poly_dict}
    second_order_comm_dict = {errorgen: _Polynomial({}, max_num_vars=len(errorgen_to_var_map)) for errorgen in second_order_comm_dict_keys}
    
    # Accumulate the coefficients contributing to each term.    
    for poly_dict in commuted_errgen_poly_dicts:
        for errorgen, poly in poly_dict.items():
            second_order_comm_dict[errorgen] += poly
    
    return second_order_comm_dict

def _error_generator_layer_pairwise_commutator_symbolic_polynomial(errorgen_layer_1, errorgen_layer_2, errorgen_to_var_map, addl_weight=1.0, identity=None):
    """
    Helper function for computing the pairwise commutator of two error generator layers symbolically, i.e. returning a data 
    structure which expresses the rates as polynomials in the original generators. 
    
    Parameters
    ----------
    errorgen_layer_1 : dict
        A dictionary whose keys are tuples of LocalStimErrorgenLabels and integers, and whose values are tuples of 
        LocalStimErrorgenLabels and floats, corresponding to a layer-by-layer mapping from input error generators
        to their final propagated values and phases. As returned by the method `ErrorGeneratorPropagator.errorgen_transform_maps`.
     
    errorgen_layer_2 : dict
        See above.
        
    errorgen_to_var_map : dict
        A dictionary whose keys are tuples of LocalStimErrorgenLabels and integer circuit layer indices
        and whose value is an integer corresponding to the corresponding variable index to use in constructed
        Polynomials.
        
    addl_weight : float
        An additional weight to add to the coefficients of the returned commutator polynomials.
        
    identity : stim.PauliString
        An optional stim.PauliString to use for comparisons to the identity.
        Passing in this kwarg isn't necessary, but can allow for reduced 
        stim.PauliString creation when calling this function many times for
        improved efficiency.
        
    Returns
    -------
    commuted_errgen_list : list of LocalStimErrorgenLabels
        List of error generator labels corresponding to the results of the 
        application of the pairwise commutators.
    
    coeff_list : list of tuples
        A list of three element tuples. The first two elements correspond to initial
        error generators whose rates should be multiplied to give a corresponding
        coefficient for the commutator output. The third term gives an addititonal overall
        phase and scale for this coefficient.
    """    
    commuted_errgen_list = []
    var_list = []
    coeff_list = []
    
    for initial_error1, final_error1 in errorgen_layer_1.items():
        for initial_error2, final_error2 in errorgen_layer_2.items():
            # get the list of error generator labels
            init_weight = addl_weight*final_error1[1]*final_error2[1]
            commuted_errgen_sublist = _eprop.error_generator_commutator(final_error1[0], final_error2[0], 
                                                                 identity=identity)
            for error_tup in commuted_errgen_sublist:
                commuted_errgen_list.append(error_tup[0])
                var_list.append((errorgen_to_var_map[initial_error1], errorgen_to_var_map[initial_error2]))
                coeff_list.append(init_weight*error_tup[1])
    
    #accumulate coefficients by output error generator and construct a combined Polynomial representation for the weight.
    # loop through all of the elements of commuted_errorgen_list and instantiate a dictionary with the requisite keys.
    commuted_errorgen_var_dict = {errorgen: [] for errorgen in commuted_errgen_list}
    commuted_errorgen_coeff_dict = {errorgen: [] for errorgen in commuted_errorgen_var_dict}
    # Accumulate the coefficients contributing to each term.    
    for errorgen, var, coeff in zip(commuted_errgen_list, var_list, coeff_list):
        commuted_errorgen_var_dict[errorgen].append(var)
        commuted_errorgen_coeff_dict[errorgen].append(coeff)
    
    max_num_vars = len(errorgen_to_var_map)
    commuted_errorgen_poly_dict = dict()
    for (key, variables), coefficients in zip(commuted_errorgen_var_dict.items(), commuted_errorgen_coeff_dict.values()):
        commuted_errorgen_poly_dict[key] = _Polynomial.from_variable_and_coefficient_lists(variables, coefficients, max_num_vars)    
    
    return commuted_errorgen_poly_dict

def error_generator_taylor_expansion_symbolic_polynomial(errorgen_dict, errorgen_to_var_map, order=1):
    """
    Compute the nth-order taylor expansion for the exponentiation of the error generator described by the input
    error generator dictionary. (Excluding the zeroth-order identity).
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `LocalStimErrorgenLabel` and whose values are `Polynomials` corresponding
        rates.
        
    errorgen_to_var_map : dict
        A dictionary whose keys are tuples of LocalStimErrorgenLabels and integer circuit layer indices
        and whose value is an integer corresponding to the corresponding variable index to use in constructed
        Polynomials.
    
    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.
    
    Returns
    -------
    list of dictionaries
        List of dictionaries whose keys are error generator labels and whose values are rates (including
        whatever scaling comes from order of taylor expansion). Each list corresponds to an order
        of the taylor expansion.
    """
    assert order == 1 or order == 2, "First and second-order symbolic taylor series approximations are currently supported."   
 
    if order == 1:
        return [errorgen_dict] #TODO, check if this should be a copy
        
    taylor_order_terms = [dict() for _ in range(order)]
    taylor_order_terms[0] = errorgen_dict
    
    if order > 1:
        # The order of the approximation determines the combinations of error generators
        # which need to be composed. (given by cartesian products of labels in errorgen_dict).
        labels_by_order = [list(product(errorgen_dict.keys(), repeat = i+1)) for i in range(1,order)]
        # Get a similar structure for the corresponding rates
        coeffs_by_order = [list(product(errorgen_dict.values(), repeat = i+1)) for i in range(1,order)]
        
        for current_order, (current_order_labels, current_order_coeffs) in enumerate(zip(labels_by_order, coeffs_by_order), start=2):
            order_scale = 1/factorial(current_order)
            composition_errgen_labels = []
            composition_errgen_coeffs = []
            rate_tup = tuple([1]*current_order)
            for label_tup, coeff_tup in zip(current_order_labels, current_order_coeffs):
                res = _eprop.iterative_error_generator_composition(label_tup, rate_tup)
                if len(res)>0: #hack
                    composed_labels, composed_rates = zip(*res)
                    composition_errgen_labels.extend(composed_labels)
                    #get the product of the coefficient polynomials in coeff_tup
                    composition_coeff_poly_product = _Polynomial.product(coeff_tup).scalar_mult(order_scale)
                    #this composite polynomial needs to have the rate appropriately scaled according to additional rate from the composition of the labels.
                    for rate in composed_rates:
                        composition_errgen_coeffs.append(composition_coeff_poly_product.scalar_mult(rate))
                
            # aggregate together any overlapping terms into a single dictionary
            composition_results_dict_keys = {errorgen: None for errorgen in composition_errgen_labels}
            max_num_vars = len(errorgen_to_var_map)
            composition_results_dict = {errorgen: _Polynomial({}, max_num_vars=max_num_vars) for errorgen in composition_results_dict_keys}
            # Accumulate the coefficients contributing to each term.    
            for errorgen, coeff_poly in zip(composition_errgen_labels, composition_errgen_coeffs):
                composition_results_dict[errorgen] += coeff_poly
            
            taylor_order_terms[current_order-1] = composition_results_dict

    return taylor_order_terms

def stabilizer_probability_correction_symbolic_polynomial(errorgen_dict, errorgen_to_var_map, tableau, desired_bitstring, order=1):
    """
    Compute the kth-order correction to the probability of the specified bit string.
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `LocalStimErrorgenLabel` and whose values are corresponding
        rates as polynomials in the original error generator rates.
    
    errorgen_to_var_map : dict
        A dictionary whose keys are tuples of LocalStimErrorgenLabels and integer circuit layer indices
        and whose value is an integer corresponding to the corresponding variable index to use in constructed
        Polynomials.
    
    tableau : stim.Tableau
        Stim tableau corresponding to a particular stabilizer state being measured.
        
    desired_bitstring : str
        String of 0's and 1's corresponding to the output bitstring being measured.

    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.


    Returns
    -------
    correction : Polynomial
        Polynomial corresponding to the correction to the output probability for the
        desired bitstring induced by the error generator (to specified order).
    """    
    correction_polys = bulk_stabilizer_probability_correction_symbolic_polynomial(errorgen_dict, errorgen_to_var_map, 
                                                                                  tableau, [desired_bitstring], order=order)
    return correction_polys[0]

def bulk_stabilizer_probability_correction_symbolic_polynomial(errorgen_dict, errorgen_to_var_map, tableau, desired_bitstrings, order=1):
    """
    Compute the kth-order correction to the probability of the specified bit string.
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `LocalStimErrorgenLabel` and whose values are corresponding
        rates as polynomials in the original error generator rates.
    
    errorgen_to_var_map : dict
        A dictionary whose keys are tuples of LocalStimErrorgenLabels and integer circuit layer indices
        and whose value is an integer corresponding to the corresponding variable index to use in constructed
        Polynomials.
    
    tableau : stim.Tableau
        Stim tableau corresponding to a particular stabilizer state being measured.
        
    desired_bitstrings : iterable of str
        iterable of strings of 0's and 1's corresponding to the output bitstrings being measured.

    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.

    Returns
    -------
    correction : Polynomial
        Polynomial corresponding to the correction to the output probability for the
        desired bitstring induced by the error generator (to specified order).
    """    
    num_random = _eprop.random_support(tableau)
    if num_random > 2148:
        raise RuntimeError('Number of random bits is greater than 1074, magnitude of probability scale will underflow!')
    scale = 1/2**(num_random) 
    
    taylor_expansion = error_generator_taylor_expansion_symbolic_polynomial(errorgen_dict, errorgen_to_var_map, order)
    combined_taylor_dict = _combined_taylor_expansion_polynomial(taylor_expansion, errorgen_to_var_map)
    complete_taylor_errgen_labels = list(combined_taylor_dict.keys())
    
    #can now do the correction computation in a single-shot.
    alphas_by_bitstring = _eprop.bulk_alpha(complete_taylor_errgen_labels, tableau, desired_bitstrings)

    #use the nonzero alphas to filter 
    nonzero_alpha_idxs_by_bitstring = [row.nonzero()[0] for row in alphas_by_bitstring]
    nonzero_alphas_by_bitstring =  [alpha_row[nonzero_alpha_idxs] for alpha_row, nonzero_alpha_idxs in 
                                    zip(alphas_by_bitstring, nonzero_alpha_idxs_by_bitstring)]
    nonzero_alpha_labels_by_bitstring = [[complete_taylor_errgen_labels[idx] for idx in nonzero_alpha_idxs] 
                                         for nonzero_alpha_idxs in  nonzero_alpha_idxs_by_bitstring]

    #gather the coefficients corresponding to each of the labels with nonzero alpha and sum this into a
    #single polynomial. Take care of the scale factor from the number of random variables at the same time.
    correction_polys = []
    for nonzero_alpha_labels, nonzero_alpha in zip(nonzero_alpha_labels_by_bitstring, nonzero_alphas_by_bitstring):
        nonzero_alpha_coeffs = [combined_taylor_dict[lbl].scalar_mult(alpha*scale) for lbl, alpha in zip(nonzero_alpha_labels, nonzero_alpha)]
        correction_polys.append(_Polynomial.sum(nonzero_alpha_coeffs))
    
    return correction_polys

def stabilizer_pauli_expectation_correction_symbolic_polynomial(errorgen_dict, errorgen_to_var_map, tableau, pauli, order = 1):
    """
    Compute the kth-order correction to the expectation value of the specified pauli.
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `LocalStimErrorgenLabel` and whose values are corresponding
        rates.
        
    errorgen_to_var_map : dict
        A dictionary whose keys are tuples of LocalStimErrorgenLabels and integer circuit layer indices
        and whose value is an integer corresponding to the corresponding variable index to use in constructed
        Polynomials.
    
    tableau : stim.Tableau
        Stim tableau corresponding to a particular stabilizer state being measured.
        
    pauli : stim.PauliString
        Pauli operator to compute expectation value correction for.

    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.
    
    Returns
    -------
    correction : Polynomial
        Polynomial corresponding to the correction to the output Pauli expectation value for the
        desired Pauli induced by the error generator (to specified order).
    """
    correction_polys = bulk_stabilizer_pauli_expectation_correction_symbolic_polynomial(errorgen_dict, errorgen_to_var_map, 
                                                                                        tableau, [pauli], order = order)
    return correction_polys[0]


def bulk_stabilizer_pauli_expectation_correction_symbolic_polynomial(errorgen_dict, errorgen_to_var_map, tableau, paulis, order = 1):
    """
    Compute the kth-order correction to the expectation value of the specified pauli.
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `LocalStimErrorgenLabel` and whose values are corresponding
        rates.
        
    errorgen_to_var_map : dict
        A dictionary whose keys are tuples of LocalStimErrorgenLabels and integer circuit layer indices
        and whose value is an integer corresponding to the corresponding variable index to use in constructed
        Polynomials.
    
    tableau : stim.Tableau
        Stim tableau corresponding to a particular stabilizer state being measured.
        
    paulis : iterable of stim.PauliString
        Pauli operator to compute expectation value correction for.

    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.
    
    Returns
    -------
    correction : Polynomial
        Polynomial corresponding to the correction to the output Pauli expectation value for the
        desired Pauli induced by the error generator (to specified order).
    """
    #accumulate the terms across orders (with short circuit logic for order 1 to save time):
    taylor_expansion = error_generator_taylor_expansion_symbolic_polynomial(errorgen_dict, errorgen_to_var_map, order)
    combined_taylor_dict = _combined_taylor_expansion_polynomial(taylor_expansion, errorgen_to_var_map)
    complete_taylor_errgen_labels = list(combined_taylor_dict.keys())

    #can now do the correction computation in a single-shot.
    alphas_by_pauli = _eprop.bulk_alpha_pauli(complete_taylor_errgen_labels, tableau, paulis)

    #use the nonzero alphas to filter 
    nonzero_alpha_idxs_by_pauli = [row.nonzero()[0] for row in alphas_by_pauli]
    nonzero_alphas_by_pauli =  [alpha_row[nonzero_alpha_idxs] for alpha_row, nonzero_alpha_idxs in 
                                    zip(alphas_by_pauli, nonzero_alpha_idxs_by_pauli)]
    nonzero_alpha_labels_by_pauli = [[complete_taylor_errgen_labels[idx] for idx in nonzero_alpha_idxs] 
                                         for nonzero_alpha_idxs in  nonzero_alpha_idxs_by_pauli]

    #gather the coefficients corresponding to each of the labels with nonzero alpha and sum this into a
    #single polynomial. Take care of the scale factor from the number of random variables at the same time.
    correction_polys = []
    for nonzero_alpha_labels, nonzero_alpha in zip(nonzero_alpha_labels_by_pauli, nonzero_alphas_by_pauli):
        nonzero_alpha_coeffs = [combined_taylor_dict[lbl].scalar_mult(alpha) for lbl, alpha in zip(nonzero_alpha_labels, nonzero_alpha)]
        correction_polys.append(_Polynomial.sum(nonzero_alpha_coeffs))
    
    return correction_polys

#helper function for forming the combined taylor-expansion polynomials.
def _combined_taylor_expansion_polynomial(taylor_expansion, errorgen_to_var_map):
    """
    Helper function for combining the polynomial dictionaries for each order of the taylor
    expansion which combines terms across orders.

    Parameters
    ----------
    taylor_expansion : list of dicts
        Dictionaries, one per order, with keys that are LocalStimErrorgenLabels and values that are 
        Polynomials corresponding to the rate.

    errorgen_to_var_map : dict
        A dictionary whose keys are tuples of LocalStimErrorgenLabels and integer circuit layer indices
        and whose value is an integer corresponding to the corresponding variable index to use in constructed
        Polynomials.

    Returns
    -------
    combined_taylor_dict
        Dictionary with keys that are LocalStimErrorgenLabels and values that are Polynomials
        corresponding to the combined rate across Taylor expansion orders.
    """
    #accumulate the terms across orders (with short circuit logic for order 1 to save time):
    if len(taylor_expansion) == 1:
        combined_taylor_dict = taylor_expansion[0]
    else:
        # Accumulate all of the dictionaries in taylor expansion into a single one, summing overlapping terms.
        combined_taylor_dict_keys = {key: None for order_dict in taylor_expansion for key in order_dict}
        max_num_vars = len(errorgen_to_var_map)
        combined_taylor_dict = {key: _Polynomial({}, max_num_vars=max_num_vars) for key in combined_taylor_dict_keys}
        for order_dict in taylor_expansion:
            for errorgen, coeff_poly in order_dict.items():
                combined_taylor_dict[errorgen] += coeff_poly
    return combined_taylor_dict