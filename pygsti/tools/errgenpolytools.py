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
    errorgen_var_gate_aggregated_map : 
    """
    
    if not isinstance(model, _OpModel):
            raise ValueError('This method does not work for non-OpModel models.')
    if isinstance(model, _ImplicitOpModel) and not isinstance(model, _LocalNoiseModel):
        raise ValueError('This method does not work for ImplicitModels that are not LocalNoiseModels.')

    if include_spam:
        circuit = model.complete_circuit(circuit)
    
    new_keys = []
    if not aggregate_shared_parameter_gates:
        #keys of errorgen_to_var map are tuples of LSE and integer layer indices.
        aggregated_error_generator_indices_by_gate = dict()
        for errorgen, layer_idx in errorgen_var_map:
            gate_contributors = errorgen_gate_contributors(model, errorgen, circuit, layer_idx)
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
            gate_contributors, gate_contributor_operators = errorgen_gate_contributors(model, errorgen, circuit, layer_idx, return_operators=True)
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
            #print(errorgen)
            #print(errorgen.basis_element_labels[0])
            #print(_truncate_lse_support(errorgen, gate_sslbls))
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
        
        #print(aggregated_error_generator_indices_by_gate)
        
        #loop through aggregated_error_generator_indices_by_gate, assign an index to each error generator equivalence class.
        #output map will still consist of maps from error generator layer index tuples, but the output index will be this joint index.
        errorgen_var_gate_aggregated_map = dict()
        var_idx = 0
        for errorgen_classes_by_gate in aggregated_error_generator_indices_by_gate.values():
            #print(errorgen_classes_by_gate)
            for errorgen_layer_idx_pairs in errorgen_classes_by_gate.values():
                #print(errorgen_layer_idx_pairs)
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
    #print(f'{circuit_layer=}')
    #print(f'{circuit.layer_label(layer_idx)=}')
    if isinstance(model, _ExplicitOpModel):
        #check if this error generator is in the error generator coefficient dictionary for this layer, and if not return the empty dictionary.
        circuit_layer_operator = model.circuit_layer_operator(circuit_layer)
        layer_errorgen_coeff_dict = circuit_layer_operator.errorgen_coefficients(label_type='local')
        
        if errorgen in layer_errorgen_coeff_dict:
            label_list_for_errorgen = [circuit.layer_label(layer_idx)]
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

def error_generator_symbolic_polynomial(errorgen_transform_map, errorgen_to_var_map):
    """
    Function for constructing an equivalent polynomial representation of an error generator
    from an input error generator transform map which is a dictionary mapping input error
    generator rates to final output generators along with their sign corrections. As returned,
    for example, by `ErrorGeneratorPropagator.errorgen_transform_maps` or 
    `ErrorGeneratorPropagator.errorgen_transform_map`.

    Parameters
    ----------
    errorgen_transform_map : dict
        Dictionary mapping tuples of LocalStimErrorgenLabels and circuit layer indices to 
        tuples of final error generators and phases. 

    errorgen_to_var_map : dict
        A dictionary whose keys are tuples of LocalStimErrorgenLabels and integer circuit layer indices
        and whose value is an integer corresponding to the corresponding variable index to use in constructed
        Polynomials.

    Returns
    -------
    errorgen_poly : dict
        A dictionary with keys that are LocalStimErrorgenLabels corresponding to output error generator
        labels from errorgen_transform_map and values corresponding to Polynomial objects for the rates
        of these error generators (as a function of the input error generators).
    """


    var_dict = {output_errgen_tup[0]: [] for output_errgen_tup in errorgen_transform_map.values()}
    coeff_dict = {key: [] for key in var_dict}
    #loop through each key of errorgen_transform_map, use the value to index into current_combined_coeff_lbls
    #and append the key's label and the phase value from the transformed value.
    for key, val in errorgen_transform_map.items():
        var_dict[val[0]].append((errorgen_to_var_map[key],))
        coeff_dict[val[0]].append(val[1])

    #for each output error generator construct the corresponding polynomials.
    errorgen_poly = dict()
    max_num_vars = len(errorgen_to_var_map)
    for (key, variables), coefficients in zip(var_dict.items(), coeff_dict.values()):
        errorgen_poly[key] = _Polynomial.from_variable_and_coefficient_lists(variables, coefficients, max_num_vars)

    return errorgen_poly


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
    
    is_polynomial_input = False
    for layer in errorgen_transform_maps:
        if layer and isinstance(next(iter(layer)), _LSE):
            is_polynomial_input = True
            max_num_vars = next(iter(layer.values())).max_num_vars
            break
    
    if is_polynomial_input:
        magnus_terms_by_order = []
        first_order_magnus_dict = {key: None for key in chain(*errorgen_transform_maps)}
        first_order_magnus_dict = {key: _Polynomial({}, max_num_vars=max_num_vars) for key in first_order_magnus_dict}
        
        errgen_iterables = [layer.items() for layer in errorgen_transform_maps]
        for errorgen, poly in chain(*errgen_iterables):
            first_order_magnus_dict[errorgen]+=poly
    
    else: #input is an error generator transform map.
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
        max_num_vars = max(errorgen_to_var_map.values()) + 1
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
    combined_magnus_order_dict = {errorgen: _Polynomial({}, max_num_vars=max(errorgen_to_var_map.values())+1) for errorgen in combined_magnus_order_dict_keys}

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
    
    is_polynomial_input = False
    for layer in errorgen_transform_maps:
        if layer and isinstance(next(iter(layer)), _LSE):
            is_polynomial_input = True
            max_num_vars = next(iter(layer.values())).max_num_vars
            break
    
    # precompute an identity string for comparisons in commutator calculations if one is not provided.
    if identity is None and errorgen_transform_maps:
        for layer in errorgen_transform_maps:
            if layer:
                if is_polynomial_input:
                    identity = stim.PauliString('I'*len(next(iter(layer)).basis_element_labels[0]))
                else:
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
    second_order_comm_dict = {errorgen: _Polynomial({}, max_num_vars=max(errorgen_to_var_map.values())+1) for errorgen in second_order_comm_dict_keys}
    
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
    is_polynomial_input = False
    for key in errorgen_layer_1:
        if key and isinstance(key, _LSE):
            is_polynomial_input = True
            max_num_vars = errorgen_layer_1[key].max_num_vars
            break
    
    if is_polynomial_input:
        commuted_errgen_list = []
        coeff_poly_list = []
        for error1, poly1 in errorgen_layer_1.items():
            for error2, poly2 in errorgen_layer_2.items():
                # get the list of error generator labels
                init_coeff_poly = poly1*poly2
                init_coeff_poly.scalar_mult(addl_weight)
                
                commuted_errgen_sublist = _eprop.error_generator_commutator(error1, error2, identity=identity)
                for error_tup in commuted_errgen_sublist:
                    commuted_errgen_list.append(error_tup[0])
                    coeff_poly_list.append(init_coeff_poly.scalar_mult(error_tup[1]))
        #accumulate coefficients by output error generator and construct a combined Polynomial representation for the weight.
        commuted_errorgen_poly_dict = {errorgen: None for errorgen in commuted_errgen_list}
        commuted_errorgen_poly_dict = {errorgen: _Polynomial({}, max_num_vars=max_num_vars) for errorgen in commuted_errorgen_poly_dict}
        
        # Accumulate the coefficients contributing to each term.    
        for errorgen, poly in zip(commuted_errgen_list, coeff_poly_list):
            commuted_errorgen_poly_dict[errorgen]+= poly
    
    else:
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

        max_num_vars = max(errorgen_to_var_map.values())+1
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
                composed_labels, composed_rates = zip(*_eprop.iterative_error_generator_composition(label_tup, rate_tup))
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

# ----------------- Cumulant Polynomial ------------------------#

def cumulant_expansion_symbolic_polynomial(errorgen_transform_maps, errorgen_to_var_map, cov_to_var_map, cumulant_order=2):
    """
    Function for computing the nth-order cumulant expansion for a set of error generator layers.

    Parameters
    ----------
    errorgen_transform_maps : list of dicts
        List of dictionaries mapping tuples of LocalStimErrorgenLabels and circuit layer indices to 
        tuples of final error generators and phases. 

    errorgen_to_var_map : dict
        A dictionary whose keys are tuples of LocalStimErrorgenLabels and integer circuit layer indices
        and whose value is an integer corresponding to the corresponding variable index to use in constructed
        Polynomials.
        
    cov_to_var_map : dict
        A dictionary whose keys are five element tuples, with the first element being a LocalStimErrorgenLabel,
        the second a gate Label object. The second set of two elements are of the same type as the first two. 
        The last argument is the difference in circuit times. These are equivalent to the arguments of most CovarianceFunction
        ModelMembers.
    
    cumulant_order : int, optional (default 2)
        Order of the cumulant expansion to apply. At present only the first-order cumulant expansion is supported.
        
    Returns
    -------
    nm_error_generator_polys : dict
        A list of dictionaries, one for each error generator layer in errorgen_transform_maps, with keys that
        are LocalStimErrorgenLabels and rates given by Polynomials in the original error generator rates and
        covariance function values at the selected order of the cumulant expansion.
    """
    #predefine the identity pauli to reduce number of instantiations.
    for layer in errorgen_transform_maps:
        if layer:
            identity = stim.PauliString('I'*len(next(iter(layer))[0].basis_element_labels[0]))
            break
    else:
        identity = None
        
    nm_error_generator_polys = []
    #TODO: See if I can't do this in a way that reuses computations across the nonmarkovian generators.
    for i in range(1, len(errorgen_transform_maps)+1):
        current_errorgen_transform_maps = errorgen_transform_maps[:i]
        nm_error_generator_polys.append(nonmarkovian_generator_symbolic_polynomial(current_errorgen_transform_maps, errorgen_to_var_map, cov_to_var_map,
                                                                                   cumulant_order, identity))
    return nm_error_generator_polys

def nonmarkovian_generator_symbolic_polynomial(errorgen_transform_maps, errorgen_to_var_map, cov_to_var_map, cumulant_order=2, identity=None):
    """
    Compute a particular non-Markovian error generator from the specified error generator layers.

    Parameters
    ----------
    errorgen_transform_maps : list of dicts
        List of dictionaries mapping tuples of LocalStimErrorgenLabels and circuit layer indices to 
        tuples of final error generators and phases. 

    errorgen_to_var_map : dict
        A dictionary whose keys are tuples of LocalStimErrorgenLabels and integer circuit layer indices
        and whose value is an integer corresponding to the corresponding variable index to use in constructed
        Polynomials.
        
    cov_to_var_map : dict
        A dictionary whose keys are five element tuples, with the first element being a LocalStimErrorgenLabel,
        the second a gate Label object. The second set of two elements are of the same type as the first two. 
        The last argument is the difference in circuit times. These are equivalent to the arguments of most CovarianceFunction
        ModelMembers.
    
    cumulant_order : int, optional (default 2)
        Order of the cumulant expansion to apply. At present only the first-order cumulant expansion is supported.

    identity : stim.PauliString, optional (default None)
        Optional pauli string corresponding to the all identity string with the correct number
        of qubits (can reduce overhead of many stim.PauliString instantations).
        
    Returns
    -------
    combined_cumulant_dict : dict
        A dictionary with LocalStimErrorgenLabel keys and values that are Polynomials corresponding to the terms and rates
        for the given order of the cumulant expansion.
    """
    if cumulant_order !=2:
        raise ValueError('Only second-order cumulant expansions are currently supported!')
    num_layers = len(errorgen_transform_maps)

    final_layer_transform_map = errorgen_transform_maps[-1]
    cumulants = []
    for starting_index in range(num_layers-1):
        current_transform_map = errorgen_transform_maps[starting_index]
        cumulants.append(error_generator_cumulant_symbolic_polynomial(final_layer_transform_map, current_transform_map, errorgen_to_var_map, 
                                                                      cov_to_var_map, cumulant_order, identity=identity))
    #add in the final layer with itself:
    cumulants.append(error_generator_cumulant_symbolic_polynomial(final_layer_transform_map, final_layer_transform_map, errorgen_to_var_map, 
                                                                      cov_to_var_map, cumulant_order, addl_weight=.5, identity=identity))
    
    #The contents of cumulants needs to get added to the first-order cumulant of the final layer, which is just the value
    #of final_layer.
    cumulants.append(error_generator_symbolic_polynomial(final_layer_transform_map, errorgen_to_var_map))
    
    combined_cumulant_keys = {key: None for key in chain(*cumulants)}
    max_num_vars = (max(errorgen_to_var_map.values()) + 1) + len(cov_to_var_map)
    combined_cumulant_dict = {errorgen: _Polynomial({}, max_num_vars=max_num_vars) for errorgen in combined_cumulant_keys}

    # Accumulate the coefficients contributing to each term.    
    for cumulant_dict in cumulants:
        for errorgen, poly in cumulant_dict.items():
            #HACK: Need better way to clear circuit time information after it is no longer needed.
            #errorgen.circuit_time = None
            combined_cumulant_dict[errorgen] += poly
    
    return combined_cumulant_dict

def error_generator_cumulant_symbolic_polynomial(errorgen_transform_map_1, errorgen_transform_map_2, 
                                                 errorgen_to_var_map, cov_to_var_map,
                                                 order=2, addl_weight=1.0, identity=None):
    """
    Function for computing the correlation function of two error generators
    represented as dictionaries of elementary error generator rates.

    Parameters
    ----------
    errorgen_transform_map_1 : dict
        Dictionary whose keys are error generator coefficients corresponding to the original
        (pre-propagation) error generators which gave rise to the elements of errgen_layer_1
        and whose values correspond to the sign that error generator's rate has picked up as a
        result of propagation.

    errorgen_transform_map_2 : dict
        See errorgen_transform_map_1.

    errorgen_to_var_map : dict
        A dictionary whose keys are tuples of LocalStimErrorgenLabels and integer circuit layer indices
        and whose value is an integer corresponding to the corresponding variable index to use in constructed
        Polynomials.
        
    cov_to_var_map : dict
        A dictionary whose keys are five element tuples, with the first element being a LocalStimErrorgenLabel,
        the second a gate Label object. The second set of two elements are of the same type as the first two. 
        The last argument is the difference in circuit times. These are equivalent to the arguments of most CovarianceFunction
        ModelMembers.

    order : int, optional (default 2)
        Order of the cumulant to compute.

    addl_weight : float, optional (default 1.0)
        Additional scaling to apply to cumulant.

    identity : stim.PauliString, optional (default None)
        Optional stim.PauliString corresponding to the all identity string with the correct number
        of qubits (can reduce overhead of many stim.PauliString instantations).

    Returns
    -------
    cumulant_errorgen_poly_dict : dict
        A dictionary whose keys are LocalStimErrorgenLabels, and whose values are Polynomials corresponding
        to the rates of those error generators as computed according to the cumulant of the two
        input error generators.
    """    
    #TODO: Enforce/check for time-ordering constraints
    if order !=2:
        raise ValueError('Only second-order cumulants are currently supported!')

    if not errorgen_transform_map_1 or not errorgen_transform_map_2:
        return dict()

    #avoid generating more stim PauliStrings than needed in composition.
    if identity is None:
        identity = stim.PauliString('I'*len(next(iter(errorgen_transform_map_1))[0].basis_element_labels[0]))

    composed_errgen_list = [] #for accumulating the tuples of weights and 
    var_list = []
    coeff_list = []
    #loop through error generator pairs in each of the dictionaries.
    for (errgen_1, layer_idx1), (output_errgen_1, sign_correction_1) in errorgen_transform_map_1.items():
        for (errgen_2, layer_idx2), (output_errgen_2, sign_correction_2) in errorgen_transform_map_2.items():
            cov_var = cov_to_var_map.get((errgen_1, errgen_1.gate_label, errgen_1.circuit_time, errgen_2, errgen_2.gate_label, errgen_2.circuit_time), None)
            if cov_var is not None:
                init_weight = addl_weight*sign_correction_1*sign_correction_2
                composed_coeff_sublist = _eprop.error_generator_composition(output_errgen_1, output_errgen_2, 
                                                                            identity=identity)
                for error_tup in composed_coeff_sublist:                
                    composed_errgen_list.append(error_tup[0])
                    var_list.append((cov_var,))
                    coeff_list.append(init_weight*error_tup[1])
                    
    #accumulate coefficients by output error generator and construct a combined Polynomial representation for the weight.
    # loop through all of the elements of commuted_errorgen_list and instantiate a dictionary with the requisite keys.
    composed_errorgen_var_dict = {errorgen: [] for errorgen in composed_errgen_list}
    composed_errorgen_coeff_dict = {errorgen: [] for errorgen in composed_errorgen_var_dict}
    
    # Accumulate the coefficients contributing to each term.    
    for errorgen, var, coeff in zip(composed_errgen_list, var_list, coeff_list):
        composed_errorgen_var_dict[errorgen].append(var)
        composed_errorgen_coeff_dict[errorgen].append(coeff)
        
    max_num_vars = (max(errorgen_to_var_map.values()) + 1) + len(cov_to_var_map)
    cumulant_errorgen_poly_dict = dict()
    for (key, variables), coefficients in zip(composed_errorgen_var_dict.items(), composed_errorgen_coeff_dict.values()):
        cumulant_errorgen_poly_dict[key] = _Polynomial.from_variable_and_coefficient_lists(variables, coefficients, max_num_vars)    
    
    return cumulant_errorgen_poly_dict

def covariance_to_polynomial_variable_maps(errorgen_transform_maps, covariance_function, cumulant_order=2, starting_index=0, return_reverse=False):
    """
    Function for computing the nth-order cumulant expansion for a set of error generator layers.

    Parameters
    ----------
    errorgen_transform_maps : list of dicts
        List of dictionaries mapping tuples of LocalStimErrorgenLabels and circuit layer indices to 
        tuples of final error generators and phases. 
    
    covariance_function : CovarianceFunction
        Covariance function to use in polynomial variable map construction.

    cumulant_order : int, optional (default 2)
        Order of the cumulant expansion to apply. At present only the first-order cumulant expansion is supported.
    
    starting_index : int, optional (default 0)
        An optional integer to use as the starting polynomial variable index.
        
    return_reverse : bool, optional (default, False)
        Optional flag that returns the reverse variable to errorgen label mapping.
        
    Returns
    -------
    cov_to_var_map : dict
        A dictionary whose keys are five-element tuples. The first element being a gate Label object, the second a LocalStimErrorgenLabel. 
        The second set of two elements are of the same type as the first two. 
        The last argument is the difference in circuit times. These are equivalent to the arguments of most CovarianceFunction
        ModelMembers. The values are integers corresponding to Polynomial variable indices to use for this covariance value. 
    """
        
    covariance_arguments = []
    for i in range(1, len(errorgen_transform_maps)+1):
        current_errorgen_transform_maps = errorgen_transform_maps[:i]
        covariance_arguments.extend(nonmarkovian_generator_covariance_arguments(current_errorgen_transform_maps, covariance_function, cumulant_order=cumulant_order))    
    
    #do another pass to get the unique arguments
    unique_covariance_arguments = {arg: None for arg in covariance_arguments}
    
    cov_to_var_map = {arg:i for i,arg in enumerate(unique_covariance_arguments, start=starting_index)}
    if return_reverse:
        var_to_cov_map = {i:arg for i,arg in enumerate(unique_covariance_arguments, start=starting_index)}
        return cov_to_var_map, var_to_cov_map
    else:
        return cov_to_var_map
    
def nonmarkovian_generator_covariance_arguments(errorgen_transform_maps, covariance_function, cumulant_order=2):
    """
    Compute a particular non-Markovian error generator from the specified error generator layers.

    Parameters
    ----------
    errorgen_transform_maps : list of dicts
        List of dictionaries mapping tuples of LocalStimErrorgenLabels and circuit layer indices to 
        tuples of final error generators and phases. 
    
    covariance_function : CovarianceFunction
        Covariance function to use in polynomial variable map construction.
        
    cumulant_order : int, optional (default 2)
        Order of the cumulant expansion to apply. At present only the first-order cumulant expansion is supported.
        
    Returns
    -------
    unique_covariance_arguments : list of tuples
        A list of five-element tuples. The first element being a gate Label object, the second a LocalStimErrorgenLabel. 
        The second set of two elements are of the same type as the first two. 
        The last argument is the difference in circuit times. These are equivalent to the arguments of most CovarianceFunction
        ModelMembers. 
    """
    if cumulant_order !=2:
        raise ValueError('Only second-order cumulant expansions are currently supported!')
    num_layers = len(errorgen_transform_maps)

    final_layer_transform_map = errorgen_transform_maps[-1]
    covariance_arguments = []
    for starting_index in range(num_layers-1):
        current_transform_map = errorgen_transform_maps[starting_index]
        covariance_arguments.extend(error_generator_cumulant_covariance_arguments(final_layer_transform_map, current_transform_map, covariance_function,
                                                                                order=cumulant_order))

    #add in the final layer with itself:
    covariance_arguments.extend(error_generator_cumulant_covariance_arguments(final_layer_transform_map, final_layer_transform_map, covariance_function, 
                                                                            order=cumulant_order))
    
    #do another pass to get the unique arguments
    unique_covariance_arguments = {arg: None for arg in covariance_arguments}
    unique_covariance_arguments = list(unique_covariance_arguments)

    return unique_covariance_arguments

def error_generator_cumulant_covariance_arguments(errorgen_transform_map_1, errorgen_transform_map_2, covariance_function,
                                                  order=2):
    """
    Function for returning a list of unique covariance function arguments that appear in the computation
    of the cumulant between the two input error generators (as represented by their transform maps).

    Parameters
    ----------
    errorgen_transform_map_1 : dict
        Dictionary whose keys are error generator coefficients corresponding to the original
        (pre-propagation) error generators which gave rise to the elements of errgen_layer_1
        and whose values correspond to the sign that error generator's rate has picked up as a
        result of propagation.

    errorgen_transform_map_2 : dict
        See errorgen_transform_map_1.
    
    covariance_function : CovarianceFunction
        Covariance function to use in polynomial variable map construction.
    
    order : int, optional (default 2)
        Order of the cumulant to compute.

    Returns
    -------
    unique_covariance_arguments : list of tuples
        A list of five-element tuples. The first element being a gate Label object, the second a LocalStimErrorgenLabel. 
        The second set of two elements are of the same type as the first two. 
        The last argument is the difference in circuit times. These are equivalent to the arguments of most CovarianceFunction
        ModelMembers. 

    """    
    #TODO: Enforce/check for time-ordering constraints
    if order !=2:
        raise ValueError('Only second-order cumulants are currently supported!')

    if not errorgen_transform_map_1 or not errorgen_transform_map_2:
        return []

    covariance_arguments = []
    #loop through error generator pairs in each of the dictionaries.
    for (errgen_1, layer_idx1), (output_errgen_1, sign_correction_1) in errorgen_transform_map_1.items():
        for (errgen_2, layer_idx2), (output_errgen_2, sign_correction_2) in errorgen_transform_map_2.items():
            cov_func_argument_tup = (errgen_1.gate_label, errgen_1, errgen_2.gate_label, errgen_2)
            #print(f'{cov_func_argument_tup=}')
            if covariance_function._errgen_label_to_param_idx.get(cov_func_argument_tup, None) is not None:

                covariance_arguments.append((errgen_1, errgen_1.gate_label, errgen_1.circuit_time, errgen_2, errgen_2.gate_label, errgen_2.circuit_time))
    #get just the unique subset of these covariance arguments.
    unique_covariance_arguments = {arg: None for arg in covariance_arguments}
    unique_covariance_arguments = list(unique_covariance_arguments)
            
    return unique_covariance_arguments

def construct_polynomial_covariance_parameter_vector(covariance_function, var_to_cov_map):
    """
    Constructs a vector of polynomial variable parameters for covariance values to use in evaluation of error generator Polynomial
    objects.
    
    Parameters
    ----------
    covariance_function : CovarianceFunction
        Covariance function to use in polynomial parameter vector construction.
        
    var_to_cov_map : dict
        A dictionary whose keys integers corresponding to Polynomial variable indices, and whose values are five element tuples, 
        with the first element being a LocalStimErrorgenLabel, the second a gate Label object. The second set of two elements 
        are of the same type as the first two. The last argument is the difference in circuit times. These are equivalent to the 
        arguments of most CovarianceFunction ModelMembers.
        
    Returns
    -------
    cov_param_vec : np.ndarray
        A vector of polynomial parameter values for covariances to use in evaluation of Polynomial objects.
    """
    
    #make sure we iterate through the variable indices in sorted order. I'm pretty sure they already would be, but still.
    sorted_var_indices = sorted(var_to_cov_map.keys())
    
    cov_param_vec = _np.zeros(len(var_to_cov_map))
    for i, var_idx in enumerate(sorted_var_indices):
        cov_arg_tup = var_to_cov_map[var_idx]
        #print(cov_arg_tup)
        cov_func_val = covariance_function(cov_arg_tup[0], cov_arg_tup[1], cov_arg_tup[2], cov_arg_tup[3], cov_arg_tup[4], 0)
        cov_param_vec[i] = cov_func_val
        
    return cov_param_vec