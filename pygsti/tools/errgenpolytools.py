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
from pygsti.baseobjs.polynomial import Polynomial as _Polynomial
from itertools import chain, product
from math import factorial
from typing import Literal, Optional, Union, Callable, Iterable
import pygsti.tools.errgenproptools as _eprop


def error_generator_to_polynomial_variable_maps(errorgen_transform_map):
    """
    Helper function which returns two dictionaries. One is a map from error generator label + layer index tuples to integers, and the second is the reverse map from integers to error
    generator label + layer index tuples.
    """
    errorgen_to_var_map = {key:i for i,key in enumerate(errorgen_transform_map.keys())}
    var_to_errorgen_map = {i:key for i,key in enumerate(errorgen_transform_map.keys())}
    return errorgen_to_var_map, var_to_errorgen_map

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
    
    first_order_magnus_var_dict = {key[0]: [] for key in chain(*errorgen_transform_maps)}
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

def error_generator_taylor_expansion_symbolic_polynomial(errorgen_dict, errorgen_to_var_map, order = 1):
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

def stabilizer_probability_correction_symbolic_polynomial(errorgen_dict, errorgen_to_var_map, tableau, desired_bitstring, order = 1):
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
    num_random = _eprop.random_support(tableau)
    if num_random > 2148:
        raise RuntimeError('Number of random bits is greater than 1074, magnitude of probability scale will underflow!')
    scale = 1/2**(num_random) 
    
    taylor_expansion = error_generator_taylor_expansion_symbolic_polynomial(errorgen_dict, errorgen_to_var_map, order)
    combined_taylor_dict = _combined_taylor_expansion_polynomial(taylor_expansion, errorgen_to_var_map)
    complete_taylor_errgen_labels = list(combined_taylor_dict.keys())
    
    #can now do the correction computation in a single-shot.
    alphas = _np.ravel(_eprop.bulk_alpha(complete_taylor_errgen_labels, tableau, [desired_bitstring]))
    #use the nonzero alphas to filter 
    nonzero_alpha_idxs = _np.nonzero(alphas)[0]
    nonzero_alpha = alphas[nonzero_alpha_idxs]
    nonzero_alpha_labels = [complete_taylor_errgen_labels[idx] for idx in nonzero_alpha_idxs]

    #gather the coefficients corresponding to each of the labels with nonzero alpha and sum this into a
    #single polynomial. Take care of the scale factor from the number of random variables at the same time.
    nonzero_alpha_coeffs = [combined_taylor_dict[lbl].scalar_mult(alpha*scale) for lbl, alpha in zip(nonzero_alpha_labels, nonzero_alpha)]
    correction_poly = _Polynomial.sum(nonzero_alpha_coeffs)
    
    return correction_poly

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
    #accumulate the terms across orders (with short circuit logic for order 1 to save time):
    taylor_expansion = error_generator_taylor_expansion_symbolic_polynomial(errorgen_dict, errorgen_to_var_map, order)
    combined_taylor_dict = _combined_taylor_expansion_polynomial(taylor_expansion, errorgen_to_var_map)
    complete_taylor_errgen_labels = list(combined_taylor_dict.keys())

    #can now do the correction computation in a single-shot.
    alphas = _np.ravel(_eprop.bulk_alpha_pauli(complete_taylor_errgen_labels, tableau, [pauli]))
    #use the nonzero alphas to filter 
    nonzero_alpha_idxs = _np.nonzero(alphas)[0]
    nonzero_alpha = alphas[nonzero_alpha_idxs]
    nonzero_alpha_labels = [complete_taylor_errgen_labels[idx] for idx in nonzero_alpha_idxs]

    #gather the coefficients corresponding to each of the labels with nonzero alpha and sum this into a
    #single polynomial. Take care of the scale factor from the number of random variables at the same time.
    nonzero_alpha_coeffs = [combined_taylor_dict[lbl].scalar_mult(alpha) for lbl, alpha in zip(nonzero_alpha_labels, nonzero_alpha)]
    correction_poly = _Polynomial.sum(nonzero_alpha_coeffs)
    
    return correction_poly    

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