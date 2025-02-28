"""
Tools for approximate non-Markovian simulation using cumulant expansion methods.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import stim
from itertools import chain 
from pygsti.tools import errgenproptools as _eprop


def cumulant_expansion(errorgen_layers, errorgen_transform_maps, cov_func, cumulant_order=2, truncation_threshold=1e-14):
    """
    Function for computing the nth-order cumulant expansion for a set of error generator layers.

    Parameters
    ----------
    errorgen_layers : list of dicts
        List of dictionaries of the error generator coefficients and rates for a circuit layer. 
        The error generator coefficients are represented using LocalStimErrorgenLabel.

    errorgen_transform_maps : dict
        Map giving the relationship between input error generators and their final
        value following propagation through the circuit. Needed to track any sign updates
        for terms with zero mean but nontrivial covariance.

    cov_func : 
        A function which maps tuples of elementary error generator labels at multiple times to
        a scalar quantity corresponding to the value of the covariance for that pair.
    
    cumulant_order : int, optional (default 2)
        Order of the cumulant expansion to apply. At present only the first-order cumulant expansion is supported.
    
    truncation_threshold : float, optional (default 1e-14)
        Threshold for which any error generators with magnitudes below this value
        are truncated.
        
    Returns
    -------
    cumulant_expansion_dict : dict
        A dictionary with the same general structure as those in `errorgen_layers`, but with the
        rates combined according to the selected order of the cumulant expansion.
    """
    #predefine the identity pauli to reduce number of instantiations.
    for layer in errorgen_layers:
        if layer:
            identity = stim.PauliString('I'*len(next(iter(layer)).basis_element_labels[0]))
            break
    else:
        identity = None

    nm_error_generators = []
    #TODO: See if I can't do this in a way that reuses computations across the nonmarkovian generators.
    for i in range(1, len(errorgen_layers)+1):
        current_errorgen_layers = errorgen_layers[:i]
        current_errorgen_sign_correction_maps = errorgen_transform_maps[:i]
        nm_error_generators.append(nonmarkovian_generator(current_errorgen_layers, current_errorgen_sign_correction_maps, cov_func, 
                                                          cumulant_order, truncation_threshold, identity))
    return nm_error_generators

def nonmarkovian_generator(errorgen_layers, errorgen_transform_maps, cov_func, cumulant_order=2, truncation_threshold=1e-14, identity=None):
    """
    Compute a particular non-Markovian error generator from the specified error generator layers.

    Parameters
    ----------
    errorgen_layers : list of dicts
        List of dictionaries of the error generator coefficients and rates for a circuit layer. 
        The error generator coefficients are represented using LocalStimErrorgenLabel.

    errorgen_transform_map : dict
        Maps giving the relationship between input error generators and their final
        value following propagation through the circuit. Needed to track any sign updates
        for terms with zero mean but nontrivial covariance.

    cov_func : 
        A function which maps tuples of elementary error generator labels at multiple times to
        a scalar quantity corresponding to the value of the covariance for that pair.
    
    cumulant_order : int, optional (default 2)
        Order of the cumulant expansion to apply. At present only the first-order cumulant expansion is supported.
    
    truncation_threshold : float, optional (default 1e-14)
        Threshold for which any error generators with magnitudes below this value
        are truncated.

    identity : stim.PauliString, optional (default None)
        Optional pauli string corresponding to the all identity string with the correct number
        of qubits (can reduce overhead of many stim.PauliString instantations).
        
    Returns
    -------
    cumulant_expansion_dict : dict
        A dictionary with the same general structure as those in `errorgen_layers`, but with the
        rates combined according to the selected order of the cumulant expansion.
    """
    if cumulant_order !=2:
        raise ValueError('Only second-order cumulant expansions are currently supported!')
    num_layers = len(errorgen_layers)

    final_layer = errorgen_layers[-1]
    final_layer_sign_map = errorgen_transform_maps[-1]
    cumulants = []
    for starting_index in range(num_layers-1):
        current_layer = errorgen_layers[starting_index]
        current_sign_map = errorgen_transform_maps[starting_index]
        cumulants.extend(error_generator_cumulant(final_layer, current_layer, final_layer_sign_map, current_sign_map, cov_func, cumulant_order, 
                                                  truncation_threshold, identity=identity))
    #add in the final layer with itself:
    cumulants.extend(error_generator_cumulant(final_layer, final_layer, final_layer_sign_map, final_layer_sign_map, cov_func, cumulant_order, 
                                              truncation_threshold, addl_scale=.5, identity=identity))
    
    #The contents of cumulants needs to get added to the first-order cumulant of the final layer, which is just the value
    #of final_layer.
    complete_coeff_set = set([tup[0] for tup in cumulants])
    complete_coeff_set.update(final_layer.keys())
    accumulated_nonmarkovian_generator = {key: 0 for key in complete_coeff_set} #make a copy of final_layer
    for coeff, rate in chain(cumulants, final_layer.items()):
        accumulated_nonmarkovian_generator[coeff] += rate
    
    return accumulated_nonmarkovian_generator


def error_generator_cumulant(errgen_layer_1, errgen_layer_2, phase_update_map_1, phase_update_map_2, cov_func, 
                             order=2, truncation_threshold=1e-14, addl_scale=1.0, identity=None):
    """
    Function for computing the correlation function of two error generators
    represented as dictionaries of elementary error generator rates.

    Parameters
    ----------
    errgen_layer_1 : dict
        Dictionary of the error generator coefficients and rates for a circuit layer. 
        The error generator coefficients are represented using LocalStimErrorgenLabel.
    
    errgen_layer_2 : list of dicts
        See errgen_layer_1.

    phase_update_map_1 : dict
        Dictionary whose keys are error generator coefficients corresponding to the original
        (pre-propagation) error generators which gave rise to the elements of errgen_layer_1
        and whose values correspond to the sign that error generator's rate has picked up as a
        result of propagation.

    phase_update_map_2 : dict
        See phase_update_map_1.

    cov_func : `CovarianceFunction`
        A function which maps tuples of elementary error generator labels at multiple times to
        a scalar quantity corresponding to the value of the covariance for that pair.

    order : int, optional (default 2)
        Order of the cumulant to compute.
    
    truncation_threshold : float, optional (default 1e-14)
        Threshold for which any error generators with magnitudes below this value
        are truncated.

    addl_scale : float, optional (default 1.0)
        Additional scaling to apply to cumulant.

    identity : stim.PauliString, optional (default None)
        Optional stim.PauliString corresponding to the all identity string with the correct number
        of qubits (can reduce overhead of many stim.PauliString instantations).

    Returns
    -------
    list of tuples
        A list of tuples 


    """
    #TODO: Enforce/check for time-ordering constraints
    if order !=2:
        raise ValueError('Only second-order cumulants are currently supported!')

    if not errgen_layer_1 or not errgen_layer_2:
        return dict()

    #avoid generating more stim PauliStrings than needed in composition.
    if identity is None:
        identity = stim.PauliString('I'*len(next(iter(errgen_layer_1)).basis_element_labels[0]))

    cumulant_coeff_list = [] #for accumulating the tuples of weights and 
    #loop through error generator pairs in each of the dictionaries.
    for errgen1 in errgen_layer_1:
        sign_correction_1 = phase_update_map_1[errgen1.initial_label]
        for errgen2 in errgen_layer_2:
            sign_correction_2 = phase_update_map_2[errgen2.initial_label]
            cov_val = cov_func(errgen1.initial_label, errgen1.gate_label, errgen1.circuit_time, errgen2.initial_label, 
                               errgen1.gate_label, errgen2.circuit_time) #can this be negative? Not for gaussian processes I don't think.
            if abs(cov_val) > truncation_threshold:
                cumulant_coeff_list.extend(_eprop.error_generator_composition(errgen1, errgen2, weight=addl_scale*sign_correction_1*sign_correction_2*cov_val, 
                                                                              identity=identity))
    
    return cumulant_coeff_list
