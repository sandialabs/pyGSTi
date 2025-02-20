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
import numpy as _np
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel as _GEEL, LocalElementaryErrorgenLabel as _LEEL
from pygsti.baseobjs import QubitSpace as _QubitSpace
from pygsti.baseobjs.basis import BuiltinBasis as _BuiltinBasis
from pygsti.baseobjs.errorgenbasis import CompleteElementaryErrorgenBasis as _CompleteElementaryErrorgenBasis, ExplicitElementaryErrorgenBasis as _ExplicitElementaryErrorgenBasis
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE
from pygsti.modelmembers.operations import LindbladErrorgen as _LinbladErrorgen
from pygsti.circuits import Circuit as _Circuit
from pygsti.tools.optools import create_elementary_errorgen_nqudit, state_to_dmvec
from functools import reduce
from itertools import chain, product
from math import factorial
from pygsti.tools import errgenproptools as _eprop


def cumulant_expansion(errorgen_layers, cov_func, cumulant_order=2, truncation_threshold=1e-14):
    """
    Function for computing the nth-order cumulant expansion for a set of error generator layers.

    Parameters
    ----------
    errorgen_layers : list of dicts
        List of dictionaries of the error generator coefficients and rates for a circuit layer. 
        The error generator coefficients are represented using LocalStimErrorgenLabel.

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
    nm_error_generators = []
    #TODO: See if I can't do this in a way that reuses computations across the nonmarkovian generators.
    for i in range(1, len(errorgen_layers)+1):
        current_errorgen_layers = errorgen_layers[:i]
        nm_error_generators.append(nonmarkovian_generator(current_errorgen_layers, cov_func, cumulant_order, truncation_threshold))
        

def nonmarkovian_generator(errorgen_layers, cov_func, cumulant_order=2, truncation_threshold=1e-14):
    """
    Compute a particular non-Markovian error generator from the specified error generator layers.

    Parameters
    ----------
    errorgen_layers : list of dicts
        List of dictionaries of the error generator coefficients and rates for a circuit layer. 
        The error generator coefficients are represented using LocalStimErrorgenLabel.

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
    if cumulant_order !=2:
        raise ValueError('Only second-order cumulant expansions are currently supported!')
    num_layers = len(errorgen_layers)

    final_layer = errorgen_layers[-1]
    cumulants = []
    for starting_index in range(num_layers):
        current_layer = errorgen_layers[starting_index]
        cumulants.extend(error_generator_cumulant(final_layer, current_layer, cov_func, cumulant_order, truncation_threshold))
        
    #The contents of cumulants needs to get added to the first-order cumulant of the final layer, which is just the value
    #of final_layer.
    complete_coeff_set = set([tup[0] for tup in cumulants])
    complete_coeff_set.update(final_layer.keys())
    accumulated_nonmarkovian_generator = {key: 0 for key in complete_coeff_set} #make a copy of final_layer
    for coeff, rate in chain(cumulants, final_layer.items()):
        accumulated_nonmarkovian_generator[coeff] += rate
    
    return accumulated_nonmarkovian_generator


def error_generator_cumulant(errgen_layer_1, errgen_layer_2, cov_func, order=2, truncation_threshold=1e-14):
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

    cov_func : 
        A function which maps tuples of elementary error generator labels at multiple times to
        a scalar quantity corresponding to the value of the covariance for that pair.

    order : int, optional (default 2)
        Order of the cumulant to compute.
    
    truncation_threshold : float, optional (default 1e-14)
        Threshold for which any error generators with magnitudes below this value
        are truncated.

    Returns
    -------


    """
    #TODO: Enforce/check for time-ordering constraints
    if order !=2:
        raise ValueError('Only second-order cumulants are currently supported!')

    if not errgen_layer_1 or not errgen_layer_2:
        return dict()

    #avoid generating more stim PauliStrings than needed in composition.
    identity = stim.PauliString('I'*len(next(iter(errgen_layer_1)).basis_element_labels[0]))

    cumulant_coeff_list = [] #for accumulating the tuples of weights and 
    #compute the 
    #loop through error generator pairs in each of the dictionaries.
    for rate1, errgen1 in errgen_layer_1.items():
        for rate2, errgen2 in errgen_layer_2.items():
            combined_rate = rate1*rate2
            cov_val = cov_func(errgen1, errgen2) #can this be negative?
            if abs(cov_val) > 0:
                combined_cov_rate = combined_rate*cov_val #TODO: revisit this
                cumulant_coeff_list.append(_eprop.error_generator_composition(errgen1, errgen2, weight=combined_cov_rate, identity=identity))
    
    return cumulant_coeff_list
    #accumulate terms:
    #accumulated_cumulant = 
    #for 