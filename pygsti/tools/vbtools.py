"""
Utility functions for featuremetric and volumetric benchmarking
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
from scipy.stats.qmc import Sobol as _Sobol

def max_depth_log2(error_per_layer, min_fidelity): 
    """
    A function for choose a maximum depth for benchmarking
    experiments based on a guess of the error per circuit
    layer.
    
    Parameters
    ----------
    error_per_layer : float
        The error per circuit layer.
        
    min_fidelity : float
        The desired minimum fidelity of the circuits 

    Returns
    -------  
    The smallest integer k such that the fidelity of
    a depth 2^k will be less than or equal to `min_fidelity`
    and a depth 2^{k-1} circuit will have a fidelity of greater
    than `min_fidelity`, if each layer in the circuit has an
    error per layer of `error_per_layer` and the noise is uniform
    depolarization. If that k is less than 0, the function returns
    -1.
    
    This function uses the approximation that the fidelities of
    depolarizing channels multiple, which is a good approximation
    except for very few qubit (narrow) circuits.
    """
    return max([-1,int(_np.ceil(_np.log2(_np.log(min_fidelity) / _np.log(1 - error_per_layer))))])
    
    
def max_depths_log2(widths, error_per_layer, min_fidelity, include_all_widths=True):
    """
    A function for choose a maximum depth for benchmarking
    experiments, versus circuit width, based on a guess of
    the error per circuit layer.i
    
    Parameters
    ----------
    widths : list or array
        The widths at which to compute a maximum depth
        
    error_per_layer : dict
        A dictionary containing the error per circuit layer
        (values) versus circuit width (keys)
        
    min_fidelity : float
        The desired minimum fidelity of the circuits.

    Returns
    -------  
    dict 
    
    A dictionary where the keys are the integers w in widths,
    and the values are the smallest integer k such that the fidelity of
    a depth 2^k will be less than or equal to `min_fidelity`
    and a depth 2^{k-1} width-w circuit will have a fidelity of greater
    than `min_fidelity`, if each layer in the circuit has an
    error per layer of `error_per_layer` and the noise is uniform
    depolarization.
    
    This function uses the approximation that the fidelities of
    depolarizing channels multiple, which is a good approximation
    except for very few qubit (narrow) circuits.
    """  

    if include_all_widths:
    	out = {w: max([0, max_depth_log2(error_per_layer[w], min_fidelity)]) for w in widths}
    else:
    	out = {}
    	for w in widths:
    		mdlog2 =  max_depth_log2(error_per_layer[w], min_fidelity)
    		if mdlog2 >= 0:
    			out[w] = mdlog2
    return out


def sample_circuit_shapes_logspace_uniform(max_depths_log2, num_samples, min_depth_log2=0):
	"""
	Samples circuit shapes (widths and depths) so that they are uniformly
	spaced in log depth space.

    Parameters
    ----------
    max_depths_log2 : dict
		The maximum depth (value) at each width (key)

    num_samples: float
		The number of circuit shapes to sample
        
    Returns
    -------  
    np.array
		The sampled circuit widths 

    np.array
		The sampled circuit depths
	"""
	widths = list(max_depths_log2.keys())
	widths_weights = _np.array([max_depths_log2[w] + 1 for w in widths])
	widths_probablities = widths_weights / _np.sum(widths_weights)
	sampled_widths = []
	sampled_depths = []
	for i in range(num_samples):
		w = _np.random.choice(widths, p=widths_probablities)
		log2_sampled_depth = -1
		while log2_sampled_depth < min_depth_log2:
			log2_sampled_depth = max_depths_log2[w] * _np.random.rand()
		d = int(_np.round(2**log2_sampled_depth))

		sampled_widths.append(w)
		sampled_depths.append(d)

	return sampled_widths, sampled_depths

def quasi_uniform_feature_vectors(num_samples, num_features, max_values, min_values, logspace, integer_valued):
	"""
	todo
	"""

	sobol = _Sobol(num_features)
	rounded_base2_num_samples = int(_np.ceil(_np.log2(num_samples)))
	samples = sobol.random_base2(rounded_base2_num_samples)

	rescaled_max_values = _np.zeros((num_features,), float)
	rescaled_min_values = _np.zeros((num_features,), float)
	for j in range(num_features):
		if logspace[j]:
			rescaled_max_values[j] = _np.log2(max_values[j])
			rescaled_min_values[j] = _np.log2(min_values[j])
		else:
			rescaled_max_values[j] = max_values[j]
			rescaled_min_values[j] = min_values[j]
	
	rescaled_samples = (rescaled_max_values - rescaled_min_values) * samples + rescaled_min_values

	sampled_feature_vectors = []
	
	for j in range(num_features):
		sampled_feature_value = rescaled_samples[:,j].copy()
		if logspace[j]:
			sampled_feature_value = 2**sampled_feature_value
		if integer_valued[j]:
			sampled_feature_value = _np.round(sampled_feature_value)
			sampled_feature_value = sampled_feature_value.astype(int)

		sampled_feature_vectors.append(sampled_feature_value)
	
	return sampled_feature_vectors


def sample_circuit_shapes_logspace_quasi_uniform(max_depths_log2, num_samples, min_depth_log2=0):
	"""
	Samples circuit shapes (widths and depths) so that they are uniformly
	spaced in log depth space.

    Parameters
    ----------
    max_depths_log2 : dict
		The maximum depth (value) at each width (key)

    num_samples: float
		The number of circuit shapes to sample
        
    Returns
    -------  
    np.array
		The sampled circuit widths 

    np.array
		The sampled circuit depths
	"""
	widths = list(max_depths_log2.keys())
	widths_weights = _np.array([max_depths_log2[w] + 1 for w in widths])
	widths_probablities = widths_weights / _np.sum(widths_weights)
	sampled_widths = []
	sampled_depths = []

	for i in range(num_samples):
		w = _np.random.choice(widths, p=widths_probablities)
		sampled_widths.append(w)

	num_each_width = {w: sampled_widths.count(w) for w in widths}
	for w in widths:
		sobol = _Sobol(1)
		rounded_up_log2_num_samples = int(_np.ceil(_np.log2(num_each_width[w])))
		unscaled_ds = sobol.random_base2(rounded_up_log2_num_samples)[:,0]
		sampled_depths_for_width[w].append(d)

	return sampled_widths, sampled_depths


def sample_random_subset(w, qubits):    
	"""
	Select a uniformly random set of w qubits from `qubits`. Returned
	list is in the same order as `qubits`.
	"""
	indices = list(np.random.choice(np.arange(0,len(qubits)), w, replace=False))
	indices.sort()
	return [qubits[i] for i in indices]
