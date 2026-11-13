"""
Utility functions for featuremetric and volumetric benchmarking
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
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
    A function for choosing a maximum depth for benchmarking
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
    depolarizing channels multiply, which is a good approximation
    except for very few qubit (narrow) circuits.

    Raises
    ------
    ValueError
        If `error_per_layer` or `min_fidelity` is not strictly between 0 and 1.
    """
    if not (0 < error_per_layer < 1):
        raise ValueError(f"error_per_layer must be between 0 and 1 (exclusive), but it is {error_per_layer}.")
    if not (0 < min_fidelity < 1):
        raise ValueError(f"min_fidelity must be between 0 and 1 (exclusive), but it is {min_fidelity}.")
    return max([-1, int(_np.ceil(_np.log2(_np.log(min_fidelity) / _np.log(1 - error_per_layer))))])


def max_depths_log2(widths, error_per_layer, min_fidelity, include_all_widths=True):
    """
    A function for choosing a maximum depth for benchmarking
    experiments, versus circuit width, based on a guess of
    the error per circuit layer.

    Parameters
    ----------
    widths : list or array
        The widths at which to compute a maximum depth

    error_per_layer : dict
        A dictionary containing the error per circuit layer
        (values) versus circuit width (keys)

    min_fidelity : float
        The desired minimum fidelity of the circuits.

    include_all_widths : bool, optional
        If True, every width in `widths` appears in the returned dictionary.
        A width at which `max_depth_log2` returns -1 gets 0, which is the
        base-2 logarithm of depth 1, not a depth of 0. A width where depth 1
        is the first power-of-two depth whose fidelity is at most
        `min_fidelity` also gets 0. If False, the widths at which
        `max_depth_log2` returns -1 are omitted.

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
    depolarizing channels multiply, which is a good approximation
    except for very few qubit (narrow) circuits.

    Raises
    ------
    ValueError
        If `min_fidelity`, or `error_per_layer[w]` for some `w` in `widths`,
        is not strictly between 0 and 1.
    """
    if include_all_widths:
        out = {w: max([0, max_depth_log2(error_per_layer[w], min_fidelity)]) for w in widths}
    else:
        out = {}
        for w in widths:
            mdlog2 = max_depth_log2(error_per_layer[w], min_fidelity)
            if mdlog2 >= 0:
                out[w] = mdlog2
    return out


def sample_circuit_shapes_logspace_uniform(max_depths_log2, num_samples, min_depth_log2=0, seed=None):
    """
    Samples circuit shapes (widths and depths) so that they are uniformly
    spaced in log depth space.

    Widths are sampled with a probability proportional to the number of
    power-of-two depths available at that width, `max_depths_log2[w] - min_depth_log2 + 1`,
    so that the sampled shapes are approximately uniformly distributed over the
    (width, log depth) region. A width whose maximum is below `min_depth_log2` is
    never sampled.

    At the sampled width `w`, the base-2 logarithm of the depth is uniform between
    `min_depth_log2` and `max_depths_log2[w]`, and the depth is 2 to that power,
    rounded to the nearest integer.

    Parameters
    ----------
    max_depths_log2 : dict
        The base-2 logarithm of the maximum depth (value) at each width (key)

    num_samples : int
        The number of circuit shapes to sample

    min_depth_log2 : int, optional
        The base-2 logarithm of the minimum depth to sample. Expected to be a
        non-negative integer: the width weights count power-of-two depths only
        for integer values, and a negative value can give depth-0 shapes.

    seed : int, optional
        A seed for the random number generator, for reproducible sampling.

    Returns
    -------
    list
        The sampled circuit widths

    list
        The sampled circuit depths

    Raises
    ------
    ValueError
        If `min_depth_log2` is larger than `max_depths_log2` at every width
        (including when `max_depths_log2` is empty), so no shape can be sampled.
    """
    rand_state = _np.random.RandomState(seed)  # Ok if seed is None
    widths = list(max_depths_log2.keys())
    widths_weights = _np.array([max_depths_log2[w] - min_depth_log2 + 1 if max_depths_log2[w] >= min_depth_log2
                                else 0 for w in widths], float)
    if _np.sum(widths_weights) == 0:
        raise ValueError(f"min_depth_log2={min_depth_log2} is larger than max_depths_log2 at every width "
                         f"({max_depths_log2}), so no circuit shape can be sampled.")
    widths_probabilities = widths_weights / _np.sum(widths_weights)
    sampled_widths = []
    sampled_depths = []
    for i in range(num_samples):
        w = rand_state.choice(widths, p=widths_probabilities)
        log2_sampled_depth = rand_state.uniform(min_depth_log2, max_depths_log2[w])
        d = int(_np.round(2**log2_sampled_depth))

        sampled_widths.append(w)
        sampled_depths.append(d)

    return sampled_widths, sampled_depths


def quasi_uniform_feature_vectors(num_samples, num_features, max_values, min_values, logspace, integer_valued,
                                  seed=None):
    """
    Samples feature vectors that are quasi-uniformly spread over a hyperrectangle.

    The samples are drawn from a Sobol sequence, which fills the hyperrectangle more
    evenly than independent uniform sampling does, and are then rescaled onto the
    specified range of each feature. A feature can be sampled uniformly in its
    logarithm rather than in its value, and can be rounded to an integer.

    Because a Sobol sequence only has its low-discrepancy property for a number of
    points that is a power of two, `num_samples` is rounded up to the next power of
    two: `2**ceil(log2(num_samples))` feature vectors are returned.

    Parameters
    ----------
    num_samples : int
        The number of feature vectors to sample. Rounded up to a power of two.

    num_features : int
        The number of features, i.e., the dimension of each feature vector.

    max_values : list or array
        The maximum value of each feature. Length `num_features`.

    min_values : list or array
        The minimum value of each feature. Length `num_features`.

    logspace : list or array of bool
        Whether each feature is sampled uniformly in its logarithm (True) or
        uniformly in its value (False). Length `num_features`.

    integer_valued : list or array of bool
        Whether each feature is rounded to the nearest integer. Length `num_features`.

    seed : int, optional
        A seed for the random scrambling of the Sobol sequence, for reproducible
        sampling. With None, every call scrambles differently.

    Returns
    -------
    list of numpy.ndarray
        A list of `num_features` arrays, the jth of which contains the sampled
        values of the jth feature. Each array has length `2**ceil(log2(num_samples))`.

    Raises
    ------
    ValueError
        If `num_samples` is less than 1.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1, but it is {num_samples}.")
    sobol = _Sobol(num_features, seed=seed)
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
        sampled_feature_value = rescaled_samples[:, j].copy()
        if logspace[j]:
            sampled_feature_value = 2**sampled_feature_value
        if integer_valued[j]:
            sampled_feature_value = _np.round(sampled_feature_value)
            sampled_feature_value = sampled_feature_value.astype(int)

        sampled_feature_vectors.append(sampled_feature_value)

    return sampled_feature_vectors
