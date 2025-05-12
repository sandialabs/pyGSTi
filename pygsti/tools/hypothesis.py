"""
Tools for general statistical hypothesis testing
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np


def bonferroni_correction(significance, numtests):
    """
    Calculates the standard Bonferroni correction.

    This is used for reducing the "local" significance for > 1 statistical
    hypothesis test to guarantee maintaining a "global" significance (i.e., a
    family-wise error rate) of `significance`.

    Parameters
    ----------
    significance : float
        Significance of each individual test.

    numtests : int
        The number of hypothesis tests performed.

    Returns
    -------
    The Boferroni-corrected local significance, given by
    `significance` / `numtests`.
    """
    local_significance = significance / numtests
    return local_significance


def sidak_correction(significance, numtests):
    """
    Sidak correction.

    TODO: docstring - better explanaition

    Parameters
    ----------
    significance : float
        Significance of each individual test.

    numtests : int
        The number of hypothesis tests performed.

    Returns
    -------
    float
    """
    adjusted_significance = 1 - (1 - significance)**(1 / numtests)
    return adjusted_significance


def generalized_bonferroni_correction(significance, weights, numtests=None,
                                      nested_method='bonferroni', tol=1e-10):
    """
    Generalized Bonferroni correction.

    Parameters
    ----------
    significance : float
        Significance of each individual test.

    weights : array-like
        An array of non-negative floating-point weights, one per individual test,
        that sum to 1.0.

    numtests : int
        The number of hypothesis tests performed.

    nested_method : {'bonferroni', 'sidak'}
        Which method is used to find the significance of the composite test.

    tol : float, optional
        Tolerance when checking that the weights add to 1.0.

    Returns
    -------
    float
    """
    weights = _np.array(weights)
    assert(_np.abs(_np.sum(weights) - 1.) < tol), "Invalid weighting! The weights must add up to 1."

    adjusted_significance = _np.zeros(len(weights), float)
    adjusted_significance = significance * weights

    if numtests is not None:

        assert(len(numtests) == len(weights)), "The number of tests must be specified for each weight!"
        for i in range(0, len(weights)):

            if nested_method == 'bonferroni':
                adjusted_significance[i] = bonferroni_correction(adjusted_significance[i], numtests[i])

            if nested_method == 'sidak':
                adjusted_significance[i] = sidak_correction(adjusted_significance[i], numtests[i])

    return adjusted_significance
