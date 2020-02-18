"""Tools for general statistical hypothesis testing"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np


def bonferroni_correction(significance, numtests):
    """
    Calculates the standard Bonferroni correction, for reducing
    the "local" significance for > 1 statistical hypothesis
    test to guarantee maintaining a "global" significance (i.e.,
    a family-wise error rate) of `significance`.

    Parameters
    ----------
    confidence : float
        The desired global significance (often 0.05).

    numtests : int
        The number of hypothesis tests

    Returns
    -------
    The Boferroni-corrected local significance, given by
    `significance` / `numtests`.
    """
    local_significance = significance / numtests

    return local_significance


def sidak_correction(significance, numtests):
    """
    Todo: docstring
    """
    adjusted_significance = 1 - (1 - significance)**(1 / numtests)

    return adjusted_significance


def generalized_bonferroni_correction(significance, weights, numtests=None,
                                      nested_method='bonferroni', tol=1e-10):
    """
    Todo: docstring
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
