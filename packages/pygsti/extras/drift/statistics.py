"""Functions for calculating test statistics thresholds and p-values for drift detection"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
from scipy.stats import chi2 as _chi2

def power_significance_threshold(significance, numtests, dof):
    """
    todo
    """
    threshold = _chi2.isf(significance/numtests,dof)/dof

    return threshold

def power_to_pvalue(power, dof):
    """
    todo
    """
    pvalue = 1 - _chi2.cdf(dof*power,dof)

    return pvalue

# def maxpower_pvalue(maxpower, num_timesteps, dof):
#     """
#     Todo: docstring
#     """
#     pvalue = 1 - _chi2.cdf(maxpower*dof,dof)**(num_timesteps-1)
    
#     return pvalue

def power_fdr_quasithreshold(significance, numstats, dof):
    """
    Todo
    """
    quasithreshold = _np.array([_chi2.isf((numstats-i)*significance/numstats,dof)/dof for i in range(numstats)])
    
    return quasithreshold