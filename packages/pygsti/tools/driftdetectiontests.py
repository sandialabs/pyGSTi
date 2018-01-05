from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""
Functions for drift detection on time series data. These functions are not automatically
imported by pyGSTi, and are mainly used as utility functions in timeseriestools.py

"""

import numpy as _np
from scipy.fftpack import dct as _dct
from scipy.fftpack import idct as _idct
from scipy.stats import chi2 as _chi2
from scipy.optimize import leastsq as _leastsq

from . import drifttools as _dtls
from . import driftdetectionthresholds as _dthresholds

####
#
#
# Functions that return various test statistics 
#
####

def summed_max_powers(ordered_modes,k=None):
    
    if k is None:
        _np.flip(_np.sort(modes),axis=0)
    else:    
        return _np.flip(_np.sort(modes),axis=0)[:k]


#
# Todo: add KS test etc.
#



### Functions for various statistical tests.

def one_sparse_fixed_frequency_test(mode,confidence=0.95):
    """
    """

    if mode**2 > _dthresholds.one_sparse_fixed_frequency_threshold(confidence=confidence):
        return True
    else:
        return False

def one_sparse_test(modes,confidence=0.95):
    """
    """
    threshold = _dthresholds.one_sparse_threshold(len(modes),confidence=confidence)
    max_power = _np.amax(modes**2)
    if max_power > threshold:
        out = True
    else:
        out = False
    return out

def one_to_k_sparse_test(modes,null_hypothesis,k=1,confidence=0.95,N=None,repeats=5,
                         thresholds=None,method=None,return_aux=False):
    
    """
    """
  
    # ordered power vector
    power = _np.flip(_np.sort(modes**2),axis=0)   
    test_statistic = _np.cumsum(power)
    if thresholds is None:
        aux_out = _dthresholds.one_to_k_sparse_threshold(null_hypothesis,k=k,confidence=confidence,
                                        N=N,repeats=repeats,method=method,return_aux=True)
        threshold = aux_out['threshold']
    else:
        threshold = thresholds
    
    out = {}
    for i in range (0, k):
        out[i+1] = False
        if test_statistic[i] > threshold[i]:
            out[i+1] = True
            
    if return_aux:
        triggered = []
        for i in out.keys():
            if out[i]:
                triggered.append(i)
        aux_out['test_results'] = out
        aux_out['tests_triggered'] = triggered
        aux_out['test_statistic'] = test_statistic
        return aux_out
    else:
        return out