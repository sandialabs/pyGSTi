from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Functions for calculating test statistics for drift detection"""

import numpy as _np
from scipy.stats import chi2 as _chi2

def maxpower_pvalue_chi2(maxpower,timesteps,spectra_averaged):
    """
    Todo: docstring
    """
    pvalue = 1 - _chi2.cdf(maxpower*spectra_averaged,spectra_averaged)**timesteps
    
    return pvalue
            
    
def maxpower_threshold_chi2(confidence,timesteps,spectra_averaged):
    """
    Todo: docstring
    """
    threshold = _chi2.isf(1-confidence**(1/timesteps),spectra_averaged)/spectra_averaged
    
    return threshold