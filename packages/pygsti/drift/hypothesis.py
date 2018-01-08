from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Functions for general statistical hypothesis testing"""

import numpy as _np

def bonferoni_correction(confidence,numtests):
    """
    Todo: docstring
    
    """
    adjusted_confidence = 1 - (1 - confidence) / numtests
      
    return adjusted_confidence

def generalized_bonferoni_correction(confidence,weights,numtests=None):
    """
    Todo: docstring
    
    """
    weights = _np.array(weights)
    assert(_np.sum(weights) == 1.), "Invalid weighting! The weights must add up to 1."
    
    adjusted_confidence = _np.zeros(len(weights),float)
    adjusted_confidence = 1 - (1 - confidence)/weights

    if numtests is not None:
        
        assert(len(numtests) == len(weights)), "The number of tests must be specified for each weight!"
        for i in range(0,len(weights)):
            
            adjusted_confidence[i] = bonferoni_correction(adjusted_confidence[i],numtests[i])
            
    return adjusted_confidence
    
            