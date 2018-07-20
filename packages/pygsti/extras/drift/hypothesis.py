from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Functions for general statistical hypothesis testing"""

import numpy as _np

def bonferroni_correction(confidence,numtests):
    """
    Calculates the standard Bonferroni correction, for raising the 
    confidence level of statistical tests when implementing more
    than a single test. This is as described on wiki.
    
    Parameters
    ----------
    confidence : float
        The desired overall confidence of the composite tests.
        
    numtests : int
        
                
    Returns
    -------
    array
        Todo.....
    
    """
    adjusted_confidence = 1 - (1 - confidence) / numtests
      
    return adjusted_confidence

def sidak_correction(confidence,numtests):
    """
    Todo: docstring
    """
    adjusted_confidence = confidence**(1/numtests)
      
    return adjusted_confidence

def generalized_bonferroni_correction(confidence, weights, numtests=None,
                                     nested_method='bonferroni',tol=1e-10):

    """
    Todo: docstring
    
    """
    weights = _np.array(weights)
    assert(_np.abs(_np.sum(weights) - 1.)<tol), "Invalid weighting! The weights must add up to 1."
    
    adjusted_confidence = _np.zeros(len(weights),float)
    adjusted_confidence = 1 - (1 - confidence)*weights

    if numtests is not None:
        
        assert(len(numtests) == len(weights)), "The number of tests must be specified for each weight!"
        for i in range(0,len(weights)):
            
            if nested_method == 'bonferroni':
                adjusted_confidence[i] = bonferroni_correction(adjusted_confidence[i],numtests[i])
                
            if nested_method == 'sidak':
                adjusted_confidence[i] = sidak_correction(adjusted_confidence[i],numtests[i])
                
    return adjusted_confidence
    
#Todo: add generalized sidak correction.          
