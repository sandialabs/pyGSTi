from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Functions for Fourier analysis of equally spaced time-series data"""

import numpy as _np
from scipy.fftpack import dct as _dct
from scipy.fftpack import idct as _idct
from scipy.stats import chi2 as _chi2
from scipy.optimize import leastsq as _leastsq
from scipy import convolve

from . import drifttools as _dtls


#def singlesequence_bootstrap(p,timesteps,counts,bootstraps=10000):
#    
#    
#    #
#    # Todo: write this function.
#    #
#    return 0
    
#def multisequence_bootstrap(return_all=True):
#    #
#    # Todo: write this function.
#    #
#    return 0

    #
#
# Todo: delete this function and replace with the function above, with an option for whether all bootstraps or
# just the average power spectrum is returned.
#
def global_bootstrap(num_spectra,N,num_outcomes,probs=None,bootstraps=500):
    """
    TODO: docstring
    """
    largest_power = _np.zeros(bootstraps)
    
    if probs is None:
        probs = _np.ones((num_outcomes,num_spectra),float)/num_outcomes

    for k in range (0,bootstraps):   

        bs = _np.array([_np.random.multinomial(1,probs[:,sequence],size=N) for sequence in range (0,num_spectra)])
        powers = _np.zeros((num_spectra,num_outcomes,N),float)
    
        for i in range (0,num_spectra):
            for j in range (0, num_outcomes):
                p = _np.mean(bs[i,:,j]) 
            
                if p == 1 or p == 0:
                    powers[i,j,:] =  _np.zeros(N,float)
                    
                else:
                    pvec = p * _np.ones(N,float)
                    powers[i,j,:] = _dct((bs[i,:,j] - p)/(_np.sqrt(pvec * (1 - pvec))),norm='ortho')**2
                
        averaged_power_spectra = _np.mean(_np.mean(powers,axis=0),axis=0)
        largest_power[k] = _np.amax(averaged_power_spectra)
        
    return largest_power
    
def global_threshold(largest_power,confidence=0.95):
    
    bootstraps = len(largest_power)
                
    # Empirical test statistics from highest to lowest,
    # and the corresponding values of the empirical 1 - cdf.
    ordered_largest_power = _np.sort(largest_power)
    oneminuscdf = 1 - _np.arange(1,bootstraps+1)/(bootstraps+1)
                
    # Truncate the empirical 1 - cdf  to the smallest 0.95 values
    truncated_ordered_largest_power = ordered_largest_power[int(_np.ceil(bootstraps*0.95)):bootstraps]
    truncated_oneminuscdf = oneminuscdf[int(_np.ceil(bootstraps*0.95)):bootstraps]
            
    soln = _np.zeros(2,float)
    p0 = [1.,-1*_np.amin(truncated_ordered_largest_power)]
    soln, success = _leastsq(threshold_errfunc, p0, args=(truncated_ordered_largest_power,truncated_oneminuscdf))
    threshold = estimated_threshold(soln,confidence)
                        
    return threshold

def one_sparse_threshold(n,confidence=0.95):
    """
    Calculates the threshold for the test statistic for detecting a one-sparse 
    signal defined by:
    
    "Power of the largest Fourier mode"
    
    This is the value of the test statistic under the null hypothesis in which the 
    Fourier modes are the DCT of a Bernoulli process from which the null hypothesis
    for the bias p_t has been removed pre-DCT (and which have all been standardized 
    to have unit variance either before or after the DCT) and using the ansatz that 
    the (standardized) Fourier modes are i.i.d. chi2 with one degree of freedom 
    random variables.
    
    Parameters
    ----------
    n : int
        The number of Fourier modes
        
    confidence : float, optional
        1 - confidence is the probability that the test gives a false positive.
        
    Returns
    -------
    float
        The value for the test statistic, based on the formula that the CDF
        for the extremal value distribution from n draws from a distribution 
        is F(x)^n, where F(x) is the CDF of the distribution. _chi2.isf(x,1)
        is scipy's implementation of inverse( 1 - F(x)) for the chi^2_1 
        distribution.
        
    """
    return _chi2.isf(1-confidence**(1/n),1)

def one_sparse_fixed_frequency_threshold(confidence=0.95):
    """
    ......
    """
    return _chi2.isf(1-confidence,1)

def threshold_fitfunc(p, x):
    """
    The exponential fit function for fitting the empirical 1 - CDF of
    bootstrapped order statistics under the null hypothesis.
    
    """
    return _np.exp(-p[0] * (x + p[1]))
        
def threshold_errfunc(p, x, y):
    """
    Error function for least squares fitting of bootstrapped order statistics data.
    
    """
    return threshold_fitfunc(p,x) - y
       
def estimated_threshold(p, confidence):
    """
    Implements the inverse of the threshold_fitfunc on 1 - confidence with parameters
    p. Gives the threshold predicted by the fitted 1 - CDF, at `confidence` level of
    confidence.
    
    """
    return -1*(p[1] + _np.log(1 - confidence) / p[0])
   



def one_to_k_sparse_unadjusted_thresholds(null_hypothesis, n, k, confidence=0.95,
                              bootstraps=None, method=None, repeats=5, return_aux=False):
    """
    null_hypothesis : float or array
        The null hypothesis that we are looking for statistically significant evidence to reject.
        This will often be constant probability, in which case this should be a float. If it is
        not a constant probability, this should be an array of probabilities in [0,1] of length
        n.
    
    n: int
        The number of data points on which we are performing the analysis. Defines the size of the
        DCT.
        
    k: int
        The maximum k-sparse test to return the trhesold for. Thresholds are returned for all 1 - k
        sparse tests.
        
    confidence: float, opt
        The confidence level for the thresholds
        
    bootstraps: int, opt
        The number of bootstraps to use to calculate the thresholds. If None, this is set to a
        suitable default value which depends on whether `method` is 'basic' or 'fitted'. In the
        case of 'fitted', the default value does not scale with any parameters, but in the case
        of 'basic' it scales with 'confidence' and for very small '1-confidence' this will make
        the runtime of the function very long.
        
    method: str, opt
        Allow values are 'None', 'basic' or 'fitted. If 'basic' the thresholds are based on
        taking many bootstraps of the k test statistics under the given null hypothesis, and then
        taking the thresholds as a fraction of 'confidence' of the way through an ordered array
        of the bootstrapped test statistics. This requires 'bootstraps' >> 1/(1-'confidence') to 
        be reliable. If 'fitted' the thresholds are based on fitting the 1 - CDF of the test
        statistics to an exponential decay. If None then it defaults to one of these two options
        based on the size of confidence.
        
    repeats: int, opt
        The number of times the full threshold-finding bootstrap is repeated. The returned threshold
        is the mean of the found thresholds. If 'return_aux' is True then the list of all thresholds
        is returned, which can be used to check the stability of the obtained thresholds.
        
    return_aux: bool, opt
        Whether to return auxillary information about the thresholding bootstraps.
        
    
              
    """
    p = None
    if type(null_hypothesis) is float:
        p = null_hypothesis
        null_hypothesis = p*_np.ones(n)

    if method is None:
        if confidence <= 0.95:
            method='basic'
        else:
            method='fitted'            
    else:
        if method != 'basic' and method != 'fitted':
            raise ValueError("'method' should be None, 'basic' or 'fitted'")
         
    if bootstraps is None:
        if method == 'basic':
            bootstraps = int(_np.ceil(500/(1-confidence)))
                
        if method=='fitted':
            bootstraps = 50000
                                
    threshold_set = _np.zeros((k,repeats),float)   
    fail = []
    soln = None
    
    for j in range (0,repeats):
            
        if p is not None:
            bs = _np.random.binomial(1,p,size=(n,bootstraps))
        else:
            if len(null_hypothesis) == n:
                bs = _np.array([_np.random.binomial(1,p,size=bootstraps) for p in null_hypothesis])
            else:
                raise ValueError("If null_hypothesis is not a float, it must be an array of length n.")

        test_statistic = _np.zeros((n,bootstraps),float)
        for i in range (0,bootstraps):
            test_statistic[:,i] = _np.cumsum(_np.flip(_np.sort(_dtls.DCT(bs[:,i],null_hypothesis=null_hypothesis)**2),axis=0))
            

        if method == 'basic':
            threshold_index = int(_np.ceil((1-confidence)*bootstraps))
            test_statistic = _np.flip(_np.sort(test_statistic,axis=1),axis=1)
            threshold_set[:,j]  = test_statistic[:k,threshold_index]
            
        if method == 'fitted':         
            test_statistic = _np.sort(test_statistic,axis=1)
            oneminuscdf = 1 - _np.arange(1,bootstraps+1)/(bootstraps+1)
        
            # Truncate the empirical 1 - cdf  to the smallest 0.95 values
            fraction_index = int(_np.ceil(bootstraps*0.95))       
            truncated_test_statistic = test_statistic[:,fraction_index:]
            truncated_oneminuscdf = oneminuscdf[fraction_index:]
                
            # Fit to an exponential decay
            soln = _np.zeros((k,2),float)
            for i in range(0,k):
                p0 = [1.,-1*_np.amin(truncated_test_statistic[i,:])]
                soln[i,:], success = _leastsq(threshold_errfunc, p0, 
                                        args=(truncated_test_statistic[i,:],truncated_oneminuscdf))
                if success !=1 and success !=2 and success !=3 and success!=4:
                    fail.append(i+1)
                threshold_set[i,j] = estimated_threshold(soln[i,:],confidence)
     
    threshold = _np.mean(threshold_set,axis=1)
    consistency = one_sparse_threshold(n,confidence=confidence) - threshold[0]
    
    if return_aux:
        aux_out = {}
        aux_out['null_hypothesis'] = null_hypothesis
        aux_out['confidence'] = confidence
        aux_out['threshold'] = threshold
        aux_out['thresholds'] = threshold_set
        aux_out['consistency'] =  consistency
        aux_out['bootstraps'] = bootstraps
        aux_out['repeats'] = repeats
        aux_out['samples'] = bootstraps*repeats
        aux_out['method'] = method
        aux_out['threshold_function_params'] = soln
        aux_out['fail'] = fail
        
        return threshold, aux_out
    else:
        return threshold


def bartlett_spectrum_one_to_k_sparse_thresholds(null_hypothesis, n, k, num_spectra, confidence=0.95,
                              bootstraps=None, method=None, repeats=5):
    #
    #
    #
    # THIS FUNCTION COULD BE EASILY MERGED INTO THE NORMAL 1-k THRESHOLD FUNCTION. IT IS ALMOST
    # IDENTITICAL!
    #


    frac = int(n/num_spectra)
    
    p = None
    if type(null_hypothesis) is float:
        p = null_hypothesis
        null_hypothesis = p*_np.ones(n)

    if method is None:
        if confidence <= 0.95:
            method='basic'
        else:
            method='fitted'            
    else:
        if method != 'basic' and method != 'fitted':
            raise ValueError("'method' should be None, 'basic' or 'fitted'")
         
    if bootstraps is None:
        if method == 'basic':
            bootstraps = int(_np.ceil(500/(1-confidence)))
                
        if method=='fitted':
            bootstraps = 50000
                                
    threshold_set = _np.zeros((k,repeats),float)   
    fail = []
    soln = None
    
    for j in range (0,repeats):
            
        if p is not None:
            bs = _np.random.binomial(1,p,size=(n,bootstraps))
        else:
            if len(null_hypothesis) == n:
                bs = _np.array([_np.random.binomial(1,p,size=bootstraps) for p in null_hypothesis])
            else:
                raise ValueError("If null_hypothesis is not a float, it must be an array of length n.")

        test_statistic = _np.zeros((frac,bootstraps),float)
        for i in range (0,bootstraps):
            spectra_bart = _dtls.bartlett_DCT_spectrum(bs[:,i],n,num_spectra,null_hypothesis=null_hypothesis)
            test_statistic[:,i] = _np.cumsum(_np.flip(_np.sort(spectra_bart),axis=0))
            

        if method == 'basic':
            threshold_index = int(_np.ceil((1-confidence)*bootstraps))
            test_statistic = _np.flip(_np.sort(test_statistic,axis=1),axis=1)
            threshold_set[:,j]  = test_statistic[:k,threshold_index]
            
        if method == 'fitted':         
            test_statistic = _np.sort(test_statistic,axis=1)
            oneminuscdf = 1 - _np.arange(1,bootstraps+1)/(bootstraps+1)
        
            # Truncate the empirical 1 - cdf  to the smallest 0.95 values
            fraction_index = int(_np.ceil(bootstraps*0.95))       
            truncated_test_statistic = test_statistic[:,fraction_index:]
            truncated_oneminuscdf = oneminuscdf[fraction_index:]
                
            # Fit to an exponential decay
            soln = _np.zeros((k,2),float)
            for i in range(0,k):
                p0 = [1.,-1*_np.amin(truncated_test_statistic[i,:])]
                soln[i,:], success = _leastsq(threshold_errfunc, p0, 
                                        args=(truncated_test_statistic[i,:],truncated_oneminuscdf))
                if success !=1 and success !=2 and success !=3 and success!=4:
                    fail.append(i+1)
                threshold_set[i,j] = estimated_threshold(soln[i,:],confidence)
     
    threshold = _np.mean(threshold_set,axis=1)

    return threshold

#def outavg_one_to_k_sparse_unadjusted_thresholds(probs, num_outcomes, n, k, 
#                                                 confidence=0.95, bootstraps=None, method='fitted', 
##                                                 repeats=5, return_aux=False):
# #       
#    if method != 'basic' and method != 'fitted':
#        raise ValueError("'method' should be None, 'basic' or 'fitted'")
#         
#    if bootstraps is None:
#        if method == 'basic':
#            bootstraps = int(_np.ceil(500/(1-confidence)))
#                
#        if method=='fitted':
#            bootstraps = 50000
#                                
#    threshold_set = _np.zeros((k,repeats),float)   
#    fail = []
#    soln = None
#    
#    for j in range (0,repeats):
#                      
#        bs = _np.random.multinomial(1,probs,size=(n,bootstraps))        
#        test_statistic = _np.zeros((n,bootstraps),float)
#        
#        for i in range (0,bootstraps):
#            
#            powers = _np.zeros((num_outcomes,n),float)
#            avg_powers = _np.zeros(n,float)
#            
#            for m in range (0, num_outcomes):
#                p = _np.mean(bs[:,i,m])
#                
#                if p == 1 or p == 0:
#                    powers[m,:] =  _np.zeros(n,float)                    
#                else:
#                    powers[m,:] = _dct((bs[:,i,m] - p)/(_np.sqrt(p * (1 - p))),norm='ortho')**2
#                    
#            avg_powers = _np.mean(powers,axis=0)                       
#            test_statistic[:,i] = _np.cumsum(_np.flip(_np.sort(avg_powers),axis=0))
#            #
#
#        if method == 'basic':
#            threshold_index = int(_np.ceil((1-confidence)*bootstraps))
#            test_statistic = _np.flip(_np.sort(test_statistic,axis=1),axis=1)
#            threshold_set[:,j]  = test_statistic[:k,threshold_index]
#            
#        if method == 'fitted':         
#            test_statistic = _np.sort(test_statistic,axis=1)
#            oneminuscdf = 1 - _np.arange(1,bootstraps+1)/(bootstraps+1)
#        
#            # Truncate the empirical 1 - cdf  to the smallest 0.95 values
#            fraction_index = int(_np.ceil(bootstraps*0.95))       
#            truncated_test_statistic = test_statistic[:,fraction_index:]
#            truncated_oneminuscdf = oneminuscdf[fraction_index:]
##                
#            # Fit to an exponential decay
#            soln = _np.zeros((k,2),float)
#            for i in range(0,k):
#                p0 = [1.,-1*_np.amin(truncated_test_statistic[i,:])]
#                soln[i,:], success = _leastsq(threshold_errfunc, p0, 
#                                        args=(truncated_test_statistic[i,:],truncated_oneminuscdf))
#                if success !=1 and success !=2 and success !=3 and success!=4:
#                    fail.append(i+1)
#                threshold_set[i,j] = estimated_threshold(soln[i,:],confidence)
#     
#    threshold = _np.mean(threshold_set,axis=1)
#   
#    return threshold