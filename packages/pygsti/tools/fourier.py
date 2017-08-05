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

#### Full time-series analysis with minimal user specification
def spectral_analysis(x, N, outcomes, test_thresholds, expected_power, filter_confidence=0.99, 
                      global_threshold=None, sequence_sets=None, sequence_sets_thresholds=None):
    """
    Implements a spectral analysis on time-series data from a set of circuits (e.g., time-stamped
    GST data). The analysis assumes a fixed time-step between each measurement outcome in the
    time-series for each gate sequence.
    
    Parameters
    ----------
    x : dict 
        The full dataset as a dictionary, with the key normally the gate sequence, and the values always
        a string of measurement outcomes. Note that time-stamps are not required, as the analysis assumes
        that the time-step between each measurement outcome in each string is the same (and that this time-step
        is the same for all gate sequences).
            
    N : int,
        The number of datapoints for each gate sequence, which must be the same for all sequences.
            
    test_thresholds : 
        The thresholds for the ....
    
    filter_confindence : float, optional
        
    global_threshold : float, optional
            A threshold value for thresholding the global averaged power spectrum.
            
    sequence_sets : dict, optional
        A dictionary of lists. The labels are names for sets of ..... . The values are
        the gate sequences in that set. An averaged DCT is returned for each of these sets
        of gate sequences.
        
    sequence_sets_threholds : dict, optional
        A dictionary of floats. Threshold values ....
        
    Returns
    -------
    dict
        A dictionary containing a range of spectral-analysis results. This includes drift-detection
        test outcomes and statistics, drift probability reconstructions, power spectra and a global
        averaged power spectra.
       
    """
    
    results = {}
    results['input'] = {}
    results['input']['data'] = x
    results['input']['N'] = N
    results['input']['outcomes'] = outcomes
    results['input']['test_thresholds'] = test_thresholds
    results['input']['expected_power'] = expected_power
    results['input']['filter_confidence'] = filter_confidence
    results['input']['global_threshold'] = global_threshold
    results['input']['sequence_sets'] = sequence_sets
    results['input']['power_estimator_index'] = power_estimator_index
    results['input']['sequence_sets_thresholds'] = sequence_sets_thresholds
    
    
    results['individual'] = {}
    results['sets'] = {}
    results['global'] = {}
    results['thresholds'] = {}
    
    # The filter threshold. Assuming fixed N, need only be calculated once
    individual_outcome_filter_threshold = _chi2.isf(1-filter_confidence**(1/N),1)
    averaged_outcome_filter_threshold = _chi2.isf(1-filter_confidence**(1/N),len(outcomes)-1)/(len(outcomes)-1)
    results['thresholds']['individual_outcome_filter_threshold'] = individual_outcome_filter_threshold
    results['thresholds']['averaged_outcome_filter_threshold'] = averaged_outcome_filter_threshold
    
    if power_estimator_index is None:
        power_estimator_index = int(_np.ceil(N/2.))
     
    # Create 0,1 datasets for all measurement outcomes.
    for key in x.keys():
        results['individual'][key] = {}
        results['individual'][key]['data'] = {}
        for i in outcomes:
            results['individual'][key]['data'][i] = _np.zeros(N,int)
            results['individual'][key]['data'][i][x[key] == i] = 1
    
    # Main loop for drift tests and Fourier filter
    for key in x.keys():

        results['individual'][key]['modes'] = {}
        results['individual'][key]['unconstrained_filtered_data'] = {}
        results['individual'][key]['constrained_filtered_data'] = {}
        results['individual'][key]['filtered_data'] = {}
        results['individual'][key]['summed_power'] = {}
        results['individual'][key]['test_outcomes'] = {}
        results['individual'][key]['number_tests_positive'] = {}
        results['individual'][key]['total_number_tests_positive'] = 0
        results['individual'][key]['power_spectrum'] = _np.zeros(N)
        results['individual'][key]['power_per_N_estimate_v1'] = {}
        results['individual'][key]['power_per_N_estimate_v2'] = {}
        results['individual'][key]['power_per_N_estimate_v1']['total'] = 0.
        results['individual'][key]['power_per_N_estimate_v2']['total'] = 0.
        results['individual'][key]['filtered_data_std'] = {}
        results['individual'][key]['p_value'] = {}
        
        for i in outcomes:
                # Calculate the DCT modes
                y = results['individual'][key]['data'][i]                       
                p = _np.mean(y)
                if p == 1 or p == 0:
                    modes =  _np.zeros(len(y),float)
                    results['individual'][key]['modes'][i] = modes
                    results['individual'][key]['unconstrained_filtered_data'][i] = y
                    results['individual'][key]['constrained_filtered_data'][i] = y
                    results['individual'][key]['filtered_data'][i] = y
                    results['individual'][key]['summed_power'][i] = _np.zeros(N)
                    results['individual'][key]['test_outcomes'][i] = _np.zeros(num_tests,bool)
                    results['individual'][key]['number_tests_positive'][i] = 0
                    results['individual'][key]['total_number_tests_positive'] = 0
                    results['individual'][key]['power_per_N_estimate_v1'][i] =  0.
                    results['individual'][key]['power_per_N_estimate_v2'][i] =  0.
                    results['individual'][key]['p_value'][i] = 1.
                    
                else:
                    pvec = p * _np.ones(len(y),float)
                    modes = _dct((y - p)/(_np.sqrt(pvec * (1 - pvec))),norm='ortho')
                    results['individual'][key]['modes'][i] = modes.copy()
                
                    # Raw Fourier Filter
                    filtered_modes = modes.copy()
                    filtered_modes[filtered_modes**2 < individual_outcome_filter_threshold] = 0.
                    unconstrained_filtered_data = _idct(_np.sqrt(pvec * (1 - pvec))*filtered_modes,norm='ortho') + p
                    results['individual'][key]['unconstrained_filtered_data'][i] = unconstrained_filtered_data
                    
                    # Estimation restricted to [0,1] via a logistic transformation                                           
                    nu = min([1-p,p])
                    constrained_filtered_data = (p - nu + (2*nu)/(1 + _np.exp(-2*(unconstrained_filtered_data-p)/nu)))
                    results['individual'][key]['constrained_filtered_data'][i] = constrained_filtered_data   
                    
                    # Calculate summed power
                    summed_power =  _np.cumsum(_np.flip(_np.sort(modes**2),axis=0))
                    results['individual'][key]['summed_power'][i] = summed_power.copy()
                                        
                    # p-value
                    results['individual'][key]['p_value'][i] = (1 - _chi2.sf(summed_power[0],1))**N
                                     
                    # k-sparse drift tests
                    num_tests = len(test_thresholds)
                    results['individual'][key]['test_outcomes'][i] = _np.ones(num_tests,bool)
                    results['individual'][key]['test_outcomes'][i][summed_power[0:num_tests] < test_thresholds] =  False
                    num_positive = len(_np.ones(num_tests)[results['individual'][key]['test_outcomes'][i]])
                    results['individual'][key]['number_tests_positive'][i] = num_positive
                    results['individual'][key]['total_number_tests_positive'] += \
                                                    results['individual'][key]['number_tests_positive'][i]
                
                    #Power analysis                
                    estimated_power_v1 = (summed_power[power_estimator_index] - expected_power[power_estimator_index])* p * (1 -
                                                                                                                             p)
                    results['individual'][key]['power_per_N_estimate_v1'][i] =  estimated_power_v1 / N
                    results['individual'][key]['power_per_N_estimate_v1']['total'] += \
                                                    results['individual'][key]['power_per_N_estimate_v1'][i]
                    index = _np.argmax(summed_power - expected_power)    
                    estimated_power_v2 = (summed_power[index] - expected_power[index])* p * (1 - p)
                    results['individual'][key]['power_per_N_estimate_v2'][i] =  estimated_power_v2 / N
                    results['individual'][key]['power_per_N_estimate_v2']['total'] += \
                                                    results['individual'][key]['power_per_N_estimate_v2'][i]

                    # Add the power spectrum for outcome i to the single power spectrum
                    results['individual'][key]['power_spectrum'] += results['individual'][key]['modes'][i]**2
        
        # Calculate power spectrum and estimates averaged over all the outcomes
        results['individual'][key]['power_spectrum'] = results['individual'][key]['power_spectrum']/len(outcomes)
        results['individual'][key]['power_per_N_estimate_v1']['total'] = results['individual'][key]['power_per_N_estimate_v1']['total']/len(outcomes)
        results['individual'][key]['power_per_N_estimate_v2']['total'] = results['individual'][key]['power_per_N_estimate_v2']['total']/len(outcomes)
        
        # normalize the estimated p_i(t) so that they sum to one
        one_norm = 0.
        for j in outcomes:
            one_norm += results['individual'][key]['constrained_filtered_data'][j]
        for j in outcomes:   
            results['individual'][key]['filtered_data'][j] = results['individual'][key]['constrained_filtered_data'][j]/one_norm
        # caculate the std of each array of filtered data
        for j in outcomes:   
            results['individual'][key]['filtered_data_std'][j] = _np.std(results['individual'][key]['filtered_data'][j])
        
    # global averaged power spectrum
    results['global']['total_number_tests_positive'] = 0.
    power_spectrums = _np.zeros((N,len(x.keys())))  
    for k in range (0,len(x.keys())):
        power_spectrums[:,k] = results['individual'][x.keys()[k]]['power_spectrum'] 
        results['global']['total_number_tests_positive'] += results['individual'][x.keys()[k]]['total_number_tests_positive']
    results['global']['averaged_power_spectrum'] = _np.mean(power_spectrums,axis=1)

    # filtered global averaged power spectrum
    if global_threshold is not None:
        results['global']['filtered_averaged_power_spectrum'] = results['global']['averaged_power_spectrum'].copy()
        results['global']['filtered_averaged_power_spectrum'][results['global']['averaged_power_spectrum'] < global_threshold]=0.

    # averaged power spectrum for sequence_sets
    if sequence_sets is not None:
        for label in sequence_sets.keys():            
            averaged_power_spectrum = _np.zeros(N) 
            for key in sequence_sets[label]:
                averaged_power_spectrum += results['individual'][key]['power_spectrum']
            averaged_power_spectrum  = averaged_power_spectrum / len(sequence_sets[label])  
            results['sets'][label] = {}
            results['sets'][label]['averaged_power_spectrum'] = averaged_power_spectrum
                        
    return results


def global_bootstrap(num_spectra,N,num_outcomes,probs=None,bootstraps=500):
    
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

def DCT(x,null_hypothesis=None):
    """
    Returns the Type-II orthogonal discrete cosine transform of y where 
    
    y[k] = (x[k] - null_hypothesis[k])/sqrt(null_hypothesis[k]*(1-null_hypothesis[k])).
    
    If null_hypothesis is None, then null_hypothesis[k] is mean(x), for all k. This is
    with the exception that when mean(x) = 0 or 1 (when the above y[k] is ill-defined),
    in which case the zero vector is returned.
    
    Parameters
    ----------
    x : array
        Bit string, on which the normalization and discrete cosine transformation is performed.
        
    null_hypothesis : array, optional
        If not None, an array to use in the normalization before the DCT. If None, it is
        taken to be an array in which every element is the mean of x.
                
    Returns
    -------
    array
        The DCT modes described above.


    """
    if null_hypothesis is None:
        null_hypothesis = _np.mean(x)
        if null_hypothesis<=0 or null_hypothesis>=1:
            return _np.zeros(len(x))
        
    else:
        if min(null_hypothesis)<=0 or max(null_hypothesis)>=1:
            raise ValueError("All element of null_hypothesis should be in (0,1)")

    return _dct((x - null_hypothesis)/_np.sqrt(null_hypothesis * (1 - null_hypothesis)),norm='ortho')

def IDCT(modes,null_hypothesis):
    """
    Inverts the DCT function.
    
    Parameters
    ----------
    modes : array
        The fourier modes to be transformed to time-domain.
        
    null_hypothesis : array
        The null_hypothesis vector. For the IDCT it is not optional, and all
        elements of this array must be in (0,1).
        
    Returns
    -------
    array
        Inverse of the DCT function
        
    """
    if min(null_hypothesis)<=0 or max(null_hypothesis)>=1:
            raise ValueError("All element of null_hypothesis should be in (0,1)")
    
    return  _idct(modes,norm='ortho')*_np.sqrt(null_hypothesis * (1 - null_hypothesis)) + null_hypothesis


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
            test_statistic[:,i] = _np.cumsum(_np.flip(_np.sort(DCT(bs[:,i],null_hypothesis=null_hypothesis)**2),axis=0))
            

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




### Functions for various statistical tests.

def one_sparse_fixed_frequency_test(mode,confidence=0.95):
    """
    """

    if mode**2 > one_sparse_fixed_frequency_threshold(confidence=confidence):
        return True
    else:
        return False

def one_sparse_test(modes,confidence=0.95):
    """
    """
    threshold = one_sparse_threshold(len(modes),confidence=confidence)
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
        aux_out = one_to_k_sparse_threshold(null_hypothesis,k=k,confidence=confidence,
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

### Probability function reconstruction + error bar finding functions 

def renormalizer(x):
    
    p = _np.mean(x)
    nu = min([1-p,p])                       
    return p - nu + (2*nu)/(1 + _np.exp(-2*(estimated_x-p)/nu))

def low_pass_filter(data,max_freq = None):
    if max_freq is None:
        max_freq = int(len(data)/10)
    modes = _dct(data,norm='ortho')
    for i in range(max_freq,len(data)):
        modes[i] = 0.0
    out = _idct(modes,norm='ortho')
    return out

def DCT_filter(x,confidence=0.99,null_hypothesis=None,method='chi2',max_keep_modes=None,
               return_aux=False):
    
    if max_keep_modes is None:
        max_keep_modes = len(x)
    
    x_mean = _np.mean(x)
    if x_mean <= 0 or x_mean >= 1:
        return x, 0
   
    else:
        if null_hypothesis is None:
            null_hypothesis = x_mean*_np.ones(len(x),float)
            
        y = DCT(x,null_hypothesis=null_hypothesis)
        if method=='chi2':
            threshold = one_sparse_threshold(len(x),confidence)
        else:
            threshold = one_to_k_sparse_threshold(null_hypothesis, 
                                                  k=1,confidence=confidence,method=method)            
        #y[y**2 < threshold] = 0
        
        #
        #
        # PROVIDE A METHOD HERE FOR KEEPING ONLY THE LARGEST MODE ABOVE THRESHOLD.
        #
        #
        
        # Reduces power by 1 --- should this be included or not? If to use, comment
        # out the y[y**2 < threshold] = 0 line.
        power = y**2
        power[power < threshold] = 0.
        power = power - 1.
        power[power < 0] = 0.
        y = _np.sign(y)*_np.sqrt(power)
        
        filtered_x = IDCT(y,null_hypothesis)
 
        if return_aux:
            out_aux = {}
            out_aux['estimate'] = filtered_x
            out_aux['num_modes'] = _np.count_nonzero(y)
            out_aux['frequencys'] = _np.flatnonzero(y)
            out_aux['confidence'] = confidence
            out_aux['null_hypothesis'] = null_hypothesis
            out_aux['method'] = method
            out_aux['max_keep_modes'] = max_keep_modes
            return out_aux
        else:
            return filtered_x
        
def DCT_iterative_filter(x,confidence=0.99,null_hypothesis=None,method='chi2',max_iteration=None,
                         max_keep_modes=None,estimate_signal_power=True,return_aux=False):
    
    if max_iteration is None:
        max_iteration = len(x)
    
    aux_out = {}
    additional_modes = 1
    iteration_number = 0
    modes_kept = 0
    
    if null_hypothesis is None:
        null_hypothesis = _np.mean(x)
    
    filtered_x = null_hypothesis
    
    while additional_modes > 0 and iteration_number < max_iteration:
        iteration_number = iteration_number + 1
        aux_out[iteration_number] = DCT_filter(x,confidence=confidence,null_hypothesis=filtered_x,
                            method=method,max_keep_modes=max_keep_modes,return_aux=True)
            
        filtered_x = aux_out[iteration_number]['estimate']
        additional_modes = aux_out[iteration_number]['num_modes']
        modes_kept = modes_kept + additional_modes
    
    #print('Iterations:', iteration_number,'Modes kepts:', modes_kept, end='. ')
    
    if estimate_signal_power:
        # Power in estimated signal. This should be scaled down *if* the filtered x is not already
        # scaled down.
        baseline_power = _np.sum((filtered_x-null_hypothesis)**2)
        residual_power = estimate_residual_power(x, filtered_x)
        signal_power = baseline_power + residual_power['estimated_power_2']
    
    else:
        signal_power = None
    
    if return_aux:
        aux_out['estimated_signal_power'] = signal_power
        aux_out['estimated_residual_power'] = residual_power['estimated_power_2']
        aux_out['filtered_signal_power'] = baseline_power
        aux_out['power_estimation_aux'] = residual_power
        aux_out['estimate'] = filtered_x
        aux_out['num_modes'] = modes_kept
        aux_out['frequencys'] = aux_out[iteration_number]['frequencys'] # this is probably wrong
        aux_out['confidence'] = confidence
        aux_out['null_hypothesis'] = null_hypothesis
        aux_out['method'] = method
        aux_out['max_keep_modes'] = max_keep_modes
        aux_out['iterations'] = iteration_number
        
        return aux_out
        
    else:
        return filtered_x

def expected_power_statistics(null_hypothesis,N=10000,return_aux=False):
    """
    To be calculated
    """
    n = len(null_hypothesis)
          
    bs = _np.array([_np.random.binomial(1,p,size=N) for p in null_hypothesis])
    power = _np.zeros((n,N),float)
    summed_power = _np.zeros((n,N),float)

    for i in range (0,N):
        # the sorted power array, from highest to lowest.
        power[:,i] = _np.flip(_np.sort(_dct(bs[:,i]-null_hypothesis,norm='ortho')**2),axis=0)
        summed_power[:,i] = _np.cumsum(power[:,i])
                            

    expected_values = _np.mean(summed_power,axis=1)
    std = _np.std(summed_power,axis=1)
    
    max_power = _np.zeros(N,float)
    normed_max_power = _np.zeros(N,float)
    for i in range (0,N):
        max_power[i] = _np.amax(summed_power[:,i] - expected_values) 

    expected_max_power = _np.mean(max_power)
    max_power_std = _np.std(max_power)
    
    out = {}
    # summed power array statistics
    out['expected_summed_power'] = expected_values
    out['summed_power_std'] = std
    out['sampled_summed_power'] = summed_power
    
    # max deviation from expected sum power statistics
    out['expected_max'] = expected_max_power
    out['max_std'] = max_power_std
    out['sampled_max'] = max_power

    return out

def estimate_residual_power(data, null_hypothesis, expected_power=None, cut_off=False, return_aux=False):
    
    # The DCT modes are *not* standardized to have unit-variance (under null-hypothesis)
    modes = _dct(data-null_hypothesis,norm='ortho')
    summed_power = _np.cumsum(_np.flip(_np.sort(modes**2),axis=0))
    
    if expected_power is None:
        power_stats = expected_power_statistics(null_hypothesis,N=10000)
    else:
        power_stats = expected_power
    expected_values = power_stats['expected_summed_power']
    summed_power_std = power_stats['summed_power_std']
    
    expected_max = power_stats['expected_max']
    max_std = power_stats['max_std']
    
    index = _np.argmax(summed_power - expected_values)
    
    #estimated_power_1 = summed_power[index] - expected_values[index] - expected_max
    estimated_power = summed_power[index] - expected_values[index]
    
    out = {}
    #out['estimated_power_1'] = estimated_power_1
    out['estimated_power'] = estimated_power
    #out['estimated_power_1_error_bars'] = 2*max_std
    out['estimated_power_error_bars'] = 2*summed_power_std[index]

    if cut_off:
        if estimated_power < 0:
            estimated_power = 0.
    if return_aux:
        power_stats['observed_summed_power'] = summed_power
        return out, power_stats
    else:
        return out
    
def estimate_power(modes, data_mean, expected_power=None, cut_off=True):
    
    # The DCT modes are *not* standardized to have unit-variance (under null-hypothesis)
    summed_power = _np.cumsum(_np.flip(_np.sort(modes**2),axis=0))
    
    if expected_power is None:
        power_stats = expected_power_statistics(null_hypothesis=data_mean*_np.ones(N),N=10000)
    else:
        power_stats = expected_power
    expected_values = power_stats['expected_summed_power']
    summed_power_std = power_stats['summed_power_std']
    
    expected_max = power_stats['expected_max']
    max_std = power_stats['max_std']
    
    index = _np.argmax(summed_power - expected_values)
    
    estimated_power = summed_power[index] - expected_values[index]
    
    out = {}
    out['estimated_power'] = estimated_power
    out['estimated_power_error_bars'] = 2*summed_power_std[index]

    if cut_off:
        if estimated_power < 0:
            estimated_power = 0.

    return out


def k_sparse_signal_generator(n,power,k,freq_cutoff=None,base = 0.5,verbose=True):
    if freq_cutoff is None:
        freq_cutoff = n-1
    amp_per_mode = _np.sqrt(power/k)
    all_modes = _np.arange(1,freq_cutoff+1)
    sampled_modes = _np.random.choice(all_modes, size=k, replace=False, p=None)
    
    modes = _np.zeros(n,float)
    for i in range (0,k):
        modes[sampled_modes[i]] = amp_per_mode*(-1)**_np.random.binomial(1,0.5)
        
        x =  _idct(modes,norm='ortho')    
    x = x + base
    if verbose:
        if _np.amax(x) > 1 or _np.amin(x) < 0:
            print("--- Elements of generated signal outside [0,1]: Run again, reduce power, or change `base` ---")
    return x

from scipy import convolve
from scipy import ones

def moving_average(sequence, width=100):
    seq_length = len(sequence)
    base = convolve(ones(seq_length), ones((int(width),))/float(width), mode='same')
    signal = convolve(sequence, ones((int(width),))/float(width), mode='same')
    return signal/base 