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

from . import fourier_utils
from . import fourier_thresholds

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
    
    power_estimator_index = int(N/10)
    num_tests = len(test_thresholds)
    
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
     
    # Create bit string datasets for all measurement outcomes.
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