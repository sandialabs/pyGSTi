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

from . import drifttools as _dtls
from . import driftdetectionthresholds as _dthresholds
from . import driftdetectiontests as _dtests

def frequencies_from_timestep(timestep,T):
     
    return _np.arange(0,T)/(2*timestep*T)

class BasicDriftResults(object):

    def __init__(self):
        
        self.data = None
        self.number_of_sequences = None
        self.number_of_timesteps = None
        self.number_of_qubits = None
        self.number_of_counts = None        
        self.timestep = None
        
        self.confidence = None
        self.frequencies = None
        
        self.all_detected = None
                
        self.global_power_spectrum = None
        self.global_significance_threshold = None                
        self.global_detected = None
                
        self.individual_modes = None
        self.individual_power_spectrum = None
        self.individual_max_powers = None
        self.individual_max_power_pvalues = None
        self.individual_max_power_threshold = None
        self.individual_drift_detected = None
        self.individual_reconstruction = None
        self.individual_reconstruction_power_per_time = None

def do_basic_drift_characterization(indata,counts,timestep=None,confidence=0.95,verbosity=2):
    """
    
    """
    # data shape should be, except when 
    #(number of sequences x num_qubits x timesteps)
    
    data_shape = _np.shape(indata)
    
    # Check that the input data is consistent with being counts in an array of dimension
    # (number of sequences x num_qubits x timesteps), or (number of sequences x timesteps).
    assert(len(data_shape) == 2 or len(data_shape) == 3), "Data format is incorrect!"
    
    # If ....
    if len(data_shape) == 2:
        data = _np.zeros((data_shape[0],1,data_shape[1]),float)
        data[:,0,:] = indata.copy()
    else:
        data = indata.copy()
    
    # Extract the number of sequences, num_qubits and timesteps from the shape of the input
    data_shape = _np.shape(data)    
    num_sequences = data_shape[0]
    num_qubits = data_shape[1]
    num_timesteps = data_shape[2] 
    
    
    if verbosity > 0:
        if num_timesteps <=50:
            string = "*** Warning: certain approximations used within this function may be unreliable when the"
            string += " number of timestamps is too low. The statistical significance"
            string += " thresholds may be inaccurate ***"   
            string +=  "\n"
            print(string)
                    
    # If the timestep is not provided, frequencies are given as integers between 0 and 1 - number of timesteps
    if timestep is None:
        frequencies = _np.arange(0,num_timesteps)
        
    # If the timestep is provided, frequencies are given in Hertz.
    else:
        frequencies = _np.zeros(num_timesteps,float)
        frequencies = frequencies_from_timestep(timestep,num_timesteps)
        
    # A bool to flag up if any drift has been detected
    was_drift_detected = False
    global_detected = False
    
    # Calculate the power spectra for all the sequencies
    modes = _np.zeros(data_shape,float)
    power_spectrum = _np.zeros(data_shape,float)
    
    for s in range(0,num_sequences):
        for q in range(0,num_qubits):               
            modes[s,q,:] = _dtls.DCT(data[s,q,:],counts=counts)
            
    power_spectrum = modes**2
    
    # Calculate the power spectrum averaged over sequencies.
    average_power_spectrum = _np.mean(power_spectrum,axis=0)
    
    # Analyze the average power spectrum, and find significant frequencies.
    threshold = _chi2.isf(1-confidence**(1/num_timesteps),num_sequences)/num_sequences
    drift_frequencies_indices = _np.zeros((num_qubits,num_timesteps),bool)
    drift_frequencies_indices[average_power_spectrum > threshold] = True 

    drift_frequencies = _np.zeros((num_qubits,num_timesteps),float)
    for q in range(0,num_qubits):
        drift_frequencies[q,:] = frequencies.copy()
    drift_frequencies = drift_frequencies[drift_frequencies_indices]
    
    # Record if the averaged power spectrum detects drift.
    if len(drift_frequencies) > 0:
        was_drift_detected = True
        global_detected = True
        
    # Analyze the individual sequences.
    max_power_threshold = _chi2.isf(1-confidence**(1/num_timesteps),1)
    max_powers = _np.zeros((num_sequences,num_qubits),float)
    max_power_pvalues = _np.zeros((num_sequences,num_qubits),float)
    drift_detected = _np.zeros((num_sequences,num_qubits),bool)
    
    raw_estimated_modes = modes.copy()
    probability_estimates = _np.zeros(data_shape,float)
    null = _np.zeros(data_shape,float)
    for s in range(0,num_sequences):
        for q in range(0,num_qubits):
            max_powers[s,q] = _np.max(power_spectrum[s,q,:])
            max_power_pvalues[s,q] = (1 - _chi2.sf(max_powers[s,q],1))**num_timesteps
            null[s,q,:] = _np.mean(data[s,q,:])*_np.ones(num_timesteps,float)/counts
            
            if max_powers[s,q] > max_power_threshold:
                was_drift_detected = True
                drift_detected[s,q] = True
                raw_estimated_modes[power_spectrum<max_power_threshold] = 0.
                
                # Here divide by counts, as when we invert the DCT we would -- in the noise-free case -- get something
                # between 0 and counts, rather than 0 and 1
                probability_estimates[s,q,:] = _dtls.IDCT(raw_estimated_modes[s,q,:], null_hypothesis=null[s,q,:], 
                                                          counts=counts)/counts
                probability_estimates[s,q,:] = _dtls.renormalizer(probability_estimates[s,q,:],method='logistic')
            else:
                probability_estimates[s,q,:] = null[s,q,:]
                
    reconstruction_power_per_time = _np.sum((probability_estimates-null)**2,axis=2)/num_timesteps
    
    #if num_qubits == 1:
    #    if len(data_shape) == 2:
    #
    #        drift_frequencies = drift_frequencies[0,:]
    #        average_power_spectrum =  average_power_spectrum[0,:]  
    #    
    #        max_powers = max_powers[:,0]
    #        max_power_pvalues = max_power_pvalues[:,0]
    #        max_power_threshold = max_power_threshold
    #        drift_detected = drift_detected[:,0]
    #        power_spectrum = power_spectrum[:,0,:]
    #        probability_estimates = probability_estimates[:,0,:]
    #        reconstruction_power_per_time = reconstruction_power_per_time[:,0]
        
    sequence_results = {}
    sequence_results['reconstruction'] = probability_estimates
    sequence_results['reconstruction_power_per_time'] = reconstruction_power_per_time
    
    global_results = {}
    global_results['drift_frequencies'] = drift_frequencies
    global_results['average_power_spectrum'] = average_power_spectrum
    global_results['significance_threshold'] = threshold
    
    # Initialize an empty results object.
    results = BasicDriftResults()
    
    # Record input information, and things fairly trivially derived from it
    results.data = data
    results.number_of_sequences = num_sequences
    results.number_of_timesteps = num_timesteps
    results.number_of_qubits = num_qubits
    results.number_of_counts = counts     
    results.timestep = timestep
        
    results.confidence = confidence
    results.frequencies = frequencies.copy()
    
    # Record flag
    results.all_detected = was_drift_detected
    
    results.global_power_spectrum = average_power_spectrum
    results.global_significance_threshold = threshold           
    results.global_detected = global_detected
    results.global_drift_frequencies = drift_frequencies
    
    results.individual_modes = modes
    results.individual_power_spectrum = power_spectrum
    results.individual_max_powers = max_powers
    results.individual_max_power_pvalues = max_power_pvalues
    results.individual_max_power_threshold = max_power_threshold
    results.individual_drift_detected = drift_detected
    results.individual_reconstruction = probability_estimates
    results.individual_reconstruction_power_per_time = reconstruction_power_per_time
    
    #if verbosity > 1:
    #    if was_drift_detected:
    #        print("Drift has been detected!")
    #    else:
    #        print("Drift has NOT been detected!")
    
    return results


#def spectral_analysis_singlesequence():
#    
#    return 

#### Full time-series analysis with minimal user specification

#class MultisequenceDriftDetectionResults(object):
#    """
#    An object...
#    """
#    def __init__(self, data, populate=True):
#        
#        counts_per_timestep =  _np.sum(data[0,0,0,:])
#        number_of_sequences = len(data[:,0,0,0])
#        number_of_num_qubits = len(data[0,:,0,0])
#        number_of_timesteps = len(data[0,0,:,0]) 
#        number_of_outcomes = len(data[0,0,0,:])
#        
#        self.number_of_sequences = number_of_sequences
#        self.number_of_timesteps = number_of_timesteps
#        self.number_of_outcomes = number_of_outcomes
#        self.number_of_num_qubits = number_of_num_qubits
#        self.counts_per_timestep = counts_per_timestep
#        
#        self.data = data
#        
#        if populate:
#            print("TO DO")
#            #
#            # Here put code to calculate everything
#            #
#        else:
#            self.modes = _np.zeros(_np.shape(data),float)
#            self.power_spectrum = _np.zeros(_np.shape(data),float)
#            self.outcome_averaged_power_spectrum = #_np.zeros((number_of_sequences,number_of_num_qubits,number_of_timesteps),float)
#            self.global_power_spectrum = _np.zeros((number_of_num_qubits,number_of_timesteps),float)
#        
#class SinglesequenceDriftDetectionResults(object):
#    """
#    An object...
#    
#    """
#    def __init__(self, data, timestamps=None, timestep=None, sequence=None):
#        """
#        
#        Data : ...
#            Currently can only be an array, but will be allowed to be a DataSet object
#        
#        timestamps : optional
#            Allows optional input of timestamps
#            
#        seqlabels : optional
#            Allows input of the sequence associated with the data.
#            
#        """
#       counts_per_timestep =  int(_np.sum(data[0,0,:]))
#        number_of_num_qubits = len(data[:,0,0])
#        number_of_timesteps = len(data[0,:,0]) 
#        number_of_outcomes = len(data[0,0,:])
#        
#        self.number_of_timesteps = number_of_timesteps
#        self.number_of_outcomes = number_of_outcomes
#        self.number_of_num_qubits = number_of_num_qubits
#        self.counts_per_timestep = counts_per_timestep
#        
#        #
#        # Todo: check whether the input is TDDataSet, and if so manipulate 
#        # as appropriate
#        #
#        # Currently it assumes it is of the array format.
#        #
#        self.data = data
#        self.timestamps = timestamps
#        self.sequence = sequence
#        
#        self.modes =  _np.zeros(_np.shape(data),float)
#        self.power_spectrum = _np.zeros(_np.shape(data),float)
#        self.ordered_power_spectrum = _np.zeros(_np.shape(data),float)
#        self.outcome_averaged_power_spectrum = _np.zeros((number_of_num_qubits,number_of_timesteps),float)
#        
#        #
#        #
#        #
#        
#
#        # These are left as None if there are no timestamps.
#        self.has_equal_spacing = None
#        self.frequencies = None
#        
#        # This is overwritten if the timestamps are also provided.
#        if timestep is not None:
#            self.timestep = timestep
#        
#        if self.timestamps is not None:
#            temp = self.timestamps[1:]
#            time_difference = temp - self.timestamps[:number_of_timesteps-1]
#            self.timestep = _np.mean(time_difference)
#            
#           # This choice is fairly arbitrary, but there is no specific tolerance that 
#            # we know is sufficient.
#            self.has_equal_spacing = _np.allclose(time_difference, self.timestep*_np.ones(number_of_timesteps-1), 
#                                                rtol=1e-02, atol=0)
#        
#        
#        
#        
#        if _np.min(self.data) < 0:
#            self.has_padded_data = True
#            self.unpadded_data = data.copy()
#            #
#            # Todo: code that pads missing data with randomly sampled data.
#            #
#        else:
#            self.has_padded_data = False
#        
#        
#        if self.timestamps is not None:
#            #
#            # Todo: calculate the frequencies in Hz.
#            #
#            self.frequencies = 0
#            
#        # Initialize an empty tests object
#        self.test_results = CompositeStatTest()
#        
#        for q in range(0,number_of_num_qubits):
#            for m in range(0,number_of_outcomes):                
#                self.modes[q,:,m] = _dtls.DCT(data[q,:,m],counts=counts_per_timestep)        
#            
#            self.power_spectrum[q,:,:] = self.modes[q,:,:]**2
#            self.outcome_averaged_power_spectrum[q,:] = _np.mean(self.power_spectrum[q,:,:],axis=1)
#            
#        self.ordered_power_spectrum = _np.flip(_np.sort(self.power_spectrum,axis=1),axis=1)
#  
#              
#        #def test_largest_powers(summed_powers=[1,2,3,4],test):
#        #    
#        #    
#        # 
#        # Todo: allow calling additional tests that have not yet been run.
#        # Must flag up something that says that test compensation will nolonger be valid, and/or flags
#        # up that additional tests have been run so statistical significance is dubious.
#        #
#
#class CompositeStatTest():
#    #
#    # ToDo: fill this in,
#    #
#    def __init__(self, test_list=None):
#        
#        self.tests = {}
#        
#        if test_list is not None:
#            for test in test_list:            
#                self.add_unimplemented_test(test)
#                
#    def add_unimplemented_test(self, name, statistic, pvalue=None, cdf=None, cdfparams=None):
#        """
#        Either the pvalue or cdf + cdfparams must be provided! However, both are not essential.
#        """
#        self.tests[name] = SingleStatTest(name, statistic, pvalue, cdf, cdfparams)
#        
#    
#    def implement_all_tests(self, confidence, multitest_correction = 'bonferonni'):
#        
#        self.x = 10
#        # Todo : here I need to code up how to run all of the tests. When bonferonni or equivalent is
#        # used, the threshold for that test should be stored (in other cases the threshold doesn't make
#        # as much sense, but could still be stored).
#    
#class SingleStatTest(object):
#    """
#    An object...
#    
#    """
#    def __init__(self, name=None, statistic=None, pvalue=None, cdf=None, cdfparams=None, 
#                 confidence=None, threshold=None, reject_null=None):
#        
#        self.name = name 
#        self.statistic = statistic
#        self.pvalue = pvalue
#        self.cdf = cdf
#        self.cdfparams = cdfparams
#        self.confidence = confidence
#        self.threshold = threshold
#        self.reject_null = reject_null
#        
#def default_tests(which_tests):
#    """
#    Returns various "standard" drift detection testing options. Note that
#    none of these options have any fundamental significance, and they are
#    mearly a range of options that are likely to be used often enough that
#    a short-hand for these settings is useful.
#    
#    """
#    
#    #
#    # Todo: write this function.
#    #
#    sumpowers_std = [1,2,3,4]
#    bartlett_std = [(1,1),(2,1),(4,1),(8,1),(16,1),(32,1)]
#       
#    if which_tests == 'maxpower-sumpowers-bartlett':
#        
#        tests = {'maxpower':None,'sumpowers':sumpowers_std,'bartlett':bartlett_std}
#        
#    if which_tests == 'maxpower':
#        
#        tests = {'maxpower':None}
#        
#    if which_tests == 'maxpower-bartlett':
#        
#        tests = {'maxpower':None,'bartlett':bartlett_std}
#            
#    return tests
#                
def singlesequence_drift_analysis(data, 
                                  timestamps=None,
                                  timestep=None,
                                  sequence=None, 
                                  possible_outcomes=[0,1], 
                                  tests='default', 
                                  test_outcome_averaged_spectrum = True, 
                                  test_statistic_distributions='bootstrap',
                                  number_of_bootstraps = 50000,
                                  test_confidence=0.95, 
                                  multitest_compensation = 'none',
                                  estimate_probabilities=True, 
                                  filter_confidence='default', 
                                  verbosity=1):
    
    #
    # Here convert data to a (num_qubits x timesteps x possible outcomes) numpy array.
    # Add checks that it is of a suitable format.
    #
    # Currently, the code assumes this format, as x is just copy of data.
    #
    x = data.copy()
    num_qubits = len(data[:,0,0])
    timesteps = len(data[0,:,0])
    outcomes = len(data[0,0,:])
    counts = _np.sum(data[0,0,:])
    
    # Create a results object, which performs the DCT, but is otherwise largely empty
    results = SinglesequenceDriftDetectionResults(x,timestamps=timestamps,timestep=timestep,sequence=sequence)
    
    # If a default set of tests has been specified, then populate the tests dictionary
    if type(tests) == unicode:
        
        if tests == 'default':
            # My current opinion on what the best default option is.
            tests = 'maxpower-sumpowers-bartlett'
            
        tests = default_tests(tests)  
        
    else:
        assert(type(tests) == dict), "If `tests` is not a standard unicode option, it must be a dict!"
        tests = tests
    
    which_tests = list(tests.keys())
    
    # Have not yet added other methods, but function is designed to be able to deal with them
    assert(test_statistic_distributions == 'bootstrap'), "Only bootstrap is currently supported!"
    
    if test_statistic_distributions == 'bootstrap':
        
        if verbosity > 0:
            print("  *Test statistic distributions to be calculated via bootstrap*")
            print("  Implementing {} bootstraps...".format(number_of_bootstraps),end='')
            
        # This is .... todo:what is it?
        p = np.mean(x,axis=1)
        
        # These are bootstraps of the unsorted DCT modes, with the first index the bootstrap number and
        # the remaining indices the same as those for 'data'
        mode_bs = _dthresholds.singlesequence_bootstrap(p, timesteps, counts, number_of_bootstraps)
        
        # Convert to powers, as that is all we use
        power_bs = mode_bootstraps**2
        
        # If we are only interested in looking at the outcome-averaged spectrum, we average over outcomes.
        if test_outcome_averaged_spectrum:
            power_bs = _np.mean(power_bs,axis=3)
        
        # Because multiple test statistics use order power vector, create this in advance.
        ordpower_bs = np.flip(np.sort(power_bs,axis=2),axis=2)
        
        if verbosity > 0:
            print("  Complete.")
            print("  Calculating test statistic distributions from the bootstraps...",end='')
        
        # These are temp dicts to hold the cdfs for all possible tests we might be doing
        sumpowers_cdf = {}
        sumpowers_cdf_params = {}
        maxpower_cdf = {}
        maxpower_cdf_param = {}
        bartlett_cdf = {}
        bartlett_cdf_params = {}
        
        # Consider each qubit in turn, and find the cdfs for the test statistics from the bootstrapped data.
        for q in range(0,num_qubits):
            
            # If we are not averaging over outcomes, then we need to find the cdf for all outcomes (generally they won't
            # be the same, as -- when there are more then two outcomes -- some outcome probabilities can be much closer
            # to 0 or 1 than other outcome probabilities).
            if not test_outcome_averaged_spectrum:
                for o in range(0,outcomes):
                    if 'sumpowers' in which_tests:
                        sumpowers_cdf[q,o], sumpowers_params[q,o] = _dthresholds.bootstrap_sumpowers_cdf(ordpower_bs[:,q,:,o], 
                                                                                       tests['sumpowers'])
                    if 'maxpower' in which_tests:
                        maxpower_cdf[q,o], maxpower_params[q,o] = _dthresholds.bootstrap_maxpower_cdf(ordpower_bos[:,q,:,o])
                
                    if 'bartlett' in which_tests:
                        bartlett_cdf[q,o], bartlett_params[q,o] = _dthresholds.bootstrap_sumpowers_cdf(ordpower_bs[:,q,:,o],
                                                                                     tests['bartlett'])
            
            # If we are averaging over outcomes, then we only need to find the cdf for the test statistics on the
            # outcome-averaged power spectrum
            if test_outcome_averaged_spectrum:
                if 'sumpowers' in which_tests:
                    sumpowers_cdf[q], sumpowers_params[q] = _dthresholds.bootstrap_sumpowers_cdf(ordpower_bs[:,q,:], 
                                                                                       tests['sumpowers'])
                if 'maxpower' in which_tests:
                    maxpower_cdf[q], maxpower_params[q] = _dthresholds.bootstrap_maxpower_cdf(ordpower_bos[:,q,:])
                
                if 'bartlett' in which_tests:
                    bartlett_cdf[q], bartlett_params[q] = _dthresholds.bootstrap_sumpowers_cdf(ordpower_bs[:,q,:],
                                                                                     tests['bartlett'])
        if verbosity > 0:
            print("   Complete.")
        
    # Create empty tests, including the cdfs. Note that this is done outside the bootstrap loop so that we can
    # have other methods for finding the cdf / cdf params   
    if verbosity > 0:
        print(" Implementing the specified statistical tests...")
    
    for q in range(0,num_qubits):
        
        if not test_outcome_averaged_spectrum:
            if 'sumpowers' in which_tests:
                statistic = 0# FUNCTION TO PUT HERE (x)
                for k in tests['sumpowers']:        
                    results.test_results.add_unimplemented_test((q,'sumpowers',k), statistic, cdf=sumpowers_cdf[q],
                                                     cdfparams=sumpowers_params[q])
            if 'maxpower' in which_tests:
                statistic = 0# FUNCTION TO PUT HERE (x)
                results.test_results.add_unimplemented_test((q,'maxpower'), statistic, cdf=maxpower_cdf[q], 
                                                     cdfparams=maxpower_params[q])
    
            if 'bartlett' in which_tests:
                statistic = 0# FUNCTION TO PUT HERE (x)
                for k in tests['bartlett']: 
                    results.test_results.add_unimplemented_test((q,'bartlett',k[0],k[1]), statistic, cdf=bartlett_cdf[q],
                                                     cdfparams=bartlett_params[q])
                    
        if test_outcome_averaged_spectrum:
            print('To do!')
    
    # Implement the tests!
    #
    # Todo : probabaly need to adjust this -- it isn't accounting for any of the settings in the function
    #
    results.test_results.implement_all_tests(confidence, multitest_correction = 'bonferonni')
        
    
    #
    # Here implement the filter
    #
    
    return results
        
def multisequence_drift_analysis(data, possible_outcomes=[0,1], individual_tests='default_convolve', 
                   global_tests='default_convolve', individual_test_thresholds='chi2', global_test_thresholds='chi2',
                   individual_test_confidence=0.95, multitest_compensation = 'none', global_test_confidence='default',
                   estimate_probabilities=True, enhanced_thresholding=True, filter_confidence='default',
                  verbosity=1):
    
    """
    data : array
        A timeseries dataset as an array, a dictionary, or a DataSet object. The form that this object must
        take depends on whether multi_qubit is True or False (BUT I HAVE NOW DELETED THIS VARIABLE, AND IT
        WILL BE INFERED FROM THE SHAPE OF DATA).
        
    possible_outcomes : list
        The possible outcomes for each measurement (This might not be necessary if data is a DataSet).
        
    individual_tests: str or list
        Specifies the tests to implement on the timeseries associated with each sequence.
        
    global_tests : str or list
        Specifies the tests to implement on the averaged power spectrum
        
    individual_test_thresholds : str or list
        Specifies the method for calculating the thresholds for all of the individual tests, or is a list 
        of threshold values.
        
    individual_test_thresholds : str or list
        Specifies the method for calculating the thresholds for all of the global tests, or is a list of 
        threshold values.
        
    individual_test_confidence : float
        The test confidence to use. Is not explicitly used if test thresholds have been provided, and
        should be set to the value ....
        
    multitest_compensation : str
        Specifies how to compensate for the increased false-positive rate due to implementing multiple
        tests.
        
    global_test_confidence : float or 'default'
        The test confidence to use. Is not explicitly used if test thresholds have been provided, and
        should be set to the value .... If 'default', then it defaults to the same
        value as the individual test confidence level.
        
    estimate_probabilities : bool
        Specifies whether to implement the Fourier filtering
        
    enhanced_thresholding : bool
        Specifies whether to use the drift detection on the global power spectrum to enhance the thresholding,
        by only testing those frequencies which have been found to be significant in the global test.
        
    filter_confidence : float or 'default'
        The confidence level used for the fourier filtering. If 'default', then it defaults to the same
        value as the test confidence level for the individual tests.
        
    
    """
    
    # Set the individual test specifications, based on the provided default
    #
    # Allow for a setting that decides whether or not to average the power spectra from DCTs of different measurement
    # outcomes, or to test them all separately.
    #
    if individual_tests is 'default_noconvolve':
        # This is the zeroth layer of the convolution tests.
        individual_tests = {'1ksparse':{0:[1,2,3,4,5]}}
        
    elif individual_tests is 'default_convolve':
        # Todo: specify these tests.
        individual_tests = {}
        
    else:
        assert(type(individual_tests) is dict)
        
    # If defaults then...
    if global_test_confidence == 'default':
        global_test_confidence = individual_test_confidence
    if filter_confidence == 'default':
        filter_confidence= individual_test_confidence
    
    #
    # Here convert data to a (number of sequences x num_qubits x timesteps x possible outcomes) numpy array.
    # Add checks that it is of a suitable format.
    #
    
    counts_per_timestep =  _np.sum(data[0,0,0,:])
    number_of_sequences = len(data[:,0,0,0])
    number_of_num_qubits = len(data[0,:,0,0])
    number_of_timesteps = len(data[0,0,:,0]) 
    number_of_outcomes = len(data[0,0,0,:])
    
    if verbosity > 0:
        print("Beginning stage 1 of ...: Calculating test thresholds")
    #
    # 
    # Threshold calculations
    #
    # Perhaps put some sanity checks on the output values here?
    
    if verbosity > 0:
        print("Beginning stage 2 of ...: Calculating ....")
        print("This stage requires calculating .... Fourier transforms of dimension...")
    
    #
    #
    # Could perhaps replace these loops with just a loop over sequences, which sends the data for each
    # sequences to the single-sequence analysis tool. However, it is probably best not to, as it is
    # likely more useful to have the multi-sequence results in an object designed for analyzing multi-sequence
    # data.
    #
    for q in range(0,number_of_num_qubits):
        for s in range(0,number_of_sequences):
            for m in range(0,number_of_outcomes):
                results.modes[s,q,:,m] = _dtls.DCT(data[s,q,:,m],counts=counts_per_timestep)
                results.power_spectrum[s,q,:,m] = results.modes[s,q,:,m]**2
                #
                # Perform any per-outcome tests
                #
                #
                # Record significant frequencies.
                #
                
            results.outcome_averaged_power_spectrum[s,q,:] = _np.mean(results.power_spectrum[s,q,:,:],axis=1)
            #
            # Perform any averaged-over-outcome tests
            #
            #
            # Record significant frequencies.
            #
            
        results.global_power_spectrum[q,:] = _np.mean(results.outcome_averaged_power_spectrum[:,q,:],axis=0)
        #
        # Perform any tests on the global power spectrum
        #  
        #
        # Record significant frequencies.
        #
        
    if verbosity > 0:
        print("Beginning stage 3 of ...: Calculating ....")
    
    for q in range(0,number_of_num_qubits):
        for s in range(0,number_of_sequences):
            for m in range(0,number_of_outcomes):
                # Do the Fourier filtering.
                #
                # Todo : allow for various methods for normalizing the filtered function.
                #
                results.modes[s,q,:,m] = results.modes[s,q,:,m]
    #
    #
    #  Put something that looks at correlation of power of estimated signal with sequence length.
    #  Or, instead, put a method in the results object that can do this.
    #
    
    return results



def spectral_analysis(x, N, outcomes, test_thresholds, filter_confidence=0.99, 
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
        #results['individual'][key]['power_per_N_estimate_v1'] = {}
        #results['individual'][key]['power_per_N_estimate_v2'] = {}
        #results['individual'][key]['power_per_N_estimate_v1']['total'] = 0.
        #results['individual'][key]['power_per_N_estimate_v2']['total'] = 0.
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
                    #results['individual'][key]['power_per_N_estimate_v1'][i] =  0.
                    #results['individual'][key]['power_per_N_estimate_v2'][i] =  0.
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
                    #estimated_power_v1 = (summed_power[power_estimator_index] - expected_power[power_estimator_index])* p * (1 -
                    #                                                                                                         p)
                    #results['individual'][key]['power_per_N_estimate_v1'][i] =  estimated_power_v1 / N
                    #results['individual'][key]['power_per_N_estimate_v1']['total'] += \
                    #                                results['individual'][key]['power_per_N_estimate_v1'][i]
                    #index = _np.argmax(summed_power - expected_power)    
                    #estimated_power_v2 = (summed_power[index] - expected_power[index])* p * (1 - p)
                    #results['individual'][key]['power_per_N_estimate_v2'][i] =  estimated_power_v2 / N
                    #results['individual'][key]['power_per_N_estimate_v2']['total'] += \
                    #                                results['individual'][key]['power_per_N_estimate_v2'][i]

                    # Add the power spectrum for outcome i to the single power spectrum
                    results['individual'][key]['power_spectrum'] += results['individual'][key]['modes'][i]**2
        
        # Calculate power spectrum and estimates averaged over all the outcomes
        results['individual'][key]['power_spectrum'] = results['individual'][key]['power_spectrum']/len(outcomes)
        #results['individual'][key]['power_per_N_estimate_v1']['total'] = results['individual'][key]['power_per_N_estimate_v1']['total']/len(outcomes)
        #results['individual'][key]['power_per_N_estimate_v2']['total'] = results['individual'][key]['power_per_N_estimate_v2']['total']/len(outcomes)
        
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
            
        y = _dtls.DCT(x,null_hypothesis=null_hypothesis)
        if method=='chi2':
            threshold = _dthresholds.one_sparse_threshold(len(x),confidence)
        else:
            threshold = _dthresholds.one_to_k_sparse_threshold(null_hypothesis, 
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
        
        filtered_x = _dtls.IDCT(y,null_hypothesis)
 
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

#def expected_power_statistics(null_hypothesis,N=10000,return_aux=False):
#    """
#    To be calculated
#    """
#    n = len(null_hypothesis)
#          
#    bs = _np.array([_np.random.binomial(1,p,size=N) for p in null_hypothesis])
#    power = _np.zeros((n,N),float)
#    summed_power = _np.zeros((n,N),float)#
#
#    for i in range (0,N):
#        # the sorted power array, from highest to lowest.
#        power[:,i] = _np.flip(_np.sort(_dct(bs[:,i]-null_hypothesis,norm='ortho')**2),axis=0)
#        summed_power[:,i] = _np.cumsum(power[:,i])
#                            #
#
#    expected_values = _np.mean(summed_power,axis=1)
#    std = _np.std(summed_power,axis=1)
#    
#    max_power = _np.zeros(N,float)
#    normed_max_power = _np.zeros(N,float)
#    for i in range (0,N):
#        max_power[i] = _np.amax(summed_power[:,i] - expected_values) #
#
#    expected_max_power = _np.mean(max_power)
#    max_power_std = _np.std(max_power)
#    
#    out = {}
#    # summed power array statistics
#    out['expected_summed_power'] = expected_values
#    out['summed_power_std'] = std
#    out['sampled_summed_power'] = summed_power
#    
#    # max deviation from expected sum power statistics
#    out['expected_max'] = expected_max_power
#    out['max_std'] = max_power_std
#    out['sampled_max'] = max_power
#
#    return out

#def estimate_residual_power(data, null_hypothesis, expected_power=None, cut_off=False, return_aux=False):
#    
#    # The DCT modes are *not* standardized to have unit-variance (under null-hypothesis)
#    modes = _dct(data-null_hypothesis,norm='ortho')
#    summed_power = _np.cumsum(_np.flip(_np.sort(modes**2),axis=0))
#    
#    if expected_power is None:
#        power_stats = expected_power_statistics(null_hypothesis,N=10000)
#    else:
#        power_stats = expected_power
#    expected_values = power_stats['expected_summed_power']
#    summed_power_std = power_stats['summed_power_std']
#    
#    expected_max = power_stats['expected_max']
#    max_std = power_stats['max_std']
#    
#    index = _np.argmax(summed_power - expected_values)
#    
#    #estimated_power_1 = summed_power[index] - expected_values[index] - expected_max
#    estimated_power = summed_power[index] - expected_values[index]
#    
#    out = {}
#    #out['estimated_power_1'] = estimated_power_1
#    out['estimated_power'] = estimated_power
#    #out['estimated_power_1_error_bars'] = 2*max_std
#    out['estimated_power_error_bars'] = 2*summed_power_std[index]#
#
#    if cut_off:
#        if estimated_power < 0:
#            estimated_power = 0.
#    if return_aux:
#        power_stats['observed_summed_power'] = summed_power
#        return out, power_stats
#    else:
#        return out
    
#def estimate_power(modes, data_mean, expected_power=None, cut_off=True):
#    
#    # The DCT modes are *not* standardized to have unit-variance (under null-hypothesis)
#    summed_power = _np.cumsum(_np.flip(_np.sort(modes**2),axis=0))
#    
#    if expected_power is None:
#        power_stats = expected_power_statistics(null_hypothesis=data_mean*_np.ones(N),N=10000)
##    else:
#        power_stats = expected_power
#    expected_values = power_stats['expected_summed_power']
#    summed_power_std = power_stats['summed_power_std']
#    
#    expected_max = power_stats['expected_max']
##    max_std = power_stats['max_std']
#    
#    index = _np.argmax(summed_power - expected_values)
#    
#    estimated_power = summed_power[index] - expected_values[index]
#    
#    out = {}
#    out['estimated_power'] = estimated_power
#    out['estimated_power_error_bars'] = 2*summed_power_std[index]#
#
#    if cut_off:
#        if estimated_power < 0:
#            estimated_power = 0.#
#
#    return out