from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Functions for Fourier analysis of equally spaced time-series data"""

from . import signal as _sig
from . import objects as _obj

import numpy as _np
#from scipy.fftpack import dct as _dct
#from scipy.fftpack import idct as _idct
from scipy.stats import chi2 as _chi2
#from scipy.optimize import leastsq as _leastsq


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
        frequencies = _sig.frequencies_sigrom_timestep(timestep,num_timesteps)
        
    # A bool to flag up if any drift has been detected
    was_drift_detected = False
    global_detected = False
    
    # Calculate the power spectra for all the sequencies
    modes = _np.zeros(data_shape,float)
    power_spectrum = _np.zeros(data_shape,float)
    
    for s in range(0,num_sequences):
        for q in range(0,num_qubits):               
            modes[s,q,:] = _sig.DCT(data[s,q,:],counts=counts)
            
    power_spectrum = modes**2
    
    # Calculate the power spectrum averaged over sequencies.
    average_power_spectrum = _np.mean(power_spectrum,axis=0)
    
    # Analyze the average power spectrum, and find significant frequencies.
    threshold = _chi2.isf(1-confidence**(1/num_timesteps),num_sequences)/num_sequences
    drift_sigrequencies_indices = _np.zeros((num_qubits,num_timesteps),bool)
    drift_sigrequencies_indices[average_power_spectrum > threshold] = True 

    drift_sigrequencies = _np.zeros((num_qubits,num_timesteps),float)
    for q in range(0,num_qubits):
        drift_sigrequencies[q,:] = frequencies.copy()
    drift_sigrequencies = drift_sigrequencies[drift_sigrequencies_indices]
    
    # Record if the averaged power spectrum detects drift.
    if len(drift_sigrequencies) > 0:
        was_drift_detected = True
        global_detected = True
        
    # Analyze the individual sequences.
    max_power_threshold = _chi2.isf(1-confidence**(1/num_timesteps),1)
    max_powers = _np.zeros((num_sequences,num_qubits),float)
    max_power_pvalues = _np.zeros((num_sequences,num_qubits),float)
    drift_detected = _np.zeros((num_sequences,num_qubits),bool)
    
    individual_drift_sigrequencies = {}
    
    raw_estimated_modes = modes.copy()
    probability_estimates = _np.zeros(data_shape,float)
    null = _np.zeros(data_shape,float)
    for s in range(0,num_sequences):
        for q in range(0,num_qubits):
            max_powers[s,q] = _np.max(power_spectrum[s,q,:])
            # Todo: check that this makes sense
            max_power_pvalues[s,q] = 1 - (1 - _chi2.sf(max_powers[s,q],1))**num_timesteps
            
            # Find the significant frequencies using the standard thresholding
            individual_drift_sigrequencies[s,q] = frequencies.copy()
            drift_sigrequencies_indices = _np.zeros((num_timesteps),bool)
            drift_sigrequencies_indices[power_spectrum[s,q,:] > max_power_threshold] = True 
            individual_drift_sigrequencies[s,q] = individual_drift_sigrequencies[s,q][drift_sigrequencies_indices]
            
            # Create the reconstructions, using the standard single-pass Fourier filter
            null[s,q,:] = _np.mean(data[s,q,:])*_np.ones(num_timesteps,float)/counts
            
            if max_powers[s,q] > max_power_threshold:
                was_drift_detected = True
                drift_detected[s,q] = True
                raw_estimated_modes[power_spectrum<max_power_threshold] = 0.
                
                # Here divide by counts, as when we invert the DCT we would -- in the noise-free case -- get something
                # between 0 and counts, rather than 0 and 1
                probability_estimates[s,q,:] = _sig.IDCT(raw_estimated_modes[s,q,:], null_hypothesis=null[s,q,:], 
                                                          counts=counts)/counts
                probability_estimates[s,q,:] = _sig.renormalizer(probability_estimates[s,q,:],method='logistic')
            else:
                probability_estimates[s,q,:] = null[s,q,:]
                
    reconstruction_power_per_time = _np.sum((probability_estimates-null)**2,axis=2)/num_timesteps
    
    #if num_qubits == 1:
    #    if len(data_shape) == 2:
    #
    #        drift_sigrequencies = drift_sigrequencies[0,:]
    #        average_power_spectrum =  average_power_spectrum[0,:]  
    #    
    #        max_powers = max_powers[:,0]
    #        max_power_pvalues = max_power_pvalues[:,0]
    #        max_power_threshold = max_power_threshold
    #        drift_detected = drift_detected[:,0]
    #        power_spectrum = power_spectrum[:,0,:]
    #        probability_estimates = probability_estimates[:,0,:]
    #        reconstruction_power_per_time = reconstruction_power_per_time[:,0]
        
    
    # Initialize an empty results object.
    results = _obj.BasicDriftResults()
    
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
    results.global_drift_sigrequencies = drift_sigrequencies
    
    results.individual_modes = modes
    results.individual_power_spectrum = power_spectrum
    results.individual_drift_sigrequencies = individual_drift_sigrequencies
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