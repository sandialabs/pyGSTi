from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Core integrated routines for detecting and characterizing drift with time-stamped data"""

from . import signal as _sig
from . import objects as _obj
from . import hypothesis as _hyp
from . import statistics as _stat
#from scipy.stats import chi2 as _chi2
import numpy as _np

def do_basic_drift_characterization(indata, counts, timestep=None, confidence=0.95, verbosity=2):
    """
    
    """
    # data shape should be, except when 
    #(number of sequences x num_qubits x timesteps)
    
    # ---------------- #
    # Format the input #
    # ---------------- #
    
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
    
    # Extract the number of sequences, qubits, and timesteps from the shape of the input
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
                    
    # Find the full set of frequencies.
    if timestep is None:
        # If the timestep is not provided, frequencies are given as integers in [0,1,..,1 - number of timesteps]
        frequencies = _np.arange(0,num_timesteps)         
    else:
        # If the timestep is provided (assumed in seconds), frequencies are given in Hertz.
        frequencies = _np.zeros(num_timesteps,float)
        frequencies = _sig.frequencies_from_timestep(timestep,num_timesteps)
    
    # ------------------------------- #
    # Per-qubit per-sequence analysis #
    # ------------------------------- #
       
    # Calculate the power spectra for all the sequencies and qubits
    pspq_modes = _np.zeros(data_shape,float)
    pspq_power_spectrum = _np.zeros(data_shape,float)
    
    for s in range(0,num_sequences):
        for q in range(0,num_qubits):               
            pspq_modes[s,q,:] = _sig.DCT(data[s,q,:],counts=counts)
            
    pspq_power_spectrum = pspq_modes**2
              
    # The significance threshold for the max power in each power spectrum. This is NOT adjusted
    # to take account of the fact that we are testing multiple sequences on multiple qubits.
    pspq_significance_threshold = _stat.maxpower_threshold_chi2(confidence,num_timesteps,1)
    
    # Initialize arrays for the per-sequence per-qubit results
    pspq_max_power = _np.zeros((num_sequences,num_qubits),float)
    pspq_pvalue = _np.zeros((num_sequences,num_qubits),float)
    pspq_drift_detected = _np.zeros((num_sequences,num_qubits),bool)       
    pspq_drift_frequencies = {}   
    pspq_reconstruction = _np.zeros(data_shape,float)    
    pspq_reconstruction_power_spectrum = _np.zeros(data_shape,float)
    
    # An array that is not returned, storing the "raw" estimate of the power spectrum, pre 
    # renormalization.
    pspq_raw_estimated_modes = pspq_modes.copy()
    # To store the no-drift estimate of the probability for each sequence. Is not returned.
    pspq_null = _np.zeros(data_shape,float)

    for s in range(0,num_sequences):
        for q in range(0,num_qubits):
            pspq_max_power[s,q] = _np.max(pspq_power_spectrum[s,q,:])
            # Todo: check that this makes sense
            pspq_pvalue[s,q] = _stat.maxpower_pvalue_chi2(pspq_max_power[s,q],num_timesteps,1)
            
            # Find the significant frequencies using the standard thresholding
            pspq_drift_frequencies[s,q] = frequencies.copy()
            indices = _np.zeros((num_timesteps),bool)
            indices[pspq_power_spectrum[s,q,:] > pspq_significance_threshold] = True 
            pspq_drift_frequencies[s,q] = pspq_drift_frequencies[s,q][indices]
            
            # Create the reconstructions, using the standard single-pass Fourier filter
            pspq_null[s,q,:] = _np.mean(data[s,q,:])*_np.ones(num_timesteps,float)/counts
            
            if pspq_max_power[s,q] > pspq_significance_threshold:
                pspq_drift_detected[s,q] = True
                pspq_raw_estimated_modes[pspq_power_spectrum < pspq_significance_threshold] = 0.
                
                # Here divide by counts, as when we invert the DCT we would -- in the noise-free case -- get something
                # between 0 and counts, rather than 0 and 1
                pspq_reconstruction[s,q,:] = _sig.IDCT(pspq_raw_estimated_modes[s,q,:], null_hypothesis=pspq_null[s,q,:],
                                                       counts=counts)/counts
                pspq_reconstruction[s,q,:] = _sig.renormalizer(pspq_reconstruction[s,q,:],method='logistic')
                pspq_reconstruction_power_spectrum[s,q,:] =  _sig.DCT(pspq_reconstruction[s,q,:], 
                                                                      null_hypothesis=pspq_null[s,q,:],
                                                                      counts=counts)
            else:
                pspq_reconstruction[s,q,:] = pspq_null[s,q,:]                
                pspq_reconstruction_power_spectrum[s,q,:] = _np.zeros(num_timesteps,float)
    
    # Store the power per timestep in the reconstruction, as this is amount-of-data independent metric for the 
    #detected drift power.
    pspq_reconstruction_powerpertimestep = _np.sum((pspq_reconstruction-pspq_null)**2,axis=2)/num_timesteps
   
    # ------------------------------------ #
    # Per-qubit sequence-averaged analysis #
    # ------------------------------------ #
    
    
    
    # Calculate the power spectrum averaged over sequencies, but not qubits
    pq_power_spectrum = _np.mean(pspq_power_spectrum,axis=0)      
    pq_reconstruction_power_spectrum = _np.mean(pspq_reconstruction_power_spectrum,axis=0)
    
    pq_max_power = _np.zeros((num_qubits),float)
    pq_pvalue = _np.zeros((num_qubits),float)
    pq_drift_detected = _np.zeros((num_qubits),bool)       
    pq_drift_frequencies = {}
    
    # Analyze the per-qubit average power spectrum, and find significant frequencies.
    # Todo: check this
    pq_significance_threshold = _stat.maxpower_threshold_chi2(confidence,num_timesteps,num_sequences)
    
    # Loop over qubits, and analysis the per-qubit spectra
    for q in range(0,num_qubits):
        
        # Find the max power and the related pvalue.
        pq_max_power[q] = _np.max(pq_power_spectrum[q,:])
         # Todo: check that this makes sense
        pq_pvalue[q] = _stat.maxpower_pvalue_chi2(pq_max_power[q],num_timesteps,num_sequences)
        
        # Find the drift frequencies
        pq_drift_frequencies[q] = frequencies.copy()
        indices = _np.zeros((num_timesteps),bool)
        indices[pq_power_spectrum[q,:] > pq_significance_threshold] = True 
        pq_drift_frequencies[q] = pq_drift_frequencies[q][indices]
        
        if pq_max_power[q] > pq_significance_threshold:
            pq_drift_detected[q] = True 
    
    # ------------------------------------ #
    # qubit and sequence averaged analysis #
    # ------------------------------------ #
    
    # Calculate the power spectrum averaged over sequencies and qubits
    global_power_spectrum = _np.mean(pq_power_spectrum,axis=0)   
    global_reconstruction_power_spectrum = _np.mean(pq_reconstruction_power_spectrum,axis=0)
    
    # Analyze the global power spectrum, and find significant frequencies.
    #
    # Todo: check this
    global_significance_threshold = _stat.maxpower_threshold_chi2(confidence,num_timesteps,num_sequences*num_qubits)
    
    # Find the drift frequencies
    global_drift_frequencies = frequencies.copy()
    indices = _np.zeros((num_timesteps),bool)
    indices[global_power_spectrum > global_significance_threshold] = True 
    global_drift_frequencies = global_drift_frequencies[indices]
    
    #
    global_max_power = _np.max(global_power_spectrum)
    # Todo: check that this makes sense
    global_pvalue = _stat.maxpower_pvalue_chi2(global_max_power,num_timesteps,num_sequences*num_qubits)
        
    global_drift_detected = False
    if global_max_power > global_drift_detected:
        global_drift_detected = True

    # ------------------------------------------------------------------------- #
    # Check whether drift is detected with composite test at the set confidence #
    # ------------------------------------------------------------------------- #
    
    if num_qubits == 1 and num_sequences > 1:
        weights = [1/2.,0.,1/2.]  
    
    elif num_qubits > 1 and num_sequences == 1:
             weights = [1/2.,1/2.,0.]
            
    elif num_qubits == 1 and num_sequences == 1:
             weights = [1.,0.,0.]      
            
    else:
        weights = [1/3.,1/3.,1/3.]
        
    numtests = [1,num_qubits,num_qubits*num_sequences]
    composite_confidence = _hyp.generalized_bonferoni_correction(confidence,weights,numtests=numtests)
    
    pspq_minimum_pvalue = _np.min(pspq_pvalue)
    pq_minimum_pvalue = _np.min(pq_pvalue)
    
    drift_detected = False
    if global_pvalue < 1-composite_confidence[0]:
        drift_detected = True
    if pq_minimum_pvalue < 1-composite_confidence[1]:
        drift_detected = True
    if pspq_minimum_pvalue < 1-composite_confidence[2]:
        drift_detected = True
    
    # ------------------ #
    # Record the results #
    # ------------------ #
    
    # Initialize an empty results object.
    results = _obj.BasicDriftResults()
    
    # Records input information, and things fairly trivially derived from it
    results.data = data
    results.number_of_sequences = num_sequences
    results.number_of_timesteps = num_timesteps
    results.number_of_qubits = num_qubits
    results.number_of_counts = counts     
    results.timestep = timestep
        
    results.confidence = confidence
    results.frequencies = frequencies.copy()
    
    results.drift_detected = drift_detected 
    
    results.pspq_modes = pspq_modes
    results.pspq_power_spectrum = pspq_power_spectrum
    results.pspq_drift_frequencies = pspq_drift_frequencies
    results.pspq_max_power = pspq_max_power
    results.pspq_pvalue = pspq_pvalue
    results.pspq_significance_threshold = pspq_significance_threshold
    results.pspq_drift_detected = pspq_drift_detected
    results.pspq_reconstruction = pspq_reconstruction
    results.pspq_reconstruction_power_spectrum = pspq_reconstruction_power_spectrum
    results.pspq_reconstruction_powerpertimestep = pspq_reconstruction_powerpertimestep
    
    results.pq_power_spectrum = pq_power_spectrum
    results.pq_significance_threshold = pq_significance_threshold  
    results.pq_max_power = pq_max_power
    results.pq_pvalue = pq_pvalue
    results.pq_drift_detected = pq_drift_detected
    results.pq_drift_frequencies = pq_drift_frequencies
    results.pq_reconstruction_power_spectrum = pq_reconstruction_power_spectrum
    
    results.global_power_spectrum = global_power_spectrum
    results.global_max_power = global_max_power
    results.global_pvalue = global_pvalue
    results.global_significance_threshold = global_significance_threshold           
    results.global_drift_detected = global_drift_detected
    results.global_drift_frequencies = global_drift_frequencies
    results.global_reconstruction_power_spectrum = global_reconstruction_power_spectrum
    
    return results