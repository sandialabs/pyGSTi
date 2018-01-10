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

import numpy as _np

def do_basic_drift_characterization(indata, counts=None, timestep=None, confidence=0.95, 
                                    indices_to_sequences=None, verbosity=2, name=None):
    """
    Todo:docstring
    """
    # data shape should be, except when 
    #(number of sequences x num_entities x timesteps)
    
    # ---------------- #
    # Format the input #
    # ---------------- #
    
    data_shape = _np.shape(indata)
    
    # Todo:
    # Check that the input data is consistent with being counts in an array of dimension
    # (number of sequences x num_entities x timesteps), or (number of sequences x timesteps).
    assert(len(data_shape) == 2 or len(data_shape) == 3 or len(data_shape) == 4), "Data format is incorrect!"
    
    # If ....
    if len(data_shape) == 2:
        
        assert(counts is not None), "This data format requires specifying `counts`, the number of counts per timestep!"
        
        if verbosity > 0:
            print("Analysis is defaulting to assuming two-outcome measurements on a single entity.")
       
        data = _np.zeros((data_shape[0],1,2,data_shape[1]),float)
        data[:,0,:] = indata.copy()
     
    if len(data_shape) == 3:
        assert(counts is not None), "This data format requires specifying `counts`, the number of counts per timestep!"
        
        if verbosity > 0:
            print("Analysis is defaulting to assuming two-outcome measurements.")
        
        data = _np.zeros((data_shape[0],data_shape[1],2,data_shape[2]),float)
        data[:,:,0,:] = indata.copy()
        data[:,:,1,:] = counts - indata.copy()
        
    if len(data_shape) == 4:
        data = indata.copy()
    
    # Extract the number of sequences, entities, and timesteps from the shape of the data array
    data_shape = _np.shape(data)    
    num_sequences = data_shape[0]
    num_entities = data_shape[1]
    num_outcomes = data_shape[2]
    num_timesteps = data_shape[3] 
    
    # --------------------------------------------------------- #
    # Prepare a results object, and store the input information #
    # --------------------------------------------------------- #
    
    # Initialize an empty results object.
    results = _obj.BasicDriftResults()
    
    # Records input information into the results object.
    results.name = name
    results.data = data
    results.number_of_sequences = num_sequences
    results.number_of_timesteps = num_timesteps
    results.number_of_entities = num_entities
    results.number_of_outcomes = num_outcomes   
    results.number_of_counts = counts     
    results.timestep = timestep       
    results.confidence = confidence
    results.indices_to_sequences = indices_to_sequences
    
    # Provides a warning if the number of timesteps is low enough that the chi2 approximations used in this 
    # function might not be good approximations.
    if verbosity > 0:
        if num_timesteps <=50:
            string = "*** Warning: certain approximations used within this function may be unreliable when the"
            string += " number of timestamps is too low. The statistical significance"
            string += " thresholds may be inaccurate ***"   
            string +=  "\n"
            print(string)
                    
    # Find the full set of frequencies, that all the power spectra are with respect to
    if timestep is None:
        # If the timestep is not provided, frequencies are given as integers in [0,1,..,1 - number of timesteps]
        frequencies = _np.arange(0,num_timesteps)         
    else:
        # If the timestep is provided (assumed in seconds), frequencies are given in Hertz.
        frequencies = _np.zeros(num_timesteps,float)
        frequencies = _sig.frequencies_from_timestep(timestep,num_timesteps)
    
    # Write the frequencies into the results object
    results.frequencies = frequencies.copy()
    
    # ------------------------------------------------- #
    #     Calculate the modes and power spectra         #
    # ------------------------------------------------- #
       
    # Calculate the power spectra for all the sequencies and entities
    pspepo_modes = _np.zeros(data_shape,float)
    pspepo_power_spectrum = _np.zeros(data_shape,float)
    
    for s in range(0,num_sequences):
        for q in range(0,num_entities): 
            for o in range(0,num_outcomes):  
                pspepo_modes[s,q,o,:] = _sig.DCT(data[s,q,o,:],counts=counts)
    
    # Calculate the power spectra
    pspepo_power_spectrum = pspepo_modes**2
    pspe_power_spectrum = _np.mean(pspepo_power_spectrum,axis=2)
    ps_power_spectrum = _np.mean(pspe_power_spectrum,axis=1)
    pe_power_spectrum = _np.mean(pspe_power_spectrum,axis=0)
    global_power_spectrum = _np.mean(pe_power_spectrum,axis=0)
    
    # The significance threshold for the max power in each power spectrum. This is NOT adjusted
    # to take account of the fact that we are testing multiple sequences on multiple entities.
    pspepo_significance_threshold = _stat.maxpower_threshold_chi2(confidence, num_timesteps, 1)
    pspe_significance_threshold = _stat.maxpower_threshold_chi2(confidence, num_timesteps,num_outcomes-1)
    ps_significance_threshold = _stat.maxpower_threshold_chi2(confidence, num_timesteps, 
                                                              num_entities*(num_outcomes-1))
    global_significance_threshold = _stat.maxpower_threshold_chi2(confidence, num_timesteps, 
                                                                  num_sequences*num_entities*(num_outcomes-1))
    
    # Initialize arrays for the per-sequence, per-entity, per-outcome results
    pspepo_max_power = _np.zeros((num_sequences,num_entities,num_outcomes),float)
    pspepo_pvalue = _np.zeros((num_sequences,num_entities,num_outcomes),float)
    pspepo_drift_detected = _np.zeros((num_sequences,num_entities,num_outcomes),bool)       
    pspepo_drift_frequencies = {}   
    pspepo_reconstruction = _np.zeros(data_shape,float)    
    pspepo_reconstruction_power_spectrum = _np.zeros(data_shape,float)
    
    # Initialize arrays for the per-sequence, per-entity results
    pspe_max_power = _np.zeros((num_sequences,num_entities),float)
    pspe_pvalue = _np.zeros((num_sequences,num_entities),float)
    pspe_drift_detected = _np.zeros((num_sequences,num_entities),bool)       
    pspe_drift_frequencies = {}   
    pspe_reconstruction_power_spectrum = _np.zeros(data_shape,float)
    
    # Initialize arrays for the per-sequence results
    ps_max_power = _np.zeros((num_sequences),float)
    ps_pvalue = _np.zeros((num_sequences),float)
    ps_drift_detected = _np.zeros((num_sequences),bool)       
    ps_drift_frequencies = {}   
    ps_reconstruction_power_spectrum = _np.zeros(data_shape,float)
    
    # Temp arrays.
    pspepo_raw_estimated_modes = pspepo_modes.copy()
    pspepo_null = _np.zeros(data_shape,float)

    for s in range(0,num_sequences):
        for q in range(0,num_entities):
            for o in range(0,num_outcomes):
                
                # --- analysis at the per-sequence per-entity per-outcome level --- #
                
                # Find the max power and associated pvalue
                pspepo_max_power[s,q,o] = _np.max(pspepo_power_spectrum[s,q,o,:])
                pspepo_pvalue[s,q,o] = _stat.maxpower_pvalue_chi2(pspepo_max_power[s,q,o],num_timesteps,1)
            
                # Find the significant frequencies using the standard thresholding
                pspepo_drift_frequencies[s,q,o] = frequencies.copy()
                indices = _np.zeros((num_timesteps),bool)
                indices[pspepo_power_spectrum[s,q,o,:] > pspepo_significance_threshold] = True 
                pspepo_drift_frequencies[s,q,o] = pspepo_drift_frequencies[s,q,o][indices]
            
                # Create the reconstructions, using the standard single-pass Fourier filter
                pspepo_null[s,q,o,:] = _np.mean(data[s,q,o,:])*_np.ones(num_timesteps,float)/counts
            
                if pspepo_max_power[s,q,o] > pspepo_significance_threshold:
                    pspepo_drift_detected[s,q,o] = True
                    pspepo_raw_estimated_modes[pspepo_power_spectrum < pspepo_significance_threshold] = 0.
                
                    # Here divide by counts, to get something in 0,1 (in the noise free case).
                    pspepo_reconstruction[s,q,o,:] = _sig.IDCT(pspepo_raw_estimated_modes[s,q,o,:], 
                                                               null_hypothesis=pspepo_null[s,q,o,:],
                                                               counts=counts)/counts
                    # Todo: check whether this renormalizer makes sense with multi-outcome data.
                    pspepo_reconstruction[s,q,o,:] = _sig.renormalizer(pspepo_reconstruction[s,q,o,:],method='logistic')
                    pspepo_reconstruction_power_spectrum[s,q,o,:] =  _sig.DCT(pspepo_reconstruction[s,q,o,:], 
                                                                      null_hypothesis=pspepo_null[s,q,o,:],
                                                                      counts=counts)
                else:
                    pspepo_reconstruction[s,q,o,:] = pspepo_null[s,q,o,:]                
                    pspepo_reconstruction_power_spectrum[s,q,o,:] = _np.zeros(num_timesteps,float)
            
            # --- analysis at the per-sequence per-entity level --- #
                
            # Find the max power and associated pvalue
            pspe_max_power[s,q] = _np.max(pspe_power_spectrum[s,q,:])
            pspe_pvalue[s,q] = _stat.maxpower_pvalue_chi2(pspe_max_power[s,q],num_timesteps,num_outcomes-1)
            
            # Find the significant frequencies using the standard thresholding
            pspe_drift_frequencies[s,q] = frequencies.copy()
            indices = _np.zeros((num_timesteps),bool)
            indices[pspe_power_spectrum[s,q,:] > pspe_significance_threshold] = True 
            pspe_drift_frequencies[s,q] = pspe_drift_frequencies[s,q][indices]
            
            if pspe_max_power[s,q] > pspe_significance_threshold:
                pspe_drift_detected[s,q] = True
        
        # --- analysis at the per-sequence level --- #
                
        # Find the max power and associated pvalue
        ps_max_power[s] = _np.max(ps_power_spectrum[s,:])
        ps_pvalue[s] = _stat.maxpower_pvalue_chi2(ps_max_power[s],num_timesteps,num_entities*(num_outcomes-1))
            
        # Find the significant frequencies using the standard thresholding
        ps_drift_frequencies[s] = frequencies.copy()
        indices = _np.zeros((num_timesteps),bool)
        indices[ps_power_spectrum[s,:] > ps_significance_threshold] = True 
        ps_drift_frequencies[s] = ps_drift_frequencies[s][indices]
            
        if ps_max_power[s] > ps_significance_threshold:
            ps_drift_detected[s] = True
    
    # --- analysis at the global level --- #
                
    # Find the max power and associated pvalue
    global_max_power = _np.max(global_power_spectrum)
    global_pvalue = _stat.maxpower_pvalue_chi2(global_max_power,num_timesteps,num_sequences*num_entities*(num_outcomes-1))    
    
    # Find the significant frequencies using the standard thresholding
    global_drift_frequencies = frequencies.copy()
    indices = _np.zeros((num_timesteps),bool)
    indices[global_power_spectrum > global_significance_threshold] = True 
    global_drift_frequencies = global_drift_frequencies[indices]
    
    if global_max_power > global_significance_threshold:
        global_drift_detected = True
    else:
        global_drift_detected = False

    # Calculate the power per timestep in the reconstruction. This is useful as this is an amount-of-data independent 
    # metric for the detected drift power.
    pspepo_reconstruction_powerpertimestep = _np.sum((pspepo_reconstruction-pspepo_null)**2,axis=3)/num_timesteps   
    pspe_reconstruction_power_spectrum = _np.mean(pspepo_reconstruction_power_spectrum,axis=2)
    pspe_reconstruction_powerpertimestep = _np.sum(pspepo_reconstruction_powerpertimestep,axis=2)
    ps_reconstruction_power_spectrum = _np.mean(pspe_reconstruction_power_spectrum,axis=1)
    ps_reconstruction_powerpertimestep = _np.mean(pspe_reconstruction_powerpertimestep,axis=1)
    global_reconstruction_power_spectrum = _np.mean(ps_reconstruction_power_spectrum,axis=0)
    global_reconstruction_powerpertimestep = _np.mean(ps_reconstruction_powerpertimestep,axis=0)
    
    # Write the results for this analysis into the results object.
    results.pspepo_modes = pspepo_modes
    results.pspepo_power_spectrum = pspepo_power_spectrum
    results.pspepo_drift_frequencies = pspepo_drift_frequencies
    results.pspepo_max_power = pspepo_max_power
    results.pspepo_pvalue = pspepo_pvalue
    results.pspepo_significance_threshold = pspepo_significance_threshold
    results.pspepo_drift_detected = pspepo_drift_detected
    results.pspepo_reconstruction = pspepo_reconstruction
    results.pspepo_reconstruction_power_spectrum = pspepo_reconstruction_power_spectrum
    results.pspepo_reconstruction_powerpertimestep = pspepo_reconstruction_powerpertimestep
    
    # Write the results for this analysis into the results object.
    results.pspe_power_spectrum = pspe_power_spectrum
    results.pspe_drift_frequencies = pspe_drift_frequencies
    results.pspe_max_power = pspe_max_power
    results.pspe_pvalue = pspe_pvalue
    results.pspe_significance_threshold = pspe_significance_threshold
    results.pspe_drift_detected = pspe_drift_detected
    results.pspe_reconstruction_power_spectrum = pspe_reconstruction_power_spectrum
    results.pspe_reconstruction_powerpertimestep = pspe_reconstruction_powerpertimestep

    # Write the results for this analysis into the results object.
    results.ps_power_spectrum = ps_power_spectrum
    results.ps_drift_frequencies = ps_drift_frequencies
    results.ps_max_power = ps_max_power
    results.ps_pvalue = ps_pvalue
    results.ps_significance_threshold = ps_significance_threshold
    results.ps_drift_detected = ps_drift_detected
    results.ps_reconstruction_power_spectrum = ps_reconstruction_power_spectrum
    results.ps_reconstruction_powerpertimestep = ps_reconstruction_powerpertimestep
    
    # Write the results for this analysis into the results object.
    results.global_power_spectrum = global_power_spectrum
    results.global_drift_frequencies = global_drift_frequencies
    results.global_max_power = global_max_power
    results.global_pvalue = global_pvalue
    results.global_significance_threshold = global_significance_threshold
    results.global_drift_detected = global_drift_detected
    results.global_reconstruction_power_spectrum = global_reconstruction_power_spectrum
    results.global_reconstruction_powerpertimestep = global_reconstruction_powerpertimestep

    # ------------------------------------ #
    #         Per-entity analysis          #
    # ------------------------------------ #
    
    # This analysis can't be implemented in 
    # the loop above, so we do it here.
    
    # Calculate the power spectrum averaged over sequencies, but not entities
    pe_max_power = _np.zeros((num_entities),float)
    pe_pvalue = _np.zeros((num_entities),float)
    pe_drift_detected = _np.zeros((num_entities),bool)       
    pe_drift_frequencies = {}
    
    # Analyze the per-entity average power spectrum, and find significant frequencies.
    # Todo: check this
    pe_significance_threshold = _stat.maxpower_threshold_chi2(confidence,num_timesteps,num_sequences)
    
    # Loop over entities, and analysis the per-entity spectra
    for q in range(0,num_entities):
        
        # Find the max power and the related pvalue.
        pe_max_power[q] = _np.max(pe_power_spectrum[q,:])
        pe_pvalue[q] = _stat.maxpower_pvalue_chi2(pe_max_power[q],num_timesteps,num_sequences*(num_outcomes-1))
        
        # Find the drift frequencies
        pe_drift_frequencies[q] = frequencies.copy()
        indices = _np.zeros((num_timesteps),bool)
        indices[pe_power_spectrum[q,:] > pe_significance_threshold] = True 
        pe_drift_frequencies[q] = pe_drift_frequencies[q][indices]
        
        if pe_max_power[q] > pe_significance_threshold:
            pe_drift_detected[q] = True 

    pe_reconstruction_power_spectrum = _np.mean(pspe_reconstruction_power_spectrum,axis=0)
    pe_reconstruction_powerpertimestep = _np.mean(pspe_reconstruction_powerpertimestep,axis=0)
    
    results.pe_power_spectrum = pe_power_spectrum
    results.pe_significance_threshold = pe_significance_threshold  
    results.pe_max_power = pe_max_power
    results.pe_pvalue = pe_pvalue
    results.pe_drift_detected = pe_drift_detected
    results.pe_drift_frequencies = pe_drift_frequencies
    results.pe_reconstruction_power_spectrum = pe_reconstruction_power_spectrum
    results.pe_reconstruction_powerpertimestep = pe_reconstruction_powerpertimestep

    # --------------------------------------------------------------------------- #
    # Check whether drift is detected with a composite test at the set confidence #
    # --------------------------------------------------------------------------- #
    # Record whether drift is detected at `confidence` confidence level, using a c
    # omposite hypothesis .... todo: details.
    
    if num_entities == 1 and num_sequences > 1:
        weights = [1/2.,0.,1/2.]  
    
    elif num_entities > 1 and num_sequences == 1:
             weights = [1/2.,1/2.,0.]
            
    elif num_entities == 1 and num_sequences == 1:
             weights = [1.,0.,0.]      
            
    else:
        weights = [1/3.,1/3.,1/3.]
        
    numtests = [1,num_entities,num_entities*num_sequences]
    composite_confidence = _hyp.generalized_bonferoni_correction(confidence,weights,numtests=numtests)
    
    pspe_minimum_pvalue = _np.min(pspe_pvalue)
    pe_minimum_pvalue = _np.min(pe_pvalue)
    
    drift_detected = False
    if global_pvalue < 1-composite_confidence[0]:
        drift_detected = True
    if pe_minimum_pvalue < 1-composite_confidence[1]:
        drift_detected = True
    if pspe_minimum_pvalue < 1-composite_confidence[2]:
        drift_detected = True
   
    # Write whether the composite hypothesis test detects drift into the results object.
    results.drift_detected = drift_detected 
    
    return results