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
from ... import objects as _objs

import numpy as _np

def do_basic_drift_characterization(ds, counts=None, timestep=None, timestamps=None,
                                    marginalize = 'none', marginalize_dict = None,
                                    outcomes=None, confidence=0.95, indices_to_sequences=None, 
                                    multitest_compensation='class', verbosity=1, name=None):
    """
    Implements a drift detection and characterization analysis on timeseries data.
    
    Parameters
    ----------
    ds : pyGSTi DataSet or numpy array
        The time series data to analyze. If this is a DataSet it should contain time series
        data (rather than, e.g., a total counts per-outcome per-GateString). If this is a
        numpy array, it must again contain time series data and it may be either 1, 2, 3
        or 4 dimensional. 
        
        The 4D case: the input array `ds` has ds[s,e,o,t] is the number of counts for outcome 
        with index o on entity e, at the tth time that sequence s is  implemented. The 
        entity index stores data for different "entities" that will be analyzed both independent
        and jointly. E.g., the sequences could be on Q qubits, and each entity could be a single 
        qubit with possible measurement outcomes '0' or '1' (i.e., the data for one qubit has 
        been obtained from hmarginalizing over the other qubits). Note that the same raw data can
        generally be arranged by entity and outcome in a range of different ways.
        
        The 3D case: As the 4D case, except that two-outcome measurements are assumed, and it is
        assumed that the, for input array `ds`, ds[s,e,t] is the counts associated with the first
        of the two measurement outcomes (this convention is only internally relevant if `outcomes`
        is specified), and that the number of outcomes for the two measurements sum to `counts`,
        at all time steps.
        
        The 2D case: As the 3D case, except that single-entity data is assumed (dropping the 2nd
        index in the array from the 3D case).
        
        The 1D case: As the 2D case, except that single-sequence data is assumsed. That is the 
        input array `ds` is a 1D array of counts, for the second out of two measurement outcomes
        at a set of different times, from a single gate sequence.
        
    counts : int, optional
        The total number of counts per-timestep, per-sequence and per-entity (i.e., the number
        of times any of the measurement outcomes is obtained per-timestep, per-sequence and 
        per-entity, which must be a constant). If `ds` is a DataSet, or it is a 4D numpy array, 
        this parameter is not used as it is extracted from the data. If `ds` is a 1D, 2D or 
        3D numpy array then this parameter is required.
        
    timestep : float, optional
        The time, in seconds, between consecutive repeats of each set of `counts` repeats of a
        each sequence. If `ds` is a DataSet this parameter is not used, as the timestep is extracted
        from the timestamps in the DataSet object (which should be in seconds to obtain frequencies
        in Hertz). If `ds` is a numpy array then it is necessary to specify this, if the user wants
        the output frequencies to be given in terms of Hertz. If `ds` is a numpy array and this is
        not specified all output frequencies are in 1/timestep with timestep defaulting to 1.
               
    timestamps : numpy array, optional
        An array of timestamps associated with the repeats of each circuit. If `ds` is a DataSet
        this is overwritten by the timestamps inside the DataSet. If `ds` is a numpy array these
        timestamps are not used for any analysis, and are simply written into the results output 
        object.
        
        #todo: add the ability to extract the timestep from timestamps.
        
    marginalize : str, optional
    
    marginalize_dict : dict, optional
    
    outcomes : list, optional 
    
    confidence : float, optional
    
    indices_to_sequences : list, optional
        
    multitest_compensation : str, optional
        Allowed values are 'none', 'class'.
     
    verbosity : int, optional
    
    name : str, optional
    
    Returns
    -------
    results : DriftResults object
        The results of the drift analysis. This contains: power spectra, statistical test outcomes,
        drift frequencies, reconstructions of drifting probabilities, all of the input information.

    """    
    # ---------------- #
    # Format the input #
    # ---------------- #
    
    if type(ds) != _objs.dataset.DataSet:
    
        data_shape = _np.shape(ds)
    
        # Check that the input data is consistent with being counts in an array of an appropriate dimension
        check1 = len(data_shape) == 1
        check2 = len(data_shape) == 2
        check3 = len(data_shape) == 3
        check4 = len(data_shape) == 4
        assert(check1 or check2 or check3 or check4), \
        "Input data format is incorrect!If the input is a numpy array it must be 1, 2, 3 or 4 dimensional."
        
        # If the input is 1D, assume that it is single-sequence, single-entity, two-outcome data.
        if len(data_shape) == 1:
            
            # There is no marginalizing possible for this data type (two-outcome data).
            assert (marginalize == 'none')
            assert(counts is not None), \
            "This data format requires specifying `counts`, the number of counts per timestep!" 
            
            if verbosity > 0:
                print("Due to input array format, analysis is defaulting to assuming:")
                print("  - single-sequence data")
                print("  - single entity data")
                print("  - two-outcome measurements")               
                print("")
            
            # Turn the data into the standard 4D format for the analysis.
            data = _np.zeros((1,1,2,data_shape[0]),float)
            data[0,0,0,:] = ds.copy()
            data[0,0,1,:] = counts - ds.copy()
                
        # If the input is 2D, assume that it is multi-sequence, single-entity, two-outcome data.
        if len(data_shape) == 2:
            
            # There is no marginalizing possible for this data type (two-outcome data).            
            assert (marginalize == 'none')
            assert(counts is not None), "This data format requires specifying `counts`, the number of counts per timestep!" 
            
            if verbosity > 0:
                print("Due to input array format, analysis is defaulting to assuming:")
                print("  - single entity data")
                print("  - two-outcome measurements")
                print("")   
                
            # Turn the data into the standard 4D format for the analysis.    
            data = _np.zeros((data_shape[0],1,2,data_shape[1]),float)
            data[:,0,0,:] = ds.copy()
            data[:,0,1,:] = counts - ds.copy()
            
        # If the input is 3D, assume that it is multi-sequence, multi-entity, two-outcome data.
        # Todo: perhaps it should default to single-entity data with the option to marginalize.
        if len(data_shape) == 3:
            
            # There is no marginalizing possible for this data type (two-outcome data).                        
            assert (marginalize == 'none')
            assert(counts is not None), "This data format requires specifying `counts`, the number of counts per timestep!"
            
            if verbosity > 0:
                print("Due to input array format, analysis is defaulting to assuming:")
                print("  - two-outcome measurements")
                print("")
                
            # Turn the data into the standard 4D format for the analysis.
            data = _np.zeros((data_shape[0],data_shape[1],2,data_shape[2]),float)
            data[:,:,0,:] = ds.copy()
            data[:,:,1,:] = counts - ds.copy()
            
        # If the input is 4D, assume nothing about the data type.  
        if len(data_shape) == 4:
            
            # Do not need to pad the input array, so just copy it.
            data = ds.copy()
            
            # Check the total number of counts is independent of sequence, timestep and entity 
            check_counts = _np.sum(data,axis=2)
            assert(_np.std(check_counts) == 0.), \
            "The total number of counts should be independent of sequence, timestep and entity"
            
            # Set the number of counts
            counts = _np.sum(data[0,0,:,0])
                     
            # Currently only allows for standard or no marginalization
            assert (marginalize == 'none' or marginalize == 'std')
            
            # If standard marginalization, [0,1] data on qubits, with the measurement outcomes
            # input as bitstring, assumed.
            if marginalize == 'std':
                
                # The marginalized set of outcomes is ['0','1']
                outcomes = ['0','1']
                num_outcomes = 2
                
                # We extract the number of qubits from the number of raw outcomes.
                num_entities = int(_np.log2(len(data[0,0,:,0]))/1)
                
                data_shape = _np.shape(data)    
                num_sequences = data_shape[0]
                num_timesteps = data_shape[3] 
                
                # This stores the marginalized data.
                data_marg = _np.zeros((num_sequences,num_entities,num_outcomes,num_timesteps),float)

                for s in range(0,num_sequences):
                    for e in range(0,num_entities):
                        for raw_o in range(0,2**num_entities):
                            
                            # If the eth bit in the binary rep of the raw outcome is '0', then
                            # the measurement outcome has the eth qubit in '0'.
                            if _np.binary_repr(raw_o,width=num_entities)[e] == '0':
                                data_marg[s,e,0,:] += data[s,0,raw_o,:]
                            else:
                                data_marg[s,e,1,:] += data[s,0,raw_o,:]
                
                # Replace the data with the marginalized version.
                data = data_marg.copy()
    
        # Extract the number of sequences, entities, and timesteps from the shape of the data array
        data_shape = _np.shape(data)    
        num_sequences = data_shape[0]
        num_entities = data_shape[1]
        num_outcomes = data_shape[2]
        num_timesteps = data_shape[3] 
        
    # This converts a DataSet to an array, as the code below uses arrays (for a good reason - as there is
    # lots of averaging over different dimensions of the array). This bit of the code seems to be by far
    # the slowest. Todo: improve this code, or the speed of access to a DataSet object.
    if type(ds) == _objs.dataset.DataSet:
        
        # Warns the user if things have been specifed which are being overwritten.
        if verbosity > 0:
            if counts != None:
                print("Warning: Input data is a pygsti DataSet, so `counts` is being overwritten!")
            if timestamps != None:
                print("Warning: Input data is a pygsti DataSet, so `timestamps` is being overwritten!")
            if indices_to_sequences != None:
                print("Warning: Input data is a pygsti DataSet, so `indices_to_sequences` is being overwritten!")
        
        num_sequences = len(list(ds.keys()))
        indices_to_sequences = list(ds.keys())
        sequences_to_indices = {}           
        
        num_timesteps = []
        counts = []

        for i in range(0,num_sequences):
            # Find the set of all timestamps and total counts
            t, c = ds[indices_to_sequences[i]].timeseries('all')
            # Record the number of timestamps for this sequence
            num_timesteps.append(len(t))
            # Check that the number of clicks is constant over all timestamps
            c = _np.array(c)
            assert(_np.std(c) == 0), "Number of total clicks must be the same at every timestamp!"
            # Record the counts-per-timestamp for this sequence
            counts.append(c[0])
    
        # Check that the number of timesteps and counts is independent of sequence.
        num_timesteps = _np.array(num_timesteps)
        counts = _np.array(counts)
        assert(_np.std(counts) == 0.), "Number of counts must be the same for all sequences!"
        assert(_np.std(num_timesteps) == 0.), "Number of timestamps must be the same for all sequences!"

        counts = counts[0]
        num_timesteps = num_timesteps[0]
        
        # Check that the marginalize specification is ok.
        assert(marginalize == 'none' or  marginalize == 'std' or marginalize == 'usr')
        if marginalize == 'usr':   
                assert(marginalize_dict is not None)
                assert(outcomes is not None)
        
        # If we are not marginalizing, we can create the array slightly more efficiently, so we have a
        # seperate code for this case. 
        if marginalize == 'none':
            
            if outcomes != None:
                print("Warning: Input data is a pygsti DataSet and marginalize == 'none', so `outcomes` is being overwritten!")
            
            # The number of outcomes, outcomes list, and number of entities.
            num_outcomes = len(list(ds.get_outcome_labels()))
            outcomes = list(ds.get_outcome_labels())
            num_entities = 1
            
            # To store the data and timestamps arrays.
            data = _np.zeros((num_sequences,num_entities,num_outcomes,num_timesteps),float)
            timestamps = _np.zeros((num_sequences,num_timesteps),float)

            for s in range(0,num_sequences):
                
                # Find the timestamps for the sequence
                times_for_seq, junk = ds[indices_to_sequences[s]].timeseries('all')
                # Write the timestamps into the timestamps array
                timestamps[s,:] = _np.array(times_for_seq)
                
                for o in range(0,num_outcomes):                        
                    # Record the timeseries data for sequence s and outcome o.
                    junk, data[s,0,o,:] = _np.array(ds[indices_to_sequences[s]].timeseries(outcomes[o],timestamps[s,:]))
        
        # Go into this section if we are marginalizing the data.
        else:
            full_outcomes = ds.get_outcome_labels()
      
            if marginalize == 'std':            
                outcomes = ['0','1']
                
            num_outcomes = len(outcomes)
            num_entities = int(_np.log2(len(full_outcomes))/_np.log2(len(outcomes)))
            
            # If 'std' then create the marginalize dictionary, mapping full bit-string outcomes to elements of ['0','1']
            if marginalize == 'std':         
                marginalize_dict = {}
                for fo in full_outcomes:
                    marginalized_outcomes_list = []
                    for i in fo[0]:
                        marginalized_outcomes_list.append(i)
             
                    marginalize_dict[fo[0]] = tuple(marginalized_outcomes_list)
            
            if marginalize == 'usr':   
                assert(marginalize_dict is not None)
            
            # Creat the inverse of the marginalize_dict, mapping (entity,outcome) pairs to full outcomes
            marginalize_invdict = {}

            for q in range(0,num_entities):
                for o in outcomes:
                    marginalize_invdict[q,o] = []
                    for key in list(marginalize_dict.keys()):
                        qopair = marginalize_dict[key]
                        if qopair[q] == o:
                            marginalize_invdict[q,o].append(key)
            
            data = _np.zeros((num_sequences,num_entities,num_outcomes,num_timesteps),float)
            timestamps = _np.zeros((num_sequences,num_timesteps),float)

            for s in range(0,num_sequences):
                times_for_seq, junk = ds[indices_to_sequences[s]].timeseries('all')
                timestamps[s,:] = _np.array(times_for_seq)
                for e in range(0,num_entities):
                    for o in range(0,num_outcomes):
                        data[s,e,o,:] = _np.zeros((num_timesteps),float)
                        for fo in marginalize_invdict[e,outcomes[o]]:
                            junk, fodata = _np.array(ds[indices_to_sequences[s]].timeseries(fo,timestamps[s,:]))
                            data[s,e,o,:] += fodata
                        
        all_time_steps = timestamps[:,1:] - timestamps[:,:num_timesteps-1]
        if verbosity > 0:
            if _np.std(all_time_steps) > 0.:
                print("Warning: data is not *exactly* equally spaced.")
              
        # We only overwrite the timestep if it is None.
        if timestep is None:
            timestep = _np.mean(all_time_steps)
        
        data_shape = _np.shape(data)
        
    # --------------------------------------------------------- #
    # Prepare a results object, and store the input information #
    # --------------------------------------------------------- #
    
    # Initialize an empty results object.
    results = _obj.DriftResults()
    
    # Records input information into the results object.
    results.name = name
    results.data = data
    results.number_of_sequences = num_sequences
    results.number_of_timesteps = num_timesteps
    results.number_of_entities = num_entities
    results.number_of_outcomes = num_outcomes   
    results.number_of_counts = counts
    results.outcomes = outcomes
    results.timestamps = timestamps
    results.timestep = timestep       
    results.confidence = confidence
    results.multitest_compensation = multitest_compensation
    results.indices_to_sequences = indices_to_sequences
    
    if indices_to_sequences is not None:
        sequences_to_indices = {}
        for i in range(0,num_sequences):
            sequences_to_indices[indices_to_sequences[i]] = i            
        results.sequences_to_indices = sequences_to_indices
        
    # Provides a warning if the number of timesteps is low enough that the chi2 approximations used in this 
    # function might not be good approximations.
    if verbosity > 0:
        if num_timesteps <=50:
            string = "Warning: Number of timesteps is less than 50."
            string += " The pvalues and statistical signficance thresholds may inaccurate"  
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
    
    pspepo_dof = 1
    pspe_dof = num_outcomes-1
    ps_dof = num_entities*(num_outcomes-1)
    pe_dof = num_sequences*(num_outcomes-1)
    global_dof = num_sequences*num_entities*(num_outcomes-1)
    
    
    global_dof_reduction = 0
    for s in range(0,num_sequences):
        if _np.sum(ps_power_spectrum[s,:]) < 1.:
            global_dof_reduction+= 1
            
    global_dof = global_dof - global_dof_reduction
    # Todo at some point: add a dof reduction for pspe, ps and pe analyses.

    pspepo_significance_threshold_1test = _stat.maxpower_threshold_chi2(confidence, num_timesteps, pspepo_dof)
    pspe_significance_threshold_1test = _stat.maxpower_threshold_chi2(confidence, num_timesteps, pspe_dof)
    ps_significance_threshold_1test = _stat.maxpower_threshold_chi2(confidence, num_timesteps, ps_dof)
    pe_significance_threshold_1test = _stat.maxpower_threshold_chi2(confidence,num_timesteps,pe_dof)
    global_significance_threshold_1test = _stat.maxpower_threshold_chi2(confidence, num_timesteps,global_dof)
    
    pspepo_numtests = num_sequences*num_entities*(num_outcomes-1)
    pspe_numtests = num_sequences*num_entities
    ps_numtests = num_sequences
    pe_numtests = num_entities
    global_numtests = 1
    
    # Todo: change some of these to the sidak correction where appropriate
    pspepo_confidence_classcompensation = _hyp.bonferroni_correction(confidence,pspepo_numtests)
    pspe_confidence_classcompensation = _hyp.bonferroni_correction(confidence,pspe_numtests)    
    ps_confidence_classcompensation = _hyp.bonferroni_correction(confidence,ps_numtests)
    pe_confidence_classcompensation = _hyp.bonferroni_correction(confidence,pe_numtests)
    global_confidence_classcompensation = _hyp.bonferroni_correction(confidence,global_numtests)
    
    pspepo_significance_threshold_classcompensation = _stat.maxpower_threshold_chi2(pspepo_confidence_classcompensation, 
                                                                                    num_timesteps, pspepo_dof)
    pspe_significance_threshold_classcompensation = _stat.maxpower_threshold_chi2(pspe_confidence_classcompensation,
                                                                                  num_timesteps, pspe_dof)
    ps_significance_threshold_classcompensation = _stat.maxpower_threshold_chi2(ps_confidence_classcompensation, num_timesteps, 
                                                                                ps_dof)
    pe_significance_threshold_classcompensation = _stat.maxpower_threshold_chi2(pe_confidence_classcompensation, 
                                                                                num_timesteps,pe_dof)
    global_significance_threshold_classcompensation = _stat.maxpower_threshold_chi2(global_confidence_classcompensation,
                                                                                    num_timesteps,global_dof)
    
    #Perhaps uncomment and add in the "all" multitest_compensation option.
    #total_numtests = global_numtests
    #if num_outcomes > 1:
    #    total_numtests += pspepo_numtests
    #if num_entities > 1:
    #    total_numtests += pspe_numtests
    #    if num_sequences > 1:
    #        total_numtests += pe_numtests
    #if num_sequences > 1:
    #    total_numtests += global_numtests
    
    if multitest_compensation == 'none':

        pspepo_significance_threshold = pspepo_significance_threshold_1test
        pspe_significance_threshold = pspe_significance_threshold_1test
        ps_significance_threshold = ps_significance_threshold_1test
        pe_significance_threshold = pe_significance_threshold_1test
        global_significance_threshold = global_significance_threshold_1test
        
    if multitest_compensation == 'class':
        
        pspepo_significance_threshold = pspepo_significance_threshold_classcompensation 
        pspe_significance_threshold = pspe_significance_threshold_classcompensation 
        ps_significance_threshold = ps_significance_threshold_classcompensation 
        pe_significance_threshold = pe_significance_threshold_classcompensation 
        global_significance_threshold = global_significance_threshold_classcompensation 
    
    # Initialize arrays for the per-sequence, per-entity, per-outcome results
    pspepo_max_power = _np.zeros((num_sequences,num_entities,num_outcomes),float)
    pspepo_pvalue = _np.zeros((num_sequences,num_entities,num_outcomes),float)
    pspepo_drift_detected = _np.zeros((num_sequences,num_entities,num_outcomes),bool)       
    pspepo_drift_frequencies = {}   
    pspepo_reconstruction = _np.zeros(data_shape,float)  
    pspepo_reconstruction_uncertainty = _np.zeros((num_sequences,num_entities,num_outcomes),float) 
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
                num_kept_modes = 1 + len(pspepo_drift_frequencies[s,q,o])
                
                # Create the reconstructions, using the standard single-pass Fourier filter
                pspepo_null[s,q,o,:] = _np.mean(data[s,q,o,:])*_np.ones(num_timesteps,float)/counts
            
                if pspepo_max_power[s,q,o] > pspepo_significance_threshold:
                    pspepo_drift_detected[s,q,o] = True
                    pspepo_raw_estimated_modes[pspepo_power_spectrum < pspepo_significance_threshold] = 0.
                
                    # Here divide by counts, to get something in 0,1 (in the noise free case).
                    pspepo_reconstruction[s,q,o,:] = _sig.IDCT(pspepo_raw_estimated_modes[s,q,o,:], 
                                                               null_hypothesis=pspepo_null[s,q,o,:],
                                                               counts=counts)/counts
                    # Todo: make this renormalizer makes sense with multi-outcome data. Currently
                    # it does not, in so much as the summed probability does not add up to 1.
                    pspepo_reconstruction[s,q,o,:] = _sig.renormalizer(pspepo_reconstruction[s,q,o,:],method='logistic')
                    pspepo_reconstruction_power_spectrum[s,q,o,:] =  _sig.DCT(pspepo_reconstruction[s,q,o,:], 
                                                                      null_hypothesis=pspepo_null[s,q,o,:],
                                                                      counts=counts)   
                else:
                    pspepo_reconstruction[s,q,o,:] = pspepo_null[s,q,o,:]                
                    pspepo_reconstruction_power_spectrum[s,q,o,:] = _np.zeros(num_timesteps,float)
                    
                pspepo_reconstruction_uncertainty[s,q,o] = _np.sqrt(num_kept_modes/num_timesteps)
            
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
    results.pspepo_confidence_classcompensation = pspepo_confidence_classcompensation
    results.pspepo_significance_threshold = pspepo_significance_threshold
    results.pspepo_significance_threshold_1test = pspepo_significance_threshold_1test
    results.pspepo_significance_threshold_classcompensation = pspepo_significance_threshold_classcompensation
    results.pspepo_drift_detected = pspepo_drift_detected
    results.pspepo_reconstruction = pspepo_reconstruction
    results.pspepo_reconstruction_uncertainty = pspepo_reconstruction_uncertainty
    results.pspepo_reconstruction_power_spectrum = pspepo_reconstruction_power_spectrum
    results.pspepo_reconstruction_powerpertimestep = pspepo_reconstruction_powerpertimestep
    results.pspepo_dof = pspepo_dof
    
    # Write the results for this analysis into the results object.
    results.pspe_power_spectrum = pspe_power_spectrum
    results.pspe_drift_frequencies = pspe_drift_frequencies
    results.pspe_max_power = pspe_max_power
    results.pspe_pvalue = pspe_pvalue
    results.pspe_confidence_classcompensation = pspe_confidence_classcompensation
    results.pspe_significance_threshold = pspe_significance_threshold
    results.pspe_significance_threshold_1test = pspe_significance_threshold_1test
    results.pspe_significance_threshold_classcompensation = pspe_significance_threshold_classcompensation
    results.pspe_drift_detected = pspe_drift_detected
    results.pspe_reconstruction_power_spectrum = pspe_reconstruction_power_spectrum
    results.pspe_reconstruction_powerpertimestep = pspe_reconstruction_powerpertimestep
    results.pspe_dof = pspe_dof
    
    # Write the results for this analysis into the results object.
    results.ps_power_spectrum = ps_power_spectrum
    results.ps_drift_frequencies = ps_drift_frequencies
    results.ps_max_power = ps_max_power
    results.ps_pvalue = ps_pvalue
    results.ps_confidence_classcompensation = ps_confidence_classcompensation
    results.ps_significance_threshold = ps_significance_threshold
    results.ps_significance_threshold_1test = ps_significance_threshold_1test
    results.ps_significance_threshold_classcompensation = ps_significance_threshold_classcompensation
    results.ps_drift_detected = ps_drift_detected
    results.ps_reconstruction_power_spectrum = ps_reconstruction_power_spectrum
    results.ps_reconstruction_powerpertimestep = ps_reconstruction_powerpertimestep
    results.ps_dof = ps_dof
    
    # Write the results for this analysis into the results object.
    results.global_power_spectrum = global_power_spectrum
    results.global_drift_frequencies = global_drift_frequencies
    results.global_max_power = global_max_power
    results.global_pvalue = global_pvalue
    results.global_confidence_classcompensation = global_confidence_classcompensation
    results.global_significance_threshold = global_significance_threshold
    results.global_significance_threshold_1test = global_significance_threshold_1test
    results.global_significance_threshold_classcompensation = global_significance_threshold_classcompensation
    results.global_drift_detected = global_drift_detected
    results.global_reconstruction_power_spectrum = global_reconstruction_power_spectrum
    results.global_reconstruction_powerpertimestep = global_reconstruction_powerpertimestep
    results.global_dof = global_dof
    results.global_dof_reduction = global_dof_reduction
    
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
    results.pe_confidence_classcompensation = pspepo_confidence_classcompensation
    results.pe_significance_threshold = pe_significance_threshold
    results.pe_significance_threshold_1test = pe_significance_threshold_1test
    results.pe_significance_threshold_classcompensation = pe_significance_threshold_classcompensation 
    results.pe_max_power = pe_max_power
    results.pe_pvalue = pe_pvalue
    results.pe_drift_detected = pe_drift_detected
    results.pe_drift_frequencies = pe_drift_frequencies
    results.pe_reconstruction_power_spectrum = pe_reconstruction_power_spectrum
    results.pe_reconstruction_powerpertimestep = pe_reconstruction_powerpertimestep
    results.pe_dof = pe_dof
    
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
    composite_confidence = _hyp.generalized_bonferroni_correction(confidence,weights,numtests=numtests)

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
