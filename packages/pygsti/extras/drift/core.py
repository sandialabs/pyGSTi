"""Core routines for detecting and characterizing drift with time-stamped data"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

from . import signal as _sig
from . import results as _dresults
from ...tools import hypothesis as _hyp
from . import statistics as _stats
from ... import objects as _obj

import numpy as _np
import warnings as _warnings
import itertools as _itertools

def do_basic_drift_characterization(ds, significance=0.05, transform='auto', setting='fastest'):

    return 0

# def do_oneQubit_drift_characterization(ds, significance=0.05, marginalize='auto', transform='DCT',
#                                      spectrafrequencies='auto', testfrequencies='all', 
#                                      enforceConstNumTimes='auto', control='FDR', 
#                                      modelSelection='local', verbosity=1, name=None)


#     do_general_drift_characterization(ds, significance=significance, marginalize=False, transform=transform,
#                                       spectrafrequencies=spectrafrequencies, testfrequencies=testfrequencies,
#                                       groupoutcomes=groupoutcomes, enforceConstNumTimes=enforceConstNumTimes,
#                                       whichTests=(('avg','avg','avg'), ('avg','per','avg')), betweenClassCorrection=True, 
#                                       inClassCorrection=('Bonferroni','BH','BH','BH'), 
#                             modelSelection='perSeqPerEnt', verbosity=1, name=None)

#     return 0

# def do_rb_drift_characterization(ds, significance=0.05, transform='DCT', control='FDR'):

#     return 0

# def do_gst_drift_characterization(ds, significance=0.05, prepStrs=None, effectStrs=None, germList=None, 
#                                   maxLengthList=None, transform='DCT', control='FDR'):

#     """
#     prepStrs : list of GateStrings
#         List of the preparation fiducial gate strings, which follow state
#         preparation.

#     effectStrs : list of GateStrings
#         List of the measurement fiducial gate strings, which precede
#         measurement.

#     germList : list of GateStrings
#         List of the germ gate strings.

#     maxLengthList : list of ints
#         List of maximum lengths. A zero value in this list has special
#         meaning, and corresponds to the LGST sequences.
#     """
#     return 0

def format_data(ds, marginalize='auto', groupoutcomes=None, enforceConstNumTimes=False, name=None):
    """"
    Formats time-series data, in the form of DataSet containing time-series data, into the format 
    required by a DriftResults object, writes this into an empty DriftResults object, and returns
    this DriftResults object. This function performs a range of useful formattings of arbitrary
    time-series data (e.g., marginalization), but does not cover all possible ways to format
    the data *and* it requires that data to first be recorded into a DataSet, which is not strictly
    necessary to perform the drift analysis. So it may sometimes be preferable to format the data 
    manually, by creating an empty DriftResults object and using the .add_formatted_data() method.

    Parameters
    ----------
    ds : DataSet object
        The DataSet, which must contain time-series data, to format.

    marginalize : {'auto',True,False}, optional
        Todo.

    groupoutcome : dict or None, optional
        Todo: add this functionality in (currently it does not do anything).


     enforceConstNumTimes : bool, optional
        If True, if the number of data points varies between GateStrings (i.e., the time series is
        a different length for different sequences), then the ... todo

    name : None or str.
        Tdo.

    Returns
    -------
    DriftResults object
        A DriftResults object containing the formatted data, that can then be used to implement
        spectral analysis of this formatted time-series data.
    """
    if groupoutcomes is not None:
        assert(not marginalize == True), "Cannot marginalize the data *and* format the data according to a `groupoutcomes` dictionary!"

    # Initialize an empty results object, with the name written in.
    results = _dresults.DriftResults(name=name)

    num_sequences = len(list(ds.keys()))
    gatestringlist = list(ds.keys())
    
    timestamps = []
    num_timesteps = []
    counts = []

    # Find the timestamps for each sequence, and the total number of counts at each timestamp.
    for i in range(num_sequences):
        # Find the set of all timestamps and total counts
        timesforseq, countsforseq = ds[gatestringlist[i]].timeseries('all')
        # Record the number of timestamps for this sequence
        num_timesteps.append(len(timesforseq))
        timestamps.append(timesforseq)
        # Check that the number of clicks is constant over all timestamps
        assert(_np.std(_np.array(countsforseq)) <= 1e-15), "The number of total clicks must be the same at every timestamp!"
        # Record the counts-per-timestamp for this sequence
        counts.append(countsforseq[0])
    
    # Check that the number of counts is independent of sequence. (we could drop this requirement if we need to).
    counts = _np.array(counts)
    assert(_np.std(counts) <= 1e-15), "Number of counts per time-step must be the same for all sequences!"
    counts = counts[0]
   
    # Check whether the number of timesteps is independent of sequence.    
    if _np.std(_np.array(num_timesteps)) > 1e-15:
        constNumTimes = False
    else:
        constNumTimes = True
        num_timesteps = num_timesteps[0]

    # If we are demanding that the number of timestamps is independent of sequence, update the timestamps
    # by dropping as many of the final timestamps as necessary to have a fixed number of timestamps.
    if enforceConstNumTimes and not constNumTimes:
        num_timesteps = min(num_timesteps)
        timestamps = [timestamps[i][0:num_timesteps] for i in range(num_sequences)] 
    else:
        # We do this, because this gets recorded in the results as whether we *have* enforced this.
        enforceConstNumTimes = False
    
    if isinstance(marginalize,str):
        assert(marginalize == 'auto')
        if len(list(ds.get_outcome_labels())) > 4:
            marginalize = True
        else:
            marginalize = False

    assert(marginalize == True or marginalize == False)

    if marginalize:
        if verbosity > 0:
            print(" - marginalizing data...")
        full_outcomes = ds.get_outcome_labels()
        for s in full_outcomes[0][0]:
            assert(s == '0' or s == '1'), "If marginalizing, this function assumes that the outcomes are strings of 0s and 1s!"
            if len(s) == 1:
                # Over-writes marginalize if the marginalization will be trivial
                marginalize = False

    if marginalize:
        num_entities = len(full_outcomes[0][0])
        outcomeslist = ['0','1']
        timeseries = {}
        for e in range(num_entities):
            timeseries[e] = {}
            tempdata = pygsti.construction.filter_dataset(ds,sectors_to_keep=i)
            for s in range(num_sequences):
                timeseries[e][s] = {}
                for o in range(2):
                    junk, timeseries[e][s][o] = tempdata[gatestringlist[s]].timeseries(outcomeslist[o],timestamps[s])

    else:
        outcomeslist = list(ds.get_outcome_labels())
        num_outcomes = len(list(ds.get_outcome_labels()))
        timeseries = {}
        timeseries[0] = {}
        for s in range(num_sequences):
            timeseries[0][s] = {}
            for o in range(num_outcomes):
                junk, timeseries[0][s][o] = ds[gatestringlist[s]].timeseries(outcomeslist[o],timestamps[s])

    results.add_formatted_data(timeseries, timestamps, gatestringlist, outcomeslist, counts, constNumTimes, 
                               enforcedConstNumTimes=enforceConstNumTimes, marginalized=marginalize)
    return results

def calculate_power_spectra(results, transform='DCT', frequenciesInHz='auto'):
    """"
    Calculates power spectra for all of the time-series data recorded in the `results`
    DriftResults objects, and writes them into these results. These power spectra are
    on the per-entity (e.g., per qubit, for marginalized data) per-sequence, per-outcome
    level.

    Parameters
    ----------
    results : DriftResults object
        A drift results object, containing data that has been formatted for a time-series
        analysis, either manually or using the `format_data()` function. The power spectra
        and Fourier modes calculated by this function are written into this results object.
        Note that if this results object already contains power spectra and modes these will
        be written over.

    transform : str, optional
        The type of Fourier transform to use. The allowed options are 'DCT', 'LSP' and 'DFT'.
        The only tested option is currently 'DCT', and the code probably doesn't work for the
        other options.

    frequenciesInHz : list/array or str, optional
        The frequencies to calculate the spectrum at. This is only used if `transform` is
        'LSP', in which case the power can be calculated at any and any number of frequencies. 
        For all other cases the frequencies, are defined by the transform and so this input
        is ignored. In the default case of 'auto' the LSP frequencies are those of the DCT/DFT.
        These frequencies should be stated in Hz, assuming that the time-stamps have been recoreded
        into the results object in seconds.

    Returns
    -------
    None
        The spectra are written into the results object and are not returned.
    """

    shape = (results.number_of_entities,results.number_of_sequences,results.number_of_outcomes,results.maxnumber_of_timesteps)
    spectra = _np.zeros(shape,float)
    
    if transform == 'DCT':

        # Regardless of the input, we write over that and use the DCT frequencies
        frequenciesInHz = _sig.frequencies_from_timestep(results.meantimestepGlobal, results.maxnumber_of_timesteps) 
        # Todo : explain
        assert(results.constNumTimes == True)
        # We can store the modes and spectra in an array, because fixed-T must be enforced.
        modes = _np.zeros(shape,float)
        for q in range(results.number_of_entities): 
            for s in range(results.number_of_sequences):
                for o in range(results.number_of_outcomes):
                    x = results.timeseries[q][s][o] 
                    modes[q,s,o,:] = _sig.DCT(x, counts=results.number_of_counts)

        spectra = modes**2

    elif transform == 'LSP':

        if isinstance(frequenciesInHz,str):
            assert(frequenciesInHz == 'auto')
            # We default to the DCT/DFT frequencies. We could allow for different frequencies for different sequences,
            # but that would prevent the spectrum averaging, so we don't.
            frequenciesInHz = _sig.frequencies_from_timestep(results.meantimestepGlobal, results.maxnumber_of_timesteps)
   
        # We can store the modes and spectra in an array regardless of fixed T, because we calculate the
        # periodgram at a fixed set of frequencies.
        modes = None
        for q in range(0,num_entities): 
            for s in range(0,num_sequences):
                for o in range(0,num_outcomes):
                    x = results.timeseries[q][s][o] 
                    t = results.timestamps[q][s][o]
                    specta[q,s,o,:] = _sig.LSP(t, x, frequenciesInHz, counts=results.number_of_counts)

    results.add_spectra(frequenciesInHz, spectra, transform, modes)

    return None

def implement_drift_detection(results, significance=0.05, testFreqInds=None,
                              whichTests=(('avg','avg','avg'), ('per','avg','avg'), 
                              ('per','per','avg')), betweenClassCorrection=True, 
                              inClassCorrection=('FWER','FWER','FDR','FDR'), name='detection',
                              overwrite=False, verbosity=1):
    """
    
    # {'global':True, 'perEnt':True, 'perSeq':False,'perSeqPerEnt':True,
                              'perEntPerSeqPerOut':True}
    #{'perEnt':'Bonferroni', 'perSeq':'BH', 'perOut':'BH', 'local':'BH'})

    Parameters
    ----------

    significance : float, optional
        The global significance level. This is ...

    testfrequ
    """

    # These assumptions are not crucial *but* the current code will probably break without them

    #
    #
    # Todo : we are violation these! In the updated test schedule.
    #
    #
    validWhichTests = []
    validWhichTests.append(('avg','avg','avg'))
    validWhichTests.append(('per','avg','avg'))
    validWhichTests.append(('per','per','avg'))
    validWhichTests.append(('per','per','per'))

    for test in whichTests: assert(test in validWhichTests)

    # These assumptions are not crucial *but* the current code will probably break without them
    validInClassCorrections = []
    validInClassCorrections.append(('FWER','FDR','FDR','FDR'))
    validInClassCorrections.append(('FWER','FWER','FDR','FDR'))
    validInClassCorrections.append(('FWER','FWER','FWER','FDR'))
    validInClassCorrections.append(('FWER','FWER','FWER','FWER'))

    assert(inClassCorrection in validInClassCorrections)

    # Work out which tests we're actually going to do.
    #print(whichTests)
    whichTestsPointers = {}

    if results.number_of_entities == 1:
        whichTestsPointers[0] = 'avg'
    if results.number_of_sequences == 1:
        whichTestsPointers[1] = 'avg'
    if results.number_of_outcomes == 2:
        whichTestsPointers[2] = 'avg'

    #print(whichTestsPointers)

    whichTestsUpdated = []
    for test in whichTests:
        equivalenttest = tuple([whichTestsPointers.get(i,test[i]) for i in range(3)])
        if equivalenttest not in whichTestsUpdated:
            whichTestsUpdated.append(equivalenttest)

    #print(whichTestsUpdated)

    existsPerTest = _np.array([False,False,False])
    for test in whichTestsUpdated:
        for i in range(3):
            if not existsPerTest[i]:
                if test[i] == 'per':
                    existsPerTest[i] = True

    # The points in the tuple to pad with 'avg'
    avgpad = {}
    for test in whichTestsUpdated:
        avgpad[test] = _np.zeros(3,bool)
        for i in range(3):
            if test[i] == 'avg':
                 avgpad[test][i] = True

    control = 'gFWER'
    if not betweenClassCorrection and len(whichTestsUpdated) > 1:
        control = 'cFWER'
    for i in range(4):
        ctype = inClassCorrection[i]
        if i == 3 or existsPerTest[i]:
            #print("Adjusting for this ctype")
            if ctype == 'none':
                control = 'none'
            elif control != 'none' and ctype == 'FDR':
                control = control[0] + 'FDR'
            #print(control)

    #print(control)

    #results.add_drift_detection_results(significance, control, whichTests, betweenClassCorrection, inClassCorrection)

    # Checks that 
    if min(results.number_of_timesteps) <= 25: 
        _warnings.warn('At least some sequences have very few timesteps (less than 25). ' + \
            'This means that the statistical signficance thresholds and p-values, ' +\
            'which rely on the central limit theorem to be valid, may be inaccurate.')

    # The baseline degrees of freedom for each type of power spectra
    #dofstuple = (results.number_of_entities,results.number_of_sequences,results.number_of_outcomes-1)
    numteststuple = (results.number_of_entities,results.number_of_sequences,results.number_of_outcomes)
    # # The number of spectra tested in the class (so not including the contribution of testing a spectra
    # # at multiple frequencies), and the baseline degrees-of-freedom per test.
    # dofPerSpectrumInClass = {}
    # numTestsInClass = {}
    # for test in whichTestsUpdated:
    #     dofPerSpectrumInClass[test] = 1
    #     numTestsInClass[test] = 1
    #     for ind, ttype in enumerate(test):
    #         if ttype == 'avg':
    #             dofPerSpectrumInClass[test] = dofPerSpectrumInClass[test]*dofstuple[ind]  
    #         else:
    #             numTestsInClass[test] = numTestsInClass[test]*numteststuple[ind]

    significancePerTestClass = {}
    if betweenClassCorrection:
        SPTC = significance/len(whichTestsUpdated)
    else:
        SPTC = significance
    # Store in a dictionary, to allow for weights in future if we want them.
    for test in whichTestsUpdated:
        significancePerTestClass[test] = SPTC

    # Todo: Deal with reducing the DOF appropriately here. Probably do when 
    # running through the tests, as have to look at all the data to check.
    # Reduce the global dofs for sequences ....    
    # global_dof_reduction = 0
    # for s in range(0,num_sequences):
    #     if _np.sum(ps_power_spectrum[s,:]) < 1.:
    #         global_dof_reduction+= 1
            
    # global_dof = global_dof - global_dof_reduction
    # # Todo: add a dof reduction for pspe, ps and pe analyses.

    if testFreqInds is not None:
        testFreqInds = _np.sort(testFreqInds)
        usedTestFreqInds = testFreqInds.copy()
    else:
        usedTestFreqInds = _np.arange(results.number_of_frequencies)

    driftdetectedinClass = {}
    driftdeteted = False
    power_significance_pseudothreshold = {}
    sigFreqInds = {}

    # Implement the statistical tests
    for test in whichTestsUpdated:

        sig = significancePerTestClass[test] 
        dof = results.get_dof_per_spectrum_in_class(*test)
        # The number of spectra tested, not the number of frequencies tested in total
        numtest = results.get_number_of_spectra_in_class(*test)
        numfreqs = len(usedTestFreqInds)
        driftdetectedinClass[test] = False

        if verbosity > 0:
            print("   - Implementing statistical test at significance {} with structure: {}".format(sig,test))
            print("      - In-class correction being used is {}".format(inClassCorrection))
            print("      - Number of tests in class is {}".format(numtest))
            print("      - Testing at {} frequencies".format(numfreqs))
            print("      - Baseline degrees-of-freedom for the power spectrum is {}".format(dof))

        spectra = results.get_spectra_set(test, freqInds=testFreqInds)

        # If we are just doing Bonferroni corrections on everything, we do this using different code
        # as it can be done faster.
        if inClassCorrection == ('FWER','FWER','FWER','FWER'):

            power_significance_pseudothreshold[test] = _stats.power_significance_threshold(sig, numtest*numfreqs, dof)
            if verbosity > 0:
                print("      - Power significance threshold is: {}".format(power_significance_pseudothreshold[test]))
            it = []
            shape = []
            if test[0] == 'per':
                it.append(range(results.number_of_entities))
                shape.append(results.number_of_entities)
            #else:
            #    it.append(('avg',))
            if test[1] == 'per':
                it.append(range(results.number_of_sequences))
                shape.append(results.number_of_sequences)
            #else:
            #    it.append(('avg',))
            if test[2] == 'per':
                it.append(range(results.number_of_outcomes))
                shape.append(results.number_of_outcomes)
            #else:
            #    it.append(('avg',))
            shape.append(len(usedTestFreqInds))
            for tup in _itertools.product(*it):
                #print(test, tup)
                tupit = 0
                paddedtup = []
                for i in range(3):
                    if avgpad[test][i]: paddedtup.append('avg')
                    else:
                        paddedtup.append(tup[tupit])
                        tupit += 1
                paddedtup = tuple(paddedtup)
                #print(paddedtup)
                sigFreqInds[paddedtup] = list(usedTestFreqInds.copy()[spectra[tup] > power_significance_pseudothreshold[test]])
                if len(sigFreqInds[paddedtup]) > 0:
                    driftdetectedinClass[test] = True
                    driftdetected = True
                #print(sigFreqInds[paddedtup])
        
        # If we're doing FDR control we go into this bit of code
        else:
            localsig = sig
            fwernum = 1
            itfwer = []
            itfdr = []
            for ind, t in enumerate(test):
                if t == 'per':
                    correction = inClassCorrection[ind]
                    if correction == 'FWER':
                        fwernum = fwernum*numteststuple[ind]
                        itfwer.append(range(numteststuple[ind]))
                    if correction == 'FDR':
                        itfdr.append(range(numteststuple[ind]))
            itfdr.append(range(len(usedTestFreqInds)))
            # The number of test statistics we are doing the Ben-Hoch procedure for.
            fdrnumtests = len(usedTestFreqInds)*numtest//fwernum
            localsig = sig/fwernum

            if verbosity > 0:
                print("      - Implementing {} Benjamini-Hockberg procedure statistical tests each containing {} tests.".format(fwernum,fdrnumtests))
                print("      - Local statistical significance for each Benjamini-Hockberg procedure is {}".format(localsig))

            power_significance_pseudothreshold[test] = {}
            
            if verbosity > 0:
                print("      - Generating Benjamini-Hockberg power quasi-threshold...",end='')
                
            quasithreshold = _stats.power_fdr_quasithreshold(localsig, fdrnumtests, dof)
            
            if verbosity > 0:
                print("complete.")
                print("      - Implementing the Benjamini-Hockberg procedures...",end='')
                if verbosity >1:
                    print('')

            #fdrtuples = [tup for tup in _itertools.product(*itfdr)]
            #print(fdrtuples)

            # Goes through each FDR set, and implements the Benjamini-Hockberg procedure.
            for tup in _itertools.product(*itfwer):

                if len(_np.shape(spectra)) > 1:
                    fdrPowers = spectra[tup].flatten()
                else:
                    fdrPowers = spectra[tup].copy()

                # The labels that will go with the elements in the flattened spectra.
                # This needs to be refreshed for each loop through with a new tuple.
                fdrtuples = [tup for tup in _itertools.product(*itfdr)]

                fdrPowers, fdrtuples = zip(*sorted(zip(fdrPowers,fdrtuples)))
                fdrPowers = _np.array(list(fdrPowers))
                #print(quasithreshold[0],quasithreshold[-1],fdrPowers[0],fdrPowers[-1])
                #print(fdrtuples[-10:])
                #print(fdrPowers[-10:])

                dif = fdrPowers - quasithreshold
                try: 
                    ind = next(i for i,v in enumerate(dif) if v > 0.0)
                    sigtuples = fdrtuples[ind:]
                    if verbosity > 1:
                        print("         - {} significant test statistics found for test-set {}!".format(fdrnumtests-ind,tup))
                    driftdetectedinClass[test] = True
                    driftdetected = True
                    power_significance_pseudothreshold[test][tup] = quasithreshold[ind]

                except:
                    if verbosity > 1:
                        print("         - 0 significant test statistics found for test-set {}.".format(tup))
                    power_significance_pseudothreshold[test][tup] = quasithreshold[-1]
                    sigtuples = []

                for stup in sigtuples:
                    tupit = 0
                    tuprebuilt = tup+stup[:-1]
                    paddedtup = []
                    for i in range(3):
                        if avgpad[test][i]: paddedtup.append('avg')
                        else:
                            paddedtup.append(tuprebuilt[tupit])
                            tupit += 1
                    paddedtup = tuple(paddedtup)
                    try:
                        sigFreqInds[paddedtup] += [stup[-1],]
                        #print("Successfully added to", tup+stup[:-1])
                    except:
                        #print("First for", tup+stup[:-1])
                        sigFreqInds[paddedtup] = [stup[-1],]              
                

            if verbosity == 1:
                print("complete.")

        if verbosity > 0:
            if driftdetectedinClass[test]:
                print("      - Drift detected! (Sufficient evidence to reject the no-drift null hypothesis).")
            else:
                print("      - Drift not detected. (Not sufficient evidence to reject the no-drift null hypothesis).")

    if betweenClassCorrection:
        betweenClassCorrection = 'bonferroni'
    inClassCorrections = {test:inClassCorrection for test in whichTestsUpdated}

    # Todo : needs to pass on the pointers.
    results.add_hypothesis_testing_results(significance=significance, testClasses=whichTestsUpdated, 
                                        betweenClassCorrection=betweenClassCorrection, inClassCorrections=inClassCorrections,
                                        control=control, driftdetected=driftdetected, driftdetectedinClass=driftdetectedinClass,
                                        testFreqInds=testFreqInds, sigFreqIndsinClass=sigFreqInds,
                                        powerSignificancePseudothreshold=power_significance_pseudothreshold,
                                        significanceForClass=significancePerTestClass, name=name, overwrite=overwrite)

    return None

def estimate_probability_trajectories(results, modelSelectionMethod=('per','per','avg'), estimator='FFLogistic', 
                                      estimatorSettings=[], verbosity=1, overwrite=False, setasDefault=True):
    """
    Todo:
    """
    # hyperparameters = {}
    # parameters = {}
    # reconstructions = {}
    # uncertainties = {}
    # auxDict = {}

    # assert(modelSelectionMethod[2] == 'avg'), "The model selection must use outcome-averaged model selection!"

    # for e in result.number_of_entities:
    #     for s in results.number_of_sequences:
            
    #         # Get the time-series data for this entity and sequence, as a dict with keys from 0 to the
    #         # number of possible measurement outcomes - 1.
    #         timeseries = results.timeseries[e][s]
    #         timestamps = results.timestamps[e][s]

    #         try:
    #             modes = results.modes[e,s,:]
    #         except:
    #             modes = None

    #         if modelSelectionMethod[0] == 'per':
    #             entkey = e
    #         else: entkey = 'avg'
    #         if modelSelectionMethod[1] == 'per':
    #             seqkey = s
    #         else: seqkey = 'avg'
                   
    #         # The hyper-parameters for the DCT (or DFT) estimators are frequency indices
    #         if results.transform != 'LSP':
    #             hyperparams = results.get_drift_frequency_indices(entity=entkey, sequence=seqkey, 
    #                                                               outcome='avg', sort=True)
    #             # We need the 0 mode as a hyper-parameter
    #             hyperparams.insert(0,0)
    #         # The hyper-parameters for the LSP estimators are real-valued frequencies.
    #         else:
    #             hyperparams = results.get_drift_frequencies(entity=entkey, sequence=seqkey, 
    #                                                         outcome='avg', sort=True)
    #             # We need the 0. frequency as a hyper-parameter
    #             hyperparams.insert(0,0.)

    #         params, recon, uncert, aux = _est.estimate_probability_trace(timeseries, timestamps, results.transform, estimator,
    #                                                                      hyperparameters, modes=modes, estimatorSettings)
            
    #         # Write the results into the dicts, to then be passed to the results DriftResults object.
    #         hyperparameters[e,s] = hyperparameters
    #         parameters[e,s] = params
    #         reconstructions[e,s] = recon
    #         uncertainties[e,s] = uncert
    #         auxDict[e,s] = aux

    # results.add_reconstructions_set(estimator, modelSelection, hyperparameters, parameters, reconstructions, 
    #                                 uncertainties, auxDict=auxDict, overwrite=overwrite, setasDefault=setasDefault)

    return None

def do_drift_characterization(ds, significance=0.05, marginalize='auto', transform='DCT',
                            spectrafrequencies='auto', testFreqInds=None,
                            groupoutcomes=None, enforceConstNumTimes='auto',
                            whichTests=(('avg','avg','avg'), ('per','avg','avg'), ('per','per','avg')), 
                            betweenClassCorrection=True, inClassCorrection=('FWER','FWER','FDR','FDR'), 
                            modelSelectionMethod=('per','per','avg'), estimator='FFLogistic', verbosity=1, name=None):
    """
    Implements a drift detection and characterization analysis on timeseries data.

    testing_method =
        DoGlobalTest=False, DoPerEntityTest, PerSequence, PerSequencePerEntity, PerSequencePerEntityPerOutcome
        
        {PerEnt:'Bonferroni','none', 'PerSeq':'FDR-BH','Bon','Sidak','none'

        PerOut:'Bonferroni','none', 'local':'FDR-BH','Bon','Sidak'}
        
        ModelSelection : 'PerSeq', 'PerEnt', 'PerSeqPerEnt' or 'Global'
    
    Parameters
    ----------
    ds : pyGSTi DataSet or numpy array
        The time series data to analyze. If this is a DataSet it should contain time series
        data (rather than, e.g., a total counts per-outcome per-GateString). If this is a
        numpy array, it must again contain time series data and it may be either 1, 2, 3
        or 4 dimensional. 

    structuresList:
        {(germ,mfudicial,pfud):[.....], }

    marginalize : str, optional
 
    significance : float, optional

    Returns
    -------
    results : DriftResults object
        The results of the drift analysis. This contains: power spectra, statistical test outcomes,
        drift frequencies, reconstructions of drifting probabilities, all of the input information.

    """ 
    # If the input is already a DriftResults objected with the data pre-formated as necessary.
    if isinstance(ds,_dresults.DriftResults):
        results = ds
        if verbosity > 0: 
            print(" - data is pre-formatted as a DriftResults object, skippng the data formatting step.")
    # The standard case of a DataSet input.
    else:
        # Todo : add various asserts here. 
        assert(isinstance(ds,_obj.dataset.DataSet)), "If the input is not pre-formatted data as a DriftResults object, it must be a pyGSTi DataSet!"
        
        # Work out what the 'auto' value should be for enforcing a fixed number of time stamps.
        if isinstance(enforceConstNumTimes,str):
            assert(enforceConstNumTimes == 'auto')
            if transform == 'LSP':
                enforceConstNumTimes = False
            else:
                enforceConstNumTimes = True

        if transform != 'LSP':
            assert(enforceConstNumTimes), "Except for the LSP transform, the code currently requires that fixed T is enforced!"

        if verbosity > 0: 
            print(" - Formatting the data...",end='')
        # Format the input, and record it inside the results object.
        results = format_data(ds, marginalize=marginalize, enforceConstNumTimes=enforceConstNumTimes, name=name)
        if verbosity > 0: print("complete.")
           
    if verbosity > 0: print(" - Calculating power spectra...",end='')
    
    # Calculate the power spectra: if we're not doing an LSP we can pass anything as the frequencies as it's just ignored.
    calculate_power_spectra(results, transform=transform, frequenciesInHz=spectrafrequencies)
    
    if verbosity > 0: print("complete.")

    if testFreqInds is not None:
        assert(set(testFreqInds).issubset(set(_np.arange(len(results.frequenciesInHz))))), "Invalid frequency indices!"

    if verbosity > 0: 
        print(" - Implementing statistical tests for drift detection...",end='')
        if verbosity > 1: print('')
    
    # Implement the drift detection with statistical hypothesis testing on the power spectra
    implement_drift_detection(results, significance=0.05, testFreqInds=testFreqInds, whichTests=whichTests, 
                              betweenClassCorrection=betweenClassCorrection, inClassCorrection=inClassCorrection,
                              verbosity=verbosity-1)
    
    if verbosity == 1: print("complete.")

    if verbosity > 0: 
        print(" - Estimating the probability trajectories...",end='')
        if verbosity > 1: print('')
    
    # Estimate the drifting probabilities.
    estimate_probability_trajectories(results, modelSelectionMethod=modelSelectionMethod, estimator=estimator, verbosity=verbosity-1)
    
    if verbosity == 1: print("complete.")

    return results