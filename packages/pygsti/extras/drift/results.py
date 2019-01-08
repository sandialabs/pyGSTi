"""Defines the DriftResults class"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

from . import signal as _sig
from . import estimate as _est
from . import statistics as _stats
from ... import objects as _obj

import numpy as _np
import copy as _copy

class DriftResults(object):
    """
    An object to contain the results of a drift detection and characterization analysis. 
    See the various .get and .plot methods for how to access the results, after they have
    been generated. For non-trivial use of this object it is first necessary to add time-series
    data (see the .add_formated_data() method), and then to add the results of drift analyses, using
    the other .add methods. This can be achieved using the core functions of the drift submodule.
    """
    def __init__(self, name=None):
        """
        Initialize a DriftResults object

        Parameters
        ----------
        name : str or None, optional
            A name for the results object.
        """
        self.name = name
        return None

    def add_formatted_data(self, timeseries, timestamps, circuitlist, outcomes, number_of_counts, 
                           constNumTimes, entitieslist=None, enforcedConstNumTimes=None, marginalized=None, 
                           overwrite=False):
        """
        Adds formatted time-series data. This is the first step in using a DriftResults object.

        Todo: add details.
        """
        if not overwrite:
            assert(not hasattr(self,"timeseries")), "This results object already contains timeseries data! To overwrite it you must set `overwrite` to True!"     
        
        # The timeseries is a list of lists of dicts. The first index is entity index (corresponding
        # to the entity at that index of the lsit self.entities). The second index is the 
        # circuit index, (corresponding to the circuit at that index of the list 
        # self.circuitlist). The dictionary keys correspond to the outcomes in self.outcomeslist.
        self.timeseries = timeseries
        self.timestamps = timestamps
        self.circuitlist = circuitlist
        self.number_of_sequences = len(circuitlist)
        self.indexforCircuit = {circuitlist[i]:i for i in range(self.number_of_sequences)}

        self.outcomes = outcomes
        self.number_of_outcomes = len(outcomes)
        self.number_of_counts = number_of_counts    
        self.constNumTimes = constNumTimes

        self.number_of_entities = len(timeseries)
        if entitieslist is not None:
            assert(len(entitieslist) == self.number_of_entities)
        else:
            entitieslist = [str(i) for i in range(self.number_of_entities)]
        self.entitieslist = entitieslist

        self.enforcedConstNumTimes = enforcedConstNumTimes
        self.marginalized = marginalized

        self.number_of_timesteps = [len(timeseries[0][i][self.outcomes[0]]) for i in range(self.number_of_sequences)]
        self.maxnumber_of_timesteps = max(self.number_of_timesteps)
        timesteps = self.get_timesteps()
        self.meantimestepGlobal = _np.mean(timesteps)
        self.stddtimestepGlobal = _np.std(timesteps)
        perseqtimesteps =  [self.get_timesteps(i) for i in range(self.number_of_sequences)]
        self.meantimestepPerSeq = [_np.mean(ts) for ts in perseqtimesteps]
        self.sttdtimestepPerSeq = [_np.std(ts) for ts in perseqtimesteps]

        return None

    def get_timesteps(self, seqInd=None):
        """
        Todo
        """
        if seqInd is None:
            timesteps = []
            for i in range(self.number_of_sequences):
                timesteps = timesteps + list(_np.array(self.timestamps[i][1:]) - _np.array(self.timestamps[i][:self.number_of_timesteps[i]-1]))
        
        else:
            timesteps = _np.array(self.timestamps[seqInd][1:]) - _np.array(self.timestamps[seqInd][:self.number_of_timesteps[seqInd]-1])

        return _np.array(_copy.deepcopy(timesteps))

    # Todo
    #def has_equally_spaced_timestamps(self, stype='absolute', rtol=1e-2):

    #     if stype == 'absolute':
    #         self.get_timesteps()

    def add_spectra(self, frequenciesInHz, spectra, transform, modes=None, overwrite=False):
        """
        Todo
        """
        if not overwrite:
            assert(not hasattr(self,"spectra")), "There are already spectra saved! To overwrite them you must set `overwrite` to True!"
        self.frequenciesInHz = frequenciesInHz.copy()
        self.number_of_frequencies = len(self.frequenciesInHz)
        self.transform = transform

        if modes is not None: 
            self.modes = {}
            self.modes['per','per','per'] = modes.copy()
        else:
            self.modes = None

        self.spectra = {}
        self.spectra['per','per','per'] = spectra.copy()
        self.spectra['per','per','avg'] = _np.mean(self.spectra['per','per','per'],axis=2)
        self.spectra['per','avg','avg'] = _np.mean(self.spectra['per','per','avg'],axis=1)
        self.spectra['avg','avg','avg'] = _np.mean(self.spectra['per','avg','avg'],axis=0)

        return None

    def get_dof_per_spectrum_in_class(self, entity, sequence, outcome):
        """
        Todo
        """
        if not hasattr(self,'_dofPerSpectrumInClass'):
            self._dofPerSpectrumInClass = {}
        try:
            return self._dofPerSpectrumInClass[entity,sequence,outcome]
        except:
            dofPerSpectrumInClass = 1
            if entity == 'avg':
                dofPerSpectrumInClass = dofPerSpectrumInClass*self.number_of_entities
            if sequence == 'avg':
                dofPerSpectrumInClass = dofPerSpectrumInClass*self.number_of_sequences
            if outcome == 'avg':
                dofPerSpectrumInClass = dofPerSpectrumInClass*(self.number_of_outcomes - 1)
            self._dofPerSpectrumInClass[entity,sequence,outcome] = dofPerSpectrumInClass
        
        return dofPerSpectrumInClass

    def get_number_of_spectra_in_class(self, entity, sequence, outcome):
        """
        Todo
        """
        if not hasattr(self,'_numSpectraInClass'):
            self._numSpectraInClass = {}
        try:
            return self._numSpectraInClass[entity,sequence,outcome]
        except:
            numSpectraInClass = 1
            if entity != 'avg':
                numSpectraInClass = numSpectraInClass*self.number_of_entities
            if sequence != 'avg':
                numSpectraInClass = numSpectraInClass*self.number_of_sequences
            if outcome != 'avg':
                numSpectraInClass = numSpectraInClass*self.number_of_outcomes
            self._numSpectraInClass[entity,sequence,outcome] = numSpectraInClass
        
        return numSpectraInClass

    def get_modes_set(self, tup, freqInds=None, store=False):
        """
        Todo
        """
        assert(hasattr(self,'modes')), "The frequency domain data has not been written into this results object!"

        if self.modes is None:
            return None

        else:               
            # The axes along which we are going to average, if we can't just query the dict.
            axis = []
            for i in range(3):
                if tup[i] == 'avg':
                    axis.append(i)

            try:
                modes = self.modes[tup]
            except:
                modes = _np.mean(self.modes['per','per','per'],axis=tuple(axis))

            if freqInds is not None:
                if len(axis) == 0: # Spectra has had no averaging
                    modes = modes[:,:,:,freqInds]
                elif len(axis) == 1: #Spectra has had averaging along 1 axis
                    modes = modes[:,:,freqInds]
                elif len(axis) == 2: # Spectra has had averaging along 2 axes
                    modes = modes[:,freqInds]
                elif len(axis) == 3: # Spectra has had averaging along all 3 axes.
                    modes = modes[freqInds]

            if store:
                assert(freqInds is None), "Only allowed to store the full modes set!"
                self.modes[tup] = modes

            return _copy.deepcopy(modes)

    def get_spectra_set(self, tup, freqInds=None, store=False):
        """
        Todo
        """
        assert(hasattr(self,'spectra')), "No spectra have been saved in this results object!"

        # The axes along which we are going to average, if we can't just query the dict.
        axis = []
        for i in range(3):
            if tup[i] == 'avg':
                axis.append(i)

        try:
            spectra = self.spectra[tup]
        except:
            spectra = _np.mean(self.spectra['per','per','per'],axis=tuple(axis))

        if freqInds is not None:
            if len(axis) == 0: # Spectra has had no averaging
                spectra = spectra[:,:,:,freqInds]
            elif len(axis) == 1: #Spectra has had averaging along 1 axis
                spectra = spectra[:,:,freqInds]
            elif len(axis) == 2: # Spectra has had averaging along 2 axes
                spectra = spectra[:,freqInds]
            elif len(axis) == 3: # Spectra has had averaging along all 3 axes.
                spectra = spectra[freqInds]

        if store:
            assert(freqInds is None), "Only allowed to store the full spectra set!"
            self.spectra[tup] = specta 

        return _copy.deepcopy(spectra)

    def get_spectrum(self, entity='avg', sequence='avg', outcome='avg'):
        """
        Todo
        """
        testclasstup = self._create_testclass_tuple(entity, sequence, outcome)
        spectra = self.get_spectra_set(testclasstup, None)
        dicttup = self._create_dict_tup(entity, sequence, outcome, pad=False)

        return _copy.deepcopy(spectra[dicttup])

    def get_maxpower(self, entity='avg', sequence='avg', outcome='avg', onlyTestedFreqs=False):
        """
        Todo
        """
        # testclasstup = self._create_testclass_tuple(entity, sequence, outcome)
        # print(testclasstup)
        # print(self._testFreqInds)
        # spectra = self.get_spectra(testclasstup, self._testFreqInds)
        # dicttup = self._create_dict_tup(entity, sequence, outcome, pad=False)

        spectrum = self.get_spectrum(entity, sequence, outcome)
        if not onlyTestedFreqs or self._testFreqInds is None:
            maxpower = _np.max(spectrum)
        else:
            maxpower = _np.max(spectrum[self._testFreqInds])

        return maxpower

    def get_maxpower_pvalue(self, entity='avg', sequence='avg', outcome='avg'):
        """
        Todo
        """
        classtup = self._create_testclass_tuple(entity, sequence, outcome)
        maxpower = self.get_maxpower(entity, sequence, outcome)
        dof = self.get_dof_per_spectrum_in_class(*classtup)
        maxpower_pvalue = _stats.power_to_pvalue(maxpower,dof)

        return maxpower_pvalue

    def add_drift_detection_results(self, significance, testClasses, betweenClassCorrection, inClassCorrections,
                                    control, driftdetected, driftdetectedinClass, testFreqInds, sigFreqIndsinClass,
                                    powerSignificancePseudothreshold, significanceForClass, name='detection', overwrite=False,
                                    settodefault=False):
        """
        Todo
        """
        if not hasattr(self,"driftdetected"):
            self.significance = {}
            self._testClasses = {}
            self._betweenClassCorrection = {}
            self._inClassCorrections = {}
            self.control = {}
            self.driftdetected = {}
            self._driftdetectedinClass = {}
            self._testFreqInds = {}
            self._sigFreqIndsinClass = {}
            self._powerSignificancePseudothreshold = {}
            self._significanceForClass = {}
            self.defaultdetectorkey = name

        if not overwrite:
            assert(name not in self.significance.keys()), "Already contains drift detection results with this name! To overwrite them you must set `overwrite` to True!"    
        
        if settodefault:
            self.defaultdetectorkey = name

        self.significance[name] = significance
        self._testClasses[name] = testClasses
        self._betweenClassCorrection[name] = betweenClassCorrection
        self._inClassCorrections[name] = inClassCorrections
        self.control[name] = control
        self.driftdetected[name] = driftdetected
        self._driftdetectedinClass[name] = driftdetectedinClass
        self._testFreqInds[name] = testFreqInds
        self._sigFreqIndsinClass[name] = sigFreqIndsinClass
        self._powerSignificancePseudothreshold[name] = powerSignificancePseudothreshold
        self._significanceForClass[name] = significanceForClass

        return None 

    def _create_testclass_tuple(self, entity, sequence, outcome):
        """
        Todo
        """
        testclasstup = []
        if entity == 'avg':
            testclasstup.append('avg')
        else:
            testclasstup.append('per')
        if sequence == 'avg':
            testclasstup.append('avg')
        else:
            testclasstup.append('per')
        if outcome == 'avg':
            testclasstup.append('avg')
        else:
            testclasstup.append('per')

        return tuple(testclasstup)

    def _get_equivalent_testclass_tuple(self, tup):
        """
        todo.
        """
        newtup = []
        if self.number_of_entities == 1:    
            newtup.append('avg')
        else:
            newtup.append(tup[0])
        if self.number_of_sequences == 1:    
            newtup.append('avg')
        else:
            newtup.append(tup[1])
        if self.number_of_sequences == 2:    
            newtup.append('avg')
        else:
            newtup.append(tup[2])

        return tuple(newtup)

    def _create_dict_tup(self, entity, sequence, outcome, pad=True):
        """
        Todo
        """
        dicttup = []
        if entity == 'avg' or self.number_of_entities == 1:
            if pad: dicttup.append('avg')
        else:
            dicttup.append(entity)

        if sequence == 'avg' or self.number_of_sequences == 1:
            if pad: dicttup.append('avg')
        else:
            if isinstance(sequence,int):
                dicttup.append(sequence)
            else:
                dicttup.append(self.indexforCircuit[sequence])

        if outcome == 'avg':
            if pad: dicttup.append('avg')
        else:
            if isinstance(outcome,int):
                dicttup.append(outcome)       
            else:
                dicttup.append(self.outcomes.index(outcome))

        return tuple(dicttup)

    def get_drift_frequency_indices(self, entity='avg', sequence='avg', outcome='avg', sort=True, detectorkey=None):
        """
        Todo
        """
        if detectorkey is None:
            detectorkey = self.defaultdetectorkey
        testclasstup = self._create_testclass_tuple(entity, sequence, outcome)

        if testclasstup not in self._testClasses[detectorkey]:
            
            equivtestclasstup = self._get_equivalent_testclass_tuple(testclasstup)
            assert(equivtestclasstup in self._testClasses[detectorkey]), "Drift dectection on this level was not performed! So drift indices cannot be returned."
            testclasstup = equivtestclasstup

        # Todo : here it should for equivalent test pointers.
        assert(testclasstup in self._testClasses[detectorkey]), "Drift dectection on this level was not performed! So drift indices cannot be returned."

        dicttup = self._create_dict_tup(entity, sequence, outcome, pad=True)
        driftfreqInds = self._sigFreqIndsinClass[detectorkey].get(dicttup,[])
        if sort:
            driftfreqInds.sort()

        return _copy.deepcopy(driftfreqInds)

    def get_drift_frequencies(self, entity='avg', sequence='avg', outcome='avg', detectorkey=None):
        """
        Todo
        """
        freqInd = self.get_drift_frequency_indices(entity=entity, sequence=sequence, outcome= outcome, sort=False, detectorkey=detectorkey)

        return _copy.deepcopy(self.frequenciesInHz[freqInd])

    def get_power_significance_threshold(self, entity='avg', sequence='avg', outcome='avg', detectorkey=None):
        """
        todo
        """
        if detectorkey is None:
            detectorkey = self.defaultdetectorkey
        testtup = self._create_testclass_tuple(entity,sequence,outcome)
        assert(testtup in self._testClasses[detectorkey]), "Can only get a significance threshold if this test class was implemented! To create an ad-hoc post-fact threshold use the functions in drift.statistics"

        thresholdset = self._powerSignificancePseudothreshold[detectorkey][testtup]
        # If it's a float, it's a "true" threshold, so we set the threshold to this.
        if isinstance(thresholdset,float):
            threshold = thresholdset 
        # If it's a dict it's either a single pseudo-threshold or a set of pseudo-threholds.
        else:
            thresholdset = list(thresholdset.values())
            # We return the largest pseudo-threshold, as this is a threshold for all cases.
            threshold = max(thresholdset)

        return threshold

    def get_power_pvalue_significance_threshold(self, entity='avg', sequence='avg', outcome='avg', detectorkey=None):
        """
        Todo
        """
        classtup = self._create_testclass_tuple(entity, sequence, outcome)
        power_threshold = self.get_power_significance_threshold(entity, sequence, outcome, detectorkey=detectorkey)
        dof = self.get_dof_per_spectrum_in_class(*classtup)
        pvalue_threshold = _stats.power_to_pvalue(power_threshold,dof)

        return pvalue_threshold

    def add_reconstruction(self, entity, sequence, model, modelSelector, estimator, auxDict={}, overwrite=False,
                           settodefault=True):
        """
        todo
        """
        if not hasattr(self,"models"):
            self.models = {}
            self.estimationAuxDict = {}

        eInd = self.entitieslist.index(entity)
        sInd = self.indexforCircuit[sequence]

        if (eInd, sInd) not in self.models.keys():
            self.models[eInd,sInd] = {}

        if (modelSelector, estimator) in self.models[eInd,sInd].keys():
            assert(overwrite), "Cannot add this model, as overwrite is False and a model with this key already exists!"

        self.models[eInd,sInd][modelSelector, estimator] =  _copy.deepcopy(model)

        if not hasattr(self,"defaultmodelkey"):
            self.defaultmodelkey = {}
        if settodefault:
            self.defaultmodelkey[eInd,sInd] = (modelSelector, estimator)
        else:
            # If there isn't yet a default, we set it to this.
            if (eInd,sInd) not in self.defaultmodelkey.keys():
                self.defaultmodelkey[eInd,sInd] = (modelSelector, estimator)

        return None


    # Todo : write this function
    def get_probability_trajectory(self,  entity, sequence, modelkey=None, times='sequence'):
        """
        This function hasn't been written yet!.
        """
        return p

         # Todo : currently the AuxDict is not stored.
        
    # def is_drift_detected(self):
        
    #     assert(hasattr(self,"drift_detected")), "Drift detection results have not yet been generated!"

    #     if self.drift_detected:
    #         print("Statistical tests set at a global significance level of: " + str(self.significance)) 
    #         print("Result: The 'no drift' hypothesis *is* rejected.")
    #     else:
    #         print("Statistical tests set at a global significance level of: " + str(self.significance))
    #         print("Result: The 'no drift' hypothesis is *not* rejected.")

  
    def plot_spectrum(self, entity='avg', sequence='avg', outcome='avg',
                      figsize=(15,3),  xlim=(None,None), ylim = (None,None), savepath=None, 
                      loc=None, addtitle=True, detectorkey=None):
        """
        Todo:       
        threshold : 'none', '1test', 'class', 'all', 'default'
        """
        # sequence_index = sequence
        
        # if self.sequences_to_indices is not None:
        #     if type(sequence) != int:
        #         if sequence in list(self.sequences_to_indices.keys()):
        #             sequence_index = self.sequences_to_indices[sequence]
            
        # if outcome != 'averaged':
        #     assert(self.outcomes is not None)
        #     assert(outcome in self.outcomes)
        #     outcome_index = self.outcomes.index(outcome)
        #     outcome_label = str(outcome)
        
        try:
            import matplotlib.pyplot as _plt
            import seaborn as _seaborn
        except ImportError:
            raise ValueError("plot_power_spectrum(...) requires you to install matplotlib and seaborn")
        
        _seaborn.set()
        _seaborn.set_style('white')
        _plt.figure(figsize=figsize)

        try:
            if detectorkey is None:
                detectorkey = self.defaultdetectorkey
        except:
            pass

        # if self.name is not None:
        #     name_in_title1 = ' and dataset '+self.name
        #     name_in_title2 = ' for dataset '+self.name
        # else:
        #     name_in_title1 = ''
        #     name_in_title2 = ''
        
        # # If sequence is not averaged, prepare the sequence label for the plot title
        # if sequence_index != 'averaged':    
        #     if self.indices_to_sequences is not None:
        #         sequence_label = str(self.indices_to_sequences[sequence_index])
        #     else:
        #         sequence_label = str(sequence_index)

        # if self.number_of_entities > 1:
        #     assert(not (outcome != 'averaged' and entity == 'averaged')), "Not permitted to average over multiple entities but not outcomes!"

       
        # # Here outcome value is ignored, as, if either S or E is averaged, must have outcome-averaged
        # if sequence_index == 'averaged' and (entity == 'averaged' or self.number_of_entities == 1):       
        #     spectrum = self.global_power_spectrum
        #     threshold1test = self.global_significance_threshold_1test
        #     thresholdclass = self.global_significance_threshold_classcompensation
        #     # Compensates for any noise-free spectra that have been averaged into the global spectrum.
        #     noiselevel = self.global_dof/(self.global_dof+self.global_dof_reduction)
        #     if threshold == 'default':
        #         threshold='1test'
        #     title = 'Global power spectrum' + name_in_title2
             
        # # Here outcome value is ignored, as, if either S or E is averaged, must have outcome-averaged   
        # elif sequence_index == 'averaged' and entity != 'averaged':       
        #     spectrum = self.pe_power_spectrum[entity,:]
        #     threshold1test = self.pe_significance_threshold_1test
        #     thresholdclass = self.pe_significance_threshold_classcompensation
        #     noiselevel = 1.
        #     if threshold == 'default':
        #         threshold='all'
        #     if self.number_of_sequences > 1:
        #         if self.number_of_outcomes > 2:
        #             title = 'Sequence and outcome averaged power spectrum for entity ' + str(entity) + name_in_title1
        #         else:
        #             title = 'Sequence-averaged power spectrum for entity ' + str(entity) + name_in_title1
        #     else:
        #         if self.number_of_outcomes > 2:
        #             title = 'Outcome-averaged power spectrum for entity ' + str(entity) + name_in_title1
        #         else:
        #             title = 'Power spectrum for entity ' + str(entity) + name_in_title1
                
        # Here outcome value is ignored, as, if either S or E is averaged, must have outcome-averaged   
        # elif sequence_index != 'averaged' and entity == 'averaged' and outcome == 'averaged':       
        #     spectrum = self.ps_power_spectrum[sequence_index,:]
        #     threshold1test = self.ps_significance_threshold_1test
        #     thresholdclass = self.ps_significance_threshold_classcompensation
        #     noiselevel = 1.
        #     if threshold == 'default':
        #         threshold='all'
                
        #     if self.number_of_entities> 1:
        #         if self.number_of_outcomes > 2:
        #             title = 'Entity and outcome averaged power spectrum for sequence ' + sequence_label + name_in_title1
        #         else:
        #             title = 'Entity-averaged power spectrum for sequence ' + sequence_label + name_in_title1
        #     else:
        #         if self.number_of_outcomes > 2:
        #             title = 'Outcome-averaged power spectrum for sequence ' + sequence_label + name_in_title1
        #         else:
        #             title = 'Power spectrum power spectrum for sequence ' + sequence_label + name_in_title1

        try:
            threshold = self.get_power_significance_threshold(entity, sequence, outcome, detectorkey=detectorkey)
            plotthreshold = True
        except:
            plotthreshold = False

        spectrum = self.get_spectrum(entity, sequence, outcome)
   
        # outcome value is ignored
        # elif sequence_index != 'averaged' and entity != 'averaged' and outcome == 'averaged':       
        #     spectrum = self.pspe_power_spectrum[sequence_index,entity,:]
        #     threshold1test = self.pspe_significance_threshold_1test
        #     thresholdclass = self.pspe_significance_threshold_classcompensation
        #     noiselevel = 1.
        #     if threshold == 'default':
        #         threshold='all'
                
        #     if self.number_of_outcomes > 2:
        #         title = 'Outcome-averaged power spectrum for sequence ' +sequence_label 
        #         title += ', entity ' + str(entity) + name_in_title1
        #     else:
        #         title = 'Power spectrum for sequence ' +sequence_label
        #         title += ', entity ' + str(entity) + name_in_title1
        
        # # outcome value is not ignored. Number of entities must be 1 (checked earlier)
        # elif sequence_index != 'averaged' and outcome != 'averaged': 
        #     if self.number_of_entities == 1:
        #         entity = 0     
        #     spectrum = self.pspepo_power_spectrum[sequence_index,entity,outcome_index,:]
        #     threshold1test = self.pspepo_significance_threshold_1test
        #     thresholdclass = self.pspepo_significance_threshold_classcompensation
        #     noiselevel = 1.
        #     if threshold == 'default':
        #         threshold='all'
                
        #     title = 'Power spectrum for sequence ' +sequence_label+ ', entity ' + str(entity) 
        #     title += ', outcome '+ outcome_label + name_in_title1
        
        # else:
        #     print("Invalid string or value for `sequence`, `entity` or `outcome`")
            
        #if self.timestep is not None:
        _plt.xlabel( "Frequence (Hertz)")
        _plt.ylabel("Power")

        #else:
        #    xlabel = "Frequence"        
        
        _plt.plot(self.frequenciesInHz[1:],spectrum[1:],'.-',label='Data spectrum')
        # Todo: Update so the noiselevel is noiselevel = self.global_dof/(self.global_dof+self.global_dof_reduction)
        noiselevel = 1
        _plt.axhline(noiselevel,color='c',label='Average shot-noise level')
        if plotthreshold:
            _plt.axhline(threshold,color='r',label='{} global stat. significance threshold'.format(self.significance[detectorkey]))

        
        # if threshold == '1test' or threshold == 'all':  
        #     _plt.plot(self.frequencies,threshold1test*_np.ones(self.number_of_timesteps),'k--', 
        #           label=str(self.significance)+' significance single-test significance threshold')
        
        # if threshold == 'class' or threshold == 'all':  
        #     _plt.plot(self.frequencies,thresholdclass*_np.ones(self.number_of_timesteps),'r--', 
        #           label=str(self.significance)+' significance multi-test significance threshold')
        
        # if ylim is None:

        #     a = _np.max(self.pspe_power_spectrum)
        #     b = _np.max(self.pe_power_spectrum)
        #     c = _np.max(self.global_power_spectrum)
        #     max_power = _np.max(_np.array([a,b,c]))
        #     a = self.pspe_significance_threshold
        #     b = self.pe_significance_threshold
        #     c = self.global_significance_threshold
        #     max_threshold = _np.max(_np.array([a,b,c]))
            
        #     if max_power > max_threshold:                
        #         ylim = [0,max_power]
                
        #     else:
        #         ylim = [0,max_threshold+1.]
                    
        # Legend
        if loc is not None: 
            _plt.legend(loc=loc)
        else: 
            _plt.legend()
        
        # limits
        #if xlim is (None,None):
        #    _plt.xlim(0,_np.max(self.frequenciesInHz))
        #else:
        _plt.xlim(xlim)
        #if ylim is (None,None):
        #    _plt.ylim((0,None))
        #else:
        _plt.ylim(ylim)

        # if addtitle:
        #     _plt.title(title,fontsize=17)
        # _plt.xlabel(xlabel,fontsize=15)
        # _plt.ylabel("Power",fontsize=15)
        # _plt.xlim(xlim)
        
        _plt.tight_layout()
        if savepath is not None:
            _plt.savefig(savepath)
        else:
            _plt.show()
            
    # def plot_most_drifty_probability(self, errorbars=True, plot_data=False, parray=None, figsize=(15,3), 
    #                                  savepath=None, loc=None, title=True):
        
    #     if self.multitest_compensation == 'none':
    #         ws = "Warning: multi-tests compensation is 'none'. This means that if there are many sequences it is likely"
    #         ws += " that some of them will have non-trivial estimates for the time-dependent probability!"
    #         print(ws)
        
    #     # Find the (sequence,entity,outcome) index with the most power in the reconstruction. This is
    #     # not necessarily the index with the largest max power in the data spectrum.
    #     most_drift_index = _np.unravel_index(_np.argmax(self.pspepo_reconstruction_powerpertimestep), 
    #                                          _np.shape(self.pspepo_reconstruction_powerpertimestep))
        
    #     self.plot_estimated_probability(int(most_drift_index[0]), int(most_drift_index[1]), int(most_drift_index[2]),
    #                                     errorbars=errorbars,
    #                                     plot_data=plot_data, target_value=None,parray=parray, figsize=figsize, 
    #                                     savepath=savepath, loc=loc, title=title)
      
    
    # Todo:
    #def add_target_probabilities(targetModel):
    #
    #    return  None
   
    def plot_probability_trajectory_estimates(self, circuitlist, entity='0', outcome=('0',), uncertainties=False,
                                              plotData=False, targetValue=None, figsize=(15,3), 
                                              savepath=None, loc=None, title=True, 
                                              estimatekey=None):
        
        # sequence_index = sequence
        
        # if self.sequences_to_indices is not None:
        #     if type(sequence) != int:
        #         if sequence in list(self.sequences_to_indices.keys()):
        #             sequence_index = self.sequences_to_indices[sequence]
                        
        # if self.outcomes is not None:
        #     if outcome in self.outcomes:
        #         outcome_index = self.outcomes.index(outcome)
        #     else:
        #         outcome_index = outcome
        # else:
        #     outcome_index = outcome
        
        try:
            import matplotlib.pyplot as _plt
        except ImportError:
            raise ValueError("This method requires you to install matplotlib")

        _plt.figure(figsize=figsize)
                
        times = []
        for opstr in circuitlist:
            gstrInd = self.indexforCircuit[opstr]
            times += list(self.timestamps[gstrInd])
        
        times.sort()       

        # else:
        #     times = _np.arange(0,self.number_of_timesteps)
        #     xlabel = 'Time (timesteps)'
        
        # if self.indices_to_sequences is not None:
        #     sequence_label = str(self.indices_to_sequences[sequence_index])
        # else:
        #     sequence_label = str(sequence_index)

        # if self.outcomes is not None:
        #     outcome_label = str(self.outcomes[outcome_index])
        # else:
        #     outcome_label = str(outcome_index)
        
        # if plotData:
        #     label = 'Data'
        #     _plt.plot(times,self.timeseries[sequence][entity][outcome_index]/self.number_of_counts,'.',label=label)
             
        entityInd = self.entitieslist.index(entity)
        outcomeInd = self.outcomes.index(outcome)

        for opstr in circuitlist:
            gstrInd = self.indexforCircuit[opstr]
            if estimatekey is None:
                mdl_estimatekey = self.defaultmodelkey[entityInd,gstrInd]
            else:
                 mdl_estimatekey = estimatekey
            p = self.models[entityInd,gstrInd][mdl_estimatekey].get_probabilities(times)[outcome]
        # error = self.pspepo_reconstruction_uncertainty[sequence_index,entity,outcome_index]
        # upper = p+error
        # lower = p-error
        # upper[upper > 1.] = 1.
        # lower[lower < 0.] = 0.
        
            _plt.plot(times,p,'-',label='{}'.format(opstr))
        
        # if errorbars:
        #     _plt.fill_between(times, upper, lower, alpha=0.2, color='r')
        
        # if target_value is not None:
        #     _plt.plot(times,target_value*_np.ones(self.number_of_timesteps),'k--',label='Ideal outcome probability')
        
        if loc is not None:
            _plt.legend(loc=loc)
        else:
            _plt.legend()
            
        #_plt.xlim(0,_np.max(times))
        _plt.ylim(-0.05,1.05)
        
        # if title:
        #     if self.number_of_entities > 1:
        #         title = "Estimated probability for sequence " + sequence_label + ", entity "
        #         title += str(entity) + " and outcome " + outcome_label
        #     else:
        #         title = "Estimated probability for sequence " + sequence_label + " and outcome " + outcome_label
         
        _plt.title("Estimated probability trajectories",fontsize=17)
        _plt.xlabel('Time (seconds)',fontsize=15)
        _plt.ylabel("Probability",fontsize=15)
        
        _plt.tight_layout()
        if savepath is not None:
            _plt.savefig(savepath)
        else:
            _plt.show()

            
    # def plot_multi_estimated_probabilities(self, sequence_list, entity=0, outcome=0, errorbars=True,
    #                                target_value=None, figsize=(15,3), savepath=None, 
    #                                loc=None, usr_labels=None, usr_title=None, xlim=None):
        
    #     sequence_index_list = sequence_list
        
    #     # Override this if ....
    #     if self.sequences_to_indices is not None:
    #         if type(sequence_list[0]) != int: 
    #             if sequence_list[0] in list(self.sequences_to_indices.keys()):
    #                 sequence_index_list = []
    #                 for seq in sequence_list:    
    #                     sequence_index_list.append(self.sequences_to_indices[seq])
            
    #     if self.outcomes is not None:
    #         if outcome in self.outcomes:
    #             outcome_index = self.outcomes.index(outcome)
    #         else:
    #             outcome_index = outcome
    #     else:
    #         outcome_index = outcome
        
    #     try:
    #         import matplotlib.pyplot as _plt
    #     except ImportError:
    #         raise ValueError("plot_power_spectrum(...) requires you to install matplotlib")

    #     _plt.figure(figsize=figsize)
        
    #     if self.timestep is not None:
    #         times = self.timestep*_np.arange(0,self.number_of_timesteps)
    #         xlabel = 'Time (seconds)'
    #     else:
    #         times = _np.arange(0,self.number_of_timesteps)
    #         xlabel = 'Time (timesteps)'
        
    #     sequence_label = {}
    #     for sequence_index in sequence_index_list:
    #         if self.indices_to_sequences is not None:
    #             sequence_label[sequence_index] = str(self.indices_to_sequences[sequence_index])
    #         else:
    #             sequence_label[sequence_index] = str(sequence_index)

    #     if self.outcomes is not None:
    #         outcome_label = str(self.outcomes[outcome_index])
    #     else:
    #         outcome_label = str(outcome_index)
        
    #     num_curves = len(sequence_index_list)
    #     c = _np.linspace(0,1,num_curves)
    #     i = 0
        
    #     for i in range(0,num_curves):
    #         sequence_index = sequence_index_list[i]
    #         p = self.pspepo_reconstruction[sequence_index,entity,outcome_index,:]
    #         error = self.pspepo_reconstruction_uncertainty[sequence_index,entity,outcome_index]
    #         upper = p+error
    #         lower = p-error
    #         upper[upper > 1.] = 1.
    #         lower[lower < 0.] = 0.
            
    #         if errorbars:
    #             _plt.fill_between(times, upper, lower, alpha=0.2, color=_plt.cm.RdYlBu(c[i]))
                
    #         label = 'Estimated $p(t)$ for sequence '+sequence_label[sequence_index]
    #         if usr_labels is not None:
    #             label = usr_labels[i]
                
    #         _plt.plot(times,p,'-',lw=2,label=label, color=_plt.cm.RdYlBu(c[i]))
            
        
    #     if target_value is not None:
    #         _plt.plot(times,target_value*_np.ones(self.number_of_timesteps),'k--',lw=2,label='Target value')
        
    #     if loc is not None:
    #         _plt.legend(loc=loc)
    #     else:
    #         _plt.legend()
        
    #     if xlim == None:
    #         _plt.xlim(0,_np.max(times))
    #     else:
    #         _plt.xlim(xlim)
    #     _plt.ylim(0,1)
        
    #     if self.number_of_entities > 1:
    #         title = "Estimated probability for entity "
    #         title += str(entity) + " and outcome " + outcome_label

    #     else:
    #         title = "Estimated probability outcome " + outcome_label
            
    #     if usr_title is not None:
    #         title = usr_title
            
    #     _plt.title(title,fontsize=17)
    #     _plt.xlabel(xlabel,fontsize=15)
    #     _plt.ylabel("Probability",fontsize=15)
        
    #     if savepath is not None:
    #         _plt.savefig(savepath)
    #     else:
    #         _plt.show()
