from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Functions for Fourier analysis of equally spaced time-series data"""

import numpy as _np

class DriftResults(object):

    def __init__(self):
        
        #--------------------------#
        # --- Input quantities --- #
        #--------------------------#
        
        self.name = None
        self.data = None
        self.indices_to_sequences = None
        
        self.number_of_sequences = None
        self.number_of_timesteps = None
        self.number_of_entities = None
        self.number_of_counts = None        
        self.timestep = None
        
        self.outcomes = None
        self.timestamps = None

        self.confidence = None
        self.multitest_compensation = None
        
        #----------------------------#
        # --- Derived quantities --- #
        #----------------------------#
        
        self.frequencies = None
        self.sequences_to_indices = None
        
        # todo:....
        self.drift_detected = None
        
        # per-sequence, per-entity, per-outcome results
        self.pspepo_modes = None
        self.pspepo_power_spectrum = None
        self.pspepo_drift_frequencies = None
        self.pspepo_max_power = None
        self.pspepo_pvalue = None
        self.pspepo_confidence_classcompensation = None
        self.pspepo_significance_threshold = None
        self.pspepo_significance_threshold_1test = None
        self.pspepo_significance_threshold_classcompensation = None
        self.pspepo_drift_detected = None
        self.pspepo_reconstruction = None
        self.pspepo_reconstruction_uncertainty = None
        self.pspepo_reconstruction_power_spectrum = None
        self.pspepo_reconstruction_powerpertimestep = None
        
        # per-sequence, per-entity, outcome-averaged results
        self.pspe_power_spectrum = None
        self.pspe_max_power = None
        self.pspe_pvalue = None
        self.pspe_confidence_classcompensation = None
        self.pspe_significance_threshold = None
        self.pspe_significance_threshold_1test = None
        self.pspe_significance_threshold_classcompensation = None
        self.pspe_drift_detected = None
        self.pspe_drift_frequencies = None
        self.pspe_reconstruction_power_spectrum = None
        self.pspe_reconstruction_powerpertimestep = None
        
        # sequence-averaged, per-entity, outcome-averaged results 
        self.pe_power_spectrum = None
        self.pe_max_power = None
        self.pe_pvalue = None
        self.pe_confidence_classcompensation = None
        self.pe_significance_threshold = None
        self.pe_significance_threshold_1test = None
        self.pe_significance_threshold_classcompensation = None
        self.pe_drift_detected = None
        self.pe_drift_frequencies = None
        self.pe_reconstruction_power_spectrum = None 
        self.pe_reconstruction_powerpertimestep = None
        
        # per-sequence, entity-averaged, outcome-averaged results 
        self.ps_power_spectrum = None
        self.ps_max_power = None
        self.ps_pvalue = None
        self.ps_confidence_classcompensation = None
        self.ps_significance_threshold = None 
        self.ps_significance_threshold_1test = None
        self.ps_significance_threshold_classcompensation = None
        self.ps_drift_detected = None
        self.ps_drift_frequencies = None
        self.ps_reconstruction_power_spectrum = None 
        self.ps_reconstruction_powerpertimestep = None
        
        # sequence-averaged, sequence-entity, outcome-averaged results 
        self.global_power_spectrum = None
        self.global_max_power = None
        self.global_pvalue = None
        self.global_confidence_classcompensation = None
        self.global_significance_threshold = None  
        self.global_significance_threshold_1test = None
        self.global_significance_threshold_classcompensation = None
        self.global_drift_detected = None
        self.global_drift_frequencies = None
        self.global_reconstruction_power_spectrum = None
        self.global_reconstruction_powerpertimestep = None
    
    def any_drift_detect(self):
        
        if self.drift_detected:
            print("Statistical tests set at a global confidence level of: " + str(self.confidence)) 
            print("Result: The 'no drift' hypothesis *is* rejected.")
        else:
            print("Statistical tests set at a global confidence level of: " + str(self.confidence))
            print("Result: The 'no drift' hypothesis is *not* rejected.")
    
    
    def plot_power_spectrum(self, sequence='averaged', entity='averaged', 
                            outcome='averaged', threshold='default', figsize=(15,3), 
                            fix_ymax = False, savepath=None, loc=None, addtitle=True):
        """
        
        threshold : 'none', '1test', 'class', 'all', 'default'
        """
        sequence_index = sequence
        
        if self.sequences_to_indices is not None:
            if type(sequence) != int:
                if sequence in list(self.sequences_to_indices.keys()):
                    sequence_index = self.sequences_to_indices[sequence]
            
        if self.outcomes is not None:
            if outcome in self.outcomes:
                outcome_index = self.outcomes.index(outcome)
            else:
                outcome_index = outcome
        else:
            outcome_index = outcome
        
        try:
            import matplotlib.pyplot as _plt
        except ImportError:
            raise ValueError("plot_power_spectrum(...) requires you to install matplotlib")
        _plt.figure(figsize=figsize)
        
        if self.name is not None:
            name_in_title1 = ' and dataset '+self.name
            name_in_title2 = ' for dataset '+self.name
        else:
            name_in_title1 = ''
            name_in_title2 = ''
        
        # If sequence is not averaged, prepare the sequence label for the plot title
        if sequence_index != 'averaged':    
            if self.indices_to_sequences is not None:
                sequence_label = str(self.indices_to_sequences[sequence_index])
            else:
                sequence_label = str(sequence_index)
        
        # If outcome is not averaged, prepare the outcome label for the plot title       
        if outcome_index != 'averaged':    
            if self.outcomes is not None:
                outcome_label = str(self.outcomes[outcome_index])
            else:
                outcome_label = str(outcome_index)
       
        # Here outcome value is ignored, as, if either S or E is averaged, must have outcome-averaged
        if sequence_index == 'averaged' and entity == 'averaged':       
            spectrum = self.global_power_spectrum
            threshold1test = self.global_significance_threshold_1test
            thresholdclass = self.global_significance_threshold_classcompensation
            # Compensates for any noise-free spectra that have been averaged into the global spectrum.
            noiselevel = self.global_dof/(self.global_dof+self.global_dof_reduction)
            if threshold == 'default':
                threshold='1test'
            title = 'Global power spectrum' + name_in_title2
            
        
        # Here outcome value is ignored, as, if either S or E is averaged, must have outcome-averaged   
        elif sequence_index == 'averaged' and entity != 'averaged':       
            spectrum = self.pe_power_spectrum[entity,:]
            threshold1test = self.pe_significance_threshold_1test
            thresholdclass = self.pe_significance_threshold_classcompensation
            noiselevel = 1.
            if threshold == 'default':
                threshold='all'
            if self.number_of_sequences > 1:
                if self.number_of_outcomes > 2:
                    title = 'Sequence and outcome averaged power spectrum for entity ' + str(entity) + name_in_title1
                else:
                    title = 'Sequence-averaged power spectrum for entity ' + str(entity) + name_in_title1
            else:
                if self.number_of_outcomes > 2:
                    title = 'Outcome-averaged power spectrum for entity ' + str(entity) + name_in_title1
                else:
                    title = 'Power spectrum for entity ' + str(entity) + name_in_title1
                
        # Here outcome value is ignored, as, if either S or E is averaged, must have outcome-averaged   
        elif sequence_index != 'averaged' and entity == 'averaged':       
            spectrum = self.ps_power_spectrum[sequence_index,:]
            threshold1test = self.ps_significance_threshold_1test
            thresholdclass = self.ps_significance_threshold_classcompensation
            noiselevel = 1.
            if threshold == 'default':
                threshold='all'
                
            if self.number_of_entities> 1:
                if self.number_of_outcomes > 2:
                    title = 'Entity and outcome averaged power spectrum for sequence ' + sequence_label + name_in_title1
                else:
                    title = 'Entity-averaged power spectrum for sequence ' + sequence_label + name_in_title1
            else:
                if self.number_of_outcomes > 2:
                    title = 'Outcome-averaged power spectrum for sequence ' + sequence_label + name_in_title1
                else:
                    title = 'Power spectrum power spectrum for sequence ' + sequence_label + name_in_title1

        
        # outcome value is not ignored
        elif sequence_index != 'averaged' and entity != 'averaged' and outcome_index == 'averaged':       
            spectrum = self.pspe_power_spectrum[sequence_index,entity,:]
            threshold1test = self.pspe_significance_threshold_1test
            thresholdclass = self.pspe_significance_threshold_classcompensation
            noiselevel = 1.
            if threshold == 'default':
                threshold='all'
                
            if self.number_of_outcomes > 2:
                title = 'Outcome-averaged power spectrum for sequence ' +sequence_label 
                title += ', entity ' + str(entity) + name_in_title1
            else:
                title = 'Power spectrum for sequence ' +sequence_label
                title += ', entity ' + str(entity) + name_in_title1
        
        # outcome value is not ignored    
        elif sequence_index != 'averaged' and entity != 'averaged' and outcome_index != 'averaged':       
            spectrum = self.pspepo_power_spectrum[sequence_index,entity,outcome_index,:]
            threshold1test = self.pspepo_significance_threshold_1test
            thresholdclass = self.pspepo_significance_threshold_classcompensation
            noiselevel = 1.
            if threshold == 'default':
                threshold='all'
                
            title = 'Power spectrum for sequence ' +sequence_label+ ', entity ' + str(entity) 
            title += ', outcome '+ outcome_label + name_in_title1
        
        else:
            print("Invalid string or value for `sequence`, `entity` or `outcome`")
            
        if self.timestep is not None:
            xlabel = "Frequence (Hertz)"
        else:
            xlabel = "Frequence"
            
         
        
        _plt.plot(self.frequencies,spectrum,'b.-',label='Data power spectrum')
        _plt.plot(self.frequencies,noiselevel*_np.ones(self.number_of_timesteps),'c--',label='Mean noise level')
        
        if threshold == '1test' or threshold == 'all':  
            _plt.plot(self.frequencies,threshold1test*_np.ones(self.number_of_timesteps),'k--', 
                  label=str(self.confidence)+' confidence single-test significance threshold')
        
        if threshold == 'class' or threshold == 'all':  
            _plt.plot(self.frequencies,thresholdclass*_np.ones(self.number_of_timesteps),'r--', 
                  label=str(self.confidence)+' confidence multi-test significance threshold')
        
        if fix_ymax:
            a = _np.max(self.pspe_power_spectrum)
            b = _np.max(self.pe_power_spectrum)
            c = _np.max(self.global_power_spectrum)
            max_power = _np.max(_np.array([a,b,c]))
            a = self.pspe_significance_threshold
            b = self.pe_significance_threshold
            c = self.global_significance_threshold
            max_threshold = _np.max(_np.array([a,b,c]))
            
            if max_power > max_threshold:                
                ylim = [0,max_power]
                
            else:
                ylim = [0,max_threshold+1.]
        
            _plt.ylim(ylim)
            
        if loc is not None:
            _plt.legend(loc=loc)
        else:
            _plt.legend()
        _plt.xlim(0,_np.max(self.frequencies))
        if addtitle:
            _plt.title(title,fontsize=17)
        _plt.xlabel(xlabel,fontsize=15)
        _plt.ylabel("Power",fontsize=15)
        
        if savepath is not None:
            _plt.savefig(savepath)
        else:
            _plt.show()
            
    def plot_most_drifty_probability(self, errorbars=True, plot_data=False, parray=None, figsize=(15,3), 
                                     savepath=None, loc=None, title=True):
        
        if self.multitest_compensation == 'none':
            ws = "Warning: multi-tests compensation is 'none'. This means that if there are many sequences it is likely"
            ws += " that some of them will have non-trivial estimates for the time-dependent probability!"
            print(ws)
        
        # Find the (sequence,entity,outcome) index with the most power in the reconstruction. This is
        # not necessarily the index with the largest max power in the data spectrum.
        most_drift_index = _np.unravel_index(_np.argmax(self.pspepo_reconstruction_powerpertimestep), 
                                             _np.shape(self.pspepo_reconstruction_powerpertimestep))
        
        self.plot_estimated_probability(int(most_drift_index[0]), int(most_drift_index[1]), int(most_drift_index[2]),
                                        errorbars=errorbars,
                                        plot_data=plot_data, target_value=None,parray=parray, figsize=figsize, 
                                        savepath=savepath, loc=loc, title=title)
        
   
    def plot_estimated_probability(self, sequence, entity=0, outcome=0, errorbars=True,
                                   plot_data=False, target_value=None, parray=None, figsize=(15,3), savepath=None, 
                                   loc=None, title=True):
        
        sequence_index = sequence
        
        if self.sequences_to_indices is not None:
            if type(sequence) != int:
                if sequence in list(self.sequences_to_indices.keys()):
                    sequence_index = self.sequences_to_indices[sequence]
                        
        if self.outcomes is not None:
            if outcome in self.outcomes:
                outcome_index = self.outcomes.index(outcome)
            else:
                outcome_index = outcome
        else:
            outcome_index = outcome
      
        
        try:
            import matplotlib.pyplot as _plt
        except ImportError:
            raise ValueError("plot_power_spectrum(...) requires you to install matplotlib")

        _plt.figure(figsize=figsize)
        
        if self.timestep is not None:
            times = self.timestep*_np.arange(0,self.number_of_timesteps)
            xlabel = 'Time (seconds)'
        else:
            times = _np.arange(0,self.number_of_timesteps)
            xlabel = 'Time (timesteps)'
        
        if self.indices_to_sequences is not None:
            sequence_label = str(self.indices_to_sequences[sequence_index])
        else:
            sequence_label = str(sequence_index)

        if self.outcomes is not None:
            outcome_label = str(self.outcomes[outcome_index])
        else:
            outcome_label = str(outcome_index)
        
        if plot_data:
            label = 'Data'
            _plt.plot(times,self.data[sequence_index,entity,outcome_index,:]/self.number_of_counts,'.',label=label)
        
        p = self.pspepo_reconstruction[sequence_index,entity,outcome_index,:]
        error = self.pspepo_reconstruction_uncertainty[sequence_index,entity,outcome_index]
        upper = p+error
        lower = p-error
        upper[upper > 1.] = 1.
        lower[lower < 0.] = 0.
            
        _plt.plot(times,p,'r-',label='Estimated $p(t)$')
        
        if errorbars:
            _plt.fill_between(times, upper, lower, alpha=0.2, color='r')
        
        if parray is not None:
            _plt.plot(times,parray[sequence_index,entity,outcome_index,:],'c--',label='True $p(t)$')
        
        if target_value is not None:
            _plt.plot(times,target_value*_np.ones(self.number_of_timesteps),'k--',label='Target value')
        
        if loc is not None:
            _plt.legend(loc=loc)
        else:
            _plt.legend()
            
        _plt.xlim(0,_np.max(times))
        _plt.ylim(0,1)
        
        if title:
            if self.number_of_entities > 1:
                title = "Estimated probability for sequence " + sequence_label + ", entity "
                title += str(entity) + " and outcome " + outcome_label
            else:
                title = "Estimated probability for sequence " + sequence_label + " and outcome " + outcome_label
         
        _plt.title(title,fontsize=17)
        _plt.xlabel(xlabel,fontsize=15)
        _plt.ylabel("Probability",fontsize=15)
        
        if savepath is not None:
            _plt.savefig(savepath)
        else:
            _plt.show()

            
    def plot_multi_estimated_probabilities(self, sequence_list, entity=0, outcome=0, errorbars=True,
                                   target_value=None, figsize=(15,3), savepath=None, 
                                   loc=None, usr_labels=None, usr_title=None, xlim=None):
        
        sequence_index_list = sequence_list
        
        # Override this if ....
        if self.sequences_to_indices is not None:
            if type(sequence_list[0]) != int: 
                if sequence_list[0] in list(self.sequences_to_indices.keys()):
                    sequence_index_list = []
                    for seq in sequence_list:    
                        sequence_index_list.append(self.sequences_to_indices[seq])
            
        if self.outcomes is not None:
            if outcome in self.outcomes:
                outcome_index = self.outcomes.index(outcome)
            else:
                outcome_index = outcome
        else:
            outcome_index = outcome
        
        try:
            import matplotlib.pyplot as _plt
        except ImportError:
            raise ValueError("plot_power_spectrum(...) requires you to install matplotlib")

        _plt.figure(figsize=figsize)
        
        if self.timestep is not None:
            times = self.timestep*_np.arange(0,self.number_of_timesteps)
            xlabel = 'Time (seconds)'
        else:
            times = _np.arange(0,self.number_of_timesteps)
            xlabel = 'Time (timesteps)'
        
        sequence_label = {}
        for sequence_index in sequence_index_list:
            if self.indices_to_sequences is not None:
                sequence_label[sequence_index] = str(self.indices_to_sequences[sequence_index])
            else:
                sequence_label[sequence_index] = str(sequence_index)

        if self.outcomes is not None:
            outcome_label = str(self.outcomes[outcome_index])
        else:
            outcome_label = str(outcome_index)
        
        num_curves = len(sequence_index_list)
        c = _np.linspace(0,1,num_curves)
        i = 0
        
        for i in range(0,num_curves):
            sequence_index = sequence_index_list[i]
            p = self.pspepo_reconstruction[sequence_index,entity,outcome_index,:]
            error = self.pspepo_reconstruction_uncertainty[sequence_index,entity,outcome_index]
            upper = p+error
            lower = p-error
            upper[upper > 1.] = 1.
            lower[lower < 0.] = 0.
            
            if errorbars:
                _plt.fill_between(times, upper, lower, alpha=0.2, color=_plt.cm.RdYlBu(c[i]))
                
            label = 'Estimated $p(t)$ for sequence '+sequence_label[sequence_index]
            if usr_labels is not None:
                label = usr_labels[i]
                
            _plt.plot(times,p,'-',lw=2,label=label, color=_plt.cm.RdYlBu(c[i]))
            
        
        if target_value is not None:
            _plt.plot(times,target_value*_np.ones(self.number_of_timesteps),'k--',lw=2,label='Target value')
        
        if loc is not None:
            _plt.legend(loc=loc)
        else:
            _plt.legend()
        
        if xlim == None:
            _plt.xlim(0,_np.max(times))
        else:
            _plt.xlim(xlim)
        _plt.ylim(0,1)
        
        if self.number_of_entities > 1:
            title = "Estimated probability for entity "
            title += str(entity) + " and outcome " + outcome_label

        else:
            title = "Estimated probability outcome " + outcome_label
            
        if usr_title is not None:
            title = usr_title
            
        _plt.title(title,fontsize=17)
        _plt.xlabel(xlabel,fontsize=15)
        _plt.ylabel("Probability",fontsize=15)
        
        if savepath is not None:
            _plt.savefig(savepath)
        else:
            _plt.show()