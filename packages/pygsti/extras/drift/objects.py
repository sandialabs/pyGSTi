from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Functions for Fourier analysis of equally spaced time-series data"""

import numpy as _np

class BasicDriftResults(object):

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

        self.confidence = None
        
        #----------------------------#
        # --- Derived quantities --- #
        #----------------------------#
        
        self.frequencies = None
        
        # todo:....
        self.drift_detected = None
        
        # per-sequence, per-entity, per-outcome results
        self.pspepo_modes = None
        self.pspepo_power_spectrum = None
        self.pspepo_drift_frequencies = None
        self.pspepo_max_power = None
        self.pspepo_pvalue = None
        self.pspepo_significance_threshold = None
        self.pspepo_drift_detected = None
        self.pspepo_reconstruction = None
        self.pspepo_reconstruction_power_spectrum = None
        self.pspepo_reconstruction_powerpertimestep = None
        
        # per-sequence, per-entity, outcome-averaged results
        self.pspe_power_spectrum = None
        self.pspe_max_power = None
        self.pspe_pvalue = None
        self.pspe_significance_threshold = None
        self.pspe_drift_detected = None
        self.pspe_drift_frequencies = None
        self.pspe_reconstruction_power_spectrum = None
        self.pspe_reconstruction_powerpertimestep = None
        
        # sequence-averaged, per-entity, outcome-averaged results 
        self.pe_power_spectrum = None
        self.pe_max_power = None
        self.pe_pvalue = None
        self.pe_significance_threshold = None      
        self.pe_drift_detected = None
        self.pe_drift_frequencies = None
        self.pe_reconstruction_power_spectrum = None 
        self.pe_reconstruction_powerpertimestep = None
        
        # per-sequence, entity-averaged, outcome-averaged results 
        self.ps_power_spectrum = None
        self.ps_max_power = None
        self.ps_pvalue = None
        self.ps_significance_threshold = None      
        self.ps_drift_detected = None
        self.ps_drift_frequencies = None
        self.ps_reconstruction_power_spectrum = None 
        self.ps_reconstruction_powerpertimestep = None
        
        # sequence-averaged, sequence-entity, outcome-averaged results 
        self.global_power_spectrum = None
        self.global_max_power = None
        self.global_pvalue = None
        self.global_significance_threshold = None       
        self.global_drift_detected = None
        self.global_drift_frequencies = None
        self.global_reconstruction_power_spectrum = None
        self.global_reconstruction_powerpertimestep = None
        
    def plot_power_spectrum(self, sequence='averaged', entity='averaged', outcome='averaged', 
                            threshold=True, figsize=(15,3), fix_ymax = False,
                            savepath=None):
        
        # Todo: deal with outcome variable
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
            
        if sequence == 'averaged' and entity == 'averaged':       
            spectrum = self.global_power_spectrum
            threshold = self.global_significance_threshold
            title = 'Global power spectrum' + name_in_title2
            
        elif sequence == 'averaged' and entity != 'averaged':       
            spectrum = self.pe_power_spectrum[entity,:]
            threshold = self.pe_significance_threshold
            title = 'Averaged power spectrum for entity ' + str(entity) + name_in_title1
            
        elif sequence != 'averaged' and entity != 'averaged':       
            spectrum = self.pspe_power_spectrum[sequence,entity,:]
            threshold = self.pspe_significance_threshold
            
            if self.indices_to_sequences is not None:
                sequence_label = str(self.indices_to_sequences[sequence])
            else:
                sequence_label = str(sequence)
            title = 'Outcome-averaged power spectrum for sequence ' +sequence_label+ ', entity ' + str(entity) + name_in_title1
            
        else:
            sequence_label = str(sequence)
        if self.timestep is not None:
            xlabel = "Frequence (Hertz)"
        else:
            xlabel = "Frequence"
        
        _plt.plot(self.frequencies,spectrum,'b.-',label='Data power spectrum')
        _plt.plot(self.frequencies,_np.ones(self.number_of_timesteps),'k--',label='Mean noise level')
        _plt.plot(self.frequencies,threshold*_np.ones(self.number_of_timesteps),'--', 
                  label=str(self.confidence)+' Significance threshold')
        
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
            
        _plt.legend()
        _plt.xlim(0,_np.max(self.frequencies))
        _plt.title(title,fontsize=17)
        _plt.xlabel(xlabel,fontsize=15)
        _plt.ylabel("Power",fontsize=15)
        
        if savepath is not None:
            _plt.savefig(savepath)
        else:
            _plt.show()
   
    def plot_estimated_probability(self, sequence, entity=0, outcome=0, plot_data=True, pt=None, figsize=(15,3),
                                  savepath=None):
        
        # Todo: various changes needed to this function
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
            sequence_label = str(self.indices_to_sequences[sequence])
        else:
            sequence_label = str(sequence)
        
        if plot_data:
            _plt.plot(times,self.data[sequence,entity,outcome,:]/self.number_of_counts,'b.',label='Data power spectrum')
        
        _plt.plot(times,self.pspepo_reconstruction[sequence,entity,outcome,:],'r-',label='Estimated p(t)')
        
        if pt is not None:
            _plt.plot(times,pt,'c--',label='True p(t)')
                
        _plt.legend()
        _plt.xlim(0,_np.max(times))
        _plt.ylim(0,1)
        
        title = "Estimated drifting probability for sequence " + sequence_label
        _plt.title(title,fontsize=17)
        _plt.xlabel(xlabel,fontsize=15)
        _plt.ylabel("Probability",fontsize=15)
        
        if savepath is not None:
            _plt.savefig(savepath)
        else:
            _plt.show()
