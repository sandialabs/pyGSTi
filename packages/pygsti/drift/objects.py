from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Functions for Fourier analysis of equally spaced time-series data"""

import numpy as _np
import matplotlib.pyplot as _plt


class BasicDriftResults(object):

    def __init__(self):
        
        # Input quantities
        self.data = None
        self.number_of_sequences = None
        self.number_of_timesteps = None
        self.number_of_qubits = None
        self.number_of_counts = None        
        self.timestep = None
        
        self.confidence = None
        self.frequencies = None
        
        # 
        self.drift_detected = None
        
        # per-sequence, per-qubit results
        self.pspq_modes = None
        self.pspq_power_spectrum = None
        self.pspq_drift_frequencies = None
        self.pspq_max_power = None
        self.pspq_pvalue = None
        self.pspq_significance_threshold = None
        self.pspq_drift_detected = None
        self.pspq_reconstruction = None
        self.pspq_reconstruction_power_spectrum = None
        self.pspq_reconstruction_powerpertimestep = None
         
        self.pq_power_spectrum = None
        self.pq_max_power = None
        self.pq_pvalue = None
        self.pq_significance_threshold = None      
        self.pq_drift_detected = None
        self.pq_drift_frequencies = None
        self.pq_reconstruction_power_spectrum = None 

        self.global_power_spectrum = None
        self.global_max_power = None
        self.global_pvalue = None
        self.global_significance_threshold = None       
        self.global_drift_detected = None
        self.global_drift_frequencies = None
        self.global_reconstruction_power_spectrum = None

    def plot_power_spectrum(self,sequence='averaged',qubit='averaged',threshold=True,figsize=(15,3)):
        
        _plt.figure(figsize=figsize)
        
        if sequence == 'averaged' and qubit == 'averaged':       
            spectrum = self.global_power_spectrum
            threshold = self.global_significance_threshold
            
        elif sequence == 'averaged' and qubit != 'averaged':       
            spectrum = self.pq_power_spectrum[qubit,:]
            threshold = self.pq_significance_threshold
            
        elif sequence != 'averaged' and qubit != 'averaged':       
            spectrum = self.pspq_power_spectrum[sequence,qubit,:]
            threshold = self.pspq_significance_threshold    
        
        if self.timestep is not None:
            xlabel = "Frequence (Hertz)"
        else:
            xlabel = "Frequence"
        
        _plt.plot(self.frequencies,spectrum,'b.-',label='Data power spectrum')
        _plt.plot(self.frequencies,_np.ones(self.number_of_timesteps),'k--',label='Mean noise level')
        _plt.plot(self.frequencies,threshold*_np.ones(self.number_of_timesteps),'--', 
                  label=str(self.confidence)+' Significance threshold')

        _plt.legend()
        _plt.xlim(0,_np.max(self.frequencies))
        _plt.title("Power spectrum",fontsize=17)
        _plt.xlabel(xlabel,fontsize=15)
        _plt.ylabel("Power",fontsize=15)
        _plt.show()

    def plot_pvalues(self,sequence='averaged',qubit='averaged',threshold=True,figsize=(15,3)):
        
        _plt.figure(figsize=figsize)
        
        if sequence == 'averaged' and qubit == 'averaged':       
            spectrum = self.global_power_spectrum
            threshold = self.global_significance_threshold
            title = 'Global power spectrum'
            
        elif sequence == 'averaged' and qubit != 'averaged':       
            spectrum = self.pq_power_spectrum[qubit,:]
            threshold = self.pq_significance_threshold
            title = 'Sequence-averaged power spectrum for qubit '+str(qubit)
            
        elif sequence != 'averaged' and qubit != 'averaged':       
            spectrum = self.pspq_power_spectrum[sequence,qubit,:]
            threshold = self.pspq_significance_threshold    
            title = 'Power spectrum for qubit ' +str(qubit)+ ' and sequence ' +str(sequence)
            
        if self.timestep is not None:
            xlabel = "Frequence (Hertz)"
        else:
            xlabel = "Frequence"
        
        _plt.plot(self.frequencies,spectrum,'b.-',label='Data power spectrum')
        _plt.plot(self.frequencies,_np.ones(self.number_of_timesteps),'k--',label='Mean noise level')
        _plt.plot(self.frequencies,threshold*_np.ones(self.number_of_timesteps),'--', 
                  label=str(self.confidence)+' Significance threshold')

        _plt.legend()
        _plt.xlim(0,_np.max(self.frequencies))
        _plt.title(title,fontsize=17)
        _plt.xlabel(xlabel,fontsize=15)
        _plt.ylabel("Power",fontsize=15)
        _plt.show()

