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
        self.individual_drift_frequencies = None
        self.individual_max_powers = None
        self.individual_max_power_pvalues = None
        self.individual_max_power_threshold = None
        self.individual_drift_detected = None
        self.individual_reconstruction = None
        self.individual_reconstruction_power_per_time = None