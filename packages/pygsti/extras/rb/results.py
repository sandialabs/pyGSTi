""" Encapsulates RB results and dataset objects """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np

class RBSummaryDataset(object):
    """
    Encapsulates a summary dataset.

    """
    def __init__(self, n, lengths, successcounts, totalcounts, circuitdepths=None, circuit2Qgcounts=None):
        
        self.number_of_qubits = n
        self.lengths = lengths
        self.successcounts = successcounts
        self.totalcounts = totalcounts
        self.circuitdepths = circuitdepths
        self.circuit2Qgcounts = circuit2Qgcounts
        
        # Create the average success probabilities.
        ASPs = []
        SPs = []
        for i in range(0,len(lengths)):
            SParray = _np.array(successcounts[i])/_np.array(totalcounts[i])
            SPs.append(list(SParray))
            ASPs.append(_np.mean(SParray))        
        self.ASPs = ASPs
        
    def generate_bootstrap(self, finite_sample_error=True):

        for i in range(samples): 

            # A new set of bootstrapped survival probabilities.
            sampled_scounts = []

            for j in range(len(lengths)):

                sampled_counts.append([])
                circuits_at_length = len(scounts[j])

                for k in range(circuits_at_length):
                    sampled_scounts[j].append(SPs[j][_np.random.randint(k_at_length)])
                if finite_sample_error:   
                    sampled_scounts[j] = _np.random.binomial(self.data.totalcounts,self.data.SPs)               
        
        BStrappeddataset = RBSummaryDataset(self.number_of_qubits, self.lengths, self.successcounts, 
                                            self.totalcounts, self.circuitdepths, self.circuit2Qgcounts)
                    
        return BStrappeddataset
    
    def std_analysis():
        
        # A wrap-around for ....

class RBResults(object):
    
    def __init__(self, data):

        self.data = data
        self.bootstraps = None
