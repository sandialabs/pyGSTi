""" Encapsulates RB results and dataset objects """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import copy as _copy

class RBSummaryDataset(object):
    """
    An object to summarize the results of RB experiments as relevant to implementing a standard RB analysis on the data. 

    This dataset does not contain most of the information in RB experiment results, because it records only the "RB length" of a circuit, 
    how many times the circuit result in "success", and -- optionally -- some basic circuit information, that can be helpful in understanding
    the results.

    """
    def __init__(self, number_of_qubits, lengths, successcounts=None, successprobabilities=None, totalcounts=None, 
                circuitdepths=None, circuit2Qgcounts=None, finitesampling=True, descriptor=None):

        """
        Initialize an RB summary dataset.

        Parameters
        ----------
        number_of_qubits: int
            The number of qubits the dataset is for. This should be the number of qubits the RB experiments where
            "holistically" performed on. This dataset type is not suitable for, e.g., simultaneous RB, which might
            consist of 1-qubit RB simultaneous on 10 qubits.

        lengths: list of ints
            A list of the "RB lengths" that the data is for. I.e., these are the "m's" in Pm = A + Bp^m.
            For direct RB this should be the number of circuit layers of native gates in the "core" circuit (i.e.,
            not including the prep/measure stabilizer circuits). For Clifford RB this should be the number of 
            Cliffords in the circuit (+ an arbitrary constant, traditionally -1, but -2 is most consistent with
            direct RB) *before* it is compiled into the native gates.

        successcounts: None or list of list of ints.
            ......
            

        """       
        assert(not (successcounts == None and successprobabilities == None)), "Either success probabilities or success counts must be provided!"
        assert(not (successcounts != None and successprobabilities != None)), "Success probabilities *and* success counts should not both be provided!"
        assert(not (successcounts != None and totalcounts == None)), "If success counts are provided total counts must be provided as well!"

        self.number_of_qubits = number_of_qubits
        self.lengths = lengths
        self.successcounts = successcounts
        self.successprobabilities = successprobabilities
        self.totalcounts = totalcounts
        self.circuitdepths = circuitdepths
        self.circuit2Qgcounts = circuit2Qgcounts
        self.finitesampling = finitesampling
        self.descriptor = descriptor

        # If they are not provided, create the success probabilities
        if successprobabilities == None:
            SPs = []
            for i in range(0,len(lengths)):
                SParray = _np.array(successcounts[i])/_np.array(totalcounts[i])
                SPs.append(list(SParray))
            self.successprobabilities = SPs 

        # Create the average success probabilities.
        ASPs = []       
        for i in range(0,len(lengths)):
            ASPs.append(_np.mean(_np.array(self.successprobabilities[i])))        
        self.ASPs = ASPs

        # If data is provided as probabilities, but we know the total counts, we populate self.successcounts
        if successcounts == None and totalcounts != None:
            SCs = []
            for i in range(0,len(lengths)):
                SCarray = _np.round(_np.array(successprobabilities[i])*_np.array(totalcounts[i]))
                SCs.append([int(k) for k in SCarray])
            self.successcounts = SCs 
        
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

    def create_smaller_dataset(self, numberofcircuits):

        newRBSdataset = _copy.deepcopy(self)
        for i in range(len(newRBSdataset.lengths)):
            if newRBSdataset.successcounts != None:
                newRBSdataset.successcounts[i] = newRBSdataset.successcounts[i][:numberofcircuits]
            if newRBSdataset.successprobabilities != None:
                newRBSdataset.successprobabilities[i] = newRBSdataset.successprobabilities[i][:numberofcircuits]
            if newRBSdataset.totalcounts != None:
                newRBSdataset.totalcounts[i] = newRBSdataset.totalcounts[i][:numberofcircuits]
            if newRBSdataset.circuitdepths != None:
                newRBSdataset.circuitdepths[i] = newRBSdataset.circuitdepths[i][:numberofcircuits]
            if newRBSdataset.circuit2Qgcounts != None:
                newRBSdataset.circuit2Qgcounts[i] = newRBSdataset.circuit2Qgcounts[i][:numberofcircuits]

        return newRBSdataset
    
    #def std_analysis():
        
        # A wrap-around for ....

    
    # Can't have this as a method, because io imports this file.    
    #def write_to_file(self,filename):
    #    """
    #    Writes the dataset to a .txt file that can be read back in using the rb.io methods.
    #    """
    #    _io.write_rb_summary_dataset_to_file(self,filename)

class RBResults(object):
    """
   Todo :docstring
    """

    def __init__(self, data):
        self.data = data
        self.bootstraps = None