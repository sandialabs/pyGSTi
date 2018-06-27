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
                 circuitdepths=None, circuit2Qgcounts=None, sortedinput=False, finitesampling=True, descriptor=None):

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
        self.finitesampling = finitesampling
        self.descriptor = descriptor
        self.bootstraps = None

        if type(totalcounts) is int:
            totalcounts = [totalcounts for i in range(len(lengths))]

         
        if not sortedinput:

            # Find and order arrays of sequence lengths at each n
            ordered_lengths = []
            for l in lengths:
                if l not in ordered_lengths:
                      ordered_lengths.append(l)
            ordered_lengths.sort()

            # Take all the raw data and put it into lists for each sequence length
            if successcounts is not None:
                scounts = []
                tcounts = []
                SPs = None
            else:
                scounts = None
                if totalcounts is not None:
                    tcounts = []
                else:
                    tcounts = None
                SPs = []

            if circuitdepths is not None:
                cdepths = []
            else:
                cdepths = None
            if circuit2Qgcounts is not None:
                c2Qgc = []
            else:
                c2Qgc = None

            for i in range(0,len(ordered_lengths)):
                if successcounts is not None:
                    scounts.append([])
                    tcounts.append([])
                else:
                    SPs.append([])
                    if totalcounts is not None:
                        tcounts.append([])
                if circuitdepths is not None:
                    cdepths.append([])
                if circuit2Qgcounts is not None:
                    c2Qgc.append([])

            for i in range(0,len(lengths)):
                index = ordered_lengths.index(lengths[i])
                if successcounts is not None:
                    scounts[index].append(successcounts[i])
                    tcounts[index].append(totalcounts[i])
                else:
                    SPs[index].append(successprobabilities[i])
                    if totalcounts is not None:
                       tcounts[index].append(totalcounts[i]) 
                if circuitdepths is not None:
                    cdepths[index].append(circuitdepths[i])
                if circuit2Qgcounts is not None:
                    c2Qgc[index].append(circuit2Qgcounts[i])

            lengths = ordered_lengths
            successcounts = scounts
            successprobabilities = SPs
            totalcounts = tcounts
            circuitdepths = cdepths
            circuit2Qgcounts = c2Qgc

        # If they are not provided, create the success probabilities
        if successprobabilities == None:
            successprobabilities = []
            for i in range(0,len(lengths)):
                SParray = _np.array(successcounts[i])/_np.array(totalcounts[i])
                successprobabilities.append(list(SParray))

        # Create the average success probabilities.
        ASPs = []       
        for i in range(0,len(lengths)):
            ASPs.append(_np.mean(_np.array(successprobabilities[i])))        

        # If data is provided as probabilities, but we know the total counts, we populate self.successcounts
        if successcounts == None and totalcounts != None:
            successcounts = []
            for i in range(0,len(lengths)):
                SCarray = _np.round(_np.array(successprobabilities[i])*_np.array(totalcounts[i]))
                successcounts.append([int(k) for k in SCarray])

        self.lengths = lengths
        self.successcounts = successcounts
        self.successprobabilities = successprobabilities
        self.totalcounts = totalcounts
        self.circuitdepths = circuitdepths
        self.circuit2Qgcounts = circuit2Qgcounts
        self.ASPs =  ASPs
        
    def add_bootstrapped_datasets(self, samples=1000):

        if self.finitesampling == True and self.totalcounts is None:
            print("Warning -- finite sampling is not taken into account!")

        if self.bootstraps is None:
            self.bootstraps = []

        for i in range(samples): 

            # A new set of bootstrapped survival probabilities.
            if self.totalcounts is not None:  
                sampled_scounts = []
            else:
                sampled_SPs = []

            for j in range(len(self.lengths)):

                sampled_scounts.append([])
                circuits_at_length = len(self.successprobabilities[j])

                for k in range(circuits_at_length):
                    sampled_SP = self.successprobabilities[j][_np.random.randint(circuits_at_length)]
                    if self.totalcounts is not None:  
                        sampled_scounts[j].append(_np.random.binomial(self.totalcounts,sampled_SP))
                    else:               
                         sampled_SPs[j].append(sampled_SP)
            
            if self.totalcounts is not None:  
                BStrappeddataset = RBSummaryDataset(self.number_of_qubits, self.lengths, successcounts=sampled_scounts, 
                                                totalcounts=self.totalcounts, circuitdepths=self.circuitdepths, 
                                                circuit2Qgcounts=self.circuit2Qgcounts, sortedinput=True, finitesampling=self.finitesampling,
                                                descriptor='data created from a non-parametric bootstrap')

            else:
                BStrappeddataset = RBSummaryDataset(self.number_of_qubits, self.lengths, successcounts=None, 
                                                totalcounts=None, successprobabilites=sampled_SPs, circuitdepths=self.circuitdepths, 
                                                circuit2Qgcounts=self.circuit2Qgcounts, sortedinput=True, finitesampling=self.finitesampling,
                                                descriptor='data created from a non-parametric bootstrap without per-circuit finite-sampling error')

            self.bootstraps.append(BStrappeddataset)





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
   

class FitResults(object):

    def __init__(self, fittype, seed, rtype, success, estimates, variable, stds=None,  bootstraps=None, bootstraps_failrate=None):

        self.fittype = fittype
        self.seed = seed
        self.rtype = rtype
        self.success = success 
        
        self.estimates = estimates
        self.variable = variable
        self.stds = stds
        
        self.std = None
        self.bootstraps = bootstraps
        self.bootstraps_failrate = bootstraps_failrate

class RBResults(object):
    """
   Todo :docstring
    """

    def __init__(self, data, rtype, fits):

        self.data = data
        self.rtype = rtype
        self.fits = fits