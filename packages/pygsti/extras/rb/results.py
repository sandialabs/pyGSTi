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
    This dataset type only records the "RB length" of a circuit, how many times the circuit resulted in "success", and, 
    optionally, some basic circuit information that can be helpful in understandingthe results. I.e., it doesn't 
    store all the details about the circuits and the counts for each circuit (use a standard DataSet object to store
    the entire output of RB experiments).
    """
    def __init__(self, number_of_qubits, lengths, success_counts=None, total_counts=None, success_probabilities=None,
                 circuit_depths=None, circuit_twoQgate_counts=None, sortedinput=False, finitesampling=True, 
                 descriptor=''):
        """
        Initialize an RB summary dataset.

        Parameters
        ----------
        number_of_qubits : int
            The number of qubits the dataset is for. This should be the number of qubits the RB experiments where
            "holistically" performed on. So, this dataset type is not suitable for, e.g., a *full* set of simultaneous 
            RB data, which consists of parallel RB on different qubits. Data of that sort can be input into
            multiple RBSummaryDataset objects.

        lengths : list of ints
            A list of the "RB lengths" that the data is for. I.e., these are the "m" values in Pm = A + Bp^m.
            E.g., for direct RB this should be the number of circuit layers of native gates in the "core" circuit 
            (i.e., not including the prep/measure stabilizer circuits). For Clifford RB this should be the number of 
            Cliffords in the circuit (+ an arbitrary constant, traditionally -1, but -2 is more consistent with
            direct RB and is the pyGSTi convention for generating CRB circuits) *before* it is compiled into the 
            native gates. This can always be the length value used to generate the circuit, if a pyGSTi RB
            circuit/experiment generation function was used to generate the circuit.

            This list should be the same length as the input results data (e.g., `success_counts` below). If
            `sortedinput` is False (the default), it is a list that has an entry for each circuit run (so values
            can appear multiple times in the list and in any order). If `sortedinput` is True is an ordered list 
            containing each and every RB length once.

        success_counts : list of ints, or list of list of ints, optional
            Success counts, i.e., the number of times a circuit returns the "success" result. Normally this
            should be a list containing ints with `success_counts[i]` containing the success counts for a circuit
            with RB length `length[i]`. This is the case when `sortedinput` is False. But, if  `sortedinput` is
            True, it is instead a list of lists of ints: the list at `success_counts[i]` contains the data for
            all circuits with RB length `lengths[i]` (in this case `lengths` is an ordered list containing each
            RB length once). `success_counts` can be None, and the data can instead be specified via 
            `success_probabilities`. But, inputing the data as success counts is the preferred option for 
            experimental data.
        
        total_counts : int, or list of ints, or list of list of ints, optional
            If not None, an int that specifies the total number of counts per circuit *or* a list that specifies
            the total counts for each element in success_counts (or success_probabilities). This is *not* optional
            if success_counts is provided, and should always be specified with experimental data.

        success_probabilities : list of floats, or list of list of floats, optional
            The same as `success_counts` except that this list specifies observed survival probabilities, rather
            than the number of success counts. Can only be specified if `success_counts` is None, and it is better
            to input experimental data as `success_counts` (but this option is useful for finite-sampling-free
            simulated data).

        circuit_depths : list of ints, or list of list of ints, optional
            Has same format has `success_counts` or `success_probabilities`. Contains circuit depths. This is
            additional auxillary information that it is often useful to have when analyzing data from any type
            of RB that includes any compilation (e.g., Clifford RB). But this is not essential.

        circuit_twoQgate_counts : list of ints, or list of list of ints, optional
            Has same format has `success_counts` or `success_probabilities`. Contains circuit 2-qubit gate counts. 
            This is additional auxillary information that it is often useful for interpretting RB results.

        sortedinput : bool, optional
            Specifies the format of the input data. If False, all data is assumed to be input as a (possibly
            unsorted) list. I.e., all of the above are lists. If True, it is assumed that the `lengths` list
            contains the sorted and unique RB lengths. The data, `success_counts` etc, is then assumed to be
            a list of lists, where the list at element i is the data for the circuits at length `length[i]`.

        finitesampling : bool, optional
            Specifies whether the data is obtained from finite-sampling of the outcome of each circuit. So, should
            always be True for experimental data.

        descriptor :  str, optional
            A string that describes what the data is for.
        """       
        assert(not (success_counts == None and success_probabilities == None)), "Either success probabilities or success counts must be provided!"
        assert(not (success_counts != None and success_probabilities != None)), "Success probabilities *and* success counts should not both be provided!"
        assert(not (success_counts != None and total_counts == None)), "If success counts are provided total counts must be provided as well!"

        if success_counts != None:
            assert(len(success_counts) == len(lengths)), "Input data shapes are inconsistent!"
        if success_probabilities != None:
            assert(len(success_probabilities) == len(lengths)), "Input data shapes are inconsistent!"
        if total_counts != None:
            # If total counts is an int, convert to a list.
            if isinstance(total_counts,int): 
                total_counts = [total_counts for i in range(len(lengths))]
            else: 
                assert(len(total_counts) == len(lengths)), "Input data shapes are inconsistent!"   

        self.number_of_qubits = number_of_qubits
        self.finitesampling = finitesampling
        self.descriptor = descriptor
        self.bootstraps = None
         
        # If the input is not already sorted, then we have to order it as a list of lists.
        if not sortedinput:

            # Find an ordered arrays of sequence lengths at each n
            ordered_lengths = []
            for l in lengths:
                # If the length hasn't yet been added, put it into the lengths list
                if l not in ordered_lengths: ordered_lengths.append(l)
            # Sort the lengths list from lowest to highest.
            ordered_lengths.sort()

            # Take all the raw data and put it into lists for each sequence length.
            if success_counts is not None:
                scounts = []
                tcounts = []
                SPs = None
            else:
                scounts = None
                if total_counts is not None: tcounts = []
                else: tcounts = None
                SPs = []

            # If there's circuit info, create lists to input it into.
            if circuit_depths is not None: cdepths = []
            else: cdepths = None
            if circuit_twoQgate_counts is not None: c2Qgc = []
            else: c2Qgc = None

            for i in range(0,len(ordered_lengths)):
                if success_counts is not None:
                    scounts.append([])
                    tcounts.append([])
                else:
                    SPs.append([])
                    # It is allowed to have total_counts unspecified with SPs input.
                    if total_counts is not None: tcounts.append([])
                if circuit_depths is not None: cdepths.append([])
                if circuit_twoQgate_counts is not None: c2Qgc.append([])

            for i in range(0,len(lengths)):
                index = ordered_lengths.index(lengths[i])
                if success_counts is not None:
                    scounts[index].append(success_counts[i])
                    tcounts[index].append(total_counts[i])
                else:
                    SPs[index].append(success_probabilities[i])
                    if total_counts is not None: tcounts[index].append(total_counts[i]) 
                if circuit_depths is not None: cdepths[index].append(circuit_depths[i])
                if circuit_twoQgate_counts is not None: c2Qgc[index].append(circuit_twoQgate_counts[i])

            lengths = ordered_lengths
            success_counts = scounts
            success_probabilities = SPs
            total_counts = tcounts
            circuit_depths = cdepths
            circuit_twoQgate_counts = c2Qgc

        # If they are not provided, create the success probabilities
        if success_probabilities == None:
            success_probabilities = []
            for i in range(0,len(lengths)):
                SParray = _np.array(success_counts[i])/_np.array(total_counts[i])
                success_probabilities.append(list(SParray))

        # Create the average success probabilities.
        ASPs = []       
        for i in range(len(lengths)): ASPs.append(_np.mean(_np.array(success_probabilities[i])))        

        # If data is provided as probabilities, but we know the total counts, we populate self.success_counts
        if success_counts == None and total_counts != None:
            success_counts = []
            for i in range(0,len(lengths)):
                SCarray = _np.round(_np.array(success_probabilities[i])*_np.array(total_counts[i]))
                success_counts.append([int(k) for k in SCarray])

        self.lengths = lengths
        self.success_counts = success_counts
        self.success_probabilities = success_probabilities
        self.total_counts = total_counts
        self.circuit_depths = circuit_depths
        self.circuit_twoQgate_counts = circuit_twoQgate_counts
        self.ASPs =  ASPs
        
    def add_bootstrapped_datasets(self, samples=1000):
        """
        Adds non-parameteric bootstrapped datasets to the self.bootstraps list (and creates that
        list if it is currently None). The bootstrap is over both the finite-sample-error of each
        circuit and over the circuits at each length.

        Parameters
        ----------
        samples : int, optional
            The number of bootstrapped datasets to construct.

        Returns
        -------
        None
        """
        if self.finitesampling == True and self.total_counts is None:
            print("Warning -- finite sampling is not taken into account!")

        if self.bootstraps is None:
            self.bootstraps = []

        for i in range(samples): 

            # A new set of bootstrapped survival probabilities.
            if self.total_counts is not None:  
                sampled_scounts = []
            else:
                sampled_SPs = []

            for j in range(len(self.lengths)):

                sampled_scounts.append([])
                # The success probabilities are always there.
                circuits_at_length = len(self.success_probabilities[j])

                for k in range(circuits_at_length):
                    sampled_SP = self.success_probabilities[j][_np.random.randint(circuits_at_length)]
                    if self.total_counts is not None:  
                        sampled_scounts[j].append(_np.random.binomial(self.total_counts[j][k],sampled_SP))
                    else:               
                         sampled_SPs[j].append(sampled_SP)
            
            if self.total_counts is not None:  
                BStrappeddataset = RBSummaryDataset(self.number_of_qubits, self.lengths, success_counts=sampled_scounts, 
                                                total_counts=self.total_counts, sortedinput=True, finitesampling=self.finitesampling,
                                                descriptor='data created from a non-parametric bootstrap')

            else:
                BStrappeddataset = RBSummaryDataset(self.number_of_qubits, self.lengths, success_counts=None, 
                                                total_counts=None, successprobabilites=sampled_SPs, sortedinput=True, 
                                                finitesampling=self.finitesampling, 
                                                descriptor='data created from a non-parametric bootstrap without per-circuit finite-sampling error')

            self.bootstraps.append(BStrappeddataset)

    def create_smaller_dataset(self, numberofcircuits):
        """
        Creates a new dataset that has discarded the data from all but the first `numberofcircuits` 
        circuits at each length. 

        Parameters
        ----------
        numberofcircuits : int
            The maximum number of circuits to keep at each length.

        Returns
        -------
        RBSummaryDataset
            A new dataset containing less data.
        """
        newRBSdataset = _copy.deepcopy(self)
        for i in range(len(newRBSdataset.lengths)):
            if newRBSdataset.success_counts != None:
                newRBSdataset.success_counts[i] = newRBSdataset.success_counts[i][:numberofcircuits]
            if newRBSdataset.success_probabilities != None:
                newRBSdataset.success_probabilities[i] = newRBSdataset.success_probabilities[i][:numberofcircuits]
            if newRBSdataset.total_counts != None:
                newRBSdataset.total_counts[i] = newRBSdataset.total_counts[i][:numberofcircuits]
            if newRBSdataset.circuit_depths != None:
                newRBSdataset.circuit_depths[i] = newRBSdataset.circuit_depths[i][:numberofcircuits]
            if newRBSdataset.circuit_twoQgate_counts != None:
                newRBSdataset.circuit_twoQgate_counts[i] = newRBSdataset.circuit_twoQgate_counts[i][:numberofcircuits]

        return newRBSdataset
   
class FitResults(object):
    """
    An object to contain the results from fitting RB data. Currently just a
    container for the results, and does not include any methods.
    """
    def __init__(self, fittype, seed, rtype, success, estimates, variable, stds=None,  
                 bootstraps=None, bootstraps_failrate=None):
        """
        Initialize a FitResults object.

        Parameters
        ----------
        fittype : str
            A string to identity the type of fit.

        seed : list
            The seed used in the fitting.

        rtype : {'IE','AGI'}
            The type of RB error rate that the 'r' in these fit results corresponds to.

        success : bool
            Whether the fit was successful.

        estimates : dict
            A dictionary containing the estimates of all parameters

        variable : dict
            A dictionary that specifies which of the parameters in "estimates" where variables
            to estimate (set to True for estimated parameters, False for fixed constants). This
            is useful when fitting to A + B*p^m and fixing one or more of these parameters: because
            then the "estimates" dict can still be queried for all three parameters.

        stds : dict, optional
            Estimated standard deviations for the parameters.

        bootstraps : dict, optional
            Bootstrapped values for the estimated parameters, from which the standard deviations
            were calculated.

        bootstraps_failrate : float, optional
            The proporition of the estimates of the parameters from bootstrapped dataset failed. 
        """
        self.fittype = fittype
        self.seed = seed
        self.rtype = rtype
        self.success = success 
        
        self.estimates = estimates
        self.variable = variable
        self.stds = stds
        
        self.bootstraps = bootstraps
        self.bootstraps_failrate = bootstraps_failrate

class RBResults(object):
    """
    An object to contain the results of an RB analysis
    """
    def __init__(self, data, rtype, fits):
        """
        Initialize an RBResults object.

        Parameters
        ----------
        data : RBSummaryDataset
            The RB summary data that the analysis was performed for.

        rtype : {'IE','AGI'}
            The type of RB error rate, corresponding to different dimension-dependent
            re-scalings of (1-p), where p is the RB decay constant in A + B*p^m.

        fits : dict
            A dictionary containing FitResults objects, obtained from one or more
            fits of the data (e.g., a fit with all A, B and p as free parameters and
            a fit with A fixed to 1/2^n).
        """
        self.data = data
        self.rtype = rtype
        self.fits = fits

    def plot(self, fitkey=None, decay=True, success_probabilities=True, size=(8,5), ylim=None, xlim=None, 
             legend=True, title=None, figpath=None):
        """
        Plots RB data and, optionally, a fitted exponential decay.

        Parameters
        ----------
        fitkey : dict key, optional
            The key of the self.fits dictionary to plot the fit for. If None, will
            look for a 'full' key (the key for a full fit to A + Bp^m if the standard
            analysis functions are used) and plot this if possible. It otherwise checks
            that there is only one key in the dict and defaults to this. If there are
            multiple keys and none of them are 'full', `fitkey` must be specified when
            `decay` is True.

        decay : bool, optional
            Whether to plot a fit, or just the data.

        success_probabilities : bool, optional
            Whether to plot the success probabilities distribution, as a violin plot. (as well
            as the *average* success probabilities at each length).

        size : tuple, optional
            The figure size

        ylim, xlim : tuple, optional
            The x and y limits for the figure.

        legend : bool, optional
            Whether to show a legend.

        title : str, optional
            A title to put on the figure.

        figpath : str, optional
            If specified, the figure is saved with this filename.
        """
        
        # Future : change to a plotly plot.
        try: import matplotlib.pyplot as _plt
        except ImportError: raise ValueError("This function requires you to install matplotlib!")

        if decay and fitkey is None:
            allfitkeys = list(self.fits.keys())
            if 'full' in allfitkeys: fitkey = 'full'
            else: 
                assert(len(allfitkeys) == 1), "There are multiple fits and none have the key 'full'. Please specify the fit to plot!"
                fitkey = allfitkeys[0]
        
        _plt.figure(figsize=size)
        _plt.plot(self.data.lengths,self.data.ASPs,'o', label='Average success probabilities')
        
        if decay:
            lengths = _np.linspace(0,max(self.data.lengths),200)
            A = self.fits[fitkey].estimates['A']
            B = self.fits[fitkey].estimates['B']
            p = self.fits[fitkey].estimates['p']
            _plt.plot(lengths,A+B*p**lengths, label = 'Fit, r = {:.2} +/- {:.1}'.format(self.fits[fitkey].estimates['r'],
                                                                                  self.fits[fitkey].stds['r']))
    
        if success_probabilities:
            _plt.violinplot(list(self.data.success_probabilities),self.data.lengths, points=10, widths=1., showmeans=False, 
                                 showextrema=False, showmedians=False) #, label='Success probabilities')
        
        if title is not None: _plt.title(title)
        _plt.ylabel("Success probability")
        _plt.xlabel("RB sequence length $(m)$")
        _plt.ylim(ylim)
        _plt.xlim(xlim)

        if legend: _plt.legend()
        
        if figpath is not None: _plt.savefig(figpath,dpi=1000)
        else: _plt.show()

        return