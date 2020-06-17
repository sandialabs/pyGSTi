""" Encapsulates RB results and dataset objects """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import copy as _copy

#from . import analysis as _analysis  # Doesn't exist!
_analysis = None  # TODO - fix or remove this dependency


def create_summary_datasets(ds, spec, datatype='adjusted', verbosity=1):
    """
    todo

    """
    structure = spec.get_structure()
    circuits = spec.get_circuits()
    lengths = list(circuits.keys())
    lengths.sort()

    success_counts = {}
    total_counts = {}
    hamming_distance_counts = {}

    for qubits in structure:

        if datatype == 'raw':
            success_counts[qubits] = {}
            total_counts[qubits] = {}
            hamming_distance_counts[qubits] = None

        elif datatype == 'adjusted':
            success_counts[qubits] = None
            total_counts[qubits] = None
            hamming_distance_counts[qubits] = {}

        else:
            raise ValueError("Requested data type ` {} ` not understood!".format(datatype))

    if verbosity == 1:
        tab = ' '
    if verbosity > 1:
        tab = '   '

    for mit, (m, circuitlist) in enumerate(circuits.items()):

        if verbosity > 0:
            print(tab + "- Processing length {} of {}".format(mit + 1, len(circuits)))

        for qubits in structure:
            if datatype == 'raw':
                success_counts[qubits][m] = []
                total_counts[qubits][m] = []
            elif datatype == 'adjusted':
                hamming_distance_counts[qubits][m] = []

        for (circ, target) in circuitlist:
            dsrow = ds[circ]
            for qubits in structure:
                if datatype == 'raw':
                    success_counts[qubits][m].append(_analysis.marginalized_success_counts(dsrow, circ, target, qubits))
                    total_counts[qubits][m].append(dsrow.total)
                elif datatype == 'adjusted':
                    hamming_distance_counts[qubits][m].append(
                        _analysis.marginalized_hamming_distance_counts(dsrow, circ, target, qubits))

    summary_data = {}
    for qubits in structure:
        #print(success_counts[qubits])
        #print(total_counts[qubits])
        #print(hamming_distance_counts[qubits])
        summary_data[qubits] = RBSummaryDataset(len(qubits), success_counts=success_counts[qubits],
                                                total_counts=total_counts[qubits],
                                                hamming_distance_counts=hamming_distance_counts[qubits])

    return summary_data


class RBSummaryDataset(object):
    """
    An object to summarize the results of RB experiments as relevant to implementing a standard RB analysis on the data.
    This dataset type only records the "RB length" of a circuit, how many times the circuit resulted in "success", and,
    optionally, some basic circuit information that can be helpful in understandingthe results. I.e., it doesn't
    store all the details about the circuits and the counts for each circuit (use a standard DataSet object to store
    the entire output of RB experiments).
    """

    def __init__(self, number_of_qubits, success_counts=None, total_counts=None, hamming_distance_counts=None,
                 aux={}, finitecounts=True, descriptor=''):
        """
        # todo : update.

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

        descriptor :  str, optional
            A string that describes what the data is for.

        """
        self.number_of_qubits = number_of_qubits
        self.finitecounts = finitecounts
        self.aux = _copy.deepcopy(aux)
        self.descriptor = descriptor

        assert(not (success_counts is not None and hamming_distance_counts is not None)), "Only one data " + \
            "type should be provided!"

        if success_counts is not None:

            self.datatype = 'success_counts'
            self.counts = _copy.deepcopy(success_counts)
            if self.finitecounts:
                assert(total_counts is not None), "The total counts per circuit is required!"
                self._total_counts = _copy.deepcopy(total_counts)
            else:
                self._total_counts = 1

        elif hamming_distance_counts is not None:

            self.datatype = 'hamming_distance_counts'
            self.counts = _copy.deepcopy(hamming_distance_counts)

            assert(total_counts is None), "The total counts per circuit should not be provided, " + \
                "as it is implicit in the Hamming distance data!"

            if self.finitecounts:
                # For Hamming distance data we just compute total counts on the fly.
                self._total_counts = None
            else:
                self._total_counts = 1

        else:
            raise ValueError("No data provided! `success_counts` or `hamming_distance_counts` must be not None!")

        lengths = list(self.counts.keys())
        lengths.sort()
        self.lengths = lengths

        # Generate "standard" and "adjusted" success probabilities

        self.SPs = []
        self.ASPs = []
        for l in self.lengths:
            SPs = [self.get_success_counts(l, i) / self.get_total_counts(l, i) for i in range(len(self.counts[l]))]
            self.SPs.append(SPs)
            self.ASPs.append(_np.mean(SPs))

        if self.datatype == 'hamming_distance_counts':
            self.adjusted_SPs = []
            self.adjusted_ASPs = []
            for l in self.lengths:
                adjSPs = [self.get_adjusted_success_probability(l, i) for i in range(len(self.counts[l]))]
                self.adjusted_SPs.append(adjSPs)
                self.adjusted_ASPs.append(_np.mean(adjSPs))

        else:
            self.adjusted_SPs = None
            self.adjusted_ASPs = None

        self.bootstraps = []

        return

    def get_adjusted_success_probability(self, length, index):
        """
        todo.
        """
        return _analysis.adjusted_success_probability(self.get_hamming_distance_distribution(length, index))

    def get_success_counts(self, length, index):
        """
        todo

        """
        if self.datatype == 'success_counts':
            return self.counts[length][index]

        else:
            return self.counts[length][index][0]

    def get_total_counts(self, length, index):
        """
        todo

        """
        if isinstance(self._total_counts, int):
            return self._total_counts

        elif self._total_counts is None:
            return _np.sum(self.counts[length][index])

        else:
            return self._total_counts[length][index]

    def get_hamming_distance_distribution(self, length, index):
        """
        todo

        """
        if self.datatype == 'hamming_distance_counts':
            return self.counts[length][index] / _np.sum(self.counts[length][index])

        else:
            raise ValueError("This is only possible for Hamming distance count data!")

    def get_success_probabilities(self, successtype='raw'):
        """
        todo.

        """
        if successtype == 'raw':
            return self.lengths, self.ASPs, self.SPs

        elif successtype == 'adjusted':
            return self.lengths, self.adjusted_ASPs, self.adjusted_SPs

    def add_bootstrapped_datasets(self, samples=1000):
        """
        Adds bootstrapped datasets. The bootstrap is over both the finite counts of each
        circuit and over the circuits at each length.

        Parameters
        ----------
        samples : int, optional
            The number of bootstrapped datasets to construct.

        Returns
        -------
        None
        """
        for i in range(len(self.bootstraps), samples):

            # A new set of bootstrapped success counts, or Hamming distance counts.
            if self.datatype == 'success_counts':

                success_counts = {}
                hamming_distance_counts = None
                total_counts = {}

                for j, l in enumerate(self.lengths):

                    success_counts[l] = []
                    if self.finitecounts:
                        total_counts[l] = []
                    else:
                        total_counts = None
                    numcircuits = len(self.SPs[j])

                    for k in range(numcircuits):

                        ind = _np.random.randint(numcircuits)
                        sampledSP = self.SPs[j][ind]
                        totalcounts = self.get_total_counts(l, ind)
                        if self.finitecounts:
                            success_counts[l].append(_np.random.binomial(totalcounts, sampledSP))
                            total_counts[l].append(totalcounts)
                        else:
                            success_counts[l].append(sampledSP)

            else:

                success_counts = None
                hamming_distance_counts = {}
                total_counts = None

                for j, l in enumerate(self.lengths):

                    hamming_distance_counts[l] = []
                    numcircuits = len(self.SPs[j])

                    for k in range(numcircuits):

                        ind = _np.random.randint(numcircuits)
                        sampledHDProbs = self.get_hamming_distance_distribution(l, ind)

                        if self.finitecounts:
                            totalcounts = self.get_total_counts(l, ind)
                            hamming_distance_counts[l].append(list(_np.random.multinomial(totalcounts, sampledHDProbs)))
                        else:
                            hamming_distance_counts[l].append(sampledHDProbs)

            bootstrapped_dataset = RBSummaryDataset(self.number_of_qubits, success_counts, total_counts,
                                                    hamming_distance_counts, finitecounts=self.finitecounts,
                                                    descriptor='data created from a non-parametric bootstrap')

            self.bootstraps.append(bootstrapped_dataset)

    # todo : add this back in.
    # def create_smaller_dataset(self, numberofcircuits):
    #     """
    #     Creates a new dataset that has discarded the data from all but the first `numberofcircuits`
    #     circuits at each length.

    #     Parameters
    #     ----------
    #     numberofcircuits : int
    #         The maximum number of circuits to keep at each length.

    #     Returns
    #     -------
    #     RBSummaryDataset
    #         A new dataset containing less data.
    #     """
    #     newRBSdataset = _copy.deepcopy(self)
    #     for i in range(len(newRBSdataset.lengths)):
    #         if newRBSdataset.success_counts is not None:
    #             newRBSdataset.success_counts[i] = newRBSdataset.success_counts[i][:numberofcircuits]
    #         if newRBSdataset.success_probabilities is not None:
    #             newRBSdataset.success_probabilities[i] = newRBSdataset.success_probabilities[i][:numberofcircuits]
    #         if newRBSdataset.total_counts is not None:
    #             newRBSdataset.total_counts[i] = newRBSdataset.total_counts[i][:numberofcircuits]
    #         if newRBSdataset.circuit_depths is not None:
    #             newRBSdataset.circuit_depths[i] = newRBSdataset.circuit_depths[i][:numberofcircuits]
    #         if newRBSdataset.circuit_twoQgate_counts is not None:
    #             newRBSdataset.circuit_twoQgate_counts[i] = newRBSdataset.circuit_twoQgate_counts[i][:numberofcircuits]

    #     return newRBSdataset
