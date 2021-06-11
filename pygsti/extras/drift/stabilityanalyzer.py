"""Defines the DriftResults class"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import copy as _copy
import itertools as _itertools
import warnings as _warnings

import numpy as _np

from . import probtrajectory as _ptraj
from . import signal as _sig
from ... import datasets as _datasets
from ...construction import datasetconstruction as _dsconst


def compute_auto_tests(shape, ids=False):
    """
    Returns the tests we'll automatically perform on time-series data, when a specific
    sets of tests is not given. Each test is specified by a tuple of length <= 3 containing
    some subset of 'dataset', 'circuit' and 'outcome', and a set of tests is a list of such
    tuples.

    Parameters
    ----------
    shape : tuple
        The "shape" of the time-sereis data that is being tested. A 3-tuple, whereby
        the shape[0] is the number of DataSets in the MultiDataSet being tested,
        shape[1] is the number of circuits, and shape[2] is the number of outcomes
        per circuit.

    ids : bool, optional.
        Whether or not we're returning the auto tests for a MultiDataSet containing
        independendet data sets. If the MultiDataSet we are testing only contains
        1 DataSet it does not matter what this is set to.

    Returns
    -------
    tuple
        A tuple containing the auto-generated tests to run.
    """
    if ids:
        auto_tests = ((), ('dataset', ), ('dataset', 'circuit'))
    else:
        auto_tests = (('dataset', ), ('dataset', 'circuit'))

    condensed_auto_tests, junk = condense_tests(shape, auto_tests, None)

    return condensed_auto_tests


def condense_tests(shape, tests, weightings=None):
    """
    Condenses a set of tests, by removing any tests that are equivalent given the meta-parameters
    of the data (e.g., the number of circuits).

    Parameters
    ----------
    shape : tuple
        The "shape" of the time-sereis data that is being tested. A 3-tuple, whereby
        the shape[0] is the number of DataSets in the MultiDataSet being tested,
        shape[1] is the number of circuits, and shape[2] is the number of outcomes
        per circuit.

    tests : list
        The set of tests to condense, based on `shape`.

    weightings : dict, optional
        A dictonary containing significance weightings for the tests, whereby the keys
        are the tests (the elemenets of the list `tests`) and the values are floats that
        are Bonferonni weightings for splitting significance.

    Returns
    -------
    tuple
        A tuple containing the auto-generated tests to run.

    if weightings is not None:

        dict
            A dictionary of weightings that has condensed the weightings in `weightings`,
            so that the weighting on a test that is being dropped (or relabelled) is
            redistributed to the equivalant test that we are implementing.

    """
    # This is the values of `shape` that correspond to a trivial axis in the data.
    trivialshape = {}
    trivialshape['dataset'] = 1
    trivialshape['circuit'] = 1
    trivialshape['outcome'] = 2

    condtests = []

    if weightings is not None: condweightings = {}
    else: condweightings = None

    for test in tests:
        condtest = []
        for i, axislabel in enumerate(('dataset', 'circuit', 'outcome')):
            if axislabel in test:
                if shape[i] > trivialshape[axislabel]:
                    condtest.append(axislabel)
        condtest = tuple(condtest)
        if condtest not in condtests:
            condtests.append(condtest)
            if weightings is not None:
                condweightings[condtest] = weightings[test]
        else:
            if weightings is not None:
                condweightings[condtest] += weightings[test]

    return condtests, condweightings


def compute_valid_tests():
    """
    Returns the set of all valid tests (specified by tuples), in the form of a
    list.
    """
    valid_tests = []
    valid_tests.append(())
    valid_tests.append(('dataset', ))
    valid_tests.append(('dataset', 'circuit'))
    valid_tests.append(('dataset', 'circuit', 'outcome'))
    valid_tests.append(('circuit', ))
    valid_tests.append(('circuit', 'outcome'))
    valid_tests.append(('outcome', ))
    valid_tests.append(('dataset', 'outcome'))

    return valid_tests


def check_valid_tests(tests):
    """
    Checks whether all the tuples in `tests` constitute valid tests, as specified.
    """
    valid_tests = compute_valid_tests()

    for test in tests: assert(test in valid_tests), "This is an invalid set of tests for drift detection!"


def compute_valid_inclass_corrections():
    """
    Returns the set of all valid `inclass_correction` dicts -- an input to the .run_instability_detection() of
    a StabilityAnalyzer. See the doctring of that method for more information on that input.
    """
    valid_inclass_corrections = []
    valid_inclass_corrections.append({'dataset': 'Bonferroni', 'circuit': 'Benjamini-Hochberg',
                                      'outcome': 'Benjamini-Hochberg', 'spectrum': 'Benjamini-Hochberg'})
    valid_inclass_corrections.append({'dataset': 'Bonferroni', 'circuit': 'Bonferroni',
                                      'outcome': 'Benjamini-Hochberg', 'spectrum': 'Benjamini-Hochberg'})
    valid_inclass_corrections.append({'dataset': 'Bonferroni', 'circuit': 'Bonferroni',
                                      'outcome': 'Bonferroni', 'spectrum': 'Benjamini-Hochberg'})
    valid_inclass_corrections.append({'dataset': 'Bonferroni', 'circuit': 'Bonferroni',
                                      'outcome': 'Bonferroni', 'spectrum': 'Bonferroni'})

    return valid_inclass_corrections


def populate_inclass_correction(inclass_correction={}):
    """
    Populates empty parts of an `inclass_correction` dictionary with auto values. This dictionary is an
    input to the .run_instability_detection() a StabilityAnalyzer. See the doctring of that method for
    more information on that input.

    The auto inclass_correction is to default to a Bonferroni correction at all levels above the lowest
    level where a correction has been specified.
    """
    autocorrection = 'Bonferroni'
    for key in ('dataset', 'circuit', 'outcome', 'spectrum'):
        if key not in inclass_correction:
            inclass_correction[key] = autocorrection
        if key in inclass_correction:
            # As soon as the correction changes from Bonferroni, we switch to that correction.
            autocorrection = inclass_correction[key]

    valid_inclass_corrections = compute_valid_inclass_corrections()
    assert(inclass_correction in valid_inclass_corrections), "This is an invalid inclass correction!"

    return inclass_correction


def compute_auto_betweenclass_weighting(tests, betweenclass_weighting=True):
    """
    Finds the automatic weighting used between classes of test, e.g., a
    "top level" Bonferroni correction, or no correction, etc.

    """
    if betweenclass_weighting:
        betweenclass_weighting = {test: 1 / len(tests) for test in tests}
    else:
        betweenclass_weighting = {test: 1 for test in tests}

    return betweenclass_weighting


def compute_auto_estimator(transform):
    """
    The default method for estimating the parameters of a parameterized probability trajectory (i.e., this is not
    the method used for the model selection, only the method used to estimate the amplitudes in the parameterized
    model). This returns the fastest method that is pretty reliable for that transform, rather than the most
    statistically well-motivated choice (which is mle in all cases).

    Parameters
    ----------
    transform : str
        The type of transform that we are auto-selecting an estimation method for.

    Returns
    -------
    str
        The name of the estimator to use.
    """
    if transform == 'dct': auto_estimator = 'filter'

    elif transform == 'lsp': auto_estimator = 'mle'

    else:
        raise ValueError("No auto estimation method available for {} transform!".format(transform))

    return auto_estimator


class StabilityAnalyzer(object):
    """
    The StabilityAnalyzer is designed to implement a stability analysis on time-series data from quantum
    circuits. It stores this time-series data, and contains methods for implementing instability detection
    and characterization, using spectral analysis techniques. These methods work on time-series data from
    any set of circuits, because they are entirely agnostic to the details of the circuits, e.g., they
    are not based on the process matrix model of GST.

    This object encapsulates stand-alone data analysis methods, but it is also the basis for implementing
    time-resolved benchmarking or tomography (e.g., time-resolved RB, GST or Ramsey spectroscopy).
    in pyGSTi.

    """

    def __init__(self, ds, transform='auto', marginalize='auto', mergeoutcomes=None, constnumtimes='auto',
                 ids=False):
        """
        Initialize a StabilityAnalyzer, by inputing time-series data and some information on how it should be
        processed.

        *** Some of the nominally allowed values for the inputs are not yet functional. For
        entirely non-functional code an assert() will flag up the input as not yet allowed, and for untested
        and perhaps unreliable code a warning will be flagged but the code will still run ***

        Parameters
        ----------
        ds : DataSet or MultiDataSet
            A DataSet containing time-series data to be analyzed for signs of instability.

        transform : str, optional
            The type of transform to use in the spectral analysis. Options are:

                - 'auto':   An attempt is made to choose the best transform given the "meta-data" of the data,
                            e.g., the variability in the time-step between data points. For beginners,
                            'auto' is the best option. If you are familiar with the underlying methods, the
                            meta-data of the input, and the relative merits of the different transform, then
                            it is probably better to choose this yourself -- as the auto-selection is not hugely
                            sophisticated.

                - 'dct' :   The Type-II Discrete Cosine Transform (with an orthogonal normalization). This is
                            the only tested option, and it is our recommended option when the data is
                            approximately equally-spaced, i.e., the time-step between each "click" for each
                            circuit is almost a constant. (the DCT transform implicitly assumes that this
                            time-step is exactly constant)

                - 'dft' :   The discrete Fourier transform (with an orthogonal normalization). *** This is an
                            experimental feature, and the results are unreliable with this transform ***

                - 'lsp' :   The Lomb-Scargle periodogram.  *** This is an experimental feature, and the code is
                            untested with this transform ***

        marginalize : str or bool, optional
            True, False or 'auto'. Whether or not to marginalize multi-qubit data, to look for instability
            in the marginalized probability distribution over the two outcomes for each qubit. Cannot be
            set to True if mergeoutcomes is not None.

        mergeoutcomes : None or Dict, optional
            If not None, a dictionary of outcome-merging dictionaries. Each dictionary contained as a
            value of `mergeoutcomes` is used to create a new DataSet, where the values have been merged
            according to that dictionary (see the aggregate_dataset_outcomes() function inside datasetconstructions.py).
            The corresponding key is used as the key for that DataSet, when it is stored in a MultiDataSet,
            and the instability analysis is implemented on each DataSet. This is a more general data
            coarse-grainin option than `marginalize`.

        constnumtimes : str or bool, optional
            True, False or 'auto'. If True then data is discarded from the end of the "clickstream" for
            each circuit until all circuits have the same length clickstream, i.e., the same number of
            data aquisition times. If 'auto' then it is set to True or False depending on the meta-data of
            the data and the type of transform being used.

        ids: True or False, optional
            Whether the multiple DataSets should be treat as generated from independent random variables.
            If the input is a DataSet and `marginalize` is False and `mergeoutcomes` is None then this
            input is irrelevant: there is only ever one DataSet being analyzed. But in general multiple
            DataSets are concurrently analyzed. This is irrelevant for independent analyses of the DataSets,
            but the analysis is capable of also implementing a joint analysis of the DataSets. This joint
            analysis is only valid on the assumption of independent DataSets, and so this analysis will not
            be permitted unless `ids` is set to True. Note that the set of N marginalized data from N-qubit
            circuits are generally not independent -- even if the circuits contain no 2-qubit gates then
            crosstalk can causes dependencies. However, as long as the dependencies are weak then settings
            this to True is likely ok.

        Returns
        -------
        StabilityAnalyzer
            A new StabilityAnalyzer object, with data formatted and written in, ready for stability
            analyzes to be implemented.

        """
        assert(isinstance(ds, _datasets.DataSet)), "The input data must be a pyGSTi DataSet!"
        tempds = ds.copy_nonstatic()  # Copy so that we can edit the dataset.
        multids = _datasets.MultiDataSet()  # This is where the formatted data is recorded

        # We need the data to have the same number of total counts per-time for all the circuits.
        assert(tempds.has_constant_totalcounts_pertime), "Data must contain" \
            + "a constant number of total counts as a function of time-step and circuit!"

        if not isinstance(constnumtimes, bool):
            assert(constnumtimes == 'auto'), "The only non-boolean option is 'auto'!"
            constnumtimes = True
        # future: update the code to allow for this, and remove this assert.
        else:
            assert(constnumtimes), "Currently the analysis requires" \
                + "`constnumtimes` to be True!"

        if constnumtimes:
            tempds = _dsconst.trim_to_constant_numtimesteps(tempds)
            assert(tempds.has_constant_totalcounts), "Data formatting has failed!"

        # Checks that the specified transform is valid, and writes it into the object.
        if transform == 'auto': transform = 'dct'
        assert(transform in ('dct', 'lsp', 'dft')), "This is not a valid choice for the transform!"
        # future: ammend when the lsp code is functional, and remove when it is tested.
        if transform == 'lsp':
            _warnings.warn("The Lomb-Scargle-based analysis is an experimental feature! The results may be unrealiable,"
                           + " and probability trajectories cannot be estimated!")
        # future: ammend when the dft code is functional, and remove when it is tested.
        if transform == 'dft':
            _warnings.warn("The DFT-based analysis is an experimental feature! The results *are* unrealiable,"
                           + " and probability trajectories cannot be estimated!")
        self.transform = transform

        # Check that we have valid and consistent `marginalize` and `mergeoutcomes`, and write thems into object
        if isinstance(marginalize, str):
            assert(marginalize == 'auto'), "`marignalize` must be a boolean or 'auto'!"
            if mergeoutcomes is not None:
                marginalize = False  # A mergOutcomesDictDict means we can't marginalize as well.
            else:
                marginalize = True  # If there is no mergOutcomesDictDict we marginalize by default.
        else:
            assert(isinstance(marginalize, bool)), "`marginalize` must be a boolean or 'auto'!"

        if mergeoutcomes is not None:
            assert(not marginalize), "Cannot marginalize when a `mergeoutcomes` is specified!"

        self._marginalize = marginalize
        self._mergeoutcomes = mergeoutcomes

        # Do any outcome merging.
        if mergeoutcomes is not None:

            for dskey, mergeoutcomesdict in mergeoutcomes.items():
                mergds = _dsconst.aggregate_dataset_outcomes(tempds, mergeoutcomesdict)
                mergds.done_adding_data()
                multids.add_dataset(dskey, mergds)

        # Do any marginalization, labelling qubits as integers.
        if marginalize:

            n = len(ds.outcome_labels[0][0])
            if n > 1:
                for i in range(n):
                    margds = _dsconst.filter_dataset(tempds, (i,), filtercircuits=False)
                    margds.done_adding_data()
                    multids.add_dataset(str(i), margds)
            else:
                multids.add_dataset('0', tempds)

        if not mergeoutcomes and not marginalize:
            multids.add_dataset('0', tempds)

        # Data formatting complete, so write it into object.
        self.data = multids

        # future: update this, as it only works because `constnumtimes` = True.
        dskey = list(self.data.keys())[0]
        circuit = list(self.data[dskey].keys())[0]
        numtimes = len(self.data[dskey][circuit].time)

        if numtimes <= 50:
            _warnings.warn('There are not enough timestamps per circuit to'
                           + ' guarantee that the analysis will be reliable!')

        # If there's only one DataSet, we set `ids` to True because its value is
        # conceptually meaningless, and it's more convenient in the code to set it to True.
        if not ids:
            if len(multids.keys()):
                ids = True

        # Decide on the dofreductions, and write it (and the ids) into object.
        if ids:
            dofreduction = {'dataset': 0, 'circuit': 0, 'outcome': 1}
        else:
            dofreduction = {'dataset': None, 'circuit': 0, 'outcome': 1}
        self._ids = ids
        self._dofreduction = dofreduction

        # Properties where power spectra are stored.
        self._contains_spectra = False  # A bool keep tracking of whether spectra have been added yet.
        self.dof = 1  # The chi2 dof of each "base" spectrum under a null hypothesis, for calculating p values etc.
        # Will become a list of frequency lists, with each element of the list being the frequencies for one or more
        # circuit's spectra.
        self._frequencies = None
        # Will become a dictionary of ``pointers`` that designate the index of the `self._frequencies` list that the
        # frequencies for a circuit correspond to. The key is the circuit index (in self.data.keys()) and the value
        # is the index in self._frequncies.
        self._freqpointers = None
        self._dofalt = {}  # A dictionary containing alternative dofs, so that it can be adjusted in special cases.

        # Properties where results of drift detection are stored.
        self._driftdetectors = []
        self._def_detection = None
        self._significance = {}
        self._tests = {}
        self._freqstest = {}
        self._condtests = {}
        self._condbetweenclass_weighting = {}
        self._test_significance = {}
        self._inclass_correction = {}
        self._betweenclass_correction = {}
        self._power_sigthreshold = {}
        self._driftfreqinds = {}
        self._driftdetected_global = {}
        self._driftdetected_class = {}

        # Properties where results of drift characterization (i.e., probability traces) are stored.
        self._def_probtrajectories = None
        self._probtrajectories = {}

        return None

    def __str__(self):

        if not self._contains_spectra:
            s = "A StabilityAnalyzer containing times-series data, but waiting for power spectra to be generated,"
            s + " and a stability analysis to be performed."
            return s

        if self._def_detection is None:
            s = "A StabilityAnalyzer containing times-series data and power spectra, but waiting for a stability"
            s + " analysis to be performed."
            return s

        detectorkey = self._def_detection
        driftdetected = self.instability_detected(detectorkey=detectorkey, test=None)
        if driftdetected:
            s = "Instability *has* been detected,"
        else:
            s = "Instability has *not* been detected,"
        s += " from tests at a global significance of {}%" .format(100 * self._significance[detectorkey])
        return s

    def compute_spectra(self, frequencies='auto', freqpointers={}):
        """"
        Generates and records power spectra. This is the first stage in instability detection
        and characterization with a StabilityAnalyzer.

        Parameters
        ----------
        frequencies : 'auto' or list, optional
            The frequencies that the power spectra are calculated for. If 'auto' these are automatically
            determined from the meta-data of the time-series data (e.g., using the mean time between data
            points) and the transform being used. If not 'auto', then a list of lists, where each list is
            a set of frequencies that are the frequencies corresponding to one or more power spectra. The
            frequencies that should be paired to a given power spectrum are specified by `freqpointers`.

            These frequencies (whether automatically calculated or explicitly input) have a fundmentally
            different meaning depending on whether the transform is time-stamp aware (here, the LSP) or not
            (here, the DCT and DFT).

            Time-stamp aware transforms take the frequencies to calculate powers at *as an input*, so the
            specified frequencies are, explicitly, the frequencies associated with the powers. The task
            of choosing the frequencies amounts to picking the best set of frequencies at which to interogate
            the true probability trajectory for components. As there are complex factors involved in this
            choice that the code has no way of knowing, sometimes it is best to choose them yourself. E.g.,
            if different frequencies are used for different circuits it isn't possible to (meaningfully)
            averaging power spectra across circuits, but this might be preferable if the time-step is
            sufficiently different between different circuits -- it depends on your aims.

            For time-stamp unaware transforms, these frequencies should be the frequencies that, given
            that we're implementing the, e.g., DCT, the generated power spectrum is *implicitly* with respect
            to. In the case of data on a fixed time-grid, i.e., equally spaced data, then there is a
            precise set of frequencies implicit in the transform (which will be accurately extracted with
            frequencies set to `auto`). Otherwise, these frequencies are explicitly at least slightly
            ad hoc, and choosing these frequencies amounts to choosing those frequencies that "best"
            approximate the properties being interogatted with fitting each, e.g., DCT basis function
            to the (timestamp-free) data. The 'auto' option bases there frequencies solely on the
            mean time step and the number of times, and is a decent option when the time stamps are roughly
            equally spaced for each circuit.

            These frequencies should be in units of 1/t where 't' is the unit of the time stamps.

        freqpointers : dict, optional
            Specifies which frequencies correspond to which power spectra. The keys are power spectra labels,
            and the values are integers that point to the index of `frequencies` (a list of lists) that the
            relevant frquencies are found at. Whenever a power spectra is not included in `freqpointers` then
            this defaults to 0. So if `frequencies` is specified and is a list containing a single list (of
            frequencies) then `freqpointers` can be left as the empty dictionary.

        Returns
        -------
        None

        """
        if isinstance(frequencies, str):
            assert(frequencies == 'auto')
            frequencies, freqpointers = _sig.compute_auto_frequencies(self.data, self.transform)
        self._frequencies = frequencies
        self._freqpointers = freqpointers

        numfrequencies = len(self._frequencies[0])
        for freqs in self._frequencies[1:]:
            assert(numfrequencies == len(freqs)), "The number of frequencies must be fixed for all circuits!"

        self._axislabels = ('dataset', 'circuit', 'outcome')  # todo: explain.

        dskeys = tuple(self.data.keys())
        circuits = tuple(self.data[dskeys[0]].keys())
        outcomes = tuple(self.data.outcome_labels)
        arrayshape = []
        arrayshape = (len(dskeys), len(circuits), len(outcomes), numfrequencies)
        self._shape = arrayshape
        self._numfrequencies = numfrequencies

        self._tupletoindex = {}  # todo: explain
        self._basespectra = _np.zeros(tuple(arrayshape), float)  # An empty array that will be populated with spectra.
        self._derivedspectra = {}  # Stores derivative spectra, eg, the spectrum obtained by averaging along all axes.

        # If the type of transform used calculates modes as well as powers, initializes an array of the relevant data
        # type to store them.
        if self.transform == 'dct':
            self._modes = _np.zeros(tuple(arrayshape), complex)
        if self.transform == 'lsp':
            self._modes = None
        if self.transform == 'dft':
            self._modes = _np.zeros(tuple(arrayshape), float)

        # This requires a fixed counts or it'll throw an error. That is also enforced in __init__() but that might be
        # removed in the future, so we check it here again.
        counts = []
        for ds in self.data.values():
            counts.append(ds.totalcounts_pertime)
        assert(_np.var(_np.array(counts)) == 0), "An equal number of counts at every time-step " \
            "in every circuit is currently required!"
        counts = counts[0]
        self.counts = counts

        for i, dskey in enumerate(dskeys):
            ds = self.data[dskey]
            for j, circuit in enumerate(circuits):
                # The time-series data to generate power spectra for.
                times, outcomedict = ds[circuit].timeseries_for_outcomes
                # Go through the outcomes and generate a power spectrum for the clickstream of each outcome
                for k, outcome in enumerate(outcomes):

                    # Record where the spectrum is written in the array.
                    self._tupletoindex[(dskey, circuit, outcome)] = (i, j, k)
                    # Finds the pointer to the frequencies for the spectrum we're about to calculate.
                    pointer = self._freqpointers.get(j, 0)
                    # The frequencies for this spectrum. Note that if we're using a transform that doesn't
                    # actually take into account a set of frequencies it is necessary that this set of
                    # frequencies is "correct" in the sense that it corresponds to what'll be calculated.
                    freqs = self._frequencies[pointer]
                    # Calculates the power spectrum for this (dskey, circuit, outcome) tuple.
                    modes, powers = _sig.spectrum(outcomedict[outcome], times=times, null_hypothesis=None,
                                                  counts=counts, frequencies=freqs, transform=self.transform,
                                                  returnfrequencies=False)

                    self._basespectra[i, j, k] = powers
                    if modes is not None: self._modes[i, j, k] = modes

                    # todo: calculate any dof alternative here (perhaps do inside the spectrum function?),
                    # and record it.

        self._contains_spectra = True

        return None

    def get_dof_reduction(self, axislabel):
        """
        Find the null hypothesis chi2 degree of freedom (DOF) reduction when averaging power spectrum along
        `axislabel`.

            - Under null hypohesis, each base spectrum is distributed according to a chi2/k with a DOF k
              for some k (obtainable via get_dof()).
            - Under the composite null hypothesis the power spectrum, averaged along the `axislabel` axis
              is chi2/(numspectra*k - l) with a DOF of (numspectra*k - l) for some l, where numspectra is the
              number of spectrum along this axis.

        This function returns `l`.

        """
        return self._dofreduction[axislabel]

    def _check_dofreduction_set(self, axislabel):
        """
        Checks whether the DOF reduction is set along an axis, i.e., it is not None. If it is None, that means
        that we do not know how to reduce the DOF when averaging along that axis.
        """
        if self._dofreduction.get(axislabel, None) is None:
            return False
        else:
            return True

    def get_dof(self, label, adjusted=False):
        """
        Returns the number of degrees of freedom (DOF) for a power in the power spectrum specified by `label`.
        This is the DOF in the chi2 distribution that the powers are (approximately) distributed according to,
        under the null hypothesis of stable circuits.

        Parameters
        ----------
        label: tuple
            The label specifying power spectrum, or type of power spectrum

        adjusted: bool, optional
            Currently does nothing. Placeholder for future improvement

        Returns
        -------
        float

        """
        if not adjusted:
            dof = 1
            for i, axislabel in enumerate(self._axislabels):
                if axislabel not in label:
                    dofreduction = self._dofreduction.get(axislabel)
                    assert(dofreduction is not None), "Cannot obtain the DOF for this type of spectrum!"
                    dof = dof * (self._shape[i] - dofreduction)
        else:
            raise NotImplementedError("This has not yet been implemented!")

        return dof

    def get_number_of_spectra(self, label):
        """
        The number of power spectra in the "class" of spectra specified by `label`, where `label` is
        a tuple containing some subset of 'dataset', 'circuit' and 'outcome'. The strings it contains
        specified those axes that are not averaged, and those not contained are axes that are averaged.
        For example, ('circuit',) is the number of power spectra after averaging over DataSets (often 1) and
        outcomes (often 2), and it is just the total number of circuits.
        """
        numspectra = 1
        for i, axislabel in enumerate(self._axislabels):
            if axislabel in label:
                numspectra = numspectra * self._shape[i]

        return numspectra

    def same_frequencies(self, dictlabel={}):
        """
        Checks whether all the "base" power spectra defined by `dictlabel` are all with respect to the same frequencies.

        Parameters
        ----------
        dictlabel : dict, optional
            Specifies the set of "base" power spectra. Keys should be a subset of 'dataset', 'circuit' and 'outcome'.
            For each string included as a key, we restrict to only those power spectra with associated with the
            corresponding value. For example, if 'circuit' is a key the value should be a Circuit and we are looking
            at only those power spectra obtained from the data for that circuit. So an empty dict means we look at every
            base spectra

        Returns
        -------
        Bool
            True only if the spectra defined by `dictlabel` are all with respect to the same frequencies.

        """
        # If there's no frequency pointers stored it's automatically true, becuase then all spectra
        # are for the frequencies stored as self._frequencies[0].
        if len(self._freqpointers) == 0: return True

        iterator = []  # A list of list-like to iterate over to consider all the spectra in question.
        for i, axislabel in enumerate(self._axislabels):
            if axislabel in dictlabel.keys():
                iterator.append([dictlabel[axislabel], ])
            else:
                iterator.append(range(self._shape[i]))

        if 'circuit' in dictlabel.keys():
            circuitindex = self._index('circuit', dictlabel['circuit'])
        else:
            circuitindex = 0
        # Find the frequency pointer for an arbitrary one of the spectra in question, and return the default of
        # 0 if there isn't one.
        reference_freqpointer = self._freqpointers.get(circuitindex, 0)
        # Iterate through the indices to all the spectra under consideration, and check their frequency pointers
        # are the same as the reference.
        for indices in _itertools.product(*iterator):
            freqpointer = self._freqpointers.get(indices, 0)
            # If the frequency pointer is different to the reference then the frequencies aren't all the same.
            if freqpointer != reference_freqpointer:
                return False

        return True

    def averaging_allowed(self, dictlabel={}, checklevel=2):
        """
        Checks whether we can average over the specified "base" power spectra.

        Parameters
        ----------
        dictlabel : dict, optional
            Specifies the set of "base" power spectra. Keys should be a subset of 'dataset', 'circuit' and 'outcome'.
            For each string included as a key, we restrict to only those power spectra with associated with the
            corresponding value. For example, if 'circuit' is a key the value should be a Circuit and we are looking
            at only those power spectra obtained from the data for that circuit. So an empty dict means we look at every
            base spectra

        checklevel : int, optiona;
            The degree of checking.
                - 0:  the function is trivial, and returns True
                - 1:  the function checks that all the power spectra to be averaged are associated with the same
                      frequencies
                - 2+: checks that we can calculate the DOF for that averaged power spectrum, so that hypothesis
                      testing can be implemented on it.

        Returns
        -------
        Bool
            True if the power spectra pass the tests for the validity of averaging over them.

        """
        if checklevel == 0:  # Does no checking if `checklevel` is 0.
            return True
        if checklevel >= 1:
            # If `checklevel` >= 1 checks that the frequencies are all the same.
            if not self.same_frequencies(dictlabel):
                return False

            # If `checklevel` >= 2 checks that DOF reduction is not None for all
            # axes on which we are trying to average.
            if checklevel >= 2:
                for i, axislabel in enumerate(self._axislabels):
                    if axislabel not in dictlabel.keys():
                        # If dimension along axis is 1 averaging is trivial so we always allow.
                        if not self._check_dofreduction_set(axislabel) and self._shape[i] > 1:
                            return False

            return True

    def _index(self, axislabel, key):
        """
        Returns the spectra array index for a key along a specific axis. For example,
        `axislabel` could be 'circuit' and 'key' and Circuit for the DataSet, and this
        will return the index in spectra array to find the spectra for that circuit.
        """
        if axislabel == 'dataset':
            return list(self.data.keys()).index(key)
        elif axislabel == 'circuit':
            return list(self.data[self.data.keys()[0]].keys()).index(key)
        elif axislabel == 'outcome':
            return self.data.outcome_labels.index(key)
        else:
            raise ValueError("axislabel must be one of `dataset`, `circuit` and `outcome`!")

    def _dictlabel_to_averaging_axes_and_array_indices(self, dictlabel):
        """
        Returns the axes to average over, and the indices of the reduced array, for a spectrum
        labelled by the input dictionary

        Parameters
        ----------
        dictlabel: dict
            A dictionary labelling a spectrum, with keys are some subset of 'dataset', 'circuit'
            and 'outcome', and values that are valid options for those keys. E.g., for 'circuit'
            this is one of the Circuits in the stored DataSet.

        Returns
        -------
        tuple
            The axes to average over. This is the axes corresponding to each of 'dataset', 'circuit'
            and 'outcome' that aren't elements of the input dictionary

        tuple
            The indices for the spectrum after any averaging. This is the indices for the values
            of the keys.
        """
        indices = []
        averageaxes = []
        for i, axislabel in enumerate(self._axislabels):
            if axislabel in dictlabel.keys():
                indices.append(self._index(axislabel, dictlabel[axislabel]))
            else:
                averageaxes.append(i)

        return tuple(averageaxes), tuple(indices)

    def get_spectrum(self, dictlabel=None, returnfrequencies=True, checklevel=2):
        """
        Returns a power spectrum.

        Parameters
        ----------
        dictlabel : dict, optional
            A power spectra has been calculated for each (DataSet, Circuit, circuit) outcome
            triple of the stored data. We can average over any of these axes. The `dictlabel`
            specifies which axes to average over, and which value to specify for the parameters
            that we are not averaging over. The keys are a subset of  'dataset', 'circuit'
            and 'outcome'. For any of these strings not included, we average over that axis.
            For each string included, the item should be a key for one of the stored DataSets,
            a Circuit in the DataSet, and a possible outcome for that Circuit, respectively.
            If None then defaults to {}, corresponding to the power spectrum obtained by
            averarging over all the "base" spectra.

        returnfrequences: bool, optional
            Whether to also return the corresponding frequencies, as the first argument.

        checklevel : int, optional
            The level of checking to do, when averaging over axes, for whether averaging over
            that axis is a meaningful thing to do. If checklevel = 0 then no checks are performed.
            If checklevel > 0 then the function checks that all the power spectra to be averaged
            are associated with the same frequencies. If checklevel > 1 we can also check that
            we can calculate the DOF for that averaged power spectrum, so that hypothesis testing
            can be implemented on it.

        Returns
        -------
        if returnfrequencies:
            array
                The frequencies associated with the returned powers.

        array
            The power spectrum.

        """
        if dictlabel is None: dictlabel = {}
        assert(self._contains_spectra is not None), "Spectra must be generated before they can be accessed!"
        if len(dictlabel) == len(self._axislabels):
            arrayindices = self._tupletoindex[(dictlabel['dataset'], dictlabel['circuit'], dictlabel['outcome'])]
            spectrum = self._basespectra[arrayindices].copy()
            if returnfrequencies:
                circuitindex = arrayindices[1]
                freq = self._frequencies[self._freqpointers.get(circuitindex, 0)]
                return freq, spectrum
            else:
                return spectrum
        else:
            return self._get_averaged_spectrum(dictlabel, returnfrequencies, checklevel)

    def _get_averaged_spectrum(self, dictlabel={}, returnfrequencies=True, checklevel=2):
        """
        A subroutine of the method `get_spectrum()`. See the docstring of that method for details.

        """
        # Check whether the requested averaging is allowed, with a check at the specified rigour level.
        assert(self.averaging_allowed(dictlabel, checklevel=checklevel)), "This averaging is not permissable! To do it \
            anyway, reduce `checklevel`."
        # Find the axes we're averaging over, and the array indices for the averaged spectra array.
        axes, indices = self._dictlabel_to_averaging_axes_and_array_indices(dictlabel)
        # Performs the averaging a picks out the specified spectrum
        spectrum = _np.mean(self._basespectra, axis=axes)[indices]

        if not returnfrequencies:
            return spectrum

        else:
            # Pick an arbitrary bases spectrum index out of those spectra averaged to obtain the
            # spectrum we are returning.
            spectrumindex = []
            for axislabel in self._axislabels:
                spectrumindex.append(dictlabel.get(axislabel, 0))

            # Find the frequencies associated with that spectrum. Note that this set of frequencies may not
            # be meaningful if `checklevel` is 0.
            freq = self._frequencies[self._freqpointers.get(tuple(spectrumindex), 0)]

            return freq, spectrum

    def get_maxpower(self, dictlabel={}, freqsubset=None):
        """
        Returns the maximum power in a power spectrum.

        Parameters
        ----------
        dictlabel : dict, optional
            The dictionary labelling the spectrum. The same format as in the get_spectrum() method.
            See the docstring of that method for more details.

        freqsubset : list, optional
            A list of indices, that specify the subset of frequency indices to maximize over.

        Returns
        -------
        float
            The maximal power in the spectrum.
        """
        spectrum = self.get_spectrum(dictlabel)
        if freqsubset is None:
            maxpower = _np.max(spectrum)
        else:
            maxpower = _np.max(spectrum[freqsubset])

        return maxpower

    def get_pvalue(self, dictlabel={}, freqsubset=None, cutoff=0):
        """
        The p-value of the maximum power in a power spectrum.

        Parameters
        ----------
        dictlabel : dict, optional
            The dictionary labelling the spectrum. The same format as in the get_spectrum() method.
            See the docstring of that method for more details.

        freqsubset : list, optional
            A list of indices, that specify the subset of frequency indices to consider.

        cutoof : float, optional
            The minimal allowed p-value.

        Returns
        -------
        float
            The p-value of the maximal power in the specified spectrum.

        """
        maxpower = self.get_maxpower(dictlabel=dictlabel, freqsubset=freqsubset)
        # future: update adjusted to True when the function allows it.
        dof = self.get_dof(tuple(dictlabel.keys()), adjusted=False)
        pvalue = _sig.power_to_pvalue(maxpower, dof)
        pvalue = max(cutoff, pvalue)

        return pvalue

    def run_instability_detection(self, significance=0.05, freqstest=None, tests='auto', inclass_correction={},
                                  betweenclass_weighting='auto', saveas='default', default=True, overwrite=False,
                                  verbosity=1):
        """
        Runs instability detection, by performing statistical hypothesis tests on the power spectra generated
        from the time-series data. Before running this method it is necessary to generate power spectra using
        the compute_spectra() method.

        Parameters
        ----------
        significance : float, optional
            The global significance level. With defaults for all other inputs (a wide range of non-default options),
            the family-wise error rate of the set of all hypothesis tests performed is controlled to this value.

        freqstest : None or list, optional
            If not not None, a list of the frequency indices at which to test the powers. Leave as None to perform
            comprehensive testing of the power spectra.

        tests : 'auto' or tuple, optional
            Specifies the set of hypothesis tests to perform. If 'auto' then an set of tests is automatically
            chosen. This set of tests will be suitable for most purposes, but sometimes it is useful to override
            this. If a tuple, the elements are "test classes", that specifies a set of hypothesis tests to run,
            and each test class is itself specified by a tuple. The tests specified by each test class in this
            tuple are all implemented. A test class is a tuple containing some subset of 'dataset', 'circuit'
            and 'outcome', which specifies a set of power spectra. Specifically, a power spectra has been calculated
            for the clickstream for every combination of eachinput DataSet (e.g., there are multiple DataSets if there
            has been marginalization of multi-qubit data), each Circuit in the DataSet, and each possible outcome in
            the DataSet. For each of "dataset", "circuit" and "outcome" *not* included in a tuple defining a test class,
            the coresponding "axis" of the 3-dimensional array of spectra is averaged over, and these spectra are then
            tested. So the tuple () specifies the "test class" whereby we test the power spectrum obtained by averaging
            all power spectra; the tuple ('dataset','circuit') specifies the "test class" whereby we average  only over
            outcomes, obtaining a single power spectrum for each DataSet and Circuit combination, which we test.

            The default option for "tests" is appropriate for most circumstances, and it consists of (), ('dataset')
            and ('dataset', 'circuit') with duplicates removed (e.g., if there is a single DataSet then () is equivalent
            to ('dataset')).

        inclass_correction : dict, optional
            A dictionary with keys 'dataset', 'circuit', 'outcome' and 'spectrum', and values that specify the type of
            multi-test correction used to account for the multiple tests being implemented. This specifies how the
            statistically significance is maintained within the tests implemented in a single "test class".

        betweenclass_weighting : 'auto' or dict, optional
            The weighting to use to maintain statistical significance between the different classes of test being
            implemented. If 'auto' then a standard Bonferroni correction is used.

        default : bool, optional
            This method can be run multiple times, to implement independent instability detection runs (if you are
            doing this, make sure you know what you're doing! For example, deciding on-the-fly what hypothesis
            tests to run is statistically very dubious!). One of these is set as the default: unless specified otherwise
            its results are returned by all of the "get" methods. One the first call to this method, this is
            is ignored and it is always set to True.

        saveas : str, optional
            This string specifies the name under which the results of this run of instability detection is saved.
            If not implementing multiple calls to this function there is no need to change this from 'default'.

        overwrite : bool, optional
            If this method has already been called, and results saved under the string "saveas", this specifies
            whether to overwrite the old results or to raise an error.

        verbosity : int, optional
            The amount of print-to-screen.

        Returns
        -------
        None

        """
        if verbosity > 0: print("Running instability detection at {} significance...".format(significance), end='')
        if verbosity >= 1: print('\n')

        if not overwrite:
            assert(saveas not in self._driftdetectors)
        # A list storing the `saveas` keys of the drift detection results.
        self._driftdetectors.append(saveas)
        # future: correct this when significance is not be maintained?
        self._significance[saveas] = significance

        assert(self._basespectra is not None), "Spectra must be generated before drift detection can be implemented! \
            First run .compute_spectra()!"

        # If there is no default detection results saved yet, these are automatically set to the default results.
        if default or (self._def_detection is None):
            self._def_detection = saveas

        if isinstance(freqstest, str):
            assert(freqstest == 'all')
        self._freqstest[saveas] = freqstest

        # Check the input `tests` is valid, and then record them.
        if not isinstance(tests, tuple):
            assert(tests == 'auto'), "If not a tuple, must be 'auto'!"
            tests = compute_auto_tests(self._shape, ids=self._ids)

        check_valid_tests(tests)
        self._tests[saveas] = tests

        # Populates the unspecfied parts of the inclass_correction with auto values, and checks the result is valid.
        inclass_correction = populate_inclass_correction(inclass_correction)

        if isinstance(betweenclass_weighting, str):
            assert(betweenclass_weighting == 'auto'), "If a string, betweenclass_weighting must be a string!"
            betweenclass_weighting = compute_auto_betweenclass_weighting(tests)

        if isinstance(compute_auto_betweenclass_weighting, bool):
            betweenclass_weighting = compute_auto_betweenclass_weighting(betweenclass_weighting)

        # future: some sort of warning if FWER, or FDR, is not being controlled?

        # Remove duplicate tests (i.e., tests that are equivalent given the data structure) and condense the
        # significance weighting so that no significance is wasted.
        condtests, condbetweenclass_weighting = condense_tests(self._shape, tests, betweenclass_weighting)

        test_significance = {}
        for test in condtests:
            test_significance[test] = significance * condbetweenclass_weighting[test]

        self._condtests[saveas] = condtests
        self._condbetweenclass_weighting[saveas] = condbetweenclass_weighting
        self._test_significance[saveas] = test_significance

        # future: add back in our delete.
        # # Work out the type of test error rate control that the testing is enforcing.
        # control = 'gBonferroni'  # Global family-wise error rate.
        # if not betweenClassCorrection and len(testsUpdated) > 1:
        #     control = 'cBonferroni'  # If we're not correcting between classes,
        # for i in range(4):
        #     ctype = inclass_correction[i]
        #     if i == 3 or existsPerTest[i]:
        #         if ctype == 'none':
        #             control = 'none'
        #         elif control != 'none' and ctype == 'Benjamini-Hochberg':
        #             control = control[0] + 'Benjamini-Hochberg'
        # sumweightings = 0
        # for test in tests:
        #     sumweightings += betweenclass_weighting[test]

        # if not (abs(1-sumweightings) < 1e-10):
        #     _warnings.warn("The weightings do not sum to 1! False positives are not necessarily being controlled!")

        # return betweenclass_weighting

        if freqstest is None:
            freqstest = _np.arange(self._shape[3])
        else:
            # future: add this functionality in. Don't think there is any major reason it can't be included.
            raise NotImplementedError("Cannot currently test only a subset of the frequencies!")
            freqstest = _np.sort(freqstest)

        sigthreshold = {}
        driftfreqinds = {}

        driftdetected_global = False
        driftdetected_class = {}

        # Implement the statistical tests
        for test in condtests:

            sig = test_significance[test]
            dof = self.get_dof(test, adjusted=False)
            # The number of spectra held in `spectra`.
            numspectra = self.get_number_of_spectra(test)
            # The total number of powers to be tested
            numtests = len(freqstest) * numspectra
            driftdetected_class[test] = False
            driftfreqinds[test] = {}

            if verbosity > 1:
                print("   - Implementing statistical test at significance {} with structure {}".format(sig, test))
                print("      - In-class correction being used is {}".format(tuple(inclass_correction.values())))
                print("      - Number of tests in class is {}".format(numtests))
                #print("      - Testing at {} frequencies".format(numfreqs))
                print("      - Baseline chi2 degrees-of-freedom for the power spectrum is {}".format(dof))

            # Work out which axes of self._spectra we need to average over to obtain the relavent spectra for this test.
            axes = []
            if 'dataset' not in test:
                axes.append(0)
            if 'circuit' not in test:
                axes.append(1)
            if 'outcome' not in test:
                axes.append(2)
            # Performs the averaging to find the spectra that will be tested.
            spectra = _np.mean(self._basespectra, axis=tuple(axes))
            # Find out how we're doing the false-positives control for this test.
            test_inclass_correction = [inclass_correction[axislabel]
                                       for axislabel in test] + [inclass_correction['spectrum'], ]
            # If we are just doing Bonferroni corrections on everything, we using the following optimized code
            if all([correction == 'Bonferroni' for correction in test_inclass_correction]):

                sigthreshold[test] = _sig.power_significance_threshold(sig, numtests, dof)
                if verbosity > 0: print("      - Power significance threshold is: {}".format(sigthreshold[test]))

                # Finds the indices of the powers above the threshold in each spectrum in `spectra`.
                for indices in _np.ndindex(_np.shape(spectra)[:-1]):
                    driftindtuple = tuple(freqstest.copy()[spectra[indices] > sigthreshold[test]])
                    # We only store the tuple in the dict if it contains at least one index.
                    if len(driftindtuple) > 0:
                        driftfreqinds[test][indices] = driftindtuple

            # If we're doing the Benjamini-Hochberg procedure we go into this bit of code. The
            # Benjamini-Hockerg part has to be nested.
            else:
                assert(inclass_correction['spectrum'] == 'Benjamini-Hochberg'), "If not `Bonferroni, only \
                     'Benjamini-Hochberg correction is allowed!"

                numBon = 1
                # Bonferroni iterators (for outer iteration).
                iterBon = []
                # Benjamini-Hochberg iterators (for inner iteration).
                iterBenjHoch = []

                # Sets the iterators for "dataset", "circuit" and "outcome" (where relevant)
                for ind, axistype in enumerate(test):

                    numspectra_for_axis = self._shape[self._axislabels.index(axistype)]
                    correction = inclass_correction[axistype]
                    #numfreqs = len(freqstest) * numspectra

                    if correction == 'Bonferroni':
                        numBon = numBon * numspectra_for_axis
                        iterBon.append(range(numspectra))

                    elif correction == 'Benjamini-Hochberg':
                        iterBenjHoch.append(range(numspectra_for_axis))

                    else:
                        raise ValueError("Only corrections allowed currently are `Bonferroni` and `Benjamini-Hockberg`")

                # Sets the iterators for "spectrum", i.e., for the powers in a single spectrum. This *must* be Benj-Hoch
                iterBenjHoch.append(range(len(freqstest)))

                # The number of test statistics we are doing *each* Benjamini-Hockberg procedure for.
                numBenjHoch = numtests // numBon
                # The significance to input into each instance of the Benjamini-Hockberg procedure.
                localsig = sig / numBon
                #print(numtests, numBenjHoch, numBon, localsig)
                #print(iterBon)
                #print(iterBenjHoch)

                if verbosity > 1:
                    print(("      - Implementing {} Benjamini-Hockberg procedure statistical tests "
                           "each containing {} tests.".format(numBon, numBenjHoch)))
                    print(("      - Local statistical significance for each Benjamini-Hockberg "
                           "procedure is {}".format(localsig)))

                # Note that this is a "pseudo-threshold" with Benjamini-Hockberg in that it depends on the data.
                # todo : this bit of code is being over-ridden later? It should be, and this should be removed.
                if numBon > 1:
                    sigthreshold[test] = None
                else:
                    sigthreshold[test] = {}

                if verbosity > 1:
                    print("      - Generating Benjamini-Hochberg power quasi-threshold...", end='')

                quasithreshold = _sig.power_significance_quasithreshold(localsig, numBenjHoch, dof,
                                                                        procedure='Benjamini-Hochberg')

                # We'll record a pseudo-treshold for each Benjamini-Hochberg procedure, in a dict.
                sigthreshold[test] = {}

                if verbosity > 1:
                    print("complete.")
                    print("      - Implementing the Benjamini-Hochberg procedures...", end='')
                    if verbosity > 1:
                        print('')
                        if verbosity > 2:
                            print('')

                # Goes through each Benjamini-Hochberg set, and implements the Benjamini-Hockberg procedure.
                for indices in _itertools.product(*iterBon):

                    # If we're not testing a single spectrum we need to flatten the >1D array.
                    if len(_np.shape(spectra)) > 1:
                        powerlist = spectra[indices].flatten()
                    # If we're testing a single spectrum, we can just copy the 1D array.
                    else: powerlist = spectra[indices].copy()

                    # The indices that will go with the elements in the flattened spectra.
                    powerindices = [tup for tup in _itertools.product(*iterBenjHoch)]

                    # Benjamini-Hochberg works by first ordering the powers from lowest to highest.
                    powerlist, powerindices = zip(*sorted(zip(powerlist, powerindices)))
                    powerlist = _np.array(list(powerlist))

                    # It then compares the ordered powers to the quasithreshold.
                    dif = powerlist - quasithreshold

                    # This will raise an error if there are no powers that are significant.
                    try:
                        # All powers beyond the first lowest-to-highest power that is above the quasithreshold
                        # are statistically significant. This is the index of that power in powerlist.
                        threshind = next(i for i, v in enumerate(dif) if v > 0.0)
                        # This is the indices for all significant powers
                        powerindices = powerindices[threshind:]

                        if verbosity > 2:
                            print("         - {} significant ".format(numBenjHoch - threshind)
                                  + "frequencies found for test-set {}!".format(indices))

                        for sigpowerind in powerindices:
                            # We append the BenjHock indices -- with the freq index removed -- to the `indices`,
                            # which is the indices of the part of the test that we are Bonferroni correcting (it is
                            # it is set in the Bonferroni level iterator).
                            spectraindex = tuple(list(indices) + list(sigpowerind[:-1]))
                            # Record the frequency index that goes with this spectra index, by saving it in a tuple.
                            if spectraindex in driftfreqinds[test].keys():
                                driftfreqinds[test][spectraindex] = tuple(
                                    list(driftfreqinds[test][spectraindex]).append(sigpowerind[-1]))
                            else:
                                driftfreqinds[test][spectraindex] = (sigpowerind[-1],)

                        # We record the pseudo-threshold in a dict.
                        sigthreshold[test][indices] = quasithreshold[threshind]

                    # If there are no powers above the quasithreshold, the next() fails in the `try`, so we enter here.
                    except:
                        # We do not need to record anything in driftfreqinds[test], as no entry means there were
                        # no significant frequencies
                        if verbosity > 2:
                            print("         - 0 significant frequencies found for test-set {}.".format(indices))

                        # We record the pseudo-threshold in a dict.
                        sigthreshold[test][indices] = quasithreshold[-1]

            if len(driftfreqinds[test]) > 0:
                driftdetected_class[test] = True
                driftdetected_global = True

            if verbosity > 1:
                if verbosity > 2:
                    print('')
                if driftdetected_class[test]:
                    print("      - Instability detected!\n")
                else:
                    print("      - Instability *not* detected.\n")

        self._freqstest[saveas] = freqstest
        self._power_sigthreshold[saveas] = sigthreshold
        self._driftfreqinds[saveas] = driftfreqinds
        self._driftdetected_global[saveas] = driftdetected_global
        self._driftdetected_class[saveas] = driftdetected_class

        if verbosity == 1:
            if driftdetected_global:
                print("      - Instability detected!")
            else:
                print("      - Instability *not* detected.")

        if verbosity == 1: print("done!")
        if verbosity > 1: print("Instability detection complete!")
        return None

    def get_statistical_significance(self, detectorkey=None):
        """
        The statistical significance level of the instability detection

        Parameters
        ----------
        detectorkey : None or string, optional
            Only relevant if more than one set of instability detection was run. The "saveas" key that
            was used when running run_instability_detection() for the detection results that you wish
            to access.

        Returns
        -------
        float
            The statistical significance level of the instability detection
        """
        if detectorkey is None: detectorkey = self._def_detection

        return self._significance[detectorkey]

    def _equivalent_implemented_test(self, test, detectorkey=None):
        """
        Finds an equivalent test that was implemented. For example, if () was implemented, the input
        is ('dataset'), and self.data contains only 1 DataSet, then () will be returned -- because then
        averaging or not averaging over DataSets is trivial, and so () and ('dataset') are equivalent
        tests. If there is no equivalent test, then None is returned.

        """
        if detectorkey is None: detectorkey = self._def_detection
        # We find the "condensed" test, which is will have been run.
        equivtest, junk = condense_tests(self._shape, [test, ], None)
        # That function returns a list of the condensed tests, and we need the (single) test tuple itself.
        equivtest = equivtest[0]
        # If this condensed test was implemented we return it
        if equivtest in self._condtests[detectorkey]: return equivtest
        # Otherwise we return None, to signify that there was no equivalent test implemented.
        else: return None

    def get_unstable_circuits(self, getmaxtvd=False, detectorkey=None, fromtests='auto', freqindices=False,
                              estimatekey=None, dskey=None):
        """
        Returns a dictionary, containing all the circuits that instability has been detected for as keys,
        with the values being the frequencies of the detected instability.

        Parameters
        ----------
        getmaxtvd: bool, optional
            Whether to also return the bound on the TVD deviation, as given by the get_max_tvd_bound() method.
            If True, then the values of the returned dictionary are a 2-element list: the first element is
            the frequencies detected for the unstable circuits, and the second element is this TVD bound.

        detectorkey : None or string, optional
            Only relevant if more than one set of instability detection was run. The "saveas" key that
            was used when running run_instability_detection() for the detection results that you wish
            to access.

        fromtests : str or tuple, optional
            The test results to use when deciding what circuits are unstable. If a tuple, it should
            be a subset of the "test classes" run, each specified by a tuple. If "auto", then all
            tests that individually look at the data from each circuit are included. This includes
            every test implemented of the null hypothesis that that circuit is stable.

        freqindices : bool, optional
            If True then the frequencies are returned as a list of integers, which are the indices
            in the frequency list corresponding to the power spectrum for that circuit. If False then
            the frequencies are returned in units of 1/t where 't' is the units of the time stamps.

        estimatekey : None or tuple, optional
            Only relevant if `getmaxtvd` is True. The name of the estimate of the probability trajectories
            to use for calculating the TVD bound.

        dskey : None or string, optional
            The DataSet to return the results for. Need if more than one DataSet has been analyzed,
            but otherwise can be left as None.

        Returns
        -------
        dict
            A dictionary of the unstable circuits.

        """
        # If we're not given a detectorkey, we default to the standard detection results.
        if detectorkey is None: detectorkey = self._def_detection

        if isinstance(fromtests, str):
            assert(fromtests == 'auto')
            validtests = compute_valid_tests()
            fromtests = []
            for test in validtests:
                if 'circuit' in test:
                    equivtest = self._equivalent_implemented_test(test, detectorkey)
                    if equivtest is not None:
                        if equivtest not in fromtests:
                            fromtests.append(equivtest)
            assert(len(fromtests) > 0), "No tests were implemented on per-circuit power spectra, so we cannot " + \
                "identify which circuits are unstable!"

        else:
            # This replaces `fromtests` with a list of equivalent tests that have explcitiyl beenn done.
            tests = []
            for test in fromtests:
                equivtest = self._equivalent_implemented_test(test, detectorkey)
                assert(equivtest is not None), "The test requested (or an equivalent test) has not been implemented!"
                if equivtest not in tests:
                    tests.append(equivtest)
            fromtests = tests

        circuits = {}
        # Goes through each test in fromtests, and collates all the circuit with drift.
        for test in fromtests:

            if dskey is not None:
                assert(len(self.data.keys()) == 1 or dskey == test['dataset'])
            # If 'circuit' is in test, then we assign different drift frequencies to each circuit. Frequency indices
            # added in this loop are "true" hypothesis test results (todo: explain). (except when the dataset
            # contains only one circuit, in which case we go into the loop below).
            if 'circuit' in test:

                indexforcircuitindex = test.index('circuit')
                for spectrumindex, freqindices in self._driftfreqinds[detectorkey][test].items():
                    circuitindex = spectrumindex[indexforcircuitindex]
                    circuit = list(self.data[self.data.keys()[0]].keys())[circuitindex]
                    # If the circuit is already a key, we append the new indices to the list.
                    if circuit in circuits.keys():
                        for ind in freqindices:
                            if ind not in circuits[circuit]:
                                circuits[circuit].append(ind)
                    # if the circuit is not already a key, we set it's value as this list.
                    else:
                        circuits[circuit] = list(freqindices)

            # Otherwise, every circuit gets assigned the drift frequencies. (This doesn't make much sense when
            # frequencies disagree).
            else:
                # Create a list of all the significant frequency indices
                freqindlist = []
                for freqindices in self._driftfreqinds[detectorkey][test].values():
                    for freqin in freqindices:
                        if freqin not in freqindlist:
                            freqindlist.append(freqin)
                # If it's a nontrivial list, we add it to the frequencies already recorded for each circuit.
                if len(freqindlist) > 0:
                    for circuit in self.data[list(self.data.keys())[0]].keys():
                        # If the circuit is already a key, we append the new indices to the list.
                        if circuit in circuits.keys():
                            for ind in freqindlist:
                                if ind not in circuits[circuit]:
                                    circuits[circuit].append(ind)
                        # if the circuit is not already a key, we set it's value as this list.
                        else:
                            circuits[circuit] = list(freqindices)

        if not freqindices:
            for circuit, freqindices in circuits.items():
                freqs = self._frequencies[self._freqpointers.get(self._index('circuit', circuit), 0)][freqindices]
                circuits[circuit] = freqs

        if getmaxtvd:
            if dskey is None:
                assert(len(self.data.keys()) == 1)
                dskey = list(self.data.keys())[0]
            if estimatekey is None:
                estimatekey = self._def_probtrajectories
            for circuit in self.data[dskey].keys():
                maxtvd = self.get_max_tvd_bound(circuit, dskey=dskey, estimatekey=estimatekey, estimator=None)
                if circuit in circuits.keys():
                    circuits[circuit] = (circuits[circuit], maxtvd)
                else:
                    if maxtvd > 0:
                        circuits[circuit] = ([], maxtvd)

        return circuits

    def get_instability_indices(self, dictlabel={}, detectorkey=None):
        """
        Returns the frequency indices that instability has been detected at in the specified
        power spectrum

        Parameters
        ----------
        dictlabel: dict, optional
            The label for the spectrum to extract the instability frequency indices for.

        detectorkey : None or string, optional
            Only relevant if more than one set of instability detection was run. The "saveas" key that
            was used when running run_instability_detection() for the detection results that you wish
            to access.

        Returns
        -------
        list
            The instability frequency indices.

        """
        # If we're not given a detectorkey, we default to the standard detection results.
        if detectorkey is None: detectorkey = self._def_detection

        # The test we're looking at the results of.
        test = tuple(dictlabel.keys())
        # The test we actually did that equivalent to this test.
        condtest = self._equivalent_implemented_test(test, detectorkey)
        assert(condtest is not None), "A test of this sort -- or an equivalent test -- has not been implemented!"

        # Find the array index. This is the values of `dictlabel`, but correctly ordered and with any elements that are
        # dropped from `condtest` (because that axislabel is to a length-1 axis) not included.
        arrayindex = []
        for axislabel in self._axislabels:
            if axislabel in condtest:
                arrayindex.append(self._index(axislabel, dictlabel[axislabel]))
        arrayindex = tuple(arrayindex)

        # Get the drift frequency indices. It's not stored if there are none, so we get the empty tuple.
        driftfreqinds = _copy.copy(self._driftfreqinds[detectorkey][condtest].get(arrayindex, ()))

        return driftfreqinds

    def get_instability_frequencies(self, dictlabel={}, detectorkey=None):
        """
        Returns the frequencies that instability has been detected at in the specified power spectrum.
        These frequencies are given in units of 1/t where 't' is the unit of the time stamps.

        Parameters
        ----------
        dictlabel: dict, optional
            The label for the spectrum to extract the instability frequencyies for.

        detectorkey : None or string, optional
            Only relevant if more than one set of instability detection was run. The "saveas" key that
            was used when running run_instability_detection() for the detection results that you wish
            to access.

        Returns
        -------
        list
            The instability frequencies

        """
        # If we're not given a detectorkey, we default to the standard detection results.
        if detectorkey is None: detectorkey = self._def_detection
        # Gets the drift indices, that we then jut need to convert to frequencies.
        freqind = self.get_instability_indices(dictlabel=dictlabel, detectorkey=detectorkey)
        # If this is for a particular circuit, find the circuit index for that circuit.
        if 'circuit' in dictlabel.keys():
            circuitindex = self._index('circuit', dictlabel['circuit'])
        # Otherwise set the circuit index to 0 (an arbitrary value)
        else:
            circuitindex = 0
        # Get the pointer to the frequencies for this circuit index
        freqpointer = self._freqpointers.get(circuitindex, 0)
        # Get the frequencies.
        driftfreqs = self._frequencies[freqpointer][list(freqind)]

        return driftfreqs

    def get_power_threshold(self, test, detectorkey=None):
        """
        The statistical significance threshold for any power spectrum in the set specified by the tuple `test`.

        test
            A tuple specifying the class of power spectra, to extract the power threshold for. Contains
            some subset of 'dataset', 'circuit' and 'outcome', and it should correspond to a test that
            was implemented.

        detectorkey : None or string, optional
            Only relevant if more than one set of instability detection was run. The "saveas" key that
            was used when running run_instability_detection() for the detection results that you wish
            to access.

        Returns
        -------
        float
            The power threshold.

        string
            The type of threshold. This is 'true' if it is true threshold, in the sense that
            it is data-independent and can be precalculated without the data. This is the sort
            of threshold one obtains when doing a Bonferroni correction. This is 'pseudo' if it
            is a threshold obtained from a p-value ordering procedure, like the Holms' method or
            the Benjamini-Hochberg procedure, because it is not a true threshold in the sane that
            it depends on the data. This is 'maxpseudo' if multiple independent p-value ordering
            procedures have been performed with the test class specified by `test`. In this case,
            all the powers that have been deemed to be statistically significant are not necessarily
            above this threshold.

        """
        # If we're not given a detectorkey, we default to the standard detection results.
        if detectorkey is None: detectorkey = self._def_detection
        # The test we actually did that equivalent to this test.
        condtest = self._equivalent_implemented_test(test, detectorkey)
        assert(condtest is not None), "A test of this sort -- or an equivalent test -- has not been implemented!" \
            + "To create an ad-hoc post-fact threshold use the functions in drift.signal"

        thresholdset = self._power_sigthreshold[detectorkey][condtest]
        # If it's a float, it's a "true" threshold, so we set the threshold to this.
        if isinstance(thresholdset, float):
            threshold = thresholdset
            thresholdtype = 'true'
        # Otherwise it's a dict, and it's either a single pseudo-threshold or a set of pseudo-threholds.
        else:
            thresholdset = list(thresholdset.values())
            # We return the largest pseudo-threshold, as this is a threshold for all cases.
            threshold = max(thresholdset)
            if len(thresholdset) == 1:
                thresholdtype = 'pseudo'
            else:
                thresholdtype = 'maxpseudo'

        return threshold, thresholdtype

    def get_pvalue_threshold(self, test, detectorkey=None):
        """
        The statistical significance threshold for any p-value of a power in the power spectra
        set specified by the tuple `test`.

        test
            A tuple specifying the class of power spectra, to extract the power threshold for. Contains
            some subset of 'dataset', 'circuit' and 'outcome', and it should correspond to a test that
            was implemented.

        detectorkey : None or string, optional
            Only relevant if more than one set of instability detection was run. The "saveas" key that
            was used when running run_instability_detection() for the detection results that you wish
            to access.

        Returns
        -------
        float
            The power threshold.

        string
            The type of threshold. This is 'true' if it is true threshold, in the sense that
            it is data-independent and can be precalculated without the data. This is the sort
            of threshold one obtains when doing a Bonferroni correction. This is 'pseudo' if it
            is a threshold obtained from a p-value ordering procedure, like the Holms' method or
            the Benjamini-Hochberg procedure, because it is not a true threshold in the sane that
            it depends on the data. This is 'maxpseudo' if multiple independent p-value ordering
            procedures have been performed with the test class specified by `test`. In this case,
            all the powers that have been deemed to be statistically significant are not necessarily
            above this threshold.

        """
        powerthreshold, thresholdtype = self.get_power_threshold(test, detectorkey=detectorkey)
        # future: update adjusted to True when the function allows it.
        dof = self.get_dof(test, adjusted=False)
        pvaluethreshold = _sig.power_to_pvalue(powerthreshold, dof)

        return pvaluethreshold, thresholdtype

    def instability_detected(self, detectorkey=None, test=None):
        """
        Whether instability was detected.

        Parameters
        ----------
        detectorkey : None or string, optional
            Only relevant if more than one set of instability detection was run. The "saveas" key that
            was used when running run_instability_detection() for the detection results that you wish
            to access.

        test : None or tuple, optional
            If None, then this method returns True if instability was detected in *any* statistical
            hypothesis test that we implemented. If a tuple, then it should be a tuple specifying
            a class of power spectra, so it contains some subset of 'dataset', 'circuit' and 'outcome',
            and it should correspond to a test that was implemented. If this tuple is specified, then
            this method returns True iff the tests specified by this tuple detected instability.

        Returns
        -------
        bool
            True if instability was detected, and returns False otherwise

        """
        # If we're not given a detectorkey, we default to the standard detection results.
        if detectorkey is None: detectorkey = self._def_detection
        if test is None:
            return self._driftdetected_global[detectorkey]
        else:
            return self._driftdetected_class[detectorkey][test]

    def run_instability_characterization(self, estimator='auto', modelselector=(None, None), default=True, verbosity=1):
        """
        Run instability characterization: estimates probability trajectories for every circuit. The estimation methods
        are based on model selection from the results of hypothesis testing, so it is is necessary to first perform this
        hypothesis testing by running the run_instability_detection method.

        Parameters
        ----------
        estimator : str, optional
            The name of the estimator to use. This is the method used to estimate the parameters of a parameterized
            model for each probability trajectory, after that parameterized model has been selected with the model
            selection methods. Allowed values are:

                - 'auto'. The estimation method is chosen automatically, default to the fast method that is also
                    reasonably reliable.

                - 'filter'. Performs a type of signal filtering: implements the transform used for generating power
                    spectra (e.g., the DCT), sets the amplitudes to zero for all freuquencies that the model selection
                    has not included in the model, inverts the transform, and then performs some minor post-processing
                    to guarantee probabilities within [0, 1]. This method is less statically well-founded than 'mle',
                    but it is faster and typically gives similar results. This method is not an option for
                    non-invertable transforms, such as the Lomb-Scargle periodogram.

                - 'mle'. Implements maximum likelihood estimation, on the parameterized model chosen by the model
                    selection. The most statistically well-founded option, but can be slower than 'filter' and relies
                    on numerical optimization.

        modelselection : tuple, optional
            The model selection method. If not None, the first element of the tuple is a string that is a "detectorkey",
            i.e., the `saveas` string for a set of instability detection results. If None then the default instability
            detection results are used. If run_instability_detection() has only been called once then there is only one
            set of results and there is no reason to set this to anything over than None. This is the instability
            detection resutls that will be used to select the models for the probability trajectories. If not None,
            the second element of the tuple is a "test class" tuple, specifying which test results to use to decide
            which frequencies are significant for each circuit. This can be typically set to None, and it will be
            chosen automatically. But if you wish to use specific test results for the model selection then this
            should be set.

        default : bool, optional
            This method can be called multiple times. This sets whether these results will be the default results
            used when probability trajectory estimates are later requested.

        verbosity : int, optional
            The amount of print-to-screen

        Returns
        -------
        None

        """
        if estimator == 'auto':
            estimator = compute_auto_estimator(self.transform)

        if self.transform == 'dct':
            assert(estimator in ('filter', 'mle')
                   ), "For the {} transform, the estimator must be `filter` of `mle`".format(self.transform)
        else:
            raise ValueError("Probability trajectory estimation is only currently possible with the DCT!")

        # Finds the default detector key.
        detectorkey = modelselector[0]
        if detectorkey is None:
            detectorkey = self._def_detection
        assert(detectorkey is not None), "There has been no instability detection performed, so cannot yet " \
            + "implement characterization! First run .run_instability_detection()"
        assert(detectorkey in self._driftdetectors), "There is no instability detection results with this key!"

        # Finds the default tests, if there is an acceptabe default tests.
        test = modelselector[1]
        if test is None:
            test = ('dataset', 'circuit')
            test = self._equivalent_implemented_test(test, detectorkey=detectorkey)
            assert(test is not None), "There has not been a test implemented that is equivalent to an acceptable" \
                + " default! Note that there *are* often other reasonable choices for `test` (the second element" \
                + " of the `modelselector` tuple) but they have more complicated statistical interpretations and" \
                + " so they are not permitted to happen without being explicitly specified."

        if (self._def_probtrajectories is None) or (default is True):
            self._def_probtrajectories = (detectorkey, test, estimator)

        dskeys = list(self.data.keys())
        circuits = self.data[dskeys[0]].keys()
        outcomes = self.data.outcome_labels

        for i, dskey in enumerate(dskeys):
            for j, circuit in enumerate(circuits):

                self._probtrajectories[i, j] = {}
                if verbosity > 1:
                    print("    - Generating estimates for dataset {} and circuit {}".format(dskey, circuit.str))

                # The most likely null hypothesis model, i.e., constant probabilities that are the observed frequencies.
                counts = self.data[dskey][circuit].counts
                total = self.data[dskey][circuit].total
                means = {o: counts.get(o, 0) / total for o in outcomes}
                nullptraj = _ptraj.ConstantProbTrajectory(outcomes, means)
                self._probtrajectories[i, j]['null'] = nullptraj

                # The hyper-parameters for the DCT models are frequency indices, along with the start and end times.
                if self.transform == 'dct':

                    # Get the required `dictlabel` for finding the drift frequencies.
                    dictlabel = {}
                    if 'dataset' in test: dictlabel['dataset'] = dskey
                    if 'circuit' in test: dictlabel['circuit'] = circuit
                    # Get the drift frequencies (doesn't include the zero frequency)
                    freqs = self.get_instability_indices(dictlabel, detectorkey=detectorkey)
                    # Add in the zero frequency, as it's a hyperparameter of the model
                    freqs = list(freqs)
                    freqs.insert(0, 0)
                    # If there is more than just the DC mode there is something non-trivial to do.
                    if len(freqs) > 0:

                        times, clickstreams = self.data[dskey][circuit].timeseries_for_outcomes
                        parameters = _sig.amplitudes_at_frequencies(freqs, clickstreams, transform=self.transform)
                        del parameters[outcomes[-1]]
                        # Divide by the counts
                        parameters = {key: list(_np.array(x) / self.counts) for key, x in parameters.items()}
                        # future: maybe these could be chosen in a better way for non-equally spaced data.
                        starttime = times[0]
                        timestep = _np.mean(_np.diff(times))
                        numtimes = len(times)
                        # Creates the "raw" filter model, where we've set all non-sig frequencies to zero.
                        filterptraj = _ptraj.CosineProbTrajectory(outcomes, freqs, parameters, starttime=starttime,
                                                                  timestep=timestep, numtimes=numtimes)
                        # Converts to the "damped" estimator, where amplitudes are reduced to guarantee valid prob.
                        filterptraj, flag = _ptraj.amplitude_compression(filterptraj, times)
                        # Records this estimate.
                        self._probtrajectories[i, j][detectorkey, test, 'filter'] = filterptraj

                        if estimator == 'mle':
                            maxlptraj = _ptraj.maxlikelihood(filterptraj, clickstreams, times, verbosity=verbosity - 1)
                            self._probtrajectories[i, j][detectorkey, test, 'mle'] = maxlptraj

                    # If it's just the DC mode, all estimators are equal to the null estimator.
                    else:
                        self._probtrajectories[i, j][detectorkey, test, 'filter'] = nullptraj
                        self._probtrajectories[i, j][detectorkey, test, 'mle'] = nullptraj

                else:
                    raise ValueError("Estimators for the {} transform are not yet implemented!".format(self.transform))

        return None

    def get_probability_trajectory_model(self, circuit, dskey=None, estimatekey=None, estimator=None):
        """
        Returns the probability trajectory for a circuit, in the form of a ProbTrajectory object.

        Parameters
        ----------
        circuit : Circuit
            The circuit to return the probability trajectories for.

        dskey : None or string, optional
            The DataSet to return the probability trajectories for. Need if more than one DataSet has
            been analyzed, but otherwise can be left as None.

        estimatekey : None or tuple, optional
            The estimate to return (typically, multiple estimates have been generated). If None, then the
            default estimate is returned. If run_instability_characterization() has been called only
            once then it is the estimate obtained by the method specified in that single call (but
            multiple estimates may have been recorded, and so are accessable, as a side-product of
            creating that estimate). If not None, a tuple where the first element is the `modelselector`
            and the second element is the `estimator`, as specified as arguments to the
            run_instability_characterization() method.

        estimator : None or string, optional
            Override for the second element of estimatekey', to easily extract the 'filter' and
            'mle' estimates, if both have been created (if 'mle' was chosen then the 'filter' estimate
            is also created as a by-product).

        Returns
        -------
        ProbTrajectory
            The estimated probability trajectory for the specified circuit.

        """
        if dskey is None:
            assert(len(self.data.keys()) == 1), \
                "There are multiple datasets, so need a dataset key, as the input `dskey`!"
            dskey = list(self.data.keys())[0]

        # Find the index for this dataset, circuit, and an arbitrary outcome.
        tup = self._tupletoindex[(dskey, circuit, self.data.outcome_labels[0])]
        dsind = tup[0]
        circind = tup[1]

        # If we're not given an estimatekey, we use the default key.
        if estimatekey is None:
            estimatekey = self._def_probtrajectories
        assert(estimatekey is not None), "There are no probability trajector estimates to get! " \
            "First must run .run_instability_characterization()."

        # If we're given an estimator name, we override that part of the `estimatekey`.
        if estimator is not None:
            estimatekey = (estimatekey[0], estimatekey[1], estimator)

        ptraj = self._probtrajectories[dsind, circind][estimatekey].copy()

        return ptraj

    def get_probability_trajectory(self, circuit, times, dskey=None, estimatekey=None, estimator=None):
        """
        Returns the probability trajectory for a circuit. See also get_probability_trajectory_model(),
        which provides are more complex, but more general-purpose, output.

        Parameters
        ----------
        circuit : Circuit
            The circuit to return the probability trajectories for.

        times : list
            The times to obtain the probabilities for.

        dskey : None or string, optional
            The DataSet to return the probability trajectories for. Need if more than one DataSet has
            been analyzed, but otherwise can be left as None.

        estimatekey : None or tuple, optional
            The estimate to return (typically, multiple estimates have been generated). If None, then the
            default estimate is returned. If run_instability_characterization() has been called only
            once then it is the estimate obtained by the method specified in that single call (but
            multiple estimates may have been recorded, and so are accessable, as a side-product of
            creating that estimate). If not None, a tuple where the first element is the `modelselector`
            and the second element is the `estimator`, as specified as arguments to the
            run_instability_characterization() method.

        estimator : None or string, optional
            Override for the second element of `estimatekey`, to easily extract the 'filter' and
            'mle' estimates, if both have been created (if 'mle' was chosen then the 'filter' estimate
            is also created as a by-product).

        Returns
        -------
        dict
            The probability trajectory at the specified times. The keys are the possible circuit outcomes
            and the value for an outcome is a list, which is the probability trajectory for that outcome.

        """
        ptraj = self.get_probability_trajectory_model(circuit, dskey, estimatekey, estimator)
        probabilities = ptraj.get_probabilities(times)

        return probabilities

    def get_max_tvd_bound(self, circuit, dskey=None, estimatekey=None, estimator=None):
        """
        Summarizes the size of the detected instability in the input circuit, as half the sum of the absolute values
        of the amplitudes in front of the non-constant basis functions in the estimate of the probability trajectories
        for that circuit.

        This is an upper bound on the maximum TVD between the instaneous probability distribution (over circuit
        outcomes) and the mean of this time-varying probability distribution, with this maximization over all times.

        Parameters
        ----------
        circuit : Circuit
            The circuit to estimate the size of the instability for.

        dskey : None or string, optional
            The DataSet to estimate the size of the instability for. Need if more than one DataSet has
            been analyzed, but otherwise can be left as None.

        estimatekey : None or tuple, optional
            The probability trajectory estimate to use. If None, then the default estimate is used.

        estimator : None or string, optional
            Overrides the second element of `estimatekey`, to easily select the 'filter' and 'mle' estimates.

        Returns
        -------
        float
            The size of the instability in the circuit, as quantified by the amplitudes in the probability trajectory
            model.
        """
        ptraj = self.get_probability_trajectory_model(circuit, dskey, estimatekey, estimator)
        params = ptraj.get_parameters()
        final_out_amplitudes = _np.zeros(len(ptraj.hyperparameters))
        summed_abs_amps = 0

        for o in params:
            final_out_amplitudes += params[o]
            summed_abs_amps += _np.sum(_np.abs(params[o][1:]))

        summed_abs_amps += _np.sum(_np.abs(final_out_amplitudes[1:]))
        maxtvd = 0.5 * summed_abs_amps

        return maxtvd

    def get_maxmax_tvd_bound(self, dskey=None, estimatekey=None, estimator=None):
        """
        The quantity returned by `get_max_tvd_bound` maximized over circuits. See the docstring of that method
        for more details.

        """
        if dskey is None:
            assert(len(self.data.keys()) == 1), \
                "There are multiple datasets, so need a dataset key, as the input `dskey`!"
            dskey = list(self.data.keys())[0]

        maxtvds = []
        for circuit in self.data[dskey].keys():
            maxtvds.append(self.get_max_tvd_bound(circuit, dskey=dskey, estimatekey=estimatekey, estimator=estimator))

        maxmaxtvd = _np.max(maxtvds)

        return maxmaxtvd
