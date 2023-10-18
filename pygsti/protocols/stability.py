"""
Stability analysis protocol objects
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti.protocols import protocol as _proto


class StabilityAnalysisDesign(_proto.ExperimentDesign):
    """
    Experimental design for stability analysis.

    Parameters
    ----------
    circuits : list
        The list of circuits to perform the stability analysis on. These
        can be anything.

    qubit_labels : tuple or "multiple", optional
        The qubits that this experiment design applies to.  These should also
        be the line labels of `circuits`.
    """

    def __init__(self, circuits, qubit_labels=None):
        self.needs_timestamps = True
        super().__init__(circuits, qubit_labels=qubit_labels)


class StabilityAnalysis(_proto.Protocol):
    """
    Stability Analysis protocol

    Parameters
    ----------
    ds : DataSet or MultiDataSet
        A DataSet containing time-series data to be analyzed for signs of instability.

    significance : float, optional
        The global significance level. With defaults for all other inputs (a wide range of non-default options),
        the family-wise error rate of the set of all hypothesis tests performed is controlled to this value.

    transform : str, optional
        The type of transform to use in the spectral analysis. Options are:

        * 'auto':   An attempt is made to choose the best transform given the "meta-data" of the data,
                    e.g., the variability in the time-step between data points. For beginners,
                    'auto' is the best option. If you are familiar with the underlying methods, the
                    meta-data of the input, and the relative merits of the different transform, then
                    it is probably better to choose this yourself -- as the auto-selection is not hugely
                    sophisticated.

        * 'dct' :   The Type-II Discrete Cosine Transform (with an orthogonal normalization). This is
                    the only tested option, and it is our recommended option when the data is
                    approximately equally-spaced, i.e., the time-step between each "click" for each
                    circuit is almost a constant. (the DCT transform implicitly assumes that this
                    time-step is exactly constant)

        * 'dft' :   The discrete Fourier transform (with an orthogonal normalization). 
                    **This is an experimental feature, and the results are unreliable with this transform**

        * 'lsp' :   The Lomb-Scargle periodogram.  
                    **This is an experimental feature, and the code is untested with this transform**

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

    estimator : str, optional
        The name of the estimator to use. This is the method used to estimate the parameters of a parameterized
        model for each probability trajectory, after that parameterized model has been selected with the model
        selection methods. Allowed values are:

        * 'auto'. The estimation method is chosen automatically, default to the fast method that is also
            reasonably reliable.
        * 'filter'. Performs a type of signal filtering: implements the transform used for generating power
            spectra (e.g., the DCT), sets the amplitudes to zero for all freuquencies that the model selection
            has not included in the model, inverts the transform, and then performs some minor post-processing
            to guarantee probabilities within [0, 1]. This method is less statically well-founded than 'mle',
            but it is faster and typically gives similar results. This method is not an option for
            non-invertable transforms, such as the Lomb-Scargle periodogram.
        * 'mle'. Implements maximum likelihood estimation, on the parameterized model chosen by the model
            selection. The most statistically well-founded option, but can be slower than 'filter' and relies
            on numerical optimization.

    modelselector : tuple, optional
        The model selection method. If not None, a "test class" tuple, specifying which test results to use to
        decide which frequencies are significant for each circuit, to then construct a parameterized model for
        each probability trajectory. This can be typically set to None, and it will be chosen automatically.
        But if you wish to use specific test results for the model selection then this should be set.

    verbosity : int, optional
        The amount of print-to-screen

    name : str, optional
        The name of this protocol, also used to (by default) name the
        results produced by this protocol.  If None, the class name will
        be used.
    """

    def __init__(self, significance=0.05, transform='auto', marginalize='auto', mergeoutcomes=None,
                 constnumtimes='auto', ids=False, frequencies='auto', freqpointers=None, freqstest=None,
                 tests='auto', inclass_correction=None, betweenclass_weighting='auto', estimator='auto',
                 modelselector=None, verbosity=1, name=None):
        """
        Implements instability ("drift") detection and characterization on timeseries data from *any* set of
        quantum circuits on *any* number of qubits. This uses the StabilityAnalyzer object, and directly
        accessing that object allows for some more complex analyzes to be performed. That object also offers
        a more step-by-step analysis procedure, which may be helpful for exploring the optional arguments of this
        analysis.

        Parameters
        ----------
        ds : DataSet or MultiDataSet
            A DataSet containing time-series data to be analyzed for signs of instability.

        significance : float, optional
            The global significance level. With defaults for all other inputs (a wide range of non-default options),
            the family-wise error rate of the set of all hypothesis tests performed is controlled to this value.

        transform : str, optional
            The type of transform to use in the spectral analysis. Options are:

            * 'auto':   An attempt is made to choose the best transform given the "meta-data" of the data,
                        e.g., the variability in the time-step between data points. For beginners,
                        'auto' is the best option. If you are familiar with the underlying methods, the
                        meta-data of the input, and the relative merits of the different transform, then
                        it is probably better to choose this yourself -- as the auto-selection is not hugely
                        sophisticated.
            * 'dct' :   The Type-II Discrete Cosine Transform (with an orthogonal normalization). This is
                        the only tested option, and it is our recommended option when the data is
                        approximately equally-spaced, i.e., the time-step between each "click" for each
                        circuit is almost a constant. (the DCT transform implicitly assumes that this
                        time-step is exactly constant)
            * 'dft' :   The discrete Fourier transform (with an orthogonal normalization). 
                        **This is an experimental feature, and the results are unreliable with this transform**
            * 'lsp' :   The Lomb-Scargle periodogram.  
                        **This is an experimental feature, and the code is untested with this transform**

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

        estimator : str, optional
            The name of the estimator to use. This is the method used to estimate the parameters of a parameterized
            model for each probability trajectory, after that parameterized model has been selected with the model
            selection methods. Allowed values are:

            * 'auto'. The estimation method is chosen automatically, default to the fast method that is also
                reasonably reliable.
            * 'filter'. Performs a type of signal filtering: implements the transform used for generating power
                spectra (e.g., the DCT), sets the amplitudes to zero for all freuquencies that the model selection
                has not included in the model, inverts the transform, and then performs some minor post-processing
                to guarantee probabilities within [0, 1]. This method is less statically well-founded than 'mle',
                but it is faster and typically gives similar results. This method is not an option for
                non-invertable transforms, such as the Lomb-Scargle periodogram.
            * 'mle'. Implements maximum likelihood estimation, on the parameterized model chosen by the model
                selection. The most statistically well-founded option, but can be slower than 'filter' and relies
                on numerical optimization.

        modelselector : tuple, optional
            The model selection method. If not None, a "test class" tuple, specifying which test results to use to
            decide which frequencies are significant for each circuit, to then construct a parameterized model for
            each probability trajectory. This can be typically set to None, and it will be chosen automatically.
            But if you wish to use specific test results for the model selection then this should be set.

        verbosity : int, optional
            The amount of print-to-screen

        Returns
        -------
        StabilityAnalysis
        """
        super().__init__(name)
        if inclass_correction is None:
            inclass_correction = {}
        if freqpointers is None:
            freqpointers = {}
        self.significance = significance
        self.transform = transform
        self.marginalize = marginalize
        self.mergeoutcomes = mergeoutcomes
        self.constnumtimes = constnumtimes
        self.ids = ids
        self.frequencies = frequencies
        self.freqpointers = freqpointers
        self.freqstest = freqstest
        self.tests = tests
        self.inclass_correction = inclass_correction
        self.betweenclass_weighting = betweenclass_weighting
        self.estimator = estimator
        self.modelselector = modelselector
        self.verbosity = verbosity
        # ...
        #self.auxfile_types['big_thing'] = 'pickle'

    def run(self, data, memlimit=None, comm=None):
        #design = data.edesign  # experiment design (specifies circuits)
        """
        Run this protocol on `data`.

        Parameters
        ----------
        data : ProtocolData
            The input data.

        memlimit : int, optional
            A rough per-processor memory limit in bytes.

        comm : mpi4py.MPI.Comm, optional
            When not ``None``, an MPI communicator used to run this protocol
            in parallel.

        Returns
        -------
        StabilityAnalysisResults
        """
        from ..extras import drift as _drift
        ds = data.dataset  # dataset

        if self.verbosity > 0: print(" - Formatting the data...", end='')
        results = _drift.StabilityAnalyzer(ds, transform=self.transform, marginalize=self.marginalize,
                                           mergeoutcomes=self.mergeoutcomes,
                                           constnumtimes=self.constnumtimes, ids=self.ids)
        if self.verbosity > 0: print("done!")

        # Calculate the power spectra.
        if self.verbosity > 0: print(" - Calculating power spectra...", end='')
        results.compute_spectra(frequencies=self.frequencies, freqpointers=self.freqpointers)
        if self.verbosity > 0: print("done!")

        # Implement the drift detection with statistical hypothesis testing on the power spectra.
        if self.verbosity > 0: print(" - Running instability detection...", end='')
        if self.verbosity > 1: print('')
        results.run_instability_detection(significance=self.significance, freqstest=self.freqstest, tests=self.tests,
                                          inclass_correction=self.inclass_correction,
                                          betweenclass_weighting=self.betweenclass_weighting,
                                          saveas='default', default=True, overwrite=False, verbosity=self.verbosity - 1)
        if self.verbosity == 1: print("done!")
        # Estimate the drifting probabilities.
        if self.verbosity > 0: print(" - Running instability characterization...", end='')
        if self.verbosity > 1: print('')

        # The model selector something slightly more complicated for this method: this function only allows us to
        # set the second part of the modelselector tuple.
        results.run_instability_characterization(estimator=self.estimator, modelselector=(None, self.modelselector),
                                                 default=True, verbosity=self.verbosity - 1)
        if self.verbosity == 1: print("done!")

        return StabilityAnalysisResults(data, self, results)  # put results in here


class StabilityAnalysisResults(_proto.ProtocolResults):
    """
    Results from the stability analysis protocol.

    **NOTE**
    Currently, this object just wraps a :class:`pygsti.extras.drift.StabilityAnalyzer`
    object, which historically performed stability analysis.  In the future, this object
    will likely take over the function of `StabilityAnalyzer`.

    Parameters
    ----------
    data : ProtocolData
        The experimental data these results are generated from.

    protocol_instance : Protocol
        The protocol that generated these results.

    stabilityanalyzer : pygsti.extras.drift.StabilityAnalyzer
        An object holding the stability analysis results.  This will likely
        be updated in the future.
    """

    def __init__(self, data, protocol_instance, stabilityanalyzer):
        """
        Initialize an empty Results object.
        """
        super().__init__(data, protocol_instance)

        self.stabilityanalyzer = stabilityanalyzer
        self.auxfile_types['stabilityanalyzer'] = 'pickle'

    def __getattr__(self, attr):
        # punt to stabilityanalyzer for now
        return getattr(self.stabilityanalyzer, attr)
