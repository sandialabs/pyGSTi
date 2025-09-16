"""
Functions for creating data
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import collections as _collections
import itertools as _itertools
import warnings as _warnings

import numpy as _np
import numpy.random as _rndm

from pygsti.circuits import circuitconstruction as _gstrc
from pygsti.data import dataset as _ds
from pygsti.baseobjs import label as _lbl, outcomelabeldict as _ld
from typing import Optional, Union


def simulate_data(model_or_dataset, circuit_list, num_samples,
                  sample_error="multinomial", seed=None, rand_state=None,
                  alias_dict=None, collision_action="aggregate",
                  record_zero_counts=True, comm=None, mem_limit=None, times=None):
    """
    Creates a DataSet using the probabilities obtained from a model.

    Parameters
    ----------
    model_or_dataset : Model or DataSet object
        The source of the underlying probabilities used to generate the data.
        If a Model, the model whose probabilities generate the data.
        If a DataSet, the data set whose frequencies generate the data.

    circuit_list : list of (tuples or Circuits) or ExperimentDesign or None
        Each tuple or Circuit contains operation labels and
        specifies a gate sequence whose counts are included
        in the returned DataSet. e.g. ``[ (), ('Gx',), ('Gx','Gy') ]``
        If an :class:`ExperimentDesign`, then the design's `.all_circuits_needing_data`
        list is used as the circuit list.

    num_samples : int or list of ints or None
        The simulated number of samples for each circuit.  This only has
        effect when  ``sample_error == "binomial"`` or ``"multinomial"``.  If an
        integer, all circuits have this number of total samples. If a list,
        integer elements specify the number of samples for the corresponding
        circuit.  If ``None``, then `model_or_dataset` must be a
        :class:`~pygsti.objects.DataSet`, and total counts are taken from it
        (on a per-circuit basis).

    sample_error : string, optional
        What type of sample error is included in the counts.  Can be:

        - "none"  - no sample error: counts are floating point numbers such
          that the exact probabilty can be found by the ratio of count / total.
        - "clip" - no sample error, but clip probabilities to [0,1] so, e.g.,
          counts are always positive.
        - "round" - same as "clip", except counts are rounded to the nearest
          integer.
        - "binomial" - the number of counts is taken from a binomial
          distribution.  Distribution has parameters p = (clipped) probability
          of the circuit and n = number of samples.  This can only be used
          when there are exactly two SPAM labels in model_or_dataset.
        - "multinomial" - counts are taken from a multinomial distribution.
          Distribution has parameters p_k = (clipped) probability of the gate
          string using the k-th SPAM label and n = number of samples.

    seed : int, optional
        If not ``None``, a seed for numpy's random number generator, which
        is used to sample from the binomial or multinomial distribution.

    rand_state : numpy.random.RandomState
        A RandomState object to generate samples from. Can be useful to set
        instead of `seed` if you want reproducible distribution samples across
        multiple random function calls but you don't want to bother with
        manually incrementing seeds between those calls.

    alias_dict : dict, optional
        A dictionary mapping single operation labels into tuples of one or more
        other operation labels which translate the given circuits before values
        are computed using `model_or_dataset`.  The resulting Dataset, however,
        contains the *un-translated* circuits as keys.

    collision_action : {"aggregate", "keepseparate"}
        Determines how duplicate circuits are handled by the resulting
        `DataSet`.  Please see the constructor documentation for `DataSet`.

    record_zero_counts : bool, optional
        Whether zero-counts are actually recorded (stored) in the returned
        DataSet.  If False, then zero counts are ignored, except for
        potentially registering new outcome labels.

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors and ensuring that the *same* dataset is
        generated on each processor.

    mem_limit : int, optional
        A rough memory limit in bytes which is used to determine job allocation
        when there are multiple processors.

    times : iterable, optional
        When not None, a list of time-stamps at which data should be sampled.
        `num_samples` samples will be simulated at each time value, meaning that
        each circuit in `circuit_list` will be evaluated with the given time
        value as its *start time*.

    Returns
    -------
    DataSet
        A static data set filled with counts for the specified circuits.
    """
    NTOL = 10
    TOL = 10**-NTOL

    if isinstance(model_or_dataset, _ds.DataSet):
        dsGen = model_or_dataset
        gsGen = None
        dataset = _ds.DataSet(collision_action=collision_action,
                              outcome_label_indices=dsGen.olIndex)  # keep same outcome labels
    else:
        gsGen = model_or_dataset
        dsGen = None
        dataset = _ds.DataSet(collision_action=collision_action)

    if alias_dict:
        alias_dict = {_lbl.Label(ky): tuple((_lbl.Label(el) for el in val))
                      for ky, val in alias_dict.items()}  # convert to use Labels

    from pygsti.protocols import ExperimentDesign as _ExperimentDesign
    if isinstance(circuit_list, _ExperimentDesign):
        circuit_list = circuit_list.all_circuits_needing_data

    if gsGen and times is None:
        if alias_dict is not None:
            trans_circuit_list = [_gstrc.translate_circuit(s, alias_dict)
                                  for s in circuit_list]
        else:
            trans_circuit_list = circuit_list
        all_probs = gsGen.bulk_probabilities(trans_circuit_list, comm=comm, mem_limit=mem_limit)
    else:
        trans_circuit_list = circuit_list

    if comm is None or comm.Get_rank() == 0:  # only root rank computes

        if sample_error in ("binomial", "multinomial") and rand_state is None:
            rndm = _rndm.RandomState(seed)  # ok if seed is None
        else:
            rndm = rand_state  # can be None

        circuit_times = times if times is not None else ["N/A dummy"]
        count_lists = _collections.OrderedDict()

        for tm in circuit_times:
            #print("Time ", tm)

            #It would be nice to be able to do something like this (time dependent calc for all probs at ptic time)
            #if gsGen and times is not None:
            #    all_probs = gsGen.bulk_probabilities(trans_circuit_list, comm=comm, mem_limit=mem_limit, time=tm)

            for k, (s, trans_s) in enumerate(zip(circuit_list, trans_circuit_list)):

                if gsGen:
                    if times is None:
                        ps = all_probs[trans_s]
                    else:
                        ps = gsGen.probabilities(trans_s, time=tm)

                    if sample_error in ("binomial", "multinomial"):
                        _adjust_probabilities_inbounds(ps, TOL)
                else:
                    ps = _collections.OrderedDict([(ol, frac) for ol, frac
                                                   in dsGen[trans_s].fractions.items()])

                if gsGen and sample_error in ("binomial", "multinomial"):
                    _adjust_unit_sum(ps, TOL)

                if num_samples is None and dsGen is not None:
                    N = dsGen[trans_s].total  # use the number of samples from the generating dataset
                    #Note: total() accounts for other intermediate-measurment branches automatically
                else:
                    try:
                        N = num_samples[k]  # try to treat num_samples as a list
                    except:
                        N = num_samples  # if not indexable, num_samples should be a single number

                nWeightedSamples = N
                counts = _sample_distribution(ps, sample_error, nWeightedSamples, rndm)
                if s not in count_lists: count_lists[s] = []
                count_lists[s].append(counts)

        if times is None:
            for s, counts_list in count_lists.items():
                for counts_dict in counts_list:
                    dataset.add_count_dict(s, counts_dict, record_zero_counts=record_zero_counts)
        else:
            for s, counts_list in count_lists.items():
                dataset.add_series_data(s, counts_list, times, record_zero_counts=record_zero_counts)

        dataset.done_adding_data()

    if comm is not None:  # broadcast to non-root procs
        dataset = comm.bcast(dataset if (comm.Get_rank() == 0) else None, root=0)

    return dataset


def _adjust_probabilities_inbounds(ps, tol):
    #Adjust to probabilities if needed (and warn if not close to in-bounds)
    # ps is a dict w/keys = outcome labels and values = probabilities
    for ol in ps:
        if ps[ol] < 0:
            if ps[ol] < -tol: _warnings.warn("Clipping probs < 0 to 0")
            ps[ol] = 0.0
        elif ps[ol] > 1:
            if ps[ol] > (1 + tol): _warnings.warn("Clipping probs > 1 to 1")
            ps[ol] = 1.0


def _adjust_unit_sum(ps, tol):
    #Check that sum ~= 1 (and nudge if needed) since binomial and
    #  multinomial random calls assume this.
    OVERTOL = 1.0 + tol
    UNDERTOL = 1.0 - tol
    psum = sum(ps.values())
    adjusted = False
    if psum > OVERTOL:
        adjusted = True
        _warnings.warn("Adjusting sum(probs) = %g > 1 to 1" % psum)
    if psum < UNDERTOL:
        adjusted = True
        _warnings.warn("Adjusting sum(probs) = %g < 1 to 1" % psum)

    if not UNDERTOL <= psum <= OVERTOL:
        for lbl in ps.keys():
            ps[lbl] /= psum
    assert(UNDERTOL <= sum(ps.values()) <= OVERTOL), 'psum={}'.format(sum(ps.values()))

    if adjusted:
        _warnings.warn('Adjustment finished')


def _sample_distribution(ps, sample_error, nSamples, rndm_state):
    counts = {}  # don't use an ordered dict here - add_count_dict will sort keys
    labels = [ol for ol, _ in sorted(list(ps.items()), key=lambda x: x[1])]
    # "outcome labels" - sort by prob for consistent generation
    if sample_error == "binomial":

        if len(labels) == 1:  # Special case when labels[0] == 1.0 (100%)
            counts[labels[0]] = nSamples
        else:
            assert(len(labels) == 2)
            ol0, ol1 = labels[0], labels[1]
            counts[ol0] = rndm_state.binomial(nSamples, ps[ol0])
            counts[ol1] = nSamples - counts[ol0]

    elif sample_error == "multinomial":
        countsArray = rndm_state.multinomial(nSamples,
                                             [ps[ol] for ol in labels], size=1)  # well-ordered list of probs
        for i, ol in enumerate(labels):
            counts[ol] = countsArray[0, i]
    else:
        for outcomeLabel, p in ps.items():
            pc = _np.clip(p, 0, 1)  # Note: *not* used in "none" case
            if sample_error == "none":
                counts[outcomeLabel] = float(nSamples * p)
            elif sample_error == "clip":
                counts[outcomeLabel] = float(nSamples * pc)
            elif sample_error == "round":
                counts[outcomeLabel] = int(round(nSamples * pc))
            else:
                raise ValueError(
                    "Invalid sample error parameter: '%s'  "
                    "Valid options are 'none', 'round', 'binomial', or 'multinomial'" % sample_error)
    return counts


def aggregate_dataset_outcomes(dataset, label_merge_dict, record_zero_counts=True):
    """
    Creates a DataSet which merges certain outcomes in input DataSet.

    This is used, for example, to aggregate a 2-qubit, 4-outcome DataSet into a
    1-qubit, 2-outcome DataSet.

    Parameters
    ----------
    dataset : DataSet object
        The input DataSet whose results will be simplified according to the rules
        set forth in label_merge_dict

    label_merge_dict : dictionary
        The dictionary whose keys define the new DataSet outcomes, and whose items
        are lists of input DataSet outcomes that are to be summed together.  For example,
        if a two-qubit DataSet has outcome labels "00", "01", "10", and "11", and
        we want to ''aggregate out'' the second qubit, we could use label_merge_dict =
        {'0':['00','01'],'1':['10','11']}.  When doing this, however, it may be better
        to use :func:`filter_dataset` which also updates the circuits.

    record_zero_counts : bool, optional
        Whether zero-counts are actually recorded (stored) in the returned
        (merged) DataSet.  If False, then zero counts are ignored, except for
        potentially registering new outcome labels.

    Returns
    -------
    merged_dataset : DataSet object
        The DataSet with outcomes merged according to the rules given in label_merge_dict.
    """
    # strings -> tuple outcome labels in keys and values of label_merge_dict
    to_outcome = _ld.OutcomeLabelDict.to_outcome  # shorthand
    label_merge_dict = {to_outcome(key): list(map(to_outcome, val))
                        for key, val in label_merge_dict.items()}

    new_outcomes = label_merge_dict.keys()
    merged_dataset = _ds.DataSet(outcome_labels=new_outcomes)
    merge_dict_old_outcomes = [outcome for sublist in label_merge_dict.values() for outcome in sublist]
    if not set(dataset.outcome_labels).issubset(merge_dict_old_outcomes):
        raise ValueError(
            "`label_merge_dict` must account for all the outcomes in original dataset."
            " It's missing directives for:\n%s" %
            '\n'.join(set(map(str, dataset.outcome_labels)) - set(map(str, merge_dict_old_outcomes)))
        )

    # New code that works for time-series data.
    for key in dataset.keys():
        times, linecounts = dataset[key].counts_as_timeseries()
        count_dict_list = []
        for i in range(len(times)):
            count_dict = {}
            for new_outcome in new_outcomes:
                count_dict[new_outcome] = 0
                for old_outcome in label_merge_dict[new_outcome]:
                    count_dict[new_outcome] += linecounts[i].get(old_outcome, 0)
            if record_zero_counts is False:
                for new_outcome in new_outcomes:
                    if count_dict[new_outcome] == 0:
                        del count_dict[new_outcome]
            count_dict_list.append(count_dict)

        merged_dataset.add_series_data(key, count_dict_list, times, aux=dataset[key].aux)

    # Old code that doesn't work for time-series data.
    #for key in dataset.keys():
    #    linecounts = dataset[key].counts
    #    count_dict = {}
    #    for new_outcome in new_outcomes:
    #        count_dict[new_outcome] = 0
    #        for old_outcome in label_merge_dict[new_outcome]:
    #            count_dict[new_outcome] += linecounts.get(old_outcome, 0)
    #    merged_dataset.add_count_dict(key, count_dict, aux=dataset[key].aux,
    #                                  record_zero_counts=record_zero_counts)

    merged_dataset.done_adding_data()
    return merged_dataset


def _create_qubit_merge_dict(num_qubits, qubits_to_keep):
    """
    Creates a dictionary appropriate for use with :func:`aggregate_dataset_outcomes`.

    The returned dictionary instructs `aggregate_dataset_outcomes` to aggregate all but
    the specified `qubits_to_keep` when the outcome labels are those of
    `num_qubits` qubits (i.e. strings of 0's and 1's).

    Parameters
    ----------
    num_qubits : int
        The total number of qubits

    qubits_to_keep : list
        A list of integers specifying which qubits should be kept, that is,
        *not* aggregated, when the returned dictionary is passed to
        `aggregate_dataset_outcomes`.

    Returns
    -------
    dict
    """
    outcome_labels = [''.join(map(str, t)) for t in _itertools.product([0, 1], repeat=num_qubits)]
    return _create_merge_dict(qubits_to_keep, outcome_labels)


def _create_merge_dict(indices_to_keep, outcome_labels):
    """
    Creates a dictionary appropriate for use with :func:`aggregate_dataset_outcomes`.

    Each element of `outcome_labels` should be a n-character string (or a
    1-tuple of such a string).  The returned dictionary's keys will be all the
    unique results of keeping only the characters indexed by `indices_to_keep`
    from each outcome label.  The dictionary's values will be a list of all the
    original outcome labels which reduce to the key value when the
    non-`indices_to_keep` characters are removed.

    For example, if `outcome_labels == ['00','01','10','11']` and
    `indices_to_keep == [1]` then this function returns the dict
    `{'0': ['00','10'], '1': ['01','11'] }`.

    Note: if the elements of `outcome_labels` are 1-tuples then so are the
    elements of the returned dictionary's values.

    Parameters
    ----------
    indices_to_keep : list
        A list of integer indices specifying which character positions should be
        kept (i.e. *not* aggregated together by `aggregate_dataset_outcomes`).

    outcome_labels : list
        A list of the outcome labels to potentially merge.  This can be a list
        of strings or of 1-tuples containing strings.

    Returns
    -------
    dict
    """
    merge_dict = _collections.defaultdict(list)
    for ol in outcome_labels:
        if isinstance(ol, str):
            reduced = ''.join([ol[i] for i in indices_to_keep])
        else:
            assert(len(ol) == 1)  # should be a 1-tuple
            reduced = (''.join([ol[0][i] for i in indices_to_keep]),)  # a tuple
        merge_dict[reduced].append(ol)
    return dict(merge_dict)  # return a *normal* dict


def filter_dataset(dataset, sectors_to_keep, sindices_to_keep=None,
                   new_sectors=None, idle=((),), record_zero_counts=True,
                   filtercircuits=True):
    """
    Creates a DataSet that is the restriction of `dataset` to `sectors_to_keep`.

    This function aggregates (sums) outcomes in `dataset` which differ only in
    sectors (usually qubits - see below) *not* in `sectors_to_keep`, and removes
    any operation labels which act specifically on sectors not in
    `sectors_to_keep` (e.g. an idle gate acting on *all* sectors because it's
    `.sslbls` is None will *not* be removed).

    Here "sectors" are state-space labels, present in the circuits of
    `dataset`.  Each sector also corresponds to a particular character position
    within the outcomes labels of `dataset`.  Thus, for this function to work,
    the outcome labels of `dataset` must all be 1-tuples whose sole element is
    an n-character string such that each character represents the outcome of
    a single sector.  If the state-space labels are integers, then they can
    serve as both a label and an outcome-string position.  The argument
    `new_sectors` may be given to rename the kept state-space labels in the
    returned `DataSet`'s circuits.

    A typical case is when the state-space is that of *n* qubits, and the
    state space labels the intergers 0 to *n-1*.  As stated above, in this
    case there is no need to specify `sindices_to_keep`.  One may want to
    "rebase" the indices to 0 in the returned data set using `new_sectors`
    (E.g. `sectors_to_keep == [4,5,6]` and `new_sectors == [0,1,2]`).

    Parameters
    ----------
    dataset : DataSet object
        The input DataSet whose data will be processed.

    sectors_to_keep : list or tuple
        The state-space labels (strings or integers) of the "sectors" to keep in
        the returned DataSet.

    sindices_to_keep : list or tuple, optional
        The 0-based indices of the labels in `sectors_to_keep` which give the
        postiions of the corresponding letters in each outcome string (see above).
        If the state space labels are integers (labeling *qubits*) thath are also
        letter-positions, then this may be left as `None`.  For example, if the
        outcome strings of `dataset` are '00','01','10',and '11' and the first
        position refers to qubit "Q1" and the second to qubit "Q2" (present in
        operation labels), then to extract just "Q2" data `sectors_to_keep` should be
        `["Q2"]` and `sindices_to_keep` should be `[1]`.

    new_sectors : list or tuple, optional
        New sectors names to map the elements of `sectors_to_keep` onto in the
        output DataSet's circuits.  None means the labels are not renamed.
        This can be useful if, for instance, you want to run a 2-qubit protocol
        that expects the qubits to be labeled "0" and "1" on qubits "4" and "5"
        of a larger set.  Simply set `sectors_to_keep == [4,5]` and
        `new_sectors == [0,1]`.

    idle : string or Label, optional
        The operation label to be used when there are no kept components of a
        "layer" (element) of a circuit.

    record_zero_counts : bool, optional
        Whether zero-counts present in the original `dataset` are recorded
        (stored) in the returned (filtered) DataSet.  If False, then such
        zero counts are ignored, except for potentially registering new
        outcome labels.

    filtercircuits : bool, optional
        Whether or not to "filter" the circuits, by removing gates that act
        outside of the `sectors_to_keep`.

    Returns
    -------
    filtered_dataset : DataSet object
        The DataSet with outcomes and circuits filtered as described above.
    """
    if sindices_to_keep is None:
        sindices_to_keep = sectors_to_keep

    #ds_merged = dataset.aggregate_outcomes(_create_merge_dict(sindices_to_keep,
    #                                                     dataset.outcome_labels),
    #                                   record_zero_counts=record_zero_counts)
    ds_merged = dataset.aggregate_std_nqubit_outcomes(sindices_to_keep, record_zero_counts)

    if filtercircuits:
        ds_merged = ds_merged.process_circuits(lambda s: _gstrc.filter_circuit(
            s, sectors_to_keep, new_sectors, idle), aggregate=True)

    return ds_merged


def trim_to_constant_numtimesteps(ds):
    """
    Trims a :class:`DataSet` so that each circuit's data comprises the same number of timesteps.

    Returns a new dataset that has data for the same number of time steps for
    every circuit. This is achieved by discarding all time-series data for every
    circuit with a time step index beyond 'min-time-step-index', where
    'min-time-step-index' is the minimum number of time steps over circuits.

    Parameters
    ----------
    ds : DataSet
        The dataset to trim.

    Returns
    -------
    DataSet
        The trimmed dataset, obtained by potentially discarding some of the data.
    """
    trimmedds = ds.copy_nonstatic()
    numtimes = []
    for circuit in ds.keys():
        numtimes.append(ds[circuit].number_of_times)
    minnumtimes = min(numtimes)

    for circuit in ds.keys():
        times, series = ds[circuit].counts_as_timeseries()
        trimmedtimes = times[0:minnumtimes]
        trimmedseries = series[0:minnumtimes]
        trimmedds.add_series_data(circuit, trimmedseries, trimmedtimes, aux=ds.auxInfo[circuit])

    trimmedds.done_adding_data()

    return trimmedds


def _subsample_timeseries_data(ds, step):
    """
    Creates a :class:`DataSet` where each circuit's data is sub-sampled.

    Returns a new dataset where, for every circuit, we only keep the data at every
    'step' timestep. Specifically, the outcomes at the ith time for each circuit are
    kept for each i such that i modulo 'step' is zero.

    Parameters
    ----------
    ds : DataSet
        The dataset to subsample

    step : int
        The sub-sampling time step.  Only data at every `step` increment
        in time is kept.

    Returns
    -------
    DataSet
        The subsampled dataset.
    """
    subsampled_ds = ds.copy_nonstatic()
    for circ in ds.keys():
        times, odicts = ds[circ].timeseries_for_outcomes
        newtimes = []
        newseries = []
        for i in range(len(times)):
            if (i % step) == 0:
                newtimes.append(times[i])
                newseries.append({key: counts[i] for key, counts in odicts.items()})

        subsampled_ds.add_series_data(circ, newseries, newtimes)
    subsampled_ds.done_adding_data()

    return subsampled_ds
