"""
Defines the DataSet class and supporting classes and functions
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import numbers as _numbers
import uuid as _uuid
#import scipy.special as _sps
#import scipy.fftpack as _fft
#from scipy.integrate import quad as _quad
#from scipy.interpolate import interp1d as _interp1d

import pickle as _pickle
import copy as _copy
import warnings as _warnings
import bisect as _bisect
import itertools as _itertools

from collections import OrderedDict as _OrderedDict
from collections import defaultdict as _defaultdict

from ..tools import listtools as _lt
from ..tools import compattools as _compat

from . import circuit as _cir
from . import labeldicts as _ld
#from . import dataset as _ds

Oindex_type = _np.uint32
Time_type = _np.float64
Repcount_type = _np.float32
_DATAROW_AUTOCACHECOUNT_THRESHOLD = 256
# thought: _np.uint16 but doesn't play well with rescaling


class _DataSetKVIterator(object):
    """
    Iterator class for op_string,_DataSetRow pairs of a DataSet

    Parameters
    ----------
    dataset : DataSet
        The parent data set.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.gsIter = dataset.cirIndex.__iter__()

        oliData = self.dataset.oliData
        timeData = self.dataset.timeData
        repData = self.dataset.repData
        auxInfo = dataset.auxInfo

        def getcache(opstr):
            return dataset.cnt_cache[opstr] if dataset.bStatic else None

        if repData is None:
            self.tupIter = ((oliData[gsi], timeData[gsi], None, getcache(opstr), auxInfo[opstr])
                            for opstr, gsi in self.dataset.cirIndex.items())
        else:
            self.tupIter = ((oliData[gsi], timeData[gsi], repData[gsi], getcache(opstr), auxInfo[opstr])
                            for opstr, gsi in self.dataset.cirIndex.items())
        #Note: gsi above will be an index for a non-static dataset and
        #  a slice for a static dataset.

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.gsIter), _DataSetRow(self.dataset, *(next(self.tupIter)))

    next = __next__


class _DataSetValueIterator(object):
    """
    Iterator class for _DataSetRow values of a DataSet

    Parameters
    ----------
    dataset : DataSet
        The parent data set.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        oliData = self.dataset.oliData
        timeData = self.dataset.timeData
        repData = self.dataset.repData
        auxInfo = dataset.auxInfo

        def getcache(opstr):
            return dataset.cnt_cache[opstr] if dataset.bStatic else None

        if repData is None:
            self.tupIter = ((oliData[gsi], timeData[gsi], None, getcache(opstr), auxInfo[opstr])
                            for opstr, gsi in self.dataset.cirIndex.items())
        else:
            self.tupIter = ((oliData[gsi], timeData[gsi], repData[gsi], getcache(opstr), auxInfo[opstr])
                            for opstr, gsi in self.dataset.cirIndex.items())
        #Note: gsi above will be an index for a non-static dataset and
        #  a slice for a static dataset.

    def __iter__(self):
        return self

    def __next__(self):
        return _DataSetRow(self.dataset, *(next(self.tupIter)))

    next = __next__


class _DataSetRow(object):
    """
    Encapsulates DataSet time series data for a single circuit.

    Outwardly, it looks similar to a list with
    `(outcome_label, time_index, repetition_count)` tuples as the values.

    Parameters
    ----------
    dataset : DataSet
        The parent data set.

    row_oli_data : numpy.ndarray
        The outcome label indices for each bin of this row.

    row_time_data : numpy.ndarray
        The timestamps for each bin of this row.

    row_rep_data : numpy.ndarray
        The repetition counts for each bin of this row (if None, assume 1 per bin).

    cached_cnts : dict
        A cached pre-computed count dictionary (for speed).

    aux : dict
        Dictionary of auxiliary information.

    Attributes
    ----------
    outcomes : list
        Returns this row's sequence of outcome labels, one per "bin" of repetition
        counts (returned by :method:`get_counts`).

    counts : dict
        a dictionary of per-outcome counts.

    allcounts : dict
        a dictionary of per-outcome counts with *all* possible outcomes as keys
        and zero values when an outcome didn't occur.  Note this can be expensive
        to compute for many-qubit data.

    fractions : dict
        a dictionary of per-outcome fractions.

    total : int
        Returns the total number of counts contained in this row.
    """

    def __init__(self, dataset, row_oli_data, row_time_data, row_rep_data,
                 cached_cnts, aux):
        self.dataset = dataset
        self.oli = row_oli_data
        self.time = row_time_data
        self.reps = row_rep_data
        self._cntcache = cached_cnts
        self.aux = aux

    @property
    def outcomes(self):
        """
        This row's sequence of outcome labels, one per "bin" of repetition counts.
        """
        return [self.dataset.ol[i] for i in self.oli]

    @outcomes.setter
    def outcomes(self, value):
        """
        This row's sequence of outcome labels, one per "bin" of repetition counts.
        """
        raise ValueError("outcomes property is read-only")

    def expanded_ol(self):
        """
        This row's sequence of outcome labels, with repetition counts expanded.

        Thus, there's one element in the returned list for *each* count.

        Returns
        -------
        list
        """
        if self.reps is not None:
            ol = []
            for oli, _, nreps in zip(self.oli, self.time, self.reps):
                nreps = _round_int_repcnt(nreps)
                ol.extend([self.dataset.ol[oli]] * nreps)
            return ol
        else: return self.outcomes

    def expanded_oli(self):
        """
        This row's sequence of outcome label indices, with repetition counts expanded.

        Thus, there's one element in the returned list for *each* count.

        Returns
        -------
        numpy.ndarray
        """
        if self.reps is not None:
            inds = []
            for oli, _, nreps in zip(self.oli, self.time, self.reps):
                nreps = _round_int_repcnt(nreps)
                inds.extend([oli] * nreps)
            return _np.array(inds, dtype=self.dataset.oliType)
        else: return self.oli.copy()

    def expanded_times(self):
        """
        This row's sequence of time stamps, with repetition counts expanded.

        Thus, there's one element in the returned list for *each* count.

        Returns
        -------
        numpy.ndarray
        """
        if self.reps is not None:
            times = []
            for _, time, nreps in zip(self.oli, self.time, self.reps):
                nreps = _round_int_repcnt(nreps)
                times.extend([time] * nreps)
            return _np.array(times, dtype=self.dataset.timeType)
        else: return self.time.copy()

    def times(self):
        """
        A list containing the unique data collection times at which there is at least one measurement result.

        Returns
        -------
        list
        """
        times = []
        last_time = None
        for t in self.time:
            if t != last_time:
                times.append(t)
                last_time = t

        return times

    def timeseries_for_outcomes(self):
        """
        Row data in a time-series format.

        This can be a much less succinct format than returned by `get_timeseries`.
        E.g., it is highly inefficient for many-qubit data.

        Returns
        -------
        times : list
            The time steps, containing the unique data collection times.

        reps : dict
            A dictionary of lists containing the number of times each
            measurement outcome was observed at the unique data collection
            times in `times`.
        """
        times = []
        last_time = None
        seriesDict = {self.dataset.olIndex[ol]: [] for ol in self.dataset.outcome_labels()}

        #REMOVED: (though this gives slightly different behavior)
        #for outcome_label in self.outcomes:
        #    if outcome_label not in seriesDict.keys():
        #        seriesDict[outcome_label] = []

        if self.reps is None:
            reps = _np.ones(len(self.time), int)
        else: reps = self.reps

        # An alternate implementation that appears to be (surprisingly?) slower...
        ##Get time bin locations
        #time_bins_borders = []
        #last_time = None
        #for i, t in enumerate(self.time):
        #    if t != last_time:
        #        time_bins_borders.append(i)
        #        last_time = t
        #time_bins_borders.append(len(self.time))
        #nTimes = len(time_bins_borders) - 1
        #
        #seriesDict = {self.dataset.olIndex[ol]: _np.zeros(nTimes, int) for ol in self.dataset.outcome_labels()}
        #
        #for i in range(nTimes):
        #    slc = slice(time_bins_borders[i],time_bins_borders[i+1])
        #    times.append( self.time[slc.start] )
        #    for oli, rep in zip(self.oli[slc], reps[slc]):
        #        seriesDict[oli][i] += rep

        for t, oli, rep in zip(self.time, self.oli, reps):

            if t != last_time:
                times.append(t)
                last_time = t

                for sd_oli in seriesDict.keys():
                    if sd_oli == oli: seriesDict[sd_oli].append(rep)
                    else: seriesDict[sd_oli].append(0)
            else:
                seriesDict[oli][-1] += rep

        return times, {ol: seriesDict[oli] for ol, oli in self.dataset.olIndex.items()}

    def get_timeseries(self):
        """
        Returns data in a time-series format.

        Returns
        -------
        times : list
            The time steps, containing the unique data collection times.
        reps : list
            A list of dictionaries containing the counts dict corresponding
            to the list of unique data collection times in `times`.
        """
        times = []
        series = []
        last_time = None

        if self.reps is None:
            reps = list(_np.ones(len(self.time), int))
        else: reps = self.reps

        for t, outcome_label, rep in zip(self.time, self.outcomes, reps):

            if t != last_time:
                times.append(t)
                last_time = t
                series.append({outcome_label: rep})

            else:
                if outcome_label in series[-1]:
                    series[-1][outcome_label] += rep
                else:
                    series[-1][outcome_label] = rep

        return times, series

    def reps_timeseries(self):
        """
        The number of measurement results at each data collection time.

        Returns
        -------
        times : list
            The time steps.

        reps : list
            The total number of counts at each time step.
        """
        times = []
        reps = []
        last_time = None

        if self.reps is None:
            return list(self.time), list(_np.ones(len(self.time), int))

        else:
            for t, rep in zip(self.time, self.reps):
                if t != last_time:
                    times.append(t)
                    last_time = t
                    reps.append(rep)
                else:
                    reps[-1] += rep

            return times, reps

    def number_of_times(self):
        """
        Returns the number of data collection times.

        Returns
        -------
        int
        """
        return len(self.times())

    def has_constant_totalcounts(self):
        """
        True if the numbers of counts is the same at all data collection times. Otherwise False.

        Returns
        -------
        bool
        """
        times, reps = self.reps_timeseries()
        firstrep = reps[0]
        fixedtotalcounts = all([firstrep == i for i in reps])

        return fixedtotalcounts

    def totalcounts_per_timestep(self):
        """
        The number of total counts per time-step, when this is constant.

        If the total counts vary over the times that there is at least
        one measurement result, then this function will raise an error.

        Returns
        -------
        int
        """
        times, reps = self.reps_timeseries()
        firstrep = reps[0]
        assert(all([firstrep == i for i in reps])), "The total counts is not the same at all time steps!"
        return firstrep

    def meantimestep(self):
        """
        The mean time-step.

        Will raise an error for data that is a trivial time-series (i.e., data all at one time).

        Returns
        -------
        float
        """
        times = _np.array(self.times())
        assert(len(times) >= 2), "Mean time-step is ill-defined when there is not multiple data times!"

        return _np.mean(_np.diff(times))

    def __iter__(self):
        if self.reps is not None:
            return ((self.dataset.ol[i], t, n) for (i, t, n) in zip(self.oli, self.time, self.reps))
        else:
            return ((self.dataset.ol[i], t, 1) for (i, t) in zip(self.oli, self.time))

    def __contains__(self, outcome_label):
        """ Checks whether data counts for `outcomelabel` are available."""
        return outcome_label in self.counts

    def __getitem__(self, index_or_outcome_label):
        if isinstance(index_or_outcome_label, _numbers.Integral):  # raw index
            i = index_or_outcome_label
            if self.reps is not None:
                return (self.dataset.ol[self.oli[i]], self.time[i], self.reps[i])
            else:
                return (self.dataset.ol[self.oli[i]], self.time[i], 1)
        elif isinstance(index_or_outcome_label, _numbers.Real):  # timestamp
            return self.counts_at_time(index_or_outcome_label)
        else:
            if len(self.dataset.olIndex) > _DATAROW_AUTOCACHECOUNT_THRESHOLD:
                #There are a lot of outcomes in this dataset - it's not worth computing
                # and caching *all* of the counts just to extract the one being asked for now.
                outcome_label = _ld.OutcomeLabelDict.to_outcome(index_or_outcome_label)
                if outcome_label not in self.dataset.olIndex:
                    raise KeyError("%s is not an index, timestamp, or outcome label!"
                                   % str(index_or_outcome_label))
                return self._get_single_count(outcome_label)

            else:
                #Compute and cache *all* of the counts, since there aren't so many of them.
                try:
                    return self.counts[index_or_outcome_label]
                except KeyError:
                    # if outcome label isn't in counts but *is* in the dataset's
                    # outcome labels then return 0 (~= return self.allcounts[...])
                    key = _ld.OutcomeLabelDict.to_outcome(index_or_outcome_label)
                    if key in self.dataset.outcome_labels(): return 0
                    raise KeyError("%s is not an index, timestamp, or outcome label!"
                                   % str(index_or_outcome_label))

    def __setitem__(self, index_or_outcome_label, val):
        if isinstance(index_or_outcome_label, _numbers.Integral):
            index = index_or_outcome_label; tup = val
            assert(len(tup) in (2, 3)), "Must set to a (<outcomeLabel>,<time>[,<repetitions>]) value"
            ol = _ld.OutcomeLabelDict.to_outcome(tup[0])  # strings -> tuple outcome labels
            self.oli[index] = self.dataset.olIndex[ol]
            self.time[index] = tup[1]

            if self.reps is not None:
                self.reps[index] = tup[2] if len(tup) == 3 else 1
            else:
                assert(len(tup) == 2 or tup[2] == 1), "Repetitions must == 1 (not tracking reps)"
        else:
            outcomeLbl = _ld.OutcomeLabelDict.to_outcome(index_or_outcome_label)  # strings -> tuple outcome labels
            count = val

            assert(all([t == self.time[0] for t in self.time])), \
                "Cannot set outcome counts directly on a DataSet with non-trivially timestamped data"
            assert(self.reps is not None), \
                "Cannot set outcome counts directly on a DataSet without repetition data"

            outcomeIndxToLookFor = self.dataset.olIndex.get(outcomeLbl, None)
            for i, outcomeIndx in enumerate(self.oli):
                if outcomeIndx == outcomeIndxToLookFor:
                    self.reps[i] = count; break
            else:  # need to add a new label & entry to reps[]
                raise NotImplementedError("Cannot create new outcome labels by assignment")

    def _get_single_count(self, outcome_label, timestamp=None):
        if timestamp is not None:
            tslc = _np.where(_np.isclose(self.time, timestamp))[0]
        else: tslc = slice(None)

        if self.reps is None:
            i = self.dataset.olIndex[outcome_label]
            return float(_np.count_nonzero(_np.equal(self.oli[tslc], i)))
        else:
            i = self.dataset.olIndex[outcome_label]
            inds = _np.nonzero(_np.equal(self.oli[tslc], i))[0]
            if len(inds) > 0:
                return float(sum(self.reps[tslc][inds]))
            else:
                return 0.0

    def _get_counts(self, timestamp=None, all_outcomes=False):
        """
        Returns this row's sequence of "repetition counts", that is, the number of
        repetitions of each outcome label in the `outcomes` list, or
        equivalently, each outcome label index in this rows `.oli` member.
        """
        #Note: when all_outcomes == False we don't add outcome labels that
        # aren't present for any of this row's elements (i.e. the #summed
        # is zero)
        cntDict = _ld.OutcomeLabelDict()
        if timestamp is not None:
            tslc = _np.where(_np.isclose(self.time, timestamp))[0]
        else: tslc = slice(None)

        nOutcomes = len(self.dataset.olIndex)
        nIndices = len(self.oli[tslc])
        if nOutcomes <= nIndices or all_outcomes:
            if self.reps is None:
                for ol, i in self.dataset.olIndex.items():
                    cnt = float(_np.count_nonzero(_np.equal(self.oli[tslc], i)))
                    if all_outcomes or cnt > 0:
                        cntDict.setitem_unsafe(ol, cnt)
            else:
                for ol, i in self.dataset.olIndex.items():
                    inds = _np.nonzero(_np.equal(self.oli[tslc], i))[0]
                    if all_outcomes or len(inds) > 0:
                        cntDict.setitem_unsafe(ol, float(sum(self.reps[tslc][inds])))
        else:
            if self.reps is None:
                for ol_index in self.oli[tslc]:
                    ol = self.dataset.ol[ol_index]
                    cntDict.setitem_unsafe(ol, 1.0 + cntDict.getitem_unsafe(ol, 0.0))
            else:
                for ol_index, reps in zip(self.oli[tslc], self.reps[tslc]):
                    ol = self.dataset.ol[ol_index]
                    cntDict.setitem_unsafe(ol, reps + cntDict.getitem_unsafe(ol, 0.0))

        return cntDict

    @property
    def counts(self):
        """
        Dictionary of per-outcome counts.
        """
        if self._cntcache: return self._cntcache  # if not None *and* len > 0
        ret = self._get_counts()
        if self._cntcache is not None:  # == and empty dict {}
            self._cntcache.update(ret)
        return ret

    @property
    def allcounts(self):
        """
        Dictionary of per-outcome counts with *all* possible outcomes as keys.

        This means that and zero values are included when an outcome didn't occur.
        Note this can be expensive to assemble for many-qubit data.
        """
        return self._get_counts(all_outcomes=True)

    @property
    def fractions(self, all_outcomes=False):
        """
        Dictionary of per-outcome fractions.
        """
        cnts = self._get_counts(all_outcomes)
        total = sum(cnts.values())
        return _OrderedDict([(k, cnt / total) for k, cnt in cnts.items()])

    @property
    def total(self):
        """
        The total number of counts contained in this row.
        """
        if self.reps is None:
            return float(len(self.oli))
        else:
            return sum(self.reps)

    #TODO: remove in favor of fractions property?
    def fraction(self, outcomelabel):
        """
        The fraction of total counts for `outcomelabel`.

        Parameters
        ----------
        outcomelabel : str or tuple
            The outcome label, e.g. `'010'` or `('0','11')`.

        Returns
        -------
        float
        """
        d = self.counts
        if outcomelabel not in d:
            return 0.0  # Note: similar to an "all_outcomes=True" default
        total = sum(d.values())
        return d[outcomelabel] / total

    def counts_at_time(self, timestamp):
        """
        Returns a dictionary of counts at a particular time

        Parameters
        ----------
        timestamp : float
            the time to get counts at.

        Returns
        -------
        int
        """
        return self._get_counts(timestamp)

    def timeseries(self, outcomelabel, timestamps=None):
        """
        Retrieve timestamps and counts for a single outcome label or for aggregated counts if `outcomelabel == "all"`.

        Parameters
        ----------
        outcomelabel : str or tuple
            The outcome label to extract a series for.  If the special value
            `"all"` is used, total (aggregated over all outcomes) counts are
            returned.

        timestamps : list or array, optional
            If not None, an array of time stamps to extract counts for,
            which will also be returned as `times`.  Times at which
            there is no data will be returned as zero-counts.

        Returns
        -------
        times, counts : numpy.ndarray
        """
        if outcomelabel == 'all':
            olis = list(self.dataset.olIndex.values())
        else:
            outcomelabel = _ld.OutcomeLabelDict.to_outcome(outcomelabel)
            olis = [self.dataset.olIndex[outcomelabel]]

        times = []
        counts = []
        last_t = -1e100
        tsIndx = 0
        for i, (t, oli) in enumerate(zip(self.time, self.oli)):

            if timestamps is not None:
                while tsIndx < len(timestamps) and t > timestamps[tsIndx] \
                        and not _np.isclose(t, timestamps[tsIndx], rtol=0., atol=1e-12):
                    times.append(timestamps[tsIndx])
                    counts.append(0)
                    tsIndx += 1

            if oli in olis and (timestamps is None or _np.isclose(t, timestamps[tsIndx], rtol=0., atol=1e-12)):
                if not _np.isclose(t, last_t, rtol=0., atol=1e-12):
                    times.append(t); tsIndx += 1
                    counts.append(0)
                    last_t = t
                counts[-1] += 1 if (self.reps is None) else self.reps[i]

        if timestamps is not None:
            while tsIndx < len(timestamps):
                times.append(timestamps[tsIndx])
                counts.append(0)
                tsIndx += 1

        return _np.array(times, self.dataset.timeType), \
            _np.array(counts, self.dataset.repType)

    def scale_inplace(self, factor):
        """
        Scales all the counts of this row by the given factor

        Parameters
        ----------
        factor : float
            scaling factor.

        Returns
        -------
        None
        """
        if self.dataset.bStatic: raise ValueError("Cannot scale rows of a *static* DataSet.")
        if self.reps is None:
            raise ValueError(("Cannot scale a DataSet without repetition "
                              "counts. Call DataSet._add_explicit_repetition_counts()"
                              " and try this again."))
        for i, cnt in enumerate(self.reps):
            self.reps[i] = cnt * factor

    def to_dict(self):
        """
        Returns the (outcomeLabel,count) pairs as a dictionary.

        Returns
        -------
        dict
        """
        return dict(self.counts)

    def to_str(self, mode="auto"):
        """
        Render this _DataSetRow as a string.

        Parameters
        ----------
        mode : {"auto","time-dependent","time-independent"}
            Whether to display the data as time-series of outcome counts
            (`"time-dependent"`) or to report per-outcome counts aggregated over
            time (`"time-independent"`).  If `"auto"` is specified, then the
            time-independent mode is used only if all time stamps in the
            _DataSetRow are equal (trivial time dependence).

        Returns
        -------
        str
        """
        if mode == "auto":
            if all([t == self.time[0] for t in self.time]):
                mode = "time-independent"
            else: mode = "time-dependent"

        assert(mode in ('time-dependent', 'time-independent')), "Invalid `mode` argument: %s" % mode

        if mode == "time-dependent":
            s = "Outcome Label Indices = " + str(self.oli) + "\n"
            s += "Time stamps = " + str(self.time) + "\n"
            if self.reps is not None:
                s += "Repetitions = " + str(self.reps) + "\n"
            else:
                s += "( no repetitions )\n"
            return s
        else:  # time-independent
            return str(self.to_dict())

    def __str__(self):
        return self.to_str()

    def __len__(self):
        return len(self.oli)


def _round_int_repcnt(nreps):
    """ Helper function to localize warning message """
    if float(nreps).is_integer():
        return int(nreps)
    else:
        _warnings.warn("Rounding fractional repetition count to next lowest whole number!")
        return int(round(nreps))


class DataSet(object):
    """
    An association between Circuits and outcome counts, serving as the input data for many QCVV protocols.

    The DataSet class associates circuits with counts or time series of
    counts for each outcome label, and can be thought of as a table with gate
    strings labeling the rows and outcome labels and/or time labeling the
    columns.  It is designed to behave similarly to a dictionary of
    dictionaries, so that counts are accessed by:

    `count = dataset[circuit][outcomeLabel]`

    in the time-independent case, and in the time-dependent case, for *integer*
    time index `i >= 0`,

    `outcomeLabel = dataset[circuit][i].outcome`
    `count = dataset[circuit][i].count`
    `time = dataset[circuit][i].time`

    Parameters
    ----------
    oli_data : list or numpy.ndarray
        When `static == True`, a 1D numpy array containing outcome label
        indices (integers), concatenated for all sequences.  Otherwise, a
        list of 1D numpy arrays, one array per gate sequence.  In either
        case, this quantity is indexed by the values of `circuit_indices`
        or the index of `circuits`.

    time_data : list or numpy.ndarray
        Same format at `oli_data` except stores floating-point timestamp
        values.

    rep_data : list or numpy.ndarray
        Same format at `oli_data` except stores integer repetition counts
        for each "data bin" (i.e. (outcome,time) pair).  If all repetitions
        equal 1 ("single-shot" timestampted data), then `rep_data` can be
        `None` (no repetitions).

    circuits : list of (tuples or Circuits)
        Each element is a tuple of operation labels or a Circuit object.  Indices for these strings
        are assumed to ascend from 0.  These indices must correspond to the time series of spam-label
        indices (above).   Only specify this argument OR circuit_indices, not both.

    circuit_indices : ordered dictionary
        An OrderedDict with keys equal to circuits (tuples of operation labels) and values equal to
        integer indices associating a row/element of counts with the circuit.  Only
        specify this argument OR circuits, not both.

    outcome_labels : list of strings or int
        Specifies the set of spam labels for the DataSet.  Indices for the spam labels
        are assumed to ascend from 0, starting with the first element of this list.  These
        indices will associate each elememtn of `timeseries` with a spam label.  Only
        specify this argument OR outcome_label_indices, not both.  If an int, specifies that
        the outcome labels should be those for a standard set of this many qubits.

    outcome_label_indices : ordered dictionary
        An OrderedDict with keys equal to spam labels (strings) and value equal to
        integer indices associating a spam label with given index.  Only
        specify this argument OR outcome_labels, not both.

    static : bool
        When True, create a read-only, i.e. "static" DataSet which cannot be modified. In
          this case you must specify the timeseries data, circuits, and spam labels.
        When False, create a DataSet that can have time series data added to it.  In this case,
          you only need to specify the spam labels.

    file_to_load_from : string or file object
        Specify this argument and no others to create a static DataSet by loading
        from a file (just like using the load(...) function).

    collision_action : {"aggregate","overwrite","keepseparate"}
        Specifies how duplicate circuits should be handled.  "aggregate"
        adds duplicate-circuit counts to the same circuit's data at the
        next integer timestamp.  "overwrite" only keeps the latest given
        data for a circuit.  "keepseparate" tags duplicate-circuits by
        setting the `.occurrence` ID of added circuits that are already
        contained in this data set to the next available positive integer.

    comment : string, optional
        A user-specified comment string that gets carried around with the
        data.  A common use for this field is to attach to the data details
        regarding its collection.

    aux_info : dict, optional
        A user-specified dictionary of per-circuit auxiliary information.
        Keys should be the circuits in this DataSet and value should
        be Python dictionaries.
    """

    def __init__(self, oli_data=None, time_data=None, rep_data=None,
                 circuits=None, circuit_indices=None,
                 outcome_labels=None, outcome_label_indices=None,
                 static=False, file_to_load_from=None, collision_action="aggregate",
                 comment=None, aux_info=None):
        """
        Initialize a DataSet.

        Parameters
        ----------
        oli_data : list or numpy.ndarray
            When `static == True`, a 1D numpy array containing outcome label
            indices (integers), concatenated for all sequences.  Otherwise, a
            list of 1D numpy arrays, one array per gate sequence.  In either
            case, this quantity is indexed by the values of `circuit_indices`
            or the index of `circuits`.

        time_data : list or numpy.ndarray
            Same format at `oli_data` except stores floating-point timestamp
            values.

        rep_data : list or numpy.ndarray
            Same format at `oli_data` except stores integer repetition counts
            for each "data bin" (i.e. (outcome,time) pair).  If all repetitions
            equal 1 ("single-shot" timestampted data), then `rep_data` can be
            `None` (no repetitions).

        circuits : list of (tuples or Circuits)
            Each element is a tuple of operation labels or a Circuit object.  Indices for these strings
            are assumed to ascend from 0.  These indices must correspond to the time series of spam-label
            indices (above).   Only specify this argument OR circuit_indices, not both.

        circuit_indices : ordered dictionary
            An OrderedDict with keys equal to circuits (tuples of operation labels) and values equal to
            integer indices associating a row/element of counts with the circuit.  Only
            specify this argument OR circuits, not both.

        outcome_labels : list of strings or int
            Specifies the set of spam labels for the DataSet.  Indices for the spam labels
            are assumed to ascend from 0, starting with the first element of this list.  These
            indices will associate each elememtn of `timeseries` with a spam label.  Only
            specify this argument OR outcome_label_indices, not both.  If an int, specifies that
            the outcome labels should be those for a standard set of this many qubits.

        outcome_label_indices : ordered dictionary
            An OrderedDict with keys equal to spam labels (strings) and value equal to
            integer indices associating a spam label with given index.  Only
            specify this argument OR outcome_labels, not both.

        static : bool
            When True, create a read-only, i.e. "static" DataSet which cannot be modified. In
              this case you must specify the timeseries data, circuits, and spam labels.
            When False, create a DataSet that can have time series data added to it.  In this case,
              you only need to specify the spam labels.

        file_to_load_from : string or file object
            Specify this argument and no others to create a static DataSet by loading
            from a file (just like using the load(...) function).

        collision_action : {"aggregate","overwrite","keepseparate"}
            Specifies how duplicate circuits should be handled.  "aggregate"
            adds duplicate-circuit counts to the same circuit's data at the
            next integer timestamp.  "overwrite" only keeps the latest given
            data for a circuit.  "keepseparate" tags duplicate-circuits by
            setting the `.occurrence` ID of added circuits that are already
            contained in this data set to the next available positive integer.

        comment : string, optional
            A user-specified comment string that gets carried around with the
            data.  A common use for this field is to attach to the data details
            regarding its collection.

        aux_info : dict, optional
            A user-specified dictionary of per-circuit auxiliary information.
            Keys should be the circuits in this DataSet and value should
            be Python dictionaries.

        Returns
        -------
        DataSet
           a new data set object.
        """
        # uuid for efficient hashing (set when done adding data or loading from file)
        self.uuid = None

        #Optionally load from a file
        if file_to_load_from is not None:
            assert(oli_data is None and time_data is None and rep_data is None
                   and circuits is None and circuit_indices is None
                   and outcome_labels is None and outcome_label_indices is None)
            self.load(file_to_load_from)
            return

        # self.cirIndex  :  Ordered dictionary where keys = Circuit objects,
        #   values = slices into oli, time, & rep arrays (static case) or
        #            integer list indices (non-static case)
        if circuit_indices is not None:
            self.cirIndex = _OrderedDict([(opstr if isinstance(opstr, _cir.Circuit) else _cir.Circuit(opstr), i)
                                          for opstr, i in circuit_indices.items()])
            #convert keys to Circuits if necessary
        elif not static:
            if circuits is not None:
                dictData = [(opstr if isinstance(opstr, _cir.Circuit) else _cir.Circuit(opstr), i)
                            for (i, opstr) in enumerate(circuits)]  # convert to Circuits if necessary
                self.cirIndex = _OrderedDict(dictData)
            else:
                self.cirIndex = _OrderedDict()
        else: raise ValueError("Must specify circuit_indices when creating a static DataSet")

        # self.olIndex  :  Ordered dictionary where
        #                  keys = outcome labels (strings or tuples),
        #                  values = integer indices mapping oli_data (integers) onto
        #                           the outcome labels.
        if outcome_label_indices is not None:
            self.olIndex = outcome_label_indices
            self.olIndex_max = max(self.olIndex.values()) if len(self.olIndex) > 0 else -1
        elif outcome_labels is not None:
            if isinstance(outcome_labels, int):
                nqubits = outcome_labels
                tup_outcomeLabels = [("".join(x),) for x in _itertools.product(*([('0', '1')] * nqubits))]
            else:
                tup_outcomeLabels = [_ld.OutcomeLabelDict.to_outcome(ol)
                                     for ol in outcome_labels]  # strings -> tuple outcome labels
            self.olIndex = _OrderedDict([(ol, i) for (i, ol) in enumerate(tup_outcomeLabels)])
            self.olIndex_max = len(tup_outcomeLabels) - 1
        else:
            self.olIndex = _OrderedDict()  # OK, as outcome labels are added as they appear
            self.olIndex_max = -1

        # self.ol :  Ordered dictionary where keys = integer indices, values = outcome
        #            labels (strings or tuples) -- just the reverse of self.olIndex
        self.ol = _OrderedDict([(i, ol) for (ol, i) in self.olIndex.items()])

        # sanity checks that indices are >= 0
        if not static:  # otherwise values() below are slices
            if self.cirIndex: assert(min(self.cirIndex.values()) >= 0)
            if self.olIndex: assert(min(self.olIndex.values()) >= 0)

        # self.oliData : when static == True a 1D numpy array containing concatenated outcome label indices.
        #                when static == False a list of 1D numpy arrays, one array per gate sequence.

        # self.timeData : when static == True a 1D numpy array containing concatenated time stamps.
        #                 when static == False a list of 1D numpy arrays, one array per gate sequence.

        # self.repData : when static == True a 1D numpy array containing concatenated repetition counts.
        #                when static == False a list of 1D numpy arrays, one array per gate sequence.
        #   (can be None, in which case no repetitions are assumed)

        if oli_data is not None:

            # check that sizes/lengths all match
            assert(len(time_data) == len(oli_data)), "time_data must be same size as oli_data"
            if rep_data is not None:
                assert(len(rep_data) == len(oli_data)), "rep_data must be same size as oli_data"

            self.oliData = oli_data
            self.timeData = time_data
            self.repData = rep_data

            if len(self.cirIndex) > 0:
                maxOlIndex = self.olIndex_max
                if static:
                    assert(max([_np.amax(self.oliData[i]) if (len(self.oliData[i]) > 0) else 0
                                for i in self.cirIndex.values()]) <= maxOlIndex)
                    # self.oliData.shape[0] > maxIndex doesn't make sense since cirIndex holds slices
                else:
                    #Note: for non-static datasets, assume *all* data in self.oliData is "in" this data set, i.e.,
                    # it can't be that this is a truncated dataset with pointers to more data than it actually owns.
                    maxIndex = max(self.cirIndex.values())
                    assert(len(self.oliData) > maxIndex)
                    if len(self.oliData) > 0:
                        assert(all([max(oliSeries) <= maxOlIndex for oliSeries in self.oliData]))
            #else cirIndex has length 0 so there are no circuits in this dataset (even though oli_data can contain data)

        elif not static:
            assert(time_data is None), "time_data must be None when oli_data is"
            assert(rep_data is None), "rep_data must be None when oli_data is"
            assert(len(self.cirIndex) == 0), "circuit specified without data!"
            self.oliData = []
            self.timeData = []
            self.repData = None

        else:
            raise ValueError("Series data must be specified when creating a static DataSet")

        # self.bStatic
        self.bStatic = static

        # collision action
        assert(collision_action in ('aggregate', 'overwrite', 'keepseparate'))
        self.collisionAction = collision_action

        # comment
        self.comment = comment

        # self.ffdata : fourier filtering data
        self.ffdata = {}

        #data types - should stay in sync with MultiDataSet
        self.oliType = Oindex_type
        self.timeType = Time_type
        self.repType = Repcount_type

        #auxiliary info
        if aux_info is None:
            self.auxInfo = _defaultdict(dict)
        else:
            self.auxInfo = _defaultdict(dict, aux_info)

        # count cache (only used when static; not saved/loaded from disk)
        if static:
            self.cnt_cache = {opstr: _ld.OutcomeLabelDict() for opstr in self.cirIndex}
        else:
            self.cnt_cache = None

    def __iter__(self):
        return self.cirIndex.__iter__()  # iterator over circuits

    def __len__(self):
        return len(self.cirIndex)

    def __contains__(self, circuit):
        """
        Test whether data set contains a given circuit.

        Parameters
        ----------
        circuit : tuple or Circuit
            A tuple of operation labels or a Circuit instance
            which specifies the the circuit to check for.

        Returns
        -------
        bool
            whether circuit was found.
        """
        if not isinstance(circuit, _cir.Circuit):
            circuit = _cir.Circuit(circuit)
        return circuit in self.cirIndex

    def __hash__(self):
        if self.uuid is not None:
            return hash(self.uuid)
        else:
            raise TypeError('Use digest hash')

    def __getitem__(self, circuit):
        return self._get_row(circuit)

    def __setitem__(self, circuit, outcome_dict_or_series):
        ca = self.collisionAction
        self.collisionAction = 'overwrite'  # overwrite data when assigning (this seems mose natural)
        try:
            ret = self._set_row(circuit, outcome_dict_or_series)
        finally:
            self.collisionAction = ca
        return ret

    def __delitem__(self, circuit):
        if not isinstance(circuit, _cir.Circuit):
            circuit = _cir.Circuit(circuit)
        self._remove([self.cirIndex[circuit]])

    def _get_row(self, circuit):
        """
        Get a row of data from this DataSet.

        Parameters
        ----------
        circuit : Circuit or tuple
            The gate sequence to extract data for.

        Returns
        -------
        _DataSetRow
        """

        #Convert to circuit 
        # needed because name-only Labels don't hash the same as strings
        # so key lookups need to be done at least with tuples of Labels.
        circuit = _cir.Circuit.cast(circuit)

        #Note: cirIndex value is either an int (non-static) or a slice (static)
        repData = self.repData[self.cirIndex[circuit]] \
            if (self.repData is not None) else None
        return _DataSetRow(self, self.oliData[self.cirIndex[circuit]],
                          self.timeData[self.cirIndex[circuit]], repData,
                          self.cnt_cache[circuit] if self.bStatic else None,
                          self.auxInfo[circuit])

    def _set_row(self, circuit, outcome_dict_or_series):
        """
        Set the counts for a row of this DataSet.

        Parameters
        ----------
        circuit : Circuit or tuple
            The gate sequence to extract data for.

        outcome_dict_or_series : dict or tuple
            The outcome count data, either a dictionary of outcome counts (with keys
            as outcome labels) or a tuple of lists.  In the latter case this can be
            a 2-tuple: (outcome-label-list, timestamp-list) or a 3-tuple:
            (outcome-label-list, timestamp-list, repetition-count-list).

        Returns
        -------
        None
        """
        circuit = _cir.Circuit.cast(circuit)
        if isinstance(outcome_dict_or_series, dict):  # a dict of counts
            self.add_count_dict(circuit, outcome_dict_or_series)

        else:  # a tuple of lists
            assert(len(outcome_dict_or_series) >= 2), \
                "Must minimally set with (outcome-label-list, time-stamp-list)"
            self.add_raw_series_data(circuit, *outcome_dict_or_series)

    def keys(self):
        """
        Returns the circuits used as keys of this DataSet.

        Returns
        -------
        list
            A list of Circuit objects which index the data
            counts within this data set.
        """
        yield from self.cirIndex.keys()

    def items(self):
        """
        Iterator over `(circuit, timeSeries)` pairs.

        Here `circuit` is a tuple of operation labels and `timeSeries` is a
        :class:`_DataSetRow` instance, which behaves similarly to a list of spam
        labels whose index corresponds to the time step.

        Returns
        -------
        _DataSetKVIterator
        """
        return _DataSetKVIterator(self)

    def values(self):
        """
        Iterator over _DataSetRow instances corresponding to the time series data for each circuit.

        Returns
        -------
        _DataSetValueIterator
        """
        return _DataSetValueIterator(self)

    def outcome_labels(self):
        """
        Get a list of *all* the outcome labels contained in this DataSet.

        Returns
        -------
        list of strings or tuples
            A list where each element is an outcome label (which can
            be a string or a tuple of strings).
        """
        return list(self.olIndex.keys())

    def gate_labels(self, prefix='G'):
        """
        Get a list of all the distinct operation labels used in the circuits of this dataset.

        Parameters
        ----------
        prefix : str
            Filter the circuit labels so that only elements beginning with
            this prefix are returned.  `None` performs no filtering.

        Returns
        -------
        list of strings
            A list where each element is a operation label.
        """
        opLabels = []
        for opLabelString in self:
            for opLabel in opLabelString:
                if not prefix or opLabel.name.startswith(prefix):
                    if opLabel not in opLabels: opLabels.append(opLabel)
        return opLabels

    def degrees_of_freedom(self, circuit_list=None, method="present_outcomes-1",
                           aggregate_times=True):
        """
        Returns the number of independent degrees of freedom in the data for the circuits in `circuit_list`.

        Parameters
        ----------
        circuit_list : list of Circuits
            The list of circuits to count degrees of freedom for.  If `None`
            then all of the `DataSet`'s strings are used.

        method : {'all_outcomes-1', 'present_outcomes-1'}
            How the degrees of freedom should be computed. 'all_outcomes-1' takes
            the number of circuits and multiplies this by the *total* number of outcomes
            (the length of what is returned by `outcome_labels()`) minus one.
            'present_outcomes-1' counts on a per-circuit basis the number of
            present (usually = non-zero) outcomes recorded minus one.  For timestamped
            data, see `aggreate_times` below.

        aggregate_times : bool, optional
            Whether counts that occur at different times should be tallied separately.
            If True, then even when counts occur at different times degrees of freedom
            are tallied on a per-circuit basis.  If False, then counts occuring at
            distinct times are treated as independent of those an any other time, and
            are tallied separately.  So, for example, if `aggregate_times` is False and
            a data row has 0- and 1-counts of 45 & 55 at time=0 and 42 and 58 at time=1
            this row would contribute *2* degrees of freedom, not 1.  It can sometimes be
            useful to set this to False when the `DataSet` holds coarse-grained data, but
            usually you want this to be left as True (especially for time-series data).

        Returns
        -------
        int
        """
        if circuit_list is None:
            circuit_list = list(self.keys())

        nDOF = 0
        Nout = len(self.olIndex)
        for opstr in circuit_list:
            dsRow = self[opstr]
            cur_t = dsRow.time[0]
            cur_outcomes = set()  # holds *distinct* outcomes at current time
            for ol, t, rep in dsRow:
                if aggregate_times or t == cur_t:
                    cur_outcomes.add(ol)
                else:
                    #assume final outcome at each time is constrained
                    nOutcomes = Nout if method == 'all_outcomes-1' else len(cur_outcomes)
                    nDOF += nOutcomes - 1; cur_outcomes = set([ol])
                    cur_t = t
            nOutcomes = Nout if method == 'all_outcomes-1' else len(cur_outcomes)
            nDOF += nOutcomes - 1  # last time stamp
        return nDOF

    def _collisionaction_update_circuit(self, circuit):
        if not isinstance(circuit, _cir.Circuit):
            circuit = _cir.Circuit(circuit)  # make sure we have a Circuit

        # if "keepseparate" mode, set occurrence id existing circuits to next available (positive) integer.
        if self.collisionAction == "keepseparate":
            if circuit in self.cirIndex:
                tagged_circuit = circuit.copy()
                i = 1; tagged_circuit.occurrence = i
                while tagged_circuit in self.cirIndex:
                    i += 1; tagged_circuit.occurrence = i
                #add data for a new (duplicate) circuit
                circuit = tagged_circuit

        # in other modes ("overwrite" and "aggregate"), strip off occurrence so duplicates are acted on appropriately
        elif circuit.occurrence is not None:
            stripped_circuit = circuit.copy()
            stripped_circuit.occurrence = None
            circuit = stripped_circuit

        return circuit

    def _add_explicit_repetition_counts(self):
        """
        Build internal repetition counts if they don't exist already.

        This method is usually unnecessary, as repetition counts are
        almost always build as soon as they are needed.

        Returns
        -------
        None
        """
        if self.repData is not None: return
        if self.bStatic:
            raise ValueError("Cannot build repetition counts in a static DataSet object")
        self.repData = []
        for oliAr in self.oliData:
            self.repData.append(_np.ones(len(oliAr), self.repType))

    def add_count_dict(self, circuit, count_dict, record_zero_counts=True, aux=None, update_ol=True):
        """
        Add a single circuit's counts to this DataSet

        Parameters
        ----------
        circuit : tuple or Circuit
            A tuple of operation labels specifying the circuit or a Circuit object

        count_dict : dict
            A dictionary with keys = outcome labels and values = counts

        record_zero_counts : bool, optional
            Whether zero-counts are actually recorded (stored) in this DataSet.
            If False, then zero counts are ignored, except for potentially
            registering new outcome labels.

        aux : dict, optional
            A dictionary of auxiliary meta information to be included with
            this set of data counts (associated with `circuit`).

        update_ol : bool, optional
            This argument is for internal use only and should be left as True.

        Returns
        -------
        None
        """
        if self.bStatic: raise ValueError("Cannot add data to a static DataSet object")

        #Convert input to an OutcomeLabelDict
        if isinstance(count_dict, _ld.OutcomeLabelDict):
            outcomeCounts = count_dict
        elif isinstance(count_dict, _OrderedDict):  # then don't sort keys
            outcomeCounts = _ld.OutcomeLabelDict(list(count_dict.items()))
        else:
            # sort key for deterministic ordering of *new* outcome labels)
            outcomeCounts = _ld.OutcomeLabelDict([
                (lbl, count_dict[lbl]) for lbl in sorted(list(count_dict.keys()))])

        outcomeLabelList = list(outcomeCounts.keys())
        countList = list(outcomeCounts.values())
        circuit = self._collisionaction_update_circuit(circuit)

        if self.collisionAction == "aggregate" and circuit in self:
            iNext = int(max(self[circuit].time)) + 1 \
                if (len(self[circuit].time) > 0) else 0
            timeStampList = [iNext] * len(countList)
            overwriteExisting = False
        else:
            timeStampList = [0] * len(countList)
            overwriteExisting = True

        self.add_raw_series_data(circuit, outcomeLabelList, timeStampList,
                                 countList, overwriteExisting, record_zero_counts,
                                 aux, update_ol, unsafe=True)
        #unsafe=True OK b/c outcome_label_list contains the keys of an OutcomeLabelDict

    def add_raw_series_data(self, circuit, outcome_label_list, time_stamp_list,
                            rep_count_list=None, overwrite_existing=True,
                            record_zero_counts=True, aux=None, update_ol=True,
                            unsafe=False):
        """
        Add a single circuit's counts to this DataSet

        Parameters
        ----------
        circuit : tuple or Circuit
            A tuple of operation labels specifying the circuit or a Circuit object

        outcome_label_list : list
            A list of outcome labels (strings or tuples).  An element's index
            links it to a particular time step (i.e. the i-th element of the
            list specifies the outcome of the i-th measurement in the series).

        time_stamp_list : list
            A list of floating point timestamps, each associated with the single
            corresponding outcome in `outcome_label_list`. Must be the same length
            as `outcome_label_list`.

        rep_count_list : list, optional
            A list of integer counts specifying how many outcomes of type given
            by `outcome_label_list` occurred at the time given by `time_stamp_list`.
            If None, then all counts are assumed to be 1.  When not None, must
            be the same length as `outcome_label_list`.

        overwrite_existing : bool, optional
            Whether to overwrite the data for `circuit` (if it exists).  If
            False, then the given lists are appended (added) to existing data.

        record_zero_counts : bool, optional
            Whether zero-counts (elements of `rep_count_list` that are zero) are
            actually recorded (stored) in this DataSet.  If False, then zero
            counts are ignored, except for potentially registering new outcome
            labels.

        aux : dict, optional
            A dictionary of auxiliary meta information to be included with
            this set of data counts (associated with `circuit`).

        update_ol : bool, optional
            This argument is for internal use only and should be left as True.

        unsafe : bool, optional
            When True, don't bother checking that outcome_label_list contains
            tuple-type outcome labels and automatically upgrading strings to
            1-tuples.  Only set this to True if you know what you're doing
            and need the marginally faster performance.

        Returns
        -------
        None
        """
        if self.bStatic: raise ValueError("Cannot add data to a static DataSet object")
        circuit = self._collisionaction_update_circuit(circuit)

        if unsafe:
            tup_outcomeLabelList = outcome_label_list
        else:
            #strings -> tuple outcome labels
            tup_outcomeLabelList = [_ld.OutcomeLabelDict.to_outcome(ol)
                                    for ol in outcome_label_list]

        #Add any new outcome labels
        self.add_outcome_labels(tup_outcomeLabelList, update_ol)

        oliArray = _np.array([self.olIndex[ol] for ol in tup_outcomeLabelList], self.oliType)
        timeArray = _np.array(time_stamp_list, self.timeType)
        assert(oliArray.shape == timeArray.shape), \
            "Outcome-label and time stamp lists must have the same length!"

        if rep_count_list is None:
            if self.repData is None: repArray = None
            else: repArray = _np.ones(len(oliArray), self.repType)
        else:
            if self.repData is None:
                #rep count data was given, but we're not currently holding repdata,
                # so we need to build this up for all existings sequences:
                self._add_explicit_repetition_counts()
            repArray = _np.array(rep_count_list, self.repType)

        if not record_zero_counts:
            # Go through repArray and remove any zeros, along with
            # corresponding elements of oliArray and timeArray
            mask = repArray != 0  # boolean array (note: == float comparison *is* desired)
            repArray = repArray[mask]
            oliArray = oliArray[mask]
            timeArray = timeArray[mask]

        if circuit in self.cirIndex:
            circuitIndx = self.cirIndex[circuit]
            if overwrite_existing:
                self.oliData[circuitIndx] = oliArray  # OVERWRITE existing time series
                self.timeData[circuitIndx] = timeArray  # OVERWRITE existing time series
                if repArray is not None: self.repData[circuitIndx] = repArray
            else:
                self.oliData[circuitIndx] = _np.concatenate((self.oliData[circuitIndx], oliArray))
                self.timeData[circuitIndx] = _np.concatenate((self.timeData[circuitIndx], timeArray))
                if repArray is not None:
                    self.repData[circuitIndx] = _np.concatenate((self.repData[circuitIndx], repArray))

        else:
            #add data for a new circuit
            assert(len(self.oliData) == len(self.timeData)), "OLI and TIME data are out of sync!!"
            circuitIndx = len(self.oliData)  # index of to-be-added circuit
            self.oliData.append(oliArray)
            self.timeData.append(timeArray)
            if repArray is not None: self.repData.append(repArray)
            self.cirIndex[circuit] = circuitIndx

        if aux is not None: self.add_auxiliary_info(circuit, aux)

    def update_ol(self):
        """
        Updates the internal outcome-label list in this dataset.

        Call this after calling add_count_dict(...) or add_raw_series_data(...)
        with `update_olIndex=False`.

        Returns
        -------
        None
        """
        self.ol = _OrderedDict([(i, sl) for (sl, i) in self.olIndex.items()])

    def add_series_data(self, circuit, count_dict_list, time_stamp_list,
                        overwrite_existing=True, record_zero_counts=True, aux=None):
        """
        Add a single circuit's counts to this DataSet

        Parameters
        ----------
        circuit : tuple or Circuit
            A tuple of operation labels specifying the circuit or a Circuit object

        count_dict_list : list
            A list of dictionaries holding the outcome-label:count pairs for each
            time step (times given by `time_stamp_list`.

        time_stamp_list : list
            A list of floating point timestamps, each associated with an entire
            dictionary of outcomes specified by `count_dict_list`.

        overwrite_existing : bool, optional
            If `True`, overwrite any existing data for the `circuit`.  If
            `False`, add the count data with the next non-negative integer
            timestamp.

        record_zero_counts : bool, optional
            Whether zero-counts (elements of the dictionaries in `count_dict_list` that
            are zero) are actually recorded (stored) in this DataSet.  If False, then
            zero counts are ignored, except for potentially registering new outcome
            labels.

        aux : dict, optional
            A dictionary of auxiliary meta information to be included with
            this set of data counts (associated with `circuit`).

        Returns
        -------
        None
        """
        expanded_outcomeList = []
        expanded_timeList = []
        expanded_repList = []

        for (cntDict, t) in zip(count_dict_list, time_stamp_list):
            if not isinstance(cntDict, _OrderedDict):
                ols = sorted(list(cntDict.keys()))
            else: ols = list(cntDict.keys())
            for ol in ols:  # loop over outcome labels
                expanded_outcomeList.append(ol)
                expanded_timeList.append(t)
                expanded_repList.append(cntDict[ol])  # could do this only for counts > 1
        return self.add_raw_series_data(circuit, expanded_outcomeList,
                                        expanded_timeList, expanded_repList,
                                        overwrite_existing, record_zero_counts, aux)

    def aggregate_outcomes(self, label_merge_dict, record_zero_counts=True):
        """
        Creates a DataSet which merges certain outcomes in this DataSet.

        Used, for example, to aggregate a 2-qubit 4-outcome DataSet into a 1-qubit 2-outcome
        DataSet.

        Parameters
        ----------
        label_merge_dict : dictionary
            The dictionary whose keys define the new DataSet outcomes, and whose items
            are lists of input DataSet outcomes that are to be summed together.  For example,
            if a two-qubit DataSet has outcome labels "00", "01", "10", and "11", and
            we want to ''aggregate out'' the second qubit, we could use label_merge_dict =
            {'0':['00','01'],'1':['10','11']}.  When doing this, however, it may be better
            to use :function:`filter_qubits` which also updates the circuits.

        record_zero_counts : bool, optional
            Whether zero-counts are actually recorded (stored) in the returned
            (merged) DataSet.  If False, then zero counts are ignored, except for
            potentially registering new outcome labels.

        Returns
        -------
        merged_dataset : DataSet object
            The DataSet with outcomes merged according to the rules given in label_merge_dict.
        """

        #static_self = self.copy()
        #static_self.done_adding_data()  # makes static, so we can assume this below

        # strings -> tuple outcome labels in keys and values of label_merge_dict
        to_outcome = _ld.OutcomeLabelDict.to_outcome  # shorthand
        label_merge_dict = {to_outcome(key): list(map(to_outcome, val))
                            for key, val in label_merge_dict.items()}

        merge_dict_old_outcomes = [outcome for sublist in label_merge_dict.values() for outcome in sublist]
        if not set(self.outcome_labels()).issubset(merge_dict_old_outcomes):
            raise ValueError(
                "`label_merge_dict` must account for all the outcomes in original dataset."
                " It's missing directives for:\n%s" %
                '\n'.join(set(map(str, self.outcome_labels())) - set(map(str, merge_dict_old_outcomes)))
            )

        new_outcomes = sorted(list(label_merge_dict.keys()))
        new_outcome_indices = _OrderedDict([(ol, i) for i, ol in enumerate(new_outcomes)])
        nNewOutcomes = len(new_outcomes)

        #Count the number of time steps so we allocate enough space
        nSteps = 0
        for key, dsrow in self.items():
            cur_t = None
            for t in dsrow.time:
                if t != cur_t:
                    nSteps += 1
                    cur_t = t

        #idea is that we create oliData, timeData, repData, and circuitIndices for the
        # merged dataset rather than looping over insertion, as this is faster
        oliData = _np.empty(nSteps * nNewOutcomes, self.oliType)
        repData = _np.empty(nSteps * nNewOutcomes, self.repType)
        timeData = _np.empty(nSteps * nNewOutcomes, self.timeType)

        oli_map = {}  # maps old outcome label indices to new ones
        for new_outcome, old_outcome_list in label_merge_dict.items():
            new_index = new_outcome_indices[new_outcome]
            for old_outcome in old_outcome_list:
                oli_map[self.olIndex[old_outcome]] = new_index

        #Future - when record_zero_counts=False these may not need to be so large
        new_olis = _np.array(range(nNewOutcomes), _np.int64)
        new_cnts = _np.zeros(nNewOutcomes, self.repType)

        if record_zero_counts:
            def add_cnts(t, cnts, offset):  # cnts is an array here
                new_cnts[:] = 0
                for nonzero_oli, cnt in cnts.items():
                    new_cnts[nonzero_oli] = cnt
                timeData[offset:offset + nNewOutcomes] = t
                oliData[offset:offset + nNewOutcomes] = new_olis
                repData[offset:offset + nNewOutcomes] = new_cnts  # a length-nNewOutcomes array
                return nNewOutcomes

        else:
            def add_cnts(t, cnts, offset):  # cnts is a dict here
                nNewCnts = len(cnts)
                #new_olis = _np.empty(nNewCnts, _np.int64)
                #new_cnts = _np.empty(nNewCnts, self.repType)
                for ii, (nonzero_oli, cnt) in enumerate(cnts.items()):
                    new_olis[ii] = nonzero_oli
                    new_cnts[ii] = cnt
                timeData[offset:offset + nNewCnts] = t
                oliData[offset:offset + nNewCnts] = new_olis[0:nNewCnts]
                repData[offset:offset + nNewCnts] = new_cnts[0:nNewCnts]
                return nNewCnts  # return the number of added counts

        k = 0  # beginning of current circuit data in 1D arrays: oliData, timeData, repData
        circuitIndices = _OrderedDict()
        for key, dsrow in self.items():

            last_t = dsrow.time[0]

            #Below code is faster version of: mapped_oli = [oli_map[x] for x in dsrow.oli]
            mapped_oli = dsrow.oli.copy()
            for from_oli, to_oli in oli_map.items():
                mapped_oli[dsrow.oli == from_oli] = to_oli

            reps = _np.ones(len(dsrow.time), self.timeType) if (self.repData is None) else dsrow.reps
            cnts = _defaultdict(lambda: 0)

            i = 0  # offset to current timeslice
            for oli, t, reps in zip(mapped_oli, dsrow.time, reps):
                if t != last_t:
                    i += add_cnts(last_t, cnts, k + i)
                    last_t = t; cnts.clear()
                cnts[oli] += reps
            if len(cnts) > 0:
                i += add_cnts(last_t, cnts, k + i)

            circuitIndices[key] = slice(k, k + i)
            k += i

        merged_dataset = DataSet(oliData[0:k], timeData[0:k], repData[0:k], circuit_indices=circuitIndices,
                                 outcome_label_indices=new_outcome_indices, static=True)
        return merged_dataset

    def aggregate_std_nqubit_outcomes(self, qubit_indices_to_keep, record_zero_counts=True):
        """
        Creates a DataSet which merges certain outcomes in this DataSet.

        Used, for example, to aggregate a 2-qubit 4-outcome DataSet into a 1-qubit 2-outcome
        DataSet.  This assumes that outcome labels are in the standard format
        whereby each qubit corresponds to a single '0' or '1' character.

        Parameters
        ----------
        qubit_indices_to_keep : list
            A list of integers specifying which qubits should be kept, that is,
            *not* aggregated.

        record_zero_counts : bool, optional
            Whether zero-counts are actually recorded (stored) in the returned
            (merged) DataSet.  If False, then zero counts are ignored, except for
            potentially registering new outcome labels.

        Returns
        -------
        merged_dataset : DataSet object
            The DataSet with outcomes merged.
        """

        label_merge_dict = _defaultdict(list)
        for ol, i in self.olIndex.items():
            assert(len(ol) == 1), "Cannot merge non-simple outcomes!"  # should be a 1-tuple
            reduced = (''.join([ol[0][i] for i in qubit_indices_to_keep]),)  # a tuple
            label_merge_dict[reduced].append(ol)
        label_merge_dict = dict(label_merge_dict)  # return a *normal* dict

        new_outcomes = sorted(list(label_merge_dict.keys()))
        new_outcome_indices = _OrderedDict([(ol, i) for i, ol in enumerate(new_outcomes)])
        nNewOutcomes = len(new_outcomes)

        #Count the number of time steps so we allocate enough space
        nSteps = 0
        for dsrow in self.values():
            cur_t = None
            for t in dsrow.time:
                if t != cur_t:
                    nSteps += 1
                    cur_t = t

        #idea is that we create oliData, timeData, repData, and circuitIndices for the
        # merged dataset rather than looping over insertion, as this is faster
        oliData = _np.empty(nSteps * nNewOutcomes, self.oliType)
        repData = _np.empty(nSteps * nNewOutcomes, self.repType)
        timeData = _np.empty(nSteps * nNewOutcomes, self.timeType)

        oli_map = {}  # maps old outcome label indices to new ones
        for new_outcome, old_outcome_list in label_merge_dict.items():
            new_index = new_outcome_indices[new_outcome]
            for old_outcome in old_outcome_list:
                oli_map[self.olIndex[old_outcome]] = new_index

        #Future - when record_zero_counts=False these may not need to be so large
        new_olis = _np.array(range(nNewOutcomes), _np.int64)
        new_cnts = _np.zeros(nNewOutcomes, self.repType)

        if record_zero_counts:
            def add_cnts(t, cnts, offset):  # cnts is an array here
                new_cnts[:] = 0
                for nonzero_oli, cnt in cnts.items():
                    new_cnts[nonzero_oli] = cnt
                timeData[offset:offset + nNewOutcomes] = t
                oliData[offset:offset + nNewOutcomes] = new_olis
                repData[offset:offset + nNewOutcomes] = new_cnts  # a length-nNewOutcomes array
                return nNewOutcomes

        else:
            def add_cnts(t, cnts, offset):  # cnts is a dict here
                nNewCnts = len(cnts)
                #new_olis = _np.empty(nNewCnts, _np.int64)
                #new_cnts = _np.empty(nNewCnts, self.repType)
                for ii, (nonzero_oli, cnt) in enumerate(cnts.items()):
                    new_olis[ii] = nonzero_oli
                    new_cnts[ii] = cnt
                timeData[offset:offset + nNewCnts] = t
                oliData[offset:offset + nNewCnts] = new_olis[0:nNewCnts]
                repData[offset:offset + nNewCnts] = new_cnts[0:nNewCnts]
                return nNewCnts  # return the number of added counts

        k = 0  # beginning of current circuit data in 1D arrays: oliData, timeData, repData
        circuitIndices = _OrderedDict()
        for key, dsrow in self.items():

            last_t = dsrow.time[0] if len(dsrow.time) > 0 else None

            if len(dsrow.oli) < len(oli_map):
                mapped_oli = _np.array([oli_map[x] for x in dsrow.oli])
            else:
                mapped_oli = dsrow.oli.copy()
                for from_oli, to_oli in oli_map.items():
                    mapped_oli[dsrow.oli == from_oli] = to_oli

            reps = _np.ones(len(dsrow.time), self.timeType) if (self.repData is None) else dsrow.reps
            cnts = _defaultdict(lambda: 0)

            i = 0  # offset to current timeslice
            for oli, t, reps in zip(mapped_oli, dsrow.time, reps):
                if t != last_t:
                    i += add_cnts(last_t, cnts, k + i)
                    last_t = t; cnts.clear()
                cnts[oli] += reps
            if len(cnts) > 0:
                i += add_cnts(last_t, cnts, k + i)

            circuitIndices[key] = slice(k, k + i)
            k += i

        merged_dataset = DataSet(oliData[0:k], timeData[0:k], repData[0:k], circuit_indices=circuitIndices,
                                 outcome_label_indices=new_outcome_indices, static=True)
        return merged_dataset

    def add_auxiliary_info(self, circuit, aux):
        """
        Add auxiliary meta information to `circuit`.

        Parameters
        ----------
        circuit : tuple or Circuit
            A tuple of operation labels specifying the circuit or a Circuit object

        aux : dict, optional
            A dictionary of auxiliary meta information to be included with
            this set of data counts (associated with `circuit`).

        Returns
        -------
        None
        """
        self.auxInfo[circuit].clear()  # needed? (could just update?)
        self.auxInfo[circuit].update(aux)

    def add_counts_from_dataset(self, other_data_set):
        """
        Append another DataSet's data to this DataSet

        Parameters
        ----------
        other_data_set : DataSet
            The dataset to take counts from.

        Returns
        -------
        None
        """
        return self.add_series_from_dataset(other_data_set)

    def add_series_from_dataset(self, other_data_set):
        """
        Append another DataSet's series data to this DataSet

        Parameters
        ----------
        other_data_set : DataSet
            The dataset to take time series data from.

        Returns
        -------
        None
        """
        if self.bStatic: raise ValueError("Cannot add data to a static DataSet object")
        for circuit, dsRow in other_data_set.items():
            self.add_raw_series_data(circuit, dsRow.outcomes, dsRow.time, dsRow.reps, False)

    def meantimestep(self):
        """
        The mean time-step, averaged over the time-step for each circuit and over circuits.

        Returns
        -------
        float
        """
        timesteps = []
        for key in self.keys():
            timesteps.append(self[key].meantimestep())

        return _np.mean(timesteps)

    def has_constant_totalcounts_pertime(self):
        """
        True if the data for every circuit has the same number of total counts at every data collection time.

        This will return True if there is a different number of total counts
        per circuit (i.e., after aggregating over time), as long as every
        circuit has the same total counts per time step (this will happen
        when the number of time-steps varies between circuit).

        Returns
        -------
        bool
        """
        for key in self.keys():
            numtotalcountspertime = None
            dsrow = self[key]
            if not dsrow.has_constant_totalcounts():
                return False
            if numtotalcountspertime is None:
                numtotalcountspertime = dsrow.totalcounts_per_timestep()
            else:
                if numtotalcountspertime != dsrow.totalcounts_per_timestep():
                    return False

        return True

    def totalcounts_pertime(self):
        """
        Total counts per time, if this is constant over times and circuits.

        When that doesn't hold, an error is raised.

        Returns
        -------
        float or int
        """
        self.has_constant_totalcounts_pertime()
        key = list(self.keys())[0]
        totalcountspertime = self[key].totalcounts_per_timestep()

        return totalcountspertime

    def has_constant_totalcounts(self):
        """
        `True` if the data for every circuit has the same number of total counts.

        Returns
        -------
        bool
        """
        reps = []
        for key in self.keys():
            reps.append(sum(list(self[key].counts.values())))
        firstrep = reps[0]
        fixedtotalcounts = all([firstrep == i for i in reps])

        return fixedtotalcounts

    def has_trivial_timedependence(self):
        """
        `True` if all the data in this DataSet occurs at time 0.

        Returns
        -------
        bool
        """
        return all([_np.all(self.timeData[gsi] == 0) for gsi in self.cirIndex.values()])

    def __str__(self):
        return self.to_str()

    def to_str(self, mode="auto"):
        """
        Render this DataSet as a string.

        Parameters
        ----------
        mode : {"auto","time-dependent","time-independent"}
            Whether to display the data as time-series of outcome counts
            (`"time-dependent"`) or to report per-outcome counts aggregated over
            time (`"time-independent"`).  If `"auto"` is specified, then the
            time-independent mode is used only if all time stamps in the
            DataSet are equal to zero (trivial time dependence).

        Returns
        -------
        str
        """
        if mode == "auto":
            mode = "time-independent" if self.has_trivial_timedependence() else "time-dependent"

        assert(mode in ('time-dependent', 'time-independent')), "Invalid `mode` argument: %s" % mode

        if mode == "time-dependent":
            s = "Dataset outcomes: " + str(self.olIndex) + "\n"
            for circuit in self:  # tuple-type operation label strings are keys
                s += "%s :\n%s\n" % (circuit.str, self[circuit].to_str(mode))
            return s + "\n"
        else:  # time-independent
            s = ""
            for circuit in self:  # tuple-type operation label strings are keys
                s += "%s  :  %s\n" % (circuit.str, self[circuit].to_str(mode))
            return s + "\n"

    def truncate(self, list_of_circuits_to_keep, missing_action='raise'):
        """
        Create a truncated dataset comprised of a subset of the circuits in this dataset.

        Parameters
        ----------
        list_of_circuits_to_keep : list of (tuples or Circuits)
            A list of the circuits for the new returned dataset.  If a
            circuit is given in this list that isn't in the original
            data set, `missing_action` determines the behavior.

        missing_action : {"raise","warn","ignore"}
            What to do when a string in `list_of_circuits_to_keep` is not in
            the data set (raise a KeyError, issue a warning, or do nothing).

        Returns
        -------
        DataSet
            The truncated data set.
        """
        missingStrs = []  # to issue warning - only used if missing_action=="warn"
        if self.bStatic:
            circuitIndices = []
            circuits = []
            used_oli = set()
            for opstr in list_of_circuits_to_keep:
                circuit = opstr if isinstance(opstr, _cir.Circuit) else _cir.Circuit(opstr)

                if circuit not in self.cirIndex:
                    if missing_action == "raise":
                        raise KeyError(("Circuit %s was not found in "
                                        "dataset being truncated and "
                                        "`missing_action` == 'raise'") % str(circuit))
                    elif missing_action == "warn":
                        missingStrs.append(circuit)
                        continue
                    elif missing_action == "ignore":
                        continue
                    else:
                        raise ValueError("Invalid `missing_action`: %s" % str(missing_action))

                #only keep track of circuits if they could be different from list_of_circuits_to_keep
                if missing_action != "raise": circuits.append(circuit)
                i = self.cirIndex[circuit]
                circuitIndices.append(i)
                used_oli.update(self.oliData[i])

            if missing_action == "raise": circuits = list_of_circuits_to_keep
            trunc_cirIndex = _OrderedDict(zip(circuits, circuitIndices))
            trunc_olIndex = _OrderedDict([(self.ol[i], i) for i in sorted(used_oli)])
            trunc_dataset = DataSet(self.oliData, self.timeData, self.repData,
                                    circuit_indices=trunc_cirIndex,
                                    outcome_label_indices=trunc_olIndex, static=True)  # reference (don't copy) counts

        else:
            trunc_dataset = DataSet(outcome_labels=[])  # let outcome labels be added automatically
            for opstr in _lt.remove_duplicates(list_of_circuits_to_keep):
                circuit = opstr if isinstance(opstr, _cir.Circuit) else _cir.Circuit(opstr)
                if circuit in self.cirIndex:
                    circuitIndx = self.cirIndex[circuit]
                    repData = self.repData[circuitIndx].copy() if (self.repData is not None) else None
                    trunc_dataset.add_raw_series_data(circuit,
                                                      [self.ol[i] for i in self.oliData[circuitIndx]],
                                                      self.timeData[circuitIndx].copy(),
                                                      repData, unsafe=True)  # copy so truncated dataset can be modified
                elif missing_action == "raise":
                    raise KeyError(("Circuit %s was not found in "
                                    "dataset being truncated and "
                                    "`missing_action` == 'raise'") % str(circuit))
                elif missing_action == "warn":
                    missingStrs.append(circuit)
                elif missing_action != "ignore":
                    raise ValueError("Invalid `missing_action`: %s" % str(missing_action))

        if len(missingStrs) > 0:
            missingStrs.append("...")  # so elipses are shown when there's more strings
            _warnings.warn(("DataSet.truncate(...) was given %s strings to keep"
                            " that weren't in the original dataset:\n%s") %
                           (len(missingStrs) - 1, "\n".join(map(str, missingStrs[0:10]))))

        return trunc_dataset

    def time_slice(self, start_time, end_time, aggregate_to_time=None):
        """
        Creates a DataSet by aggregating the counts within the [`start_time`,`end_time`) interval.

        Parameters
        ----------
        start_time : float
            The starting time.

        end_time : float
            The ending time.

        aggregate_to_time : float, optional
            If not None, a single timestamp to give all the data in
            the specified range, resulting in time-independent
            `DataSet`.  If None, then the original timestamps are
            preserved.

        Returns
        -------
        DataSet
        """
        tot = 0
        ds = DataSet(outcome_label_indices=self.olIndex)
        for opStr, dsRow in self.items():

            if dsRow.reps is None:
                reps = _np.ones(dsRow.oli.shape, self.repType)
            else: reps = dsRow.reps

            count_dict = {ol: 0 for ol in self.olIndex.keys()}
            times = []; ols = []; repCnts = []
            for oli, t, rep in zip(dsRow.oli, dsRow.time, reps):

                ol = self.ol[oli]  # index -> outcome label
                if start_time <= t < end_time:
                    if aggregate_to_time is not None:
                        count_dict[ol] += rep
                    else:
                        times.append(t)
                        ols.append(ol)
                        repCnts.append(rep)
                    tot += rep

            if aggregate_to_time is not None:
                ols = [k for k in count_dict.keys() if count_dict[k] > 0]
                repCnts = [count_dict[k] for k in ols]
                times = [aggregate_to_time] * len(repCnts)

            ds.add_raw_series_data(opStr, ols, times, repCnts)

        if tot == 0:
            _warnings.warn("No counts in the requested time range: empty DataSet created")
        ds.done_adding_data()
        return ds

    def process_circuits(self, processor_fn, aggregate=False):
        """
        Manipulate this DataSet's circuits (keys) according to `processor_fn`.

        All of the DataSet's gate sequence labels are updated by running each
        through `processor_fn`.  This can be useful when "tracing out" qubits
        in a dataset containing multi-qubit data.

        Parameters
        ----------
        processor_fn : function
            A function which takes a single Circuit argument and returns
            another (or the same) Circuit.  This function may also return
            `None`, in which case the data for that string is deleted.

        aggregate : bool, optional
            When `True`, aggregate the data for ciruits that `processor_fn`
            assigns to the same "new" circuit.  When `False`, use the data
            from the *last* original circuit that maps to a given "new" circuit.

        Returns
        -------
        None
        """
        if self.bStatic: raise ValueError("Cannot process_circuits on a static DataSet object")

        to_delete = []
        new_cirIndex = _OrderedDict()
        for opstr, indx in self.cirIndex.items():
            new_gstr = processor_fn(opstr)
            if new_gstr is None:
                to_delete.append(indx)
            elif new_gstr not in new_cirIndex or not aggregate:
                assert(isinstance(new_gstr, _cir.Circuit)), "`processor_fn` must return a Circuit!"
                new_cirIndex[new_gstr] = indx
            else:  # aggregate data from indx --> new_cirIndex[new_gstr]
                # A subset of what is in add_raw_series_data(...), but we
                # don't need to do many of the checks there since the
                # incoming data is known to have no new outcome labels, etc.
                assert(isinstance(new_gstr, _cir.Circuit)), "`processor_fn` must return a Circuit!"
                iSrc, iDest = indx, new_cirIndex[new_gstr]
                self.oliData[iDest] = _np.concatenate((self.oliData[iDest], self.oliData[iSrc]))
                self.timeData[iDest] = _np.concatenate((self.timeData[iDest], self.timeData[iSrc]))
                if self.repData is not None:
                    self.repData[iDest] = _np.concatenate((self.repData[iDest], self.repData[iSrc]))
                #FUTURE: just add counts for same timestamp & same outcome
                #  label data? (and in add_raw_series_data(...) too).

                # mark indx for deletion (don't do it yet, as this will
                # mess up the values in new_cirIndex)
                to_delete.append(indx)

        self.cirIndex = new_cirIndex
        self._remove(to_delete)

        #Note: self.cnt_cache just remains None (a non-static DataSet)

        #Process self.auxInfo
        auxInfo = _defaultdict(dict)
        for opstr in self.auxInfo.keys():
            new_gstr = processor_fn(opstr)
            if new_gstr is None:
                continue
            elif new_gstr not in auxInfo or not aggregate:
                auxInfo[new_gstr] = self.auxInfo[opstr]
            else:  # "aggregate" auxinfo by merging dictionaries
                #FUTURE: better merging - do something for key collisions?
                auxInfo[new_gstr].update(self.auxInfo[opstr])
        self.auxInfo = auxInfo

    def remove(self, circuits, missing_action="raise"):
        """
        Remove (delete) the data for `circuits` from this :class:`DataSet`.

        Parameters
        ----------
        circuits : iterable
            An iterable over Circuit-like objects specifying the keys
            (circuits) to remove.

        missing_action : {"raise","warn","ignore"}
            What to do when a string in `circuits` is not in this data
            set (raise a KeyError, issue a warning, or do nothing).

        Returns
        -------
        None
        """
        missingStrs = []  # to issue warning - only used if missing_action=="warn"
        gstr_indices = []; auxkeys_to_remove = []
        for opstr in circuits:
            if not isinstance(opstr, _cir.Circuit):
                opstr = _cir.Circuit(opstr)

            if opstr in self:
                gstr_indices.append(self.cirIndex[opstr])
                if opstr in self.auxInfo:
                    auxkeys_to_remove.append(opstr)
            elif missing_action == "raise":
                raise KeyError(("Circuit %s does not exist and therefore "
                                "cannot be removed when `missing_action` == "
                                "'raise'") % str(opstr))
            elif missing_action == "warn":
                missingStrs.append(opstr)
            elif missing_action != "ignore":
                raise ValueError("Invalid `missing_action`: %s" % str(missing_action))

        # the actual removal operations
        self._remove(gstr_indices)
        for ky in auxkeys_to_remove:
            del self.auxInfo[ky]

        if len(missingStrs) > 0:  # Print a warning with missing strings
            missingStrs.append("...")  # so elipses are shown when there's more strings
            _warnings.warn(("DataSet.remove(...) cannot remove %s strings because"
                            " they don't exist in the original dataset:\n%s") %
                           (len(missingStrs) - 1, "\n".join(map(str, missingStrs[0:10]))))

    def _remove(self, gstr_indices):
        """ Removes the data in indices given by gstr_indices """
        if self.bStatic: raise ValueError("Cannot _remove on a static DataSet object")

        #Removing elements from oli_data, time_data, and rep_data is easy since
        # these are just lists.  Hard part is adjusting cirIndex values: we
        # need to subtract k from index n, where k is the number of indices
        # in `gstr_indices` less than n.
        inds = sorted(list(gstr_indices))

        #remove indices from lists (high->low)
        for i in reversed(inds):
            del self.oliData[i]
            del self.timeData[i]
            if self.repData:
                del self.repData[i]

        #remove elements of cirIndex assoc. w/deleted indices
        keys_to_delete = []; inds_set = set(inds)
        for k, v in self.cirIndex.items():
            if v in inds_set:
                keys_to_delete.append(k)
        for k in keys_to_delete:
            del self.cirIndex[k]

        #adjust remaining indices in cirIndex
        inds_ar = _np.array(inds, _np.int64)
        for k in self.cirIndex.keys():
            cnt = _bisect.bisect(inds_ar, self.cirIndex[k])  # cnt == number of removed
            self.cirIndex[k] -= cnt                         # indices < self.cirIndex[k]

    def copy(self):
        """
        Make a copy of this DataSet.

        Returns
        -------
        DataSet
        """
        if self.bStatic:
            return self  # doesn't need to be copied since data can't change
        else:
            copyOfMe = DataSet(outcome_labels=self.outcome_labels(),
                               collision_action=self.collisionAction)
            copyOfMe.cirIndex = _copy.deepcopy(self.cirIndex)
            copyOfMe.oliData = [el.copy() for el in self.oliData]
            copyOfMe.timeData = [el.copy() for el in self.timeData]
            if self.repData is not None:
                copyOfMe.repData = [el.copy() for el in self.repData]
            else: copyOfMe.repData = None

            copyOfMe.oliType = self.oliType
            copyOfMe.timeType = self.timeType
            copyOfMe.repType = self.repType
            copyOfMe.cnt_cache = None
            copyOfMe.auxInfo = self.auxInfo.copy()
            return copyOfMe

    def copy_nonstatic(self):
        """
        Make a non-static copy of this DataSet.

        Returns
        -------
        DataSet
        """
        if self.bStatic:
            copyOfMe = DataSet(outcome_labels=self.outcome_labels(),
                               collision_action=self.collisionAction)
            copyOfMe.cirIndex = _OrderedDict([(opstr, i) for i, opstr in enumerate(self.cirIndex.keys())])
            copyOfMe.oliData = []
            copyOfMe.timeData = []
            copyOfMe.repData = None if (self.repData is None) else []
            for slc in self.cirIndex.values():
                copyOfMe.oliData.append(self.oliData[slc].copy())
                copyOfMe.timeData.append(self.timeData[slc].copy())
                if self.repData is not None:
                    copyOfMe.repData.append(self.repData[slc].copy())

            copyOfMe.oliType = self.oliType
            copyOfMe.timeType = self.timeType
            copyOfMe.repType = self.repType
            copyOfMe.cnt_cache = None
            copyOfMe.auxInfo = self.auxInfo.copy()
            return copyOfMe
        else:
            return self.copy()

    def done_adding_data(self):
        """
        Promotes a non-static DataSet to a static (read-only) DataSet.

        This method should be called after all data has been added.

        Returns
        -------
        None
        """
        if self.bStatic: return
        #Convert normal dataset to static mode.
        #  olIndex stays the same
        #  cirIndex changes to hold slices into 1D arrays
        #  oli_data, time_data, & rep_data change from being lists of arrays to
        #    single 1D arrays.

        if len(self.oliData) > 0:
            new_cirIndex = _OrderedDict()
            curIndx = 0
            to_concat_oli = []
            to_concat_time = []
            to_concat_rep = []
            for circuit, indx in self.cirIndex.items():
                seriesLen = len(self.oliData[indx])

                to_concat_oli.append(self.oliData[indx])  # just build up lists of
                to_concat_time.append(self.timeData[indx])  # reference, not copies
                assert(seriesLen == len(self.timeData[indx])), "TIME & OLI out of sync!"

                if self.repData is not None:
                    to_concat_rep.append(self.repData[indx])
                    assert(seriesLen == len(self.repData[indx])), "REP & OLI out of sync!"

                new_cirIndex[circuit] = slice(curIndx, curIndx + seriesLen)
                curIndx += seriesLen

            self.cirIndex = new_cirIndex
            self.oliData = _np.concatenate(to_concat_oli)
            self.timeData = _np.concatenate(to_concat_time)
            if self.repData is not None:
                self.repData = _np.concatenate(to_concat_rep)

        else:
            #leave cirIndex alone (should be empty anyway?)
            self.oliData = _np.empty((0,), self.oliType)
            self.timeData = _np.empty((0,), self.timeType)
            if self.repData is not None:
                self.repData = _np.empty((0,), self.repType)

        self.cnt_cache = {opstr: _ld.OutcomeLabelDict() for opstr in self.cirIndex}
        self.bStatic = True
        self.uuid = _uuid.uuid4()

    def __getstate__(self):
        toPickle = {'cirIndexKeys': list(map(_cir.CompressedCircuit, self.cirIndex.keys())),
                    'cirIndexVals': list(self.cirIndex.values()),
                    'olIndex': self.olIndex,
                    'olIndex_max': self.olIndex_max,
                    'ol': self.ol,
                    'bStatic': self.bStatic,
                    'oliData': self.oliData,
                    'timeData': self.timeData,
                    'repData': self.repData,
                    'oliType': _np.dtype(self.oliType).str,
                    'timeType': _np.dtype(self.timeType).str,
                    'repType': _np.dtype(self.repType).str,
                    'collisionAction': self.collisionAction,
                    'uuid': self.uuid,
                    'auxInfo': self.auxInfo,
                    'comment': self.comment}
        return toPickle

    def __setstate__(self, state_dict):
        bStatic = state_dict['bStatic']

        if "gsIndexKeys" in state_dict:
            _warnings.warn("Unpickling a deprecated-format DataSet.  Please re-save/pickle asap.")
            cirIndexKeys = [cgstr.expand() for cgstr in state_dict['gsIndexKeys']]
            cirIndex = _OrderedDict(list(zip(cirIndexKeys, state_dict['gsIndexVals'])))
        else:
            cirIndexKeys = [cgstr.expand() for cgstr in state_dict['cirIndexKeys']]
            cirIndex = _OrderedDict(list(zip(cirIndexKeys, state_dict['cirIndexVals'])))

        if "slIndex" in state_dict:
            #print("DB: UNPICKLING AN OLD DATASET"); print("Keys = ",state_dict.keys())
            _warnings.warn("Unpickling a *very* deprecated-format DataSet.  Please re-save/pickle asap.")

            #Turn spam labels into outcome labels
            self.cirIndex = _OrderedDict()
            self.olIndex = _OrderedDict([((str(sl),), i) for sl, i in state_dict['slIndex'].items()])
            self.ol = _OrderedDict([(i, ol) for (ol, i) in self.olIndex.items()])
            self.oliData = []
            self.timeData = []
            self.repData = []
            self.comment = ''

            self.oliType = Oindex_type
            self.timeType = Time_type
            self.repType = Repcount_type

            self.bStatic = False  # for adding data
            for opstr, indx in cirIndex.items():
                count_row = state_dict['counts'][indx]
                count_dict = _OrderedDict([(ol, count_row[i]) for ol, i in self.olIndex.items()])
                self.add_count_dict(opstr, count_dict)
            if not self.bStatic: self.done_adding_data()

        else:  # Normal case
            self.bStatic = bStatic
            self.cirIndex = cirIndex
            self.olIndex = state_dict['olIndex']
            self.olIndex_max = state_dict.get('olIndex_max',
                                              max(self.olIndex.values()) if len(self.olIndex) > 0 else -1)
            self.ol = state_dict['ol']
            self.oliData = state_dict['oliData']
            self.timeData = state_dict['timeData']
            self.repData = state_dict['repData']
            self.oliType = _np.dtype(state_dict['oliType'])
            self.timeType = _np.dtype(state_dict['timeType'])
            self.repType = _np.dtype(state_dict['repType'])
            self.comment = state_dict.get('comment', '')
            if bStatic:  # always empty - don't save this, just init
                self.cnt_cache = {opstr: _ld.OutcomeLabelDict() for opstr in self.cirIndex}
            else: self.cnt_cache = None

        self.auxInfo = state_dict.get('auxInfo', _defaultdict(dict))
        if not isinstance(self.auxInfo, _defaultdict) and isinstance(self.auxInfo, dict):
            self.auxInfo = _defaultdict(dict, self.auxInfo)
            # some types of serialization (e.g. JSON) just save a *normal* dict
            # so promote to a defaultdict if needed..

        self.collisionAction = state_dict.get('collisionAction', 'aggregate')
        self.uuid = state_dict.get('uuid', None)

    def save(self, file_or_filename):
        """
        Save this DataSet to a file.

        Parameters
        ----------
        file_or_filename : string or file object
            If a string,  interpreted as a filename.  If this filename ends
            in ".gz", the file will be gzip compressed.

        Returns
        -------
        None
        """

        toPickle = {'cirIndexKeys': list(map(_cir.CompressedCircuit, self.cirIndex.keys())) if self.cirIndex else [],
                    'cirIndexVals': list(self.cirIndex.values()) if self.cirIndex else [],
                    'olIndex': self.olIndex,
                    'olIndex_max': self.olIndex_max,
                    'ol': self.ol,
                    'bStatic': self.bStatic,
                    'oliType': self.oliType,
                    'timeType': self.timeType,
                    'repType': self.repType,
                    'useReps': bool(self.repData is not None),
                    'collisionAction': self.collisionAction,
                    'uuid': self.uuid,
                    'auxInfo': self.auxInfo,
                    'comment': self.comment}  # Don't pickle counts numpy data b/c it's inefficient
        if not self.bStatic: toPickle['nRows'] = len(self.oliData)

        bOpen = isinstance(file_or_filename, str)
        if bOpen:
            if file_or_filename.endswith(".gz"):
                import gzip as _gzip
                f = _gzip.open(file_or_filename, "wb")
            else:
                f = open(file_or_filename, "wb")
        else:
            f = file_or_filename

        _pickle.dump(toPickle, f)
        if self.bStatic:
            _np.save(f, self.oliData)
            _np.save(f, self.timeData)
            if self.repData is not None:
                _np.save(f, self.repData)
        else:
            for row in self.oliData: _np.save(f, row)
            for row in self.timeData: _np.save(f, row)
            if self.repData is not None:
                for row in self.repData: _np.save(f, row)
        if bOpen: f.close()

    def load(self, file_or_filename):
        """
        Load DataSet from a file, clearing any data is contained previously.

        Parameters
        ----------
        file_or_filename : str or buffer
            The file or filename to load from.

        Returns
        -------
        None
        """
        bOpen = isinstance(file_or_filename, str)
        if bOpen:
            if file_or_filename.endswith(".gz"):
                import gzip as _gzip
                f = _gzip.open(file_or_filename, "rb")
            else:
                f = open(file_or_filename, "rb")
        else:
            f = file_or_filename

        with _compat.patched_uuid():
            state_dict = _pickle.load(f)

        if "gsIndexKeys" in state_dict:
            _warnings.warn("Loading a deprecated-format DataSet.  Please re-save asap.")
            state_dict['cirIndexKeys'] = state_dict['gsIndexKeys']
            state_dict['cirIndexVals'] = state_dict['gsIndexVals']
            del state_dict['gsIndexKeys']
            del state_dict['gsIndexVals']

        def expand(x):  # to be backward compatible
            """ Expand a compressed circuit """
            if isinstance(x, _cir.CompressedCircuit): return x.expand()
            elif hasattr(x, '__class__') and x.__class__.__name__ == "dummy_CompressedGateString":
                return _cir.Circuit(_cir.CompressedCircuit.expand_op_label_tuple(x._tup), stringrep=x.str)
                #for backward compatibility, needed for Python2 only, which doesn't call __new__ when
                # unpickling protocol=0 (the default) info.
            else:
                _warnings.warn("Deprecated dataset format.  Please re-save "
                               "this dataset soon to avoid future incompatibility.")
                return _cir.Circuit(_cir.CompressedCircuit.expand_op_label_tuple(x))
        cirIndexKeys = [expand(cgstr) for cgstr in state_dict['cirIndexKeys']]

        #cirIndexKeys = [ cgs.expand() for cgs in state_dict['cirIndexKeys'] ]
        self.cirIndex = _OrderedDict(list(zip(cirIndexKeys, state_dict['cirIndexVals'])))
        self.olIndex = state_dict['olIndex']
        self.olIndex_max = state_dict.get('olIndex_max',
                                          max(self.olIndex.values()) if len(self.olIndex) > 0 else -1)
        self.ol = state_dict['ol']
        self.bStatic = state_dict['bStatic']
        self.oliType = state_dict['oliType']
        self.timeType = state_dict['timeType']
        self.repType = state_dict['repType']
        self.collisionAction = state_dict['collisionAction']
        self.uuid = state_dict['uuid']
        self.auxInfo = state_dict.get('auxInfo', _defaultdict(dict))  # backward compat
        self.comment = state_dict.get('comment', '')  # backward compat

        useReps = state_dict['useReps']

        if self.bStatic:
            self.oliData = _np.lib.format.read_array(f)  # _np.load(f) doesn't play nice with gzip
            self.timeData = _np.lib.format.read_array(f)  # _np.load(f) doesn't play nice with gzip
            if useReps:
                self.repData = _np.lib.format.read_array(f)  # _np.load(f) doesn't play nice with gzip
            self.cnt_cache = {opstr: _ld.OutcomeLabelDict() for opstr in self.cirIndex}  # init cnt_cache afresh
        else:
            self.oliData = []
            for _ in range(state_dict['nRows']):
                self.oliData.append(_np.lib.format.read_array(f))  # _np.load(f) doesn't play nice with gzip

            self.timeData = []
            for _ in range(state_dict['nRows']):
                self.timeData.append(_np.lib.format.read_array(f))  # _np.load(f) doesn't play nice with gzip

            if useReps:
                self.repData = []
                for _ in range(state_dict['nRows']):
                    self.repData.append(_np.lib.format.read_array(f))  # _np.load(f) doesn't play nice with gzip
            else:
                self.repData = None
            self.cnt_cache = None

        if bOpen: f.close()

    def rename_outcome_labels(self, old_to_new_dict):
        """
        Replaces existing output labels with new ones as per `old_to_new_dict`.

        Parameters
        ----------
        old_to_new_dict : dict
            A mapping from old/existing outcome labels to new ones.  Strings
            in keys or values are automatically converted to 1-tuples.  Missing
            outcome labels are left unaltered.

        Returns
        -------
        None
        """
        mapdict = {}
        for old, new in old_to_new_dict.items():
            if isinstance(old, str): old = (old,)
            if isinstance(new, str): new = (new,)
            mapdict[old] = new

        new_olIndex = _OrderedDict()
        for ol, i in self.olIndex.items():
            if ol in mapdict:
                new_olIndex[mapdict[ol]] = i
            else:
                new_olIndex[ol] = i

        #Note: rebuild reverse-dict self.ol:
        self.olIndex = new_olIndex
        self.ol = _OrderedDict([(i, ol) for (ol, i) in self.olIndex.items()])

    def add_std_nqubit_outcome_labels(self, nqubits):
        """
        Adds all the "standard" outcome labels (e.g. '0010') on `nqubits` qubits.

        This is useful to ensure that, even if not all outcomes appear in the
        data, that all are recognized as being potentially valid outcomes (and
        so attempts to get counts for these outcomes will be 0 rather than
        raising an error).

        Parameters
        ----------
        nqubits : int
            The number of qubits.  For example, if equal to 3 the outcome labels
            '000', '001', ... '111' are added.

        Returns
        -------
        None
        """
        self.add_outcome_labels((("".join(x),) for x in _itertools.product(*([('0', '1')] * nqubits))))

    def add_outcome_labels(self, outcome_labels, update_ol=True):
        """
        Adds new valid outcome labels.

        Ensures that all the elements of `outcome_labels` are stored as
        valid outcomes for circuits in this DataSet, adding new outcomes
        as necessary.

        Parameters
        ----------
        outcome_labels : list or generator
            A list or generator of string- or tuple-valued outcome labels.

        update_ol : bool, optional
            Whether to update internal mappings to reflect the new outcome labels.
            Leave this as True unless you really know what you're doing.

        Returns
        -------
        None
        """
        added = False
        iNext = self.olIndex_max
        for ol in outcome_labels:
            if ol not in self.olIndex:
                iNext += 1
                self.olIndex[ol] = iNext; added = True
        if added and update_ol:  # rebuild self.ol because olIndex has changed
            self.update_ol()
        self.olIndex_max = iNext
