""" Defines the DataSet class and supporting classes and functions """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

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
import sys as _sys

from collections import OrderedDict as _OrderedDict
from collections import defaultdict as _DefaultDict

from ..tools import listtools as _lt
from ..tools import compattools as _compat

from . import circuit as _cir
from . import labeldicts as _ld
#from . import dataset as _ds

Oindex_type = _np.uint32
Time_type = _np.float64
Repcount_type = _np.float32
 # thought: _np.uint16 but doesn't play well with rescaling

class DataSet_KeyValIterator(object):
    """ Iterator class for op_string,DataSetRow pairs of a DataSet """
    def __init__(self, dataset):
        self.dataset = dataset
        self.gsIter = dataset.cirIndex.__iter__()
        oliData = self.dataset.oliData
        timeData = self.dataset.timeData
        repData = self.dataset.repData
        cntcache = self.dataset.cnt_cache
        auxInfo = dataset.auxInfo

        def getcache(opstr):
            return dataset.cnt_cache[opstr] if dataset.bStatic else None

        if repData is None:
            self.tupIter = ( (oliData[ gsi ], timeData[ gsi ], None, getcache(opstr), auxInfo[opstr])
                             for opstr,gsi in self.dataset.cirIndex.items() )
        else:
            self.tupIter = ( (oliData[ gsi ], timeData[ gsi ], repData[ gsi ], getcache(opstr), auxInfo[opstr])
                             for opstr,gsi in self.dataset.cirIndex.items() )
        #Note: gsi above will be an index for a non-static dataset and
        #  a slice for a static dataset.

    def __iter__(self):
        return self

    def __next__(self): # Python 3: def __next__(self)
        return next(self.gsIter), DataSetRow(self.dataset, *(next(self.tupIter)) )

    next = __next__


class DataSet_ValIterator(object):
    """ Iterator class for DataSetRow values of a DataSet """
    def __init__(self, dataset):
        self.dataset = dataset
        oliData = self.dataset.oliData
        timeData = self.dataset.timeData
        repData = self.dataset.repData
        cntcache = self.dataset.cnt_cache
        auxInfo = dataset.auxInfo

        def getcache(opstr):
            return dataset.cnt_cache[opstr] if dataset.bStatic else None

        if repData is None:
            self.tupIter = ( (oliData[ gsi ], timeData[ gsi ], None, getcache(opstr), auxInfo[opstr])
                             for opstr,gsi in self.dataset.cirIndex.items() )
        else:
            self.tupIter = ( (oliData[ gsi ], timeData[ gsi ], repData[ gsi ], getcache(opstr), auxInfo[opstr])
                             for opstr,gsi in self.dataset.cirIndex.items() )
        #Note: gsi above will be an index for a non-static dataset and
        #  a slice for a static dataset.

    def __iter__(self):
        return self

    def __next__(self): # Python 3: def __next__(self)
        return DataSetRow(self.dataset, *(next(self.tupIter)) )

    next = __next__


class DataSetRow(object):
    """
    Encapsulates DataSet time series data for a single circuit.  Outwardly
    looks similar to a list with `(outcome_label, time_index, repetition_count)`
    tuples as the values.
    """
    def __init__(self, dataset, rowOliData, rowTimeData, rowRepData,
                 cached_cnts, aux):
        self.dataset = dataset
        self.oli = rowOliData
        self.time = rowTimeData
        self.reps = rowRepData
        self._cntcache = cached_cnts
        self.aux = aux

    @property
    def outcomes(self):
        """
        Returns this row's sequence of outcome labels, one per "bin" of repetition
        counts (returned by :method:`get_counts`).
        """
        return [self.dataset.ol[i] for i in self.oli]

    @outcomes.setter
    def outcomes(self, value):
        raise ValueError("outcomes property is read-only")

    def get_expanded_ol(self):
        """
        Returns this row's sequence of outcome labels, with repetition counts
        expanded, so there's one element in the returned list for *each* count.
        """
        if self.reps is not None:
            ol = []
            for oli, _, nreps in zip(self.oli,self.time,self.reps):
                nreps = _round_int_repcnt(nreps)
                ol.extend( [self.dataset.ol[oli]]*nreps )
            return ol
        else: return self.outcomes

    def get_expanded_oli(self):
        """
        Returns this row's sequence of outcome label indices, with repetition counts
        expanded, so there's one element in the returned list for *each* count.
        """
        if self.reps is not None:
            inds = []
            for oli, _, nreps in zip(self.oli,self.time,self.reps):
                nreps = _round_int_repcnt(nreps)
                inds.extend( [oli]*nreps )
            return _np.array(inds, dtype=self.dataset.oliType)
        else: return self.oli.copy()

    def get_expanded_times(self):
        """
        Returns this row's sequence of time stamps, with repetition counts
        expanded, so there's one element in the returned list for *each* count.
        """
        if self.reps is not None:
            times = []
            for _, time, nreps in zip(self.oli,self.time,self.reps):
                nreps = _round_int_repcnt(nreps)
                times.extend( [time]*nreps )
            return _np.array(times, dtype=self.dataset.timeType)
        else: return self.time.copy()

    def __iter__(self):
        if self.reps is not None:
            return ( (self.dataset.ol[i],t,n) for (i,t,n) in zip(self.oli,self.time,self.reps) )
        else:
            return ( (self.dataset.ol[i],t,1) for (i,t) in zip(self.oli,self.time) )

    def has_key(self, outcomeLabel):
        """ Checks whether data counts for `outcomelabel` are available."""
        return outcomeLabel in self.counts

    def __getitem__(self,indexOrOutcomeLabel):
        if isinstance(indexOrOutcomeLabel, _numbers.Integral): #raw index
            i = indexOrOutcomeLabel
            if self.reps is not None:
                return ( self.dataset.ol[ self.oli[i] ], self.time[i], self.reps[i] )
            else:
                return ( self.dataset.ol[ self.oli[i] ], self.time[i], 1 )
        elif isinstance(indexOrOutcomeLabel, _numbers.Real): #timestamp
            return self.counts_at_time(indexOrOutcomeLabel)
        else:
            try:
                return self.counts[indexOrOutcomeLabel]
            except KeyError:
                # if outcome label isn't in counts but *is* in the dataset's
                # outcome labels then return 0 (~= return self.allcounts[...])
                key = _ld.OutcomeLabelDict.to_outcome(indexOrOutcomeLabel)
                if key in self.dataset.get_outcome_labels(): return 0
                raise KeyError("%s is not an index, timestamp, or outcome label!"
                               % str(indexOrOutcomeLabel))

    def __setitem__(self,indexOrOutcomeLabel,val):
        if isinstance(indexOrOutcomeLabel, _numbers.Integral):
            index = indexOrOutcomeLabel; tup = val
            assert(len(tup) in (2,3) ), "Must set to a (<outcomeLabel>,<time>[,<repetitions>]) value"
            ol = _ld.OutcomeLabelDict.to_outcome(tup[0]) #strings -> tuple outcome labels
            self.oli[index] = self.dataset.olIndex[ ol ]
            self.time[index] = tup[1]

            if self.reps is not None:
                self.reps[index] = tup[2] if len(tup) == 3 else 1
            else:
                assert(len(tup) == 2 or tup[2] == 1),"Repetitions must == 1 (not tracking reps)"
        else:
            outcomeLbl = _ld.OutcomeLabelDict.to_outcome(indexOrOutcomeLabel) #strings -> tuple outcome labels
            count = val

            assert( all([t == self.time[0] for t in self.time]) ), \
                "Cannot set outcome counts directly on a DataSet with non-trivially timestamped data"
            assert(self.reps is not None), \
                "Cannot set outcome counts directly on a DataSet without repetition data"

            outcomeIndxToLookFor = self.dataset.olIndex.get(outcomeLbl,None)
            for i,outcomeIndx in enumerate(self.oli):
                if outcomeIndx == outcomeIndxToLookFor:
                    self.reps[i] = count; break
            else: # need to add a new label & entry to reps[]
                raise NotImplementedError("Cannot create new outcome labels by assignment")

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
            tslc = _np.where(_np.isclose(self.time,timestamp))[0]
        else: tslc = slice(None)

        if self.reps is None:
            for ol,i in self.dataset.olIndex.items():
                cnt = float(_np.count_nonzero( _np.equal(self.oli[tslc],i) ))
                if all_outcomes or cnt > 0: cntDict[ol] = cnt
        else:
            for ol,i in self.dataset.olIndex.items():
                inds = _np.nonzero(_np.equal(self.oli[tslc],i))[0]
                if all_outcomes or len(inds) > 0:
                    cntDict[ol] = float( sum(self.reps[tslc][inds]))
        return cntDict

    @property
    def counts(self):
        if self._cntcache: return self._cntcache # if not None *and* len > 0
        ret = self._get_counts()
        if self._cntcache is not None: # == and empty dict {}
            self._cntcache.update(ret)
        return ret

    @property
    def allcounts(self):
        return self._get_counts(all_outcomes=True)

    @property
    def fractions(self, all_outcomes=False):
        """
        Returns this row's sequence of "repetition counts", that is, the number of
        repetitions of each outcome label in the `outcomes` list, or
        equivalently, each outcome label index in this rows `.oli` member.
        """
        cnts = self._get_counts(all_outcomes)
        total = sum(cnts.values())
        return _OrderedDict( [(k,cnt/total) for k,cnt in cnts.items()] )

    @property
    def total(self):
        """ Returns the total number of counts contained in this row."""
        if self.reps is None:
            return float(len(self.oli))
        else:
            return sum(self.reps)

    #TODO: remove in favor of fractions property?
    def fraction(self,outcomelabel):
        """ Returns the fraction of total counts for `outcomelabel`."""
        d = self.counts
        if outcomelabel not in d:
            return 0.0 # Note: similar to an "all_outcomes=True" default
        total = sum(d.values())
        return d[outcomelabel]/total

    def counts_at_time(self, timestamp):
        """ Returns a dictionary of counts at a particular time """
        return self._get_counts(timestamp)

    def timeseries(self, outcomelabel, timestamps=None):
        """
        Returns timestamps and counts for a single outcome label
        or for aggregated counts if `outcomelabel == "all"`.

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
        for i,(t,oli) in enumerate(zip(self.time,self.oli)):

            if timestamps is not None:
                while tsIndx < len(timestamps) and t > timestamps[tsIndx] \
                      and not _np.isclose(t,timestamps[tsIndx], rtol=0., atol=1e-12):
                    times.append(timestamps[tsIndx])
                    counts.append(0)
                    tsIndx += 1

            if oli in olis and (timestamps is None or _np.isclose(t,timestamps[tsIndx], rtol=0., atol=1e-12)):
                if not _np.isclose(t,last_t, rtol=0., atol=1e-12):
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


    def scale(self, factor):
        """ Scales all the counts of this row by the given factor """
        if self.dataset.bStatic: raise ValueError("Cannot scale rows of a *static* DataSet.")
        if self.reps is None:
            raise ValueError(("Cannot scale a DataSet without repetition "
                              "counts. Call DataSet.build_repetition_counts()"
                              " and try this again."))
        for i,cnt in enumerate(self.reps):
            self.reps[i] = cnt*factor


    def as_dict(self):
        """ Returns the (outcomeLabel,count) pairs as a dictionary."""
        return dict( self.counts )

    def to_str(self, mode="auto"):
        """
        Render this DataSetRow as a string.

        Parameters
        ----------
        mode : {"auto","time-dependent","time-independent"}
            Whether to display the data as time-series of outcome counts
            (`"time-dependent"`) or to report per-outcome counts aggregated over
            time (`"time-independent"`).  If `"auto"` is specified, then the
            time-independent mode is used only if all time stamps in the
            DataSetRow are equal (trivial time dependence).

        Returns
        -------
        str
        """
        if mode == "auto":
            if all([t == self.time[0] for t in self.time]):
                mode = "time-independent"
            else: mode = "time-dependent"

        assert(mode in ('time-dependent','time-independent')),"Invalid `mode` argument: %s" % mode

        if mode == "time-dependent":
            s  = "Outcome Label Indices = " + str(self.oli) + "\n"
            s += "Time stamps = " + str(self.time) + "\n"
            if self.reps is not None:
                s += "Repetitions = " + str(self.reps) + "\n"
            else:
                s += "( no repetitions )\n"
            return s
        else: # time-independent
            return str(self.as_dict())

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
    """

    def __init__(self, oliData=None, timeData=None, repData=None,
                 circuits=None, circuitIndices=None,
                 outcomeLabels=None, outcomeLabelIndices=None,
                 bStatic=False, fileToLoadFrom=None, collisionAction="aggregate",
                 comment=None):
        """
        Initialize a DataSet.

        Parameters
        ----------
        oliData : list or numpy.ndarray
            When `bStatic == True`, a 1D numpy array containing outcome label
            indices (integers), concatenated for all sequences.  Otherwise, a
            list of 1D numpy arrays, one array per gate sequence.  In either
            case, this quantity is indexed by the values of `circuitIndices`
            or the index of `circuits`.

        timeData : list or numpy.ndarray
            Same format at `oliData` except stores floating-point timestamp
            values.

        repData : list or numpy.ndarray
            Same format at `oliData` except stores integer repetition counts
            for each "data bin" (i.e. (outcome,time) pair).  If all repetitions
            equal 1 ("single-shot" timestampted data), then `repData` can be
            `None` (no repetitions).

        circuits : list of (tuples or Circuits)
            Each element is a tuple of operation labels or a Circuit object.  Indices for these strings
            are assumed to ascend from 0.  These indices must correspond to the time series of spam-label
            indices (above).   Only specify this argument OR circuitIndices, not both.

        circuitIndices : ordered dictionary
            An OrderedDict with keys equal to circuits (tuples of operation labels) and values equal to
            integer indices associating a row/element of counts with the circuit.  Only
            specify this argument OR circuits, not both.

        outcomeLabels : list of strings
            Specifies the set of spam labels for the DataSet.  Indices for the spam labels
            are assumed to ascend from 0, starting with the first element of this list.  These
            indices will associate each elememtn of `timeseries` with a spam label.  Only
            specify this argument OR outcomeLabelIndices, not both.

        outcomeLabelIndices : ordered dictionary
            An OrderedDict with keys equal to spam labels (strings) and value equal to
            integer indices associating a spam label with given index.  Only
            specify this argument OR outcomeLabels, not both.

        bStatic : bool
            When True, create a read-only, i.e. "static" DataSet which cannot be modified. In
              this case you must specify the timeseries data, circuits, and spam labels.
            When False, create a DataSet that can have time series data added to it.  In this case,
              you only need to specify the spam labels.

        fileToLoadFrom : string or file object
            Specify this argument and no others to create a static DataSet by loading
            from a file (just like using the load(...) function).

        collisionAction : {"aggregate","keepseparate"}
            Specifies how duplicate circuits should be handled.  "aggregate"
            adds duplicate-sequence counts to the same circuit's data at the
            next integer timestamp.  "keepseparate" tags duplicate-sequences by
            appending a final "#<number>" operation label to the duplicated gate
            sequence, which can then be accessed via the `get_row` and `set_row`
            functions.

        comment : string, optional
            A user-specified comment string that gets carried around with the
            data.  A common use for this field is to attach to the data details
            regarding its collection.

        Returns
        -------
        DataSet
           a new data set object.
        """
        # uuid for efficient hashing (set when done adding data or loading from file)
        self.uuid = None

        #Optionally load from a file
        if fileToLoadFrom is not None:
            assert(oliData is None and timeData is None and repData is None \
                   and circuits is None and circuitIndices is None \
                   and outcomeLabels is None and outcomeLabelIndices is None)
            self.load(fileToLoadFrom)
            return

        # self.cirIndex  :  Ordered dictionary where keys = Circuit objects,
        #   values = slices into oli, time, & rep arrays (static case) or
        #            integer list indices (non-static case)
        if circuitIndices is not None:
            self.cirIndex = _OrderedDict( [(opstr if isinstance(opstr,_cir.Circuit) else _cir.Circuit(opstr),i)
                                          for opstr,i in circuitIndices.items()] )
                                         #convert keys to Circuits if necessary
        elif not bStatic:
            if circuits is not None:
                dictData = [ (opstr if isinstance(opstr,_cir.Circuit) else _cir.Circuit(opstr),i) \
                             for (i,opstr) in enumerate(circuits) ] #convert to Circuits if necessary
                self.cirIndex = _OrderedDict( dictData )
            else:
                self.cirIndex = _OrderedDict()
        else: raise ValueError("Must specify circuitIndices when creating a static DataSet")

        # self.olIndex  :  Ordered dictionary where
        #                  keys = outcome labels (strings or tuples),
        #                  values = integer indices mapping oliData (integers) onto
        #                           the outcome labels.
        if outcomeLabelIndices is not None:
            self.olIndex = outcomeLabelIndices
        elif outcomeLabels is not None:
            tup_outcomeLabels = [ _ld.OutcomeLabelDict.to_outcome(ol)
                                  for ol in outcomeLabels] #strings -> tuple outcome labels
            self.olIndex = _OrderedDict( [(ol,i) for (i,ol) in enumerate(tup_outcomeLabels) ] )
        else:
            self.olIndex = _OrderedDict() #OK, as outcome labels are added as they appear

        # self.ol :  Ordered dictionary where keys = integer indices, values = outcome
        #            labels (strings or tuples) -- just the reverse of self.olIndex
        self.ol = _OrderedDict( [(i,ol) for (ol,i) in self.olIndex.items()] )

        # sanity checks that indices are >= 0
        if not bStatic: #otherwise values() below are slices
            if self.cirIndex:  assert( min(self.cirIndex.values()) >= 0)
            if self.olIndex:  assert( min(self.olIndex.values()) >= 0)

        # self.oliData : when bStatic == True a 1D numpy array containing concatenated outcome label indices.
        #                when bStatic == False a list of 1D numpy arrays, one array per gate sequence.

        # self.timeData : when bStatic == True a 1D numpy array containing concatenated time stamps.
        #                 when bStatic == False a list of 1D numpy arrays, one array per gate sequence.

        # self.repData : when bStatic == True a 1D numpy array containing concatenated repetition counts.
        #                when bStatic == False a list of 1D numpy arrays, one array per gate sequence.
        #   (can be None, in which case no repetitions are assumed)

        if oliData is not None:

            # check that sizes/lengths all match
            assert(len(timeData) == len(oliData)), "timeData must be same size as oliData"
            if repData is not None:
                assert(len(repData) == len(oliData)), "repData must be same size as oliData"

            self.oliData = oliData
            self.timeData = timeData
            self.repData = repData

            if len(self.cirIndex) > 0:
                maxOlIndex = max(self.olIndex.values())
                if bStatic:
                    assert( _np.amax(self.oliData) <= maxOlIndex )
                    # self.oliData.shape[0] > maxIndex doesn't make sense since cirIndex holds slices
                else:
                    maxIndex = max(self.cirIndex.values())
                    assert( len(self.oliData) > maxIndex )
                    if len(self.oliData) > 0:
                        assert( all( [ max(oliSeries) <= maxOlIndex for oliSeries in self.oliData ] ) )
            #else cirIndex has length 0 so there are no circuits in this dataset (even though oliData can contain data)

        elif not bStatic:
            assert( timeData is None ), "timeData must be None when oliData is"
            assert( repData is None ), "repData must be None when oliData is"
            assert( len(self.cirIndex) == 0), "circuit specified without data!"
            self.oliData = []
            self.timeData = []
            self.repData = None

        else:
            raise ValueError("Series data must be specified when creating a static DataSet")

        # self.bStatic
        self.bStatic = bStatic

        # collision action
        assert(collisionAction in ('aggregate','keepseparate'))
        self.collisionAction = collisionAction

        # comment
        self.comment = comment

        # self.ffdata : fourier filtering data
        self.ffdata = {}

        #data types - should stay in sync with MultiDataSet
        self.oliType  = Oindex_type
        self.timeType = Time_type
        self.repType  = Repcount_type

        #auxiliary info
        self.auxInfo = _DefaultDict( dict )

        # count cache (only used when static; not saved/loaded from disk)
        if bStatic:
            self.cnt_cache = { opstr:_ld.OutcomeLabelDict() for opstr in self.cirIndex }
        else:
            self.cnt_cache = None


    def __iter__(self):
        return self.cirIndex.__iter__() #iterator over circuits

    def __len__(self):
        return len(self.cirIndex)

    def __contains__(self, circuit):
        return self.has_key(circuit)

    def __hash__(self):
        if self.uuid is not None:
            return hash(self.uuid)
        else:
            raise TypeError('Use digest hash')

    def __getitem__(self, circuit):
        return self.get_row(circuit)

    def __setitem__(self, circuit, outcomeDictOrSeries):
        return self.set_row(circuit, outcomeDictOrSeries)

    def __delitem__(self, circuit):
        if not isinstance(circuit,_cir.Circuit):
            circuit = _cir.Circuit(circuit)
        self._remove( [self.cirIndex[circuit]] )

    def get_row(self, circuit, occurrence=0):
        """
        Get a row of data from this DataSet.  This gives the same
        functionality as [ ] indexing except you can specify the
        occurrence number separately from the gate sequence.

        Parameters
        ----------
        circuit : Circuit or tuple
            The gate sequence to extract data for.

        occurrence : int, optional
            0-based occurrence index, specifying which occurrence of
            a repeated gate sequence to extract data for.

        Returns
        -------
        DataSetRow
        """

        #Convert to circuit - needed for occurrence > 0 case and
        # because name-only Labels still don't hash the same as strings
        # so key lookups need to be done at least with tuples of Labels.
        if not isinstance(circuit,_cir.Circuit):
            circuit = _cir.Circuit(circuit)

        if occurrence > 0:
            circuit = circuit + _cir.Circuit(("#%d" % occurrence,))

        #Note: cirIndex value is either an int (non-static) or a slice (static)
        repData = self.repData[ self.cirIndex[circuit] ] \
                  if (self.repData is not None) else None
        return DataSetRow(self, self.oliData[ self.cirIndex[circuit] ],
                          self.timeData[ self.cirIndex[circuit] ], repData,
                          self.cnt_cache[ circuit ] if self.bStatic else None,
                          self.auxInfo[circuit])

    def set_row(self, circuit, outcomeDictOrSeries, occurrence=0):
        """
        Set the counts for a row of this DataSet.  This gives the same
        functionality as [ ] indexing except you can specify the
        occurrence number separately from the gate sequence.

        Parameters
        ----------
        circuit : Circuit or tuple
            The gate sequence to extract data for.

        countDict : dict
            The dictionary of counts (data).

        occurrence : int, optional
            0-based occurrence index, specifying which occurrence of
            a repeated gate sequence to extract data for.
        """
        if not isinstance(circuit,_cir.Circuit):
            circuit = _cir.Circuit(circuit)

        if occurrence > 0:
            circuit = _cir.Circuit(circuit) + _cir.Circuit(("#%d" % occurrence,))

        if isinstance(outcomeDictOrSeries, dict): # a dict of counts
            self.add_count_dict(circuit, outcomeDictOrSeries)

        else: # a tuple of lists
            assert(len(outcomeDictOrSeries) >= 2), \
                "Must minimally set with (outcome-label-list, time-stamp-list)"
            self.add_raw_series_data(circuit, *outcomeDictOrSeries)


    def keys(self, stripOccurrenceTags=False):
        """
        Returns the circuits used as keys of this DataSet.

        Parameters
        ----------
        stripOccurrenceTags : bool, optional
            Only applicable if `collisionAction` has been set to
            "keepseparate", when this argument is set to True
            any final "#<number>" elements of (would-be dupilcate)
            circuits are stripped so that the returned list
            may have *duplicate* entries.

        Returns
        -------
        list
            A list of Circuit objects which index the data
            counts within this data set.
        """
        if stripOccurrenceTags and self.collisionAction == "keepseparate":
            # Note: assumes keys are Circuits containing Labels
            return [ (opstr[:-1] if (len(opstr)>0 and opstr[-1].name.startswith("#")) else opstr)
                     for opstr in self.cirIndex.keys() ]
        else:
            return list(self.cirIndex.keys())


    def has_key(self, circuit):
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
        if not isinstance(circuit,_cir.Circuit):
            circuit = _cir.Circuit(circuit)
        return circuit in self.cirIndex


    def items(self):
        """
        Iterator over (circuit, timeSeries) pairs,
        where circuit is a tuple of operation labels
        and timeSeries is a DataSetRow instance,
        which behaves similarly to a list of
        spam labels whose index corresponds to
        the time step.
        """
        return DataSet_KeyValIterator(self)

    def values(self):
        """
        Iterator over DataSetRow instances corresponding
        to the time series data for each circuit.
        """
        return DataSet_ValIterator(self)


    def get_outcome_labels(self):
        """
        Get a list of *all* the outcome labels contained in this DataSet.

        Returns
        -------
        list of strings or tuples
          A list where each element is an outcome label (which can
          be a string or a tuple of strings).
        """
        return list(self.olIndex.keys())


    def get_gate_labels(self, prefix='G'):
        """
        Get a list of all the distinct operation labels used
        in the circuits of this dataset.

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
        opLabels = [ ]
        for opLabelString in self:
            for opLabel in opLabelString:
                if not prefix or opLabel.name.startswith(prefix):
                    if opLabel not in opLabels: opLabels.append(opLabel)
        return opLabels


    def get_degrees_of_freedom(self, circuitList=None, method="all_outcomes-1",
                               aggregate_times=True):
        """
        Returns the number of independent degrees of freedom in the data for
        the circuits in `circuitList`.

        Parameters
        ----------
        circuitList : list of Circuits
            The list of circuits to count degrees of freedom for.  If `None`
            then all of the `DataSet`'s strings are used.

        method : {'all_outcomes-1', 'present_outcomes-1'}
            How the degrees of freedom should be computed. 'all_outcomes-1' takes
            the number of circuits and multiplies this by the *total* number of outcomes
            (the length of what is returned by `get_outcome_labels()`) minus one.  
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
        if circuitList is None:
            circuitList = list(self.keys())

        nDOF = 0
        Nout = len(self.olIndex)
        for opstr in circuitList:
            dsRow = self[opstr]
            cur_t = dsRow.time[0]
            cur_outcomes = set() # holds *distinct* outcomes at current time
            for ol,t,rep in dsRow:
                if aggregate_times or t == cur_t:
                    cur_outcomes.add(ol)
                else:
                    #assume final outcome at each time is constrained
                    nOutcomes = Nout if method == 'all_outcomes-1' else len(cur_outcomes)
                    nDOF += nOutcomes-1; cur_outcomes = set()
                    cur_t = t
            nOutcomes = Nout if method == 'all_outcomes-1' else len(cur_outcomes)
            nDOF += nOutcomes-1; #last time stamp
        return nDOF


    def _keepseparate_update_circuit(self, circuit):
        if not isinstance(circuit, _cir.Circuit):
            circuit = _cir.Circuit(circuit) #make sure we have a Circuit

        # if "keepseparate" mode, add tag onto end of circuit
        if circuit in self.cirIndex and self.collisionAction == "keepseparate":
            i=0; tagged_circuit = circuit
            while tagged_circuit in self.cirIndex:
                i+=1; tagged_circuit = circuit + _cir.Circuit(("#%d" % i,))
            #add data for a new (duplicate) circuit
            circuit = tagged_circuit

        return circuit

    def build_repetition_counts(self):
        """
        Build internal repetition counts if they don't exist already.

        This method is usually unnecessary, as repetition counts are
        almost always build as soon as they are needed.
        """
        if self.repData is not None: return
        if self.bStatic:
            raise ValueError("Cannot build repetition counts in a static DataSet object")
        self.repData = []
        for oliAr in self.oliData:
            self.repData.append( _np.ones(len(oliAr), self.repType) )


    def add_count_dict(self, circuit, countDict, overwriteExisting=True,
                       recordZeroCnts=False, aux=None):
        """
        Add a single circuit's counts to this DataSet

        Parameters
        ----------
        circuit : tuple or Circuit
            A tuple of operation labels specifying the circuit or a Circuit object

        countDict : dict
            A dictionary with keys = outcome labels and values = counts

        overwriteExisting : bool, optional
            If `True`, overwrite any existing data for the `circuit`.  If
            `False`, add this count data with the next non-negative integer
            timestamp.

        recordZeroCnts : bool, optional
            Whether zero-counts are actually recorded (stored) in this DataSet.
            If False, then zero counts are ignored, except for potentially
            registering new outcome labels.

        aux : dict, optional
            A dictionary of auxiliary meta information to be included with
            this set of data counts (associated with `circuit`).

        Returns
        -------
        None
        """
        if self.bStatic: raise ValueError("Cannot add data to a static DataSet object")

        #Convert input to an OutcomeLabelDict
        if isinstance(countDict, _ld.OutcomeLabelDict):
            outcomeCounts = countDict
        elif isinstance(countDict, _OrderedDict): #then don't sort keys
            outcomeCounts = _ld.OutcomeLabelDict( list(countDict.items()) )
        else:
            # sort key for deterministic ordering of *new* outcome labels)
            outcomeCounts = _ld.OutcomeLabelDict( [
                (lbl,countDict[lbl]) for lbl in sorted(list(countDict.keys()))])

        outcomeLabelList = list(outcomeCounts.keys())
        countList = list(outcomeCounts.values())

        # if "keepseparate" mode, add tag onto end of circuit
        circuit = self._keepseparate_update_circuit(circuit)

        if not overwriteExisting and circuit in self:
            iNext = int(max(self[circuit].time)) + 1 \
                    if (len(self[circuit].time) > 0) else 0
            timeStampList = [iNext]*len(countList)
        else:
            timeStampList = [0]*len(countList)

        self.add_raw_series_data(circuit, outcomeLabelList, timeStampList,
                                 countList, overwriteExisting, recordZeroCnts,
                                 aux)
        

    def add_raw_series_data(self, circuit, outcomeLabelList, timeStampList,
                            repCountList=None, overwriteExisting=True,
                            recordZeroCnts=True, aux=None):
        """
        Add a single circuit's counts to this DataSet

        Parameters
        ----------
        circuit : tuple or Circuit
            A tuple of operation labels specifying the circuit or a Circuit object

        outcomeLabelList : list
            A list of outcome labels (strings or tuples).  An element's index
            links it to a particular time step (i.e. the i-th element of the
            list specifies the outcome of the i-th measurement in the series).

        timeStampList : list
            A list of floating point timestamps, each associated with the single
            corresponding outcome in `outcomeLabelList`. Must be the same length
            as `outcomeLabelList`.

        repCountList : list, optional
            A list of integer counts specifying how many outcomes of type given
            by `outcomeLabelList` occurred at the time given by `timeStampList`.
            If None, then all counts are assumed to be 1.  When not None, must
            be the same length as `outcomeLabelList`.

        overwriteExisting : bool, optional
            Whether to overwrite the data for `circuit` (if it exists).  If
            False, then the given lists are appended (added) to existing data.

        recordZeroCnts : bool, optional
            Whether zero-counts (elements of `repCountList` that are zero) are
            actually recorded (stored) in this DataSet.  If False, then zero
            counts are ignored, except for potentially registering new outcome
            labels.

        aux : dict, optional
            A dictionary of auxiliary meta information to be included with
            this set of data counts (associated with `circuit`).

        Returns
        -------
        None
        """
        if self.bStatic: raise ValueError("Cannot add data to a static DataSet object")

        # if "keepseparate" mode, add tag onto end of circuit
        circuit = self._keepseparate_update_circuit(circuit)

        #strings -> tuple outcome labels
        tup_outcomeLabelList = [ _ld.OutcomeLabelDict.to_outcome(ol)
                                 for ol in outcomeLabelList]

        #Add any new outcome labels
        added = False
        for ol in tup_outcomeLabelList:
            if ol not in self.olIndex:
                iNext = max(self.olIndex.values())+1 if len(self.olIndex) > 0 else 0
                self.olIndex[ol] = iNext; added=True
        if added: #rebuild self.ol because olIndex has changed
            self.ol = _OrderedDict( [(i,sl) for (sl,i) in self.olIndex.items()] )

        oliArray = _np.array([ self.olIndex[ol] for ol in tup_outcomeLabelList ] , self.oliType)
        timeArray = _np.array(timeStampList, self.timeType)
        assert(oliArray.shape == timeArray.shape), \
            "Outcome-label and time stamp lists must have the same length!"

        if repCountList is None:
            if self.repData is None: repArray = None
            else: repArray = _np.ones(len(oliArray), self.repType)
        else:
            if self.repData is None:
                #rep count data was given, but we're not currently holding repdata,
                # so we need to build this up for all existings sequences:
                self.build_repetition_counts()
            repArray = _np.array(repCountList, self.repType)

        if not recordZeroCnts:
            # Go through repArray and remove any zeros, along with
            # corresponding elements of oliArray and timeArray
            mask = repArray != 0 # boolean array (note: == float comparison *is* desired)
            repArray  = repArray[mask]
            oliArray  = oliArray[mask]
            timeArray = timeArray[mask]

        if circuit in self.cirIndex:
            circuitIndx = self.cirIndex[circuit]
            if overwriteExisting:
                self.oliData[ circuitIndx ] = oliArray #OVERWRITE existing time series
                self.timeData[ circuitIndx ] = timeArray #OVERWRITE existing time series
                if repArray is not None: self.repData[ circuitIndx ] = repArray
            else:
                self.oliData[ circuitIndx ] = _np.concatenate((self.oliData[circuitIndx],oliArray))
                self.timeData[ circuitIndx ] = _np.concatenate((self.timeData[circuitIndx],timeArray))
                if repArray is not None:
                    self.repData[ circuitIndx ] = _np.concatenate((self.repData[circuitIndx],repArray))

        else:
            #add data for a new circuit
            assert( len(self.oliData) == len(self.timeData) ), "OLI and TIME data are out of sync!!"
            circuitIndx = len(self.oliData) #index of to-be-added circuit
            self.oliData.append( oliArray )
            self.timeData.append( timeArray )
            if repArray is not None: self.repData.append( repArray )
            self.cirIndex[ circuit ] = circuitIndx

        if aux is not None: self.add_auxiliary_info(circuit, aux)


    def add_series_data(self, circuit, countDictList, timeStampList,
                        overwriteExisting=True, aux=None):
        """
        Add a single circuit's counts to this DataSet

        Parameters
        ----------
        circuit : tuple or Circuit
            A tuple of operation labels specifying the circuit or a Circuit object

        countDictList : list
            A list of dictionaries holding the outcome-label:count pairs for each
            time step (times given by `timeStampList`.

        timeStampList : list
            A list of floating point timestamps, each associated with an entire
            dictionary of outcomes specified by `countDictList`.

        overwriteExisting : bool, optional
            If `True`, overwrite any existing data for the `circuit`.  If
            `False`, add the count data with the next non-negative integer
            timestamp.

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

        for (cntDict, t) in zip(countDictList, timeStampList):
            if not isinstance(cntDict, _OrderedDict):
                ols = sorted(list(cntDict.keys()))
            else: ols = list(cntDict.keys())
            for ol in ols: # loop over outcome labels
                expanded_outcomeList.append(ol)
                expanded_timeList.append(t)
                expanded_repList.append(cntDict[ol]) #could do this only for counts > 1
        return self.add_raw_series_data(circuit, expanded_outcomeList,
                                        expanded_timeList, expanded_repList,
                                        overwriteExisting, aux=aux)

    
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
        self.auxInfo[circuit].clear() # needed? (could just update?)
        self.auxInfo[circuit].update(aux) 


    def add_counts_from_dataset(self, otherDataSet):
        """
        Append another DataSet's data to this DataSet

        Parameters
        ----------
        otherDataSet : DataSet
            The dataset to take counts from.

        Returns
        -------
        None
        """
        return self.add_series_from_dataset(otherDataSet)


    def add_series_from_dataset(self, otherDataSet):
        """
        Append another DataSet's series data to this DataSet

        Parameters
        ----------
        otherDataSet : DataSet
            The dataset to take time series data from.

        Returns
        -------
        None
        """
        if self.bStatic: raise ValueError("Cannot add data to a static DataSet object")
        for circuit,dsRow in otherDataSet.items():
            self.add_raw_series_data(circuit, dsRow.outcomes, dsRow.time, dsRow.reps, False)


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
            if all([_np.all(self.timeData[gsi] == 0) for gsi in self.cirIndex.values()]):
                mode = "time-independent"
            else: mode = "time-dependent"

        assert(mode in ('time-dependent','time-independent')),"Invalid `mode` argument: %s" % mode

        if mode == "time-dependent":
            s = "Dataset outcomes: " + str(self.olIndex) + "\n"
            for circuit in self: # tuple-type operation label strings are keys
                s += "%s :\n%s\n" % (circuit.str, self[circuit].to_str(mode))
            return s + "\n"
        else: # time-independent
            s = ""
            for circuit in self: # tuple-type operation label strings are keys
                s += "%s  :  %s\n" % (circuit.str, self[circuit].to_str(mode))
            return s + "\n"


    def truncate(self, listOfCircuitsToKeep, missingAction='raise'):
        """
        Create a truncated dataset comprised of a subset of the circuits
        in this dataset.

        Parameters
        ----------
        listOfCircuitsToKeep : list of (tuples or Circuits)
            A list of the circuits for the new returned dataset.  If a
            circuit is given in this list that isn't in the original
            data set, `missingAction` determines the behavior.

        missingAction : {"raise","warn","ignore"}
            What to do when a string in `listOfCircuitsToKeep` is not in
            the data set (raise a KeyError, issue a warning, or do nothing).

        Returns
        -------
        DataSet
            The truncated data set.
        """
        missingStrs = [] # to issue warning - only used if missingAction=="warn"
        if self.bStatic:
            circuitIndices = []
            circuits = []
            for opstr in listOfCircuitsToKeep:
                circuit = opstr if isinstance(opstr, _cir.Circuit) else _cir.Circuit(opstr)

                if circuit not in self.cirIndex:
                    if missingAction == "raise":
                        raise KeyError(("Circuit %s was not found in "
                                        "dataset being truncated and "
                                        "`missingAction` == 'raise'") % str(circuit))
                    elif missingAction == "warn":
                        missingStrs.append(circuit)
                        continue
                    elif missingAction == "ignore":
                        continue
                    else:
                        raise ValueError("Invalid `missingAction`: %s" % str(missingAction))

                #only keep track of circuits if they could be different from listOfCircuitsToKeep
                if missingAction != "raise": circuits.append( circuit )
                circuitIndices.append( self.cirIndex[circuit] )

            if missingAction == "raise": circuits = listOfCircuitsToKeep
            trunc_cirIndex = _OrderedDict( zip(circuits, circuitIndices) )
            trunc_dataset = DataSet(self.oliData, self.timeData, self.repData,
                                    circuitIndices=trunc_cirIndex,
                                    outcomeLabelIndices=self.olIndex, bStatic=True) #don't copy counts, just reference
            
        else:
            trunc_dataset = DataSet(outcomeLabels=self.get_outcome_labels())
            for opstr in _lt.remove_duplicates(listOfCircuitsToKeep):
                circuit = opstr if isinstance(opstr, _cir.Circuit) else _cir.Circuit(opstr)
                if circuit in self.cirIndex:
                    circuitIndx = self.cirIndex[circuit]
                    repData = self.repData[ circuitIndx ].copy() if (self.repData is not None) else None
                    trunc_dataset.add_raw_series_data( circuit, [ self.ol[i] for i in self.oliData[ circuitIndx ] ],
                                                   self.timeData[ circuitIndx ].copy(), repData) #Copy operation so truncated dataset can be modified
                elif missingAction == "raise":
                    raise KeyError(("Circuit %s was not found in "
                                    "dataset being truncated and "
                                    "`missingAction` == 'raise'") % str(circuit))
                elif missingAction == "warn":
                    missingStrs.append(circuit)
                elif missingAction != "ignore":
                    raise ValueError("Invalid `missingAction`: %s" % str(missingAction))

        if len(missingStrs) > 0:
            missingStrs.append("...") # so elipses are shown when there's more strings
            _warnings.warn(("DataSet.truncate(...) was given %s strings to keep"
                            " that weren't in the original dataset:\n%s") % 
                           (len(missingStrs)-1, "\n".join(map(str,missingStrs[0:10]))))

        return trunc_dataset


    def time_slice(self, startTime, endTime, aggregateToTime=None):
        """
        Creates a DataSet by aggregating the counts within the
        [`startTime`,`endTime`) interval.

        Parameters
        ----------
        startTime, endTime : float or int
            The time-stamps to use for the beginning (inclusive) and end
            (exclusive) of the time interval.

        aggregateToTime : float, optional
            If not None, a single timestamp to give all the data in
            the specified range, resulting in time-independent
            `DataSet`.  If None, then the original timestamps are
            preserved.

        Returns
        -------
        DataSet
        """
        tot = 0
        ds = DataSet(outcomeLabelIndices=self.olIndex)
        for opStr,dsRow in self.items():

            if dsRow.reps is None:
                reps = _np.ones(dsRow.oli.shape, self.repType)
            else: reps = dsRow.reps

            count_dict = {ol: 0 for ol in self.olIndex.keys()}
            times = []; ols = []; repCnts = []
            for oli, t, rep in zip(dsRow.oli, dsRow.time, reps):

                ol = self.ol[oli] # index -> outcome label
                if startTime <= t < endTime:
                    if aggregateToTime is not None:
                        count_dict[ol] += rep
                    else:
                        times.append(t)
                        ols.append(ol)
                        repCnts.append(rep)
                    tot += rep

            if aggregateToTime is not None:
                ols = [ k for k in count_dict.keys() if count_dict[k] > 0 ]
                repCnts = [ count_dict[k] for k in ols ]
                times = [ aggregateToTime ]*len(repCnts)

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
        for opstr,indx in self.cirIndex.items():
            new_gstr = processor_fn(opstr)
            if new_gstr is None:
                to_delete.append(indx)
            elif new_gstr not in new_cirIndex or aggregate == False:
                assert(isinstance(new_gstr, _cir.Circuit)), "`processor_fn` must return a Circuit!"
                new_cirIndex[ new_gstr  ] = indx
            else: # aggregate data from indx --> new_cirIndex[new_gstr]
                # A subset of what is in add_raw_series_data(...), but we
                # don't need to do many of the checks there since the
                # incoming data is known to have no new outcome labels, etc.
                assert(isinstance(new_gstr, _cir.Circuit)), "`processor_fn` must return a Circuit!"
                iSrc, iDest = indx, new_cirIndex[new_gstr]
                self.oliData[ iDest ] = _np.concatenate((self.oliData[iDest],self.oliData[iSrc]))
                self.timeData[ iDest ] = _np.concatenate((self.timeData[iDest],self.timeData[iSrc]))
                if self.repData is not None:
                    self.repData[ iDest ] = _np.concatenate((self.repData[iDest],self.repData[iSrc]))
                #FUTURE: just add counts for same timestamp & same outcome
                #  label data? (and in add_raw_series_data(...) too).

                # mark indx for deletion (don't do it yet, as this will
                # mess up the values in new_cirIndex)
                to_delete.append(indx)

        self.cirIndex = new_cirIndex
        self._remove(to_delete)

        #Note: self.cnt_cache just remains None (a non-static DataSet)

        #Process self.auxInfo
        auxInfo = _DefaultDict( dict )
        for opstr in self.auxInfo.keys():
            new_gstr = processor_fn(opstr)
            if new_gstr is None:
                continue
            elif new_gstr not in auxInfo or aggregate == False:
                auxInfo[new_gstr] = self.auxInfo[opstr]
            else: # "aggregate" auxinfo by merging dictionaries
                #FUTURE: better merging - do something for key collisions?
                auxInfo[new_gstr].update( self.auxInfo[opstr] )
        self.auxInfo = auxInfo



    def remove(self, circuits, missingAction="raise"):
        """
        Remove (delete) the data for `circuits` from this :class:`DataSet`.

        Parameters
        ----------
        circuits : iterable
            An iterable over Circuit-like objects specifying the keys
            (circuits) to remove.

        missingAction : {"raise","warn","ignore"}
            What to do when a string in `circuits` is not in this data
            set (raise a KeyError, issue a warning, or do nothing).

        Returns
        -------
        None
        """
        missingStrs = [] # to issue warning - only used if missingAction=="warn"
        gstr_indices = []; auxkeys_to_remove = []
        for opstr in circuits:
            if not isinstance(opstr,_cir.Circuit):
                opstr = _cir.Circuit(opstr)

            if self.has_key(opstr):
                gstr_indices.append( self.cirIndex[opstr] )
                if opstr in self.auxInfo:
                    auxkeys_to_remove.append(opstr)
            elif missingAction == "raise":
                raise KeyError(("Circuit %s does not exist and therefore "
                                "cannot be removed when `missingAction` == "
                                "'raise'") % str(opstr))
            elif missingAction == "warn":
                missingStrs.append(opstr)
            elif missingAction != "ignore":
                raise ValueError("Invalid `missingAction`: %s" % str(missingAction))

        # the actual removal operations
        self._remove(gstr_indices) 
        for ky in auxkeys_to_remove:
             del self.auxInfo[ky]

        if len(missingStrs) > 0: #Print a warning with missing strings
            missingStrs.append("...") # so elipses are shown when there's more strings
            _warnings.warn(("DataSet.remove(...) cannot remove %s strings because"
                            " they don't exist in the original dataset:\n%s") % 
                           (len(missingStrs)-1, "\n".join(map(str,missingStrs[0:10]))))
            

    def _remove(self, gstr_indices):
        """ Removes the data in indices given by gstr_indices """
        if self.bStatic: raise ValueError("Cannot _remove on a static DataSet object")

        #Removing elements from oliData, timeData, and repData is easy since
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
        for k,v in self.cirIndex.items():
            if v in inds_set: 
                keys_to_delete.append(k)
        for k in keys_to_delete:
            del self.cirIndex[k]

        #adjust remaining indices in cirIndex
        inds_ar = _np.array(inds, _np.int64)
        for k in self.cirIndex.keys():
            cnt = _bisect.bisect(inds_ar, self.cirIndex[k]) # cnt == number of removed 
            self.cirIndex[k] -= cnt                         # indices < self.cirIndex[k]


            
    def copy(self):
        """ Make a copy of this DataSet. """
        if self.bStatic:
            return self # doesn't need to be copied since data can't change
        else:
            copyOfMe = DataSet(outcomeLabels=self.get_outcome_labels(),
                               collisionAction=self.collisionAction)
            copyOfMe.cirIndex = _copy.deepcopy(self.cirIndex)
            copyOfMe.oliData = [ el.copy() for el in self.oliData ]
            copyOfMe.timeData = [ el.copy() for el in self.timeData ]
            if self.repData is not None:
                copyOfMe.repData = [ el.copy() for el in self.repData ]
            else: copyOfMe.repData = None

            copyOfMe.oliType  =self.oliType
            copyOfMe.timeType = self.timeType
            copyOfMe.repType  = self.repType
            copyOfMe.cnt_cache = None
            copyOfMe.auxInfo = self.auxInfo.copy()
            return copyOfMe


    def copy_nonstatic(self):
        """ Make a non-static copy of this DataSet. """
        if self.bStatic:
            copyOfMe = DataSet(outcomeLabels=self.get_outcome_labels(),
                               collisionAction=self.collisionAction)
            copyOfMe.cirIndex = _OrderedDict([ (opstr,i) for i,opstr in enumerate(self.cirIndex.keys()) ])
            copyOfMe.oliData = []
            copyOfMe.timeData = []
            copyOfMe.repData = None if (self.repData is None) else []
            for slc in self.cirIndex.values():
                copyOfMe.oliData.append( self.oliData[slc].copy() )
                copyOfMe.timeData.append( self.timeData[slc].copy() )
                if self.repData is not None:
                    copyOfMe.repData.append( self.repData[slc].copy() )

            copyOfMe.oliType  = self.oliType
            copyOfMe.timeType = self.timeType
            copyOfMe.repType  = self.repType
            copyOfMe.cnt_cache = None
            copyOfMe.auxInfo = self.auxInfo.copy()
            return copyOfMe
        else:
            return self.copy()


    def done_adding_data(self):
        """
        Promotes a non-static DataSet to a static (read-only) DataSet.  This
         method should be called after all data has been added.
        """
        if self.bStatic: return
        #Convert normal dataset to static mode.
        #  olIndex stays the same
        #  cirIndex changes to hold slices into 1D arrays
        #  oliData, timeData, & repData change from being lists of arrays to
        #    single 1D arrays.

        if len(self.oliData) > 0:
            new_cirIndex = _OrderedDict()
            curIndx = 0
            to_concat_oli = []
            to_concat_time = []
            to_concat_rep = []
            for circuit, indx in self.cirIndex.items():
                seriesLen = len(self.oliData[indx])

                to_concat_oli.append( self.oliData[indx] )   #just build up lists of
                to_concat_time.append( self.timeData[indx] ) # reference, not copies
                assert(seriesLen == len(self.timeData[indx])), "TIME & OLI out of sync!"

                if self.repData is not None:
                    to_concat_rep.append( self.repData[indx] )
                    assert(seriesLen == len(self.repData[indx])), "REP & OLI out of sync!"

                new_cirIndex[circuit] = slice(curIndx, curIndx+seriesLen)
                curIndx += seriesLen

            self.cirIndex = new_cirIndex
            self.oliData = _np.concatenate( to_concat_oli )
            self.timeData = _np.concatenate( to_concat_time )
            if self.repData is not None:
                self.repData = _np.concatenate( to_concat_rep )

        else:
            #leave cirIndex alone (should be empty anyway?)
            self.oliData = _np.empty( (0,), self.oliType)
            self.timeData = _np.empty( (0,), self.timeType)
            if self.repData is not None:
                self.repData = _np.empty( (0,), self.repType)

        self.cnt_cache = { opstr:_ld.OutcomeLabelDict() for opstr in self.cirIndex }
        self.bStatic = True
        self.uuid = _uuid.uuid4()

    def __getstate__(self):
        toPickle = { 'cirIndexKeys': list(map(_cir.CompressedCircuit, self.cirIndex.keys())),
                     'cirIndexVals': list(self.cirIndex.values()),
                     'olIndex': self.olIndex,
                     'ol': self.ol,
                     'bStatic': self.bStatic,
                     'oliData': self.oliData,
                     'timeData': self.timeData,
                     'repData': self.repData,
                     'oliType': _np.dtype(self.oliType).str,
                     'timeType': _np.dtype(self.timeType).str,
                     'repType': _np.dtype(self.repType).str,
                     'collisionAction': self.collisionAction,
                     'uuid' : self.uuid,
                     'auxInfo': self.auxInfo,
                     'comment': self.comment}
        return toPickle

    def __setstate__(self, state_dict):
        bStatic = state_dict['bStatic']
        
        if "gsIndexKeys" in state_dict:
            _warnings.warn("Unpickling a deprecated-format DataSet.  Please re-save/pickle asap.")
            cirIndexKeys = [ cgstr.expand() for cgstr in state_dict['gsIndexKeys'] ]
            cirIndex = _OrderedDict( list(zip( cirIndexKeys, state_dict['gsIndexVals'])) )
        else:
            cirIndexKeys = [ cgstr.expand() for cgstr in state_dict['cirIndexKeys'] ]
            cirIndex = _OrderedDict( list(zip( cirIndexKeys, state_dict['cirIndexVals'])) )


        if "slIndex" in state_dict:
            #print("DB: UNPICKLING AN OLD DATASET"); print("Keys = ",state_dict.keys())
            _warnings.warn("Unpickling a *very* deprecated-format DataSet.  Please re-save/pickle asap.")

            #Turn spam labels into outcome labels
            self.cirIndex = _OrderedDict()
            self.olIndex = _OrderedDict( [ ((str(sl),),i) for sl,i in state_dict['slIndex'].items() ] )
            self.ol = _OrderedDict( [(i,ol) for (ol,i) in self.olIndex.items()] )
            self.oliData = []
            self.timeData = []
            self.repData = []
            self.comment = ''

            self.oliType  = Oindex_type
            self.timeType = Time_type
            self.repType  = Repcount_type

            self.bStatic = False # for adding data
            for opstr, indx in cirIndex.items():
                count_row = state_dict['counts'][indx]
                count_dict = _OrderedDict( [(ol,count_row[i]) for ol,i in self.olIndex.items()] )
                self.add_count_dict(opstr, count_dict)
            if not self.bStatic: self.done_adding_data()

        else:  #Normal case
            self.bStatic = bStatic
            self.cirIndex = cirIndex
            self.olIndex = state_dict['olIndex']
            self.ol = state_dict['ol']
            self.oliData  = state_dict['oliData']
            self.timeData = state_dict['timeData']
            self.repData  = state_dict['repData']
            self.oliType  = _np.dtype(state_dict['oliType'])
            self.timeType = _np.dtype(state_dict['timeType'])
            self.repType  = _np.dtype(state_dict['repType'])
            self.comment  = state_dict.get('comment','')
            if bStatic: #always empty - don't save this, just init
                self.cnt_cache = { opstr:_ld.OutcomeLabelDict() for opstr in self.cirIndex }
            else: self.cnt_cache = None

        self.auxInfo = state_dict.get('auxInfo', _DefaultDict(dict) )
        if not isinstance(self.auxInfo, _DefaultDict) and isinstance(self.auxInfo,dict):
            self.auxInfo = _DefaultDict(dict, self.auxInfo)
            # some types of serialization (e.g. JSON) just save a *normal* dict
            # so promote to a defaultdict if needed..
            
        self.collisionAction = state_dict.get('collisionAction','aggregate')
        self.uuid = state_dict.get('uuid',None)



    def save(self, fileOrFilename):
        """
        Save this DataSet to a file.

        Parameters
        ----------
        fileOrFilename : string or file object
            If a string,  interpreted as a filename.  If this filename ends
            in ".gz", the file will be gzip compressed.

        Returns
        -------
        None
        """

        toPickle = { 'cirIndexKeys': list(map(_cir.CompressedCircuit, self.cirIndex.keys())) if self.cirIndex else [],
                     'cirIndexVals': list(self.cirIndex.values()) if self.cirIndex else [],
                     'olIndex': self.olIndex,
                     'ol': self.ol,
                     'bStatic': self.bStatic,
                     'oliType': self.oliType,
                     'timeType': self.timeType,
                     'repType': self.repType,
                     'useReps': bool(self.repData is not None),
                     'collisionAction': self.collisionAction,
                     'uuid' : self.uuid,
                     'auxInfo': self.auxInfo,
                     'comment': self.comment } #Don't pickle counts numpy data b/c it's inefficient
        if not self.bStatic: toPickle['nRows'] = len(self.oliData)

        bOpen = _compat.isstr(fileOrFilename)
        if bOpen:
            if fileOrFilename.endswith(".gz"):
                import gzip as _gzip
                f = _gzip.open(fileOrFilename,"wb")
            else:
                f = open(fileOrFilename,"wb")
        else:
            f = fileOrFilename

        _pickle.dump(toPickle,f)
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

    def load(self, fileOrFilename):
        """
        Load DataSet from a file, clearing any data is contained previously.

        Parameters
        ----------
        fileOrFilename string or file object.
            If a string,  interpreted as a filename.  If this filename ends
            in ".gz", the file will be gzip uncompressed as it is read.

        Returns
        -------
        None
        """
        bOpen = _compat.isstr(fileOrFilename)
        if bOpen:
            if fileOrFilename.endswith(".gz"):
                import gzip as _gzip
                f = _gzip.open(fileOrFilename,"rb")
            else:
                f = open(fileOrFilename,"rb")
        else:
            f = fileOrFilename

        try:
            state_dict = _pickle.load(f)
        except AttributeError:
            # HACK TO ALLOW UUIDs saved on python3.7 work with earlier python versions that don't have uuid.SafeUUID
            # HACK - maybe move this to leagacyio to deal with Python 3 versions < 3.7 not having SafeUUID?
            class dummy_SafeUUID(object): 
                def __new__(self,*args):  
                    return _uuid.UUID.__new__(uuid.UUID,*args)
            _sys.modules['uuid'].SafeUUID = dummy_SafeUUID
            state_dict = _pickle.load(f)
            del _sys.modules['uuid'].SafeUUID
            

        if "gsIndexKeys" in state_dict:
            _warnings.warn("Loading a deprecated-format DataSet.  Please re-save asap.")
            state_dict['cirIndexKeys'] = state_dict['gsIndexKeys']
            state_dict['cirIndexVals'] = state_dict['gsIndexVals']
            del state_dict['gsIndexKeys']
            del state_dict['gsIndexVals']

            if _sys.version_info < (3, 0): #for backward compatibility, needed for Python2 only
                new_aux_info = _DefaultDict(dict)  # where GateStrings don't get up-converted to Circuits b/c __new__ isn't called
                for gstr,val in state_dict['auxInfo'].items():
                    new_aux_info[ _cir.Circuit(gstr._tup, stringrep=gstr._str) ] = val
                state_dict['auxInfo'] = new_aux_info
        
        def expand(x): #to be backward compatible
            """ Expand a compressed circuit """
            if isinstance(x,_cir.CompressedCircuit): return x.expand()
            elif hasattr(x,'__class__') and x.__class__.__name__ == "dummy_CompressedGateString":
                return _cir.Circuit(_cir.CompressedCircuit.expand_op_label_tuple(x._tup), stringrep=x.str)
                  #for backward compatibility, needed for Python2 only, which doesn't call __new__ when
                  # unpickling protocol=0 (the default) info.
            else:
                _warnings.warn("Deprecated dataset format.  Please re-save " +
                               "this dataset soon to avoid future incompatibility.")
                return _cir.Circuit(_cir.CompressedCircuit.expand_op_label_tuple(x))
        cirIndexKeys = [ expand(cgstr) for cgstr in state_dict['cirIndexKeys'] ]

        #cirIndexKeys = [ cgs.expand() for cgs in state_dict['cirIndexKeys'] ]
        self.cirIndex = _OrderedDict( list(zip( cirIndexKeys, state_dict['cirIndexVals'])) )
        self.olIndex = state_dict['olIndex']
        self.ol      = state_dict['ol']
        self.bStatic = state_dict['bStatic']
        self.oliType = state_dict['oliType']
        self.timeType= state_dict['timeType']
        self.repType = state_dict['repType']
        self.collisionAction = state_dict['collisionAction']
        self.uuid    = state_dict['uuid']
        self.auxInfo = state_dict.get('auxInfo', _DefaultDict(dict)) #backward compat
        self.comment = state_dict.get('comment', '') # backward compat

        useReps = state_dict['useReps']

        if self.bStatic:
            self.oliData = _np.lib.format.read_array(f) #_np.load(f) doesn't play nice with gzip
            self.timeData = _np.lib.format.read_array(f) #_np.load(f) doesn't play nice with gzip
            if useReps:
                self.repData = _np.lib.format.read_array(f) #_np.load(f) doesn't play nice with gzip
            self.cnt_cache = { opstr:_ld.OutcomeLabelDict() for opstr in self.cirIndex } # init cnt_cache afresh
        else:
            self.oliData = []
            for _ in range(state_dict['nRows']):
                self.oliData.append( _np.lib.format.read_array(f) ) #_np.load(f) doesn't play nice with gzip

            self.timeData = []
            for _ in range(state_dict['nRows']):
                self.timeData.append( _np.lib.format.read_array(f) ) #_np.load(f) doesn't play nice with gzip

            if useReps:
                self.repData = []
                for _ in range(state_dict['nRows']):
                    self.repData.append( _np.lib.format.read_array(f) ) #_np.load(f) doesn't play nice with gzip
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
        for old,new in old_to_new_dict.items():
            if _compat.isstr(old): old = (old,)
            if _compat.isstr(new): new = (new,)
            mapdict[old] = new
            
        new_olIndex = _OrderedDict()
        for ol,i in self.olIndex.items():
            if ol in mapdict:
                new_olIndex[mapdict[ol]] = i
            else:
                new_olIndex[ol] = i

        #Note: rebuild reverse-dict self.ol:
        self.olIndex = new_olIndex
        self.ol = _OrderedDict( [(i,ol) for (ol,i) in self.olIndex.items()] )
