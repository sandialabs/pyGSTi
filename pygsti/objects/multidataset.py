""" Defines the MultiDataSet class and supporting classes and functions """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import pickle as _pickle
import copy as _copy
from collections import OrderedDict as _OrderedDict
from collections import defaultdict as _DefaultDict

from .dataset import DataSet as _DataSet
from . import circuit as _cir
from . import labeldicts as _ld


class MultiDataSet_KeyValIterator(object):
    """ Iterator class for datasetName,DataSet pairs of a MultiDataSet """

    def __init__(self, multidataset):
        self.multidataset = multidataset
        self.oliDictIter = multidataset.oliDict.__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        datasetName = next(self.oliDictIter)
        oliData = self.multidataset.oliDict[datasetName]
        timeData = self.multidataset.timeDict[datasetName]
        if self.multidataset.repDict:
            repData = self.multidataset.repDict[datasetName]
        else:
            repData = None

        ds = _DataSet(oliData, timeData, repData,
                      circuitIndices=self.multidataset.cirIndex,
                      outcomeLabelIndices=self.multidataset.olIndex,
                      collisionAction=self.multidataset.collisionActions[datasetName],
                      bStatic=True, auxInfo=None)
        ds.auxInfo = self.multidataset.auxInfo  # avoids shallow-copying dict
        return datasetName, ds

    next = __next__


class MultiDataSet_ValIterator(object):
    """ Iterator class for DataSets of a MultiDataSet """

    def __init__(self, multidataset):
        self.multidataset = multidataset
        self.oliDictIter = multidataset.oliDict.__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        datasetName = next(self.oliDictIter)
        oliData = self.multidataset.oliDict[datasetName]
        timeData = self.multidataset.timeDict[datasetName]
        if self.multidataset.repDict:
            repData = self.multidataset.repDict[datasetName]
        else:
            repData = None

        ds = _DataSet(oliData, timeData, repData,
                      circuitIndices=self.multidataset.cirIndex,
                      outcomeLabelIndices=self.multidataset.olIndex,
                      collisionAction=self.multidataset.collisionActions[datasetName],
                      bStatic=True, auxInfo=None)
        ds.auxInfo = self.multidataset.auxInfo  # avoids shallow-copying dict
        return ds

    next = __next__


class MultiDataSet(object):
    """
    The MultiDataSet class allows for the combined access and storage of
    several static DataSets that contain the same circuits (in the same
    order) AND the same time-dependence structure (if applicable).

    It is designed to behave similarly to a dictionary of DataSets, so that
    a DataSet is obtained by:

    `dataset = multiDataset[datasetName]`

    where `datasetName` may be a string OR a tuple.
    """

    def __init__(self, oliDict=None, timeDict=None, repDict=None,
                 circuitIndices=None,
                 outcomeLabels=None, outcomeLabelIndices=None,
                 fileToLoadFrom=None, collisionActions=None,
                 comment=None, comments=None, auxInfo=None):
        """
        Initialize a MultiDataSet.

        Parameters
        ----------
        oliDict : ordered dictionary, optional
          Keys specify dataset names.  Values are 1D numpy arrays which specify
          outcome label indices.  Each value is indexed by the values of
          `circuitIndices`.

        timeDict : ordered dictionary, optional
          Same format as `oliDict` except stores arrays of floating-point time
          stamp data.

        repDict :  ordered dictionary, optional
          Same format as `oliDict` except stores arrays of integer repetition
          counts (can be `None` if there are no repetitions)

        circuitIndices : ordered dictionary, optional
          An OrderedDict with keys equal to circuits (tuples of operation labels) and values equal to
          integer indices associating a row/element of counts with the circuit.

        outcomeLabels : list of strings
          Specifies the set of spam labels for the DataSet.  Indices for the spam labels
          are assumed to ascend from 0, starting with the first element of this list.  These
          indices will associate each elememtn of `timeseries` with a spam label.  Only
          specify this argument OR outcomeLabelIndices, not both.

        outcomeLabelIndices : ordered dictionary
          An OrderedDict with keys equal to spam labels (strings) and value equal to
          integer indices associating a spam label with given index.  Only
          specify this argument OR outcomeLabels, not both.

        fileToLoadFrom : string or file object, optional
          Specify this argument and no others to create a MultiDataSet by loading
          from a file (just like using the load(...) function).

        collisionActions : dictionary, optional
            Specifies how duplicate circuits should be handled for the data
            sets.  Keys must match those of `oliDict` and values are "aggregate"
            or "keepseparate".  See documentation for :class:`DataSet`.  If None,
            then "aggregate" is used for all sets by default.

        comment : string, optional
            A user-specified comment string that gets carried around with the
            data.  A common use for this field is to attach to the data details
            regarding its collection.

        comments : dict, optional
            A user-specified dictionary of comments, one per dataset.  Keys
            are dataset names (same as `oliDict` keys).

        auxInfo : dict, optional
            A user-specified dictionary of per-circuit auxiliary information.
            Keys should be the circuits in this MultiDataSet and value should
            be Python dictionaries.

        Returns
        -------
        MultiDataSet
           a new multi data set object.
        """

        #Optionally load from a file
        if fileToLoadFrom is not None:
            assert(oliDict is None and timeDict is None and repDict is None
                   and circuitIndices is None and outcomeLabels is None and outcomeLabelIndices is None)
            self.load(fileToLoadFrom)
            return

        # self.cirIndex  :  Ordered dictionary where keys = circuits (tuples), values = integer indices into counts
        if circuitIndices is not None:
            self.cirIndex = circuitIndices
        else:
            self.cirIndex = None

            # self.olIndex  :  Ordered dictionary where
        #                  keys = outcome labels (strings or tuples),
        #                  values = integer indices mapping oliData (integers) onto
        #                           the outcome labels.
        if outcomeLabelIndices is not None:
            self.olIndex = outcomeLabelIndices
        elif outcomeLabels is not None:
            tup_outcomeLabels = [_ld.OutcomeLabelDict.to_outcome(ol)
                                 for ol in outcomeLabels]  # strings -> tuple outcome labels
            self.olIndex = _OrderedDict([(ol, i) for (i, ol) in enumerate(tup_outcomeLabels)])
        else:
            self.olIndex = None

        #if self.cirIndex:  #Note: tests if not none and nonempty
        #    assert( min(self.cirIndex.values()) >= 0) # values are *slices* so can't test like this
        if self.olIndex:  # Note: tests if not none and nonempty
            assert(min(self.olIndex.values()) >= 0)

        # self.*Dict : dictionaries of 1D numpy arrays, each corresponding to a DataSet.
        if oliDict is not None:
            self.oliDict = _OrderedDict()
            self.timeDict = _OrderedDict()
            self.repDict = _OrderedDict() if repDict else None

            assert(timeDict is not None), "Must specify `timeDict` also!"

            for name, outcomeInds in oliDict.items():
                assert(name in timeDict), "`timeDict` arg is missing %s key!" % name
                self.oliDict[name] = outcomeInds  # copy OrderedDict but share arrays
                self.timeDict[name] = timeDict[name]
                if repDict: self.repDict[name] = repDict[name]

            if collisionActions is None: collisionActions = {}  # allow None to function as an empty dict
            self.collisionActions = _OrderedDict([(name, collisionActions.get(name, "aggregate"))
                                                  for name in self.oliDict.keys()])

            if comments is None: comments = {}  # allow None to function as an empty dict
            self.comments = _OrderedDict([(name, comments.get(name, None))
                                          for name in self.oliDict.keys()])

            if self.olIndex:  # Note: tests if not none and nonempty
                maxOlIndex = max(self.olIndex.values())
                for outcomeInds in self.oliDict.values():
                    assert(_np.amax(outcomeInds) <= maxOlIndex)
        else:
            assert(timeDict is None), "Must specify `oliDict` also!"
            assert(repDict is None), "Must specify `oliDict` also!"
            self.oliDict = _OrderedDict()
            self.timeDict = _OrderedDict()
            self.repDict = None
            self.collisionActions = _OrderedDict()
            self.comments = _OrderedDict()

        # comment
        self.comment = comment

        #auxiliary info
        if auxInfo is None:
            self.auxInfo = _DefaultDict(dict)
        else:
            self.auxInfo = _DefaultDict(dict, auxInfo)

        #data types - should stay in sync with DataSet
        self.oliType = _np.uint32
        self.timeType = _np.float64
        self.repType = _np.float32
        # thought: _np.uint16 but doesn't play well with rescaling

    def get_outcome_labels(self):
        """
        Get a list of *all* the outcome labels contained in this MultiDataSet.

        Returns
        -------
        list of strings or tuples
          A list where each element is an outcome label (which can
          be a string or a tuple of strings).
        """
        if self.olIndex is not None:
            return list(self.olIndex.keys())
        else: return None

    def __iter__(self):
        return self.oliDict.__iter__()  # iterator over dataset names

    def __len__(self):
        return len(self.oliDict)

    def __getitem__(self, datasetName):  # return a static DataSet
        repData = self.repDict[datasetName] if self.repDict else None
        ds = _DataSet(self.oliDict[datasetName],
                      self.timeDict[datasetName], repData,
                      circuitIndices=self.cirIndex,
                      outcomeLabelIndices=self.olIndex, bStatic=True,
                      collisionAction=self.collisionActions[datasetName],
                      auxInfo=None)
        ds.auxInfo = self.auxInfo  # avoids shallow-copying dict
        return ds

    def __setitem__(self, datasetName, dataset):
        self.add_dataset(datasetName, dataset)

    def __contains__(self, datasetName):
        return datasetName in self.oliDict

    def keys(self):
        """ Returns a list of the keys (dataset names) of this MultiDataSet """
        return list(self.oliDict.keys())

    def items(self):
        """ Iterator over (dataset name, DataSet) pairs """
        return MultiDataSet_KeyValIterator(self)

    def values(self):
        """ Iterator over DataSets corresponding to each dataset name """
        return MultiDataSet_ValIterator(self)

    def get_datasets_aggregate(self, *datasetNames):
        """
        Generate a new DataSet by combining the outcome counts of multiple
        member Datasets.  Data with the same time-stamp and outcome are
        merged into a single "bin" in the returned :class:`DataSet`.

        Parameters
        ----------
        datasetNames : one or more dataset names.

        Returns
        -------
        DataSet
            a single DataSet containing the summed counts of each of the datasets
            named by the parameters.
        """

        if len(datasetNames) == 0: raise ValueError("Must specify at least one dataset name")
        for datasetName in datasetNames:
            if datasetName not in self:
                raise ValueError("No dataset with the name '%s' exists" % datasetName)

        #add data for each gate sequence to build up aggregate lists
        gstrSlices = _OrderedDict()
        agg_oli = []; agg_time = []; agg_rep = []; slc_i = 0
        for opstr, slc in self.cirIndex.items():
            concat_oli = _np.concatenate([self.oliDict[datasetName][slc]
                                          for datasetName in datasetNames], axis=0)
            concat_time = _np.concatenate([self.timeDict[datasetName][slc]
                                           for datasetName in datasetNames], axis=0)
            if self.repDict:
                concat_rep = _np.concatenate([self.repDict[datasetName][slc]
                                              for datasetName in datasetNames], axis=0)
            else:
                concat_rep = None

            # Merge same-timestamp, same-outcome data
            sortedInds = [i for i, el in sorted(enumerate(concat_time), key=lambda x:x[1])]
            sorted_oli = []; sorted_time = []; sorted_rep = []
            last_time = concat_time[sortedInds[0]]; cur_reps = {}
            for i in sortedInds:
                if concat_time[i] != last_time:
                    # dump cur_reps at last_time and reset
                    for oli, reps in cur_reps.items():
                        sorted_time.append(last_time)
                        sorted_oli.append(oli)
                        sorted_rep.append(reps)
                    last_time = concat_time[i]; cur_reps = {}

                #add i-th element data to cur_reps
                reps = concat_rep[i] if (concat_rep is not None) else 1
                if concat_oli[i] in cur_reps:
                    cur_reps[concat_oli[i]] += reps
                else:
                    cur_reps[concat_oli[i]] = reps

            # dump cur_reps at last_time
            for oli, reps in cur_reps.items():
                sorted_time.append(last_time)
                sorted_oli.append(oli)
                sorted_rep.append(reps)

            agg_oli.extend(sorted_oli)
            agg_time.extend(sorted_time)
            agg_rep.extend(sorted_rep)

            gstrSlices[opstr] = slice(slc_i, slc_i + len(sorted_oli))
            slc_i += len(sorted_oli)

        agg_oli = _np.array(agg_oli, self.oliType)
        agg_time = _np.array(agg_time, self.timeType)
        agg_rep = _np.array(agg_rep, self.repType)
        if _np.max(agg_rep) == 1: agg_rep = None  # don't store trivial reps

        ds = _DataSet(agg_oli, agg_time, agg_rep,
                      circuitIndices=gstrSlices,
                      outcomeLabelIndices=self.olIndex, bStatic=True,
                      auxInfo=None)  # leave collisionAction as default "aggregate"
        ds.auxInfo = self.auxInfo  # avoids shallow-copying dict
        return ds

    def add_dataset(self, datasetName, dataset, update_auxinfo=True):
        """
        Add a DataSet to this MultiDataSet.  The dataset
        must be static and conform with the circuits and
        time-dependent structure passed upon construction or
        those inherited from the first dataset added.

        Parameters
        ----------
        datasetName : string
            The name to give the added dataset (i.e. the key the new
            data set will be referenced by).

        dataset : DataSet
            The data set to add.

        update_auxinfo : bool, optional
            Whether the auxiliary information (if any exists) in `dataset` is added to
            the information already stored in this `MultiDataSet`.
        """

        #Check if dataset is compatible
        if not dataset.bStatic:
            raise ValueError("Cannot add dataset: only static DataSets can be added to a MultiDataSet")
        if self.cirIndex is not None and set(dataset.cirIndex.keys()) != set(self.cirIndex.keys()):
            raise ValueError("Cannot add dataset: circuits do not match")

        if self.cirIndex is None:
            self.cirIndex = dataset.cirIndex.copy()  # copy b/c we may modify our cirIndex later

        if self.olIndex is None:
            self.olIndex = dataset.olIndex.copy()  # copy b/c we may modify our olIndex later

        # Check if outcome labels use the same indexing; if not, update dataset.oliData
        ds_oliData = dataset.oliData  # default - just use dataset's outcome indices as is...
        update_map = {}  # default - no updates needed
        if dataset.olIndex != self.olIndex:
            next_olIndex = max(self.olIndex.values()) + 1
            for ol, i in dataset.olIndex.items():
                if ol not in self.olIndex:  # then add a new outcome label
                    self.olIndex[ol] = next_olIndex; next_olIndex += 1
                if self.olIndex[ol] != i:  # if we need to update dataset's indices
                    update_map[i] = self.olIndex[ol]  # old -> new index
            if update_map:
                # update dataset's outcome indices to conform to self.olIndex
                ds_oliData = _np.array([update_map.get(oli, oli)
                                        for oli in dataset.oliData], self.oliType)

        #Do easy additions
        self.collisionActions[datasetName] = dataset.collisionAction
        self.comments[datasetName] = dataset.comment

        #Add dataset's data to self.oliDict, self.timeDict and self.repDict,
        # updating cirIndex and existing data if needed
        if len(self.oliDict) == 0:
            # this is the first added DataSet, so we can just overwrite
            # cirIndex even if it isn't None.
            self.cirIndex = dataset.cirIndex.copy()  # copy b/c we may modify our cirIndex later

            #And then add data:
            self.oliDict[datasetName] = ds_oliData
            self.timeDict[datasetName] = dataset.timeData
            if dataset.repData is not None:
                self.repDict = _OrderedDict()  # OK since this is the first DataSet added
                self.repDict[datasetName] = dataset.repData

        elif dataset.cirIndex == self.cirIndex:
            # self.cirIndex is fine as is - no need to reconcile anything
            self.oliDict[datasetName] = ds_oliData
            self.timeDict[datasetName] = dataset.timeData
            if dataset.repData is not None:
                if self.repDict is None:  # then build self.repDict
                    firstKey = list(self.oliDict.keys())[0]
                    dataLen = len(self.oliDict[firstKey])  # current length of self's data arrays
                    self.repDict = _OrderedDict(
                        [(nm, _np.ones(dataLen, self.repType)) for nm in self])
                self.repDict[datasetName] = dataset.repData

        else:
            # Merge data due to differences in self.cirIndex and dataset.cirIndex, which can
            # occur because dataset may have zeros in different places from self.
            firstKey = list(self.oliDict.keys())[0]
            dataLen = len(self.oliDict[firstKey])  # current length of self's data arrays

            #We'll need 0-rep elements to reconcile, so can't have repDict == None
            if self.repDict is None:  # then create self.repDict
                self.repDict = _OrderedDict(
                    [(nm, _np.ones(dataLen, self.repType)) for nm in self])

            ds_timeData = dataset.timeData
            ds_repData = dataset.repData if (dataset.repData is not None) \
                else _np.ones(len(ds_oliData), self.repType)

            #Create new arrays to populate with dataset's data
            self.oliDict[datasetName] = _np.zeros(dataLen, self.oliType)
            self.timeDict[datasetName] = _np.zeros(dataLen, self.timeType)
            self.repDict[datasetName] = _np.zeros(dataLen, self.repType)

            #sort keys in order of increasing slice-start position (w.r.t. self)
            sorted_cirIndex = sorted(list(self.cirIndex.items()), key=lambda x: x[1].start)

            off = 0
            for opstr, slc in sorted_cirIndex:
                other_slc = dataset.cirIndex[opstr]  # we know key exists from check above
                l1 = slc.stop - slc.start; assert(slc.step is None)
                l2 = other_slc.stop - other_slc.start; assert(slc.step is None)

                #Update cirIndex - (expands slice if l2 > l1; adds in offset)
                l = max(l1, l2)
                new_slc = slice(off + slc.start, off + slc.start + l)
                self.cirIndex[opstr] = new_slc

                #Update existing data & new data arrays
                if l2 > l1:  # insert 0-reps into self's data arrays
                    nToInsert = l2 - l1; insertAt = off + slc.stop
                    for nm in self:
                        oliVal = self.oliDict[nm][insertAt - 1]    # just repeat the final bin's outcome
                        timeVal = self.timeDict[nm][insertAt - 1]  # index and timestamp in 0-count bins
                        self.oliDict[nm] = _np.insert(self.oliDict[nm], insertAt,
                                                      oliVal * _np.ones(nToInsert, self.oliType))
                        self.timeDict[nm] = _np.insert(self.timeDict[nm], insertAt,
                                                       timeVal * _np.ones(nToInsert, self.timeType))
                        self.repDict[nm] = _np.insert(self.repDict[nm], insertAt,
                                                      _np.zeros(nToInsert, self.repType))
                    off += nToInsert

                #Copy dataset's data into new_ arrays (already padded as needed above)
                if l2 >= l1:  # no need to pad dataset's array
                    self.oliDict[datasetName][new_slc] = ds_oliData[other_slc]
                    self.timeDict[datasetName][new_slc] = ds_timeData[other_slc]
                    self.repDict[datasetName][new_slc] = ds_repData[other_slc]
                else:  # l2 < l1 - need to pad dataset arrays w/0-counts
                    new_slc_part1 = slice(new_slc.start, new_slc.start + l2)
                    new_slc_part2 = slice(new_slc.start + l2, new_slc.stop)
                    self.oliDict[datasetName][new_slc_part1] = ds_oliData[other_slc]
                    self.timeDict[datasetName][new_slc_part1] = ds_timeData[other_slc]
                    self.repDict[datasetName][new_slc_part1] = ds_repData[other_slc]

                    timeVal = ds_timeData[other_slc.stop - 1]  # index and timestamp in 0-count bins
                    oliVal = ds_oliData[other_slc.stop - 1]       # just repeat the final bin's outcome
                    self.oliDict[datasetName][new_slc_part2] = oliVal
                    self.timeDict[datasetName][new_slc_part2] = timeVal
                    # (leave self.repDict[datasetName] with zeros in remaining "part2" of slice)

        # Update auxInfo
        if update_auxinfo and dataset.auxInfo:
            if len(self.auxInfo) == 0:
                self.auxInfo.update(dataset.auxInfo)
            else:
                for circuit, aux in dataset.auxInfo.items():
                    self.auxInfo[circuit].update(aux)

    def __str__(self):
        s = "MultiDataSet containing: %d datasets, each with %d strings\n" % (
            len(self), len(self.cirIndex) if self.cirIndex is not None else 0)
        s += " Dataset names = " + ", ".join(list(self.keys())) + "\n"
        s += " Outcome labels = " + ", ".join(map(str, self.olIndex.keys()) if self.olIndex is not None else [])
        if self.cirIndex is not None:
            s += "\nGate strings: \n" + "\n".join(map(str, list(self.cirIndex.keys())))
        return s + "\n"

    def copy(self):
        """ Make a copy of this MultiDataSet """
        return MultiDataSet(self.oliDict, self.timeDict, self.repDict,
                            circuitIndices=_copy.deepcopy(self.cirIndex) if (self.cirIndex is not None) else None,
                            outcomeLabelIndices=_copy.deepcopy(self.olIndex) if (self.olIndex is not None) else None,
                            collisionActions=self.collisionActions, comments=_copy.deepcopy(self.comments),
                            comment=(self.comment + " copy") if self.comment else None, auxInfo=self.auxInfo)

    def __getstate__(self):
        toPickle = {'cirIndexKeys': list(map(_cir.CompressedCircuit,
                                             list(self.cirIndex.keys()))) if self.cirIndex else [],
                    'cirIndexVals': list(self.cirIndex.values()) if self.cirIndex else [],
                    'olIndex': self.olIndex,
                    'oliDict': self.oliDict,
                    'timeDict': self.timeDict,
                    'repDict': self.repDict,
                    'collisionActions': self.collisionActions,
                    'auxInfo': self.auxInfo,
                    'comments': self.comments,
                    'comment': self.comment}
        return toPickle

    def __setstate__(self, state_dict):
        cirIndexKeys = [cgs.expand() for cgs in state_dict['cirIndexKeys']]
        self.cirIndex = _OrderedDict(list(zip(cirIndexKeys, state_dict['cirIndexVals'])))
        self.olIndex = state_dict['olIndex']
        self.oliDict = state_dict['oliDict']
        self.timeDict = state_dict['timeDict']
        self.repDict = state_dict['repDict']
        self.collisionActions = state_dict['collisionActions']
        self.comments = state_dict['comments']
        self.comment = state_dict['comment']

        self.auxInfo = state_dict.get('auxInfo', _DefaultDict(dict))
        if not isinstance(self.auxInfo, _DefaultDict) and isinstance(self.auxInfo, dict):
            self.auxInfo = _DefaultDict(dict, self.auxInfo)
            # some types of serialization (e.g. JSON) just save a *normal* dict
            # so promote to a defaultdict if needed..

    def save(self, fileOrFilename):
        """
        Save this MultiDataSet to a file.

        Parameters
        ----------
        fileOrFilename : file or string
            Either a filename or a file object.  In the former case, if the
            filename ends in ".gz", the file will be gzip compressed.
        """

        toPickle = {'cirIndexKeys': list(map(_cir.CompressedCircuit,
                                             list(self.cirIndex.keys()))) if self.cirIndex else [],
                    'cirIndexVals': list(self.cirIndex.values()) if self.cirIndex else [],
                    'olIndex': self.olIndex,
                    'oliKeys': list(self.oliDict.keys()),
                    'timeKeys': list(self.timeDict.keys()),
                    'repKeys': list(self.repDict.keys()) if self.repDict else None,
                    'collisionActions': self.collisionActions,
                    'auxInfo': self.auxInfo,
                    'comments': self.comments,
                    'comment': self.comment}  # Don't pickle *Dict numpy data b/c it's inefficient
        # Compatability for unicode-literal filenames
        bOpen = not (hasattr(fileOrFilename, 'write'))
        if bOpen:
            if fileOrFilename.endswith(".gz"):
                import gzip as _gzip
                f = _gzip.open(fileOrFilename, "wb")
            else:
                f = open(fileOrFilename, "wb")
        else:
            f = fileOrFilename

        _pickle.dump(toPickle, f)
        for _, data in self.oliDict.items():
            _np.save(f, data)

        for _, data in self.timeDict.items():
            _np.save(f, data)

        if self.repDict:
            for _, data in self.repDict.items():
                _np.save(f, data)

        if bOpen: f.close()

    def load(self, fileOrFilename):
        """
        Load MultiDataSet from a file, clearing any data is contained previously.

        Parameters
        ----------
        fileOrFilename : file or string
            Either a filename or a file object.  In the former case, if the
            filename ends in ".gz", the file will be gzip uncompressed as it is read.
        """
        # Compatability for unicode-literal filenames
        bOpen = not (hasattr(fileOrFilename, 'write'))
        if bOpen:
            if fileOrFilename.endswith(".gz"):
                import gzip as _gzip
                f = _gzip.open(fileOrFilename, "rb")
            else:
                f = open(fileOrFilename, "rb")
        else:
            f = fileOrFilename

        state_dict = _pickle.load(f)

        def expand(x):
            """ Expand a comproessed circuit """
            assert isinstance(x, _cir.CompressedCircuit)
            return x.expand()
            #else: #to be backward compatible
            #  _warnings.warn("Deprecated dataset format.  Please re-save " +
            #                 "this dataset soon to avoid future incompatibility.")
            #  return _cir.Circuit(_cir.CompressedCircuit.expand_op_label_tuple(x))
        cirIndexKeys = [expand(cgs) for cgs in state_dict['cirIndexKeys']]

        #cirIndexKeys = [ cgs.expand() for cgs in state_dict['cirIndexKeys'] ]
        self.cirIndex = _OrderedDict(list(zip(cirIndexKeys, state_dict['cirIndexVals'])))
        self.olIndex = state_dict['olIndex']
        self.collisionActions = state_dict['collisionActions']
        self.auxInfo = state_dict.get('auxInfo', _DefaultDict(dict))  # backward compat
        self.comments = state_dict["comments"]
        self.comment = state_dict["comment"]

        self.oliDict = _OrderedDict()
        for key in state_dict['oliKeys']:
            self.oliDict[key] = _np.lib.format.read_array(f)  # np.load(f) doesn't play nice with gzip

        self.timeDict = _OrderedDict()
        for key in state_dict['timeKeys']:
            self.timeDict[key] = _np.lib.format.read_array(f)

        if state_dict['repKeys']:
            self.repDict = _OrderedDict()
            for key in state_dict['repKeys']:
                self.repDict[key] = _np.lib.format.read_array(f)
        else:
            self.repDict = None

        if bOpen: f.close()

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
