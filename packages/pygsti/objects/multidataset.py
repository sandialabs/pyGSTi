from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines the MultiDataSet class and supporting classes and functions """

import numpy as _np
import pickle as _pickle
from collections import OrderedDict as _OrderedDict

from .dataset import DataSet as _DataSet
from . import gatestring as _gs


class MultiDataSet_KeyValIterator(object):
    """ Iterator class for datasetName,DataSet pairs of a MultiDataSet """
    def __init__(self, multidataset):
        self.multidataset = multidataset
        self.countsDictIter = multidataset.countsDict.__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        datasetName = next(self.countsDictIter)
        return datasetName, _DataSet(self.multidataset.countsDict[datasetName], 
                                     gateStringIndices=self.multidataset.gsIndex,
                                     spamLabelIndices=self.multidataset.slIndex,
                                     collisionAction=self.multidataset.collisionActions[datasetName],
                                     bStatic=True)

    next = __next__


class MultiDataSet_ValIterator(object):
    """ Iterator class for DataSets of a MultiDataSet """
    def __init__(self, multidataset):
        self.multidataset = multidataset
        self.countsDictIter = multidataset.countsDict.__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        datasetName = next(self.countsDictIter)
        return _DataSet(self.multidataset.countsDict[datasetName],
                        gateStringIndices=self.multidataset.gsIndex,
                        spamLabelIndices=self.multidataset.slIndex, bStatic=True,
                        collisionAction=self.multidataset.collisionActions[datasetName],)

    next = __next__

class MultiDataSet(object):
    """
    The MultiDataSet class allows for the combined access and storage of
    several static DataSets that contain the same gate strings (in the same order).
    It is designed to behave similarly to a dictionary of DataSets, so that
    a DataSet is obtained by (Note that datasetName may be a tuple):

    dataset = multiDataset[datasetName]
    """

    def __init__(self, countsDict=None,
                 gateStrings=None, gateStringIndices=None,
                 spamLabels=None, spamLabelIndices=None,
                 fileToLoadFrom=None, collisionActions=None):
        """
        Initialize a MultiDataSet.

        Parameters
        ----------
        countsDict : ordered dictionary, optional
          Keys specify dataset names.  Values are 2D numpy arrays which specify counts. Rows of the arrays
          correspond to gate strings and columns to spam labels.

        gateStrings : list of (tuples or GateStrings), optional
          Each element is a tuple of gate labels or a GateString object.  Indices for these strings
          are assumed to ascend from 0.  These indices must correspond to rows/elements of counts (above).
          Only specify this argument OR gateStringIndices, not both.

        gateStringIndices : ordered dictionary, optional
          An OrderedDict with keys equal to gate strings (tuples of gate labels) and values equal to
          integer indices associating a row/element of counts with the gate string.  Only
          specify this argument OR gateStrings, not both.

        spamLabels : list of strings, optional
          Specifies the set of spam labels for the DataSet.  Indices for the spam labels
          are assumed to ascend from 0, starting with the first element of this list.  These
          indices will index columns of the counts array/list.  Only specify this argument
          OR spamLabelIndices, not both.

        spamLabelIndices : ordered dictionary, optional
          An OrderedDict with keys equal to spam labels (strings) and value  equal to
          integer indices associating a spam label with a column of counts.  Only
          specify this argument OR spamLabels, not both.

        fileToLoadFrom : string or file object, optional
          Specify this argument and no others to create a MultiDataSet by loading
          from a file (just like using the load(...) function).

        collisionActions : dictionary, optional
            Specifies how duplicate gate sequences should be handled for the data
            sets specified by `countsDict`.  Keys must match those of `countsDict`
            and values are "aggregate" or "keepseparate".  See documentation for
            `DataSet`.  If None, then "aggregate" is used for all sets by default.
        """

        #Optionally load from a file
        if fileToLoadFrom is not None:
            assert(countsDict is None and gateStrings is None and gateStringIndices is None and spamLabels is None and spamLabelIndices is None)
            self.load(fileToLoadFrom)
            return

        # self.gsIndex  :  Ordered dictionary where keys = gate strings (tuples), values = integer indices into counts
        if gateStringIndices is not None:
            self.gsIndex = gateStringIndices
        elif gateStrings is not None:
            dictData = [ (gs if isinstance(gs,_gs.GateString) else _gs.GateString(gs),i) \
                           for (i,gs) in enumerate(gateStrings) ] #convert to GateStrings if necessary
            self.gsIndex = _OrderedDict( dictData )
        else:
            self.gsIndex = None

        # self.slIndex  :  Ordered dictionary where keys = spam labels (strings), values = integer indices into counts
        if spamLabelIndices is not None:
            self.slIndex = spamLabelIndices
        elif spamLabels is not None:
            self.slIndex = _OrderedDict( [(sl,i) for (i,sl) in enumerate(spamLabels) ] )
        else:
            self.slIndex = None

        if self.gsIndex:  #Note: tests if not none and nonempty
            assert( min(self.gsIndex.values()) >= 0)
        if self.slIndex:  #Note: tests if not none and nonempty
            assert( min(self.slIndex.values()) >= 0)

        # self.countsDict : a dictionary of 2D numpy arrays, each corresponding to a DataSet.  Rows = gate strings, Cols = spam labels
        #                   ( keys = dataset names, values = 2D counts array of corresponding dataset )
        if countsDict is not None:
            self.countsDict = _OrderedDict( [ (name,counts) for name,counts in countsDict.items() ] ) #copy OrderedDict but share counts arrays
            if collisionActions is None: collisionActions = {} #allow None to function as an empty dict
            self.collisionActions = _OrderedDict( [ (name,collisionActions.get(name,"aggregate"))
                                                    for name in self.countsDict.keys() ] )

            if self.gsIndex:  #Note: tests if not none and nonempty
                #minIndex = min(self.gsIndex.values())
                maxIndex = max(self.gsIndex.values())
                for _,counts in self.countsDict.items():
                    assert( counts.shape[0] > maxIndex and counts.shape[1] == len(self.slIndex) )
        else:
            self.countsDict = _OrderedDict()
            self.collisionActions = _OrderedDict()

    def get_spam_labels(self):
        """
        Get the spam labels of this MultiDataSet.

        Returns
        -------
        list of strings
          A list where each element is a spam label.
          Returns None when the MultiDataSet is not
          yet initialized with any data or spam labels.
        """
        if self.slIndex is not None:
            return list(self.slIndex.keys())
        else: return None


    def __iter__(self):
        return self.countsDict.__iter__() #iterator over dataset names

    def __len__(self):
        return len(self.countsDict)

    def __getitem__(self, datasetName):  #return a static DataSet
        return _DataSet(self.countsDict[datasetName],
                        gateStringIndices=self.gsIndex,
                        spamLabelIndices=self.slIndex, bStatic=True,
                        collisionAction=self.collisionActions[datasetName])

    def __setitem__(self, datasetName, dataset):
        self.add_dataset(datasetName, dataset)

    def __contains__(self, datasetName):
        return datasetName in self.countsDict

    def keys(self):
        """ Returns a list of the keys (dataset names) of this MultiDataSet """
        return list(self.countsDict.keys())

    def has_key(self, datasetName):
        """ Test whether this MultiDataSet contains a given dataset name """
        return datasetName in self.countsDict

    def iteritems(self):
        """ Iterator over (dataset name, DataSet) pairs """
        return MultiDataSet_KeyValIterator(self)

    def itervalues(self):
        """ Iterator over DataSets corresponding to each dataset name """
        return MultiDataSet_ValIterator(self)

    def get_datasets_sum(self, *datasetNames):
        """
        Generate a new DataSet by combining the counts of multiple member Datasets.

        Parameters
        ----------
        datasetNames : one or more dataset names.

        Returns
        -------
        DataSet
            a single DataSet containing the summed counts of each of the datasets
            named by the parameters.
        """
        summedCounts = None
        if len(datasetNames) == 0: raise ValueError("Must specify at least one dataset name")
        for datasetName in datasetNames:
            if datasetName not in self:
                raise ValueError("No dataset with the name '%s' exists" % datasetName)

            if summedCounts is None:
                summedCounts = self.countsDict[datasetName].copy()
            else:
                summedCounts += self.countsDict[datasetName]

        return _DataSet(summedCounts, gateStringIndices=self.gsIndex,
                        spamLabelIndices=self.slIndex, bStatic=True)
                        #leave collisionAction as default "aggregate"

    def add_dataset(self, datasetName, dataset):
        """
        Add a DataSet to this MultiDataSet.  The dataset
        must be static and conform with the gate strings passed
        upon construction or those inherited from the first
        dataset added.

        Parameters
        ----------
        datasetName : string
            The name to give the added dataset (i.e. the key the new
            data set will be referenced by).

        dataset : DataSet
            The data set to add.
        """

        #first test if dataset is compatible
        if not dataset.bStatic:
            raise ValueError("Cannot add dataset: only static DataSets can be added to a MultiDataSet")
        if self.gsIndex is not None and dataset.gsIndex != self.gsIndex:
            raise ValueError("Cannot add dataset: gate strings and/or their indices do not match")
        if self.slIndex is not None and dataset.slIndex != self.slIndex:
            raise ValueError("Cannot add dataset: spam labels and/or their indices do not match")

        if self.gsIndex:  #Note: tests if not none and nonempty
            maxIndex = max(self.gsIndex.values())
            assert( dataset.counts.shape[0] > maxIndex and dataset.counts.shape[1] == len(self.slIndex) )

        self.countsDict[datasetName] = dataset.counts
        self.collisionActions[datasetName] = dataset.collisionAction

        if self.gsIndex is None:
            self.gsIndex = dataset.gsIndex
            if len(self.gsIndex) > 0:
                assert( min(self.gsIndex.values()) >= 0)

        if self.slIndex is None:
            self.slIndex = dataset.slIndex
            if len(self.slIndex) > 0:
                assert( min(self.slIndex.values()) >= 0)


    def add_dataset_counts(self, datasetName, datasetCounts,
                           collisionAction="aggregate"):
        """
        Directly add a full set of counts for a specified dataset.

        Parameters
        ----------
        datasetName : string
            Counts are added for this data set.  This can be a new name, in
            which case this method adds a new data set to the MultiDataSet.

        datasetCounts: numpy array
            A 2D array with rows = gate strings and cols = spam labels, to this
            MultiDataSet.  The shape of dataSetCounts is checked for compatibility.

        collisionAction : {"aggregate", "keepseparate"}
            Specifies how duplicate gate sequences should be handled for this
            data set.  This is applicable only if and when the dataset is copied
            to a non-static one.  "aggregate" adds duplicate-sequence counts,
            whereas "keepseparate" tags duplicate-sequence data with by appending
            a final "#<number>" gate label to the duplicated gate sequence.
        """

        if self.gsIndex:  #Note: tests if not none and nonempty
            maxIndex = max(self.gsIndex.values())
            assert( datasetCounts.shape[0] > maxIndex and datasetCounts.shape[1] == len(self.slIndex) )
        self.countsDict[datasetName] = datasetCounts
        self.collisionActions[datasetName] = collisionAction

    def __str__(self):
        s  = "MultiDataSet containing: %d datasets, each with %d strings\n" % (len(self), len(self.gsIndex) if self.gsIndex is not None else 0)
        s += " Dataset names = " + ", ".join(list(self.keys())) + "\n"
        s += " SPAM labels = " + ", ".join(list(self.slIndex.keys()) if self.slIndex is not None else [])
        if self.gsIndex is not None:
            s += "\nGate strings: \n" + "\n".join( map(str,list(self.gsIndex.keys())) )
        return s + "\n"


    def copy(self):
        """ Make a copy of this MultiDataSet """
        return MultiDataSet(self.countsDict, gateStringIndices=self.gsIndex, spamLabelIndices=self.slIndex,
                            collisionActions=self.collisionActions)


    def __getstate__(self):
        toPickle = { 'gsIndexKeys': list(map(_gs.CompressedGateString, list(self.gsIndex.keys()))) if self.gsIndex else [],
                     'gsIndexVals': list(self.gsIndex.values()) if self.gsIndex else [],
                     'slIndex': self.slIndex,
                     'countsDict': self.countsDict,
                     'collisionActions': self.collisionActions }
        return toPickle

    def __setstate__(self, state_dict):
        gsIndexKeys = [ cgs.expand() for cgs in state_dict['gsIndexKeys'] ]
        self.gsIndex = _OrderedDict( list(zip(gsIndexKeys, state_dict['gsIndexVals'])) )
        self.slIndex = state_dict['slIndex']
        self.countsDict = state_dict['countsDict']
        self.collisionActions = state_dict['collisionActions']

    def save(self, fileOrFilename):
        """
        Save this MultiDataSet to a file.

        Parameters
        ----------
        fileOrFilename : file or string
            Either a filename or a file object.  In the former case, if the
            filename ends in ".gz", the file will be gzip compressed.
        """

        toPickle = { 'gsIndexKeys': list(map(_gs.CompressedGateString, list(self.gsIndex.keys()))) if self.gsIndex else [],
                     'gsIndexVals': list(self.gsIndex.values()) if self.gsIndex else [],
                     'slIndex': self.slIndex,
                     'countsKeys': list(self.countsDict.keys()),
                     'collisionActions' : self.collisionActions }  #Don't pickle countsDict numpy data b/c it's inefficient
        # Compatability for unicode-literal filenames
        bOpen = not (hasattr(fileOrFilename, 'write'))
        if bOpen:
            if fileOrFilename.endswith(".gz"):
                import gzip as _gzip
                f = _gzip.open(fileOrFilename,"wb")
            else:
                f = open(fileOrFilename,"wb")
        else:
            f = fileOrFilename

        _pickle.dump(toPickle,f)
        for _,data in self.countsDict.items():
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
                f = _gzip.open(fileOrFilename,"rb")
            else:
                f = open(fileOrFilename,"rb")
        else:
            f = fileOrFilename

        state_dict = _pickle.load(f)
        def expand(x): #to be backward compatible
            assert isinstance(x,_gs.CompressedGateString)
            return x.expand()
            #else:
            #  _warnings.warn("Deprecated dataset format.  Please re-save " +
            #                 "this dataset soon to avoid future incompatibility.")
            #  return _gs.GateString(_gs.CompressedGateString.expand_gate_label_tuple(x))
        gsIndexKeys = [ expand(cgs) for cgs in state_dict['gsIndexKeys'] ]

        #gsIndexKeys = [ cgs.expand() for cgs in state_dict['gsIndexKeys'] ]
        self.gsIndex = _OrderedDict( list(zip(gsIndexKeys, state_dict['gsIndexVals'])) )
        self.slIndex = state_dict['slIndex']
        self.collisionActions = state_dict['collisionActions']
        self.countsDict = _OrderedDict()
        for key in state_dict['countsKeys']:
            self.countsDict[key] = _np.lib.format.read_array(f) #np.load(f) doesn't play nice with gzip
        if bOpen: f.close()
