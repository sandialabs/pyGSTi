""" Defines the MultiDataSet class and supporting classes and functions """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import pickle as _pickle
import copy as _copy
from collections import OrderedDict as _OrderedDict

from ..tools import compattools as _compat

from .dataset import DataSet as _DataSet
from . import gatestring as _gs



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

        return datasetName, _DataSet(oliData, timeData, repData,
                                     gateStringIndices=self.multidataset.gsIndex,
                                     outcomeLabelIndices=self.multidataset.olIndex,
                                     collisionAction=self.multidataset.collisionActions[datasetName],
                                     bStatic=True)

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

        return _DataSet(oliData, timeData, repData,
                        gateStringIndices=self.multidataset.gsIndex,
                        outcomeLabelIndices=self.multidataset.olIndex,
                        collisionAction=self.multidataset.collisionActions[datasetName],
                        bStatic=True)

    next = __next__

    
class MultiDataSet(object):
    """
    The MultiDataSet class allows for the combined access and storage of
    several static DataSets that contain the same gate strings (in the same
    order) AND the same time-dependence structure (if applicable).

    It is designed to behave similarly to a dictionary of DataSets, so that
    a DataSet is obtained by:

    `dataset = multiDataset[datasetName]`

    where `datasetName` may be a string OR a tuple.
    """

    def __init__(self, oliDict=None, timeDict=None, repDict=None,
                 gateStringIndices=None,
                 outcomeLabels=None, outcomeLabelIndices=None,
                 fileToLoadFrom=None, collisionActions=None,
                 comment=None, comments=None):
        """
        Initialize a MultiDataSet.

        Parameters
        ----------
        oliDict : ordered dictionary, optional
          Keys specify dataset names.  Values are 1D numpy arrays which specify
          outcome label indices.  Each value is indexed by the values of
          `gateStringIndices`.

        timeDict : ordered dictionary, optional
          Same format as `oliDict` except stores arrays of floating-point time
          stamp data.

        repDict :  ordered dictionary, optional
          Same format as `oliDict` except stores arrays of integer repetition
          counts (can be `None` if there are no repetitions)

        gateStringIndices : ordered dictionary, optional
          An OrderedDict with keys equal to gate strings (tuples of gate labels) and values equal to
          integer indices associating a row/element of counts with the gate string.

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
            Specifies how duplicate gate sequences should be handled for the data
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


        Returns
        -------
        MultiDataSet
           a new multi data set object.
        """

        #Optionally load from a file
        if fileToLoadFrom is not None:
            assert(oliDict is None and timeDict is None and repDict is None and
                   gateStringIndices is None and outcomeLabels is None and outcomeLabelIndices is None)
            self.load(fileToLoadFrom)
            return

        # self.gsIndex  :  Ordered dictionary where keys = gate strings (tuples), values = integer indices into counts
        if gateStringIndices is not None:
            self.gsIndex = gateStringIndices
        else:
            self.gsIndex = None


                # self.olIndex  :  Ordered dictionary where
        #                  keys = outcome labels (strings or tuples),
        #                  values = integer indices mapping oliData (integers) onto
        #                           the outcome labels.
        if outcomeLabelIndices is not None:
            self.olIndex = outcomeLabelIndices
        elif outcomeLabels is not None:
            tup_outcomeLabels = [ ((ol,) if _compat.isstr(ol) else ol)
                                  for ol in outcomeLabels] #strings -> tuple outcome labels
            self.olIndex = _OrderedDict( [(ol,i) for (i,ol) in enumerate(tup_outcomeLabels) ] )
        else:
            self.olIndex = None

        #if self.gsIndex:  #Note: tests if not none and nonempty
        #    assert( min(self.gsIndex.values()) >= 0) # values are *slices* so can't test like this
        if self.olIndex:  #Note: tests if not none and nonempty
            assert( min(self.olIndex.values()) >= 0)

        # self.*Dict : dictionaries of 1D numpy arrays, each corresponding to a DataSet.        
        if oliDict is not None:
            self.oliDict = _OrderedDict()
            self.timeDict = _OrderedDict()
            self.repDict = _OrderedDict() if repDict else None

            assert(timeDict is not None), "Must specify `timeDict` also!"
        
            for name,outcomeInds in oliDict.items():
                assert(name in timeDict), "`timeDict` arg is missing %s key!" % name
                self.oliDict[name] = outcomeInds  #copy OrderedDict but share arrays
                self.timeDict[name] = timeDict[name]
                if repDict: self.repDict[name] = repDict[name]

            if collisionActions is None: collisionActions = {} #allow None to function as an empty dict
            self.collisionActions = _OrderedDict( [ (name,collisionActions.get(name,"aggregate"))
                                                    for name in self.oliDict.keys() ] )

            if comments is None: comments = {} #allow None to function as an empty dict
            self.comments = _OrderedDict( [ (name,comments.get(name,None))
                                            for name in self.oliDict.keys() ] )

            if self.olIndex:  #Note: tests if not none and nonempty
                maxOlIndex = max(self.olIndex.values())
                for outcomeInds in self.oliDict.values():
                    assert( _np.amax(outcomeInds) <= maxOlIndex )
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

        #data types - should stay in sync with DataSet
        self.oliType = _np.uint8
        self.timeType = _np.float64
        self.repType = _np.uint16



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
        return self.oliDict.__iter__() #iterator over dataset names

    def __len__(self):
        return len(self.oliDict)

    def __getitem__(self, datasetName):  #return a static DataSet
        repData = self.repDict[datasetName] if self.repDict else None
        return _DataSet(self.oliDict[datasetName],
                        self.timeDict[datasetName], repData, 
                        gateStringIndices=self.gsIndex,
                        outcomeLabelIndices=self.olIndex, bStatic=True,
                        collisionAction=self.collisionActions[datasetName])

    def __setitem__(self, datasetName, dataset):
        self.add_dataset(datasetName, dataset)

    def __contains__(self, datasetName):
        return datasetName in self.oliDict

    def keys(self):
        """ Returns a list of the keys (dataset names) of this MultiDataSet """
        return list(self.oliDict.keys())

    def has_key(self, datasetName):
        """ Test whether this MultiDataSet contains a given dataset name """
        return datasetName in self.oliDict

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
        for gstr, slc in self.gsIndex.items():
            concat_oli = _np.concatenate( [ self.oliDict[datasetName][slc]
                                            for datasetName in datasetNames ], axis=0 )
            concat_time = _np.concatenate( [ self.timeDict[datasetName][slc]
                                            for datasetName in datasetNames ], axis=0 )
            if self.repDict:
                concat_rep = _np.concatenate( [ self.repDict[datasetName][slc]
                                                for datasetName in datasetNames ], axis=0 )
            else:
                concat_rep = None

            # Merge same-timestamp, same-outcome data
            sortedInds = [i for i,el in sorted(enumerate(concat_time),key=lambda x:x[1])]
            sorted_oli = []; sorted_time = []; sorted_rep = []
            last_time =concat_time[sortedInds[0]]; cur_reps = {}
            for i in sortedInds:
                if concat_time[i] != last_time:
                    # dump cur_reps at last_time and reset
                    for oli,reps in cur_reps.items():
                        sorted_time.append( last_time )
                        sorted_oli.append( oli )
                        sorted_rep.append( reps )
                    last_time = concat_time[i]; cur_reps = {}
                    
                #add i-th element data to cur_reps
                reps = concat_rep[i] if (concat_rep is not None) else 1
                if concat_oli[i] in cur_reps:
                    cur_reps[ concat_oli[i] ] += reps
                else:
                    cur_reps[ concat_oli[i] ] = reps
                    
            # dump cur_reps at last_time
            for oli,reps in cur_reps.items():
                sorted_time.append( last_time )
                sorted_oli.append( oli )
                sorted_rep.append( reps )

            agg_oli.extend( sorted_oli )
            agg_time.extend( sorted_time )
            agg_rep.extend( sorted_rep )

            gstrSlices[gstr] = slice(slc_i, slc_i+len(sorted_oli))
            slc_i += len(sorted_oli)

        agg_oli = _np.array(agg_oli, self.oliType)
        agg_time = _np.array(agg_time, self.timeType)
        agg_rep = _np.array(agg_rep, self.repType)
        if _np.max(agg_rep) == 1: agg_rep = None #don't store trivial reps

        return _DataSet(agg_oli, agg_time, agg_rep,
                        gateStringIndices=gstrSlices,
                        outcomeLabelIndices=self.olIndex, bStatic=True)
                        #leave collisionAction as default "aggregate"

                        
    def add_dataset(self, datasetName, dataset):
        """
        Add a DataSet to this MultiDataSet.  The dataset
        must be static and conform with the gate strings and
        time-dependent structure passed upon construction or
        those inherited from the first dataset added.

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
        if self.olIndex is not None and dataset.olIndex != self.olIndex:
            raise ValueError("Cannot add dataset: outcome labels and/or their indices do not match")

        #if self.gsIndex:  #Note: tests if not none and nonempty
        #    maxIndex = max(self.gsIndex.values())
        #    assert( dataset.counts.shape[0] > maxIndex and dataset.counts.shape[1] == len(self.slIndex) )
        if len(self.oliDict) > 0:
            firstKey =  list(self.oliDict.keys())[0]
            dataLen = len(self.oliDict[firstKey])
            assert( len(dataset.oliData) == dataLen ), "Incompatible data sizes!"

            if (dataset.repData is not None) and self.repDict is None:
                # buildup trivial repDatas for all existing datasets
                self.repDict = _OrderedDict(
                    [ (nm,_np.ones(dataLen, self.repType)) for nm in self])
                
        elif dataset.repData is not None:
            self.repDict = _OrderedDict()
                
        self.oliDict[datasetName] = dataset.oliData
        self.timeDict[datasetName] = dataset.timeData
        if dataset.repData is not None:
            self.repDict[datasetName] = dataset.repData
            
        self.collisionActions[datasetName] = dataset.collisionAction
        self.comments[datasetName] = dataset.comment

        if self.gsIndex is None:
            self.gsIndex = dataset.gsIndex

        if self.olIndex is None:
            self.olIndex = dataset.olIndex


    #REMOVE FOR NOW - Maybe revive later
    #def add_dataset_counts(self, datasetName, datasetCounts,
    #                       collisionAction="aggregate"):
    #    """
    #    Directly add a full set of counts for a specified dataset.
    #
    #    Parameters
    #    ----------
    #    datasetName : string
    #        Counts are added for this data set.  This can be a new name, in
    #        which case this method adds a new data set to the MultiDataSet.
    #
    #    datasetCounts: numpy array
    #        A 2D array with rows = gate strings and cols = spam labels, to this
    #        MultiDataSet.  The shape of dataSetCounts is checked for compatibility.
    #
    #    collisionAction : {"aggregate", "keepseparate"}
    #        Specifies how duplicate gate sequences should be handled for this
    #        data set.  This is applicable only if and when the dataset is copied
    #        to a non-static one.  "aggregate" adds duplicate-sequence counts,
    #        whereas "keepseparate" tags duplicate-sequence data with by appending
    #        a final "#<number>" gate label to the duplicated gate sequence.
    #    """
    #
    #    if self.gsIndex:  #Note: tests if not none and nonempty
    #        maxIndex = max(self.gsIndex.values())
    #        assert( datasetCounts.shape[0] > maxIndex and datasetCounts.shape[1] == len(self.slIndex) )
    #    self.countsDict[datasetName] = datasetCounts
    #    self.collisionActions[datasetName] = collisionAction

    def __str__(self):
        s  = "MultiDataSet containing: %d datasets, each with %d strings\n" % (len(self), len(self.gsIndex) if self.gsIndex is not None else 0)
        s += " Dataset names = " + ", ".join(list(self.keys())) + "\n"
        s += " Outcome labels = " + ", ".join(map(str,self.olIndex.keys()) if self.olIndex is not None else [])
        if self.gsIndex is not None:
            s += "\nGate strings: \n" + "\n".join( map(str,list(self.gsIndex.keys())) )
        return s + "\n"


    def copy(self):
        """ Make a copy of this MultiDataSet """
        return MultiDataSet(self.oliDict, self.timeDict, self.repDict,
                            gateStringIndices=_copy.deepcopy(self.gsIndex) if (self.gsIndex is not None) else None,
                            outcomeLabelIndices=_copy.deepcopy(self.olIndex) if (self.olIndex is not None) else None,
                            collisionActions=self.collisionActions, comments=_copy.deepcopy(self.comments),
                            comment=(self.comment + " copy") if self.comment else None )


    def __getstate__(self):
        toPickle = { 'gsIndexKeys': list(map(_gs.CompressedGateString, list(self.gsIndex.keys()))) if self.gsIndex else [],
                     'gsIndexVals': list(self.gsIndex.values()) if self.gsIndex else [],
                     'olIndex': self.olIndex,
                     'oliDict': self.oliDict,
                     'timeDict': self.timeDict,
                     'repDict': self.repDict,
                     'collisionActions': self.collisionActions,
                     'comments': self.comments,
                     'comment': self.comment }
        return toPickle

    def __setstate__(self, state_dict):
        gsIndexKeys = [ cgs.expand() for cgs in state_dict['gsIndexKeys'] ]
        self.gsIndex = _OrderedDict( list(zip(gsIndexKeys, state_dict['gsIndexVals'])) )
        self.olIndex = state_dict['olIndex']
        self.oliDict = state_dict['oliDict']
        self.timeDict = state_dict['timeDict']
        self.repDict = state_dict['repDict']
        self.collisionActions = state_dict['collisionActions']
        self.comments = state_dict['comments']
        self.comment = state_dict['comment']

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
                     'olIndex': self.olIndex,
                     'oliKeys': list(self.oliDict.keys()),
                     'timeKeys': list(self.timeDict.keys()),
                     'repKeys': list(self.repDict.keys()) if self.repDict else None,
                     'collisionActions' : self.collisionActions,
                     'comments': self.comments,
                     'comment': self.comment }  #Don't pickle *Dict numpy data b/c it's inefficient
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
        for _,data in self.oliDict.items():
            _np.save(f, data)
            
        for _,data in self.timeDict.items():
            _np.save(f, data)
            
        if self.repDict:
            for _,data in self.repDict.items():
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
        def expand(x): 
            """ Expand a comproessed gate string """
            assert isinstance(x,_gs.CompressedGateString)
            return x.expand()
            #else: #to be backward compatible
            #  _warnings.warn("Deprecated dataset format.  Please re-save " +
            #                 "this dataset soon to avoid future incompatibility.")
            #  return _gs.GateString(_gs.CompressedGateString.expand_gate_label_tuple(x))
        gsIndexKeys = [ expand(cgs) for cgs in state_dict['gsIndexKeys'] ]

        #gsIndexKeys = [ cgs.expand() for cgs in state_dict['gsIndexKeys'] ]
        self.gsIndex = _OrderedDict( list(zip(gsIndexKeys, state_dict['gsIndexVals'])) )
        self.olIndex = state_dict['olIndex']
        self.collisionActions = state_dict['collisionActions']
        self.comments = state_dict["comments"]
        self.comment = state_dict["comment"]
        
        self.oliDict = _OrderedDict()
        for key in state_dict['oliKeys']:
            self.oliDict[key] = _np.lib.format.read_array(f) #np.load(f) doesn't play nice with gzip

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
