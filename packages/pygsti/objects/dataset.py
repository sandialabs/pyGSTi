from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines the DataSet class and supporting classes and functions """

import numpy as _np
import pickle as _pickle
import warnings as _warnings
from collections import OrderedDict as _OrderedDict

from ..tools import listtools as _lt

from . import gatestring as _gs


class DataSet_KeyValIterator(object):
    """ Iterator class for gate_string,DataSetRow pairs of a DataSet """
    def __init__(self, dataset):
        self.dataset = dataset
        self.gsIter = dataset.gsIndex.__iter__()
        self.countIter = dataset.counts.__iter__()

    def __iter__(self):
        return self

    def __next__(self): # Python 3: def __next__(self)
        return next(self.gsIter), DataSetRow(self.dataset, next(self.countIter))

    next = __next__


class DataSet_ValIterator(object):
    """ Iterator class for DataSetRow values of a DataSet """
    def __init__(self, dataset):
        self.dataset = dataset
        self.countIter = dataset.counts.__iter__()

    def __iter__(self):
        return self

    def __next__(self): # Python 3: def __next__(self)
        return DataSetRow(self.dataset, next(self.countIter))

    next = __next__

class DataSetRow(object):
    """
    Encapsulates DataSet count data for a single gate string.  Outwardly
      looks similar to a dictionary with spam labels as keys and counts as
      values.
    """
    def __init__(self, dataset, rowData):
        self.dataset = dataset
        self.rowData = rowData

    def __iter__(self):
        return self.dataset.slIndex.__iter__() #iterator over spam labels

    def __contains__(self, spamlabel):
        return spamlabel in self.dataset.slIndex

    def keys(self):
        """ Returns spam labels (strings) for which data counts are available."""
        return list(self.dataset.slIndex.keys())

    def has_key(self, spamlabel):
        """ Checks whether data counts for spamlabel (a string) is available."""
        return spamlabel in self.dataset.slIndex

    def iteritems(self):
        """ Iterates over (spam label, count) pairs. """
        return DataSetRow_KeyValIterator(self.dataset, self.rowData)

    def values(self):
        """ Returns spamlabel counts as a numpy array."""
        return self.rowData

    def total(self):
        """ Returns the total counts."""
        return float(sum(self.rowData))

    def fraction(self,spamlabel):
        """ Returns the fraction of total counts for spamlabel."""
        return self[spamlabel] / self.total()

    def __getitem__(self,spamlabel):
        return self.rowData[ self.dataset.slIndex[spamlabel] ]

    def __setitem__(self,spamlabel,count):
        self.rowData[ self.dataset.slIndex[spamlabel] ] = count

    def as_dict(self):
        """ Returns the (spamlabel,count) pairs as a dictionary."""
        return dict( list(zip(list(self.dataset.slIndex.keys()),self.rowData)) )

    def __str__(self):
        return str(self.as_dict())


class DataSetRow_KeyValIterator(object):
    """ Iterates over spamLabel,count pairs of a DataSetRow """
    def __init__(self, dataset, rowData):
        self.spamLabelIter = dataset.slIndex.__iter__()
        self.rowDataIter = rowData.__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.spamLabelIter), next(self.rowDataIter)

    next = __next__


class DataSet(object):
    """
    The DataSet class associates gate strings with counts for each spam label,
    and can be thought of as a table with gate strings labeling the rows and
    spam labels labeling the columns.  It is designed to behave similarly to a
    dictionary of dictionaries, so that counts are accessed by:
    count = dataset[gateString][spamLabel]
    """

    def __init__(self, counts=None, gateStrings=None, gateStringIndices=None,
                 spamLabels=None, spamLabelIndices=None,  bStatic=False, fileToLoadFrom=None,
                 collisionAction="aggregate"):
        """
        Initialize a DataSet.

        Parameters
        ----------
        counts : 2D numpy array (static case) or list of 1D numpy arrays (non-static case)
            Specifies spam label counts.  In static case, rows of counts correspond to gate
            strings and columns to spam labels.  In non-static case, different arrays
            correspond to gate strings and each array contains counts for the spam labels.

        gateStrings : list of (tuples or GateStrings)
            Each element is a tuple of gate labels or a GateString object.  Indices for these strings
            are assumed to ascend from 0.  These indices must correspond to rows/elements of counts (above).
            Only specify this argument OR gateStringIndices, not both.

        gateStringIndices : ordered dictionary
            An OrderedDict with keys equal to gate strings (tuples of gate labels) and values equal to
            integer indices associating a row/element of counts with the gate string.  Only
            specify this argument OR gateStrings, not both.

        spamLabels : list of strings
            Specifies the set of spam labels for the DataSet.  Indices for the spam labels
            are assumed to ascend from 0, starting with the first element of this list.  These
            indices will index columns of the counts array/list.  Only specify this argument
            OR spamLabelIndices, not both.

        spamLabelIndices : ordered dictionary
            An OrderedDict with keys equal to spam labels (strings) and value  equal to
            integer indices associating a spam label with a column of counts.  Only
            specify this argument OR spamLabels, not both.

        bStatic : bool
            When True, create a read-only, i.e. "static" DataSet which cannot be modified. In
              this case you must specify the counts, gate strings, and spam labels.
            When False, create a DataSet that can have counts added to it.  In this case,
              you only need to specify the spam labels.

        fileToLoadFrom : string or file object
            Specify this argument and no others to create a static DataSet by loading
            from a file (just like using the load(...) function).

        collisionAction : {"aggregate","keepseparate"}
            Specifies how duplicate gate sequences should be handled.  "aggregate"
            adds duplicate-sequence counts, whereas "keepseparate" tags duplicate-
            sequence data with by appending a final "#<number>" gate label to the
            duplicated gate sequence, which can then be accessed via the
            `get_row` and `set_row` functions.


        Returns
        -------
        DataSet
           a new data set object.

        """

        #Optionally load from a file
        if fileToLoadFrom is not None:
            assert(counts is None and gateStrings is None and gateStringIndices is None and spamLabels is None and spamLabelIndices is None)
            self.load(fileToLoadFrom)
            return

        # self.gsIndex  :  Ordered dictionary where keys = GateString objects, values = integer indices into counts
        if gateStringIndices is not None:
            self.gsIndex = gateStringIndices
        elif gateStrings is not None:
            dictData = [ (gs if isinstance(gs,_gs.GateString) else _gs.GateString(gs),i) \
                           for (i,gs) in enumerate(gateStrings) ] #convert to GateStrings if necessary
            self.gsIndex = _OrderedDict( dictData )

        elif not bStatic:
            self.gsIndex = _OrderedDict()
        else: raise ValueError("Must specify either gateStrings or gateStringIndices when creating a static DataSet")

        # self.slIndex  :  Ordered dictionary where keys = spam labels (strings), values = integer indices into counts
        if spamLabelIndices is not None:
            self.slIndex = spamLabelIndices
        elif spamLabels is not None:
            self.slIndex = _OrderedDict( [(sl,i) for (i,sl) in enumerate(spamLabels) ] )
        else: raise ValueError("Must specify either spamLabels or spamLabelIndices when creating a DataSet")

        if self.gsIndex:  assert( min(self.gsIndex.values()) >= 0)
        if self.slIndex:  assert( min(self.slIndex.values()) >= 0)

        # self.counts  :  when bStatic == True a single 2D numpy array.  Rows = gate strings, Cols = spam labels
        #                 when bStatic == False a list of 1D numpy arrays. Each array has length = num of spam labels
        if counts is not None:
            self.counts = counts

            if len(self.gsIndex) > 0:
                maxIndex = max(self.gsIndex.values())
                if bStatic:
                    assert( self.counts.shape[0] > maxIndex and self.counts.shape[1] == len(self.slIndex) )
                else:
                    assert( len(self.counts) > maxIndex and all( [ len(el) == len(self.slIndex) for el in self.counts ] ) )
            #else gsIndex has length 0 so there are no gate strings in this dataset (even though counts can contain data)

        elif not bStatic:
            assert( len(self.gsIndex) == 0)
            self.counts = []

        else:
            raise ValueError("data counts must be specified when creating a static DataSet")

        # self.bStatic
        self.bStatic = bStatic

        # collision action
        assert(collisionAction in ('aggregate','keepseparate'))
        self.collisionAction = collisionAction


    def __iter__(self):
        return self.gsIndex.__iter__() #iterator over gate strings

    def __len__(self):
        return len(self.gsIndex)

    def __getitem__(self, gatestring):
        return self.get_row(gatestring)

    def __setitem__(self, gatestring, countDict):
        return self.set_row(gatestring, countDict)

    def __contains__(self, gatestring):
        return gatestring in self.gsIndex

    def get_row(self, gatestring, occurance=0):
        """
        Get a row of data from this DataSet.  This gives the same
        functionality as [ ] indexing except you can specify the
        occurance number separately from the gate sequence.
        
        Parameters
        ----------
        gatestring : GateString or tuple
            The gate sequence to extract data for.

        occurance : int, optional
            0-based occurance index, specifying which occurance of
            a repeated gate sequence to extract data for.

        Returns
        -------
        DataSetRow
        """
        if occurance > 0: 
            gatestring = gatestring + _gs.GateString(("#%d" % occurance,))
        return DataSetRow(self, self.counts[ self.gsIndex[gatestring] ])


    def set_row(self, gatestring, countDict, occurance=0):
        """
        Set the counts for a row of this DataSet.  This gives the same
        functionality as [ ] indexing except you can specify the
        occurance number separately from the gate sequence.
        
        Parameters
        ----------
        gatestring : GateString or tuple
            The gate sequence to extract data for.

        countDict : dict
            The dictionary of counts (data).

        occurance : int, optional
            0-based occurance index, specifying which occurance of
            a repeated gate sequence to extract data for.
        """
        if occurance > 0: 
            gatestring = gatestring + _gs.GateString(("#%d" % occurance,))
        if gatestring in self:
            row = DataSetRow(self, self.counts[ self.gsIndex[gatestring] ])
            for spamLabel,cnt in countDict.items():
                row[spamLabel] = cnt
        else:
            self.add_count_dict(gatestring, countDict)
        

    def keys(self, stripOccuranceTags=False):
        """
        Returns the gate strings used as keys of this DataSet.

        Parameters
        ----------
        stripOccuranceTags : bool, optional
            Only applicable if `collisionAction` has been set to
            "keepseparate", when this argument is set to True
            any final "#<number>" elements of (would-be dupilcate)
            gate sequences are stripped so that the returned list
            may have *duplicate* entries.

        Returns
        -------
        list
            A list of GateString objects which index the data
            counts within this data set.            
        """
        if stripOccuranceTags and self.collisionAction == "keepseparate":
            return [ (gs[:-1] if (len(gs)>0 and gs[-1].startswith("#")) else gs) 
                     for gs in self.gsIndex.keys() ]
        else:
            return list(self.gsIndex.keys())
        

    def has_key(self, gatestring):
        """
        Test whether data set contains a given gate string.

        Parameters
        ----------
        gatestring : tuple or GateString
            A tuple of gate labels or a GateString instance
            which specifies the the gate string to check for.

        Returns
        -------
        bool
            whether gatestring was found.
        """
        return gatestring in self.gsIndex

    def iteritems(self):
        """
        Iterator over (gateString, countData) pairs,
          where gateString is a tuple of gate labels
          and countData is a DataSetRow instance,
          which behaves similarly to a dictionary
          with spam labels as keys and counts as
          values.
        """
        return DataSet_KeyValIterator(self)

    def itervalues(self):
        """
        Iterator over DataSetRow instances corresponding
          to the count data for each gate string.
        """
        return DataSet_ValIterator(self)

    def get_spam_labels(self):
        """
        Get the spam labels of this DataSet.

        Returns
        -------
        list of strings
          A list where each element is a spam label.
        """
        return list(self.slIndex.keys())

    def get_gate_labels(self):
        """
        Get a list of all the distinct gate labels used
          in the gate strings of this dataset.

        Returns
        -------
        list of strings
          A list where each element is a gate label.
        """
        gateLabels = [ ]
        for gateLabelString in self:
            for gateLabel in gateLabelString:
                if gateLabel not in gateLabels: gateLabels.append(gateLabel)
        return gateLabels

    def add_count_dict(self, gateString, countDict):
        """
        Add a single gate string's counts to this DataSet

        Parameters
        ----------
        gateString : tuple or GateString
          A tuple of gate labels specifying the gate string or a GateString object

        countDict : dict
          A dictionary with keys = spam labels and values = counts

        Returns
        -------
        None
        """
        if self.bStatic: raise ValueError("Cannot add data to a static DataSet object")
        countList = [ _np.nan ] * len(self.slIndex)
        for (spamLabel,count) in countDict.items():
            if spamLabel not in self.get_spam_labels():
                raise ValueError("Error adding data to Dataset: invalid spam label %s" % spamLabel)
            countList[ self.slIndex[spamLabel] ] = count
        if _np.nan in countList:
            raise ValueError("Error adding data to Dataset: not all spam labels were specified")
        self.add_count_list(gateString, countList)


    def add_count_list(self, gateString, countList):
        """
        Add a single gate string's counts to this DataSet.

        Parameters
        ----------
        gateString : tuple or GateString
          A tuple of gate labels specifying the gate string or a GateString object

        countsList : list
          A list/tuple of counts in the same order as the DataSet's spam labels

        Returns
        -------
        None
        """
        if self.bStatic: raise ValueError("Cannot add data to a static DataSet object")
        if not isinstance(gateString, _gs.GateString):
            gateString = _gs.GateString(gateString) #make sure we have a GateString

        if round(sum(countList)) == 0: return #don't add zero counts to a dataset

        assert( len(countList) == len(self.slIndex))
        countArray = _np.array(countList, 'd')

        if gateString in self.gsIndex:
            if self.collisionAction == "aggregate":
                gateStringIndx = self.gsIndex[gateString]
                self.counts[ gateStringIndx ] += countArray
            elif self.collisionAction == "keepseparate":
                #find next available gatestring:
                i=0; tagged_gateString = gateString
                while tagged_gateString in self.gsIndex:
                    i+=1; tagged_gateString = gateString + _gs.GateString(("#%d" % i,))
                #add data for a new (duplicate) gatestring
                gateStringIndx = len(self.counts) #index of to-be-added gate string
                self.counts.append( countArray )
                self.gsIndex[ tagged_gateString ] = gateStringIndx
                
        else:
            #add data for a new gatestring
            gateStringIndx = len(self.counts) #index of to-be-added gate string
            self.counts.append( countArray )
            self.gsIndex[ gateString ] = gateStringIndx

    def add_counts_1q(self, gateString, nPlus, nMinus):
        """
        Single-qubit version of addCountsDict, for convenience when
          the DataSet contains two spam labels, 'plus' and 'minus'.

        Parameters
        ----------
        gateString : tuple or GateString
          A tuple of gate labels specifying the gate string or a GateString object,
            e.g. ('I','x')

        nPlus : int
          The number of counts for the 'plus' spam label.

        nMinus : int
          The number of counts for the 'minus' spam label.

        Returns
        -------
        None
        """
        if self.bStatic: raise ValueError("Cannot add data to a static DataSet object")
        if not isinstance(gateString, _gs.GateString):
            gateString = _gs.GateString(gateString) #make sure we have a GateString

        if gateString in self.gsIndex and self.collisionAction == "aggregate":
            current_dsRow = self[ gateString ]
            oldP = current_dsRow['plus'] / float( current_dsRow['plus'] + current_dsRow['minus'] )
            newP = nPlus / float(nPlus + nMinus)
            if abs(oldP-newP) > 0.1:
                print('Warning! When attempting to combine data for the gate string '+ \
                    str(gateString) +', I encountered a discrepency of '+ str(abs(oldP-newP)*100.0) + \
                    ' percent! To resolve this issue, I am not going to ignore the latter data.')
                return

        self.add_count_dict(gateString, {'plus':nPlus, 'minus':nMinus} )

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
        if self.bStatic: raise ValueError("Cannot add data to a static DataSet object")
        assert(self.get_spam_labels() == otherDataSet.get_spam_labels())
        for (gateLabelString,dsRow) in otherDataSet.iteritems():
            self.add_count_list(gateLabelString, list(dsRow.values()) )

    def __str__(self):
        s = ""
        for gateString in self: # tuple-type gate label strings are keys
            s += "%s  :  %s\n" % (gateString, self[gateString])
            #s += "%d  :  %s\n" % (len(gateString), self[gateString]) #Uncomment to print string lengths instead of strings themselves
        return s + "\n"

    def truncate(self, listOfGateStringsToKeep, bThrowErrorIfStringIsMissing=True):
        """
        Create a truncated dataset comprised of a subset of the counts in this dataset.

        Parameters
        ----------
        listOfGateStringsToKeep : list of (tuples or GateStrings)
            A list of the gate strings for the new returned dataset.  If a
            gate string is given in this list that isn't in the original
            data set, bThrowErrorIfStringIsMissing determines the behavior.

        bThrowErrorIfStringIsMissing : bool, optional
            When true, a ValueError exception is raised when a strin)g
            if verbosity > 0:
            in listOfGateStringsToKeep is not in the data set.

        Returns
        -------
        DataSet
            The truncated data set.
        """
        if self.bStatic:
            gateStringIndices = []
            gateStrings = []
            for gs in listOfGateStringsToKeep:
                gateString = gs if isinstance(gs, _gs.GateString) else _gs.GateString(gs)

                if gateString not in self.gsIndex:
                    if bThrowErrorIfStringIsMissing:
                        raise ValueError("Gate string %s was not found in dataset begin truncated and bThrowErrorIfStringIsMissing == True" % str(gateString))
                    else: continue

                #only keep track of gate strings if they could be different from listOfGateStringsToKeep
                if not bThrowErrorIfStringIsMissing: gateStrings.append( gateString )
                gateStringIndices.append( self.gsIndex[gateString] )

            if bThrowErrorIfStringIsMissing: gateStrings = listOfGateStringsToKeep
            trunc_gsIndex = _OrderedDict( list(zip(gateStrings, gateStringIndices)) )
            trunc_dataset = DataSet(self.counts, gateStringIndices=trunc_gsIndex, spamLabelIndices=self.slIndex, bStatic=True) #don't copy counts, just reference
            #trunc_dataset = StaticDataSet(self.counts.take(gateStringIndices,axis=0), gateStrings=gateStrings, spamLabelIndices=self.slIndex)

        else:
            trunc_dataset = DataSet(spamLabels=self.get_spam_labels())
            for gateString in _lt.remove_duplicates(listOfGateStringsToKeep):
                if gateString in self.gsIndex:
                    gateStringIndx = self.gsIndex[gateString]
                    trunc_dataset.add_count_list( gateString, self.counts[ gateStringIndx ].copy() ) #Copy operation so trucated dataset can be modified
                elif bThrowErrorIfStringIsMissing:
                    raise ValueError("Gate string %s was not found in dataset begin truncated and bThrowErrorIfStringIsMissing == True" % str(gateString))

        return trunc_dataset

    def copy(self):
        """ Make a copy of this DataSet. """
        if self.bStatic:
            return self # doesn't need to be copied since data can't change
        else:
            copyOfMe = DataSet(spamLabels=self.get_spam_labels(),
                               collisionAction=self.collisionAction)
            copyOfMe.gsIndex = self.gsIndex.copy()
            copyOfMe.counts = [ el.copy() for el in self.counts ]
            return copyOfMe


    def copy_nonstatic(self, collisionAction=None):
        """ Make a non-static copy of this DataSet. """
        if self.bStatic:
            copyOfMe = DataSet(spamLabels=self.get_spam_labels(),
                               collisionAction=self.collisionAction)
            copyOfMe.gsIndex = self.gsIndex.copy()
            copyOfMe.counts = [ el.copy() for el in self.counts ]
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
        #  gsIndex and slIndex stay the same ; counts is transformed to a 2D numpy array
        if len(self.counts) > 0:
            newCounts = _np.concatenate( [el.reshape(1,-1) for el in self.counts], axis=0 )
        else:
            newCounts = _np.empty( (0,len(self.slIndex)), 'd')
        self.counts, self.bStatic = newCounts, True


    def __getstate__(self):
        toPickle = { 'gsIndexKeys': [_gs.CompressedGateString(key) for key in self.gsIndex.keys()], #list(map(_gs.CompressedGateString, list(self.gsIndex.keys()))),
                     'gsIndexVals': list(self.gsIndex.values()),
                     'slIndex': self.slIndex,
                     'bStatic': self.bStatic,
                     'counts': self.counts,
                     'collisionAction': self.collisionAction}
        return toPickle

    def __setstate__(self, state_dict):
        gsIndexKeys = [ cgs.expand() for cgs in state_dict['gsIndexKeys'] ]
        self.gsIndex = _OrderedDict( list(zip( gsIndexKeys, state_dict['gsIndexVals'])) )
        self.slIndex = state_dict['slIndex']
        self.counts = state_dict['counts']
        self.bStatic = state_dict['bStatic']
        self.collisionAction = state_dict.get('collisionAction',"aggregate") #backwards compatibility


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

        toPickle = { 'gsIndexKeys': [_gs.CompressedGateString(key) for key in (self.gsIndex.keys() if self.gsIndex else [])],  #list(map(_gs.CompressedGateString, list(self.gsIndex.keys()))) if self.gsIndex else [],
                     'gsIndexVals': list(self.gsIndex.values()) if self.gsIndex else [],
                     'slIndex': self.slIndex,
                     'bStatic': self.bStatic,
                     'collisionAction': self.collisionAction} 
                     #Don't pickle counts numpy data b/c it's inefficient
        if not self.bStatic: toPickle['nRows'] = len(self.counts)

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
        if self.bStatic:
            _np.save(f, self.counts)
        else:
            for rowArray in self.counts:
                _np.save(f, rowArray)
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
            if isinstance(x,_gs.CompressedGateString): return x.expand()
            else:
                _warnings.warn("Deprecated dataset format.  Please re-save " +
                               "this dataset soon to avoid future incompatibility.")
                return _gs.GateString(_gs.CompressedGateString.expand_gate_label_tuple(x))
        gsIndexKeys = [ expand(cgs) for cgs in state_dict['gsIndexKeys'] ]

        #gsIndexKeys = [ cgs.expand() for cgs in state_dict['gsIndexKeys'] ]
        self.gsIndex = _OrderedDict( list(zip( gsIndexKeys, state_dict['gsIndexVals'])) )
        self.slIndex = state_dict['slIndex']
        self.bStatic = state_dict['bStatic']
        self.collisionAction = state_dict.get("collisionAction","aggregate") #backward compatibility

        if self.bStatic:
            self.counts = _np.lib.format.read_array(f) #_np.load(f) doesn't play nice with gzip
        else:
            self.counts = []
            for i in range(state_dict['nRows']): #pylint: disable=unused-variable
                self.counts.append( _np.lib.format.read_array(f) ) #_np.load(f) doesn't play nice with gzip
        if bOpen: f.close()


#def upgrade_old_dataset(oldDataset):
#    """ Deprecated: Returns a DataSet based on an old-version dataset object """
#    if len(oldDataset.keys()) > 0:
#      spamLabels = oldDataset[ oldDataset.keys()[0] ].n.keys()
#      newDataset = DataSet(spamLabels=spamLabels)
#      for gs,datum in oldDataset.iteritems():
#        newDataset.add_count_dict( gs, datum.n )
#    else:
#      newDataset = DataSet(spamLabels=[]) #if it's an empty dataset, no spam labels
#    newDataset.done_adding_data()
#    return newDataset
#
#def upgrade_old_data_set_pickle(filename):
#    """ Deprecated: Upgrades an old-version dataset object pickle file."""
#    import sys as _sys
#    import OldDataSet as _OldDataSet
#    import cPickle as _pickle
#
#    currentDataSetModule = _sys.modules['DataSet']
#    _sys.modules['DataSet'] = _OldDataSet  #replace DataSet module with old one so unpickling can work
#    try:     oldDataset = _pickle.load( open(filename,"rb") )
#    finally: _sys.modules['DataSet'] = currentDataSetModule
#
#    newDataset = upgrade_old_dataset(oldDataset)
#
#    _pickle.dump( newDataset, open(filename + ".upd","wb") )
#    print "Successfully updated ==> %s" % (filename + ".upd")
