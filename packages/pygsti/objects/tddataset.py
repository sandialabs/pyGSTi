from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation              
#    This Software is released under the GPL license detailed    
#    in the file "license.txt" in the top-level pyGSTi directory 
#*****************************************************************
""" Defines the DataSet class and supporting classes and functions """

import numpy as _np
import scipy.special as _sps
import scipy.fftpack as _fft
from scipy.integrate import quad as _quad
from scipy.interpolate import interp1d as _interp1d
import pickle as _pickle
import warnings as _warnings
from collections import OrderedDict as _OrderedDict

from ..tools import listtools as _lt

from . import gatestring as _gs
from . import dataset as _ds


class TDDataSet_KeyValIterator(object):
  """ Iterator class for gate_string,TDDataSetRow pairs of a TDDataSet """
  def __init__(self, dataset):
    self.dataset = dataset
    self.gsIter = dataset.gsIndex.__iter__()
    sliData = self.dataset.sliData
    timeData = self.dataset.timeData
    repData = self.dataset.repData
    
    if repData is None:
      self.tupIter = ( (sliData[ gsi ], timeData[ gsi ], None)
                       for gsi in self.dataset.gsIndex.values() )
    else:
      self.tupIter = ( (sliData[ gsi ], timeData[ gsi ], repData[ gsi ])
                       for gsi in self.dataset.gsIndex.values() )
    #Note: gsi above will be an index for a non-static dataset and
    #  a slice for a static dataset.

  def __iter__(self):
    return self

  def __next__(self): # Python 3: def __next__(self)
    return next(self.gsIter), TDDataSetRow(self.dataset, *(next(self.tupIter)) )

  next = __next__
  

class TDDataSet_ValIterator(object):
  """ Iterator class for TDDataSetRow values of a TDDataSet """
  def __init__(self, dataset):
    self.dataset = dataset
    sliData = self.dataset.sliData
    timeData = self.dataset.timeData
    repData = self.dataset.repData
    
    if repData is None:
      self.tupIter = ( (sliData[ gsi ], timeData[ gsi ], None)
                       for gsi in self.dataset.gsIndex.values() )
    else:
      self.tupIter = ( (sliData[ gsi ], timeData[ gsi ], repData[ gsi ])
                       for gsi in self.dataset.gsIndex.values() )
    #Note: gsi above will be an index for a non-static dataset and
    #  a slice for a static dataset.

  def __iter__(self):
    return self

  def __next__(self): # Python 3: def __next__(self)
    return TDDataSetRow(self.dataset, *(next(self.tupIter)) )

  next = __next__


class TDDataSetRow(object):
  """ 
  Encapsulates TDDataSet time series data for a single gate string.  Outwardly
    looks similar to a list with spam labels as the values.
  """
  def __init__(self, dataset, rowSliData, rowTimeData, rowRepData):
    self.dataset = dataset
    self.sli = rowSliData
    self.time = rowTimeData
    self.reps = rowRepData

  def get_sl(self):
    return [self.dataset.sl[i] for i in self.sli]

  def get_expanded_sl(self):
    if self.reps is not None:
      sl = []
      for sli, _, nreps in zip(self.sli,self.time,self.reps):
        sl.extend( [self.dataset.sl[sli]]*nreps )
      return sl
    else: return self.get_sl()

  def get_expanded_sli(self):
    if self.reps is not None:
      inds = []
      for sli, _, nreps in zip(self.sli,self.time,self.reps):
        inds.extend( [sli]*nreps )
      return _np.array(inds, dtype=self.dataset.sliType)
    else: return self.sli.copy()

  def get_expanded_times(self):
    if self.reps is not None:
      times = []
      for _, time, nreps in zip(self.sli,self.time,self.reps):
        times.extend( [time]*nreps )
      return _np.array(times, dtype=self.dataset.timeType)
    else: return self.time.copy()

  def __iter__(self):
    if self.reps is not None:      
      return ( (self.dataset.sl[i],t,n) for (i,t,n) in zip(self.sli,self.time,self.reps) )
    else:
      return ( (self.dataset.sl[i],t,1) for (i,t) in zip(self.sli,self.time) )

  def __getitem__(self,index):
    if self.reps is not None:
      return ( self.dataset.sl[ self.sli[index] ], self.time[index], self.reps[index] )
    else:
      return ( self.dataset.sl[ self.sli[index] ], self.time[index], 1 )

  def __setitem__(self,index,tup):
    assert(len(tup) in (2,3) ), "Must set to a (<spamLabel>,<time>[,<repetitions>]) value"
    self.sli[index] = self.dataset.slIndex[ tup[0] ]
    self.time[index] = tup[1]
      
    if self.reps is not None:
      self.reps[index] = tup[2] if len(tup) == 3 else 1
    else:
      assert(len(tup) == 2 or tup[2] == 1),"Repetitions must == 1 (not tracking reps)"

  def get_counts(self):
    cntDict = _OrderedDict()
    if self.reps is None:
      for sl,i in self.dataset.slIndex.items():
        cntDict[sl] = float(_np.count_nonzero( _np.equal(self.sli,i) ))
    else:
      for sl,i in self.dataset.slIndex.items():        
        cntDict[sl] = float( sum(self.reps[
          _np.nonzero(_np.equal(self.sli,i))[0]]))
    return cntDict
    
  def total(self):
    """ Returns the total number of counts."""
    if self.reps is None:
      return float(len(self.sli))
    else:
      return sum(self.reps)

  def fraction(self,spamlabel):
    """ Returns the fraction of total counts for spamlabel."""
    d = self.get_counts()
    total = sum(d.values())
    return d[spamlabel]/total

  def __str__(self):
    s  = "Spam Label Indices = " + str(self.sli) + "\n"
    s += "Time stamps = " + str(self.time) + "\n"
    if self.reps is not None:
      s += "Repetitions = " + str(self.reps) + "\n"
    else:
      s += "( no repetitions )\n"
    return s

  def __len__(self):
    return len(self.sli)

  


class TDDataSet(object):
  """ 
  The TDDataSet ("Time-Dependent DataSet") class associates gate strings with
  spam-label time series, one times series for each gate string.  It can be
  thought of as a table of spam labels (or their indicies/abbreviations) with
  gate strings labeling the rows and time step indices labeling the columns.
  It is designed to behave similarly to a dictionary of lists, so that the
  spam label for the i-th outcome of the sequence `gateString` is accessed by:
  spamLabel = dataset[gateString][i]
  """

  def __init__(self, sliData=None, timeData=None, repData=None, gateStrings=None, gateStringIndices=None, 
               spamLabels=None, spamLabelIndices=None,  bStatic=False, fileToLoadFrom=None):
    """ 
    Initialize a TDDataSet ("Time-dependent DataSet").
      
    Parameters
    ----------
    timeseries : 2D numpy array (static case) or list of 1D numpy arrays (non-static case)
        Specifies spam label indices.  In static case, rows of indices correspond to gate
        strings and columns to time steps.  In non-static case, different arrays
        correspond to gate strings and each array contains index data for all the time steps.

    gateStrings : list of (tuples or GateStrings)
        Each element is a tuple of gate labels or a GateString object.  Indices for these strings
        are assumed to ascend from 0.  These indices must correspond to the time series of spam-label
        indices (above).   Only specify this argument OR gateStringIndices, not both.

    gateStringIndices : ordered dictionary
        An OrderedDict with keys equal to gate strings (tuples of gate labels) and values equal to
        integer indices associating a row/element of counts with the gate string.  Only
        specify this argument OR gateStrings, not both.

    spamLabels : list of strings
        Specifies the set of spam labels for the DataSet.  Indices for the spam labels
        are assumed to ascend from 0, starting with the first element of this list.  These
        indices will associate each elememtn of `timeseries` with a spam label.  Only
        specify this argument OR spamLabelIndices, not both.

    spamLabelIndices : ordered dictionary
        An OrderedDict with keys equal to spam labels (strings) and value equal to 
        integer indices associating a spam label with given index.  Only 
        specify this argument OR spamLabels, not both.

    bStatic : bool
        When True, create a read-only, i.e. "static" TDDataSet which cannot be modified. In
          this case you must specify the timeseries data, gate strings, and spam labels.
        When False, create a DataSet that can have time series data added to it.  In this case,
          you only need to specify the spam labels.
      
    fileToLoadFrom : string or file object
        Specify this argument and no others to create a static TDDataSet by loading
        from a file (just like using the load(...) function).

    Returns
    -------
    TDDataSet
       a new data set object.

    """

    #Optionally load from a file
    if fileToLoadFrom is not None:
      assert(sliData is None and timeData is None and repData is None \
             and gateStrings is None and gateStringIndices is None \
             and spamLabels is None and spamLabelIndices is None)
      self.load(fileToLoadFrom)
      return

    # self.gsIndex  :  Ordered dictionary where keys = GateString objects,
    #   values = slices into sli, time, & rep arrays (static case) or
    #            integer list indices (non-static case)
    if gateStringIndices is not None:
      self.gsIndex = gateStringIndices
    elif not bStatic:
      if gateStrings is not None:
        dictData = [ (gs if isinstance(gs,_gs.GateString) else _gs.GateString(gs),i) \
                     for (i,gs) in enumerate(gateStrings) ] #convert to GateStrings if necessary
        self.gsIndex = _OrderedDict( dictData )
      else:
        self.gsIndex = _OrderedDict()
    else: raise ValueError("Must specify gateStringIndices when creating a static TDDataSet")

    # self.slIndex  :  Ordered dictionary where keys = spam labels (strings), values = integer indices mapping 
    #                    sliData (integers) onto the spam labels.
    if spamLabelIndices is not None:
      self.slIndex = spamLabelIndices
    elif spamLabels is not None:
      self.slIndex = _OrderedDict( [(sl,i) for (i,sl) in enumerate(spamLabels) ] )
    else: raise ValueError("Must specify either spamLabels or spamLabelIndices when creating a TDDataSet")

    # self.sl :  Ordered dictionary where keys = integer indices, values = spam labels (strings)
    self.sl = _OrderedDict( [(i,sl) for (sl,i) in self.slIndex.items()] )

    # sanity checks that indices are >= 0
    if not bStatic: #otherwise values() below are slices
      if self.gsIndex:  assert( min(self.gsIndex.values()) >= 0)
      if self.slIndex:  assert( min(self.slIndex.values()) >= 0)

    # self.sliData : when bStatic == True a 1D numpy array containing concatenated spam label indices.
    #                when bStatic == False a list of 1D numpy arrays, one array per gate sequence.

    # self.timeData : when bStatic == True a 1D numpy array containing concatenated time stamps.
    #                 when bStatic == False a list of 1D numpy arrays, one array per gate sequence.

    # self.repData : when bStatic == True a 1D numpy array containing concatenated repetition counts.
    #                when bStatic == False a list of 1D numpy arrays, one array per gate sequence.
    #   (can be None, in which case no repetitions are assumed)

    if sliData is not None:

      # check that sizes/lengths all match
      assert(len(timeData) == len(sliData)), "timeData must be same size as sliData"
      if repData is not None:
        assert(len(repData) == len(sliData)), "repData must be same size as sliData"

      self.sliData = sliData
      self.timeData = timeData
      self.repData = repData

      if len(self.gsIndex) > 0:
        maxSLIndex = max(self.slIndex.values())
        if bStatic:
          assert( _np.amax(self.sliData) <= maxSLIndex )
          # self.sliData.shape[0] > maxIndex doesn't make sense since gsIndex holds slices
        else:
          maxIndex = max(self.gsIndex.values())
          assert( len(self.sliData) > maxIndex )
          if len(self.sliData) > 0:
              assert( all( [ max(sliSeries) <= maxSLIndex for sliSeries in self.sliData ] ) )
      #else gsIndex has length 0 so there are no gate strings in this dataset (even though sliData can contain data)

    elif not bStatic:
      assert( timeData is None ), "timeData must be None when sliData is"
      assert( repData is None ), "repData must be None when sliData is"
      assert( len(self.gsIndex) == 0)
      self.sliData = []
      self.timeData = []
      self.repData = None

    else:
      raise ValueError("Series data must be specified when creating a static TDDataSet")
    
    # self.bStatic
    self.bStatic = bStatic

    # self.ffdata : fourier filtering data
    self.ffdata = {}

    #data types
    self.sliType = _np.uint8
    self.timeType = _np.float64
    self.repType = _np.uint16


  def __iter__(self):
    return self.gsIndex.__iter__() #iterator over gate strings

  def __len__(self):
    return len(self.gsIndex)

  def __getitem__(self, gatestring):
    #Note: gsIndex value is either an int (non-static) or a slice (static)
    repData = self.repData[ self.gsIndex[gatestring] ] \
              if (self.repData is not None) else None
    return TDDataSetRow(self, self.sliData[ self.gsIndex[gatestring] ],
                        self.timeData[ self.gsIndex[gatestring] ], repData)
  
  def __setitem__(self, gatestring, sli_stamp_rep_tuple):
    assert(len(sli_stamp_rep_tuple) >= 2), \
      "Must minimally set with (spam-label-list, time-stamp-list)"
    self.add_series_data(gatestring, *sli_stamp_rep_tuple)

  def __contains__(self, gatestring):
    return gatestring in self.gsIndex

  def keys(self):
    """ 
    Returns the gate strings of this TDDataSet as tuples 
        of gate labels (not GateString objects).
    """
    return self.gsIndex.keys()

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
    Iterator over (gateString, timeSeries) pairs,
      where gateString is a tuple of gate labels
      and timeSeries is a TDDataSetRow instance, 
      which behaves similarly to a list of
      spam labels whose index corresponds to 
      the time step.
    """
    return TDDataSet_KeyValIterator(self)
    
  def itervalues(self):
    """ 
    Iterator over TDDataSetRow instances corresponding
      to the time series data for each gate string.
    """
    return TDDataSet_ValIterator(self)

  def get_spam_labels(self):
    """ 
    Get the spam labels of this DataSet.

    Returns
    -------
    list of strings
      A list where each element is a spam label.
    """
    return self.slIndex.keys()

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


  def add_series_data(self, gateString, spamLabelList, timeStampList, repCountList=None,
                      overwriteExisting=True):
    """ 
    Add a single gate string's counts to this DataSet

    Parameters
    ----------
    gateString : tuple or GateString
      A tuple of gate labels specifying the gate string or a GateString object

    spamLabelList : list
      A list of spam labels (strings).  An element's index links it to a
      particular time step (i.e. the i-th element of the list specifies
      the outcome of the i-th measurement in the series).


    Returns
    -------
    None
    """
    if self.bStatic: raise ValueError("Cannot add data to a static TDDataSet object")
    if not isinstance(gateString, _gs.GateString):
      gateString = _gs.GateString(gateString) #make sure we have a GateString
    
    sliArray = _np.array([ self.slIndex[sl] for sl in spamLabelList ] , self.sliType)
    timeArray = _np.array(timeStampList, self.timeType)
    assert(sliArray.shape == timeArray.shape), \
      "Spam-label and time stamp lists must have the same length!"
    
    if repCountList is None:
      if self.repData is None: repArray = None
      else: repArray = _np.ones(len(sliArray), self.repType)
    else:
      if self.repData is None:
        #rep count data was given, but we're not currently holding repdata,
        # so we need to build this up for all existings sequences:
        self.repData = []
        for sliAr in self.sliData:
          self.repData.append( _np.ones(len(sliAr), self.repType) )
      repArray = _np.array(repCountList, self.repType)

    if gateString in self.gsIndex:
      gateStringIndx = self.gsIndex[gateString]
      if overwriteExisting:
        self.sliData[ gateStringIndx ] = sliArray #OVERWRITE existing time series
        self.timeData[ gateStringIndx ] = timeArray #OVERWRITE existing time series
        if repArray is not None: self.repData[ gateStringIndx ] = repArray
      else:
        self.sliData[ gateStringIndx ] = _np.concatenate((self.sliData[gateStringIndx],sliArray))
        self.timeData[ gateStringIndx ] = _np.concatenate((self.timeData[gateStringIndx],timeArray))
        if repArray is not None:
          self.repData[ gateStringIndx ] = _np.concatenate((self.repData[gateStringIndx],repArray))
  
    else:
      #add data for a new gatestring
      assert( len(self.sliData) == len(self.timeData) ), "SLI and TIME data are out of sync!!"
      gateStringIndx = len(self.sliData) #index of to-be-added gate string
      self.sliData.append( sliArray )
      self.timeData.append( timeArray )
      if repArray is not None: self.repData.append( repArray )
      self.gsIndex[ gateString ] = gateStringIndx

      
  def add_series_from_dataset(self, otherTDDataSet):
    """ 
    Append another TDDataSet's series data to this TDDataSet

    Parameters
    ----------
    otherDataSet : TDDataSet
        The dataset to take time series data from.

    Returns
    -------
    None
    """
    if self.bStatic: raise ValueError("Cannot add data to a static TDDataSet object")
    for gateString,dsRow in otherDataSet.iteritems():
      self.add_series_data(gateString, dsRow.get_sl(), dsRow.time, dsRow.reps, False)

      
  def __str__(self):
    s = ""
    for gateString in self: # tuple-type gate label strings are keys
      s += "%s  :  %s\n" % (gateString, self[gateString])
      #s += "%d  :  %s\n" % (len(gateString), self[gateString]) #Uncomment to print string lengths instead of strings themselves
    return s + "\n"

  def truncate(self, listOfGateStringsToKeep, bThrowErrorIfStringIsMissing=True):
    """ 
    Create a truncated dataset comprised of a subset of the gate strings
    in this dataset.

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
    TDDataSet
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
      trunc_gsIndex = _OrderedDict( zip(gateStrings, gateStringIndices) )
      trunc_dataset = TDDataSet(self.sliData, self.timeData, self.repData,
                                gateStringIndices=trunc_gsIndex,
                                spamLabelIndices=self.slIndex, bStatic=True) #don't copy counts, just reference
    else:
      trunc_dataset = TDDataSet(spamLabels=self.get_spam_labels())
      for gateString in _lt.remove_duplicates(listOfGateStringsToKeep):
        if gateString in self.gsIndex:
          gateStringIndx = self.gsIndex[gateString]
          repData = self.repData[ gateStringIndx ].copy() if (self.repData is not None) else None
          trunc_dataset.add_series_data( gateString, [ self.sl[i] for i in self.sliData[ gateStringIndx ] ],
                                         self.timeData[ gateStringIndx ].copy(), repData) #Copy operation so truncated dataset can be modified
        elif bThrowErrorIfStringIsMissing:
          raise ValueError("Gate string %s was not found in dataset begin truncated and bThrowErrorIfStringIsMissing == True" % str(gateString))

    return trunc_dataset

  def copy(self):
    """ Make a copy of this TDDataSet. """
    if self.bStatic: 
      return self # doesn't need to be copied since data can't change
    else:
      copyOfMe = TDDataSet(spamLabels=self.get_spam_labels())
      copyOfMe.gsIndex = self.gsIndex.copy()
      copyOfMe.sliData = [ el.copy() for el in self.sliData ]
      copyOfMe.timeData = [ el.copy() for el in self.timeData ]
      if self.repData is not None:
        copyOfMe.repData = [ el.copy() for el in self.repData ]
      else: copyOfMe.repData = None
      
      copyOfMe.sliType  =self.sliType
      copyOfMe.timeType = self.timeType
      copyOfMe.repType  = self.repType
      return copyOfMe


  def copy_nonstatic(self):
    """ Make a non-static copy of this TDDataSet. """
    if self.bStatic: 
      copyOfMe = TDDataSet(spamLabels=self.get_spam_labels())
      copyOfMe.gsIndex = _OrderedDict([ (gstr,i) for i,gstr in enumerate(self.gsIndex.keys()) ])
      copyOfMe.sliData = [] 
      copyOfMe.timeData = []
      copyOfMe.repData = None if (self.repData is None) else []
      for slc in self.gsIndex.values():
        copyOfMe.sliData.append( self.sliData[slc].copy() )
        copyOfMe.timeData.append( self.timeData[slc].copy() )
        if self.repData is not None:
          copyOfMe.repData.append( self.repData[slc].copy() )

      copyOfMe.sliType  =self.sliType
      copyOfMe.timeType = self.timeType
      copyOfMe.repType  = self.repType
      return copyOfMe
    else:
      return self.copy()


  def done_adding_data(self):
    """ 
    Promotes a non-static TDDataSet to a static (read-only) TDDataSet.  This
     method should be called after all data has been added.
    """     
    if self.bStatic: return
    #Convert normal dataset to static mode.
    #  slIndex stays the same
    #  gsIndex changes to hold slices into 1D arrays
    #  sliData, timeData, & repData change from being lists of arrays to
    #    single 1D arrays.
    
    if len(self.sliData) > 0:
      new_gsIndex = _OrderedDict()
      curIndx = 0
      to_concat_sli = []
      to_concat_time = []
      to_concat_rep = []
      for gatestring, indx in self.gsIndex.items():
        seriesLen = len(self.sliData[indx])

        to_concat_sli.append( self.sliData[indx] )   #just build up lists of
        to_concat_time.append( self.timeData[indx] ) # reference, not copies
        assert(seriesLen == len(self.timeData[indx])), "TIME & SLI out of sync!"
        
        if self.repData is not None:
          to_concat_rep.append( self.repData[indx] )
          assert(seriesLen == len(self.repData[indx])), "REP & SLI out of sync!"
          
        new_gsIndex[gatestring] = slice(curIndx, curIndx+seriesLen)
        curIndx += seriesLen

      self.gsIndex = new_gsIndex
      self.sliData = _np.concatenate( to_concat_sli )
      self.timeData = _np.concatenate( to_concat_time )
      if self.repData is not None:
        self.repData = _np.concatenate( to_concat_rep )
        
    else:
      #leave gsIndex alone (should be empty anyway?)
      self.sliData = _np.empty( (0,), self.sliType)
      self.timeData = _np.empty( (0,), self.timeType)
      if self.repData is not None:
        self.repData = _np.empty( (0,), self.repType)
      
    self.bStatic = True

  def compute_fourier_filtering(self, n_sigma=3, slope_compensation=False, resample_factor=5, verbosity=0):
    """ Compute and store significant fourier coefficients -- EXPERIMENTAL TODO DOCSTRING"""
    if not self.bStatic: 
      raise ValueError("TDDataSet object must be *static* to compute fourier filters")

    #Computes the order statistic

    # Compute the log factorial function for numbers up to 2**15
    # we could go higher, but this is already a lot of data
    log_factorial = _np.append([0.0],_np.cumsum(list(map(_np.log, _np.arange(2**15)+1))))
    
    def order(x,sigma, n,r):
        # Compute the distribution of the r^th smallest of n draws of a 
        # mean-zero normal distribution with variance sigma^2
        
        a = (1./2 - n) * _np.log(2) 
        b = -x**2 / (2 * sigma**2)
        c = (n-r)*_np.log(1-_sps.erf(x/(_np.sqrt(2)*sigma)))
        d = (r-1)*_np.log(1+_sps.erf(x/(_np.sqrt(2)*sigma)))
    
        e = log_factorial[n]
        f = log_factorial[n-r]
        g = log_factorial[r-1]
    
        h = _np.log(_np.pi*sigma**2)/2.
    
        return _np.exp(a + b + c + d + e - f - g - h)

    def order_mean_var(sigma, n, r, mean_in = None):
        # Compute the mean variance of the order statistic
        order_mean = _quad(lambda x: x * order(x,sigma, n, r), -6*sigma, 6*sigma)[0]
        return order_mean, _quad(lambda x: x**2 * order(x, sigma, n, r), -6*sigma, 6*sigma)[0] - order_mean**2

    def fourier_filter(tdata, ydata, n_sigma=3., keep=None, return_aux=False, verbose=False, 
                       truncate=True, slope_compensation=False, resampleFactor=5):
        """
        Process a list of binary data to produce an estimate of the time-dependent 
        probability underlying the data.  Fourier components are kept if their 
        magnitude is more than sigma from the expected value.
        """

        #first turn the data into an equally spaced sequence of data values
        try:
          f = _interp1d(tdata, ydata, kind='cubic', assume_sorted=False)
        except _np.linalg.linalg.LinAlgError: #cubic can fail for few data points
          f = _interp1d(tdata, ydata, kind='linear', assume_sorted=False)
        ts = _np.linspace(tdata[0],tdata[-1], len(tdata)*resampleFactor)
        #delta = _np.mean([ tdata[i]-tdata[i-1] for i in range(1,len(tdata)) ])
        data  = _np.array([f(t) for t in ts],'d')

        # Estimate the stationary probability from the data and compute the expected
        # variance for the fourier coefficients
        p_mean = _np.mean(data)
        p_sigma = _np.sqrt(len(data) * (p_mean - p_mean**2) / 2.)

        min_p_sigma = 1e-6 # hack for now to keep p_sigma from equaling zero.
        p_sigma = max(p_sigma, min_p_sigma)
        
        # Compensate for slope
        if slope_compensation:
            slope = 2 * (_np.mean(data[int(len(data)/2.):]) - _np.mean(data[0:int(len(data)/2.)]))
            data = data - slope * _np.linspace(-.5,.5,len(data))
            
        # DFT the data
        fft_data = _fft.rfft(data)
        filtered_fft = _np.array(fft_data.copy())
        
        if keep is None:
            # Determine the threshold by computing the distribution of the extremal 
            # order statistic.  The threshold is taken to be the mean
            order_mean, order_var = order_mean_var(p_sigma, len(data), 1)
            order_sig = _np.sqrt(order_var)
            threshold = abs(order_mean) + n_sigma * order_sig
        else:
            # Determine the threshold to be consistent with keeping "keep" fourier modes
            threshold = list(sorted(abs(fft_data), reverse=True))[keep]
    
        n_kept = sum(abs(filtered_fft) >= threshold)
        if verbose:
            print("Threshold:" + str(threshold))
            print("Keeping {} Fourier modes".format(n_kept))
    
        filtered_fft[ abs(filtered_fft) < threshold ] = 0
        filtered_data = _np.array(_fft.irfft(filtered_fft))
        filtered_tdata = ts #from interpolation
    
        if slope_compensation:
            filtered_data += slope * _np.linspace(-.5,.5,len(data))
       
        if truncate:
            filtered_data = _np.maximum(_np.minimum(filtered_data,1-1.e-6),1.e-6)
        
        if return_aux:
            return filtered_data, filtered_tdata, fft_data, threshold, n_kept
        else:
            return filtered_data, filtered_tdata

    # For each spam label, get 1's and 0's string to be "fourier filtered" 
    # into a time-dependent probability function.
    self.ffdata = {}
    for gateStr in self.gsIndex.keys():
      dsRow = self[gateStr]
      ff_data_dict = {}
      for spamLabel,spamIndx in self.slIndex.items():
        ydata = _np.where(dsRow.get_expanded_sli() == spamIndx, 1, 0 )
        tdata = dsRow.get_expanded_times()

        # Filter modes less than n_sigma sigma from expected maximum.
        filtered, filtered_ts, fft_data, threshold, n_kept = fourier_filter(
          tdata, ydata, n_sigma=n_sigma, return_aux=True,
          slope_compensation=slope_compensation, resampleFactor=resample_factor,
          verbose=(verbosity > 0))
        ff_data_dict[spamLabel] = (filtered, filtered_ts, fft_data, threshold, n_kept)

      self.ffdata[gateStr] = ff_data_dict


  def create_dataset_at_time(self, timeval):
    """ 
    Creates a DataSet at time `timeval` via fourier-filtering this data.

    Parameters
    ----------
    timeval : float or int
        The time-stamp value at which to create a DataSet.

    Returns
    -------
    DataSet
    """
    ds = _ds.DataSet(spamLabelIndices=self.slIndex)
    for gateStr,i in self.gsIndex.items():
      ff_data_dict = self.ffdata[gateStr]
      nTimeSteps = self[gateStr].total()
      avgKept = _np.average( [ff_data_dict[sl][4] for sl in self.slIndex ] )
      N = int(round(float(nTimeSteps)/avgKept)) # ~ clicks per fourier mode

      count_dict = {}
      for sl in self.slIndex:
        (filtered, ts, fft_data, threshold, n_kept) = ff_data_dict[sl]
        try:
          f = _interp1d(ts, filtered, kind='cubic')
        except _np.linalg.linalg.LinAlgError: #cubic can fail for few data points
          f = _interp1d(ts, filtered, kind='linear')
        p = f(timeval)
        count_dict[sl] = N*p

      ds.add_count_dict(gateStr, count_dict)
    ds.done_adding_data()
    return ds

  
  def create_dataset_from_time_range(self, startTime, endTime):
    """ 
    Creates a DataSet by aggregating the counts within the
    [`startTime`,`endTime`) interval.

    Parameters
    ----------
    startTime, endTime : float or int
        The time-stamps to use for the beginning (inclusive) and end
        (exclusive) of the time interval.

    Returns
    -------
    DataSet
    """
    tot = 0
    ds = _ds.DataSet(spamLabelIndices=self.slIndex)
    for gateStr,dsRow in self.iteritems():

      if dsRow.reps is None:
        reps = _np.ones(dsRow.sli.shape, self.repType)
      else: reps = dsRow.reps

      count_dict = {i: 0 for i in self.slIndex.values()}
      for spamIndx, t, rep in zip(dsRow.sli, dsRow.time, reps):
        if startTime <= t < endTime:
          count_dict[spamIndx] += rep
          tot += rep

      ds.add_count_dict(gateStr,
                        {self.sl[i]: cnt for i,cnt in count_dict.items()})

    if tot == 0:
      _warnings.warn("No counts in the requested time range: empty DataSet created")
    ds.done_adding_data()                                                                                                           
    return ds         


  def __getstate__(self):
    toPickle = { 'gsIndexKeys': map(_gs.CompressedGateString, self.gsIndex.keys()),
                 'gsIndexVals': list(self.gsIndex.values()),
                 'slIndex': self.slIndex,
                 'sl': self.sl,
                 'bStatic': self.bStatic,
                 'sliData': self.sliData,
                 'timeData': self.timeData,
                 'repData': self.repData,
                 'sliType': self.sliType,
                 'timeType': self.timeType,
                 'repType': self.repType}
    return toPickle

  def __setstate__(self, state_dict):
    gsIndexKeys = [ cgs.expand() for cgs in state_dict['gsIndexKeys'] ]
    self.gsIndex = _OrderedDict( list(zip( gsIndexKeys, state_dict['gsIndexVals'])) )
    self.slIndex = state_dict['slIndex']
    self.sl = state_dict['sl']
    self.bStatic = state_dict['bStatic']
    self.sliData  = state_dict['sliData']
    self.timeData = state_dict['timeData']
    self.repData  = state_dict['repData']
    self.sliType  = state_dict['sliType']
    self.timeType = state_dict['timeType']
    self.repType  = state_dict['repType']


  def save(self, fileOrFilename):
    """ 
    Save this TDDataSet to a file.

    Parameters
    ----------
    fileOrFilename : string or file object
        If a string,  interpreted as a filename.  If this filename ends 
        in ".gz", the file will be gzip compressed.

    Returns
    -------
    None
    """
    
    toPickle = { 'gsIndexKeys': map(_gs.CompressedGateString, self.gsIndex.keys()) if self.gsIndex else [],
                 'gsIndexVals': list(self.gsIndex.values()) if self.gsIndex else [],
                 'slIndex': self.slIndex,
                 'sl': self.sl,
                 'bStatic': self.bStatic,
                 'sliType': self.sliType,
                 'timeType': self.timeType,
                 'repType': self.repType,
                 'useReps': bool(self.repData is None)} #Don't pickle counts numpy data b/c it's inefficient
    if not self.bStatic: toPickle['nRows'] = len(self.sliData)
    
    bOpen = (type(fileOrFilename) == str)
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
      _np.save(f, self.sliData)
      _np.save(f, self.timeData)
      if self.repData is not None:
        _np.save(f, self.repData)
    else: 
      for row in self.sliData: _np.save(f, row)
      for row in self.timeData: _np.save(f, row)
      if self.repData is not None:
        for row in self.repData: _np.save(f, row)
    if bOpen: f.close()

  def load(self, fileOrFilename):
    """
    Load TDDataSet from a file, clearing any data is contained previously.

    Parameters
    ----------
    fileOrFilename string or file object.
        If a string,  interpreted as a filename.  If this filename ends 
        in ".gz", the file will be gzip uncompressed as it is read.

    Returns
    -------
    None
    """
    bOpen = (type(fileOrFilename) == str)
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
    self.sl      = state_dict['sl']
    self.bStatic = state_dict['bStatic']
    self.sliType = state_dict['sliType']
    self.timeType= state_dict['timeType']
    self.repType = state_dict['repType']
    useReps = state_dict['useReps']

    if self.bStatic:
      self.sliData = _np.lib.format.read_array(f) #_np.load(f) doesn't play nice with gzip
      self.timeData = _np.lib.format.read_array(f) #_np.load(f) doesn't play nice with gzip
      if useReps:
        self.repData = _np.lib.format.read_array(f) #_np.load(f) doesn't play nice with gzip
    else:
      self.sliData = []
      for i in range(state_dict['nRows']):
        self.sliData.append( _np.lib.format.read_array(f) ) #_np.load(f) doesn't play nice with gzip

      self.timeData = []
      for i in range(state_dict['nRows']):
        self.timeData.append( _np.lib.format.read_array(f) ) #_np.load(f) doesn't play nice with gzip

      if useReps:
        self.repData = []
        for i in range(state_dict['nRows']):
          self.repData.append( _np.lib.format.read_array(f) ) #_np.load(f) doesn't play nice with gzip
      else:
        self.repData = None

    if bOpen: f.close()
