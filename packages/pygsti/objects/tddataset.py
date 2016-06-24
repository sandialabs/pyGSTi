#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation              
#    This Software is released under the GPL license detailed    
#    in the file "license.txt" in the top-level pyGSTi directory 
#*****************************************************************
""" Defines the DataSet class and supporting classes and functions """

from __future__ import print_function

import numpy as _np
import scipy.special as _sps
import scipy.fftpack as _fft
from scipy.integrate import quad as _quad
import cPickle as _pickle
import warnings as _warnings
from collections import OrderedDict as _OrderedDict

from ..tools import listtools as _lt

import gatestring as _gs
import dataset as _ds


class TDDataSet_KeyValIterator(object):
  """ Iterator class for gate_string,TDDataSetRow pairs of a TDDataSet """
  def __init__(self, dataset):
    self.dataset = dataset
    self.gsIter = dataset.gsIndex.__iter__()
    self.timeseriesIter = dataset.timeseries.__iter__()

  def __iter__(self):
    return self

  def next(self): # Python 3: def __next__(self)
    return self.gsIter.next(), TDDataSetRow(self.dataset, self.timeseriesIter.next())
  

class TDDataSet_ValIterator(object):
  """ Iterator class for TDDataSetRow values of a TDDataSet """
  def __init__(self, dataset):
    self.dataset = dataset
    self.timeseriesIter = dataset.timeseries.__iter__()

  def __iter__(self):
    return self

  def next(self): # Python 3: def __next__(self)
    return TDDataSetRow(self.dataset, self.timeseriesIter.next())


class TDDataSetRow(object):
  """ 
  Encapsulates TDDataSet time series data for a single gate string.  Outwardly
    looks similar to a list with spam labels as the values.
  """
  def __init__(self, dataset, rowData):
    self.dataset = dataset
    self.rowData = rowData

  def __iter__(self):
    return ( self.dataset.sl[i] for i in self.rowData )

  def total(self):
    """ Returns the total counts."""
    return float(len(self.rowData))

  def fraction(self,spamlabel):
    """ Returns the fraction of total counts for spamlabel."""
    cnt = _np.count_nonzero( _np.equal(self.rowData,
                                       self.dataset.slIndex[spamlabel]) )
    return float(cnt) / self.total()

  def __getitem__(self,index):
    return self.dataset.sl[ self.rowData[index] ]

  def __setitem__(self,index,spamlabel):
    self.rowData[ index ] = self.dataset.slIndex[spamlabel]

  def __str__(self):
    return str(self.rowData)

  def __len__(self):
    return len(self.rowData)

  


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

  def __init__(self, timeseries=None, gateStrings=None, gateStringIndices=None, 
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
      assert(timeseries is None and gateStrings is None and gateStringIndices is None and spamLabels is None and spamLabelIndices is None)
      self.load(fileToLoadFrom)
      return

    # self.gsIndex  :  Ordered dictionary where keys = GateString objects, values = integer indices into timeseries
    if gateStringIndices is not None:
      self.gsIndex = gateStringIndices
    elif gateStrings is not None:
      dictData = [ (gs if isinstance(gs,_gs.GateString) else _gs.GateString(gs),i) \
                     for (i,gs) in enumerate(gateStrings) ] #convert to GateStrings if necessary
      self.gsIndex = _OrderedDict( dictData )

    elif not bStatic:
      self.gsIndex = _OrderedDict() 
    else: raise ValueError("Must specify either gateStrings or gateStringIndices when creating a static TDDataSet")

    # self.slIndex  :  Ordered dictionary where keys = spam labels (strings), values = integer indices mapping 
    #                    timeseries data (integers) onto the spam labels.
    if spamLabelIndices is not None:
      self.slIndex = spamLabelIndices
    elif spamLabels is not None:
      self.slIndex = _OrderedDict( [(sl,i) for (i,sl) in enumerate(spamLabels) ] )
    else: raise ValueError("Must specify either spamLabels or spamLabelIndices when creating a TDDataSet")

    # self.sl :  Ordered dictionary where keys = integer indices, values = spam labels (strings)
    self.sl = _OrderedDict( [(i,sl) for (sl,i) in self.slIndex.iteritems()] )


    if self.gsIndex:  assert( min(self.gsIndex.values()) >= 0)
    if self.slIndex:  assert( min(self.slIndex.values()) >= 0)

    # self.timeseries  :  when bStatic == True a single 2D numpy array.  Rows = gate strings, Cols = time steps      
    #                     when bStatic == False a list of 1D numpy arrays. Each array has length = num of time steps
    if timeseries is not None:
      self.timeseries = timeseries

      if len(self.gsIndex) > 0:
        maxIndex = max(self.gsIndex.values())
        maxSLIndex = max(self.slIndex.values())
        if bStatic:
          assert( self.timeseries.shape[0] > maxIndex and _np.amax(self.timeseries) <= maxSLIndex )
        else:
          assert( len(self.timeseries) > maxIndex )
          if len(self.timeseries) > 0:
              nTimeSteps = len(self.timeseries[0])
              assert( all( [ len(series) == nTimeSteps for series in self.timeseries ] ) )
              assert( all( [ max(series) <= maxSLIndex for series in self.timeseries ] ) )
      #else gsIndex has length 0 so there are no gate strings in this dataset (even though timeseries can contain data)

    elif not bStatic:
      assert( len(self.gsIndex) == 0)
      self.timeseries = []

    else:
      raise ValueError("time series data must be specified when creating a static TDDataSet")
    
    # self.bStatic
    self.bStatic = bStatic

    # self.ffdata : fourier filtering data
    self.ffdata = []


  def __iter__(self):
    return self.gsIndex.__iter__() #iterator over gate strings

  def __len__(self):
    return len(self.gsIndex)

  def __getitem__(self, gatestring):
    return TDDataSetRow(self, self.timeseries[ self.gsIndex[gatestring] ])

  def __setitem__(self, gatestring, spamLabelList):
    if gatestring in self:
      assert(len(spamLabelList) == len(self.timeseries[0])) #all time series must have the same length
      row = TDDataSetRow(self, self.timeseries[ self.gsIndex[gatestring] ])
      for i,spamLabel in enumerate(spamLabelList):
        row[spamLabel] = cnt
    else:
      self.add_timeseries_list(gatestring, spamLabelList)

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


  def num_timesteps(self):
    """ 
    Get the number of time steps in this data set.

    Returns
    -------
    int
    """
    if len(self.timeseries) > 0:
      return len(self.timeseries[0])
    else: return 0


  def add_timeseries_list(self, gateString, spamLabelList):
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

    if len(self.timeseries) > 0:
      nTimeSteps = len(self.timeseries[0])
      assert(len(spamLabelList) == len(self.timeseries[0])) #all time series must have the same length
    
    seriesArray = _np.array([ self.slIndex[sl] for sl in spamLabelList ] , 'i')

    if gateString in self.gsIndex:
      gateStringIndx = self.gsIndex[gateString]
      self.timeseries[ gateStringIndx ] = seriesArray #OVERWRITE existing time series
    else:
      #add data for a new gatestring
      gateStringIndx = len(self.timeseries) #index of to-be-added gate string
      self.timeseries.append( seriesArray )
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
    assert(self.get_spam_labels() == otherDataSet.get_spam_labels())
    assert(self.keys() == otherDataSet.keys())
    for i in range(self.keys()):
      self.timeseries[i] = _np.concatenate( (self.timeseries[i], otherDataSet.timeseries[i]) )

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
      trunc_dataset = TDDataSet(self.timeseries, gateStringIndices=trunc_gsIndex, spamLabelIndices=self.slIndex, bStatic=True) #don't copy counts, just reference
      #trunc_dataset = StaticDataSet(self.counts.take(gateStringIndices,axis=0), gateStrings=gateStrings, spamLabelIndices=self.slIndex)

    else:
      trunc_dataset = TDDataSet(spamLabels=self.get_spam_labels())
      for gateString in _lt.remove_duplicates(listOfGateStringsToKeep):
        if gateString in self.gsIndex:
          gateStringIndx = self.gsIndex[gateString]
          trunc_dataset.add_timeseries_list( gateString, self.timeseries[ gateStringIndx ].copy() ) #Copy operation so truncated dataset can be modified
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
      copyOfMe.timeseries = [ el.copy() for el in self.timeseries ]
      return copyOfMe


  def copy_nonstatic(self):
    """ Make a non-static copy of this TDDataSet. """
    if self.bStatic: 
      copyOfMe = TDDataSet(spamLabels=self.get_spam_labels())
      copyOfMe.gsIndex = self.gsIndex.copy()
      copyOfMe.timeseries = [ el.copy() for el in self.timeseries ]
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
    #  gsIndex and slIndex stay the same ; counts is transformed to a 2D numpy array
    if len(self.timeseries) > 0:
      newTimeseries = _np.concatenate( [el.reshape(1,-1) for el in self.timeseries], axis=0 )
    else:
      newTimeseries = _np.empty( (0,0), 'i')
    self.timeseries, self.bStatic = newTimeseries, True

  def compute_fourier_filtering(self, n_sigma=3, slope_compensation=False, verbosity=0):
    """ Compute and store significant fourier coefficients -- EXPERIMENTAL TODO DOCSTRING"""
    if not self.bStatic: 
      raise ValueError("TDDataSet object must be *static* to compute fourier filters")

    #Computes the order statistic

    # Compute the log factorial function for numbers up to 2**15
    # we could go higher, but this is already a lot of data
    log_factorial = _np.append([0.0],_np.cumsum(map(_np.log, _np.arange(2**15)+1)))
    
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

    def fourier_filter(data, n_sigma=3., keep=None, return_aux=False, verbose=False, 
                   truncate=True, slope_compensation=False):
        """
        Process a list of binary data to produce an estimate of the time-dependent 
        probability underlying the data.  Fourier components are kept if their 
        magnitude is more than sigma from the expected value.
        """
        
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
    
        if slope_compensation:
            filtered_data += slope * _np.linspace(-.5,.5,len(data))
       
        if truncate:
            filtered_data = _np.maximum(_np.minimum(filtered_data,1-1.e-6),1.e-6)
        
        if return_aux:
            return filtered_data, fft_data, threshold, n_kept
        else:
            return filtered_data

    # For each spam label, get 1's and 0's string to be "fourier filtered" 
    # into a time-dependent probability function.
    for tseries in self.timeseries:
      ff_data_dict = {}
      for spamLabel,spamIndx in self.slIndex.iteritems():
        tdata = _np.where( tseries == spamIndx, 1, 0 )
          
        # Filter modes less than 5 sigma from expected maximum.
        filtered, fft_data, threshold, n_kept = fourier_filter(
          tdata, n_sigma=n_sigma, return_aux=True,
          slope_compensation=slope_compensation, verbose=(verbosity > 0))
        ff_data_dict[spamLabel] = (filtered, fft_data, threshold, n_kept)

      self.ffdata.append( ff_data_dict )


  def create_dataset_at_time(self, timeIndex):
    ds = _ds.DataSet(spamLabelIndices=self.slIndex)
    for gateStr,i in self.gsIndex.iteritems():
      ff_data_dict = self.ffdata[i]
      nTimeSteps = self.num_timesteps()
      avgKept = _np.average( [ff_data_dict[sl][3] for sl in self.slIndex ] )
      N = int(round(float(nTimeSteps)/avgKept)) # ~ clicks per fourier mode

      count_dict = {}
      for sl in self.slIndex:
        (filtered, fft_data, threshold, n_kept) = ff_data_dict[sl]
        p = filtered[timeIndex]
        count_dict[sl] = N*p

      ds.add_count_dict(gateStr, count_dict)
    ds.done_adding_data()
    return ds


  def __getstate__(self):
    toPickle = { 'gsIndexKeys': map(_gs.CompressedGateString, self.gsIndex.keys()),
                 'gsIndexVals': self.gsIndex.values(),
                 'slIndex': self.slIndex,
                 'sl': self.sl,
                 'bStatic': self.bStatic,
                 'timeseries': self.timeseries }
    return toPickle

  def __setstate__(self, state_dict):
    gsIndexKeys = [ cgs.expand() for cgs in state_dict['gsIndexKeys'] ]
    self.gsIndex = _OrderedDict( zip( gsIndexKeys, state_dict['gsIndexVals']) )
    self.slIndex = state_dict['slIndex']
    self.sl = state_dict['sl']
    self.timeseries = state_dict['timeseries']
    self.bStatic = state_dict['bStatic']


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
                 'gsIndexVals': self.gsIndex.values() if self.gsIndex else [],
                 'slIndex': self.slIndex,
                 'sl': self.sl,
                 'bStatic': self.bStatic } #Don't pickle counts numpy data b/c it's inefficient
    if not self.bStatic: toPickle['nRows'] = len(self.timeseries)
    
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
      _np.save(f, self.timeseries)
    else: 
      for rowArray in self.timeseries:
        _np.save(f, rowArray)
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
    self.gsIndex = _OrderedDict( zip( gsIndexKeys, state_dict['gsIndexVals']) )
    self.slIndex = state_dict['slIndex']
    self.sl      = state_dict['sl']
    self.bStatic = state_dict['bStatic']

    if self.bStatic:
      self.timeseries = _np.lib.format.read_array(f) #_np.load(f) doesn't play nice with gzip
    else:
      self.timeseries = []
      for i in range(state_dict['nRows']):
        self.timeseries.append( _np.lib.format.read_array(f) ) #_np.load(f) doesn't play nice with gzip
    if bOpen: f.close()
