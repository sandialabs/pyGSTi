""" Gate string list creation functions using repeated-germs limited by a max-length."""
import itertools as _itertools
from ..objects import spamspec as _ss
from ..tools import listtools as _lt
import gatestringconstruction as _gsc
import spamspecconstruction as _ssc

def make_lsgst_lists(gateLabels, fiducialList, germList, maxLengthList,
                     rhoEPairs=None, truncScheme="whole germ powers"):
    """
    Create a set of gate string lists for LSGST based on germs and max-lengths.

    Constructs a series of successively larger lists by iteratively adding to
    to a running list.  If maxLengthList[0] == 0 then the starting list is the 
    list of LGST strings, otherwise the starting list is empty.  For each 
    nonzero element of maxLengthList, call it L, add strings of the form:
    
    Case: truncScheme == 'whole germ powers':
      fiducial1 + pygsti.construction.repeat_with_max_length(germ,L) + fiducial2

    Case: truncScheme == 'truncated germ powers':
      fiducial1 + pygsti.construction.repeat_and_truncate(germ,L) + fiducial2

    Case: truncScheme == 'length as exponent':
      fiducial1 + germ^L + fiducial2

    to the running list and add the resulting list to the list (with 
    duplicates removed) of gate string lists that is ultimately returned.

    Parameters
    ----------
    gateLabels : list or tuple
        List of gate labels to determine needed LGST strings.  Only relevant
        when maxLengthList[0] == 0.

    fiducialList : list of GateStrings
        List of the fiducial gate strings.
        
    germList : list of GateStrings
        List of the germ gate strings.

    maxLengthList : list of ints
        List of maximum lengths.  If maxLengthList[0] == 0 this results in 
        special behavior where LGST strings are included as the first 
        returned list.

    rhoEPairs : list of 2-tuples, optional
        Specifies a subset of all fiducial string pairs (fiducial1, fiducial2)
        to be used in the gate string lists.  Each element of rhoEPairs is a 
        (iFiducial1, iFidicial2) 2-tuple of integers, each indexing a string
        within fiducialList so that fiducial1 = fiducialList[iFiducial1], etc.

    truncScheme : str, optional
        Truncation scheme used to interpret what the list of maximum lengths
        means. If unsure, leave as default. Allowed values are:
        
        - 'whole germ powers' -- germs are repeated an integer number of 
          times such that the length is less than or equal to the max.
        - 'truncated germ powers' -- repeated germ string is truncated
          to be exactly equal to the max (partial germ at end is ok).
        - 'length as exponent' -- max. length is instead interpreted
          as the germ exponent (the number of germ repetitions).

    Returns
    -------
    list of (lists of GateStrings)
        The i-th list corresponds to a gate string list containing
        repeated germs limited to lengths maxLengthList[0] through (and including)
        maxLengthList[i].  Note that a "0" exponent corresponds to the LGST
        strings.
    """
    return make_lsgst_lists_asymmetric_fids(gateLabels, fiducialList, fiducialList,
                                            germList, maxLengthList, rhoEPairs,
                                            truncScheme)
    
def make_lsgst_lists_asymmetric_fids(gateLabels, rhoStrs, EStrs, germList, maxLengthList,
                                     rhoEPairs=None, truncScheme="whole germ powers"):
    '''
    Same as make_lsgst_lists, except for asymmetric fiducial sets, specified by rhoStrs and EStrs.
    '''
    lgstStrings = _gsc.list_lgst_gatestrings( _ssc.build_spam_specs(rhoStrs = rhoStrs, EStrs = EStrs),
                                              gateLabels)
    lsgst_list = _gsc.gatestring_list([ () ]) #running list of all strings so far

    if rhoEPairs is not None:
        fiducialPairs = [ (rhoStrs[i],EStrs[j]) for (i,j) in rhoEPairs ]
    else:
        fiducialPairs = list(_itertools.product(rhoStrs, EStrs))
    
    if maxLengthList[0] == 0:
        lsgst_listOfLists = [ lgstStrings ]
        maxLengthList = maxLengthList[1:]
    else: lsgst_listOfLists = [ ]

    Rfn = _getTruncFunction(truncScheme)
        
    for maxLen in maxLengthList:
        lsgst_list += _gsc.create_gatestring_list("f[0]+R(germ,N)+f[1]",
                                                f=fiducialPairs,
                                                germ=germList, N=maxLen,
                                                R=Rfn, order=('germ','f'))
        lsgst_listOfLists.append( _lt.remove_duplicates(lgstStrings + lsgst_list) )

    #print "%d LSGST sets w/lengths" % len(lsgst_listOfLists),map(len,lsgst_listOfLists)
    return lsgst_listOfLists


def make_elgst_lists(gateLabels, germList, maxLengthList,
                     truncScheme="whole germ powers"):
    """
    Create a set of gate string lists for eLGST based on germs and max-lengths.

    Constructs a series of successively larger lists by iteratively adding to
    to a running list.  If maxLengthList[0] == 0 then the starting list is the 
    list of length-1 gate label strings, otherwise the starting list is empty.
    For each nonzero element of maxLengthList, call it L, add strings of the form:

    Case: truncScheme == 'whole germ powers':
      pygsti.construction.repeat_with_max_length(germ,L)

    Case: truncScheme == 'truncated germ powers':
      pygsti.construction.repeat_and_truncate(germ,L)

    Case: truncScheme == 'length as exponent':
      germ^L
    
    to the running list, and add the resulting list (with duplicates removed) 
    to the list of gate string lists that is ultimately returned.

    Parameters
    ----------
    gateLabels : list or tuple
        List of gate labels .Only relevant when expList[0] == 0.

    germList : list of GateStrings
        List of the germ gate strings.

    maxLengthList : list of ints
        List of the maximum lengths.  If maxLengthList[0] == 0 this results in
        special behavior where the length-1 gate label strings are included as
        the first returned list.

    truncScheme : str, optional
        Truncation scheme used to interpret what the list of maximum lengths
        means.  If unsure, leave as default. Allowed values are:
        
        - 'whole germ powers' -- germs are repeated an integer number of 
          times such that the length is less than or equal to the max.
        - 'truncated germ powers' -- repeated germ string is truncated
          to be exactly equal to the max (partial germ at end is ok).
        - 'length as exponent' -- max. length is instead interpreted
          as the germ exponent (the number of germ repetitions).


    Returns
    -------
    list of (lists of GateStrings)
        The i-th list corresponds to a gate string list containing
        repeated germs limited to lengths maxLengthList[0] through (and including)
        maxLengthList[i].  Note that a "0" exponent corresponds to the gate
        label strings.
    """
    singleGates = _gsc.gatestring_list([(g,) for g in gateLabels])
    elgst_list = _gsc.gatestring_list([ () ])  #running list of all strings so far
    
    if maxLengthList[0] == 0:
        elgst_listOfLists = [ singleGates ]
        maxLengthList = maxLengthList[1:]
    else: elgst_listOfLists = [ ]

    Rfn = _getTruncFunction(truncScheme)

    for maxLen in maxLengthList:
        elgst_list += _gsc.create_gatestring_list("R(germ,N)", germ=germList, N=maxLen, R=Rfn)
        elgst_listOfLists.append( _lt.remove_duplicates(singleGates + elgst_list) )

    #print "%d eLGST sets w/lengths" % len(elgst_listOfLists),map(len,elgst_listOfLists)
    return elgst_listOfLists


def _getTruncFunction(truncScheme):
    if truncScheme == "whole germ powers":
        Rfn = _gsc.repeat_with_max_length
    elif truncScheme == "truncated germ powers":
        Rfn = _gsc.repeat_and_truncate
    elif truncScheme == "length as exponent":
        def Rfn(germ,N): return germ*N
    else:
        raise ValueError("Invalid truncation scheme: %s" % truncSheme)
    return Rfn



    
