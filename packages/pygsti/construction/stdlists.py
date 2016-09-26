from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Gate string list creation functions using repeated-germs limited by a max-length."""

import numpy.random as _rndm
import itertools as _itertools
from ..tools import listtools as _lt
from . import gatestringconstruction as _gsc
from . import spamspecconstruction as _ssc


def make_lsgst_lists(gateLabels, prepStrs, effectStrs, germList, maxLengthList,
                     fidPairs=None, truncScheme="whole germ powers", nest=True,
                     keepFraction=1, keepSeed=None):
    """
    Create a set of gate string lists for LSGST based on germs and max-lengths.

    Constructs a series (a list) of gate string lists used by long-sequence GST
    (LSGST) algorithms.  If maxLengthList[0] == 0 then the starting list is the
    list of LGST strings, otherwise the starting list is empty.  For each
    nonzero element of maxLengthList, call it L, a list of gate strings is
    created with the form:

    Case: truncScheme == 'whole germ powers':
      prepStr + pygsti.construction.repeat_with_max_length(germ,L) + effectStr

    Case: truncScheme == 'truncated germ powers':
      prepStr + pygsti.construction.repeat_and_truncate(germ,L) + effectStr

    Case: truncScheme == 'length as exponent':
      prepStr + germ^L + effectStr

    If nest == True, the above list is iteratively *added* (w/duplicates
    removed) to the current list of gate strings to form a final list for the
    given L.  This results in successively larger lists, each of which
    contains all the elements of previous-L lists.  If nest == False then
    the above list *is* the final list for the given L.

    Parameters
    ----------
    gateLabels : list or tuple
        List of gate labels to determine needed LGST strings.  Only relevant
        when maxLengthList[0] == 0.

    prepStrs : list of GateStrings
        List of the preparation fiducial gate strings, which follow state
        preparation.

    effectStrs : list of GateStrings
        List of the measurement fiducial gate strings, which precede
        measurement.

    germList : list of GateStrings
        List of the germ gate strings.

    maxLengthList : list of ints
        List of maximum lengths.  If maxLengthList[0] == 0 this results in
        special behavior where LGST strings are included as the first
        returned list.

    fidPairs : list of 2-tuples or dict, optional
        Specifies a subset of all fiducial string pairs (prepStr, effectStr)
        to be used in the gate string lists.  If a list, each element of 
        fidPairs is a (iPrepStr, iEffectStr) 2-tuple of integers, each 
        indexing a string within prepStrs and effectStrs, respectively, so 
        that prepStr = prepStrs[iPrepStr] and effectStr = 
        effectStrs[iEffectStr].  If a dictionary, keys are germs (elements
        of germList) and values are lists of 2-tuples specifying the pairs
        to use for that germ.

    truncScheme : str, optional
        Truncation scheme used to interpret what the list of maximum lengths
        means. If unsure, leave as default. Allowed values are:

        - 'whole germ powers' -- germs are repeated an integer number of
          times such that the length is less than or equal to the max.
        - 'truncated germ powers' -- repeated germ string is truncated
          to be exactly equal to the max (partial germ at end is ok).
        - 'length as exponent' -- max. length is instead interpreted
          as the germ exponent (the number of germ repetitions).

    nest : boolean, optional
        If Frue, the returned gate string lists are "nested", meaning
        that each successive list of gate strings contains all the gate
        strings found in previous lists (and usually some additional
        new ones).  If False, then the returned string list for maximum
        length == L contains *only* those gate strings specified in the
        description above, and *not* those for previous values of L.

    keepFraction : float, optional
        The fraction of fiducial pairs selected for each germ-power base
        string.  The default includes all fiducial pairs.  Note that
        for each germ-power the selected pairs are *different* random
        sets of all possible pairs (unlike fidPairs, which specifies the
        *same* fiducial pairs for *all* same-germ base strings).  If
        fidPairs is used in conjuction with keepFraction, the pairs
        specified by fidPairs are always selected, and any additional
        pairs are randomly selected.

    keepSeed : int, optional
        The seed used for random fiducial pair selection (only relevant
        when keepFraction < 1).

    Returns
    -------
    list of (lists of GateStrings)
        The i-th list corresponds to a gate string list containing repeated
        germs limited to length maxLengthList[i].  If nest == True, then
        repeated germs limited to previous max-lengths are also included.
        Note that a "0" maximum-length corresponds to the LGST strings.
    """
    lgstStrings = _gsc.list_lgst_gatestrings( _ssc.build_spam_specs(prepStrs = prepStrs, effectStrs = effectStrs),
                                              gateLabels)
    lsgst_list = _gsc.gatestring_list([ () ]) #running list of all strings so far

    if keepFraction < 1.0:
        rndm = _rndm.RandomState(keepSeed) # ok if seed is None
        nPairs = len(prepStrs)*len(effectStrs)
        nPairsToKeep = int(round(float(keepFraction) * nPairs))
    else: rndm = None

    if isinstance(fidPairs, dict) or hasattr(fidPairs, "keys"):
        fiducialPairs = { germ: [ (prepStrs[i],effectStrs[j]) 
                                  for (i,j) in fidPairs[germ] ]
                          for germ in germList }
        fidPairDict = fidPairs
    else:
        if fidPairs is not None:   #assume fidPairs is a list
            fidPairDict = { germ:fidPairs for germ in germList }
            lst = [ (prepStrs[i],effectStrs[j]) for (i,j) in fidPairs ]
        else:
            fidPairDict = None
            lst = list(_itertools.product(prepStrs, effectStrs))
        fiducialPairs = { germ:lst for germ in germList }

        
    if maxLengthList[0] == 0:
        lsgst_listOfLists = [ lgstStrings ]
        maxLengthList = maxLengthList[1:]
    else: lsgst_listOfLists = [ ]

    Rfn = _getTruncFunction(truncScheme)

    for maxLen in maxLengthList:

        lst = []
        for germ in germList:

            if rndm is None:
                fiducialPairsThisIter = fiducialPairs[germ]

            elif fidPairDict is not None:
                pair_indx_tups = fidPairDict[germ]
                remainingPairs = [ (prepStrs[i],effectStrs[j])
                                   for i in range(len(prepStrs))
                                   for j in range(len(effectStrs))
                                   if (i,j) not in pair_indx_tups ]
                nPairsRemaining = len(remainingPairs)
                nPairsToChoose = nPairsToKeep-len(pair_indx_tups)
                nPairsToChoose = max(0,min(nPairsToChoose,nPairsRemaining))
                assert(0 <= nPairsToChoose <= nPairsRemaining)
                # FUTURE: issue warnings when clipping nPairsToChoose?

                fiducialPairsThisIter = fiducialPairs[germ] + \
                    [ remainingPairs[k] for k in
                      sorted(rndm.choice(nPairsRemaining,nPairsToChoose,
                                         replace=False))]

            else: # rndm is not None and fidPairDict is None
                assert(nPairsToKeep <= nPairs) # keepFraction must be <= 1.0
                fiducialPairsThisIter = \
                    [ fiducialPairs[germ][k] for k in
                      sorted(rndm.choice(nPairs,nPairsToKeep,replace=False))]

            lst += _gsc.create_gatestring_list("f[0]+R(germ,N)+f[1]",
                                              f=fiducialPairsThisIter,
                                              germ=germ, N=maxLen,
                                              R=Rfn, order=('f',))
        if nest:
            lsgst_list += lst #add new strings to running list
            lsgst_listOfLists.append( _lt.remove_duplicates(lgstStrings + lsgst_list) )
        else:
            lsgst_listOfLists.append( _lt.remove_duplicates(lst) )

    #print "%d LSGST sets w/lengths" % len(lsgst_listOfLists),map(len,lsgst_listOfLists)
    return lsgst_listOfLists


def make_lsgst_experiment_list(gateLabels, prepStrs, effectStrs, germList,
                               maxLengthList, fidPairs=None,
                               truncScheme="whole germ powers", keepFraction=1,
                               keepSeed=None):
    """
    Create a list of all the gate strings (i.e. the experiments) required for
    long-sequence GST (LSGST) algorithms.

    Returns a single list containing, without duplicates, all the gate
    strings required throughout all the iterations of LSGST given by
    maxLengthList.  Thus, the returned list is equivalently the list of
    the experiments required to run LSGST using the supplied parameters,
    and so commonly used when construting data set templates or simulated
    data sets.  The breakdown of which gate strings are used for which
    iteration(s) of LSGST is given by make_lsgst_lists(...).

    Parameters
    ----------
    gateLabels : list or tuple
        List of gate labels to determine needed LGST strings.  Only relevant
        when maxLengthList[0] == 0.

    prepStrs : list of GateStrings
        List of the preparation fiducial gate strings, which follow state
        preparation.

    effectStrs : list of GateStrings
        List of the measurement fiducial gate strings, which precede
        measurement.

    germList : list of GateStrings
        List of the germ gate strings.

    maxLengthList : list of ints
        List of maximum lengths.  If maxLengthList[0] == 0 this results in
        special behavior where LGST strings are included as the first
        returned list.

    fidPairs : list of 2-tuples, optional
        Specifies a subset of all fiducial string pairs (prepStr, effectStr)
        to be used in the gate string lists.  If a list, each element of 
        fidPairs is a (iPrepStr, iEffectStr) 2-tuple of integers, each 
        indexing a string within prepStrs and effectStrs, respectively, so 
        that prepStr = prepStrs[iPrepStr] and effectStr = 
        effectStrs[iEffectStr].  If a dictionary, keys are germs (elements
        of germList) and values are lists of 2-tuples specifying the pairs
        to use for that germ.

    truncScheme : str, optional
        Truncation scheme used to interpret what the list of maximum lengths
        means. If unsure, leave as default. Allowed values are:

        - 'whole germ powers' -- germs are repeated an integer number of
          times such that the length is less than or equal to the max.
        - 'truncated germ powers' -- repeated germ string is truncated
          to be exactly equal to the max (partial germ at end is ok).
        - 'length as exponent' -- max. length is instead interpreted
          as the germ exponent (the number of germ repetitions).

    keepFraction : float, optional
        The fraction of fiducial pairs selected for each germ-power base
        string.  The default includes all fiducial pairs.  Note that
        for each germ-power the selected pairs are *different* random
        sets of all possible pairs (unlike fidPairs, which specifies the
        *same* fiduicial pairs for *all* same-germ base strings).  If
        fidPairs is used in conjuction with keepFraction, the pairs
        specified by fidPairs are always selected, and any additional
        pairs are randomly selected.

    keepSeed : int, optional
        The seed used for random fiducial pair selection (only relevant
        when keepFraction < 1).


    Returns
    -------
    list of GateStrings
    """
    nest = True # => the final list contains all of the strings
    return make_lsgst_lists(gateLabels, prepStrs, effectStrs, germList,
                            maxLengthList, fidPairs, truncScheme, nest,
                            keepFraction, keepSeed)[-1]



def make_elgst_lists(gateLabels, germList, maxLengthList,
                     truncScheme="whole germ powers", nest=True):
    """
    Create a set of gate string lists for eLGST based on germs and max-lengths

    Constructs a series (a list) of gate string lists used by the extended LGST
    (eLGST) algorithm.  If maxLengthList[0] == 0 then the starting list is the
    list of length-1 gate label strings, otherwise the starting list is empty.
    For each nonzero element of maxLengthList, call it L, a list of gate strings is
    created with the form:

    Case: truncScheme == 'whole germ powers':
      pygsti.construction.repeat_with_max_length(germ,L)

    Case: truncScheme == 'truncated germ powers':
      pygsti.construction.repeat_and_truncate(germ,L)

    Case: truncScheme == 'length as exponent':
      germ^L

    If nest == True, the above list is iteratively *added* (w/duplicates
    removed) to the current list of gate strings to form a final list for the
    given L.  This results in successively larger lists, each of which
    contains all the elements of previous-L lists.  If nest == False then
    the above list *is* the final list for the given L.

    Parameters
    ----------
    gateLabels : list or tuple
        List of gate labels. Only relevant when maxLengthList[0] == 0.

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

    nest : boolean, optional
        If Frue, the returned gate string lists are "nested", meaning
        that each successive list of gate strings contains all the gate
        strings found in previous lists (and usually some additional
        new ones).  If False, then the returned string list for maximum
        length == L contains *only* those gate strings specified in the
        description above, and *not* those for previous values of L.


    Returns
    -------
    list of (lists of GateStrings)
        The i-th list corresponds to a gate string list containing repeated
        germs limited to length maxLengthList[i].  If nest == True, then
        repeated germs limited to previous max-lengths are also included.
        Note that a "0" maximum-length corresponds to the gate
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
        lst = _gsc.create_gatestring_list("R(germ,N)", germ=germList, N=maxLen, R=Rfn)
        if nest:
            elgst_list += lst #add new strings to running list
            elgst_listOfLists.append( _lt.remove_duplicates(singleGates + elgst_list) )
        else:
            elgst_listOfLists.append( _lt.remove_duplicates(lst) )

    #print "%d eLGST sets w/lengths" % len(elgst_listOfLists),map(len,elgst_listOfLists)
    return elgst_listOfLists


def make_elgst_experiment_list(gateLabels, germList, maxLengthList,
                               truncScheme="whole germ powers"):
    """
    Create a list of all the gate strings (i.e. the experiments) required for
    the extended LGST (eLGST) algorithm.

    Returns a single list containing, without duplicates, all the gate
    strings required throughout all the iterations of eLGST given by
    maxLengthList.  Thus, the returned list is equivalently the list of
    the experiments required to run eLGST using the supplied parameters,
    and so commonly used when construting data set templates or simulated
    data sets.  The breakdown of which gate strings are used for which
    iteration(s) of eLGST is given by make_elgst_lists(...).

    Parameters
    ----------
    gateLabels : list or tuple
        List of gate labels. Only relevant when maxLengthList[0] == 0.

    germList : list of GateStrings
        List of the germ gate strings.

    maxLengthList : list of ints
        List of the maximum lengths.  If maxLengthList[0] == 0 this results in
        special behavior where the length-1 gate label strings are included as
        the first returned list.

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
    list of GateStrings
    """

    #When nest == True the final list contains all of the strings
    return make_elgst_lists(gateLabels, germList,
                            maxLengthList, truncScheme, nest=True)[-1]



def _getTruncFunction(truncScheme):
    if truncScheme == "whole germ powers":
        Rfn = _gsc.repeat_with_max_length
    elif truncScheme == "truncated germ powers":
        Rfn = _gsc.repeat_and_truncate
    elif truncScheme == "length as exponent":
        def Rfn(germ,N): return germ*N
    else:
        raise ValueError("Invalid truncation scheme: %s" % truncScheme)
    return Rfn
