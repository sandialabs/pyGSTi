""" Gate string list creation functions using repeated-germs limited by a max-length."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy.random as _rndm
import itertools as _itertools
import warnings as _warnings
from ..tools import listtools as _lt
from ..objects import LsGermsStructure as _LsGermsStructure
from ..objects import GateSet as _GateSet
from ..objects import GateString as _GateString
from ..baseobjs import VerbosityPrinter as _VerbosityPrinter
from . import gatestringconstruction as _gsc


def make_lsgst_lists(gateLabelSrc, prepStrs, effectStrs, germList, maxLengthList,
                     fidPairs=None, truncScheme="whole germ powers", nest=True,
                     keepFraction=1, keepSeed=None, includeLGST=True,
                     germLengthLimits=None):
    """
    Create a set of gate string lists for LSGST based on germs and max-lengths.

    Constructs a series (a list) of gate string lists used by long-sequence GST
    (LSGST) algorithms.  If `includeLGST == True` then the starting list is the
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
    gateLabelSrc : list or GateSet
        List of gate labels to determine needed LGST strings.  If a GateSet,
        then the gate set's gate and instrument labels are used. Only
        relevant when `includeLGST == True`.

    prepStrs : list of GateStrings
        List of the preparation fiducial gate strings, which follow state
        preparation.

    effectStrs : list of GateStrings
        List of the measurement fiducial gate strings, which precede
        measurement.

    germList : list of GateStrings
        List of the germ gate strings.

    maxLengthList : list of ints
        List of maximum lengths. A zero value in this list has special
        meaning, and corresponds to the LGST sequences.

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
        If True, the returned gate string lists are "nested", meaning
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

    includeLGST : boolean, optional
        If true, then the starting list (only applicable when
        `nest == True`) is the list of LGST strings rather than the
        empty list.  This means that when `nest == True`, the LGST
        sequences will be included in all the lists.

    germLengthLimits : dict, optional
        A dictionary limiting the max-length values used for specific germs.
        Keys are germ sequences and values are integers.  For example, if
        this argument is `{('Gx',): 4}` and `maxLengthList = [1,2,4,8,16]`,
        then the germ `('Gx',)` is only repeated using max-lengths of 1, 2,
        and 4 (whereas other germs use all the values in `maxLengthList`).


    Returns
    -------
    list of (lists of GateStrings)
        The i-th list corresponds to a gate string list containing repeated
        germs limited to length maxLengthList[i].  If nest == True, then
        repeated germs limited to previous max-lengths are also included.
        Note that a "0" maximum-length corresponds to the LGST strings.
    """
    if germLengthLimits is None: germLengthLimits = {}
    if nest == True and includeLGST == True and len(maxLengthList) > 0 and maxLengthList[0] == 0:
        _warnings.warn("Setting the first element of a max-length list to zero"
                       + " to ensure the inclusion of LGST sequences has been"
                       + " replaced by the `includeLGST` parameter which"
                       + " defaults to `True`.  Thus, in most cases, you can"
                       + " simply remove the leading 0 and start your"
                       + " max-length list at 1 now."
                       + "")

    if isinstance(gateLabelSrc, _GateSet):
        gateLabels = list(gateLabelSrc.gates.keys()) + \
                     list(gateLabelSrc.instruments.keys())
    else: gateLabels = gateLabelSrc

    lgst_list = _gsc.list_lgst_gatestrings(prepStrs, effectStrs, gateLabels)

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


    #running list of all strings so far (LGST strings or empty)
    lsgst_list = lgst_list[:] if includeLGST else _gsc.gatestring_list([ () ])
    lsgst_listOfLists = [ ] # list of lists to return

    Rfn = _getTruncFunction(truncScheme)

    for maxLen in maxLengthList:

        lst = []
        if maxLen == 0:
            #Special LGST case
            lst += lgst_list[:]
        else:
            #Typical case of germs repeated to maxLen using Rfn
            for germ in germList:
                if maxLen > germLengthLimits.get(germ,1e100): continue

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
            lsgst_listOfLists.append( _lt.remove_duplicates(lsgst_list) )
        else:
            lsgst_listOfLists.append( _lt.remove_duplicates(lst) )

    #print "%d LSGST sets w/lengths" % len(lsgst_listOfLists),map(len,lsgst_listOfLists)
    return lsgst_listOfLists


def make_lsgst_structs(gateLabelSrc, prepStrs, effectStrs, germList, maxLengthList,
                       fidPairs=None, truncScheme="whole germ powers", nest=True,
                       keepFraction=1, keepSeed=None, includeLGST=True,
                       gateLabelAliases=None, sequenceRules=None,
                       dscheck=None, actionIfMissing="raise", germLengthLimits=None,
                       verbosity=0):
    """
    Create a set of gate string structures for LSGST.

    Constructs a series (a list) of gate string structures used by long-sequence
    GST (LSGST) algorithms.  If `includeLGST == True` then the starting
    structure contains the LGST strings, otherwise the starting structure is
    empty.  For each nonzero element of maxLengthList, call it L, a set of
    gate strings is created with the form:

    Case: truncScheme == 'whole germ powers':
      prepStr + pygsti.construction.repeat_with_max_length(germ,L) + effectStr

    Case: truncScheme == 'truncated germ powers':
      prepStr + pygsti.construction.repeat_and_truncate(germ,L) + effectStr

    Case: truncScheme == 'length as exponent':
      prepStr + germ^L + effectStr

    If nest == True, the above set is iteratively *added* (w/duplicates
    removed) to the current gate string structure to form a final structure for
    the given L.  This results in successively larger structures, each of which
    contains all the elements of previous-L structures.  If nest == False then
    the above set *is* the final structure for the given L.

    Parameters
    ----------
    gateLabelSrc : list or GateSet
        List of gate labels to determine needed LGST strings.  If a GateSet,
        then the gate set's gate and instrument labels are used. Only
        relevant when `includeLGST == True`.

    prepStrs : list of GateStrings
        List of the preparation fiducial gate strings, which follow state
        preparation.

    effectStrs : list of GateStrings
        List of the measurement fiducial gate strings, which precede
        measurement.

    germList : list of GateStrings
        List of the germ gate strings.

    maxLengthList : list of ints
        List of maximum lengths. A zero value in this list has special
        meaning, and corresponds to the LGST sequences.

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
        If True, the returned gate string lists are "nested", meaning
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

    includeLGST : boolean, optional
        If true, then the starting list (only applicable when
        `nest == True`) is the list of LGST strings rather than the
        empty list.  This means that when `nest == True`, the LGST
        sequences will be included in all the lists.

    gateLabelAliases : dictionary, optional
        Dictionary whose keys are gate label "aliases" and whose values are tuples
        corresponding to what that gate label should be expanded into before querying
        the dataset.  This information is stored within the returned gate string
        structures.  Defaults to the empty dictionary (no aliases defined)
        e.g. gateLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    sequenceRules : list, optional
        A list of `(find,replace)` 2-tuples which specify string replacement
        rules.  Both `find` and `replace` are tuples of gate labels
        (or `GateString` objects).

    dscheck : DataSet, optional
        A data set which is checked for each of the generated gate strings. When
        a generated sequence is missing from this `DataSet`, action is taken
        according to `actionIfMissing`.

    actionIfMissing : {"raise","drop"}, optional
        The action to take when a generated gate sequence is missing from
        `dscheck` (only relevant when `dscheck` is not None).  "raise" causes
        a ValueError to be raised; "drop" causes the missing sequences to be
        dropped from the returned set.

    germLengthLimits : dict, optional
        A dictionary limiting the max-length values used for specific germs.
        Keys are germ sequences and values are integers.  For example, if
        this argument is `{('Gx',): 4}` and `maxLengthList = [1,2,4,8,16]`,
        then the germ `('Gx',)` is only repeated using max-lengths of 1, 2,
        and 4 (whereas other germs use all the values in `maxLengthList`).

    verbosity : int, optional
        The level of output to print to stdout.


    Returns
    -------
    list of LsGermsStructure objects
        The i-th object corresponds to a gate string list containing repeated
        germs limited to length maxLengthList[i].  If nest == True, then
        repeated germs limited to previous max-lengths are also included.
        Note that a "0" maximum-length corresponds to the LGST strings.
    """

    printer = _VerbosityPrinter.build_printer(verbosity)
    if germLengthLimits is None: germLengthLimits = {}

    if nest == True and includeLGST == True and len(maxLengthList) > 0 and maxLengthList[0] == 0:
        _warnings.warn("Setting the first element of a max-length list to zero"
                       + " to ensure the inclusion of LGST sequences has been"
                       + " replaced by the `includeLGST` parameter which"
                       + " defaults to `True`.  Thus, in most cases, you can"
                       + " simply remove the leading 0 and start your"
                       + " max-length list at 1 now."
                       + "")

    if isinstance(gateLabelSrc, _GateSet):
        gateLabels = list(gateLabelSrc.gates.keys()) + \
                     list(gateLabelSrc.instruments.keys())
    else: gateLabels = gateLabelSrc

    lgst_list = _gsc.list_lgst_gatestrings(prepStrs, effectStrs, gateLabels)

    allPossiblePairs = list(_itertools.product(range(len(prepStrs)),
                                               range(len(effectStrs))))

    if keepFraction < 1.0:
        rndm = _rndm.RandomState(keepSeed) # ok if seed is None
        nPairs = len(prepStrs)*len(effectStrs)
        nPairsToKeep = int(round(float(keepFraction) * nPairs))
    else: rndm = None

    if isinstance(fidPairs, dict) or hasattr(fidPairs, "keys"):
        fidPairDict = fidPairs #assume a dict of per-germ pairs
    else:
        if fidPairs is not None:   #assume fidPairs is a list
            fidPairDict = { germ:fidPairs for germ in germList }
        else:
            fidPairDict = None

    truncFn = _getTruncFunction(truncScheme)

    empty_germ = _GateString( (), "{}" )
    if includeLGST: germList = [empty_germ] + germList

    #running structure of all strings so far (LGST strings or empty)
    running_gss = _LsGermsStructure([],germList,prepStrs,
                                    effectStrs,gateLabelAliases,
                                    sequenceRules)

    missing_lgst = []

    if includeLGST and len(maxLengthList) == 0:
        #Add *all* LGST sequences as unstructured if we don't add them below
        missing_lgst = running_gss.add_unindexed(lgst_list, dscheck)

    lsgst_listOfStructs = [ ] # list of gate string structures to return
    missing_list = []
    totStrs = len(running_gss.allstrs)

    for i,maxLen in enumerate(maxLengthList):

        if nest: #add to running_gss and copy at end
            gss = running_gss #don't copy (yet)
            gss.Ls.append(maxLen)
        else: #create a new gss for just this maxLen
            gss = _LsGermsStructure([maxLen],germList,prepStrs,
                                    effectStrs,gateLabelAliases,
                                    sequenceRules)
        if maxLen == 0:
            #Special LGST case
            missing_lgst = gss.add_unindexed(lgst_list, dscheck)
        else:
            if includeLGST and i == 0: #first maxlen, so add LGST seqs as empty germ
                #Note: no FPR on LGST strings
                missing_list.extend( gss.add_plaquette(empty_germ, maxLen, empty_germ,
                                                       allPossiblePairs, dscheck) )
                missing_lgst = gss.add_unindexed(lgst_list, dscheck) # only adds those not already present

            #Typical case of germs repeated to maxLen using Rfn
            for germ in germList:
                if germ == empty_germ: continue #handled specially above
                if maxLen > germLengthLimits.get(germ,1e100): continue
                germ_power = truncFn(germ,maxLen)

                if rndm is None:
                    if fidPairDict is not None:
                        fiducialPairsThisIter = fidPairDict.get(
                            germ,allPossiblePairs)
                    else:
                        fiducialPairsThisIter = allPossiblePairs

                elif fidPairDict is not None:
                    pair_indx_tups = fidPairDict.get(germ,allPossiblePairs)
                    remainingPairs = [ (i,j)
                                       for i in range(len(prepStrs))
                                       for j in range(len(effectStrs))
                                       if (i,j) not in pair_indx_tups ]
                    nPairsRemaining = len(remainingPairs)
                    nPairsToChoose = nPairsToKeep-len(pair_indx_tups)
                    nPairsToChoose = max(0,min(nPairsToChoose,nPairsRemaining))
                    assert(0 <= nPairsToChoose <= nPairsRemaining)
                    # FUTURE: issue warnings when clipping nPairsToChoose?

                    fiducialPairsThisIter = fidPairDict[germ] + \
                        [ remainingPairs[k] for k in
                          sorted(rndm.choice(nPairsRemaining,nPairsToChoose,
                                             replace=False))]

                else: # rndm is not None and fidPairDict is None
                    assert(nPairsToKeep <= nPairs) # keepFraction must be <= 1.0
                    fiducialPairsThisIter = \
                        [ allPossiblePairs[k] for k in
                          sorted(rndm.choice(nPairs,nPairsToKeep,replace=False))]

                missing_list.extend( gss.add_plaquette(germ_power, maxLen, germ,
                                                       fiducialPairsThisIter, dscheck) )

        if nest: gss = gss.copy() #pinch off a copy of running_gss
        gss.done_adding_strings()
        lsgst_listOfStructs.append( gss )
        totStrs += len(gss.allstrs) #only relevant for non-nested case

    if nest: #then totStrs computation about overcounts -- just take string count of final stage
        totStrs = len(running_gss.allstrs)

    printer.log("--- Gate Sequence Creation ---", 1)
    printer.log(" %d sequences created" % totStrs,2)
    if dscheck:
        printer.log(" Dataset has %d entries: %d utilized, %d requested sequences were missing"
                    % (len(dscheck), totStrs, len(missing_list)), 2)
    if len(missing_list) > 0 or len(missing_lgst) > 0:
        missing_msgs = ["Prep: %s, Germ: %s, L: %d, Meas: %s, Seq: %s" % tup
                        for tup in missing_list] + \
                       ["LGST Seq: %s" % gstr for gstr in missing_lgst ]
        printer.log("The following sequences were missing from the dataset:",4)
        printer.log("\n".join(missing_msgs), 4)
        if actionIfMissing == "raise":
            raise ValueError("Missing data! %d missing gate sequences" % len(missing_msgs))
        elif actionIfMissing == "drop":
            pass
        else:
            raise ValueError("Invalid `actionIfMissing` argument: %s" % actionIfMissing)


    for i,struct in enumerate(lsgst_listOfStructs):
        if nest:
            assert(struct.Ls == maxLengthList[0:i+1]) #Make sure lengths are correct!
        else:
            assert(struct.Ls == maxLengthList[i:i+1]) #Make sure lengths are correct!
    return lsgst_listOfStructs


def make_lsgst_experiment_list(gateLabelSrc, prepStrs, effectStrs, germList,
                               maxLengthList, fidPairs=None,
                               truncScheme="whole germ powers", keepFraction=1,
                               keepSeed=None, includeLGST=True):
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
    gateLabelSrc : list or GateSet
        List of gate labels to determine needed LGST strings.  If a GateSet,
        then the gate set's gate and instrument labels are used. Only
        relevant when `includeLGST == True`.

    prepStrs : list of GateStrings
        List of the preparation fiducial gate strings, which follow state
        preparation.

    effectStrs : list of GateStrings
        List of the measurement fiducial gate strings, which precede
        measurement.

    germList : list of GateStrings
        List of the germ gate strings.

    maxLengthList : list of ints
        List of maximum lengths.

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

    includeLGST : boolean, optional
        If true, then ensure that LGST sequences are included in the
        returned list.


    Returns
    -------
    list of GateStrings
    """
    nest = True # => the final list contains all of the strings
    return make_lsgst_lists(gateLabelSrc, prepStrs, effectStrs, germList,
                            maxLengthList, fidPairs, truncScheme, nest,
                            keepFraction, keepSeed, includeLGST)[-1]



def make_elgst_lists(gateLabelSrc, germList, maxLengthList,
                     truncScheme="whole germ powers", nest=True,
                     includeLGST=True):
    """
    Create a set of gate string lists for eLGST based on germs and max-lengths

    Constructs a series (a list) of gate string lists used by the extended LGST
    (eLGST) algorithm.  If `includeLGST == True` then the starting list is the
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
    gateLabelSrc : list or GateSet
        List of gate labels to determine needed LGST strings.  If a GateSet,
        then the gate set's gate and instrument labels are used. Only
        relevant when `includeLGST == True`.

    germList : list of GateStrings
        List of the germ gate strings.

    maxLengthList : list of ints
        List of maximum lengths. A zero value in this list has special
        meaning, and corresponds to the length-1 gate label strings.

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
        If True, the returned gate string lists are "nested", meaning
        that each successive list of gate strings contains all the gate
        strings found in previous lists (and usually some additional
        new ones).  If False, then the returned string list for maximum
        length == L contains *only* those gate strings specified in the
        description above, and *not* those for previous values of L.

    includeLGST : boolean, optional
        If true, then the starting list (only applicable when
        `nest == True`) is the list of length-1 gate label strings
        rather than the empty list.  This means that when
        `nest == True`, the length-1 sequences will be included in all
        the lists.


    Returns
    -------
    list of (lists of GateStrings)
        The i-th list corresponds to a gate string list containing repeated
        germs limited to length maxLengthList[i].  If nest == True, then
        repeated germs limited to previous max-lengths are also included.
        Note that a "0" maximum-length corresponds to the gate
        label strings.
    """
    if isinstance(gateLabelSrc, _GateSet):
        gateLabels = list(gateLabelSrc.gates.keys()) + \
                     list(gateLabelSrc.instruments.keys())
    else: gateLabels = gateLabelSrc

    singleGates = _gsc.gatestring_list([(g,) for g in gateLabels])

    #running list of all strings so far (length-1 strs or empty)
    elgst_list = singleGates[:] if includeLGST else _gsc.gatestring_list([()])
    elgst_listOfLists = [ ] # list of lists to return

    Rfn = _getTruncFunction(truncScheme)

    for maxLen in maxLengthList:
        if maxLen == 0:
            #Special length-1 string case
            lst = singleGates[:]
        else:
            #Typical case of germs repeated to maxLen using Rfn
            lst = _gsc.create_gatestring_list("R(germ,N)", germ=germList, N=maxLen, R=Rfn)

        if nest:
            elgst_list += lst #add new strings to running list
            elgst_listOfLists.append( _lt.remove_duplicates(singleGates + elgst_list) )
        else:
            elgst_listOfLists.append( _lt.remove_duplicates(lst) )

    #print "%d eLGST sets w/lengths" % len(elgst_listOfLists),map(len,elgst_listOfLists)
    return elgst_listOfLists


def make_elgst_experiment_list(gateLabelSrc, germList, maxLengthList,
                               truncScheme="whole germ powers",
                               includeLGST=True):
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
    gateLabelSrc : list or GateSet
        List of gate labels to determine needed LGST strings.  If a GateSet,
        then the gate set's gate and instrument labels are used. Only
        relevant when `includeLGST == True`.

    germList : list of GateStrings
        List of the germ gate strings.

    maxLengthList : list of ints
        List of maximum lengths. A zero value in this list has special
        meaning, and corresponds to the length-1 gate label strings.

    truncScheme : str, optional
        Truncation scheme used to interpret what the list of maximum lengths
        means. If unsure, leave as default. Allowed values are:

        - 'whole germ powers' -- germs are repeated an integer number of
          times such that the length is less than or equal to the max.
        - 'truncated germ powers' -- repeated germ string is truncated
          to be exactly equal to the max (partial germ at end is ok).
        - 'length as exponent' -- max. length is instead interpreted
          as the germ exponent (the number of germ repetitions).

    includeLGST : boolean, optional
        If true, then ensure that length-1 sequences are included in
        the returned list.


    Returns
    -------
    list of GateStrings
    """

    #When nest == True the final list contains all of the strings
    nest = True
    return make_elgst_lists(gateLabelSrc, germList,
                            maxLengthList, truncScheme, nest,
                            includeLGST)[-1]



def _getTruncFunction(truncScheme):
    if truncScheme == "whole germ powers":
        Rfn = _gsc.repeat_with_max_length
    elif truncScheme == "truncated germ powers":
        Rfn = _gsc.repeat_and_truncate
    elif truncScheme == "length as exponent":
        Rfn = lambda germ,N : germ*N
    else:
        raise ValueError("Invalid truncation scheme: %s" % truncScheme)
    return Rfn
