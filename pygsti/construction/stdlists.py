"""
Circuit list creation functions using repeated-germs limited by a max-length.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy.random as _rndm
import itertools as _itertools
import warnings as _warnings
from ..tools import listtools as _lt
from ..tools.legacytools import deprecated_fn as _deprecated_fn
from ..objects import LsGermsStructure as _LsGermsStructure
from ..objects import Model as _Model
from ..objects import Circuit as _Circuit
from ..objects import BulkCircuitList as _BulkCircuitList
from ..objects.verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from . import circuitconstruction as _gsc


def make_raw_lsgst_lists(op_label_src, prep_strs, effect_strs, germ_list, max_length_list,
                         fid_pairs=None, trunc_scheme="whole germ powers", nest=True,
                         keep_fraction=1, keep_seed=None, include_lgst=True,
                         germ_length_limits=None):
    """
    Create a set of circuit lists for LSGST based on germs and max-lengths.

    Constructs a series (a list) of circuit lists used by long-sequence GST
    (LSGST) algorithms.  If `include_lgst == True` then the starting list is the
    list of LGST strings, otherwise the starting list is empty.  For each
    nonzero element of max_length_list, call it L, a list of circuits is
    created with the form:

    Case: trunc_scheme == 'whole germ powers':
      prepStr + pygsti.construction.repeat_with_max_length(germ,L) + effectStr

    Case: trunc_scheme == 'truncated germ powers':
      prepStr + pygsti.construction.repeat_and_truncate(germ,L) + effectStr

    Case: trunc_scheme == 'length as exponent':
      prepStr + germ^L + effectStr

    If nest == True, the above list is iteratively *added* (w/duplicates
    removed) to the current list of circuits to form a final list for the
    given L.  This results in successively larger lists, each of which
    contains all the elements of previous-L lists.  If nest == False then
    the above list *is* the final list for the given L.

    Parameters
    ----------
    op_label_src : list or Model
        List of operation labels to determine needed LGST strings.  If a Model,
        then the model's gate and instrument labels are used. Only
        relevant when `include_lgst == True`.

    prep_strs : list of Circuits
        List of the preparation fiducial circuits, which follow state
        preparation.

    effect_strs : list of Circuits
        List of the measurement fiducial circuits, which precede
        measurement.

    germ_list : list of Circuits
        List of the germ circuits.

    max_length_list : list of ints
        List of maximum lengths. A zero value in this list has special
        meaning, and corresponds to the LGST sequences.

    fid_pairs : list of 2-tuples or dict, optional
        Specifies a subset of all fiducial string pairs (prepStr, effectStr)
        to be used in the circuit lists.  If a list, each element of
        fid_pairs is a (iPrepStr, iEffectStr) 2-tuple of integers, each
        indexing a string within prep_strs and effect_strs, respectively, so
        that prepStr = prep_strs[iPrepStr] and effectStr =
        effect_strs[iEffectStr].  If a dictionary, keys are germs (elements
        of germ_list) and values are lists of 2-tuples specifying the pairs
        to use for that germ.

    trunc_scheme : str, optional
        Truncation scheme used to interpret what the list of maximum lengths
        means. If unsure, leave as default. Allowed values are:

        - 'whole germ powers' -- germs are repeated an integer number of
          times such that the length is less than or equal to the max.
        - 'truncated germ powers' -- repeated germ string is truncated
          to be exactly equal to the max (partial germ at end is ok).
        - 'length as exponent' -- max. length is instead interpreted
          as the germ exponent (the number of germ repetitions).

    nest : boolean, optional
        If True, the returned circuits lists are "nested", meaning
        that each successive list of circuits contains all the gate
        strings found in previous lists (and usually some additional
        new ones).  If False, then the returned circuit list for maximum
        length == L contains *only* those circuits specified in the
        description above, and *not* those for previous values of L.

    keep_fraction : float, optional
        The fraction of fiducial pairs selected for each germ-power base
        string.  The default includes all fiducial pairs.  Note that
        for each germ-power the selected pairs are *different* random
        sets of all possible pairs (unlike fid_pairs, which specifies the
        *same* fiducial pairs for *all* same-germ base strings).  If
        fid_pairs is used in conjuction with keep_fraction, the pairs
        specified by fid_pairs are always selected, and any additional
        pairs are randomly selected.

    keep_seed : int, optional
        The seed used for random fiducial pair selection (only relevant
        when keep_fraction < 1).

    include_lgst : boolean, optional
        If true, then the starting list (only applicable when
        `nest == True`) is the list of LGST strings rather than the
        empty list.  This means that when `nest == True`, the LGST
        sequences will be included in all the lists.

    germ_length_limits : dict, optional
        A dictionary limiting the max-length values used for specific germs.
        Keys are germ sequences and values are integers.  For example, if
        this argument is `{('Gx',): 4}` and `max_length_list = [1,2,4,8,16]`,
        then the germ `('Gx',)` is only repeated using max-lengths of 1, 2,
        and 4 (whereas other germs use all the values in `max_length_list`).

    Returns
    -------
    list of (lists of Circuits)
        The i-th list corresponds to a circuit list containing repeated
        germs limited to length max_length_list[i].  If nest == True, then
        repeated germs limited to previous max-lengths are also included.
        Note that a "0" maximum-length corresponds to the LGST strings.
    """
    if germ_length_limits is None: germ_length_limits = {}
    if nest and include_lgst and len(max_length_list) > 0 and max_length_list[0] == 0:
        _warnings.warn("Setting the first element of a max-length list to zero"
                       + " to ensure the inclusion of LGST sequences has been"
                       + " replaced by the `include_lgst` parameter which"
                       + " defaults to `True`.  Thus, in most cases, you can"
                       + " simply remove the leading 0 and start your"
                       + " max-length list at 1 now."
                       + "")

    if isinstance(op_label_src, _Model):
        opLabels = op_label_src.get_primitive_op_labels() + op_label_src.get_primitive_instrument_labels()
    else: opLabels = op_label_src

    lgst_list = _gsc.list_lgst_circuits(prep_strs, effect_strs, opLabels)

    if keep_fraction < 1.0:
        rndm = _rndm.RandomState(keep_seed)  # ok if seed is None
        nPairs = len(prep_strs) * len(effect_strs)
        nPairsToKeep = int(round(float(keep_fraction) * nPairs))
    else: rndm = None

    if isinstance(fid_pairs, dict) or hasattr(fid_pairs, "keys"):
        fiducialPairs = {germ: [(prep_strs[i], effect_strs[j])
                                for (i, j) in fid_pairs[germ]]
                         for germ in germ_list}
        fidPairDict = fid_pairs
    else:
        if fid_pairs is not None:  # assume fid_pairs is a list
            fidPairDict = {germ: fid_pairs for germ in germ_list}
            lst = [(prep_strs[i], effect_strs[j]) for (i, j) in fid_pairs]
        else:
            fidPairDict = None
            lst = list(_itertools.product(prep_strs, effect_strs))
        fiducialPairs = {germ: lst for germ in germ_list}

    #running list of all strings so far (LGST strings or empty)
    lsgst_list = lgst_list[:] if include_lgst else _gsc.circuit_list([()])
    lsgst_listOfLists = []  # list of lists to return

    Rfn = _get_trunc_function(trunc_scheme)

    for maxLen in max_length_list:

        lst = []
        if maxLen == 0:
            #Special LGST case
            lst += lgst_list[:]
        else:
            #Typical case of germs repeated to maxLen using Rfn
            for germ in germ_list:
                if maxLen > germ_length_limits.get(germ, 1e100): continue

                if rndm is None:
                    fiducialPairsThisIter = fiducialPairs[germ]

                elif fidPairDict is not None:
                    pair_indx_tups = fidPairDict[germ]
                    remainingPairs = [(prep_strs[i], effect_strs[j])
                                      for i in range(len(prep_strs))
                                      for j in range(len(effect_strs))
                                      if (i, j) not in pair_indx_tups]
                    nPairsRemaining = len(remainingPairs)
                    nPairsToChoose = nPairsToKeep - len(pair_indx_tups)
                    nPairsToChoose = max(0, min(nPairsToChoose, nPairsRemaining))
                    assert(0 <= nPairsToChoose <= nPairsRemaining)
                    # FUTURE: issue warnings when clipping nPairsToChoose?

                    fiducialPairsThisIter = fiducialPairs[germ] + \
                        [remainingPairs[k] for k in
                         sorted(rndm.choice(nPairsRemaining, nPairsToChoose,
                                            replace=False))]

                else:  # rndm is not None and fidPairDict is None
                    assert(nPairsToKeep <= nPairs)  # keep_fraction must be <= 1.0
                    fiducialPairsThisIter = \
                        [fiducialPairs[germ][k] for k in
                         sorted(rndm.choice(nPairs, nPairsToKeep, replace=False))]

                lst += _gsc.create_circuit_list("f[0]+R(germ,N)+f[1]",
                                                f=fiducialPairsThisIter,
                                                germ=germ, N=maxLen,
                                                R=Rfn, order=('f',))
        if nest:
            lsgst_list += lst  # add new strings to running list
            lsgst_listOfLists.append(_lt.remove_duplicates(lsgst_list))
        else:
            lsgst_listOfLists.append(_lt.remove_duplicates(lst))

    #print "%d LSGST sets w/lengths" % len(lsgst_listOfLists),map(len,lsgst_listOfLists)
    return lsgst_listOfLists


@_deprecated_fn('make_lsgst_lists(...)')
def make_lsgst_structs(op_label_src, prep_strs, effect_strs, germ_list, max_length_list,
                       fid_pairs=None, trunc_scheme="whole germ powers", nest=True,
                       keep_fraction=1, keep_seed=None, include_lgst=True,
                       op_label_aliases=None, sequence_rules=None,
                       dscheck=None, action_if_missing="raise", germ_length_limits=None,
                       verbosity=0):
    """
    Deprecated function.
    """
    bulk_circuit_lists = make_lsgst_lists(op_label_src, prep_strs, effect_strs, germ_list, max_length_list,
                                          fid_pairs, trunc_scheme, nest,
                                          keep_fraction, keep_seed, include_lgst,
                                          op_label_aliases, sequence_rules,
                                          dscheck, action_if_missing, germ_length_limits, verbosity)
    return [bcl.circuit_structure for bcl in bulk_circuit_lists]


def make_lsgst_lists(op_label_src, prep_strs, effect_strs, germ_list, max_length_list,
                     fid_pairs=None, trunc_scheme="whole germ powers", nest=True,
                     keep_fraction=1, keep_seed=None, include_lgst=True,
                     op_label_aliases=None, sequence_rules=None,
                     dscheck=None, action_if_missing="raise", germ_length_limits=None,
                     verbosity=0):
    """
    Create a set of long-sequence GST circuit lists (including structure).

    Constructs a series (a list) of circuit structures used by long-sequence
    GST (LSGST) algorithms.  If `include_lgst == True` then the starting
    structure contains the LGST strings, otherwise the starting structure is
    empty.  For each nonzero element of max_length_list, call it L, a set of
    circuits is created with the form:

    Case: trunc_scheme == 'whole germ powers':
      prepStr + pygsti.construction.repeat_with_max_length(germ,L) + effectStr

    Case: trunc_scheme == 'truncated germ powers':
      prepStr + pygsti.construction.repeat_and_truncate(germ,L) + effectStr

    Case: trunc_scheme == 'length as exponent':
      prepStr + germ^L + effectStr

    If nest == True, the above set is iteratively *added* (w/duplicates
    removed) to the current circuit structure to form a final structure for
    the given L.  This results in successively larger structures, each of which
    contains all the elements of previous-L structures.  If nest == False then
    the above set *is* the final structure for the given L.

    Parameters
    ----------
    op_label_src : list or Model
        List of operation labels to determine needed LGST strings.  If a Model,
        then the model's gate and instrument labels are used. Only
        relevant when `include_lgst == True`.

    prep_strs : list of Circuits
        List of the preparation fiducial circuits, which follow state
        preparation.

    effect_strs : list of Circuits
        List of the measurement fiducial circuits, which precede
        measurement.

    germ_list : list of Circuits
        List of the germ circuits.

    max_length_list : list of ints
        List of maximum lengths. A zero value in this list has special
        meaning, and corresponds to the LGST sequences.

    fid_pairs : list of 2-tuples or dict, optional
        Specifies a subset of all fiducial string pairs (prepStr, effectStr)
        to be used in the circuit lists.  If a list, each element of
        fid_pairs is a (iPrepStr, iEffectStr) 2-tuple of integers, each
        indexing a string within prep_strs and effect_strs, respectively, so
        that prepStr = prep_strs[iPrepStr] and effectStr =
        effect_strs[iEffectStr].  If a dictionary, keys are germs (elements
        of germ_list) and values are lists of 2-tuples specifying the pairs
        to use for that germ.

    trunc_scheme : str, optional
        Truncation scheme used to interpret what the list of maximum lengths
        means. If unsure, leave as default. Allowed values are:

        - 'whole germ powers' -- germs are repeated an integer number of
          times such that the length is less than or equal to the max.
        - 'truncated germ powers' -- repeated germ string is truncated
          to be exactly equal to the max (partial germ at end is ok).
        - 'length as exponent' -- max. length is instead interpreted
          as the germ exponent (the number of germ repetitions).

    nest : boolean, optional
        If True, the returned circuit lists are "nested", meaning
        that each successive list of circuits contains all the gate
        strings found in previous lists (and usually some additional
        new ones).  If False, then the returned string list for maximum
        length == L contains *only* those circuits specified in the
        description above, and *not* those for previous values of L.

    keep_fraction : float, optional
        The fraction of fiducial pairs selected for each germ-power base
        string.  The default includes all fiducial pairs.  Note that
        for each germ-power the selected pairs are *different* random
        sets of all possible pairs (unlike fid_pairs, which specifies the
        *same* fiducial pairs for *all* same-germ base strings).  If
        fid_pairs is used in conjuction with keep_fraction, the pairs
        specified by fid_pairs are always selected, and any additional
        pairs are randomly selected.

    keep_seed : int, optional
        The seed used for random fiducial pair selection (only relevant
        when keep_fraction < 1).

    include_lgst : boolean, optional
        If true, then the starting list (only applicable when
        `nest == True`) is the list of LGST strings rather than the
        empty list.  This means that when `nest == True`, the LGST
        sequences will be included in all the lists.

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset.  This information is stored within the returned circuit
        structures.  Defaults to the empty dictionary (no aliases defined)
        e.g. op_label_aliases['Gx^3'] = ('Gx','Gx','Gx')

    sequence_rules : list, optional
        A list of `(find,replace)` 2-tuples which specify string replacement
        rules.  Both `find` and `replace` are tuples of operation labels
        (or `Circuit` objects).

    dscheck : DataSet, optional
        A data set which is checked for each of the generated circuits. When
        a generated sequence is missing from this `DataSet`, action is taken
        according to `action_if_missing`.

    action_if_missing : {"raise","drop"}, optional
        The action to take when a generated gate sequence is missing from
        `dscheck` (only relevant when `dscheck` is not None).  "raise" causes
        a ValueError to be raised; "drop" causes the missing sequences to be
        dropped from the returned set.

    germ_length_limits : dict, optional
        A dictionary limiting the max-length values used for specific germs.
        Keys are germ sequences and values are integers.  For example, if
        this argument is `{('Gx',): 4}` and `max_length_list = [1,2,4,8,16]`,
        then the germ `('Gx',)` is only repeated using max-lengths of 1, 2,
        and 4 (whereas other germs use all the values in `max_length_list`).

    verbosity : int, optional
        The level of output to print to stdout.

    Returns
    -------
    list of LsGermsStructure objects
        The i-th object corresponds to a circuit list containing repeated
        germs limited to length max_length_list[i].  If nest == True, then
        repeated germs limited to previous max-lengths are also included.
        Note that a "0" maximum-length corresponds to the LGST strings.
    """

    printer = _VerbosityPrinter.build_printer(verbosity)
    if germ_length_limits is None: germ_length_limits = {}

    if nest and include_lgst and len(max_length_list) > 0 and max_length_list[0] == 0:
        _warnings.warn("Setting the first element of a max-length list to zero"
                       + " to ensure the inclusion of LGST sequences has been"
                       + " replaced by the `include_lgst` parameter which"
                       + " defaults to `True`.  Thus, in most cases, you can"
                       + " simply remove the leading 0 and start your"
                       + " max-length list at 1 now."
                       + "")

    if isinstance(op_label_src, _Model):
        opLabels = op_label_src.get_primitive_op_labels() + op_label_src.get_primitive_instrument_labels()
    else: opLabels = op_label_src

    lgst_list = _gsc.list_lgst_circuits(prep_strs, effect_strs, opLabels)

    allPossiblePairs = list(_itertools.product(range(len(prep_strs)),
                                               range(len(effect_strs))))

    if keep_fraction < 1.0:
        rndm = _rndm.RandomState(keep_seed)  # ok if seed is None
        nPairs = len(prep_strs) * len(effect_strs)
        nPairsToKeep = int(round(float(keep_fraction) * nPairs))
    else: rndm = None

    if isinstance(fid_pairs, dict) or hasattr(fid_pairs, "keys"):
        fidPairDict = fid_pairs  # assume a dict of per-germ pairs
    else:
        if fid_pairs is not None:  # assume fid_pairs is a list
            fidPairDict = {germ: fid_pairs for germ in germ_list}
        else:
            fidPairDict = None

    truncFn = _get_trunc_function(trunc_scheme)

    line_labels = germ_list[0].line_labels if len(germ_list) > 0 \
        else (prep_strs + effect_strs)[0].line_labels   # if an empty germ list, base line_labels off of fiducials

    empty_germ = _Circuit((), line_labels, stringrep="{}")
    if include_lgst: germ_list = [empty_germ] + germ_list

    #running structure of all strings so far (LGST strings or empty)
    running_cs = _LsGermsStructure([], germ_list, prep_strs,
                                   effect_strs, op_label_aliases,
                                   sequence_rules)

    missing_lgst = []

    if include_lgst and len(max_length_list) == 0:
        #Add *all* LGST sequences as unstructured if we don't add them below
        missing_lgst = running_cs.add_unindexed(lgst_list, dscheck)

    lsgst_listOfStructs = []  # list of circuit structures to return
    missing_list = []
    totStrs = len(running_cs.allstrs)
    #import time as _time; t0=_time.time() # DEBUG

    for i, maxLen in enumerate(max_length_list):
        #print("Maxlen = ",maxLen, " %.2fs" % (_time.time()-t0)) # DEBUG - and remove import time above
        if nest:  # add to running_cs and copy at end
            cs = running_cs  # don't copy (yet)
            cs.Ls.append(maxLen)
        else:  # create a new cs for just this maxLen
            cs = _LsGermsStructure([maxLen], germ_list, prep_strs,
                                   effect_strs, op_label_aliases,
                                   sequence_rules)
        if maxLen == 0:
            #Special LGST case
            missing_lgst = cs.add_unindexed(lgst_list, dscheck)
        else:
            if include_lgst and i == 0:  # first maxlen, so add LGST seqs as empty germ
                #Note: no FPR on LGST strings
                missing_list.extend(cs.add_plaquette(empty_germ, maxLen, empty_germ,
                                                     allPossiblePairs, dscheck))
                missing_lgst = cs.add_unindexed(lgst_list, dscheck)  # only adds those not already present
                #assert(('Gx','Gi0','Gi0') not in cs.allstrs) # DEBUG

            #Typical case of germs repeated to maxLen using r_fn
            for ii, germ in enumerate(germ_list):
                if germ == empty_germ: continue  # handled specially above
                if maxLen > germ_length_limits.get(germ, 1e100): continue
                germ_power = truncFn(germ, maxLen)

                if rndm is None:
                    if fidPairDict is not None:
                        fiducialPairsThisIter = fidPairDict.get(
                            germ, allPossiblePairs)
                    else:
                        fiducialPairsThisIter = allPossiblePairs

                elif fidPairDict is not None:
                    pair_indx_tups = fidPairDict.get(germ, allPossiblePairs)
                    remainingPairs = [(i, j)
                                      for i in range(len(prep_strs))
                                      for j in range(len(effect_strs))
                                      if (i, j) not in pair_indx_tups]
                    nPairsRemaining = len(remainingPairs)
                    nPairsToChoose = nPairsToKeep - len(pair_indx_tups)
                    nPairsToChoose = max(0, min(nPairsToChoose, nPairsRemaining))
                    assert(0 <= nPairsToChoose <= nPairsRemaining)
                    # FUTURE: issue warnings when clipping nPairsToChoose?

                    fiducialPairsThisIter = fidPairDict[germ] + \
                        [remainingPairs[k] for k in
                         sorted(rndm.choice(nPairsRemaining, nPairsToChoose,
                                            replace=False))]

                else:  # rndm is not None and fidPairDict is None
                    assert(nPairsToKeep <= nPairs)  # keep_fraction must be <= 1.0
                    fiducialPairsThisIter = \
                        [allPossiblePairs[k] for k in
                         sorted(rndm.choice(nPairs, nPairsToKeep, replace=False))]

                missing_list.extend(cs.add_plaquette(germ_power, maxLen, germ,
                                                     fiducialPairsThisIter, dscheck))

        if nest: cs = cs.copy()  # pinch off a copy of running_cs
        cs.done_adding_strings()
        lsgst_listOfStructs.append(cs)
        totStrs += len(cs.allstrs)  # only relevant for non-nested case

    if nest:  # then totStrs computation about overcounts -- just take string count of final stage
        totStrs = len(running_cs.allstrs)

    printer.log("--- Circuit Creation ---", 1)
    printer.log(" %d sequences created" % totStrs, 2)
    if dscheck:
        printer.log(" Dataset has %d entries: %d utilized, %d requested sequences were missing"
                    % (len(dscheck), totStrs, len(missing_list)), 2)
    if len(missing_list) > 0 or len(missing_lgst) > 0:
        MAX = 10  # Maximum missing-seq messages to display
        missing_msgs = ["Prep: %s, Germ: %s, L: %d, Meas: %s, Seq: %s" % tup
                        for tup in missing_list[0:MAX + 1]] + \
                       ["LGST Seq: %s" % opstr for opstr in missing_lgst[0:MAX + 1]]
        if len(missing_list) > MAX or len(missing_lgst) > MAX:
            missing_msgs.append(" ... (more missing sequences not show) ... ")
        printer.log("The following sequences were missing from the dataset:", 4)
        printer.log("\n".join(missing_msgs), 4)
        if action_if_missing == "raise":
            raise ValueError("Missing data! %d missing circuits" % len(missing_msgs))
        elif action_if_missing == "drop":
            pass
        else:
            raise ValueError("Invalid `action_if_missing` argument: %s" % action_if_missing)

    for i, struct in enumerate(lsgst_listOfStructs):
        if nest:
            assert(struct.Ls == max_length_list[0:i + 1])  # Make sure lengths are correct!
        else:
            assert(struct.Ls == max_length_list[i:i + 1])  # Make sure lengths are correct!

    #Turn circuit structures into BulkCircuitList objects
    bulk_circuit_lists = [_BulkCircuitList(cs, op_label_aliases) for cs in lsgst_listOfStructs]
    return bulk_circuit_lists


def make_lsgst_experiment_list(op_label_src, prep_strs, effect_strs, germ_list,
                               max_length_list, fid_pairs=None,
                               trunc_scheme="whole germ powers", keep_fraction=1,
                               keep_seed=None, include_lgst=True):
    """
    List all the circuits (i.e. experiments) required for long-sequence GST (LSGST).

    Returns a single list containing, without duplicates, all the gate
    strings required throughout all the iterations of LSGST given by
    max_length_list.  Thus, the returned list is equivalently the list of
    the experiments required to run LSGST using the supplied parameters,
    and so commonly used when construting data set templates or simulated
    data sets.  The breakdown of which circuits are used for which
    iteration(s) of LSGST is given by make_lsgst_lists(...).

    Parameters
    ----------
    op_label_src : list or Model
        List of operation labels to determine needed LGST strings.  If a Model,
        then the model's gate and instrument labels are used. Only
        relevant when `include_lgst == True`.

    prep_strs : list of Circuits
        List of the preparation fiducial circuits, which follow state preparation.

    effect_strs : list of Circuits
        List of the measurement fiducial circuits, which precede  measurement.

    germ_list : list of Circuits
        List of the germ circuits.

    max_length_list : list of ints
        List of maximum lengths.

    fid_pairs : list of 2-tuples, optional
        Specifies a subset of all fiducial string pairs (prepStr, effectStr)
        to be used in the circuit lists.  If a list, each element of
        fid_pairs is a (iPrepStr, iEffectStr) 2-tuple of integers, each
        indexing a string within prep_strs and effect_strs, respectively, so
        that prepStr = prep_strs[iPrepStr] and effectStr =
        effect_strs[iEffectStr].  If a dictionary, keys are germs (elements
        of germ_list) and values are lists of 2-tuples specifying the pairs
        to use for that germ.

    trunc_scheme : str, optional
        Truncation scheme used to interpret what the list of maximum lengths
        means. If unsure, leave as default. Allowed values are:

        - 'whole germ powers' -- germs are repeated an integer number of
          times such that the length is less than or equal to the max.
        - 'truncated germ powers' -- repeated germ string is truncated
          to be exactly equal to the max (partial germ at end is ok).
        - 'length as exponent' -- max. length is instead interpreted
          as the germ exponent (the number of germ repetitions).

    keep_fraction : float, optional
        The fraction of fiducial pairs selected for each germ-power base
        string.  The default includes all fiducial pairs.  Note that
        for each germ-power the selected pairs are *different* random
        sets of all possible pairs (unlike fid_pairs, which specifies the
        *same* fiduicial pairs for *all* same-germ base strings).  If
        fid_pairs is used in conjuction with keep_fraction, the pairs
        specified by fid_pairs are always selected, and any additional
        pairs are randomly selected.

    keep_seed : int, optional
        The seed used for random fiducial pair selection (only relevant
        when keep_fraction < 1).

    include_lgst : boolean, optional
        If true, then ensure that LGST sequences are included in the
        returned list.

    Returns
    -------
    list of Circuits
    """
    nest = True  # => the final list contains all of the strings
    return make_lsgst_lists(op_label_src, prep_strs, effect_strs, germ_list,
                            max_length_list, fid_pairs, trunc_scheme, nest,
                            keep_fraction, keep_seed, include_lgst)[-1]


@_deprecated_fn('ELGST is not longer implemented in pyGSTi.')
def make_elgst_lists(op_label_src, germ_list, max_length_list,
                     trunc_scheme="whole germ powers", nest=True,
                     include_lgst=True):
    """
    Create a set of circuit lists for eLGST based on germs and max-lengths

    Constructs a series (a list) of circuit lists used by the extended LGST
    (eLGST) algorithm.  If `include_lgst == True` then the starting list is the
    list of length-1 operation label strings, otherwise the starting list is empty.
    For each nonzero element of max_length_list, call it L, a list of circuits is
    created with the form:

    Case: trunc_scheme == 'whole germ powers':
      pygsti.construction.repeat_with_max_length(germ,L)

    Case: trunc_scheme == 'truncated germ powers':
      pygsti.construction.repeat_and_truncate(germ,L)

    Case: trunc_scheme == 'length as exponent':
      germ^L

    If nest == True, the above list is iteratively *added* (w/duplicates
    removed) to the current list of circuits to form a final list for the
    given L.  This results in successively larger lists, each of which
    contains all the elements of previous-L lists.  If nest == False then
    the above list *is* the final list for the given L.

    Parameters
    ----------
    op_label_src : list or Model
        List of operation labels to determine needed LGST strings.  If a Model,
        then the model's gate and instrument labels are used. Only
        relevant when `include_lgst == True`.

    germ_list : list of Circuits
        List of the germ circuits.

    max_length_list : list of ints
        List of maximum lengths. A zero value in this list has special
        meaning, and corresponds to the length-1 operation label strings.

    trunc_scheme : str, optional
        Truncation scheme used to interpret what the list of maximum lengths
        means.  If unsure, leave as default. Allowed values are:

        - 'whole germ powers' -- germs are repeated an integer number of
          times such that the length is less than or equal to the max.
        - 'truncated germ powers' -- repeated germ string is truncated
          to be exactly equal to the max (partial germ at end is ok).
        - 'length as exponent' -- max. length is instead interpreted
          as the germ exponent (the number of germ repetitions).

    nest : boolean, optional
        If True, the returned circuit lists are "nested", meaning
        that each successive list of circuits contains all the gate
        strings found in previous lists (and usually some additional
        new ones).  If False, then the returned string list for maximum
        length == L contains *only* those circuits specified in the
        description above, and *not* those for previous values of L.

    include_lgst : boolean, optional
        If true, then the starting list (only applicable when
        `nest == True`) is the list of length-1 operation label strings
        rather than the empty list.  This means that when
        `nest == True`, the length-1 sequences will be included in all
        the lists.

    Returns
    -------
    list of (lists of Circuits)
        The i-th list corresponds to a circuit list containing repeated
        germs limited to length max_length_list[i].  If nest == True, then
        repeated germs limited to previous max-lengths are also included.
        Note that a "0" maximum-length corresponds to the gate
        label strings.
    """
    if isinstance(op_label_src, _Model):
        opLabels = op_label_src.get_primitive_op_labels() + op_label_src.get_primitive_instrument_labels()
    else: opLabels = op_label_src

    singleOps = _gsc.circuit_list([(g,) for g in opLabels])

    #running list of all strings so far (length-1 strs or empty)
    elgst_list = singleOps[:] if include_lgst else _gsc.circuit_list([()])
    elgst_listOfLists = []  # list of lists to return

    Rfn = _get_trunc_function(trunc_scheme)

    for maxLen in max_length_list:
        if maxLen == 0:
            #Special length-1 string case
            lst = singleOps[:]
        else:
            #Typical case of germs repeated to maxLen using Rfn
            lst = _gsc.create_circuit_list("R(germ,N)", germ=germ_list, N=maxLen, R=Rfn)

        if nest:
            elgst_list += lst  # add new strings to running list
            elgst_listOfLists.append(_lt.remove_duplicates(singleOps + elgst_list))
        else:
            elgst_listOfLists.append(_lt.remove_duplicates(lst))

    #print "%d eLGST sets w/lengths" % len(elgst_listOfLists),map(len,elgst_listOfLists)
    return elgst_listOfLists


@_deprecated_fn('ELGST is not longer implemented in pyGSTi.')
def make_elgst_experiment_list(op_label_src, germ_list, max_length_list,
                               trunc_scheme="whole germ powers",
                               include_lgst=True):
    """
    List of all the circuits (i.e. experiments) required for extended LGST (eLGST).

    Returns a single list containing, without duplicates, all the gate
    strings required throughout all the iterations of eLGST given by
    max_length_list.  Thus, the returned list is equivalently the list of
    the experiments required to run eLGST using the supplied parameters,
    and so commonly used when construting data set templates or simulated
    data sets.  The breakdown of which circuits are used for which
    iteration(s) of eLGST is given by make_elgst_lists(...).

    Parameters
    ----------
    op_label_src : list or Model
        List of operation labels to determine needed LGST strings.  If a Model,
        then the model's gate and instrument labels are used. Only
        relevant when `include_lgst == True`.

    germ_list : list of Circuits
        List of the germ circuits.

    max_length_list : list of ints
        List of maximum lengths. A zero value in this list has special
        meaning, and corresponds to the length-1 operation label strings.

    trunc_scheme : str, optional
        Truncation scheme used to interpret what the list of maximum lengths
        means. If unsure, leave as default. Allowed values are:

        - 'whole germ powers' -- germs are repeated an integer number of
          times such that the length is less than or equal to the max.
        - 'truncated germ powers' -- repeated germ string is truncated
          to be exactly equal to the max (partial germ at end is ok).
        - 'length as exponent' -- max. length is instead interpreted
          as the germ exponent (the number of germ repetitions).

    include_lgst : boolean, optional
        If true, then ensure that length-1 sequences are included in
        the returned list.

    Returns
    -------
    list of Circuits
    """

    #When nest == True the final list contains all of the strings
    nest = True
    return make_elgst_lists(op_label_src, germ_list,
                            max_length_list, trunc_scheme, nest,
                            include_lgst)[-1]


def _get_trunc_function(trunc_scheme):
    if trunc_scheme == "whole germ powers":
        r_fn = _gsc.repeat_with_max_length
    elif trunc_scheme == "truncated germ powers":
        r_fn = _gsc.repeat_and_truncate
    elif trunc_scheme == "length as exponent":
        def r_fn(germ, n): return germ * n
    else:
        raise ValueError("Invalid truncation scheme: %s" % trunc_scheme)
    return r_fn
