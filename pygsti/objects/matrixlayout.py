"""
Defines the MatrixEvalTree class which implements an evaluation tree.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from .verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from ..tools import slicetools as _slct
from ..tools import listtools as _lt
from .bulkcircuitlist import BulkCircuitList as _BulkCircuitList
from .distlayout import _DistributableAtom
from .distlayout import DistributableCOPALayout as _DistributableCOPALayout

import numpy as _np
import collections as _collections
import time as _time  # DEBUG TIMERS


def _create_tree(circuits_to_evaluate):
    """
    Note: circuits_to_evaluate can be either a list or an integer-keyed dict (for faster lookups), as we
    only take its length and index it.

    Returns
    -------
    eval_tree : list
        A list of instructions (tuples), where each element contains
        information about evaluating a particular circuit:
        (iDest, iLeft, iRight).
        In particular, eval_tree[iDest] = eval_tree[iLeft] + eval_tree[iRight] as *sequences*
        so that matrix(eval_tree[iDest]) = matrixOf(eval_tree[iRight]) * matrixOf(eval_tree[iLeft])
    """
    #Evaluation tree:
    # A list of instructions (tuples), where each element contains
    #  information about evaluating a particular operation sequence:
    #  (iDest, iLeft, iRight)
    # and the order of the elements specifies the evaluation order.
    # In particular, the evalTree[iDest] = eval_tree[iLeft] + eval_tree[iRight]
    #   so that matrix(evalTree[iDest]) = matrixOf(eval_tree[iRight]) * matrixOf(eval_tree[iLeft])
    eval_tree = []

    #Evaluation dictionary:
    # keys == operation sequences that have been evaluated so far
    # values == index of operation sequence (key) within eval_tree
    evalDict = {}

    #Process circuits in order of length, so that we always place short strings
    # in the right place (otherwise assert stmt below can fail)
    indices_sorted_by_circuit_len = \
        sorted(list(range(len(circuits_to_evaluate))),
               key=lambda i: len(circuits_to_evaluate[i]))

    next_scratch_index = len(circuits_to_evaluate)
    for k in indices_sorted_by_circuit_len:
        circuit = circuits_to_evaluate[k]
        L = len(circuit)

        #Single gate (or zero-gate) computations are assumed to be atomic, and be computed independently.
        #  These labels serve as the initial values, and each operation sequence is assumed to be a tuple of
        #  operation labels.
        if L == 0:
            eval_tree.append((k, None, ()))  # iLeft = None => evaluate iRight as a label
            evalDict[()] = k
            continue

        elif L == 1:
            eval_tree.append((k, None, circuit[0]))  # iLeft = None => evaluate iRight as a label
            evalDict[circuit] = k
            continue

        start = 0; bite = 1
        while start < L:

            #Take a bite out of circuit, starting at `start` that is in evalDict
            for b in range(L - start, 0, -1):
                if circuit[start:start + b] in evalDict:
                    bite = b; break
            else:
                # Can't even take a bite of length 1, so add the next op-label to the tree and take b=1 bite.
                eval_tree.append((next_scratch_index, None, circuit[start]))
                evalDict[circuit[start:start + 1]] = next_scratch_index; next_scratch_index += 1
                bite = 1

            bFinal = bool(start + bite == L)
            #print("DB: start=",start,": found ",circuit[start:start+bite],
            #      " (len=%d) in evalDict" % bite, "(final=%s)" % bFinal)

            if start == 0:  # first in-evalDict bite - no need to add anything to self yet
                iCur = evalDict[circuit[0:bite]]
                #print("DB: taking bite: ", circuit[0:bite], "indx = ",iCur)
                if bFinal:
                    if iCur != k:  # then we have a duplicate final operation sequence
                        iEmptyStr = evalDict.get((), None)
                        assert(iEmptyStr is not None)  # duplicate final strs require
                        # the empty string to be included in the tree too!
                        #assert(self[k] is None)  # make sure we haven't put anything here yet
                        eval_tree.append((k, iCur, iEmptyStr))
                        #self[k] = (iCur, iEmptyStr)  # compute the duplicate using by
                        #self.eval_order.append(k)  # multiplying by the empty string.
            else:
                # add (iCur, iBite)
                assert(circuit[0:start + bite] not in evalDict)
                iBite = evalDict[circuit[start:start + bite]]
                if bFinal:  # place (iCur, iBite) at location k
                    iNew = k
                    evalDict[circuit[0:start + bite]] = iNew
                    #assert(self[iNew] is None)  # make sure we haven't put anything here yet
                    #self[k] = (iCur, iBite)
                    eval_tree.append((k, iCur, iBite))
                else:
                    iNew = next_scratch_index
                    evalDict[circuit[0:start + bite]] = iNew
                    eval_tree.append((iNew, iCur, iBite))
                    next_scratch_index += 1
                    #self.append((iCur, iBite))

                #print("DB: add %s (index %d)" % (str(circuit[0:start+bite]),iNew))
                #self.eval_order.append(iNew)
                iCur = iNew
            start += bite
            #nBites += 1

        #assert(k in self.eval_order or k in self.init_indices)
    return eval_tree


def _walk_subtree(treedict, indx, running_inds):
    running_inds.add(indx)
    (iDest, iLeft, iRight) = treedict[indx]
    if iLeft is not None:
        _walk_subtree(treedict, iLeft, running_inds)
        _walk_subtree(treedict, iRight, running_inds)


def _create_single_item_trees(eval_tree, num_elements):
    # num_elements == number of elements *to evaluate* (can be < len(eval_tree))
    #  Create disjoint set of subtrees generated by single items
    need_to_compute = _np.zeros(len(eval_tree), 'bool')
    need_to_compute[0:num_elements] = True

    treedict = {iDest: (iDest, iLeft, iRight) for iDest, iLeft, iRight in eval_tree}

    singleItemTreeSetList = []  # each element represents a subtree, and
    # is a set of the indices owned by that subtree
    for i in reversed(range(num_elements)):
        if not need_to_compute[i]: continue  # move to the last element
        #of eval_tree that needs to be computed (i.e. is not in a subTree)

        subTreeIndices = set()  # create subtree for uncomputed item
        _walk_subtree(treedict, i, subTreeIndices)
        for k in subTreeIndices:
            need_to_compute[k] = False  # mark all the elements of
            #the new tree as computed

        # Add this single-item-generated tree as a new subtree. Later
        #  we merge and/or split these trees based on constraints.
        singleItemTreeSetList.append(subTreeIndices)
    return singleItemTreeSetList


def _find_splitting(eval_tree, num_elements, max_sub_tree_size, num_sub_trees, verbosity):
    """
    Find a partition of the indices of `circuit_tree` to define a set of sub-trees with the desire properties.

    This is done in order to reduce the maximum size of any tree (useful for
    limiting memory consumption or for using multiple cores).  Must specify
    either max_sub_tree_size or num_sub_trees.

    Parameters
    ----------
    eval_tree : list
        A list of (iDest, iLeft, iRight) instructions.

    num_elements : int
        The number of elements `eval_tree` is meant to compute (this means that any
        tree indices `>= num_elements` are considered "scratch" space.

    max_sub_tree_size : int, optional
        The maximum size (i.e. list length) of each sub-tree.  If the
        original tree is smaller than this size, no splitting will occur.
        If None, then there is no limit.

    num_sub_trees : int, optional
        The maximum size (i.e. list length) of each sub-tree.  If the
        original tree is smaller than this size, no splitting will occur.

    verbosity : int, optional
        How much detail to send to stdout.

    Returns
    -------
    list
        A list of sets of elements to place in sub-trees.
    """
    tm = _time.time()
    printer = _VerbosityPrinter.build_printer(verbosity)

    if (max_sub_tree_size is None and num_sub_trees is None) or \
       (max_sub_tree_size is not None and num_sub_trees is not None):
        raise ValueError("Specify *either* max_sub_tree_size or num_sub_trees")
    if num_sub_trees is not None and num_sub_trees <= 0:
        raise ValueError("num_sub_trees must be > 0!")

    #Don't split at all if it's unnecessary
    if max_sub_tree_size is None or len(eval_tree) < max_sub_tree_size:
        if num_sub_trees is None or num_sub_trees == 1:
            return [set(range(len(eval_tree)))]  # no splitting needed

    #First pass - identify which indices go in which subtree
    #   Part 1: create disjoint set of subtrees generated by single items
    singleItemTreeSetList = _create_single_item_trees(eval_tree, num_elements)
    #each element represents a subtree, and
    # is a set of the indices owned by that subtree
    nSingleItemTrees = len(singleItemTreeSetList)

    printer.log("EvalTree.split created singles in %.0fs" %
                (_time.time() - tm)); tm = _time.time()

    #   Part 2: determine whether we need to split/merge "single" trees
    if num_sub_trees is not None:

        #Merges: find the best merges to perform if any are required
        if nSingleItemTrees > num_sub_trees:

            #Find trees that have least intersection to begin:
            # The goal is to find a set of single-item trees such that
            # none of them intersect much with any other of them.
            #
            # Algorithm:
            #   - start with a set of the one tree that has least
            #       intersection with any other tree.
            #   - iteratively add the tree that has the least intersection
            #       with the trees in the existing set
            iStartingTrees = []

            def get_start_indices(max_intersect):
                """ Builds an initial set of indices by merging single-
                    item trees that don't intersect too much (intersection
                    is less than `max_intersect`.  Returns a list of the
                    single-item tree indices and the final set of indices."""
                starting = [0]  # always start with 0th tree
                startingSet = singleItemTreeSetList[0].copy()
                for i, s in enumerate(singleItemTreeSetList[1:], start=1):
                    if len(startingSet.intersection(s)) <= max_intersect:
                        starting.append(i)
                        startingSet.update(s)
                return starting, startingSet

            left, right = 0, max(map(len, singleItemTreeSetList))
            while left < right:
                mid = (left + right) // 2
                iStartingTrees, startingTreeEls = get_start_indices(mid)
                nStartingTrees = len(iStartingTrees)
                if nStartingTrees < num_sub_trees:
                    left = mid + 1
                elif nStartingTrees > num_sub_trees:
                    right = mid
                else: break  # nStartingTrees == num_sub_trees!

            if len(iStartingTrees) < num_sub_trees:
                iStartingTrees, startingTreeEls = get_start_indices(mid + 1)
            if len(iStartingTrees) > num_sub_trees:
                iStartingTrees = iStartingTrees[0:num_sub_trees]
                startingTreeEls = set()
                for i in iStartingTrees:
                    startingTreeEls.update(singleItemTreeSetList[i])

            printer.log("EvalTree.split fast-found starting trees in %.0fs" %
                        (_time.time() - tm)); tm = _time.time()

            #else:
            #    raise ValueError("Invalid start select method: %s" % start_select_method)

            #Merge all the non-starting trees into the starting trees
            # so that we're left with the desired number of trees
            subTreeSetList = [singleItemTreeSetList[i] for i in iStartingTrees]
            assert(len(subTreeSetList) == num_sub_trees)

            indicesLeft = list(range(nSingleItemTrees))
            for i in iStartingTrees:
                del indicesLeft[indicesLeft.index(i)]

            printer.log("EvalTree.split deleted initial indices in %.0fs" %
                        (_time.time() - tm)); tm = _time.time()

            #merge_method = "fast"
            #Another possible algorith (but slower)
            #if merge_method == "best":
            #    while len(indicesLeft) > 0:
            #        iToMergeInto,_ = min(enumerate(map(len,subTreeSetList)),
            #                             key=lambda x: x[1]) #argmin
            #        setToMergeInto = subTreeSetList[iToMergeInto]
            #        #intersectionSizes = [ len(setToMergeInto.intersection(
            #        #            singleItemTreeSetList[i])) for i in indicesLeft ]
            #        #iMaxIntsct = _np.argmax(intersectionSizes)
            #        iMaxIntsct,_ = max( enumerate( ( len(setToMergeInto.intersection(
            #                            singleItemTreeSetList[i])) for i in indicesLeft )),
            #                          key=lambda x: x[1]) #argmax
            #        setToMerge = singleItemTreeSetList[indicesLeft[iMaxIntsct]]
            #        subTreeSetList[iToMergeInto] = \
            #              subTreeSetList[iToMergeInto].union(setToMerge)
            #        del indicesLeft[iMaxIntsct]
            #
            #elif merge_method == "fast":
            most_at_once = 10
            while len(indicesLeft) > 0:
                iToMergeInto, _ = min(enumerate(map(len, subTreeSetList)),
                                      key=lambda x: x[1])  # argmin
                setToMergeInto = subTreeSetList[iToMergeInto]
                intersectionSizes = sorted([(ii, len(setToMergeInto.intersection(
                    singleItemTreeSetList[i]))) for ii, i in enumerate(indicesLeft)],
                    key=lambda x: x[1], reverse=True)
                toDelete = []
                for i in range(min(most_at_once, len(indicesLeft))):
                    #if len(subTreeSetList[iToMergeInto]) >= desiredLength: break
                    iMaxIntsct, _ = intersectionSizes[i]
                    setToMerge = singleItemTreeSetList[indicesLeft[iMaxIntsct]]
                    subTreeSetList[iToMergeInto].update(setToMerge)
                    toDelete.append(iMaxIntsct)
                for i in sorted(toDelete, reverse=True):
                    del indicesLeft[i]

            #else:
            #    raise ValueError("Invalid merge method: %s" % merge_method)

            assert(len(subTreeSetList) == num_sub_trees)
            printer.log("EvalTree.split merged trees in %.0fs" %
                        (_time.time() - tm)); tm = _time.time()

        #Splits (more subtrees desired than there are single item trees!)
        else:
            #Splits: find the best splits to perform
            #TODO: how to split a tree intelligently -- for now, just do
            # trivial splits by making empty trees.
            subTreeSetList = singleItemTreeSetList[:]
            nSplitsNeeded = num_sub_trees - nSingleItemTrees
            while nSplitsNeeded > 0:
                # LATER...
                # for iSubTree,subTreeSet in enumerate(subTreeSetList):
                subTreeSetList.append([])  # create empty subtree
                nSplitsNeeded -= 1

    else:
        assert(max_sub_tree_size is not None)
        subTreeSetList = []

        #Merges: find the best merges to perform if any are allowed given
        # the maximum tree size
        min_sub_tree_size = max(list(map(len, singleItemTreeSetList)))
        if min_sub_tree_size > max_sub_tree_size:
            raise ValueError("Max. sub tree size (%d) is too low (<%d)!"
                             % (max_sub_tree_size, min_sub_tree_size))

        for singleItemTreeSet in singleItemTreeSetList:
            #See if we should merge this single-item-generated tree with
            # another one or make it a new subtree.
            newTreeSize = len(singleItemTreeSet)
            maxIntersectSize = None; iMaxIntersectSize = None
            for k, existingSubTreeSet in enumerate(subTreeSetList):
                mergedSize = len(existingSubTreeSet) + newTreeSize
                if mergedSize <= max_sub_tree_size:
                    intersectionSize = \
                        len(singleItemTreeSet.intersection(existingSubTreeSet))
                    if maxIntersectSize is None or \
                            maxIntersectSize < intersectionSize:
                        maxIntersectSize = intersectionSize
                        iMaxIntersectSize = k

            if iMaxIntersectSize is not None:
                # then we merge the new tree with this existing set
                subTreeSetList[iMaxIntersectSize] = \
                    subTreeSetList[iMaxIntersectSize].union(singleItemTreeSet)
            else:  # we create a new subtree
                subTreeSetList.append(singleItemTreeSet)

    return subTreeSetList


#TODO: update this or REMOVE it -- maybe move to unit tests?
#def check_tree(evaltree, original_list): #generate_circuit_list(self, permute=True):
#    """
#    Generate a list of the final operation sequences this tree evaluates.
#
#    This method essentially "runs" the tree and follows its
#      prescription for sequentailly building up longer strings
#      from shorter ones.  When permute == True, the resulting list
#      should be the same as the one passed to initialize(...), and
#      so this method may be used as a consistency check.
#
#    Parameters
#    ----------
#    permute : bool, optional
#        Whether to permute the returned list of strings into the
#        same order as the original list passed to initialize(...).
#        When False, the computed order of the operation sequences is
#        given, which is matches the order of the results from calls
#        to `Model` bulk operations.  Non-trivial permutation
#        occurs only when the tree is split (in order to keep
#        each sub-tree result a contiguous slice within the parent
#        result).
#
#    Returns
#    -------
#    list of gate-label-tuples
#        A list of the operation sequences evaluated by this tree, each
#        specified as a tuple of operation labels.
#    """
#    circuits = [None] * len(self)
#
#    #Set "initial" (single- or zero- gate) strings
#    for i, opLabel in zip(self.get_init_indices(), self.get_init_labels()):
#        if opLabel == "": circuits[i] = ()  # special case of empty label
#        else: circuits[i] = (opLabel,)
#
#    #Build rest of strings
#    for i in self.get_evaluation_order():
#        iLeft, iRight = self[i]
#        circuits[i] = circuits[iLeft] + circuits[iRight]
#
#    #Permute to get final list:
#    nFinal = self.num_final_strings()
#    if self.original_index_lookup is not None and permute:
#        finalCircuits = [None] * nFinal
#        for iorig, icur in self.original_index_lookup.items():
#            if iorig < nFinal: finalCircuits[iorig] = circuits[icur]
#        assert(None not in finalCircuits)
#        return finalCircuits
#    else:
#        assert(None not in circuits[0:nFinal])
#        return circuits[0:nFinal]

class _MatrixCOPALayoutAtom(_DistributableAtom):
    """
    Object that acts as "atomic unit" of instructions-for-applying a COPA strategy.
    """

    def __init__(self, unique_complete_circuits, unique_nospam_circuits, circuits_by_unique_nospam_circuits,
                 ds_circuits, group, model_shlp, dataset, offset, elindex_outcome_tuples):

        expanded_nospam_circuit_outcomes = _collections.OrderedDict()
        for i in group:
            nospam_c = unique_nospam_circuits[i]
            for orig_i in circuits_by_unique_nospam_circuits[nospam_c]:  # orig circuits that add SPAM to nospam_c
                observed_outcomes = None if (dataset is None) else dataset[ds_circuits[orig_i]].outcomes
                expc_outcomes = unique_complete_circuits[orig_i].expand_instruments_and_separate_povm(
                    model_shlp, observed_outcomes)

                for sep_povm_c, outcomes in expc_outcomes:
                    prep_lbl = sep_povm_c.circuit_without_povm[0]
                    exp_nospam_c = sep_povm_c.circuit_without_povm[1:]  # sep_povm_c *always* has prep lbl
                    spam_tuples = [(prep_lbl, elabel) for elabel in sep_povm_c.effect_labels]
                    outcome_by_spamtuple = {st: (outcome, orig_i) for st, outcome in zip(spam_tuples, outcomes)}

                    if exp_nospam_c not in expanded_nospam_circuit_outcomes:
                        expanded_nospam_circuit_outcomes[exp_nospam_c] = outcome_by_spamtuple
                    else:
                        expanded_nospam_circuit_outcomes[exp_nospam_c].update(outcome_by_spamtuple)

        expanded_nospam_circuits = {i: cir for i, cir in enumerate(expanded_nospam_circuit_outcomes.keys())}
        self.tree = _create_tree(expanded_nospam_circuits)
        self._num_nonscratch_tree_items = len(expanded_nospam_circuits)

        # self.trees elements give instructions for evaluating ("caching") no-spam quantities (e.g. products).
        # Now we assign final element indices to the circuit outcomes corresponding to a given no-spam ("tree")
        # quantity plus a spam-tuple. We order the final indices so that all the outcomes corresponding to a
        # given spam-tuple are contiguous.

        tree_indices_by_spamtuple = _collections.defaultdict(list)  # "tree" indices index expanded_nospam_circuits
        for i, c in expanded_nospam_circuits.items():
            for spam_tuple in expanded_nospam_circuit_outcomes[c].keys():
                tree_indices_by_spamtuple[spam_tuple].append(i)

        #Assign element indices, starting at `offset`
        # now that we know how many of each spamtuple there are, assign final element indices.
        initial_offset = offset
        self.indices_by_spamtuple = {}  # values are (element_indices, tree_indices) tuples.
        for spam_tuple, tree_indices in tree_indices_by_spamtuple.items():
            self.indices_by_spamtuple[spam_tuple] = (slice(offset, offset + len(tree_indices)), tree_indices)
            offset += len(tree_indices)
            #TODO: allow tree_indices to be None or a slice?

        element_slice = slice(initial_offset, offset)
        num_elements = offset - initial_offset

        for spam_tuple, (element_indices, tree_indices) in self.indices_by_spamtuple.items():
            for elindex, tree_index in zip(_slct.indices(element_indices), tree_indices):
                outcome_by_spamtuple = expanded_nospam_circuit_outcomes[expanded_nospam_circuits[tree_index]]
                outcome, orig_i = outcome_by_spamtuple[spam_tuple]
                elindex_outcome_tuples[orig_i].append((elindex, outcome))

        super().__init(element_slice, num_elements)

    def nonscratch_cache_view(self, a, axis=None):
        """
        Create a view of array `a` restricting it to only the *final* results computed by this tree.

        This need not be the entire array because there could be intermediate results
        (e.g. "scratch space") that are excluded.

        Parameters
        ----------
        a : ndarray
            An array of results computed using this EvalTree,
            such that the `axis`-th dimension equals the full
            length of the tree.  The other dimensions of `a` are
            unrestricted.

        axis : int, optional
            Specified the axis along which the selection of the
            final elements is performed. If None, than this
            selection if performed on flattened `a`.

        Returns
        -------
        ndarray
            Of the same shape as `a`, except for along the
            specified axis, whose dimension has been reduced
            to filter out the intermediate (non-final) results.
        """
        if axis is None:
            return a[0:self._num_nonscratch_tree_items]
        else:
            sl = [slice(None)] * a.ndim
            sl[axis] = slice(0, self._num_nonscratch_tree_items)
            ret = a[tuple(sl)]
            assert(ret.base is a or ret.base is a.base)  # check that what is returned is a view
            assert(ret.size == 0 or _np.may_share_memory(ret, a))
            return ret

    @property
    def cache_size(self):
        return len(self.tree)


class MatrixCOPALayout(_DistributableCOPALayout):
    """
    TODO: update docstring

    An Evaluation Tree that structures circuits for efficient multiplication of process matrices.

    MatrixEvalTree instances create and store the decomposition of a list of circuits into
    a sequence of 2-term products of smaller strings.  Ideally, this sequence would
    prescribe the way to obtain the entire list of circuits, starting with just the single
    gates, using the fewest number of multiplications, but this optimality is not
    guaranteed.

    Parameters
    ----------
    items : list, optional
        Initial items.  This argument should only be used internally
        in the course of serialization.

    num_strategy_subcomms : int, optional
        The number of processor groups (communicators) to divide the "atomic" portions
        of this strategy (a circuit probability array layout) among when calling `distribute`.
        By default, the communicator is not divided.  This default behavior is fine for cases
        when derivatives are being taken, as multiple processors are used to process differentiations
        with respect to different variables.  If no derivaties are needed, however, this should be
        set to (at least) the number of processors.
    """

    def __init__(self, circuits, model_shlp, dataset=None, max_sub_tree_size=None,
                 num_sub_trees=None, additional_dimensions=(), verbosity=0):

        # 1. pre-process => get complete circuits => spam-tuples list for each no-spam circuit (no expanding yet)
        # 2. decide how to divide no-spam circuits into groups corresponding to sub-strategies
        #    - create tree of no-spam circuits (may contain instruments, etc, just not SPAM)
        #    - heuristically find groups of circuits that meet criteria
        # 3. separately create a tree of no-spam expanded circuits originating from each group => self.trees
        #    (self.trees is always a list)
        # 4. assign "cache" and element indices so that a) all elements of a tree are contiguous
        #    and b) elements with the same spam-tuple are continguous.
        # 5. initialize base class with given per-original-circuit element indices.

        unique_circuits, to_unique = self._compute_unique_circuits(circuits)
        aliases = circuits.op_label_aliases if isinstance(circuits, _BulkCircuitList) else None
        ds_circuits = _lt.apply_aliases_to_circuit_list(unique_circuits, aliases)
        unique_complete_circuits = [model_shlp.complete_circuit(c) for c in unique_circuits]

        circuits_by_unique_nospam_circuits = _collections.OrderedDict()
        for i, c in enumerate(unique_complete_circuits):
            nospam_c = model_shlp.strip(c)
            if nospam_c in circuits_by_unique_nospam_circuits:
                circuits_by_unique_nospam_circuits[nospam_c].append(i)
            else:
                circuits_by_unique_nospam_circuits[nospam_c] = [i]
        unique_nospam_circuits = list(circuits_by_unique_nospam_circuits.keys())

        circuit_tree = _create_tree(unique_nospam_circuits)
        groups = _find_splitting(circuit_tree, len(unique_nospam_circuits),
                                 max_sub_tree_size, num_sub_trees, verbosity)  # a list of tuples/sets?
        # (elements of `groups` contain indices into `unique_nospam_circuits`)

        atoms = []
        elindex_outcome_tuples = {orig_i: list() for orig_i in range(len(unique_circuits))}

        offset = 0
        for group in groups:
            atoms.append(_MatrixCOPALayoutAtom(unique_complete_circuits, unique_nospam_circuits,
                                               circuits_by_unique_nospam_circuits, ds_circuits, group,
                                               model_shlp, dataset, offset, elindex_outcome_tuples))
            offset += atoms[-1].size

        super().__init__(circuits, unique_circuits, to_unique, elindex_outcome_tuples, unique_complete_circuits,
                         atoms, additional_dimensions)

    def copy(self):
        """
        Create a copy of this evaluation strategy.

        Returns
        -------
        MatrixCOPAEvalStrategy
        """
        raise NotImplementedError("TODO! update this!")
        #newTree = self._copy_base(MatrixEvalTree(self[:]))
        #newTree.opLabels = self.opLabels[:]
        #newTree.init_indices = self.init_indices[:]
        #newTree.simplified_circuit_spamTuples = self.simplified_circuit_spamTuples[:]
        ##newTree.finalStringToElsMap = self.finalStringToElsMap[:]
        #newTree.spamtuple_indices = self.spamtuple_indices.copy()
        #return newTree

    #def cache_size(self):
    #    """
    #    Returns the size of the persistent "cache".
    #
    #    This cache holds partial results used during the computation of all
    #    the strings in this tree.
    #
    #    Returns
    #    -------
    #    int
    #    """
    #    return len(self.eval_tree)

    #def get_min_tree_size(self):
    #    """
    #    Returns the minimum sub tree size required to compute each of the tree entries individually.
    #
    #    This minimum size is the smallest "max_sub_tree_size" that can be passed to
    #    split(), as any smaller value will result in at least one entry being
    #    uncomputable.
    #
    #    Returns
    #    -------
    #    int
    #    """
    #    singleItemTreeSetList = self._create_single_item_trees()
    #    return max(list(map(len, singleItemTreeSetList)))

    #PRIVATE
    #def get_analysis_plot_infos(self):
    #    """
    #    Returns debug plot information.
    #
    #    This is useful for assessing the quality of a tree. This
    #    function is not guaranteed to work.
    #
    #    Returns
    #    -------
    #    dict
    #    """
    #
    #    analysis = {}
    #    firstIndxSeen = list(range(len(self)))
    #    lastIndxSeen = list(range(len(self)))
    #    subTreeSize = [-1] * len(self)
    #
    #    xs = []; ys = []
    #    for i in range(len(self)):
    #        subTree = []
    #        self._walk_subtree(i, subTree)
    #        subTreeSize[i] = len(subTree)
    #        ys.extend([i] * len(subTree) + [None])
    #        xs.extend(list(sorted(subTree) + [None]))
    #
    #        for k, t in enumerate(self):
    #            iLeft, iRight = t
    #            if i in (iLeft, iRight):
    #                lastIndxSeen[i] = k
    #
    #    analysis['SubtreeUsagePlot'] = {'xs': xs, 'ys': ys, 'title': "Indices used by the subtree rooted at each index",
    #                                    'xlabel': "Indices used", 'ylabel': 'Subtree root index'}
    #    analysis['SubtreeSizePlot'] = {'xs': list(range(len(self))),
    #                                   'ys': subTreeSize,
    #                                   'title': "Size of subtree rooted at each index",
    #                                   'xlabel': "Subtree root index",
    #                                   'ylabel': 'Subtree size'}
    #
    #    xs = []; ys = []
    #    for i, rng in enumerate(zip(firstIndxSeen, lastIndxSeen)):
    #        ys.extend([i, i, None])
    #        xs.extend([rng[0], rng[1], None])
    #    analysis['IndexUsageIntervalsPlot'] = {'xs': xs, 'ys': ys, 'title': "Usage Intervals of each index",
    #                                           'xlabel': "Index Interval", 'ylabel': 'Index'}
    #
    #    return analysis

    #def recompute_spamtuple_indices(self, local=False):
    #    """
    #    Recompute this tree's `.spamtuple_indices` array.
    #
    #    Parameters
    #    ----------
    #    local : bool, optional
    #        If True, then the indices computed will index
    #        this tree's final array (even if it's a subtree).
    #        If False (the default), then a subtree's indices
    #        will index the *parent* tree's final array.
    #
    #    Returns
    #    -------
    #    None
    #    """
    #    self.spamtuple_indices = _compute_spamtuple_indices(
    #        self.simplified_circuit_spamTuples,
    #        None if local else self.myFinalElsToParentFinalElsMap)
    #
    #def _get_full_eval_order(self):
    #    """Includes init_indices in matrix-based evaltree case... HACK """
    #    return self.init_indices + self.eval_order
    #
    #def _update_eval_order_helpers(self, index_permutation):
    #    """Update anything pertaining to the "full" evaluation order - e.g. init_inidces in matrix-based case (HACK)"""
    #    self.init_indices = [index_permutation[iCur] for iCur in self.init_indices]
    #
    #def _update_element_indices(self, new_indices_in_old_order, old_indices_in_new_order, element_indices_dict):
    #    """
    #    Update any additional members because this tree's elements are being permuted.
    #    In addition, return an updated version of `element_indices_dict` a dict whose keys are
    #    the tree's (unpermuted) circuit indices and whose values are the final element indices for
    #    each circuit.
    #    """
    #    self.simplified_circuit_spamTuples, updated_elIndices = \
    #        self._permute_simplified_circuit_xs(self.simplified_circuit_spamTuples,
    #                                            element_indices_dict, old_indices_in_new_order)
    #    self.simplified_circuit_nEls = list(map(len, self.simplified_circuit_spamTuples))
    #    self.recompute_spamtuple_indices(local=True)  # local shouldn't matter here - just for clarity
    #
    #    return updated_elIndices


#def _compute_spamtuple_indices(simplified_circuit_spam_tuples,
#                               subtree_final_els_to_parent_final_els_map=None):
#    """
#    Returns a dictionary whose keys are the distinct spamTuples
#    found in `simplified_circuit_spam_tuples` and whose values are
#    (finalIndices, finalTreeSlice) tuples where:
#
#    finalIndices = the "element" indices in any final filled quantities
#                   which combines both spam and gate-sequence indices.
#                   If this tree is a subtree, then these final indices
#                   refer to the *parent's* final elements if
#                   `subtree_final_els_to_parent_final_els_map` is given, otherwise
#                   they refer to the subtree's final indices (usually desired).
#    treeIndices = indices into the tree's final circuit list giving
#                  all of the (raw) operation sequences which need to be computed
#                  for the current spamTuple (this list has the SAME length
#                  as finalIndices).
#    """
#    spamtuple_indices = _collections.OrderedDict(); el_off = 0
#    for i, spamTuples in enumerate(  # i == final operation sequence index
#            simplified_circuit_spam_tuples):
#        for j, spamTuple in enumerate(spamTuples, start=el_off):  # j == final element index
#            if spamTuple not in spamtuple_indices:
#                spamtuple_indices[spamTuple] = ([], [])
#            f = subtree_final_els_to_parent_final_els_map[j] \
#                if (subtree_final_els_to_parent_final_els_map is not None) else j  # parent's final
#            spamtuple_indices[spamTuple][0].append(f)
#            spamtuple_indices[spamTuple][1].append(i)
#        el_off += len(spamTuples)
#
#    def to_slice(x, max_len=None):
#        s = _slct.list_to_slice(x, array_ok=True, require_contiguous=False)
#        if max_len is not None and isinstance(s, slice) and (s.start, s.stop, s.step) == (0, max_len, None):
#            return slice(None, None)  # check for entire range
#        else:
#            return s
#
#    nRawSequences = len(simplified_circuit_spam_tuples)
#    nElements = el_off if (subtree_final_els_to_parent_final_els_map is None) \
#        else None  # (we don't know how many els the parent has!)
#    return _collections.OrderedDict(
#        [(spamTuple, (to_slice(f_inds, nElements), to_slice(g_inds, nRawSequences)))
#         for spamTuple, (f_inds, g_inds) in spamtuple_indices.items()])
