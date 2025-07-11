"""
Defines the EvalTree class.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations
import bisect as _bisect
import time as _time  # DEBUG TIMERS
import warnings as _warnings

import numpy as _np

from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti.baseobjs.verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from pygsti.baseobjs.label import LabelTupTup, Label
from pygsti.modelmembers.operations import create_from_superop_mx
from pygsti.modelmembers.operations import LinearOperator as _LinearOperator
import itertools
from typing import Sequence
import time



def _walk_subtree(treedict, indx, running_inds):
    running_inds.add(indx)
    (iDest, iLeft, iRight) = treedict[indx]
    if iLeft is not None:
        _walk_subtree(treedict, iLeft, running_inds)
        _walk_subtree(treedict, iRight, running_inds)


class EvalTree(list):
    @classmethod
    def create(cls, circuits_to_evaluate):  # a class method instead of __init__ because we inherit from list
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
        eval_tree = cls()  # makes an empty list

        #Evaluation dictionary:
        # keys == operation sequences that have been evaluated so far
        # values == index of operation sequence (key) within eval_tree
        evalDict = {}  # _collections.defaultdict(dict)
        evalDict_keys = []  # the sorted keys of evalDict

        #Process circuits in order of length, so that we always place short strings
        # in the right place (otherwise assert stmt below can fail)
        indices_sorted_by_circuit_len = \
            sorted(list(range(len(circuits_to_evaluate))),
                   key=lambda i: len(circuits_to_evaluate[i]))

        next_scratch_index = len(circuits_to_evaluate)
        for k in indices_sorted_by_circuit_len:

            circuit = circuits_to_evaluate[k]
            layertup = circuit.layertup if isinstance(circuit, _Circuit) else circuit
            L = len(circuit)

            #Single gate (or zero-gate) computations are assumed to be atomic, and be computed independently.
            #  These labels serve as the initial values, and each operation sequence is assumed to be a tuple of
            #  operation labels.
            if L == 0:
                eval_tree.append((k, None, None))  # iLeft = iRight = None => no-op (length-0 circuit)
                if L not in evalDict:
                    evalDict[L] = {}
                    _bisect.insort(evalDict_keys, L)  # inserts L into evalDict_keys while maintaining sorted order
                evalDict[L][None] = k
                continue

            elif L == 1:
                eval_tree.append((k, None, layertup[0]))  # iLeft = None => evaluate iRight as a label
                if L not in evalDict:
                    evalDict[L] = {}
                    _bisect.insort(evalDict_keys, L)  # inserts L into evalDict_keys while maintaining sorted order
                evalDict[L][layertup] = k
                continue

            def best_bite_length(tup, possible_bitelens):
                for b in possible_bitelens:
                    if tup[0:b] in evalDict[b]:
                        return b
                return 0

            #db_added_scratch = 0
            start = 0; bite = 1
            possible_bs = list(reversed(evalDict_keys))  # copy list
            while start < L:

                #Take a bite out of circuit, starting at `start` that is in evalDict
                maxb = L - start
                possible_bs = [b for b in possible_bs if b <= maxb]
                best_bite_and_score = (None, 0)
                for b in possible_bs:  # range(L - start, 0, -1):
                    if layertup[start:start + b] in evalDict[b]:
                        # score of taking this bite = this bite's length + length of next bite
                        #if start + b == L: break  # maximal score, so stop looking (this finishes circuit)
                        score = b + best_bite_length(layertup[start + b:],
                                                     [bb for bb in possible_bs if bb <= L - (start + b)])
                        if score > best_bite_and_score[1]: best_bite_and_score = (b, score)
                        if score == L: break  # this is a maximal score, so stop looking

                if best_bite_and_score[0] is not None:
                    bite = best_bite_and_score[0]
                else:
                    # Can't even take a bite of length 1, so add the next op-label to the tree and take b=1 bite.
                    eval_tree.append((next_scratch_index, None, layertup[start]))
                    if 1 not in evalDict:
                        evalDict[1] = {}
                        _bisect.insort(evalDict_keys, 1)
                    evalDict[1][layertup[start:start + 1]] = next_scratch_index; next_scratch_index += 1
                    bite = 1

                bFinal = bool(start + bite == L)
                evalDict_bite = evalDict[bite]
                #print("DB: start=", start, ": found ", layertup[start:start + bite],
                #      " (len=%d) in evalDict" % bite, "(final=%s)" % bFinal)

                if start == 0:  # first in-evalDict bite - no need to add anything to self yet
                    iCur = evalDict_bite[layertup[0:bite]]
                    #print("DB: taking initial bite:", layertup[0:bite], "indx =", iCur)
                    if bFinal:
                        if iCur != k:  # then we have a duplicate final operation sequence
                            if 0 not in evalDict:
                                evalDict[0] = {}
                                _bisect.insort(evalDict_keys, 0)
                            iEmptyStr = evalDict[0].get(None, None)
                            if iEmptyStr is None:  # then we need to add the empty string
                                # duplicate final strs require the empty string to be included in the tree
                                iEmptyStr = next_scratch_index; next_scratch_index += 1
                                evalDict[0][None] = iEmptyStr
                                eval_tree.append((iEmptyStr, None, None))  # iLeft = iRight = None => no-op
                            #assert(self[k] is None)  # make sure we haven't put anything here yet
                            eval_tree.append((k, iCur, iEmptyStr))
                            #self[k] = (iCur, iEmptyStr)  # compute the duplicate using by
                            #self.eval_order.append(k)  # multiplying by the empty string.
                else:
                    # add (iCur, iBite)
                    assert(layertup[0:start + bite] not in evalDict_bite)
                    iBite = evalDict_bite[layertup[start:start + bite]]
                    if start + bite not in evalDict:
                        evalDict[start + bite] = {}
                        _bisect.insort(evalDict_keys, start + bite)

                    if bFinal:  # place (iCur, iBite) at location k
                        iNew = k
                        evalDict[start + bite][layertup[0:start + bite]] = iNew  # note: start + bite == L
                        #assert(self[iNew] is None)  # make sure we haven't put anything here yet
                        #self[k] = (iCur, iBite)
                        eval_tree.append((k, iCur, iBite))
                        #print("DB: add final %s (index %d)" % (str(layertup[0:start + bite]), iNew))
                    else:
                        iNew = next_scratch_index
                        evalDict[start + bite][layertup[0:start + bite]] = iNew
                        eval_tree.append((iNew, iCur, iBite))
                        next_scratch_index += 1
                        #print("DB: add scratch %s (index %d)" % (str(layertup[0:start + bite]), iNew))
                        #db_added_scratch += 1

                    iCur = iNew
                start += bite
                #nBites += 1

        if len(circuits_to_evaluate) > 0:
            test_ratios = (100, 10, 3); ratio = len(eval_tree) / len(circuits_to_evaluate)
            for test_ratio in test_ratios:
                if ratio >= test_ratio and len(circuits_to_evaluate) > 1:  # no warning for 1-circuit case
                    _warnings.warn(("Created an evaluation tree that is inefficient: tree-size > %d * #circuits !\n"
                                    "This is likely due to the fact that the circuits being simulated do not have a\n"
                                    "periodic structure. Consider using a different forward simulator "
                                    "(e.g. MapForwardSimulator).") % test_ratio)
                    break  # don't print multiple warnings about the same inefficient tree

        return eval_tree

    def _create_single_item_trees(self, num_elements):
        # num_elements == number of elements *to evaluate* (can be < len(self))
        #  Create disjoint set of subtrees generated by single items
        need_to_compute = _np.zeros(len(self), 'bool')
        need_to_compute[0:num_elements] = True

        treedict = {iDest: (iDest, iLeft, iRight) for iDest, iLeft, iRight in self}

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

    def find_splitting(self, num_elements, max_sub_tree_size, num_sub_trees, verbosity):
        """
        Find a partition of the indices of `circuit_tree` to define a set of sub-trees with the desire properties.

        This is done in order to reduce the maximum size of any tree (useful for
        limiting memory consumption or for using multiple cores).  Must specify
        either max_sub_tree_size or num_sub_trees.

        Parameters
        ----------
        num_elements : int
            The number of elements `self` is meant to compute (this means that any
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
        printer = _VerbosityPrinter.create_printer(verbosity)

        if max_sub_tree_size is None and num_sub_trees is None:
            return [set(range(num_elements))]  # no splitting needed

        if max_sub_tree_size is not None and num_sub_trees is not None:
            raise ValueError("Cannot specify both max_sub_tree_size and num_sub_trees")
        if num_sub_trees is not None and num_sub_trees <= 0:
            raise ValueError("num_sub_trees must be > 0!")

        #Don't split at all if it's unnecessary
        if max_sub_tree_size is None or len(self) < max_sub_tree_size:
            if num_sub_trees is None or num_sub_trees == 1:
                return [set(range(num_elements))]  # no splitting needed

        #First pass - identify which indices go in which subtree
        #   Part 1: create disjoint set of subtrees generated by single items
        singleItemTreeSetList = self._create_single_item_trees(num_elements)

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

                def _get_start_indices(max_intersect):
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
                    iStartingTrees, startingTreeEls = _get_start_indices(mid)
                    nStartingTrees = len(iStartingTrees)
                    if nStartingTrees < num_sub_trees:
                        left = mid + 1
                    elif nStartingTrees > num_sub_trees:
                        right = mid
                    else: break  # nStartingTrees == num_sub_trees!

                if len(iStartingTrees) < num_sub_trees:
                    iStartingTrees, startingTreeEls = _get_start_indices(mid + 1)
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
                #Another possible algorithm (but slower)
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

        #Remove all "scratch" indices, as we want a partition just of the "final" items:
        subTreeSetList = [set(filter(lambda x: x < num_elements, s)) for s in subTreeSetList]

        #Remove duplicated "final" items, as only a single tree (the first one to claim it)
        # should be assigned each final item, even if other trees need to compute that item as scratch.
        # BUT: keep these removed final items as helpful scratch items, as these items, though
        #      not needed, can help in the creating of a balanced evaluation tree.
        claimed_final_indices = set(); disjointLists = []; helpfulScratchLists = []
        for subTreeSet in subTreeSetList:
            disjointLists.append(subTreeSet - claimed_final_indices)
            helpfulScratchLists.append(subTreeSet - disjointLists[-1])  # the final items that were duplicated
            claimed_final_indices.update(subTreeSet)

        assert(sum(map(len, disjointLists)) == num_elements), "sub-tree sets are not disjoint!"
        return disjointLists, helpfulScratchLists


#region Longest Common Subsequence

def _best_matching_only(A: Sequence, B: Sequence) -> int:
    """
    Returns:
    -----
    int - the length of the longest matching prefix between A and B.
    """
    i = 0
    n = len(A)
    m = len(B)
    while i < n and i < m:
        if A[i] != B[i]:
            return len(A[:i])
        i += 1
    return len(A[:i])


def _lcs_dp_version(A: Sequence, B: Sequence):
    """
    Compute the longest common substring between A and B using
    dynamic programming.

    This will use O(n \times m) space and take O(n \times m \times max(m, n)) time.
    """
    table = _np.zeros((len(A) + 1, len(B) + 1))
    n, m = table.shape
    for i in range(n-2, -1, -1):
        for j in range(m-2, -1, -1):
            opt1 = 0
            if A[i] == B[j]:
                opt1 = _best_matching_only(A[i:], B[j:])
            opt2 = table[i, j+1]
            opt3 = table[i+1, j]
            table[i,j] = max(opt1, opt2, opt3)
    return table


def _conduct_one_round_of_lcs_simplification(circuits, table_data_and_sequences, internal_tables_and_sequences, starting_cache_num, cache_struct, round_num: int=0):
    if table_data_and_sequences:
        table, sequences = table_data_and_sequences
    else:
        table, sequences = _compute_lcs_for_every_pair_of_circuits(circuits)

    if internal_tables_and_sequences:
        internal_subtable, internal_subsequences = internal_tables_and_sequences
    else:
        internal_subtable, internal_subsequences = build_internal_tables(circuits)

    best_index = _np.where(table == _np.max(table))
    best_internal_index = _np.where(internal_subtable == _np.max(internal_subtable))
    updated_circuits = circuits
    cache_num = starting_cache_num

    # Build sequence dict
    all_subsequences_to_replace: dict[tuple, dict[int, list[int]]] = {}

    if _np.max(internal_subtable) >= _np.max(table):
        # We are only going to replace if this was the longest substring.
        for cir_ind in best_internal_index[0]:
            for seq in internal_subsequences[cir_ind]:
                key = tuple(seq)
                if key in all_subsequences_to_replace:
                    all_subsequences_to_replace[key][cir_ind] = internal_subsequences[cir_ind][seq]
                else:
                    all_subsequences_to_replace[key] = {cir_ind: internal_subsequences[cir_ind][seq]}

    if _np.max(table) >= _np.max(internal_subtable):
        for ii in range(len(best_index[0])):
            starting_point, starting_point_2, length = sequences[(best_index[0][ii], best_index[1][ii])]
            cir_index = best_index[0][ii]
            cir_index2 = best_index[1][ii]
            seq = updated_circuits[cir_index][starting_point: int(starting_point + length+1)]

            key = tuple(seq)
            if key in all_subsequences_to_replace:
                if cir_index not in all_subsequences_to_replace[key]:
                    # We did not already handle this with internal subsequences.
                    all_subsequences_to_replace[key][cir_index] = [starting_point]
                if cir_index2 not in all_subsequences_to_replace[key]:
                    all_subsequences_to_replace[key][cir_index2] = [starting_point_2]

            else:
                all_subsequences_to_replace[key] = {cir_index: [starting_point], cir_index2: [starting_point_2]}


    # Handle the updates.
    old_cache_num = cache_num
    for seq, cdict in all_subsequences_to_replace.items():
        w = len(seq)
        if  w > 1 or (not isinstance(seq[0], int)):
            # We have reached an item which we can just compute.
            for cir_ind in cdict:
                my_cir = updated_circuits[cir_ind]
                sp = 0
                while sp+w <= len(my_cir):
                    if list(my_cir[sp: sp+w]) == list(seq):
                        my_cir[sp: sp + w] = [cache_num]

                    sp += 1
                updated_circuits[cir_ind] = my_cir

                cache_struct[cir_ind] = updated_circuits[cir_ind]

            updated_circuits.append(list(seq))
            cache_struct[cache_num] = updated_circuits[cache_num]

            cache_num += 1

    sequences_introduced_in_this_round = _np.arange(cache_num - old_cache_num) + old_cache_num

    return updated_circuits, cache_num, cache_struct, sequences_introduced_in_this_round


def _find_starting_positions_using_dp_table(dp_table) -> tuple[int, int, int]:
    """
    Finds the indices of the starting points of the sequences in A and B.

    Returns:
    ---------
    int - starting index in A of LCS(A,B)
    int - starting index in B of LCS(A,B)
    int - length of LCS(A,B)
    """
    n, m = dp_table.shape
    i = 0
    j = 0
    while i < n-1 and j < m -1:
        curr = dp_table[i,j]
        opt1 = dp_table[i+1, j+1]
        opt2 = dp_table[i+1, j]
        opt3 = dp_table[i, j+1]
        options = [opt1, opt2, opt3]
        if _np.all(curr == options):
            i += 1
            j += 1
        elif opt2 > opt1 and opt2 > opt3:
            i += 1
        elif opt3 > opt2 and opt3 > opt1:
            j += 1
        else:
            # All three options are equal. So we should march the diagonal.
            i += 1
            j += 1
            return i-1, j-1, dp_table[i,j]
    return None, None, None


def _compute_lcs_for_every_pair_of_circuits(circuit_list: list[_Circuit]):
    """
    Computes the LCS for every pair of circuits A,B in circuit_list
    """
    best_subsequences = {}
    best_lengths = _np.zeros((len(circuit_list), len(circuit_list)))
    curr_best = 0
    for i in range(len(circuit_list)-1, -1, -1): # Lets do this in reverse order
        cir0 = circuit_list[i]
        if len(cir0) >= curr_best:
            # Could be the best.
            for j in range(i-1, -1, -1):
                cir1 = circuit_list[j]
                if len(cir1) >= curr_best:
                    table = _lcs_dp_version(cir0, cir1)
                    best_lengths[i,j] = table[0,0]
                    best_subsequences[(i,j)] = _find_starting_positions_using_dp_table(table)
                    curr_best = max(best_lengths[i,j], curr_best)
                else:
                    best_lengths[i,j] = -1
                    best_subsequences[(i,j)] = (None, None, None)
        else:
            # Skipped because cannot be the best yet.
            best_lengths[i,j] = -1
            best_subsequences[(i,j)] = (None, None, None)
    return best_lengths, best_subsequences


def _longest_common_internal_subsequence(A: Sequence) -> tuple[int, dict[tuple, list[int]]]:
    """
    Compute the longest common subsequence within a single circuit A.

    Cost ~ O(L^3 / 8) where L is the length of A

    Returns:
    ---------
    int - length of longest common subsequences within A
    dict[tuple, list[int]] - dictionary of subsequences to starting positions within A.
    """
    n = len(A)
    best = 0
    best_ind = {}
    changed = False
    for w in range(1, int(_np.floor(n / 2) + 1)):
        for sp in range(n - w):
            window = A[sp: sp + w]
            for match in range(sp+ w, n-w + 1):
                if A[match: match + w] == window:
                    if best == w:
                        if tuple(window) in best_ind:
                            best_ind[tuple(window)].add(match)
                        else:
                            best_ind[tuple(window)] = {sp, match}
                    else:
                        best_ind = {tuple(window): {sp, match}}
                        changed = True
                        best = w
        if not changed:
            return best, best_ind
    return best, best_ind


def build_internal_tables(circuit_list):
    """
    Compute all the longest common internal sequences for each circuit A in circuit_list

    Total cost is O(C L^3).
    """

    C = len(circuit_list)
    the_table = _np.zeros(C)
    seq_table = [[] for _ in range(C)]

    curr_best = 1
    for i in range(C):
        if len(circuit_list[i]) >= curr_best:
            the_table[i], seq_table[i] = _longest_common_internal_subsequence(circuit_list[i])
            curr_best = max(curr_best, the_table[i])
    return the_table, seq_table

#endregion Longest Common Subsequence


#region Split circuit list into lists of subcircuits

def _add_in_idle_gates_to_circuit(circuit: _Circuit, idle_gate_name: str = "I") -> _Circuit:
    """
    Add in explicit idles to the labels for each layer.
    """

    tmp = circuit.copy(editable=True)
    num_layers = circuit.num_layers

    for i in range(num_layers):
        tmp[i] = Label(tmp.layer_label_with_idles(i, idle_gate_name))

    if tmp._static:
        tmp.done_editing()
    return tmp


def _compute_qubit_to_lanes_mapping_for_circuit(circuit, num_qubits: int) -> tuple[dict[int, int], dict[int, tuple[int]]]:
    """
    Returns
    --------
    Dictionary mapping qubit number to lane number in the circuit.
    """

    qubits_to_potentially_entangled_others = {i: set((i,)) for i in range(num_qubits)}
    num_layers = circuit.num_layers
    for layer_ind in range(num_layers):
        layer = circuit.layer(layer_ind)
        for op in layer:
            qubits_used = op.qubits
            for qb in qubits_used:
                qubits_to_potentially_entangled_others[qb].update(set(qubits_used))

    lanes = {}
    lan_num = 0
    visited: dict[int, int] = {}
    def reachable_nodes(starting_point: int, graph_qubits_to_neighbors: dict[int, set[int]], visited: dict[int, set[int]]):
        """
        Find which nodes are reachable from this starting point.
        """
        if starting_point in visited:
            return visited[starting_point]
        else:
            assert starting_point in graph_qubits_to_neighbors
            visited[starting_point] = graph_qubits_to_neighbors[starting_point]
            output = set(visited[starting_point])
            for child in graph_qubits_to_neighbors[starting_point]:
                if child != starting_point:
                    output.update(output, reachable_nodes(child, graph_qubits_to_neighbors, visited))
            visited[starting_point] = output
            return output

    available_starting_points = list(sorted(qubits_to_potentially_entangled_others.keys()))
    while available_starting_points:
        sp = available_starting_points[0]
        nodes = reachable_nodes(sp, qubits_to_potentially_entangled_others, visited)
        for node in nodes:
            available_starting_points.remove(node)
        lanes[lan_num] = nodes
        lan_num += 1

    def compute_qubits_to_lanes(lanes_to_qubits: dict[int, set[int]]) -> dict[int, int]:
        """
        Determine a mapping from qubit to the lane it is in for this specific circuit.
        """
        out = {}
        for key, val in lanes_to_qubits.items():
            for qb in val:
                out[qb] = key
        return out

    return compute_qubits_to_lanes(lanes), lanes


def _compute_subcircuits(circuit, qubits_to_lanes: dict[int, int]) -> list[list[LabelTupTup]]:
    """
    Split a circuit into multiple subcircuits which do not talk across lanes.
    """

    lanes_to_gates = [[] for _ in range(_np.unique(list(qubits_to_lanes.values())).shape[0])]

    num_layers = circuit.num_layers
    for layer_ind in range(num_layers):
        layer = circuit.layer(layer_ind)
        group = []
        group_lane = None
        sorted_layer = sorted(layer, key=lambda x: x.qubits[0])

        for op in sorted_layer:
            # We need this to be sorted by the qubit number so we do not get that a lane was split Q1 Q3 Q2 in the layer where Q1 and Q2 are in the same lane.
            qubits_used = op.qubits # This will be a list of qubits used.
            # I am assuming that the qubits are indexed numerically and not by strings.
            lane = qubits_to_lanes[qubits_used[0]]

            if group_lane is None:
                group_lane = lane
                group.append(op)
            elif group_lane == lane:
                group.append(op)
            else:
                lanes_to_gates[group_lane].append(LabelTupTup(tuple(group)))
                group_lane = lane
                group = [op]

        if len(group) > 0:
            # We have a left over group.
            lanes_to_gates[group_lane].append(LabelTupTup(tuple(group)))

    return lanes_to_gates


def setup_circuit_list_for_LCS_computations(
        circuit_list: list[_Circuit],
        implicit_idle_gate_name: str = "I") -> tuple[list[dict[int, int]],
                                                    dict[tuple[_Circuit], list[tuple[int, int]]],
                                                    dict[tuple[int, ...], set[_Circuit]]]:
    """
    Split a circuit list into a list of subcircuits by lanes. These lanes are non-interacting partions of a circuit.

    Also return a sequence detailing the number of lanes in each circuit.
    Then, a sequence detailing the number of qubits in each lane for a circuit.
    """

    # output = []
    # cir_id_to_lanes = []

    # We want to split the circuit list into a dictionary of subcircuits where each sub_cir in the dict[key] act exclusively on the same qubits.
    # I need a mapping from subcircuit to actual circuit. This is uniquely defined by circuit_id and then lane id.

    sub_cir_to_cir_id_and_lane_id: dict[tuple[_Circuit], list[tuple[int, int]]] = {}
    line_labels_to_circuit_list: dict[tuple[int, ...], set[_Circuit]] = {}
    cir_ind_and_lane_id_to_sub_cir: dict[int, dict[int, _Circuit]] = {}

    for i, cir in enumerate(circuit_list):

        if implicit_idle_gate_name:
            cir = _add_in_idle_gates_to_circuit(cir, implicit_idle_gate_name)

        qubits_to_lane, lanes_to_qubits = _compute_qubit_to_lanes_mapping_for_circuit(cir, cir.num_lines)
        sub_cirs = _compute_subcircuits(cir, qubits_to_lane)

        assert len(sub_cirs) == len(lanes_to_qubits)
        for j in range(len(sub_cirs)):
            sc = _Circuit(sub_cirs[j])
            lbls = sc._line_labels
            if lbls in line_labels_to_circuit_list:
                line_labels_to_circuit_list[lbls].append(sc)
            else:
                line_labels_to_circuit_list[lbls] = [sc]
            if sc in sub_cir_to_cir_id_and_lane_id:
                sub_cir_to_cir_id_and_lane_id[sc].append((i,j))
            else:
                sub_cir_to_cir_id_and_lane_id[sc] = [(i,j)]
            if i in cir_ind_and_lane_id_to_sub_cir:
                cir_ind_and_lane_id_to_sub_cir[i][j] = sc
            else:
                cir_ind_and_lane_id_to_sub_cir[i] = {j: sc}

        # output.extend(sub_cirs)
        # cir_id_to_lanes.append(lanes_to_qubits)
    return cir_ind_and_lane_id_to_sub_cir, sub_cir_to_cir_id_and_lane_id, line_labels_to_circuit_list

#endregion Split Circuits by lanes helpers


#region Lane Collapsing Helpers

def model_and_gate_to_dense_rep(model, opTuple) -> _np.ndarray:
    """
    Look up the dense representation of a gate in the model.
    """


    if hasattr(model, "operations"):
        return model.operations[opTuple].to_dense()
    elif hasattr(model, "operation_blks"):
        if opTuple[0] not in model.operation_blks["gates"]:
            breakpoint()
        return model.operation_blks["gates"][opTuple[0]].to_dense()
    else:
        raise ValueError("Missing attribute")


def get_dense_representation_of_gate_with_perfect_swap_gates(model, op: Label, saved: dict[int | LabelTupTup, _np.ndarray], swap_dense: _np.ndarray) -> _np.ndarray:
    op_term = 1
    if op.num_qubits == 2:
        # We may need to do swaps.
        if op in saved:
            op_term = saved[op]
        elif op.qubits[1] < op.qubits[0]:
            # This is in the wrong order.
            op_term = model_and_gate_to_dense_rep(model, op)
            op_term = swap_dense @ (op_term) @ swap_dense
            saved[op] = op_term # Save so we only need to this operation once.
        else:
            op_term = model_and_gate_to_dense_rep(model, op)
    else:
        op_term = model_and_gate_to_dense_rep(model, op)
    return op_term


def combine_two_gates(cumulative_term, next_dense_matrix):
    """
    Note that the visual representation was

    State Prep | CumulativeTerm | NextDense | Measure

    which in matrix multiplication requires Measure @ (NextDense @ Cumulative) @ State Prep.
    """
    return next_dense_matrix @ cumulative_term

#endregion Lane Collapsing Helpers


class EvalTreeBasedUponLongestCommonSubstring():

    def __init__(self, circuit_list: list[LabelTupTup], qubit_starting_loc: int = 0):
        """
        Construct an evaluation order tree for a circuit list that minimizes the number of rounds of computation.
        """

        self.circuit_to_save_location = {tuple(cir): i for i,cir in enumerate(circuit_list)}

        external_matches = _compute_lcs_for_every_pair_of_circuits(circuit_list)
        
        best_external_match = _np.max(external_matches[0])
        self.orig_circuits = {i: circuit_list[i] for i in range(len(circuit_list))}
        self.qubit_start_point = qubit_starting_loc


        internal_matches = build_internal_tables(circuit_list)
        best_internal_match = _np.max(internal_matches[0])

        max_rounds = int(max(best_external_match,best_internal_match))

        C = len(circuit_list)
        sequence_intro = {0: _np.arange(C)}

        cache_pos = C
        cache = {i: circuit_list[i] for i in range(len(circuit_list))}

        new_circuit_list = [cir for cir in circuit_list] # Get a deep copy since we will modify it here.

        i = 0
        while max_rounds > 1:
            new_circuit_list, cache_pos, cache, sequence_intro[i+1] = _conduct_one_round_of_lcs_simplification(new_circuit_list, external_matches, internal_matches, cache_pos, cache, i)
            i += 1
            external_matches = _compute_lcs_for_every_pair_of_circuits(new_circuit_list)

            if best_internal_match < best_external_match and best_external_match < 2 * best_internal_match:
                # We are not going to get a better internal match.
                pass
            else:
                internal_matches = build_internal_tables(new_circuit_list)

            best_external_match = _np.max(external_matches[0])
            best_internal_match = _np.max(internal_matches[0])

            max_rounds = int(max(best_external_match,best_internal_match))

        self.circuit_list = new_circuit_list
        self.cache = cache
        self.num_circuits = C
        self.from_other = False

        self.sequence_intro = sequence_intro

        self.swap_gate = _np.array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.23259516e-32,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,
                                      0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  1.23259516e-32,  0.00000000e+00, 0.00000000e+00, -1.23259516e-32],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, 0.00000000e+00, 1.23259516e-32,
                                      0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,
                                      1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.23259516e-32, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00],
                                    [ 1.23259516e-32,  0.00000000e+00,  0.00000000e+00, -1.23259516e-32,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,
                                      0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.23259516e-32],
                                   
                                    [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,
                                     0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  1.23259516e-32,  0.00000000e+00, 0.00000000e+00],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,0.00000000e+00,  0.00000000e+00,  1.00000000e+00, 0.00000000e+00,  0.00000000e+00,
                                     0.00000000e+00, 0.00000000e+00,  1.23259516e-32,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.23259516e-32,  0.00000000e+00,
                                     0.00000000e+00, 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00],
                                    [ 0.00000000e+00,  1.23259516e-32,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,0.00000000e+00,  0.00000000e+00,
                                     0.00000000e+00,0.00000000e+00,  0.00000000e+00,  0.00000000e+00,0.00000000e+00,  1.00000000e+00,  0.00000000e+00, 0.00000000e+00],
                                   
                                    [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,
                                     0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.23259516e-32, 0.00000000e+00],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00,  0.00000000e+00,
                                     0.00000000e+00, -1.23259516e-32,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00],
                                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.23259516e-32, 0.00000000e+00,  0.00000000e+00,
                                    0.00000000e+00, 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,0.00000000e+00],
                                    [ 0.00000000e+00,  0.00000000e+00,  1.23259516e-32, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,
                                     0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.00000000e+00, 0.00000000e+00],
       
                                    [ 1.23259516e-32,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,
                                     0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.23259516e-32,  0.00000000e+00,  0.00000000e+00, 1.23259516e-32],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  1.23259516e-32,  0.00000000e+00, 0.00000000e+00,  1.00000000e+00,
                                     0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00],
                                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,0.00000000e+00,  0.00000000e+00,
                                    1.23259516e-32, 0.00000000e+00,  0.00000000e+00,  1.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00],
                                    [-1.23259516e-32, 0.00000000e+00,  0.00000000e+00, 1.23259516e-32,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,
                                     0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.23259516e-32,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])

        # Assumes a perfect swap gate!
        # self.swap_gate = create_from_superop_mx(swap_gate, "static standard", stdname="Gswap")            

    def from_other_eval_tree(self, other: EvalTreeBasedUponLongestCommonSubstring, qubit_label_exchange: dict[int, int]):
        """
        Construct a tree from another tree.
        """
        
        self.cache = other.cache
        self.num_circuits = other.num_circuits
        self.sequence_intro = other.sequence_intro
        self.swap_gate = other.swap_gate
        self.circuit_list = other.circuit_list
        self.orig_circuit_list = other.orig_circuit_list
        self.circuit_to_save_location = other.circuit_to_save_location
        self.from_other = other

        for ind in self.cache:
            for i, term in enumerate(self.cache[ind]):
                if isinstance(term, int):
                    pass # The tree will stay the same.
                elif isinstance(term, LabelTupTup):
                    new_term = ()
                    for op in term:
                        new_qu = (qubit_label_exchange[qu] for qu in op.qubits)
                        new_op = (op.name, *new_qu)
                        new_term = (*new_term, new_op)
                    self.cache[ind][i] = Label(new_term)

        
        for icir in range(len(self.orig_circuit_list)):
            self.orig_circuit_list[icir] = self.trace_through_cache_to_build_circuit(icir)

        updated = {}
        for cir, loc in self.circuit_to_save_location.items():
            new_cir = ()
            for layer in cir:
                new_layer = ()
                for op in layer:
                    new_op = (op[0], *(qubit_label_exchange[qu] for qu in op[1:]))
                    new_layer = (*new_layer, new_op)
                new_cir = (*new_cir, new_layer)
            updated[new_cir] = loc
        self.circuit_to_save_location = updated

    def collapse_circuits_to_process_matrices(self, model, num_qubits_in_default: int):
        """
        Compute the total product cache. Note that this may still have a tensor product
        structure that the operator needs to combine again if they want to have the full 'dense' matrix.
        """


        round_keys = sorted(_np.unique(list(self.sequence_intro.keys())))[::-1]
        saved: dict[int, _LinearOperator] = {}
        


        def cache_lookup_and_product(cumulative_term, term_to_extend_with: int):
            if cumulative_term is None:
                # look up result.
                return saved[term]
            elif isinstance(term, int) and cumulative_term is not None:
                return combine_two_gates(cumulative_term, saved[term_to_extend_with]) 



        def collapse_cache_line(cumulative_term, term_to_extend_with: int | LabelTupTup):

            if isinstance(term_to_extend_with, int):
                return cache_lookup_and_product(cumulative_term, term_to_extend_with)

            else:
                val = 1
                qubits_used = [i for i in range(num_qubits_in_default)]
                while qubits_used:
                    qu = qubits_used[0]
                    gate_matrix = _np.eye(4)
                    found = False
                    op_ind = self.qubit_start_point # Handle circuits with only qubits (i, i+k) where k is number of qubits in the subsystem.
                    while not found and op_ind < len(term):
                        op = term[op_ind]
                        if qu in op.qubits:
                            gate_matrix = get_dense_representation_of_gate_with_perfect_swap_gates(model, op, saved, self.swap_gate)
                            found = True
                            # We assume that the qubits need to overlap for a specific gate.
                            # i.e. One cannot have op.qubits = (0, 2) in a system with a qubits (0,1,2).
                            qubits_used = qubits_used[len(op.qubits):]
                        op_ind += 1
                    val = _np.kron(val, gate_matrix)
                    if not found:
                        # Remove that qubit from list to check.
                        qubits_used = qubits_used[1:]

                if val.shape != expected_shape:
                    breakpoint()
                if cumulative_term is None:
                    return val
                else:
                    return combine_two_gates(cumulative_term, val)

        expected_shape = (4**num_qubits_in_default, 4**num_qubits_in_default)
        for key in round_keys:
            for cind in self.sequence_intro[key]:
                cumulative_term = None
                for term in self.cache[cind]:
                    cumulative_term = collapse_cache_line(cumulative_term, term)
                        
                if cumulative_term is None:
                    saved[cind] = _np.eye(4**num_qubits_in_default) # identity of the appropriate size.
                else:
                    saved[cind] = cumulative_term
        if __debug__:
            # We may store more in the cache in order to handle multi-qubit gates which are out of the normal order.
            for key in self.cache:
                assert key in saved
        
        # {tuple(self.trace_through_cache_to_build_circuit(icir)): icir for icir in range(len(self.orig_circuit_list)) if icir < self.num_circuits}
    
        return saved, self.circuit_to_save_location 

    def trace_through_cache_to_build_circuit(self, cache_ind: int) -> list[tuple]:

        output = ()
        for term in self.cache[cache_ind]:

            if isinstance(term, Label):
                output = (*output, term)
            elif isinstance(term, int):
                # Recurse down.
                next_term = self.trace_through_cache_to_build_circuit(term)
                output = (*output, *next_term)

        return list(output)

    """        
    def _evaluate_product_rule(self, cind: int, rn: int):

        sequence = self.cache[cind]
        num_terms = len(sequence)
        sub_tree_cache, sub_rounds = self.deriv_ordering_cache[num_terms]

        for sub_r in sorted(sub_rounds.keys())[::-1]:
            sub_sequence = None
            for sub_cind in sub_rounds[sub_r]:
        
                for term in sub_tree_cache[sub_cind]:
                    if isinstance(term, tuple):
                        # Then, this may be a partial derivative or an character in original sequence.
                        if len(term) == 2:
                            # Then this is taking a partial derivative.
                            natural_term = term[1][0]
                            if natural_term in self.derivative_cache:
                                cumulative_term = cumulative_term @ self.derivative_cache[natural_term]
                            else:
                                # This should be a natural derivative.
                                self.derivative_cache[natural_term] = term.deriv_wrt_params(None)
                                cumulative_term = cumulative_term @ self.derivative_cache[natural_term]

                        # It is just an index to sequence for where to look in the cache.
                        next_ind = term[0]
                        sequence_val = sequence[next_ind]

                        if isinstance(term, int) and cumulative_term is None:
                            # look up result.
                            cumulative_term = saved[term]
                        elif isinstance(term, int) and not (cumulative_term is None):
                            cumulative_term = saved[term] @ cumulative_term
                        elif isinstance(term, LabelTupTup):
                            val = 1
                            for op in term:
                            op_term = 1
                            if op.num_qubits == 2:
                                # We may need to do swaps.
                                if op in saved:
                                    op_term = saved[op]
                                elif op.qubits[1] < op.qubits[0]:
                                    # This is in the wrong order.
                                    swap_term = model.operation_blks["gates"][("Gswap",0,1)].to_dense() # assume this is perfect.
                                    op_term = model.operation_blks["gates"][op].to_dense()
                                    op_term = swap_term @ op_term @ swap_term.T
                                    saved[op] = op_term # Save so we only need to this operation once.
                                else:
                                    op_term = model.operation_blks["gates"][op].to_dense()
                            else:
                                op_term = model.operation_blks["gates"][op].to_dense()
                            val = _np.kron(val, op_term)
                        #val = model.operation_blks["gates"][term[0]].to_dense()
                        if cumulative_term is None:
                            cumulative_term = val
                        else:
                            cumulative_term = val @ cumulative_term
    """


class CollectionOfLCSEvalTrees():

    def __init__(self, line_lbls_to_circuit_list, sub_cir_to_full_cir_id_and_lane_id, cir_id_and_lane_id_to_sub_cir):
        
        self.trees: dict[tuple[int, ...], EvalTreeBasedUponLongestCommonSubstring] = {}

        ASSUME_MATCHING_QUBIT_SIZE_MATCHING_TREE = False

        size_to_tree: dict[int, tuple[int, ...]] = {}

        self.line_lbls_to_cir_list = line_lbls_to_circuit_list

        starttime = time.time()
        for key, vals in line_lbls_to_circuit_list.items():
            sub_cirs = [list(cir) for cir in vals]
            if ASSUME_MATCHING_QUBIT_SIZE_MATCHING_TREE:
                if len(key) not in size_to_tree:
                    self.trees[key] = EvalTreeBasedUponLongestCommonSubstring(sub_cirs)
                    size_to_tree[len(key)] = key
                else:
                    sample = EvalTreeBasedUponLongestCommonSubstring(sub_cirs[:2]) # Build a small version to be corrected later.
                    other_key = size_to_tree[len(key)]
                    sample.from_other_eval_tree(self.trees[other_key], {other_key[i]: key[i] for i in range(len(key))})
                    self.trees[key] = sample
            else:
                self.trees[key] = EvalTreeBasedUponLongestCommonSubstring(sub_cirs, sorted(key)[0])
                
        endtime = time.time()

        print(" Time to compute all the evaluation orders (s): ", endtime - starttime)


        self.sub_cir_to_full_cir_id_and_lane_id = sub_cir_to_full_cir_id_and_lane_id
        self.cir_id_and_lane_id_to_sub_cir = cir_id_and_lane_id_to_sub_cir

        self.cir_id_to_tensor_order = {}
        self.compute_tensor_orders()

        self.saved_results = {}
        self.sub_cir_to_ind_in_results: dict[tuple[int, ...], dict[_Circuit, int]] = {}

    def collapse_circuits_to_process_matrices(self, model):
        # Just collapse all of them.
        
        self.saved_results = {}
        for key in self.trees:
            self.saved_results[key], self.sub_cir_to_ind_in_results[key] = self.trees[key].collapse_circuits_to_process_matrices(model, len(key))

    def reconstruct_full_matrices(self):

        if len(self.saved_results) == 0:
            return
        
        # Now we can do the combination.

        num_cirs = len(self.cir_id_and_lane_id_to_sub_cir)

        output = []
        for icir in range(num_cirs):
            lane_circuits = []
            for i in range(len(self.cir_id_and_lane_id_to_sub_cir[icir])):
                cir = self.cir_id_and_lane_id_to_sub_cir[icir][i]
                lblkey = cir._line_labels

                if len(cir.layertup) == 0:

                    lane_circuits.append(_np.eye(4**(len(lblkey))))
                else:
                    if cir.layertup not in self.sub_cir_to_ind_in_results[lblkey]:
                        print(lblkey)
                        print(cir)
                        breakpoint()
                    ind_in_results = self.sub_cir_to_ind_in_results[lblkey][cir.layertup]
                    lane_circuits.append(self.saved_results[lblkey][ind_in_results])
            output.append(lane_circuits)

        # Need a map from lane id to computed location.
        for icir in range(num_cirs):

            order = self.cir_id_to_tensor_order[icir]
            
            
            while order:
                sp = order[0]
                output[icir][sp] = _np.kron(output[icir][sp], output[icir][sp+1])
                output[icir][sp+1:] = output[icir][sp+2:]
                
                # Adjust future indices
                tmp = []
                for new_val in order[1:]:
                    tmp.append((new_val - 1)*(new_val > sp) + (new_val) * (new_val < sp))
                order = tmp

            output[icir] = output[icir][0]
        return output
    
    def compute_tensor_orders(self):

        num_cirs = len(self.cir_id_and_lane_id_to_sub_cir)

        cache_struct = {}

        for cir_id in range(num_cirs):
            qubit_list = ()
            for lane_id in range(len(self.cir_id_and_lane_id_to_sub_cir[cir_id])):
                subcir = self.cir_id_and_lane_id_to_sub_cir[cir_id][lane_id]
                qubit_list = (*qubit_list, len(subcir._line_labels))
            self.cir_id_to_tensor_order[cir_id] = self.best_order_for_tensor_contraction(qubit_list, cache_struct)

        return
            
    def best_order_for_tensor_contraction(self, qubit_list: tuple[int, ...], cache):
        

        if qubit_list in cache:
            return cache[qubit_list]

        best_cost = _np.inf
        best_order = []

        for order in itertools.permutations(range(len(qubit_list)-1), len(qubit_list)-1):

            my_list = [qb for qb in qubit_list] # force deep copy.
            my_starting_points = [sp for sp in order]
            cost = 0
            early_exit = False
            while my_starting_points and not early_exit:
                sp = my_starting_points.pop(0)

                cost += self._tensor_cost_model(my_list[sp], my_list[sp+1])
                if cost <= best_cost:
                    # modify sp for future.
                    tmp = []
                    for new_val in my_starting_points:
                        tmp.append((new_val - 1)*(new_val > sp) + (new_val) * (new_val < sp))
                    my_starting_points = tmp

                    q2 = my_list.pop(sp+1)
                    my_list[sp] += q2
                else:
                    early_exit = True # This round is done because the partial sum was too big.

            if cost < best_cost and not early_exit:
                best_cost = cost
                best_order = list(order)

        # Store off the information.
        cache[qubit_list] = best_order

        return best_order

    def _tensor_cost_model(self, num_qubits1, num_qubits2):
        """
        Assumes kronecker product of 2 square matrices.
        """

        return (4**num_qubits1)**2 * (4**num_qubits2)**2
