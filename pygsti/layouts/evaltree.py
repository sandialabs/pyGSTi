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

from pygsti.circuits.circuit import Circuit as _Circuit, LayerTupLike
from pygsti.baseobjs.verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from pygsti.baseobjs.label import LabelTupTup, Label, LabelTup
from pygsti.modelmembers.operations import create_from_superop_mx
from pygsti.modelmembers.operations import LinearOperator as _LinearOperator
import itertools
from pygsti.tools.sequencetools import (
    conduct_one_round_of_lcs_simplification,
    _compute_lcs_for_every_pair_of_sequences,
    create_tables_for_internal_LCS,
    simplify_internal_first_one_round
)

from pygsti.circuits.split_circuits_into_lanes import compute_qubit_to_lane_and_lane_to_qubits_mappings_for_circuit, compute_subcircuits
import time
import scipy.linalg as la
import scipy.sparse.linalg as sparla
from typing import List, Optional, Iterable
from pygsti.tools.tqdm import our_tqdm


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




#region Split circuit list into lists of subcircuits

def _add_in_idle_gates_to_circuit(circuit: _Circuit, idle_gate_name: str|Label = 'I') -> _Circuit:
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





def setup_circuit_list_for_LCS_computations(
        circuit_list: list[_Circuit],
        implicit_idle_gate_name: str|Label = 'I'
    ) -> tuple[
        dict[int, dict[int, _Circuit]],
        dict[LayerTupLike, list[tuple[int, int]]],
        dict[tuple[int, ...],list[LayerTupLike]]
    ]:
    """
    Split a circuit list into a list of subcircuits by lanes. These lanes are non-interacting partions of a circuit.

    Also return a sequence detailing the number of lanes in each circuit.
    Then, a sequence detailing the number of qubits in each lane for a circuit.
    """

    # We want to split the circuit list into a dictionary of subcircuits where each sub_cir in the dict[key] act exclusively on the same qubits.
    # I need a mapping from subcircuit to actual circuit. This is uniquely defined by circuit_id and then lane id.

    cir_ind_and_lane_id_to_sub_cir: dict[int, dict[int, _Circuit]] = {}
    sub_cir_to_cir_id_and_lane_id:  dict[LayerTupLike, list[tuple[int, int]]] = {}
    line_labels_to_layertup_lists:  dict[tuple[int, ...], list[LayerTupLike]] = {}

    for i, cir in enumerate(circuit_list):

        if implicit_idle_gate_name:
            cir = _add_in_idle_gates_to_circuit(cir, implicit_idle_gate_name)

        qubit_to_lane, lane_to_qubits = compute_qubit_to_lane_and_lane_to_qubits_mappings_for_circuit(cir)
        sub_cirs = compute_subcircuits(cir, qubit_to_lane, lane_to_qubits)

        if not implicit_idle_gate_name:
            if not all([len(sc) == len(sub_cirs[0]) for sc in sub_cirs]):
                raise ValueError("Each lane does not have the same number of layers. Therefore, a lane has an implicit idle gate. Please add in idle gates explicitly to the circuit.")

        assert len(sub_cirs) == len(lane_to_qubits)
        for j in range(len(sub_cirs)):
            sc = _Circuit(sub_cirs[j],line_labels=tuple(lane_to_qubits[j]),)
            lbls = sc._line_labels
            if lbls in line_labels_to_layertup_lists:
                line_labels_to_layertup_lists[lbls].append(sc.layertup)
            else:
                line_labels_to_layertup_lists[lbls] = [sc.layertup]
            if sc.layertup in sub_cir_to_cir_id_and_lane_id:
                sub_cir_to_cir_id_and_lane_id[sc.layertup].append((i,j))
            else:
                sub_cir_to_cir_id_and_lane_id[sc.layertup] = [(i,j)]
            if i in cir_ind_and_lane_id_to_sub_cir:
                cir_ind_and_lane_id_to_sub_cir[i][j] = sc
            else:
                cir_ind_and_lane_id_to_sub_cir[i] = {j: sc}

    return cir_ind_and_lane_id_to_sub_cir, sub_cir_to_cir_id_and_lane_id, line_labels_to_layertup_lists

#endregion Split Circuits by lanes helpers


#region Lane Collapsing Helpers

def get_dense_representation_of_gate_with_perfect_swap_gates(model, op: LabelTup, saved: dict[int | LabelTup | LabelTupTup, _np.ndarray], swap_dense: _np.ndarray) -> _np.ndarray:
    """
    Assumes that a gate which operates on 2 qubits does not have the right orientation if label is (qu_i+1, qu_i).
    """
    if op.num_qubits == 2:
        # We may need to do swaps.
        op_term : _np.ndarray = _np.array([1.])
        if op in saved:
            op_term = saved[op]
        elif op.qubits[1] < op.qubits[0]:  # type: ignore
            # This is in the wrong order.
            op_term = model._layer_rules.get_dense_process_matrix_represention_for_gate(model, op)
            op_term = swap_dense @ (op_term) @ swap_dense
            saved[op] = op_term # Save so we only need to this operation once.
        else:
            op_term = model._layer_rules.get_dense_process_matrix_represention_for_gate(model, op)
        return op_term
    return model._layer_rules.get_dense_process_matrix_represention_for_gate(model, op)


def matrix_matrix_cost_estimate(matrix_size: tuple[int, int]) -> int:
    """
    Estimate cost of A @ B when both are square and dense.
    """
    n = matrix_size[0]
    return 2 * n**3


#endregion Lane Collapsing Helpers


class EvalTreeBasedUponLongestCommonSubstring():

    def __init__(self, circuit_list: list[LabelTupTup], qubit_starting_loc: int = 0):
        """
        Construct an evaluation order tree for a circuit list that minimizes the number of rounds of computation.
        """

        self.circuit_to_save_location = {tuple(cir): i for i,cir in enumerate(circuit_list)}

        self.orig_circuits = {i: circuit_list[i] for i in range(len(circuit_list))}
        self.qubit_start_point = qubit_starting_loc


        internal_matches = create_tables_for_internal_LCS(circuit_list)
        best_internal_match = _np.max(internal_matches[0])

        max_rounds = best_internal_match

        C = len(circuit_list)
        sequence_intro = {0: _np.arange(C)}

        cache_pos = C
        cache = {i: circuit_list[i] for i in range(len(circuit_list))}

        new_circuit_list = [cir for cir in circuit_list] # Get a deep copy since we will modify it here.

        # Let's try simplifying internally first.
        self.internal_first = False
        seq_ind_to_cache_index = {i: i for i in range(C)}
        if self.internal_first:
            i = 0
            cache_pos = -1
            while max_rounds > 1:

                breakpoint()
                tmp = simplify_internal_first_one_round(new_circuit_list, 
                                                        internal_matches,
                                                        cache_pos,
                                                        cache,
                                                        seq_ind_to_cache_index)
                new_circuit_list, cache_pos, cache, sequence_intro[i-1] = tmp
                i -= 1
                internal_matches = create_tables_for_internal_LCS(new_circuit_list)

                max_rounds = _np.max(internal_matches[0])
        external_matches = _compute_lcs_for_every_pair_of_sequences(new_circuit_list,
                                                                    None,
                                                                    None,
                                                                    set(_np.arange(len(new_circuit_list))),
                                                                    max([len(cir) for cir in new_circuit_list])-1)


        best_external_match = _np.max(external_matches[0])

        max_rounds = int(max(best_external_match,best_internal_match))
        i = 0
        cache_pos = len(new_circuit_list)
        while max_rounds > 1:
            tmp = conduct_one_round_of_lcs_simplification(new_circuit_list,
                                                          external_matches,
                                                          internal_matches,
                                                          cache_pos,
                                                          cache,
                                                          seq_ind_to_cache_index)
            new_circuit_list, cache_pos, cache, sequence_intro[i+1], ext_table, external_sequences, dirty_inds = tmp
            i += 1
            dirty_inds = set(_np.arange(len(new_circuit_list))) # TODO: fix to only correct those which are actually dirty.
            external_matches = _compute_lcs_for_every_pair_of_sequences(new_circuit_list,
                                                                        ext_table,
                                                                        external_sequences,
                                                                        dirty_inds,
                                                                        max_rounds)

            if best_internal_match < best_external_match and best_external_match < 2 * best_internal_match:
                # We are not going to get a better internal match.
                pass
            elif not self.internal_first:
                internal_matches = create_tables_for_internal_LCS(new_circuit_list)

            best_external_match = _np.max(external_matches[0])
            best_internal_match = _np.max(internal_matches[0])

            max_rounds = int(max(best_external_match,best_internal_match))

        self.cache = cache
        self.num_circuits = C
        self.from_other = False

        self.sequence_intro = sequence_intro

        from pygsti.modelmembers.operations import StaticStandardOp
        self.swap_gate = StaticStandardOp('Gswap', basis='pp').to_dense().round(16)

        self.cache_ind_to_alphabet_vals_referenced: dict[int, set[LabelTupTup]] = {}


        # Useful for repeated calculations seen in a derivative calculation.
        for key in self.cache:
            self.compute_depends_on(key, self.cache_ind_to_alphabet_vals_referenced)

        alphabet_val_to_cache_inds_to_update: dict[LabelTupTup, set[int]] = {}

        for cache_ind, vals in self.cache_ind_to_alphabet_vals_referenced.items():
            for val in vals:
                if val in alphabet_val_to_cache_inds_to_update:
                    alphabet_val_to_cache_inds_to_update[val].add(cache_ind)
                else:
                    alphabet_val_to_cache_inds_to_update[val] = set([cache_ind])

        self.results: dict[int | LabelTupTup, _np.ndarray] = {}

        self.alphabet_val_to_sorted_cache_inds: dict[LabelTupTup, list[int]] = {}

        for val, cache_inds in alphabet_val_to_cache_inds_to_update.items():
            rnd_nums = {}
            for cache_ind in cache_inds:
                for rnd_num in self.sequence_intro:
                    if cache_ind in self.sequence_intro[rnd_num]:
                        rnd_nums[cache_ind] = rnd_num
                        break

            sorted_inds = sorted(cache_inds, key =lambda x : rnd_nums[x])[::-1] # We want to iterate large to small.

            self.alphabet_val_to_sorted_cache_inds[val] = sorted_inds

      
    def from_other_eval_tree(self, other: EvalTreeBasedUponLongestCommonSubstring, qubit_label_exchange: dict[int, int]):
        """
        Construct a tree from another tree.
        """
        
        self.cache = other.cache
        self.num_circuits = other.num_circuits
        self.sequence_intro = other.sequence_intro
        self.swap_gate = other.swap_gate
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

    def collapse_circuits_to_process_matrices(self, model, num_qubits_in_default: int, alphabet_piece_changing: Optional[LabelTupTup] = None):
        """
        Compute the total product cache. Note that this may still have a tensor product
        structure that the operator needs to combine again if they want to have the full 'dense' matrix.
        """

        if alphabet_piece_changing is not None:

            if alphabet_piece_changing not in self.alphabet_val_to_sorted_cache_inds:
                # Nothing needs to change here.
                return self.results, self.circuit_to_save_location
            
            cache_inds = self.alphabet_val_to_sorted_cache_inds[alphabet_piece_changing]

            local_changes = {k: v.copy() for k, v in self.results.items()}
            for cache_ind in cache_inds:
                cumulative_term = None
                for term in self.cache[cache_ind]:
                    cumulative_term = self._collapse_cache_line(model, cumulative_term, term, local_changes, num_qubits_in_default)

                # Do not overwrite the results cache
                # so that we can use it again on a different derivative.
                if cumulative_term is None:
                    local_changes[cache_ind] = _np.eye(4**num_qubits_in_default)
                    # NOTE: unclear when (if ever) this should be a noisy idle gate.
                else:
                    local_changes[cache_ind] = cumulative_term
            return local_changes, self.circuit_to_save_location


        else:
            round_keys = sorted(_np.unique(list(self.sequence_intro.keys())))[::-1]
            # saved: dict[int | LabelTupTup, _np.ndarray] = {}

            if self.internal_first:

                round_keys = _np.unique(list(self.sequence_intro.keys()))

                pos_inds = _np.where(round_keys >0)
                pos_keys = round_keys[pos_inds]
                pos_keys = sorted(pos_keys)[::-1]

                neg_inds = _np.where(round_keys < 0)
                neg_keys = round_keys[neg_inds]
                neg_keys = sorted(neg_keys)

                round_keys = pos_keys + neg_keys + _np.array([0])
            
            for key in round_keys:
                for cache_ind in self.sequence_intro[key]:
                    cumulative_term = None
                    for term in self.cache[cache_ind]:
                        cumulative_term = self._collapse_cache_line(model, cumulative_term, term, self.results, num_qubits_in_default)
                            
                    if cumulative_term is None:
                        self.results[cache_ind] = _np.eye(4**num_qubits_in_default)
                        # NOTE: unclear when (if ever) this should be a noisy idle gate.
                    else:
                        self.results[cache_ind] = cumulative_term
        if __debug__:
            # We may store more in the cache in order to handle multi-qubit gates which are out of the normal order.
            for key in self.cache:
                assert key in self.results
        
        # {tuple(self.trace_through_cache_to_build_circuit(icir)): icir for icir in range(len(self.orig_circuit_list)) if icir < self.num_circuits}
    
        return self.results, self.circuit_to_save_location
    
    def compute_depends_on(self, val: int | LabelTupTup, visited: dict[int, set[LabelTupTup]]) -> set[LabelTupTup]:

        if not isinstance(val, int):
            return set([val])
        elif val in visited:
            return visited[val]
        else:
            tmp = set()
            for child in self.cache[val]:
                ret_val = self.compute_depends_on(child, visited)
                tmp = tmp.union(ret_val)
            visited[val] = tmp
            return tmp


    def combine_for_visualization(self, val, visited):

        if not isinstance(val, int):
            return [val]
        elif val in visited:
            return visited[val]
        else:
            tmp = []
            for child in self.cache[val]:
                tmp.append(self.combine_for_visualization(child, visited))
            visited[val] = tmp
            return tmp

    def handle_results_cache_lookup_and_product(self,
                            cumulative_term: None | _np.ndarray,
                            term_to_extend_with: int | LabelTupTup,
                            results_cache: dict[int | LabelTupTup, _np.ndarray]) -> _np.ndarray:

        if cumulative_term is None:
            # look up result.
            return results_cache[term_to_extend_with]
        return results_cache[term_to_extend_with] @ cumulative_term 


    def _collapse_cache_line(self, model, cumulative_term: None | _np.ndarray,
                            term_to_extend_with: int | LabelTupTup,
                            results_cache: dict[int | LabelTupTup, _np.ndarray],
                            num_qubits_in_default: int) -> _np.ndarray:
        """
        Reduce a cache line to a single process matrix.

        This should really only be called from collapse_circuits_to_process_matrices.

        """

        if isinstance(term_to_extend_with, int):
            assert term_to_extend_with in results_cache
            return self.handle_results_cache_lookup_and_product(cumulative_term, term_to_extend_with, results_cache)
        if term_to_extend_with in results_cache:
            return self.handle_results_cache_lookup_and_product(cumulative_term, term_to_extend_with, results_cache)
        else:
            val = 1
            qubits_available = [i + self.qubit_start_point for i in range(num_qubits_in_default)]
            matrix_reps = {op.qubits: get_dense_representation_of_gate_with_perfect_swap_gates(model, op,
                                            results_cache, self.swap_gate) for op in term_to_extend_with}
            qubit_used = []
            for key in matrix_reps.keys():
                qubit_used.extend(key)

            assert len(qubit_used) == len(set(qubit_used))
            unused_qubits = set(qubits_available) - set(qubit_used)

            implicit_idle_reps = {(qu,): get_dense_representation_of_gate_with_perfect_swap_gates(model,
                                        Label("Fake_Gate_To_Get_Tensor_Size_Right", qu), # A fake gate to look up and use the appropriate idle gate.
                                        results_cache, self.swap_gate) for qu in unused_qubits}

            while qubits_available:

                qu = qubits_available[0]
                if qu in unused_qubits:
                    val = _np.kron(val, implicit_idle_reps[(qu,)])
                    qubits_available = qubits_available[1:]
                else:
                    # It must be a part of a non-trivial gate.
                    gatekey = [key for key in matrix_reps if qu in key][0]
                    val = _np.kron(val, matrix_reps[gatekey])

                    qubits_available = qubits_available[len(gatekey):]

            results_cache[term_to_extend_with] = val
            if cumulative_term is None:
                return val
            # Cache if off.
            return results_cache[term_to_extend_with] @ cumulative_term


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

    def flop_cost_of_evaluating_tree(self, matrix_size: tuple[int, int]):
        """
        We assume that each matrix matrix multiply is the same size.
        """

        assert matrix_size[0] == matrix_size[1]

        total_flop_cost = 0
        for cache_ind in self.cache:
            num_mm_on_this_cache_line = len(self.cache[cache_ind]) - 1
            total_flop_cost += (matrix_matrix_cost_estimate(matrix_size)) * num_mm_on_this_cache_line

        return total_flop_cost


class CollectionOfLCSEvalTrees():

    def __init__(self, line_lbls_to_circuit_list: dict[tuple[int, ...], list[LabelTupTup]],
                 sub_cir_to_full_cir_id_and_lane_id,
                 cir_id_and_lane_id_to_sub_cir):
        
        self.trees: dict[tuple[int, ...], EvalTreeBasedUponLongestCommonSubstring] = {}

        ASSUME_MATCHING_QUBIT_SIZE_MATCHING_TREE = False

        size_to_tree: dict[int, tuple[int, ...]] = {}

        self.line_lbls_to_cir_list = line_lbls_to_circuit_list

        starttime = time.time()
        for key, vals in our_tqdm(line_lbls_to_circuit_list.items(), " Building Longest Common Substring Caches"):
            sub_cirs = []
            for cir in vals:
                sub_cirs.append(list(cir))
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

        self.cir_id_to_tensor_order: dict[int, list[list[int], int]] = {}
        self.compute_tensor_orders()

        self.saved_results = {}
        self.sub_cir_to_ind_in_results: dict[tuple[int, ...], dict[_Circuit, int]] = {}

    def collapse_circuits_to_process_matrices(self, model, alphabet_piece_changing: Optional[LabelTupTup] = None):
        """
        Collapse all circuits to their process matrices. If alphabet_piece_changing is not None, then
        we assume we have already collapsed this system once before and so only need to update part of the eval tree.
        """
        # Just collapse all of them.


        self.saved_results = {}
        for key in self.trees:
            num_qubits = len(key) if key[0] != ('*',) else key[1] # Stored in the data structure.
            tree = self.trees[key]
            out1, out2 = tree.collapse_circuits_to_process_matrices(model, num_qubits, alphabet_piece_changing)
            # self.saved_results[key], self.sub_cir_to_ind_in_results[key] = self.trees[key].collapse_circuits_to_process_matrices(model, len(key))
            self.saved_results[key] = out1
            self.sub_cir_to_ind_in_results[key] = out2

    def reconstruct_full_matrices(self) -> Optional[List[KronStructured]]:
        """
        Construct a tensor product structure for each individual circuit
        """

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

                ind_in_results = self.sub_cir_to_ind_in_results[lblkey][cir.layertup]
                lane_circuits.append(self.saved_results[lblkey][ind_in_results])
            output.append(KronStructured(lane_circuits))
        return output
    
    def flop_estimate(self, return_collapse: bool = False, return_tensor_matvec: bool = False):


        cost_collapse = 0
        for key in self.trees:
            num_qubits = len(key) if key[0] != ('*',) else key[1] # Stored in the data structure.
            tree = self.trees[key]
            cost_collapse += tree.flop_cost_of_evaluating_tree(tuple([4**num_qubits, 4**num_qubits]))
        

        tensor_cost = 0
        num_cirs = len(self.cir_id_and_lane_id_to_sub_cir)

        for cir_id in range(num_cirs):
            qubit_list = ()
            for lane_id in range(len(self.cir_id_and_lane_id_to_sub_cir[cir_id])):
                subcir = self.cir_id_and_lane_id_to_sub_cir[cir_id][lane_id]
                qubit_list = (*qubit_list, len(subcir._line_labels))
            qubit_list = list(qubit_list)
            total_num = _np.sum(qubit_list)

            tensor_cost += cost_to_compute_tensor_matvec_without_reordering(qubit_list, total_num)

        if return_collapse:
            return tensor_cost + cost_collapse, cost_collapse
        elif return_tensor_matvec:
            return tensor_cost + cost_collapse, tensor_cost

        return tensor_cost + cost_collapse

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
            
    def best_order_for_tensor_contraction(self,
                    qubit_list: tuple[int, ...],
                    cache: dict[tuple[int, ...], tuple[list[int], int]]) -> tuple[list[int], int]:
        """
        Find the tensor contraction order that minizes the cost of contracting to a dense system with
        a total number of qubits equal to the len(qubit_list)
        """
        

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
        cache[qubit_list] = best_order, best_cost

        return best_order, best_cost

    def _tensor_cost_model(self, num_qubits1, num_qubits2):
        """
        Assumes kronecker product of 2 square matrices.
        """

        return (4**num_qubits1)**2 * (4**num_qubits2)**2
    
    def _flop_estimate_to_collapse_to_each_circuit_to_process_matrix(self) -> tuple[int, list[int], list[int]]:
        """
        Compute the number of flops needed to collapse each circuit into a single process matrix.

        Returns:
        ---------
            cost - int total cost to collapse and reform
            collapse_lane_cost - list[int] cost to collapse a lane
            tensor_cost - list[int] cost to recombine a circuit into its full size.
        """


        num_cirs = len(self.cir_id_and_lane_id_to_sub_cir)

        collapse_lane_cost = []

        for lbl_key, my_tree in self.trees.items():
            collapse_lane_cost.append(my_tree.flop_cost_of_evaluating_tree([4**len(lbl_key), 4**len(lbl_key)]))

        tensor_cost = []
        for icir in range(num_cirs):
            
            _order, cost = self.cir_id_to_tensor_order[icir]
            tensor_cost.append(cost)

        return sum(tensor_cost) + sum(collapse_lane_cost), collapse_lane_cost, tensor_cost
    




class RealLinOp:
    
    # Function implementations below are merely defaults.
    # Don't hesitate to override them if need be.

    __array_priority__ = 100

    @property
    def ndim(self):
        return 2

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def T(self):
        return self._adjoint

    def item(self):
        # If self.size == 1, return a scalar representation of this linear operator.
        # Otherwise, error.
        raise NotImplementedError()

    def __matmul__(self, other):
        return self._linop @ other
    
    def __rmatmul__(self, other):
        return other @ self._linop


def is_2d_square(arg):
    if not hasattr(arg, 'shape'):
        return False
    if len(arg.shape) != 2:
        return False
    return arg.shape[0] == arg.shape[1]


class DyadicKronStructed(RealLinOp):

    def __init__(self, A, B, adjoint=None):
        assert A.ndim == 2
        assert B.ndim == 2
        self.A = A
        self.B = B
        self._A_is_trivial = A.size == 1
        self._B_is_trivial = B.size == 1
        self._shape = ( A.shape[0]*B.shape[0], A.shape[1]*B.shape[1] )
        self._size = self.shape[0] * self.shape[1]
        self._fwd_matvec_core_shape = (B.shape[1], A.shape[1])
        self._adj_matvec_core_shape = (B.shape[0], A.shape[0])
        self._dtype = A.dtype
        self._linop =  sparla.LinearOperator(dtype=self.dtype, shape=self.shape, matvec=self.matvec, rmatvec=self.rmatvec)
        self._adjoint = DyadicKronStructed(A.T, B.T, adjoint=self) if adjoint is None else adjoint

    def item(self):
        # This will raise a ValueError if self.size > 1.
        return self.A.item() * self.B.item()
    
    def matvec(self, other):
        inshape = other.shape
        assert other.size == self.shape[1]
        if self._A_is_trivial:
            return self.A.item() * (self.B @ other)
        if self._B_is_trivial:
            return self.B.item() * (self.A @ other)
        out = self.B @ _np.reshape(other, self._fwd_matvec_core_shape, order='F') @ self.A.T
        out = _np.reshape(out, inshape, order='F')
        return out

    def rmatvec(self, other):
        inshape = other.shape
        assert other.size == self.shape[0]
        if self._A_is_trivial:
            return self.A.item() * (self.B.T @ other)
        if self._B_is_trivial:
            return self.B.item() * (self.A.T @ other)
        out = self.B.T @ _np.reshape(other, self._adj_matvec_core_shape, order='F') @ self.A
        out = _np.reshape(out, inshape, order='F')
        return out
    
    @staticmethod
    def build_polyadic(kron_operands):
        if len(kron_operands) == 2:
            out = DyadicKronStructed(kron_operands[0], kron_operands[1])
            return out
        # else, recurse
        arg = DyadicKronStructed.build_polyadic(kron_operands[1:])
        out = DyadicKronStructed(kron_operands[0], arg)
        return out


class KronStructured(RealLinOp):

    def __init__(self, kron_operands):
        self.kron_operands = kron_operands
        assert all([op.ndim == 2 for op in kron_operands])
        self.shapes = _np.array([op.shape for op in kron_operands])
        self._shape = tuple(int(i) for i in _np.prod(self.shapes, axis=0))
        forward = DyadicKronStructed.build_polyadic(self.kron_operands)
        self._linop   = forward._linop
        self._adjoint = forward.T
        self._dtype = self.kron_operands[0].dtype


def cost_to_compute_tensor_matvec_without_reordering(qubit_list: list[int], total_num_qubits: int):

    assert _np.sum(qubit_list) == total_num_qubits

    if len(qubit_list) == 1:
        # Basic matvec.
        cost = 2 * (4**qubit_list[0]**2)
        return cost
    
    elif len(qubit_list) == 2:
        # vec((A \tensor B) u) = vec(B U A.T)
        term1 = 2*(4**qubit_list[1]**2) * (4**qubit_list[0]) # MM of BU.
        term2 = 2 * (4**qubit_list[0]**2) * (4**qubit_list[1]) # MM of U A.T
        return term1 + term2
    
    else:
        # Just pop off the last term
        # (B_1 \tensor B_2 ... \tensor B_n) u = (B_n \tensor B_n-1 ... \tensor B_2) U (B_1).T

        right = cost_to_compute_tensor_matvec_without_reordering(qubit_list[:1], qubit_list[0])
        right *= 4**(_np.sum(qubit_list[1:]))
        left = cost_to_compute_tensor_matvec_without_reordering(qubit_list[1:],
                                                                total_num_qubits - qubit_list[0])
        left *= 4**(qubit_list[0])
        return left + right