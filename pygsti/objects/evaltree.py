"""
Defines the EvalTree class.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import time as _time  # DEBUG TIMERS
import collections as _collections
import bisect as _bisect
import warnings as _warnings

from .verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from .circuit import Circuit as _Circuit


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
                        #REMOVE: print("b=%d is possible: score = %d" % (b, score))
                        if score > best_bite_and_score[1]: best_bite_and_score = (b, score)
                        if score == L: break  # this is a maximal score, so stop looking
                        #OLD REMOVE: bite = b; break

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
                #REMOVE: print("best bite = ",bite)

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

            #DEBUG REMOVE
            #if db_added_scratch > 100:
            #    print("DB: Processing circuit: ", circuit.str)
            #    print("Index %d added %d scratch entries" % (k, db_added_scratch))
            #    import bpdb; bpdb.set_trace()

        if len(circuits_to_evaluate) > 0:
            test_ratios = (100, 10, 3); ratio = len(eval_tree) / len(circuits_to_evaluate)
            for test_ratio in test_ratios:
                if ratio >= test_ratio:
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
