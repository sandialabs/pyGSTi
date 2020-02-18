""" Defines the MatrixEvalTree class which implements an evaluation tree. """
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
from .evaltree import EvalTree

import numpy as _np
import collections as _collections
import time as _time  # DEBUG TIMERS


class MatrixEvalTree(EvalTree):
    """
    An Evaluation Tree.  Instances of this class specify how to
      perform bulk Model operations.

    EvalTree instances create and store the decomposition of a list
      of operation sequences into a sequence of 2-term products of smaller
      strings.  Ideally, this sequence would prescribe the way to
      obtain the entire list of operation sequences, starting with just the
      single gates, using the fewest number of multiplications, but
      this optimality is not guaranteed.
    """

    def __init__(self, items=[]):
        """ Create a new, empty, evaluation tree. """

        # indices for initial computation that is viewed separately
        # from the "main evaluation" given by eval_order
        self.init_indices = []

        # a list of spamTuple-lists, one for each final operation sequence
        self.simplified_circuit_spamTuples = None
        #self.finalStringToElsMap = None

        # a dictionary of final-gate-string index lists keyed by
        # each distinct spamTuple
        self.spamtuple_indices = None

        # list of the operation labels
        self.opLabels = []

        super(MatrixEvalTree, self).__init__(items)

    def initialize(self, simplified_circuit_elabels, numSubTreeComms=1):
        """
          Initialize an evaluation tree using a set of operation sequences.
          This function must be called before using an EvalTree.

          Parameters
          ----------
          TODO: docstring - fix this!
          circuit_list : list of (tuples or Circuits)
              A list of tuples of operation labels or Circuit
              objects, specifying the operation sequences that
              should be present in the evaluation tree.

          numSubTreeComms : int, optional
              The number of processor groups (communicators)
              to divide the subtrees of this EvalTree among
              when calling `distribute`.  By default, the
              communicator is not divided.

          Returns
          -------
          None
        """
        #tStart = _time.time() #DEBUG TIMER

        #Extra processing step - matrix eval tree deals with simple circuits *without* their preps
        # since it's trivial to compute probabilities for different state preps when you have the
        # process matrix.  The values of the simplified_circuit_list then become lists of *spamtuples*
        # rather than just lists of effect labels.
        simplified_circuit_list = _collections.OrderedDict()
        for simple_circuit_with_prep, elabels in simplified_circuit_elabels.items():
            if elabels == [None]:  # special case when there is no prep
                simplified_circuit_list[simple_circuit_with_prep] = elabels
            else:
                rhoLbl = simple_circuit_with_prep[0]  # assume first circuit layer is a prep
                simple_circuit_no_prep = simple_circuit_with_prep[1:]
                simplified_circuit_list[simple_circuit_no_prep] = [(rhoLbl, eLbl) for eLbl in elabels]

        # opLabels : A list of all the length-0 & 1 operation labels to be stored
        #  at the beginning of the tree.  This list must include all the gate
        #  labels contained in the elements of simplified_circuit_list
        #  (including a special empty-string sentinel at the beginning).
        self.opLabels = [""] + self._get_opLabels(simplified_circuit_elabels)
        if numSubTreeComms is not None:
            self.distribution['numSubtreeComms'] = numSubTreeComms

        circuit_list = [tuple(mdl) for mdl in simplified_circuit_list.keys()]
        self.simplified_circuit_spamTuples = list(simplified_circuit_list.values())
        self.simplified_circuit_nEls = list(map(len, self.simplified_circuit_spamTuples))
        self.num_final_els = sum([len(v) for v in self.simplified_circuit_spamTuples])
        #self._compute_finalStringToEls() #depends on simplified_circuit_spamTuples
        self.recompute_spamtuple_indices(bLocal=True)  # bLocal shouldn't matter here

        #Evaluation dictionary:
        # keys == operation sequences that have been evaluated so far
        # values == index of operation sequence (key) within evalTree
        evalDict = {}

        #Evaluation tree:
        # A list of tuples, where each element contains
        #  information about evaluating a particular operation sequence:
        #  (iLeft, iRight)
        # and the order of the elements specifies the evaluation order.
        # In particular, the circuit = evalTree[iLeft] + evalTree[iRight]
        #   so that matrix(circuit) = matrixOf(evalTree[iRight]) * matrixOf(evalTree[iLeft])
        del self[:]  # clear self (a list)

        #Final Indices
        # The first len(circuit_list) elements of the tree correspond
        # to computing the operation sequences requested in circuit_list.  Doing
        # this make later extraction much easier (views can be used), but
        # requires a non-linear order of evaluation, held in the eval_order list.
        self.eval_order = []

        #initialize self as a list of Nones
        self.num_final_strs = len(circuit_list)
        self[:] = [None] * self.num_final_strs

        #Single gate (or zero-gate) computations are assumed to be atomic, and be computed independently.
        #  These labels serve as the initial values, and each operation sequence is assumed to be a tuple of
        #  operation labels.
        self.init_indices = []  # indices to put initial zero & single gate results
        for opLabel in self.opLabels:
            tup = () if opLabel == "" else (opLabel,)  # special case of empty label == no gate
            if tup in circuit_list:
                indx = circuit_list.index(tup)
                self[indx] = (None, None)  # iLeft = iRight = None for always-evaluated zero string
            else:
                indx = len(self)
                self.append((None, None))  # iLeft = iRight = None for always-evaluated zero string
            self.init_indices.append(indx)
            evalDict[tup] = indx

        #print("DB: initial eval dict = ",evalDict)

        #Process circuits in order of length, so that we always place short strings
        # in the right place (otherwise assert stmt below can fail)
        indices_sorted_by_circuit_len = \
            sorted(list(range(len(circuit_list))),
                   key=lambda i: len(circuit_list[i]))

        #avgBiteSize = 0
        #useCounts = {}
        for k in indices_sorted_by_circuit_len:
            circuit = circuit_list[k]
            L = len(circuit)
            if L == 0:
                iEmptyStr = evalDict.get((), None)
                assert(iEmptyStr is not None)  # duplicate () final strs require
                if k != iEmptyStr:            # the empty string to be included in the tree too!
                    assert(self[k] is None)
                    self[k] = (iEmptyStr, iEmptyStr)  # compute the duplicate () using by
                    self.eval_order.append(k)  # multiplying by the empty string.

            start = 0; bite = 1
            #nBites = 0
            #print("\nDB: string = ",circuit, "(len=%d)" % len(circuit))

            while start < L:

                #Take a bite out of circuit, starting at `start` that is in evalDict
                for b in range(L - start, 0, -1):
                    if circuit[start:start + b] in evalDict:
                        bite = b; break
                else: assert(False), ("EvalTree Error: probably caused because "
                                      "your operation sequences contain gates that your model does not")
                #Logic error - loop above should always exit when b == 1

                #iInFinal = k if bool(start + bite == L) else -1
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
                            assert(self[k] is None)  # make sure we haven't put anything here yet
                            self[k] = (iCur, iEmptyStr)  # compute the duplicate using by
                            self.eval_order.append(k)  # multiplying by the empty string.
                else:
                    # add (iCur, iBite)
                    assert(circuit[0:start + bite] not in evalDict)
                    iBite = evalDict[circuit[start:start + bite]]
                    if bFinal:  # place (iCur, iBite) at location k
                        iNew = k
                        evalDict[circuit[0:start + bite]] = iNew
                        assert(self[iNew] is None)  # make sure we haven't put anything here yet
                        self[k] = (iCur, iBite)
                    else:
                        iNew = len(self)
                        evalDict[circuit[0:start + bite]] = iNew
                        self.append((iCur, iBite))

                    #print("DB: add %s (index %d)" % (str(circuit[0:start+bite]),iNew))
                    self.eval_order.append(iNew)
                    iCur = iNew
                start += bite
                #nBites += 1

            #if nBites > 0: avgBiteSize += L / float(nBites)
            assert(k in self.eval_order or k in self.init_indices)

        #avgBiteSize /= float(len(circuit_list))
        #print "DEBUG: Avg bite size = ",avgBiteSize

        #see if there are superfluous tree nodes: those with iFinal == -1 and
        self.myFinalToParentFinalMap = None  # this tree has no "children",
        self.myFinalElsToParentFinalElsMap = None  # i.e. has not been created by a 'split'
        self.parentIndexMap = None
        self.original_index_lookup = None
        self.subTrees = []  # no subtrees yet
        assert(self.generate_circuit_list() == circuit_list)
        assert(None not in circuit_list)

    def cache_size(self):
        """
        Returns the size of the persistent "cache" of partial results
        used during the computation of all the strings in this tree.
        """
        return len(self)

    def generate_circuit_list(self, permute=True):
        """
        Generate a list of the final operation sequences this tree evaluates.

        This method essentially "runs" the tree and follows its
          prescription for sequentailly building up longer strings
          from shorter ones.  When permute == True, the resulting list
          should be the same as the one passed to initialize(...), and
          so this method may be used as a consistency check.

        Parameters
        ----------
        permute : bool, optional
           Whether to permute the returned list of strings into the
           same order as the original list passed to initialize(...).
           When False, the computed order of the operation sequences is
           given, which is matches the order of the results from calls
           to `Model` bulk operations.  Non-trivial permutation
           occurs only when the tree is split (in order to keep
           each sub-tree result a contiguous slice within the parent
           result).

        Returns
        -------
        list of gate-label-tuples
            A list of the operation sequences evaluated by this tree, each
            specified as a tuple of operation labels.
        """
        circuits = [None] * len(self)

        #Set "initial" (single- or zero- gate) strings
        for i, opLabel in zip(self.get_init_indices(), self.get_init_labels()):
            if opLabel == "": circuits[i] = ()  # special case of empty label
            else: circuits[i] = (opLabel,)

        #Build rest of strings
        for i in self.get_evaluation_order():
            iLeft, iRight = self[i]
            circuits[i] = circuits[iLeft] + circuits[iRight]

        #Permute to get final list:
        nFinal = self.num_final_strings()
        if self.original_index_lookup is not None and permute:
            finalCircuits = [None] * nFinal
            for iorig, icur in self.original_index_lookup.items():
                if iorig < nFinal: finalCircuits[iorig] = circuits[icur]
            assert(None not in finalCircuits)
            return finalCircuits
        else:
            assert(None not in circuits[0:nFinal])
            return circuits[0:nFinal]

    def get_min_tree_size(self):
        """
        Returns the minimum sub tree size required to compute each
        of the tree entries individually.  This minimum size is the
        smallest "maxSubTreeSize" that can be passed to split(),
        as any smaller value will result in at least one entry being
        uncomputable.
        """
        singleItemTreeSetList = self._createSingleItemTrees()
        return max(list(map(len, singleItemTreeSetList)))

    def split(self, elIndicesDict, maxSubTreeSize=None, numSubTrees=None, verbosity=0):
        """
        Split this tree into sub-trees in order to reduce the
          maximum size of any tree (useful for limiting memory consumption
          or for using multiple cores).  Must specify either maxSubTreeSize
          or numSubTrees.

        Parameters
        ----------
        elIndicesDict : dict
            A dictionary whose keys are integer original-circuit indices
            and whose values are slices or index arrays of final-element-
            indices (typically this dict is returned by calling
            :method:`Model.simplify_circuits`).  Since splitting a
            tree often involves permutation of the raw string ordering
            and thereby the element ordering, an updated version of this
            dictionary, with all permutations performed, is returned.

        maxSubTreeSize : int, optional
            The maximum size (i.e. list length) of each sub-tree.  If the
            original tree is smaller than this size, no splitting will occur.
            If None, then there is no limit.

        numSubTrees : int, optional
            The maximum size (i.e. list length) of each sub-tree.  If the
            original tree is smaller than this size, no splitting will occur.

        verbosity : int, optional
            How much detail to send to stdout.

        Returns
        -------
        OrderedDict
            A updated version of elIndicesDict
        """
        #dbList = self.generate_circuit_list()
        tm = _time.time()
        printer = _VerbosityPrinter.build_printer(verbosity)

        if (maxSubTreeSize is None and numSubTrees is None) or \
           (maxSubTreeSize is not None and numSubTrees is not None):
            raise ValueError("Specify *either* maxSubTreeSize or numSubTrees")
        if numSubTrees is not None and numSubTrees <= 0:
            raise ValueError("EvalTree split() error: numSubTrees must be > 0!")

        #Don't split at all if it's unnecessary
        if maxSubTreeSize is None or len(self) < maxSubTreeSize:
            if numSubTrees is None or numSubTrees == 1: return elIndicesDict

        self.subTrees = []
        printer.log("EvalTree.split done initial prep in %.0fs" %
                    (_time.time() - tm)); tm = _time.time()

        #First pass - identify which indices go in which subtree
        #   Part 1: create disjoint set of subtrees generated by single items
        singleItemTreeSetList = self._createSingleItemTrees()
        #each element represents a subtree, and
        # is a set of the indices owned by that subtree
        nSingleItemTrees = len(singleItemTreeSetList)

        printer.log("EvalTree.split created singles in %.0fs" %
                    (_time.time() - tm)); tm = _time.time()

        #   Part 2: determine whether we need to split/merge "single" trees
        if numSubTrees is not None:

            #Merges: find the best merges to perform if any are required
            if nSingleItemTrees > numSubTrees:

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

                #Another possible Algorithm (but was very slow...)
                #start_select_method = "fast"
                #if start_select_method == "best":
                #    availableIndices = list(range(nSingleItemTrees))
                #    i_min,_ = min( enumerate(  #index of a tree in the minimal intersection
                #            ( min((len(s1.intersection(s2)) for s2 in singleItemTreeSetList[i+1:]))
                #              for i,s1 in enumerate(singleItemTreeSetList[:-1]) )),
                #                   key=lambda x: x[1]) #argmin using generators (np.argmin doesn't work)
                #    iStartingTrees.append(i_min)
                #    startingTreeEls = singleItemTreeSetList[i_min].copy()
                #    del availableIndices[i_min]
                #
                #    while len(iStartingTrees) < numSubTrees:
                #        ii_min,_ = min( enumerate(
                #            ( len(startingTreeEls.intersection(singleItemTreeSetList[i]))
                #              for i in availableIndices )), key=lambda x: x[1]) #argmin
                #        i_min = availableIndices[ii_min]
                #        iStartingTrees.append(i_min)
                #        startingTreeEls.update( singleItemTreeSetList[i_min] )
                #        del availableIndices[ii_min]
                #
                #    printer.log("EvalTree.split found starting trees in %.0fs" %
                #                (_time.time()-tm)); tm = _time.time()
                #
                #elif start_select_method == "fast":

                def get_start_indices(maxIntersect):
                    """ Builds an initial set of indices by merging single-
                        item trees that don't intersect too much (intersection
                        is less than `maxIntersect`.  Returns a list of the
                        single-item tree indices and the final set of indices."""
                    starting = [0]  # always start with 0th tree
                    startingSet = singleItemTreeSetList[0].copy()
                    for i, s in enumerate(singleItemTreeSetList[1:], start=1):
                        if len(startingSet.intersection(s)) <= maxIntersect:
                            starting.append(i)
                            startingSet.update(s)
                    return starting, startingSet

                left, right = 0, max(map(len, singleItemTreeSetList))
                while left < right:
                    mid = (left + right) // 2
                    iStartingTrees, startingTreeEls = get_start_indices(mid)
                    nStartingTrees = len(iStartingTrees)
                    if nStartingTrees < numSubTrees:
                        left = mid + 1
                    elif nStartingTrees > numSubTrees:
                        right = mid
                    else: break  # nStartingTrees == numSubTrees!

                if len(iStartingTrees) < numSubTrees:
                    iStartingTrees, startingTreeEls = get_start_indices(mid + 1)
                if len(iStartingTrees) > numSubTrees:
                    iStartingTrees = iStartingTrees[0:numSubTrees]
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
                assert(len(subTreeSetList) == numSubTrees)

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

                assert(len(subTreeSetList) == numSubTrees)
                printer.log("EvalTree.split merged trees in %.0fs" %
                            (_time.time() - tm)); tm = _time.time()

            #Splits (more subtrees desired than there are single item trees!)
            else:
                #Splits: find the best splits to perform
                #TODO: how to split a tree intelligently -- for now, just do
                # trivial splits by making empty trees.
                subTreeSetList = singleItemTreeSetList[:]
                nSplitsNeeded = numSubTrees - nSingleItemTrees
                while nSplitsNeeded > 0:
                    # LATER...
                    # for iSubTree,subTreeSet in enumerate(subTreeSetList):
                    subTreeSetList.append([])  # create empty subtree
                    nSplitsNeeded -= 1

        else:
            assert(maxSubTreeSize is not None)
            subTreeSetList = []

            #Merges: find the best merges to perform if any are allowed given
            # the maximum tree size
            for singleItemTreeSet in singleItemTreeSetList:
                if len(singleItemTreeSet) > maxSubTreeSize:
                    raise ValueError("Max. sub tree size (%d) is too low (<%d)!"
                                     % (maxSubTreeSize, self.get_min_tree_size()))

                #See if we should merge this single-item-generated tree with
                # another one or make it a new subtree.
                newTreeSize = len(singleItemTreeSet)
                maxIntersectSize = None; iMaxIntersectSize = None
                for k, existingSubTreeSet in enumerate(subTreeSetList):
                    mergedSize = len(existingSubTreeSet) + newTreeSize
                    if mergedSize <= maxSubTreeSize:
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

        #TODO: improve tree efficiency via better splitting?
        # print "DEBUG TREE SPLITTING:"
        # for k,dbTreeSet in enumerate(subTreeSetList):
        #    print "Tree %d (size %d): " % (k,len(dbTreeSet)), \
        #        [ len(dbTreeSet.intersection(x)) for kk,x in enumerate(subTreeSetList) if kk != k ]
        # cnts = [0]*len(self)
        # for k,dbTreeSet in enumerate(subTreeSetList):
        #    for i in dbTreeSet:
        #        cnts[i] += 1
        # sorted_cnts = sorted( list(enumerate(cnts)), key=lambda x: x[1], reverse=True)
        # print "Top index : cnts"
        # for ii,(i,cnt) in enumerate(sorted_cnts):
        #    print ii,":", i,", ",cnt
        # raise ValueError("STOP")

        #bDebug = False
        #if bDebug: print("Parent nFinal = ",self.num_final_strings(), " len=",len(self))
        printer.log("EvalTree.split done first pass in %.0fs" %
                    (_time.time() - tm)); tm = _time.time()

        #Second pass - create subtrees from index sets
        # (common logic provided by base class up to providing a few helper fns)

        def permute_parent_element(perm, el):
            """Applies a permutation to an element of the tree """
            # perm[oldIndex] = newIndex
            return (perm[el[0]] if (el[0] is not None) else None,
                    perm[el[1]] if (el[1] is not None) else None)

        def create_subtree(parentIndices, numFinal, fullEvalOrder, sliceIntoParentsFinalArray, parentTree):
            """
            Creates a subtree given requisite information:

            Parameters
            ----------
            parentIndices : list
                The ordered list of (parent-tree) indices to be included in
                the created subtree.

            numFinal : int
                The number of "final" elements, i.e. those that are used to
                construct the final array of results and not just an intermediate.
                The first numFinal elemements of parentIndices are "final", and
                'sliceIntoParentsFinalArray' tells you which final indices of
                the parent they map to.

            fullEvalOrder : list
                A list of the integers between 0 and len(parentIndics)-1 which
                gives the evaluation order of the subtree *including* evaluation
                of any initial elements.

            sliceIntoParentsFinalArray : slice
                Described above - map between to-be-created subtree's final
                elements and parent-tree indices.

            parentTree : EvalTree
                The parent tree itself.
            """
            subTree = MatrixEvalTree()
            subTree.myFinalToParentFinalMap = sliceIntoParentsFinalArray
            subTree.num_final_strs = numFinal
            subTree[:] = [None] * len(parentIndices)

            mapParentIndxToSubTreeIndx = {k: ik for ik, k in enumerate(parentIndices)}

            for ik in fullEvalOrder:  # includes any initial indices
                k = parentIndices[ik]  # original tree index
                (oLeft, oRight) = parentTree[k]  # original tree indices

                if (oLeft is None) and (oRight is None):
                    iLeft = iRight = None
                    #assert(len(subTree.opLabels) == len(subTree)) #make sure all oplabel items come first
                    subTree.opLabels.append(parentTree.opLabels[
                        parentTree.init_indices.index(k)])
                    subTree.init_indices.append(ik)
                else:
                    iLeft = mapParentIndxToSubTreeIndx[oLeft]
                    iRight = mapParentIndxToSubTreeIndx[oRight]
                    subTree.eval_order.append(ik)

                assert(subTree[ik] is None)
                subTree[ik] = (iLeft, iRight)

                #if ik < subTreeNumFinal:
                #    assert(k < self.num_final_strings()) # it should be a final element in parent too!
                #    subTree.myFinalToParentFinalMap[ik] = k

            subTree.parentIndexMap = parentIndices  # parent index of *each* subtree index
            subTree.simplified_circuit_spamTuples = [self.simplified_circuit_spamTuples[k]
                                                     for k in _slct.indices(subTree.myFinalToParentFinalMap)]
            subTree.simplified_circuit_nEls = list(map(len, subTree.simplified_circuit_spamTuples))
            #subTree._compute_finalStringToEls() #depends on simplified_circuit_spamTuples

            final_el_startstops = []; i = 0
            for spamTuples in parentTree.simplified_circuit_spamTuples:
                final_el_startstops.append((i, i + len(spamTuples)))
                i += len(spamTuples)

            toConcat = [_np.arange(*final_el_startstops[k])
                        for k in _slct.indices(subTree.myFinalToParentFinalMap)]
            if len(toConcat) > 0:
                subTree.myFinalElsToParentFinalElsMap = _np.concatenate(toConcat)
            else:
                subTree.myFinalElsToParentFinalElsMap = _np.empty(0, _np.int64)
            #Note: myFinalToParentFinalMap maps only between *final* elements
            #   (which are what is held in simplified_circuit_spamTuples)

            subTree.num_final_els = sum([len(v) for v in subTree.simplified_circuit_spamTuples])
            subTree.recompute_spamtuple_indices(bLocal=False)

            return subTree

        updated_elIndices = self._finish_split(elIndicesDict, subTreeSetList,
                                               permute_parent_element, create_subtree)

        #print("PT5 = %.3fs" % (_time.time()-t0)); t0 = _time.time() # REMOVE
        printer.log("EvalTree.split done second pass in %.0fs" %
                    (_time.time() - tm)); tm = _time.time()

        return updated_elIndices

    def _walkSubTree(self, indx, out):
        if indx not in out: out.append(indx)
        (iLeft, iRight) = self[indx]
        if iLeft is not None: self._walkSubTree(iLeft, out)
        if iRight is not None: self._walkSubTree(iRight, out)

    def _createSingleItemTrees(self):
        #  Create disjoint set of subtrees generated by single items
        need_to_compute = _np.zeros(len(self), 'bool')
        need_to_compute[0:self.num_final_strings()] = True

        singleItemTreeSetList = []  # each element represents a subtree, and
        # is a set of the indices owned by that subtree
        for i in reversed(range(self.num_final_strings())):
            if not need_to_compute[i]: continue  # move to the last element
            #of evalTree that needs to be computed (i.e. is not in a subTree)

            subTreeIndices = []  # create subtree for uncomputed item
            self._walkSubTree(i, subTreeIndices)
            newTreeSet = set(subTreeIndices)
            for k in subTreeIndices:
                need_to_compute[k] = False  # mark all the elements of
                #the new tree as computed

            # Add this single-item-generated tree as a new subtree. Later
            #  we merge and/or split these trees based on constraints.
            singleItemTreeSetList.append(newTreeSet)
        return singleItemTreeSetList

    def get_analysis_plot_infos(self):
        """
        Returns debug plot information useful for
        assessing the quality of a tree. This
        function is not guaranteed to work.
        """

        analysis = {}
        firstIndxSeen = list(range(len(self)))
        lastIndxSeen = list(range(len(self)))
        subTreeSize = [-1] * len(self)

        xs = []; ys = []
        for i in range(len(self)):
            subTree = []
            self._walkSubTree(i, subTree)
            subTreeSize[i] = len(subTree)
            ys.extend([i] * len(subTree) + [None])
            xs.extend(list(sorted(subTree) + [None]))

            for k, t in enumerate(self):
                iLeft, iRight = t
                if i in (iLeft, iRight):
                    lastIndxSeen[i] = k

        analysis['SubtreeUsagePlot'] = {'xs': xs, 'ys': ys, 'title': "Indices used by the subtree rooted at each index",
                                        'xlabel': "Indices used", 'ylabel': 'Subtree root index'}
        analysis['SubtreeSizePlot'] = {'xs': list(range(len(self))),
                                       'ys': subTreeSize,
                                       'title': "Size of subtree rooted at each index",
                                       'xlabel': "Subtree root index",
                                       'ylabel': 'Subtree size'}

        xs = []; ys = []
        for i, rng in enumerate(zip(firstIndxSeen, lastIndxSeen)):
            ys.extend([i, i, None])
            xs.extend([rng[0], rng[1], None])
        analysis['IndexUsageIntervalsPlot'] = {'xs': xs, 'ys': ys, 'title': "Usage Intervals of each index",
                                               'xlabel': "Index Interval", 'ylabel': 'Index'}

        return analysis

    def copy(self):
        """ Create a copy of this evaluation tree. """
        newTree = self._copyBase(MatrixEvalTree(self[:]))
        newTree.opLabels = self.opLabels[:]
        newTree.init_indices = self.init_indices[:]
        newTree.simplified_circuit_spamTuples = self.simplified_circuit_spamTuples[:]
        #newTree.finalStringToElsMap = self.finalStringToElsMap[:]
        newTree.spamtuple_indices = self.spamtuple_indices.copy()
        return newTree

    def recompute_spamtuple_indices(self, bLocal=False):
        """
        Recompute this tree's `.spamtuple_indices` array.

        Parameters
        ----------
        bLocal : bool, optional
            If True, then the indices computed will index
            this tree's final array (even if it's a subtree).
            If False (the default), then a subtree's indices
            will index the *parent* tree's final array.

        Returns
        -------
        None
        """
        self.spamtuple_indices = _compute_spamtuple_indices(
            self.simplified_circuit_spamTuples,
            None if bLocal else self.myFinalElsToParentFinalElsMap)

    def _get_full_eval_order(self):
        """Includes init_indices in matrix-based evaltree case... HACK """
        return self.init_indices + self.eval_order

    def _update_eval_order_helpers(self, indexPermutation):
        """Update anything pertaining to the "full" evaluation order - e.g. init_inidces in matrix-based case (HACK)"""
        self.init_indices = [indexPermutation[iCur] for iCur in self.init_indices]

    def _update_element_indices(self, new_indices_in_old_order, old_indices_in_new_order, element_indices_dict):
        """
        Update any additional members because this tree's elements are being permuted.
        In addition, return an updated version of `element_indices_dict` a dict whose keys are
        the tree's (unpermuted) circuit indices and whose values are the final element indices for
        each circuit.
        """
        self.simplified_circuit_spamTuples, updated_elIndices = \
            self._permute_simplified_circuit_Xs(self.simplified_circuit_spamTuples,
                                                element_indices_dict, old_indices_in_new_order)
        self.simplified_circuit_nEls = list(map(len, self.simplified_circuit_spamTuples))
        self.recompute_spamtuple_indices(bLocal=True)  # bLocal shouldn't matter here - just for clarity

        return updated_elIndices


def _compute_spamtuple_indices(simplified_circuit_spamTuples,
                               subtreeFinalElsToParentFinalElsMap=None):
    """
    Returns a dictionary whose keys are the distinct spamTuples
    found in `simplified_circuit_spamTuples` and whose values are
    (finalIndices, finalTreeSlice) tuples where:

    finalIndices = the "element" indices in any final filled quantities
                   which combines both spam and gate-sequence indices.
                   If this tree is a subtree, then these final indices
                   refer to the *parent's* final elements if
                   `subtreeFinalElsToParentFinalElsMap` is given, otherwise
                   they refer to the subtree's final indices (usually desired).
    treeIndices = indices into the tree's final circuit list giving
                  all of the (raw) operation sequences which need to be computed
                  for the current spamTuple (this list has the SAME length
                  as finalIndices).
    """
    spamtuple_indices = _collections.OrderedDict(); el_off = 0
    for i, spamTuples in enumerate(  # i == final operation sequence index
            simplified_circuit_spamTuples):
        for j, spamTuple in enumerate(spamTuples, start=el_off):  # j == final element index
            if spamTuple not in spamtuple_indices:
                spamtuple_indices[spamTuple] = ([], [])
            f = subtreeFinalElsToParentFinalElsMap[j] \
                if (subtreeFinalElsToParentFinalElsMap is not None) else j  # parent's final
            spamtuple_indices[spamTuple][0].append(f)
            spamtuple_indices[spamTuple][1].append(i)
        el_off += len(spamTuples)

    def to_slice(x, maxLen=None):
        s = _slct.list_to_slice(x, array_ok=True, require_contiguous=False)
        if maxLen is not None and isinstance(s, slice) and (s.start, s.stop, s.step) == (0, maxLen, None):
            return slice(None, None)  # check for entire range
        else:
            return s

    nRawSequences = len(simplified_circuit_spamTuples)
    nElements = el_off if (subtreeFinalElsToParentFinalElsMap is None) \
        else None  # (we don't know how many els the parent has!)
    return _collections.OrderedDict(
        [(spamTuple, (to_slice(fInds, nElements), to_slice(gInds, nRawSequences)))
         for spamTuple, (fInds, gInds) in spamtuple_indices.items()])
