""" Defines the MapEvalTree class which implements an evaluation tree. """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import collections as _collections
import copy as _copy

from .verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from ..tools import slicetools as _slct
from .evaltree import EvalTree

import time as _time  # DEBUG TIMERS


class MapEvalTree(EvalTree):
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
        # list of the operation labels
        self.opLabels = []

        # Trivially init other members - to be filled in by initialize() or by subtree creation
        self.simplified_circuit_elabels = None
        self.element_offsets_for_circuit = None
        self.eLbl_indices_per_circuit = None
        self.final_indices_per_circuit = None
        self.rholabels = None
        self.cachesize = None

        super(MapEvalTree, self).__init__(items)

    def initialize(self, simplified_circuit_list, numSubTreeComms=1, maxCacheSize=None):
        """
          Initialize an evaluation tree using a set of operation sequences.
          This function must be called before using an EvalTree.

          Parameters
          ----------
          TODO: docstring update needed
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

        # opLabels : A list of all the distinct operation labels found in
        #              simplified_circuit_list.  Used in calc classes
        #              as a convenient precomputed quantity.
        self.opLabels = self._get_opLabels(simplified_circuit_list)
        if numSubTreeComms is not None:
            self.distribution['numSubtreeComms'] = numSubTreeComms

        circuit_list = [tuple(simple_circuit) for simple_circuit in simplified_circuit_list.keys()]
        self.simplified_circuit_elabels = list(simplified_circuit_list.values())
        self.simplified_circuit_nEls = list(map(len, self.simplified_circuit_elabels))
        self.element_offsets_for_circuit = _np.cumsum(
            [0] + [nEls for nEls in self.simplified_circuit_nEls])[:-1]
        self.elabels, self.eLbl_indices_per_circuit, self.final_indices_per_circuit = \
            self._build_elabels_lookups()

        self.rholabels = set()
        for c, elabels in simplified_circuit_list.items():
            if elabels != [None]:  # so we know c[0] is a prep label
                self.rholabels.add(c[0])
        self.rholabels = sorted(list(self.rholabels))

        self.num_final_els = sum([len(v) for v in self.simplified_circuit_elabels])
        #self._compute_finalStringToEls() #depends on simplified_circuit_spamTuples
        #UNNEEDED? self.recompute_spamtuple_indices(bLocal=True)  # bLocal shouldn't matter here

        #Evaluation tree:
        # A list of tuples, where each element contains
        #  information about evaluating a particular operation sequence:
        #  (iStart, tuple_of_following_gatelabels )
        # and self.eval_order specifies the evaluation order.
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

        #Sort the operation sequences "alphabetically", so that it's trivial to find common prefixes
        sorted_strs = sorted(list(enumerate(circuit_list)), key=lambda x: x[1])

        #DEBUG
        #print("SORTED"); print("\n".join(map(str,sorted_strs)))

        #PASS1: figure out what's worth keeping in the cache:
        if maxCacheSize is None or maxCacheSize > 0:
            curCacheSize = 0
            cacheIndices = []  # indices into circuit_list/self of the strings to cache
            dummy_self = [None] * self.num_final_strs; cache_hits = {}
            for k, (iStr, circuit) in enumerate(sorted_strs):
                L = len(circuit)
                for i in range(curCacheSize - 1, -1, -1):  # from curCacheSize-1 -> 0
                    candidate = circuit_list[cacheIndices[i]]
                    Lc = len(candidate)
                    if L >= Lc > 0 and circuit[0:Lc] == candidate:
                        iStart = i
                        remaining = circuit[Lc:]
                        if iStr in cache_hits: cache_hits[cacheIndices[i]] += 1  # tally cache hit
                        else: cache_hits[cacheIndices[i]] = 1  # TODO: use default dict?
                        break
                else:  # no break => no prefix
                    iStart = None
                    remaining = circuit[:]

                cacheIndices.append(iStr)
                iCache = curCacheSize
                curCacheSize += 1; assert(len(cacheIndices) == curCacheSize)
                dummy_self[iStr] = (iStart, remaining, iCache)

        #PASS #2: for real this time: construct tree but only cache items w/hits
        curCacheSize = 0
        cacheIndices = []  # indices into circuit_list/self of the strings to cache
        # (store persistently as prefixes of other string -- this need not be all
        #  of the strings in the tree)

        for k, (iStr, circuit) in enumerate(sorted_strs):
            L = len(circuit)

            #find longest existing prefix for circuit by working backwards
            # and finding the first string that *is* a prefix of this string
            # (this will necessarily be the longest prefix, given the sorting)
            for i in range(curCacheSize - 1, -1, -1):  # from curCacheSize-1 -> 0
                candidate = circuit_list[cacheIndices[i]]
                Lc = len(candidate)
                if L >= Lc > 0 and circuit[0:Lc] == candidate:  # ">=" allows for duplicates
                    iStart = i  # NOTE: this is an index into the *cache*, not necessarily self
                    remaining = circuit[Lc:]
                    break
            else:  # no break => no prefix
                iStart = None
                remaining = circuit[:]

            # if/where this string should get stored in the cache
            if (maxCacheSize is None or curCacheSize < maxCacheSize) and cache_hits.get(iStr, 0) > 0:
                cacheIndices.append(iStr)
                iCache = curCacheSize
                curCacheSize += 1; assert(len(cacheIndices) == curCacheSize)
            else:  # don't store in the cache
                iCache = None

            #Add info for this string
            self[iStr] = (iStart, remaining, iCache)
            self.eval_order.append(iStr)

        #FUTURE: could perform a second pass, and if there is
        # some threshold number of elements which share the
        # *same* iStart and the same beginning of the
        # 'remaining' part then add a new "extra" element
        # (beyond the #circuits index) which computes
        # the shared prefix and insert this into the eval
        # order.

        self.cachesize = curCacheSize
        self.myFinalToParentFinalMap = None  # this tree has no "children",
        self.myFinalElsToParentFinalElsMap = None  # i.e. has not been created by a 'split'
        self.parentIndexMap = None
        self.original_index_lookup = None
        self.subTrees = []  # no subtrees yet
        assert(self.generate_circuit_list() == circuit_list)
        assert(None not in circuit_list)

    def _remove_from_cache(self, indx):
        """ Removes self[indx] from cache (if it's in it)"""
        remStart, remRemain, remCache = self[indx]
        # iStart, remaining string, and iCache of element to remove
        if remCache is None: return  # not in cache to begin with!

        for i in range(len(self)):
            iStart, remainingStr, iCache = self[i]
            if iCache == remCache:
                assert(i == indx)
                self[i] = (iStart, remainingStr, None)  # not in cache anymore
                continue

            if iCache is not None and iCache > remCache:
                iCache -= 1  # shift left all cache indices after one removed
            if iStart == remCache:
                iStart = remStart
                remainingStr = remRemain + remainingStr
            elif iStart is not None and iStart > remCache:
                iStart -= 1  # shift left all cache indices after one removed
            self[i] = (iStart, remainingStr, iCache)
        self.cachesize -= 1

    def _build_elabels_lookups(self):
        all_elabels = set()
        for elabels in self.simplified_circuit_elabels:
            all_elabels.update(elabels)

        #Convert set of unique effect labels to a list so ordering is fixed
        all_elabels = sorted(list(all_elabels))

        # Create lookup so elabel_lookup[eLbl] gives index of eLbl
        # within all_elabels (for faster lookup, Cython routines in ptic)
        elabel_lookup = {elbl: i for i, elbl in enumerate(all_elabels)}

        #Create arrays that tell us, for a given rholabel, what the elabel indices
        # are for each simplified circuit.  This is obviously convenient for computing
        # outcome probabilities.
        eLbl_indices_per_circuit = {}
        final_indices_per_circuit = {}
        for i, elabels in enumerate(self.simplified_circuit_elabels):
            element_offset = self.element_offsets_for_circuit[i]  # offset to i-th simple circuits elements
            for j, eLbl in enumerate(elabels):
                if i in eLbl_indices_per_circuit:
                    eLbl_indices_per_circuit[i].append(elabel_lookup[eLbl])
                    final_indices_per_circuit[i].append(element_offset + j)
                else:
                    eLbl_indices_per_circuit[i] = [elabel_lookup[eLbl]]
                    final_indices_per_circuit[i] = [element_offset + j]

        return all_elabels, eLbl_indices_per_circuit, final_indices_per_circuit

    def squeeze(self, maxCacheSize):
        """
        Remove items from cache (if needed) so it contains less than or equal
        to `maxCacheSize` elements.

        Paramteters
        -----------
        maxCacheSize : int

        Returns
        -------
        None
        """
        assert(maxCacheSize >= 0)

        if maxCacheSize == 0:  # special but common case
            curCacheSize = self.cache_size()
            cacheinds = [None] * curCacheSize
            for i in self.get_evaluation_order():
                iStart, remainingStr, iCache = self[i]
                if iStart is not None:
                    remainingStr = self[cacheinds[iStart]][1] + remainingStr
                if iCache is not None:
                    cacheinds[iCache] = i
                self[i] = (None, remainingStr, None)
            self.cachesize = 0
            return

        #Otherwise, if maxCacheSize > 0, remove cache elements one at a time:
        while self.cache_size() > maxCacheSize:

            #Figure out what's in cache and # of times each one is hit
            curCacheSize = self.cache_size()
            hits = [0] * curCacheSize
            cacheinds = [None] * curCacheSize
            for i in range(len(self)):
                iStart, remainingStr, iCache = self[i]
                if iStart is not None: hits[iStart] += 1
                if iCache is not None: cacheinds[iCache] = i

            #Find a min-cost item to remove
            minCost = None; iMinTree = None; iMinCache = None
            for i in range(curCacheSize):
                cost = hits[i] * len(self[cacheinds[i]][1])
                # hits * len(remainder) ~= # more applies if we
                # remove i-th cache element.
                if iMinTree is None or cost < minCost:
                    minCost = cost; iMinTree = cacheinds[i]
                    iMinCache = i  # in cache
            assert(self[iMinTree][2] == iMinCache)  # sanity check

            #Remove references to iMin element
            self._remove_from_cache(iMinTree)

    def trim_nonfinal_els(self):
        """
        Removes from this tree all non-final elements (used to facilitate
        computation sometimes)
        """
        nFinal = self.num_final_strings()
        self._delete_els(list(range(nFinal, len(self))))

        #remove any unreferenced cache elements
        curCacheSize = self.cache_size()
        hits = [0] * curCacheSize
        cacheinds = [None] * curCacheSize
        for i in range(len(self)):
            iStart, remainingStr, iCache = self[i]
            if iStart is not None: hits[iStart] += 1
            if iCache is not None: cacheinds[iCache] = i
        for hits, i in zip(hits, cacheinds):
            if hits == 0: self._remove_from_cache(i)

    def _delete_els(self, elsToRemove):
        """
        Delete a self[i] for i in elsToRemove.
        """
        if len(elsToRemove) == 0: return

        last = elsToRemove[0]
        for i in elsToRemove[1:]:
            assert(i > last), "elsToRemove *must* be sorted in ascending order!"
            last = i

        #remove from cache
        for i in elsToRemove:
            self._remove_from_cache(i)

        order = self.eval_order
        for i in reversed(elsToRemove):
            del self[i]  # remove from self

            #remove & update eval order
            order = [((k - 1) if k > i else k) for k in order if k != i]
        self.eval_order = order

    def cache_size(self):
        """
        Returns the size of the persistent "cache" of partial results
        used during the computation of all the strings in this tree.
        """
        return self.cachesize

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

        cachedStrings = [None] * self.cache_size()

        #Build rest of strings
        for i in self.get_evaluation_order():
            iStart, remainingStr, iCache = self[i]
            if iStart is None:
                circuits[i] = remainingStr
            else:
                circuits[i] = cachedStrings[iStart] + remainingStr

            if iCache is not None:
                cachedStrings[iCache] = circuits[i]

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

    def get_num_applies(self):
        """
        Gets the number of "apply" operations required to compute this tree.

        Returns
        -------
        int
        """
        ops = 0
        for _, remainder, _ in self:
            ops += len(remainder)
        return ops

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
        evalOrder = self.get_evaluation_order()
        printer.log("EvalTree.split done initial prep in %.0fs" %
                    (_time.time() - tm)); tm = _time.time()

        def nocache_create_equal_size_subtrees():
            """ A shortcut for special case when there is no cache so each
                circuit can be evaluated independently """
            N = len(self)
            subTrees = [set(range(i, N, numSubTrees)) for i in range(numSubTrees)]
            totalCost = N
            return subTrees, totalCost

        def create_subtrees(maxCost, maxCostRate=0, costMetric="size"):
            """
            Find a set of subtrees by iterating through the tree
            and placing "break" points when the cost of evaluating the
            subtree exceeds some 'maxCost'.  This ensure ~ equal cost
            trees, but doesn't ensure any particular number of them.

            maxCostRate can be set to implement a varying maxCost
            over the course of the iteration.
            """

            if costMetric == "applys":
                def cost_fn(rem): return len(rem)  # length of remainder = #-apply ops needed
            elif costMetric == "size":
                def cost_fn(rem): return 1  # everything costs 1 in size of tree
            else: raise ValueError("Uknown cost metric: %s" % costMetric)

            subTrees = []
            curSubTree = set([evalOrder[0]])
            curTreeCost = cost_fn(self[evalOrder[0]][1])  # remainder length of 0th evaluant
            totalCost = 0
            cacheIndices = [None] * self.cache_size()

            for k in evalOrder:
                iStart, remainder, iCache = self[k]

                if iCache is not None:
                    cacheIndices[iCache] = k

                #compute the cost (additional #applies) which results from
                # adding this element to the current tree.
                cost = cost_fn(remainder)
                inds = set([k])

                if iStart is not None and cacheIndices[iStart] not in curSubTree:
                    #we need to add the tree elements traversed by
                    #following iStart
                    j = iStart  # index into cache
                    while j is not None:
                        iStr = cacheIndices[j]  # cacheIndices[ iStart ]
                        inds.add(iStr)
                        cost += cost_fn(self[iStr][1])  # remainder
                        j = self[iStr][0]  # iStart

                if curTreeCost + cost < maxCost:
                    #Just add current string to current tree
                    curTreeCost += cost
                    curSubTree.update(inds)
                else:
                    #End the current tree and begin a new one
                    #print("cost %d+%d exceeds %d" % (curTreeCost,cost,maxCost))
                    subTrees.append(curSubTree)
                    curSubTree = set([k])

                    cost = cost_fn(remainder); j = iStart
                    while j is not None:  # always traverse back iStart
                        iStr = cacheIndices[j]
                        curSubTree.add(iStr)
                        cost += cost_fn(self[iStr][1])  # remainder
                        j = self[iStr][0]  # iStart
                    totalCost += curTreeCost
                    curTreeCost = cost
                    #print("Added new tree w/initial cost %d" % (cost))

                maxCost += maxCostRate

            subTrees.append(curSubTree)
            totalCost += curTreeCost
            return subTrees, totalCost

        ##################################################################
        # Part I: find a list of where the current tree should be broken #
        ##################################################################

        if numSubTrees is not None and self.cache_size() == 0:
            #print("Split: EQUAL SUBTREES!") #REMOVE
            subTreeSetList, totalCost = nocache_create_equal_size_subtrees()
            #printer.log("EvalTree.split PT1 %.1fs" %
            #            (_time.time()-tm)); tm = _time.time()  #REMOVE

        elif numSubTrees is not None:

            #OLD METHOD: optimize max-cost to get the right number of trees
            # (but this can yield trees with unequal lengths or cache sizes,
            # which is what we're often after for memory reasons)
            costMet = "size"  # cost metric
            if costMet == "applies":
                maxCost = self.get_num_applies() / numSubTrees
            else: maxCost = len(self) / numSubTrees
            maxCostLowerBound, maxCostUpperBound = maxCost, None
            maxCostRate, rateLowerBound, rateUpperBound = 0, -1.0, +1.0
            #OLD (& incorrect) vals were 0, -1.0/len(self), +1.0/len(self),
            #   though current -1,1 vals are probably overly conservative...
            resultingSubtrees = numSubTrees + 1  # just to prime the loop
            iteration = 0

            #Iterate until the desired number of subtrees have been found.
            while resultingSubtrees != numSubTrees:
                subTreeSetList, totalCost = create_subtrees(maxCost, maxCostRate, costMet)
                resultingSubtrees = len(subTreeSetList)
                #print("DEBUG: resulting numTrees = %d (cost %g) w/maxCost = %g [%s,%s] & rate = %g [%g,%g]" % \
                #     (resultingSubtrees, totalCost, maxCost, str(maxCostLowerBound), str(maxCostUpperBound),
                #      maxCostRate, rateLowerBound, rateUpperBound))

                #DEBUG
                #totalSet = set()
                #for s in subTreeSetList:
                #    totalSet.update(s)
                #print("DB: total set length = ",len(totalSet))
                #assert(len(totalSet) == len(self))

                #Perform binary search in maxCost then maxCostRate to find
                # desired final subtree count.
                if maxCostUpperBound is None or abs(maxCostLowerBound - maxCostUpperBound) > 1.0:
                    # coarse adjust => vary maxCost
                    last_maxCost = maxCost
                    if resultingSubtrees <= numSubTrees:  # too few trees: reduce maxCost
                        maxCost = (maxCost + maxCostLowerBound) / 2.0
                        maxCostUpperBound = last_maxCost
                    else:  # too many trees: raise maxCost
                        if maxCostUpperBound is None:
                            maxCost = totalCost  # / numSubTrees
                        else:
                            maxCost = (maxCost + maxCostUpperBound) / 2.0
                            maxCostLowerBound = last_maxCost
                else:
                    # fine adjust => vary maxCostRate
                    last_maxRate = maxCostRate
                    if resultingSubtrees <= numSubTrees:  # too few trees reduce maxCostRate
                        maxCostRate = (maxCostRate + rateLowerBound) / 2.0
                        rateUpperBound = last_maxRate
                    else:  # too many trees: increase maxCostRate
                        maxCostRate = (maxCostRate + rateUpperBound) / 2.0
                        rateLowerBound = last_maxRate

                iteration += 1
                assert(iteration < 100), "Unsuccessful splitting for 100 iterations!"

        else:  # maxSubTreeSize is not None
            subTreeSetList, totalCost = create_subtrees(
                maxSubTreeSize, maxCostRate=0, costMetric="size")

        ##########################################################
        # Part II: create subtrees from index sets
        ##########################################################
        # (common logic provided by base class up to providing a few helper fns)

        def permute_parent_element(perm, el):
            """Applies a permutation to an element of the tree """
            # perm[oldIndex] = newIndex
            #return (perm[el[0]] if (el[0] is not None) else None, el[1], el[2])
            return (el[0], el[1], el[2])  # no need to permute the cache element ([0])

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
                A list of the integers between 0 and len(parentIndices)-1 which
                gives the evaluation order of the subtree *including* evaluation
                of any initial elements.

            sliceIntoParentsFinalArray : slice
                Described above - map between to-be-created subtree's final
                elements and parent-tree indices.

            parentTree : EvalTree
                The parent tree itself.
            """
            #t0 = _time.time() #REMOVE
            subTree = MapEvalTree()
            subTree.myFinalToParentFinalMap = sliceIntoParentsFinalArray
            subTree.num_final_strs = numFinal
            subTree[:] = [None] * len(parentIndices)

            curCacheSize = 0
            subTreeCacheIndices = {}

            for ik in fullEvalOrder:  # includes any initial indices
                k = parentIndices[ik]  # original tree index

                oStart, remainder, oCache = self[k]  # original tree data

                if oCache is not None:  # this element was in parent's cache,
                    subTreeCacheIndices[oCache] = curCacheSize  # maps parent's cache indices to subtree's
                    iCache = curCacheSize
                    curCacheSize += 1
                else:
                    iCache = None

                iStart = None if (oStart is None) else \
                    subTreeCacheIndices[oStart]
                subTree.eval_order.append(ik)

                assert(subTree[ik] is None)
                subTree[ik] = (iStart, remainder, iCache)

            #t1 = _time.time()  #REMOVE
            subTree.cachesize = curCacheSize
            subTree.parentIndexMap = parentIndices  # parent index of each subtree index
            subTree.simplified_circuit_elabels = [self.simplified_circuit_elabels[k]
                                                  for k in _slct.indices(subTree.myFinalToParentFinalMap)]
            subTree.simplified_circuit_nEls = list(map(len, subTree.simplified_circuit_elabels))
            subTree.element_offsets_for_circuit = _np.cumsum(
                [0] + [len(elabelList) for elabelList in subTree.simplified_circuit_elabels])[:-1]
            subTree.elabels, subTree.eLbl_indices_per_circuit, subTree.final_indices_per_circuit = \
                subTree._build_elabels_lookups()
            # don't bother trying to thin this out for now - just take the parent's list
            subTree.rholabels = self.rholabels
            #subTree._compute_finalStringToEls() #depends on simplified_circuit_spamTuples

            #t2 = _time.time() #REMOVE
            final_el_startstops = []; i = 0
            for nEls in parentTree.simplified_circuit_nEls:
                final_el_startstops.append((i, i + nEls))
                i += nEls
            #t3 = _time.time() #REMOVE
            if len(_slct.indices(subTree.myFinalToParentFinalMap)) > 0:
                subTree.myFinalElsToParentFinalElsMap = _np.concatenate(
                    [_np.arange(*final_el_startstops[k])
                     for k in _slct.indices(subTree.myFinalToParentFinalMap)])
                #Note: myFinalToParentFinalMap maps only between *final* elements
                #   (which are what is held in simplified_circuit_spamTuples)
            else:  # no final elements (a "dummy" tree, useful just to keep extra procs busy)
                subTree.myFinalElsToParentFinalElsMap = _np.arange(0, 0)  # empty array

            #t4 = _time.time() #REMOVE
            subTree.num_final_els = sum([len(v) for v in subTree.simplified_circuit_elabels])
            #t5 = _time.time() #REMOVE
            #UNNEEDED? subTree.recompute_spamtuple_indices(bLocal=False) # REMOVE
            #t6 = _time.time() #REMOVE

            subTree.trim_nonfinal_els()
            #t7 = _time.time() #REMOVE
            circuits = subTree.generate_circuit_list(permute=False)
            subTree.opLabels = self._get_opLabels(
                {c: elbls for c, elbls in zip(circuits, subTree.simplified_circuit_elabels)})

            #t8 = _time.time() #REMOVE
            # print("DB: create_subtree timing: "
            #       "t1=%.3fs, t2=%.3fs, t3=%.3fs, t4=%.3fs, t5=%.3fs, t6=%.3fs, t7=%.3fs, t8=%.3fs"
            #       % (t1-t0,t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t7-t6,t8-t7))

            return subTree

        #printer.log("EvalTree.split PT2 %.1fs" %
        #            (_time.time()-tm)); tm = _time.time()  #REMOVE
        updated_elIndices = self._finish_split(elIndicesDict, subTreeSetList,
                                               permute_parent_element, create_subtree,
                                               all_final=bool(self.cache_size() == 0))

        printer.log("EvalTree.split done second pass in %.0fs" %
                    (_time.time() - tm)); tm = _time.time()
        return updated_elIndices

    def _update_element_indices(self, new_indices_in_old_order, old_indices_in_new_order, element_indices_dict):
        """
        Update any additional members because this tree's elements are being permuted.
        In addition, return an updated version of `element_indices_dict` a dict whose keys are
        the tree's (unpermuted) circuit indices and whose values are the final element indices for
        each circuit.
        """
        self.simplified_circuit_elabels, updated_elIndices = \
            self._permute_simplified_circuit_Xs(self.simplified_circuit_elabels,
                                                element_indices_dict, old_indices_in_new_order)
        self.simplified_circuit_nEls = list(map(len, self.simplified_circuit_elabels))

        # Update element_offsets_for_circuit, etc
        self.element_offsets_for_circuit = _np.cumsum(
            [0] + [len(elabelList) for elabelList in self.simplified_circuit_elabels])[:-1]
        self.elabels, self.eLbl_indices_per_circuit, self.final_indices_per_circuit = \
            self._build_elabels_lookups()

        return updated_elIndices

    def copy(self):
        """ Create a copy of this evaluation tree. """
        cpy = self._copyBase(MapEvalTree(self[:]))
        cpy.cachesize = self.cachesize  # member specific to MapEvalTree
        cpy.opLabels = self.opLabels[:]
        cpy.simplified_circuit_elabels = _copy.deepcopy(self.simplified_circuit_elabels)
        cpy.element_offsets_for_circuit = self.element_offsets_for_circuit.copy()
        cpy.elabels = self.elabels[:]
        cpy.eLbl_indices_per_circuit = _copy.deepcopy(self.eLbl_indices_per_circuit)
        cpy.final_indices_per_circuit = _copy.deepcopy(self.final_indices_per_circuit)
        cpy.rholabels = self.rholabels[:]
        return cpy
