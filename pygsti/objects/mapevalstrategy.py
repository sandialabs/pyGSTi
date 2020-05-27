"""
Defines the MapEvalTree class which implements an evaluation tree.
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
import collections as _collections
import copy as _copy

from .verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from ..tools import slicetools as _slct
from .evaltree import EvalTree

import time as _time  # DEBUG TIMERS


def _create_prefix_table(circuits_to_evaluate, max_cache_size):
    """
    Creates a "prefix table" for evaluating a set of circuits.

    The table is list of tuples, where each element contains
    instructions for evaluating a particular operation sequence:
    
    (iDest, iStart, tuple_of_following_items, iCache)

    Means that circuit[iDest] = cached_circuit[iStart] + tuple_of_following_items,
    and that the resulting state should be stored at cache index iCache (for
    later reference as an iStart value).  The ordering of the returned list
    specifies the evaluation order.

    `iDest` is always in the range [0,len(circuits_to_evaluate)-1], and
    indexes the result computed for each of the circuits.

    Returns
    -------
    tuple
        A tuple of `(table_contents, cache_size)` where `table_contents` is a list
        of tuples as given above and `cache_size` is the total size of the state
        cache used to hold intermediate results.
    """
    #Sort the operation sequences "alphabetically", so that it's trivial to find common prefixes
    sorted_circuits_to_evaluate = sorted(list(enumerate(circuits_to_evaluate)), key=lambda x: x[1])

    if max_cache_size is None or max_cache_size > 0:
        #CACHE assessment pass: figure out what's worth keeping in the cache.
        # In this pass, we cache *everything* and keep track of how many times each
        # original index (after it's cached) is utilized as a prefix for another circuit.
        # Not: this logic could be much better, e.g. computing a cost savings for each
        #  potentially-cached item and choosing the best ones, and proper accounting
        #  for chains of cached items.
        cacheIndices = []  # indices into circuits_to_evaluate of the results to cache
        cache_hits = _collections.defaultdict(lambda: 0)
        for i, circuit in sorted_circuits_to_evaluate:
            L = len(circuit)
            for cached_index in reversed(cacheIndices):
                candidate = circuits_to_evaluate[cached_index]
                Lc = len(candidate)
                if L >= Lc > 0 and circuit[0:Lc] == candidate:  # a cache hit!
                    cache_hits[cached_index] += 1
                    break  # stop looking through cache
            cacheIndices.append(i)  # cache *everything* in this pass

    # Build prefix table: construct list, only caching items with hits > 0 (up to max_cache_size)
    cacheIndices = []  # indices into circuits_to_evaluate of the results to cache
    table_contents = []
    curCacheSize = 0

    for i, circuit in sorted_circuits_to_evaluate:
        L = len(circuit)

        #find longest existing prefix for circuit by working backwards
        # and finding the first string that *is* a prefix of this string
        # (this will necessarily be the longest prefix, given the sorting)
        for i_in_cache in range(curCacheSize - 1, -1, -1):  # from curCacheSize-1 -> 0
            candidate = circuits_to_evaluate[cacheIndices[i_in_cache]]
            Lc = len(candidate)
            if L >= Lc > 0 and circuit[0:Lc] == candidate:  # ">=" allows for duplicates
                iStart = i_in_cache  # an index into the *cache*, not into circuits_to_evaluate
                remaining = circuit[Lc:]
                break
        else:  # no break => no prefix
            iStart = None
            remaining = circuit[:]

        # if/where this string should get stored in the cache
        if (max_cache_size is None or curCacheSize < max_cache_size) and cache_hits.get(i, 0) > 0:
            iCache = len(cacheIndices)
            cacheIndices.append(i); curCacheSize += 1
        else:  # don't store in the cache
            iCache = None

        #Add instruction for computing this circuit
        table_contents.append((i, iStart, remaining, iCache))

    #FUTURE: could perform a second pass, and if there is
    # some threshold number of elements which share the
    # *same* iStart and the same beginning of the
    # 'remaining' part then add a new "extra" element
    # (beyond the #circuits index) which computes
    # the shared prefix and insert this into the eval
    # order.

    return table_contents, curCacheSize


def _find_splitting(prefix_table, max_sub_table_size=None, num_sub_tables=None, verbosity=0):
    """
    Find a partition of the indices of `prefix_table` to define a set of sub-tables with the desire properties.

    This is done in order to reduce the maximum size of any tree (useful for
    limiting memory consumption or for using multiple cores).  Must specify
    either max_sub_tree_size or num_sub_trees.

    Parameters
    ----------
    prefix_table : tuple
        A prefix table. TODO: docstring - more detail.

    max_sub_table_size : int, optional
        The maximum size (i.e. list length) of each sub-table.  If the
        original table is smaller than this size, no splitting will occur.
        If None, then there is no limit.

    num_sub_tables : int, optional
        The maximum size (i.e. list length) of each sub-table.  If the
        original table is smaller than this size, no splitting will occur.

    verbosity : int, optional
        How much detail to send to stdout.

    Returns
    -------
    list
        A list of sets of elements to place in sub-tables.
    """
    tm = _time.time()
    printer = _VerbosityPrinter.build_printer(verbosity)
    table_contents, cachesize = prefix_table

    if (max_sub_table_size is None and num_sub_tables is None) or \
       (max_sub_table_size is not None and num_sub_tables is not None):
        raise ValueError("Specify *either* max_sub_table_size or num_sub_tables")
    if num_sub_tables is not None and num_sub_tables <= 0:
        raise ValueError("Error: num_sub_tables must be > 0!")

    #Don't split at all if it's unnecessary
    if max_sub_table_size is None or len(self) < max_sub_table_size:
        if num_sub_tables is None or num_sub_tables == 1:
            return [set(range(len(table_contents)))]

    def nocache_create_equal_size_subtables():
        """ A shortcut for special case when there is no cache so each
            circuit can be evaluated independently """
        N = len(table_contents)
        subTables = [set(range(i, N, num_sub_tables)) for i in range(num_sub_tables)]
        totalCost = N
        return subTables, totalCost

    def create_subtables(max_cost, max_cost_rate=0, cost_metric="size"):
        """
        Find a set of subtables by iterating through the table
        and placing "break" points when the cost of evaluating the
        subtable exceeds some 'max_cost'.  This ensure ~ equal cost
        tables, but doesn't ensure any particular number of them.

        max_cost_rate can be set to implement a varying max_cost
        over the course of the iteration.
        """

        if cost_metric == "applys":
            def cost_fn(rem): return len(rem)  # length of remainder = #-apply ops needed
        elif cost_metric == "size":
            def cost_fn(rem): return 1  # everything costs 1 in size of table
        else: raise ValueError("Uknown cost metric: %s" % cost_metric)

        subTables = []
        curSubTable = set([table_contents[0][0]])  # destination index of 0th evaluant
        curTableCost = cost_fn(table_contents[0][2])  # remainder length of 0th evaluant
        totalCost = 0
        cacheIndices = [None] * cachesize
        contents_by_idest = {tup[0]: tup for tup in table_contents}  # for fast lookup by a circuit index

        def traverse_index(iDest, iStart, remainder, current_table):
            """Get the set of indices you'd need to add along with and including `k`, and their cost."""
            cost = cost_fn(remainder)
            inds = set([iDest])

            if iStart is not None and cacheIndices[iStart] not in current_table:
                #we need to add the table elements traversed by following iStart
                j = iStart  # index into cache
                while j is not None:
                    j_circuit = cacheIndices[j]  # cacheIndices[ iStart ]
                    inds.add(j_circuit)
                    _, jStart, jrem, _ = contents_by_idest[j_circuit]
                    cost += cost_fn(jrem)  # remainder
                    j = jStart
            return inds, cost

        for iDest, iStart, remainder, iCache in table_contents:
            if iCache is not None:
                cacheIndices[iCache] = iDest

            #compute the cost (additional #applies) which results from
            # adding this element (iDest) to the current sub-table.
            inds, cost = traverse_index(iDest, iStart, remainder)

            if curTableCost + cost < max_cost:
                #Just add current string to current table
                curTableCost += cost
                curSubTable.update(inds)
            else:
                #End the current table and begin a new one
                #print("cost %d+%d exceeds %d" % (curTableCost,cost,max_cost))
                subTables.append(curSubTable)
                curSubTable, curTableCost = traverse_index(iDest, iStart, remainder)
                #print("Added new table w/initial cost %d" % (cost))

            max_cost += max_cost_rate

        subTables.append(curSubTable)
        totalCost += curTableCost
        return subTables, totalCost

    ##################################################################
    # Find a list of where the current table should be broken        #
    ##################################################################

    if num_sub_tables is not None and self.cache_size() == 0:
        subTableSetList, totalCost = nocache_create_equal_size_subtables()

    elif num_sub_tables is not None:

        #Optimize max-cost to get the right number of tables
        # (but this can yield tables with unequal lengths or cache sizes,
        # which is what we're often after for memory reasons)
        if cost_metric == "applies":
            maxCost = self.get_num_applies() / num_sub_tables
        else: maxCost = len(self) / num_sub_tables
        maxCostLowerBound, maxCostUpperBound = maxCost, None
        maxCostRate, rateLowerBound, rateUpperBound = 0, -1.0, +1.0
        #OLD (& incorrect) vals were 0, -1.0/len(self), +1.0/len(self),
        #   though current -1,1 vals are probably overly conservative...
        resultingSubtables = num_sub_tables + 1  # just to prime the loop
        iteration = 0

        #Iterate until the desired number of subtables have been found.
        while resultingSubtables != num_sub_tables:
            subTableSetList, totalCost = create_subtables(maxCost, maxCostRate, costMet)
            resultingSubtables = len(subTableSetList)
            #print("DEBUG: resulting numTables = %d (cost %g) w/maxCost = %g [%s,%s] & rate = %g [%g,%g]" % \
            #     (resultingSubtables, totalCost, maxCost, str(maxCostLowerBound), str(maxCostUpperBound),
            #      maxCostRate, rateLowerBound, rateUpperBound))

            #DEBUG
            #totalSet = set()
            #for s in subTableSetList:
            #    totalSet.update(s)
            #print("DB: total set length = ",len(totalSet))
            #assert(len(totalSet) == len(self))

            #Perform binary search in maxCost then maxCostRate to find
            # desired final subtable count.
            if maxCostUpperBound is None or abs(maxCostLowerBound - maxCostUpperBound) > 1.0:
                # coarse adjust => vary maxCost
                last_maxCost = maxCost
                if resultingSubtables <= num_sub_tables:  # too few tables: reduce maxCost
                    maxCost = (maxCost + maxCostLowerBound) / 2.0
                    maxCostUpperBound = last_maxCost
                else:  # too many tables: raise maxCost
                    if maxCostUpperBound is None:
                        maxCost = totalCost  # / num_sub_tables
                    else:
                        maxCost = (maxCost + maxCostUpperBound) / 2.0
                        maxCostLowerBound = last_maxCost
            else:
                # fine adjust => vary maxCostRate
                last_maxRate = maxCostRate
                if resultingSubtables <= num_sub_tables:  # too few tables reduce maxCostRate
                    maxCostRate = (maxCostRate + rateLowerBound) / 2.0
                    rateUpperBound = last_maxRate
                else:  # too many tables: increase maxCostRate
                    maxCostRate = (maxCostRate + rateUpperBound) / 2.0
                    rateLowerBound = last_maxRate

            iteration += 1
            assert(iteration < 100), "Unsuccessful splitting for 100 iterations!"

    else:  # max_sub_table_size is not None
        subTableSetList, totalCost = create_subtables(
            max_sub_table_size, max_cost_rate=0, cost_metric="size")

    return subTableSetList


#TODO: update this or REMOVE it -- maybe move to unit tests?
#def _check_prefix_table(prefix_table):  #generate_circuit_list(self, permute=True):
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
#    cachedStrings = [None] * self.cache_size()
#
#    #Build rest of strings
#    for i in self.get_evaluation_order():
#        iStart, remainingStr, iCache = self[i]
#        if iStart is None:
#            circuits[i] = remainingStr
#        else:
#            circuits[i] = cachedStrings[iStart] + remainingStr
#
#        if iCache is not None:
#            cachedStrings[iCache] = circuits[i]
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


class MapCOPAEvalStrategy(_COPAEvalStrategy):
    """
    TODO: docstring (update)
    An Evaluation Tree that structures a circuit list for map-based calculations.

    Parameters
    ----------
    simplified_circuit_elabels : dict
        A dictionary of `(circuit, elabels)` tuples specifying
        the circuits that should be present in the evaluation tree.
        `circuit` is a *simplified* circuit whose first layer is a
        preparation label. `elabels` is a list of all the POVM
        effect labels (corresponding to outcomes) for the
        circuit (only a single label is needed rather than a
        POVM-label, effect-label pair because these are *simplified*
        effect labels).

    num_sub_tree_comms : int, optional
        The number of processor groups (communicators)
        to divide the subtrees of this EvalTree among
        when calling `distribute`.  By default, the
        communicator is not divided.

    max_cache_size : int, optional
        Maximum cache size, used for holding common circuit prefixes.
    """

    def __init__(self, circuits, model_shlp, dataset=None, num_sub_table_comms=1, max_cache_size=None,
                 max_sub_table_size=None, num_sub_tables=None, verbosity=0):

        unique_circuits, to_unique = self._compute_unique_circuits(circuits)
        aliases = circuits.op_label_aliases if isinstance(circuits, _BulkCircuitList) else None
        ds_circuits = _lt.apply_aliases_to_circuit_list(unique_circuits, aliases)
        unique_complete_circuits = [model_shlp.complete_circuit(c) for c in unique_circuits]

        circuit_table = _create_prefix_table(unique_complete_circuits, max_cache_size)
        groups = self._find_splitting(circuit_table, max_sub_table_size, num_sub_tables, verbosity)  # a list of tuples/sets?

        self.tables = []
        elindex_outcome_tuples = {orig_i: list() for orig_i in range(len(unique_circuits))}

        offset = 0
        for group in groups:

            expanded_circuit_outcomes_by_orig = _collections.OrderedDict()
            expanded_circuit_outcomes = _collections.OrderedDict()
            for i in group:
                observed_outcomes = None if (dataset is None) else dataset[ds_circuits[i]].outcomes
                d = unique_complete_circuits[i].expand_instruments_and_separate_povm(model_shlp, observed_outcomes)
                expanded_circuit_outcomes_by_orig[i] = d
                expanded_circuit_outcomes.update(d)

            expanded_circuits = {i: cir for i, cir in enumerate(expanded_circuit_outcomes.keys())}
            self.tables.append(_create_prefix_table(expanded_circuits, max_cache_size))

            #Create circuit element <=> integer index lookups for speed
            all_rholabels = set()
            all_oplabels = set()
            all_elabels = set()
            for expanded_circuit_outcomes in expanded_circuit_outcomes_by_orig.values():
                for sep_povm_c in expanded_circuit_outcomes:
                    if sep_povm_c.effect_labels == [None]:  # special case
                        all_oplabels.update(sep_povm_c[:])
                    else:
                        all_rholabels.add(sep_povm_c[0])
                        all_oplabels.update(sep_povm_c[1:])
                        all_elabels.update(sep_povm_c.effect_labels)
            rho_labels = sorted(all_rholabels)
            op_labels = sorted(all_oplabels)
            elabel_lookup = {elbl: i for i, elbl in enumerate(all_elabels)}

            #Lookup arrays for faster replib computation.
            table_offset = 0
            elbl_indices_by_expcircuit = {}
            elindices_by_expcircuit = {}

            #Assign element indices, starting at `offset`
            for orig_i, expanded_circuit_outcomes in expanded_circuit_outcomes_by_orig.items():
                for table_relindex, (sep_povm_c, outcomes) in enumerate(expanded_circuit_outcomes.items()):
                    i = table_offset + table_relindex
                    elindices = list(range(offset, offset + len(outcomes)))
                    elbl_indices_by_expcircuit[i] = [elabel_lookup[elbl] for elbl in sep_povm_c.effect_labels]
                    elindices_by_expcircuit[i] = elindices
                    offset += len(outcomes)

                    elindex_outcome_tuples[orig_i].extend(list(zip(elindices, outcomes)))
                table_offset += len(expanded_circuit_outcomes)

        super().__init__(circuits, unique_circuits, to_unique, elindex_outcome_tuples, unique_complete_circuits,
                         num_sub_table_comms)
        self.cachesize = None

        #self.cachesize = curCacheSize
        #self.myFinalToParentFinalMap = None  # this tree has no "children",
        #self.myFinalElsToParentFinalElsMap = None  # i.e. has not been created by a 'split'
        #self.parentIndexMap = None
        #self.original_index_lookup = None
        #self.subTrees = []  # no subtrees yet
        #assert(self.generate_circuit_list() == circuit_list)
        #assert(None not in circuit_list)

    #def _remove_from_cache(self, indx):
    #    """ Removes self[indx] from cache (if it's in it)"""
    #    remStart, remRemain, remCache = self[indx]
    #    # iStart, remaining string, and iCache of element to remove
    #    if remCache is None: return  # not in cache to begin with!
    #
    #    for i in range(len(self)):
    #        iStart, remainingStr, iCache = self[i]
    #        if iCache == remCache:
    #            assert(i == indx)
    #            self[i] = (iStart, remainingStr, None)  # not in cache anymore
    #            continue
    #
    #        if iCache is not None and iCache > remCache:
    #            iCache -= 1  # shift left all cache indices after one removed
    #        if iStart == remCache:
    #            iStart = remStart
    #            remainingStr = remRemain + remainingStr
    #        elif iStart is not None and iStart > remCache:
    #            iStart -= 1  # shift left all cache indices after one removed
    #        self[i] = (iStart, remainingStr, iCache)
    #    self.cachesize -= 1

    #def _build_elabels_lookups(self):
    #    all_elabels = set()
    #    for elabels in self.simplified_circuit_elabels:
    #        all_elabels.update(elabels)
    #
    #    #Convert set of unique effect labels to a list so ordering is fixed
    #    all_elabels = sorted(list(all_elabels))
    #
    #    # Create lookup so elabel_lookup[eLbl] gives index of eLbl
    #    # within all_elabels (for faster lookup, Cython routines in ptic)
    #    elabel_lookup = {elbl: i for i, elbl in enumerate(all_elabels)}
    #
    #    #Create arrays that tell us, for a given rholabel, what the elabel indices
    #    # are for each simplified circuit.  This is obviously convenient for computing
    #    # outcome probabilities.
    #    eLbl_indices_per_circuit = {}
    #    final_indices_per_circuit = {}
    #    for i, elabels in enumerate(self.simplified_circuit_elabels):
    #        element_offset = self.element_offsets_for_circuit[i]  # offset to i-th simple circuits elements
    #        for j, eLbl in enumerate(elabels):
    #            if i in eLbl_indices_per_circuit:
    #                eLbl_indices_per_circuit[i].append(elabel_lookup[eLbl])
    #                final_indices_per_circuit[i].append(element_offset + j)
    #            else:
    #                eLbl_indices_per_circuit[i] = [elabel_lookup[eLbl]]
    #                final_indices_per_circuit[i] = [element_offset + j]
    #
    #    return all_elabels, eLbl_indices_per_circuit, final_indices_per_circuit

    #def squeeze(self, max_cache_size):
    #    """
    #    Remove items from cache (if needed) so it contains less than or equal to `max_cache_size` elements.
    #
    #    Parameters
    #    ----------
    #    max_cache_size : int
    #        The maximum cache size.
    #
    #    Returns
    #    -------
    #    None
    #    """
    #    assert(max_cache_size >= 0)
    #
    #    if max_cache_size == 0:  # special but common case
    #        curCacheSize = self.cache_size()
    #        cacheinds = [None] * curCacheSize
    #        for i in self.get_evaluation_order():
    #            iStart, remainingStr, iCache = self[i]
    #            if iStart is not None:
    #                remainingStr = self[cacheinds[iStart]][1] + remainingStr
    #            if iCache is not None:
    #                cacheinds[iCache] = i
    #            self[i] = (None, remainingStr, None)
    #        self.cachesize = 0
    #        return
    #
    #    #Otherwise, if max_cache_size > 0, remove cache elements one at a time:
    #    while self.cache_size() > max_cache_size:
    #
    #        #Figure out what's in cache and # of times each one is hit
    #        curCacheSize = self.cache_size()
    #        hits = [0] * curCacheSize
    #        cacheinds = [None] * curCacheSize
    #        for i in range(len(self)):
    #            iStart, remainingStr, iCache = self[i]
    #            if iStart is not None: hits[iStart] += 1
    #            if iCache is not None: cacheinds[iCache] = i
    #
    #        #Find a min-cost item to remove
    #        minCost = None; iMinTree = None; iMinCache = None
    #        for i in range(curCacheSize):
    #            cost = hits[i] * len(self[cacheinds[i]][1])
    #            # hits * len(remainder) ~= # more applies if we
    #            # remove i-th cache element.
    #            if iMinTree is None or cost < minCost:
    #                minCost = cost; iMinTree = cacheinds[i]
    #                iMinCache = i  # in cache
    #        assert(self[iMinTree][2] == iMinCache)  # sanity check
    #
    #        #Remove references to iMin element
    #        self._remove_from_cache(iMinTree)

    #def trim_nonfinal_els(self):  #INPLACE
    #    """
    #    Removes from this tree all non-final elements (to facilitate computation sometimes).
    #
    #    Returns
    #    -------
    #    None
    #    """
    #    nFinal = self.num_final_strings()
    #    self._delete_els(list(range(nFinal, len(self))))
    #
    #    #remove any unreferenced cache elements
    #    curCacheSize = self.cache_size()
    #    hits = [0] * curCacheSize
    #    cacheinds = [None] * curCacheSize
    #    for i in range(len(self)):
    #        iStart, remainingStr, iCache = self[i]
    #        if iStart is not None: hits[iStart] += 1
    #        if iCache is not None: cacheinds[iCache] = i
    #    for hits, i in zip(hits, cacheinds):
    #        if hits == 0: self._remove_from_cache(i)

    #def _delete_els(self, els_to_remove):
    #    """
    #    Delete a self[i] for i in els_to_remove.
    #    """
    #    if len(els_to_remove) == 0: return
    #
    #    last = els_to_remove[0]
    #    for i in els_to_remove[1:]:
    #        assert(i > last), "els_to_remove *must* be sorted in ascending order!"
    #        last = i
    #
    #    #remove from cache
    #    for i in els_to_remove:
    #        self._remove_from_cache(i)
    #
    #    order = self.eval_order
    #    for i in reversed(els_to_remove):
    #        del self[i]  # remove from self
    #
    #        #remove & update eval order
    #        order = [((k - 1) if k > i else k) for k in order if k != i]
    #    self.eval_order = order

    def cache_size(self):
        """
        Returns the size of the persistent "cache" of partial results.

        This cache is used during the computation of all the strings in
        this tree.

        Returns
        -------
        int
        """
        return self.cachesize

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

    #def _update_element_indices(self, new_indices_in_old_order, old_indices_in_new_order, element_indices_dict):
    #    """
    #    Update any additional members because this tree's elements are being permuted.
    #    In addition, return an updated version of `element_indices_dict` a dict whose keys are
    #    the tree's (unpermuted) circuit indices and whose values are the final element indices for
    #    each circuit.
    #    """
    #    self.simplified_circuit_elabels, updated_elIndices = \
    #        self._permute_simplified_circuit_xs(self.simplified_circuit_elabels,
    #                                            element_indices_dict, old_indices_in_new_order)
    #    self.simplified_circuit_nEls = list(map(len, self.simplified_circuit_elabels))
    #
    #    # Update element_offsets_for_circuit, etc
    #    self.element_offsets_for_circuit = _np.cumsum(
    #        [0] + [len(elabelList) for elabelList in self.simplified_circuit_elabels])[:-1]
    #    self.elabels, self.eLbl_indices_per_circuit, self.final_indices_per_circuit = \
    #        self._build_elabels_lookups()
    #
    #    return updated_elIndices

    def copy(self):
        """
        Create a copy of this evaluation tree.

        Returns
        -------
        MapEvalTree
        """
        cpy = self._copy_base(MapEvalTree(self[:]))
        cpy.cachesize = self.cachesize  # member specific to MapEvalTree
        cpy.opLabels = self.opLabels[:]
        cpy.simplified_circuit_elabels = _copy.deepcopy(self.simplified_circuit_elabels)
        cpy.element_offsets_for_circuit = self.element_offsets_for_circuit.copy()
        cpy.elabels = self.elabels[:]
        cpy.eLbl_indices_per_circuit = _copy.deepcopy(self.eLbl_indices_per_circuit)
        cpy.final_indices_per_circuit = _copy.deepcopy(self.final_indices_per_circuit)
        cpy.rholabels = self.rholabels[:]
        return cpy
