"""
Defines the PrefixTable class.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import collections as _collections


class PrefixTable(object):
    """
    An ordered list ("table") of circuits to evaluate, where common prefixes can be cached.

    """

    def __init__(self, circuits_to_evaluate, max_cache_size):
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
        circuits_to_evaluate_fastlookup = {i: cir for i, cir in enumerate(circuits_to_evaluate)}

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
                    candidate = circuits_to_evaluate_fastlookup[cached_index]
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
                candidate = circuits_to_evaluate_fastlookup[cacheIndices[i_in_cache]]
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
        self.contents = table_contents
        self.cache_size = curCacheSize

    def __len__(self):
        return len(self.contents)

    def find_splitting(self, max_sub_table_size=None, num_sub_tables=None, cost_metric="size", verbosity=0):
        """
        Find a partition of the indices of this table to define a set of sub-tables with the desire properties.

        This is done in order to reduce the maximum size of any tree (useful for
        limiting memory consumption or for using multiple cores).  Must specify
        either max_sub_tree_size or num_sub_trees.

        Parameters
        ----------
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
        table_contents = self.contents
        if max_sub_table_size is None and num_sub_tables is None:
            return [set(range(len(table_contents)))]  # no splitting needed

        if max_sub_table_size is not None and num_sub_tables is not None:
            raise ValueError("Cannot specify both max_sub_table_size and num_sub_tables")
        if num_sub_tables is not None and num_sub_tables <= 0:
            raise ValueError("Error: num_sub_tables must be > 0!")

        #Don't split at all if it's unnecessary
        if max_sub_table_size is None or len(table_contents) < max_sub_table_size:
            if num_sub_tables is None or num_sub_tables == 1:
                return [set(range(len(table_contents)))]

        def nocache_create_equal_size_subtables():
            """ A shortcut for special case when there is no cache so each
                circuit can be evaluated independently """
            N = len(table_contents)
            subTables = [set(range(i, N, num_sub_tables)) for i in range(num_sub_tables)]
            totalCost = N
            return subTables, totalCost

        def create_subtables(max_cost, max_cost_rate=0):
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
            cacheIndices = [None] * self.cache_size
            contents_by_idest = {tup[0]: tup for tup in table_contents}  # for fast lookup by a circuit index

            def traverse_index(idest, istart, remainder, current_table):
                """Get the set of indices you'd need to add along with and including `k`, and their cost."""
                cost = cost_fn(remainder)
                inds = set([idest])

                if istart is not None and cacheIndices[istart] not in current_table:
                    #we need to add the table elements traversed by following istart
                    j = istart  # index into cache
                    while j is not None:
                        j_circuit = cacheIndices[j]  # cacheIndices[ istart ]
                        inds.add(j_circuit)
                        _, jStart, jrem, _ = contents_by_idest[j_circuit]
                        cost += cost_fn(jrem)  # remainder
                        j = jStart
                return inds, cost

            for idest, istart, remainder, iCache in table_contents:
                if iCache is not None:
                    cacheIndices[iCache] = idest

                #compute the cost (additional #applies) which results from
                # adding this element (idest) to the current sub-table.
                inds, cost = traverse_index(idest, istart, remainder)

                if curTableCost + cost < max_cost:
                    #Just add current string to current table
                    curTableCost += cost
                    curSubTable.update(inds)
                else:
                    #End the current table and begin a new one
                    #print("cost %d+%d exceeds %d" % (curTableCost,cost,max_cost))
                    subTables.append(curSubTable)
                    curSubTable, curTableCost = traverse_index(idest, istart, remainder)
                    #print("Added new table w/initial cost %d" % (cost))

                max_cost += max_cost_rate

            subTables.append(curSubTable)
            totalCost += curTableCost
            return subTables, totalCost

        def get_num_applies(content):
            """
            Gets the number of "apply" operations required to compute this prefix tree (an int)
            """
            ops = 0
            for _, _, remainder, _ in content:
                ops += len(remainder)
            return ops

        ##################################################################
        # Find a list of where the current table should be broken        #
        ##################################################################

        if num_sub_tables is not None and self.cache_size == 0:
            subTableSetList, totalCost = nocache_create_equal_size_subtables()

        elif num_sub_tables is not None:

            #Optimize max-cost to get the right number of tables
            # (but this can yield tables with unequal lengths or cache sizes,
            # which is what we're often after for memory reasons)
            if cost_metric == "applies":
                maxCost = get_num_applies(table_contents) / num_sub_tables
            else: maxCost = len(table_contents) / num_sub_tables
            maxCostLowerBound, maxCostUpperBound = maxCost, None
            maxCostRate, rateLowerBound, rateUpperBound = 0, -1.0, +1.0
            #OLD (& incorrect) vals were 0, -1.0/len(self), +1.0/len(self),
            #   though current -1,1 vals are probably overly conservative...
            resultingSubtables = num_sub_tables + 1  # just to prime the loop
            iteration = 0

            #Iterate until the desired number of subtables have been found.
            while resultingSubtables != num_sub_tables:
                subTableSetList, totalCost = create_subtables(maxCost, maxCostRate)
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
