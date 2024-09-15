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

from pygsti.circuits.circuit import SeparatePOVMCircuit as _SeparatePOVMCircuit


class PrefixTable(object):
    """
    An ordered list ("table") of circuits to evaluate, where common prefixes can be cached.

    """

    def __init__(self, circuits_to_evaluate, max_cache_size, circuit_parameter_dependencies=None):
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

        Parameters
        ----------


        circuit_parameter_sensitivities : 
            A map between the circuits in circuits_to_evaluate and the indices of the model parameters
            to which these circuits depend.

        Returns
        -------
        tuple
            A tuple of `(table_contents, cache_size)` where `table_contents` is a list
            of tuples as given above and `cache_size` is the total size of the state
            cache used to hold intermediate results.
        """
        #print(f'{circuits_to_evaluate=}')
        #print(f'{circuit_parameter_dependencies=}')
        #Sort the operation sequences "alphabetically", so that it's trivial to find common prefixes
        #circuits_to_evaluate_fastlookup = {i: cir for i, cir in enumerate(circuits_to_evaluate)}
        circuits_to_sort_by = [cir.circuit_without_povm if isinstance(cir, _SeparatePOVMCircuit) else cir
                               for cir in circuits_to_evaluate]  # always Circuits - not SeparatePOVMCircuits
        sorted_circuits_to_sort_by = sorted(list(enumerate(circuits_to_sort_by)), key=lambda x: x[1])
        sorted_circuits_to_evaluate = [(i, circuits_to_evaluate[i]) for i, _ in sorted_circuits_to_sort_by]
        
        #print(f'{sorted_circuits_to_evaluate[-1][1].circuit_without_povm=}')

        #If the circuit parameter dependencies have been specified sort these in the same order used for
        #circuits_to_evaluate.
        if circuit_parameter_dependencies is not None:
            sorted_circuit_parameter_dependencies = [circuit_parameter_dependencies[i] for i, _ in sorted_circuits_to_evaluate]
        else:
            sorted_circuit_parameter_dependencies = None

        distinct_line_labels = set([cir.line_labels for cir in circuits_to_sort_by])
        if len(distinct_line_labels) == 1:  # if all circuits have the *same* line labels, we can just compare tuples
            circuit_reps_to_compare_and_lengths = {i: (cir.layertup, len(cir))
                                                   for i, cir in enumerate(circuits_to_sort_by)}
        else:
            circuit_reps_to_compare_and_lengths = {i: (cir, len(cir)) for i, cir in enumerate(circuits_to_sort_by)}
        #print(f'{max_cache_size=}')
        if max_cache_size is None or max_cache_size > 0:
            #CACHE assessment pass: figure out what's worth keeping in the cache.
            # In this pass, we cache *everything* and keep track of how many times each
            # original index (after it's cached) is utilized as a prefix for another circuit.
            # Not: this logic could be much better, e.g. computing a cost savings for each
            #  potentially-cached item and choosing the best ones, and proper accounting
            #  for chains of cached items.
            cacheIndices = []  # indices into circuits_to_evaluate of the results to cache
            cache_hits = _collections.defaultdict(lambda: 0)

            for i, _ in sorted_circuits_to_evaluate:
                circuit, L = circuit_reps_to_compare_and_lengths[i]  # can be a Circuit or a label tuple
                for cached_index in reversed(cacheIndices):
                    candidate, Lc = circuit_reps_to_compare_and_lengths[cached_index]
                    if L >= Lc > 0 and circuit[0:Lc] == candidate:  # a cache hit!
                        cache_hits[cached_index] += 1
                        break  # stop looking through cache
                cacheIndices.append(i)  # cache *everything* in this pass

        # Build prefix table: construct list, only caching items with hits > 0 (up to max_cache_size)
        cacheIndices = []  # indices into circuits_to_evaluate of the results to cache
        table_contents = []
        curCacheSize = 0

        for i, circuit in sorted_circuits_to_evaluate:
            circuit_rep, L = circuit_reps_to_compare_and_lengths[i]

            #find longest existing prefix for circuit by working backwards
            # and finding the first string that *is* a prefix of this string
            # (this will necessarily be the longest prefix, given the sorting)
            for i_in_cache in range(curCacheSize - 1, -1, -1):  # from curCacheSize-1 -> 0
                candidate, Lc = circuit_reps_to_compare_and_lengths[cacheIndices[i_in_cache]]
                if L >= Lc > 0 and circuit_rep[0:Lc] == candidate:  # ">=" allows for duplicates
                    iStart = i_in_cache  # an index into the *cache*, not into circuits_to_evaluate
                    remaining = circuit_rep[Lc:]  # *always* a SeparatePOVMCircuit or Circuit
                    break
            else:  # no break => no prefix
                iStart = None
                remaining = circuit_rep

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
        self.circuit_param_dependence = sorted_circuit_parameter_dependencies

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

        def create_subtables(max_cost, max_cost_rate=0, max_num=None):
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
            curSubTable = set()  # destination index of 0th evaluant
            curTableCost = 0  # remainder length of 0th evaluant
            totalCost = 0
            cacheIndices = [None] * self.cache_size
            contents_by_idest = {tup[0]: tup for tup in table_contents}  # for fast lookup by a circuit index
            used_indices = set()  # so we don't add an index to two sub-tables (even though they may
            # need to compute the index in the end)

            def traverse_index(idest, istart, remainder, current_table):
                """Get the set of indices you'd need to add along with and including `k`, and their cost."""
                cost = cost_fn(remainder)
                inds = set([idest])
                assert(idest not in used_indices)

                if istart is not None and cacheIndices[istart] not in current_table:
                    #we need to add the table elements traversed by following istart
                    j = istart  # index into cache
                    while j is not None:
                        j_circuit = cacheIndices[j]  # cacheIndices[ istart ]
                        if j_circuit not in used_indices: inds.add(j_circuit)
                        _, jStart, jrem, _ = contents_by_idest[j_circuit]
                        cost += cost_fn(jrem)  # remainder
                        j = jStart
                return inds, cost

            for idest, istart, remainder, iCache in table_contents:
                if iCache is not None:
                    cacheIndices[iCache] = idest

                #compute the cost (additional #applies) which results from
                # adding this element (idest) to the current sub-table.
                inds, cost = traverse_index(idest, istart, remainder, curSubTable)

                if curTableCost + cost < max_cost or (max_num is not None and len(subTables) == max_num - 1):
                    #Just add current string to current table
                    curTableCost += cost
                    curSubTable.update(inds)
                    used_indices.update(inds)
                else:
                    #End the current table and begin a new one
                    #print("cost %d+%d exceeds %d" % (curTableCost,cost,max_cost))
                    subTables.append(curSubTable)
                    totalCost += curTableCost
                    curSubTable, curTableCost = traverse_index(idest, istart, remainder, set())
                    used_indices.update(curSubTable)
                    #print("Added new table w/initial cost %d" % (cost))

                max_cost += max_cost_rate

            subTables.append(curSubTable)
            totalCost += curTableCost
            return subTables, totalCost

        def _get_num_applies(content):
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
                maxCost = _get_num_applies(table_contents) / num_sub_tables
            else: maxCost = len(table_contents) / num_sub_tables
            maxCostLowerBound, maxCostUpperBound = maxCost, None
            maxCostRate, rateLowerBound, rateUpperBound = 0, -1.0, +1.0
            failsafe_maxcost_and_rate = None
            #OLD (& incorrect) vals were 0, -1.0/len(self), +1.0/len(self),
            #   though current -1,1 vals are probably overly conservative...
            resultingSubtables = num_sub_tables + 1  # just to prime the loop
            iteration = 0
            #print("DEBUG: targeting %d sub-tables" % num_sub_tables)

            #Iterate until the desired number of subtables have been found.
            while resultingSubtables != num_sub_tables:
                subTableSetList, totalCost = create_subtables(maxCost, maxCostRate)
                resultingSubtables = len(subTableSetList)
                #print("DEBUG: resulting sub-tables = %d (cost %g) w/maxCost = %g [%s,%s] & rate = %g [%g,%g]" %
                #      (resultingSubtables, totalCost, maxCost, str(maxCostLowerBound), str(maxCostUpperBound),
                #       maxCostRate, rateLowerBound, rateUpperBound))

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
                        failsafe_maxcost_and_rate = (maxCost, maxCostRate)  # just in case
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
                        failsafe_maxcost_and_rate = (maxCost, maxCostRate)  # just in case
                        maxCostRate = (maxCostRate + rateUpperBound) / 2.0
                        rateLowerBound = last_maxRate

                iteration += 1
                if iteration >= 100:
                    #Force the correct number of tables using the max_cost & rate that produced the smallest number of
                    # sub-tables greater than the target number.
                    print("WARNING: Forcing splitting into %d tables after 100 iterations (achieved %d)!"
                          % (num_sub_tables, resultingSubtables))
                    subTableSetList, totalCost = create_subtables(failsafe_maxcost_and_rate[0],
                                                                  failsafe_maxcost_and_rate[1], max_num=num_sub_tables)
                    assert(len(subTableSetList) == num_sub_tables)  # ensure the loop exits now (~= break)
                    break  # or could set resultingSubtables = len(subTableSetList)

        else:  # max_sub_table_size is not None
            subTableSetList, totalCost = create_subtables(
                max_sub_table_size, max_cost_rate=0, cost_metric="size")

        assert(sum(map(len, subTableSetList)) == len(self)), "sub-table sets are not disjoint!"
        return subTableSetList
    

class PrefixTableJacobian(object):
    """
    An ordered list ("table") of circuits to evaluate, where common prefixes can be cached.
    Specialized for purposes of jacobian calculations.

    """

    def __init__(self, circuits_to_evaluate, max_cache_size, parameter_circuit_dependencies=None):
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

        Parameters
        ----------


        circuit_parameter_sensitivities : 
            A map between the circuits in circuits_to_evaluate and the indices of the model parameters
            to which these circuits depend.

        Returns
        -------
        tuple
            A tuple of `(table_contents, cache_size)` where `table_contents` is a list
            of tuples as given above and `cache_size` is the total size of the state
            cache used to hold intermediate results.
        """
        #print(f'{circuits_to_evaluate=}')
        #print(f'{circuit_parameter_dependencies=}')
        #Sort the operation sequences "alphabetically", so that it's trivial to find common prefixes        
        circuits_to_sort_by = [cir.circuit_without_povm if isinstance(cir, _SeparatePOVMCircuit) else cir
                               for cir in circuits_to_evaluate]  # always Circuits - not SeparatePOVMCircuits
        sorted_circuits_to_sort_by = sorted(list(enumerate(circuits_to_sort_by)), key=lambda x: x[1])
        sorted_circuits_to_evaluate = [(i, circuits_to_evaluate[i]) for i, _ in sorted_circuits_to_sort_by]
        #print(f'{sorted_circuits_to_evaluate=}')
        #create a map from sorted_circuits_to_sort_by by can be used to quickly sort each of the parameter
        #dependency lists.
        fast_sorting_map = {circuits_to_evaluate[i]:j for j, (i, _) in enumerate(sorted_circuits_to_sort_by)} 

        #also need a map from circuits to their original indices in circuits_to_evaluate
        #for the purpose of setting the correct destination indices in the evaluation instructions.
        circuit_to_orig_index_map = {circuit: i for i,circuit in enumerate(circuits_to_evaluate)}

        #use this map to sort the parameter_circuit_dependencies sublists.
        sorted_parameter_circuit_dependencies = []
        sorted_parameter_circuit_dependencies_orig_indices = []
        for sublist in parameter_circuit_dependencies:
            sorted_sublist = [None]*len(sorted_circuits_to_evaluate)
            for ckt in sublist:
                sorted_sublist[fast_sorting_map[ckt]] = ckt
            
            #filter out instances of None to get the correctly sized and sorted
            #sublist.
            filtered_sorted_sublist = [val for val in sorted_sublist if val is not None]
            orig_index_sublist = [circuit_to_orig_index_map[ckt] for ckt in filtered_sorted_sublist]
            
            sorted_parameter_circuit_dependencies.append(filtered_sorted_sublist)
            sorted_parameter_circuit_dependencies_orig_indices.append(orig_index_sublist)

        sorted_circuit_reps = []
        sorted_circuit_lengths = []
        for sublist in sorted_parameter_circuit_dependencies:
            circuit_reps, circuit_lengths = self._circuits_to_compare(sublist)
            sorted_circuit_reps.append(circuit_reps)
            sorted_circuit_lengths.append(circuit_lengths)

        #Intuition: The sorted circuit lists should likely break into equivalence classes, wherein multiple
        #parameters will have the same dependent circuits. This is because in typical models parameters
        #appear in blocks corresponding to a particular gate label, and so most of the time it should be the
        #case that the list fractures into all those circuits containing a particular label.
        #This intuition probably breaks down for ImplicitOpModels with complicated layer rules for which
        #the breaking into equivalence classes may have limited savings.
        unique_parameter_circuit_dependency_classes = {}
        for i, sublist in enumerate(sorted_circuit_reps):
            if unique_parameter_circuit_dependency_classes.get(sublist, None) is None:
                unique_parameter_circuit_dependency_classes[sublist] = [i]
            else:
                unique_parameter_circuit_dependency_classes[sublist].append(i)
        
        self.unique_parameter_circuit_dependency_classes = unique_parameter_circuit_dependency_classes
        #print(unique_parameter_circuit_dependency_classes)

        #the keys of the dictionary already give the needed circuit rep lists for 
        #each class, also grab the appropriate list of length for each class.
        sorted_circuit_lengths_by_class = [sorted_circuit_lengths[class_indices[0]] 
                                           for class_indices in unique_parameter_circuit_dependency_classes.values()]
        
        #also need representatives fo the entries in sorted_parameter_circuit_dependencies for each class,
        #and for sorted_parameter_circuit_dependencies_orig_indices
        sorted_parameter_circuit_dependencies_by_class = [sorted_parameter_circuit_dependencies[class_indices[0]] 
                                                          for class_indices in unique_parameter_circuit_dependency_classes.values()]
        sorted_parameter_circuit_dependencies_orig_indices_by_class = [sorted_parameter_circuit_dependencies_orig_indices[class_indices[0]] 
                                                                       for class_indices in unique_parameter_circuit_dependency_classes.values()]
        
        #now we can just do the calculation for each of these equivalence classes.

        #get the cache hits for all of the parameter circuit dependency sublists
        if max_cache_size is None or max_cache_size > 0:
            cache_hits_by_class = []
            #CACHE assessment pass: figure out what's worth keeping in the cache.
            # In this pass, we cache *everything* and keep track of how many times each
            # original index (after it's cached) is utilized as a prefix for another circuit.
            # Not: this logic could be much better, e.g. computing a cost savings for each
            #  potentially-cached item and choosing the best ones, and proper accounting
            #  for chains of cached items.
            for circuit_reps, circuit_lengths in zip(unique_parameter_circuit_dependency_classes.keys(), 
                                                     sorted_circuit_lengths_by_class):
                cache_hits_by_class.append(self._cache_hits(circuit_reps, circuit_lengths))
        else:
            cache_hits_by_class = [None]*len(unique_parameter_circuit_dependency_classes)
                
        #next construct a prefix table for each sublist.
        table_contents_by_class = []
        cache_size_by_class = []
        for sublist, cache_hits, circuit_reps, circuit_lengths, orig_indices in zip(sorted_parameter_circuit_dependencies_by_class, 
                                                                                    cache_hits_by_class,
                                                                                    unique_parameter_circuit_dependency_classes.keys(),
                                                                                    sorted_circuit_lengths_by_class,
                                                                                    sorted_parameter_circuit_dependencies_orig_indices_by_class):
            table_contents, curCacheSize = self._build_table(sublist, cache_hits,
                                                             max_cache_size, circuit_reps, circuit_lengths,
                                                             orig_indices)
            table_contents_by_class.append(table_contents)
            cache_size_by_class.append(curCacheSize)
            #print(f'{table_contents=}')
            #raise Exception
        #FUTURE: could perform a second pass, and if there is
        # some threshold number of elements which share the
        # *same* iStart and the same beginning of the
        # 'remaining' part then add a new "extra" element
        # (beyond the #circuits index) which computes
        # the shared prefix and insert this into the eval
        # order.

        #map back from equivalence classes to by parameter.
        table_contents_by_parameter = [None]*len(parameter_circuit_dependencies)
        cache_size_by_parameter = [None]*len(parameter_circuit_dependencies)
        for table_contents, cache_size, param_class in zip(table_contents_by_class, cache_size_by_class,
                                                           unique_parameter_circuit_dependency_classes.values()):
            for idx in param_class:
                table_contents_by_parameter[idx] = table_contents
                cache_size_by_parameter[idx] = cache_size

        self.contents_by_parameter = table_contents_by_parameter
        self.cache_size_by_parameter = cache_size_by_parameter
        self.parameter_circuit_dependencies = sorted_parameter_circuit_dependencies

    def _circuits_to_compare(self, sorted_circuits_to_evaluate):
        circuit_reps = [None]*len(sorted_circuits_to_evaluate)
        circuit_lens = [None]*len(sorted_circuits_to_evaluate)
        for i, cir in enumerate(sorted_circuits_to_evaluate):
            if isinstance(cir, _SeparatePOVMCircuit):
                circuit_reps[i] = cir.circuit_without_povm.layertup
                circuit_lens[i] = len(circuit_reps[i])
            else:
                circuit_reps[i] = cir.layertup
                circuit_lens[i] = len(circuit_reps[i])
        return tuple(circuit_reps), tuple(circuit_lens)


    def _cache_hits(self, circuit_reps, circuit_lengths):

        #CACHE assessment pass: figure out what's worth keeping in the cache.
        # In this pass, we cache *everything* and keep track of how many times each
        # original index (after it's cached) is utilized as a prefix for another circuit.
        # Not: this logic could be much better, e.g. computing a cost savings for each
        #  potentially-cached item and choosing the best ones, and proper accounting
        #  for chains of cached items.
        
        cacheIndices = []  # indices into circuits_to_evaluate of the results to cache
        cache_hits = [0]*len(circuit_reps)

        for i in range(len(circuit_reps)):
            circuit = circuit_reps[i] 
            L = circuit_lengths[i]  # can be a Circuit or a label tuple
            for cached_index in reversed(cacheIndices):
                candidate = circuit_reps[cached_index]
                Lc = circuit_lengths[cached_index]
                if L >= Lc > 0 and circuit[0:Lc] == candidate:  # a cache hit!
                    cache_hits[cached_index] += 1
                    break  # stop looking through cache
            cacheIndices.append(i)  # cache *everything* in this pass
    
        return cache_hits


    def _build_table(self, sorted_circuits_to_evaluate, cache_hits, max_cache_size, circuit_reps, circuit_lengths,
                     orig_indices):

        # Build prefix table: construct list, only caching items with hits > 0 (up to max_cache_size)
        cacheIndices = []  # indices into circuits_to_evaluate of the results to cache
        table_contents = [None]*len(sorted_circuits_to_evaluate)
        curCacheSize = 0

        for j, (i, circuit) in zip(orig_indices,enumerate(sorted_circuits_to_evaluate)):
            circuit_rep = circuit_reps[i] 
            L = circuit_lengths[i]

            #find longest existing prefix for circuit by working backwards
            # and finding the first string that *is* a prefix of this string
            # (this will necessarily be the longest prefix, given the sorting)
            for i_in_cache in range(curCacheSize - 1, -1, -1):  # from curCacheSize-1 -> 0
                candidate = circuit_reps[cacheIndices[i_in_cache]]
                Lc = circuit_lengths[cacheIndices[i_in_cache]]
                if L >= Lc > 0 and circuit_rep[0:Lc] == candidate:  # ">=" allows for duplicates
                    iStart = i_in_cache  # an index into the *cache*, not into circuits_to_evaluate
                    remaining = circuit_rep[Lc:]  # *always* a SeparatePOVMCircuit or Circuit
                    break
            else:  # no break => no prefix
                iStart = None
                remaining = circuit_rep

            # if/where this string should get stored in the cache
            if (max_cache_size is None or curCacheSize < max_cache_size) and cache_hits[i]:
                iCache = len(cacheIndices)
                cacheIndices.append(i); curCacheSize += 1
            else:  # don't store in the cache
                iCache = None

            #Add instruction for computing this circuit
            table_contents[i] = (j, iStart, remaining, iCache)
        
        return table_contents, curCacheSize, 
