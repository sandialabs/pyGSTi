"""
Defines the PrefixTable class.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import collections as _collections
import networkx as _nx
from math import ceil
from pygsti.baseobjs import Label as _Label
from pygsti.circuits.circuit import SeparatePOVMCircuit as _SeparatePOVMCircuit
from pygsti.tools.tqdm import our_tqdm


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

        #Sort the operation sequences "alphabetically", so that it's trivial to find common prefixes
        circuits_to_sort_by = [cir.circuit_without_povm if isinstance(cir, _SeparatePOVMCircuit) else cir
                               for cir in circuits_to_evaluate]  # always Circuits - not SeparatePOVMCircuits
        #with the current logic in _build_table a candidate circuit is only treated as a possible prefix if
        #it is shorter than the one it is being evaluated as a prefix for. So it should work to sort these
        #circuits by length for the purposes of the current logic.
        sorted_circuits_to_sort_by = sorted(list(enumerate(circuits_to_sort_by)), key=lambda x: len(x[1]))
        orig_indices, sorted_circuits_to_evaluate = zip(*[(i, circuits_to_evaluate[i]) for i, _ in sorted_circuits_to_sort_by])
        
        self.sorted_circuits_to_evaluate = sorted_circuits_to_evaluate
        self.orig_indices = orig_indices

        #get the circuits in a form readily usable for comparisons
        circuit_reps, circuit_lens = _circuits_to_compare(sorted_circuits_to_evaluate)
        self.circuit_reps = circuit_reps


        if max_cache_size is None or max_cache_size > 0:
            #CACHE assessment pass: figure out what's worth keeping in the cache.
            # In this pass, we cache *everything* and keep track of how many times each
            # original index (after it's cached) is utilized as a prefix for another circuit.
            # Not: this logic could be much better, e.g. computing a cost savings for each
            #  potentially-cached item and choosing the best ones, and proper accounting
            #  for chains of cached items.
            cache_hits = _cache_hits(self.circuit_reps, circuit_lens)
        else:
            cache_hits = [None]*len(self.circuit_reps)

        table_contents, curCacheSize = _build_table(sorted_circuits_to_evaluate, cache_hits,
                                                    max_cache_size, self.circuit_reps, circuit_lens,
                                                    orig_indices)

        #FUTURE: could perform a second pass, and if there is
        # some threshold number of elements which share the
        # *same* iStart and the same beginning of the
        # 'remaining' part then add a new "extra" element
        # (beyond the #circuits index) which computes
        # the shared prefix and insert this into the eval
        # order.
        self.contents = table_contents
        self.cache_size = curCacheSize
        self.circuits_evaluated = circuits_to_sort_by 


    def __len__(self):
        return len(self.contents)
    
    def num_state_propagations(self):
        """
        Return the number of state propagation operations (excluding the action of POVM effects) 
        required for the evaluation strategy given by this PrefixTable.
        """
        return sum(self.num_state_propagations_by_circuit().values())
    
    def num_state_propagations_by_circuit(self):
        """
        Return the number of state propagation operations per-circuit 
        (excluding the action of POVM effects) required for the evaluation strategy 
        given by this PrefixTable, returned as a dictionary with keys corresponding to
        circuits and values corresponding to the number of state propagations
        required for that circuit.
        """
        state_props_by_circuit = {}
        for i, istart, remainder, _ in self.contents:
            if len(self.circuits_evaluated[i][0])>0 and self.circuits_evaluated[i][0] == _Label('rho0') and istart is None:
                state_props_by_circuit[self.circuits_evaluated[i]] = len(remainder)-1
            else:
                state_props_by_circuit[self.circuits_evaluated[i]] = len(remainder)
            
        return state_props_by_circuit
    
    def num_state_propagations_by_circuit_no_caching(self):
        """
        Return the number of state propagation operations per-circuit 
        (excluding the action of POVM effects) required for an evaluation strategy 
        without caching, returned as a dictionary with keys corresponding to
        circuits and values corresponding to the number of state propagations
        required for that circuit.
        """
        state_props_by_circuit = {}
        for circuit in self.circuits_evaluated:
            if len(circuit)>0 and circuit[0] == _Label('rho0'):
                state_props_by_circuit[circuit] = len(circuit[1:])
            else:
                state_props_by_circuit[circuit] = len(circuit)
        return state_props_by_circuit
    
    def num_state_propagations_no_caching(self):
        """
        Return the total number of state propagation operations
        (excluding the action of POVM effects) required for an evaluation strategy 
        without caching.
        """
        return sum(self.num_state_propagations_by_circuit_no_caching().values())

    def find_splitting_new(self, max_sub_table_size=None, num_sub_tables=None, initial_cost_metric='size',
                           rebalancing_cost_metric='propagations', imbalance_threshold=1.2, minimum_improvement_threshold=0.1,
                           verbosity=0):
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

        imbalance_threshold : float, optional (default 1.2)
            This number serves as a tolerance parameter for a final load balancing refinement
            to the splitting. The value coresponds to a threshold value of the ratio of the heaviest
            to the lightest subtree such that ratios below this value are considered sufficiently
            balanced and processing stops.

        minimum_improvement_threshold : float, optional (default 0.1)
            A parameter for the final load balancing refinement process that sets a minimum balance
            improvement (improvement to the ratio of the sizes of two subtrees) such that a rebalancing
            step is considered worth performing (even if it would otherwise bring the imbalance parameter
            described above in `imbalance_threshold` below the target value) .
                    
        verbosity : int, optional (default 0)
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
            
        #construct a tree structure describing the prefix strucure of the circuit set.
        circuit_tree = _build_prefix_tree(self.sorted_circuits_to_evaluate, self.circuit_reps, self.orig_indices)
        circuit_tree_nx = circuit_tree.to_networkx_graph()
        
        if num_sub_tables is not None:
            max_max_sub_table_size = len(self.sorted_circuits_to_evaluate)
            initial_max_sub_table_size = ceil(len(self.sorted_circuits_to_evaluate)/num_sub_tables)
            cut_edges, new_roots, tree_levels, subtree_weights = tree_partition_kundu_misra(circuit_tree_nx, max_weight=initial_max_sub_table_size,
                                                                            weight_key= 'cost' if initial_cost_metric=='size' else 'prop_cost',
                                                                            return_levels_and_weights=True)

            if len(new_roots) > num_sub_tables: #iteratively row the maximum subtree size until we either hit or are less than the target.
                last_seen_sub_max_sub_table_size_val = None
                feasible_range = [initial_max_sub_table_size+1, max_max_sub_table_size-1]
                #bisect on max_sub_table_size until we find the smallest value for which len(new_roots) <= num_sub_tables
                while feasible_range[0] < feasible_range[1]:
                    current_max_sub_table_size = (feasible_range[0] + feasible_range[1])//2
                    cut_edges, new_roots = tree_partition_kundu_misra(circuit_tree_nx, max_weight=current_max_sub_table_size,
                                                                      weight_key='cost' if initial_cost_metric=='size' else 'prop_cost',
                                                                      test_leaves=False, precomp_levels=tree_levels, precomp_weights=subtree_weights)                    
                    if len(new_roots) > num_sub_tables:
                        feasible_range[0] = current_max_sub_table_size+1
                    else:
                        last_seen_sub_max_sub_table_size_val = (cut_edges, new_roots) #In the multiple root setting I am seeing some strange
                        #non-monotonicity, so add this as a fall back in case the final result anomalously has len(roots)>num_sub_tables
                        feasible_range[1] = current_max_sub_table_size
                if len(new_roots)>num_sub_tables and last_seen_sub_max_sub_table_size_val is not None: #fallback
                    cut_edges, new_roots = last_seen_sub_max_sub_table_size_val

                #only apply the cuts now that we have found our starting point.
                partitioned_tree = _copy_networkx_graph(circuit_tree_nx)
                 #update the propagation cost attribute of the promoted nodes.
                #only do this at this point to reduce the need for copying
                for edge in cut_edges:
                    partitioned_tree.nodes[edge[1]]['prop_cost'] += partitioned_tree.edges[edge[0], edge[1]]['promotion_cost']
                partitioned_tree.remove_edges_from(cut_edges)
               
            #if we have hit the number of partitions, great, we're done!
            if len(new_roots) == num_sub_tables:
                #only apply the cuts now that we have found our starting point.
                partitioned_tree = _copy_networkx_graph(circuit_tree_nx)
                 #update the propagation cost attribute of the promoted nodes.
                #only do this at this point to reduce the need for copying
                for edge in cut_edges:
                    partitioned_tree.nodes[edge[1]]['prop_cost'] += partitioned_tree.edges[edge[0], edge[1]]['promotion_cost']
                partitioned_tree.remove_edges_from(cut_edges)
                pass
            #if we have fewer subtables then we need to look whether or not we should strictly
            #hit the number of partitions, or whether we allow for fewer than the requested number to be returned.
            if len(new_roots) < num_sub_tables:
                #Perform bisection operations on the heaviest subtrees until we hit the target number.
                while len(new_roots) < num_sub_tables:
                    partitioned_tree, new_roots, cut_edges = _bisection_pass(partitioned_tree, cut_edges, new_roots, num_sub_tables,
                                                                             weight_key='cost' if rebalancing_cost_metric=='size' else 'prop_cost')
            #add in a final refinement pass to improve the balancing across subtrees.
            partitioned_tree, new_roots, addl_cut_edges = _refinement_pass(partitioned_tree, new_roots, 
                                                                           weight_key='cost' if rebalancing_cost_metric=='size' else 'prop_cost', 
                                                                           imbalance_threshold= imbalance_threshold, 
                                                                           minimum_improvement_threshold= minimum_improvement_threshold)
        else:
            cut_edges, new_roots = tree_partition_kundu_misra(circuit_tree_nx, max_weight = max_sub_table_size,
                                                              weight_key='cost' if initial_cost_metric=='size' else 'prop_cost')
            partitioned_tree = _copy_networkx_graph(circuit_tree_nx)
            for edge in cut_edges:
                    partitioned_tree.nodes[edge[1]]['prop_cost'] += partitioned_tree.edges[edge[0], edge[1]]['promotion_cost']
            partitioned_tree.remove_edges_from(cut_edges)
        
        #Collect the original circuit indices for each of the parititioned subtrees.
        orig_index_groups = []
        for root in new_roots:
            if isinstance(root,tuple):
                ckts = []
                for elem in root:
                    ckts.extend(_collect_orig_indices(partitioned_tree, elem))
                orig_index_groups.append(ckts)
            else:
                orig_index_groups.append(_collect_orig_indices(partitioned_tree, root))

        return orig_index_groups
        

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

            if cost_metric == "applies":
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
        #Sort the operation sequences "alphabetically", so that it's trivial to find common prefixes        
        circuits_to_sort_by = [cir.circuit_without_povm if isinstance(cir, _SeparatePOVMCircuit) else cir
                               for cir in circuits_to_evaluate]  # always Circuits - not SeparatePOVMCircuits
        sorted_circuits_to_sort_by = sorted(list(enumerate(circuits_to_sort_by)), key=lambda x: len(x[1]))
        sorted_circuits_to_evaluate = [(i, circuits_to_evaluate[i]) for i, _ in sorted_circuits_to_sort_by]
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
            circuit_reps, circuit_lengths = _circuits_to_compare(sublist)
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
                cache_hits_by_class.append(_cache_hits(circuit_reps, circuit_lengths))
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
            table_contents, curCacheSize = _build_table(sublist, cache_hits,
                                                        max_cache_size, circuit_reps, circuit_lengths,
                                                        orig_indices)
            table_contents_by_class.append(table_contents)
            cache_size_by_class.append(curCacheSize)

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


#---------Helper Functions------------#

def _circuits_to_compare(sorted_circuits_to_evaluate):
        
    bare_circuits = [cir.circuit_without_povm if isinstance(cir, _SeparatePOVMCircuit) else cir
                            for cir in sorted_circuits_to_evaluate]
    distinct_line_labels = set([cir.line_labels for cir in bare_circuits])
    
    circuit_lens = [None]*len(sorted_circuits_to_evaluate)
    if len(distinct_line_labels) == 1:
        circuit_reps = [None]*len(sorted_circuits_to_evaluate)
        for i, cir in enumerate(bare_circuits):
            circuit_reps[i] = cir.layertup
            circuit_lens[i] = len(circuit_reps[i])
    else:
        circuit_reps = bare_circuits
        for i, cir in enumerate(sorted_circuits_to_evaluate):
            circuit_lens[i] = len(circuit_reps[i])

    return tuple(circuit_reps), tuple(circuit_lens)

def _cache_hits(circuit_reps, circuit_lengths):

    #CACHE assessment pass: figure out what's worth keeping in the cache.
    # In this pass, we cache *everything* and keep track of how many times each
    # original index (after it's cached) is utilized as a prefix for another circuit.
    # Not: this logic could be much better, e.g. computing a cost savings for each
    #  potentially-cached item and choosing the best ones, and proper accounting
    #  for chains of cached items.
    
    cacheIndices = []  # indices into circuits_to_evaluate of the results to cache
    cache_hits = [0]*len(circuit_reps)

    for i in our_tqdm(range(len(circuit_reps)), 'Prefix table : _cache_hits '):
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

def _build_table(sorted_circuits_to_evaluate, cache_hits, max_cache_size, circuit_reps, circuit_lengths,
                 orig_indices):

    # Build prefix table: construct list, only caching items with hits > 0 (up to max_cache_size)
    cacheIndices = []  # indices into circuits_to_evaluate of the results to cache
    num_sorted_circuits = len(sorted_circuits_to_evaluate)
    table_contents = [None]*num_sorted_circuits
    curCacheSize = 0
    for i in our_tqdm(range(num_sorted_circuits), 'Prefix table : _build_table '):
        j = orig_indices[i]
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

    return table_contents, curCacheSize

#helper method for building a tree showing the connections between different circuits
#for the purposes of prefix-based evaluation.
def _build_prefix_tree(sorted_circuits_to_evaluate, circuit_reps, orig_indices):
    #assume the input circuits have already been sorted by length.
    circuit_tree = Tree()
    for j, (i, _) in zip(orig_indices,enumerate(sorted_circuits_to_evaluate)):
        circuit_rep = circuit_reps[i]
        #the first layer should be a state preparation. If this isn't in a root in the
        #tree add it.
        root_node = circuit_tree.get_root_node(circuit_rep[0])
        if root_node is None and len(circuit_rep)>0:
            #cost is the number of propagations, so exclude the initial state prep
            root_node = RootNode(circuit_rep[0], cost=0) 
            circuit_tree.add_root(root_node)
        
        current_node = root_node
        for layerlbl in circuit_reps[i][1:]:
            child_node = current_node.get_child_node(layerlbl)
            if child_node is None:
                child_node = ChildNode(layerlbl, parent=current_node)
            current_node = child_node
        #when we get to the end of the circuit add a pointer on the
        #final node to the original index of this circuit in the
        #circuit list.
        current_node.add_orig_index(j)

    return circuit_tree


#----------------------Helper classes for managing circuit evaluation tree. --------------#
class TreeNode:
    def __init__(self, value, children=None, orig_indices=None):
        """
        Parameters
        ----------
        value : any
            The value to be stored in the node.

        children : list, optional (default is None)
            A list of child nodes. If None, initializes an empty list.

        orig_indices : list, optional (default is None)
            A list of original indices. If None, initializes an empty list.
        """
        self.value = value
        self.children = [] if children is None else children
        self.orig_indices = [] if orig_indices is None else orig_indices #make this a list to allow for duplicates

    def add_child(self, child_node):
        """
        Add a child node to the current node.

        Parameters
        ----------
        child_node : TreeNode
            The child node to be added.
        """

        self.children.append(child_node)

    def remove_child(self, child_node):
        """
        Remove a child node from the current node.

        Parameters
        ----------
        child_node : TreeNode
            The child node to be removed.
        """
        self.children = [child for child in self.children if child is not child_node]

    def get_child_node(self, value):
        """
        Get the child node associated with the input value. If that node is not present, return None.

        Parameters
        ----------
        value : any
            The value to search for in the child nodes.

        Returns
        -------
        TreeNode or None
            The child node with the specified value, or None if not found.
        """

        for node in self.children:
            if node.value == value:
                return node
        #if we haven't returned already it is because there wasn't a corresponding root,
        #so return None
        return None

    def add_orig_index(self, value):
        """
        Add an original index to the node.

        Parameters
        ----------
        value : int
            The original index to be added.
        """
        self.orig_indices.append(value)

    def traverse(self):
        """
        Traverse the tree in pre-order and return a list of node values.

        Returns
        -------
        list
            A list of node values in pre-order traversal.
        """

        nodes = []
        stack = [self]
        while stack:
            node = stack.pop()
            nodes.append(node.value)
            stack.extend(reversed(node.children))  # Add children to stack in reverse order for pre-order traversal
        return nodes

    def get_descendants(self):
        """
        Get all descendant node values of the current node in pre-order traversal.

        Returns
        -------
        list
            A list of descendant node values.
        """
        descendants = []
        stack = self.children[:]
        while stack:
            node = stack.pop()
            descendants.append(node.value)
            stack.extend(reversed(node.children))  # Add children to stack in reverse order for pre-order traversal
        return descendants
    
    def total_orig_indices(self):
        """
        Calculate the total number of orig_indices values for this node and all of its descendants.
        """
        total = len(self.orig_indices)
        for child in self.get_descendants():
            total += len(child.orig_indices)
        return total

    def print_tree(self, level=0, prefix=""):
        """
        Print the tree structure starting from the current node.

        Parameters
        ----------
        level : int, optional (default 0)
            The current level in the tree.
        prefix : str, optional (default "")
            The prefix for the current level.
        """
        connector = "├── " if level > 0 else ""
        print(prefix + connector + str(self.value) +', ' + str(self.orig_indices))
        for i, child in enumerate(self.children):
            if i == len(self.children) - 1:
                child.print_tree(level + 1, prefix + ("    " if level > 0 else ""))
            else:
                child.print_tree(level + 1, prefix + ("│   " if level > 0 else ""))

#create a class for RootNodes that includes additional initial cost information.
class RootNode(TreeNode):
    """
    Class for representing a root node for a tree, along with the corresponding metadata
    specific to root nodes.
    """

    def __init__(self, value, cost=0, tree=None, children=None, orig_indices=None):
        """
        Initialize a RootNode with a value, optional cost, optional tree, optional children, and optional original indices.

        Parameters
        ----------
        value : any
            The value to be stored in the node.
        cost : int, optional (default is 0)
            The initial cost associated with the root node.
        tree : Tree, optional (default is None)
            The tree to which this root node belongs.
        children : list, optional (default is None)
            A list of child nodes. If None, initializes an empty list.
        orig_indices : list, optional (default is None)
            A list of original indices. If None, initializes an empty list.
        """
        super().__init__(value, children, orig_indices)
        self.cost = cost
        self.tree = tree
        
class ChildNode(TreeNode):
    """
    Class for representing a child node for a tree, along with the corresponding metadata
    specific to child nodes.
    """
    def __init__(self, value, parent=None, children=None, orig_indices=None):
        """
        Parameters
        ----------
        value : any
            The value to be stored in the node.
        parent : TreeNode, optional (default is None)
            The parent node.
        children : list, optional (default is None)
            A list of child nodes. If None, initializes an empty list.
        orig_indices : list, optional (default is None)
            A list of original indices. If None, initializes an empty list.
        """
        super().__init__(value, children, orig_indices)
        self.parent = parent
        if parent is not None:
            parent.add_child(self)

    def get_ancestors(self):
        """
        Get all ancestor nodes of the current node up to the root node.

        Returns
        -------
        list
            A list of ancestor nodes.
        """
        ancestors = []
        node = self
        while node:
            ancestors.append(node)
            if isinstance(node, RootNode):
                break
            node = node.parent
        return ancestors

    def calculate_promotion_cost(self):
        """
        Calculate the cost of promoting this child node to a root node. This
        corresponds to the sum of the cost of this node's current root, plus
        the total number of ancestors (less the root).
        """
        ancestors = self.get_ancestors()
        ancestor_count = len(ancestors) - 1
        current_root = self.get_root()
        current_root_cost = current_root.cost
        return ancestor_count + current_root_cost

    def promote_to_root(self):
        """
        Promote this child node to a root node, updating the tree structure accordingly.
        """
        # Calculate the cost (I know this is code duplication, but in this case
        #we need the intermediate values as well).
        ancestors = self.get_ancestors()
        ancestor_count = len(ancestors) - 1
        current_root = self.get_root()
        current_root_cost = current_root.cost
        new_root_cost = ancestor_count + current_root_cost

        # Remove this node from its parent's children
        if self.parent:
            self.parent.remove_child(self)

        # Create a new RootNode
        ancestor_values = [ancestor.value for ancestor in reversed(ancestors)]
        if isinstance(ancestor_values[0], tuple):
            ancestor_values = list(ancestor_values[0]) + ancestor_values[1:]
        new_root_value = tuple(ancestor_values)
        new_root = RootNode(new_root_value, cost=new_root_cost, tree=current_root.tree, children=self.children,
                            orig_indices=self.orig_indices)

        # Update the children of the new RootNode
        for child in new_root.children:
            child.parent = new_root

        # Add the new RootNode to the tree
        if new_root.tree:
            new_root.tree.add_root(new_root)

        # Delete this ChildNode
        del self

    def get_root(self):
        """
        Get the root node of the current node.

        Returns
        -------
        RootNode
            The root node of the current node.
        """
        node = self
        while node.parent and not isinstance(node.parent, RootNode):
            node = node.parent
        return node.parent

class Tree:
    """
    Container class for storing a tree structure (technically a forest, as there
    can be multiple roots).
    """
    def __init__(self, roots=None):
        """
        Parameters
        ----------
        roots: list of RootNode, optional (default None)
            List of roots for this tree structure.
        """
        self.roots = []
        self.root_set = set(self.roots)
    
    def get_root_node(self, value):
        """
        Get the root node associated with the input value. If that node is not present, return None.

        Parameters
        ----------
        value : any
            The value to search for in the root nodes.

        Returns
        -------
        RootNode or None
            The root node with the specified value, or None if not found.
        """

        for node in self.roots:
            if node.value == value:
                return node
        #if we haven't returned already it is because there wasn't a corresponding root,
        #so return None
        return None

    def add_root(self, root_node):
        """
        Add a root node to the tree.

        Parameters
        ----------
        root_node : RootNode
            The root node to be added.
        """

        root_node.tree = self
        self.roots.append(root_node)
        self.root_set.add(root_node)

    def remove_root(self, root_node):
        """
        Remove a root node from the tree.

        Parameters
        ----------
        root_node : RootNode
            The root node to be removed.
        """

        root_node.tree = None
        self.roots = [root for root in self.roots if root is not root_node]
    
    def total_orig_indices(self):
        """
        Calculate the total number of original indices for all root nodes and their descendants.
        """
        return sum([root.total_orig_indices() for root in self.roots])

    def traverse(self):
        """
        Traverse the entire tree in pre-order and return a list of node values.

        Returns
        -------
        list
            A list of node values in pre-order traversal.
        """
        nodes = []
        for root in self.roots:
            nodes.extend(root.traverse())
        return nodes
    
    def count_nodes(self):
        """
        Count the total number of nodes in the tree.
        """
        count = 0
        stack = self.roots[:]
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children)
        return count

    def print_tree(self):
        """
        Print the entire tree structure.
        """
        for root in self.roots:
            root.print_tree()
    
    def calculate_cost(self):
        """
        Calculate the total cost of the tree, including root costs and promotion costs for child nodes.
        See `RootNode` and `ChildNode`.
        """
        total_cost = sum([root.cost for root in self.roots])
        total_nodes = self.count_nodes()
        total_child_nodes = total_nodes - len(self.roots)
        return total_cost + total_child_nodes
    
    def to_networkx_graph(self):
        """
        Convert the tree to a NetworkX directed graph with node and edge attributes.

        Returns
        -------
        networkx.DiGraph
            The NetworkX directed graph representation of the tree.
        """
        G = _nx.DiGraph()
        stack = [(None, root) for root in self.roots]
        insertion_order = 0
        while stack:
            parent, node = stack.pop()
            node_id = id(node)
            prop_cost = node.cost if isinstance(node, RootNode) else 1
            G.add_node(node_id, cost=len(node.orig_indices), orig_indices=tuple(node.orig_indices), 
                       label=node.value, prop_cost = prop_cost,
                       insertion_order=insertion_order)
            insertion_order+=1
            if parent is not None:
                parent_id = id(parent)
                edge_cost = node.calculate_promotion_cost()
                G.add_edge(parent_id, node_id, promotion_cost=edge_cost)
            for child in node.children:
                stack.append((node, child))

        #if there are multiple roots then add an additional virtual root node as the
        #parent for all of these roots to enable partitioning with later algorithms.
        if len(self.roots)>1:
            G.add_node('virtual_root', cost = 0, orig_indices=(), label = (), prop_cost=0, insertion_order=-1)
            for root in self.roots:
                G.add_edge('virtual_root', id(root), promotion_cost=0)

        return G

#--------------- Tree Partitioning Algorithm Helpers (+NetworkX Utilities)-----------------#

def _draw_graph(G, node_label_key='label', edge_label_key='promotion_cost', figure_size=(10,10)):
    """
    Draw the NetworkX graph with node labels.
    
    Parameters
    ----------
    G : networkx.Graph
        The networkx Graph object to draw.
    
    node_label_key : str, optional (default 'label')
        Optional key for the node attribute to use for the node labels.
    
    edge_label_key : str, optional (default 'cost')
        Optional key for the edge attribute to use for the edge labels.
    
    figure_size : tuple of floats, optional (default (10,10))
        An optional size specifier passed into the matplotlib figure
        constructor to set the plot size.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        msg = 'The function `_draw_graph` requires the installation of matplotlib.'\
              +'Please ensure this is properly installed and try again.'
        print(msg)
        raise e

    plt.figure(figsize=figure_size)
    pos = _nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Granksep=5 -Gnodesep=10")
    labels = _nx.get_node_attributes(G, node_label_key)
    _nx.draw(G, pos, labels=labels, with_labels=True, node_size=500, node_color='lightblue', font_size=6, font_weight='bold')
    edge_labels = _nx.get_edge_attributes(G, edge_label_key)
    _nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

def _copy_networkx_graph(G):
    """
    Create a new independent copy of a NetworkX directed graph with node and edge attributes that
    match the original graph. Specialized to copying graphs with the known attributes set by the
    `to_networkx_graph` method of the `Tree` class.

    Parameters
    ----------
    G : networkx.DiGraph
        The original NetworkX directed graph.

    Returns
    -------
    networkx.DiGraph
        The new independent copy of the NetworkX directed graph.
    """
    new_G = _nx.DiGraph()

    # Copy nodes with attributes
    for node, data in G.nodes(data=True):
        new_G.add_node(node, cost = data['cost'], orig_indices=data['orig_indices'], 
                       label= data['label'] , prop_cost = data['prop_cost'],
                       insertion_order=data['insertion_order'])

    # Copy edges with attributes
    for u, v, data in G.edges(data=True):
        new_G.add_edge(u, v, promotion_cost = data['promotion_cost'])

    return new_G

def _find_root(tree):
    """
    Find the root node of a directed tree.

    Parameters
    ----------
    tree : networkx.DiGraph
        The directed tree.
    
    Returns
    -------
    networkx node corresponding to the root.
    """

    # The root node will have no incoming edges
    for node in tree.nodes():
        if tree.in_degree(node) == 0:
            return node
    raise ValueError("The input graph is not a valid tree (no root found).")

def _compute_subtree_weights(tree, root, weight_key):
    """
    This function computes the total weight of each subtree in a directed tree.
    The weight of a subtree is defined as the sum of the weights of all nodes
    in that subtree, including the root of the subtree.

    Parameters
    ----------
    tree : networkx.DiGraph
        The directed tree.
    
    root: networkx node 
        The root node of the tree.
    
    weight_key : str
        A string corresponding to the node attribute to use as the weights.

    Returns
    -------
    A dictionary where keys are nodes and values are the total weights of the subtrees rooted at those nodes.
    """

    subtree_weights = {} # {node: 0 for node in tree.nodes()}
    stack = [root]
    visited = set()

    # First pass: calculate the subtree weights in a bottom-up manner
    while stack:
        node = stack.pop()
        if node in visited:
            # All children have been processed, now process the node itself
            subtree_weight = tree.nodes[node][weight_key]
            for child in tree.successors(node):
                subtree_weight += subtree_weights[child]
            subtree_weights[node] = subtree_weight
        else:
            # Process the node after its children
            visited.add(node)
            stack.append(node)
            for child in tree.successors(node):
                if child not in visited:
                    stack.append(child)

    return subtree_weights

def _partition_levels(tree, root):
    """
    Partition the nodes of a rooted directed tree into levels based on their distance from the root.

    Parameters
    ----------
    tree : networkx.DiGraph 
        The directed tree.
    root : networkx node
        The root node of the tree.

    Returns
    -------
    list of sets: 
        A list where each set contains nodes that are equidistant from the root.
    """
    # Initialize a dictionary to store the level of each node
    levels = {}
    # Initialize a queue for BFS
    queue = _collections.deque([(root, 0)])
    
    while queue:
        node, level = queue.popleft()
        if level not in levels:
            levels[level] = set()
        levels[level].add(node)
        
        for child in tree.successors(node):
            queue.append((child, level + 1))
    
    tree_nodes = tree.nodes
    # Convert the levels dictionary to a list of sets ordered by level
    sorted_levels = []
    for level in sorted(levels.keys()):
        # Sort nodes at each level by 'insertion_order' attribute
        sorted_nodes = sorted(levels[level], key=lambda node: tree_nodes[node]['insertion_order'])
        sorted_levels.append(sorted_nodes)
    
    return sorted_levels


def _partition_levels_and_compute_subtree_weights(tree, root, weight_key):
    """
    Partition the nodes of a rooted directed tree into levels based on their distance from the root
    and compute the total weight of each subtree.

    Parameters
    ----------
    tree : networkx.DiGraph 
        The directed tree.
    root : networkx node
        The root node of the tree.
    weight_key : str
        A string corresponding to the node attribute to use as the weights.

    Returns
    -------
    tuple:
        - list of sets: A list where each set contains nodes that are equidistant from the root.
        - dict: A dictionary where keys are nodes and values are the total weights of the subtrees rooted at those nodes.
    """
    # Initialize a dictionary to store the level of each node
    levels = {}
    # Initialize a dictionary to store the subtree weights
    subtree_weights = {}
    # Initialize a queue for BFS
    queue = _collections.deque([(root, 0)])
    # Initialize a stack for DFS to compute subtree weights
    stack = []
    visited = set()

    #I think this returns a view, so grab this ahead of time in case
    #there is overhead with that.
    tree_nodes = tree.nodes

    #successors is kind of an expensive call and we use at least twice
    #per node, so let's just compute it once and cache in a dict.
    node_successors = {node: list(tree.successors(node)) for node in tree_nodes}

    while queue:
        node, level = queue.popleft()
        if node not in visited:
            visited.add(node)
            if level not in levels:
                levels[level] = set()
            levels[level].add(node)
            stack.append(node)
            for child in node_successors[node]:
                queue.append((child, level + 1))

    # Compute subtree weights in a bottom-up manner
    while stack:
        node = stack.pop()
        subtree_weight = tree_nodes[node][weight_key]
        for child in node_successors[node]:
            subtree_weight += subtree_weights[child]
        subtree_weights[node] = subtree_weight

    # Convert the levels dictionary to a list of sets ordered by level
    sorted_levels = []
    for level in sorted(levels.keys()):
        # Sort nodes at each level by 'insertion_order' attribute
        sorted_nodes = sorted(levels[level], key=lambda node: tree_nodes[node]['insertion_order'])
        sorted_levels.append(sorted_nodes)

    return sorted_levels, subtree_weights


def _find_leaves(tree):
    """
    Find all leaf nodes in a directed tree.

    Parameters
    ----------
    tree : networkx.DiGraph
        The directed tree.

    Returns
    -------
    A list of leaf nodes.
    """
    leaf_nodes = set([node for node in tree.nodes() if tree.out_degree(node) == 0])
    return leaf_nodes
        
def _path_to_root(tree, node, root):
    """
    Return a list of nodes along the path from the given node to the root.

    Parameters
    ----------
    tree : networkx.DiGraph 
        The directed tree.
    node : networkx node
        The starting node.
    root : networkx node
        The root node of the tree.

    Returns
    -------
    A list of nodes along the path from the given node to the root.
    """
    path = []
    current_node = node

    while current_node != root:
        path.append(current_node)
        #note: for a tree structure there should be just one predecessor
        #so not worried about nondeterminism, if we every apply this to another
        #graph structure this needs to be reevaluated.
        predecessors = list(tree.predecessors(current_node))
        current_node = predecessors[0]
    path.append(root)

    return path

def _get_subtree(tree, root):
    """
    Return a new graph corresponding to the subtree rooted at the given node.

    Parameters
    ----------
    tree : networkx.DiGraph
        The directed tree.
    
    root : networkx node
        The root node of the subtree.

    Returns
    -------
    subtree : networkx.DiGraph
        A new directed graph corresponding to the subtree rooted at the given node.
    """
    # Create a new directed graph for the subtree
    subtree = _nx.DiGraph()
    
    # Use a queue to perform BFS and add nodes and edges to the subtree
    queue = [root]
    while queue:
        node = queue.pop(0)
        subtree.add_node(node, **tree.nodes[node])
        for child in tree.successors(node):
            subtree.add_edge(node, child, **tree.edges[node, child])
            queue.append(child)
    
    return subtree

def _collect_orig_indices(tree, root):
    """
    Collect all values of the 'orig_indices' node attributes in the subtree rooted at the given node.
    The 'orig_indices' values are tuples, and the function flattens these tuples into a single list.

    Parameters
    ----------
    tree : networkx.DiGraph
        The directed tree.
    
    root : networkx node
        The root node of the subtree.

    Returns
    -------
    list
        A flattened list of all values of the 'orig_indices' node attributes in the subtree.
    """
    orig_indices_list = []
    queue = [root]
    
    #TODO: See if this would be any faster with one of the dfs/bfs iterators in networkx
    while queue:
        node = queue.pop()
        orig_indices_list.extend(tree.nodes[node]['orig_indices'])
        for child in tree.successors(node):
            queue.append(child)
    
    return sorted(orig_indices_list) #sort it to account for any nondeterministic traversal order.

def _process_node_km(node, tree, subtree_weights, cut_edges, max_weight, root, new_roots):
    """
    Helper function for Kundu-Misra algorithm. This function processes each node
    by cutting edges with the highest weight children until the node's subtree weight
    is below the maximum weight threshold, updating the subtree weights of any ancestors
    as needed.
    """

    #if the subtree weight of this node is less than max weight we can stop right away
    #and avoid the sorting of the child weights.
    if subtree_weights[node]<=max_weight:
        return
    
    tree_nodes = tree.nodes
    #otherwise we will sort the weights of the child nodes to get the heaviest weight ones.
    #sorting by insertion order to ensure determinism.
    weighted_children = [(child, subtree_weights[child]) for child in 
                         sorted(tree.successors(node), key=lambda node: tree_nodes[node]['insertion_order']) ]
    sorted_weighted_children = sorted(weighted_children, key = lambda x: x[1], reverse=True)
    
    #get the path of nodes up to the root which need to have their weights updated upon edge removal.
    nodes_to_update = _path_to_root(tree, node, root)
        
    #remove the weightiest children until the weight is below the maximum weight.
    removed_child_index = 0 #track the index of the child being removed.
    while subtree_weights[node]>max_weight:
        removed_child =  sorted_weighted_children[removed_child_index][0]
        #add the edge to this child to the list of those cut.
        cut_edges.append((node, removed_child))
        new_roots.append(removed_child)
        removed_child_weight = subtree_weights[removed_child]
        #update the subtree weight of the current node and all parents up to the root.
        for node_to_update in nodes_to_update:
            subtree_weights[node_to_update]-= removed_child_weight
        #update index:
        removed_child_index+=1

def tree_partition_kundu_misra(tree, max_weight, weight_key='cost', test_leaves = True,
                               return_levels_and_weights=False, precomp_levels = None,
                               precomp_weights = None):
    """
    Algorithm for optimal minimum cardinality k-partition of tree (a partition
    of a tree into cluster of size at most k) based on a slightly less sophisticated
    implementation of the algorithm from "A Linear Tree Partitioning Algorithm"
    by Kundu and Misra (SIAM J. Comput. Vol. 6, No. 1, March 1977). Less sophisiticated
    because the strictly linear time implementation uses linear-time median estimation
    routine, while this implementation uses sorting (n log(n)-time), in practice it is
    likely that the highly-optimized C implementation of sorting would beat an uglier
    python implementation of median finding for most problem instances of interest anyhow.

    Parameters
    ----------
    tree : networkx.DiGraph
        An input graph representing the directed tree to perform partitioning on.
    
    max_weight : int
        Maximum node weight allowed for each partition.
    
    weight_key : str, optional (default 'cost')
        An optional string denoting the node attribute label to use for node weights
        in partitioning.

    test_leaves : bool, optional (default True)
        When True an initial test is performed to ensure that the weight of the leaves are all
        less than the maximum weight. Only turn off if you know for certain this is true.

    return_levels_and_weights : bool, optional (default False)
        If True return the constructed tree level structure (the lists of nodes partitioned
        by distance from the root) and subtree weights.

    precomp_levels : list of sets, optional (default None)
        A list where each set contains nodes that are equidistant from the root.

    precomp_weights : dict, optional (default None)
        A dictionary where keys are nodes and values are the total weights of the subtrees rooted at those nodes.
        
    Returns
    -------
    partitioned_tree : networkx.DiGraph
        A new DiGraph corresponding to the partitioned tree. I.e. a copy of the original
        tree with the requisite edge cuts performed.
 
    cut_edges : list of tuples
        A list of the parent-child node pairs whose edges were cut in partitioning the tree.

    
    """
    #create a copy of the input tree:
    #tree = _copy_networkx_graph(tree)
    
    cut_edges = [] #list of cut edges.
    new_roots = [] #list of the subtree root node in the partitioned tree

    #find the root node of tree:
    root = _find_root(tree)
    new_roots.append(root)

    tree_nodes = tree.nodes

    if test_leaves:
        #find the leaves:
        leaves = _find_leaves(tree)
        #make sure that the weights of the leaves are all less than the maximum weight.
        msg = 'The maximum node weight for at least one leaf is greater than the maximum weight, no partition possible.'
        assert all([tree_nodes[leaf][weight_key]<=max_weight for leaf in leaves]), msg
        
    #precompute a list of subtree weights which will be dynamically updated as we make cuts. Also
    #parition tree into levels based on distance from root.
    if precomp_levels is None and precomp_weights is None:
        tree_levels, subtree_weights = _partition_levels_and_compute_subtree_weights(tree, root, weight_key)
    else:
        tree_levels = precomp_levels if precomp_levels is not None else _partition_levels(tree, root)
        subtree_weights = precomp_weights.copy() if precomp_weights is not None else _compute_subtree_weights(tree, root, weight_key)
        
    #the subtree_weights get modified in-place by _process_node_km, so create a copy for the return value.
    if return_levels_and_weights:
        subtree_weights_orig = subtree_weights.copy()

    #begin processing the nodes level-by-level.
    for level in reversed(tree_levels):
        for node in level:
            _process_node_km(node, tree, subtree_weights, cut_edges, max_weight, root, new_roots)

    #sort the new root nodes in case there are determinism issues
    new_roots = sorted(new_roots, key=lambda node: tree_nodes[node]['insertion_order'])    
    
    if return_levels_and_weights:
        return cut_edges, new_roots, tree_levels, subtree_weights_orig
    else:
        return cut_edges, new_roots

def _bisect_tree(tree, subtree_root, subtree_weights, weight_key, root_cost = 0, target_proportion = .5):
    #perform a bisection on the subtree. Loop through the tree beginning at the root,
    #and find as cheap as possible of an edge which when cut approximately bisects the tree based on cost.
    
    heaviest_subtree_levels = _partition_levels(tree, subtree_root)
    new_subtree_cost = {}
    
    new_subtree_cost[subtree_root] =  subtree_weights[subtree_root]
    for i, level in enumerate(heaviest_subtree_levels[1:]): #skip the root.
        for node in level:
            #calculate the cost of a new subtree rooted at this node. This is the current cost
            #plus the current level plus the propagation cost of the current root.
            new_subtree_cost[node] = subtree_weights[node] + i + root_cost if weight_key == 'prop_cost' else subtree_weights[node]
    
    #find the node that results in as close as possible to a bisection of the subtree
    #in terms of propagation cost.
    target_prop_cost = new_subtree_cost[subtree_root] * target_proportion
    closest_node = subtree_root
    closest_distance = new_subtree_cost[subtree_root]
    for node, cost in new_subtree_cost.items(): #since the nodes in each level are sorted this should be alright for determinism.
        current_distance = abs(cost - target_prop_cost)
        if current_distance < closest_distance:
            closest_distance = current_distance
            closest_node = node
    #we now have the node which when promoted to a root produces the tree closest to a bisection in terms of propagation
    #cost possible. Let's perform that bisection now.
    if closest_node is not subtree_root:
        #since a tree should only be one predecessor, so don't need to worry about determinism.
        cut_edge = (list(tree.predecessors(closest_node))[0], closest_node)
        return cut_edge, (new_subtree_cost[closest_node], subtree_weights[subtree_root] - subtree_weights[closest_node])
    else:
        return None, None

def _bisection_pass(partitioned_tree, cut_edges, new_roots, num_sub_tables, weight_key):
    partitioned_tree = _copy_networkx_graph(partitioned_tree)
    subtree_weights = [(root, _compute_subtree_weights(partitioned_tree, root, weight_key)) for root in new_roots]
    sorted_subtree_weights = sorted(subtree_weights, key=lambda x: x[1][x[0]], reverse=True)

    #perform a bisection on the heaviest subtree. Loop through the tree beginning at the root,
    #and find as cheap as possible of an edge which when cut approximately bisects the tree based on cost.
    for i in range(len(sorted_subtree_weights)):
        heaviest_subtree_root = sorted_subtree_weights[i][0]
        heaviest_subtree_weights = sorted_subtree_weights[i][1]
        root_cost =  partitioned_tree.nodes[heaviest_subtree_root][weight_key] if weight_key == 'prop_cost' else 0
        cut_edge, new_subtree_costs = _bisect_tree(partitioned_tree, heaviest_subtree_root, heaviest_subtree_weights, weight_key, root_cost)
        if cut_edge is not None:
            cut_edges.append(cut_edge)
            new_roots.append(cut_edge[1])
            #cut the prescribed edge.
            partitioned_tree.remove_edge(cut_edge[0], cut_edge[1])
        #check whether we need to continue paritioning subtrees.
        if len(new_roots) == num_sub_tables:
            break
    #sort the new root nodes in case there are determinism issues
    new_roots = sorted(new_roots, key=lambda node: partitioned_tree.nodes[node]['insertion_order'])    

    return partitioned_tree, new_roots, cut_edges

def _refinement_pass(partitioned_tree, roots, weight_key, imbalance_threshold=1.2, minimum_improvement_threshold = .1):
    #refine the partitioning to improve the balancing of the specified weights across the
    #subtrees.
    #start by recomputing the latest subtree weights and ranking them from heaviest to lightest.
    partitioned_tree = _copy_networkx_graph(partitioned_tree)
    subtree_weights = [(root, _compute_subtree_weights(partitioned_tree, root, weight_key)) for root in roots]
    sorted_subtree_weights = sorted(subtree_weights, key=lambda x: x[1][x[0]], reverse=True)

    partitioned_tree_nodes = partitioned_tree.nodes

    #Strategy: pair heaviest and lightest subtrees and identify the subtree in the heaviest that could be
    #snipped out and added to the lightest to bring their weights as close as possible.
    #Next do this for the second heaviest and second lightest, etc. 
    #Only do so while the imbalance threshold, the ratio between the heaviest and lightest subtrees, is
    #above a specified threshold.
    heavy_light_pairs = _pair_elements(sorted_subtree_weights)
    heavy_light_pair_indices = _pair_elements(list(range(len(sorted_subtree_weights))))
    heavy_light_weights = [(sorted_subtree_weights[i][1][sorted_subtree_weights[i][0]], sorted_subtree_weights[j][1][sorted_subtree_weights[j][0]])
                          for i,j in heavy_light_pair_indices]
    heavy_light_ratios = [weight_1/weight_2 for weight_1,weight_2 in heavy_light_weights]

    heavy_light_pairs_to_balance = heavy_light_pairs if len(sorted_subtree_weights)%2==0 else heavy_light_pairs[0:-1]
    new_roots = []
    addl_cut_edges = []
    pair_iter = iter(range(len(heavy_light_pairs_to_balance)))
    for i in pair_iter:
        #if the ratio is above the threshold then try a rebalancing
        #step.
        if heavy_light_ratios[i] > imbalance_threshold:
            #calculate the fraction of the heavy tree that would be needed to bring the weight of the
            #lighter tree in line.
            root_cost =  partitioned_tree_nodes[heavy_light_pairs[i][0][0]][weight_key] if weight_key == 'prop_cost' else 0

            rebalancing_target_fraction = (0.5*(heavy_light_weights[i][0] - heavy_light_weights[i][1]))/heavy_light_weights[i][0]
            cut_edge, new_subtree_weights =_bisect_tree(partitioned_tree, heavy_light_pairs[i][0][0], heavy_light_pairs[i][0][1], 
                                                        weight_key, root_cost = root_cost,
                                                        target_proportion = rebalancing_target_fraction)
            #before applying the edge cut check whether the edge we found was close enough
            # to bring us below the threshold.
            if cut_edge is not None:
                new_light_tree_weight = new_subtree_weights[0] + heavy_light_weights[i][1]
                new_heavy_tree_weight = new_subtree_weights[1]
                new_heavy_light_ratio = new_heavy_tree_weight/new_light_tree_weight
                if new_heavy_light_ratio > imbalance_threshold and \
                    (heavy_light_ratios[i] - new_heavy_light_ratio)<minimum_improvement_threshold:
                    #We're only as good as the worst balancing, so if we are unable to 
                    #balance below the threshold and the improvement is below some minimum threshold 
                    #then we won't make an update and will terminate. 
                    #Maybe we should throw a warning too?
                    #but it isn't clear whether that would just be confusing to end-users who wouldn't
                    #know what was meant or if it was important.
                    #also add the roots of any of the pairs we haven't yet processed.
                    remaining_indices = [i] + [j for j in pair_iter]
                    for idx in remaining_indices:
                        new_roots.extend((heavy_light_pairs[idx][0][0], heavy_light_pairs[idx][1][0]))
                    break

                else:
                    #append the original root of the heavy tree, and a tuple of roots for the light plus the
                    #bisected part of the heavy.
                    new_roots.append(heavy_light_pairs[i][0][0])
                    new_roots.append((heavy_light_pairs[i][1][0], cut_edge[1]))
                    addl_cut_edges.append(cut_edge)
                    #apply the cut
                    partitioned_tree.remove_edge(cut_edge[0],cut_edge[1])
            else:
                #if the cut edge is None append the original heavy and light roots.
                new_roots.extend((heavy_light_pairs[i][0][0], heavy_light_pairs[i][1][0]))
        #since we're pairing up subsequent pairs of heavy and light
        #elements, once we see one which is sufficiently balanced we
        #know the rest must be.
        else:
            remaining_indices = [i] + [j for j in pair_iter]
            for idx in remaining_indices:
                new_roots.extend((heavy_light_pairs[idx][0][0], heavy_light_pairs[idx][1][0]))
            break

    #if the number of subtrees was odd to start we need to append on the median weight element which hasn't
    #been processed.
    if len(sorted_subtree_weights)%2!=0:
        new_roots.append(heavy_light_pairs[-1][0][0])

    return partitioned_tree, new_roots, addl_cut_edges


#helper function for pairing up heavy and light subtrees.
def _pair_elements(lst):
    paired_list = []
    length = len(lst)
    
    for i in range((length + 1) // 2):
        if i == length - i - 1:
            paired_list.append((lst[i], lst[i]))
        else:
            paired_list.append((lst[i], lst[length - i - 1]))
    
    return paired_list

                     