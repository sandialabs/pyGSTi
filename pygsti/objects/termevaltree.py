"""
Defines the TermEvalTree class which implements an evaluation tree.
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
import copy as _copy

from .verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from ..tools import slicetools as _slct
from ..tools import mpitools as _mpit
from .evaltree import EvalTree
from .opcalc import compact_deriv as _compact_deriv, bulk_eval_compact_polys_complex as _bulk_eval_compact_polys_complex

import time as _time  # DEBUG TIMERS


class TermEvalTree(EvalTree):
    """
    An Evaluation Tree for Taylor-term-based (path integral) calculations.

    Parameters
    ----------
    items : list, optional
        Initial items.  This argument should only be used internally
        in the course of serialization.
    """

    def __init__(self, items=[]):
        """ Create a new, empty, evaluation tree. """
        # Trivially init other members - to be filled in by initialize() or by subtree creation
        self.simplified_circuit_elabels = None

        super(TermEvalTree, self).__init__(items)

    def initialize(self, simplified_circuit_elabels, num_sub_tree_comms=1, max_cache_size=None):
        """
        Initialize an evaluation tree using a set of "simplified" circuits.

        This function must be called before using this evaluation tree.

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
            The maximum cache size allowed. Currently this is
            unused and should be treated as zero.

        Returns
        -------
        None
        """
        #tStart = _time.time() #DEBUG TIMER

        # opLabels : A list of all the distinct operation labels found in
        #              simplified_circuit_elabels.  Used in calc classes
        #              as a convenient precomputed quantity.
        self.opLabels = self._get_op_labels(simplified_circuit_elabels)
        if num_sub_tree_comms is not None:
            self.distribution['numSubtreeComms'] = num_sub_tree_comms

        circuit_list = [tuple(opstr) for opstr in simplified_circuit_elabels.keys()]
        self.simplified_circuit_elabels = list(simplified_circuit_elabels.values())
        self.simplified_circuit_nEls = list(map(len, self.simplified_circuit_elabels))
        self.num_final_els = sum([len(v) for v in self.simplified_circuit_elabels])
        #self._compute_finalStringToEls() #depends on simplified_circuit_spamTuples
        #UNNEEDED? self.recompute_spamtuple_indices(local=True)  # local shouldn't matter here

        #Evaluation tree:
        # A list of tuples, where each element contains
        #  information about evaluating a particular operation sequence:
        #  (iStart, tuple_of_following_gatelabels )
        # and self.eval_order specifies the evaluation order.
        del self[:]  # clear self (a list)
        self[:] = circuit_list
        self.num_final_strs = len(circuit_list)

        #DON'T sort b/c then we'd need to keep track of element ordering
        # so that we can build arrays of probabilities for all the elements
        # in the appropriate order.
        # FUTURE TODO: clean up this class to take advantage of the fact that the evaluation order is linear.

        self.eval_order = list(range(self.num_final_strs))

        #Storage for polynomial expressions for probabilities and
        # their derivatives

        # cache of the high-magnitude terms (actually their represenations), which
        # together with the per-circuit threshold given in `percircuit_p_polys`,
        # defines a set of paths to use in probability computations.
        self.pathset = None
        self.percircuit_p_polys = {}  # keys = circuits, values = (threshold, compact_polys)

        self.merged_compact_polys = None
        self.merged_achievedsopm_compact_polys = None

        self.myFinalToParentFinalMap = None  # this tree has no "children",
        self.myFinalElsToParentFinalElsMap = None  # i.e. has not been created by a 'split'
        self.parentIndexMap = None
        self.original_index_lookup = None
        self.subTrees = []  # no subtrees yet
        assert(self.generate_circuit_list() == circuit_list)
        assert(None not in circuit_list)

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

        #Build rest of strings
        for i in self.get_evaluation_order():
            circuits[i] = self[i]

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

    def split(self, el_indices_dict, max_sub_tree_size=None, num_sub_trees=None, verbosity=0):
        """
        Split this tree into sub-trees.

        This is done in order to reduce the maximum size of any tree (useful for
        limiting memory consumption or for using multiple cores).  Must specify
        either max_sub_tree_size or num_sub_trees.

        Parameters
        ----------
        el_indices_dict : dict
            A dictionary whose keys are integer original-circuit indices
            and whose values are slices or index arrays of final-element-
            indices (typically this dict is returned by calling
            :method:`Model.simplify_circuits`).  Since splitting a
            tree often involves permutation of the raw string ordering
            and thereby the element ordering, an updated version of this
            dictionary, with all permutations performed, is returned.

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
        OrderedDict
            A updated version of el_indices_dict
        """
        #dbList = self.generate_circuit_list()
        tm = _time.time()
        printer = _VerbosityPrinter.build_printer(verbosity)

        if (max_sub_tree_size is None and num_sub_trees is None) or \
           (max_sub_tree_size is not None and num_sub_trees is not None):
            raise ValueError("Specify *either* max_sub_tree_size or num_sub_trees")
        if num_sub_trees is not None and num_sub_trees <= 0:
            raise ValueError("EvalTree split() error: num_sub_trees must be > 0!")

        #Don't split at all if it's unnecessary
        if max_sub_tree_size is None or len(self) < max_sub_tree_size:
            if num_sub_trees is None or num_sub_trees == 1: return el_indices_dict

        self.subTrees = []
        evalOrder = self.get_evaluation_order()
        printer.log("EvalTree.split done initial prep in %.0fs" %
                    (_time.time() - tm)); tm = _time.time()

        def create_subtrees(max_cost, max_cost_rate=0, cost_metric="size"):
            """
            Find a set of subtrees by iterating through the tree
            and placing "break" points when the cost of evaluating the
            subtree exceeds some 'max_cost'.  This ensure ~ equal cost
            trees, but doesn't ensure any particular number of them.

            max_cost_rate can be set to implement a varying max_cost
            over the course of the iteration.
            """

            if cost_metric == "applys":
                def cost_fn(rem): return len(rem)  # length of remainder = #-apply ops needed
            elif cost_metric == "size":
                def cost_fn(rem): return 1  # everything costs 1 in size of tree
            else: raise ValueError("Uknown cost metric: %s" % cost_metric)

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

                if curTreeCost + cost < max_cost:
                    #Just add current string to current tree
                    curTreeCost += cost
                    curSubTree.update(inds)
                else:
                    #End the current tree and begin a new one
                    #print("cost %d+%d exceeds %d" % (curTreeCost,cost,max_cost))
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

                max_cost += max_cost_rate

            subTrees.append(curSubTree)
            totalCost += curTreeCost
            return subTrees, totalCost

        ##################################################################
        # Part I: find a list of where the current tree should be broken #
        ##################################################################

        subTreeSetList = []
        if num_sub_trees is not None:

            subTreeSize = len(self) // num_sub_trees
            for i in range(num_sub_trees):
                end = (i + 1) * subTreeSize if (i < num_sub_trees - 1) else len(self)
                subTreeSetList.append(set(range(i * subTreeSize, end)))

        else:  # max_sub_tree_size is not None
            k = 0
            while k < len(self):
                end = min(k + max_sub_tree_size, len(self))
                subTreeSetList.append(set(range(k, end)))
                k = end

        ##########################################################
        # Part II: create subtrees from index sets
        ##########################################################
        # (common logic provided by base class up to providing a few helper fns)

        def permute_parent_element(perm, el):
            """Applies a permutation to an element of the tree """
            # perm[oldIndex] = newIndex
            return el  # no need to permute operation sequence

        def create_subtree(parent_indices, num_final, full_eval_order, slice_into_parents_final_array, parent_tree):
            """
            Creates a subtree given requisite information:

            Parameters
            ----------
            parent_indices : list
                The ordered list of (parent-tree) indices to be included in
                the created subtree.

            num_final : int
                The number of "final" elements, i.e. those that are used to
                construct the final array of results and not just an intermediate.
                The first num_final elemements of parent_indices are "final", and
                'slice_into_parents_final_array' tells you which final indices of
                the parent they map to.

            full_eval_order : list
                A list of the integers between 0 and len(parent_indices)-1 which
                gives the evaluation order of the subtree *including* evaluation
                of any initial elements.

            slice_into_parents_final_array : slice
                Described above - map between to-be-created subtree's final
                elements and parent-tree indices.

            parent_tree : EvalTree
                The parent tree itself.
            """
            subTree = TermEvalTree()
            subTree.myFinalToParentFinalMap = slice_into_parents_final_array
            subTree.num_final_strs = num_final
            subTree[:] = [None] * len(parent_indices)
            subTree.highmag_termrep_cache = {}
            subTree.circuitsetup_cache = {}
            subTree.percircuit_p_polys = {}
            subTree.merged_compact_polys = None

            for ik in full_eval_order:  # includes any initial indices
                k = parent_indices[ik]  # original tree index
                circuit = self[k]  # original tree data
                subTree.eval_order.append(ik)
                assert(subTree[ik] is None)
                subTree[ik] = circuit

            subTree.parentIndexMap = parent_indices  # parent index of each subtree index
            subTree.simplified_circuit_elabels = [self.simplified_circuit_elabels[kk]
                                                  for kk in _slct.indices(subTree.myFinalToParentFinalMap)]
            subTree.simplified_circuit_nEls = list(map(len, self.simplified_circuit_elabels))
            #subTree._compute_finalStringToEls() #depends on simplified_circuit_spamTuples

            final_el_startstops = []; i = 0
            for elabels in parent_tree.simplified_circuit_elabels:
                final_el_startstops.append((i, i + len(elabels)))
                i += len(elabels)
            subTree.myFinalElsToParentFinalElsMap = _np.concatenate(
                [_np.arange(*final_el_startstops[kk])
                 for kk in _slct.indices(subTree.myFinalToParentFinalMap)])
            #Note: myFinalToParentFinalMap maps only between *final* elements
            #   (which are what is held in simplified_circuit_spamTuples)

            subTree.num_final_els = sum([len(v) for v in subTree.simplified_circuit_elabels])
            #NEEDED? subTree.recompute_spamtuple_indices(local=False)

            circuits = subTree.generate_circuit_list(permute=False)
            subTree.opLabels = self._get_op_labels(
                {c: elbls for c, elbls in zip(circuits, subTree.simplified_circuit_elabels)})

            return subTree

        updated_elIndices = self._finish_split(el_indices_dict, subTreeSetList,
                                               permute_parent_element, create_subtree)

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
            self._permute_simplified_circuit_xs(self.simplified_circuit_elabels,
                                                element_indices_dict, old_indices_in_new_order)
        self.simplified_circuit_nEls = list(map(len, self.simplified_circuit_elabels))
        return updated_elIndices

    def cache_size(self):
        """
        Returns the size of the persistent "cache" of partial results.

        In the case of a TermEvalTree, this is always 0.

        Returns
        -------
        int
        """
        return 0

    def copy(self):
        """
        Create a copy of this evaluation tree.

        Returns
        -------
        TermEvalTree
        """
        cpy = self._copy_base(TermEvalTree(self[:]))
        cpy.opLabels = self.opLabels[:]
        cpy.simplified_circuit_elabels = _copy.deepcopy(self.simplified_circuit_elabels)
        return cpy

    def get_achieved_and_max_sopm(self, calc):
        """
        Compute the achieved and maximum possible sum-of-path-magnitudes.

        This gives a sense of how accurately the current path set is able
        to compute probabilities.

        Parameters
        ----------
        calc : TermForwardSimulator
            The forward simulator object.

        Returns
        -------
        numpy.ndarray
        """
        achieved_sopm = []
        max_sopm = []

        for i in self.get_evaluation_order():  # uses *linear* evaluation order so we know final indices are sequential
            circuit = self[i]
            # must have selected a set of paths for this to be populated!
            current_threshold, _ = self.percircuit_p_polys[circuit]

            rholabel = circuit[0]
            opstr = circuit[1:]
            elabels = self.simplified_circuit_elabels[i]

            achieved, maxx = calc.circuit_achieved_and_max_sopm(rholabel,
                                                                elabels,
                                                                opstr,
                                                                self.pathset.highmag_termrep_cache,
                                                                calc.sos.opcache,
                                                                current_threshold)
            achieved_sopm.extend(list(achieved))
            max_sopm.extend(list(maxx))

        assert(len(achieved_sopm) == len(max_sopm) == self.num_final_elements())
        return _np.array(achieved_sopm, 'd'), _np.array(max_sopm, 'd')

    def get_sopm_gaps(self, calc):
        """
        Compute the sum-of-path-magnitude gaps.

        The "gap" here is the difference between the maximum possible and
        achieved values.

        Parameters
        ----------
        calc : TermForwardSimulator
            The forward simulator object.

        Returns
        -------
        numpy.ndarray
        """
        achieved_sopm, max_sopm = self.get_achieved_and_max_sopm_gaps(calc)
        return max_sopm - achieved_sopm

    def get_achieved_and_max_sopm_jacobian(self, calc):
        """
        Compute the jacobian of the achieved and maximum possible sum-of-path-magnitudes.

        Parameters
        ----------
        calc : TermForwardSimulator
            The forward simulator object.

        Returns
        -------
        achieved_sopm_jacobian: numpy.ndarray
            The jacobian of the achieved sum-of-path-magnitudes.

        max_sopm_jacobian: numpy.ndarray
            The jacobian of the maximum possible sum-of-path-magnitudes.
        """
        nEls = self.num_final_elements()
        polys = self.merged_achievedsopm_compact_polys
        dpolys = _compact_deriv(polys[0], polys[1], _np.arange(calc.Np))
        d_achieved_mags = _bulk_eval_compact_polys_complex(
            dpolys[0], dpolys[1], _np.abs(calc.paramvec), (nEls, calc.Np))
        assert(_np.linalg.norm(_np.imag(d_achieved_mags)) < 1e-8)
        d_achieved_mags = d_achieved_mags.real
        d_achieved_mags[:, (calc.paramvec < 0)] *= -1

        d_max_sopms = _np.empty((nEls, calc.Np), 'd')
        k = 0  # current element position for loop below

        opcache = calc.sos.opcache
        # uses *linear* evaluation order so we know final indices are sequential
        for iCircuit in self.get_evaluation_order():
            circuit = self[iCircuit]

            rholabel = circuit[0]
            opstr = circuit[1:]
            elabels = self.simplified_circuit_elabels[iCircuit]

            #Get MAX-SOPM for circuit outcomes and thereby the SOPM gap (via MAX - achieved)
            # Here we take d(MAX) (above merged_achievedsopm_compact_polys give d(achieved)).  Since each
            # MAX-SOPM value is a product of max term magnitudes, to get deriv we use the chain rule:
            partial_ops = [opcache[rholabel] if rholabel in opcache else calc.sos.get_prep(rholabel)]
            for glbl in opstr:
                partial_ops.append(opcache[glbl] if glbl in opcache else calc.sos.get_operation(glbl))
            Eops = [(opcache[elbl] if elbl in opcache else calc.sos.get_effect(elbl)) for elbl in elabels]
            partial_op_maxmag_values = [op.get_total_term_magnitude() for op in partial_ops]
            Eop_maxmag_values = [Eop.get_total_term_magnitude() for Eop in Eops]
            maxmag_partial_product = _np.product(partial_op_maxmag_values)
            maxmag_products = [maxmag_partial_product * Eop_val for Eop_val in Eop_maxmag_values]

            deriv = _np.zeros((len(elabels), calc.Np), 'd')
            for i in range(len(partial_ops)):  # replace i-th element of product with deriv
                dop_local = partial_ops[i].get_total_term_magnitude_deriv()
                dop_global = _np.zeros(calc.Np, 'd')
                dop_global[partial_ops[i].gpindices] = dop_local
                dop_global /= partial_op_maxmag_values[i]

                for j in range(len(elabels)):
                    deriv[j, :] += dop_global * maxmag_products[j]

            for j in range(len(elabels)):  # replace final element with appropriate derivative
                dop_local = Eops[j].get_total_term_magnitude_deriv()
                dop_global = _np.zeros(calc.Np, 'd')
                dop_global[Eops[j].gpindices] = dop_local
                dop_global /= Eop_maxmag_values[j]
                deriv[j, :] += dop_global * maxmag_products[j]

            d_max_sopms[k:k + len(elabels), :] = deriv
            k += len(elabels)

        return d_achieved_mags, d_max_sopms

    def get_sopm_gaps_jacobian(self, calc):
        """
        Compute the jacobian of the (maximum-possible - achieved) sum-of-path-magnitudes.

        Parameters
        ----------
        calc : TermForwardSimulator
            The forward simulator object.

        Returns
        -------
        numpy.ndarray
            The jacobian of the sum-of-path-magnitudes gap.
        """
        d_achieved_mags, d_max_sopms = self.get_achieved_and_max_sopm_jacobian(calc)
        dgaps = d_max_sopms - d_achieved_mags
        return dgaps

    def find_minimal_paths_set(self, calc, comm, mem_limit, exit_after_this_many_failures=0):
        """
        Find the minimal (smallest) path set that achieves the desired accuracy conditions.

        Parameters
        ----------
        calc : TermForwardSimulator
            The forward simulator object.

        comm : mpy4py.MPI.Comm
            An MPI communicator for dividing the compuational task.

        mem_limit : int
            Rough memory limit (per processor) in bytes.

        exit_after_this_many_failures : int, optional
           If > 0, give up after this many circuits fail to meet the desired accuracy criteria.
           This short-circuits doomed attempts to find a good path set so they don't take too long.

        Returns
        -------
        PathSet
        """
        tot_npaths = 0
        tot_target_sopm = 0
        tot_achieved_sopm = 0

        #We're only testing how many failures there are, don't update the "locked in" persistent
        # set of paths given by self.percircuit_p_polys and self.pathset.highmag_termrep_cache - just use a
        # temporary repcache.
        repcache = {}
        circuitsetup_cache = {}
        thresholds = {}

        num_failed = 0  # number of circuits which fail to achieve the target sopm
        failed_circuits = []

        for i in self.get_evaluation_order():  # uses *linear* evaluation order so we know final indices are sequential
            circuit = self[i]
            rholabel = circuit[0]
            opstr = circuit[1:]
            elabels = self.simplified_circuit_elabels[i]

            npaths, threshold, target_sopm, achieved_sopm = \
                calc.compute_pruned_pathmag_threshold(rholabel,
                                                      elabels,
                                                      opstr,
                                                      repcache,
                                                      calc.sos.opcache,
                                                      circuitsetup_cache,
                                                      comm,
                                                      mem_limit,
                                                      None)  # guess?
            thresholds[circuit] = threshold

            if achieved_sopm < target_sopm:
                num_failed += 1
                failed_circuits.append((i, circuit))  # (circuit,npaths, threshold, target_sopm, achieved_sopm))
                if exit_after_this_many_failures > 0 and num_failed == exit_after_this_many_failures:
                    return UnsplitTreeTermPathSet(self, None, None, None, 0, 0, num_failed)

            tot_npaths += npaths
            tot_target_sopm += target_sopm
            tot_achieved_sopm += achieved_sopm

        #if comm is None or comm.Get_rank() == 0:
        rankStr = "Rank%d: " % comm.Get_rank() if comm is not None else ""
        nC = self.num_final_strings()
        max_npaths = calc.max_paths_per_outcome * self.num_final_elements()
        print(("%sPruned path-integral: kept %d paths (%.1f%%) w/magnitude %.4g "
               "(target=%.4g, #circuits=%d, #failed=%d)") %
              (rankStr, tot_npaths, 100 * tot_npaths / max_npaths, tot_achieved_sopm, tot_target_sopm, nC, num_failed))
        print("%s  (avg per circuit paths=%d, magnitude=%.4g, target=%.4g)" %
              (rankStr, tot_npaths // nC, tot_achieved_sopm / nC, tot_target_sopm / nC))

        return UnsplitTreeTermPathSet(self, thresholds, repcache,
                                      circuitsetup_cache, tot_npaths,
                                      max_npaths, num_failed)

    def get_paths_set(self):
        """
        The current path set.

        Returns
        -------
        PathSet
        """
        return self.pathset

    def select_paths_set(self, calc, pathset, comm, mem_limit):
        """
        Selects (makes "current") a path set *and* computes polynomials the new set.

        Parameters
        ----------
        calc : TermForwardSimulator
            The forward simulator object.

        pathset : PathSet
            The path set to select.

        comm : mpy4py.MPI.Comm
            An MPI communicator for dividing the compuational task.

        mem_limit : int
            Rough memory limit (per processor) in bytes.

        Returns
        -------
        None
        """

        #TODO: update this outdated docstring
        # We're finding and "locking in" a set of paths to use in subsequent evaluations.  This
        # means we're going to re-compute the high-magnitude terms for each operation (in
        # self.pathset.highmag_termrep_cache) and re-compute the thresholds (in self.percircuit_p_polys)
        # for each circuit (using the computed high-magnitude terms).  This all occurs for
        # the particular current value of the parameter vector (found via calc.to_vector());
        # these values determine what is a "high-magnitude" term and the path magnitudes that are
        # summed to get the overall sum-of-path-magnitudes for a given circuit outcome.

        self.pathset = pathset
        self.percircuit_p_polys = {}
        repcache = self.pathset.highmag_termrep_cache
        circuitsetup_cache = self.pathset.circuitsetup_cache
        thresholds = self.pathset.thresholds

        all_compact_polys = []  # holds one compact polynomial per final *element*

        for i in self.get_evaluation_order():  # uses *linear* evaluation order so we know final indices are sequential
            circuit = self[i]
            #print("Computing pruned poly %d" % i)

            threshold = thresholds[circuit]
            rholabel = circuit[0]
            opstr = circuit[1:]
            elabels = self.simplified_circuit_elabels[i]

            raw_polyreps = calc.prs_as_pruned_polyreps(threshold,
                                                       rholabel,
                                                       elabels,
                                                       opstr,
                                                       repcache,
                                                       calc.sos.opcache,
                                                       circuitsetup_cache,
                                                       comm,
                                                       mem_limit)

            compact_polys = [polyrep.compact_complex() for polyrep in raw_polyreps]
            self.percircuit_p_polys[circuit] = (threshold, compact_polys)
            all_compact_polys.extend(compact_polys)  # ok b/c *linear* evaluation order

        tapes = all_compact_polys  # each "compact polynomials" is a (vtape, ctape) 2-tupe
        vtape = _np.concatenate([t[0] for t in tapes])  # concat all the vtapes
        ctape = _np.concatenate([t[1] for t in tapes])  # concat all teh ctapes
        self.merged_compact_polys = (vtape, ctape)  # Note: ctape should always be complex here

        return

    def cache_p_polys(self, calc, comm):
        """
        Compute and cache the compact-form polynomials that evaluate to this tree's probabilities.

        These polynomials corresponding to all this tree's operation sequences sandwiched
        between each state preparation and effect.  The result is cached to speed
        up subsequent calls.

        Parameters
        ----------
        calc : TermForwardSimulator
            A calculator object for computing the raw polynomials (if necessary)

        comm : mpi4py.MPI.Comm
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        Returns
        -------
        None
        """
        #Otherwise compute poly
        all_compact_polys = []  # holds one compact polynomial per final *element*
        for i in self.get_evaluation_order():  # uses *linear* evaluation order so we know final indices are sequential
            circuit = self[i]

            if circuit in self.percircuit_p_polys:
                compact_polys = self.percircuit_p_polys[circuit]
            else:
                rholabel = circuit[0]
                opstr = circuit[1:]
                elabels = self.simplified_circuit_elabels[i]
                compact_polys = calc.prs_as_compact_polys(rholabel, elabels, opstr, comm)

            all_compact_polys.extend(compact_polys)  # ok b/c *linear* evaluation order

        tapes = all_compact_polys  # each "compact polynomials" is a (vtape, ctape) 2-tupe
        vtape = _np.concatenate([t[0] for t in tapes])  # concat all the vtapes
        ctape = _np.concatenate([t[1] for t in tapes])  # concat all teh ctapes

        self.merged_compact_polys = (vtape, ctape)  # Note: ctape should always be complex here


class TermPathSet(object):
    """
    A set of error-term paths.

    Each such path is comprised of a single "term" (usually a Taylor term of an
    error-generator expansion) for each gate operation or circuit layer (more
    generally, each factor within the product that evaluates to the probability).

    A set of paths is specified by giving a path-magnitude threshold for each
    circuit in an evaluation tree.  All paths with magnitude less than this threshold
    are a part of the path set.  The term magnitudes that determine a path magnitude
    are held in Term objects resulting from a Model at a particular parameter-space
    point.  Representations of these term objects (actually just the "high-magnitude" ones,
    as determined by a different, term-magnitude, threshold) are also held
    within the path set.

    Parameters
    ----------
    evaltree : TermEvalTree
        The evaluation tree that is associated with this path set.

    npaths : int
        The number of total paths.

    maxpaths : int
        The maximum-allowed-paths limit that was in place when this
        path set was created.

    nfailed : int
        The number of circuits that failed to meet the desired accuracy
        (path-magnitude gap) requirements.
    """

    def __init__(self, evaltree, npaths, maxpaths, nfailed):
        self.tree = evaltree
        self.npaths = npaths
        self.max_allowed_paths = maxpaths
        self.num_failures = nfailed  # number of failed *circuits* (not outcomes)

    def get_allowed_path_fraction(self):
        """
        The fraction of maximal allowed paths that are in this path set.

        Returns
        -------
        float
        """
        return self.npaths / self.max_allowed_paths


class UnsplitTreeTermPathSet(TermPathSet):
    """
    A path set for an un-split :class:`TermEvalTree`.

    Parameters
    ----------
    evaltree : TermEvalTree
        The evaluation tree this path set is associated with.

    thresholds : dict
        A dictionary whose keys are circuits and values are path-magnitude thresholds.
        These thresholds store what

    highmag_termrep_cache : dict
        A dictionary whose keys are gate or circuit-layer labels and whose values are
        internally-used "rep-cache" elements that each hold a list of the term representations
        for that gate having a "high" magnitude (magnitude above some threshold).  This
        cache is an essential link between the path-magnitude thresholds in `thresholds` and
        the actual set of paths that are evaluated by `evaltree` (e.g. updating this cache by
        re-computing term magnitudes at a new parameter-space point will also update the set
        of paths that are evaluated given the *same* set of `thresholds`).

    circuitsetup_cache : dict
        A dictionary that caches per-circuit setup information and can be used to
        speed up multiple calls which use the same circuits.

    npaths : int
        The number of total paths.

    maxpaths : int
        The maximum-allowed-paths limit that was in place when this
        path set was created.

    nfailed : int
        The number of circuits that failed to meet the desired accuracy
        (path-magnitude gap) requirements.
    """
    def __init__(self, evaltree, thresholds, highmag_termrep_cache,
                 circuitsetup_cache, npaths, maxpaths, nfailed):
        TermPathSet.__init__(self, evaltree, npaths, maxpaths, nfailed)
        self.thresholds = thresholds
        self.highmag_termrep_cache = highmag_termrep_cache
        self.circuitsetup_cache = circuitsetup_cache


class SplitTreeTermPathSet(TermPathSet):
    """
    A path set for a split :class:`TermEvalTree`.

    Parameters
    ----------
    evaltree : TermEvalTree
        The evaluation tree this path set is associated with.

    local_subtree_pathsets : list
        A list of path sets for each of the *local* sub-trees (i.e. the
        sub-trees assigned to the current processor).

    comm : mpi4py.MPI.Comm
        When not None, an MPI communicator for distributing the computation
        across multiple processors.
    """
    def __init__(self, evaltree, local_subtree_pathsets, comm):

        #Get local-subtree totals
        nTotPaths = sum([sps.npaths for sps in local_subtree_pathsets])
        nTotFailed = sum([sps.num_failures for sps in local_subtree_pathsets])
        nAllowed = sum([sps.max_allowed_paths for sps in local_subtree_pathsets])

        #Get global totals
        nTotFailed = _mpit.sum_across_procs(nTotFailed, comm)
        nTotPaths = _mpit.sum_across_procs(nTotPaths, comm)
        nAllowed = _mpit.sum_across_procs(nAllowed, comm)

        TermPathSet.__init__(self, evaltree, nTotPaths, nAllowed, nTotFailed)
        self.local_subtree_pathsets = local_subtree_pathsets
