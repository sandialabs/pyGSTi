""" Defines the TermEvalTree class which implements an evaluation tree. """
from __future__ import division, print_function, absolute_import, unicode_literals
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

from ..baseobjs import VerbosityPrinter as _VerbosityPrinter
from ..tools import slicetools as _slct
from .evaltree import EvalTree

try:
    from .fastopcalc import fast_compact_deriv as _compact_deriv
    from .fastopcalc import fast_bulk_eval_compact_polys_complex as _bulk_eval_compact_polys_complex

    #DEBUG from .polynomial import compact_deriv as _compact_deriv
    # from . import fastopcalc
    # from . import polynomial
    # def _compact_deriv(vtape, ctape, wrtParams):
    #     v1,c1 = fastopcalc.fast_compact_deriv(vtape,ctape,wrtParams)
    #     v2,c2 = polynomial.compact_deriv(vtape,ctape,wrtParams)
    #     print("SIZES = ",v1.shape, c1.shape, v2.shape, c2.shape)
    #     assert(_np.linalg.norm(v1-v2) < 1e-6)
    #     assert(_np.linalg.norm(c1-c2) < 1e-6)
    #     return v1,c1

except ImportError:
    from .polynomial import compact_deriv as _compact_deriv
    from .polynomial import bulk_eval_compact_polys as _bulk_eval_compact_polys
    def _bulk_eval_compact_polys_complex(vtape, ctape, paramvec, dest_shape):
        return _bulk_eval_compact_polys(vtape, ctape, paramvec, dest_shape, "complex")


import time as _time  # DEBUG TIMERS


class TermEvalTree(EvalTree):
    """
    An Evaluation Tree for term-based calcualtions.
    """

    def __init__(self, items=[]):
        """ Create a new, empty, evaluation tree. """
        # list of the operation labels
        self.opLabels = []

        # Trivially init other members - to be filled in by initialize() or by subtree creation
        self.simplified_circuit_elabels = None

        super(TermEvalTree, self).__init__(items)

    def initialize(self, simplified_circuit_list, numSubTreeComms=1, maxCacheSize=None):
        """
        Initialize an evaluation tree using a set of complied operation sequences.
        This function must be called before using this EvalTree.

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

        circuit_list = [tuple(opstr) for opstr in simplified_circuit_list.keys()]
        self.simplified_circuit_elabels = list(simplified_circuit_list.values())
        self.simplified_circuit_nEls = list(map(len, self.simplified_circuit_elabels))
        self.num_final_els = sum([len(v) for v in self.simplified_circuit_elabels])
        #self._compute_finalStringToEls() #depends on simplified_circuit_spamTuples
        #UNNEEDED? self.recompute_spamtuple_indices(bLocal=True)  # bLocal shouldn't matter here

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

        # REMOVE #Evaluate the operation sequences "alphabetically" - not needed now, but may
        # REMOVE # be useful later for prefixing?
        # REMOVE sorted_strs = sorted(list(enumerate(circuit_list)), key=lambda x: x[1])
        # REMOVE self.eval_order = [iStr for (iStr, circuit) in sorted_strs]

        self.eval_order = list(range(self.num_final_strs))

        #Storage for polynomial expressions for probabilities and
        # their derivatives

        # TODO REMOVE unused caches
        #self.raw_polys = {}
        #self.all_p_polys = {}
        #self.p_polys = {}
        #self.dp_polys = {}
        #self.hp_polys = {}

        # cache of the high-magnitude terms (actually their represenations), which
        # together with the per-circuit threshold given in `percircuit_p_polys`,
        # defines a set of paths to use in probability computations.
        self.highmag_termrep_cache = {}
        self.circuitsetup_cache = {}
        self.percircuit_p_polys = {}  # keys = circuits, values = (threshold, compact_polys)
        
        self.merged_compact_polys = None
        self.merged_maxsopm_compact_polys = None

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

        subTreeSetList = []
        if numSubTrees is not None:

            subTreeSize = len(self) // numSubTrees
            for i in range(numSubTrees):
                end = (i + 1) * subTreeSize if (i < numSubTrees - 1) else len(self)
                subTreeSetList.append(set(range(i * subTreeSize, end)))

        else:  # maxSubTreeSize is not None
            k = 0
            while k < len(self):
                end = min(k + maxSubTreeSize, len(self))
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
            subTree = TermEvalTree()
            subTree.myFinalToParentFinalMap = sliceIntoParentsFinalArray
            subTree.num_final_strs = numFinal
            subTree[:] = [None] * len(parentIndices)
            # REMOVE subTree.all_p_polys = {}
            # REMOVE subTree.p_polys = {}
            # REMOVE subTree.dp_polys = {}
            # REMOVE subTree.hp_polys = {}
            subTree.highmag_termrep_cache = {}
            subTree.circuitsetup_cache = {}
            subTree.percircuit_p_polys = {}
            subTree.merged_compact_polys = None

            for ik in fullEvalOrder:  # includes any initial indices
                k = parentIndices[ik]  # original tree index
                circuit = self[k]  # original tree data
                subTree.eval_order.append(ik)
                assert(subTree[ik] is None)
                subTree[ik] = circuit

            subTree.parentIndexMap = parentIndices  # parent index of each subtree index
            subTree.simplified_circuit_elabels = [self.simplified_circuit_elabels[kk]
                                                     for kk in _slct.indices(subTree.myFinalToParentFinalMap)]
            subTree.simplified_circuit_nEls = list(map(len, self.simplified_circuit_elabels))
            #subTree._compute_finalStringToEls() #depends on simplified_circuit_spamTuples

            final_el_startstops = []; i = 0
            for elabels in parentTree.simplified_circuit_elabels:
                final_el_startstops.append((i, i + len(elabels)))
                i += len(elabels)
            subTree.myFinalElsToParentFinalElsMap = _np.concatenate(
                [_np.arange(*final_el_startstops[kk])
                 for kk in _slct.indices(subTree.myFinalToParentFinalMap)])
            #Note: myFinalToParentFinalMap maps only between *final* elements
            #   (which are what is held in simplified_circuit_spamTuples)

            subTree.num_final_els = sum([len(v) for v in subTree.simplified_circuit_elabels])
            #NEEDED? subTree.recompute_spamtuple_indices(bLocal=False)

            circuits = subTree.generate_circuit_list(permute=False)
            subTree.opLabels = self._get_opLabels({c: elbls for c,elbls in zip(circuits,subTree.simplified_circuit_elabels)})

            return subTree

        old_indices_in_new_order = self._finish_split(elIndicesDict, subTreeSetList,
                                                      permute_parent_element, create_subtree)

        self.simplified_circuit_elabels, updated_elIndices = \
            self._permute_simplified_circuit_Xs(self.simplified_circuit_elabels, elIndicesDict, old_indices_in_new_order)
        self.simplified_circuit_nEls = list(map(len, self.simplified_circuit_elabels))

        printer.log("EvalTree.split done second pass in %.0fs" %
                    (_time.time() - tm)); tm = _time.time()
        return updated_elIndices

    def cache_size(self):
        """
        Returns the size of the persistent "cache" of partial results
        used during the computation of all the strings in this tree.
        """
        return 0

    def copy(self):
        """ Create a copy of this evaluation tree. """
        cpy = self._copyBase(TermEvalTree(self[:]))
        cpy.opLabels = self.opLabels[:]
        cpy.simplified_circuit_elabels = _copy.deepcopy(self.simplified_circuit_elabels)
        return cpy

    def num_circuit_sopm_failures_using_current_paths(self, calc, pathmagnitude_gap, return_gaps=False):  # TODO REMOVE restrict_to args? HERE
        """ TODO: docstring """
        num_failed = 0  # number of circuits which fail to achieve the target sopm
        failed_circuits = []
        per_circuit_gaps = []
        #db_tested = 0;  #DEBUG, REMOVE
        
        for i in self.get_evaluation_order():  # uses *linear* evaluation order so we know final indices are sequential
            circuit = self[i]
            if circuit not in self.percircuit_p_polys:
                continue  # if circuit not a "current" circuit, so don't count as a failure
            current_threshold, _ = self.percircuit_p_polys[circuit]
            #db_tested += 1 #REMOVE

            rholabel = circuit[0]
            opstr = circuit[1:]
            elabels = self.simplified_circuit_elabels[i]

            #DEBUG TODO REMOVE
            #print("NUM CIRCUIT FAILURES for: ",circuit)
            #print(" - threshold = ",current_threshold)
            #print(" - repcache = ",id(self.highmag_termrep_cache))
            #print(" - opcache = ",id(calc.sos.opcache))
            #print(" - rholabel = ",rholabel)
            #print(" - elabels = ",elabels)

            gaps, DEBUG1,DEBUG2 = calc.circuit_pathmagnitude_gap(rholabel, elabels, opstr, self.highmag_termrep_cache,
                                                  calc.sos.opcache, current_threshold)
            num_failed += 1 if _np.count_nonzero(gaps > pathmagnitude_gap) > 0 else 0  #just count # failed *circuits*
            per_circuit_gaps.append( max(gaps) )
            if _np.count_nonzero(gaps > pathmagnitude_gap) > 0:
                failed_circuits.append(circuit)

        if return_gaps:
            return num_failed, failed_circuits, per_circuit_gaps
        else:
            return num_failed, failed_circuits

    def get_sopm_gaps_using_current_paths(self, calc):  # TODO REMOVE restrict_to args? HERE
        gaps = []

        #DEBUG - TODO REMOVE - testing merged_maxsopm_compact_polys values
        #nEls = self.num_final_elements()
        #polys = self.merged_maxsopm_compact_polys
        #maxsopms = _bulk_eval_compact_polys_complex(
        #    polys[0], polys[1], _np.abs(calc.paramvec), (nEls,))  # shape (nElements,) -- could make this a *fill*
        #debug_i = 0
        #debug_maxsopm_list = []
        #debug_mags_list = []
        #debug_threshold_list = []
        #debug_evalindex_list = []

        for i in self.get_evaluation_order():  # uses *linear* evaluation order so we know final indices are sequential
            circuit = self[i]
            current_threshold, _ = self.percircuit_p_polys[circuit] # must have selected a set of paths for this to be populated!

            rholabel = circuit[0]
            opstr = circuit[1:]
            elabels = self.simplified_circuit_elabels[i]

            circuit_gaps, debug_maxsopm, debug_mags = calc.circuit_pathmagnitude_gap(rholabel, elabels, opstr, self.highmag_termrep_cache,
                                                                                     calc.sos.opcache, current_threshold)

            #DEBUG TODO REMOVE
            #if i == 38:
            #    import bpdb; bpdb.set_trace()
            #    print(current_threshold, circuit_gaps, debug_maxsopm, debug_mags)
            
            #FUTURE: maybe use maxsopms within calc.circuit_pathmagnitude_gap?
            #assert(_np.allclose(maxsopms[debug_i:debug_i+len(debug_mags)], debug_mags))
            #debug_i += len(debug_mags)

            #For DEBUGGING TODO REMOVE (when fn had a debug arg)
            #if debug:
            #    #gaps.extend(list(debug_mags))  #DEBUG
            #    gaps.extend(list(debug_maxsopm))  #DEBUG
            #else:
            #debug_maxsopm_list.extend(list(debug_maxsopm))
            #debug_mags_list.extend(list(debug_mags))
            #debug_threshold_list.extend([current_threshold]*len(debug_mags))
            #debug_evalindex_list.extend([i]*len(debug_mags))
            
            gaps.extend(list(circuit_gaps))

        #DEBUG TODO REMOVE
        #import os, pickle
        #if not os.path.exists("debug_gaps.pkl"):
        #    pickle.dump( (gaps, debug_maxsopm_list, debug_mags_list, debug_threshold_list, debug_evalindex_list), open("debug_gaps.pkl","wb"))
        #    print("WROTE DEBUG FILE debug_gaps.pkl")
        #else:
        #    gaps1, debug_maxsopm_list1, debug_mags_list1, debug_threshold_list1, debug_evalindex_list1 = pickle.load(open("debug_gaps.pkl","rb"))
        #    print("CHECKING for cases where gap1 < gap2...")
        #    for i, (gap1, gap2) in enumerate(zip(gaps1, gaps)):
        #        if gap2 > gap1:
        #            print("GAP2 greater: ",i, gap1, gap2)
        #    import bpdb; bpdb.set_trace()

        assert(len(gaps) == self.num_final_elements())
        return _np.array(gaps, 'd')

    def get_sopm_gaps_jacobian_using_current_paths(self, calc):
        gaps = []

        nEls = self.num_final_elements()
        polys = self.merged_maxsopm_compact_polys
        #from .polynomial import bulk_load_compact_polys  # DEBUG!!!
        #poly_objects = bulk_load_compact_polys(polys[0], polys[1], False, calc.Np) # DEBUG!!!
        dpolys = _compact_deriv(polys[0], polys[1], _np.arange(calc.Np))
        #dpoly_objects = bulk_load_compact_polys(dpolys[0], dpolys[1], False, calc.Np) # DEBUG!!!
        d_achieved_mags = _bulk_eval_compact_polys_complex(
            dpolys[0], dpolys[1], _np.abs(calc.paramvec), (nEls,calc.Np))
        assert(_np.linalg.norm(_np.imag(d_achieved_mags)) < 1e-8)
        d_achieved_mags = d_achieved_mags.real
        d_achieved_mags[:, (calc.paramvec < 0)] *= -1
        #import bpdb; bpdb.set_trace()

        d_max_sopms = _np.empty( (nEls,calc.Np), 'd')
        k = 0  # current element position for loop below

        opcache = calc.sos.opcache
        for iCircuit in self.get_evaluation_order():  # uses *linear* evaluation order so we know final indices are sequential
            circuit = self[iCircuit]

            rholabel = circuit[0]
            opstr = circuit[1:]
            elabels = self.simplified_circuit_elabels[iCircuit]

            #Get MAX-SOPM for circuit outcomes and thereby the target SOPM (via MAX - gap)
            # Here we take d(MAX) (above merged_maxsopm_compact_polys give d(gap)).  Since each
            # MAX-SOPM value is a product of max term magnitudes, to get deriv we use the chain rule:
            partial_ops = [ opcache[rholabel] if rholabel in opcache else calc.sos.get_prep(rholabel) ]
            for glbl in opstr:
                partial_ops.append( opcache[glbl] if glbl in opcache else calc.sos.get_operation(glbl) )
            Eops = [ (opcache[elbl] if elbl in opcache else calc.sos.get_effect(elbl)) for elbl in elabels ]
            partial_op_maxmag_values = [ op.get_total_term_magnitude() for op in partial_ops ]
            Eop_maxmag_values = [ Eop.get_total_term_magnitude() for Eop in Eops ]
            maxmag_partial_product = _np.product(partial_op_maxmag_values)
            maxmag_products = [ maxmag_partial_product * Eop_val for Eop_val in Eop_maxmag_values ]
            
            deriv = _np.zeros( (len(elabels), calc.Np), 'd')
            for i in range(len(partial_ops)):  # replace i-th element of product with deriv
                dop_local = partial_ops[i].get_total_term_magnitude_deriv()
                dop_global = _np.zeros(calc.Np, 'd')
                dop_global[partial_ops[i].gpindices] = dop_local
                dop_global /= partial_op_maxmag_values[i]
                
                for j in range(len(elabels)):
                    deriv[j,:] += dop_global * maxmag_products[j]

            for j in range(len(elabels)): # replace final element with appropriate derivative
                dop_local = Eops[j].get_total_term_magnitude_deriv()
                dop_global = _np.zeros(calc.Np, 'd')
                dop_global[Eops[j].gpindices] = dop_local
                dop_global /= Eop_maxmag_values[j]
                deriv[j,:] += dop_global * maxmag_products[j]

            d_max_sopms[k:k+len(elabels),:] = deriv
            k += len(elabels)

        dgaps = d_max_sopms - d_achieved_mags
        return dgaps


    #def num_circuit_sopm_failures_after_adapting_paths(self, calc, comm, memLimit, pathmagnitude_gap,
    #                                                   min_term_mag, max_paths, exit_after_first_failure=True):
    
    def find_minimal_paths_set(self, calc, comm, memLimit, exit_after_this_many_failures=1):
        """TODO: docstring: returns caches but only when the # failures <= exit_after_this_many_failures """

        tot_npaths = 0
        tot_target_sopm = 0
        tot_achieved_sopm = 0
        
        #We're only testing how many failures there are, don't update the "locked in" persistent
        # set of paths given by self.percircuit_p_polys and self.highmag_termrep_cache - just use a
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
                                                      memLimit,
                                                      None) # guess?
            thresholds[circuit] = threshold
            
            if achieved_sopm < target_sopm:
                num_failed += 1
                failed_circuits.append( (i,circuit) )  #(circuit,npaths, threshold, target_sopm, achieved_sopm))
                if exit_after_this_many_failures > 0 and num_failed == exit_after_this_many_failures:
                    return None, None, None, num_failed #, failed_circuits
                
            tot_npaths += npaths
            tot_target_sopm += target_sopm
            tot_achieved_sopm += achieved_sopm


        #if comm is None or comm.Get_rank() == 0:
        rankStr = "Rank%d: " % comm.Get_rank() if comm is not None else ""
        nC = self.num_final_strings()
        print("%sPruned path-integral: kept %d paths w/magnitude %.4g (target=%.4g, #circuits=%d, #failed=%d)" %
              (rankStr, tot_npaths, tot_achieved_sopm, tot_target_sopm, nC, num_failed))
        print("%s  (avg per circuit paths=%d, magnitude=%.4g, target=%.4g)" %
              (rankStr, tot_npaths // nC, tot_achieved_sopm / nC, tot_target_sopm / nC))

        return thresholds, repcache, circuitsetup_cache, num_failed #, failed_circuits

    #def cache_p_pruned_polys(self, calc, comm, memLimit, pathmagnitude_gap,
    #                         min_term_mag, max_paths, adapt_paths):

    def select_paths_set(self, calc, thresholds, highmag_termrep_cache, circuitsetup_cache, comm, memLimit):
        """ TODO: docstring  - selects *and* computes polys for the given "path set" defined by the arguments."""

        #TODO: update this outdated docstring
        # We're finding and "locking in" a set of paths to use in subsequent evaluations.  This
        # means we're going to re-compute the high-magnitude terms for each operation (in
        # self.highmag_termrep_cache) and re-compute the thresholds (in self.percircuit_p_polys)
        # for each circuit (using the computed high-magnitude terms).  This all occurs for
        # the particular current value of the parameter vector (found via calc.to_vector());
        # these values determine what is a "high-magnitude" term and the path magnitudes that are
        # summed to get the overall sum-of-path-magnitudes for a given circuit outcome.

        self.percircuit_p_polys = {}
        self.highmag_termrep_cache = highmag_termrep_cache
        self.circuitsetup_cache = circuitsetup_cache
        repcache = self.highmag_termrep_cache
        circuitsetup_cache = self.circuitsetup_cache

        ##DEBUG - TODO REMOVE -- for debugging a particular case where were getting unexpected
        ## failures and wanted to rule out parts of select_path_set as the cause
        #for i in self.get_evaluation_order():  # uses *linear* evaluation order so we know final indices are sequential
        #    circuit = self[i]
        #    threshold = thresholds[circuit]
        #    self.percircuit_p_polys[circuit] = (threshold, None)
        #assert(self.num_circuit_sopm_failures_using_current_paths(calc, calc.pathmagnitude_gap)[0] == 0), "STOP1"
        #self.percircuit_p_polys = {}

        all_compact_polys = []  # holds one compact polynomial per final *element*
        all_maxsopm_compact_polys = []
        num_failed = 0  # number of circuits which fail to achieve the target sopm

        for i in self.get_evaluation_order():  # uses *linear* evaluation order so we know final indices are sequential
            circuit = self[i]
            #print("Computing pruned poly %d" % i)

            #TODO REMOVE
            # if circuit in self.percircuit_p_polys:
            #     current_threshold, compact_polys = self.percircuit_p_polys[circuit]
            # else:
            #     current_threshold, compact_polys = None, None

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
                                                       memLimit)

            raw_maxsopm_polyreps = calc.prs_as_pruned_polyreps(threshold,
                                                               rholabel,
                                                               elabels,
                                                               opstr,
                                                               repcache,
                                                               calc.sos.opcache,
                                                               circuitsetup_cache,
                                                               comm,
                                                               memLimit,
                                                               mode="max-sopm")

            compact_polys = [polyrep.compact_complex() for polyrep in raw_polyreps]
            self.percircuit_p_polys[circuit] = (threshold, compact_polys)
            all_compact_polys.extend(compact_polys)  # ok b/c *linear* evaluation order

            compact_maxsopm_polys = [polyrep.compact_complex() for polyrep in raw_maxsopm_polyreps]
            #print("DB: coeff lengths = ",[len(polyrep.int_coeffs) for polyrep in raw_maxsopm_polyreps],
            #      "vs", [len(polyrep.int_coeffs) for polyrep in raw_polyreps])
            #assert(False), "STOP"
            all_maxsopm_compact_polys.extend(compact_maxsopm_polys)
                
        tapes = all_compact_polys  # each "compact polynomials" is a (vtape, ctape) 2-tupe
        vtape = _np.concatenate([t[0] for t in tapes])  # concat all the vtapes
        ctape = _np.concatenate([t[1] for t in tapes])  # concat all teh ctapes
        self.merged_compact_polys = (vtape, ctape)  # Note: ctape should always be complex here

        tapes = all_maxsopm_compact_polys  # each "compact polynomials" is a (vtape, ctape) 2-tupe
        vtape = _np.concatenate([t[0] for t in tapes])  # concat all the vtapes
        ctape = _np.concatenate([t[1] for t in tapes])  # concat all teh ctapes
        self.merged_maxsopm_compact_polys = (vtape, ctape)  # Note: ctape should always be complex here

        return
        

    def cache_p_polys(self, calc, comm):
        """
        Get the compact-form polynomials that evaluate to the probabilities
        corresponding to all this tree's operation sequences sandwiched between
        `rholabel` and each of the `elabels`.  The result is cached to speed
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

    #UNUSED - could perhaps use these to cache derivative polys in the future so TermForwardSimulator
    # doesn't need to always compute them, but this would typically be needed for many-qubit cases
    # when there wouldn't be adequate storage to hold the cached polys... so probably TODO REMOVE this.
    #def get_dp_polys(self, calc, wrtSlice, comm):
    #    """
    #    Similar to :method:`get_p_polys` except returns the compact-form
    #    polynomials that evaluate to the Jacobian of the probabilities
    #    with respect to the parameters given by `wrtSlice`.  The result is
    #    cached to speed up subsequent calls.
    #
    #    Parameters
    #    ----------
    #    calc : TermForwardSimulator
    #        A calculator object for computing the raw polynomials (if necessary)
    #
    #    rholabel : Label
    #        The (simplified) state preparation label.
    #
    #    elabels : list
    #        A list of (simplified) POVM effect labels.
    #
    #    wrtSlice : slice
    #        The parameter slice to differentiate with respect to.
    #
    #    comm : mpi4py.MPI.Comm
    #        When not None, an MPI communicator for distributing the computation
    #        across multiple processors.
    #
    #    Returns
    #    -------
    #    list
    #        A list of `len(elabels)` tuples.  Each tuple is a `(vtape,ctape)`
    #        2-tuple containing the concatenated compact-form tapes of all N*K
    #        polynomials for that (rholabel,elabel) pair, where N is the number
    #        of operation sequences in this tree and K is the number of parameters
    #        we've differentiated with respect to (~`len(wrtSlice)`).
    #    """
    #    slcTup = (wrtSlice.start, wrtSlice.stop, wrtSlice.step) \
    #        if (wrtSlice is not None) else (None, None, None)
    #    slcInds = _slct.indices(wrtSlice if (wrtSlice is not None) else slice(0, calc.Np))
    #    slcInds = _np.ascontiguousarray(slcInds, _np.int64)  # for Cython arg mapping
    #
    #    #Check if everything is computed already
    #    if all([((rholabel, elabel, slcTup) in self.dp_polys) for elabel in elabels]):
    #        return [self.dp_polys[(rholabel, elabel, slcTup)] for elabel in elabels]
    #
    #    #print("*** getDP POLYS ***"); t0= _time.time()
    #
    #    #Otherwise compute poly
    #    ret = []
    #    compact_polys = self.get_p_polys(calc, rholabel, elabels, comm)
    #    for i, elabel in enumerate(elabels):
    #        if (rholabel, elabel, slcTup) not in self.dp_polys:
    #            vtape, ctape = _compact_deriv(compact_polys[i][0], compact_polys[i][1], slcInds)
    #            self.dp_polys[(rholabel, elabel, slcTup)] = (vtape, ctape)
    #        ret.append(self.dp_polys[(rholabel, elabel, slcTup)])
    #
    #    #OLD - using raw polys
    #    #polys = self.get_raw_polys(calc, rholabel, elabels, comm)
    #    #for i,elabel in enumerate(elabels):
    #    #    if (rholabel,elabel,slcTup) not in self.dp_polys:
    #    #        tapes = [ p.deriv(k).compact() for p in polys[i] for k in slcInds ]
    #    #        vtape = _np.concatenate( [ t[0] for t in tapes ] )
    #    #        ctape = _np.concatenate( [ t[1] for t in tapes ] )
    #    #        self.dp_polys[ (rholabel,elabel,slcTup) ] = (vtape, ctape)
    #    #    ret.append( self.dp_polys[ (rholabel,elabel,slcTup) ] )
    #
    #    #print("*** DONE DP POLYS in %.1fs ***" % (_time.time()-t0))
    #    return ret
    #
    #def get_hp_polys(self, calc, rholabel, elabels, wrtSlice1, wrtSlice2, comm):
    #    """
    #    Similar to :method:`get_p_polys` except returns the compact-form
    #    polynomials that evaluate to the Hessian of the probabilities
    #    with respect to the parameters given by `wrtSlice1` and `wrtSlice2`.
    #    The result is cached to speed up subsequent calls.
    #
    #    Parameters
    #    ----------
    #    calc : TermForwardSimulator
    #        A calculator object for computing the raw polynomials (if necessary)
    #
    #    rholabel : Label
    #        The (simplified) state preparation label.
    #
    #    elabels : list
    #        A list of (simplified) POVM effect labels.
    #
    #    wrtSlice1, wrtSlice2 : slice
    #        The parameter slices to differentiate with respect to.
    #
    #    comm : mpi4py.MPI.Comm
    #        When not None, an MPI communicator for distributing the computation
    #        across multiple processors.
    #
    #    Returns
    #    -------
    #    list
    #        A list of `len(elabels)` tuples.  Each tuple is a `(vtape,ctape)`
    #        2-tuple containing the concatenated compact-form tapes of all N*K1*K2
    #        polynomials for that (rholabel,elabel) pair, where N is the number
    #        of operation sequences in this tree and K1,K2 are the number of parameters
    #        we've differentiated with respect to.
    #    """
    #    slcTup1 = (wrtSlice1.start, wrtSlice1.stop, wrtSlice1.step) \
    #        if (wrtSlice1 is not None) else (None, None, None)
    #    slcTup2 = (wrtSlice2.start, wrtSlice2.stop, wrtSlice2.step) \
    #        if (wrtSlice2 is not None) else (None, None, None)
    #    slcInds1 = _slct.indices(wrtSlice1 if (wrtSlice1 is not None) else slice(0, calc.Np))
    #    slcInds2 = _slct.indices(wrtSlice2 if (wrtSlice2 is not None) else slice(0, calc.Np))
    #
    #    #Check if everything is computed already
    #    if all([((rholabel, elabel, slcTup1, slcTup2) in self.hp_polys) for elabel in elabels]):
    #        return [self.hp_polys[(rholabel, elabel, slcTup1, slcTup2)] for elabel in elabels]
    #
    #    #Otherwise compute poly -- FUTURE: do this faster w/
    #    # some self.prs_as_polys(rholabel, elabels, circuit, ...) function
    #    #TODO: add use of caches & compact polys here -- this fn is OUTDATED
    #    ret = []
    #    for elabel in elabels:
    #        if (rholabel, elabel, slcTup1, slcTup2) not in self.hp_polys:
    #            polys = [calc.pr_as_poly((rholabel, elabel), opstr, comm)
    #                     for opstr in self.generate_circuit_list(permute=False)]
    #            dpolys = [p.deriv(k) for p in polys for k in slcInds2]
    #            tapes = [p.deriv(k).compact() for p in dpolys for k in slcInds1]
    #            vtape = _np.concatenate([t[0] for t in tapes])
    #            ctape = _np.concatenate([t[1] for t in tapes])
    #            self.hp_polys[(rholabel, elabel, slcTup1, slcTup2)] = (vtape, ctape)
    #        ret.append(self.hp_polys[(rholabel, elabel, slcTup1, slcTup2)])
    #    return ret
