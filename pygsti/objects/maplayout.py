"""
Defines the MapCOPALayout class.
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
from ..tools import listtools as _lt
from .bulkcircuitlist import BulkCircuitList as _BulkCircuitList
from .distlayout import _DistributableAtom
from .distlayout import DistributableCOPALayout as _DistributableCOPALayout
from .prefixtable import PrefixTable as _PrefixTable

import time as _time  # DEBUG TIMERS


class _MapCOPALayoutAtom(_DistributableAtom):
    """
    Object that acts as "atomic unit" of instructions-for-applying a COPA strategy.
    """

    def __init__(self, unique_complete_circuits, ds_circuits, unique_to_orig, group, model_shlp,
                 dataset, offset, elindex_outcome_tuples, max_cache_size):

        expanded_circuit_outcomes_by_unique = _collections.OrderedDict()
        expanded_circuit_outcomes = _collections.OrderedDict()
        for i in group:
            observed_outcomes = None if (dataset is None) else dataset[ds_circuits[i]].outcomes
            d = unique_complete_circuits[i].expand_instruments_and_separate_povm(model_shlp, observed_outcomes)
            expanded_circuit_outcomes_by_unique[i] = d
            expanded_circuit_outcomes.update(d)

        expanded_circuits = list(expanded_circuit_outcomes.keys())
        self.table = _PrefixTable(expanded_circuits, max_cache_size)

        #Create circuit element <=> integer index lookups for speed
        all_rholabels = set()
        all_oplabels = set()
        all_elabels = set()
        for expanded_circuit_outcomes in expanded_circuit_outcomes_by_unique.values():
            for sep_povm_c in expanded_circuit_outcomes:
                if sep_povm_c.effect_labels == [None]:  # special case -- needed (for bulk_product?)
                    all_oplabels.update(sep_povm_c.circuit_without_povm[:])
                else:
                    all_rholabels.add(sep_povm_c.circuit_without_povm[0])
                    all_oplabels.update(sep_povm_c.circuit_without_povm[1:])
                    all_elabels.update(sep_povm_c.full_effect_labels)

        self.rho_labels = sorted(all_rholabels)
        self.op_labels = sorted(all_oplabels)
        self.full_effect_labels = all_elabels
        self.elabel_lookup = {elbl: i for i, elbl in enumerate(self.full_effect_labels)}

        #Lookup arrays for faster replib computation.
        table_offset = 0
        self.orig_indices_by_expcircuit = {}  # record original circuit index so dataset row can be retrieved
        self.elbl_indices_by_expcircuit = {}
        self.elindices_by_expcircuit = {}
        self.outcomes_by_expcircuit = {}

        #Assign element indices, starting at `offset`
        initial_offset = offset
        for unique_i, expanded_circuit_outcomes in expanded_circuit_outcomes_by_unique.items():
            for table_relindex, (sep_povm_c, outcomes) in enumerate(expanded_circuit_outcomes.items()):
                i = table_offset + table_relindex  # index of expanded circuit (table item)
                elindices = list(range(offset, offset + len(outcomes)))
                self.elbl_indices_by_expcircuit[i] = [self.elabel_lookup[lbl] for lbl in sep_povm_c.full_effect_labels]
                self.elindices_by_expcircuit[i] = elindices
                self.outcomes_by_expcircuit[i] = outcomes
                self.orig_indices_by_expcircuit[i] = unique_to_orig[unique_i]
                offset += len(outcomes)

                # fill in running dict of per-circuit element indices and outcomes:
                elindex_outcome_tuples[unique_i].extend(list(zip(elindices, outcomes)))
            table_offset += len(expanded_circuit_outcomes)

        super().__init__(slice(initial_offset, offset), offset - initial_offset)

    @property
    def cache_size(self):
        return self.table.cache_size


class MapCOPALayout(_DistributableCOPALayout):
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

    num_strategy_subcomms : int, optional
        The number of processor groups (communicators) to divide the "atomic" portions
        of this strategy (a circuit probability array layout) among when calling `distribute`.
        By default, the communicator is not divided.  This default behavior is fine for cases
        when derivatives are being taken, as multiple processors are used to process differentiations
        with respect to different variables.  If no derivaties are needed, however, this should be
        set to (at least) the number of processors.

    max_cache_size : int, optional
        Maximum cache size, used for holding common circuit prefixes.
    """

    def __init__(self, circuits, model_shlp, dataset=None, max_cache_size=None,
                 max_sub_table_size=None, num_sub_tables=None, additional_dimensions=(), verbosity=0):

        unique_circuits, to_unique = self._compute_unique_circuits(circuits)
        aliases = circuits.op_label_aliases if isinstance(circuits, _BulkCircuitList) else None
        ds_circuits = _lt.apply_aliases_to_circuits(unique_circuits, aliases)
        unique_complete_circuits = [model_shlp.complete_circuit(c) for c in unique_circuits]

        circuit_table = _PrefixTable(unique_complete_circuits, max_cache_size)
        groups = circuit_table.find_splitting(max_sub_table_size, num_sub_tables, verbosity)

        atoms = []
        elindex_outcome_tuples = {unique_i: list() for unique_i in range(len(unique_circuits))}
        to_orig = {unique_i: orig_i for orig_i, unique_i in to_unique.items()}  # unique => original indices

        offset = 0
        for group in groups:
            atoms.append(_MapCOPALayoutAtom(unique_complete_circuits, ds_circuits, to_orig, group,
                                            model_shlp, dataset, offset, elindex_outcome_tuples, max_cache_size))
            offset += atoms[-1].num_elements

        super().__init__(circuits, unique_circuits, to_unique, elindex_outcome_tuples, unique_complete_circuits,
                         atoms, additional_dimensions)

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

    #def cache_size(self):
    #    """
    #    Returns the size of the persistent "cache" of partial results.
    #
    #    This cache is used during the computation of all the strings in
    #    this tree.
    #
    #    Returns
    #    -------
    #    int
    #    """
    #    return self.cachesize

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
