"""
Defines the MatrixCOPALayout class.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from ..tools import slicetools as _slct
from ..tools import listtools as _lt
from .bulkcircuitlist import BulkCircuitList as _BulkCircuitList
from .distlayout import _DistributableAtom
from .distlayout import DistributableCOPALayout as _DistributableCOPALayout
from .evaltree import EvalTree as _EvalTree

import numpy as _np
import collections as _collections


class _MatrixCOPALayoutAtom(_DistributableAtom):
    """
    Object that acts as "atomic unit" of instructions-for-applying a COPA strategy.
    """

    def __init__(self, unique_complete_circuits, unique_nospam_circuits, circuits_by_unique_nospam_circuits,
                 ds_circuits, group, model_shlp, dataset, offset, elindex_outcome_tuples):

        expanded_nospam_circuit_outcomes = _collections.OrderedDict()
        for i in group:
            nospam_c = unique_nospam_circuits[i]
            for orig_i in circuits_by_unique_nospam_circuits[nospam_c]:  # orig circuits that add SPAM to nospam_c
                observed_outcomes = None if (dataset is None) else dataset[ds_circuits[orig_i]].outcomes
                expc_outcomes = unique_complete_circuits[orig_i].expand_instruments_and_separate_povm(
                    model_shlp, observed_outcomes)

                for sep_povm_c, outcomes in expc_outcomes.items():
                    prep_lbl = sep_povm_c.circuit_without_povm[0]
                    exp_nospam_c = sep_povm_c.circuit_without_povm[1:]  # sep_povm_c *always* has prep lbl
                    spam_tuples = [(prep_lbl, elabel) for elabel in sep_povm_c.full_effect_labels]
                    outcome_by_spamtuple = {st: (outcome, orig_i) for st, outcome in zip(spam_tuples, outcomes)}

                    if exp_nospam_c not in expanded_nospam_circuit_outcomes:
                        expanded_nospam_circuit_outcomes[exp_nospam_c] = outcome_by_spamtuple
                    else:
                        expanded_nospam_circuit_outcomes[exp_nospam_c].update(outcome_by_spamtuple)

        expanded_nospam_circuits = {i: cir for i, cir in enumerate(expanded_nospam_circuit_outcomes.keys())}
        self.tree = _EvalTree.create(expanded_nospam_circuits)
        self._num_nonscratch_tree_items = len(expanded_nospam_circuits)  # put this in EvalTree?

        # self.tree's elements give instructions for evaluating ("caching") no-spam quantities (e.g. products).
        # Now we assign final element indices to the circuit outcomes corresponding to a given no-spam ("tree")
        # quantity plus a spam-tuple. We order the final indices so that all the outcomes corresponding to a
        # given spam-tuple are contiguous.

        tree_indices_by_spamtuple = _collections.defaultdict(list)  # "tree" indices index expanded_nospam_circuits
        for i, c in expanded_nospam_circuits.items():
            for spam_tuple in expanded_nospam_circuit_outcomes[c].keys():
                tree_indices_by_spamtuple[spam_tuple].append(i)

        #Assign element indices, starting at `offset`
        # now that we know how many of each spamtuple there are, assign final element indices.
        initial_offset = offset
        self.indices_by_spamtuple = {}  # values are (element_indices, tree_indices) tuples.
        for spam_tuple, tree_indices in tree_indices_by_spamtuple.items():
            self.indices_by_spamtuple[spam_tuple] = (slice(offset, offset + len(tree_indices)),
                                                     _slct.list_to_slice(tree_indices, array_ok=True))
            offset += len(tree_indices)
            #TODO: allow tree_indices to be None or a slice?

        element_slice = slice(initial_offset, offset)
        num_elements = offset - initial_offset

        for spam_tuple, (element_indices, tree_indices) in self.indices_by_spamtuple.items():
            for elindex, tree_index in zip(_slct.indices(element_indices), _slct.to_array(tree_indices)):
                outcome_by_spamtuple = expanded_nospam_circuit_outcomes[expanded_nospam_circuits[tree_index]]
                outcome, orig_i = outcome_by_spamtuple[spam_tuple]
                elindex_outcome_tuples[orig_i].append((elindex, outcome))

        super().__init__(element_slice, num_elements)

    def nonscratch_cache_view(self, a, axis=None):
        """
        Create a view of array `a` restricting it to only the *final* results computed by this tree.

        This need not be the entire array because there could be intermediate results
        (e.g. "scratch space") that are excluded.

        Parameters
        ----------
        a : ndarray
            An array of results computed using this EvalTree,
            such that the `axis`-th dimension equals the full
            length of the tree.  The other dimensions of `a` are
            unrestricted.

        axis : int, optional
            Specified the axis along which the selection of the
            final elements is performed. If None, than this
            selection if performed on flattened `a`.

        Returns
        -------
        ndarray
            Of the same shape as `a`, except for along the
            specified axis, whose dimension has been reduced
            to filter out the intermediate (non-final) results.
        """
        if axis is None:
            return a[0:self._num_nonscratch_tree_items]
        else:
            sl = [slice(None)] * a.ndim
            sl[axis] = slice(0, self._num_nonscratch_tree_items)
            ret = a[tuple(sl)]
            assert(ret.base is a or ret.base is a.base)  # check that what is returned is a view
            assert(ret.size == 0 or _np.may_share_memory(ret, a))
            return ret

    @property
    def cache_size(self):
        return len(self.tree)


class MatrixCOPALayout(_DistributableCOPALayout):
    """
    TODO: update docstring

    An Evaluation Tree that structures circuits for efficient multiplication of process matrices.

    MatrixEvalTree instances create and store the decomposition of a list of circuits into
    a sequence of 2-term products of smaller strings.  Ideally, this sequence would
    prescribe the way to obtain the entire list of circuits, starting with just the single
    gates, using the fewest number of multiplications, but this optimality is not
    guaranteed.

    Parameters
    ----------
    items : list, optional
        Initial items.  This argument should only be used internally
        in the course of serialization.

    num_strategy_subcomms : int, optional
        The number of processor groups (communicators) to divide the "atomic" portions
        of this strategy (a circuit probability array layout) among when calling `distribute`.
        By default, the communicator is not divided.  This default behavior is fine for cases
        when derivatives are being taken, as multiple processors are used to process differentiations
        with respect to different variables.  If no derivaties are needed, however, this should be
        set to (at least) the number of processors.
    """

    def __init__(self, circuits, model_shlp, dataset=None, max_sub_tree_size=None,
                 num_sub_trees=None, additional_dimensions=(), verbosity=0):

        #OUTDATED: TODO - revise this:
        # 1. pre-process => get complete circuits => spam-tuples list for each no-spam circuit (no expanding yet)
        # 2. decide how to divide no-spam circuits into groups corresponding to sub-strategies
        #    - create tree of no-spam circuits (may contain instruments, etc, just not SPAM)
        #    - heuristically find groups of circuits that meet criteria
        # 3. separately create a tree of no-spam expanded circuits originating from each group => self.atoms
        # 4. assign "cache" and element indices so that a) all elements of a tree are contiguous
        #    and b) elements with the same spam-tuple are continguous.
        # 5. initialize base class with given per-original-circuit element indices.

        unique_circuits, to_unique = self._compute_unique_circuits(circuits)
        aliases = circuits.op_label_aliases if isinstance(circuits, _BulkCircuitList) else None
        ds_circuits = _lt.apply_aliases_to_circuits(unique_circuits, aliases)
        unique_complete_circuits = [model_shlp.complete_circuit(c) for c in unique_circuits]

        circuits_by_unique_nospam_circuits = _collections.OrderedDict()
        for i, c in enumerate(unique_complete_circuits):
            _, nospam_c, _ = model_shlp.split_circuit(c)
            if nospam_c in circuits_by_unique_nospam_circuits:
                circuits_by_unique_nospam_circuits[nospam_c].append(i)
            else:
                circuits_by_unique_nospam_circuits[nospam_c] = [i]
        unique_nospam_circuits = list(circuits_by_unique_nospam_circuits.keys())

        circuit_tree = _EvalTree.create(unique_nospam_circuits)
        groups = circuit_tree.find_splitting(len(unique_nospam_circuits),
                                             max_sub_tree_size, num_sub_trees, verbosity)  # a list of tuples/sets?
        # (elements of `groups` contain indices into `unique_nospam_circuits`)

        atoms = []
        elindex_outcome_tuples = {orig_i: list() for orig_i in range(len(unique_circuits))}

        offset = 0
        for group in groups:
            atoms.append(_MatrixCOPALayoutAtom(unique_complete_circuits, unique_nospam_circuits,
                                               circuits_by_unique_nospam_circuits, ds_circuits, group,
                                               model_shlp, dataset, offset, elindex_outcome_tuples))
            offset += atoms[-1].num_elements

        super().__init__(circuits, unique_circuits, to_unique, elindex_outcome_tuples, unique_complete_circuits,
                         atoms, additional_dimensions)

    def copy(self):
        """
        Create a copy of this layout.

        Returns
        -------
        MatrixCOPALayout
        """
        raise NotImplementedError("TODO! update this!")
        #newTree = self._copy_base(MatrixEvalTree(self[:]))
        #newTree.opLabels = self.opLabels[:]
        #newTree.init_indices = self.init_indices[:]
        #newTree.simplified_circuit_spamTuples = self.simplified_circuit_spamTuples[:]
        ##newTree.finalStringToElsMap = self.finalStringToElsMap[:]
        #newTree.spamtuple_indices = self.spamtuple_indices.copy()
        #return newTree

    #def cache_size(self):
    #    """
    #    Returns the size of the persistent "cache".
    #
    #    This cache holds partial results used during the computation of all
    #    the strings in this tree.
    #
    #    Returns
    #    -------
    #    int
    #    """
    #    return len(self.eval_tree)

    #def get_min_tree_size(self):
    #    """
    #    Returns the minimum sub tree size required to compute each of the tree entries individually.
    #
    #    This minimum size is the smallest "max_sub_tree_size" that can be passed to
    #    split(), as any smaller value will result in at least one entry being
    #    uncomputable.
    #
    #    Returns
    #    -------
    #    int
    #    """
    #    singleItemTreeSetList = self._create_single_item_trees()
    #    return max(list(map(len, singleItemTreeSetList)))

    #PRIVATE
    #def get_analysis_plot_infos(self):
    #    """
    #    Returns debug plot information.
    #
    #    This is useful for assessing the quality of a tree. This
    #    function is not guaranteed to work.
    #
    #    Returns
    #    -------
    #    dict
    #    """
    #
    #    analysis = {}
    #    firstIndxSeen = list(range(len(self)))
    #    lastIndxSeen = list(range(len(self)))
    #    subTreeSize = [-1] * len(self)
    #
    #    xs = []; ys = []
    #    for i in range(len(self)):
    #        subTree = []
    #        self._walk_subtree(i, subTree)
    #        subTreeSize[i] = len(subTree)
    #        ys.extend([i] * len(subTree) + [None])
    #        xs.extend(list(sorted(subTree) + [None]))
    #
    #        for k, t in enumerate(self):
    #            iLeft, iRight = t
    #            if i in (iLeft, iRight):
    #                lastIndxSeen[i] = k
    #
    #    analysis['SubtreeUsagePlot'] = {'xs': xs, 'ys': ys, 'title': "Indices used by the subtree rooted at each index",
    #                                    'xlabel': "Indices used", 'ylabel': 'Subtree root index'}
    #    analysis['SubtreeSizePlot'] = {'xs': list(range(len(self))),
    #                                   'ys': subTreeSize,
    #                                   'title': "Size of subtree rooted at each index",
    #                                   'xlabel': "Subtree root index",
    #                                   'ylabel': 'Subtree size'}
    #
    #    xs = []; ys = []
    #    for i, rng in enumerate(zip(firstIndxSeen, lastIndxSeen)):
    #        ys.extend([i, i, None])
    #        xs.extend([rng[0], rng[1], None])
    #    analysis['IndexUsageIntervalsPlot'] = {'xs': xs, 'ys': ys, 'title': "Usage Intervals of each index",
    #                                           'xlabel': "Index Interval", 'ylabel': 'Index'}
    #
    #    return analysis

    #def recompute_spamtuple_indices(self, local=False):
    #    """
    #    Recompute this tree's `.spamtuple_indices` array.
    #
    #    Parameters
    #    ----------
    #    local : bool, optional
    #        If True, then the indices computed will index
    #        this tree's final array (even if it's a subtree).
    #        If False (the default), then a subtree's indices
    #        will index the *parent* tree's final array.
    #
    #    Returns
    #    -------
    #    None
    #    """
    #    self.spamtuple_indices = _compute_spamtuple_indices(
    #        self.simplified_circuit_spamTuples,
    #        None if local else self.myFinalElsToParentFinalElsMap)
    #
    #def _get_full_eval_order(self):
    #    """Includes init_indices in matrix-based evaltree case... HACK """
    #    return self.init_indices + self.eval_order
    #
    #def _update_eval_order_helpers(self, index_permutation):
    #    """Update anything pertaining to the "full" evaluation order - e.g. init_inidces in matrix-based case (HACK)"""
    #    self.init_indices = [index_permutation[iCur] for iCur in self.init_indices]
    #
    #def _update_element_indices(self, new_indices_in_old_order, old_indices_in_new_order, element_indices_dict):
    #    """
    #    Update any additional members because this tree's elements are being permuted.
    #    In addition, return an updated version of `element_indices_dict` a dict whose keys are
    #    the tree's (unpermuted) circuit indices and whose values are the final element indices for
    #    each circuit.
    #    """
    #    self.simplified_circuit_spamTuples, updated_elIndices = \
    #        self._permute_simplified_circuit_xs(self.simplified_circuit_spamTuples,
    #                                            element_indices_dict, old_indices_in_new_order)
    #    self.simplified_circuit_nEls = list(map(len, self.simplified_circuit_spamTuples))
    #    self.recompute_spamtuple_indices(local=True)  # local shouldn't matter here - just for clarity
    #
    #    return updated_elIndices


#def _compute_spamtuple_indices(simplified_circuit_spam_tuples,
#                               subtree_final_els_to_parent_final_els_map=None):
#    """
#    Returns a dictionary whose keys are the distinct spamTuples
#    found in `simplified_circuit_spam_tuples` and whose values are
#    (finalIndices, finalTreeSlice) tuples where:
#
#    finalIndices = the "element" indices in any final filled quantities
#                   which combines both spam and gate-sequence indices.
#                   If this tree is a subtree, then these final indices
#                   refer to the *parent's* final elements if
#                   `subtree_final_els_to_parent_final_els_map` is given, otherwise
#                   they refer to the subtree's final indices (usually desired).
#    treeIndices = indices into the tree's final circuit list giving
#                  all of the (raw) operation sequences which need to be computed
#                  for the current spamTuple (this list has the SAME length
#                  as finalIndices).
#    """
#    spamtuple_indices = _collections.OrderedDict(); el_off = 0
#    for i, spamTuples in enumerate(  # i == final operation sequence index
#            simplified_circuit_spam_tuples):
#        for j, spamTuple in enumerate(spamTuples, start=el_off):  # j == final element index
#            if spamTuple not in spamtuple_indices:
#                spamtuple_indices[spamTuple] = ([], [])
#            f = subtree_final_els_to_parent_final_els_map[j] \
#                if (subtree_final_els_to_parent_final_els_map is not None) else j  # parent's final
#            spamtuple_indices[spamTuple][0].append(f)
#            spamtuple_indices[spamTuple][1].append(i)
#        el_off += len(spamTuples)
#
#    def to_slice(x, max_len=None):
#        s = _slct.list_to_slice(x, array_ok=True, require_contiguous=False)
#        if max_len is not None and isinstance(s, slice) and (s.start, s.stop, s.step) == (0, max_len, None):
#            return slice(None, None)  # check for entire range
#        else:
#            return s
#
#    nRawSequences = len(simplified_circuit_spam_tuples)
#    nElements = el_off if (subtree_final_els_to_parent_final_els_map is None) \
#        else None  # (we don't know how many els the parent has!)
#    return _collections.OrderedDict(
#        [(spamTuple, (to_slice(f_inds, nElements), to_slice(g_inds, nRawSequences)))
#         for spamTuple, (f_inds, g_inds) in spamtuple_indices.items()])
