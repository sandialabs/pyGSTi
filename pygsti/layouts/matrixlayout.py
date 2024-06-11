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

import collections as _collections

import numpy as _np

from pygsti.layouts.distlayout import DistributableCOPALayout as _DistributableCOPALayout
from pygsti.layouts.distlayout import _DistributableAtom
from pygsti.layouts.evaltree import EvalTree as _EvalTree
from pygsti.circuits.circuitlist import CircuitList as _CircuitList
from pygsti.tools import listtools as _lt
from pygsti.tools import slicetools as _slct


class _MatrixCOPALayoutAtom(_DistributableAtom):
    """
    The atom ("atomic unit") for dividing up the element dimension in a :class:`MatrixCOPALayout`.

    Parameters
    ----------
    unique_complete_circuits : list
        A list that contains *all* the "complete" circuits for the parent layout.  This
        atom only owns a subset of these, as given by `group` below.

    unique_nospam_circuits : list
        A list that contains the unique circuits within `unique_complete_circuits` once
        their state preparations and measurements are removed.  A subset of these circuits
        (see `group` below) are what fundamentally define the circuit outcomes that this atom
        includes: it includes *all* the circuit outcomes of those circuits.

    circuits_by_unique_nospam_circuits : dict
       A dictionary with keys equal to the elements of `unique_nospam_circuits` and values
       that are lists of indices into `unique_complete_circuits`.  Thus, this dictionary
       maps each distinct circuit-without-SPAM circuit to the list of complete circuits
       within `unique_complete_circuits` that correspond to it.

    ds_circuits : list
        A list of circuits parallel to `unique_complete_circuits` of these circuits
        as they should be accessed from `dataset`.  This applies any aliases and
        removes implied SPAM elements relative to `unique_complete_circuits`.

    group : set
        The set of indices into `unique_nospam_circuits` that define the circuit
        outcomes owned by this atom.

    helpful_scratch : set
        A set of indices into `unique_nospam_circuits` that specify circuits that
        aren't owned by this atom but are helpful in building up an efficient evaluation
        tree.

    model : Model
        The model being used to construct this layout.  Used for expanding instruments
        within the circuits.

    dataset : DataSet
        The dataset, used to include only observed circuit outcomes in this atom
        and therefore the parent layout.
    """

    def __init__(self, unique_complete_circuits, unique_nospam_circuits, circuits_by_unique_nospam_circuits,
                 ds_circuits, group, helpful_scratch, model, dataset):

        #Note: group gives unique_nospam_circuits indices, which circuits_by_unique_nospam_circuits
        # turns into "unique complete circuit" indices, which the layout via it's to_unique can map
        # to original circuit indices.
        def add_expanded_circuits(indices, add_to_this_dict):
            _expanded_nospam_circuit_outcomes = add_to_this_dict
            for i in indices:
                nospam_c = unique_nospam_circuits[i]
                for unique_i in circuits_by_unique_nospam_circuits[nospam_c]:  # "unique" circuits: add SPAM to nospam_c
                    observed_outcomes = None if (dataset is None) else dataset[ds_circuits[unique_i]].unique_outcomes
                    expc_outcomes = unique_complete_circuits[unique_i].expand_instruments_and_separate_povm(
                        model, observed_outcomes)
                    #Note: unique_complete_circuits may have duplicates (they're only unique *pre*-completion)

                    for sep_povm_c, outcomes in expc_outcomes.items():  # for each expanded cir from unique_i-th circuit
                        prep_lbl = sep_povm_c.circuit_without_povm[0]
                        exp_nospam_c = sep_povm_c.circuit_without_povm[1:]  # sep_povm_c *always* has prep lbl
                        spam_tuples = [(prep_lbl, elabel) for elabel in sep_povm_c.full_effect_labels]
                        outcome_by_spamtuple = _collections.OrderedDict([(st, outcome)
                                                                         for st, outcome in zip(spam_tuples, outcomes)])

                        #Now add these outcomes to `expanded_nospam_circuit_outcomes` - note that multiple "unique_i"'s
                        # may exist for the same expanded & without-spam circuit (exp_nospam_c) and so we need to
                        # keep track of a list of unique_i indices for each circut and spam tuple below.
                        if exp_nospam_c not in _expanded_nospam_circuit_outcomes:
                            _expanded_nospam_circuit_outcomes[exp_nospam_c] = _collections.OrderedDict(
                                [(st, (outcome, [unique_i])) for st, outcome in zip(spam_tuples, outcomes)])
                        else:
                            for st, outcome in outcome_by_spamtuple.items():
                                if st in _expanded_nospam_circuit_outcomes[exp_nospam_c]:
                                    existing_outcome, existing_unique_is = \
                                        _expanded_nospam_circuit_outcomes[exp_nospam_c][st]
                                    assert(existing_outcome == outcome), "Outcome should be same when spam tuples are!"
                                    assert(unique_i not in existing_unique_is)  # SLOW - remove?
                                    existing_unique_is.append(unique_i)
                                else:
                                    _expanded_nospam_circuit_outcomes[exp_nospam_c][st] = (outcome, [unique_i])

        # keys = expanded circuits w/out SPAM layers; values = spamtuple => (outcome, unique_is) dictionary that
        # keeps track of which "unique" circuit indices having each spamtuple / outcome.
        expanded_nospam_circuit_outcomes = _collections.OrderedDict()
        add_expanded_circuits(group, expanded_nospam_circuit_outcomes)
        expanded_nospam_circuits = _collections.OrderedDict(
            [(i, cir) for i, cir in enumerate(expanded_nospam_circuit_outcomes.keys())])

        # add suggested scratch to the "final" elements as far as the tree creation is concerned
        # - this allows these scratch element to help balance the tree.
        expanded_nospam_circuit_outcomes_plus_scratch = expanded_nospam_circuit_outcomes.copy()
        add_expanded_circuits(helpful_scratch, expanded_nospam_circuit_outcomes_plus_scratch)
        expanded_nospam_circuits_plus_scratch = _collections.OrderedDict(
            [(i, cir) for i, cir in enumerate(expanded_nospam_circuit_outcomes_plus_scratch.keys())])

        double_expanded_nospam_circuits_plus_scratch = _collections.OrderedDict()
        for i, cir in expanded_nospam_circuits_plus_scratch.items():
            cir = cir.copy(editable=True)
            cir.expand_subcircuits()  # expand sub-circuits for a more efficient tree
            cir.done_editing()
            double_expanded_nospam_circuits_plus_scratch[i] = cir

        self.tree = _EvalTree.create(double_expanded_nospam_circuits_plus_scratch)
        #print("Atom tree: %d circuits => tree of size %d" % (len(expanded_nospam_circuits), len(self.tree)))

        self._num_nonscratch_tree_items = len(expanded_nospam_circuits)  # put this in EvalTree?

        # self.tree's elements give instructions for evaluating ("caching") no-spam quantities (e.g. products).
        # Now we assign final element indices to the circuit outcomes corresponding to a given no-spam ("tree")
        # quantity plus a spam-tuple. We order the final indices so that all the outcomes corresponding to a
        # given spam-tuple are contiguous.

        tree_indices_by_spamtuple = _collections.OrderedDict()  # "tree" indices index expanded_nospam_circuits
        for i, c in expanded_nospam_circuits.items():
            for spam_tuple in expanded_nospam_circuit_outcomes[c].keys():
                if spam_tuple not in tree_indices_by_spamtuple: tree_indices_by_spamtuple[spam_tuple] = []
                tree_indices_by_spamtuple[spam_tuple].append(i)

        #Assign element indices, starting at `offset`
        # now that we know how many of each spamtuple there are, assign final element indices.
        local_offset = 0
        self.indices_by_spamtuple = _collections.OrderedDict()  # values are (element_indices, tree_indices) tuples.
        for spam_tuple, tree_indices in tree_indices_by_spamtuple.items():
            self.indices_by_spamtuple[spam_tuple] = (slice(local_offset, local_offset + len(tree_indices)),
                                                     _slct.list_to_slice(tree_indices, array_ok=True))
            local_offset += len(tree_indices)
            #TODO: allow tree_indices to be None or a slice?

        element_slice = None  # slice(offset, offset + local_offset)  # *global* (of parent layout) element-index slice
        num_elements = local_offset

        elindex_outcome_tuples = _collections.OrderedDict([
            (unique_i, list()) for unique_i in range(len(unique_complete_circuits))])

        for spam_tuple, (element_indices, tree_indices) in self.indices_by_spamtuple.items():
            for elindex, tree_index in zip(_slct.indices(element_indices), _slct.to_array(tree_indices)):
                outcome_by_spamtuple = expanded_nospam_circuit_outcomes[expanded_nospam_circuits[tree_index]]
                outcome, unique_is = outcome_by_spamtuple[spam_tuple]
                for unique_i in unique_is:
                    elindex_outcome_tuples[unique_i].append((elindex, outcome))  # *local* element indices
        self.elindex_outcome_tuples = elindex_outcome_tuples

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
        """The cache size of this atom."""
        return len(self.tree)


class MatrixCOPALayout(_DistributableCOPALayout):
    """
    A circuit outcome probability array (COPA) layout for circuit simulation by process matrix multiplication.

    A distributed layout that divides a list of circuits into several "evaluation trees"
    that compute subsets of the circuit outcomes by multiplying together process matrices.
    Often these evaluation trees correspond to available processors, but it can be useful
    to divide computations in order to lessen the amount of intermediate memory required.

    MatrixCOPALayout instances create and store the decomposition of a list of circuits into
    a sequence of 2-term products of smaller strings.  Ideally, this sequence would
    prescribe the way to obtain the entire list of circuits, starting with just the single
    gates, using the fewest number of multiplications, but this optimality is not
    guaranteed.

    Parameters
    ----------
    circuits : list
        A list of:class:`Circuit` objects representing the circuits this layout will include.

    model : Model
        The model that will be used to compute circuit outcome probabilities using this layout.
        This model is used to complete and expand the circuits in `circuits`.

    dataset : DataSet, optional
        If not None, restrict the circuit outcomes stored by this layout to only the
        outcomes observed in this data set.

    num_sub_trees : int, optional
        The number of groups ("sub-trees") to divide the circuits into.  This is the
        number of *atoms* for this layout.

    num_tree_processors : int, optional
        The number of atom-processors, i.e. groups of processors that process sub-trees.

    num_param_dimension_processors : tuple, optional
        A 1- or 2-tuple of integers specifying how many parameter-block processors are
        used when dividing the physical processors into a grid.  The first and second
        elements correspond to counts for the first and second parameter dimensions,
        respecively.

    param_dimensions : tuple, optional
        The number of parameters along each parameter dimension.  Can be an
        empty, 1-, or 2-tuple of integers which dictates how many parameter dimensions this
        layout supports.

    param_dimension_blk_sizes : tuple, optional
        The parameter block sizes along each present parameter dimension, so this should
        be the same shape as `param_dimensions`.  A block size of `None` means that there
        should be no division into blocks, and that each block processor computes all of
        its parameter indices at once.

    resource_alloc : ResourceAllocation, optional
        The resources available for computing circuit outcome probabilities.

    verbosity : int or VerbosityPrinter
        Determines how much output to send to stdout.  0 means no output, higher
        integers mean more output.
    """

    def __init__(self, circuits, model, dataset=None, num_sub_trees=None, num_tree_processors=1,
                 num_param_dimension_processors=(), param_dimensions=(),
                 param_dimension_blk_sizes=(), resource_alloc=None, verbosity=0):

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
        aliases = circuits.op_label_aliases if isinstance(circuits, _CircuitList) else None
        ds_circuits = _lt.apply_aliases_to_circuits(unique_circuits, aliases)
        unique_complete_circuits, split_unique_circuits = model.complete_circuits(unique_circuits, return_split=True)
        #Note: "unique" means a unique circuit *before* circuit-completion, so there could be duplicate
        # "unique circuits" after completion, e.g. "rho0Gx" and "Gx" could both complete to "rho0GxMdefault_0".

        circuits_by_unique_nospam_circuits = _collections.OrderedDict()
        for i, (_, nospam_c, _) in enumerate(split_unique_circuits):
            if nospam_c in circuits_by_unique_nospam_circuits:
                circuits_by_unique_nospam_circuits[nospam_c].append(i)
            else:
                circuits_by_unique_nospam_circuits[nospam_c] = [i]
        unique_nospam_circuits = list(circuits_by_unique_nospam_circuits.keys())
        
        # Split circuits into groups that will make good subtrees (all procs do this)
        max_sub_tree_size = None  # removed from being an argument (unused)
        if (num_sub_trees is not None and num_sub_trees > 1) or max_sub_tree_size is not None:
            circuit_tree = _EvalTree.create(unique_nospam_circuits)
            groups, helpful_scratch = circuit_tree.find_splitting(len(unique_nospam_circuits),
                                                                  max_sub_tree_size, num_sub_trees, verbosity - 1)
        else:
            groups = [set(range(len(unique_nospam_circuits)))]
            helpful_scratch = [set()]
        # (elements of `groups` contain indices into `unique_nospam_circuits`)

        def _create_atom(args):
            group, helpful_scratch_group = args
            return _MatrixCOPALayoutAtom(unique_complete_circuits, unique_nospam_circuits,
                                         circuits_by_unique_nospam_circuits, ds_circuits,
                                         group, helpful_scratch_group, model, dataset)

        super().__init__(circuits, unique_circuits, to_unique, unique_complete_circuits,
                         _create_atom, list(zip(groups, helpful_scratch)), num_tree_processors,
                         num_param_dimension_processors, param_dimensions,
                         param_dimension_blk_sizes, resource_alloc, verbosity)
