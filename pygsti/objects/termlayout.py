"""
Defines the TermCOPALayout class.
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
from .circuitlist import CircuitList as _CircuitList
from .distlayout import _DistributableAtom
from .distlayout import DistributableCOPALayout as _DistributableCOPALayout


class _TermCOPALayoutAtom(_DistributableAtom):
    """
    The atom ("atomic unit") for dividing up the element dimension in a :class:`TermCOPALayout`.

    This class noteably holds the current "path-set" used to evaluate circuit probabilites,
    as well as compact representations of the polynomials for evaluating these probabilities.

    Parameters
    ----------
    unique_complete_circuits : list
        A list that contains *all* the "complete" circuits for the parent layout.  This
        atom only owns a subset of these, as given by `group` below.

    ds_circuits : list
        A parallel list of circuits as they should be accessed from `dataset`.
        This applies any aliases and removes implied SPAM elements relative to
        `unique_complete_circuits`.

    group : set
        The set of unique-circuit indices (i.e. indices into `unique_complete_circuits`)
        that this atom owns.

    model : Model
        The model being used to construct this layout.  Used for expanding instruments
        within the circuits.

    dataset : DataSet
        The dataset, used to include only observed circuit outcomes in this atom
        and therefore the parent layout.
    """

    def __init__(self, unique_complete_circuits, ds_circuits, group, model, dataset):

        expanded_circuit_outcomes_by_unique = _collections.OrderedDict()
        expanded_circuit_outcomes = _collections.OrderedDict()
        for i in group:
            observed_outcomes = None if (dataset is None) else dataset[ds_circuits[i]].outcomes
            d = unique_complete_circuits[i].expand_instruments_and_separate_povm(model, observed_outcomes)
            expanded_circuit_outcomes_by_unique[i] = d
            expanded_circuit_outcomes.update(d)

        self.expanded_circuits = list(expanded_circuit_outcomes.keys())  # a list of SeparatePOVMCircuits

        #Create circuit element <=> integer index lookups for speed
        all_rholabels = set()
        all_oplabels = set()
        all_elabels = set()
        for expanded_circuit_outcomes in expanded_circuit_outcomes_by_unique.values():
            for sep_povm_c in expanded_circuit_outcomes:
                all_rholabels.add(sep_povm_c.circuit_without_povm[0])
                all_oplabels.update(sep_povm_c.circuit_without_povm[1:])
                all_elabels.update(sep_povm_c.full_effect_labels)

        self.rho_labels = sorted(all_rholabels)
        self.op_labels = sorted(all_oplabels)
        self.full_effect_labels = all_elabels
        self.elabel_lookup = {elbl: i for i, elbl in enumerate(self.full_effect_labels)}

        #Lookup arrays for faster replib computation.
        table_offset = 0
        self.elbl_indices_by_expcircuit = {}
        self.elindices_by_expcircuit = {}

        elindex_outcome_tuples = _collections.OrderedDict([
            (unique_i, list()) for unique_i in range(len(unique_complete_circuits))])

        #Assign element indices, "global" indices starting at `offset`
        local_offset = 0
        for unique_i, expanded_circuit_outcomes in expanded_circuit_outcomes_by_unique.items():
            for table_relindex, (sep_povm_c, outcomes) in enumerate(expanded_circuit_outcomes.items()):
                i = table_offset + table_relindex  # index of expanded circuit (table item)
                elindices = list(range(local_offset, local_offset + len(outcomes)))
                self.elbl_indices_by_expcircuit[i] = [self.elabel_lookup[lbl] for lbl in sep_povm_c.full_effect_labels]
                self.elindices_by_expcircuit[i] = elindices  # *local* indices (0 is 1st element computed by this atom)
                local_offset += len(outcomes)

                # fill in running dict of per-circuit *local* element indices and outcomes:
                elindex_outcome_tuples[unique_i].extend([(eli, out) for eli, out in zip(elindices, outcomes)])
            table_offset += len(expanded_circuit_outcomes)

        self.elindex_outcome_tuples = elindex_outcome_tuples
        element_slice = None  # *global* (of parent layout) element-index slice - set by parent

        # cache of the high-magnitude terms (actually their represenations), which
        # together with the per-circuit threshold given in `percircuit_p_polys`,
        # defines a set of paths to use in probability computations.
        self.pathset = None
        self.percircuit_p_polys = {}  # keys = circuits, values = (threshold, compact_polys)

        self.merged_compact_polys = None
        self.merged_achievedsopm_compact_polys = None

        super().__init__(element_slice, local_offset)

    @property
    def cache_size(self):
        """The cache size of this atom."""
        return 0


class TermCOPALayout(_DistributableCOPALayout):
    """
    A circuit outcome probability array (COPA) layout for circuit simulation by taylor-term path integration.

    A simple distributed layout that divides a list of circuits among available
    processors.  This layout is designed for Taylor-term based calculations for which
    there is no straightforward caching/performance-enhancement mechanism.  The
    path-integral specific implementation is present in the atoms, which hold current
    path sets and compact polynomials for speeding the evaluation of circuit outcome
    probabilities.

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

    num_sub_tables : int, optional
        The number of groups ("sub-tables") to divide the circuits into.  This is the
        number of *atoms* for this layout.

    num_table_processors : int, optional
        The number of atom-processors, i.e. groups of processors that process sub-tables.

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

    def __init__(self, circuits, model, dataset=None,
                 num_sub_tables=None, num_table_processors=1,
                 num_param_dimension_processors=(), param_dimensions=(), param_dimension_blk_sizes=(),
                 resource_alloc=None, verbosity=0):

        unique_circuits, to_unique = self._compute_unique_circuits(circuits)
        aliases = circuits.op_label_aliases if isinstance(circuits, _CircuitList) else None
        ds_circuits = _lt.apply_aliases_to_circuits(unique_circuits, aliases)
        unique_complete_circuits = [model.complete_circuit(c) for c in unique_circuits]

        #Create evenly divided groups of indices of unique_complete_circuits
        max_sub_table_size = None  # was an argument but never used; remove in future
        assert(max_sub_table_size is None), "No support for size-limited subtables yet!"
        ngroups = num_sub_tables
        groups = [set(sub_array) for sub_array in _np.array_split(range(len(unique_complete_circuits)), ngroups)]

        #atoms = []
        #elindex_outcome_tuples = {unique_i: list() for unique_i in range(len(unique_circuits))}
        #
        #offset = 0
        #for group in groups:
        #    atoms.append(_TermCOPALayoutAtom(unique_complete_circuits, ds_circuits, group, model,
        #                                     dataset, offset, elindex_outcome_tuples))
        #    offset += atoms[-1].num_elements

        def _create_atom(group):
            return _TermCOPALayoutAtom(unique_complete_circuits, ds_circuits, group, model, dataset)

        super().__init__(circuits, unique_circuits, to_unique, unique_complete_circuits,
                         _create_atom, groups, num_table_processors,
                         num_param_dimension_processors, param_dimensions,
                         param_dimension_blk_sizes, resource_alloc, verbosity)

    @property
    def pathset(self):
        """
        This layout's current path set.

        Returns
        -------
        TermPathSet or None
        """
        from .termforwardsim import TermPathSet as _TermPathSet
        local_atom_pathsets = [atom.pathset for atom in self.atoms]
        if None in local_atom_pathsets:  # None signifies that we're not using path-sets (not "pruned" term-fwdsim mode)
            return None  # None behaves a bit like NaN - if there's a None, just return None.
        else:
            return _TermPathSet(local_atom_pathsets, self.resource_alloc().comm)
