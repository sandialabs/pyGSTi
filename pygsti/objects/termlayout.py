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
from .bulkcircuitlist import BulkCircuitList as _BulkCircuitList
from .distlayout import _DistributableAtom
from .distlayout import DistributableCOPALayout as _DistributableCOPALayout


class _TermCOPALayoutAtom(_DistributableAtom):
    """
    Object that acts as "atomic unit" of instructions-for-applying a COPA strategy.
    """

    def __init__(self, unique_complete_circuits, ds_circuits, group, model_shlp, dataset,
                 offset, elindex_outcome_tuples):

        expanded_circuit_outcomes_by_orig = _collections.OrderedDict()
        expanded_circuit_outcomes = _collections.OrderedDict()
        for i in group:
            observed_outcomes = None if (dataset is None) else dataset[ds_circuits[i]].outcomes
            d = unique_complete_circuits[i].expand_instruments_and_separate_povm(model_shlp, observed_outcomes)
            expanded_circuit_outcomes_by_orig[i] = d
            expanded_circuit_outcomes.update(d)

        self.expanded_circuits = list(expanded_circuit_outcomes.keys())  # a list of SeparatePOVMCircuits

        #Create circuit element <=> integer index lookups for speed
        all_rholabels = set()
        all_oplabels = set()
        all_elabels = set()
        for expanded_circuit_outcomes in expanded_circuit_outcomes_by_orig.values():
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

        #Assign element indices, starting at `offset`
        initial_offset = offset
        for orig_i, expanded_circuit_outcomes in expanded_circuit_outcomes_by_orig.items():
            for table_relindex, (sep_povm_c, outcomes) in enumerate(expanded_circuit_outcomes.items()):
                i = table_offset + table_relindex  # index of expanded circuit (table item)
                elindices = list(range(offset, offset + len(outcomes)))
                self.elbl_indices_by_expcircuit[i] = [self.elabel_lookup[lbl] for lbl in sep_povm_c.full_effect_labels]
                self.elindices_by_expcircuit[i] = elindices
                offset += len(outcomes)

                # fill in running dict of per-circuit element indices and outcomes:
                elindex_outcome_tuples[orig_i].extend(list(zip(elindices, outcomes)))
            table_offset += len(expanded_circuit_outcomes)

        # cache of the high-magnitude terms (actually their represenations), which
        # together with the per-circuit threshold given in `percircuit_p_polys`,
        # defines a set of paths to use in probability computations.
        self.pathset = None
        self.percircuit_p_polys = {}  # keys = circuits, values = (threshold, compact_polys)

        self.merged_compact_polys = None
        self.merged_achievedsopm_compact_polys = None

        super().__init__(slice(initial_offset, offset), offset - initial_offset)

    @property
    def cache_size(self):
        return 0


class TermCOPALayout(_DistributableCOPALayout):
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
    """

    def __init__(self, circuits, model_shlp, dataset=None, max_sub_table_size=None, num_sub_tables=None,
                 additional_dimensions=(), verbosity=0):

        unique_circuits, to_unique = self._compute_unique_circuits(circuits)
        aliases = circuits.op_label_aliases if isinstance(circuits, _BulkCircuitList) else None
        ds_circuits = _lt.apply_aliases_to_circuits(unique_circuits, aliases)
        unique_complete_circuits = [model_shlp.complete_circuit(c) for c in unique_circuits]

        #Create evenly divided groups of indices of unique_complete_circuits
        assert(max_sub_table_size is None), "No support for size-limited subtables yet!"
        ngroups = num_sub_tables
        groups = [set(sub_array) for sub_array in _np.array_split(range(len(unique_complete_circuits)), ngroups)]

        atoms = []
        elindex_outcome_tuples = {orig_i: list() for orig_i in range(len(unique_circuits))}

        offset = 0
        for group in groups:
            atoms.append(_TermCOPALayoutAtom(unique_complete_circuits, ds_circuits, group, model_shlp,
                                             dataset, offset, elindex_outcome_tuples))
            offset += atoms[-1].num_elements

        super().__init__(circuits, unique_circuits, to_unique, elindex_outcome_tuples, unique_complete_circuits,
                         atoms, additional_dimensions)

    def pathset(self, comm=None):
        """
        This layout's current path set.

        Parameters
        ----------
        layout : TermCOPALayout
            The layout specifiying the quantities (circuit outcome probabilities) to be
            computed, and related information.

        comm : mpi4py.MPI.Comm, optional
            The comm used for distributing this layout among multiple processors.

        Returns
        -------
        TermPathSet or None
        """
        from .termforwardsim import TermPathSet as _TermPathSet
        myAtomIndices, atomOwners, mySubComm = self.distribute(comm)
        local_atom_pathsets = [self.atoms[i].pathset for i in myAtomIndices]
        if None in local_atom_pathsets:  # None signifies that we're not using path-sets (not "pruned" term-fwdsim mode)
            return None  # None behaves a bit like NaN - if there's a None, just return None.
        else:
            return _TermPathSet(local_atom_pathsets, comm)
