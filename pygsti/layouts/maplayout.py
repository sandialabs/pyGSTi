"""
Defines the MapCOPALayout class.
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
import importlib as _importlib
import numpy as _np

from pygsti.layouts.distlayout import DistributableCOPALayout as _DistributableCOPALayout
from pygsti.layouts.distlayout import _DistributableAtom
from pygsti.layouts.prefixtable import PrefixTable as _PrefixTable, PrefixTableJacobian as _PrefixTableJacobian
from pygsti.circuits.circuitlist import CircuitList as _CircuitList
from pygsti.tools import listtools as _lt


class _MapCOPALayoutAtom(_DistributableAtom):
    """
    The atom ("atomic unit") for dividing up the element dimension in a :class:`MapCOPALayout`.

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

    max_cache_size : int
        The maximum allowed cache size, given as number of quantum states.
    """

    def __init__(self, unique_complete_circuits, ds_circuits, group, model,
                 dataset, max_cache_size, 
                 circuit_param_dependencies=None, param_circuit_dependencies=None, 
                 expanded_complete_circuit_cache = None):
        
        expanded_circuit_info_by_unique = dict()
        expanded_circuit_set = dict() # only use SeparatePOVMCircuit keys as ordered set

        if expanded_complete_circuit_cache is None:
            expanded_complete_circuit_cache = dict()

        for i in group:
            d = expanded_complete_circuit_cache.get(unique_complete_circuits[i], None)
            if d is None:
                unique_observed_outcomes = None if (dataset is None) else dataset[ds_circuits[i]].unique_outcomes
                d = model.expand_instruments_and_separate_povm(unique_complete_circuits[i], unique_observed_outcomes)
            expanded_circuit_info_by_unique[i] = d  # a dict of SeparatePOVMCircuits => tuples of outcome labels
            expanded_circuit_set.update(d)            

        expanded_circuits = list(expanded_circuit_set.keys())
        
        self.table = _PrefixTable(expanded_circuits, max_cache_size)

        #only Build the Jacobian prefix table if we are using the generic evotype.
        if model.sim.calclib is _importlib.import_module("pygsti.forwardsims.mapforwardsim_calc_generic"):
            #create a list for storing the model parameter dependencies of expanded circuits
            expanded_param_circuit_depend = [{} for _ in range(len(param_circuit_dependencies))]
            for i in group:
                for param_idx in circuit_param_dependencies[i]:
                    expanded_param_circuit_depend[param_idx].update(expanded_circuit_info_by_unique[i])
            expanded_param_circuit_depend = [list(param_circuit_depend_dict.keys()) for  param_circuit_depend_dict in expanded_param_circuit_depend]

            self.jac_table = _PrefixTableJacobian(expanded_circuits, max_cache_size, expanded_param_circuit_depend)

        #Create circuit element <=> integer index lookups for speed
        all_rholabels = set()
        all_oplabels = set()
        all_elabels = set()
        all_povmlabels = set()
        for expanded_circuit_infos in expanded_circuit_info_by_unique.values():
            for sep_povm_c in expanded_circuit_infos:
                all_rholabels.add(sep_povm_c.circuit_without_povm[0])
                all_oplabels.update(sep_povm_c.circuit_without_povm[1:])
                all_elabels.update(sep_povm_c.full_effect_labels)
                all_povmlabels.add(sep_povm_c.povm_label)

        self.rho_labels = sorted(all_rholabels)
        self.op_labels = sorted(all_oplabels)
        self.povm_labels = sorted(all_povmlabels)
        self.full_effect_labels = all_elabels
        self.elabel_lookup = {elbl: i for i, elbl in enumerate(self.full_effect_labels)}

        #Lookup arrays for faster replib computation.
        table_offset = 0
        self.orig_indices_by_expcircuit = {}  # record original circuit index so dataset row can be retrieved
        self.unique_indices_by_expcircuit = {}
        self.elbl_indices_by_expcircuit = {}
        self.elindices_by_expcircuit = {}
        self.outcomes_by_expcircuit = {}
        self.povm_and_elbls_by_expcircuit = {}

        elindex_outcome_tuples = {unique_i: list() for unique_i in range(len(unique_complete_circuits))}

        #Assign element indices, "global" indices starting at `offset`
        local_offset = 0
        for unique_i, expanded_circuit_infos in expanded_circuit_info_by_unique.items():
            for table_relindex, (sep_povm_c, outcomes) in enumerate(expanded_circuit_infos.items()):
                i = table_offset + table_relindex  # index of expanded circuit (table item)
                elindices = list(range(local_offset, local_offset + len(outcomes)))
                self.elbl_indices_by_expcircuit[i] = [self.elabel_lookup[lbl] for lbl in sep_povm_c.full_effect_labels]
                self.povm_and_elbls_by_expcircuit[i] = (sep_povm_c.povm_label,) + tuple(sep_povm_c.effect_labels)
                self.elindices_by_expcircuit[i] = elindices  # *local* indices (0 is 1st element computed by this atom)
                self.outcomes_by_expcircuit[i] = outcomes
                self.unique_indices_by_expcircuit[i] = unique_i
                local_offset += len(outcomes)

                # fill in running dict of per-circuit *local* element indices and outcomes:
                elindex_outcome_tuples[unique_i].extend([(eli, out) for eli, out in zip(elindices, outcomes)])
            table_offset += len(expanded_circuit_infos)
        self.elindex_outcome_tuples = elindex_outcome_tuples
        element_slice = None  # *global* (of parent layout) element-index slice - set by parent

        super().__init__(element_slice, local_offset)

    def _update_indices(self, old_unique_is_by_new_unique_is):
        """
        Updates any internal indices held as a result of the unique-circuit indices of the layout changing.

        This function is called during layout construction to alert the atom that the layout
        being created will only hold a subset of the `unique_complete_circuits` provided to
        to the atom's `__init__` method.  Thus, if the atom keeps indices to unique circuits
        within the layout, it should update these indices accordingly.

        Parameters
        ----------
        old_unique_is_by_new_unique_is : list
            The indices within the `unique_complete_circuits` given to `__init__` that index the
            unique circuits of the created layout - thus, these  that will become (in order) all of
            the unique circuits of the created layout.

        Returns
        -------
        None
        """
        new_unique_indices_by_expcircuit = {}
        old_to_new = {old_i: new_i for new_i, old_i in enumerate(old_unique_is_by_new_unique_is)}
        for i, unique_i in self.unique_indices_by_expcircuit.items():
            new_unique_indices_by_expcircuit[i] = old_to_new[unique_i]
        self.unique_indices_by_expcircuit = new_unique_indices_by_expcircuit

    @property
    def cache_size(self):
        """The cache size of this atom."""
        return self.table.cache_size


class MapCOPALayout(_DistributableCOPALayout):
    """
    A circuit outcome probability array (COPA) layout for circuit simulation by state vector maps.

    A simple distributed layout that divides a list of circuits among available
    processors and optionally supports caching "prefix" states that result from
    common prefixes found in the circuits.

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

    max_cache_size : int, optional
        The maximum number of "prefix" quantum states that may be cached for performance.
        If `None`, there is no limit to how large the cache may be.

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
    
    circuit_partition_cost_functions : tuple of str, optional (default ('size', 'propagations'))
        A tuple of strings denoting cost function to use in each of the two stages of the algorithm
        for determining the partitions of the complete circuit set amongst atoms.
        Allowed options are 'size', which corresponds to balancing the number of circuits, 
        and 'propagations', which corresponds to balancing the number of state propagations.

    verbosity : int or VerbosityPrinter
        Determines how much output to send to stdout.  0 means no output, higher
        integers mean more output.

    layout_creation_circuit_cache : dict, optional (default None)
        An optional dictionary containing pre-computed circuit structures/modifications which
        can be used to reduce the overhead of repeated circuit operations during layout creation.
    
    load_balancing_parameters : tuple of floats, optional (default (1.2, .1))
        A tuple of floats used as load balancing parameters when splitting a layout across atoms,
        as in the multi-processor setting when using MPI. These parameters correspond to the `imbalance_threshold`
        and `minimum_improvement_threshold` parameters described in the method `find_splitting_new`
        of the `PrefixTable` class.
    """

    def __init__(self, circuits, model, dataset=None, max_cache_size=None,
                 num_sub_tables=None, num_table_processors=1, num_param_dimension_processors=(),
                 param_dimensions=(), param_dimension_blk_sizes=(), resource_alloc=None, 
                 circuit_partition_cost_functions=('size', 'propagations'), verbosity=0, 
                 layout_creation_circuit_cache=None, load_balancing_parameters = (1.2, .1)):

        unique_circuits, to_unique = self._compute_unique_circuits(circuits)
        aliases = circuits.op_label_aliases if isinstance(circuits, _CircuitList) else None
        ds_circuits = _lt.apply_aliases_to_circuits(unique_circuits, aliases)

        #extract subcaches from layout_creation_circuit_cache:
        if layout_creation_circuit_cache is None:
            layout_creation_circuit_cache = dict()
        self.completed_circuit_cache = layout_creation_circuit_cache.get('completed_circuits', None)
        self.split_circuit_cache = layout_creation_circuit_cache.get('split_circuits', None)
        self.expanded_and_separated_circuits_cache = layout_creation_circuit_cache.get('expanded_and_separated_circuits', None)
        
        if self.completed_circuit_cache is None:
            unique_complete_circuits = model.complete_circuits(unique_circuits)
            split_circuits = model.split_circuits(unique_complete_circuits, split_prep=False)
        else:
            unique_complete_circuits = []
            for c in unique_circuits:
                comp_ckt = self.completed_circuit_cache.get(c, None)
                if comp_ckt is not None:
                    unique_complete_circuits.append(comp_ckt)
                else:
                    unique_complete_circuits.append(model.complete_circuit(c))
            split_circuits = []
            for c, c_complete in zip(unique_circuits,unique_complete_circuits):
                split_ckt = self.split_circuit_cache.get(c, None)
                if split_ckt is not None:
                    split_circuits.append(split_ckt)
                else:
                    split_circuits.append(model.split_circuit(c_complete, split_prep=False))

        #construct a map for the parameter dependence for each of the unique_complete_circuits.
        #returns a dictionary who's keys are the unique completed circuits, and whose
        #values are lists of model parameters upon which that circuit depends.
        if model.sim.calclib is _importlib.import_module("pygsti.forwardsims.mapforwardsim_calc_generic") and model.param_interposer is None:
            circ_param_map, param_circ_map = model.circuit_parameter_dependence(unique_complete_circuits, return_param_circ_map=True)
            uniq_comp_circs_param_depend = list(circ_param_map.values())
            uniq_comp_param_circs_depend = param_circ_map
        else : 
            circ_param_map = None
            param_circ_map = None
            uniq_comp_circs_param_depend = None
            uniq_comp_param_circs_depend = None
        #construct list of unique POVM-less circuits.
        unique_povmless_circuits = [ckt_tup[1] for ckt_tup in split_circuits]

        max_sub_table_size = None  # was an argument but never used; remove in future
        if (num_sub_tables is not None and num_sub_tables > 1) or max_sub_table_size is not None:
            circuit_table = _PrefixTable(unique_povmless_circuits, max_cache_size)
            self.complete_circuit_table = circuit_table
            groups = circuit_table.find_splitting_new(max_sub_table_size, num_sub_tables, verbosity=verbosity,
                                                      initial_cost_metric=circuit_partition_cost_functions[0],
                                                      rebalancing_cost_metric=circuit_partition_cost_functions[1],
                                                      imbalance_threshold = load_balancing_parameters[0],
                                                      minimum_improvement_threshold = load_balancing_parameters[1])
            #groups = circuit_table.find_splitting(max_sub_table_size, num_sub_tables, verbosity=verbosity)
        else:
            groups = [list(range(len(unique_complete_circuits)))]
        

        def _create_atom(group):
            return _MapCOPALayoutAtom(unique_complete_circuits, ds_circuits, group,
                                      model, dataset, max_cache_size,
                                      circuit_param_dependencies= uniq_comp_circs_param_depend,
                                      param_circuit_dependencies= uniq_comp_param_circs_depend,
                                      expanded_complete_circuit_cache=self.expanded_and_separated_circuits_cache)

        super().__init__(circuits, unique_circuits, to_unique, unique_complete_circuits,
                         _create_atom, groups, num_table_processors,
                         num_param_dimension_processors, param_dimensions,
                         param_dimension_blk_sizes, resource_alloc, verbosity)

        # For time dependent calcs:
        # connect unique -> orig indices of final layout now that base class has created it
        # (don't do this before because the .circuits of this local layout may not be *all* the circuits,
        # or in the same order - this is only true in the *global* layout.
        unique_to_orig = {unique_i: orig_i for orig_i, unique_i in self._to_unique.items()}  # unique => orig. indices
        for atom in self.atoms:
            for expanded_circuit_i, unique_i in atom.unique_indices_by_expcircuit.items():
                atom.orig_indices_by_expcircuit[expanded_circuit_i] = unique_to_orig[unique_i]