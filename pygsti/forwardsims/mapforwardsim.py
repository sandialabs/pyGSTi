"""
Defines the MapForwardSimulator calculator class
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import importlib as _importlib
import warnings as _warnings

import numpy as _np
from numpy import linalg as _nla

from pygsti.forwardsims.distforwardsim import DistributableForwardSimulator as _DistributableForwardSimulator
from pygsti.forwardsims.forwardsim import ForwardSimulator as _ForwardSimulator
from pygsti.forwardsims.forwardsim import _bytes_for_array_types
from pygsti.layouts.maplayout import MapCOPALayout as _MapCOPALayout
from pygsti.baseobjs.profiler import DummyProfiler as _DummyProfiler
from pygsti.baseobjs.resourceallocation import ResourceAllocation as _ResourceAllocation
from pygsti.baseobjs.verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from pygsti.tools import sharedmemtools as _smt
from pygsti.tools import slicetools as _slct
from pygsti.tools.matrixtools import _fas
from pygsti.tools import listtools as _lt
from pygsti.circuits import CircuitList as _CircuitList

_dummy_profiler = _DummyProfiler()

# FUTURE: use enum
_SUPEROP = 0
_UNITARY = 1
CLIFFORD = 2


class SimpleMapForwardSimulator(_ForwardSimulator):
    """
    A forward simulator that uses matrix-vector products to compute circuit outcome probabilities.

    This is "simple" in that it adds a minimal implementation to its :class:`ForwardSimulator`
    base class.  Because of this, it lacks some of the efficiency of a :class:`MapForwardSimulator`
    object, and is mainly useful as a reference implementation and check for other simulators.
    """
    # NOTE: It is currently not a *distributed* forward simulator, but after the addition of
    # the `as_layout` method to distributed atoms, this class could instead derive from
    # DistributableForwardSimulator and (I think) not need any more implementation.
    # If this is done, then MapForwardSimulator wouldn't need to separately subclass DistributableForwardSimulator

    def _compute_circuit_outcome_probabilities(self, array_to_fill, circuit, outcomes, resource_alloc, time=None):
        expanded_circuit_outcomes = self.model.expand_instruments_and_separate_povm(circuit, outcomes)
        outcome_to_index = {outc: i for i, outc in enumerate(outcomes)}
        for spc, spc_outcomes in expanded_circuit_outcomes.items():  # spc is a SeparatePOVMCircuit
            # Note: `spc.circuit_without_povm` *always* begins with a prep label.
            indices = [outcome_to_index[o] for o in spc_outcomes]
            if time is None:  # time-independent state propagation

                rhorep = self.model.circuit_layer_operator(spc.circuit_without_povm[0], 'prep')._rep
                povmrep = self.model.circuit_layer_operator(spc.povm_label, 'povm')._rep
                rhorep = self.calclib.propagate_staterep(rhorep,
                                                         [self.model.circuit_layer_operator(ol, 'op')._rep
                                                          for ol in spc.circuit_without_povm[1:]])
                if povmrep is None:
                    ereps = [self.model.circuit_layer_operator(elabel, 'povm')._rep for elabel in spc.full_effect_labels]
                    array_to_fill[indices] = [erep.probability(rhorep) for erep in ereps]  # outcome probabilities
                else:
                    # using spc.effect_labels ensures returned probabilities are in same order as spc_outcomes
                    array_to_fill[indices] = povmrep.probabilities(rhorep, None, spc.effect_labels)

            else:
                t = time  # Note: time in labels == duration
                rholabel = spc.circuit_without_povm[0]
                op = self.model.circuit_layer_operator(rholabel, 'prep'); op.set_time(t); t += rholabel.time
                state = op._rep
                for ol in spc.circuit_without_povm[1:]:
                    op = self.model.circuit_layer_operator(ol, 'op'); op.set_time(t); t += ol.time
                    state = op._rep.acton(state)
                ps = []
                for elabel in spc.full_effect_labels:
                    op = self.model.circuit_layer_operator(elabel, 'povm'); op.set_time(t)
                    # Note: don't advance time (all effects occur at same time)
                    ps.append(op._rep.probability(state))
                array_to_fill[indices] = ps

    def _set_evotype(self, evotype):
        """ Called when the evotype being used (defined by the parent model) changes.
            `evotype` will be `None` when the current model is None"""
        if evotype is not None:
            try:
                self.calclib = _importlib.import_module("pygsti.forwardsims.mapforwardsim_calc_" + evotype.name)
            except ImportError:
                self.calclib = _importlib.import_module("pygsti.forwardsims.mapforwardsim_calc_generic")
        else:
            self.calclib = None

    def __getstate__(self):
        state = super(SimpleMapForwardSimulator, self).__getstate__()
        if 'calclib' in state: del state['calclib']
        #Note: I don't think we need to implement __setstate__ since the model also needs to be reset,
        # and this is done by the parent model which will cause _set_evotype to be called.
        return state


class MapForwardSimulator(_DistributableForwardSimulator, SimpleMapForwardSimulator):
    """
    Computes circuit outcome probabilities using circuit layer maps that act on a state.

    Interfaces with a model via its `circuit_layer_operator` method and applies the resulting
    operators in order to propagate states and finally compute outcome probabilities.  Derivatives
    are computed using finite-differences, and the prefix tables construbed by :class:`MapCOPALayout`
    layout object are used to avoid duplicating (some) computation.

    Parameters
    ----------
    model : Model, optional
        The parent model of this simulator.  It's fine if this is `None` at first,
        but it will need to be set (by assigning `self.model` before using this simulator.

    max_cache_size : int, optional
        The maximum number of "prefix" quantum states that may be cached for performance
        (within the layout).  If `None`, there is no limit to how large the cache may be.

    num_atoms : int, optional
        The number of atoms (sub-prefix-tables) to use when creating the layout (i.e. when calling
        :meth:`create_layout`).  This determines how many units the element (circuit outcome
        probability) dimension is divided into, and doesn't have to correclate with the number of
        processors.  When multiple processors are used, if `num_atoms` is less than the number of
        processors then `num_atoms` should divide the number of processors evenly, so that
        `num_atoms // num_procs` groups of processors can be used to divide the computation
        over parameter dimensions.

    processor_grid : tuple optional
        Specifies how the total number of processors should be divided into a number of
        atom-processors, 1st-parameter-deriv-processors, and 2nd-parameter-deriv-processors.
        Each level of specification is optional, so this can be a 1-, 2-, or 3- tuple of
        integers (or None).  Multiplying the elements of `processor_grid` together should give
        at most the total number of processors.

    param_blk_sizes : tuple, optional
        The parameter block sizes along the first or first & second parameter dimensions - so
        this can be a 0-, 1- or 2-tuple of integers or `None` values.  A block size of `None`
        means that there should be no division into blocks, and that each block processor
        computes all of its parameter indices at once.
    """

    @classmethod
    def _array_types_for_method(cls, method_name):
        # The array types of *intermediate* or *returned* values within various class methods (for memory estimates)
        if method_name == '_bulk_fill_probs_block': return ('zd',)  # cache of rho-vectors
        if method_name == '_bulk_fill_dprobs_block': return ('zd',)  # cache of rho-vectors
        if method_name == '_bulk_fill_hprobs_block': return ('zd',)  # cache of rho-vectors

        if method_name == 'bulk_fill_timedep_loglpp': return ()
        if method_name == 'bulk_fill_timedep_dloglpp': return ('p',)  # just an additional parameter vector
        if method_name == 'bulk_fill_timedep_chi2': return ()
        if method_name == 'bulk_fill_timedep_dchi2': return ('p',)  # just an additional parameter vector
        return super()._array_types_for_method(method_name)

    def __init__(self, model=None, max_cache_size=None, num_atoms=None, processor_grid=None, param_blk_sizes=None,
                 derivative_eps=1e-7, hessian_eps=1e-5):
        #super().__init__(model, num_atoms, processor_grid, param_blk_sizes)
        _DistributableForwardSimulator.__init__(self, model, num_atoms, processor_grid, param_blk_sizes)
        self._max_cache_size = max_cache_size
        self.derivative_eps = derivative_eps  # for finite difference derivative calculations
        self.hessian_eps = hessian_eps

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'max_cache_size': self._max_cache_size,
                      'derivative_epsilon': self.derivative_eps,
                      'hessian_epsilon': self.hessian_eps,
                      # (don't serialize parent model or processor distribution info)
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        #Note: resets processor-distribution information
        return cls(None, state['max_cache_size'],
                   derivative_eps=state.get('derivative_epsilon', 1e-7),
                   hessian_eps=state.get('hessian_epsilon', 1e-5))

    def copy(self):
        """
        Return a shallow copy of this MapForwardSimulator

        Returns
        -------
        MapForwardSimulator
        """
        return MapForwardSimulator(self.model, self._max_cache_size, self._num_atoms,
                                   self._processor_grid, self._pblk_sizes)

    def create_layout(self, circuits, dataset=None, resource_alloc=None, array_types=('E',),
                      derivative_dimensions=None, verbosity=0, layout_creation_circuit_cache=None,
                      circuit_partition_cost_functions=('size', 'propagations'),
                      load_balancing_parameters=(1.15,.1)):
        """
        Constructs an circuit-outcome-probability-array (COPA) layout for a list of circuits.

        Parameters
        ----------
        circuits : list
            The circuits whose outcome probabilities should be included in the layout.

        dataset : DataSet
            The source of data counts that will be compared to the circuit outcome
            probabilities.  The computed outcome probabilities are limited to those
            with counts present in `dataset`.

        resource_alloc : ResourceAllocation
            A available resources and allocation information.  These factors influence how
            the layout (evaluation strategy) is constructed.

        array_types : tuple, optional
            A tuple of string-valued array types.  See :meth:`ForwardSimulator.create_layout`.

        derivative_dimensions : int or tuple[int], optional
            Optionally, the parameter-space dimension used when taking first
            and second derivatives with respect to the cirucit outcome probabilities.  This must be
            non-None when `array_types` contains `'ep'` or `'epp'` types.
            If a tuple, then must be length 1.

        verbosity : int or VerbosityPrinter
            Determines how much output to send to stdout.  0 means no output, higher
            integers mean more output.
        
        layout_creation_circuit_cache : dict, optional (default None)
            A precomputed dictionary serving as a cache for completed circuits. I.e. circuits 
            with prep labels and POVM labels appended. Along with other useful pre-computed 
            circuit structures used in layout creation.

        circuit_partition_cost_functions : tuple of str, optional (default ('size', 'propagations'))
            A tuple of strings denoting cost function to use in each of the two stages of the algorithm
            for determining the partitions of the complete circuit set amongst atoms.
            Allowed options are 'size', which corresponds to balancing the number of circuits, 
            and 'propagations', which corresponds to balancing the number of state propagations.

        load_balancing_parameters : tuple of floats, optional (default (1.2, .1))
            A tuple of floats used as load balancing parameters when splitting a layout across atoms,
            as in the multi-processor setting when using MPI. These parameters correspond to the `imbalance_threshold`
            and `minimum_improvement_threshold` parameters described in the method `find_splitting_new`
            of the `PrefixTable` class.

        Returns
        -------
        MapCOPALayout
        """
        resource_alloc = _ResourceAllocation.cast(resource_alloc)
        printer = _VerbosityPrinter.create_printer(verbosity, resource_alloc)
        mem_limit = resource_alloc.mem_limit - resource_alloc.allocated_memory \
            if (resource_alloc.mem_limit is not None) else None  # *per-processor* memory limit
        nprocs = resource_alloc.comm_size
        comm = resource_alloc.comm
        if isinstance(derivative_dimensions, int):
            num_params = derivative_dimensions
        elif isinstance(derivative_dimensions, tuple):
            assert len(derivative_dimensions) == 1
            num_params = derivative_dimensions[0]
        else:
            num_params = self.model.num_params
        C = 1.0 / (1024.0**3)

        if mem_limit is not None:
            if mem_limit <= 0:
                raise MemoryError("Attempted layout creation w/memory limit = %g <= 0!" % mem_limit)
            printer.log("Layout creation w/mem limit = %.2fGB" % (mem_limit * C))

        #Start with how we'd like to split processors up (without regard to memory limit):        
        #The current implementation of map (should) benefit more from having a matching between the number of atoms
        #and the number of processors, at least for up to around two-qubits.
        default_natoms = nprocs # heuristic
        #TODO: factor in the mem_limit value to more intelligently set the default number of atoms.

        natoms, na, npp, param_dimensions, param_blk_sizes = self._compute_processor_distribution(
            array_types, nprocs, num_params, len(circuits), default_natoms=default_natoms)  
        
        printer.log(f'Num Param Processors {npp}')
        
        printer.log("MapLayout: %d processors divided into %s (= %d) grid along circuit and parameter directions." %
                    (nprocs, ' x '.join(map(str, (na,) + npp)), _np.prod((na,) + npp)))
        printer.log("   %d atoms, parameter block size limits %s" % (natoms, str(param_blk_sizes)))
        assert(_np.prod((na,) + npp) <= nprocs), "Processor grid size exceeds available processors!"

        layout = _MapCOPALayout(circuits, self.model, dataset, self._max_cache_size, natoms, na, npp,
                                param_dimensions, param_blk_sizes, resource_alloc,circuit_partition_cost_functions,
                                verbosity, layout_creation_circuit_cache= layout_creation_circuit_cache,
                                load_balancing_parameters=load_balancing_parameters)

        if mem_limit is not None:
            loc_nparams1 = num_params / npp[0] if len(npp) > 0 else 0
            loc_nparams2 = num_params / npp[1] if len(npp) > 1 else 0
            blk1 = param_blk_sizes[0] if len(param_blk_sizes) > 0 else 0
            blk2 = param_blk_sizes[1] if len(param_blk_sizes) > 1 else 0
            if blk1 is None: blk1 = loc_nparams1
            if blk2 is None: blk2 = loc_nparams2
            global_layout = layout.global_layout
            if comm is not None:
                from mpi4py import MPI
                max_local_els = comm.allreduce(layout.num_elements, op=MPI.MAX)    # layout.max_atom_elements
                max_atom_els = comm.allreduce(layout.max_atom_elements, op=MPI.MAX)
                max_local_circuits = comm.allreduce(layout.num_circuits, op=MPI.MAX)
                max_atom_cachesize = comm.allreduce(layout.max_atom_cachesize, op=MPI.MAX)
            else:
                max_local_els = layout.num_elements
                max_atom_els = layout.max_atom_elements
                max_local_circuits = layout.num_circuits
                max_atom_cachesize = layout.max_atom_cachesize
            mem_estimate = _bytes_for_array_types(array_types, global_layout.num_elements, max_local_els, max_atom_els,
                                                  global_layout.num_circuits, max_local_circuits,
                                                  layout._param_dimensions, (loc_nparams1, loc_nparams2),
                                                  (blk1, blk2), max_atom_cachesize, self.model.dim)

            #def approx_mem_estimate(nc, np1, np2):
            #    approx_cachesize = (num_circuits / nc) * 1.3  # inflate expected # of circuits per atom => cache_size
            #    return _bytes_for_array_types(array_types, num_elements, num_elements / nc,
            #                                  num_circuits, num_circuits / nc,
            #                                  (num_params, num_params), (num_params / np1, num_params / np2),
            #                                  approx_cachesize, self.model.dim)

            GB = 1.0 / 1024.0**3
            if mem_estimate > mem_limit:
                raise MemoryError("Not enough memory for desired layout! (limit=%.1fGB, required=%.1fGB)" % (
                    mem_limit * GB, mem_estimate * GB))
            else:
                printer.log("   Esimated memory required = %.1fGB" % (mem_estimate * GB))

        return layout
    
    @staticmethod
    def create_copa_layout_circuit_cache(circuits, model, dataset=None):
        """
        Helper function for pre-computing/pre-processing circuits structures
        used in matrix layout creation.
        """
        cache = dict()
        completed_circuits = model.complete_circuits(circuits)

        cache['completed_circuits'] = {ckt: comp_ckt for ckt, comp_ckt in zip(circuits, completed_circuits)}

        split_circuits = model.split_circuits(completed_circuits, split_prep=False)    
        cache['split_circuits'] = {ckt: split_ckt for ckt, split_ckt in zip(circuits, split_circuits)}
        

        if dataset is not None:
            aliases = circuits.op_label_aliases if isinstance(circuits, _CircuitList) else None
            ds_circuits = _lt.apply_aliases_to_circuits(circuits, aliases)
            unique_outcomes_list = []
            for ckt in ds_circuits:
                ds_row = dataset[ckt]
                unique_outcomes_list.append(ds_row.unique_outcomes if ds_row is not None else None)
        else:
            unique_outcomes_list = [None]*len(circuits)

        expanded_circuit_outcome_list = model.bulk_expand_instruments_and_separate_povm(circuits, 
                                                                                        observed_outcomes_list = unique_outcomes_list, 
                                                                                        completed_circuits= completed_circuits)
        
        expanded_circuit_cache = {ckt: expanded_ckt for ckt,expanded_ckt in zip(completed_circuits, expanded_circuit_outcome_list)}                
        cache['expanded_and_separated_circuits'] = expanded_circuit_cache

        return cache


    def _bulk_fill_probs_atom(self, array_to_fill, layout_atom, resource_alloc):
        # Note: *don't* set dest_indices arg = layout.element_slice, as this is already done by caller
        resource_alloc.check_can_allocate_memory(layout_atom.cache_size * self.model.dim)
        self.calclib.mapfill_probs_atom(self, array_to_fill, slice(0, array_to_fill.shape[0]),  # all indices
                                        layout_atom, resource_alloc)

    def _bulk_fill_dprobs_atom(self, array_to_fill, dest_param_slice, layout_atom, param_slice, resource_alloc):
        # Note: *don't* set dest_indices arg = layout.element_slice, as this is already done by caller
        resource_alloc.check_can_allocate_memory(layout_atom.cache_size * self.model.dim * _slct.length(param_slice))
        self.calclib.mapfill_dprobs_atom(self, array_to_fill, slice(0, array_to_fill.shape[0]), dest_param_slice,
                                         layout_atom, param_slice, resource_alloc, self.derivative_eps)

    def _bulk_fill_hprobs_atom(self, array_to_fill, dest_param_slice1, dest_param_slice2, layout_atom,
                               param_slice1, param_slice2, resource_alloc):
        # Note: *don't* set dest_indices arg = layout.element_slice, as this is already done by caller
        resource_alloc.check_can_allocate_memory(layout_atom.cache_size * self.model.dim
                                                 * _slct.length(param_slice1) * _slct.length(param_slice2))
        self._mapfill_hprobs_atom(array_to_fill, slice(0, array_to_fill.shape[0]), dest_param_slice1,
                                  dest_param_slice2, layout_atom, param_slice1, param_slice2, resource_alloc,
                                  self.hessian_eps)

    #Not used enough to warrant pushing to evotypes yet... just keep a slow version
    def _mapfill_hprobs_atom(self, array_to_fill, dest_indices, dest_param_indices1, dest_param_indices2,
                             layout_atom, param_indices1, param_indices2, resource_alloc, eps):

        """
        Helper function for populating hessian values by block.
        """
        shared_mem_leader = resource_alloc.is_host_leader if (resource_alloc is not None) else True

        if param_indices1 is None:
            param_indices1 = list(range(self.model.num_params))
        if param_indices2 is None:
            param_indices2 = list(range(self.model.num_params))
        if dest_param_indices1 is None:
            dest_param_indices1 = list(range(_slct.length(param_indices1)))
        if dest_param_indices2 is None:
            dest_param_indices2 = list(range(_slct.length(param_indices2)))

        param_indices1 = _slct.to_array(param_indices1)
        dest_param_indices1 = _slct.to_array(dest_param_indices1)
        #dest_param_indices2 = _slct.to_array(dest_param_indices2)  # OK if a slice

        #Get a map from global parameter indices to the desired
        # final index within mx_to_fill (fpoffset = final parameter offset)
        iParamToFinal = {i: dest_index for i, dest_index in zip(param_indices1, dest_param_indices1)}

        nEls = layout_atom.num_elements
        nP2 = _slct.length(param_indices2) if isinstance(param_indices2, slice) else len(param_indices2)
        dprobs, shm = _smt.create_shared_ndarray(resource_alloc, (nEls, nP2), 'd')
        dprobs2, shm2 = _smt.create_shared_ndarray(resource_alloc, (nEls, nP2), 'd')
        self.calclib.mapfill_dprobs_atom(self, dprobs, slice(0, nEls), None, layout_atom, param_indices2,
                                         resource_alloc, eps)

        orig_vec = self.model.to_vector().copy()
        for i in range(self.model.num_params):
            if i in iParamToFinal:
                iFinal = iParamToFinal[i]
                vec = orig_vec.copy(); vec[i] += eps
                self.model.from_vector(vec, close=True)
                self.calclib.mapfill_dprobs_atom(self, dprobs2, slice(0, nEls), None, layout_atom,
                                                 param_indices2, resource_alloc, eps)
                if shared_mem_leader:
                    _fas(array_to_fill, [dest_indices, iFinal, dest_param_indices2], (dprobs2 - dprobs) / eps)
        self.model.from_vector(orig_vec)
        _smt.cleanup_shared_ndarray(shm)
        _smt.cleanup_shared_ndarray(shm2)

    ## ---------------------------------------------------------------------------------------------
    ## TIME DEPENDENT functionality ----------------------------------------------------------------
    ## ---------------------------------------------------------------------------------------------

    def bulk_fill_timedep_chi2(self, array_to_fill, layout, ds_circuits, num_total_outcomes, dataset,
                               min_prob_clip_for_weighting, prob_clip_interval, ds_cache=None):
        """
        Compute the chi2 contributions for an entire tree of circuits, allowing for time dependent operations.

        Computation is performed by summing together the contributions for each time the circuit is
        run, as given by the timestamps in `dataset`.

        Parameters
        ----------
        array_to_fill : numpy ndarray
            an already-allocated 1D numpy array of length equal to the
            total number of computed elements (i.e. layout.num_elements)

        layout : CircuitOutcomeProbabilityArrayLayout
            A layout for `array_to_fill`, describing what circuit outcome each
            element corresponds to.  Usually given by a prior call to :meth:`create_layout`.

        ds_circuits : list of Circuits
            the circuits to use as they should be queried from `dataset` (see
            below).  This is typically the same list of circuits used to
            construct `layout` potentially with some aliases applied.

        num_total_outcomes : list or array
            a list of the total number of *possible* outcomes for each circuit
            (so `len(num_total_outcomes) == len(ds_circuits_to_use)`).  This is
            needed for handling sparse data, where `dataset` may not contain
            counts for all the possible outcomes of each circuit.

        dataset : DataSet
            the data set used to compute the chi2 contributions.

        min_prob_clip_for_weighting : float, optional
            Sets the minimum and maximum probability p allowed in the chi^2
            weights: N/(p*(1-p)) by clipping probability p values to lie within
            the interval [ min_prob_clip_for_weighting, 1-min_prob_clip_for_weighting ].

        prob_clip_interval : 2-tuple or None, optional
            (min,max) values used to clip the predicted probabilities to.
            If None, no clipping is performed.

        Returns
        -------
        None
        """
        def compute_timedep(layout_atom, resource_alloc):
            dataset_rows = {i_expanded: dataset[ds_circuits[i]]
                            for i_expanded, i in layout_atom.orig_indices_by_expcircuit.items()}
            num_outcomes = {i_expanded: num_total_outcomes[i]
                            for i_expanded, i in layout_atom.orig_indices_by_expcircuit.items()}
            self.calclib.mapfill_TDchi2_terms(self, array_to_fill, layout_atom.element_slice, num_outcomes,
                                              layout_atom, dataset_rows, min_prob_clip_for_weighting,
                                              prob_clip_interval, resource_alloc, outcomes_cache=None)

        atom_resource_alloc = layout.resource_alloc('atom-processing')
        atom_resource_alloc.host_comm_barrier()  # ensure all procs have finished w/shared memory before we begin

        for atom in layout.atoms:  # layout only holds local atoms
            compute_timedep(atom, atom_resource_alloc)

        atom_resource_alloc.host_comm_barrier()  # don't exit until all procs' array_to_fill is ready

    def bulk_fill_timedep_dchi2(self, array_to_fill, layout, ds_circuits, num_total_outcomes, dataset,
                                min_prob_clip_for_weighting, prob_clip_interval, chi2_array_to_fill=None,
                                ds_cache=None):
        """
        Compute the chi2 jacobian contributions for an entire tree of circuits, allowing for time dependent operations.

        Similar to :meth:`bulk_fill_timedep_chi2` but compute the *jacobian*
        of the summed chi2 contributions for each circuit with respect to the
        model's parameters.

        Parameters
        ----------
        array_to_fill : numpy ndarray
            an already-allocated ExM numpy array where E is the total number of
            computed elements (i.e. layout.num_elements) and M is the
            number of model parameters.

        layout : CircuitOutcomeProbabilityArrayLayout
            A layout for `array_to_fill`, describing what circuit outcome each
            element corresponds to.  Usually given by a prior call to :meth:`create_layout`.

        ds_circuits : list of Circuits
            the circuits to use as they should be queried from `dataset` (see
            below).  This is typically the same list of circuits used to
            construct `layout` potentially with some aliases applied.

        num_total_outcomes : list or array
            a list of the total number of *possible* outcomes for each circuit
            (so `len(num_total_outcomes) == len(ds_circuits_to_use)`).  This is
            needed for handling sparse data, where `dataset` may not contain
            counts for all the possible outcomes of each circuit.

        dataset : DataSet
            the data set used to compute the chi2 contributions.

        min_prob_clip_for_weighting : float, optional
            Sets the minimum and maximum probability p allowed in the chi^2
            weights: `N/(p*(1-p))` by clipping probability p values to lie within
            the interval [ min_prob_clip_for_weighting, 1-min_prob_clip_for_weighting ].

        prob_clip_interval : 2-tuple or None, optional
            (min,max) values used to clip the predicted probabilities to.
            If None, no clipping is performed.

        chi2_array_to_fill : numpy array, optional
            when not None, an already-allocated length-E numpy array that is filled
            with the per-circuit chi2 contributions, just like in
            `bulk_fill_timedep_chi2(...)`.

        Returns
        -------
        None
        """
        outcomes_cache = {}  # for performance

        def dchi2(dest_mx, dest_indices, dest_param_indices, num_tot_outcomes, layout_atom,
                  dataset_rows, wrt_slice, fill_comm):
            self.calclib.mapfill_TDdchi2_terms(self, dest_mx, dest_indices, dest_param_indices, num_tot_outcomes,
                                               layout_atom, dataset_rows, min_prob_clip_for_weighting,
                                               prob_clip_interval, wrt_slice, fill_comm, outcomes_cache)

        def chi2(dest_mx, dest_indices, num_tot_outcomes, layout_atom, dataset_rows, fill_comm):
            return self.calclib.mapfill_TDchi2_terms(self, dest_mx, dest_indices, num_tot_outcomes, layout_atom,
                                                     dataset_rows, min_prob_clip_for_weighting, prob_clip_interval,
                                                     fill_comm, outcomes_cache)

        return self._bulk_fill_timedep_deriv(layout, dataset, ds_circuits, num_total_outcomes,
                                             array_to_fill, dchi2, chi2_array_to_fill, chi2)

    def bulk_fill_timedep_loglpp(self, array_to_fill, layout, ds_circuits, num_total_outcomes, dataset,
                                 min_prob_clip, radius, prob_clip_interval, ds_cache=None):
        """
        Compute the log-likelihood contributions (within the "poisson picture") for an entire tree of circuits.

        Computation is performed by summing together the contributions for each time the circuit is run,
        as given by the timestamps in `dataset`.

        Parameters
        ----------
        array_to_fill : numpy ndarray
            an already-allocated 1D numpy array of length equal to the
            total number of computed elements (i.e. layout.num_elements)

        layout : CircuitOutcomeProbabilityArrayLayout
            A layout for `array_to_fill`, describing what circuit outcome each
            element corresponds to.  Usually given by a prior call to :meth:`create_layout`.

        ds_circuits : list of Circuits
            the circuits to use as they should be queried from `dataset` (see
            below).  This is typically the same list of circuits used to
            construct `layout` potentially with some aliases applied.

        num_total_outcomes : list or array
            a list of the total number of *possible* outcomes for each circuit
            (so `len(num_total_outcomes) == len(ds_circuits_to_use)`).  This is
            needed for handling sparse data, where `dataset` may not contain
            counts for all the possible outcomes of each circuit.

        dataset : DataSet
            the data set used to compute the logl contributions.

        min_prob_clip : float, optional
            The minimum probability treated normally in the evaluation of the
            log-likelihood.  A penalty function replaces the true log-likelihood
            for probabilities that lie below this threshold so that the
            log-likelihood never becomes undefined (which improves optimizer
            performance).

        radius : float, optional
            Specifies the severity of rounding used to "patch" the
            zero-frequency terms of the log-likelihood.

        prob_clip_interval : 2-tuple or None, optional
            (min,max) values used to clip the predicted probabilities to.
            If None, no clipping is performed.

        Returns
        -------
        None
        """
        def compute_timedep(layout_atom, resource_alloc):
            dataset_rows = {i_expanded: dataset[ds_circuits[i]]
                            for i_expanded, i in layout_atom.orig_indices_by_expcircuit.items()}
            num_outcomes = {i_expanded: num_total_outcomes[i]
                            for i_expanded, i in layout_atom.orig_indices_by_expcircuit.items()}
            self.calclib.mapfill_TDloglpp_terms(self, array_to_fill, layout_atom.element_slice, num_outcomes,
                                                layout_atom, dataset_rows, min_prob_clip,
                                                radius, prob_clip_interval, resource_alloc, outcomes_cache=None)

        atom_resource_alloc = layout.resource_alloc('atom-processing')
        atom_resource_alloc.host_comm_barrier()  # ensure all procs have finished w/shared memory before we begin

        for atom in layout.atoms:  # layout only holds local atoms
            compute_timedep(atom, atom_resource_alloc)

        atom_resource_alloc.host_comm_barrier()  # don't exit until all procs' array_to_fill is ready

    def bulk_fill_timedep_dloglpp(self, array_to_fill, layout, ds_circuits, num_total_outcomes, dataset,
                                  min_prob_clip, radius, prob_clip_interval, logl_array_to_fill=None,
                                  ds_cache=None):
        """
        Compute the ("poisson picture")log-likelihood jacobian contributions for an entire tree of circuits.

        Similar to :meth:`bulk_fill_timedep_loglpp` but compute the *jacobian*
        of the summed logl (in posison picture) contributions for each circuit
        with respect to the model's parameters.

        Parameters
        ----------
        array_to_fill : numpy ndarray
            an already-allocated ExM numpy array where E is the total number of
            computed elements (i.e. layout.num_elements) and M is the
            number of model parameters.

        layout : CircuitOutcomeProbabilityArrayLayout
            A layout for `array_to_fill`, describing what circuit outcome each
            element corresponds to.  Usually given by a prior call to :meth:`create_layout`.

        ds_circuits : list of Circuits
            the circuits to use as they should be queried from `dataset` (see
            below).  This is typically the same list of circuits used to
            construct `layout` potentially with some aliases applied.

        num_total_outcomes : list or array
            a list of the total number of *possible* outcomes for each circuit
            (so `len(num_total_outcomes) == len(ds_circuits_to_use)`).  This is
            needed for handling sparse data, where `dataset` may not contain
            counts for all the possible outcomes of each circuit.

        dataset : DataSet
            the data set used to compute the logl contributions.

        min_prob_clip : float
            a regularization parameter for the log-likelihood objective function.

        radius : float
            a regularization parameter for the log-likelihood objective function.

        prob_clip_interval : 2-tuple or None, optional
            (min,max) values used to clip the predicted probabilities to.
            If None, no clipping is performed.

        logl_array_to_fill : numpy array, optional
            when not None, an already-allocated length-E numpy array that is filled
            with the per-circuit logl contributions, just like in
            `bulk_fill_timedep_loglpp(...)` .

        Returns
        -------
        None
        """
        outcomes_cache = {}  # for performance

        def dloglpp(array_to_fill, dest_indices, dest_param_indices, num_tot_outcomes, layout_atom,
                    dataset_rows, wrt_slice, fill_comm):
            return self.calclib.mapfill_TDdloglpp_terms(self, array_to_fill, dest_indices, dest_param_indices,
                                                        num_tot_outcomes, layout_atom, dataset_rows, min_prob_clip,
                                                        radius, prob_clip_interval, wrt_slice, fill_comm,
                                                        outcomes_cache)

        def loglpp(array_to_fill, dest_indices, num_tot_outcomes, layout_atom, dataset_rows, fill_comm):
            return self.calclib.mapfill_TDloglpp_terms(self, array_to_fill, dest_indices, num_tot_outcomes, layout_atom,
                                                       dataset_rows, min_prob_clip, radius, prob_clip_interval,
                                                       fill_comm, outcomes_cache)

        return self._bulk_fill_timedep_deriv(layout, dataset, ds_circuits, num_total_outcomes,
                                             array_to_fill, dloglpp, logl_array_to_fill, loglpp)

    
    #Utility method for generating process matrices for circuits. Should not be used for forward
    #simulation when using the MapForwardSimulator.
    def product(self, circuit, scale=False):
        """
        Compute the product of a specified sequence of operation labels.

        Note: LinearOperator matrices are multiplied in the reversed order of the tuple. That is,
        the first element of circuit can be thought of as the first gate operation
        performed, which is on the far right of the product of matrices.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
            The sequence of operation labels.

        scale : bool, optional
            When True, return a scaling factor (see below).

        Returns
        -------
        product : numpy array
            The product or scaled product of the operation matrices.
        scale : float
            Only returned when scale == True, in which case the
            actual product == product * scale.  The purpose of this
            is to allow a trace or other linear operation to be done
            prior to the scaling.
        """
        _warnings.warn('Generating dense process matrix representations of circuits or gates \n'
                       'can be inefficient and should be avoided for the purposes of forward \n'
                       'simulation/calculation of circuit outcome probability distributions \n' 
                       'when using the MapForwardSimulator.')
        
        # Smallness tolerances, used internally for conditional scaling required
        # to control bulk products, their gradients, and their Hessians.
        _PSMALL = 1e-100
        
        if scale:
            scaledGatesAndExps = {}
            scale_exp = 0
            G = _np.identity(self.model.evotype.minimal_dim(self.model.state_space))
            for lOp in circuit:
                if lOp not in scaledGatesAndExps:
                    opmx = self.model.circuit_layer_operator(lOp, 'op').to_dense(on_space='minimal')
                    ng = max(_nla.norm(opmx), 1.0)
                    scaledGatesAndExps[lOp] = (opmx / ng, _np.log(ng))

                gate, ex = scaledGatesAndExps[lOp]
                H = _np.dot(gate, G)   # product of gates, starting with identity
                scale_exp += ex   # scale and keep track of exponent
                if H.max() < _PSMALL and H.min() > -_PSMALL:
                    nG = max(_nla.norm(G), _np.exp(-scale_exp))
                    G = _np.dot(gate, G / nG); scale_exp += _np.log(nG)  # LEXICOGRAPHICAL VS MATRIX ORDER
                else: G = H

            old_err = _np.seterr(over='ignore')
            scale = _np.exp(scale_exp)
            _np.seterr(**old_err)

            return G, scale

        else:
            G = _np.identity(self.model.evotype.minimal_dim(self.model.state_space))
            for lOp in circuit:
                G = _np.dot(self.model.circuit_layer_operator(lOp, 'op').to_dense(on_space='minimal'), G)
                # above line: LEXICOGRAPHICAL VS MATRIX ORDER
            return G