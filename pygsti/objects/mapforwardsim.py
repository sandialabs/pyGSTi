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

import warnings as _warnings
import numpy as _np
import time as _time
import itertools as _itertools

from ..tools import mpitools as _mpit
from ..tools import slicetools as _slct
from ..tools.matrixtools import _fas
from ..tools import symplectic as _symp
from ..tools import sharedmemtools as _smt
from .profiler import DummyProfiler as _DummyProfiler
from .label import Label as _Label
from .maplayout import MapCOPALayout as _MapCOPALayout
from .forwardsim import ForwardSimulator as _ForwardSimulator
from .forwardsim import _bytes_for_array_types
from .distforwardsim import DistributableForwardSimulator as _DistributableForwardSimulator
from .distlayout import DistributableCOPALayout as _DistributableCOPALayout
from .resourceallocation import ResourceAllocation as _ResourceAllocation
from .verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from . import replib


_dummy_profiler = _DummyProfiler()

# FUTURE: use enum
_SUPEROP = 0
_UNITARY = 1
CLIFFORD = 2


class SimpleMapForwardSimulator(_ForwardSimulator):

    def _compute_circuit_outcome_probabilities(self, array_to_fill, circuit, outcomes, resource_alloc, time=None):
        expanded_circuit_outcomes = circuit.expand_instruments_and_separate_povm(self.model, outcomes)
        outcome_to_index = {outc: i for i, outc in enumerate(outcomes)}
        for spc, spc_outcomes in expanded_circuit_outcomes.items():  # spc is a SeparatePOVMCircuit
            # Note: `spc.circuit_without_povm` *always* begins with a prep label.
            indices = [outcome_to_index[o] for o in spc_outcomes]
            if time is None:  # time-independent state propagation
                rhorep = self.model.circuit_layer_operator(spc.circuit_without_povm[0], 'prep')._rep
                ereps = [self.model.circuit_layer_operator(elabel, 'povm')._rep for elabel in spc.full_effect_labels]
                rhorep = replib.propagate_staterep(rhorep,
                                                   [self.model.circuit_layer_operator(ol, 'op')._rep
                                                    for ol in spc.circuit_without_povm[1:]])
                array_to_fill[indices] = [erep.probability(rhorep) for erep in ereps]  # outcome probabilities
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


class MapForwardSimulator(_DistributableForwardSimulator, SimpleMapForwardSimulator):

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

    def __init__(self, model=None, max_cache_size=None, num_atoms=None, processor_grid=None, param_blk_sizes=None):
        """  TODO: docstring - at least need num_atoms, processor_grid, & param_blk_sizes docs"""
        super().__init__(model, num_atoms, processor_grid, param_blk_sizes)
        self._max_cache_size = max_cache_size

    def copy(self):
        """
        Return a shallow copy of this MapForwardSimulator

        Returns
        -------
        MapForwardSimulator
        """
        return MapForwardSimulator(self.model, self._max_cache_size)

    def create_layout(self, circuits, dataset=None, resource_alloc=None, array_types=('E',),
                      derivative_dimension=None, verbosity=0):

        resource_alloc = _ResourceAllocation.cast(resource_alloc)
        comm = resource_alloc.comm
        mem_limit = resource_alloc.mem_limit - resource_alloc.allocated_memory \
            if (resource_alloc.mem_limit is not None) else None  # *per-processor* memory limit
        printer = _VerbosityPrinter.create_printer(verbosity, comm)
        nprocs = 1 if comm is None else comm.Get_size()
        num_params = derivative_dimension if (derivative_dimension is not None) else self.model.num_params
        C = 1.0 / (1024.0**3)

        if mem_limit is not None:
            if mem_limit <= 0:
                raise MemoryError("Attempted layout creation w/memory limit = %g <= 0!" % mem_limit)
            printer.log("Layout creation w/mem limit = %.2fGB" % (mem_limit * C))

        #Start with how we'd like to split processors up (without regard to memory limit):

        # when there are lots of processors, the from_vector calls dominante over the actual fwdsim,
        # but we can reduce from_vector calls by having np1, np2 > 0 (each param requires a from_vector
        # call when using finite diffs) - so we want to choose nc = Ng < nprocs and np1 > 1 (so nc * np1 = nprocs).
        #work_per_proc = self.model.dim**2

        natoms, na, npp, param_dimensions, param_blk_sizes = self._compute_processor_distribution(
            array_types, nprocs, num_params, len(circuits), default_natoms=2 * self.model.dim)  # heuristic?

        printer.log("MapLayout: %d processors divided into %s (= %d) grid along circuit and parameter directions." %
                    (nprocs, ' x '.join(map(str, (na,) + npp)), _np.product((na,) + npp)))
        printer.log("   %d atoms, parameter block size limits %s" % (natoms, str(param_blk_sizes)))
        assert(_np.product((na,) + npp) <= nprocs), "Processor grid size exceeds available processors!"

        layout = _MapCOPALayout(circuits, self.model, dataset, None, natoms, na, npp,
                                param_dimensions, param_blk_sizes, resource_alloc, verbosity)

        if mem_limit is not None:
            loc_nparams1 = num_params / npp[0] if len(npp) > 0 else 0
            loc_nparams2 = num_params / npp[1] if len(npp) > 1 else 0
            blk1 = param_blk_sizes[0] if len(param_blk_sizes) > 0 else 0
            blk2 = param_blk_sizes[1] if len(param_blk_sizes) > 1 else 0
            if blk1 is None: blk1 = loc_nparams1
            if blk2 is None: blk1 = loc_nparams2
            global_layout = layout.global_layout if isinstance(layout, _DistributableCOPALayout) else layout
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

            if mem_estimate > mem_limit:
                raise MemoryError("Not enough memory for desired layout!")

        return layout

    def _bulk_fill_probs_block(self, array_to_fill, layout_atom, resource_alloc):
        # Note: *don't* set dest_indices arg = layout.element_slice, as this is already done by caller
        resource_alloc.check_can_allocate_memory(layout_atom.cache_size * self.model.dim)
        replib.DM_mapfill_probs_block(self, array_to_fill, slice(0, array_to_fill.shape[0]),  # all indices
                                      layout_atom, resource_alloc)

    def _bulk_fill_dprobs_block(self, array_to_fill, dest_param_slice, layout_atom, param_slice, resource_alloc):
        # Note: *don't* set dest_indices arg = layout.element_slice, as this is already done by caller
        resource_alloc.check_can_allocate_memory(layout_atom.cache_size * self.model.dim * _slct.length(param_slice))
        replib.DM_mapfill_dprobs_block(self, array_to_fill, slice(0, array_to_fill.shape[0]), dest_param_slice,
                                       layout_atom, param_slice, resource_alloc)

    def _bulk_fill_hprobs_block(self, array_to_fill, dest_param_slice1, dest_param_slice2, layout_atom,
                                param_slice1, param_slice2, resource_alloc):
        # Note: *don't* set dest_indices arg = layout.element_slice, as this is already done by caller
        resource_alloc.check_can_allocate_memory(layout_atom.cache_size * self.model.dim
                                                 * _slct.length(param_slice1) * _slct.length(param_slice2))
        self._dm_mapfill_hprobs_block(array_to_fill, slice(0, array_to_fill.shape[0]), dest_param_slice1,
                                      dest_param_slice2, layout_atom, param_slice1, param_slice2, resource_alloc)

    #Not used enough to warrant pushing to replibs yet... just keep a slow version
    def _dm_mapfill_hprobs_block(self, array_to_fill, dest_indices, dest_param_indices1, dest_param_indices2,
                                 layout_atom, param_indices1, param_indices2, resource_alloc):

        """
        Helper function for populating hessian values by block.
        """
        eps = 1e-4  # hardcoded?
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
        dprobs, shm = _smt.create_shared_ndarray(resource_alloc, (nEls, nP2), 'd', track_memory=False)
        dprobs2, shm2 = _smt.create_shared_ndarray(resource_alloc, (nEls, nP2), 'd', track_memory=False)
        replib.DM_mapfill_dprobs_block(self, dprobs, slice(0, nEls), None, layout_atom, param_indices2, resource_alloc)

        orig_vec = self.model.to_vector().copy()
        for i in range(self.model.num_params):
            if i in iParamToFinal:
                iFinal = iParamToFinal[i]
                vec = orig_vec.copy(); vec[i] += eps
                self.model.from_vector(vec, close=True)
                replib.DM_mapfill_dprobs_block(self, dprobs2, slice(0, nEls), None, layout_atom,
                                               param_indices2, resource_alloc)
                if shared_mem_leader:
                    _fas(array_to_fill, [dest_indices, iFinal, dest_param_indices2], (dprobs2 - dprobs) / eps)
        self.model.from_vector(orig_vec)
        _smt.cleanup_shared_ndarray(shm)
        _smt.cleanup_shared_ndarray(shm2)

    ## ---------------------------------------------------------------------------------------------
    ## TIME DEPENDENT functionality ----------------------------------------------------------------
    ## ---------------------------------------------------------------------------------------------

    def bulk_fill_timedep_chi2(self, array_to_fill, layout, ds_circuits, num_total_outcomes, dataset,
                               min_prob_clip_for_weighting, prob_clip_interval, resource_alloc=None,
                               ds_cache=None):
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
            element corresponds to.  Usually given by a prior call to :method:`create_layout`.

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

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        Returns
        -------
        None
        """
        def compute_timedep(layout_atom, sub_resource_alloc):
            dataset_rows = {i_expanded: dataset[ds_circuits[i]]
                            for i_expanded, i in layout_atom.orig_indices_by_expcircuit.items()}
            num_outcomes = {i_expanded: num_total_outcomes[i]
                            for i_expanded, i in layout_atom.orig_indices_by_expcircuit.items()}
            replib.DM_mapfill_TDchi2_terms(self, array_to_fill, layout_atom.element_slice, num_outcomes,
                                           layout_atom, dataset_rows, min_prob_clip_for_weighting,
                                           prob_clip_interval, sub_resource_alloc, outcomes_cache=None)

        atomOwners = self._compute_on_atoms(layout, compute_timedep, resource_alloc)

        #collect/gather results
        all_atom_element_slices = [atom.element_slice for atom in layout.atoms]
        _mpit.gather_slices(all_atom_element_slices, atomOwners, array_to_fill, [], 0, resource_alloc.comm)
        #note: pass mx_to_fill, dim=(KS,), so gather mx_to_fill[felInds] (axis=0)

    def bulk_fill_timedep_dchi2(self, array_to_fill, layout, ds_circuits, num_total_outcomes, dataset,
                                min_prob_clip_for_weighting, prob_clip_interval, chi2_array_to_fill=None,
                                resource_alloc=None, ds_cache=None):
        """
        Compute the chi2 jacobian contributions for an entire tree of circuits, allowing for time dependent operations.

        Similar to :method:`bulk_fill_timedep_chi2` but compute the *jacobian*
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
            element corresponds to.  Usually given by a prior call to :method:`create_layout`.

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

        chi2_array_to_fill : numpy array, optional
            when not None, an already-allocated length-E numpy array that is filled
            with the per-circuit chi2 contributions, just like in
            bulk_fill_timedep_chi2(...).

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        Returns
        -------
        None
        """
        outcomes_cache = {}  # for performance

        def dchi2(dest_mx, dest_indices, dest_param_indices, num_tot_outcomes, layout_atom,
                  dataset_rows, wrt_slice, fill_comm):
            replib.DM_mapfill_TDdchi2_terms(self, dest_mx, dest_indices, dest_param_indices, num_tot_outcomes,
                                            layout_atom, dataset_rows, min_prob_clip_for_weighting,
                                            prob_clip_interval, wrt_slice, fill_comm, outcomes_cache)

        def chi2(dest_mx, dest_indices, num_tot_outcomes, layout_atom, dataset_rows, fill_comm):
            return replib.DM_mapfill_TDchi2_terms(self, dest_mx, dest_indices, num_tot_outcomes, layout_atom,
                                                  dataset_rows, min_prob_clip_for_weighting, prob_clip_interval,
                                                  fill_comm, outcomes_cache)

        return self._bulk_fill_timedep_deriv(layout, dataset, ds_circuits, num_total_outcomes,
                                             array_to_fill, dchi2, chi2_array_to_fill, chi2,
                                             resource_alloc)

    def bulk_fill_timedep_loglpp(self, array_to_fill, layout, ds_circuits, num_total_outcomes, dataset,
                                 min_prob_clip, radius, prob_clip_interval, resource_alloc=None,
                                 ds_cache=None):
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
            element corresponds to.  Usually given by a prior call to :method:`create_layout`.

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

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        Returns
        -------
        None
        """
        def compute_timedep(layout_atom, sub_resource_alloc):
            dataset_rows = {i_expanded: dataset[ds_circuits[i]]
                            for i_expanded, i in layout_atom.orig_indices_by_expcircuit.items()}
            num_outcomes = {i_expanded: num_total_outcomes[i]
                            for i_expanded, i in layout_atom.orig_indices_by_expcircuit.items()}
            replib.DM_mapfill_TDloglpp_terms(self, array_to_fill, layout_atom.element_slice, num_outcomes,
                                             layout_atom, dataset_rows, min_prob_clip,
                                             radius, prob_clip_interval, sub_resource_alloc, outcomes_cache=None)

        atomOwners = self._compute_on_atoms(layout, compute_timedep, resource_alloc)

        #collect/gather results
        all_atom_element_slices = [atom.element_slice for atom in layout.atoms]
        _mpit.gather_slices(all_atom_element_slices, atomOwners, array_to_fill, [], 0, resource_alloc.comm)
        #note: pass mx_to_fill, dim=(KS,), so gather mx_to_fill[felInds] (axis=0)

    def bulk_fill_timedep_dloglpp(self, array_to_fill, layout, ds_circuits, num_total_outcomes, dataset,
                                  min_prob_clip, radius, prob_clip_interval, logl_array_to_fill=None,
                                  resource_alloc=None, ds_cache=None):
        """
        Compute the ("poisson picture")log-likelihood jacobian contributions for an entire tree of circuits.

        Similar to :method:`bulk_fill_timedep_loglpp` but compute the *jacobian*
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
            element corresponds to.  Usually given by a prior call to :method:`create_layout`.

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
            bulk_fill_timedep_loglpp(...).

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        Returns
        -------
        None
        """
        outcomes_cache = {}  # for performance

        def dloglpp(array_to_fill, dest_indices, dest_param_indices, num_tot_outcomes, layout_atom,
                    dataset_rows, wrt_slice, fill_comm):
            return replib.DM_mapfill_TDdloglpp_terms(self, array_to_fill, dest_indices, dest_param_indices,
                                                     num_tot_outcomes, layout_atom, dataset_rows, min_prob_clip,
                                                     radius, prob_clip_interval, wrt_slice, fill_comm, outcomes_cache)

        def loglpp(array_to_fill, dest_indices, num_tot_outcomes, layout_atom, dataset_rows, fill_comm):
            return replib.DM_mapfill_TDloglpp_terms(self, array_to_fill, dest_indices, num_tot_outcomes, layout_atom,
                                                    dataset_rows, min_prob_clip, radius, prob_clip_interval,
                                                    fill_comm, outcomes_cache)

        return self._bulk_fill_timedep_deriv(layout, dataset, ds_circuits, num_total_outcomes,
                                             array_to_fill, dloglpp, logl_array_to_fill, loglpp,
                                             resource_alloc)
