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
from .profiler import DummyProfiler as _DummyProfiler
from .label import Label as _Label
from .maplayout import MapCOPALayout as _MapCOPALayout
from .forwardsim import ForwardSimulator as _ForwardSimulator
from .distforwardsim import DistributableForwardSimulator as _DistributableForwardSimulator
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
                rhorep = self.model.prep(spc.circuit_without_povm[0])._rep
                ereps = [self.model.effect(elabel)._rep for elabel in spc.effect_labels]
                rhorep = replib.propagate_staterep(rhorep,
                                                   [self.sos.operation(ol)._rep for ol in spc.circuit_without_povm[1:]])
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

    def __init__(self, model=None, max_cache_size=None):
        super().__init__(model)
        self._max_cache_size = max_cache_size

    def copy(self):
        """
        Return a shallow copy of this MapForwardSimulator

        Returns
        -------
        MapForwardSimulator
        """
        return MapForwardSimulator(self.model, self._max_cache_size)

    def create_layout(self, circuits, dataset=None, resource_alloc=None, array_types=('p',),
                      derivative_dimension=None, verbosity=0):

        resource_alloc = _ResourceAllocation.cast(resource_alloc)
        comm = resource_alloc.comm
        mem_limit = resource_alloc.mem_limit  # *per-processor* memory limit
        printer = _VerbosityPrinter.create_printer(verbosity, comm)
        nprocs = 1 if comm is None else comm.Get_size()
        num_params = derivative_dimension if (derivative_dimension is not None) else self.model.num_params()
        C = 1.0 / (1024.0**3)

        if mem_limit is not None:
            if mem_limit <= 0:
                raise MemoryError("Attempted layout creation w/memory limit = %g <= 0!" % mem_limit)
            printer.log("Layout creation w/mem limit = %.2fGB" % (mem_limit * C))

        def create_layout_candidate(num_atoms):
            return _MapCOPALayout(circuits, self.model, dataset, None, None, num_atoms,
                                  (num_params, num_params), verbosity)

        #Start with how we'd like to split processors up (without regard to memory limit):
        np1 = 1; np2 = 1; nc = Ng = min(nprocs, len(circuits))

        #Create initial layout, and get the "final memory" that is required to hold the final results
        # for each array type.  This amount doesn't depend on how the layout is "split" into atoms.
        layout_cache = {}  # cache of layout candidates indexed on # (minimal) atoms, to avoid re-computation
        layout_cache[nc] = create_layout_candidate(nc)

        if mem_limit is not None:

            final_mem = sum([layout_cache[nc].memory_estimate(array_type) for array_type in array_types])
            gather_mem_limit = mem_limit * 0.01  # better?
            cache_mem_limit = mem_limit - final_mem - gather_mem_limit
            if cache_mem_limit < 0: raise MemoryError("Not enough memory to hold final results!")

            d = self.model.dim
            bytes_per_element = _np.dtype('d').itemsize
            num_circuits = len(circuits)

            def _cache_mem(cache_size, wrtblk1_size, wrtblk2_size):  # based on cache size and param block sizes
                mem = 0
                for array_type in array_types:
                    if array_type == "p": mem += cache_size * d * bytes_per_element
                    elif array_type == "dp": mem += 2 * cache_size * d * bytes_per_element
                    elif array_type == "hp": mem += cache_size * d * wrtblk2_size * bytes_per_element
                    else: raise ValueError("Invalid array type: %s" % array_type)
                return mem

            def cache_mem_estimate(nc, np1, np2, n_comms):
                """ Estimate of memory required by "cache" - the only memory that dependes on the layout distribution"""
                if nc not in layout_cache: layout_cache[nc] = create_layout_candidate(nc)
                trial_layout = layout_cache[nc]

                #Each atom holds its own cache, and when these include derivatives these are computed by *block*
                max_cache_size = max([atom.cache_size for atom in trial_layout.atoms])
                blk1, blk2 = num_params / np1, num_params / np2  # float blk sizes ok for now
                return _cache_mem(max_cache_size, blk1, blk2)

            def approx_cache_mem_estimate(nc, np1, np2, n_comms):
                approx_cache_size = (len(circuits) / nc) * 0.7
                if self._max_cache_size is not None:
                    approx_cache_size = min(approx_cache_size, self._max_cache_size)
                return _cache_mem(approx_cache_size, num_params / np1, num_params / np2)

            cmem = cache_mem_estimate(nc, np1, np2, Ng)  # initial estimate (to screen)
            #printer.log(f" mem({nc} atoms, {np1},{np2} param-grps, {Ng} proc-grps) = {(final_mem + cmem) * C}GB")
            printer.log(" mem(%d atoms, %d,%d param-grps, %d proc-grps) = %.2fGB" %
                        (nc, np1, np2, Ng, (final_mem + cmem) * C))

            #Increase nc in amounts of Ng (so nc % Ng == 0).  Start with approximation, then switch to slow mode.
            while approx_cache_mem_estimate(nc, np1, np2, Ng) > cache_mem_limit:
                if nc == num_circuits:  # even "maximal" splitting doesn't work!
                    raise MemoryError("Cannot split or layout enough to achieve memory limit")
                nc += Ng
                if nc > num_circuits: nc = num_circuits

            cmem = cache_mem_estimate(nc, np1, np2, Ng)
            #printer.log(f" mem({nc} atoms, {np1},{np2} param-grps, {Ng} proc-grps) = {(final_mem + cmem) * C}GB")
            printer.log(" mem(%d atoms, %d,%d param-grps, %d proc-grps) = %.2fGB" %
                        (nc, np1, np2, Ng, (final_mem + cmem) * C))
            while cmem > cache_mem_limit:
                nc += Ng; _next = cache_mem_estimate(nc, np1, np2, Ng, log=True)
                if(_next >= cmem): raise MemoryError("Not enough memory: splitting unproductive")
                cmem = _next

                #Note: could do these while loops smarter, e.g. binary search-like?
                #  or assume mem_estimate scales linearly in ng? E.g:
                #     if mem_limit < estimate:
                #         reductionFactor = float(estimate) / float(mem_limit)
                #         maxTreeSize = int(nstrs / reductionFactor)
        else:
            gather_mem_limit = None

        layout = layout_cache[nc]

        paramBlkSize1 = num_params / np1
        paramBlkSize2 = num_params / np2  # the *average* param block size
        # (in general *not* an integer), which ensures that the intended # of
        # param blocks is communicated to forwardsim routines (taking ceiling or
        # floor can lead to inefficient MPI distribution)

        bNp2Matters = bool("hp" in array_types)
        nparams = (num_params, num_params) if bNp2Matters else num_params
        np = (np1, np2) if bNp2Matters else np1
        paramBlkSizes = (paramBlkSize1, paramBlkSize2) if bNp2Matters else paramBlkSize1
        #printer.log((f"Created map-sim layout for {len(circuits)} circuits over {nprocs} processors:\n"
        #             f" Layout comprised of {nc} atoms, processed in {Ng} groups of ~{nprocs // Ng} processors each.\n"
        #             f" {nparams} parameters divided into {np} blocks of ~{paramBlkSizes} params."))
        printer.log(("Created map-sim layout for %d circuits over %d processors:\n"
                     " Layout comprised of %d atoms, processed in %d groups of ~%d processors each.\n"
                     " %d parameters divided into %d blocks of ~%s params.") %
                    (len(circuits), nprocs, nc, Ng, nprocs // Ng, nparams, np, str(paramBlkSizes)))

        if np1 == 1:  # (paramBlkSize == num_params)
            paramBlkSize1 = None  # == all parameters, and may speed logic in dprobs, etc.
        else:
            if comm is not None:  # check that all procs have *same* paramBlkSize1
                blkSizeTest = comm.bcast(paramBlkSize1, root=0)
                assert(abs(blkSizeTest - paramBlkSize1) < 1e-3)

        if np2 == 1:  # (paramBlkSize == num_params)
            paramBlkSize2 = None  # == all parameters, and may speed logic in hprobs, etc.
        else:
            if comm is not None:  # check that all procs have *same* paramBlkSize2
                blkSizeTest = comm.bcast(paramBlkSize2, root=0)
                assert(abs(blkSizeTest - paramBlkSize2) < 1e-3)

        layout.set_distribution_params(Ng, (paramBlkSize1, paramBlkSize2), gather_mem_limit)
        return layout

    def _bulk_fill_probs_block(self, array_to_fill, layout_atom, resource_alloc):
        # Note: *don't* set dest_indices arg = layout.element_slice, as this is already done by caller
        replib.DM_mapfill_probs_block(self, array_to_fill, slice(0, array_to_fill.shape[0]),  # all indices
                                      layout_atom, resource_alloc.comm)

    def _bulk_fill_dprobs_block(self, array_to_fill, dest_param_slice, layout_atom, param_slice, resource_alloc):
        # Note: *don't* set dest_indices arg = layout.element_slice, as this is already done by caller
        replib.DM_mapfill_dprobs_block(self, array_to_fill, slice(0, array_to_fill.shape[0]), dest_param_slice,
                                       layout_atom, param_slice, resource_alloc.comm)

    def _bulk_fill_hprobs_block(self, array_to_fill, dest_param_slice1, dest_param_slice2, layout_atom,
                                param_slice1, param_slice2, resource_alloc):
        # Note: *don't* set dest_indices arg = layout.element_slice, as this is already done by caller
        self._dm_mapfill_hprobs_block(array_to_fill, slice(0, array_to_fill.shape[0]), dest_param_slice1,
                                      dest_param_slice2, layout_atom, param_slice1, param_slice2, resource_alloc.comm)

    #Not used enough to warrant pushing to replibs yet... just keep a slow version
    def _dm_mapfill_hprobs_block(self, array_to_fill, dest_indices, dest_param_indices1, dest_param_indices2,
                                 layout_atom, param_indices1, param_indices2, comm):

        """
        Helper function for populating hessian values by block.
        """
        eps = 1e-4  # hardcoded?

        if param_indices1 is None:
            param_indices1 = list(range(self.model.num_params()))
        if param_indices2 is None:
            param_indices2 = list(range(self.model.num_params()))
        if dest_param_indices1 is None:
            dest_param_indices1 = list(range(_slct.length(param_indices1)))
        if dest_param_indices2 is None:
            dest_param_indices2 = list(range(_slct.length(param_indices2)))

        param_indices1 = _slct.to_array(param_indices1)
        dest_param_indices1 = _slct.to_array(dest_param_indices1)
        #dest_param_indices2 = _slct.to_array(dest_param_indices2)  # OK if a slice

        all_slices, my_slice, owners, subComm = \
            _mpit.distribute_slice(slice(0, len(param_indices1)), comm)

        my_param_indices = param_indices1[my_slice]
        st = my_slice.start

        #Get a map from global parameter indices to the desired
        # final index within mx_to_fill (fpoffset = final parameter offset)
        iParamToFinal = {i: dest_param_indices1[st + ii] for ii, i in enumerate(my_param_indices)}

        nEls = layout_atom.num_elements
        nP2 = _slct.length(param_indices2) if isinstance(param_indices2, slice) else len(param_indices2)
        dprobs = _np.empty((nEls, nP2), 'd')
        dprobs2 = _np.empty((nEls, nP2), 'd')
        replib.DM_mapfill_dprobs_block(self, dprobs, slice(0, nEls), None, layout_atom, param_indices2, comm)

        orig_vec = self.model.to_vector().copy()
        for i in range(self.model.num_params()):
            if i in iParamToFinal:
                iFinal = iParamToFinal[i]
                vec = orig_vec.copy(); vec[i] += eps
                self.model.from_vector(vec, close=True)
                replib.DM_mapfill_dprobs_block(self, dprobs2, slice(0, nEls), None, layout_atom,
                                               param_indices2, subComm)
                _fas(array_to_fill, [dest_indices, iFinal, dest_param_indices2], (dprobs2 - dprobs) / eps)
        self.model.from_vector(orig_vec)

        #Now each processor has filled the relavant parts of mx_to_fill, so gather together:
        _mpit.gather_slices(all_slices, owners, array_to_fill, [], axes=1, comm=comm)

    ## ---------------------------------------------------------------------------------------------
    ## TIME DEPENDENT functionality ----------------------------------------------------------------
    ## ---------------------------------------------------------------------------------------------

    def bulk_fill_timedep_chi2(self, array_to_fill, layout, ds_circuits, num_total_outcomes, dataset,
                               min_prob_clip_for_weighting, prob_clip_interval, resource_alloc=None):
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
                                           prob_clip_interval, sub_resource_alloc)

        atomOwners = self._compute_on_atoms(layout, compute_timedep, resource_alloc)

        #collect/gather results
        all_atom_element_slices = [atom.element_slice for atom in layout.atoms]
        _mpit.gather_slices(all_atom_element_slices, atomOwners, array_to_fill, [], 0, resource_alloc.comm)
        #note: pass mx_to_fill, dim=(KS,), so gather mx_to_fill[felInds] (axis=0)

    def bulk_fill_timedep_dchi2(self, array_to_fill, layout, ds_circuits, num_total_outcomes, dataset,
                                min_prob_clip_for_weighting, prob_clip_interval, chi2_array_to_fill=None,
                                wrt_filter=None, resource_alloc=None):
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

        wrt_filter : list of ints, optional
            If not None, a list of integers specifying which parameters
            to include in the derivative dimension. This argument is used
            internally for distributing calculations across multiple
            processors and to control memory usage.

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        Returns
        -------
        None
        """
        def dchi2(dest_mx, dest_indices, dest_param_indices, num_tot_outcomes, layout_atom,
                  dataset_rows, wrt_slice, fill_comm):
            replib.DM_mapfill_TDdchi2_terms(self, dest_mx, dest_indices, dest_param_indices, num_tot_outcomes,
                                            layout_atom, dataset_rows, min_prob_clip_for_weighting,
                                            prob_clip_interval, wrt_slice, fill_comm)

        def chi2(dest_mx, dest_indices, num_tot_outcomes, layout_atom, dataset_rows, fill_comm):
            return replib.DM_mapfill_TDchi2_terms(self, dest_mx, dest_indices, num_tot_outcomes, layout_atom,
                                                  dataset_rows, min_prob_clip_for_weighting, prob_clip_interval,
                                                  fill_comm)

        return self._bulk_fill_timedep_deriv(layout, dataset, ds_circuits, num_total_outcomes,
                                             array_to_fill, dchi2, chi2_array_to_fill, chi2,
                                             wrt_filter, resource_alloc)

    def bulk_fill_timedep_loglpp(self, array_to_fill, layout, ds_circuits, num_total_outcomes, dataset,
                                 min_prob_clip, radius, prob_clip_interval, resource_alloc=None):
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
                                             radius, prob_clip_interval, sub_resource_alloc)

        atomOwners = self._compute_on_atoms(layout, compute_timedep, resource_alloc)

        #collect/gather results
        all_atom_element_slices = [atom.element_slice for atom in layout.atoms]
        _mpit.gather_slices(all_atom_element_slices, atomOwners, array_to_fill, [], 0, resource_alloc.comm)
        #note: pass mx_to_fill, dim=(KS,), so gather mx_to_fill[felInds] (axis=0)

    def bulk_fill_timedep_dloglpp(self, array_to_fill, layout, ds_circuits, num_total_outcomes, dataset,
                                  min_prob_clip, radius, prob_clip_interval, logl_array_to_fill=None,
                                  wrt_filter=None, resource_alloc=None):
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

        wrt_filter : list of ints, optional
            If not None, a list of integers specifying which parameters
            to include in the derivative dimension. This argument is used
            internally for distributing calculations across multiple
            processors and to control memory usage.

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        Returns
        -------
        None
        """
        def dloglpp(array_to_fill, dest_indices, dest_param_indices, num_tot_outcomes, layout_atom,
                    dataset_rows, wrt_slice, fill_comm):
            return replib.DM_mapfill_TDdloglpp_terms(self, array_to_fill, dest_indices, dest_param_indices,
                                                     num_tot_outcomes, layout_atom, dataset_rows, min_prob_clip,
                                                     radius, prob_clip_interval, wrt_slice, fill_comm)

        def loglpp(array_to_fill, dest_indices, num_tot_outcomes, layout_atom, dataset_rows, fill_comm):
            return replib.DM_mapfill_TDloglpp_terms(self, array_to_fill, dest_indices, num_tot_outcomes, layout_atom,
                                                    dataset_rows, min_prob_clip, radius, prob_clip_interval, fill_comm)

        return self._bulk_fill_timedep_deriv(layout, dataset, ds_circuits, num_total_outcomes,
                                             array_to_fill, dloglpp, logl_array_to_fill, loglpp,
                                             wrt_filter, resource_alloc)
