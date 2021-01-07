"""
Defines the DistributableForwardSimulator calculator class
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
import warnings as _warnings

from .forwardsim import ForwardSimulator as _ForwardSimulator
from .resourceallocation import ResourceAllocation as _ResourceAllocation
from ..tools import mpitools as _mpit
from ..tools import slicetools as _slct
from ..tools import sharedmemtools as _smt


class DistributableForwardSimulator(_ForwardSimulator):
    """
    Assumes layout is a :class:`DistributableCOPALayout`
    """

    @classmethod
    def _array_types_for_method(cls, method_name):
        # give array types for this method because it's currently used publically in objective function's hessian
        if method_name == '_bulk_hprobs_by_block_singleatom':
            return ('epp', 'epp') + cls._array_types_for_method('_bulk_fill_hprobs_singleatom')
        if method_name == '_bulk_fill_hprobs_singleatom':
            return cls._array_types_for_method('_bulk_fill_probs_block') \
                + cls._array_types_for_method('_bulk_fill_dprobs_block') \
                + cls._array_types_for_method('_bulk_fill_hprobs_block')
        return super()._array_types_for_method(method_name)

    def __init__(self, model=None, num_atoms=None, processor_grid=None, param_blk_sizes=None):
        super().__init__(model)
        self._num_atoms = num_atoms
        self._processor_grid = processor_grid
        self._pblk_sizes = param_blk_sizes
        self._default_distribute_method = "circuits"

    def _set_param_block_size(self, wrt_filter, wrt_block_size, comm):
        if wrt_filter is None:
            blkSize = wrt_block_size  # could be None
            if (comm is not None) and (comm.Get_size() > 1):
                comm_blkSize = self.model.num_params / comm.Get_size()
                blkSize = comm_blkSize if (blkSize is None) \
                    else min(comm_blkSize, blkSize)  # override with smaller comm_blkSize
        else:
            blkSize = None  # wrt_filter dictates block
        return blkSize

    def _bulk_fill_probs(self, array_to_fill, layout, resource_alloc):
        """Note: we expect that array_to_fill points to the memory specifically for this processor
           (a subset of the memory for the host when memory is shared) """
        atom_resource_alloc = resource_alloc.layout_allocs['atom-processing']
        atom_resource_alloc.host_comm_barrier()  # ensure all procs have finished w/shared memory before we reinit

        for atom in layout.atoms:  # layout only holds local atoms
            self._bulk_fill_probs_block(array_to_fill[atom.element_slice], atom, atom_resource_alloc)

        atom_resource_alloc.host_comm_barrier()  # don't exit until all procs' array_to_fill is ready
        # (may need to wait for the host leader to write to this proc's array_to_fill, as _block
        #  functions just ensure the lead proc eventually writes to the memory))

    def _bulk_fill_dprobs(self, array_to_fill, layout, pr_array_to_fill, resource_alloc):
        """Note: we expect that array_to_fill points to the memory specifically for this processor
           (a subset of the memory for the host when memory is shared) """
        blkSize = layout.param_dimension_blk_sizes[0]
        atom_resource_alloc = resource_alloc.layout_allocs['atom-processing']
        param_resource_alloc = resource_alloc.layout_allocs['param-processing']
        
        atom_resource_alloc.host_comm_barrier()  # ensure all procs have finished w/shared memory before we reinit
        # Note: use *largest* host comm that we fill - so 'atom' comm, not 'param' comm

        host_param_slice = None  # layout.host_param_slice  # array_to_fill is already just this slice of the host mem
        global_param_slice = layout.global_param_slice

        for atom in layout.atoms:
            if pr_array_to_fill is not None:
                self._bulk_fill_probs_block(pr_array_to_fill[atom.element_slice], atom, atom_resource_alloc)

            if blkSize is None:  # wrt_filter gives entire computed parameter block
                #Compute all requested derivative columns at once
                self._bulk_fill_dprobs_block(array_to_fill[atom.element_slice, :], host_param_slice, atom,
                                             global_param_slice, param_resource_alloc)

            else:  # Divide columns into blocks of at most blkSize
                Np = _slct.length(global_param_slice)  # total number of parameters we're computing
                nBlks = int(_np.ceil(Np / blkSize))  # num blocks required to achieve desired average size == blkSize
                blocks = _mpit.slice_up_range(Np, nBlks)  # blocks contain indices into final_array[host_param_slice]

                for block in blocks:
                    host_param_slice_part = block # _slct.shift(block, host_param_slice.start)  # into host's memory
                    global_param_slice_part = _slct.shift(block, global_wrtSlice.start)  # actual parameter indices
                    self._bulk_fill_dprobs_block(array_to_fill[atom.element_slice, :], host_param_slice_part, atom,
                                                 global_param_slice_part, param_resource_alloc)

        atom_resource_alloc.host_comm_barrier()  # don't exit until all procs' array_to_fill is ready

    def _bulk_fill_hprobs(self, array_to_fill, layout,
                          pr_array_to_fill, deriv1_array_to_fill, deriv2_array_to_fill,
                          resource_alloc):
        """Note: we expect that array_to_fill points to the memory specifically for this processor
           (a subset of the memory for the host when memory is shared) """
        blkSize1 = layout.param_dimension_blk_sizes[0]
        blkSize2 = layout.param_dimension_blk_sizes[1]

        #Assume we're being called with a resource_alloc that's been setup by a distributed layout:
        atom_resource_alloc = resource_alloc.layout_allocs['atom-processing']
        param_resource_alloc = resource_alloc.layout_allocs['param-processing']
        param2_resource_alloc = resource_alloc.layout_allocs['param2-processing']

        atom_resource_alloc.host_comm_barrier()  # ensure all procs have finished w/shared memory before we reinit
        # Note: use *largest* host comm that we fill - so 'atom' comm, not 'param' comm

        host_param_slice = None # layout.host_param_slice  # array_to_fill is already just this slice of the host mem
        host_param2_slice = None # layout.host_param2_slice  # array_to_fill is already just this slice of the host mem
        global_param_slice = layout.global_param_slice
        global_param2_slice = layout.global_param2_slice

        for atom in layout.atoms:
            self._bulk_fill_hprobs_singleatom(array_to_fill, atom, pr_array_to_fill,
                                              deriv1_array_to_fill, deriv2_array_to_fill,
                                              host_param_slice, host_param2_slice,
                                              global_param_slice, global_param2_slice,
                                              blkSize1, blkSize2, atom_resource_alloc,
                                              param_resource_alloc, param2_resource_alloc)

        atom_resource_alloc.host_comm_barrier()  # don't exit until all procs' array_to_fill is ready

    def _bulk_fill_hprobs_singleatom(self, array_to_fill, atom, pr_array_to_fill,
                                     deriv1_array_to_fill, deriv2_array_to_fill,
                                     dest_slice1, dest_slice2,
                                     wrt_slice1, wrt_slice2,
                                     wrt_blksize1, wrt_blksize2, atom_resource_alloc,
                                     param_resource_alloc, param2_resource_alloc):

        if pr_array_to_fill is not None:
            self._bulk_fill_probs_block(pr_array_to_fill[atom.element_slice], atom, atom_resource_alloc)

        if wrt_blksize1 is None and wrt_blksize2 is None:  # wrt_filter1 & wrt_filter2 dictate block
            #Compute all requested derivative columns at once
            if deriv1_array_to_fill is not None:
                self._bulk_fill_dprobs_block(deriv1_array_to_fill[atom.element_slice, :], dest_slice1, atom,
                                             wrt_slice1, param_resource_alloc)
            if deriv2_array_to_fill is not None:
                if deriv1_array_to_fill is not None and wrt_slice1 == wrt_slice2:
                    deriv2_array_to_fill[atom.element_slice, dest_slice2] = \
                        deriv1_array_to_fill[atom.element_slice, dest_slice1]
                else:
                    self._bulk_fill_dprobs_block(deriv2_array_to_fill[atom.element_slice, :], dest_slice2, atom,
                                                 wrt_slice2, param2_resource_alloc)

            self._bulk_fill_hprobs_block(array_to_fill[atom.element_slice, :, :], dest_slice1, dest_slice2, atom,
                                         wrt_slice1, wrt_slice2, param2_resource_alloc)

        else:  # Divide columns into blocks of at most blkSize
            assert(wrt_slice1 is None and wrt_slice2 is None)  # cannot specify both wrt_slice and wrt_blksize
            Np1 = _slct.length(dest_slice1)
            Np2 = _slct.length(dest_slice2)
            nBlks1 = int(_np.ceil(Np1 / wrt_blksize1))
            nBlks2 = int(_np.ceil(Np2 / wrt_blksize2))
            # num blocks required to achieve desired average size == blkSize1 or blkSize2
            blocks1 = _mpit.slice_up_range(Np1, nBlks1)
            blocks2 = _mpit.slice_up_range(Np2, nBlks2)

            #in this case, where we've just divided the entire range(Np) into blocks, the two deriv mxs
            # will always be the same whenever they're desired (they'll both cover the entire range of params)
            derivArToFill = deriv1_array_to_fill if (deriv1_array_to_fill is not None) else deriv2_array_to_fill

            for block1 in blocks1:
                dest_slice1_part = _slct.shift(block1, dest_slice1.start)  # into host's memory
                wrt_slice1_part = _slct.shift(block1, wrt_slice1.start)  # actual parameter indices

                if deriv1_array_to_fill is not None:
                    self._bulk_fill_dprobs_block(deriv1_array_to_fill[atom.element_slice, :], dest_slice1_part, atom,
                                                 wrt_slice1_part, param_resource_alloc)

                for block2 in blocks2:
                    dest_slice2_part = _slct.shift(block2, dest_slice2.start)  # into host's memory
                    wrt_slice2_part = _slct.shift(block2, wrt_slice2.start)  # actual parameter indices

                    self._bulk_fill_hprobs_block(array_to_fill[atom.element_slice, :],
                                                 dest_slice1_part, dest_slice2_part, atom,
                                                 wrt_slice1_part, wrt_slice2_part, param2_resource_alloc)

            #Fill deriv2_array_to_fill if we need to.
            if deriv2_array_to_fill is not None:
                if deriv1_array_to_fill is not None and wrt_slice1 == wrt_slice2:
                    deriv2_array_to_fill[atom.element_slice, dest_slice2] = \
                        deriv1_array_to_fill[atom.element_slice, dest_slice1]
                else:
                    for block2 in blocks2:
                        dest_slice2_part = _slct.shift(block2, dest_slice2.start)  # into host's memory
                        wrt_slice2_part = _slct.shift(block2, wrt_slice2.start)  # actual parameter indices
                        self._bulk_fill_dprobs_block(deriv2_array_to_fill[atom.element_slice, :], dest_slice2_part,
                                                     atom, wrt_slice2_part, param_resource_alloc)

    def _bulk_hprobs_by_block_singleatom(self, atom, wrt_slices_list, return_dprobs_12, resource_alloc):

        #FUTURE could make a resource_alloc.check_can_allocate_memory call here for ('epp', 'epp')?
        nElements = atom.num_elements
        for wrtSlice1, wrtSlice2 in wrt_slices_list:

            if return_dprobs_12:
                dprobs1, dprobs1_shm = _smt.create_shared_ndarray(resource_alloc, (nElements, _slct.length(wrtSlice1)),
                                                                  'd', zero_out=True)
                dprobs2, dprobs2_shm = _smt.create_shared_ndarray(resource_alloc, (nElements, _slct.length(wrtSlice2)),
                                                                  'd', zero_out=True)
            else:
                dprobs1 = dprobs2 = dprobs1_shm = dprobs2_shm = None

            hprobs, hprobs_shm = _smt.create_shared_ndarray(
                resource_alloc, (nElements, _slct.length(wrtSlice1), _slct.length(wrtSlice2)),
                'd', zero_out=True)

            self._bulk_fill_hprobs_singleatom(hprobs, atom, None, dprobs1, dprobs2, 
                                              None, None, # slice(0, hprobs.shape[1]), slice(0, hprobs.shape[2]),
                                              wrtSlice1, wrtSlice2, None, None, resource_alloc, 
                                              resource_alloc, resource_alloc)
            #Note: we give all three resource_alloc's as our local `resource_alloc` above because all the arrays
            # have been allocated based on just this subset of processors, unlike a call to bulk_fill_hprobs(...)
            # where the probs & dprobs are memory allocated and filled by a larger group of processors.  (the main
            # function of these args is to know which procs work together to fill the *same* values and which of
            # these are on the *same* host so that only one per host actually writes to the assumed-shared memory.

            if return_dprobs_12:
                dprobs12 = dprobs1[:, :, None] * dprobs2[:, None, :]  # (KM,N,1) * (KM,1,N') = (KM,N,N')
                yield wrtSlice1, wrtSlice2, hprobs, dprobs12
            else:
                yield wrtSlice1, wrtSlice2, hprobs

            _smt.cleanup_shared_ndarray(dprobs1_shm)
            _smt.cleanup_shared_ndarray(dprobs2_shm)
            _smt.cleanup_shared_ndarray(hprobs_shm)

    def _bulk_fill_timedep_deriv(self, layout, dataset, ds_circuits, num_total_outcomes,
                                 deriv_array_to_fill, deriv_fill_fn, array_to_fill=None,
                                 fill_fn=None, resource_alloc=None):
        """
        A helper method for computing (filling) the derivative of a time-dependent quantity.

        A generic method providing the scaffolding used when computing (filling) the
        derivative of a time-dependent quantity.  In particular, it distributes the
        computation among the subtrees of `eval_tree` and relies on the caller to supply
        "compute_cache" and "compute_dcache" functions which just need to compute the
        quantitiy being filled and its derivative given a sub-tree and a parameter-slice.

        Parameters
        ----------
        layout : TermCOPALayout
            The layout specifiying the quantities (circuit outcome probabilities) to be
            computed, and related information.

        dataset : DataSet
            the data set passed on to the computation functions.

        ds_circuits : list of Circuits
            the circuits to use as they should be queried from `dataset` (see
            below).  This is typically the same list of circuits used to
            construct `layout` potentially with some aliases applied.

        num_total_outcomes : list or array
            a list of the total number of *possible* outcomes for each circuit
            (so `len(num_total_outcomes) == len(ds_circuits)`).  This is
            needed for handling sparse data, where `dataset` may not contain
            counts for all the possible outcomes of each circuit.

        deriv_array_to_fill : numpy ndarray
            an already-allocated ExM numpy array where E is the total number of
            computed elements (i.e. layout.num_elements) and M is the
            number of model parameters.

        deriv_fill_fn : function
            a function used to compute the objective funtion jacobian.

        array_to_fill : numpy array, optional
            when not None, an already-allocated length-E numpy array that is filled
            with the per-circuit contributions computed using `fn` below.

        fill_fn : function, optional
            a function used to compute the objective function.

        resource_alloc : ResourceAllocation
            Available resources for this computation. Includes the number of processors
            (MPI comm) and memory limit.

        Returns
        -------
        None
        """
        #Note: this function is similar to _bulk_fill_dprobs, and may be able to consolidated in the FUTURE.

        resource_alloc = _ResourceAllocation.cast(resource_alloc)
        assert(resource_alloc.host_comm is None), "Shared memory is not supported in time-dependent calculations (yet)"

        blkSize = layout.additional_dimension_blk_sizes[0]
        atom_resource_alloc = resource_alloc.layout_allocs['atom-processing']
        param_resource_alloc = resource_alloc.layout_allocs['param-processing']

        host_param_slice = layout.host_param_slice
        global_param_slice = layout.global_param_slice

        for atom in layout.atoms:
            elInds = atom.element_slice

            #NOTE: this block uses atom.orig_indices_by_expcircuit, which is specific to _MapCOPALayoutAtom - TODO
            dataset_rows = {i_expanded: dataset[ds_circuits[i]]
                            for i_expanded, i in atom.orig_indices_by_expcircuit.items()}
            num_outcomes = {i_expanded: num_total_outcomes[i]
                            for i_expanded, i in atom.orig_indices_by_expcircuit.items()}

            if array_to_fill is not None:
                fill_fn(array_to_fill, elInds, num_outcomes, atom, dataset_rows, atom_resource_alloc)

            if blkSize is None:  # wrt_filter gives entire computed parameter block
                #Fill derivative cache info
                deriv_fill_fn(deriv_array_to_fill, elInds, host_param_slice, num_outcomes, atom,
                              dataset_rows, global_param_slice, param_resource_alloc)
                #profiler.mem_check("bulk_fill_dprobs: post fill")

            else:  # Divide columns into blocks of at most blkSize
                Np = _slct.length(host_param_slice)  # total number of parameters we're computing
                nBlks = int(_np.ceil(Np / blkSize))
                # num blocks required to achieve desired average size == blkSize
                blocks = _mpit.slice_up_range(Np, nBlks)

                for block in blocks:
                    host_param_slice_part = _slct.shift(block, host_param_slice.start)  # into host's memory
                    global_param_slice_part = _slct.shift(block, global_wrtSlice.start)  # actual parameter indices
                    deriv_fill_fn(deriv_array_to_fill, elInds, host_param_slice_part, num_outcomes, atom,
                                  dataset_rows, global_param_slice_part, param_resource_alloc)
                    #profiler.mem_check("bulk_fill_dprobs: post fill blk")

    def _run_on_atoms(self, layout, fn, resource_alloc):
        """Runs `fn` on all the atoms of `layout`, returning a list of the local (current processor) return values."""
        myAtomIndices, atomOwners, sub_resource_alloc = layout.distribute(resource_alloc)
        local_results = []  # list of the return values just from the atoms run on *this* processor

        for iAtom in myAtomIndices:
            atom = layout.atoms[iAtom]
            local_results.append(fn(atom, sub_resource_alloc))

        return local_results

    def _compute_on_atoms(self, layout, fn, resource_alloc):
        """Similar to _run_on_atoms, but returns a dict mapping atom indices (within layout.atoms) to
           owning-processor ranks (the "owners" dict).  Assumes `fn` returns None.  """
        myAtomIndices, atomOwners, sub_resource_alloc = layout.distribute(resource_alloc)

        for iAtom in myAtomIndices:
            atom = layout.atoms[iAtom]
            fn(atom, sub_resource_alloc)

        return atomOwners

    def _compute_processor_distribution(self, array_types, nprocs, num_params, num_circuits, default_natoms):
        """ Computes commonly needed processor-grid info for distributed layout creation (a helper function)"""
        bNp1Matters = bool("EP" in array_types or "EPP" in array_types or "ep" in array_types or "epp" in array_types)
        bNp2Matters = bool("EPP" in array_types or "epp" in array_types)

        param_dimensions = (num_params,) * (int(bNp1Matters) + int(bNp2Matters))
        param_blk_sizes = (None,) * len(param_dimensions) if (self._pblk_sizes is None) \
                          else self._pblk_sizes[0:len(param_dimensions)]  # automatically set these?

        if self._processor_grid is not None:
            assert(_np.product(self._processor_grid) <= nprocs), "`processor_grid` must multiply to # of procs!"
            na = self._processor_grid[0]
            natoms = max(na, self._num_atoms) if (self._num_atoms is not None) else na
            npp = ()
            if bNp1Matters: npp += (self._processor_grid[1],)
            if bNp2Matters: npp += (self._processor_grid[2],)
        else:
            if self._num_atoms is not None:
                natoms = self._num_atoms
            else:
                natoms = default_natoms
            natoms = min(natoms, num_circuits)  # don't have more atoms than circuits

            pblk = nprocs
            if bNp2Matters: 
                na = _np.gcd(pblk, natoms); pblk //= na
                np1 = _np.gcd(pblk, num_params); pblk //= np1
                np2 = _mpit.closest_divisor(pblk, num_params); pblk //= np2  # last dim: don't demand we divide params evenly
                npp = (np1, np2)
            elif bNp1Matters:
                na = _np.gcd(pblk, natoms); pblk //= na
                np1 = _mpit.closest_divisor(pblk, num_params); pblk //= np1  # last dim: don't demand we divide params evenly
                npp = (np1,)
            else:
                na = _mpit.closest_divisor(pblk, natoms); pblk //= na  # last dim: don't demand we divide atoms evenly
                npp = ()
        return natoms, na, npp, param_dimensions, param_blk_sizes

