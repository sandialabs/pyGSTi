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

from pygsti.forwardsims.forwardsim import ForwardSimulator as _ForwardSimulator
from pygsti.forwardsims.forwardsim import _array_type_parameter_dimension_letters
from pygsti.tools import mpitools as _mpit
from pygsti.tools import slicetools as _slct
from pygsti.tools import sharedmemtools as _smt


class DistributableForwardSimulator(_ForwardSimulator):
    """
    A base class for forward simulators that use distributed COPA layouts.

    This class contains implements the methods of :class:`ForwardSimulator` assuming that the
    layout is a :class:`DistributableCOPALayout` object, and leaves a set of a simpler methods
    for derived classes to implement.

    In particular, because a distributed layout divides computations by assigning segments of
    the full element- and parameter-dimensions to individual processors, derived classes just
    implement the `_bulk_fill_*probs_atom` methods which compute a single section of the entire
    output array, and don't need to worry about dealing with the distribution in element and
    parameter directions.

    Parameters
    ----------
    model : Model, optional
        The parent model of this simulator.  It's fine if this is `None` at first,
        but it will need to be set (by assigning `self.model` before using this simulator.

    num_atoms : int, optional
        The number of atoms to use when creating a layout (i.e. when calling :method:`create_layout`).
        This determines how many units the element (circuit outcome probability) dimension is divided
        into, and doesn't have to correclate with the number of processors.  When multiple processors
        are used, if `num_atoms` is less than the number of processors it should divide the number of
        processors evenly, so that `num_atoms // num_procs` groups of processors can be used to divide
        the computation over parameter dimensions.

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
        # give array types for this method because it's currently used publically in objective function's hessian
        if method_name == '_iter_atom_hprobs_by_rectangle':
            return ('abb', 'abb') + cls._array_types_for_method('_bulk_fill_hprobs_dprobs_atom')
        if method_name == '_bulk_fill_hprobs_dprobs_atom':
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

    def _bulk_fill_probs(self, array_to_fill, layout):
        """Note: we expect that array_to_fill points to the memory specifically for this processor
           (a subset of the memory for the host when memory is shared) """
        atom_resource_alloc = layout.resource_alloc('atom-processing')
        atom_resource_alloc.host_comm_barrier()  # ensure all procs have finished w/shared memory before we reinit

        for atom in layout.atoms:  # layout only holds local atoms
            self._bulk_fill_probs_atom(array_to_fill[atom.element_slice], atom, atom_resource_alloc)

        atom_resource_alloc.host_comm_barrier()  # don't exit until all procs' array_to_fill is ready
        # (may need to wait for the host leader to write to this proc's array_to_fill, as _block
        #  functions just ensure the lead proc eventually writes to the memory))

    def _bulk_fill_probs_atom(self, array_to_fill, layout_atom, resource_alloc):
        # if atom can be converted to a (sub)-layout, then we can just use machinery of base
        # class (note: layouts hold their own resource-alloc, atom's don't)
        self._bulk_fill_probs_block(array_to_fill, layout_atom.as_layout(resource_alloc))

    def _bulk_fill_dprobs(self, array_to_fill, layout, pr_array_to_fill):
        """Note: we expect that array_to_fill points to the memory specifically for this processor
           (a subset of the memory for the host when memory is shared) """
        blkSize = layout.param_dimension_blk_sizes[0]
        atom_resource_alloc = layout.resource_alloc('atom-processing')
        param_resource_alloc = layout.resource_alloc('param-processing')

        atom_resource_alloc.host_comm_barrier()  # ensure all procs have finished w/shared memory before we reinit
        # Note: use *largest* host comm that we fill - so 'atom' comm, not 'param' comm

        host_param_slice = None  # layout.host_param_slice  # array_to_fill is already just this slice of the host mem
        global_param_slice = layout.global_param_slice

        for atom in layout.atoms:
            #assert(_slct.length(atom.element_slice) == atom.num_elements)  # for debugging
            #print("DEBUG: Atom %d of %d slice=%s" % (iDB, len(layout.atoms), str(atom.element_slice)))

            if pr_array_to_fill is not None:
                self._bulk_fill_probs_atom(pr_array_to_fill[atom.element_slice], atom, atom_resource_alloc)

            if blkSize is None:  # avoid unnecessary slice_up_range and block loop logic in 'else' block
                #Compute all of our derivative columns at once
                self._bulk_fill_dprobs_atom(array_to_fill[atom.element_slice, :], host_param_slice, atom,
                                            global_param_slice, param_resource_alloc)

            else:  # Divide columns into blocks of at most blkSize
                Np = _slct.length(global_param_slice)  # total number of parameters we're computing
                nBlks = int(_np.ceil(Np / blkSize))  # num blocks required to achieve desired average size == blkSize
                blocks = _mpit.slice_up_range(Np, nBlks)  # blocks contain indices into final_array[host_param_slice]

                for block in blocks:
                    host_param_slice_part = block  # _slct.shift(block, host_param_slice.start)  # into host's memory
                    global_param_slice_part = _slct.shift(block, global_param_slice.start)  # actual parameter indices
                    self._bulk_fill_dprobs_atom(array_to_fill[atom.element_slice, :], host_param_slice_part, atom,
                                                global_param_slice_part, param_resource_alloc)

        atom_resource_alloc.host_comm_barrier()  # don't exit until all procs' array_to_fill is ready

    def _bulk_fill_dprobs_atom(self, array_to_fill, dest_param_slice, layout_atom, param_slice, resource_alloc):
        # if atom can be converted to a (sub)-layout, then we can just use machinery of base
        # class (note: layouts hold their own resource-alloc, atom's don't)
        self._bulk_fill_dprobs_block(array_to_fill, dest_param_slice,
                                     layout_atom.as_layout(resource_alloc), param_slice)

    def _bulk_fill_hprobs(self, array_to_fill, layout,
                          pr_array_to_fill, deriv1_array_to_fill, deriv2_array_to_fill):
        """Note: we expect that array_to_fill points to the memory specifically for this processor
           (a subset of the memory for the host when memory is shared) """
        blkSize1 = layout.param_dimension_blk_sizes[0]
        blkSize2 = layout.param_dimension_blk_sizes[1]

        #Assume we're being called with a resource_alloc that's been setup by a distributed layout:
        atom_resource_alloc = layout.resource_alloc('atom-processing')
        param_resource_alloc = layout.resource_alloc('param-processing')
        param2_resource_alloc = layout.resource_alloc('param2-processing')

        atom_resource_alloc.host_comm_barrier()  # ensure all procs have finished w/shared memory before we reinit
        # Note: use *largest* host comm that we fill - so 'atom' comm, not 'param' comm

        host_param_slice = None  # layout.host_param_slice  # array_to_fill is already just this slice of the host mem
        host_param2_slice = None  # layout.host_param2_slice  # array_to_fill is already just this slice of the host mem
        global_param_slice = layout.global_param_slice
        global_param2_slice = layout.global_param2_slice

        for atom in layout.atoms:

            if pr_array_to_fill is not None:
                self._bulk_fill_probs_atom(pr_array_to_fill[atom.element_slice], atom, atom_resource_alloc)

            if blkSize1 is None and blkSize2 is None:  # run 'else' block without unnecessary logic
                #Compute all our derivative columns at once
                if deriv1_array_to_fill is not None:
                    self._bulk_fill_dprobs_atom(deriv1_array_to_fill[atom.element_slice, :], host_param_slice,
                                                atom, global_param_slice, param_resource_alloc)
                if deriv2_array_to_fill is not None:
                    if deriv1_array_to_fill is not None and global_param_slice == global_param2_slice:
                        deriv2_array_to_fill[atom.element_slice, :] = deriv1_array_to_fill[atom.element_slice, :]
                    else:
                        self._bulk_fill_dprobs_atom(deriv2_array_to_fill[atom.element_slice, :], host_param2_slice,
                                                    atom, global_param2_slice, param2_resource_alloc)

                self._bulk_fill_hprobs_atom(array_to_fill[atom.element_slice, :, :], host_param_slice,
                                            host_param2_slice, atom, global_param_slice, global_param2_slice,
                                            param2_resource_alloc)

            else:  # Divide columns into blocks of at most shape (blkSize1, blkSize2)
                assert(blkSize1 is not None and blkSize2 is not None), \
                    "Both (or neither) of the Hessian block sizes must be specified!"
                Np1 = _slct.length(global_param_slice)
                Np2 = _slct.length(global_param2_slice)
                nBlks1 = int(_np.ceil(Np1 / blkSize1))
                nBlks2 = int(_np.ceil(Np2 / blkSize2))
                # num blocks required to achieve desired average size == blkSize1 or blkSize2
                blocks1 = _mpit.slice_up_range(Np1, nBlks1)
                blocks2 = _mpit.slice_up_range(Np2, nBlks2)

                for block1 in blocks1:
                    host_param_slice_part = block1  # _slct.shift(block1, host_param_slice.start)  # into host's memory
                    global_param_slice_part = _slct.shift(block1, global_param_slice.start)  # actual parameter indices

                    if deriv1_array_to_fill is not None:
                        self._bulk_fill_dprobs_atom(deriv1_array_to_fill[atom.element_slice, :], host_param_slice_part,
                                                    atom, global_param_slice_part, param_resource_alloc)

                    for block2 in blocks2:
                        host_param2_slice_part = block2  # into host's memory
                        global_param2_slice_part = _slct.shift(block2, global_param2_slice.start)  # parameter indices
                        self._bulk_fill_hprobs_atom(array_to_fill[atom.element_slice, :],
                                                    host_param_slice_part, host_param2_slice_part, atom,
                                                    global_param_slice_part, global_param2_slice_part,
                                                    param2_resource_alloc)

                #Fill deriv2_array_to_fill if we need to.
                if deriv2_array_to_fill is not None:
                    if deriv1_array_to_fill is not None and global_param_slice == global_param2_slice:
                        deriv2_array_to_fill[atom.element_slice, :] = deriv1_array_to_fill[atom.element_slice, :]
                    else:
                        for block2 in blocks2:
                            host_param2_slice_part = block2  # into host's memory
                            global_param2_slice_part = _slct.shift(block2, global_param2_slice.start)  # param indices
                            self._bulk_fill_dprobs_atom(deriv2_array_to_fill[atom.element_slice, :],
                                                        host_param2_slice_part, atom,
                                                        global_param2_slice_part, param_resource_alloc)

        atom_resource_alloc.host_comm_barrier()  # don't exit until all procs' array_to_fill is ready

    def _bulk_fill_hprobs_atom(self, array_to_fill, dest_param_slice1, dest_param_slice2, layout_atom,
                               param_slice1, param_slice2, resource_alloc):
        # if atom can be converted to a (sub)-layout, then we can just use machinery of base
        # class (note: layouts hold their own resource-alloc, atom's don't)
        self._bulk_fill_hprobs_block(array_to_fill, dest_param_slice1, dest_param_slice2,
                                     layout_atom.as_layout(resource_alloc), param_slice1, param_slice2)

    def _bulk_fill_hprobs_dprobs_atom(self, array_to_fill, deriv1_array_to_fill, deriv2_array_to_fill, atom,
                                      param_slice1, param_slice2, resource_alloc):
        #Note: this function can be called similarly to _bulk_fill_hprobs_atom or _bulk_fill_dprobs_atom
        # in that array_to_fill is assumed to already be sized to the atom's elements, i.e.
        # we provide array_to_fill and not array_to_fill[atom.alement_slice,...] when calling subroutines.

        host_param_slice1 = host_param_slice2 = None  # array_to_fill is already just this slice of the host mem
        if deriv1_array_to_fill is not None:
            self._bulk_fill_dprobs_atom(deriv1_array_to_fill, host_param_slice1, atom,
                                        param_slice1, resource_alloc)
        if deriv2_array_to_fill is not None:
            if deriv1_array_to_fill is not None and param_slice1 == param_slice2:
                deriv2_array_to_fill[:, :] = deriv1_array_to_fill[:, :]
            else:
                self._bulk_fill_dprobs_atom(deriv2_array_to_fill, host_param_slice2, atom,
                                            param_slice2, resource_alloc)

        self._bulk_fill_hprobs_atom(array_to_fill, host_param_slice1, host_param_slice2, atom,
                                    param_slice1, param_slice2, resource_alloc)

    def _iter_hprobs_by_rectangle(self, layout, wrt_slices_list, return_dprobs_12):
        # Just needed for compatibility - so base `iter_hprobs_by_rectangle` knows to loop over atoms
        # Similar to _iter_atom_hprobs_by_rectangle but runs over all atoms before yielding and
        #  yielded array has leading dim == # of local elements instead of just 1 atom's # elements.
        nElements = layout.num_elements
        resource_alloc = layout.resource_alloc()
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

            for atom in layout.atoms:
                self._bulk_fill_hprobs_dprobs_atom(hprobs[atom.element_slice, :, :],
                                                   dprobs1[atom.element_slice, :] if (dprobs1 is not None) else None,
                                                   dprobs2[atom.element_slice, :] if (dprobs2 is not None) else None,
                                                   atom, wrtSlice1, wrtSlice2, resource_alloc)
            #Note: we give resource_alloc as our local `resource_alloc` above because all the arrays
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

    def _iter_atom_hprobs_by_rectangle(self, atom, wrt_slices_list, return_dprobs_12, resource_alloc):

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

            # Note: no need to index w/ [atom.element_slice,...] (compare with _iter_hprobs_by_rectangles)
            # since these arrays are already sized to this particular atom (not to all the host's atoms)
            self._bulk_fill_hprobs_dprobs_atom(hprobs, dprobs1, dprobs2, atom,
                                               wrtSlice1, wrtSlice2, resource_alloc)
            #Note: we give resource_alloc as our local `resource_alloc` above because all the arrays
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
                                 fill_fn=None):
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

        Returns
        -------
        None
        """
        #Note: this function is similar to _bulk_fill_dprobs, and may be able to consolidated in the FUTURE.

        blkSize = layout.param_dimension_blk_sizes[0]
        atom_resource_alloc = layout.resource_alloc('atom-processing')
        param_resource_alloc = layout.resource_alloc('param-processing')

        assert(atom_resource_alloc.host_comm is None), \
            "Shared memory is not supported in time-dependent calculations (yet)"

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
                    global_param_slice_part = _slct.shift(block, global_param_slice.start)  # actual parameter indices
                    deriv_fill_fn(deriv_array_to_fill, elInds, host_param_slice_part, num_outcomes, atom,
                                  dataset_rows, global_param_slice_part, param_resource_alloc)
                    #profiler.mem_check("bulk_fill_dprobs: post fill blk")

    def _run_on_atoms(self, layout, fn, resource_alloc):
        """Runs `fn` on all the atoms of `layout`, returning a list of the local (current processor) return values."""
        local_results = []  # list of the return values just from the atoms run on *this* processor

        for atom in layout.atoms:
            local_results.append(fn(atom, resource_alloc))

        return local_results

    def _compute_processor_distribution(self, array_types, nprocs, num_params, num_circuits, default_natoms):
        """ Computes commonly needed processor-grid info for distributed layout creation (a helper function)"""
        parameter_dim_letters = _array_type_parameter_dimension_letters()
        param_dim_cnts = [sum([array_type.count(l) for l in parameter_dim_letters]) for array_type in array_types]
        max_param_dims = max(param_dim_cnts) if len(param_dim_cnts) > 0 else 0

        bNp1Matters = bool(max_param_dims > 0)
        bNp2Matters = bool(max_param_dims > 1)

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
                np2 = _mpit.closest_divisor(pblk, num_params); pblk //= np2  # last dim: don't demand we divide evenly
                npp = (np1, np2)
            elif bNp1Matters:
                na = _np.gcd(pblk, natoms); pblk //= na
                np1 = _mpit.closest_divisor(pblk, num_params); pblk //= np1  # last dim: don't demand we divide evenly
                npp = (np1,)
            else:
                na = _mpit.closest_divisor(pblk, natoms); pblk //= na  # last dim: don't demand we divide atoms evenly
                npp = ()
        return natoms, na, npp, param_dimensions, param_blk_sizes
