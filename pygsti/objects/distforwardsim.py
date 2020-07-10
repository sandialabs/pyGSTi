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


class DistributableForwardSimulator(_ForwardSimulator):
    """
    Assumes layout is a :class:`DistributableCOPALayout`
    """

    def __init__(self, model=None):
        super().__init__(model)
        self._default_distribute_method = "circuits"

    def _set_param_block_size(self, wrt_filter, wrt_block_size, comm):
        if wrt_filter is None:
            blkSize = wrt_block_size  # could be None
            if (comm is not None) and (comm.Get_size() > 1):
                comm_blkSize = self.model.num_params() / comm.Get_size()
                blkSize = comm_blkSize if (blkSize is None) \
                    else min(comm_blkSize, blkSize)  # override with smaller comm_blkSize
        else:
            blkSize = None  # wrt_filter dictates block
        return blkSize

    def _bulk_fill_probs(self, array_to_fill, layout, resource_alloc):
        myAtomIndices, atomOwners, mySubComm = layout.distribute(resource_alloc.comm)
        sub_resource_alloc = _ResourceAllocation(comm=mySubComm)

        for iAtom in myAtomIndices:
            atom = layout.atoms[iAtom]
            self._bulk_fill_probs_block(array_to_fill[atom.element_slice], atom, sub_resource_alloc)

        #collect/gather results
        all_atom_element_slices = [atom.element_slice for atom in layout.atoms]
        _mpit.gather_slices(all_atom_element_slices, atomOwners, array_to_fill, [], 0,
                            resource_alloc.comm, layout.gather_mem_limit)

    def _bulk_fill_dprobs(self, array_to_fill, layout, pr_array_to_fill, resource_alloc, wrt_filter):
        myAtomIndices, atomOwners, mySubComm = layout.distribute(resource_alloc.comm)
        sub_resource_alloc = _ResourceAllocation(comm=mySubComm)
        wrt_block_size = layout.additional_dimension_blk_sizes[0]
        Np = self.model.num_params()

        if wrt_filter is not None:
            assert(wrt_block_size is None)  # Cannot specify both wrt_filter and wrt_block_size
            wrtSlice = _slct.list_to_slice(wrt_filter)  # for now, require the filter specify a slice
        else:
            wrtSlice = None

        for iAtom in myAtomIndices:
            atom = layout.atoms[iAtom]
            if pr_array_to_fill is not None:
                self._bulk_fill_probs_block(pr_array_to_fill[atom.element_slice], atom, sub_resource_alloc)

            #Set wrt_block_size to use available processors if it isn't specified
            blkSize = self._set_param_block_size(wrt_filter, wrt_block_size, mySubComm)

            if blkSize is None:  # wrt_filter gives entire computed parameter block
                #Compute all requested derivative columns at once
                self._bulk_fill_dprobs_block(array_to_fill[atom.element_slice, :], None, atom,
                                             wrtSlice, sub_resource_alloc)

            else:  # Divide columns into blocks of at most blkSize
                assert(wrt_filter is None)  # cannot specify both wrt_filter and blkSize
                nBlks = int(_np.ceil(Np / blkSize))
                # num blocks required to achieve desired average size == blkSize
                blocks = _mpit.slice_up_range(Np, nBlks)

                #distribute derivative computation across blocks
                myBlkIndices, blkOwners, blkComm = \
                    _mpit.distribute_indices(list(range(nBlks)), mySubComm)
                if blkComm is not None:
                    _warnings.warn("Note: more CPUs(%d)" % mySubComm.Get_size()
                                   + " than derivative columns(%d)!" % Np
                                   + " [blkSize = %.1f, nBlks=%d]" % (blkSize, nBlks))  # pragma: no cover
                blk_resource_alloc = _ResourceAllocation(comm=blkComm)

                for iBlk in myBlkIndices:
                    paramSlice = blocks[iBlk]  # specifies which deriv cols calc_and_fill computes
                    self._bulk_fill_dprobs_block(array_to_fill[atom.element_slice, :], paramSlice, atom,
                                                 paramSlice, blk_resource_alloc)

                #gather results
                _mpit.gather_slices(blocks, blkOwners, array_to_fill, [atom.element_slice],
                                    1, mySubComm, layout.gather_mem_limit)
                #note: gathering axis 1 of mx_to_fill[:,fslc], dim=(ks,M)

        #collect/gather results
        all_atom_element_slices = [atom.element_slice for atom in layout.atoms]
        _mpit.gather_slices(all_atom_element_slices, atomOwners, array_to_fill, [], 0,
                            resource_alloc.comm, layout.gather_mem_limit)

        if pr_array_to_fill is not None:
            _mpit.gather_slices(all_atom_element_slices, atomOwners, pr_array_to_fill, [], 0,
                                resource_alloc.comm, layout.gather_mem_limit)
            #note: pass pr_mx_to_fill, dim=(KS,), so gather pr_mx_to_fill[felInds] (axis=0)

    def _bulk_fill_hprobs(self, array_to_fill, layout,
                          pr_array_to_fill, deriv1_array_to_fill, deriv2_array_to_fill,
                          resource_alloc, wrt_filter1, wrt_filter2):
        myAtomIndices, atomOwners, mySubComm = layout.distribute(resource_alloc.comm)
        sub_resource_alloc = _ResourceAllocation(comm=mySubComm)

        if wrt_filter1 is not None:
            assert(layout.additional_dimension_blk_sizes[0] is None)  # Can't specify both wrt_filter and wrt_block_size
            wrtSlice1 = _slct.list_to_slice(wrt_filter1)  # for now, require the filter specify a slice
        else:
            wrtSlice1 = None

        if wrt_filter2 is not None:
            assert(layout.additional_dimension_blk_sizes[1] is None)  # Can't specify both wrt_filter and wrt_block_size
            wrtSlice2 = _slct.list_to_slice(wrt_filter2)  # for now, require the filter specify a slice
        else:
            wrtSlice2 = None

        for iAtom in myAtomIndices:
            atom = layout.atoms[iAtom]

            #Set wrt_block_size to use available processors if it isn't specified
            blkSize1 = self._set_param_block_size(wrt_filter1, layout.additional_dimension_blk_sizes[0], mySubComm)
            blkSize2 = self._set_param_block_size(wrt_filter2, layout.additional_dimension_blk_sizes[1], mySubComm)

            self._bulk_fill_hprobs_singleatom(array_to_fill, atom,
                                              pr_array_to_fill, deriv1_array_to_fill, deriv2_array_to_fill,
                                              sub_resource_alloc, wrtSlice1, wrtSlice2, blkSize1, blkSize2,
                                              layout.gather_mem_limit)

        #collect/gather results
        all_atom_element_slices = [atom.element_slice for atom in layout.atoms]
        _mpit.gather_slices(all_atom_element_slices, atomOwners, array_to_fill, [], 0,
                            resource_alloc.comm, layout.gather_mem_limit)

        if deriv1_array_to_fill is not None:
            _mpit.gather_slices(all_atom_element_slices, atomOwners, deriv1_array_to_fill, [], 0,
                                resource_alloc.comm, layout.gather_mem_limit)
        if deriv2_array_to_fill is not None:
            _mpit.gather_slices(all_atom_element_slices, atomOwners, deriv2_array_to_fill, [], 0,
                                resource_alloc.comm, layout.gather_mem_limit)
        if pr_array_to_fill is not None:
            _mpit.gather_slices(all_atom_element_slices, atomOwners, pr_array_to_fill, [], 0,
                                resource_alloc.comm, layout.gather_mem_limit)

    def _bulk_fill_hprobs_singleatom(self, array_to_fill, atom,
                                     pr_array_to_fill, deriv1_array_to_fill, deriv2_array_to_fill,
                                     resource_alloc, wrt_slice1, wrt_slice2, wrt_blksize1, wrt_blksize2,
                                     gather_mem_limit):
        Np = self.model.num_params()
        if pr_array_to_fill is not None:
            self._bulk_fill_probs_block(pr_array_to_fill[atom.element_slice], atom, resource_alloc)

        if wrt_blksize1 is None and wrt_blksize2 is None:  # wrt_filter1 & wrt_filter2 dictate block
            #Compute all requested derivative columns at once
            if deriv1_array_to_fill is not None:
                self._bulk_fill_dprobs_block(deriv1_array_to_fill[atom.element_slice, :], None, atom,
                                             wrt_slice1, resource_alloc)
            if deriv2_array_to_fill is not None:
                if deriv1_array_to_fill is not None and wrt_slice1 == wrt_slice2:
                    deriv2_array_to_fill[atom.element_slice, :] = deriv1_array_to_fill[atom.element_slice, :]
                else:
                    self._bulk_fill_dprobs_block(deriv2_array_to_fill[atom.element_slice, :], None, atom,
                                                 wrt_slice2, resource_alloc)

            self._bulk_fill_hprobs_block(array_to_fill[atom.element_slice, :, :], None, None, atom,
                                         wrt_slice1, wrt_slice2, resource_alloc)

        else:  # Divide columns into blocks of at most blkSize
            assert(wrt_slice1 is None and wrt_slice2 is None)  # cannot specify both wrt_slice and wrt_blksize
            nBlks1 = int(_np.ceil(Np / wrt_blksize1))
            nBlks2 = int(_np.ceil(Np / wrt_blksize2))
            # num blocks required to achieve desired average size == blkSize1 or blkSize2
            blocks1 = _mpit.slice_up_range(Np, nBlks1)
            blocks2 = _mpit.slice_up_range(Np, nBlks2)

            #distribute derivative computation across blocks
            myBlk1Indices, blk1Owners, blk1Comm = \
                _mpit.distribute_indices(list(range(nBlks1)), resource_alloc.comm)
            blk1_resource_alloc = _ResourceAllocation(comm=blk1Comm)

            myBlk2Indices, blk2Owners, blk2Comm = \
                _mpit.distribute_indices(list(range(nBlks2)), blk1Comm)
            blk2_resource_alloc = _ResourceAllocation(comm=blk2Comm)

            if blk2Comm is not None:
                _warnings.warn("Note: more CPUs(%d)" % resource_alloc.comm.Get_size()
                               + " than hessian elements(%d)!" % (Np**2)
                               + " [blkSize = {%.1f,%.1f}, nBlks={%d,%d}]" % (blkSize1, blkSize2, nBlks1, nBlks2))  # pragma: no cover # noqa

            #in this case, where we've just divided the entire range(Np) into blocks, the two deriv mxs
            # will always be the same whenever they're desired (they'll both cover the entire range of params)
            derivArToFill = deriv1_array_to_fill if (deriv1_array_to_fill is not None) else deriv2_array_to_fill

            for iBlk1 in myBlk1Indices:
                paramSlice1 = blocks1[iBlk1]
                if derivArToFill is not None:
                    self._bulk_fill_dprobs_block(derivArToFill[atom.element_slice, :], paramSlice1, atom,
                                                 paramSlice1, blk1_resource_alloc)

                for iBlk2 in myBlk2Indices:
                    paramSlice2 = blocks2[iBlk2]
                    self._bulk_fill_hprobs_block(array_to_fill[atom.element_slice, :], paramSlice1, paramSlice2,
                                                 atom, paramSlice1, paramSlice2, blk2_resource_alloc)

                #gather column results: gather axis 2 of mx_to_fill[felInds,blocks1[iBlk1]], dim=(ks,blk1,M)
                _mpit.gather_slices(blocks2, blk2Owners, array_to_fill, [atom.element_slice, blocks1[iBlk1]],
                                    2, blk1Comm, gather_mem_limit)

            #gather row results; gather axis 1 of mx_to_fill[felInds], dim=(ks,M,M)
            _mpit.gather_slices(blocks1, blk1Owners, array_to_fill, [atom.element_slice],
                                1, resource_alloc.comm, gather_mem_limit)
            if derivArToFill is not None:
                _mpit.gather_slices(blocks1, blk1Owners, derivArToFill, [atom.element_slice],
                                    1, resource_alloc.comm, gather_mem_limit)

            #in this case, where we've just divided the entire range(Np) into blocks, the two deriv mxs
            # will always be the same whenever they're desired (they'll both cover the entire range of params)
            if deriv1_array_to_fill is not None:
                deriv1_array_to_fill[atom.element_slice, :] = derivArToFill[atom.element_slice, :]
            if deriv2_array_to_fill is not None:
                deriv2_array_to_fill[atom.element_slice, :] = derivArToFill[atom.element_slice, :]

    def _bulk_hprobs_by_block_singleatom(self, atom, wrt_slices_list, return_dprobs_12, resource_alloc,
                                         gather_mem_limit):

        nElements = atom.num_elements
        for wrtSlice1, wrtSlice2 in wrt_slices_list:

            if return_dprobs_12:
                dprobs1 = _np.zeros((nElements, _slct.length(wrtSlice1)), 'd')
                dprobs2 = _np.zeros((nElements, _slct.length(wrtSlice2)), 'd')
            else:
                dprobs1 = dprobs2 = None
            hprobs = _np.zeros((nElements, _slct.length(wrtSlice1),
                                _slct.length(wrtSlice2)), 'd')
            
            self._bulk_fill_hprobs_singleatom(hprobs, atom, None, dprobs1, dprobs2, resource_alloc,
                                              wrtSlice1, wrtSlice2, None, None, gather_mem_limit)

            if return_dprobs_12:
                dprobs12 = dprobs1[:, :, None] * dprobs2[:, None, :]  # (KM,N,1) * (KM,1,N') = (KM,N,N')
                yield wrtSlice1, wrtSlice2, hprobs, dprobs12
            else:
                yield wrtSlice1, wrtSlice2, hprobs

    def _bulk_fill_timedep_deriv(self, layout, dataset, ds_circuits, num_total_outcomes,
                                 deriv_array_to_fill, deriv_fill_fn, array_to_fill=None, fill_fn=None,
                                 wrt_filter=None, resource_alloc=None):
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

        wrt_filter : list of ints, optional
            If not None, a list of integers specifying which parameters
            to include in the derivative dimension. This argument is used
            internally for distributing calculations across multiple
            processors and to control memory usage.

        resource_alloc : ResourceAllocation
            Available resources for this computation. Includes the number of processors
            (MPI comm) and memory limit.

        Returns
        -------
        None
        """
        #Note: this function is similar to _bulk_fill_dprobs, and may be able to consolidated in the FUTURE.

        resource_alloc = _ResourceAllocation.cast(resource_alloc)
        myAtomIndices, atomOwners, mySubComm = layout.distribute(resource_alloc.comm)
        #sub_resource_alloc = _ResourceAllocation(comm=mySubComm)  # FUTURE: pass this to *_fn instead of mySubComm?
        wrt_block_size = layout.additional_dimension_blk_sizes[0]
        Np = self.model.num_params()

        if wrt_filter is not None:
            assert(wrt_block_size is None)  # Cannot specify both wrt_filter and wrt_block_size
            wrtSlice = _slct.list_to_slice(wrt_filter)  # for now, require the filter specify a slice
        else:
            wrtSlice = None

        for iAtom in myAtomIndices:
            atom = layout.atoms[iAtom]
            elInds = atom.element_slice

            #NOTE: this block uses atom.orig_indices_by_expcircuit, which is specific to _MapCOPALayoutAtom - TODO
            dataset_rows = {i_expanded: dataset[ds_circuits[i]]
                            for i_expanded, i in atom.orig_indices_by_expcircuit.items()}
            num_outcomes = {i_expanded: num_total_outcomes[i]
                            for i_expanded, i in atom.orig_indices_by_expcircuit.items()}

            if array_to_fill is not None:
                fill_fn(array_to_fill, elInds, num_outcomes, atom, dataset_rows, mySubComm)

            #Set wrt_block_size to use available processors if it isn't specified
            blkSize = self._set_param_block_size(wrt_filter, wrt_block_size, mySubComm)

            if blkSize is None:  # wrt_filter gives entire computed parameter block
                #Fill derivative cache info
                deriv_fill_fn(deriv_array_to_fill, elInds, None, num_outcomes, atom,
                              dataset_rows, wrtSlice, mySubComm)
                #profiler.mem_check("bulk_fill_dprobs: post fill")

            else:  # Divide columns into blocks of at most blkSize
                assert(wrt_filter is None)  # cannot specify both wrt_filter and blkSize
                nBlks = int(_np.ceil(Np / blkSize))
                # num blocks required to achieve desired average size == blkSize
                blocks = _mpit.slice_up_range(Np, nBlks)

                #distribute derivative computation across blocks
                myBlkIndices, blkOwners, blkComm = \
                    _mpit.distribute_indices(list(range(nBlks)), mySubComm)
                if blkComm is not None:
                    _warnings.warn("Note: more CPUs(%d)" % mySubComm.Get_size()
                                   + " than derivative columns(%d)!" % Np
                                   + " [blkSize = %.1f, nBlks=%d]" % (blkSize, nBlks))  # pragma: no cover

                for iBlk in myBlkIndices:
                    paramSlice = blocks[iBlk]  # specifies which deriv cols calc_and_fill computes
                    deriv_fill_fn(deriv_array_to_fill, elInds, paramSlice, num_outcomes, atom,
                                  dataset_rows, paramSlice, blkComm)
                    #profiler.mem_check("bulk_fill_dprobs: post fill blk")

                #gather results
                _mpit.gather_slices(blocks, blkOwners, deriv_array_to_fill, [elInds],
                                    1, mySubComm, layout.gather_mem_limit)
                #note: gathering axis 1 of deriv_mx_to_fill[:,fslc], dim=(ks,M)
                #profiler.mem_check("bulk_fill_dprobs: post gather blocks")

        #collect/gather results
        all_atom_element_slices = [atom.element_slice for atom in layout.atoms]
        _mpit.gather_slices(all_atom_element_slices, atomOwners, deriv_array_to_fill, [], 0,
                            resource_alloc.comm, layout.gather_mem_limit)
        #note: pass deriv_mx_to_fill, dim=(KS,M), so gather deriv_mx_to_fill[felInds] (axis=0)

        if array_to_fill is not None:
            _mpit.gather_slices(all_atom_element_slices, atomOwners, array_to_fill, [], 0,
                                resource_alloc.comm, layout.gather_mem_limit)
            #note: pass mx_to_fill, dim=(KS,), so gather mx_to_fill[felInds] (axis=0)

        #profiler.mem_check("bulk_fill_timedep_dchi2: post gather subtrees")
        #
        #profiler.add_time("bulk_fill_timedep_dchi2: total", tStart)
        #profiler.add_count("bulk_fill_timedep_dchi2 count")
        #profiler.mem_check("bulk_fill_timedep_dchi2: end")

    def _run_on_atoms(self, layout, fn, resource_alloc):
        """Runs `fn` on all the atoms of `layout`, returning a list of the local (current processor) return values."""
        myAtomIndices, atomOwners, mySubComm = layout.distribute(resource_alloc.comm)
        sub_resource_alloc = _ResourceAllocation(comm=mySubComm)
        local_results = []  # list of the return values just from the atoms run on *this* processor

        for iAtom in myAtomIndices:
            atom = layout.atoms[iAtom]
            local_results.append(fn(atom, sub_resource_alloc))

        return local_results

    def _compute_on_atoms(self, layout, fn, resource_alloc):
        """Similar to _run_on_atoms, but returns a dict mapping atom indices (within layout.atoms) to
           owning-processor ranks (the "owners" dict).  Assumes `fn` returns None.  """
        myAtomIndices, atomOwners, mySubComm = layout.distribute(resource_alloc.comm)
        sub_resource_alloc = _ResourceAllocation(comm=mySubComm)

        for iAtom in myAtomIndices:
            atom = layout.atoms[iAtom]
            fn(atom, sub_resource_alloc)

        return atomOwners
