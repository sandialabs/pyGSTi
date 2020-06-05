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

    def __init__(self, model):
        super().__init__(model)
        self._default_distribute_method = "circuits"

    def _set_param_block_size(self, wrt_filter, wrt_block_size, comm):
        if wrt_filter is None:
            blkSize = wrt_block_size  # could be None
            if (comm is not None) and (comm.Get_size() > 1):
                comm_blkSize = self.Np / comm.Get_size()
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
                nBlks = int(_np.ceil(self.Np / blkSize))
                # num blocks required to achieve desired average size == blkSize
                blocks = _mpit.slice_up_range(self.Np, nBlks)

                #distribute derivative computation across blocks
                myBlkIndices, blkOwners, blkComm = \
                    _mpit.distribute_indices(list(range(nBlks)), mySubComm)
                if blkComm is not None:
                    _warnings.warn("Note: more CPUs(%d)" % mySubComm.Get_size()
                                   + " than derivative columns(%d)!" % self.Np
                                   + " [blkSize = %.1f, nBlks=%d]" % (blkSize, nBlks))  # pragma: no cover

                for iBlk in myBlkIndices:
                    paramSlice = blocks[iBlk]  # specifies which deriv cols calc_and_fill computes
                    self._bulk_fill_dprobs_block(array_to_fill[atom.element_slice, :], paramSlice, atom,
                                                 paramSlice, sub_resource_alloc)

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
            if pr_array_to_fill is not None:
                self._bulk_fill_probs_block(pr_array_to_fill[atom.element_slice], atom, sub_resource_alloc)

            #Set wrt_block_size to use available processors if it isn't specified
            blkSize1 = self._set_param_block_size(wrt_filter1, layout.additional_dimension_blk_sizes[0], mySubComm)
            blkSize2 = self._set_param_block_size(wrt_filter2, layout.additional_dimension_blk_sizes[1], mySubComm)

            if blkSize1 is None and blkSize2 is None:  # wrt_filter1 & wrt_filter2 dictate block
                #Compute all requested derivative columns at once
                if deriv1_array_to_fill is not None:
                    self._bulk_fill_dprobs_block(deriv1_array_to_fill[atom.element_slice, :], None, atom,
                                                 wrtSlice1, sub_resource_alloc)
                if deriv2_array_to_fill is not None:
                    if deriv1_array_to_fill is not None and wrtSlice1 == wrtSlice2:
                        deriv2_array_to_fill[atom.element_slice, :] = deriv1_array_to_fill[atom.element_slice, :]
                    else:
                        self._bulk_fill_dprobs_block(deriv2_array_to_fill[atom.element_slice, :], None, atom,
                                                     wrtSlice2, sub_resource_alloc)

                self._bulk_fill_hprobs_block(array_to_fill[atom.element_slice, :, :], None, None, atom,
                                             wrtSlice1, wrtSlice2, sub_resource_alloc)

            else:  # Divide columns into blocks of at most blkSize
                assert(wrt_filter1 is None and wrt_filter2 is None)  # cannot specify both wrt_filter and blkSize
                nBlks1 = int(_np.ceil(self.Np / blkSize1))
                nBlks2 = int(_np.ceil(self.Np / blkSize2))
                # num blocks required to achieve desired average size == blkSize1 or blkSize2
                blocks1 = _mpit.slice_up_range(self.Np, nBlks1)
                blocks2 = _mpit.slice_up_range(self.Np, nBlks2)

                #distribute derivative computation across blocks
                myBlk1Indices, blk1Owners, blk1Comm = \
                    _mpit.distribute_indices(list(range(nBlks1)), mySubComm)
                blk1_resource_alloc = _ResourceAllocation(comm=blk1Comm)

                myBlk2Indices, blk2Owners, blk2Comm = \
                    _mpit.distribute_indices(list(range(nBlks2)), blk1Comm)
                blk2_resource_alloc = _ResourceAllocation(comm=blk2Comm)

                if blk2Comm is not None:
                    _warnings.warn("Note: more CPUs(%d)" % mySubComm.Get_size()
                                   + " than hessian elements(%d)!" % (self.Np**2)
                                   + " [blkSize = {%.1f,%.1f}, nBlks={%d,%d}]" % (blkSize1, blkSize2, nBlks1, nBlks2))  # pragma: no cover # noqa

                #in this case, where we've just divided the entire range(self.Np) into blocks, the two deriv mxs
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
                                        2, blk1Comm, layout.gather_mem_limit)

                #gather row results; gather axis 1 of mx_to_fill[felInds], dim=(ks,M,M)
                _mpit.gather_slices(blocks1, blk1Owners, array_to_fill, [atom.element_slice],
                                    1, mySubComm, layout.gather_mem_limit)
                if derivArToFill is not None:
                    _mpit.gather_slices(blocks1, blk1Owners, derivArToFill, [atom.element_slice],
                                        1, mySubComm, layout.gather_mem_limit)

                #in this case, where we've just divided the entire range(self.Np) into blocks, the two deriv mxs
                # will always be the same whenever they're desired (they'll both cover the entire range of params)
                if deriv1_array_to_fill is not None:
                    deriv1_array_to_fill[atom.element_slice, :] = derivArToFill[atom.element_slice, :]
                if deriv2_array_to_fill is not None:
                    deriv2_array_to_fill[atom.element_slice, :] = derivArToFill[atom.element_slice, :]

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
