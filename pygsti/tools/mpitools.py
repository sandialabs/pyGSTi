"""
Functions for working with MPI processor distributions
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import itertools as _itertools
import warnings as _warnings

import numpy as _np

from pygsti.baseobjs import _compatibility as _compat
from pygsti.tools import sharedmemtools as _smt
from pygsti.tools import slicetools as _slct
from pygsti.tools.matrixtools import _fas, _findx, _findx_shape
from pygsti.tools.matrixtools import prime_factors as _prime_factors


def distribute_indices(indices, comm, allow_split_comm=True):
    """
    Partition an array of indices (any type) evenly among `comm`'s processors.

    Parameters
    ----------
    indices : list
        An array of items (any type) which are to be partitioned.

    comm : mpi4py.MPI.Comm or ResourceAllocation
        The communicator which specifies the number of processors and
        which may be split into returned sub-communicators.  If a
        :class:`ResourceAllocation` object, node information is also
        taken into account when available (for shared memory compatibility).

    allow_split_comm : bool
        If True, when there are more processors than indices,
        multiple processors will be given the *same* set of local
        indices and `comm` will be split into sub-communicators,
        one for each group of processors that are given the same
        indices.  If False, then "extra" processors are simply given
        nothing to do, i.e. empty lists of local indices.

    Returns
    -------
    loc_indices : list
        A list containing the elements of `indices` belonging to the current
        processor.
    owners : dict
        A dictionary mapping the elements of `indices` to integer ranks, such
        that `owners[el]` gives the rank of the processor responsible for
        communicating that element's results to the other processors.  Note that
        in the case when `allow_split_comm=True` and multiple procesors have
        computed the results for a given element, only a single (the first)
        processor rank "owns" the element, and is thus responsible for sharing
        the results.  This notion of ownership is useful when gathering the
        results.
    loc_comm : mpi4py.MPI.Comm or ResourceAllocation or None
        The local communicator for the group of processors which have been
        given the same `loc_indices` to compute, obtained by splitting `comm`.
        If `loc_indices` is unique to the current processor, or if
        `allow_split_comm` is False, None is returned.
    """
    from ..baseobjs.resourceallocation import ResourceAllocation as _ResourceAllocation
    if isinstance(comm, _ResourceAllocation):
        ralloc = comm
        comm = ralloc.comm
    else:
        ralloc = None

    if comm is None:
        nprocs, rank = 1, 0
    else:
        nprocs = comm.Get_size()
        rank = comm.Get_rank()

    loc_indices, owners, _ = distribute_indices_base(indices, nprocs, rank,
                                                     allow_split_comm)

    #Split comm into sub-comms when there are more procs than
    # indices, resulting in all procs getting only a
    # single index and multiple procs getting the *same*
    # (single) index.
    #if nprocs > 1 and len(indices)==1 and (comm is not None) and allow_split_comm:
    #    loc_comm = comm #split is unnecessary
    #el
    if nprocs > len(indices) and (comm is not None) and allow_split_comm:
        color = loc_indices[0] if isinstance(loc_indices[0], int) \
            else (int(hash(loc_indices[0])) >> 32)  # mpi4py only allows 32-bit ints
        loc_comm = comm.Split(color=color, key=rank)
    else:
        loc_comm = None

    if ralloc is not None:  # then return a ResourceAllocation instead of a comm
        loc_comm = _ResourceAllocation(loc_comm, ralloc.mem_limit, ralloc.profiler,
                                       ralloc.distribute_method, ralloc.allocated_memory)
        if ralloc.host_comm is not None:
            loc_comm.build_hostcomms()  # signals that we want to use shared intra-host memory

    return loc_indices, owners, loc_comm


def distribute_indices_base(indices, nprocs, rank, allow_split_comm=True):
    """
    Partition an array of "indices" evenly among a given number of "processors"

    This function is similar to :func:`distribute_indices`, but allows for more
    a more generalized notion of what a "processor" is, since the number of
    processors and rank are given independently and do not have to be
    associated with an MPI comm.  Note also that `indices` can be an arbitrary
    list of items, making this function very general.

    Parameters
    ----------
    indices : list
        An array of items (any type) which are to be partitioned.

    nprocs : int
        The number of "processors" to distribute the elements of
        `indices` among.

    rank : int
        The rank of the current "processor" (must be an integer
        between 0 and `nprocs-1`).  Note that this value is not
        obtained from any MPI communicator.

    allow_split_comm : bool
        If True, when there are more processors than indices,
        multiple processors will be given the *same* set of local
        indices.  If False, then extra processors are simply given
        nothing to do, i.e. empty lists of local indices.

    Returns
    -------
    loc_indices : list
        A list containing the elements of `indices` belonging to the current
        processor (i.e. the one specified by `rank`).
    owners : dict
        A dictionary mapping the elements of `indices` to integer ranks, such
        that `owners[el]` gives the rank of the processor responsible for
        communicating that element's results to the other processors.  Note that
        in the case when `allow_split_comm=True` and multiple procesors have
        computed the results for a given element, only a single (the first)
        processor rank "owns" the element, and is thus responsible for sharing
        the results.  This notion of ownership is useful when gathering the
        results.
    """
    nIndices = len(indices)
    if nIndices == 0:  # special case when == 0
        return [], {}, ()

    if nprocs >= nIndices:
        if allow_split_comm:
            nloc_std = nprocs // nIndices  # this many processors per index, w/out any "extra"
            extra = nprocs - nloc_std * nIndices  # extra procs
            # indices 0 to extra-1 get (nloc_std+1) processors each
            # indices extra to nIndices-1 get nloc_std processors each
            if rank < extra * (nloc_std + 1):
                k = rank // (nloc_std + 1)
                loc_indices = [indices[k]]
                peer_ranks = tuple(range(k * (nloc_std + 1), (k + 1) * (nloc_std + 1)))
            else:
                k = (rank - extra * (nloc_std + 1)) // nloc_std
                loc_indices = [indices[extra + k]]
                peer_ranks = tuple(range(extra * (nloc_std + 1) + k * nloc_std,
                                         extra * (nloc_std + 1) + (k + 1) * nloc_std))

            # owners dict gives rank of first (chief) processor for each index
            # (the "owner" of a given index is responsible for communicating
            #  results for that index to the other processors)
            owners = {indices[i]: i * (nloc_std + 1) for i in range(extra)}
            owners.update({indices[i]: extra * (nloc_std + 1) + (i - extra) * nloc_std
                           for i in range(extra, nIndices)})
        else:
            #Not allowed to assign multiple procs the same local index
            # (presumably b/c there is no way to sub-divide the work
            #  performed for a single index among multiple procs)
            if rank < nIndices:
                loc_indices = [indices[rank]]
            else:
                loc_indices = []  # extra procs do nothing
            owners = {indices[i]: i for i in range(nIndices)}
            peer_ranks = ()

    else:
        nloc_std = nIndices // nprocs
        extra = nIndices - nloc_std * nprocs  # extra indices
        # so assign (nloc_std+1) indices to first extra procs
        if rank < extra:
            nloc = nloc_std + 1
            nstart = rank * (nloc_std + 1)
        else:
            nloc = nloc_std
            nstart = extra * (nloc_std + 1) + (rank - extra) * nloc_std
        loc_indices = [indices[i] for i in range(nstart, nstart + nloc)]

        owners = {}  # which rank "owns" each index
        for r in range(extra):
            nstart = r * (nloc_std + 1)
            for i in range(nstart, nstart + (nloc_std + 1)):
                owners[indices[i]] = r
        for r in range(extra, nprocs):
            nstart = extra * (nloc_std + 1) + (r - extra) * nloc_std
            for i in range(nstart, nstart + nloc_std):
                owners[indices[i]] = r
        peer_ranks = ()

    return loc_indices, owners, peer_ranks


def slice_up_slice(slc, num_slices):
    """
    Divides up `slc` into `num_slices` slices.

    Parameters
    ----------
    slc : slice
        The slice to be divided.

    num_slices : int
        The number of slices to divide the range into.

    Returns
    -------
    list of slices
    """
    assert(slc.step is None)  # currently, no support for step != None slices
    if slc.start is None or slc.stop is None:
        return slice_up_range(0, num_slices)
    else:
        return slice_up_range(slc.stop - slc.start, num_slices, slc.start)


def slice_up_range(n, num_slices, start=0):
    """
    Divides up `range(start,start+n)` into `num_slices` slices.

    Parameters
    ----------
    n : int
        The number of (consecutive) indices in the range to be divided.

    num_slices : int
        The number of slices to divide the range into.

    start : int, optional
        The starting entry of the range, so that the range to be
        divided is `range(start,start+n)`.

    Returns
    -------
    list of slices
    """
    base = n // num_slices  # base slice size
    m1 = n - base * num_slices  # num base+1 size slices
    m2 = num_slices - m1     # num base size slices
    assert(((base + 1) * m1 + base * m2) == n)

    off = start
    ret = [slice(off + (base + 1) * i, off + (base + 1) * (i + 1)) for i in range(m1)]
    off += (base + 1) * m1
    ret += [slice(off + base * i, off + base * (i + 1)) for i in range(m2)]
    assert(len(ret) == num_slices)
    return ret


def distribute_slice(s, comm, allow_split_comm=True):
    """
    Partition a continuous slice evenly among `comm`'s processors.

    This function is similar to :func:`distribute_indices`, but
    is specific to the case when the indices being distributed
    are a consecutive set of integers (specified by a slice).

    Parameters
    ----------
    s : slice
        The slice to be partitioned.

    comm : mpi4py.MPI.Comm or ResourceAllocation
        The communicator which specifies the number of processors and
        which may be split into returned sub-communicators.  If a
        :class:`ResourceAllocation` object, node information is also
        taken into account when available (for shared memory compatibility).

    allow_split_comm : bool
        If True, when there are more processors than slice indices,
        multiple processors will be given the *same* local slice
        and `comm` will be split into sub-communicators, one for each
        group of processors that are given the same local slice.
        If False, then "extra" processors are simply given
        nothing to do, i.e. an empty local slice.

    Returns
    -------
    slices : list of slices
        The list of *unique* slices assigned to different processors.  It's
        possible that a single slice (i.e. element of `slices`) is assigned
        to multiple processors (when there are more processors than indices
        in `s`.
    loc_slice : slice
        A slice specifying the indices belonging to the current processor.
    owners : dict
        A dictionary giving the owning rank of each slice.  Values are integer
        ranks and keys are integers into `slices`, specifying which slice.
    loc_comm : mpi4py.MPI.Comm or ResourceAllocation or None
        The local communicator/ResourceAllocation for the group of processors
        which have been given the same `loc_slice` to compute, obtained by
        splitting `comm`.  If `loc_slice` is unique to the current processor,
        or if `allow_split_comm` is False, None is returned.
    """
    from ..baseobjs.resourceallocation import ResourceAllocation as _ResourceAllocation
    if isinstance(comm, _ResourceAllocation):
        ralloc = comm
        comm = ralloc.comm
    else:
        ralloc = None

    if comm is None:
        nprocs, rank = 1, 0
    else:
        nprocs = comm.Get_size()
        rank = comm.Get_rank()

    slices = slice_up_slice(s, min(nprocs, _slct.length(s)))
    assert(len(slices) <= nprocs)
    loc_iSlices, slcOwners, _ = \
        distribute_indices_base(list(range(len(slices))), nprocs, rank,
                                allow_split_comm)
    assert(len(loc_iSlices) <= 1)  # should not assign more than one slice to
    # each proc by design (there are only nprocs slices)

    if len(loc_iSlices) == 1:
        loc_slice = slices[loc_iSlices[0]]

        #Split comm into sub-comms when there are more procs than
        # indices, resulting in all procs getting only a
        # single index and multiple procs getting the *same*
        # (single) index.
        if nprocs > _slct.length(s) and (comm is not None) and allow_split_comm:
            loc_comm = comm.Split(color=loc_iSlices[0], key=rank)
        else:
            loc_comm = None
    else:  # len(loc_iSlices) == 0 (nothing for this proc to do)
        loc_slice = slice(0, 0)
        loc_comm = None

    if ralloc is not None:  # then return a ResourceAllocation instead of a comm
        loc_comm = _ResourceAllocation(loc_comm, ralloc.mem_limit, ralloc.profiler,
                                       ralloc.distribute_method, ralloc.allocated_memory)
        if ralloc.host_comm is not None:
            loc_comm.build_hostcomms()  # signals that we want to use shared intra-host memory

    return slices, loc_slice, slcOwners, loc_comm


def gather_slices(slices, slice_owners, ar_to_fill,
                  ar_to_fill_inds, axes, comm, max_buffer_size=None):
    """
    Gathers data within a numpy array, `ar_to_fill`, according to given slices.

    Upon entry it is assumed that the different processors within `comm` have
    computed different parts of `ar_to_fill`, namely different slices of the
    `axis`-th axis.  At exit, data has been gathered such that all processors
    have the results for the entire `ar_to_fill` (or at least for all the slices
    given).

    Parameters
    ----------
    slices : list
        A list of all the slices (computed by *any* of the processors, not
        just the current one).  Each element of `slices` may be either a
        single slice or a tuple of slices (when gathering across multiple
        dimensions).

    slice_owners : dict
        A dictionary mapping the index of a slice (or tuple of slices)
        within `slices` to an integer rank of the processor responsible
        for communicating that slice's data to the rest of the processors.

    ar_to_fill : numpy.ndarray
        The array which contains partial data upon entry and the gathered
        data upon exit.

    ar_to_fill_inds : list
        A list of slice or index-arrays specifying the (fixed) sub-array of
        `ar_to_fill` that should be gathered into.  The elements of
        `ar_to_fill_inds` are taken to be indices for the leading dimension
        first, and any unspecified dimensions or `None` elements are
        assumed to be unrestricted (as if `slice(None,None)`).  Note that
        the combination of `ar_to_fill` and `ar_to_fill_inds` is essentally like
        passing `ar_to_fill[ar_to_fill_inds]` to this function, except it will
        work with index arrays as well as slices.

    axes : int or tuple of ints
        The axis or axes of `ar_to_fill` on which the slices apply (which axis
        do the slices in `slices` refer to?).  Note that `len(axes)` must
        be equal to the number of slices (i.e. the tuple length) of each
        element of `slices`.

    comm : mpi4py.MPI.Comm or ResourceAllocation or None
        The communicator specifying the processors involved and used
        to perform the gather operation.  If a :class:`ResourceAllocation`
        is provided, then inter-host communication is used when available
        to facilitate use of shared intra-host memory.

    max_buffer_size : int or None
        The maximum buffer size in bytes that is allowed to be used
        for gathering data.  If None, there is no limit.

    Returns
    -------
    None
    """
    from ..baseobjs.resourceallocation import ResourceAllocation as _ResourceAllocation
    if isinstance(comm, _ResourceAllocation):
        ralloc = comm
        comm = ralloc.comm

        #For use with shared intra-host (intra-node) memory:
        # my_interhost_ranks = ranks of comm, 1 per host, that this processor uses to send/receive data between hosts
        # broadcast_comm = the comm of my_interhost_ranks used to send/receive data.
        if ralloc.interhost_ranks is not None:
            my_interhost_ranks = set(ralloc.interhost_ranks)
            broadcast_rank_map = {comm_rank: broadcast_comm_rank
                                  for broadcast_comm_rank, comm_rank in enumerate(ralloc.interhost_ranks)}
            broadcast_comm = ralloc.interhost_comm
        else:
            my_interhost_ranks = None
            broadcast_rank_map = {i: i for i in range(comm.Get_size())} if (comm is not None) else {0: 0}  # trivial map
            broadcast_comm = comm
    else:
        ralloc = None
        my_interhost_ranks = None
        broadcast_rank_map = {i: i for i in range(comm.Get_size())} if (comm is not None) else {0: 0}  # trivial map
        broadcast_comm = comm

    if comm is None: return  # no gathering needed!

    # To be safe, since use of broadcast_comm below means we don't always need to wait for all procs
    # to finish what they were doing last, which could involve updating a shared ar_to_fill so that
    # values accessed by the already-finished front-running processors are affected!
    comm.barrier()

    #Perform broadcasts for each slice in order
    my_rank = comm.Get_rank()

    axes = (axes,) if _compat.isint(axes) else axes

    #print("DB: Rank %d (%d): BEGIN GATHER SLICES: interhost=%s, group=%s" %
    #      (my_rank, broadcast_comm.rank, str(my_interhost_ranks), str(broadcast_comm.Get_group())))

    # # if ar_to_fill_inds only contains slices (or is empty), then we can slice ar_to_fill once up front
    # # and not use generic arIndx in loop below (slower, especially with lots of procs)
    # if all([isinstance(indx, slice) for indx in ar_to_fill_inds]):
    #     ar_to_fill = ar_to_fill[tuple(ar_to_fill_inds)]  # Note: this *doesn't* reduce its .ndim
    #     ar_to_fill_inds = ()  # now ar_to_fill requires no further indexing

    arIndx = [slice(None, None)] * ar_to_fill.ndim
    arIndx[0:len(ar_to_fill_inds)] = ar_to_fill_inds
    max_indices = [None] * len(axes)
    if max_buffer_size is not None:  # no maximum of buffer size
        chunkBytes = ar_to_fill.nbytes  # start with the entire array as the "chunk"
        for iaxis, axis in enumerate(axes):
            # Consider restricting the chunk size along the iaxis-th axis.
            #  If we can achieve the desired max_buffer_size by restricting
            #  just along this axis, great.  Otherwise, restrict to at most
            #  1 index along this axis and keep going.
            bytes_per_index = chunkBytes / ar_to_fill.shape[axis]
            max_inds = int(max_buffer_size / bytes_per_index)
            if max_inds == 0:
                max_indices[iaxis] = 1
                chunkBytes /= ar_to_fill.shape[axis]
            else:
                max_indices[iaxis] = max_inds
                break
        else:
            _warnings.warn("gather_slices: Could not achieve max_buffer_size")

    # NOTE: Tried doing something faster (Allgatherv) when slices elements are simple slices (not tuples of slices).
    # This ultimately showed that our repeated use of Bcast isn't any slower than fewer calls to Allgatherv,
    # and since the Allgatherv case complicates the code and ignores the memory limit, it's best to just drop it.

    # Broadcast slices one-by-one (slower, but more general):
    for iSlice, slcOrSlcTup in enumerate(slices):
        owner = slice_owners[iSlice]  # owner's rank
        if my_interhost_ranks is not None and owner not in my_interhost_ranks:
            # if the "source" (owner) of the data isn't a part of my "circle" of ranks, then we
            # don't need to send or receive this data - other ranks on the same hosts will do it.
            continue

        slcTup = (slcOrSlcTup,) if isinstance(slcOrSlcTup, slice) else slcOrSlcTup
        assert(len(slcTup) == len(axes))

        #Get the a list of the (sub-)slices along each axis, whose product
        # (along the specified axes) gives the entire block given by slcTup
        axisSlices = []
        for iaxis, axis in enumerate(axes):
            slc = slcTup[iaxis]
            if max_indices[iaxis] is None or max_indices[iaxis] >= _slct.length(slc):
                axisSlices.append([slc])  # arIndx[axis] = slc
            else:
                axisSlices.append(_slct.divide(slc, max_indices[iaxis]))

        for axSlcs in _itertools.product(*axisSlices):
            #create arIndx from per-axis (sub-)slices and broadcast
            for iaxis, axis in enumerate(axes):
                arIndx[axis] = axSlcs[iaxis]

            #broadcast arIndx slice
            buf = _findx(ar_to_fill, arIndx, True) if (my_rank == owner) \
                else _np.empty(_findx_shape(ar_to_fill, arIndx), ar_to_fill.dtype)
            if my_interhost_ranks is None or len(my_interhost_ranks) > 1:
                #print("DB: Rank %d (%d) Broadcast: arIndx = %s, owner=%d root=%d" %
                #      (my_rank, broadcast_comm.rank, str(arIndx), owner, broadcast_rank_map[owner]))
                broadcast_comm.Bcast(buf, root=broadcast_rank_map[owner])
                if my_rank != owner: _fas(ar_to_fill, arIndx, buf)
            buf = None  # free buffer mem asap
    #print("DB: Rank %d: END GATHER SLICES" % my_rank)

    # Important: wait for everything to finish before proceeding
    #  (when broadcast_comm != comm some procs may run ahead - see comment above)
    comm.barrier()


def gather_slices_by_owner(current_slices, ar_to_fill, ar_to_fill_inds,
                           axes, comm, max_buffer_size=None):
    """
    Gathers data within a numpy array, `ar_to_fill`, according to given slices.

    Upon entry it is assumed that the different processors within `comm` have
    computed different parts of `ar_to_fill`, namely different slices of the
    axes indexed by `axes`. At exit, data has been gathered such that all processors
    have the results for the entire `ar_to_fill` (or at least for all the slices
    given).

    Parameters
    ----------
    current_slices : list
        A list of all the slices computed by the *current* processor.
        Each element of `slices` may be either a single slice or a
        tuple of slices (when gathering across multiple dimensions).

    ar_to_fill : numpy.ndarray
        The array which contains partial data upon entry and the gathered
        data upon exit.

    ar_to_fill_inds : list
        A list of slice or index-arrays specifying the (fixed) sub-array of
        `ar_to_fill` that should be gathered into.  The elements of
        `ar_to_fill_inds` are taken to be indices for the leading dimension
        first, and any unspecified dimensions or `None` elements are
        assumed to be unrestricted (as if `slice(None,None)`).  Note that
        the combination of `ar_to_fill` and `ar_to_fill_inds` is essentally like
        passing `ar_to_fill[ar_to_fill_inds]` to this function, except it will
        work with index arrays as well as slices.

    axes : int or tuple of ints
        The axis or axes of `ar_to_fill` on which the slices apply (which axis
        do the slices in `slices` refer to?).  Note that `len(axes)` must
        be equal to the number of slices (i.e. the tuple length) of each
        element of `slices`.

    comm : mpi4py.MPI.Comm or None
        The communicator specifying the processors involved and used
        to perform the gather operation.

    max_buffer_size : int or None
        The maximum buffer size in bytes that is allowed to be used
        for gathering data.  If None, there is no limit.

    Returns
    -------
    None
    """

    #Note: same beginning as gather_slices (TODO: consolidate?)
    if comm is None: return  # no gathering needed!

    #Perform broadcasts for each slice in order
    my_rank = comm.Get_rank()
    arIndx = [slice(None, None)] * ar_to_fill.ndim
    arIndx[0:len(ar_to_fill_inds)] = ar_to_fill_inds

    axes = (axes,) if _compat.isint(axes) else axes

    max_indices = [None] * len(axes)
    if max_buffer_size is not None:  # no maximum of buffer size
        chunkBytes = ar_to_fill.nbytes  # start with the entire array as the "chunk"
        for iaxis, axis in enumerate(axes):
            # Consider restricting the chunk size along the iaxis-th axis.
            #  If we can achieve the desired max_buffer_size by restricting
            #  just along this axis, great.  Otherwise, restrict to at most
            #  1 index along this axis and keep going.
            bytes_per_index = chunkBytes / ar_to_fill.shape[axis]
            max_inds = int(max_buffer_size / bytes_per_index)
            if max_inds == 0:
                max_indices[iaxis] = 1
                chunkBytes /= ar_to_fill.shape[axis]
            else:
                max_indices[iaxis] = max_inds
                break
        else:
            _warnings.warn("gather_slices_by_owner: Could not achieve max_buffer_size")
    # -- end part that is the same as gather_slices

    #Get a list of the slices to broadcast, indexed by the rank of the owner proc
    slices_by_owner = comm.allgather(current_slices)
    for owner, slices in enumerate(slices_by_owner):
        for slcOrSlcTup in slices:
            slcTup = (slcOrSlcTup,) if isinstance(slcOrSlcTup, slice) else slcOrSlcTup
            assert(len(slcTup) == len(axes))

            #Get the a list of the (sub-)slices along each axis, whose product
            # (along the specified axes) gives the entire block given by slcTup
            axisSlices = []
            for iaxis, axis in enumerate(axes):
                slc = slcTup[iaxis]
                if max_indices[iaxis] is None or max_indices[iaxis] >= _slct.length(slc):
                    axisSlices.append([slc])  # arIndx[axis] = slc
                else:
                    axisSlices.append(_slct.divide(slc, max_indices[iaxis]))

            for axSlcs in _itertools.product(*axisSlices):
                #create arIndx from per-axis (sub-)slices and broadcast
                for iaxis, axis in enumerate(axes):
                    arIndx[axis] = axSlcs[iaxis]

                #broadcast arIndx slice
                buf = _findx(ar_to_fill, arIndx, True) if (my_rank == owner) \
                    else _np.empty(_findx_shape(ar_to_fill, arIndx), ar_to_fill.dtype)
                comm.Bcast(buf, root=owner)
                if my_rank != owner: _fas(ar_to_fill, arIndx, buf)
                buf = None  # free buffer mem asap


def gather_indices(indices, index_owners, ar_to_fill, ar_to_fill_inds,
                   axes, comm, max_buffer_size=None):
    """
    Gathers data within a numpy array, `ar_to_fill`, according to given indices.

    Upon entry it is assumed that the different processors within `comm` have
    computed different parts of `ar_to_fill`, namely different slices or
    index-arrays of the `axis`-th axis.  At exit, data has been gathered such
    that all processors have the results for the entire `ar_to_fill` (or at least
    for all the indices given).

    Parameters
    ----------
    indices : list
        A list of all the integer-arrays or slices (computed by *any* of
        the processors, not just the current one).  Each element of `indices`
        may be either a single slice/index-array or a tuple of such
        elements (when gathering across multiple dimensions).

    index_owners : dict
        A dictionary mapping the index of an element within `slices` to an
        integer rank of the processor responsible for communicating that
        slice/index-array's data to the rest of the processors.

    ar_to_fill : numpy.ndarray
        The array which contains partial data upon entry and the gathered
        data upon exit.

    ar_to_fill_inds : list
        A list of slice or index-arrays specifying the (fixed) sub-array of
        `ar_to_fill` that should be gathered into.  The elements of
        `ar_to_fill_inds` are taken to be indices for the leading dimension
        first, and any unspecified dimensions or `None` elements are
        assumed to be unrestricted (as if `slice(None,None)`).  Note that
        the combination of `ar_to_fill` and `ar_to_fill_inds` is essentally like
        passing `ar_to_fill[ar_to_fill_inds]` to this function, except it will
        work with index arrays as well as slices.

    axes : int or tuple of ints
        The axis or axes of `ar_to_fill` on which the slices apply (which axis
        do the elements of `indices` refer to?).  Note that `len(axes)` must
        be equal to the number of sub-indices (i.e. the tuple length) of each
        element of `indices`.

    comm : mpi4py.MPI.Comm or None
        The communicator specifying the processors involved and used
        to perform the gather operation.

    max_buffer_size : int or None
        The maximum buffer size in bytes that is allowed to be used
        for gathering data.  If None, there is no limit.

    Returns
    -------
    None
    """
    if comm is None: return  # no gathering needed!

    #Perform broadcasts for each slice in order
    my_rank = comm.Get_rank()
    arIndx = [slice(None, None)] * ar_to_fill.ndim
    arIndx[0:len(ar_to_fill_inds)] = ar_to_fill_inds

    axes = (axes,) if _compat.isint(axes) else axes

    max_indices = [None] * len(axes)
    if max_buffer_size is not None:  # no maximum of buffer size
        chunkBytes = ar_to_fill.nbytes  # start with the entire array as the "chunk"
        for iaxis, axis in enumerate(axes):
            # Consider restricting the chunk size along the iaxis-th axis.
            #  If we can achieve the desired max_buffer_size by restricting
            #  just along this axis, great.  Otherwise, restrict to at most
            #  1 index along this axis and keep going.
            bytes_per_index = chunkBytes / ar_to_fill.shape[axis]
            max_inds = int(max_buffer_size / bytes_per_index)
            if max_inds == 0:
                max_indices[iaxis] = 1
                chunkBytes /= ar_to_fill.shape[axis]
            else:
                max_indices[iaxis] = max_inds
                break
        else:
            _warnings.warn("gather_indices: Could not achieve max_buffer_size")

    for iIndex, indOrIndTup in enumerate(indices):
        owner = index_owners[iIndex]  # owner's rank
        indTup = (indOrIndTup,) if not isinstance(indOrIndTup, tuple) else indOrIndTup
        assert(len(indTup) == len(axes))

        def to_slice_list(index_array_or_slice):
            """Breaks a slice or index array into a list of slices"""
            if isinstance(index_array_or_slice, slice):
                return [index_array_or_slice]  # easy!

            lst = index_array_or_slice
            if len(lst) == 0: return [slice(0, 0)]

            slc_lst = []
            i = 0; N = len(lst)
            while i < N:
                start = lst[i]
                step = lst[i + 1] - lst[i] if i + 1 < N else None
                while i + 1 < N and lst[i + 1] - lst[i] == step: i += 1
                stop = lst[i] + 1
                slc_lst.append(slice(start, stop, None if step == 1 else step))
                i += 1

            return slc_lst

        #Get the a list of the (sub-)indices along each axis, whose product
        # (along the specified axes) gives the entire block given by slcTup
        axisSlices = []
        for iaxis, axis in enumerate(axes):
            ind = indTup[iaxis]
            sub_slices = []

            #break `ind`, which may be either a single slice or an index array,
            # into a list of slices that are broadcast one at a time (sometimes
            # these `ind_slice` slices themselves need to be broken up further
            # to obey max_buffer_size).
            for islice in to_slice_list(ind):
                if max_indices[iaxis] is None or max_indices[iaxis] >= _slct.length(islice):
                    sub_slices.append(islice)  # arIndx[axis] = slc
                else:
                    sub_slices.extend(_slct.divide(islice, max_indices[iaxis]))
            axisSlices.append(sub_slices)

        for axSlcs in _itertools.product(*axisSlices):
            #create arIndx from per-axis (sub-)slices and broadcast
            for iaxis, axis in enumerate(axes):
                arIndx[axis] = axSlcs[iaxis]

            #broadcast arIndx slice
            buf = _findx(ar_to_fill, arIndx, True) if (my_rank == owner) \
                else _np.empty(_findx_shape(ar_to_fill, arIndx), ar_to_fill.dtype)
            comm.Bcast(buf, root=owner)
            if my_rank != owner: _fas(ar_to_fill, arIndx, buf)
            buf = None  # free buffer mem asap


def distribute_for_dot(a_shape, b_shape, comm):
    """
    Prepares for one or muliple distributed dot products given the dimensions to be dotted.

    The returned values should be passed as `loc_slices` to :func:`mpidot`.

    Parameters
    ----------
    a_shape, b_shape : tuple
        The shapes of the arrays that will be dotted together in ensuing :func:`mpidot`
        calls (see above).

    comm : mpi4py.MPI.Comm or ResourceAllocation or None
        The communicator used to perform the distribution.

    Returns
    -------
    row_slice, col_slice : slice
        The "local" row slice of "A" and column slice of "B" belonging to the current
        processor, which computes result[row slice, col slice].  These should be passed
        to :func:`mpidot`.
    slice_tuples_by_rank : list
        A list of the `(row_slice, col_slice)` owned by each processor, ordered by rank.
        If a `ResourceAllocation` is given that utilizes shared memory, then this list is
        for the ranks in this processor's inter-host communication group.  This should
        be passed as the `slice_tuples_by_rank` argument of :func:`mpidot`.
    """
    from ..baseobjs.resourceallocation import ResourceAllocation as _ResourceAllocation
    if isinstance(comm, _ResourceAllocation):
        ralloc = comm; comm = ralloc.comm
    else:
        ralloc = None

    if comm is None:  # then the single proc owns all rows & cols
        loc_row_slice = slice(0, a_shape[0])
        loc_col_slice = slice(0, b_shape[1])

    elif b_shape[1] > comm.size:  # then there are enough procs to just distribute B's cols
        loc_row_slice = slice(0, a_shape[0])
        _, loc_col_slice, _, _ = distribute_slice(
            slice(0, b_shape[1]), comm, False)  # local B-column range as a slice

    else:  # below should work even when comm.size > a_shape[0] * b_shape[1]
        #distribute rows
        _, loc_row_slice, _, loc_row_comm = distribute_slice(
            slice(0, a_shape[0]), comm, True)  # local A-row range as a slice

        #distribute columns among procs in sub-comm
        _, loc_col_slice, _, _ = distribute_slice(
            slice(0, b_shape[1]), loc_row_comm, False)  # local B-column range as a slice

    if ralloc is None:
        broadcast_comm = comm  # the comm used to communicate results within mpidot(...)
    else:
        broadcast_comm = comm if (ralloc.interhost_comm is None) else ralloc.interhost_comm

    if broadcast_comm is not None:
        slice_tuples_by_rank = broadcast_comm.allgather((loc_row_slice, loc_col_slice))
    else:
        slice_tuples_by_rank = None

    return loc_row_slice, loc_col_slice, slice_tuples_by_rank


def mpidot(a, b, loc_row_slice, loc_col_slice, slice_tuples_by_rank, comm,
           out=None, out_shm=None):
    """
    Performs a distributed dot product, dot(a,b).

    Parameters
    ----------
    a : numpy.ndarray
        First array to dot together.

    b : numpy.ndarray
        Second array to dot together.

    loc_row_slice, loc_col_slice : slice
        Specify the row or column indices, respectively, of the
        resulting dot product that are computed by this processor (the
        rows of `a` and columns of `b` that are used). Obtained from
        :func:`distribute_for_dot`.

    slice_tuples_by_rank : list
        A list of (row_slice, col_slice) tuples, one per processor within this
        processors broadcast group, ordered by rank.  Provided by :func:`distribute_for_dot`.

    comm : mpi4py.MPI.Comm or ResourceAllocation or None
        The communicator used to parallelize the dot product.  If a
        :class:`ResourceAllocation` object is given, then a shared
        memory result will be returned when appropriate.

    out : numpy.ndarray, optional
        If not None, the array to use for the result.  This should be the
        same type of array (size, and whether it's shared or not) as this
        function would have created if `out` were `None`.

    out_shm : multiprocessing.shared_memory.SharedMemory, optinal
        The shared memory object corresponding to `out` when it uses
        shared memory.

    Returns
    -------
    result : numpy.ndarray
        The resulting array
    shm : multiprocessing.shared_memory.SharedMemory
        A shared memory object needed to cleanup the shared memory.  If
        a normal array is created, this is `None`.  Provide this to
        :func:`cleanup_shared_ndarray` to ensure `ar` is deallocated properly.
    """
    # R_ij = sum_k A_ik * B_kj
    from ..baseobjs.resourceallocation import ResourceAllocation as _ResourceAllocation
    if isinstance(comm, _ResourceAllocation):
        ralloc = comm
        comm = ralloc.comm
    else:
        ralloc = None

    if comm is None or comm.Get_size() == 1:
        return _np.dot(a, b), None

    if out is None:
        if ralloc is None:
            result, result_shm = _np.zeros((a.shape[0], b.shape[1]), a.dtype), None
        else:
            result, result_shm = _smt.create_shared_ndarray(ralloc, (a.shape[0], b.shape[1]), a.dtype,
                                                            zero_out=True)
    else:
        result = out
        result_shm = out_shm

    rshape = (_slct.length(loc_row_slice), _slct.length(loc_col_slice))
    loc_result_flat = _np.empty(rshape[0] * rshape[1], a.dtype)
    loc_result = loc_result_flat.view(); loc_result.shape = rshape
    loc_result[:, :] = _np.dot(a[loc_row_slice, :], b[:, loc_col_slice])

    # broadcast_com defines the group of processors this processor communicates with.
    # Without shared memory, this is *all* the other processors.  With shared memory, this
    # is one processor on each host.  This code is identical to that in distribute_for_dot.
    if ralloc is None:
        broadcast_comm = comm
    else:
        broadcast_comm = comm if (ralloc.interhost_comm is None) else ralloc.interhost_comm

    comm.barrier()  # wait for all ranks to do their work (get their loc_result)
    for r, (cur_row_slice, cur_col_slice) in enumerate(slice_tuples_by_rank):
        # for each member of the group that will communicate results
        cur_shape = (_slct.length(cur_row_slice), _slct.length(cur_col_slice))
        buf = loc_result_flat if (broadcast_comm.rank == r) else _np.empty(cur_shape[0] * cur_shape[1], a.dtype)
        broadcast_comm.Bcast(buf, root=r)
        if broadcast_comm.rank != r: buf.shape = cur_shape
        else: buf = loc_result  # already of correct shape
        result[cur_row_slice, cur_col_slice] = buf
    comm.barrier()  # wait for all ranks to finish writing to result

    #assert(_np.linalg.norm(_np.dot(a,b) - result)/(_np.linalg.norm(result) + result.size) < 1e-6),\
    #    "DEBUG: %g, %g, %d" % (_np.linalg.norm(_np.dot(a,b) - result), _np.linalg.norm(result), result.size)
    return result, result_shm


def parallel_apply(f, l, comm):
    """
    Apply a function, f to every element of a list, l in parallel, using MPI.

    Parameters
    ----------
    f : function
        function of an item in the list l

    l : list
        list of items as arguments to f

    comm : MPI Comm
        MPI communicator object for organizing parallel programs

    Returns
    -------
    results : list
        list of items after f has been applied
    """
    locArgs, _, locComm = distribute_indices(l, comm)
    if locComm is None or locComm.Get_rank() == 0:  # only first proc in local comm group
        locResults = [f(arg) for arg in locArgs]  # needs to do anything
    else: locResults = []
    results = comm.allgather(locResults)  # Certain there is a better way to do this (see above)
    results = list(_itertools.chain.from_iterable(results))  # list-of-lists -> single list
    return results


def mpi4py_comm():
    """
    Get a comm object

    Returns
    -------
    MPI.Comm
        Comm object to be passed down to parallel pygsti routines
    """
    from mpi4py import MPI  # not at top so can import pygsti on cluster login nodes
    return MPI.COMM_WORLD


def sum_across_procs(x, comm):
    """
    Sum a value across all processors in `comm`.

    Parameters
    ----------
    x : object
        Local value - the current processor's contrubution to the sum.

    comm : mpi4py.MPI.Comm
        MPI communicator

    Returns
    -------
    object
        Of the same type as the `x` objects that were summed.
    """
    if comm is not None:
        from mpi4py import MPI  # not at top so can import pygsti on cluster login nodes
        return comm.allreduce(x, MPI.SUM)
    else:
        return x


def processor_group_size(nprocs, number_of_tasks):
    """
    Find the number of groups to divide `nprocs` processors into to tackle `number_of_tasks` tasks.

    When `number_of_tasks` > `nprocs` the smallest integer multiple of `nprocs` is returned that
    equals or exceeds `number_of_tasks` is returned.

    When `number_of_tasks` < `nprocs` the smallest divisor of `nprocs` that equals or exceeds
    `number_of_tasks` is returned.

    Parameters
    ----------
    nprocs : int
        The number of processors to divide into groups.

    number_of_tasks : int or float
        The number of tasks to perform, which can also be seen as the *desired* number of
        processor groups.  If a floating point value is given the next highest integer is
        used.

    Returns
    -------
    int
    """
    if number_of_tasks >= nprocs:
        return nprocs * int(_np.ceil(1. * number_of_tasks / nprocs))
    else:
        fctrs = sorted(_prime_factors(nprocs)); i = 1
        if int(_np.ceil(number_of_tasks)) in fctrs:
            return int(_np.ceil(number_of_tasks))  # we got lucky
        while _np.prod(fctrs[0:i]) < number_of_tasks: i += 1
        return _np.prod(fctrs[0:i])


def sum_arrays(local_array, owners, comm):
    """
    Sums arrays across all "owner" processors.

    Parameters
    ----------
    local_array : numpy.ndarray
        The array contributed by this processor.  This array will be *zeroed out*
        on processors whose ranks are not in `owners`.

    owners : list or set
        The ranks whose contributions should be summed.  These
        are the ranks of the processors that "own" the responsibility to
        communicate their local array to the rest of the processors.

    comm : mpi4py.MPI.Comm
        MPI communicator

    Returns
    -------
    numpy.ndarray
        The summed local arrays.
    """
    if comm is None or comm.size == 1: return local_array
    from mpi4py import MPI  # not at top so can import pygsti on cluster login nodes
    if comm.rank not in owners:
        local_array.fill(0)  # zero-out array so it doesn't contribute to the sum (better way?)
    result = _np.empty(local_array.shape, local_array.dtype)
    comm.Allreduce(local_array, result, op=MPI.SUM)
    return result


def closest_divisor(a, b):
    """
    Returns the divisor of `a` that is closest to `b`.

    Parameters
    ----------
    a, b : int

    Returns
    -------
    int
    """
    if b >= a or b == 0: return a  # b=0 is special case.
    for test in range(b, 0, -1):
        if a % test == 0: return test
    assert(False), "Should never get here - a %% 1 == 0 always! (a=%s, b=%s)" % (str(a), str(b))
