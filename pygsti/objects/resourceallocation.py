"""
Resource allocation manager
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
import os as _os
import numpy as _np
import socket as _socket
import collections as _collections
from contextlib import contextmanager as _contextmanager
from pygsti.objects.profiler import DummyProfiler as _DummyProfiler
from hashlib import blake2b as _blake2b
from ..tools import sharedmemtools as _smt

_dummy_profiler = _DummyProfiler()
_GB = 1.0 / (1024.0)**3  # converts bytes to GB


class ResourceAllocation(object):
    """
    Describes available resources and how they should be allocated.

    This includes the number of processors and amount of memory,
    as well as a strategy for how computations should be distributed
    among them.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm, optional
        MPI communicator holding the number of available processors.

    mem_limit : int, optional
        A rough per-processor memory limit in bytes.

    profiler : Profiler, optional
        A lightweight profiler object for tracking resource usage.

    distribute_method : str, optional
        The name of a distribution strategy.
    """

    @classmethod
    def cast(cls, arg):
        """
        Cast `arg` to a :class:`ResourceAllocation` object.

        If `arg` already is a :class:`ResourceAllocation` instance, it
        just returned.  Otherwise this function attempts to create a new
        instance from `arg`.

        Parameters
        ----------
        arg : ResourceAllocation or dict
            An object that can be cast to a :class:`ResourceAllocation`.

        Returns
        -------
        ResourceAllocation
        """
        if arg is None:
            return cls()
        elif isinstance(arg, ResourceAllocation):
            return arg
        else:  # assume argument is a dict of args
            return cls(arg.get('comm', None), arg.get('mem_limit', None),
                       arg.get('profiler', None), arg.get('distribute_method', 'default'))

    def __init__(self, comm=None, mem_limit=None, profiler=None, distribute_method="default", allocated_memory=0):
        self.comm = comm
        self.mem_limit = mem_limit
        self.host_comm = None  # comm of the processors local to each processor's host (distinct hostname)
        self.host_ranks = None  # tuple of the self.comm ranks that belong to self.host_comm
        self.interhost_comm = None  # comm used to spread results to other hosts; 1 proc on each host (node)
        self.interhost_ranks = None  # tuple of the self.comm ranks that belong to self.interhost_comm
        self.host_index = 0  # index of the host this proc belongs to (~= hostname)
        self.host_index_for_rank = None  # a dict mapping self.comm.rank => host_index
        self.layout_distribution = None
        self.jac_distribution_method = None
        self.jac_slice = None
        if profiler is not None:
            self.profiler = profiler
        else:
            self.profiler = _dummy_profiler
        self.distribute_method = distribute_method
        self.reset(allocated_memory)

    def build_hostcomms(self):
        if self.comm is None:
            self.host_comm = None
            self.host_ranks = None
            self.interhost_comm = None
            self.interhost_ranks = None
            self.host_index = 0
            self.host_index_for_rank = None
            return

        my_rank = self.comm.rank
        my_color = None  # set to the index of my_rank within the ranks_by_hostname value that contains my_rank
        my_hostname = _gethostname()
        my_hostid = int(_blake2b(my_hostname.encode('utf-8'), digest_size=4).hexdigest(), 16) % (1 << 31)
        self.host_comm = self.comm.Split(color=int(my_hostid), key=int(my_rank))  # Note: 32-bit ints only for mpi4py
        self.host_ranks = tuple(self.host_comm.allgather(my_rank))  # store all the original ranks on our host
        #print("CREATED HOSTCOMM: ",my_hostname, my_hostid, self.host_comm.size, self.host_comm.rank)

        hostnames_by_rank = self.comm.allgather(my_hostname)  # ~"node id" of each processor in self.comm
        ranks_by_hostname = _collections.OrderedDict()
        for rank, hostname in enumerate(hostnames_by_rank):
            if hostname not in ranks_by_hostname: ranks_by_hostname[hostname] = []
            if rank == my_rank: my_color = len(ranks_by_hostname[hostname])
            ranks_by_hostname[hostname].append(rank)
        hostname_indices = {hostname: i for i, hostname in enumerate(ranks_by_hostname.keys())}
        self.host_index = hostname_indices[my_hostname]
        self.host_index_for_rank = {rank: hostname_indices[hostname] for rank, hostname in enumerate(hostnames_by_rank)}

        #check to make sure that each host id that is present occurs the same number of times
        assert(len(set(map(len, ranks_by_hostname.values()))) == 1), \
            ("Could not build an inter-host comm because procs-per-node is not uniform.  Ranks by hostname =\n%s"
             % str(ranks_by_hostname))

        #create sub-comm that groups disjoint sets of processors across all the (present) nodes
        self.interhost_comm = self.comm.Split(color=my_color, key=self.host_index)
        self.interhost_ranks = tuple(self.interhost_comm.allgather(my_rank))  # store all the original ranks by color
        assert(self.interhost_comm.rank == self.host_index)  # because of key= in Split call above

    @property
    def is_host_leader(self):
        """True if this processors is the rank-0 "leader" of its host (node).  False otherwise. """
        return bool(self.host_comm is None or self.host_comm.rank == 0)

    def host_comm_barrier(self):
        """
        Calls self.host_comm.barrier() when self.host_comm is not None.

        This convenience function provides an often-used barrier that
        follows code where a single "leader" processor modifies a memory
        block shared between all members of `self.host_comm`, and the
        other processors must wait until this modification is performed
        before proceeding with their own computations.

        Returns
        -------
        None
        """
        if self.host_comm is not None:
            self.host_comm.barrier()

    def copy(self):
        """
        Copy this object.

        Returns
        -------
        ResourceAllocation
        """
        return ResourceAllocation(self.comm, self.mem_limit, self.profiler, self.distribute_method)

    def reset(self, allocated_memory=0):
        """
        Resets internal allocation counters to given values (defaults to zero).

        Parameters
        ----------
        allocated_memory : int64
            The value to set the memory allocation counter to.

        Returns
        -------
        None
        """
        self.allocated_memory = allocated_memory

    def add_tracked_memory(self, num_elements, dtype='d'):
        """
        Adds `nelements * itemsize` bytes to the total amount of allocated memory being tracked.

        If the total (tracked) memory exceeds `self.mem_limit` a :class:`MemoryError`
        exception is raised.

        Parameters
        ----------
        num_elements : int
            The number of elements to track allocation of.

        dtype : numpy.dtype, optional
            The type of elements, needed to compute the number of bytes per element.

        Returns
        -------
        None
        """
        nbytes = num_elements * _np.dtype(dtype).itemsize
        self.allocated_memory += nbytes
        if self.mem_limit is not None and self.allocated_memory > self.mem_limit:
            raise MemoryError("User-supplied memory limit of %.2fGB has been exceeded! (tracked_mem +  %.2fGB = %.2GB)"
                              % (self.mem_limit * _GB, nbytes * _GB, self.allocated_memory * _GB))

    def check_can_allocate_memory(self, num_elements, dtype='d'):
        """
        Checks that allocating `nelements` doesn't cause the memory limit to be exceeded.

        This memory isn't tracked - it's just added to the current tracked memory and a
        :class:`MemoryError` exception is raised if the result exceeds `self.mem_limit`.

        Parameters
        ----------
        num_elements : int
            The number of elements to track allocation of.

        dtype : numpy.dtype, optional
            The type of elements, needed to compute the number of bytes per element.

        Returns
        -------
        None
        """
        nbytes = num_elements * _np.dtype(dtype).itemsize
        if self.mem_limit is not None and self.allocated_memory + nbytes > self.mem_limit:
            raise MemoryError("User-supplied memory limit of %.2fGB has been exceeded! (testing %.2fGB + %.2fGB)"
                              % (self.mem_limit * _GB, self.allocated_memory * _GB, nbytes * _GB))

    @_contextmanager
    def temporarily_track_memory(self, num_elements, dtype='d'):
        """
        Temporarily adds `nelements` to tracked memory (a context manager).

        A :class:`MemoryError` exception is raised if the tracked memory exceeds `self.mem_limit`.

        Parameters
        ----------
        num_elements : int
            The number of elements to track allocation of.

        dtype : numpy.dtype, optional
            The type of elements, needed to compute the number of bytes per element.

        Returns
        -------
        contextmanager
        """
        nbytes = num_elements * _np.dtype(dtype).itemsize
        self.allocated_memory += nbytes
        if self.mem_limit is not None and self.allocated_memory > self.mem_limit:
            raise MemoryError("User-supplied memory limit of %.2fGB has been exceeded!"
                              % (self.mem_limit / (1024.0**3)))
        yield
        self.allocated_memory -= nbytes

    def layout_distribution(self, layout):
        """
        Cached `layout.distribute(self)` call.

        Parameters
        ----------
        layout : DistributableCOPALayout
            The layout to distributed.

        Returns
        -------
        myAtomIndices : list
            A list of integer indices specifying which atoms this
            processor is responsible for.
        atomOwners : dict
            A dictionary whose keys are integer atom indices and
            whose values are processor ranks, which indicates which
            processor is responsible for communicating the final
            results of each atom.
        mySubResourceAlloc : ResourceAllocation
            The communicator for the processor group that is responsible
            for computing the same `myAtomIndices` list.  This
            communicator is used for further processor division (e.g.
            for parallelization across derivative columns).
        """
        if self._layout_distribution is None:
            self._layout_distribution = layout.distribute(self)
        return self._layout_distribution  # myAtomIndices, atomOwners, sub_resource_alloc

    def gather_base(self, result, local, slice_of_global, unit_ralloc=None, all_gather=False):
        """
        TODO: docstring -- notes:
        result must be allocated as a shared array using *this* ralloc or a larger one
        unit_alloc specifies a comm/ralloc of the group of processors that all compute
           the same logical result -- so only the unit_ralloc.rank == 0 processors
           will contribute to the sum (but all procs get the result)
        NOTE2: slice_of_global may also be an *index array*
        """
        if self.comm is None:
            assert(result.shape == local.shape)
            result[(slice(None, None),) * local.ndim] = local
            return

        participating = unit_ralloc is None or unit_ralloc.comm is None or unit_ralloc.comm.rank == 0
        gather_comm = self.interhost_comm if (self.host_comm is not None) else self.comm
        slices = gather_comm.gather(slice_of_global if participating else None, root=0)

        if all_gather:
            gathered_data = gather_comm.allgather(local)  # could change this to Allgather (?)
        else:
            gathered_data = gather_comm.gather(local, root=0)  # could change this to Gather (?)

        if gather_comm.rank == 0 or all_gather:
            for slc_or_indx_array, data in zip(slices, gathered_data):
                if slc_or_indx_array is None: continue  # signals a non-unit-leader proc that shouldn't do anything
                result[slc_or_indx_array] = data

        self.comm.barrier()  # make sure result is completely filled before returniing
        return

    def gather(self, result, local, slice_of_global, unit_ralloc=None):
        """ TODO: docstring """
        return self.gather_base(result, local, slice_of_global, unit_ralloc, False)

    def allgather(self, result, local, slice_of_global, unit_ralloc=None):
        """ TODO: docstring """
        return self.gather_base(result, local, slice_of_global, unit_ralloc, True)

    def allreduce_sum(self, result, local, unit_ralloc=None):
        """
        TODO: docstring -- notes:
        result must be allocated as a shared array using *this* ralloc or a larger one
        unit_alloc specifies a comm/ralloc of the group of processors that all compute
           the same logical result -- so only the unit_ralloc.rank == 0 processors
           will contribute to the sum (but all procs get the result)
        """
        from mpi4py import MPI
        participating_local = local if (unit_ralloc is None or unit_ralloc.comm is None or unit_ralloc.comm.rank == 0) \
            else _np.zeros(local.shape, local.dtype)
        if self.host_comm is not None:
            #Barrier-sum on host within shared mem
            if self.host_comm.rank == 0: result.fill(0)  # zero out
            self.host_comm.barrier()  # make sure all zero-outs above complete
            for i in range(self.host_comm.size):
                if i == self.host_comm.rank:
                    result += participating_local  # adds *in place* (relies on numpy implementation)
                self.host_comm.barrier()  # synchonize adding to shared mem
            if self.host_comm.rank == 0:
                summed_across_hosts = self.interhost_comm.allreduce(result, op=MPI.SUM)
                result[(slice(None, None),) * result.ndim] = summed_across_hosts
            self.host_comm.barrier()  # wait for allreduce and assignment to complete on non-hostroot procs
        elif self.comm is not None:
            result[(slice(None, None),) * result.ndim] = self.comm.allreduce(participating_local, op=MPI.SUM)
        else:
            result[(slice(None, None),) * result.ndim] = participating_local

    def allreduce_sum_simple(self, local, unit_ralloc=None):
        """
        TODO: docstring -- notes:
        "simple" b/c no shared memory is used, and result is *returned*
        local is just a scalar float or array
        unit_alloc specifies a comm/ralloc of the group of processors that all compute
           the same logical result -- so only the unit_ralloc.rank == 0 processors
           will contribute to the sum (but all procs get the result returned)
        """
        from mpi4py import MPI
        if self.comm is None: return local

        participating = bool(unit_ralloc is None or unit_ralloc.comm is None or unit_ralloc.comm.rank == 0)
        if hasattr(local, 'shape'):
            participating_local = local if participating else _np.zeros(local.shape, 'd')
        else:
            participating_local = local if participating else 0.0
        return self.comm.allreduce(participating_local, op=MPI.SUM)

    def allreduce_max(self, result, local, unit_ralloc=None):
        """
        TODO: docstring -- notes:
        result must be allocated as a shared array using *this* ralloc or a larger one
        unit_alloc specifies a comm/ralloc of the group of processors that all compute
           the same logical result -- so only the unit_ralloc.rank == 0 processors
           will contribute to the sum (but all procs get the result)
        """
        from mpi4py import MPI
        participating = unit_ralloc is None or unit_ralloc.comm is None or unit_ralloc.comm.rank == 0

        if self.host_comm is not None:
            #Barrier-max on host within shared mem
            if self.host_comm.rank == 0: result.fill(-1e100)  # sentinel
            self.host_comm.barrier()  # make sure all zero-outs above complete
            for i in range(self.host_comm.size):
                if i == self.host_comm.rank and participating:
                    _np.maximum(result, local, out=result)
                self.host_comm.barrier()  # synchonize adding to shared mem
            if self.host_comm.rank == 0:
                maxed_across_hosts = self.interhost_comm.allreduce(result, op=MPI.MAX)
                result[(slice(None, None),) * result.ndim] = maxed_across_hosts
            self.host_comm.barrier()  # wait for allreduce and assignment to complete on non-hostroot procs
        elif self.comm is not None:
            participating_local = local if participating else -1e100
            result[(slice(None, None),) * result.ndim] = self.comm.allreduce(participating_local, op=MPI.MAX)
        else:
            result[(slice(None, None),) * result.ndim] = local

    def bcast(self, value, root=0):
        """
        TODO: docstring
        Broadcast `value` from root *host* to all other hosts in this resource allocation.

        Note: `value` must be a numpy array.
        """
        if self.host_comm is not None:
            bcast_shape, bcast_dtype = self.comm.bcast((value.shape, value.dtype) if self.comm.rank == root else None,
                                                       root=root)
            ar, ar_shm = _smt.create_shared_ndarray(self, bcast_shape, bcast_dtype)
            if self.comm.rank == root:
                ar[(slice(None, None),) * value.ndim] = value  # put our value into the shared memory.

            self.host_comm.barrier()  # wait until shared mem is written to on all root-host procs
            interhost_root = self.host_index_for_rank[root]  # (b/c host_index == interhost.rank)
            ret = self.interhost_comm.bcast(ar, root=interhost_root)
            self.comm.barrier()  # wait until everyone's values are ready
            _smt.cleanup_shared_ndarray(ar_shm)
            return ret
        elif self.comm is not None:
            return self.comm.bcast(value, root=root)
        else:
            return value

    def __getstate__(self):
        # Can't pickle comm objects
        to_pickle = self.__dict__.copy()
        to_pickle['comm'] = None  # will cause all unpickled ResourceAllocations comm=`None`
        return to_pickle


def _gethostname():
    """ Mimics multiple hosts on a single host, mostly for debugging"""
    hostname = _socket.gethostname()
    if _os.environ.get('PYGSTI_MAX_HOST_PROCS', None):
        max_vhost_procs = int(_os.environ['PYGSTI_MAX_HOST_PROCS'])
        try:
            from mpi4py import MPI
            hostname += "_vhost%d" % (MPI.COMM_WORLD.rank // max_vhost_procs)
        except ImportError:
            pass
    return hostname
