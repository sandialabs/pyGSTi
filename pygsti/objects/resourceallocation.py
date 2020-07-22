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
import numpy as _np
from contextlib import contextmanager as _contextmanager
from pygsti.objects.profiler import DummyProfiler as _DummyProfiler

_dummy_profiler = _DummyProfiler()


class ResourceAllocation(object):
    """
    Describes available resources and how they should be allocated.

    This includes the number of processors and amount of memory,
    as well as a strategy for how computations should be distributed
    among them.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm, optional
v        MPI communicator holding the number of available processors.

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

    def __init__(self, comm=None, mem_limit=None, profiler=None, distribute_method="default"):
        self.comm = comm
        self.mem_limit = mem_limit
        if profiler is not None:
            self.profiler = profiler
        else:
            self.profiler = _dummy_profiler
        self.distribute_method = distribute_method
        self.reset()  # begin with no memory allocated

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
        self.allocated_memory = 0

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
            raise MemoryError("User-supplied memory limit of %.2fGB has been exceeded!"
                              % (self.mem_limit / (1024.0**3)))

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
            raise MemoryError("User-supplied memory limit of %.2fGB has been exceeded!"
                              % (self.mem_limit / (1024.0**3)))

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

    def __getstate__(self):
        # Can't pickle comm objects
        to_pickle = self.__dict__.copy()
        to_pickle['comm'] = None  # will cause all unpickled ResourceAllocations comm=`None`
        return to_pickle
