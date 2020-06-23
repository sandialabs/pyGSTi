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

    def __init__(self, comm=None, mem_limit=None, profiler=None, distribute_method="default"):
        self.comm = comm
        self.mem_limit = mem_limit
        if profiler is not None:
            self.profiler = profiler
        else:
            self.profiler = _dummy_profiler
        self.distribute_method = distribute_method

    def copy(self):
        """
        Copy this object.

        Returns
        -------
        ResourceAllocation
        """
        return ResourceAllocation(self.comm, self.mem_limit, self.profiler, self.distribute_method)
