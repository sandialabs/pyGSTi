#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
"""
Profiler utilities.
"""
import cProfile
from mpi4py import MPI


def profile(filename=None, comm=MPI.COMM_WORLD):
    """
    A decorator for profiling (using cProfile) a function.

    Parameters
    ----------
    filename : str, optional
        Filename to dump profiler stats to.

    comm : mpi4py.MPI.Comm, optional
        Communicator so that different processors dump to different files.
    """
    def prof_decorator(f):
        def wrap_f(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            result = f(*args, **kwargs)
            pr.disable()

            if filename is None:
                pr.print_stats()
            else:
                rankstr = ".{}".format(comm.rank) if comm is not None else ""
                filename_r = filename + rankstr
                pr.dump_stats(filename_r)

            return result
        return wrap_f
    return prof_decorator
