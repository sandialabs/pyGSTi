from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Defines the Profiler class and supporting functionality"""

import time as _time
import os as _os

try:
    import psutil as _psutil
    def _get_mem_usage():
        p = _psutil.Process(_os.getpid())
        return p.memory_info()[0]

except ImportError:
    import sys as _sys
    import resource as _resource
    def _get_mem_usage():
        mem = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
          # peak memory usage (bytes on OS X, kilobytes on Linux)
        if sys.platform != 'darwin': mem *= 1024 #now always in bytes
        return mem

#from mpi4py import MPI
#MPI.COMM_WORLD

BtoGB = 1.0/(1024.0**3) #convert bytes -> GB

class Profiler(object):
    """
    Profiler objects are used for tracking both time and memory usage.
    """

    def __init__(self,comm=None,default_print_memcheck=False):
        """
        Construct a new Profiler instance.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm, optional
           MPI communicator so only profile and print messages on root proc.
        """
        self.comm = comm
        self.timers = {}
        self.counters = {}
        self.mem_checkpoints = {}
        self.print_memchecks = default_print_memcheck

    def add_time(self, name, start_time):
        """
        Adds an elapsed time to a named "timer"-type accumulator.
        
        Parameters
        ----------
        name : string
           The name of the timer to add elapsed time into (if the name doesn't
           exist, one is created and initialized to the elapsed time).
           
        start_time : float
           The starting time used to compute the elapsed, i.e. the value 
           `time.time()-start_time`, which is added to the named timer.

        Returns
        -------
        None
        """
        val = _time.time() - start_time
        if name in self.timers:
            self.timers[name] += val
        else:
            self.timers[name] = val


    def add_count(self, name, inc=1):
        """
        Adds a given value to a named "counter"-type accumulator.
        
        Parameters
        ----------
        name : string
           The name of the counter to add `val` into (if the name doesn't exist,
           one is created and initialized to `val`).
           
        inc : int, optional
           The increment (the value to add to the counter).

        Returns
        -------
        None
        """
        if name in self.counters:
            self.timers[name] += inc
        else:
            self.timers[name] = inc


    def mem_check(self, name,printme=None):
        """
        Record the memory usage at this point and tag with a `name`.
        
        Parameters
        ----------
        name : string
           The name of the memory checkpoint.  (Later, memory information can
           be organized by checkpoint name.)
           
        printme : bool, optional
           Whether or not to print the memory usage during this function call
           (if None, the default, then the value of `default_print_memcheck`
           specified during Profiler construction is used).

        Returns
        -------
        None
        """
        usage = _get_mem_usage()
        timestamp = _time.time()
        if name in self.mem_checkpoints:
            self.mem_checkpoints[name].append( (timestamp,usage) )
        else:
            self.mem_checkpoints[name] = [ (timestamp,usage) ]

        bPrint = self.print_memchecks if (printme is None) else printme
        if bPrint and (self.comm is None or self.comm.Get_rank() == 0):
            print("MEM USAGE [%s] = %.2f GB" % (name,usage*BtoGB))
        


class DummyProfiler(object):
    """
    A class which implements the same interface as Profiler but 
    which doesn't actually do any profiling (consists of stub functions).
    """

    def __init__(self):
        """
        Construct a new DummyProfiler instance.
        """

    def add_time(self, name, start_time):
        """Stub function that does nothing"""
        pass

    def add_count(self, name, inc=1):
        """Stub function that does nothing"""
        pass

    def mem_check(self, name,printme=None):
        """Stub function that does nothing"""
        pass

#Create a global instance for use as a default profiler in functions elsewhere
no_profiler = DummyProfiler()

    
