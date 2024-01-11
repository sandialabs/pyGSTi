## user-exposure: minimal (EGN - internal use only.)
"""
Defines the Profiler class and supporting functionality
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import inspect as _inspect
import itertools as _itertools
import os as _os
import time as _time

import numpy as _np

try:
    # Get memory usage using psutil, if available
    import psutil as _psutil

    def _get_mem_usage():
        p = _psutil.Process(_os.getpid())
        return p.memory_info()[0]

except ImportError:
    try:
        # If psutil is unavailable, get memory usage using resource (only available on Unix platforms)
        import sys as _sys
        import resource as _resource

        def _get_mem_usage():
            mem = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
            # peak memory usage (bytes on OS X, kilobytes on Linux)
            if _sys.platform != 'darwin': mem *= 1024  # now always in bytes
            return mem
    except ImportError:
        # No memory usage polling available
        def _get_mem_usage():
            raise EnvironmentError("Memory profiling is not available (hint: try `pip install psutil`)")


def _get_root_mem_usage(comm):
    """ Returns the memory usage on the 0th processor """
    mem = _get_mem_usage()
    if comm is not None:
        if comm.Get_rank() == 0:
            comm.bcast(mem, root=0)
        else:
            mem = comm.bcast(None, root=0)
    return mem


def _get_max_mem_usage(comm):
    """ Returns the memory usage on the 0th processor """
    mem = _get_mem_usage()
    if comm is not None:
        memlist = comm.allgather(mem)
        mem = max(memlist)
    return mem


_BtoGB = 1.0 / (1024.0**3)  # convert bytes -> GB


class Profiler(object):
    """
    Profiler objects are used for tracking both time and memory usage.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm optional
        The active MPI communicator.

    default_print_memcheck : bool, optional
        Whether to print memory checks.
    """

    def __init__(self, comm=None, default_print_memcheck=False):
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

    def add_time(self, name, start_time, prefix=0):
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

        prefix : int, optional
            Prefix to the timer name the current stack depth and this number
            of function names, starting with the current function and moving
            the call stack.  When zero, no prefix is added. For example,
            with `prefix == 1`, "Total" might map to " 3: myFunc: Total".

        Returns
        -------
        None
        """
        if prefix > 0:
            stack = _inspect.stack()
            try:
                depth = len(stack) - 1  # -1 to discount current fn (add_time)
                functions = " : ".join(_inspect.getframeinfo(frm[0]).filename
                                       for frm in reversed(stack[1:1 + prefix]))
                name = "%2d: %s: %s" % (depth, functions, name)
            finally:
                stack = None  # make sure frames get cleaned up properly

        val = _time.time() - start_time
        if name in self.timers:
            self.timers[name] += val
        else:
            self.timers[name] = val

        #if self.comm is None or self.comm.Get_rank() == 0:
        #    print("TIME [%s] += %.2fs (total = %.2fs)" % (name,val,self.timers[name]))

    def add_count(self, name, inc=1, prefix=0):
        """
        Adds a given value to a named "counter"-type accumulator.

        Parameters
        ----------
        name : string
            The name of the counter to add `val` into (if the name doesn't exist,
            one is created and initialized to `val`).

        inc : int, optional
            The increment (the value to add to the counter).

        prefix : int, optional
            Prefix to the timer name the current stack depth and this number
            of function names, starting with the current function and moving
            the call stack.  When zero, no prefix is added. For example,
            with `prefix == 1`, "Total" might map to " 3: myFunc: Total".

        Returns
        -------
        None
        """
        if prefix > 0:
            stack = _inspect.stack()
            try:
                depth = len(stack) - 1  # -1 to discount current fn (add_count)
                functions = " : ".join(_inspect.getframeinfo(frm[0]).filename
                                       for frm in reversed(stack[1:1 + prefix]))
                name = "%2d: %s: %s" % (depth, functions, name)
            finally:
                stack = None  # make sure frames get cleaned up properly

        if name in self.counters:
            self.counters[name] += inc
        else:
            self.counters[name] = inc

    def memory_check(self, name, printme=None, prefix=0):
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

        prefix : int, optional
            Prefix to the timer name the current stack depth and this number
            of function names, starting with the current function and moving
            the call stack.  When zero, no prefix is added. For example,
            with `prefix == 1`, "Total" might map to " 3: myFunc: Total".

        Returns
        -------
        None
        """
        if prefix > 0:
            stack = _inspect.stack()
            try:
                depth = len(stack) - 1  # -1 to discount current fn (memory_check)
                functions = " : ".join(_inspect.getframeinfo(frm[0]).filename
                                       for frm in reversed(stack[1:1 + prefix]))
                name = "%2d: %s: %s" % (depth, functions, name)
            finally:
                stack = None  # make sure frames get cleaned up properly

        usage = _get_mem_usage()
        timestamp = _time.time()
        if name in self.mem_checkpoints:
            self.mem_checkpoints[name].append((timestamp, usage))
        else:
            self.mem_checkpoints[name] = [(timestamp, usage)]

        bPrint = self.print_memchecks if (printme is None) else printme
        if bPrint: self.print_memory(name)

    def print_memory(self, name, show_minmax=False):
        """
        Prints the current memory usage (but doesn't store it).

        Useful for debugging, this function prints the current memory
        usage - optionally giving the mininum, maximum, and average
        across all the processors.

        Parameters
        ----------
        name : string
            A label to print before the memory usage number(s).

        show_minmax : bool, optional
            If True and there are multiple processors, print the
            min, average, and max memory usage from among the processors.
            Note that this will invoke MPI collective communication and so
            this `print_memory` call **must** be executed by all the processors.
            If False and there are multiple processors, only the rank 0
            processor prints output.

        Returns
        -------
        None
        """
        usage = _get_mem_usage()
        if self.comm is not None:
            if show_minmax:
                memlist = self.comm.gather(usage, root=0)
                if self.comm.Get_rank() == 0:
                    avg_usage = sum(memlist) * _BtoGB / self.comm.Get_size()
                    min_usage = min(memlist) * _BtoGB
                    max_usage = max(memlist) * _BtoGB
                    print("MEM USAGE [%s] = %.2f GB, %.2f GB, %.2f GB" %
                          (name, min_usage, avg_usage, max_usage))
            elif self.comm.Get_rank() == 0:
                print("MEM USAGE [%s] = %.2f GB" % (name, usage * _BtoGB))
        else:
            print("MEM USAGE [%s] = %.2f GB" % (name, usage * _BtoGB))

    def print_message(self, msg, all_ranks=False):
        """
        Prints a message to stdout, possibly from all ranks.

        A utility function used in debugging, this function offers a
        convenient way to print a message on only the root processor
        or on all processors.

        Parameters
        ----------
        msg : string
            The message to print.

        all_ranks : bool, optional
            If True, all processors will print `msg`, preceded by their
            rank label (e.g. "Rank4: ").  If False, only the rank 0
            processor will print the message.

        Returns
        -------
        None
        """
        if self.comm is not None:
            if all_ranks:
                print("Rank%d: %s" % (self.comm.Get_rank(), msg))
            elif self.comm.Get_rank() == 0:
                print(msg)
        else: print(msg)

    def _format_times(self, sort_by="name"):
        """
        Formats a string to report the timer values recorded in this Profiler.

        Parameters
        ----------
        sort_by : {"name","time"}
            What to sort list of timers by.

        Returns
        -------
        str
        """
        s = "---> Times (by %s): \n" % sort_by
        if sort_by == "name":
            timerNames = sorted(list(self.timers.keys()))
        elif sort_by == "time":
            timerNames = sorted(list(self.timers.keys()),
                                key=lambda x: self.timers[x])
        else:
            raise ValueError("Invalid 'sort_by' argument: %s" % sort_by)

        for nm in timerNames:
            s += "  %s : %.1fs\n" % (nm, self.timers[nm])
        s += "\n"
        return s

    def _format_counts(self, sort_by="name"):
        """
        Formats a string to report the counter values recorded in this Profiler.

        Parameters
        ----------
        sort_by : {"name","count"}
            What to sort list of counts by.

        Returns
        -------
        str
        """
        s = "---> Counters (by %s): \n" % sort_by
        if sort_by == "name":
            counterNames = sorted(list(self.counters.keys()))
        elif sort_by == "count":
            counterNames = sorted(list(self.counters.keys()),
                                  key=lambda x: self.counters[x])
        else:
            raise ValueError("Invalid 'sort_by' argument: %s" % sort_by)

        for nm in counterNames:
            s += "  %s : %d\n" % (nm, self.counters[nm])
        s += "\n"
        return s

    def _format_memory(self, sort_by="name"):
        """
        Formats a string to report the memory usage checkpoints recorded in this Profiler.

        Parameters
        ----------
        sort_by : {"name","usage","timestamp"}
            What to sort list of counts by.

        Returns
        -------
        str
        """
        if len(self.mem_checkpoints) == 0:
            return "No memory checkpoints"

        #for key in self.mem_checkpoints:
        #    print("ITEM:",self.mem_checkpoints[key])
        #    assert(False)
        #print("LIST: ",list(self.mem_checkpoints.values()))
        max_memory = max([usage for timestamp, usage in
                          _itertools.chain(*self.mem_checkpoints.values())])
        s = "---> Max Memory usage = %.2fGB\n" % (max_memory * _BtoGB)
        s += "---> Memory usage (by %s): \n" % sort_by

        if sort_by == "timestamp":
            # special case in that we print each event, not just the average usage per checkpoint
            raise NotImplementedError("TODO")

        avg_usages = {k: _np.mean([u for t, u in infos]) for k, infos
                      in self.mem_checkpoints.items()}

        if sort_by == "name":
            chkptNames = sorted(list(self.mem_checkpoints.keys()))
        elif sort_by == "usage":
            chkptNames = sorted(list(avg_usages.keys()),
                                key=lambda x: avg_usages[x])
        else:
            raise ValueError("Invalid 'sort_by' argument: %s" % sort_by)

        for nm in chkptNames:
            usages = [u for t, u in self.mem_checkpoints[nm]]
            s += "  %s : %.2fGB (min=%.2f,max=%.2f)\n" % \
                (nm, avg_usages[nm] * _BtoGB, min(usages) * _BtoGB, max(usages) * _BtoGB)
        s += "\n"
        return s

    def __getstate__(self):
        #Return the state (for pickling) -- *don't* pickle Comm object
        to_pickle = self.__dict__.copy()
        del to_pickle['comm']  # one *cannot* pickle Comm objects
        return to_pickle

    def __setstate__(self, state_dict):
        self.__dict__.update(state_dict)
        self.comm = None  # initialize to None upon unpickling


class DummyProfiler(object):
    """
    A dummy profiler that doesn't do anything.

    A class which implements the same interface as Profiler but
    which doesn't actually do any profiling (consists of stub functions).
    """

    def __init__(self):
        """
        Construct a new DummyProfiler instance.
        """

    def add_time(self, name, start_time, prefix=0):
        """
        Stub function that does nothing

        Parameters
        ----------
        name : string
            The name of the timer to add elapsed time into (if the name doesn't
            exist, one is created and initialized to the elapsed time).

        start_time : float
            The starting time used to compute the elapsed, i.e. the value
            `time.time()-start_time`, which is added to the named timer.

        prefix : int, optional
            Prefix to the timer name the current stack depth and this number
            of function names, starting with the current function and moving
            the call stack.  When zero, no prefix is added. For example,
            with `prefix == 1`, "Total" might map to " 3: myFunc: Total".

        Returns
        -------
        None
        """
        pass

    def add_count(self, name, inc=1, prefix=0):
        """
        Stub function that does nothing

        Parameters
        ----------
        name : string
            The name of the counter to add `val` into (if the name doesn't exist,
            one is created and initialized to `val`).

        inc : int, optional
            The increment (the value to add to the counter).

        prefix : int, optional
            Prefix to the timer name the current stack depth and this number
            of function names, starting with the current function and moving
            the call stack.  When zero, no prefix is added. For example,
            with `prefix == 1`, "Total" might map to " 3: myFunc: Total".

        Returns
        -------
        None
        """
        pass

    def memory_check(self, name, printme=None, prefix=0):
        """
        Stub function that does nothing

        Parameters
        ----------
        name : string
            The name of the memory checkpoint.  (Later, memory information can
            be organized by checkpoint name.)

        printme : bool, optional
            Whether or not to print the memory usage during this function call
            (if None, the default, then the value of `default_print_memcheck`
            specified during Profiler construction is used).

        prefix : int, optional
            Prefix to the timer name the current stack depth and this number
            of function names, starting with the current function and moving
            the call stack.  When zero, no prefix is added. For example,
            with `prefix == 1`, "Total" might map to " 3: myFunc: Total".

        Returns
        -------
        None
        """
        pass
