import unittest
import pygsti
import numpy as np
import pickle
import time

from pygsti.modelpacks.legacy import std1Q_XYI
import pygsti.construction as pc
from pygsti.objects import profiler

from ..testutils import BaseTestCase, compare_files, temp_files

class ProfilerTestCase(BaseTestCase):

    def setUp(self):
        super(ProfilerTestCase, self).setUp()

    def test_profler_methods(self):
        comm=None
        mem = profiler._get_root_mem_usage(comm)
        mem = profiler._get_max_mem_usage(comm)

        start_time = time.time()
        p = profiler.Profiler(comm, default_print_memcheck=True)
        p.add_time("My Name", start_time, prefix=1)
        p.add_count("My Count", inc=1, prefix=1)
        p.add_count("My Count", inc=2, prefix=1)
        p.mem_check("My Memcheck", prefix=1)
        p.mem_check("My Memcheck", prefix=1)
        p.print_mem("My Memcheck just to print")
        p.print_mem("My Memcheck just to print", show_minmax=True)
        p.print_msg("My Message")
        p.print_msg("My Message", all_ranks=True)

        s = p.format_times(sortBy="name")
        s = p.format_times(sortBy="time")
        with self.assertRaises(ValueError):
            p.format_times(sortBy="foobar")

        s = p.format_counts(sortBy="name")
        s = p.format_counts(sortBy="count")
        with self.assertRaises(ValueError):
            p.format_counts(sortBy="foobar")

        s = p.format_memory(sortBy="name")
        s = p.format_memory(sortBy="usage")
        with self.assertRaises(ValueError):
            p.format_memory(sortBy="foobar")
        with self.assertRaises(NotImplementedError):
            p.format_memory(sortBy="timestamp")
        empty = profiler.Profiler(comm, default_print_memcheck=True)
        self.assertEqual(empty.format_memory(sortBy="timestamp"),"No memory checkpoints")

    def test_profiler_pickling(self):
        comm=None
        start_time = time.time()
        p = profiler.Profiler(comm, default_print_memcheck=True)
        p.add_time("My Name", start_time, prefix=1)
        p.add_count("My Count", inc=1, prefix=1)
        p.add_count("My Count", inc=2, prefix=1)
        p.mem_check("My Memcheck", prefix=1)

        s = pickle.dumps(p)
        p2 = pickle.loads(s)


if __name__ == '__main__':
    unittest.main(verbosity=2)
