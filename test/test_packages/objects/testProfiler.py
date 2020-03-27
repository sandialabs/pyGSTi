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

        s = p.format_times(sort_by="name")
        s = p.format_times(sort_by="time")
        with self.assertRaises(ValueError):
            p.format_times(sort_by="foobar")

        s = p.format_counts(sort_by="name")
        s = p.format_counts(sort_by="count")
        with self.assertRaises(ValueError):
            p.format_counts(sort_by="foobar")

        s = p.format_memory(sort_by="name")
        s = p.format_memory(sort_by="usage")
        with self.assertRaises(ValueError):
            p.format_memory(sort_by="foobar")
        with self.assertRaises(NotImplementedError):
            p.format_memory(sort_by="timestamp")
        empty = profiler.Profiler(comm, default_print_memcheck=True)
        self.assertEqual(empty.format_memory(sort_by="timestamp"),"No memory checkpoints")

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
