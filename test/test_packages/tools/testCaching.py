from ..testutils import BaseTestCase, compare_files, temp_files
import unittest
import pygsti
import pickle
import time
from pygsti.baseobjs import SmartCache, smart_cached

@smart_cached
def fib(x):
    assert x >= 0
    if x == 1 or x == 0:
        return 1
    else:
        return fib(x - 1) + fib(x - 2)

@smart_cached
def slow_fib(x):
    time.sleep(1)
    assert x >= 0
    if x == 1 or x == 0:
        return 1
    else:
        return fib(x - 1) + fib(x - 2)

class CachingBaseTestCase(BaseTestCase):
    def test_smart_caching(self):
        for i in range(10, 20):
            fib(i)
        for i in range(10, 20):
            fib(i)
        slow_fib(20)
        slow_fib(20)

    def test_obj(self):
        cache = SmartCache()
        cache.low_overhead_cached_compute(slow_fib, (20,))

    def test_status(self):
        printer = pygsti.objects.VerbosityPrinter(1)
        fib.cache.status(printer)
        slow_fib.cache.status(printer)
        printer = pygsti.objects.VerbosityPrinter(0)
        SmartCache.global_status(printer)

    def test_pickle(self):
        a = pickle.dumps(slow_fib.cache)
        newcache = pickle.loads(a)


if __name__ == '__main__':
    unittest.main(verbosity=2)
