import pickle
import time

import pygsti
from pygsti.baseobjs import smartcache as sc
from ..util import BaseCase


@sc.smart_cached
def fib(x):
    assert x >= 0
    if x == 1 or x == 0:
        return 1
    else:
        return fib(x - 1) + fib(x - 2)


@sc.smart_cached
def slow_fib(x):
    # TODO is this useful?
    time.sleep(0.01)
    assert x >= 0
    if x == 1 or x == 0:
        return 1
    else:
        return fib(x - 1) + fib(x - 2)


class SmartCacheTester(BaseCase):
    def test_smart_caching(self):
        for i in range(10, 20):
            fib(i)
        for i in range(10, 20):
            fib(i)
        slow_fib(20)
        slow_fib(20)

    def test_obj(self):
        cache = sc.SmartCache()
        cache.low_overhead_cached_compute(slow_fib, (20,))

    def test_status(self):
        printer = pygsti.baseobjs.VerbosityPrinter(1)
        fib.cache.status(printer)
        slow_fib.cache.status(printer)
        printer = pygsti.baseobjs.VerbosityPrinter(0)
        sc.SmartCache.global_status(printer)

    def test_pickle(self):
        a = pickle.dumps(slow_fib.cache)
        newcache = pickle.loads(a)
        # TODO assert correctness
