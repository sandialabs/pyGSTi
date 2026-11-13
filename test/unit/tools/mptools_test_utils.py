"""
Helpers for tests of code that dispatches work to worker processes through
`pygsti.tools.mptools.SharedWorkerPool`.

They let a test assert that a parallel code path really ran, rather than silently falling back to
serial execution because the workload was below a parallelization threshold. Used by
`test_errgenproptools.py` and (through `errgenprop_test_utils`) `test_errorgenpropagation.py`.
This module is not collected by pytest.
"""
from contextlib import contextmanager
from unittest import mock

from pygsti.tools import mptools as _mpt


class PoolDispatchSpy:
    """
    Context manager recording the names of the callables dispatched to an *active* worker pool
    through `SharedWorkerPool.map` / `imap_unordered`, i.e. recording which parallel code paths ran
    (as opposed to falling back to serial execution because a workload was below the thresholds).

    The names are collected in the `dispatched` list. Only dispatches through a `SharedWorkerPool`
    are seen: work sent to a plain `multiprocessing.Pool` passed in by the caller is not recorded.
    """
    def __enter__(self):
        self.dispatched = []
        self._patches = []
        for name in ('map', 'imap_unordered'):
            orig = getattr(_mpt.SharedWorkerPool, name)

            def wrapper(pool_self, func, *args, _orig=orig, **kwargs):
                result = _orig(pool_self, func, *args, **kwargs)  # (lazy pools start inside this call)
                if pool_self.is_started:
                    self.dispatched.append(getattr(func, '__name__', type(func).__name__))
                return result
            patch = mock.patch.object(_mpt.SharedWorkerPool, name, wrapper)
            patch.start()
            self._patches.append(patch)
        return self

    def __exit__(self, *exc):
        for patch in self._patches:
            patch.stop()
        return False


@contextmanager
def assert_pool_dispatch(testcase, expected_dispatch):
    """
    Run the block under a `PoolDispatchSpy` and afterwards assert, using `testcase`, that work was
    dispatched to worker processes through each of the callables named in `expected_dispatch`.
    Guards parallel/serial equivalence tests against silently comparing serial results with serial
    results. Yields the spy.
    """
    with PoolDispatchSpy() as spy:
        yield spy
    for name in expected_dispatch:
        testcase.assertIn(name, spy.dispatched, f"the parallel path ({name}) did not run; dispatched: {spy.dispatched}")
