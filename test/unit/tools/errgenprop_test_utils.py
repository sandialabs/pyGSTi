"""
Helpers for tests of the parallelized error generator propagation code
(`pygsti.tools.errgenproptools` and `pygsti.errorgenpropagation`), used by `test_errgenproptools.py`
and `test/unit/objects/test_errorgenpropagation.py`. This module is not collected by pytest.
"""
from contextlib import contextmanager

from pygsti.tools import errorgenpropagation as _eprop

from .mptools_test_utils import assert_pool_dispatch


@contextmanager
def forced_parallel(*operations):
    """
    Context manager forcing the named error generator propagation operations (any of 'alpha',
    'magnus', 'layer_prop', 'commuting_product'; all of them if none are given) to take their
    parallel code path regardless of workload size, restoring the default configuration on exit.
    """
    _eprop.configure_errorgen_propagation_parallelization(force_parallel=list(operations) if operations else True)
    try:
        yield
    finally:
        _eprop.configure_errorgen_propagation_parallelization(reset_to_defaults=True)


@contextmanager
def parallel_path_exercised(testcase, operations, expected_dispatch):
    """
    Force the parallel path of `operations` (see `forced_parallel`) within the block, and afterwards
    assert that work was really dispatched to worker processes through each of the callables named in
    `expected_dispatch` (see `mptools_test_utils.assert_pool_dispatch`). Yields the dispatch spy.
    """
    with forced_parallel(*operations), assert_pool_dispatch(testcase, expected_dispatch) as spy:
        yield spy
