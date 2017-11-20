from ..testutils import BaseTestCase, compare_files, temp_files
import unittest
import numpy as np
import scipy
import pygsti

from pygsti.tools import cache_by_hashed_args, timed_block

from collections import defaultdict
from functools import partial

@cache_by_hashed_args
def fib(x):
    assert x >= 0
    if x == 1 or x == 0:
        return 1
    else:
        return fib(x - 1) + fib(x - 2)

class OptToolsBaseTestCase(BaseTestCase):
    def test_caching(self):
        fib(20)

        @cache_by_hashed_args
        def process_kwargs(**kwargs):
            return ' '.join(str(k) + str(v) for k, v in kwargs.items())

        with self.assertRaises(ValueError):
            process_kwargs(foo='bar')

        @cache_by_hashed_args
        def takes_dictionary(dictionary):
            return dictionary['key']

        takes_dictionary(dict(key='value'))

    def test_timed_block(self):
        with timed_block('40th fibonacci'):
            print(fib(40))

        timeDict = defaultdict(list)
        with timed_block('100th fibonacci', timeDict): # Duration saved to dict under label "100th fibonacci"
            print(fib(100))



if __name__ == '__main__':
    unittest.main(verbosity=2)
