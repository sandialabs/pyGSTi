from __future__ import division, print_function, absolute_import, unicode_literals
from time import time
from contextlib import contextmanager

@contextmanager
def timed_block(label):
    start = time()
    try:
        yield
    finally:
        end = time()
        print('{} block took {} seconds'.format(label, str(end-start)))
    '''
    start = perf_counter()
    try:
        yield
    finally:
        end = perf_counter()
        print('{} block took {} seconds'.format(label, str(end-start)))
    '''
