from __future__ import division, print_function, absolute_import, unicode_literals
from time import time
from contextlib import contextmanager

@contextmanager
def timed_block(label, timeDict=None):
    start = time()
    try:
        yield
    finally:
        end = time()
        t = end - start
        if timeDict is not None:
            timeDict[label].append(t)
        else:
            print('{} block took {} seconds'.format(label, str(t)))
    '''
    start = perf_counter()
    try:
        yield
    finally:
        end = perf_counter()
        print('{} block took {} seconds'.format(label, str(end-start)))
    '''
