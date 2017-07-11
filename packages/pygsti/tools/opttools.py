from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

'''
This module defines tools for optimization and profiling
'''
from functools  import partial, wraps
from time       import time
from contextlib import contextmanager

# note that this decorator ignores **kwargs
def cache_by_hashed_args(obj):
    cache = obj.cache = {}

    @wraps(obj)
    def memoizer(*args, **kwargs):
        if len(kwargs) > 0:
            raise ValueError('Cannot currently memoize on kwargs')
        try:
            if args not in cache:
                cache[args] = obj(*args, **kwargs)
            return cache[args]
        except TypeError:
            print('Warning: arguments for cached function could not be cached')
            return obj(*args, **kwargs)
    return memoizer

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
