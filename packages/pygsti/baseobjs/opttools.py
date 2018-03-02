""" This module defines tools for optimization and profiling """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************


from time        import time
from contextlib  import contextmanager
from collections import defaultdict
from datetime    import datetime
from functools   import wraps
import warnings

# note that this decorator ignores **kwargs
def cache_by_hashed_args(obj):
    """ Decorator for caching a function values """
    cache = obj.cache = {}

    @wraps(obj)
    def _memoizer(*args, **kwargs):
        if len(kwargs) > 0:
            #instead of an error, just don't cache in this case
            warnings.warn('Cannot currently memoize on kwargs') 
            return obj(*args, **kwargs)
        try:
            if args not in cache:
                cache[args] = obj(*args, **kwargs)
            return cache[args]
        except TypeError:
            print('Warning: arguments for cached function could not be cached')
            return obj(*args, **kwargs)
    return _memoizer


@contextmanager
def timed_block(label, timeDict=None, printer=None, verbosity=2, roundPlaces=6, preMessage=None, formatStr=None):
    """
    Context manager that times a block of code

    Parameters
    ----------
    label : str
        An identifying label for this timed block.

    timeDict : dict, optional
        A dictionary to store the final time in, under the key `label`.

    printer : VerbosityPrinter, optional
        A printer object to log the timer's message.  If None, this message will
        be printed directly.

    verbosity : int, optional
        The verbosity level at which to print the time message (if `printer` is
        given).

    roundPlaces : int, opitonal
        How many decimal places of precision to print time with (in seconds).

    preMessage : str, optional
        A format string to print out before the timer's message, which
        formats the `label` arguent, e.g. `"My label is {}"`.

    formatStr : str, optional
        A format string used to format the label before the resulting "rendered
        label" is used as the first argument in the final formatting string
        `"{} took {} seconds"`.
    """
    def put(message):
        """Prints message"""
        if printer is None:
            print(message)
        else:
            printer.log(message, verbosity)

    if preMessage is not None:
        put(preMessage.format(label))
    start = time()
    try:
        yield
    finally:
        end = time()
        t = end - start
        if timeDict is not None:
            if isinstance(timeDict, defaultdict):
                timeDict[label].append(t)
            else:
                timeDict[label] = t
        else:
            if formatStr is not None:
                label = formatStr.format(label)
            put('{} took {} seconds'.format(label, str(round(t, roundPlaces))))

def time_hash():
    """Get string-version of current time"""
    return str(datetime.now())
