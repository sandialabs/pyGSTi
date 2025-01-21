"""
This module defines tools for optimization and profiling
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from time import time

from pygsti.tools.legacytools import deprecate


# note that this decorator ignores **kwargs


@deprecate('functools.lru_cache')
def cache_by_hashed_args(obj):
    """
    Decorator for caching a function values

    .. deprecated:: v0.9.8.3
        :func:`cache_by_hashed_args` will be removed in pyGSTi
        v0.9.9. Use :func:`functools.lru_cache` instead.

    Parameters
    ----------
    obj : function
        function to decorate

    Returns
    -------
    function
    """
    return lru_cache(maxsize=128)(obj)


@contextmanager
def timed_block(label, time_dict=None, printer=None, verbosity=2, round_places=6, pre_message=None, format_str=None):
    """
    Context manager that times a block of code

    Parameters
    ----------
    label : str
        An identifying label for this timed block.

    time_dict : dict, optional
        A dictionary to store the final time in, under the key `label`.

    printer : VerbosityPrinter, optional
        A printer object to log the timer's message.  If None, this message will
        be printed directly.

    verbosity : int, optional
        The verbosity level at which to print the time message (if `printer` is
        given).

    round_places : int, opitonal
        How many decimal places of precision to print time with (in seconds).

    pre_message : str, optional
        A format string to print out before the timer's message, which
        formats the `label` arguent, e.g. `"My label is {}"`.

    format_str : str, optional
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

    if pre_message is not None:
        put(pre_message.format(label))
    start = time()
    try:
        yield
    finally:
        end = time()
        t = end - start
        if time_dict is not None:
            if isinstance(time_dict, defaultdict):
                time_dict[label].append(t)
            else:
                time_dict[label] = t
        else:
            if format_str is not None:
                label = format_str.format(label)
            put('{} took {} seconds'.format(label, str(round(t, round_places))))


def time_hash():
    """
    Get string-version of current time

    Returns
    -------
    str
    """
    return str(datetime.now())
