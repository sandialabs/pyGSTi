from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Utility functions for working with Python slice objects"""

def length(s):
    """
    Returns the length (the number of indices) contained in a slice.

    Parameters
    ----------
    s : slice
      The slice to operate upon.

    Returns
    -------
    int
    """
    if s.step is None: return s.stop-s.start
    else: return len(range(s.start,s.stop.s.step))

def shift(s,offset):
    """
    Returns a new slice whose start and stop points are shifted by `offset`.

    Parameters
    ----------
    s : slice
      The slice to operate upon.

    Returns
    -------
    slice
    """
    return slice(s.start+offset,s.stop+offset,s.step)

def indices(s):
    """
    Returns a list of the indices specified by slice `s`.

    Parameters
    ----------
    s : slice
      The slice to operate upon.

    Returns
    -------
    list of ints
    """
    if s.step is None: 
        return list(range(s.start,s.stop))
    return list(range(s.start,s.stop,s.step))

def list_to_slice(lst):
    """
    Returns a slice corresponding to a given list of (integer) indices. If
    the list of indices is not contiguous, an AssertionError is raised.

    Parameters
    ----------
    lst : list
      The list of integers to convert to a slice (must be contiguous).

    Returns
    -------
    slice
    """
    assert(lst == list(range(lst[0],lst[-1]+1)))
    return slice(lst[0],lst[-1]+1)
