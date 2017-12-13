"""Utility functions for working with Python slice objects"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

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
    if not isinstance(s,slice): return len(s)
    if s.start is None or s.stop is None: 
        return 0
    if s.step is None: 
        return s.stop - s.start
    else: 
        return len(range(s.start, s.stop, s.step))

def shift(s, offset):
    """
    Returns a new slice whose start and stop points are shifted by `offset`.

    Parameters
    ----------
    s : slice
      The slice to operate upon.

    offset : int
      The amount to shift the start and stop members of `s`.

    Returns
    -------
    slice
    """
    return slice(s.start + offset, s.stop + offset, s.step)


def intersect(s1, s2):
    """
    Returns the intersection of two slices (which must have the same step).

    Parameters
    ----------
    s1, s2 : slice
      The slices to intersect.

    Returns
    -------
    slice
    """
    assert (s1.step is None and s2.step is None) or s1.step == s2.step, \
        "Only implemented for same-step slices"
    if s1.start is None: 
        start = s2.start
    elif s2.start is None: 
        start = s1.start
    else: 
        start = max(s1.start,s2.start)

    if s1.stop is None: 
        stop = s2.stop
    elif s2.stop is None: 
        stop = s1.stop
    else: 
        stop = min(s1.stop,s2.stop)

    if stop is not None and start is not None and stop < start:
        stop = start

    return slice(start, stop, s1.step)


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
    if s.start is None or s.stop is None:
        return []
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
    if isinstance(lst, slice): return lst 
    if not lst: return slice(0,0)
    assert lst == list(range(lst[0],lst[-1]+1))
    return slice(lst[0],lst[-1]+1)


def divide(slc, maxLen):
    """
    Divides a slice into sub-slices based on a maximum length (for each
    sub-slice).

    For example:
    `divide(slice(0,10,2), 2) == [slice(0,4,2), slice(4,8,2), slice(8,10,2)]`

    Parameters
    ----------
    slc : slice
        The slice to divide

    maxLen : int
        The maximum length (i.e. number of indices) allowed in a sub-slice.

    Returns
    -------
    list of slices
    """
    sub_slices = []
    sub_start = slc.start
    step = 1 if (slc.step is None) else slc.step
    while sub_start < slc.stop:
        # Note: len(range(start,stop,step)) == stop-start+(step-1) // step
        sub_slices.append( slice(sub_start, min(sub_start+maxLen*step,slc.stop),
                                 slc.step) )
        sub_start += maxLen*step
    return sub_slices
