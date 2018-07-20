"""Utility functions for working with Python slice objects"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np

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

def list_to_slice(lst, array_ok=False, require_contiguous=True):
    """
    Returns a slice corresponding to a given list of (integer) indices,
    if this is possible.  If not, `array_ok` determines the behavior.

    Parameters
    ----------
    lst : list
      The list of integers to convert to a slice (must be contiguous
      if `require_contiguous == True`).

    array_ok : bool, optional
      If True, an integer array (of type `numpy.ndarray`) is returned
      when `lst` does not correspond to a single slice.  Otherwise,
      an AssertionError is raised.

    require_contiguous : bool, optional
      If True, then lst must correspond to a contiguous (step=1)
      slice, or else an AssertionError is raised.

    Returns
    -------
    numpy.ndarray or slice
    """
    if isinstance(lst, slice):
        if require_contiguous:
            assert(lst.step is None or lst.step == 1), \
                "Slice must be contiguous!"
        return lst
    if lst is None or len(lst) == 0: return slice(0,0)
    start=lst[0]

    if len(lst) == 1: return slice(start,start+1)
    step=lst[1]-lst[0]; stop = start + step*len(lst)
    if require_contiguous:
        assert(step == 1), "Slice must be contiguous!"

    if list(lst) == list(range(start,stop,step)):
        if step == 1: step = None
        return slice(start,stop,step)

    if array_ok: return _np.array(lst,_np.int64)
    else: raise ValueError("List does not correspond to a slice!")

def as_array(slcOrListLike):
    """
    Returns `slcOrListLike` as an index array (an integer numpy.ndarray).
    """
    if isinstance(slcOrListLike, slice):
        return _np.array(indices(slcOrListLike), _np.int64)
    else:
        return _np.array(slcOrListLike, _np.int64)


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
