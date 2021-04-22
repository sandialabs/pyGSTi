"""
Utility functions for working with Python slice objects
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

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
    if not isinstance(s, slice): return len(s)
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
    if s == slice(0, 0, None): return s  # special "null slice": shifted(null_slice) == null_slice
    return slice(s.start + offset, s.stop + offset, s.step)


def intersect(s1, s2):
    """
    Returns the intersection of two slices (which must have the same step).

    Parameters
    ----------
    s1 : slice
        First slice.

    s2 : slice
        Second slice.

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
        start = max(s1.start, s2.start)

    if s1.stop is None:
        stop = s2.stop
    elif s2.stop is None:
        stop = s1.stop
    else:
        stop = min(s1.stop, s2.stop)

    if stop is not None and start is not None and stop < start:
        stop = start

    return slice(start, stop, s1.step)


def intersect_within(s1, s2):
    """
    Returns the intersection of two slices (which must have the same step).
    *and* the sub-slice of `s1` and `s2` that specifies the intersection.

    Furthermore, `s2` may be an array of indices, in which case the returned
    slices become arrays as well.

    Parameters
    ----------
    s1 : slice
        First slice.  Must have definite boundaries (start & stop
        cannot be `None`).

    s2 : slice or numpy.ndarray
        Second slice or index array.

    Returns
    -------
    intersection : slice or numpy.ndarray
        The intersection of `s1` and `s2`.

    subslice1 : slice or numpy.ndarray
        The portion of `s1` that yields `intersection`.

    subslice2 : slice or numpy.ndarray
        The portion of `s2` that yields `intersection`.
    """
    assert(s1.start is not None and s1.stop is not None), \
        "`s1` = %s must have definite boundaries - start & stop cannot be None!" % str(s1)

    if isinstance(s2, slice):
        assert (s1.step is None and s2.step is None) or s1.step == s2.step == 1, \
            "Only implemented for step == 1 slices"
        assert(s2.start is not None and s2.stop is not None), \
            "`s2` = %s must have definite boundaries - start & stop cannot be None!" % str(s2)

        if s1.start < s2.start:
            start = s2.start
            sub1_start = s2.start - s1.start
            sub2_start = 0
        else:
            start = s1.start
            sub1_start = 0
            sub2_start = s1.start - s2.start

        if s1.stop < s2.stop:
            stop = s1.stop
            sub1_stop = s1.stop - s1.start
            sub2_stop = s1.stop - s2.start
        else:
            stop = s2.stop
            sub1_stop = s2.stop - s1.start
            sub2_stop = s2.stop - s2.start

        if start <= stop:  # then there's a nonzero intersection
            return slice(start, stop), slice(sub1_start, sub1_stop), slice(sub2_start, sub2_stop)
        else:  # no intersection - return all empty slices
            return slice(0, 0), slice(0, 0), slice(0, 0)

    else:  # s2 is an array of integer indices

        intersect_indices = []
        sub1_indices = []
        sub2_indices = []
        for ii, i in enumerate(s2):
            if s1.start <= i < s1.stop:
                intersect_indices.append(i)
                sub1_indices.append(i - s1.start)
                sub2_indices.append(ii)
        return _np.array(intersect_indices), _np.array(sub1_indices), _np.array(sub2_indices)


def indices(s, n=None):
    """
    Returns a list of the indices specified by slice `s`.

    Parameters
    ----------
    s : slice
        The slice to operate upon.

    n : int, optional
        The number of elements in the array being indexed,
        used for computing *negative* start/stop points.

    Returns
    -------
    list of ints
    """
    if s.start is None and s.stop is None:
        return []

    if s.start is None:
        start = 0
    elif s.start < 0:
        assert(n is not None), "Must supply `n` to obtain indices of a slice with negative start point!"
        start = n + s.start
    else: start = s.start

    if s.stop is None:
        assert(n is not None), "Must supply `n` to obtain indices of a slice with unspecified stop point!"
        stop = n
    elif s.stop < 0:
        assert(n is not None), "Must supply `n` to obtain indices of a slice with negative stop point!"
        stop = n + s.stop
    else: stop = s.stop

    if s.step is None:
        return list(range(start, stop))
    return list(range(start, stop, s.step))


def list_to_slice(lst, array_ok=False, require_contiguous=True):
    """
    Returns a slice corresponding to a given list of (integer) indices, if this is possible.

    If not, `array_ok` determines the behavior.

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
        If True, then lst will only be converted to a contiguous (step=1)
        slice, otherwise either a ValueError is raised (if `array_ok`
        is False) or an array is returned.

    Returns
    -------
    numpy.ndarray or slice
    """
    if isinstance(lst, slice):
        if require_contiguous:
            if not(lst.step is None or lst.step == 1):
                if array_ok:
                    return _np.array(range(lst.start, lst.stop, 1 if (lst.step is None) else lst.step))
                else:
                    raise ValueError("Slice must be contiguous!")
        return lst
    if lst is None or len(lst) == 0: return slice(0, 0)
    start = lst[0]

    if len(lst) == 1: return slice(start, start + 1)
    step = lst[1] - lst[0]; stop = start + step * len(lst)

    if list(lst) == list(range(start, stop, step)):
        if require_contiguous and step != 1:
            if array_ok: return _np.array(lst, _np.int64)
            else: raise ValueError("Slice must be contiguous (or array_ok must be True)!")
        if step == 1: step = None
        return slice(start, stop, step)
    elif array_ok:
        return _np.array(lst, _np.int64)
    else:
        raise ValueError("List does not correspond to a slice!")


def to_array(slc_or_list_like):
    """
    Returns `slc_or_list_like` as an index array (an integer numpy.ndarray).

    Parameters
    ----------
    slc_or_list_like : slice or list
        A slice, list, or array.

    Returns
    -------
    numpy.ndarray
    """
    if isinstance(slc_or_list_like, slice):
        return _np.array(indices(slc_or_list_like), _np.int64)
    else:
        return _np.array(slc_or_list_like, _np.int64)


def divide(slc, max_len):
    """
    Divides a slice into sub-slices based on a maximum length (for each sub-slice).

    For example:
    `divide(slice(0,10,2), 2) == [slice(0,4,2), slice(4,8,2), slice(8,10,2)]`

    Parameters
    ----------
    slc : slice
        The slice to divide

    max_len : int
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
        sub_slices.append(slice(sub_start, min(sub_start + max_len * step, slc.stop),
                                slc.step))
        sub_start += max_len * step
    return sub_slices


def slice_of_slice(slc, base_slc):
    """
    A slice that is the composition of `base_slc` and `slc`.

    So that when indexing an array `a`,
    `a[slice_of_slice(slc, base_slc)] == a[base_slc][slc]`

    Parameters
    ----------
    slc : slice
        the slice to take out of `base_slc`.

    base_slc : slice
        the original "base" slice to act upon.

    Returns
    -------
    slice
    """
    #NOTE: this is very similar to shift(slc, base_slc.start) above - consolidate or remove this function?
    assert(slc.step in (1, None) and base_slc.step in (1, None)), \
        "This function only works with step == 1 slices so far"
    if slc.start is None and slc.stop is None: return base_slc
    if base_slc.start is None and base_slc.stop is None: return slc

    if base_slc.start is None and slc.start is None:
        start = None
    else:
        start = (base_slc.start if (base_slc.start is not None) else 0) \
            + (slc.start if (slc.start is not None) else 0)

    if base_slc.stop is None and slc.stop is None:
        stop = None
    else:
        stop = (base_slc.start if (base_slc.start is not None) else 0) \
            + (slc.stop if (slc.stop is not None) else 0)
        assert(base_slc.stop is None or stop <= base_slc.stop)
    return slice(start, stop)


def slice_hash(slc):
    return (slc.start, slc.stop, slc.step)
