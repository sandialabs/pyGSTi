"""
Utility functions for working with lists
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import itertools as _itertools

import numpy as _np


def remove_duplicates_in_place(l, index_to_test=None):
    """
    Remove duplicates from the list passed as an argument.

    Parameters
    ----------
    l : list
        The list to remove duplicates from.

    index_to_test : int, optional
        If not None, the index within the elements of l to test.  For
        example, if all the elements of l contain 2 tuples (x,y) then
        set index_to_test == 1 to remove tuples with duplicate y-values.

    Returns
    -------
    None
    """
    s = set(); n = 0
    if index_to_test is None:
        for x in l:
            if x not in s:
                s.add(x)
                l[n] = x; n += 1
    else:
        for x in l:
            t = x[index_to_test]

            if t not in s:
                s.add(t)
                l[n] = x; n += 1

    del l[n:]


def remove_duplicates(l, index_to_test=None):
    """
    Remove duplicates from the a list and return the result.

    Parameters
    ----------
    l : iterable
        The list/set to remove duplicates from.

    index_to_test : int, optional
        If not None, the index within the elements of l to test.  For
        example, if all the elements of l contain 2 tuples (x,y) then
        set index_to_test == 1 to remove tuples with duplicate y-values.

    Returns
    -------
    list
        the list after duplicates have been removed.
    """
    s = set(); ret = []
    if index_to_test is None:
        for x in l:
            if x not in s:
                s.add(x)
                ret.append(x)
    else:
        for x in l:
            t = x[index_to_test]
            #TODO: create a special duplicate removal function for use with
            #  WeighedOpStrings ...
            if t not in s:
                s.add(t)
                ret.append(x)
    return ret


def compute_occurrence_indices(lst):
    """
    A 0-based list of integers specifying which occurrence, i.e. enumerated duplicate, each list item is.

    For example, if `lst` = [ 'A','B','C','C','A'] then the
    returned list will be   [  0 , 0 , 0 , 1 , 1 ].  This may be useful
    when working with `DataSet` objects that have `collisionAction`
    set to "keepseparate".

    Parameters
    ----------
    lst : list
        The list to process.

    Returns
    -------
    list
    """
    lookup = {}; ret = []
    for x in lst:
        if x not in lookup:
            lookup[x] = 0
        else:
            lookup[x] += 1
        ret.append(lookup[x])
    return ret


def find_replace_tuple(t, alias_dict):
    """
    Replace elements of t according to rules in `alias_dict`.

    Parameters
    ----------
    t : tuple or list
        The object to perform replacements upon.

    alias_dict : dictionary
        Dictionary whose keys are potential elements of `t` and whose values
        are tuples corresponding to a sub-sequence that the given element should
        be replaced with.  If None, no replacement is performed.

    Returns
    -------
    tuple
    """
    t = tuple(t)
    if alias_dict is None: return t
    for label, expandedStr in alias_dict.items():
        while label in tuple(t):
            i = t.index(label)
            t = t[:i] + tuple(expandedStr) + t[i + 1:]
    return t


def find_replace_tuple_list(list_of_tuples, alias_dict):
    """
    Applies :func:`find_replace_tuple` on each element of `list_of_tuples`.

    Parameters
    ----------
    list_of_tuples : list
        A list of tuple objects to perform replacements upon.

    alias_dict : dictionary
        Dictionary whose keys are potential elements of `t` and whose values
        are tuples corresponding to a sub-sequence that the given element should
        be replaced with.  If None, no replacement is performed.

    Returns
    -------
    list
    """
    return [find_replace_tuple(t, alias_dict) for t in list_of_tuples]


def apply_aliases_to_circuits(list_of_circuits, alias_dict):
    """
    Applies `alias_dict` to the circuits in `list_of_circuits`.

    Parameters
    ----------
    list_of_circuits : list
        A list of circuits to make replacements in.

    alias_dict : dict
        A dictionary whose keys are layer Labels (or equivalent tuples or
        strings), and whose values are Circuits or tuples of labels.

    Returns
    -------
    list
    """
    if len(list_of_circuits) == 0 or not alias_dict:
        return list_of_circuits
    return [c.replace_layers_with_aliases(alias_dict) for c in list_of_circuits]


def sorted_partitions(n):
    """
    Iterate over all sorted (decreasing) partitions of integer `n`.

    A partition of `n` here is defined as a list of one or more non-zero
    integers which sum to `n`.  Sorted partitions (those iterated over here)
    have their integers in decreasing order.

    Parameters
    ----------
    n : int
        The number to partition.
    """

    if n == 0:  # special case
        yield _np.zeros(0, _np.int64); return

    p = _np.zeros(n, _np.int64)
    k = 0    # Index of last element in a partition
    p[k] = n  # Initialize first partition as number itself

    # This loop first yields current partition, then generates next
    # partition. The loop stops when the current partition has all 1s
    while True:
        yield p[0:k + 1]

        # Find the rightmost non-one value in p[]. Also, update the
        # rem_val so that we know how much value can be accommodated
        rem_val = 0
        while k >= 0 and p[k] == 1:
            rem_val += p[k]
            k -= 1

        # if k < 0, all the values are 1 so there are no more partitions
        if k < 0: return

        # Decrease the p[k] found above and adjust the rem_val
        p[k] -= 1
        rem_val += 1

        # If rem_val is more, then the sorted order is violated.  Divide
        # rem_val in different values of size p[k] and copy these values at
        # different positions after p[k]
        while rem_val > p[k]:
            p[k + 1] = p[k]
            rem_val -= p[k]
            k += 1

        # Copy rem_val to next position and increment position
        p[k + 1] = rem_val
        k += 1


def partitions(n):
    """
    Iterate over all partitions of integer `n`.

    A partition of `n` here is defined as a list of one or more non-zero
    integers which sum to `n`.  Every partition is iterated over exacty
    once - there are no duplicates/repetitions.

    Parameters
    ----------
    n : int
        The number to partition.
    """
    for p in sorted_partitions(n):
        previous = tuple()
        for pp in _itertools.permutations(p[::-1]):  # flip p so it's in *ascending* order
            if pp > previous:  # only *unique* permutations
                previous = pp  # (relies in itertools implementatin detail that
                yield pp      # any permutations of a sorted iterable are in
                # sorted order unless they are duplicates of prior permutations


def partition_into(n, nbins):
    """
    Iterate over all partitions of integer `n` into `nbins` bins.

    Here, unlike in :func:`partition`, a "partition" is allowed to contain
    zeros.  For example, (4,1,0) is a valid partition of 5 using 3 bins.  This
    function fixes the number of bins and iterates over all possible length-
    `nbins` partitions while allowing zeros.  This is equivalent to iterating
    over all usual partitions of length at most `nbins` and inserting zeros into
    all possible places for partitions of length less than `nbins`.

    Parameters
    ----------
    n : int
        The number to partition.

    nbins : int
        The fixed number of bins, equal to the length of all the
        partitions that are iterated over.
    """
    if n == 0:
        a = _np.zeros(nbins, _np.int64)
        yield tuple(a)

    elif n == 1:
        a = _np.zeros(nbins, _np.int64)
        for i in range(nbins):
            a[i] = 1
            yield tuple(a)
            a[i] = 0

    elif n == 2:
        a = _np.zeros(nbins, _np.int64)
        for i in range(nbins):
            a[i] = 2
            yield tuple(a)
            a[i] = 0

        for i in range(nbins):
            a[i] = 1
            for j in range(i + 1, nbins):
                a[j] = 1
                yield tuple(a)
                a[j] = 0
            a[i] = 0

    else:
        for p in _partition_into_slow(n, nbins):
            yield p


def _partition_into_slow(n, nbins):
    """
    Helper function for `partition_into` that performs the same task for
    a general number `n`.
    """
    for p in sorted_partitions(n):
        if len(p) > nbins: continue  # don't include partitions of length > nbins
        previous = tuple()
        p = _np.concatenate((p, _np.zeros(nbins - len(p), _np.int64)))  # pad with zeros
        for pp in _itertools.permutations(p[::-1]):
            if pp > previous:  # only *unique* permutations
                previous = pp  # (relies in itertools implementatin detail that
                yield pp      # any permutations of a sorted iterable are in
                # sorted order unless they are duplicates of prior permutations


def incd_product(*args):
    """
    Like `itertools.product` but returns the first modified (incremented) index along with the product tuple itself.

    Parameters
    ----------
    `*args` : iterables
        Any number of iterable things that we're taking the product of.
    """
    lists = [list(a) for a in args]  # so we can get new iterators to each argument
    iters = [iter(l) for l in lists]
    N = len(lists)
    incr = 0  # the first index that was changed (incremented) since the last iteration
    try:
        t = [next(i) for i in iters]
    except StopIteration:  # at least one list is empty
        yield incr, ()  # just yield one item w/empty tuple, like itertools.product
        return
    yield incr, tuple(t)  # first yield is special b/c establishes baseline (incr==0)

    incr = N - 1
    while incr >= 0:
        try:  # to increment index incr
            t[incr] = next(iters[incr])
        except StopIteration:  # if exhaused, increment iterator to left
            incr -= 1
        else:  # reset all iterators to right of incremented one and yield
            for i in range(incr + 1, N):
                iters[i] = iter(lists[i])
                t[i] = next(iters[i])  # won't raise error b/c all lists have len >= 1
            yield incr, tuple(t)
            incr = N - 1  # next time try to increment the last index again
    return


def lists_to_tuples(obj):
    """
    Recursively replaces lists with tuples.

    Can be useful for fixing tuples that were serialized to json or mongodb.
    Recurses on lists, tuples, and dicts within `obj`.

    Parameters
    ----------
    obj : object
        Object to convert.

    Returns
    -------
    object
    """
    if isinstance(obj, (list, tuple)):
        return tuple((lists_to_tuples(el) for el in obj))
    elif isinstance(obj, dict):
        return {lists_to_tuples(k): lists_to_tuples(v) for k, v in obj.items()}
    else:
        return obj
