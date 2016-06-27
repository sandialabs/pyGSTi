from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Utility functions for working with lists """

def remove_duplicates_in_place(l,indexToTest=None):
    """
    Remove duplicates from the list passed as an argument.

    In the special case when l contains WeightedGateString instances, the
    duplicates are removed in such a way that the largest weight instance
    of any set of duplicates is kept.

    Parameters
    ----------
    l : list
        The list to remove duplicates from.

    indexToTest : int, optional
        If not None, the index within the elements of l to test.  For
        example, if all the elements of l contain 2 tuples (x,y) then
        set indexToTest == 1 to remove tuples with duplicate y-values.

    Returns
    -------
    None
    """
    s = set(); n = 0
    for x in l:
        t = x if indexToTest is None else x[indexToTest]

        #TODO: create a special duplicate removal function for use with
        #  WeighedGateStrings and include the below commented block:
        #Special case of weighted gate strings: if collisions
        # keep the hightest weight string
        #if isinstance(t, _WeightedGateString) and t in s:
        #    for gs in l[0:n]:
        #        if gs == t:
        #            if isinstance(gs, _WeightedGateString):
        #                gs.weight = max(gs.weight, t.weight)
        #            break

        if t not in s:
            s.add(t)
            l[n] = x; n += 1

    del l[n:]

def remove_duplicates(l,indexToTest=None):
    """
    Remove duplicates from the a list and return the result.

    In the special case when l contains WeightedGateString instances, the
    duplicates are removed in such a way that the largest weight instance
    of any set of duplicates is kept.

    Parameters
    ----------
    l : list
        The list to remove duplicates from.

    indexToTest : int, optional
        If not None, the index within the elements of l to test.  For
        example, if all the elements of l contain 2 tuples (x,y) then
        set indexToTest == 1 to remove tuples with duplicate y-values.

    Returns
    -------
    list
        the list after duplicates have been removed.
    """
    lcopy = l[:]; remove_duplicates_in_place(lcopy,indexToTest)
    return lcopy
