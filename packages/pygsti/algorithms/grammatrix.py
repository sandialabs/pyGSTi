""" Utility functions related to Gram matrix construction."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

from .. import construction as _construction
from .core import gram_rank_and_evals as _gramRankAndEvals
from ..objects import ComplementSPAMVec as _ComplementSPAMVec


########################################################
## Gram matrix stuff
########################################################

def get_max_gram_basis(gateLabels, dataset, maxLength=0):
    """
    Compute a maximal set of gate strings that can be used as a basis for a Gram
      matrix.  That is, a maximal set of strings {S_i} such that the gate
      strings { S_i S_j } are all present in dataset.  If maxLength > 0, then
      restrict len(S_i) <= maxLength.

    Parameters
    ----------
    gateLabels : list or tuple
      the gate labels to use in Gram matrix basis strings

    dataset : DataSet
      the dataset to use when constructing the Gram matrix

    maxLength : int, optional
      the maximum string length considered for Gram matrix basis
      elements.  Defaults to 0 (no limit).

    Returns
    -------
    list of tuples
      where each tuple contains gate labels and specifies a single gate string.
    """

    datasetStrings = list(dataset.keys())
    minLength = min( [len(s) for s in datasetStrings] )
    if maxLength <= 0:
        maxLength = max( [len(s) for s in datasetStrings] )
    possibleStrings = _construction.gen_all_gatestrings(gateLabels, (minLength+1)//2, maxLength//2)

    def _have_all_data(strings):
        for a in strings:
            for b in strings:
                if tuple(list(a) + list(b)) not in dataset:
                    return False
        return True

    max_string_set = [ ]
    for p in possibleStrings:
        if _have_all_data(max_string_set + [p]):
            max_string_set.append(p)

    return max_string_set


def max_gram_rank_and_evals(dataset, targetGateset, maxBasisStringLength=10,
                            fixedLists=None):
    """
    Compute the rank and singular values of a maximal Gram matrix,that is, the
    Gram matrix using a basis computed by:
    get_max_gram_basis(dataset.get_gate_labels(), dataset, maxBasisStringLength).

    Parameters
    ----------
    dataset : DataSet
      the dataset to use when constructing the Gram matrix

    targetGateset : GateSet
      A gateset used to make sense of gate strings and for the construction of
      a theoretical gram matrix and spectrum.

    maxBasisStringLength : int, optional
      the maximum string length considered for Gram matrix basis
      elements.  Defaults to 10.

    fixedLists : (prepStrs, effectStrs), optional
      2-tuple of gate string lists, specifying the preparation and
      measurement fiducials to use when constructing the Gram matrix,
      and thereby bypassing the search for such lists.


    Returns
    -------
    rank : integer
    singularvalues : numpy array
    targetsingularvalues : numpy array
    """
    if fixedLists is not None:
        maxRhoStrs, maxEStrs = fixedLists
    else:
        maxRhoStrs = maxEStrs = get_max_gram_basis(dataset.get_gate_labels(),
                                                  dataset, maxBasisStringLength)

    return _gramRankAndEvals(dataset, maxRhoStrs, maxEStrs, targetGateset)
