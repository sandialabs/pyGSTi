""" Utility functions related to Gram matrix construction."""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from .. import construction as _construction
from .core import gram_rank_and_evals as _gramRankAndEvals
from ..objects import ComplementSPAMVec as _ComplementSPAMVec


########################################################
## Gram matrix stuff
########################################################

def get_max_gram_basis(opLabels, dataset, maxLength=0):
    """
    Compute a maximal set of operation sequences that can be used as a basis for a Gram
      matrix.  That is, a maximal set of strings {S_i} such that the gate
      strings { S_i S_j } are all present in dataset.  If maxLength > 0, then
      restrict len(S_i) <= maxLength.

    Parameters
    ----------
    opLabels : list or tuple
      the operation labels to use in Gram matrix basis strings

    dataset : DataSet
      the dataset to use when constructing the Gram matrix

    maxLength : int, optional
      the maximum string length considered for Gram matrix basis
      elements.  Defaults to 0 (no limit).

    Returns
    -------
    list of tuples
      where each tuple contains operation labels and specifies a single operation sequence.
    """

    datasetStrings = list(dataset.keys())
    minLength = min([len(s) for s in datasetStrings])
    if maxLength <= 0:
        maxLength = max([len(s) for s in datasetStrings])
    possibleStrings = _construction.gen_all_circuits(opLabels, (minLength + 1) // 2, maxLength // 2)

    def _have_all_data(strings):
        for a in strings:
            for b in strings:
                if tuple(list(a) + list(b)) not in dataset:
                    return False
        return True

    max_string_set = []
    for p in possibleStrings:
        if _have_all_data(max_string_set + [p]):
            max_string_set.append(p)

    return max_string_set


def max_gram_rank_and_evals(dataset, targetModel, maxBasisStringLength=10,
                            fixedLists=None):
    """
    Compute the rank and singular values of a maximal Gram matrix,that is, the
    Gram matrix using a basis computed by:
    get_max_gram_basis(dataset.get_gate_labels(), dataset, maxBasisStringLength).

    Parameters
    ----------
    dataset : DataSet
      the dataset to use when constructing the Gram matrix

    targetModel : Model
      A model used to make sense of operation sequences and for the construction of
      a theoretical gram matrix and spectrum.

    maxBasisStringLength : int, optional
      the maximum string length considered for Gram matrix basis
      elements.  Defaults to 10.

    fixedLists : (prepStrs, effectStrs), optional
      2-tuple of operation sequence lists, specifying the preparation and
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

    return _gramRankAndEvals(dataset, maxRhoStrs, maxEStrs, targetModel)
