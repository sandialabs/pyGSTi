"""
Utility functions related to Gram matrix construction.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti.algorithms.core import gram_rank_and_eigenvalues as _gram_rank_and_evals
from pygsti import circuits as _circuits


########################################################
## Gram matrix stuff
########################################################

def max_gram_basis(op_labels, dataset, max_length=0):
    """
    Compute a maximal set of basis circuits for a Gram matrix.

    That is, a maximal set of strings {S_i} such that the gate
    strings { S_i S_j } are all present in dataset.  If max_length > 0, then
    restrict len(S_i) <= max_length.

    Parameters
    ----------
    op_labels : list or tuple
        the operation labels to use in Gram matrix basis strings

    dataset : DataSet
        the dataset to use when constructing the Gram matrix

    max_length : int, optional
        the maximum string length considered for Gram matrix basis
        elements.  Defaults to 0 (no limit).

    Returns
    -------
    list of tuples
        where each tuple contains operation labels and specifies a single circuit.
    """

    datasetStrings = list(dataset.keys())
    minLength = min([len(s) for s in datasetStrings])
    if max_length <= 0:
        max_length = max([len(s) for s in datasetStrings])
    possibleStrings = _circuits.iter_all_circuits(op_labels, (minLength + 1) // 2, max_length // 2)

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


def max_gram_rank_and_eigenvalues(dataset, target_model, max_basis_string_length=10,
                                  fixed_lists=None):
    """
    Compute the rank and singular values of a maximal Gram matrix.

    That is, compute the rank and singular values of the Gram matrix computed using the basis:
    max_gram_basis(dataset.gate_labels(), dataset, max_basis_string_length).

    Parameters
    ----------
    dataset : DataSet
        the dataset to use when constructing the Gram matrix

    target_model : Model
        A model used to make sense of circuits and for the construction of
        a theoretical gram matrix and spectrum.

    max_basis_string_length : int, optional
        the maximum string length considered for Gram matrix basis
        elements.  Defaults to 10.

    fixed_lists : (prep_fiducials, effect_fiducials), optional
        2-tuple of :class:`Circuit` lists, specifying the preparation and
        measurement fiducials to use when constructing the Gram matrix,
        and thereby bypassing the search for such lists.

    Returns
    -------
    rank : integer
    singularvalues : numpy array
    targetsingularvalues : numpy array
    """
    if fixed_lists is not None:
        maxRhoStrs, maxEStrs = fixed_lists
    else:
        maxRhoStrs = maxEStrs = max_gram_basis(dataset.gate_labels(),
                                               dataset, max_basis_string_length)

    return _gram_rank_and_evals(dataset, maxRhoStrs, maxEStrs, target_model)
