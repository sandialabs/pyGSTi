from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Utility functions related to Gram matrix construction."""

from .. import construction as _construction
from .core import gram_rank_and_evals as _gramRankAndEvals


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

    def have_all_data(strings,datasetStrs):
        for a in strings:
            for b in strings:
                if tuple(list(a) + list(b)) not in datasetStrs:
                    return False
        return True

    max_string_set = [ ]
    for p in possibleStrings:
        if have_all_data(max_string_set + [p], datasetStrings):
            max_string_set.append(p)

    return max_string_set


def max_gram_rank_and_evals(dataset, maxBasisStringLength=10,
                            targetGateset=None, spamDict=None,
                            fixedLists=None):
    """
    Compute the rank and singular values of a maximal Gram matrix,that is, the
    Gram matrix using a basis computed by:
    get_max_gram_basis(dataset.get_gate_labels(), dataset, maxBasisStringLength).

    Parameters
    ----------
    dataset : DataSet
      the dataset to use when constructing the Gram matrix

    maxBasisStringLength : int, optional
      the maximum string length considered for Gram matrix basis
      elements.  Defaults to 10.

    targetGateset : GateSet, optional
      A gateset used to specify the SPAM labels used to connect
      the dataset values to rhoVec and EVec indices.

    spamDict : dictionary, optional
      Dictionary mapping (rhoVec_index,EVec_index) integer tuples to string spam labels.
      Defaults to the spam dictionary of targetGateset
      e.g. spamDict[(0,0)] == "plus"

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
        maxStringSet = get_max_gram_basis(dataset.get_gate_labels(), dataset, maxBasisStringLength)
        maxRhoStrs = maxEStrs = maxStringSet

    if spamDict is None:
        if targetGateset is not None:
            spamDict = targetGateset.get_reverse_spam_defs()
            rhoLabels = list(targetGateset.preps.keys())
            eLabels = list(targetGateset.effects.keys()) # 'remainder' should *not* be an effectLabel here
        else:
            firstSpamLabel = dataset.get_spam_labels()[0]
            spamDict = {('dummy_rho','dummy_E'): firstSpamLabel}
            rhoLabels = ['dummy_rho']; eLabels = ['dummy_E']
            # Note: it doesn't actually matter what strings we use here

    specs = _construction.build_spam_specs(prepStrs=maxRhoStrs, effectStrs=maxEStrs,
                                           prep_labels=rhoLabels, effect_labels=eLabels)
    return _gramRankAndEvals(dataset, specs, targetGateset, spamDict)
