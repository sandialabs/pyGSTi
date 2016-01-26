#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation              
#    This Software is released under the GPL license detailed    
#    in the file "license.txt" in the top-level pyGSTi directory 
#*****************************************************************
""" Utility functions related to Gram matrix construction."""

from .. import tools as _tools
from .. import construction as _construction
from core import gram_rank_and_evals as _gramRankAndEvals


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

    datasetStrings = dataset.keys()
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


def max_gram_rank_and_evals(dataset, maxBasisStringLength=10):
    """
    Compute the rank and eigenvalues of a maximal Gram matrix,that is, the
    Gram matrix using a basis computed by:
    get_max_gram_basis(dataset.get_gate_labels(), dataset, maxBasisStringLength).

    Parameters
    ----------
    dataset : DataSet
      the dataset to use when constructing the Gram matrix

    maxBasisStringLength : int, optional
      the maximum string length considered for Gram matrix basis
      elements.  Defaults to 10.
    
    Returns
    -------
    rank : integer
    eigenvalues : numpy array
    """
    maxStringSet = get_max_gram_basis(dataset.get_gate_labels(), dataset, maxBasisStringLength)
    specs = _construction.build_spam_specs(fiducialGateStrings=maxStringSet) 
    # Note: specs use by default just the 0-th rho and Evec indices, so 
    #  we just need to have a spamDict that associates (0,0) with some spam label
    #  in the dataset -- we just take the first one.
    firstSpamLabel = dataset.get_spam_labels()[0]
    return _gramRankAndEvals(dataset, specs, spamDict={(0,0): firstSpamLabel})

