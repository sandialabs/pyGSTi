""" Utility functions related to Gram matrix construction."""
from GateStringTools import genAllGateStrings as _genAllGateStrings
from Core import gramRankAndEvals as _gramRankAndEvals
from Core import getRhoAndESpecs as _getRhoAndESpecs

########################################################
## Gram matrix stuff
########################################################

def getMaxGramBasis(gateLabels, dataset, maxLength=0):
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
    possibleStrings = _genAllGateStrings(gateLabels, (minLength+1)//2, maxLength//2)
  
    def haveAllData(strings,datasetStrs):
      for a in strings:
        for b in strings:
          if tuple(list(a) + list(b)) not in datasetStrs:
            return False
      return True
    
    max_string_set = [ ]
    for p in possibleStrings:
      if haveAllData(max_string_set + [p], datasetStrings):
        max_string_set.append(p)
  
    return max_string_set


def maxGramRankAndEvals(dataset, maxBasisStringLength=10):
    """
    Compute the rank and eigenvalues of a maximal Gram matrix,that is, the
    Gram matrix using a basis computed by:
    getMaxGramBasis(dataset.getGateLabels(), dataset, maxBasisStringLength).

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
    maxStringSet = getMaxGramBasis(dataset.getGateLabels(), dataset, maxBasisStringLength)
    specs = _getRhoAndESpecs(fiducialGateStrings=maxStringSet) 
    # Note: specs use by default just the 0-th rho and Evec indices, so 
    #  we just need to have a spamDict that associates (0,0) with some spam label
    #  in the dataset -- we just take the first one.
    firstSpamLabel = dataset.getSpamLabels()[0]
    return _gramRankAndEvals(dataset, specs, spamDict={(0,0): firstSpamLabel})

