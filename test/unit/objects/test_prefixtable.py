import numpy as np

from ..util import BaseCase


#TODO: create a prefixtable and use this function to check it -- this was
# taken from an internal checking function within prefixtable.py

#def _check_prefix_table(prefix_table):  #generate_circuit_list(self, permute=True):
#    """
#    Generate a list of the final operation sequences this tree evaluates.
#
#    This method essentially "runs" the tree and follows its
#      prescription for sequentailly building up longer strings
#      from shorter ones.  When permute == True, the resulting list
#      should be the same as the one passed to initialize(...), and
#      so this method may be used as a consistency check.
#
#    Parameters
#    ----------
#    permute : bool, optional
#        Whether to permute the returned list of strings into the
#        same order as the original list passed to initialize(...).
#        When False, the computed order of the operation sequences is
#        given, which is matches the order of the results from calls
#        to `Model` bulk operations.  Non-trivial permutation
#        occurs only when the tree is split (in order to keep
#        each sub-tree result a contiguous slice within the parent
#        result).
#
#    Returns
#    -------
#    list of gate-label-tuples
#        A list of the operation sequences evaluated by this tree, each
#        specified as a tuple of operation labels.
#    """
#    circuits = [None] * len(self)
#
#    cachedStrings = [None] * self.cache_size()
#
#    #Build rest of strings
#    for i in self.get_evaluation_order():
#        iStart, remainingStr, iCache = self[i]
#        if iStart is None:
#            circuits[i] = remainingStr
#        else:
#            circuits[i] = cachedStrings[iStart] + remainingStr
#
#        if iCache is not None:
#            cachedStrings[iCache] = circuits[i]
#
#    #Permute to get final list:
#    nFinal = self.num_final_strings()
#    if self.original_index_lookup is not None and permute:
#        finalCircuits = [None] * nFinal
#        for iorig, icur in self.original_index_lookup.items():
#            if iorig < nFinal: finalCircuits[iorig] = circuits[icur]
#        assert(None not in finalCircuits)
#        return finalCircuits
#    else:
#        assert(None not in circuits[0:nFinal])
#        return circuits[0:nFinal]
