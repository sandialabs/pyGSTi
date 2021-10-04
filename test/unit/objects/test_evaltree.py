import numpy as np

from ..util import BaseCase


#TODO: create an evaltree and use this function to check it -- this was
# taken from an internal checking function within evaltree.py
#def check_tree(evaltree, original_list): #generate_circuit_list(self, permute=True):
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
#    #Set "initial" (single- or zero- gate) strings
#    for i, opLabel in zip(self.get_init_indices(), self.get_init_labels()):
#        if opLabel == "": circuits[i] = ()  # special case of empty label
#        else: circuits[i] = (opLabel,)
#
#    #Build rest of strings
#    for i in self.get_evaluation_order():
#        iLeft, iRight = self[i]
#        circuits[i] = circuits[iLeft] + circuits[iRight]
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
