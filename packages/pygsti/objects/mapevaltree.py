from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines the EvalTree class which implements an evaluation tree. """

from . import gatestring as _gs
from ..tools import mpitools as _mpit
from .verbosityprinter import VerbosityPrinter
from .evaltree import EvalTree

import numpy as _np
import time as _time #DEBUG TIMERS

class MapEvalTree(EvalTree):
    """
    An Evaluation Tree.  Instances of this class specify how to
      perform bulk GateSet operations.

    EvalTree instances create and store the decomposition of a list
      of gate strings into a sequence of 2-term products of smaller
      strings.  Ideally, this sequence would prescribe the way to
      obtain the entire list of gate strings, starting with just the
      single gates, using the fewest number of multiplications, but
      this optimality is not guaranteed.
    """
    def __init__(self, items=[]):
        """ Create a new, empty, evaluation tree. """
        super(MapEvalTree, self).__init__(items)

    def initialize(self, gateLabels, gatestring_list, numSubTreeComms=1):
        """
          Initialize an evaluation tree using a set of gate strings.
          This function must be called before using an EvalTree.

          Parameters
          ----------
          gateLabels : list of strings
              A list of all the single gate labels to
              be stored at the beginning of the tree.  This
              list must include all the gate labels contained
              in the elements of gatestring_list.

          gatestring_list : list of (tuples or GateStrings)
              A list of tuples of gate labels or GateString
              objects, specifying the gate strings that
              should be present in the evaluation tree.

          numSubTreeComms : int, optional
              The number of processor groups (communicators)
              to divide the subtrees of this EvalTree among
              when calling `distribute`.  By default, the
              communicator is not divided.

          Returns
          -------
          None
        """
        #tStart = _time.time() #DEBUG TIMER
        self.gateLabels = gateLabels
        if numSubTreeComms is not None:
            self.distribution['numSubtreeComms'] = numSubTreeComms

        if len(gatestring_list ) > 0 and isinstance(gatestring_list[0],_gs.GateString):
            gatestring_list = [gs.tup for gs in gatestring_list]

        #Evaluation tree:
        # A list of tuples, where each element contains
        #  information about evaluating a particular gate string:
        #  (iStart, tuple_of_following_gatelabels )
        # and self.eval_order specifies the evaluation order.
        del self[:] #clear self (a list)

        #Final Indices
        # The first len(gatestring_list) elements of the tree correspond
        # to computing the gate strings requested in gatestring_list.  Doing
        # this make later extraction much easier (views can be used), but
        # requires a non-linear order of evaluation, held in the eval_order list.
        self.eval_order = []

        #initialize self as a list of Nones
        self.num_final_strs = len(gatestring_list)
        self[:] = [None]*self.num_final_strs

        #Sort the gate strings "alphabetically", so that it's trivial to find common prefixes
        sorted_strs = sorted(list(enumerate(gatestring_list)),key=lambda x: x[1])

        #DEBUG
        #print("SORTED"); print("\n".join(map(str,sorted_strs)))

        lastStr = None
        for k,(iStr,gateString) in enumerate(sorted_strs):
            L = len(gateString)
            
            #find longest existing prefix for gateString by working backwards
            # and finding the first string that *is* a prefix of this string
            # (this will necessarily be the longest prefix, given the sorting)
            for i in range(k-1,-1,-1): #from k-1 -> 0
                ic, candidate = sorted_strs[i]
                Lc = len(candidate)
                if L > Lc > 0 and gateString[0:Lc] == candidate:
                    iStart = ic
                    remaining = gateString[Lc:]
                    break
            else: #no break => no prefix
                iStart = None
                remaining = gateString[:]

            #Add info for this string
            self[iStr] = (iStart, remaining)
            self.eval_order.append(iStr)
            
        #FUTURE: could perform a second pass, and if there is
        # some threshold number of elements which share the
        # *same* iStart and the same beginning of the
        # 'remaining' part then add a new "extra" element
        # (beyond the #gatestrings index) which computes
        # the shared prefix and insert this into the eval
        # order.
                        
        self.myFinalToParentFinalMap = None #this tree has no "children",
        self.parentIndexMap = None          # i.e. has not been created by a 'split'
        self.original_index_lookup = None
        self.subTrees = [] #no subtrees yet
        assert(self.generate_gatestring_list() == gatestring_list)
        assert(None not in gatestring_list)


    def generate_gatestring_list(self, permute=True):
        """
        Generate a list of the final gate strings this tree evaluates.

        This method essentially "runs" the tree and follows its
          prescription for sequentailly building up longer strings
          from shorter ones.  When permute == True, the resulting list
          should be the same as the one passed to initialize(...), and
          so this method may be used as a consistency check.

        Parameters
        ----------
        permute : bool, optional
           Whether to permute the returned list of strings into the 
           same order as the original list passed to initialize(...).
           When False, the computed order of the gate strings is 
           given, which is matches the order of the results from calls
           to `GateSet` bulk operations.  Non-trivial permutation
           occurs only when the tree is split (in order to keep 
           each sub-tree result a contiguous slice within the parent
           result).

        Returns
        -------
        list of gate-label-tuples
            A list of the gate strings evaluated by this tree, each
            specified as a tuple of gate labels.
        """
        gateStrings = [None]*len(self)

        #Build rest of strings
        for i in self.get_evaluation_order():
            iStart, remainingStr = self[i]
            if iStart is None:
                gateStrings[i] = remainingStr
            else:
                gateStrings[i] = gateStrings[iStart] + remainingStr
            
        #Permute to get final list:
        nFinal = self.num_final_strings()
        if self.original_index_lookup is not None and permute == True:
            finalGateStrings = [None]*nFinal
            for iorig,icur in self.original_index_lookup.items():
                if iorig < nFinal: finalGateStrings[iorig] = gateStrings[icur]
            assert(None not in finalGateStrings)
            return finalGateStrings
        else:
            assert(None not in gateStrings[0:nFinal])
            return gateStrings[0:nFinal]


    def get_num_applies(self):
        """
        Gets the number of "apply" operations required to compute this tree.

        Returns
        -------
        int
        """
        ops = 0
        for iStart, remainder in self:
            ops += len(remainder)
        return ops


    def split(self, maxSubTreeSize=None, numSubTrees=None, verbosity=0):
        """
        Split this tree into sub-trees in order to reduce the
          maximum size of any tree (useful for limiting memory consumption
          or for using multiple cores).  Must specify either maxSubTreeSize
          or numSubTrees.

        Parameters
        ----------
        maxSubTreeSize : int, optional
            The maximum size (i.e. list length) of each sub-tree.  If the
            original tree is smaller than this size, no splitting will occur.
            If None, then there is no limit.

        numSubTrees : int, optional
            The maximum size (i.e. list length) of each sub-tree.  If the
            original tree is smaller than this size, no splitting will occur.

        verbosity : int, optional
            How much detail to send to stdout.

        Returns
        -------
        None
        """
        #dbList = self.generate_gatestring_list()
        tm = _time.time()
        printer = VerbosityPrinter.build_printer(verbosity)

        if (maxSubTreeSize is None and numSubTrees is None) or \
           (maxSubTreeSize is not None and numSubTrees is not None):
            raise ValueError("Specify *either* maxSubTreeSize or numSubTrees")
        if numSubTrees is not None and numSubTrees <= 0:
            raise ValueError("EvalTree split() error: numSubTrees must be > 0!")

        #Don't split at all if it's unnecessary
        if maxSubTreeSize is None or len(self) < maxSubTreeSize:
            if numSubTrees is None or numSubTrees == 1: return

        self.subTrees = []
        subTreesFinalList = [None]*self.num_final_strings()
        evalOrder = self.get_evaluation_order()
        printer.log("EvalTree.split done initial prep in %.0fs" %
                    (_time.time()-tm)); tm = _time.time()

        def create_subtrees(maxCost, maxCostRate=0, costMetric="applys"):
            """ 
            Find a set of subtrees by iterating through the tree
            and placing "break" points when the cost of evaluating the
            subtree exceeds some 'maxCost'.  This ensure ~ equal cost
            trees, but doesn't ensure any particular number of them.
            
            maxCostRate can be set to implement a varying maxCost
            over the course of the iteration.
            """

            if costMetric == "applys":
                cost_fn = lambda rem: len(rem) #length of remainder = #-apply ops needed
            elif costMetric == "size":
                cost_fn = lambda rem: 1 # everything costs 1 in size of tree
            else: raise ValueError("Uknown cost metric: %s" % costMetric)

            subTrees = []
            curSubTree = set([evalOrder[0]])
            curTreeCost = cost_fn(self[evalOrder[0]][1]) #remainder length of 0th evaluant
            totalCost = 0
            
            for i,k in enumerate(evalOrder):
                iStart,remainder = self[k]

                #compute the cost (additional #applies) which results from
                # adding this element to the current tree.
                cost = cost_fn(remainder)
                inds = set([k])

                if iStart is not None and iStart not in curSubTree:
                    #we need to add the tree elements traversed by
                    #following iStart
                    j = iStart
                    while j is not None:
                        inds.add(j)
                        cost += cost_fn(self[j][1]) # remainder
                        j = self[j][0] #iStart
                        
                if curTreeCost + cost < maxCost:
                    #Just add current string to current tree
                    curTreeCost += cost
                    curSubTree.update(inds)
                else:
                    #End the current tree and begin a new one
                    #print("cost %d+%d exceeds %d" % (curTreeCost,cost,maxCost))
                    subTrees.append(curSubTree)
                    curSubTree = set([k])
                    
                    cost = cost_fn(remainder); j = iStart
                    while j is not None: # always traverse back iStart
                        curSubTree.add(j)
                        cost += cost_fn(self[j][1]) #remainder
                        j = self[j][0] #iStart
                    totalCost += curTreeCost
                    curTreeCost = cost
                    #print("Added new tree w/initial cost %d" % (cost))
                    
                maxCost += maxCostRate

            subTrees.append(curSubTree)
            totalCost += curTreeCost
            return subTrees, totalCost


        ##################################################################
        # Part I: find a list of where the current tree should be broken #
        ##################################################################
        startIndices = None #eval-order indices of starting indices for subtrees
                        
        if numSubTrees is not None:
            maxCost = self.get_num_applies() / numSubTrees
            maxCostLowerBound, maxCostUpperBound = maxCost, None
            maxCostRate, rateLowerBound, rateUpperBound = 0, -1.0/len(self), +1.0/len(self)
            resultingSubtrees = numSubTrees+1 #just to prime the loop
            iteration = 0

            #Iterate until the desired number of subtrees have been found.
            while resultingSubtrees != numSubTrees:
                subTreeSetList, totalCost = create_subtrees(maxCost, maxCostRate)
                resultingSubtrees = len(subTreeSetList)
                #print("DEBUG: resulting numTrees = %d (cost %g) w/maxCost = %g [%s,%s] & rate = %g [%g,%g]" % \
                #     (resultingSubtrees, totalCost, maxCost, str(maxCostLowerBound), str(maxCostUpperBound),
                #      maxCostRate, rateLowerBound, rateUpperBound))

                #DEBUG
                #totalSet = set()
                #for s in subTreeSetList:
                #    totalSet.update(s)
                #print("DB: total set length = ",len(totalSet))
                #assert(len(totalSet) == len(self))

                #Perform binary search in maxCost then maxCostRate to find
                # desired final subtree count.
                if maxCostUpperBound is None or abs(maxCostLowerBound-maxCostUpperBound) > 1.0:
                    last_maxCost = maxCost
                    if resultingSubtrees <= numSubTrees: #too few trees: reduce maxCost
                        maxCost = (maxCost + maxCostLowerBound)/2.0
                        maxCostUpperBound = last_maxCost
                    else: #too many trees: raise maxCost
                        if maxCostUpperBound is None:
                            maxCost = totalCost / numSubTrees
                        else:
                            maxCost = (maxCost + maxCostUpperBound)/2.0
                            maxCostLowerBound = last_maxCost
                else:
                    last_maxRate = maxCostRate
                    if resultingSubtrees <= numSubTrees: # too few trees reduce maxCostRate
                        maxCostRate = (maxCostRate + rateLowerBound)/2.0
                        rateUpperBound = last_maxRate
                    else: # too many trees: increase maxCostRate
                        maxCostRate = (maxCostRate + rateUpperBound)/2.0
                        rateLowerBound = last_maxRate
                        
                iteration += 1
                assert(iteration < 100), "Unsuccessful splitting for 100 iterations!"
                        

        else: # maxSubTreeSize is not None
            subTreeSetList, totalCost = create_subtrees(
                maxSubTreeSize, maxCostRate=0, costMetric="size")

        ##########################################################
        # Part II: create subtrees from index sets
        ##########################################################
        # (common logic provided by base class up to providing a few helper fns)
        
        def permute_parent_element(perm, el):
            """Applies a permutation to an element of the tree """
            # perm[oldIndex] = newIndex
            return (perm[el[0]] if (el[0] is not None) else None, el[1])
    
        def create_subtree(parentIndices, numFinal, fullEvalOrder, sliceIntoParentsFinalArray, parentTree):
            """ 
            Creates a subtree given requisite information:

            Parameters
            ----------
            parentIndices : list 
                The ordered list of (parent-tree) indices to be included in
                the created subtree.

            numFinal : int
                The number of "final" elements, i.e. those that are used to
                construct the final array of results and not just an intermediate.
                The first numFinal elemements of parentIndices are "final", and
                'sliceIntoParentsFinalArray' tells you which final indices of
                the parent they map to.

            fullEvalOrder : list
                A list of the integers between 0 and len(parentIndics)-1 which
                gives the evaluation order of the subtree *including* evaluation
                of any initial elements.

            sliceIntoParentsFinalArray : slice
                Described above - map between to-be-created subtree's final
                elements and parent-tree indices.

            parentTree : EvalTree
                The parent tree itself.
            """
            subTree = MapEvalTree()
            subTree.myFinalToParentFinalMap = sliceIntoParentsFinalArray
            subTree.num_final_strs = numFinal
            subTree[:] = [None]*len(parentIndices)

            mapParentIndxToSubTreeIndx = { k: ik for ik,k in enumerate(parentIndices) }
    
            for ik in fullEvalOrder: #includes any initial indices
                k = parentIndices[ik] #original tree index

                (oStart,remainder) = self[k] #original tree data

                iStart  = None if (oStart is None) else \
                          mapParentIndxToSubTreeIndx[ oStart ]
                subTree.eval_order.append(ik)

                assert(subTree[ik] is None)
                subTree[ik] = (iStart,remainder)

            subTree.parentIndexMap = parentIndices #parent index of each subtree index
            return subTree
    
        self._finish_split(subTreeSetList, permute_parent_element, create_subtree)
        printer.log("EvalTree.split done second pass in %.0fs" %
                    (_time.time()-tm)); tm = _time.time()
        return

    def copy(self):
        """ Create a copy of this evaluation tree. """
        return self._copyBase( MapEvalTree(self[:]) )
