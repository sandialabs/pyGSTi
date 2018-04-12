""" Defines the TermEvalTree class which implements an evaluation tree. """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np

from . import gatestring as _gs
from ..baseobjs import VerbosityPrinter as _VerbosityPrinter
from ..tools import slicetools as _slct
from .evaltree import EvalTree

import time as _time #DEBUG TIMERS

class TermEvalTree(EvalTree):
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
        super(TermEvalTree, self).__init__(items)

    def initialize(self, gateLabels, compiled_gatestring_list, numSubTreeComms=1, maxCacheSize=None):
        """
          TODO: docstring -- and other tree's initialize methods?
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

        gatestring_list = [tuple(gs) for gs in compiled_gatestring_list.keys()]
        self.compiled_gatestring_spamTuples = list(compiled_gatestring_list.values())
        self.num_final_els = sum([len(v) for v in self.compiled_gatestring_spamTuples])
        #self._compute_finalStringToEls() #depends on compiled_gatestring_spamTuples
        self.recompute_spamtuple_indices(bLocal=True) # bLocal shouldn't matter here

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

        #Sort the gate strings "alphabetically" - not needed now, but may
        # be useful later for prefixing...
        sorted_strs = sorted(list(enumerate(gatestring_list)),key=lambda x: x[1])

        for k,(iStr,gateString) in enumerate(sorted_strs):
            #Add info for this string
            self[iStr] = gateString
            self.eval_order.append(iStr)

        #Storage for polynomial expressions for probabilities and
        # their derivatives
        self.raw_polys = {}
        self.p_polys = {}
        self.dp_polys = {}
        self.hp_polys = {}

        self.myFinalToParentFinalMap = None #this tree has no "children",
        self.myFinalElsToParentFinalElsMap = None # i.e. has not been created by a 'split'
        self.parentIndexMap = None
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
            gateStrings[i] = self[i]
            
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

        
    def split(self, elIndicesDict, maxSubTreeSize=None, numSubTrees=None, verbosity=0):
        """
        Split this tree into sub-trees in order to reduce the
          maximum size of any tree (useful for limiting memory consumption
          or for using multiple cores).  Must specify either maxSubTreeSize
          or numSubTrees.

        Parameters
        ----------
        elIndicesDict : dict
            A dictionary whose keys are integer original-gatestring indices
            and whose values are slices or index arrays of final-element-
            indices (typically this dict is returned by calling
            :method:`GateSet.compile_gatestrings`).  Since splitting a 
            tree often involves permutation of the raw string ordering
            and thereby the element ordering, an updated version of this
            dictionary, with all permutations performed, is returned.

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
        OrderedDict
            A updated version of elIndicesDict
        """
        #dbList = self.generate_gatestring_list()
        tm = _time.time()
        printer = _VerbosityPrinter.build_printer(verbosity)

        if (maxSubTreeSize is None and numSubTrees is None) or \
           (maxSubTreeSize is not None and numSubTrees is not None):
            raise ValueError("Specify *either* maxSubTreeSize or numSubTrees")
        if numSubTrees is not None and numSubTrees <= 0:
            raise ValueError("EvalTree split() error: numSubTrees must be > 0!")

        #Don't split at all if it's unnecessary
        if maxSubTreeSize is None or len(self) < maxSubTreeSize:
            if numSubTrees is None or numSubTrees == 1: return elIndicesDict

        self.subTrees = []
        evalOrder = self.get_evaluation_order()
        printer.log("EvalTree.split done initial prep in %.0fs" %
                    (_time.time()-tm)); tm = _time.time()

        def create_subtrees(maxCost, maxCostRate=0, costMetric="size"):
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
            cacheIndices = [None]*self.cache_size()

            for k in evalOrder:
                iStart,remainder,iCache = self[k]

                if iCache is not None:
                    cacheIndices[iCache] = k

                #compute the cost (additional #applies) which results from
                # adding this element to the current tree.
                cost = cost_fn(remainder)
                inds = set([k])

                if iStart is not None and cacheIndices[iStart] not in curSubTree:
                    #we need to add the tree elements traversed by
                    #following iStart
                    j = iStart #index into cache
                    while j is not None:
                        iStr = cacheIndices[j] # cacheIndices[ iStart ]
                        inds.add(iStr)
                        cost += cost_fn(self[iStr][1]) # remainder
                        j = self[iStr][0] # iStart

                        
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
                        iStr = cacheIndices[j]
                        curSubTree.add(iStr)
                        cost += cost_fn(self[iStr][1]) #remainder
                        j = self[iStr][0] # iStart
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

        subTreeSetList = []
        if numSubTrees is not None:

            subTreeSize = len(self) // numSubTrees
            for i in range(numSubTrees):
                end = (i+1)*subTreeSize if (i < numSubTrees-1) else len(self)
                subTreeSetList.append( set(range(i*subTreeSize,end)) )

        else: # maxSubTreeSize is not None
            k = 0
            while k < len(self):
                end = min(k+maxSubTreeSize,len(self))
                subTreeSetList.append( set(range(k,end)) )
                k = end

                
        ##########################################################
        # Part II: create subtrees from index sets
        ##########################################################
        # (common logic provided by base class up to providing a few helper fns)
        
        def permute_parent_element(perm, el):
            """Applies a permutation to an element of the tree """
            # perm[oldIndex] = newIndex
            return el # no need to permute gate string
    
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
                A list of the integers between 0 and len(parentIndices)-1 which
                gives the evaluation order of the subtree *including* evaluation
                of any initial elements.

            sliceIntoParentsFinalArray : slice
                Described above - map between to-be-created subtree's final
                elements and parent-tree indices.

            parentTree : EvalTree
                The parent tree itself.
            """
            subTree = TermEvalTree()
            subTree.myFinalToParentFinalMap = sliceIntoParentsFinalArray
            subTree.num_final_strs = numFinal
            subTree[:] = [None]*len(parentIndices)
            subTree.p_polys = None
            subTree.dp_polys = None
            subTree.hp_polys = None

            mapParentIndxToSubTreeIndx = { k: ik for ik,k in enumerate(parentIndices) }
            curCacheSize = 0
            subTreeCacheIndices = {}
    
            for ik in fullEvalOrder: #includes any initial indices
                k = parentIndices[ik] #original tree index
                gatestring = self[k] #original tree data
                subTree.eval_order.append(ik)
                assert(subTree[ik] is None)
                subTree[ik] = gatestring

            subTree.parentIndexMap = parentIndices #parent index of each subtree index
            subTree.compiled_gatestring_spamTuples = [ self.compiled_gatestring_spamTuples[k]
                                                       for k in _slct.indices(subTree.myFinalToParentFinalMap) ]
            #subTree._compute_finalStringToEls() #depends on compiled_gatestring_spamTuples
            
            final_el_startstops = []; i=0
            for spamTuples in parentTree.compiled_gatestring_spamTuples:
                final_el_startstops.append( (i,i+len(spamTuples)) )
                i += len(spamTuples)
            subTree.myFinalElsToParentFinalElsMap = _np.concatenate(
                [ _np.arange(*final_el_startstops[k])
                  for k in _slct.indices(subTree.myFinalToParentFinalMap) ] )
            #Note: myFinalToParentFinalMap maps only between *final* elements
            #   (which are what is held in compiled_gatestring_spamTuples)
            
            subTree.num_final_els = sum([len(v) for v in subTree.compiled_gatestring_spamTuples])
            subTree.recompute_spamtuple_indices(bLocal=False)

            return subTree
    
        updated_elIndices = self._finish_split(elIndicesDict, subTreeSetList,
                                               permute_parent_element, create_subtree)
        printer.log("EvalTree.split done second pass in %.0fs" %
                    (_time.time()-tm)); tm = _time.time()
        return updated_elIndices

    def cache_size(self):
        """ 
        Returns the size of the persistent "cache" of partial results
        used during the computation of all the strings in this tree.
        """
        return 0

    def copy(self):
        """ Create a copy of this evaluation tree. """
        cpy = self._copyBase( TermEvalTree(self[:]) )
        return cpy

    def get_raw_polys(self, calc, rholabel, elabels, comm):
        #Check if everything is computed already
        if all([ ((rholabel,elabel) in self.raw_polys) for elabel in elabels]):
            return [self.raw_polys[(rholabel,elabel)] for elabel in elabels]

        print("DB: **** COMPUTING RAW POLYS FOR: ",rholabel,elabels, " **********")
        #Otherwise compute poly -- FUTURE: do this faster w/
        # some self.prs_as_polys(rholabel, elabels, gatestring, ...) function
        ret = []
        for elabel in elabels:
            if (rholabel,elabel) not in self.raw_polys:
                polys = [ calc.pr_as_poly((rholabel,elabel), gstr, comm)
                          for gstr in self.generate_gatestring_list(permute=False) ]
                self.raw_polys[ (rholabel,elabel) ] = polys
            ret.append( self.raw_polys[ (rholabel,elabel) ] )
        return ret

    
    def get_p_polys(self, calc, rholabel, elabels, comm):
        #Check if everything is computed already
        if all([ ((rholabel,elabel) in self.p_polys) for elabel in elabels]):
            return [self.p_polys[(rholabel,elabel)] for elabel in elabels]

        #Otherwise compute poly -- FUTURE: do this faster w/
        # some self.prs_as_polys(rholabel, elabels, gatestring, ...) function
        ret = []
        polys = self.get_raw_polys(calc, rholabel, elabels, comm)
        for i,elabel in enumerate(elabels):
            if (rholabel,elabel) not in self.p_polys:
                tapes = [ poly.compact() for poly in polys[i] ]
                vtape = _np.concatenate( [ t[0] for t in tapes ] )
                ctape = _np.concatenate( [ t[1] for t in tapes ] )
                self.p_polys[ (rholabel,elabel) ] = (vtape, ctape)
            ret.append( self.p_polys[ (rholabel,elabel) ] )
        return ret


    def get_dp_polys(self, calc, rholabel, elabels, wrtSlice, comm):
        slcTup = (wrtSlice.start,wrtSlice.stop,wrtSlice.step) \
                 if (wrtSlice is not None) else (None,None,None)
        slcInds = _slct.indices(wrtSlice if (wrtSlice is not None) else slice(0,calc.Np))
            
        #Check if everything is computed already
        if all([ ((rholabel,elabel,slcTup) in self.dp_polys) for elabel in elabels]):
            return [self.dp_polys[(rholabel,elabel,slcTup)] for elabel in elabels]

        #Otherwise compute poly -- FUTURE: do this faster w/
        # some self.prs_as_polys(rholabel, elabels, gatestring, ...) function
        ret = []
        polys = self.get_raw_polys(calc, rholabel, elabels, comm)
        for i,elabel in enumerate(elabels):
            if (rholabel,elabel,slcTup) not in self.dp_polys:
                tapes = [ p.deriv(k).compact() for p in polys[i] for k in slcInds ]
                vtape = _np.concatenate( [ t[0] for t in tapes ] )
                ctape = _np.concatenate( [ t[1] for t in tapes ] )
                self.dp_polys[ (rholabel,elabel,slcTup) ] = (vtape, ctape)
            ret.append( self.dp_polys[ (rholabel,elabel,slcTup) ] )
        return ret

    def get_hp_polys(self, calc, rholabel, elabels, wrtSlice1, wrtSlice2, comm):
        slcTup1 = (wrtSlice1.start,wrtSlice1.stop,wrtSlice1.step) \
                 if (wrtSlice1 is not None) else (None,None,None)
        slcTup2 = (wrtSlice2.start,wrtSlice2.stop,wrtSlice2.step) \
                 if (wrtSlice2 is not None) else (None,None,None)
        slcInds1 = _slct.indices(wrtSlice1 if (wrtSlice1 is not None) else slice(0,calc.Np))
        slcInds2 = _slct.indices(wrtSlice2 if (wrtSlice2 is not None) else slice(0,calc.Np))
            
        #Check if everything is computed already
        if all([ ((rholabel,elabel,slcTup1,slcTup2) in self.hp_polys) for elabel in elabels]):
            return [self.hp_polys[(rholabel,elabel,slcTup1,slcTup2)] for elabel in elabels]

        #Otherwise compute poly -- FUTURE: do this faster w/
        # some self.prs_as_polys(rholabel, elabels, gatestring, ...) function
        ret = []
        for elabel in elabels:
            if (rholabel,elabel,slcTup1,slcTup2) not in self.hp_polys:
                polys = [ calc.pr_as_poly((rholabel,elabel), gstr, comm)
                          for gstr in self.generate_gatestring_list(permute=False) ]
                dpolys = [ p.deriv(k) for p in polys for k in slcInds2 ]
                tapes = [ dp.deriv(k).compact() for p in dpolys for k in slcInds1 ]
                vtape = _np.concatenate( [ t[0] for t in tapes ] )
                ctape = _np.concatenate( [ t[1] for t in tapes ] )
                self.hp_polys[ (rholabel,elabel,slcTup1,slcTup2) ] = (vtape, ctape)
            ret.append( self.hp_polys[ (rholabel,elabel,slcTup1,slcTup2) ] )
        return ret
