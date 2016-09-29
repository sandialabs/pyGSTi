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

import numpy as _np


class EvalTree(list):
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
        self.gateLabels = []
        self.init_indices = []
        self.eval_order = []
        self.myFinalToParentFinalMap = None
        self.num_final_strs = 0
        self.subTrees = []
        self.parentIndexMap = None
        self.original_index_lookup = None
        self.distribution = {}
        super(EvalTree, self).__init__(items)

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
        self.gateLabels = gateLabels
        if numSubTreeComms is not None:
            self.distribution['numSubtreeComms'] = numSubTreeComms

        if len(gatestring_list ) > 0 and isinstance(gatestring_list[0],_gs.GateString):
            gatestring_list = [gs.tup for gs in gatestring_list]

        #Evaluation dictionary:
        # keys == gate strings that have been evaluated so far
        # values == index of gate string (key) within evalTree
        evalDict = { }

        #Evaluation tree:
        # A list of tuples, where each element contains
        #  information about evaluating a particular gate string:
        #  (iLeft, iRight)
        # and the order of the elements specifies the evaluation order.
        # In particular, the gateString = evalTree[iLeft] + evalTree[iRight]
        #   so that matrix(gateString) = matrixOf(evalTree[iRight]) * matrixOf(evalTree[iLeft])
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

        #Single gate (or zero-gate) computations are assumed to be atomic, and be computed independently.
        #  These labels serve as the initial values, and each gate string is assumed to be a tuple of
        #  gate labels.
        self.init_indices = [] #indices to put initial zero & single gate results
        for gateLabel in self.gateLabels:
            tup = () if gateLabel == "" else (gateLabel,) #special case of empty label == no gate
            if tup in gatestring_list:
                indx = gatestring_list.index(tup) 
                self[indx] = (None,None) #iLeft = iRight = None for always-evaluated zero string
            else:
                indx = len(self)
                self.append( (None,None) ) #iLeft = iRight = None for always-evaluated zero string
            self.init_indices.append( indx )
            evalDict[ tup ] = indx

        #Process gatestrings in order of length, so that we always place short strings
        # in the right place (otherwise assert stmt below can fail)
        indices_sorted_by_gatestring_len = \
            sorted(list(range(len(gatestring_list))),
                   key=lambda i: len(gatestring_list[i]))

        #avgBiteSize = 0
        #useCounts = {}
        #OLD (sequential): for (k,gateString) in enumerate(gatestring_list):
        for k in indices_sorted_by_gatestring_len:
            gateString = gatestring_list[k]
            L = len(gateString)
            #if L == 0:
            #    self[0] = (None,None,k) #set in-final-list index for zero-length string (special case)
            #    finalIndxList[k] = 0

            start = 0; bite = 1
            #nBites = 0
            #print "DB: string = ",gateString, " tree length = ",len(self)

            while start < L:

                #Take a bite out of gateString, starting at `start` that is in evalDict
                for b in range(L-start,0,-1):
                    if gateString[start:start+b] in evalDict:
                        bite = b; break
                else: assert(False) #Logic error - loop above should always exit when b == 1

                #print "DB: start=", start, ": found ", gateString[start:start+bite], " in evalDict"
                #iInFinal = k if bool(start + bite == L) else -1
                bFinal = bool(start + bite == L)

                if start == 0: #first in-evalDict bite - no need to add anything to self yet
                    iCur = evalDict[ gateString[0:bite] ]
                    if bFinal: assert(iCur == k) #make sure this bite is in the right place!
                else:
                    # add (iCur, iBite)
                    assert(gateString[0:start+bite] not in evalDict)
                    iBite = evalDict[ gateString[start:start+bite] ]
                    if bFinal: #place (iCur, iBite) at location k
                        iNew = k
                        evalDict[ gateString[0:start+bite] ] = iNew
                        assert(self[iNew] is None) #make sure we haven't put anything here yet
                        self[k] = (iCur, iBite)
                    else:
                        iNew = len(self)
                        evalDict[ gateString[0:start+bite] ] = iNew
                        self.append( (iCur,iBite) )

                    #print "DB: appending %s (index %d)" % (str(gateString[0:start+bite]),iNew)
                    self.eval_order.append(iNew)
                    iCur = iNew
                start += bite
                #nBites += 1

            #if nBites > 0: avgBiteSize += L / float(nBites)

        #avgBiteSize /= float(len(gatestring_list))
        #print "DEBUG: Avg bite size = ",avgBiteSize

        #see if there are superfluous tree nodes: those with iFinal == -1 and
        self.myFinalToParentFinalMap = None #this tree has no "children",
        self.parentIndexMap = None          # i.e. has not been created by a 'split'
        self.original_index_lookup = None
        self.subTrees = [] #no subtrees yet


    def initialize_orig(self, gateLabels, gatestring_list, numSubTreeComms=1):
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
        self.gateLabels = gateLabels
        if numSubTreeComms is not None:
            self.distribution['numSubtreeComms'] = numSubTreeComms

        if len(gatestring_list ) > 0 and isinstance(gatestring_list[0],_gs.GateString):
            gatestring_list = [gs.tup for gs in gatestring_list]

        #Evaluation dictionary:
        # keys == gate strings that have been evaluated so far
        # values == index of gate string (key) within evalTree
        evalDict = { }

        #Evaluation tree:
        # A list of tuples, where each element contains
        #  information about evaluating a particular gate string:
        #  (iLeft, iRight, iInFinalList)
        # and the order of the elements specifies the evaluation order.
        # In particular, the gateString = evalTree[iLeft] + evalTree[iRight]
        #   so that matrix(gateString) = matrixOf(evalTree[iRight]) * matrixOf(evalTree[iLeft])
        #  and iInFinalList is the index of this gatestring in the gatestring_list
        # passed to this function, or -1 if it is not in the list.
        del self[:] #clear self (a list)

        #Final Index List
        # A list of integers whose i-th element is the index into evalTree
        #  corresponding to the i-th gatestring in gatestring_list.
        finalIndxList = [ None ] * len(gatestring_list)

        #Single gate (or zero-gate) computations are assumed to be atomic, and be computed independently.
        #  These labels serve as the initial values, and each gate string is assumed to be a tuple of gate labels.
        for gateLabel in self.gateLabels:
            if gateLabel == "": #special case of empty label == no gate
                evalDict[ () ] = len(self)
            else:
                evalDict[ (gateLabel,) ] = len(self)
            self.append( (None,None,-1) ) #iLeft = iRight = None for always-evaluated zero string

        #avgBiteSize = 0
        #useCounts = {}
        for (k,gateString) in enumerate(gatestring_list):
            L = len(gateString)
            if L == 0:
                self[0] = (None,None,k) #set in-final-list index for zero-length string (special case)
                finalIndxList[k] = 0

            start = 0; bite = 1
            #nBites = 0
            #print "DB: string = ",gateString, " tree length = ",len(self)

            while start < L:
                for b in range(L-start,0,-1):
                    if gateString[start:start+b] in evalDict:
                        bite = b; break
                else: assert(False) #Logic error - loop above should always exit when b == 1

                #print "DB: start=", start, ": found ", gateString[start:start+bite], " in evalDict"
                iInFinal = k if bool(start + bite == L) else -1

                if start == 0: #first in-evalDict bite - no need to add anything to self yet
                    iCur = evalDict[ gateString[0:bite] ]
                    if iInFinal >= 0:
                        if self[iCur][2] == -1: #set in-final-list index in existing tree node
                            self[iCur] = (self[iCur][0],self[iCur][1],iInFinal)
                        finalIndxList[iInFinal] = iCur
                else:
                    assert(gateString[0:start+bite] not in evalDict)
                    iBite = evalDict[ gateString[start:start+bite] ]
                    iNew  = len(self)
                    evalDict[ gateString[0:start+bite] ] = iNew
                    self.append( (iCur,iBite,iInFinal) )
                    #print "DB: appending %s (index %d)" % (str(gateString[0:start+bite]),iNew)
                    if iInFinal >= 0: finalIndxList[iInFinal] = iNew
                    iCur = iNew
                start += bite
                #nBites += 1

            #if nBites > 0: avgBiteSize += L / float(nBites)

        #avgBiteSize /= float(len(gatestring_list))
        #print "DEBUG: Avg bite size = ",avgBiteSize

        #see if there are superfluous tree nodes: those with iFinal == -1 and
        self.finalList = finalIndxList
        self.inplaceCopyList = build_permute_copy_order(finalIndxList)
        self.myFinalToParentFinalMap = None #this tree has no "children",
        self.parentIndexMap = None          # i.e. has not been created by a 'split'
        self.subTrees = [] #no subtrees yet        


    def copy(self):
        """ Create a copy of this evaluation tree. """
        newTree = EvalTree(self[:])
        newTree.gateLabels = self.gateLabels[:]
        newTree.init_indices = self.init_indices[:]
        newTree.eval_order = self.eval_order[:]
        #newTree.finalList = self.finalList
        #newTree.inplaceCopyList = self.inplaceCopyList
        newTree.myFinalToParentFinalMap = self.myFinalToParentFinalMap
        newTree.parentIndexMap = self.parentIndexMap[:]
        newTree.subTrees = [ st.copy() for st in self.subTrees ]
        newTree.original_index_lookup = self.original_index_lookup[:]
        return newTree

    def get_init_labels(self):
        """ Return a tuple of the gate labels (strings)
            which form the beginning of the tree.
        """
        return tuple(self.gateLabels)

    def get_init_indices(self):
        """ Return a tuple of the indices corresponding
             to the initial gate labels (strings)
             which form the beginning of the tree.
        """
        return tuple(self.init_indices)

    def get_evaluation_order(self):
        """ Return a list of indices specifying the
             order in which elements of this EvalTree
             should be visited when doing a computation
             (after computing the initial indices).
        """
        return self.eval_order

    def final_view(self, a, axis=None):
        """ TODO: docstring """
        #OLD (which *copies* instead of just returning a view)
        #finalIndxList = self.get_list_of_final_value_tree_indices()
        #return a.take(finalIndxList, axis=axis )
        
        if axis is None:
            return a[0:self.num_final_strings()]
        else:
            sl = [slice(None)] * a.ndim
            sl[axis] = slice(0,self.num_final_strings())
            ret = a[sl]
            assert(ret.base is a) #check that what is returned is a view
            assert(_np.may_share_memory(ret,a))
            return ret

    def final_slice(self):
        """ TODO: docstring """
        if self.myFinalToParentFinalMap is not None:
            return self.myFinalToParentFinalMap
        else:
            return slice(0,self.num_final_strings())
        



    #def get_tree_index_of_final_value(self, finalValueIndex):
    #    """
    #    Return the index within the tree list of the
    #      gate string that had index finalValueIndex in
    #      the gatestring_list passed to initialize.
    #    """
    #    return self.finalList[finalValueIndex]

    #def get_list_of_final_value_tree_indices(self, returnCopyList=False):
    #    """
    #    Get a list of indices (ints) which specifying the
    #      tree indices corresponding to each gate string
    #      in the gatestring_list passed to initialize.
    #
    #    Parameters
    #    ----------
    #    returnCopyList : bool, optional
    #       When true, returns list of copy operations in addition to a
    #       list of indices which give instructions for in-place "taking"
    #       of the final indices from an array containing computed values
    #       for all of this tree's indices.
    #
    #
    #    Returns
    #    -------
    #    indexList : list
    #        List of indices with length len(gatestring_list passed to initialize).
    #
    #    copyList : list
    #        A list of (iDest,iSource) 2-tuples specifying an order for in-place
    #        copying that will transform an array of computed tree values to one where
    #        the leading len(indexList) values correspond to indexList.  `copyList`
    #        is meant to be give as an argument to `inplace_take` to perform an
    #        in-place numpy.take operation.
    #
    #    Note
    #    ----
    #    A reference to the EvalTree's internal
    #      list is returned, and so the caller should copy
    #      this list before modifying it.
    #    """
    #
    #    #Note: no copy of returned lists, so caller could modify them!
    #    if returnCopyList:
    #        return self.finalList, self.inplaceCopyList
    #    else:
    #        return self.finalList


#    def _initializeBETA(self, gateLabels, gatestring_list):
#        """
#          Experimental alternate tree initialization algorithm.
#          This doesn't currently work.
#        """
#        self.gateLabels = gateLabels
#
#        def repetitions(s):
#            r = _re.compile(r"(.+?)\1+")
#            for match in r.finditer(s):
#                yield (match.group(1), len(match.group(0))/len(match.group(1)))
#
#        #Evaluation dictionary:
#        # keys == gate strings that have been evaluated so far
#        # values == index of gate string (key) within evalTree
#        evalDict = { }
#
#        #Evaluation tree:
#        # A list of tuples, where each element contains
#        #  information about evaluating a particular gate string:
#        #  (iLeft, iRight, iInFinalList)
#        # and the order of the elements specifies the evaluation order.
#        # In particular, the gateString = evalTree[iLeft] + evalTree[iRight]
#        #  and iInFinalList is the index of this gatestring in the gatestring_list
#        # passed to this function, or -1 if it is not in the list.
#        del self[:] #clear self (a list)
#
#        #Final Index List
#        # A list of integers whose i-th element is the index into evalTree
#        #  corresponding to the i-th gatestring in gatestring_list.
#        finalIndxList = [ None ] * len(gatestring_list)
#
#        #Single gate (or zero-gate) computations are assumed to be atomic, and be computed independently.
#        #  These labels serve as the initial values, and each gate string is assumed to be a tuple of gate labels.
#        singleGateLabels = []
#        for gateLabel in self.gateLabels:
#            if gateLabel == "": #special case of empty label == no gate
#                evalDict[ () ] = len(self)
#            else:
#                evalDict[ (gateLabel,) ] = len(self)
#                singleGateLabels.append(gateLabel)
#            self.append( (None,None,-1) ) #iLeft = iRight = None for always-evaluated zero string
#
#        #Collect list of what sub-strings get repeated a lot
#        repDict = {}
#        for (k,gateString) in enumerate(gatestring_list):
#            #print "String %d (len %d): " % (k,len(gateString)),
#            for repStr,repCnt in repetitions( gateString.to_pythonstr(singleGateLabels) ):
#                repGateStr = _gs.GateString.from_pythonstr(repStr,singleGateLabels)
#                if repDict.has_key(repGateStr):
#                    if repCnt not in repDict[repGateStr][0]:
#                        repDict[repGateStr][0].append(repCnt)
#                        repDict[repGateStr][1].append(1)
#                    else:
#                        indx = repDict[repGateStr][0].index(repCnt)
#                        repDict[repGateStr][1][indx] += 1
#                else:
#                    repDict[repGateStr] = [ [repCnt],[1] ] #list of unique rep counts, list of multiplicities
#                #print "%s^%d" % (str(repGateStr),repCnt),
#            #print ""
#        for repGateStr,repCntList in repDict.iteritems():
#            print repGateStr, ":", repCntList
#
#        self.finalList = []
#        self.myFinalToParentFinalMap = None #this tree has no "children", i.e. has not been created by a 'split'
#        self.subTrees = []

    def num_final_strings(self):
        """
        Returns the integer number of "final" gate strings, equal
          to the length of the gatestring_list passed to initialize.
        """
        return self.num_final_strs

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

        #Set "initial" (single- or zero- gate) strings
        for i,gateLabel in zip(self.get_init_indices(), self.get_init_labels()):
            if gateLabel == "": gateStrings[i] = () #special case of empty label
            else: gateStrings[i] = (gateLabel,)

        #Build rest of strings
        for i in self.get_evaluation_order():
            iLeft, iRight = self[i]
            gateStrings[i] = gateStrings[iLeft] + gateStrings[iRight]
            
        #Permute to get final list:
        nFinal = self.num_final_strings()
        if self.original_index_lookup is not None and permute == True:
            finalGateStrings = [None]*nFinal
            for iorig,icur in self.original_index_lookup.items():
                if iorig < nFinal: finalGateStrings[iorig] = gateStrings[icur]
            return finalGateStrings
        else:
            return gateStrings[0:nFinal]

    def permute_original_to_computation(self, a):
        """
        Permute an array's elements using mapping from the "original"
        gate string ordering to the "computation" ordering.
        
        This function converts arrays with elements corresponding
        to gate strings in the "original" ordering (i.e. the 
        ordering in the list passed to `initialize(...)`) to the
        ordering used in tree computation (i.e. by a `GateSet`'s
        bulk computation routines).

        Paramters
        ---------
        a : numpy array
           The array whose permuted elements are returned.

        Returns
        -------
        numpy array
        """
        assert(a.shape[0] == self.num_final_strings())
        nFinal = self.num_final_strings()
        ret = a.copy()

        if self.original_index_lookup is not None:
            for iorig,icur in self.original_index_lookup.items():
                if iorig < nFinal: ret[icur] = a[iorig]

        return ret


    def permute_computation_to_original(self, a):
        """
        Permute an array's elements using mapping from the "computation"
        gate string ordering to the "original" ordering.
        
        This function converts arrays with elements corresponding
        to gate strings in the ordering used in tree computation 
        (i.e. the ordering returned by `GateSet`'s bulk computation routines)
        to the "original" ordering (i.e. the ordering of the gate string list
        passed to `initialize(...)`).

        Paramters
        ---------
        a : numpy array
           The array whose permuted elements are returned.

        Returns
        -------
        numpy array
        """
        assert(a.shape[0] == self.num_final_strings())
        nFinal = self.num_final_strings()
        ret = a.copy()

        if self.original_index_lookup is not None:
            for iorig,icur in self.original_index_lookup.items():
                if iorig < nFinal: ret[iorig] = a[icur]

        return ret
        


#Future MPI API??
#    def is_distributed(self):
#        pass
#
#    #get_sub_trees => returns only *local* subtrees (call these "branches"?)
#    # OR add "get_local_tree(...)" -- then calc routines always, only
#    #  compute that single tree, then ask if they need to gather in the end?
#

    def distribute(self, comm, verbosity=0):
        # split tree into local subtrees, each which contains one/group of
        # processors (group can then parallelize derivative calcs over
        # gate set parameters) 

        rank = 0 if (comm is None) else comm.Get_rank()
        nprocs = 1 if (comm is None) else comm.Get_size()
        nSubtreeComms = self.distribution.get('numSubtreeComms',1)
        nSubtrees = len(self.get_sub_trees())

        assert(nSubtreeComms <= nprocs) # => len(mySubCommIndices) == 1
        mySubCommIndices, subCommOwners, mySubComm = \
            _mpit.distribute_indices(list(range(nSubtreeComms)), comm)
        assert(len(mySubCommIndices) == 1)
        mySubCommIndex = mySubCommIndices[0]

        assert(nSubtreeComms <= nSubtrees) # don't allow more comms than trees
        mySubtreeIndices, subTreeOwners = _mpit.distribute_indices_base(
            list(range(nSubtrees)), nSubtreeComms, mySubCommIndex)

        # subTreeOwners contains index of owner subComm, but we really want
        #  the owning processor, i.e. the owner of the subComm
        subTreeOwners = { iSubTree: subCommOwners[subTreeOwners[iSubTree]]
                          for iSubTree in subTreeOwners }

        printer = VerbosityPrinter.build_printer(verbosity, comm)
        printer.log("*** Distributing %d subtrees into %d sub-comms (%s processors) ***"% \
                        (nSubtrees, nSubtreeComms, nprocs))

        return mySubtreeIndices, subTreeOwners, mySubComm


    def get_min_tree_size(self):
        """
        Returns the minimum sub tree size required to compute each
        of the tree entries individually.  This minimum size is the
        smallest "maxSubTreeSize" that can be passed to split(),
        as any smaller value will result in at least one entry being
        uncomputable.
        """
        singleItemTreeSetList = self._createSingleItemTrees()
        return max(list(map(len,singleItemTreeSetList)))

    def split(self, maxSubTreeSize=None, numSubTrees=None):
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

        Returns
        -------
        None
        """
        dbList = self.generate_gatestring_list()

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

        #First pass - identify which indices go in which subtree
        #   Part 1: create disjoint set of subtrees generated by single items
        singleItemTreeSetList = self._createSingleItemTrees()
          #each element represents a subtree, and
          # is a set of the indices owned by that subtree
        nSingleItemTrees = len(singleItemTreeSetList)

        #   Part 2: determine whether we need to split/merge "single" trees
        if numSubTrees is not None:

            #Merges: find the best merges to perform if any are required
            if nSingleItemTrees > numSubTrees:

                #find trees that have least intersection to begin
                intersectSizes = {}
                for i in range(nSingleItemTrees):
                    s1 = singleItemTreeSetList[i]
                    for j in range(i+1,nSingleItemTrees):
                        s2 = singleItemTreeSetList[j]
                        intersectSizes[(i,j)] = len(s1.intersection(s2))

                sortedIntersects = sorted(iter(intersectSizes.items()),
                                            key=lambda x: x[1])
                iStartingTrees = []
                for (i,j),_ in sortedIntersects:
                    if i in iStartingTrees or j in iStartingTrees: continue
                    iStartingTrees.append(i)
                    if len(iStartingTrees) == numSubTrees:  break
                    iStartingTrees.append(j)
                    if len(iStartingTrees) == numSubTrees:  break
                else:
                    raise ValueError("Could not find set of starting trees!")
                subTreeSetList = [singleItemTreeSetList[i] for i in iStartingTrees]
                assert(len(subTreeSetList) == numSubTrees)

                indicesLeft = list(range(nSingleItemTrees))
                for i in iStartingTrees:
                    del indicesLeft[indicesLeft.index(i)]

                while len(indicesLeft) > 0:
                    iToMergeInto = _np.argmin(list(map(len,subTreeSetList)))
                    setToMergeInto = subTreeSetList[iToMergeInto]
                    intersectionSizes = [ len(setToMergeInto.intersection(
                                singleItemTreeSetList[i])) for i in indicesLeft ]
                    iMaxIntsct = _np.argmax(intersectionSizes)
                    setToMerge = singleItemTreeSetList[indicesLeft[iMaxIntsct]]
                    subTreeSetList[iToMergeInto] = \
                          subTreeSetList[iToMergeInto].union(setToMerge)
                    del indicesLeft[iMaxIntsct]

                assert(len(subTreeSetList) == numSubTrees)

            #Splits:
            else:
                #Splits: find the best splits to perform
                #TODO: how to split a tree intelligently -- for now, just do
                # trivial splits by making empty trees.
                subTreeSetList = singleItemTreeSetList[:]
                nSplitsNeeded = numSubTrees - nSingleItemTrees
                while nSplitsNeeded > 0:
                    # LATER...
                    # for iSubTree,subTreeSet in enumerate(subTreeSetList):
                    subTreeSetList.append( [] ) # create empty subtree
                    nSplitsNeeded -= 1

        else:
            assert(maxSubTreeSize is not None)
            subTreeSetList = []

            #Merges: find the best merges to perform if any are allowed given
            # the maximum tree size
            for singleItemTreeSet in singleItemTreeSetList:
                if len(singleItemTreeSet) > maxSubTreeSize:
                    raise ValueError("Max. sub tree size (%d) is too low (<%d)!"
                                   % (maxSubTreeSize, self.get_min_tree_size()))

                #See if we should merge this single-item-generated tree with
                # another one or make it a new subtree.
                newTreeSize = len(singleItemTreeSet)
                maxIntersectSize = None; iMaxIntersectSize = None
                for k,existingSubTreeSet in enumerate(subTreeSetList):
                    mergedSize = len(existingSubTreeSet) + newTreeSize
                    if mergedSize <= maxSubTreeSize:
                        intersectionSize = \
                            len(singleItemTreeSet.intersection(existingSubTreeSet))
                        if maxIntersectSize is None or \
                                maxIntersectSize < intersectionSize:
                            maxIntersectSize = intersectionSize
                            iMaxIntersectSize = k

                if iMaxIntersectSize is not None:
                    # then we merge the new tree with this existing set
                    subTreeSetList[iMaxIntersectSize] = \
                      subTreeSetList[iMaxIntersectSize].union(singleItemTreeSet)
                else: # we create a new subtree
                    subTreeSetList.append( singleItemTreeSet )

        #TODO: improve tree efficiency via better splitting?
        #print "DEBUG TREE SPLITTING:"
        #for k,dbTreeSet in enumerate(subTreeSetList):
        #    print "Tree %d (size %d): " % (k,len(dbTreeSet)), [ len(dbTreeSet.intersection(x)) for kk,x in enumerate(subTreeSetList) if kk != k ]
        #cnts = [0]*len(self)
        #for k,dbTreeSet in enumerate(subTreeSetList):
        #    for i in dbTreeSet:
        #        cnts[i] += 1
        #sorted_cnts = sorted( list(enumerate(cnts)), key=lambda x: x[1], reverse=True)
        #print "Top index : cnts"
        #for ii,(i,cnt) in enumerate(sorted_cnts):
        #    print ii,":", i,", ",cnt
        #raise ValueError("STOP")
                    
        from mpi4py import MPI
        rank = -1 #MPI.COMM_WORLD.Get_rank()
        if rank == 0: print("Parent nFinal = ",self.num_final_strings(), " len=",len(self))

        #Second pass - create subtrees from index sets
        need_to_compute = _np.zeros( len(self), 'bool' )
        need_to_compute[0:self.num_final_strings()] = True

          #First, reorder the parent tree's elements so that the final
          # elements of the subtrees map to contiguous slices of the
          # parent tree's final elements.
        parentIndexRevPerm = [] # parentIndexRevPerm[newIndex] = currentIndex (i.e. oldIndex)
        subTreeIndicesList = []
        numFinalList = []
        for iSubTree,subTreeSet in enumerate(subTreeSetList):
            subTreeIndices = list(subTreeSet)
            if rank==0: print("SUBTREE0: %s (len=%d)" % (str(subTreeIndices),len(subTreeIndices)))
            if rank==0: print("  NEED: %s" % ",".join([ "1" if b else "0" for b in need_to_compute]))
            subTreeIndices.sort() # order subtree gatestrings (really just their indices) so
                                  # that all "final" strings come first.
            #Compute # of "final" strings in this subtree (count # of indices < num_final_strs)
            subTreeNumFinal = _np.sum(_np.array(subTreeIndices) < self.num_final_strings())

            #Swap the indices of "final" strings that have already been computed past the end
            # of the "final strings" region of the subtree's list (i.e. the subtree itself).
            already_computed = _np.logical_not( need_to_compute[ subTreeIndices[0:subTreeNumFinal] ] )
            already_computed_inds = _np.nonzero(already_computed)[0] # (sorted ascending)
            if rank==0: print("SUBTREE1: %s (nFinal=%d - %d)" % (str(subTreeIndices),
                                                     subTreeNumFinal, len(already_computed_inds)))
            if rank==0: print("  - already computed = ", [subTreeIndices[i] for i in already_computed_inds])

            iFirstNonFinal = subTreeNumFinal
            for k in already_computed_inds:
                if k >= iFirstNonFinal: continue #already a non-final el
                elif k == iFirstNonFinal-1: #index is last "final" el - just shift boundary
                    iFirstNonFinal -= 1 #now index is "non-final"
                else: # k < iFirstNonFinal-1, so find a desired "final" el at boundary to swap it with
                    iLastFinal = iFirstNonFinal-1
                    while iLastFinal >= 0 and (iLastFinal in already_computed_inds):
                        iLastFinal -= 1 #the element at iLastFinal happens to be one that we wanted to be non-final, so remove it
                    assert(iLastFinal >= 0) #Logic error!
                    subTreeIndices[iLastFinal],subTreeIndices[k] = \
                        subTreeIndices[k],subTreeIndices[iLastFinal] #Swap k <-> iLastFinal
                    iFirstNonFinal = iLastFinal # move boundary to make k's new location non-final

            subTreeNumFinal = iFirstNonFinal # the final <-> non-final boundary
            parentIndexRevPerm.extend( subTreeIndices[0:subTreeNumFinal] )
            subTreeIndicesList.append( subTreeIndices )
            numFinalList.append( subTreeNumFinal )
            need_to_compute[ subTreeIndices[0:subTreeNumFinal] ] = False
            if rank==0: print("FINAL SUBTREE: %s (nFinal=%d)" % (str(subTreeIndices),subTreeNumFinal))
                    
        #Permute parent tree indices according to parentIndexPerm
        assert(len(parentIndexRevPerm) == self.num_final_strings())
        parentIndexRevPerm.extend( list(range(self.num_final_strings(), len(self))) ) 
          #don't permute non-final indices (no need)
        
        #Create forward permutation map: currentIndex -> newIndex
        parentIndexPerm = [ None ] * len(parentIndexRevPerm)
        for inew,icur in enumerate(parentIndexRevPerm):
            parentIndexPerm[icur] = inew
        assert( None not in parentIndexPerm) #all indices should be mapped somewhere!
        assert( self.original_index_lookup is None )
        self.original_index_lookup = { icur: inew for inew,icur in enumerate(parentIndexRevPerm) }

        if rank==0: print("PERM REV MAP = ", parentIndexRevPerm)
        if rank==0: print("PERM MAP = ", parentIndexPerm)

        #Permute parent indices
        self.init_indices = [ parentIndexPerm[iCur] for iCur in self.init_indices ]
        self.eval_order = [ parentIndexPerm[iCur] for iCur in self.eval_order ]
        self[:] = [ (parentIndexPerm[self[iCur][0]] if (self[iCur][0] is not None) else None,
                     parentIndexPerm[self[iCur][1]] if (self[iCur][1] is not None) else None)
                    for iCur in parentIndexRevPerm ]
        assert(self.myFinalToParentFinalMap is None)
        assert(self.parentIndexMap is None)


        #Update indices in subtree lists
        newList = []; finalSlices = []; sStart = 0
        for subTreeIndices,numFinal in zip(subTreeIndicesList,numFinalList):
            newSubTreeIndices = [ parentIndexPerm[i] for i in subTreeIndices ] 
            assert(newSubTreeIndices[0:numFinal] == list(range(sStart,sStart+numFinal)))
              #final elements should be a sequential slice of parent indices
            finalSlices.append( slice(sStart,sStart+numFinal) )
            newList.append(newSubTreeIndices)
            sStart += numFinal #increment slice start position
        subTreeIndicesList = newList

        #for iSubTree,subTreeSet in enumerate(subTreeSetList):
            #subTreeIndices = list(subTreeSet)
            #subTreeIndices.sort() # order subtree gatestrings (really just their indices) so
            #                      # that all "final" strings come first.
            #
            ##Compute # of "final" strings in this subtree (count # of indices < num_final_strs)
            #subTreeNumFinal = _np.sum(_np.array(subTreeIndices) < self.num_final_strings())

        for iSubTree,(subTreeIndices,subTreeNumFinal,slc) \
                in enumerate(zip(subTreeIndicesList,numFinalList,finalSlices)):

            fullEvalOrder = self.init_indices + self.eval_order #list concatenation
            subTreeEvalOrder = sorted(range(len(subTreeIndices)),
                                      key=lambda j: fullEvalOrder.index(subTreeIndices[j]))
            mapIndxToSubTreeIndx = { k: ik for ik,k in enumerate(subTreeIndices) }

            subTree = EvalTree()
            subTree.myFinalToParentFinalMap = slc
            subTree.num_final_strs = subTreeNumFinal
            subTree[:] = [None]*len(subTreeIndices)

            for ik in subTreeEvalOrder:
                k = subTreeIndices[ik] #original tree index
                (oLeft,oRight) = self[k] #original tree indices

                if (oLeft is None) and (oRight is None):
                    iLeft = iRight = None
                    #assert(len(subTree.gateLabels) == len(subTree)) #make sure all gatelabel items come first
                    subTree.gateLabels.append( self.gateLabels[ 
                            self.init_indices.index(k)] )
                    subTree.init_indices.append(ik)
                else:
                    iLeft  = mapIndxToSubTreeIndx[ oLeft ]
                    iRight = mapIndxToSubTreeIndx[ oRight ]
                    subTree.eval_order.append(ik)

                assert(subTree[ik] is None)
                subTree[ik] = (iLeft,iRight)

                #if ik < subTreeNumFinal:
                #    assert(k < self.num_final_strings()) # it should be a final element in parent too!
                #    subTree.myFinalToParentFinalMap[ik] = k

            subTree.parentIndexMap = subTreeIndices #parent index of each subtree index
            self.subTrees.append( subTree )
            if rank==0: print("NEW SUBTREE: indices=%s, len=%d, nFinal=%d"  %
                              (subTree.parentIndexMap, len(subTree), subTree.num_final_strings()))

        dbList2 = self.generate_gatestring_list()
        if rank==0: print("DBLIST = ",dbList)
        if rank==0: print("DBLIST2 = ",dbList2)
        assert(dbList == dbList2)

        return



    def is_split(self):
        """ Returns boolean indicating whether tree is split into sub-trees or not. """
        return len(self.subTrees) > 0

    def get_sub_trees(self):
        """
        Returns a list of all the sub-trees (also EvalTree instances) of
          this tree.  If this tree is not split, returns a single-element
          list containing just the tree.
        """
        if self.is_split():
            return self.subTrees
        else:
            return [self] #return self as the only "subTree" when not split

    def _walkSubTree(self,indx,out):
        if indx not in out: out.append(indx)
        (iLeft,iRight) = self[indx]
        if iLeft is not None: self._walkSubTree(iLeft,out)
        if iRight is not None: self._walkSubTree(iRight,out)

    def _createSingleItemTrees(self):
        #  Create disjoint set of subtrees generated by single items
        need_to_compute = _np.zeros( len(self), 'bool' )
        need_to_compute[0:self.num_final_strings()] = True

        singleItemTreeSetList = [] #each element represents a subtree, and
                            # is a set of the indices owned by that subtree
        for i in reversed(range(self.num_final_strings())):
            if not need_to_compute[i]: continue # move to the last element
              #of evalTree that needs to be computed (i.e. is not in a subTree)

            subTreeIndices = [] # create subtree for uncomputed item
            self._walkSubTree(i,subTreeIndices)
            newTreeSet = set(subTreeIndices)
            for k in subTreeIndices:
                need_to_compute[k] = False #mark all the elements of
                                           #the new tree as computed

            # Add this single-item-generated tree as a new subtree. Later
            #  we merge and/or split these trees based on constraints.
            singleItemTreeSetList.append( newTreeSet )
        return singleItemTreeSetList


    def print_analysis(self):
        """
        Print a brief analysis of this tree. Used for
        debugging and assessing tree quality.
        """

        #Analyze tree
        if not self.is_split():
            print("Size of evalTree = %d" % len(self))
            print("Size of gatestring_list = %d" % self.num_final_strings())

            #TODO: maybe revive this later if it ever becomes useful again.
            # Currently left un-upgraded after evaltree was changed to hold
            # all final indices at its beginning. (need to iterate in eval_order
            #  not not just over enumerate(self) as is done below).
            #lastOccurrance = [-1] * len(self)
            #nRefs = [-1] * len(self)
            #for i,tup in enumerate(self):
            #    iLeft,iRight = tup
            #
            #    if iLeft is not None:
            #        nRefs[iLeft] += 1
            #        lastOccurrance[iLeft] = i
            #
            #    if iRight is not None:
            #        nRefs[iRight] += 1
            #        lastOccurrance[iRight] = i
            #
            ##print "iTree  nRefs lastOcc  iFinal  inUse"
            #maxInUse = nInUse = 0
            #for i,tup in enumerate(self):
            #    nInUse += 1
            #    for j in range(i):
            #        if lastOccurrance[j] == i and self[j][2] == -1: nInUse -= 1
            #    maxInUse = max(maxInUse,nInUse)
            #    #print "%d  %d  %d  %d  %d" % (i, nRefs[i], lastOccurrance[i], tup[2], nInUse)
            #print("Max in use at once = (smallest tree size for mem) = %d" % maxInUse)

        else: #tree is split
            print("Size of original tree = %d" % len(self))
            print("Size of original gatestring_list = %d" % self.num_final_strings())
            print("Tree is split into %d sub-trees" % len(self.subTrees))
            print("Sub-tree lengths = ", list(map(len,self.subTrees)), " (Sum = %d)" % sum(map(len,self.subTrees)))
            for i,t in enumerate(self.subTrees):
                print(">> sub-tree %d: " % i)
                t.print_analysis()


    def get_analysis_plot_infos(self):
        """
        Returns debug plot information useful for
        assessing the quality of a tree. This
        function is not guaranteed to work.
        """

        analysis = {}
        firstIndxSeen = list(range(len(self)))
        lastIndxSeen = list(range(len(self)))
        subTreeSize = [-1]*len(self)

        xs = []; ys = []
        for i in range(len(self)):
            subTree = []
            self._walkSubTree(i,subTree)
            subTreeSize[i] = len(subTree)
            ys.extend( [i]*len(subTree) + [None] )
            xs.extend( list(sorted(subTree) + [None]) )

            for k,t in enumerate(self):
                (iLeft,iRight,_) = t
                if i in (iLeft,iRight):
                    lastIndxSeen[i] = k

        analysis['SubtreeUsagePlot'] = { 'xs': xs, 'ys': ys, 'title': "Indices used by the subtree rooted at each index",
                                                'xlabel': "Indices used", 'ylabel': 'Subtree root index' }
        analysis['SubtreeSizePlot'] = { 'xs': list(range(len(self))), 'ys': subTreeSize, 'title': "Size of subtree rooted at each index",
                                                'xlabel': "Subtree root index", 'ylabel': 'Subtree size' }

        xs = [];  ys = []
        for i,rng in enumerate(zip(firstIndxSeen,lastIndxSeen)):
            ys.extend( [i,i,None] )
            xs.extend( [rng[0],rng[1],None] )
        analysis['IndexUsageIntervalsPlot'] = { 'xs': xs, 'ys': ys, 'title': "Usage Intervals of each index",
                                                'xlabel': "Index Interval", 'ylabel': 'Index' }

        return analysis
