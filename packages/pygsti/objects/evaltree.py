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
import time as _time #DEBUG TIMERS

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
        # list of the gate labels
        self.gateLabels = []

        # indices for initial computation that is viewed separately
        # from the "main evaluation" given by eval_order
        self.init_indices = []

        # list of indices specifying what order they should be evaluated in,
        # *after* evaluating all of the initial indices (init_indices)
        self.eval_order = []

        # Number of "final" or "requested" strings, which may be less
        # then len(self) since some un-requested results may be stored
        # as useful intermediates.
        self.num_final_strs = 0

        # The list of "child" sub-trees (if this tree is spilt)
        self.subTrees = []

        # a dict to hold various MPI distribution info
        self.distribution = {} 
                
        # ********* Only non-None for sub-trees ******************
        
          # The mapping between this tree's final indices and its parent's
        self.myFinalToParentFinalMap = None
          # The parent's index of each of this tree's indices
        self.parentIndexMap = None
          # A dictionary whose keys are the "original" (as-given to initialize)
          # indices and whose values are the new "permuted" indices.  So if you
          # want to know where in a tree the ith-element of gatestring_list (as
          # passed to initialize(...) is, it's at index original_index_lookup[i]
        self.original_index_lookup = None
        
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
        raise NotImplementedError("initialize(...) must be implemented by a derived class") 


    def _copyBase(self,newTree):
        """ copy EvalTree members to a new tree (used by derived classes "copy" fns) """
        newTree.gateLabels = self.gateLabels[:]
        newTree.init_indices = self.init_indices[:]
        newTree.eval_order = self.eval_order[:]
        newTree.num_final_strs = self.num_final_strs
        newTree.myFinalToParentFinalMap = self.myFinalToParentFinalMap
        newTree.parentIndexMap = self.parentIndexMap[:] \
            if (self.parentIndexMap is not None) else None
        newTree.subTrees = [ st.copy() for st in self.subTrees ]
        newTree.original_index_lookup = self.original_index_lookup[:] \
            if (self.original_index_lookup is not None) else None
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
        """ 
        Returns a view of array `a` restricting it to only the
        *final* results computed by this tree (not the intermediate
        results).

        Parameters
        ----------
        a : ndarray
            An array of results computed using this EvalTree, 
            such that the `axis`-th dimension equals the full
            length of the tree.  The other dimensions of `a` are
            unrestricted.

        axis : int, optional
            Specified the axis along which the selection of the
            final elements is performed. If None, than this 
            selection if performed on flattened `a`.

        Returns
        -------
        ndarray
            Of the same shape as `a`, except for along the 
            specified axis, whose dimension has been reduced
            to filter out the intermediate (non-final) results.
        """
        if axis is None:
            return a[0:self.num_final_strings()]
        else:
            sl = [slice(None)] * a.ndim
            sl[axis] = slice(0,self.num_final_strings())
            ret = a[sl]
            assert(ret.base is a or ret.base is a.base) #check that what is returned is a view
            assert(ret.size == 0 or _np.may_share_memory(ret,a))
            return ret


    def final_slice(self, parent_tree):
        """
        Return a slice that identifies the segment of `parent_tree`'s
        final values that correspond to this tree's final values.

        Note that if `parent_tree` is not None, this tree must be a
        sub-tree of it (which means `parent_tree` is split).

        Parameters
        ----------
        parent_tree : EvalTree
            This tree's parent tree.  If the parent tree is None
            or isn't split (which sometimes means that this 
            "sub-tree" is the same as the parent tree - see 
            `get_sub_trees`), then a slice for the entire set of
            final values is returned, which is appropriate in this
            case.

        Returns
        -------
        slice
        """
        if (self.myFinalToParentFinalMap is not None) and \
                parent_tree.is_split():
            return self.myFinalToParentFinalMap
        else:
            return slice(0,self.num_final_strings())
        

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
        raise NotImplementedError("generate_gatestring_list(...) not implemented!")


    def permute_original_to_computation(self, a, axis=0):
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

        axis : int, optional
           The axis to permute.  By default, the first dimension is used.

        Returns
        -------
        numpy array
        """
        assert(a.shape[axis] == self.num_final_strings())
        nFinal = self.num_final_strings()
        ret = a.copy()

        def mkindx(i):
            mi = [slice(None)]*a.ndim; mi[axis] = i
            return mi

        if self.original_index_lookup is not None:
            for iorig,icur in self.original_index_lookup.items():                
                if iorig < nFinal: 
                    ret[mkindx(icur)] = a[mkindx(iorig)]

        return ret


    def permute_computation_to_original(self, a, axis=0):
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

        axis : int, optional
           The axis to permute.  By default, the first dimension is used.

        Returns
        -------
        numpy array
        """
        assert(a.shape[axis] == self.num_final_strings())
        nFinal = self.num_final_strings()
        ret = a.copy()

        def mkindx(i):
            mi = [slice(None)]*a.ndim; mi[axis] = i
            return mi

        if self.original_index_lookup is not None:
            for iorig,icur in self.original_index_lookup.items():                
                if iorig < nFinal: 
                    ret[mkindx(iorig)] = a[mkindx(icur)]

        return ret
        

    def distribute(self, comm, verbosity=0):
        """
        Distributes this tree's sub-trees across multiple processors.
        
        This function checks how many processors are present in
        `comm` and divides this tree's subtrees into groups according to
        the number of subtree comms provided as an argument to 
        `initialize`.  Note that this does *not* always divide the subtrees
        among the processors as much as possible, as this is not always what is
        desired (computing several subtrees serially using multiple
        processors simultaneously can be faster, due to better balancing, than
        computing all the subtrees simultaneously using smaller processor groups
        for each).  

        For example, if the number of subtree comms given to 
        `initialize` == 1, then all the subtrees are assigned to the one and
        only processor group containing all the processors of `comm`.  If the
        number of subtree comms == 2 and there are 4 subtrees and 8 processors,
        then `comm` is divided into 2 groups of 4 processors each, and two 
        subtrees are assigned to each processor group.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm
            When not None, an MPI communicator for distributing subtrees
            across processor groups

        verbosity : int, optional
            How much detail to send to stdout.
            
        Returns
        -------
        mySubtreeIndices : list
            A list of integer indices specifying which subtrees this
            processor is responsible for.
        subTreeOwners : dict
            A dictionary whose keys are integer subtree indices and 
            whose values are processor ranks, which indicates which 
            processor is responsible for communicating the final 
            results of each subtree.
        mySubComm : mpi4py.MPI.Comm or None
            The communicator for the processor group that is responsible
            for computing the same `mySubTreeIndices` list.  This 
            communicator is used for further processor division (e.g.
            for parallelization across derivative columns).
        """
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
        raise NotImplementedError("split(...) not implemented!")


    def _finish_split(self, subTreeSetList, permute_parent_element, create_subtree):
        # Create subtrees from index sets
        need_to_compute = _np.zeros( len(self), 'bool' ) #flags so we don't duplicate computation of needed quantities
        need_to_compute[0:self.num_final_strings()] = True #  b/c multiple subtrees need them as intermediates

          #First, reorder the parent tree's elements so that the final
          # elements of the subtrees map to contiguous slices of the
          # parent tree's final elements.
        parentIndexRevPerm = [] # parentIndexRevPerm[newIndex] = currentIndex (i.e. oldIndex)
        subTreeIndicesList = []
        numFinalList = []
        for iSubTree,subTreeSet in enumerate(subTreeSetList):
            subTreeIndices = list(subTreeSet)
            #if bDebug: print("SUBTREE0: %s (len=%d)" % (str(subTreeIndices),len(subTreeIndices)))
            #if bDebug: print("  NEED: %s" % ",".join([ "1" if b else "0" for b in need_to_compute]))
            subTreeIndices.sort() # order subtree gatestrings (really just their indices) so
                                  # that all "final" strings come first.
            #Compute # of "final" strings in this subtree (count # of indices < num_final_strs)
            subTreeNumFinal = _np.sum(_np.array(subTreeIndices) < self.num_final_strings())

            #Swap the indices of "final" strings that have already been computed past the end
            # of the "final strings" region of the subtree's list (i.e. the subtree itself).
            # (some "required"/"final"strings may have already been computed by a previous subtree)
            already_computed = _np.logical_not( need_to_compute[ subTreeIndices[0:subTreeNumFinal] ] )
            already_computed_inds = _np.nonzero(already_computed)[0] # (sorted ascending)
            #if bDebug: print("SUBTREE1: %s (nFinal=%d - %d)" % (str(subTreeIndices),
            #                                         subTreeNumFinal, len(already_computed_inds)))
            #if bDebug: print("  - already computed = ", [subTreeIndices[i] for i in already_computed_inds])

            iFirstNonFinal = subTreeNumFinal
            for k in already_computed_inds:
                if k >= iFirstNonFinal: continue #already a non-final el
                elif k == iFirstNonFinal-1: #index is last "final" el - just shift boundary
                    iFirstNonFinal -= 1 #now index is "non-final"
                else: # k < iFirstNonFinal-1, so find a desired "final" el at boundary to swap it with
                    iLastFinal = iFirstNonFinal-1
                    while iLastFinal > k and (iLastFinal in already_computed_inds):
                        iLastFinal -= 1 #the element at iLastFinal happens to be one that we wanted to be non-final, so remove it
                    if iLastFinal != k:
                        subTreeIndices[iLastFinal],subTreeIndices[k] = \
                            subTreeIndices[k],subTreeIndices[iLastFinal] #Swap k <-> iLastFinal
                    iFirstNonFinal = iLastFinal # move boundary to make k's new location non-final

            subTreeNumFinal = iFirstNonFinal # the final <-> non-final boundary
            parentIndexRevPerm.extend( subTreeIndices[0:subTreeNumFinal] )
            subTreeIndicesList.append( subTreeIndices )
            numFinalList.append( subTreeNumFinal )
            need_to_compute[ subTreeIndices[0:subTreeNumFinal] ] = False
            #if bDebug: print("FINAL SUBTREE: %s (nFinal=%d)" % (str(subTreeIndices),subTreeNumFinal))
                    
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

        #if bDebug: print("PERM REV MAP = ", parentIndexRevPerm)
        #if bDebug: print("PERM MAP = ", parentIndexPerm)

        #Permute parent indices
        self.init_indices = [ parentIndexPerm[iCur] for iCur in self.init_indices ]
        self.eval_order = [ parentIndexPerm[iCur] for iCur in self.eval_order ]
        self[:] = [ permute_parent_element(parentIndexPerm, self[iCur])
                    for iCur in parentIndexRevPerm ]
        assert(self.myFinalToParentFinalMap is None)
        assert(self.parentIndexMap is None)

        #Permute subtree indices (i.e. lists of subtree indices)
        newList = []; finalSlices = []; sStart = 0
        for subTreeIndices,numFinal in zip(subTreeIndicesList,numFinalList):
            newSubTreeIndices = [ parentIndexPerm[i] for i in subTreeIndices ] 
            assert(newSubTreeIndices[0:numFinal] == list(range(sStart,sStart+numFinal)))
              #final elements should be a sequential slice of parent indices
            finalSlices.append( slice(sStart,sStart+numFinal) )
            newList.append(newSubTreeIndices)
            sStart += numFinal #increment slice start position
        subTreeIndicesList = newList # => subTreeIndicesList is now permuted

        #Now (finally) create the actual subtrees, which requires
        # taking parent-indices and mapping them the subtree-indices
        for iSubTree,(subTreeIndices,subTreeNumFinal,slc) \
                in enumerate(zip(subTreeIndicesList,numFinalList,finalSlices)):

            fullEvalOrder = self.init_indices + self.eval_order #list concatenation
            subTreeEvalOrder = sorted(range(len(subTreeIndices)),
                                      key=lambda j: fullEvalOrder.index(subTreeIndices[j]))

            subtree = create_subtree(subTreeIndices, subTreeNumFinal, subTreeEvalOrder, slc, self)
            self.subTrees.append( subtree )
            #if bDebug: print("NEW SUBTREE: indices=%s, len=%d, nFinal=%d"  %
            #                  (subTree.parentIndexMap, len(subTree), subTree.num_final_strings()))

        #dbList2 = self.generate_gatestring_list()
        #if bDebug: print("DBLIST = ",dbList)
        #if bDebug: print("DBLIST2 = ",dbList2)
        #assert(dbList == dbList2)
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
