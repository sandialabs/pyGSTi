""" Defines the EvalTree class which implements an evaluation tree. """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import collections as _collections

from ..tools import mpitools as _mpit
from ..tools import slicetools as _slct
from ..baseobjs import VerbosityPrinter as _VerbosityPrinter

import numpy as _np
#import time as _time #DEBUG TIMERS

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

        # Number of "final" or "requested" elements, which separately
        # counts each spamTuple of each of the final gate strings.
        self.num_final_els = 0
        
        # The list of "child" sub-trees (if this tree is spilt)
        self.subTrees = []

        # a dict to hold various MPI distribution info
        self.distribution = {}

        # a list of spamTuple-lists, one for each final gate string
        self.compiled_gatestring_spamTuples = None
        #self.finalStringToElsMap = None

        # a dictionary of final-gate-string index lists keyed by 
        # each distinct spamTuple
        self.spamtuple_indices = None
                
        # ********* Only non-None for sub-trees ******************
        
          # The mapping between this tree's final gate string indices and its parent's
        self.myFinalToParentFinalMap = None

          # The mapping between this tree's final element indices and its parent's
        self.myFinalElsToParentFinalElsMap = None
        
          # The parent's index of each of this tree's *final* indices          
        self.parentIndexMap = None
        
          # A dictionary whose keys are the "original" (as-given to initialize)
          # indices and whose values are the new "permuted" indices.  So if you
          # want to know where in a tree the ith-element of gatestring_list (as
          # passed to initialize(...) is, it's at index original_index_lookup[i]
        self.original_index_lookup = None

        super(EvalTree, self).__init__(items)

        
    def initialize(self, compiled_gatestring_list, numSubTreeComms=1):
        """
          Initialize an evaluation tree using a set of gate strings.
          This function must be called before using an EvalTree.

          Parameters
          ----------
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

    def _get_gateLabels(self, compiled_gatestring_list):
        """ 
        Returns a list of the distinct gate labels in 
        `compiled_gatestring_list` - a dictionary w/keys = "raw" gate strings OR a list of them.
        """
        gateLabels = set()
        for raw_gstr in compiled_gatestring_list: # will work for dict keys too
            gateLabels.update( raw_gstr )
        return sorted(gateLabels)


    def _copyBase(self,newTree):
        """ copy EvalTree members to a new tree (used by derived classes "copy" fns) """
        newTree.gateLabels = self.gateLabels[:]
        newTree.init_indices = self.init_indices[:]
        newTree.eval_order = self.eval_order[:]
        newTree.num_final_strs = self.num_final_strs
        newTree.num_final_els = self.num_final_els
        newTree.myFinalToParentFinalMap = self.myFinalToParentFinalMap # a slice
        newTree.myFinalElsToParentFinalElsMap = self.myFinalElsToParentFinalElsMap.copy() \
                            if (self.myFinalElsToParentFinalElsMap is not None) else None
        newTree.parentIndexMap = self.parentIndexMap[:] \
            if (self.parentIndexMap is not None) else None
        newTree.subTrees = [ st.copy() for st in self.subTrees ]
        newTree.original_index_lookup = self.original_index_lookup[:] \
            if (self.original_index_lookup is not None) else None
        newTree.compiled_gatestring_spamTuples = self.compiled_gatestring_spamTuples[:]
        #newTree.finalStringToElsMap = self.finalStringToElsMap[:]
        newTree.spamtuple_indices = self.spamtuple_indices.copy()
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
            ret = a[tuple(sl)]
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


    def final_element_indices(self, parent_tree):
        """
        Return a slice or index array that identifies the segment of
        `parent_tree`'s final "element" values that correspond to this tree's
        final values, *including* spam indices.

        Note that if `parent_tree` is not None, this tree must be a
        sub-tree of it (which means `parent_tree` is split).

        Parameters
        ----------
        parent_tree : EvalTree
            This tree's parent tree.  If the parent tree is None
            or isn't split (which sometimes means that this 
            "sub-tree" is the same as the parent tree - see 
            `get_sub_trees`), then an index for the entire set of
            final values is returned, which is appropriate in this
            case.

        Returns
        -------
        slice
        """
        if (self.myFinalElsToParentFinalElsMap is not None) and \
                parent_tree.is_split():
            return self.myFinalElsToParentFinalElsMap
        else:
            return slice(0,self.num_final_elements())


    def num_final_strings(self):
        """
        Returns the integer number of "final" gate strings, equal
          to the number of keys in the `compiled_gatestring_list`
          passed to :method:`initialize`.
        """
        return self.num_final_strs

    def num_final_elements(self):
        """
        Returns the integer number of "final" elements, equal
          to the number of (gatestring, spamTuple) pairs contained in
          the `compiled_gatestring_list` passed to :method:`initialize`.
        """
        return self.num_final_els

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

        def _mkindx(i):
            mi = [slice(None)]*a.ndim; mi[axis] = i
            return tuple(mi)

        if self.original_index_lookup is not None:
            for iorig,icur in self.original_index_lookup.items():                
                if iorig < nFinal: 
                    ret[_mkindx(icur)] = a[_mkindx(iorig)]

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

        def _mkindx(i):
            mi = [slice(None)]*a.ndim; mi[axis] = i
            return tuple(mi)

        if self.original_index_lookup is not None:
            for iorig,icur in self.original_index_lookup.items():                
                if iorig < nFinal: 
                    ret[_mkindx(iorig)] = a[_mkindx(icur)]

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

        #rank = 0 if (comm is None) else comm.Get_rank()
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

        printer = _VerbosityPrinter.build_printer(verbosity, comm)
        printer.log("*** Distributing %d subtrees into %d sub-comms (%s processors) ***"% \
                        (nSubtrees, nSubtreeComms, nprocs))

        return mySubtreeIndices, subTreeOwners, mySubComm


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
        raise NotImplementedError("split(...) not implemented!")

    def recompute_spamtuple_indices(self, bLocal=False):
        """ 
        Recompute this tree's `.spamtuple_indices` array.

        Parameters
        ----------
        bLocal : bool, optional
            If True, then the indices computed will index 
            this tree's final array (even if it's a subtree).
            If False (the default), then a subtree's indices
            will index the *parent* tree's final array.

        Returns
        -------
        None
        """
        self.spamtuple_indices = _compute_spamtuple_indices(
            self.compiled_gatestring_spamTuples,
            None if bLocal else self.myFinalElsToParentFinalElsMap)
        

    def _finish_split(self, elIndicesDict, subTreeSetList, permute_parent_element, create_subtree):
        # Create subtrees from index sets
        need_to_compute = _np.zeros( len(self), 'bool' ) #flags so we don't duplicate computation of needed quantities
        need_to_compute[0:self.num_final_strings()] = True #  b/c multiple subtrees need them as intermediates

        #print("DEBUG Tree split: ")
        #print("  subTreeSetList = ",subTreeSetList)
        #print("  elIndices = ",elIndicesDict)

          #First, reorder the parent tree's elements so that the final
          # elements of the subtrees map to contiguous slices of the
          # parent tree's final elements.
        parentIndexRevPerm = [] # parentIndexRevPerm[newIndex] = currentIndex (i.e. oldIndex)
        subTreeIndicesList = []
        numFinalList = []
        for subTreeSet in subTreeSetList:
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
            if subTreeNumFinal == 0: continue # this subtree only contributes non-final elements -> skip

            parentIndexRevPerm.extend( subTreeIndices[0:subTreeNumFinal] )
            subTreeIndicesList.append( subTreeIndices )
            numFinalList.append( subTreeNumFinal )
            need_to_compute[ subTreeIndices[0:subTreeNumFinal] ] = False
            #if bDebug: print("FINAL SUBTREE: %s (nFinal=%d)" % (str(subTreeIndices),subTreeNumFinal))
                    
        #Permute parent tree indices according to parentIndexPerm
        # parentIndexRevPerm maps: newIndex -> currentIndex, so looking at it as a list
        #  gives the new (permuted) elements
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

        #print("DEBUG: PERM REV MAP = ", parentIndexRevPerm,
        #      "(first %d are 'final')" % self.num_final_strings())
        #print("DEBUG: PERM MAP = ", parentIndexPerm)

        #Permute parent indices
        self.init_indices = [ parentIndexPerm[iCur] for iCur in self.init_indices ]
        self.eval_order = [ parentIndexPerm[iCur] for iCur in self.eval_order ]
        self[:] = [ permute_parent_element(parentIndexPerm, self[iCur])
                    for iCur in parentIndexRevPerm ]


        # Setting compiled_gatestring_spamTuples, (re)sets the element ordering,
        # so before doint this compute the old_to_new mapping and update
        # elIndicesDict.
        old_finalStringToElsMap = []; i=0
        for k,spamTuples in enumerate(self.compiled_gatestring_spamTuples):
            old_finalStringToElsMap.append( list(range(i,i+len(spamTuples))) )
            i += len(spamTuples)

        permute_newToOld = []
        for iOldStr in parentIndexRevPerm[0:self.num_final_strings()]:
            permute_newToOld.extend( old_finalStringToElsMap[iOldStr] )
        permute_oldToNew = { iOld:iNew for iNew,iOld in enumerate(permute_newToOld) }

        #print("DEBUG: old_finalStrToEls = ",old_finalStringToElsMap)
        #print("DEBUG: permute_newToOld = ",permute_newToOld)
        #print("DEBUG: permute_oldToNew = ",permute_oldToNew)

        updated_elIndices = _collections.OrderedDict()
        for ky,indices in elIndicesDict.items():
            updated_elIndices[ky] = _slct.list_to_slice(
                [ permute_oldToNew[x] for x in
                  (_slct.indices(indices) if isinstance(indices,slice) else indices)] )

        # Now update compiled_gatestring_spamTuples
        self.compiled_gatestring_spamTuples = [ self.compiled_gatestring_spamTuples[iCur]
                                                for iCur in parentIndexRevPerm[0:self.num_final_strings()] ]
        self.recompute_spamtuple_indices(bLocal=True) #bLocal shouldn't matter here - just for clarity

        #Assert this tree (self) is *not* split
        assert(self.myFinalToParentFinalMap is None)
        assert(self.myFinalElsToParentFinalElsMap is None)
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
        #print("DEBUG: updated elIndices = ",updated_elIndices)
        
        return updated_elIndices


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

    #NOT NEEDED?
    #def _compute_finalStringToEls(self):
    #    #Create a mapping from each final gate string (index) to
    #    # a slice of final element indices
    #    self.finalStringToElsMap = []; i=0
    #    for k,spamTuples in enumerate(self.compiled_gatestring_spamTuples):
    #        self.finalStringToElsMap.append( slice(i,i+len(spamTuples)) )
    #        i += len(spamTuples)

        
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

                
def _compute_spamtuple_indices(compiled_gatestring_spamTuples,
                               subtreeFinalElsToParentFinalElsMap=None):
    """ 
    Returns a dictionary whose keys are the distinct spamTuples
    found in `compiled_gatestring_spamTuples` and whose values are
    (finalIndices, finalTreeSlice) tuples where:

    finalIndices = the "element" indices in any final filled quantities
                   which combines both spam and gate-sequence indices.
                   If this tree is a subtree, then these final indices
                   refer to the *parent's* final elements if 
                   `subtreeFinalElsToParentFinalElsMap` is given, otherwise
                   they refer to the subtree's final indices (usually desired).
    treeIndices = indices into the tree's final gatestring list giving
                  all of the (raw) gate sequences which need to be computed
                  for the current spamTuple (this list has the SAME length
                  as finalIndices).
    """
    spamtuple_indices = _collections.OrderedDict(); el_off = 0
    for i,spamTuples in enumerate(  # i == final gate string index
            compiled_gatestring_spamTuples):
        for j,spamTuple in enumerate(spamTuples,start=el_off): # j == final element index
            if spamTuple not in spamtuple_indices:
                spamtuple_indices[spamTuple] = ([],[])
            f = subtreeFinalElsToParentFinalElsMap[j] \
                if (subtreeFinalElsToParentFinalElsMap is not None) else j #parent's final
            spamtuple_indices[spamTuple][0].append(f)
            spamtuple_indices[spamTuple][1].append(i)
        el_off += len(spamTuples)

    def to_slice(x):
        s = _slct.list_to_slice(x,array_ok=True,require_contiguous=False)
        if isinstance(s, slice) and (s.start,s.stop,s.step) == \
           (0,len(compiled_gatestring_spamTuples),None):
            return slice(None,None) #check for entire range
        else:
            return s

    return _collections.OrderedDict(
        [ (spamTuple, (to_slice(fInds), to_slice(gInds)))
          for spamTuple,(fInds,gInds) in spamtuple_indices.items() ] )
