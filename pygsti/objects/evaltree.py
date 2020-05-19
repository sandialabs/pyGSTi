"""
Defines the EvalTree class which implements an evaluation tree.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import collections as _collections

from ..tools import mpitools as _mpit
from ..tools import slicetools as _slct
from .verbosityprinter import VerbosityPrinter as _VerbosityPrinter

import numpy as _np
import warnings as _warnings
#import time as _time #DEBUG TIMERS


class EvalTree(list):
    """
    An Evaluation Tree.  A structure that assists in performing bulk Model operations.

    EvalTree instances create and store a particular structure on or decomposition of
    a list of circuits that facilitates fast computation of outcome probabilities.

    Parameters
    ----------
    items : list, optional
        Initial items.  This argument should only be used internally
        in the course of serialization.
    """

    def __init__(self, items=[]):
        """ Create a new, empty, evaluation tree. """

        # list of indices specifying what order they should be evaluated in,
        # *after* evaluating all of the initial indices (init_indices)
        self.eval_order = []

        # list giving the number of elements (~effect labels) associated
        # with each simplified circuit.
        self.simplified_circuit_nEls = None

        # Number of "final" or "requested" strings, which may be less
        # then len(self) since some un-requested results may be stored
        # as useful intermediates.
        self.num_final_strs = 0

        # Number of "final" or "requested" elements, which separately
        # counts each spam_tuple of each of the final operation sequences.
        self.num_final_els = 0

        # The list of "child" sub-trees (if this tree is spilt)
        self.subTrees = []

        # a dict to hold various MPI distribution info
        self.distribution = {}

        # ********* Only non-None for sub-trees ******************

        # The mapping between this tree's final operation sequence indices and its parent's
        self.myFinalToParentFinalMap = None

        # The mapping between this tree's final element indices and its parent's
        self.myFinalElsToParentFinalElsMap = None

        # The parent's index of each of this tree's *final* indices
        self.parentIndexMap = None

        # list of the operation labels
        self.opLabels = []

        # A dictionary whose keys are the "original" (as-given to initialize)
        # indices and whose values are the new "permuted" indices.  So if you
        # want to know where in a tree the ith-element of circuit_list (as
        # passed to initialize(...) is, it's at index original_index_lookup[i]
        self.original_index_lookup = None

        super(EvalTree, self).__init__(items)

    def initialize(self, simplified_circuit_elabels, num_sub_tree_comms=1):
        """
        Initialize an evaluation tree using a set of "simplified" circuits.

        This function must be called before using this evaluation tree.

        Parameters
        ----------
        simplified_circuit_elabels : dict
            A dictionary of simplified circuits, i.e. circuits that do not include
            POVM measurements and do not contain any instruments.  Keys are "raw"
            operation sequences (circuits with state preparations and no instruments)
            and values are lists of "simplified" effect labels (i.e. a single label
            that identifies a POVM and outcome).

        num_sub_tree_comms : int, optional
            The number of processor groups (communicators)
            to divide the subtrees of this EvalTree among
            when calling `distribute`.  By default, the
            communicator is not divided.

        Returns
        -------
        None
        """
        raise NotImplementedError("initialize(...) must be implemented by a derived class")

    def _get_op_labels(self, simplified_circuit_elabels):
        """
        Returns a list of the distinct operation labels in
        `simplified_circuit_elabels` - a dictionary w/keys = "raw" operation sequences OR a list of them.
        """
        opLabels = set()

        for simple_circuit_with_prep, elabels in simplified_circuit_elabels.items():
            if elabels == [None]:  # special case when circuit contains no prep
                opLabels.update(simple_circuit_with_prep)
            else:
                opLabels.update(simple_circuit_with_prep[1:])  # don't include prep

        return sorted(opLabels)

    def _copy_base(self, new_tree):
        """ copy EvalTree members to a new tree (used by derived classes "copy" fns) """
        new_tree.eval_order = self.eval_order[:]
        new_tree.num_final_strs = self.num_final_strs
        new_tree.num_final_els = self.num_final_els
        new_tree.myFinalToParentFinalMap = self.myFinalToParentFinalMap  # a slice
        new_tree.myFinalElsToParentFinalElsMap = self.myFinalElsToParentFinalElsMap.copy() \
            if (self.myFinalElsToParentFinalElsMap is not None) else None
        new_tree.parentIndexMap = self.parentIndexMap[:] \
            if (self.parentIndexMap is not None) else None
        new_tree.subTrees = [st.copy() for st in self.subTrees]
        new_tree.original_index_lookup = self.original_index_lookup[:] \
            if (self.original_index_lookup is not None) else None
        new_tree.simplified_circuit_nEls = self.simplified_circuit_nEls[:]
        return new_tree

    def get_init_labels(self):
        """
        Return a tuple of the operation labels which form the beginning of the tree.

        Returns
        -------
        tuple
        """
        return tuple(self.opLabels)

    def get_init_indices(self):
        """
        Return a tuple of the indices corresponding to the initial operation labels.

        These "initial labels" are returned by :method:`get_init_labels` and
        form the beginning of the tree.

        Returns
        -------
        tuple
        """
        return tuple(self.init_indices)

    def get_evaluation_order(self):
        """
        Return a list of indices specifying the evaluation order.

        This is the order in which elements of this EvalTree should be visited when
        doing a computation (after computing the initial indices).

        Returns
        -------
        list
        """
        return self.eval_order

    def final_view(self, a, axis=None):
        """
        Create a view of array `a` restricting it to only the *final* results computed by this tree.

        This need not be the entire array because there could be intermediate results
        (e.g. "scratch space") that are excluded.

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
            sl[axis] = slice(0, self.num_final_strings())
            ret = a[tuple(sl)]
            assert(ret.base is a or ret.base is a.base)  # check that what is returned is a view
            assert(ret.size == 0 or _np.may_share_memory(ret, a))
            return ret

    def final_slice(self, parent_tree):
        """
        The slice of `parent_tree`'s final values corresponding to this tree's final values.

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
            Identifies the segment of `parent_tree`'s final values that
            correspond to this tree's final values.
        """
        if (self.myFinalToParentFinalMap is not None) and \
                parent_tree.is_split():
            return self.myFinalToParentFinalMap
        else:
            return slice(0, self.num_final_strings())

    def final_element_indices(self, parent_tree):
        """
        The slice (or index array) of `parent_tree`'s final "element" values corresponding to this tree's final values.

        These element values differ from the "values" of :method:`final_view` in that elements
        correspond to circuits *including* SPAM, and so each element corresponds to a single
        outcome probability..

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
        slice or numpy.ndarray
            Identifies the segment of `parent_tree`'s final "element" values that
            correspond to this tree's final "element" values (i.e. *including* spam indices).
        """
        if (self.myFinalElsToParentFinalElsMap is not None) and \
                parent_tree.is_split():
            return self.myFinalElsToParentFinalElsMap
        else:
            return slice(0, self.num_final_elements())

    def num_final_strings(self):
        """
        Returns the integer number of "final" circuits.

        This is equal to the number of keys in the `simplified_circuit_elabels`
        passed to :method:`initialize`.

        Returns
        -------
        int
        """
        return self.num_final_strs

    def num_final_elements(self):
        """
        Returns the integer number of "final" elements.

        This is equal to the number of (circuit, elabels) pairs contained in
        the `simplified_circuit_elabels` passed to :method:`initialize`.

        Returns
        -------
        int
        """
        return self.num_final_els

    def generate_circuit_list(self, permute=True):
        """
        Generate a list of the final operation sequences this tree evaluates.

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
            When False, the computed order of the operation sequences is
            given, which is matches the order of the results from calls
            to `Model` bulk operations.  Non-trivial permutation
            occurs only when the tree is split (in order to keep
            each sub-tree result a contiguous slice within the parent
            result).

        Returns
        -------
        list of gate-label-tuples
            A list of the operation sequences evaluated by this tree, each
            specified as a tuple of operation labels.
        """
        raise NotImplementedError("generate_circuit_list(...) not implemented!")

    def permute_original_to_computation(self, a, axis=0):
        """
        Permute an array's elements from "original" to "computational" circuit ordering.

        This function converts arrays with elements corresponding
        to circuits in the "original" ordering (i.e. the
        ordering in the list passed to `initialize(...)`) to the
        ordering used in tree computation (i.e. by a `Model`'s
        bulk computation routines).

        Parameters
        ----------
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
            mi = [slice(None)] * a.ndim; mi[axis] = i
            return tuple(mi)

        if self.original_index_lookup is not None:
            for iorig, icur in self.original_index_lookup.items():
                if iorig < nFinal:
                    ret[_mkindx(icur)] = a[_mkindx(iorig)]

        return ret

    def permute_computation_to_original(self, a, axis=0):
        """
        Permute an array's elements from "computational" to "original" circuit ordering.

        This function converts arrays with elements corresponding
        to circuits in the ordering used in tree computation
        (i.e. the ordering returned by `Model`'s bulk computation routines)
        to the "original" ordering (i.e. the ordering of the circuit list
        passed to `initialize(...)`).

        Parameters
        ----------
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
            mi = [slice(None)] * a.ndim; mi[axis] = i
            return tuple(mi)

        if self.original_index_lookup is not None:
            for iorig, icur in self.original_index_lookup.items():
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
        # model parameters)

        #rank = 0 if (comm is None) else comm.Get_rank()
        nprocs = 1 if (comm is None) else comm.Get_size()
        nSubtreeComms = self.distribution.get('numSubtreeComms', 1)
        nSubtrees = len(self.get_sub_trees())

        assert(nSubtreeComms <= nprocs)  # => len(mySubCommIndices) == 1
        mySubCommIndices, subCommOwners, mySubComm = \
            _mpit.distribute_indices(list(range(nSubtreeComms)), comm)
        assert(len(mySubCommIndices) == 1)
        mySubCommIndex = mySubCommIndices[0]

        assert(nSubtreeComms <= nSubtrees)  # don't allow more comms than trees
        mySubtreeIndices, subTreeOwners = _mpit.distribute_indices_base(
            list(range(nSubtrees)), nSubtreeComms, mySubCommIndex)

        # subTreeOwners contains index of owner subComm, but we really want
        #  the owning processor, i.e. the owner of the subComm
        subTreeOwners = {iSubTree: subCommOwners[subTreeOwners[iSubTree]]
                         for iSubTree in subTreeOwners}

        printer = _VerbosityPrinter.build_printer(verbosity, comm)
        printer.log("*** Distributing %d subtrees into %d sub-comms (%s processors) ***" %
                    (nSubtrees, nSubtreeComms, nprocs))

        return mySubtreeIndices, subTreeOwners, mySubComm

    def split(self, el_indices_dict, max_sub_tree_size=None, num_sub_trees=None, verbosity=0):
        """
        Split this tree into sub-trees.

        This is done in order to reduce the maximum size of any tree (useful for
        limiting memory consumption or for using multiple cores).  Must specify
        either max_sub_tree_size or num_sub_trees.

        Parameters
        ----------
        el_indices_dict : dict
            A dictionary whose keys are integer original-circuit indices
            and whose values are slices or index arrays of final-element-
            indices (typically this dict is returned by calling
            :method:`Model.simplify_circuits`).  Since splitting a
            tree often involves permutation of the raw string ordering
            and thereby the element ordering, an updated version of this
            dictionary, with all permutations performed, is returned.

        max_sub_tree_size : int, optional
            The maximum size (i.e. list length) of each sub-tree.  If the
            original tree is smaller than this size, no splitting will occur.
            If None, then there is no limit.

        num_sub_trees : int, optional
            The maximum size (i.e. list length) of each sub-tree.  If the
            original tree is smaller than this size, no splitting will occur.

        verbosity : int, optional
            How much detail to send to stdout.

        Returns
        -------
        OrderedDict
            A updated version of el_indices_dict
        """
        raise NotImplementedError("split(...) not implemented!")

    def _finish_split(self, el_indices_dict, sub_tree_set_list, permute_parent_element,
                      create_subtree, all_final=False):
        # Create subtrees from index sets
        import time as _time
        need_to_compute = _np.zeros(len(self), 'bool')  # flags so we don't duplicate computation of needed quantities
        need_to_compute[0:self.num_final_strings()] = True  # b/c multiple subtrees need them as intermediates

        #print("DEBUG Tree split: ")
        #print("  sub_tree_set_list = ",sub_tree_set_list)
        #print("  el_indices = ",el_indices_dict)

        #First, reorder the parent tree's elements so that the final
        # elements of the subtrees map to contiguous slices of the
        # parent tree's final elements.
        parentIndexRevPerm = []  # parentIndexRevPerm[newIndex] = currentIndex (i.e. oldIndex)
        subTreeIndicesList = []
        numFinalList = []
        #t0 = _time.time() #REMOVE
        for subTreeSet in sub_tree_set_list:
            subTreeIndices = list(subTreeSet)
            #if bDebug: print("SUBTREE0: %s (len=%d)" % (str(subTreeIndices),len(subTreeIndices)))
            #if bDebug: print("  NEED: %s" % ",".join([ "1" if b else "0" for b in need_to_compute]))
            subTreeIndices.sort()  # order subtree circuits (really just their indices) so
            # that all "final" strings come first.

            if all_final:
                subTreeNumFinal = len(subTreeIndices)
            else:
                #Compute # of "final" strings in this subtree (count # of indices < num_final_strs)
                subTreeNumFinal = _np.sum(_np.array(subTreeIndices) < self.num_final_strings())

                #Swap the indices of "final" strings that have already been computed past the end
                # of the "final strings" region of the subtree's list (i.e. the subtree itself).
                # (some "required"/"final"strings may have already been computed by a previous subtree)
                already_computed = _np.logical_not(need_to_compute[subTreeIndices[0:subTreeNumFinal]])
                already_computed_inds = _np.nonzero(already_computed)[0]  # (sorted ascending)
                #if bDebug: print("SUBTREE1: %s (nFinal=%d - %d)" % (str(subTreeIndices),
                #                                         subTreeNumFinal, len(already_computed_inds)))
                #if bDebug: print("  - already computed = ", [subTreeIndices[i] for i in already_computed_inds])

                iFirstNonFinal = subTreeNumFinal
                for k in already_computed_inds:
                    if k >= iFirstNonFinal: continue  # already a non-final el
                    elif k == iFirstNonFinal - 1:  # index is last "final" el - just shift boundary
                        iFirstNonFinal -= 1  # now index is "non-final"
                    else:  # k < iFirstNonFinal-1, so find a desired "final" el at boundary to swap it with
                        iLastFinal = iFirstNonFinal - 1
                        while iLastFinal > k and (iLastFinal in already_computed_inds):
                            # the element at iLastFinal happens to be one that we wanted to be non-final, so remove it
                            iLastFinal -= 1
                        if iLastFinal != k:
                            subTreeIndices[iLastFinal], subTreeIndices[k] = \
                                subTreeIndices[k], subTreeIndices[iLastFinal]  # Swap k <-> iLastFinal
                        iFirstNonFinal = iLastFinal  # move boundary to make k's new location non-final

                subTreeNumFinal = iFirstNonFinal  # the final <-> non-final boundary
                if subTreeNumFinal == 0:
                    _warnings.warn(("A 'dummy' subtree was created that doesn't compute any "
                                    "actual ('final') elements.  This is fine, but usually "
                                    "means you're using more processors than you need to."))
                    #OLD: continue # this subtree only contributes non-final elements -> skip

            parentIndexRevPerm.extend(subTreeIndices[0:subTreeNumFinal])
            subTreeIndicesList.append(subTreeIndices)
            numFinalList.append(subTreeNumFinal)
            need_to_compute[subTreeIndices[0:subTreeNumFinal]] = False
            #if bDebug: print("FINAL SUBTREE: %s (nFinal=%d)" % (str(subTreeIndices),subTreeNumFinal))

        #print("PT1 = %.3fs" % (_time.time()-t0)); t0 = _time.time() # REMOVE

        #Permute parent tree indices according to parentIndexPerm
        # parentIndexRevPerm maps: newIndex -> currentIndex, so looking at it as a list
        #  gives the new (permuted) elements
        assert(len(parentIndexRevPerm) == self.num_final_strings())
        parentIndexRevPerm.extend(list(range(self.num_final_strings(), len(self))))
        #don't permute non-final indices (no need)

        #Create forward permutation map: currentIndex -> newIndex
        parentIndexPerm = [None] * len(parentIndexRevPerm)
        for inew, icur in enumerate(parentIndexRevPerm):
            parentIndexPerm[icur] = inew
        assert(None not in parentIndexPerm)  # all indices should be mapped somewhere!
        assert(self.original_index_lookup is None)
        self.original_index_lookup = {icur: inew for inew, icur in enumerate(parentIndexRevPerm)}
        #print("PT2 = %.3fs" % (_time.time()-t0)); t0 = _time.time()  # REMOVE

        #print("DEBUG: PERM REV MAP = ", parentIndexRevPerm,
        #      "(first %d are 'final')" % self.num_final_strings())
        #print("DEBUG: PERM MAP = ", parentIndexPerm)

        #Permute parent indices
        # HACK to allow .init_indices to be updated in Matrix tree case
        self._update_eval_order_helpers(parentIndexPerm)
        updated_elIndices = self._update_element_indices(parentIndexPerm, parentIndexRevPerm, el_indices_dict)
        self.eval_order = [parentIndexPerm[iCur] for iCur in self.eval_order]
        self[:] = [permute_parent_element(parentIndexPerm, self[iCur])
                   for iCur in parentIndexRevPerm]
        #print("PT3 = %.3fs" % (_time.time()-t0)); t0 = _time.time()  # REMOVE

        #Assert this tree (self) is *not* split
        assert(self.myFinalToParentFinalMap is None)
        assert(self.myFinalElsToParentFinalElsMap is None)
        assert(self.parentIndexMap is None)

        #Permute subtree indices (i.e. lists of subtree indices)
        newList = []; finalSlices = []; sStart = 0
        for subTreeIndices, numFinal in zip(subTreeIndicesList, numFinalList):
            newSubTreeIndices = [parentIndexPerm[i] for i in subTreeIndices]
            assert(newSubTreeIndices[0:numFinal] == list(range(sStart, sStart + numFinal)))
            #final elements should be a sequential slice of parent indices
            finalSlices.append(slice(sStart, sStart + numFinal))
            newList.append(newSubTreeIndices)
            sStart += numFinal  # increment slice start position
        subTreeIndicesList = newList  # => subTreeIndicesList is now permuted

        #print("PT6 = %.3fs" % (_time.time()-t0)); t0 = _time.time() # REMOVE
        #Now (finally) create the actual subtrees, which requires
        # taking parent-indices and mapping them the subtree-indices
        fullEvalOrder_lookup = {k: i for i, k in enumerate(self._get_full_eval_order())}  # list concatenation
        for iSubTree, (subTreeIndices, subTreeNumFinal, slc) \
                in enumerate(zip(subTreeIndicesList, numFinalList, finalSlices)):

            subTreeEvalOrder = sorted(range(len(subTreeIndices)),
                                      key=lambda j: fullEvalOrder_lookup[subTreeIndices[j]])

            subtree = create_subtree(subTreeIndices, subTreeNumFinal, subTreeEvalOrder, slc, self)
            self.subTrees.append(subtree)
            #if bDebug: print("NEW SUBTREE: indices=%s, len=%d, nFinal=%d"  %
            #                  (subTree.parentIndexMap, len(subTree), subTree.num_final_strings()))

        # print("PT7 = %.3fs" % (_time.time()-t0)); t0 = _time.time() # REMOVE
        #dbList2 = self.generate_circuit_list()
        #if bDebug: print("DBLIST = ",dbList)
        #if bDebug: print("DBLIST2 = ",dbList2)
        #assert(dbList == dbList2)
        #return parentIndexRevPerm
        return updated_elIndices

    def _get_full_eval_order(self):
        """Includes init_indices in matrix-based evaltree case... HACK """
        return self.eval_order

    def _update_eval_order_helpers(self, index_permutation):
        """Update anything pertaining to the "full" evaluation order
           - e.g. init_inidces in matrix-based case (HACK)"""
        pass

    def _update_element_indices(self, new_indices_in_old_order, old_indices_in_new_order, element_indices_dict):
        """
        Update any additional members because this tree's elements are being permuted.
        In addition, return an updated version of `element_indices_dict` a dict whose keys are
        the tree's (unpermuted) circuit indices and whose values are the final element indices for
        each circuit.
        """
        #default is to leave element indices alone, and assume that permuting the circuit
        # indices doesn't affect how the elements are ordered (usually this is false, and
        # the derived class should implement this).
        return element_indices_dict.copy()

    def _permute_simplified_circuit_xs(self, simplified_circuit_xs, element_indices, old_indices_in_new_order):
        """
        Updates simplified_circuit_xs by the old->new circuit mapping specified by
        `old_indices_in_new_order`.  Because the order of simplified_circuit_xs implied
        an element ordering, this routine also returns an updated_elIndices mapping
        that provides an updated element-index array (i.e. el_indices[circuitIndex] = slice of element indices)

        Parameters
        ----------
        simplified_circuit_xs : list
            list of n_circuits (#final circuits) lists, each
            of whose length gives the number of elements for that circuit (the values are
            immaterial - it could be an elabel or spamtuple, etc).

        old_indices_in_new_order : list
            giving the new ordering of the old indices.

        Returns
        -------
        updated_simplified_circuit_Xs : list
        updated_element_indices : OrderedDict
        """

        # Setting simplified_circuit_xs, (re)sets the element ordering,
        # so before doing this compute the old_to_new mapping and update
        # el_indices_dict.

        # just assume that len(simplified_circuit_xs[k]) gives number of elements
        # for circuit k.

        old_finalStringToElsMap = []; i = 0
        for k, Xs in enumerate(simplified_circuit_xs):
            old_finalStringToElsMap.append(list(range(i, i + len(Xs))))
            i += len(Xs)

        permute_newToOld = []
        for iOldStr in old_indices_in_new_order[0:self.num_final_strings()]:
            permute_newToOld.extend(old_finalStringToElsMap[iOldStr])
        permute_oldToNew = {iOld: iNew for iNew, iOld in enumerate(permute_newToOld)}

        updated_elIndices = _collections.OrderedDict()
        for ky, indices in element_indices.items():
            updated_elIndices[ky] = _slct.list_to_slice(
                [permute_oldToNew[x] for x in
                 (_slct.indices(indices) if isinstance(indices, slice) else indices)])

        # Now update simplified_circuit_xs
        updated_simplified_circuit_Xs = [simplified_circuit_xs[iCur]
                                         for iCur in old_indices_in_new_order[0:self.num_final_strings()]]
        return updated_simplified_circuit_Xs, updated_elIndices

    def is_split(self):
        """
        Whether tree is split into sub-trees or not.

        Returns
        -------
        bool
        """
        return len(self.subTrees) > 0

    def get_sub_trees(self):
        """
        A list of all the sub-trees (also EvalTree instances) of this tree.

        If this tree is not split, returns a single-element list containing just the tree.

        Returns
        -------
        list
        """
        if self.is_split():
            return self.subTrees
        else:
            return [self]  # return self as the only "subTree" when not split

    #NOT NEEDED?
    #def _compute_finalStringToEls(self):
    #    #Create a mapping from each final operation sequence (index) to
    #    # a slice of final element indices
    #    self.finalStringToElsMap = []; i=0
    #    for k,spamTuples in enumerate(self.simplified_circuit_spamTuples):
    #        self.finalStringToElsMap.append( slice(i,i+len(spamTuples)) )
    #        i += len(spamTuples)

    #PRIVATE
    def print_analysis(self):
        """
        Print a brief analysis of this tree.

        Used for debugging and assessing tree quality.

        Returns
        -------
        None
        """

        #Analyze tree
        if not self.is_split():
            print("Size of eval_tree = %d" % len(self))
            print("Size of circuit_list = %d" % self.num_final_strings())

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

        else:  # tree is split
            print("Size of original tree = %d" % len(self))
            print("Size of original circuit_list = %d" % self.num_final_strings())
            print("Tree is split into %d sub-trees" % len(self.subTrees))
            print("Sub-tree lengths = ", list(map(len, self.subTrees)), " (Sum = %d)" % sum(map(len, self.subTrees)))
            #for i,t in enumerate(self.subTrees):
            #    print(">> sub-tree %d: " % i)
            #    t.print_analysis()
