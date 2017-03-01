from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines the GateSetCalculator class"""

import warnings as _warnings
import numpy as _np
import numpy.linalg as _nla
import time as _time
import collections as _collections

from ..tools import gatetools as _gt
from ..tools import mpitools as _mpit
from ..tools import slicetools as _slct
from .profiler import DummyProfiler as _DummyProfiler
_dummy_profiler = _DummyProfiler()

# Smallness tolerances, used internally for conditional scaling required
# to control bulk products, their gradients, and their Hessians.
PSMALL = 1e-100
DSMALL = 1e-100
HSMALL = 1e-100

#TODO:
# Gate -> GateMatrix
# New "Gate" base class, new "GateMap" class
class GateMapCalculator(GateSetCalculator):
    """
    Encapsulates a calculation tool used by gate set objects to perform product
    and derivatives-of-product calculations.

    This is contained in a class separate from GateSet to allow for additional
    gate set classes (e.g. ones which use entirely different -- non-gate-local
    -- parameterizations of gate matrices and SPAM vectors) access to these
    fundamental operations.
    """

    def __init__(self, dim, gates, preps, effects, povm_identity, spamdefs,
                 remainderLabel, identityLabel):
        """
        Construct a new GateSetCalculator object.

        Parameters
        ----------
        dim : int
            The gate-dimension.  All gate matrices should be dim x dim, and all
            SPAM vectors should be dim x 1.

        gates, preps, effects : OrderedDict
            Ordered dictionaries of Gate, SPAMVec, and SPAMVec objects,
            respectively.  Must be *ordered* dictionaries to specify a
            well-defined column ordering when taking derivatives.

        povm_identity : SPAMVec
            Identity vector (shape must be dim x 1) used when spamdefs
            contains the value (<rho_label>,"remainder"), which specifies
            a POVM effect that is the identity minus the sum of all the
            effect vectors in effects.

        spamdefs : OrderedDict
            A dictionary whose keys are the allowed SPAM labels, and whose
            values are 2-tuples comprised of a state preparation label
            followed by a POVM effect label (both of which are strings,
            and keys of preps and effects, respectively, except for the
            special case when eith both or just the effect label is set
            to "remainder").

        remainderLabel : string
            A string that may appear in the values of spamdefs to designate
            special behavior.

        identityLabel : string
            The string used to designate the identity POVM vector.
        """
        super(GateMatrixCalculator, self).__init__(
            dim, gates, preps, effects, povm_identity, spamdefs,
            remainderLabel, identityLabel)


    def _pr_nr(self, spamLabel, gatestring, clipTo, bUseScaling):
        """ non-remainder version of pr(...) overridden by derived clases """
        
        (rholabel,elabel) = self.spamdefs[spamLabel]
        rho = self.preps[rholabel]
        E   = _np.conjugate(_np.transpose(self._get_evec(elabel)))

        for lGate in gatestring:
            rho = self.gates[lGate].acton(rho) # LEXICOGRAPHICAL VS MATRIX ORDER
        p = _np.dot(E,rho)

        if _np.isnan(p):
            if len(gatestring) < 10:
                strToPrint = str(gatestring)
            else:
                strToPrint = str(gatestring[0:10]) + " ... (len %d)" % len(gatestring)
            _warnings.warn("pr(%s) == nan" % strToPrint)

        if clipTo is not None:
            return _np.clip(p,clipTo[0],clipTo[1])
        else: return p

        
    def _dpr_nr(self, spamLabel, gatestring, returnPr, clipTo):
        """ non-remainder version of dpr(...) overridden by derived clases """
        
        #Finite difference derivative
        eps = 1e-7 #hardcoded?
        p = self.pr(spamLabel, gatestring, clipTo)
        dp = _np.empty( (1,self.tot_params), 'd' )
        k = 0

        def fd_deriv(dct, kk):
            for lbl in dct.keys():
                orig_vec = dct[lbl].to_vector()
                Np = dct[lbl].num_params()
                for i in range(Np):
                    vec = orig_vec.copy(); vec[i] += eps
                    dct[lbl].from_vector(vec)
                    dp[kk] = (self.pr(spamLabel, gatestring, clipTo)-p)/eps
                    kk += 1
                dct[lbl].from_vector(orig_vec)
            return kk
        
        k = fd_deriv(self.preps,k) #prep derivs
        k = fd_deriv(self.effects,k) #effect derivs
        k = fd_deriv(self.gates,k) #gate derivs
                
        if returnPr:
            if clipTo is not None:  p = _np.clip( p, clipTo[0], clipTo[1] )
            return dp, p
        else: return dp


    def _hpr_nr(self, spamLabel, gatestring, returnPr, returnDeriv, clipTo):
        """ non-remainder version of hpr(...) overridden by derived clases """
        
        #Finite difference hessian
        eps = 1e-7 #hardcoded?
        if returnPr:
            dp,p = self.dpr(spamLabel, gatestring, returnPr, clipTo)
        else:
            dp = self.dpr(spamLabel, gatestring, returnPr, clipTo)
        hp = _np.empty( (1,self.tot_params, self.tot_params), 'd' )
        k = 0

        def fd_hessian(dct, kk):
            for lbl in dct.keys():
                orig_vec = dct[lbl].to_vector()
                Np = dct[lbl].num_params()
                for i in range(Np):
                    vec = orig_vec.copy(); vec[i] += eps
                    dct[lbl].from_vector(vec)
                    dp[kk,:] = (self.dpr(spamLabel, gatestring, False, clipTo)-dp)/eps
                    kk += 1
                dct[lbl].from_vector(orig_vec)
            return kk
        
        k = fd_hessian(self.preps,k) #prep derivs
        k = fd_hessian(self.effects,k) #effect derivs
        k = fd_hessian(self.gates,k) #gate derivs
                
        if returnPr and clipTo is not None:
            p = _np.clip( p, clipTo[0], clipTo[1] )

        if returnDeriv:
            if returnPr: return hp, dp, p
            else:        return hp, dp
        else:
            if returnPr: return hp, p
            else:        return hp


    def _compute_pr_cache(self, rho, E, evalTree, comm):
        #HERE
        pass
    def _compute_dpr_cache(self, rho, E, evalTree, wrtSlice, comm):
        pass
    def _compute_hpr_cache(self, rho, E, evalTree, wrtSlice1, wrtSlice2, comm):
        pass

    def bulk_fill_probs(self, mxToFill, spam_label_rows,
                        evalTree, clipTo=None, check=False, comm=None):

        """
        Identical to bulk_probs(...) except results are
        placed into rows of a pre-allocated array instead
        of being returned in a dictionary.

        Specifically, the probabilities for all gate strings
        and a given SPAM label are placed into the row of
        mxToFill specified by spam_label_rows[spamLabel].

        Parameters
        ----------
        mxToFill : numpy ndarray
          an already-allocated KxS numpy array, where K is larger
          than the maximum value in spam_label_rows and S is equal
          to the number of gate strings (i.e. evalTree.num_final_strings())

        spam_label_rows : dictionary
          a dictionary with keys == spam labels and values which
          are integer row indices into mxToFill, specifying the
          correspondence between rows of mxToFill and spam labels.

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        clipTo : 2-tuple, optional
           (min,max) to clip return value if not None.

        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is performed over
           subtrees of evalTree (if it is split).

        Returns
        -------
        None
        """
        remainder_row_index = self._get_remainder_row_index(spam_label_rows)

        #get distribution across subtrees (groups if needed)
        subtrees = evalTree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = evalTree.distribute(comm)

        #eval on each local subtree
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]
            fslc = evalSubTree.final_slice(evalTree)

            #Free memory from previous subtree iteration before computing caches
            prCache = None

            def calc_and_fill(spamLabel, isp, fslc, pslc1, pslc2, sumInto):
                tm = _time.time()
                
                #Fill cache info
                rho,E = self._rhoE_from_spamLabel(spamLabel)
                prCache = self._compute_pr_cache(rho, E, evalSubTree, mySubComm)

                #use cached data to final values
                ps = evalSubTree.final_view( prCache, axis=0) # ( nGateStrings, )

                if sumInto:
                    mxToFill[isp,fslc] += ps
                else:
                    mxToFill[isp,fslc] = ps
                profiler.add_time("bulk_fill_probs: calc_and_fill", tm)

            self._fill_result_tuple( (mxToFill,), spam_label_rows,
                                     fslc, slice(None), slice(None), calc_and_fill )

        #collect/gather results
        subtreeFinalSlices = [ t.final_slice(evalTree) for t in subtrees]
        _mpit.gather_slices(subtreeFinalSlices, subTreeOwners, mxToFill,
                            1, comm) 
        #note: pass mxToFill, dim=(K,S), so gather mxToFill[:,fslc] (axis=1)

        if clipTo is not None:
            _np.clip( mxToFill, clipTo[0], clipTo[1], out=mxToFill ) # in-place clip

#Will this work?? TODO
#        if check:
#            self._check(evalTree, spam_label_rows, mxToFill, clipTo=clipTo)


    def bulk_fill_dprobs(self, mxToFill, spam_label_rows, evalTree,
                         prMxToFill=None,clipTo=None,check=False,
                         comm=None, wrtFilter=None, wrtBlockSize=None,
                         profiler=None, gatherMemLimit=None):

        """
        Identical to bulk_dprobs(...) except results are
        placed into rows of a pre-allocated array instead
        of being returned in a dictionary.

        Specifically, the probability derivatives for all gate
        strings and a given SPAM label are placed into
        mxToFill[ spam_label_rows[spamLabel] ].
        Optionally, probabilities can be placed into
        prMxToFill[ spam_label_rows[spamLabel] ]

        Parameters
        ----------
        mxToFill : numpy array
          an already-allocated KxSxM numpy array, where K is larger
          than the maximum value in spam_label_rows, S is equal
          to the number of gate strings (i.e. evalTree.num_final_strings()),
          and M is the length of the vectorized gateset.

        spam_label_rows : dictionary
          a dictionary with keys == spam labels and values which
          are integer row indices into mxToFill, specifying the
          correspondence between rows of mxToFill and spam labels.

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        prMxToFill : numpy array, optional
          when not None, an already-allocated KxS numpy array that is filled
          with the probabilities as per spam_label_rows, similar to
          bulk_fill_probs(...).

        clipTo : 2-tuple, optional
          (min,max) to clip returned probability to if not None.
          Only relevant when prMxToFill is not None.

        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is first performed over
           subtrees of evalTree (if it is split), and then over blocks (subsets)
           of the parameters being differentiated with respect to (see
           wrtBlockSize).

        wrtFilter : list of ints, optional
          If not None, a list of integers specifying which parameters
          to include in the derivative dimension. This argument is used
          internally for distributing calculations across multiple
          processors and to control memory usage.  Cannot be specified
          in conjuction with wrtBlockSize.

        wrtBlockSize : int or float, optional
          The maximum number of derivative columns to compute *products*
          for simultaneously.  None means compute all requested columns
          at once.  The  minimum of wrtBlockSize and the size that makes
          maximal use of available processors is used as the final block size.
          This argument must be None if wrtFilter is not None.  Set this to
          non-None to reduce amount of intermediate memory required.

        profiler : Profiler, optional
          A profiler object used for to track timing and memory usage.

        gatherMemLimit : int, optional
          A memory limit in bytes to impose upon the "gather" operations
          performed as a part of MPI processor syncronization.

        Returns
        -------
        None
        """

        tStart = _time.time()
        if profiler is None: profiler = _dummy_profiler

        remainder_row_index = self._get_remainder_row_index(spam_label_rows)

        if wrtFilter is not None:
            assert(wrtBlockSize is None) #Cannot specify both wrtFilter and wrtBlockSize
            wrtSlice = _slct.list_to_slice(wrtFilter) #for now, require the filter specify a slice (could break up into contiguous parts later?)
        else:
            wrtSlice = None

        profiler.mem_check("bulk_fill_dprobs: begin (expect ~ %.2fGB)" 
                           % (mxToFill.nbytes/(1024.0**3)) )

        #get distribution across subtrees (groups if needed)
        subtrees = evalTree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = evalTree.distribute(comm)

        #eval on each local subtree
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]
            fslc = evalSubTree.final_slice(evalTree)

            #Free memory from previous subtree iteration before computing caches
            prCache = dprCache = None
            paramSlice = slice(None)

            def calc_and_fill(spamLabel, isp, fslc, pslc1, pslc2, sumInto):
                tm = _time.time()
                rho,E = self._rhoE_from_spamLabel(spamLabel)
                
                if prMxToFill is not None:
                    prCache = self._compute_pr_cache(evalSubTree, mySubComm)
                    ps = evalSubTree.final_view( prCache, axis=0) # ( nGateStrings, )
                    if sumInto:
                        prMxToFill[isp,fslc] += ps
                    else:
                        prMxToFill[isp,fslc] = ps

                #Fill cache info
                dprCache = self._compute_dpr_cache(rho, E, evalSubTree, paramSlice, mySubComm)
                dps = evalSubTree.final_view( dprCache, axis=0) # ( nGateStrings, )

                if sumInto:
                    mxToFill[isp,fslc,pslc1] += dps
                else:
                    mxToFill[isp,fslc,pslc1] = dps
                profiler.add_time("bulk_fill_dprobs: calc_and_fill", tm)

                
            #Set wrtBlockSize to use available processors if it isn't specified
            if wrtFilter is None:
                blkSize = wrtBlockSize #could be None
                if (mySubComm is not None) and (mySubComm.Get_size() > 1):
                    comm_blkSize = self.tot_gate_params / mySubComm.Get_size()
                    blkSize = comm_blkSize if (blkSize is None) \
                        else min(comm_blkSize, blkSize) #override with smaller comm_blkSize
            else:
                blkSize = None # wrtFilter dictates block


            if blkSize is None:
                #Fill derivative cache info
                paramSlice = wrtSlice #specifies which deriv cols calc_and_fill computes
                
                #Compute all requested derivative columns at once
                self._fill_result_tuple( (prMxToFill, mxToFill), spam_label_rows,
                                         fslc, slice(None), slice(None), calc_and_fill )
                profiler.mem_check("bulk_fill_dprobs: post fill")

            else: # Divide columns into blocks of at most blkSize
                assert(wrtFilter is None) #cannot specify both wrtFilter and blkSize
                nBlks = int(_np.ceil(self.tot_params / blkSize))
                  # num blocks required to achieve desired average size == blkSize
                blocks = _mpit.slice_up_range(self.tot_params, nBlks)

                #distribute derivative computation across blocks
                myBlkIndices, blkOwners, blkComm = \
                    _mpit.distribute_indices(list(range(nBlks)), mySubComm)
                if blkComm is not None:
                    _warnings.warn("Note: more CPUs(%d)" % mySubComm.Get_size()
                       +" than derivative columns(%d)!" % self.tot_gate_params 
                       +" [blkSize = %.1f, nBlks=%d]" % (blkSize,nBlks))

                for iBlk in myBlkIndices:
                    paramSlice = blocks[iBlk] #specifies which deriv cols calc_and_fill computes
                    self._fill_result_tuple( 
                        (mxToFill,), spam_label_rows, fslc, 
                        blocks[iBlk], slice(None), calc_and_fill )
                    profiler.mem_check("bulk_fill_dprobs: post fill blk")
                    dProdCache = dGs = None #free mem

                #gather results
                tm = _time.time()
                _mpit.gather_slices(blocks, blkOwners, mxToFill[:,fslc],
                                    2, mySubComm, gatherMemLimit)
                #note: gathering axis 2 of mxToFill[:,fslc], dim=(K,s,M)
                profiler.add_time("MPI IPC", tm)
                profiler.mem_check("bulk_fill_dprobs: post gather blocks")

        #collect/gather results
        tm = _time.time()
        subtreeFinalSlices = [ t.final_slice(evalTree) for t in subtrees]
        _mpit.gather_slices(subtreeFinalSlices, subTreeOwners, mxToFill,
                            1, comm, gatherMemLimit) 
        #note: pass mxToFill, dim=(K,S,M), so gather mxToFill[:,fslc] (axis=1)

        if prMxToFill is not None:
            _mpit.gather_slices(subtreeFinalSlices, subTreeOwners, prMxToFill,
                                1, comm) 
        #note: pass prMxToFill, dim=(K,S), so gather prMxToFill[:,fslc] (axis=1)

        profiler.add_time("MPI IPC", tm)
        profiler.mem_check("bulk_fill_dprobs: post gather subtrees")

        if clipTo is not None and prMxToFill is not None:
            _np.clip( prMxToFill, clipTo[0], clipTo[1], out=prMxToFill ) # in-place clip

        #TODO: will this work?
        #if check:
        #    self._check(evalTree, spam_label_rows, prMxToFill, mxToFill,
        #                clipTo=clipTo)
        profiler.add_time("bulk_fill_dprobs: total", tStart)
        profiler.add_count("bulk_fill_dprobs count")
        profiler.mem_check("bulk_fill_dprobs: end")



    def bulk_fill_hprobs(self, mxToFill, spam_label_rows, evalTree,
                         prMxToFill=None, deriv1MxToFill=None, deriv2MxToFill=None, 
                         clipTo=None, check=False,comm=None, wrtFilter1=None, wrtFilter2=None,
                         wrtBlockSize1=None, wrtBlockSize2=None, gatherMemLimit=None):

        """
        Identical to bulk_hprobs(...) except results are
        placed into rows of a pre-allocated array instead
        of being returned in a dictionary.

        Specifically, the probability hessians for all gate
        strings and a given SPAM label are placed into
        mxToFill[ spam_label_rows[spamLabel] ].
        Optionally, probabilities and/or derivatives can be placed into
        prMxToFill[ spam_label_rows[spamLabel] ] and
        derivMxToFill[ spam_label_rows[spamLabel] ] respectively.

        Parameters
        ----------
        mxToFill : numpy array
          an already-allocated KxSxMxM numpy array, where K is larger
          than the maximum value in spam_label_rows, S is equal
          to the number of gate strings (i.e. evalTree.num_final_strings()),
          and M is the length of the vectorized gateset.

        spam_label_rows : dictionary
          a dictionary with keys == spam labels and values which
          are integer row indices into mxToFill, specifying the
          correspondence between rows of mxToFill and spam labels.

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        prMxToFill : numpy array, optional
          when not None, an already-allocated KxS numpy array that is filled
          with the probabilities as per spam_label_rows, similar to
          bulk_fill_probs(...).

        derivMxToFill1, derivMxToFill2 : numpy array, optional
          when not None, an already-allocated KxSxM numpy array that is filled
          with the probability derivatives as per spam_label_rows, similar to
          bulk_fill_dprobs(...), but where M is the number of gateset parameters
          selected for the 1st and 2nd differentiation, respectively (i.e. by
          wrtFilter1 and wrtFilter2).

        clipTo : 2-tuple
          (min,max) to clip returned probability to if not None.
          Only relevant when prMxToFill is not None.

        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is first performed over
           subtrees of evalTree (if it is split), and then over blocks (subsets)
           of the parameters being differentiated with respect to (see
           wrtBlockSize).

        wrtFilter1, wrtFilter2 : list of ints, optional
          If not None, a list of integers specifying which gate set parameters
          to differentiate with respect to in the first (row) and second (col)
          derivative operations, respectively.

        wrtBlockSize2, wrtBlockSize2 : int or float, optional
          The maximum number of 1st (row) and 2nd (col) derivatives to compute
          *products* for simultaneously.  None means compute all requested
          rows or columns at once.  The  minimum of wrtBlockSize and the size
          that makes maximal use of available processors is used as the final
          block size.  These arguments must be None if the corresponding
          wrtFilter is not None.  Set this to non-None to reduce amount of
          intermediate memory required.

        gatherMemLimit : int, optional
          A memory limit in bytes to impose upon the "gather" operations
          performed as a part of MPI processor syncronization.


        Returns
        -------
        None
        """
        remainder_row_index = self._get_remainder_row_index(spam_label_rows)

        if wrtFilter1 is not None:
            assert(wrtBlockSize1 is None and wrtBlockSize2 is None) #Cannot specify both wrtFilter and wrtBlockSize
            wrtSlice1 = _slct.list_to_slice(wrtFilter1) #for now, require the filter specify a slice (could break up into contiguous parts later?)
        else:
            wrtSlice1 = None

        if wrtFilter2 is not None:
            assert(wrtBlockSize1 is None and wrtBlockSize2 is None) #Cannot specify both wrtFilter and wrtBlockSize
            wrtSlice2 = _slct.list_to_slice(wrtFilter2) #for now, require the filter specify a slice (could break up into contiguous parts later?)
        else:
            wrtSlice2 = None

        #get distribution across subtrees (groups if needed)
        subtrees = evalTree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = evalTree.distribute(comm)

        #eval on each local subtree
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]
            fslc = evalSubTree.final_slice(evalTree)

            #Free memory from previous subtree iteration before computing caches
            prCache = dprCache1 = dprCache2 = hprCache = None
            paramSlice1 = slice(None)
            paramSlice2 = slice(None)

            def calc_and_fill(spamLabel, isp, fslc, pslc1, pslc2, sumInto):
                rho,E = self._rhoE_from_spamLabel(spamLabel)

                if prMxToFill is not None:
                    prCache = self._compute_pr_cache(evalSubTree, mySubComm)
                    ps = evalSubTree.final_view( prCache, axis=0) # ( nGateStrings, )
                    if sumInto:
                        prMxToFill[isp,fslc] += ps
                    else:
                        prMxToFill[isp,fslc] = ps

                if derivMxToFill1 is not None:
                    dprCache = self._compute_dpr_cache(rho, E, evalSubTree, paramSlice1, mySubComm)
                    dps1 = evalSubTree.final_view( dprCache, axis=0) # ( nGateStrings, )
                    if sumInto:
                        derivMxToFill1[isp,fslc,pslc1] += dps1
                    else:
                        derivMxToFill1[isp,fslc,pslc1] = dps1

                if derivMxToFill2 is not None:
                    if derivMxToFill1 is not None and paramSlice1 == paramSlice2:
                        dps2 = dps1
                    else:
                        dprCache = self._compute_dpr_cache(rho, E, evalSubTree, paramSlice2, mySubComm)
                        dps2 = evalSubTree.final_view( dprCache, axis=0) # ( nGateStrings, )
                        
                    if sumInto:
                        derivMxToFill2[isp,fslc,pslc2] += dps2
                    else:
                        derivMxToFill2[isp,fslc,pslc2] = dps2

                #Fill cache info
                hprCache = self._compute_hpr_cache(rho, E, evalSubTree, paramSlice1, paramSlice2, mySubComm)
                hps = evalSubTree.final_view( hprCache, axis=0) # ( nGateStrings, )

                if sumInto:
                    mxToFill[isp,fslc,pslc1,pslc2] += hps
                else:
                    mxToFill[isp,fslc,pslc1,pslc2]  = hps


            #Set wrtBlockSize to use available processors if it isn't specified
            if wrtFilter1 is None and wrtFilter2 is None:
                blkSize1 = wrtBlockSize1 #could be None
                blkSize2 = wrtBlockSize2 #could be None
                if (mySubComm is not None) and (mySubComm.Get_size() > 1):
                    comm_blkSize = self.tot_gate_params / mySubComm.Get_size()
                    blkSize1 = comm_blkSize if (blkSize1 is None) \
                        else min(comm_blkSize, blkSize1) #override with smaller comm_blkSize
                    blkSize2 = comm_blkSize if (blkSize2 is None) \
                        else min(comm_blkSize, blkSize2) #override with smaller comm_blkSize
            else:
                blkSize1 = blkSize2 = None # wrtFilter1 & wrtFilter2 dictates block


            if blkSize1 is None and blkSize2 is None:
                #Fill hessian cache info
                paramSlice1 = wrtSlice1 #specifies which deriv cols calc_and_fill computes
                paramSlice2 = wrtSlice2 #specifies which deriv cols calc_and_fill computes

                #Compute all requested derivative columns at once
                self._fill_result_tuple((prMxToFill, deriv1MxToFill, deriv2MxToFill, mxToFill),
                                        spam_label_rows, fslc, slice(None),
                                        slice(None), calc_and_fill)

            else: # Divide columns into blocks of at most blkSize
                assert(wrtFilter1 is None and wrtFilter2 is None) #cannot specify both wrtFilter and blkSize
                nBlks1 = int(_np.ceil(self.tot_params / blkSize1))
                nBlks2 = int(_np.ceil(self.tot_params / blkSize2))
                  # num blocks required to achieve desired average size == blkSize1 or blkSize2
                blocks1 = _mpit.slice_up_range(self.tot_params, nBlks1)
                blocks2 = _mpit.slice_up_range(self.tot_params, nBlks2)

                #distribute derivative computation across blocks
                myBlk1Indices, blk1Owners, blk1Comm = \
                    _mpit.distribute_indices(list(range(nBlks1)), mySubComm)

                myBlk2Indices, blk2Owners, blk2Comm = \
                    _mpit.distribute_indices(list(range(nBlks2)), blk1Comm)

                if blk2Comm is not None:
                    _warnings.warn("Note: more CPUs(%d)" % mySubComm.Get_size()
                       +" than hessian elements(%d)!" % (self.tot_gate_params**2)
                       +" [blkSize = {%.1f,%.1f}, nBlks={%d,%d}]" % (blkSize1,blkSize2,nBlks1,nBlks2))

                for iBlk1 in myBlk1Indices:
                    paramSlice1 = blocks1[iBlk1]

                    for iBlk2 in myBlk2Indices:
                        paramSlice2 = blocks2[iBlk2]
                        effectSlice2 = _slct.shift( _slct.intersect(blocks2[iBlk2],slice(self.tot_rho_params,self.tot_spam_params)), -self.tot_rho_params)
                        gateSlice2 = _slct.shift( _slct.intersect(blocks2[iBlk2],slice(self.tot_spam_params,None)), -self.tot_spam_params)

                        self._fill_result_tuple((prMxToFill, deriv1MxToFill, deriv2MxToFill, mxToFill),
                                                spam_label_rows, fslc, blocks1[iBlk1], blocks2[iBlk2],
                                                calc_and_fill)
    
                    #gather column results: gather axis 3 of mxToFill[:,fslc,blocks1[iBlk1]], dim=(K,s,blk1,M)
                    _mpit.gather_slices(blocks2, blk2Owners, mxToFill[:,fslc,blocks1[iBlk1]],
                                        3, blk1Comm, gatherMemLimit)

                #gather row results; gather axis 2 of mxToFill[:,fslc], dim=(K,s,M,M)
                _mpit.gather_slices(blocks1, blk1Owners, mxToFill[:,fslc],
                                    2, mySubComm, gatherMemLimit)
                if deriv1MxToFill is not None:
                    _mpit.gather_slices(blocks1, blk1Owners, deriv1MxToFill[:,fslc],
                                        2, mySubComm, gatherMemLimit)
                if deriv2MxToFill is not None:
                    _mpit.gather_slices(blocks2, blk2Owners, deriv2MxToFill[:,fslc],
                                        2, blk1Comm, gatherMemLimit) 
                   #Note: deriv2MxToFill gets computed on every inner loop completion
                   # (to save mem) but isn't gathered until now (but using blk1Comm).
                   # (just as prMxToFill is computed fully on each inner loop *iteration*!)
            
        #collect/gather results
        subtreeFinalSlices = [ t.final_slice(evalTree) for t in subtrees]
        _mpit.gather_slices(subtreeFinalSlices, subTreeOwners, 
                            mxToFill, 1, comm, gatherMemLimit) 
        if deriv1MxToFill is not None:
            _mpit.gather_slices(subtreeFinalSlices, subTreeOwners,
                                deriv1MxToFill, 1, comm, gatherMemLimit) 
        if deriv2MxToFill is not None:
            _mpit.gather_slices(subtreeFinalSlices, subTreeOwners,
                                deriv2MxToFill, 1, comm, gatherMemLimit) 
        if prMxToFill is not None:
            _mpit.gather_slices(subtreeFinalSlices, subTreeOwners,
                                prMxToFill, 1, comm) 


        if clipTo is not None and prMxToFill is not None:
            _np.clip( prMxToFill, clipTo[0], clipTo[1], out=prMxToFill ) # in-place clip

        #TODO: check if this works
        #if check:
        #    self._check(evalTree, spam_label_rows,
        #                prMxToFill, deriv1MxToFill, mxToFill, clipTo)
