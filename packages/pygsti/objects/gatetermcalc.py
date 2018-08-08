""" Defines the GateTermCalc calculator class"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import warnings as _warnings
import numpy as _np
import time as _time
import itertools as _itertools
import functools as _functools
import operator as _operator

from ..tools import mpitools as _mpit
from ..tools import slicetools as _slct
from ..tools import compattools as _compat
from ..tools import listtools as _lt
from ..tools.matrixtools import _fas
from ..baseobjs import DummyProfiler as _DummyProfiler
from ..baseobjs import Label as _Label
from .termevaltree import TermEvalTree as _TermEvalTree
from .gatecalc import GateCalc
from .polynomial import Polynomial as _Polynomial

# For debug: sometimes helpful as it prints (python-only) tracebacks from segfaults
#import faulthandler
#faulthandler.enable()

try:
    from . import fastreplib as replib
except ImportError:
    from . import replib

try:
    from . import fastgatecalc as _fastgatecalc
    def _bulk_eval_compact_polys(vtape, ctape, paramvec, dest_shape):
        if _np.iscomplexobj(ctape):
            ret = _fastgatecalc.fast_bulk_eval_compact_polys_complex(
                vtape, ctape, paramvec, dest_shape)
            assert(_np.linalg.norm(_np.imag(ret)) < 1e-6 ) # DEBUG CHECK
            return _np.real( ret )
        else:
            return _np.real( _fastgatecalc.fast_bulk_eval_compact_polys(
                vtape, ctape, paramvec, dest_shape) )
except ImportError:
    from .polynomial import bulk_eval_compact_polys as _bulk_eval_compact_polys

_dummy_profiler = _DummyProfiler()

class GateTermCalc(GateCalc):
    """
    Encapsulates a calculation tool used by gate set objects that evaluates
    probabilities to some order in a small (error) parameter using Gates
    that can be expanded into terms of different orders and PureStateSPAMVecs.
    """

    def __init__(self, dim, gates, preps, effects, paramvec, autogator, max_order, cache=None):
        """
        Construct a new GateTermCalc object.

        Parameters
        ----------
        dim : int
            The gate-dimension.  All gate matrices should be dim x dim, and all
            SPAM vectors should be dim x 1.

        gates, preps, effects : OrderedDict
            Ordered dictionaries of Gate, SPAMVec, and SPAMVec objects,
            respectively.  Must be *ordered* dictionaries to specify a
            well-defined column ordering when taking derivatives.

        paramvec : ndarray
            The parameter vector of the GateSet.

        autogator : AutoGator
            An auto-gator object that may be used to construct virtual gates
            for use in computations.

        max_order : int 
            The maximum order of error-rate terms to include in probability
            computations.

        cache : dict, optional
            A dictionary of pre-computed compact polynomial objects.  Keys are
            `(max_order, rholabel, elabel, gatestring)` tuples, where 
            `max_order` is an integer, `rholabel` and `elabel` are
            :class:`Label` objects, and `gatestring` is a :class:`GateString`.
            Computed values are added to any dictionary that is supplied, so
            supplying an empty dictionary and using this calculator will cause
            the dictionary to be filled with values.
        """
        # self.unitary_evolution = False # Unused - idea was to have this flag
        #    allow unitary-evolution calcs to be term-based, which essentially
        #    eliminates the "pRight" portion of all the propagation calcs, and
        #    would require pLeft*pRight => |pLeft|^2
        self.max_order = max_order
        self.cache = cache
        super(GateTermCalc, self).__init__(
            dim, gates, preps, effects, paramvec, autogator)
        if self.evotype not in ("svterm","cterm"):
            raise ValueError(("Evolution type %s is incompatbile with "
                              "term-based calculations" % self.evotype))
        
    def copy(self):
        """ Return a shallow copy of this GateMatrixCalc """
        return GateTermCalc(self.dim, self.gates, self.preps,
                            self.effects, self.paramvec,
                            self.autogator, self.max_order)

        
    def _rhoE_from_spamTuple(self, spamTuple):
        assert( len(spamTuple) == 2 )
        if isinstance(spamTuple[0],_Label):
            rholabel,elabel = spamTuple
            rho = self.preps[rholabel]
            E   = self.effects[elabel]
        else:
            # a "custom" spamLabel consisting of a pair of SPAMVec (or array)
            #  objects: (prepVec, effectVec)
            rho, E = spamTuple
        return rho,E

    def _rhoEs_from_labels(self, rholabel, elabels):
        """ Returns SPAMVec *objects*, so must call .todense() later """
        rho = self.preps[rholabel]
        Es = [ self.effects[elabel] for elabel in elabels ]
        #No support for "custom" spamlabel stuff here
        return rho,Es

    def propagate_state(self, rho, factors, adjoint=False):
        # TODO UPDATE
        """ 
        State propagation by GateMap objects which have 'acton'
        methods.  This function could easily be overridden to 
        perform some more sophisticated state propagation
        (i.e. Monte Carlo) in the future.

        Parameters
        ----------
        rho : SPAMVec
           The spam vector representing the initial state.

        gatestring : GateString or tuple
           A tuple of labels specifying the gate sequence to apply.

        Returns
        -------
        SPAMVec
        """
        if adjoint:
            for f in factors:
                rho = f.adjoint_acton(rho) # LEXICOGRAPHICAL VS MATRIX ORDER
        else:
            for f in factors:
                rho = f.acton(rho) # LEXICOGRAPHICAL VS MATRIX ORDER
        return rho

    def prs_as_polys(self, rholabel, elabels, gatestring, comm=None, memLimit=None):
        """
        Computes polynomials of the probabilities for multiple spam-tuples of
        `gatestring`, sharing the same state preparation (so with different
        POVM effects).

        Parameters
        ----------
        rho_label : Label
            The state preparation label.
        
        elabels : list
            A list of :class:`Label` objects giving the *compiled* effect labels.

        gatestring : GateString or tuple
            A tuple-like object of *compiled* gates (e.g. may include
            instrument elements like 'Imyinst_0')
        
        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        memLimit : int, optional
            A memory limit in bytes to impose on the computation.
        
        Returns
        -------
        list
            A list of Polynomial objects.
        """
        #Cache hold *compact* polys now: see prs_as_compact_polys
        #cache_keys = [(self.max_order, rholabel, elabel, gatestring) for elabel in tuple(elabels)]
        #if self.cache is not None and all([(ck in self.cache) for ck in cache_keys]):
        #    return [ self.cache[ck] for ck in cache_keys ]
        
        fastmode = True
        if self.evotype == "svterm":
            poly_reps = replib.SV_prs_as_polys(self, rholabel, elabels, gatestring, comm, memLimit, fastmode)
        else: # "cterm" (stabilizer-based term evolution)
            poly_reps = replib.SB_prs_as_polys(self, rholabel, elabels, gatestring, comm, memLimit, fastmode)
        prps = [ _Polynomial.fromrep(rep) for rep in poly_reps ]

        #Cache hold *compact* polys now: see prs_as_compact_polys
        #if self.cache is not None:
        #    for ck,poly in zip(cache_keys,prps):
        #        self.cache[ck] = poly
        return prps
        

    def pr_as_poly(self, spamTuple, gatestring, comm=None, memLimit=None):
        """
        Compute probability of a single "outcome" (spam-tuple) for a single
        gate string.

        Parameters
        ----------
        spamTuple : (rho_label, compiled_effect_label)
            Specifies the prep and POVM effect used to compute the probability.

        gatestring : GateString or tuple
            A tuple-like object of *compiled* gates (e.g. may include
            instrument elements like 'Imyinst_0')
        
        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        memLimit : int, optional
            A memory limit in bytes to impose on the computation.
        
        Returns
        -------
        Polynomial
        """
        return self.prs_as_polys(spamTuple[0], [spamTuple[1]], gatestring,
                                 comm, memLimit)[0]

    def prs_as_compact_polys(self, rholabel, elabels, gatestring, comm=None, memLimit=None):
        """
        Computes compact-form polynomials of the probabilities for multiple
        spam-tuples of `gatestring`, sharing the same state preparation (so
        with different POVM effects).

        Parameters
        ----------
        rholabel : Label
            The state preparation label.
        
        elabels : list
            A list of :class:`Label` objects giving the *compiled* effect labels.

        gatestring : GateString or tuple
            A tuple-like object of *compiled* gates (e.g. may include
            instrument elements like 'Imyinst_0')
        
        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        memLimit : int, optional
            A memory limit in bytes to impose on the computation.
        
        Returns
        -------
        list
            A list of Polynomial objects.
        """
        cache_keys = [(self.max_order, rholabel, elabel, gatestring) for elabel in tuple(elabels)]
        if self.cache is not None and all([(ck in self.cache) for ck in cache_keys]):
            return [ self.cache[ck] for ck in cache_keys ]

        raw_prps = self.prs_as_polys(rholabel, elabels, gatestring, comm, memLimit)
        prps = [ poly.compact(force_complex=True) for poly in raw_prps ]
          # create compact polys w/*complex* coeffs always since we're likely
          # going to concatenate a bunch of them.

        if self.cache is not None:
            for ck,poly in zip(cache_keys,prps):
                self.cache[ck] = poly
        return prps
        

    # TODO REMOVE (UNUSED)
    #def pr(self, spamTuple, gatestring, clipTo, bScale):
    #    """
    #    Compute probability of a single "outcome" (spam-tuple) for a single
    #    gate string.
    #
    #    Parameters
    #    ----------
    #    spamTuple : (rho_label, compiled_effect_label)
    #        Specifies the prep and POVM effect used to compute the probability.
    #
    #    gatestring : GateString or tuple
    #        A tuple-like object of *compiled* gates (e.g. may include
    #        instrument elements like 'Imyinst_0')
    #
    #    clipTo : 2-tuple
    #      (min,max) to clip returned probability to if not None.
    #      Only relevant when prMxToFill is not None.
    #
    #    bScale : bool, optional
    #      Unused.  Present to match function signature of other calculators.
    #
    #    Returns
    #    -------
    #    probability: float
    #    """
    #    #OLD: poly = self.pr_as_poly(spamTuple, gatestring)
    #    #OLD: p = _np.real_if_close(cpoly.evaluate(self.paramvec))
    #    cpoly = self.prs_as_compact_polys(spamTuple[0], [spamTuple[1]], gatestring)[0]
    #    val = _bulk_eval_compact_polys(cpoly[0], cpoly[1], self.paramvec, (1,))[0]
    #    p = _np.real_if_close(val)
    #    if clipTo is not None:  p = _np.clip( p, clipTo[0], clipTo[1] )
    #    return float(p)


    def prs(self, rholabel, elabels, gatestring, clipTo, bUseScaling=False):
        """
        Compute probabilities of a multiple "outcomes" (spam-tuples) for a single
        gate string.  The spam tuples may only vary in their effect-label (their
        prep labels must be the same)

        Parameters
        ----------
        rholabel : Label
            The state preparation label.
        
        elabels : list
            A list of :class:`Label` objects giving the *compiled* effect labels.

        gatestring : GateString or tuple
            A tuple-like object of *compiled* gates (e.g. may include
            instrument elements like 'Imyinst_0')

        clipTo : 2-tuple
          (min,max) to clip returned probability to if not None.
          Only relevant when prMxToFill is not None.

        bUseScaling : bool, optional
          Unused.  Present to match function signature of other calculators.

        Returns
        -------
        numpy.ndarray
            An array of floating-point probabilities, corresponding to
            the elements of `elabels`.
        """
        cpolys = self.prs_as_compact_polys(rholabel, elabels, gatestring)
        vals = [ _bulk_eval_compact_polys(cpoly[0], cpoly[1], self.paramvec, (1,))[0]
                 for cpoly in cpolys ]
        ps = _np.array([ _np.real_if_close(val) for val in vals ])
        if clipTo is not None:  ps = _np.clip( ps, clipTo[0], clipTo[1] )
        return ps
    

    def dpr(self, spamTuple, gatestring, returnPr, clipTo):
        """
        Compute the derivative of a probability generated by a gate string and
        spam tuple as a 1 x M numpy array, where M is the number of gateset
        parameters.

        Parameters
        ----------
        spamTuple : (rho_label, compiled_effect_label)
            Specifies the prep and POVM effect used to compute the probability.

        gatestring : GateString or tuple
            A tuple-like object of *compiled* gates (e.g. may include
            instrument elements like 'Imyinst_0')

        returnPr : bool
          when set to True, additionally return the probability itself.

        clipTo : 2-tuple
          (min,max) to clip returned probability to if not None.
          Only relevant when prMxToFill is not None.

        Returns
        -------
        derivative : numpy array
            a 1 x M numpy array of derivatives of the probability w.r.t.
            each gateset parameter (M is the length of the vectorized gateset).

        probability : float
            only returned if returnPr == True.
        """
        dp = _np.empty( (1,self.Np), 'd' )
        
        poly = self.pr_as_poly(spamTuple, gatestring, comm=None, memLimit=None)
        for i in range(self.Np):
            dpoly_di = poly.deriv(i)
            dp[0,i] = dpoly.evaluate(self.paramvec)
            
        if returnPr:
            p = poly.evaluate(self.paramvec)
            if clipTo is not None:  p = _np.clip( p, clipTo[0], clipTo[1] )
            return dp, p
        else: return dp


    def hpr(self, spamTuple, gatestring, returnPr, returnDeriv, clipTo):
        """
        Compute the Hessian of a probability generated by a gate string and
        spam tuple as a 1 x M x M array, where M is the number of gateset
        parameters.

        Parameters
        ----------
        spamTuple : (rho_label, compiled_effect_label)
            Specifies the prep and POVM effect used to compute the probability.

        gatestring : GateString or tuple
            A tuple-like object of *compiled* gates (e.g. may include
            instrument elements like 'Imyinst_0')

        returnPr : bool
          when set to True, additionally return the probability itself.

        returnDeriv : bool
          when set to True, additionally return the derivative of the
          probability.

        clipTo : 2-tuple
          (min,max) to clip returned probability to if not None.
          Only relevant when prMxToFill is not None.

        Returns
        -------
        hessian : numpy array
            a 1 x M x M array, where M is the number of gateset parameters.
            hessian[0,j,k] is the derivative of the probability w.r.t. the
            k-th then the j-th gateset parameter.

        derivative : numpy array
            only returned if returnDeriv == True. A 1 x M numpy array of
            derivatives of the probability w.r.t. each gateset parameter.

        probability : float
            only returned if returnPr == True.
        """
        hp = _np.empty( (1,self.Np, self.Np), 'd' )
        if returnDeriv:
            dp = _np.empty( (1,self.Np), 'd' )
        
        poly = self.pr_as_poly(spamTuple, gatestring, comm=None, memLimit=None)
        for j in range(self.Np):
            dpoly_dj = poly.deriv(j)
            if returnDeriv:
                dp[0,j] = dpoly_dj.evaluate(self.paramvec)
                
            for i in range(self.Np):
                dpoly_didj = dpoly_dj.deriv(i)
                hp[0,i,j] = dpoly_didj.evaluate(self.paramvec)

        if returnPr:
            p = poly.evaluate(self.paramvec)
            if clipTo is not None:  p = _np.clip( p, clipTo[0], clipTo[1] )
            
            if returnDeriv: return hp, dp, p
            else:           return hp, p
        else:
            if returnDeriv: return hp, dp
            else:           return hp


    def default_distribute_method(self):
        """ 
        Return the preferred MPI distribution mode for this calculator.
        """
        return "gatestrings"

        
    def construct_evaltree(self):
        """
        Constructs an EvalTree object appropriate for this calculator.
        """
        return _TermEvalTree()


    def estimate_mem_usage(self, subcalls, cache_size, num_subtrees,
                           num_subtree_proc_groups, num_param1_groups,
                           num_param2_groups, num_final_strs):
        """
        Estimate the memory required by a given set of subcalls to computation functions.

        Parameters
        ----------
        subcalls : list of strs
            A list of the names of the subcalls to estimate memory usage for.

        cache_size : int
            The size of the evaluation tree that will be passed to the
            functions named by `subcalls`.

        num_subtrees : int
            The number of subtrees to split the full evaluation tree into.

        num_subtree_proc_groups : int
            The number of processor groups used to (in parallel) iterate through
            the subtrees.  It can often be useful to have fewer processor groups
            then subtrees (even == 1) in order to perform the parallelization
            over the parameter groups.
        
        num_param1_groups : int
            The number of groups to divide the first-derivative parameters into.
            Computation will be automatically parallelized over these groups.

        num_param2_groups : int
            The number of groups to divide the second-derivative parameters into.
            Computation will be automatically parallelized over these groups.

        num_final_strs : int
            The number of final strings (may be less than or greater than
            `cacheSize`) the tree will hold.
        
        Returns
        -------
        int
            The memory estimate in bytes.
        """
        np1,np2 = num_param1_groups, num_param2_groups
        FLOATSIZE = 8 # in bytes: TODO: a better way

        dim = self.dim
        wrtLen1 = (self.Np+np1-1) // np1 # ceiling(num_params / np1)
        wrtLen2 = (self.Np+np2-1) // np2 # ceiling(num_params / np2)

        mem = 0
        for fnName in subcalls:
            if fnName == "bulk_fill_probs":
                mem += num_final_strs # pr cache final (* #elabels!)

            elif fnName == "bulk_fill_dprobs":
                mem += num_final_strs * wrtLen1 # dpr cache final (* #elabels!)

            elif fnName == "bulk_fill_hprobs":
                mem += num_final_strs * wrtLen1 * wrtLen2  # hpr cache final (* #elabels!)
                
            else:
                raise ValueError("Unknown subcall name: %s" % fnName)
        
        return mem * FLOATSIZE


    
    def bulk_fill_probs(self, mxToFill, evalTree, clipTo=None, check=False,
                        comm=None):
        """
        Compute the outcome probabilities for an entire tree of gate strings.

        This routine fills a 1D array, `mxToFill` with the probabilities
        corresponding to the *compiled* gate strings found in an evaluation
        tree, `evalTree`.  An initial list of (general) :class:`GateString`
        objects is *compiled* into a lists of gate-only sequences along with
        a mapping of final elements (i.e. probabilities) to gate-only sequence
        and prep/effect pairs.  The evaluation tree organizes how to efficiently
        compute the gate-only sequences.  This routine fills in `mxToFill`, which
        must have length equal to the number of final elements (this can be 
        obtained by `evalTree.num_final_elements()`.  To interpret which elements
        correspond to which strings and outcomes, you'll need the mappings 
        generated when the original list of `GateStrings` was compiled.

        Parameters
        ----------
        mxToFill : numpy ndarray
          an already-allocated 1D numpy array of length equal to the
          total number of computed elements (i.e. evalTree.num_final_elements())

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the *compiled* gate
           strings to compute the bulk operation on.

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

        #get distribution across subtrees (groups if needed)
        subtrees = evalTree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = evalTree.distribute(comm)

        #eval on each local subtree
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]
            nStrs = evalSubTree.num_final_strings()

            def calc_and_fill(rholabel, elabels, fIndsList, gIndsList, pslc1, pslc2, sumInto):
                """ Compute and fill result quantities for given arguments """
                polys = evalSubTree.get_p_polys(self, rholabel, elabels, mySubComm) # computes polys if necessary

                for i,(fInds,gInds) in enumerate(zip(fIndsList,gIndsList)):
                    #use cached data to final values
                    prCache = _bulk_eval_compact_polys(polys[i][0], polys[i][1], self.paramvec, (nStrs,) ) # ( nGateStrings,)
                    ps = evalSubTree.final_view( prCache, axis=0) # ( nGateStrings,)
                    _fas(mxToFill, [fInds], ps[gInds], add=sumInto)

            self._fill_result_tuple_collectrho((mxToFill,), evalSubTree,
                                     slice(None), slice(None), calc_and_fill )

        #collect/gather results
        subtreeElementIndices = [ t.final_element_indices(evalTree) for t in subtrees]
        _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                             mxToFill,[], 0, comm)
        #note: pass mxToFill, dim=(KS,), so gather mxToFill[felInds] (axis=0)

        if clipTo is not None:
            _np.clip( mxToFill, clipTo[0], clipTo[1], out=mxToFill ) # in-place clip

#Will this work?? TODO
#        if check:
#            self._check(evalTree, spam_label_rows, mxToFill, clipTo=clipTo)



    def bulk_fill_dprobs(self, mxToFill, evalTree,
                         prMxToFill=None,clipTo=None,check=False,
                         comm=None, wrtFilter=None, wrtBlockSize=None,
                         profiler=None, gatherMemLimit=None):
        """
        Compute the outcome probability-derivatives for an entire tree of gate
        strings.

        Similar to `bulk_fill_probs(...)`, but fills a 2D array with
        probability-derivatives for each "final element" of `evalTree`.

        Parameters
        ----------
        mxToFill : numpy ndarray
          an already-allocated ExM numpy array where E is the total number of
          computed elements (i.e. evalTree.num_final_elements()) and M is the 
          number of gate set parameters.

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the *compiled* gate
           strings to compute the bulk operation on.

        prMxToFill : numpy array, optional
          when not None, an already-allocated length-E numpy array that is filled
          with probabilities, just like in bulk_fill_probs(...).

        clipTo : 2-tuple, optional
           (min,max) to clip return value if not None.

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

        if wrtFilter is not None:
            assert(wrtBlockSize is None) #Cannot specify both wrtFilter and wrtBlockSize
            wrtSlice = _slct.list_to_slice(wrtFilter) #for now, require the filter specify a slice
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
            felInds = evalSubTree.final_element_indices(evalTree)
            nStrs = evalSubTree.num_final_strings()

            #Free memory from previous subtree iteration before computing caches
            paramSlice = slice(None)
            fillComm = mySubComm #comm used by calc_and_fill

            def calc_and_fill(rholabel, elabels, fIndsList, gIndsList, pslc1, pslc2, sumInto):
                """ Compute and fill result quantities for given arguments """
                tm = _time.time()
                
                if prMxToFill is not None:
                    polys = evalSubTree.get_p_polys(self, rholabel, elabels, fillComm)
                    for i,(fInds,gInds) in enumerate(zip(fIndsList,gIndsList)):
                        prCache = _bulk_eval_compact_polys(polys[i][0], polys[i][1], self.paramvec, (nStrs,) ) # ( nGateStrings,)
                        ps = evalSubTree.final_view( prCache, axis=0) # ( nGateStrings,)
                        _fas(prMxToFill, [fInds], ps[gInds], add=sumInto)

                #Fill cache info
                dpolys = evalSubTree.get_dp_polys(self, rholabel, elabels, paramSlice, fillComm)
                nP = self.Np if (paramSlice is None or paramSlice.start is None) else _slct.length(paramSlice)
                for i,(fInds,gInds) in enumerate(zip(fIndsList,gIndsList)):
                    dprCache = _bulk_eval_compact_polys(dpolys[i][0], dpolys[i][1], self.paramvec, (nStrs,nP) )
                    dps = evalSubTree.final_view( dprCache, axis=0) # ( nGateStrings, nDerivCols)
                    _fas(mxToFill, [fInds, pslc1], dps[gInds], add=sumInto)
                profiler.add_time("bulk_fill_dprobs: calc_and_fill", tm)

                
            #Set wrtBlockSize to use available processors if it isn't specified
            if wrtFilter is None:
                blkSize = wrtBlockSize #could be None
                if (mySubComm is not None) and (mySubComm.Get_size() > 1):
                    comm_blkSize = self.Np / mySubComm.Get_size()
                    blkSize = comm_blkSize if (blkSize is None) \
                        else min(comm_blkSize, blkSize) #override with smaller comm_blkSize
            else:
                blkSize = None # wrtFilter dictates block


            if blkSize is None:
                #Fill derivative cache info
                paramSlice = wrtSlice #specifies which deriv cols calc_and_fill computes
                
                #Compute all requested derivative columns at once
                self._fill_result_tuple_collectrho( (prMxToFill, mxToFill), evalSubTree,
                                                    slice(None), slice(None), calc_and_fill )
                profiler.mem_check("bulk_fill_dprobs: post fill")

            else: # Divide columns into blocks of at most blkSize
                assert(wrtFilter is None) #cannot specify both wrtFilter and blkSize
                nBlks = int(_np.ceil(self.Np / blkSize))
                  # num blocks required to achieve desired average size == blkSize
                blocks = _mpit.slice_up_range(self.Np, nBlks)

                #distribute derivative computation across blocks
                myBlkIndices, blkOwners, blkComm = \
                    _mpit.distribute_indices(list(range(nBlks)), mySubComm)
                if blkComm is not None:
                    _warnings.warn("Note: more CPUs(%d)" % mySubComm.Get_size()
                       +" than derivative columns(%d)!" % self.Np
                       +" [blkSize = %.1f, nBlks=%d]" % (blkSize,nBlks)) # pragma: no cover
                fillComm = blkComm #comm used by calc_and_fill

                for iBlk in myBlkIndices:
                    paramSlice = blocks[iBlk] #specifies which deriv cols calc_and_fill computes
                    self._fill_result_tuple_collectrho( 
                        (mxToFill,), evalSubTree,
                        blocks[iBlk], slice(None), calc_and_fill )
                    profiler.mem_check("bulk_fill_dprobs: post fill blk")

                #gather results
                tm = _time.time()
                _mpit.gather_slices(blocks, blkOwners, mxToFill,[felInds],
                                    1, mySubComm, gatherMemLimit)
                #note: gathering axis 1 of mxToFill[:,fslc], dim=(ks,M)
                profiler.add_time("MPI IPC", tm)
                profiler.mem_check("bulk_fill_dprobs: post gather blocks")

        #collect/gather results
        tm = _time.time()
        subtreeElementIndices = [ t.final_element_indices(evalTree) for t in subtrees]
        _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                             mxToFill,[], 0, comm, gatherMemLimit)
        #note: pass mxToFill, dim=(KS,M), so gather mxToFill[felInds] (axis=0)

        if prMxToFill is not None:
            _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                             prMxToFill,[], 0, comm)
            #note: pass prMxToFill, dim=(KS,), so gather prMxToFill[felInds] (axis=0)

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



    def bulk_fill_hprobs(self, mxToFill, evalTree,
                         prMxToFill=None, deriv1MxToFill=None, deriv2MxToFill=None, 
                         clipTo=None, check=False,comm=None, wrtFilter1=None, wrtFilter2=None,
                         wrtBlockSize1=None, wrtBlockSize2=None, gatherMemLimit=None):
        """
        Compute the outcome probability-Hessians for an entire tree of gate
        strings.

        Similar to `bulk_fill_probs(...)`, but fills a 3D array with
        probability-Hessians for each "final element" of `evalTree`.

        Parameters
        ----------
        mxToFill : numpy ndarray
          an already-allocated ExMxM numpy array where E is the total number of
          computed elements (i.e. evalTree.num_final_elements()) and M1 & M2 are
          the number of selected gate-set parameters (by wrtFilter1 and wrtFilter2).

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the *compiled* gate
           strings to compute the bulk operation on.

        prMxToFill : numpy array, optional
          when not None, an already-allocated length-E numpy array that is filled
          with probabilities, just like in bulk_fill_probs(...).

        derivMxToFill1, derivMxToFill2 : numpy array, optional
          when not None, an already-allocated ExM numpy array that is filled
          with probability derivatives, similar to bulk_fill_dprobs(...), but
          where M is the number of gateset parameters selected for the 1st and 2nd
          differentiation, respectively (i.e. by wrtFilter1 and wrtFilter2).

        clipTo : 2-tuple, optional
           (min,max) to clip return value if not None.

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

        profiler : Profiler, optional
          A profiler object used for to track timing and memory usage.

        gatherMemLimit : int, optional
          A memory limit in bytes to impose upon the "gather" operations
          performed as a part of MPI processor syncronization.

        Returns
        -------
        None
        """

        if wrtFilter1 is not None:
            assert(wrtBlockSize1 is None and wrtBlockSize2 is None) #Cannot specify both wrtFilter and wrtBlockSize
            wrtSlice1 = _slct.list_to_slice(wrtFilter1) #for now, require the filter specify a slice
        else:
            wrtSlice1 = None

        if wrtFilter2 is not None:
            assert(wrtBlockSize1 is None and wrtBlockSize2 is None) #Cannot specify both wrtFilter and wrtBlockSize
            wrtSlice2 = _slct.list_to_slice(wrtFilter2) #for now, require the filter specify a slice
        else:
            wrtSlice2 = None

        #get distribution across subtrees (groups if needed)
        subtrees = evalTree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = evalTree.distribute(comm)

        #eval on each local subtree
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]
            felInds = evalSubTree.final_element_indices(evalTree)
            fillComm = mySubComm
            nStrs = evalSubTree.num_final_strings()

            #Free memory from previous subtree iteration before computing caches
            paramSlice1 = slice(None)
            paramSlice2 = slice(None)

            def calc_and_fill(rholabel, elabels, fIndsList, gIndsList, pslc1, pslc2, sumInto):
                """ Compute and fill result quantities for given arguments """

                                
                if prMxToFill is not None:
                    polys = evalSubTree.get_p_polys(self, rholabel, elabels, fillComm)
                    for i,(fInds,gInds) in enumerate(zip(fIndsList,gIndsList)):
                        prCache = _bulk_eval_compact_polys(polys[i][0], polys[i][1], self.paramvec, (nStrs,) ) # ( nGateStrings,)
                        ps = evalSubTree.final_view( prCache, axis=0) # ( nGateStrings,)
                        _fas(prMxToFill, [fInds], ps[gInds], add=sumInto)

                nP1 = self.Np if (paramSlice1 is None or paramSlice1.start is None) else _slct.length(paramSlice1)
                nP2 = self.Np if (paramSlice2 is None or paramSlice2.start is None) else _slct.length(paramSlice2)
                        
                if deriv1MxToFill is not None:
                    dpolys = evalSubTree.get_dp_polys(self, rholabel, elabels, paramSlice, fillComm)
                    for i,(fInds,gInds) in enumerate(zip(fIndsList,gIndsList)):
                        dprCache = _bulk_eval_compact_polys(dpolys[i][0], dpolys[i][1], self.paramvec, (nStrs,nP1)) # ( nGateStrings, nDerivCols)
                        dps1 = evalSubTree.final_view( dprCache, axis=0) # ( nGateStrings, nDerivCols)
                        _fas(deriv1MxToFill, [fInds,pslc1], dps1[gInds], add=sumInto)
                    
                if deriv2MxToFill is not None:
                    if deriv1MxToFill is not None and paramSlice1 == paramSlice2:
                        dps2 = dps1
                        for i,(fInds,gInds) in enumerate(zip(fIndsList,gIndsList)):
                            _fas(deriv2MxToFill, [fInds,pslc2], dps2[gInds], add=sumInto)
                    else:
                        dpolys = evalSubTree.get_dp_polys(self, rholabel, elabels, paramSlice, fillComm)
                        for i,(fInds,gInds) in enumerate(zip(fIndsList,gIndsList)):
                            dprCache = _bulk_eval_compact_polys(dpolys[i][0], dpolys[i][1], self.paramvec, (nStrs,nP2)) # ( nGateStrings, nDerivCols)
                            dps2 = evalSubTree.final_view( dprCache, axis=0) # ( nGateStrings, nDerivCols)
                            _fas(deriv2MxToFill, [fInds,pslc2], dps2[gInds], add=sumInto)

                #Fill cache info
                hpolys = evalSubTree.get_hp_polys(self, rholabel, elabels, paramSlice1, paramSlice2, fillComm)
                for i,(fInds,gInds) in enumerate(zip(fIndsList,gIndsList)):
                    hprCache = _bulk_eval_compact_polys(hpolys[i][0], hpolys[i][1], self.paramvec, (nStrs,nP1,nP2)) # ( nGateStrings, nDerivCols1, nDerivCols2)
                    hps = evalSubTree.final_view( hprCache, axis=0) # ( nGateStrings, nDerivCols1, nDerivCols2)
                    _fas(mxToFill, [fInds,pslc1,pslc2], hps[gInds], add=sumInto)

            #Set wrtBlockSize to use available processors if it isn't specified
            if wrtFilter1 is None and wrtFilter2 is None:
                blkSize1 = wrtBlockSize1 #could be None
                blkSize2 = wrtBlockSize2 #could be None
                if (mySubComm is not None) and (mySubComm.Get_size() > 1):
                    comm_blkSize = self.Np / mySubComm.Get_size()
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
                self._fill_result_tuple_collectrho(
                    (prMxToFill, deriv1MxToFill, deriv2MxToFill, mxToFill),
                    evalSubTree, slice(None), slice(None), calc_and_fill)

            else: # Divide columns into blocks of at most blkSize
                assert(wrtFilter1 is None and wrtFilter2 is None) #cannot specify both wrtFilter and blkSize
                nBlks1 = int(_np.ceil(self.Np / blkSize1))
                nBlks2 = int(_np.ceil(self.Np / blkSize2))
                  # num blocks required to achieve desired average size == blkSize1 or blkSize2
                blocks1 = _mpit.slice_up_range(self.Np, nBlks1)
                blocks2 = _mpit.slice_up_range(self.Np, nBlks2)

                #distribute derivative computation across blocks
                myBlk1Indices, blk1Owners, blk1Comm = \
                    _mpit.distribute_indices(list(range(nBlks1)), mySubComm)

                myBlk2Indices, blk2Owners, blk2Comm = \
                    _mpit.distribute_indices(list(range(nBlks2)), blk1Comm)

                if blk2Comm is not None:
                    _warnings.warn("Note: more CPUs(%d)" % mySubComm.Get_size()
                       +" than hessian elements(%d)!" % (self.Np**2)
                       +" [blkSize = {%.1f,%.1f}, nBlks={%d,%d}]" % (blkSize1,blkSize2,nBlks1,nBlks2)) # pragma: no cover
                fillComm = blk2Comm #comm used by calc_and_fill

                for iBlk1 in myBlk1Indices:
                    paramSlice1 = blocks1[iBlk1]

                    for iBlk2 in myBlk2Indices:
                        paramSlice2 = blocks2[iBlk2]
                        self._fill_result_tuple_collectrho
                        ((prMxToFill, deriv1MxToFill, deriv2MxToFill, mxToFill),
                         evalSubTree, blocks1[iBlk1], blocks2[iBlk2], calc_and_fill)
    
                    #gather column results: gather axis 2 of mxToFill[felInds,blocks1[iBlk1]], dim=(ks,blk1,M)
                    _mpit.gather_slices(blocks2, blk2Owners, mxToFill,[felInds,blocks1[iBlk1]],
                                        2, blk1Comm, gatherMemLimit)

                #gather row results; gather axis 1 of mxToFill[felInds], dim=(ks,M,M)
                _mpit.gather_slices(blocks1, blk1Owners, mxToFill,[felInds],
                                    1, mySubComm, gatherMemLimit)
                if deriv1MxToFill is not None:
                    _mpit.gather_slices(blocks1, blk1Owners, deriv1MxToFill,[felInds],
                                        1, mySubComm, gatherMemLimit)
                if deriv2MxToFill is not None:
                    _mpit.gather_slices(blocks2, blk2Owners, deriv2MxToFill,[felInds],
                                        1, blk1Comm, gatherMemLimit) 
                   #Note: deriv2MxToFill gets computed on every inner loop completion
                   # (to save mem) but isn't gathered until now (but using blk1Comm).
                   # (just as prMxToFill is computed fully on each inner loop *iteration*!)
            
        #collect/gather results
        subtreeElementIndices = [ t.final_element_indices(evalTree) for t in subtrees]
        _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                             mxToFill,[], 0, comm, gatherMemLimit)
        if deriv1MxToFill is not None:
            _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                                 deriv1MxToFill,[], 0, comm, gatherMemLimit)
        if deriv2MxToFill is not None:
            _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                                 deriv2MxToFill,[], 0, comm, gatherMemLimit)
        if prMxToFill is not None:
            _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                                 prMxToFill,[], 0, comm)


        if clipTo is not None and prMxToFill is not None:
            _np.clip( prMxToFill, clipTo[0], clipTo[1], out=prMxToFill ) # in-place clip

        #TODO: check if this works
        #if check:
        #    self._check(evalTree, spam_label_rows,
        #                prMxToFill, deriv1MxToFill, mxToFill, clipTo)
