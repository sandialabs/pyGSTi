""" Defines the GateMapCalc calculator class"""
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

from ..tools import mpitools as _mpit
from ..tools import slicetools as _slct
from ..tools import compattools as _compat
from ..tools.matrixtools import _fas
from ..tools import symplectic as _symp
from ..baseobjs import DummyProfiler as _DummyProfiler
from ..baseobjs import Label as _Label
from .mapevaltree import MapEvalTree as _MapEvalTree
from .gatecalc import GateCalc

try:
    from . import fastreplib as replib
except ImportError:
    from . import replib



_dummy_profiler = _DummyProfiler()

# FUTURE: use enum (make sure it's supported in Python2.7?)
SUPEROP  = 0
UNITARY  = 1
CLIFFORD = 2


#TODO:
# Gate -> GateMatrix
# New "Gate" base class, new "GateMap" class
class GateMapCalc(GateCalc):
    """
    Encapsulates a calculation tool used by gate set objects to perform product
    and derivatives-of-product calculations.

    This is contained in a class separate from GateSet to allow for additional
    gate set classes (e.g. ones which use entirely different -- non-gate-local
    -- parameterizations of gate matrices and SPAM vectors) access to these
    fundamental operations.
    """    
    def __init__(self, dim, gates, preps, effects, paramvec, autogator):
        """
        Construct a new GateMapCalc object.

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
        """
        super(GateMapCalc, self).__init__(
            dim, gates, preps, effects, paramvec, autogator)
        if self.evotype not in ("statevec","densitymx","stabilizer"):
            raise ValueError(("Evolution type %s is incompatbile with "
                              "map-based calculations" % self.evotype))

        
    def copy(self):
        """ Return a shallow copy of this GateMatrixCalc """
        return GateMapCalc(self.dim, self.gates, self.preps,
                              self.effects, self.paramvec, self.autogator)


    #UNUSED TODO REMOVE
    #Same as GateMatrixCalc, but not general enough to be in base class
    #def _rhoE_from_spamTuple(self, spamTuple):
    #    assert( len(spamTuple) == 2 )
    #    if isinstance(spamTuple[0],_Label): 
    #        rholabel,elabel = spamTuple
    #        if self.evotype in ("densitymx","statevec"):  # FUTURE: use enum (make sure it's supported in Python2.7?)
    #            typ = complex if self.evotype == "statevec" else 'd'
    #            scratch = _np.empty(self.preps[rholabel].dim, typ) # allocate local scratch
    #            rho = self.preps[rholabel].todense(scratch).copy() # copy b/c use scratch again (next line)
    #            E   = _np.conjugate(_np.transpose(self.effects[elabel].todense(scratch)))
    #        else: # CLIFFORD
    #            rho = self.preps[rholabel].todense()
    #            E = self.effects[elabel] # just return raw effect object
    #    else:
    #        # a "custom" spamLabel consisting of a pair of SPAMVec (or array)
    #        #  objects: (prepVec, effectVec)
    #        rho, Eraw = spamTuple
    #        E   = _np.conjugate(_np.transpose(Eraw))
    #    return rho,E
    #
    
    def _rhoEs_from_labels(self, rholabel, elabels):
        """ Returns SPAMVec *objects*, so must call .todense() later """
        rho = self.preps[rholabel]
        Es = [ self.effects[elabel] for elabel in elabels ]
        #No support for "custom" spamlabel stuff here
        return rho,Es

    #OLD: TODO REMOVE
    #def propagate_state(self, rho, gatestring):
    #    """ 
    #    State propagation by GateMap objects which have 'acton'
    #    methods.  This function could easily be overridden to 
    #    perform some more sophisticated state propagation
    #    (i.e. Monte Carlo) in the future.
    #
    #    Parameters
    #    ----------
    #    rho : SPAMVec
    #       The spam vector representing the initial state.
    #
    #    gatestring : GateString or tuple
    #       A tuple of labels specifying the gate sequence to apply.
    #
    #    Returns
    #    -------
    #    SPAMVec
    #    """
    #    #from .label import Label #DEBUG
    #    #print("INIT: \n",rho) #DEBUG
    #    for lbl in gatestring:
    #        rho = self.gates[lbl].acton(rho) # LEXICOGRAPHICAL VS MATRIX ORDER
    #        #print("AFTER %s: \n" % str(lbl),rho) #DEBUG HERE
    #    return rho


    #TODO REMOVE (UNUSED)
    #def pr(self, spamTuple, gatestring, clipTo, bUseScaling=False):
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
    #    bUseScaling : bool, optional
    #      Whether to use a post-scaled product internally.  If False, this
    #      routine will run slightly faster, but with a chance that the
    #      product will overflow and the subsequent trace operation will
    #      yield nan as the returned probability.
    #
    #    Returns
    #    -------
    #    probability: float
    #    """
    #    rholabel,elabel = spamTuple # can't handle custom rho/e -- this seems ok...
    #    rhorep = self.preps[rholabel].torep('prep')
    #    erep = self.effects[elabel].torep('effect')
    #    rhorep = replib.propagate_staterep(rhorep, [self._getgate(gl).torep() for gl in gatestring])
    #    p = erep.probability(rhorep) #outcome probability
    #
    #    #OLD DEPRECATED REPS TODO REMOVE
    #    #rho,E = self._rhoE_from_spamTuple(spamTuple)
    #    #rho = self.propagate_state(rho, gatestring)
    #    ## DEBUG print( " - state = ", rho.s)
    #    ## DEBUG print( "         = ", rho.ps)
    #    ## DEBUG print( "         = ", rho.a)
    #    #if self.evotype == "statevec":
    #    #    p_old = float(abs(_np.dot(E,rho))**2)
    #    #elif self.evotype == "densitymx":
    #    #    p_old = float(_np.dot(E,rho))
    #    #else: # evotype == "stabilizer"
    #    #    #print("MEASURE!!")
    #    #    p_old = rho.measurement_probability(E.outcomes)
    #    #    #a_old = rho.extract_amplitude(E.outcomes)
    #    #    # DEBUG print("AMP DEBUG COMP = ",amp,a_old)
    #    #    #assert(_np.isclose(amp,a_old)),"New code is giving a different amplitude result!"
    #    #if not (_np.isnan(p) and _np.isnan(p_old)):
    #    #    assert(_np.isclose(p,p_old)),"New code is giving a different result!"
    #
    #    if _np.isnan(p):
    #        if len(gatestring) < 10:
    #            strToPrint = str(gatestring)
    #        else:
    #            strToPrint = str(gatestring[0:10]) + " ... (len %d)" % len(gatestring)
    #        _warnings.warn("pr(%s) == nan" % strToPrint)
    #
    #    if clipTo is not None:
    #        return _np.clip(p,clipTo[0],clipTo[1])
    #    else: return p

        
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
        rhorep = self.preps[rholabel].torep('prep')
        ereps = [ self.effects[elabel].torep('effect') for elabel in elabels ]
        rhorep = replib.propagate_staterep(rhorep, [self._getgate(gl).torep() for gl in gatestring])
        ps = _np.array([ erep.probability(rhorep) for erep in ereps ], 'd')
          #outcome probabilities

        if _np.any(_np.isnan(ps)):
            if len(gatestring) < 10:
                strToPrint = str(gatestring)
            else:
                strToPrint = str(gatestring[0:10]) + " ... (len %d)" % len(gatestring)
            _warnings.warn("pr(%s) == nan" % strToPrint)

        if clipTo is not None:
            return _np.clip(ps,clipTo[0],clipTo[1])
        else: return ps

        
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
        
        #Finite difference derivative
        eps = 1e-7 #hardcoded?
        p = self.prs(spamTuple[0], [spamTuple[1]], gatestring, clipTo)[0]
        dp = _np.empty( (1,self.Np), 'd' )

        orig_vec = self.to_vector().copy()
        for i in range(self.Np):
            vec = orig_vec.copy(); vec[i] += eps
            self.from_vector(vec)
            dp[0,i] = (self.prs(spamTuple[0], [spamTuple[1]], gatestring, clipTo)-p)[0]/eps
        self.from_vector(orig_vec)
                
        if returnPr:
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
        
        #Finite difference hessian
        eps = 1e-4 #hardcoded?
        if returnPr:
            dp,p = self.dpr(spamTuple, gatestring, returnPr, clipTo)
        else:
            dp = self.dpr(spamTuple, gatestring, returnPr, clipTo)
        hp = _np.empty( (1,self.Np, self.Np), 'd' )

        orig_vec = self.to_vector().copy()
        for i in range(self.Np):
            vec = orig_vec.copy(); vec[i] += eps
            self.from_vector(vec)
            hp[0,i,:] = (self.dpr(spamTuple, gatestring, False, clipTo)-dp)/eps
        self.from_vector(orig_vec)
                
        if returnPr and clipTo is not None:
            p = _np.clip( p, clipTo[0], clipTo[1] )

        if returnDeriv:
            if returnPr: return hp, dp, p
            else:        return hp, dp
        else:
            if returnPr: return hp, p
            else:        return hp


    def _compute_pr_cache(self, rholabel, elabels, evalTree, comm, scratch=None):
        return replib.DM_compute_pr_cache(self, rholabel, elabels, evalTree, comm)

        #DEPRECATED REPS - just call replib version now - but below (commented) version
        # works for state-vec and stabilizer modes too - and we still need to essentially
        # duplicate DM_compute_pr_cache to SV and SB modes...
        ##tStart = _time.time()
        #dim = self.dim
        #cacheSize = evalTree.cache_size()
        #rhoVec,EVecs = self._rhoEs_from_labels(rholabel, elabels)
        #ret = _np.empty((len(evalTree),len(elabels)),'d')
        #
        ##Get rho & rhoCache
        #if self.evotype in ("statevec", "densitymx"):
        #    #Scratch type (for holding spam/state vectors)
        #    typ = complex if self.evotype == "statevec" else 'd'
        #    
        #    if scratch is None:
        #        rho_cache = _np.zeros((cacheSize, dim), typ)
        #    else:
        #        assert(scratch.shape == (cacheSize,dim))
        #        rho_cache = scratch #to avoid reallocation
        #        
        #    Escratch = _np.empty(dim,typ) # memory for E.todense() if it wants it
        #    rho = rhoVec.todense(Escratch).copy() #rho can use same scratch space (enables fastkron)
        #                                          # copy b/c use Escratch again (below)
        #else: # CLIFFORD case
        #    rho = rhoVec.todense()
        #    if scratch is None:
        #        rho_cache = [None]*cacheSize # so we can store (s,p) tuples in cache
        #    else:
        #        assert(len(scratch) == cacheSize)
        #        rho_cache = scratch
        #               
        #
        ##comm is currently ignored
        ##TODO: if evalTree is split, distribute among processors
        #for i in evalTree.get_evaluation_order():
        #    iStart,remainder,iCache = evalTree[i]
        #    if iStart is None:  init_state = rho
        #    else:               init_state = rho_cache[iStart] #[:,None]
        #
        #    final_state = self.propagate_state(init_state, remainder)
        #    if iCache is not None: rho_cache[iCache] = final_state # [:,0] #store this state in the cache
        #
        #    if self.evotype == "statevec":
        #        for j,E in enumerate(EVecs):
        #            ret[i,j] = _np.abs(_np.vdot(E.todense(Escratch),final_state))**2
        #    elif self.evotype == "densitymx":
        #        for j,E in enumerate(EVecs):
        #            ret[i,j] = _np.vdot(E.todense(Escratch),final_state)
        #            #OLD (slower): _np.dot(_np.conjugate(E.todense(Escratch)).T,final_state)
        #            # FUTURE: optionally pre-compute todense() results for speed if mem is available?
        #    else: # evotype == "stabilizer" case
        #        #TODO: compute using tree-like fanout, only fanning when necessary. -- at least when there are O(d=2^nQ) effects
        #        for j,E in enumerate(EVecs):
        #            ret[i,j] = rho.measurement_probability(E.outcomes)
        #
        ##print("DEBUG TIME: pr_cache(dim=%d, cachesize=%d) in %gs" % (self.dim, cacheSize,_time.time()-tStart)) #DEBUG
        #
        ##CHECK
        ##print("DB: ",ret); print("DB: ",ret2)
        #assert(_np.linalg.norm(ret-ret2) < 1e-6)
        #
        #return ret
    
    
    def _compute_dpr_cache(self, rholabel, elabels, evalTree, wrtSlice, comm, scratch=None):
        return replib.DM_compute_dpr_cache(self, rholabel, elabels, evalTree, wrtSlice, comm, scratch)

        #DEPRECATED REPS - just call replib version now - but below (commented) version
        # works for state-vec and stabilizer modes too - and we still need to essentially
        # duplicate this function to SV and SB modes...
        ##Compute finite difference derivatives, one parameter at a time.
        #tStart = _time.time() #DEBUG
        #param_indices = range(self.Np) if (wrtSlice is None) else _slct.indices(wrtSlice)
        #nDerivCols = len(param_indices) # *all*, not just locally computed ones
        #
        #dim = self.dim
        #cacheSize = evalTree.cache_size()
        #dpr_cache  = _np.zeros((len(evalTree), len(elabels), nDerivCols),'d')
        #
        ## Allocate cache space if needed
        #if self.evotype in ("statevec", "densitymx"):
        #    typ = complex if self.evotype == "statevec" else 'd'
        #
        #    if scratch is None:
        #        rho_cache  = _np.zeros((cacheSize, dim), typ)
        #    else:
        #        assert(scratch.shape == (cacheSize,dim))
        #        rho_cache  = scratch
        #else: # evotype == "stabilizer" case
        #    if scratch is None:
        #        rho_cache = [None]*cacheSize # so we can store (s,p) tuples in cache
        #    else:
        #        assert(len(scratch) == cacheSize)
        #        rho_cache = scratch
        #        
        #eps = 1e-7 #hardcoded?
        #pCache = self._compute_pr_cache(rholabel,elabels,evalTree,comm,rho_cache)
        #
        #all_slices, my_slice, owners, subComm = \
        #        _mpit.distribute_slice(slice(0,len(param_indices)), comm)
        #
        #my_param_indices = param_indices[my_slice]
        #st = my_slice.start #beginning of where my_param_indices results
        #                    # get placed into dpr_cache
        #
        ##Get a map from global parameter indices to the desired
        ## final index within dpr_cache
        #iParamToFinal = { i: st+ii for ii,i in enumerate(my_param_indices) }
        #
        #orig_vec = self.to_vector().copy()
        #for i in range(self.Np):
        #    #print("dprobs cache %d of %d" % (i,self.Np))
        #    if i in iParamToFinal:
        #        iFinal = iParamToFinal[i]
        #        vec = orig_vec.copy(); vec[i] += eps
        #        self.from_vector(vec)
        #        dpr_cache[:,:,iFinal] = ( self._compute_pr_cache(
        #            rholabel,elabels,evalTree,subComm,rho_cache) - pCache)/eps
        #self.from_vector(orig_vec)
        #
        ##Now each processor has filled the relavant parts of dpr_cache,
        ## so gather together:
        #_mpit.gather_slices(all_slices, owners, dpr_cache,[], axes=2, comm=comm)
        ## DEBUG LINE USED FOR MONITORION N-QUBIT GST TESTS
        ##print("DEBUG TIME: dpr_cache(Np=%d, dim=%d, cachesize=%d, treesize=%d, napplies=%d) in %gs" % 
        ##      (self.Np, self.dim, cacheSize, len(evalTree), evalTree.get_num_applies(), _time.time()-tStart)) #DEBUG
        #
        ##CHECK
        ##assert(_np.linalg.norm(dpr_cache-dpr_cache2) < 1e-6)
        #if _np.linalg.norm(dpr_cache-dpr_cache2) > 1e-6:
        #    print("DPR_CACHE MISMATCH: ", _np.linalg.norm(dpr_cache-dpr_cache2), " shape=",dpr_cache.shape)
        #
        #return dpr_cache

    def _compute_hpr_cache(self, rholabel, elabels, evalTree, wrtSlice1, wrtSlice2, comm):
        #Compute finite difference hessians, one parameter at a time.

        param_indices1 = range(self.Np) if (wrtSlice1 is None) else _slct.indices(wrtSlice1)
        param_indices2 = range(self.Np) if (wrtSlice2 is None) else _slct.indices(wrtSlice2)
        nDerivCols1 = len(param_indices1) # *all*, not just locally computed ones
        nDerivCols2 = len(param_indices2) # *all*, not just locally computed ones
        
        dim = self.dim
        cacheSize = evalTree.cache_size()
        hpr_cache  = _np.zeros((len(evalTree),len(elabels), nDerivCols1, nDerivCols2),'d')

        # Allocate scratch space for compute_dpr_cache
        if self.evotype in ("statevec", "densitymx"):
            typ = complex if self.evotype == "statevec" else 'd'
            dpr_scratch  = _np.zeros((cacheSize, dim), typ)
        else: # evotype == "stabilizer" case
            dpr_scratch = [None]*cacheSize # so we can store (s,p) tuples in cache
            
        eps = 1e-4 #hardcoded?
        dpCache = self._compute_dpr_cache(rholabel,elabels,evalTree,wrtSlice2,comm,
                                          dpr_scratch).copy()
           #need copy here b/c scratch space is used by sub-calls to
           # _compute_dpr_cache below in finite difference computation.
           
        all_slices, my_slice, owners, subComm = \
                _mpit.distribute_slice(slice(0,len(param_indices1)), comm)

        my_param_indices = param_indices1[my_slice]
        st = my_slice.start #beginning of where my_param_indices results
                            # get placed into dpr_cache
        
        #Get a map from global parameter indices to the desired
        # final index within dpr_cache
        iParamToFinal = { i: st+ii for ii,i in enumerate(my_param_indices) }

        orig_vec = self.to_vector().copy()
        for i in range(self.Np):
            if i in iParamToFinal:
                iFinal = iParamToFinal[i]
                vec = orig_vec.copy(); vec[i] += eps
                self.from_vector(vec)
                hpr_cache[:,:,iFinal,:] = ( self._compute_dpr_cache(
                    rholabel,elabels,evalTree,wrtSlice2,subComm,dpr_scratch) - dpCache)/eps
        self.from_vector(orig_vec)

        #Now each processor has filled the relavant parts of dpr_cache,
        # so gather together:
        _mpit.gather_slices(all_slices, owners, hpr_cache,[], axes=2, comm=comm)

        return hpr_cache

    def default_distribute_method(self):
        """ 
        Return the preferred MPI distribution mode for this calculator.
        """
        return "gatestrings"

    
    def estimate_cache_size(self, nGateStrings):
        """
        Return an estimate of the ideal/desired cache size given a number of 
        gate strings.

        Returns
        -------
        int
        """
        return int( 0.7 * nGateStrings )

    
    def construct_evaltree(self):
        """
        Constructs an EvalTree object appropriate for this calculator.
        """
        return _MapEvalTree()


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
                mem += cache_size * dim # pr cache intermediate
                mem += num_final_strs # pr cache final (* #elabels!)

            elif fnName == "bulk_fill_dprobs":
                mem += cache_size * dim # dpr cache scratch
                mem += cache_size * dim # pr cache intermediate
                mem += num_final_strs * wrtLen1 # dpr cache final (* #elabels!)

            elif fnName == "bulk_fill_hprobs":
                mem += cache_size * dim # dpr cache intermediate (scratch)
                mem += cache_size * wrtLen2 * dim * 2 # dpr cache (x2)
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

            def calc_and_fill(rholabel, elabels, fIndsList, gIndsList, pslc1, pslc2, sumInto):
                """ Compute and fill result quantities for given arguments """
                #Fill cache info
                prCache = self._compute_pr_cache(rholabel, elabels, evalSubTree, mySubComm)
                
                #use cached data to final values
                ps = evalSubTree.final_view( prCache, axis=0) # ( nGateStrings, len(elabels))
                for i,(fInds,gInds) in enumerate(zip(fIndsList,gIndsList)):
                    _fas(mxToFill, [fInds], ps[gInds,i], add=sumInto)

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

            #Free memory from previous subtree iteration before computing caches
            paramSlice = slice(None)
            fillComm = mySubComm #comm used by calc_and_fill

            def calc_and_fill(rholabel, elabels, fIndsList, gIndsList, pslc1, pslc2, sumInto):
                """ Compute and fill result quantities for given arguments """
                tm = _time.time()
                
                if prMxToFill is not None:
                    prCache = self._compute_pr_cache(rholabel, elabels, evalSubTree, fillComm)
                    ps = evalSubTree.final_view( prCache, axis=0) # ( nGateStrings, len(elabels))
                    for i,(fInds,gInds) in enumerate(zip(fIndsList,gIndsList)):
                        _fas(prMxToFill, [fInds], ps[gInds,i], add=sumInto)

                #Fill cache info
                dprCache = self._compute_dpr_cache(rholabel, elabels, evalSubTree, paramSlice, fillComm)
                dps = evalSubTree.final_view( dprCache, axis=0) # ( nGateStrings, len(elabels), nDerivCols)
                for i,(fInds,gInds) in enumerate(zip(fIndsList,gIndsList)):
                    _fas(mxToFill, [fInds, pslc1], dps[gInds,i], add=sumInto)
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

            #Free memory from previous subtree iteration before computing caches
            paramSlice1 = slice(None)
            paramSlice2 = slice(None)

            def calc_and_fill(rholabel, elabels, fIndsList, gIndsList, pslc1, pslc2, sumInto):
                """ Compute and fill result quantities for given arguments """
                
                if prMxToFill is not None:
                    prCache = self._compute_pr_cache(rholabel, elabels, evalSubTree, fillComm)
                    ps = evalSubTree.final_view( prCache, axis=0) # ( nGateStrings, len(elabels))
                    for i,(fInds,gInds) in enumerate(zip(fIndsList,gIndsList)):
                        _fas(prMxToFill, [fInds], ps[gInds,i], add=sumInto)

                if deriv1MxToFill is not None:
                    dprCache = self._compute_dpr_cache(rholabel, elabels, evalSubTree, paramSlice1, fillComm)
                    dps1 = evalSubTree.final_view( dprCache, axis=0) # ( nGateStrings, len(elabels), nDerivCols)
                    for i,(fInds,gInds) in enumerate(zip(fIndsList,gIndsList)):
                        _fas(deriv1MxToFill, [fInds,pslc1], dps1[gInds,i], add=sumInto)

                if deriv2MxToFill is not None:
                    if deriv1MxToFill is not None and paramSlice1 == paramSlice2:
                        dps2 = dps1
                    else:
                        dprCache = self._compute_dpr_cache(rholabel, elabels, evalSubTree, paramSlice2, fillComm)
                        dps2 = evalSubTree.final_view( dprCache, axis=0) # ( nGateStrings, len(elabels), nDerivCols)
                    for i,(fInds,gInds) in enumerate(zip(fIndsList,gIndsList)):
                        _fas(deriv2MxToFill, [fInds,pslc2], dps2[gInds,i], add=sumInto)

                #Fill cache info
                hprCache = self._compute_hpr_cache(rholabel, elabels, evalSubTree, paramSlice1, paramSlice2, fillComm)
                hps = evalSubTree.final_view( hprCache, axis=0) # ( nGateStrings, len(elabels), nDerivCols1, nDerivCols2)
                
                for i,(fInds,gInds) in enumerate(zip(fIndsList,gIndsList)):
                    _fas(mxToFill, [fInds,pslc1,pslc2], hps[gInds,i], add=sumInto)


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
