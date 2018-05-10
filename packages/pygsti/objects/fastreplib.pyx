# encoding: utf-8
# cython: profile=False
# cython: linetrace=False
# filename: fastactonlib.pyx

import sys
import numpy as np
from libc.stdlib cimport malloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
#from libcpp.unordered_map cimport unordered_map
from cython.operator cimport dereference as deref, preincrement as inc
cimport numpy as np
cimport cython

from ..tools import mpitools as _mpit
from ..tools import slicetools as _slct



cdef extern from "fastreps.h" namespace "CReps":    
    cdef cppclass DMStateCRep:
        DMStateCRep() except +
        DMStateCRep(int) except +
        DMStateCRep(double*,int,bool) except +
        int _dim
        double* _dataptr

    cdef cppclass DMEffectCRep:
        DMEffectCRep() except +
        DMEffectCRep(double*,int) except +
        double amplitude(DMStateCRep* state)
        int _dim
        double* _dataptr

    cdef cppclass DMGateCRep:
        DMGateCRep() except +
        DMGateCRep(double*,int) except +
        DMStateCRep* acton(DMStateCRep*, DMStateCRep*);
        int _dim
        double* _dataptr


ctypedef DMGateCRep* DMGateCRep_ptr
ctypedef DMStateCRep* DMStateCRep_ptr
ctypedef DMEffectCRep* DMEffectCRep_ptr
        
cdef class StateRep:
    pass

cdef class DMStateRep(StateRep):
    cdef DMStateCRep* c_state
    #cdef np.ndarray dataview 
    cdef double [:] data_view
    #cdef int dim

    def __cinit__(self, np.ndarray[double, ndim=1, mode='c'] data):
        #print("DMStateRep init: refcnt(data)=",sys.getrefcount(data),data.shape[0] )
        cdef np.ndarray[double, ndim=1, mode='c'] np_cbuf = np.ascontiguousarray(data, dtype='d')
        cdef double [:] view = data
        self.data_view = view # holds reference to data so it doesn't get garbage collected - or could copy=true
        ##self.dataview = data # also holds reference to data so it doesn't get garbage collected - or could copy=true
        #print("DMStateRep init2: refcnt(data)=",sys.getrefcount(data) )
        self.c_state = new DMStateCRep(<double*>np_cbuf.data,<int>np_cbuf.shape[0],<bool>0)
        #self.c_state = new DMStateCRep(<double*>data.data,<int>data.shape[0],<bool>0)
        
        #self.data_view = data
        #self.dim = data.shape[0]
        #print("DMStateRep DONE init")

    def __dealloc__(self):
        del self.c_state

    def __str__(self):
        return str([self.c_state._dataptr[i] for i in range(self.c_state._dim)])


cdef class EffectRep:
    #cdef double camplitude(self, StateRep state): pass
    def amplitude(self, state):  pass

cdef class DMEffectRep(EffectRep):
    cdef DMEffectCRep c_effect
    #cdef double [:] data_view
    #cdef int dim

    def __cinit__(self, np.ndarray[double, ndim=1] data):
        #print("DMEffectRep init: refcnt(data)=",sys.getrefcount(data) )
        cdef np.ndarray[double, ndim=1, mode='c'] np_cbuf = np.ascontiguousarray(data, dtype='d')
        self.c_effect = DMEffectCRep(<double*>np_cbuf.data,<int>np_cbuf.shape[0])

        #self.data_view = data
        #self.dim = data.shape[0]

    #cdef double camplitude(self, StateRep state):
    #    cdef double ret = 0.0
    #    for i in range(self.dim):
    #        ret += self.data_view[i] * state.data_view[i]
    #    return ret
            
    def amplitude(self, StateRep state not None):
        cdef DMStateRep st = <DMStateRep?>state
        return self.c_effect.amplitude(st.c_state)
        

cdef class GateRep:
    #cdef cacton(self, StateRep state, StateRep out_state): pass
    def acton(self, state):  pass
        
cdef class DMGateRep(GateRep):
    cdef DMGateCRep c_gate
    #cdef double [:,:] data_view
    #cdef int dim

    def __cinit__(self, np.ndarray[double, ndim=2] data):
        #print("DMGateRep init")
        cdef np.ndarray[double, ndim=2, mode='c'] np_cbuf = np.ascontiguousarray(data, dtype='d')
        self.c_gate = DMGateCRep(<double*>np_cbuf.data,<int>np_cbuf.shape[0])
        #self.data_view = data
        #self.dim = data.shape[0]

#    cdef cacton(self, StateRep state, StateRep out_state):
#        print("cacton begin")
#        cdef int i,j
#        cdef DMStateRep instate = <DMStateRep?>state
#        cdef DMStateRep outstate = <DMStateRep?>out_state
#        
#        for i in range(self.dim):
#            outstate.data_view[i] = 0
#            for j in range(self.dim):
#                outstate.data_view[i] += self.data_view[i,j]*instate.data_view[j]
#        print("cacton end")

#    def acton_effect_test(self, EffectRep state not None):
#        cdef np.ndarray[double, ndim=1, mode="c"] nparr = np.empty(4, dtype='d')
#        cdef np.ndarray[double, ndim=1, mode="c"] nparr2 = np.empty(4, dtype='d')
#        cdef DMEffectRep st = <DMEffectRep?>state
#        print("PRE PTR = ",<long>st.c_effect._dataptr)
#        print("ACTON PTRS = ",<long>st.c_effect._dataptr, <long>nparr.data, <long>nparr2.data)


    def acton(self, StateRep state not None):
        #print("acton: refcnt(state)=",sys.getrefcount(state) )

        #DEBUG
        #cdef np.ndarray[double, ndim=1, mode="c"] nparr = np.empty(4, dtype='d')
        #cdef np.ndarray[double, ndim=1, mode="c"] nparr2 = np.empty(4, dtype='d')

        cdef DMStateRep st = <DMStateRep?>state
        #print("DEBUG PTRS = ",<long>st.c_state._dataptr, <long>nparr.data, <long>nparr2.data)
        
        cdef DMStateRep out_state = DMStateRep(np.empty(self.c_gate._dim, dtype='d'))
        #print("ACTON PTRS = ",<long>st.c_state._dataptr, <long>out_state.c_state._dataptr)
        assert(st.c_state._dataptr != out_state.c_state._dataptr) # DEBUG
        #print("ACTON IN = ",str(st))
        #print("ACTON WITH = \n",str(self))

        self.c_gate.acton(st.c_state, out_state.c_state)
        #Cython version of above c-func so we can debug-print during it...
        #for i in range(self.c_gate._dim):
        #  out_state.c_state._dataptr[i] = 0.0
        #  k = i*self.c_gate._dim
        #  print("ROW ",i," k=",k)
        #  for j in range(self.c_gate._dim):
        #    out_state.c_state._dataptr[i] += self.c_gate._dataptr[k+j] * st.c_state._dataptr[j]
        #    print("  COL ",j,": ",self.c_gate._dataptr[k+j]," * ",st.c_state._dataptr[j], "-->", out_state.c_state._dataptr[i])
        
        #print("ACTON OUT = ",str(out_state))
        return out_state

    def __str__(self):
        s = ""
        cdef int i,j,k 
        for i in range(self.c_gate._dim):
            k = i*self.c_gate._dim
            for j in range(self.c_gate._dim):
                s += str(self.c_gate._dataptr[k+j]) + " "
            s += "\n"
        return s



    
def propagate_staterep(staterep, gatereps):
    ret = staterep
    for gaterep in gatereps:
        ret = gaterep.acton(ret)
    return ret




cdef vector[vector[int]] convert_evaltree(evalTree, gate_lookup):
    # c_evalTree :
    # an array of int-arrays; each int-array is [i,iStart,iCache,<remainder gate indices>]
    cdef vector[int] intarray
    cdef vector[vector[int]] c_evalTree = vector[vector[int]](len(evalTree))
    for kk,ii in enumerate(evalTree.get_evaluation_order()):
        iStart,remainder,iCache = evalTree[ii]
        if iStart is None: iStart = -1 # so always an int
        if iCache is None: iCache = -1 # so always an int
        intarray = vector[int](3 + len(remainder))
        intarray[0] = ii
        intarray[1] = iStart
        intarray[2] = iCache
        for jj,gl in enumerate(remainder,start=3):
            intarray[jj] = gate_lookup[gl]
        c_evalTree[kk] = intarray
        
    return c_evalTree

cdef vector[DMStateCRep*] create_rhocache(int cacheSize, int state_dim):
    cdef int i
    cdef vector[DMStateCRep*] rho_cache = vector[DMStateCRep_ptr](cacheSize)
    for i in range(cacheSize): # fill cache with empty but alloc'd states
        rho_cache[i] = new DMStateCRep(state_dim)
    return rho_cache

cdef void free_rhocache(vector[DMStateCRep*] rho_cache):
    cdef unsigned int i
    for i in range(rho_cache.size()): # fill cache with empty but alloc'd states
        del rho_cache[i]


cdef vector[DMGateCRep*] convert_gatereps(gatereps):
    # c_gatereps : an array of DMGateCReps
    cdef vector[DMGateCRep*] c_gatereps = vector[DMGateCRep_ptr](len(gatereps))
    for ii,grep in gatereps.items(): # (ii = python variable)
        c_gatereps[ii] = &(<DMGateRep?>grep).c_gate
    return c_gatereps

cdef DMStateCRep* convert_rhorep(rhorep):
    # extract c-reps from rhorep and ereps => c_rho and c_ereps
    return (<DMStateRep?>rhorep).c_state

cdef vector[DMEffectCRep*] convert_ereps(ereps):
    cdef vector[DMEffectCRep*] c_ereps = vector[DMEffectCRep_ptr](len(ereps))
    for i in range(len(ereps)):
        c_ereps[i] = &(<DMEffectRep>ereps[i]).c_effect
    return c_ereps


def DM_compute_pr_cache(calc, rholabel, elabels, evalTree, comm, scratch=None): # TODO remove scratch


    rhoVec,EVecs = calc._rhoEs_from_labels(rholabel, elabels)
    pCache = np.empty((len(evalTree),len(EVecs)),'d')

    #Get (extension-type) representation objects
    rhorep = rhoVec.torep('prep')
    ereps = [ E.torep('effect') for E in EVecs]  # could cache these? then have torep keep a non-dense rep that can be quickly kron'd for a tensorprod spamvec
    gate_lookup = { lbl:i for i,lbl in enumerate(calc.gates.keys()) } # gate labels -> ints for faster lookup
    gatereps = { i:calc.gates[lbl].torep() for lbl,i in gate_lookup.items() }
    
    # convert to C-mode:  evaltree, gate_lookup, gatereps
    cdef c_evalTree = convert_evaltree(evalTree, gate_lookup)
    cdef DMStateCRep *c_rho = convert_rhorep(rhorep)
    cdef vector[DMGateCRep*] c_gatereps = convert_gatereps(gatereps)
    cdef vector[DMEffectCRep*] c_ereps = convert_ereps(ereps)

    # create rho_cache = vector of DMStateCReps
    cdef vector[DMStateCRep*] rho_cache = create_rhocache(evalTree.cache_size(), c_rho._dim)

    #OLD cdef double[:,:] ret_view = ret
    dm_compute_pr_cache(pCache, c_evalTree, c_gatereps, c_rho, c_ereps, &rho_cache, comm)

    free_rhocache(rho_cache)  #delete cache entries
    return pCache
        


cdef dm_compute_pr_cache(double[:,:] ret,
                         vector[vector[int]] c_evalTree,
                         vector[DMGateCRep*] c_gatereps,
                         DMStateCRep *c_rho, vector[DMEffectCRep*] c_ereps,
                         vector[DMStateCRep*]* prho_cache, comm): # any way to transmit comm?
    #Note: we need to take in rho_cache as a pointer b/c we may alter the values its
    # elements point to (instead of copying the states) - we just guarantee that in the end
    # all of the cache entries are filled with allocated (by 'new') states that the caller
    # can deallocate at will.
    cdef int k,l, i, istart, icache
    cdef double p
    cdef int dim = c_rho._dim
    cdef DMStateCRep *init_state
    cdef DMStateCRep *prop1
    cdef DMStateCRep *tprop
    cdef DMStateCRep *final_state
    cdef DMStateCRep *prop2 = new DMStateCRep(dim)
    cdef DMStateCRep *shelved = new DMStateCRep(dim)

    #print "BEGIN"

    #Invariants required for proper memory management:
    # - upon loop entry, prop2 is allocated and prop1 is not (it doesn't "own" any memory)
    # - all rho_cache entries have been allocated via "new"
    for k in range(c_evalTree.size()):
        intarray = c_evalTree[k]
        i = intarray[0]
        istart = intarray[1]
        icache = intarray[2]

        if istart == -1:  init_state = c_rho
        else:             init_state = deref(prho_cache)[istart]
        
        #DEBUG
        #print "LOOP i=",i," istart=",istart," icache=",icache," remcnt=",(intarray.size()-3)
        #print [ init_state._dataptr[t] for t in range(4) ]
    
        #Propagate state rep
        # prop2 should already be alloc'd; need to "allocate" prop1 - either take from cache or from "shelf"
        #OLD: prop1 = new DMStateCRep(init_state._dataptr, dim, <bool>1) 
        prop1 = shelved if icache == -1 else deref(prho_cache)[icache]
        for j in range(dim): prop1._dataptr[j] = init_state._dataptr[j] # copy init_state -> prop1   FUTURE: a method of DMStateCRep?
        #print " prop1:";  print [ prop1._dataptr[t] for t in range(4) ]
        for l in range(3,intarray.size()): #during loop, both prop1 & prop2 are alloc'd        
            c_gatereps[intarray[l]].acton(prop1,prop2)
            #print " post-act prop2:"; print [ prop2._dataptr[t] for t in range(4) ]
            tprop = prop1; prop1 = prop2; prop2 = tprop # swap prop1 <-> prop2
        final_state = prop1 # output = prop1 (after swap from loop above)
        # Note: prop2 is the other alloc'd state and this maintains invariant
        #print " final:"; print [ final_state._dataptr[t] for t in range(4) ]
        
        for j in range(c_ereps.size()):
            # p = erep.probability(final_state) #outcome probability
            p = c_ereps[j].amplitude(final_state) #outcome probability
            #print("processing ",i,j,p)
            ret[i,j] = p

        if icache != -1:
            deref(prho_cache)[icache] = final_state # store this state in the cache
        else: # our 2nd state was pulled from the shelf before; return it
            shelved = final_state
            final_state = NULL


    #delete our temp states
    del prop2
    del shelved


    
def DM_compute_dpr_cache(calc, rholabel, elabels, evalTree, wrtSlice, comm, scratch=None):

    cdef double eps = 1e-7 #hardcoded?

    #Compute finite difference derivatives, one parameter at a time.
    param_indices = range(calc.Np) if (wrtSlice is None) else _slct.indices(wrtSlice)
    nDerivCols = len(param_indices) # *all*, not just locally computed ones

    rhoVec,EVecs = calc._rhoEs_from_labels(rholabel, elabels)
    pCache = np.empty((len(evalTree),len(elabels)),'d')
    dpr_cache  = np.zeros((len(evalTree), len(elabels), nDerivCols),'d')

    #Get (extension-type) representation objects
    rhorep = calc.preps[rholabel].torep('prep')
    ereps = [ calc.effects[el].torep('effect') for el in elabels]
    gate_lookup = { lbl:i for i,lbl in enumerate(calc.gates.keys()) } # gate labels -> ints for faster lookup
    gatereps = { i:calc.gates[lbl].torep() for lbl,i in gate_lookup.items() }
    
    # convert to C-mode:  evaltree, gate_lookup, gatereps
    cdef c_evalTree = convert_evaltree(evalTree, gate_lookup)
    cdef DMStateCRep *c_rho = convert_rhorep(rhorep)
    cdef vector[DMGateCRep*] c_gatereps = convert_gatereps(gatereps)
    cdef vector[DMEffectCRep*] c_ereps = convert_ereps(ereps)

    # create rho_cache = vector of DMStateCReps
    cdef vector[DMStateCRep*] rho_cache = create_rhocache(evalTree.cache_size(), c_rho._dim)

    dm_compute_pr_cache(pCache, c_evalTree, c_gatereps, c_rho, c_ereps, &rho_cache, comm)
    pCache_delta = pCache.copy() # for taking finite differences

    all_slices, my_slice, owners, subComm = \
            _mpit.distribute_slice(slice(0,len(param_indices)), comm)

    my_param_indices = param_indices[my_slice]
    st = my_slice.start #beginning of where my_param_indices results
                        # get placed into dpr_cache
    
    #Get a map from global parameter indices to the desired
    # final index within dpr_cache
    iParamToFinal = { i: st+ii for ii,i in enumerate(my_param_indices) }

    orig_vec = calc.to_vector().copy()
    for i in range(calc.Np):
        #print("dprobs cache %d of %d" % (i,self.Np))
        if i in iParamToFinal:
            iFinal = iParamToFinal[i]
            vec = orig_vec.copy(); vec[i] += eps
            calc.from_vector(vec)

            #rebuild reps (not evaltree or gate_lookup)
            rhorep = calc.preps[rholabel].torep('prep')
            ereps = [ calc.effects[el].torep('effect') for el in elabels]
            gatereps = { i:calc.gates[lbl].torep() for lbl,i in gate_lookup.items() }
            c_rho = convert_rhorep(rhorep)
            c_ereps = convert_ereps(ereps)
            c_gatereps = convert_gatereps(gatereps)

            dm_compute_pr_cache(pCache_delta, c_evalTree, c_gatereps, c_rho, c_ereps, &rho_cache, comm)
            dpr_cache[:,:,iFinal] = (pCache_delta - pCache)/eps

    calc.from_vector(orig_vec)
    free_rhocache(rho_cache)
    
    #Now each processor has filled the relavant parts of dpr_cache,
    # so gather together:
    _mpit.gather_slices(all_slices, owners, dpr_cache,[], axes=2, comm=comm)
    
    # DEBUG LINE USED FOR MONITORION N-QUBIT GST TESTS
    #print("DEBUG TIME: dpr_cache(Np=%d, dim=%d, cachesize=%d, treesize=%d, napplies=%d) in %gs" % 
    #      (self.Np, self.dim, cacheSize, len(evalTree), evalTree.get_num_applies(), _time.time()-tStart)) #DEBUG

    return dpr_cache


    

#OLD LOOP - maybe good for python version?
#    #comm is currently ignored
#    #TODO: if evalTree is split, distribute among processors
#    for i,iStart,remainder,iCache in ievalTree:
#        if iStart is None:  init_state = rhorep
#        else:               init_state = rho_cache[iStart] #[:,None]
#
#        
#        #Propagate state rep
#        #print("init ",i,str(init_state))
#        #print("applying")
#        #for r in remainder:
#        #    print("mx:")
#        #    print(gatereps[r])
#        final_state = propagate_staterep(init_state, [gatereps[r] for r in remainder] )
#        #print("final ",i,str(final_state))
#        if iCache is not None: rho_cache[iCache] = final_state # [:,0] #store this state in the cache
#
#        for j,erep in enumerate(ereps):
#            # p = erep.probability(final_state) #outcome probability
#            p = erep.amplitude(final_state) #outcome probability
#            #print("processing ",i,j,p)
#            ret[i,j] = p



        #if self.evotype == "statevec":
        #    for j,E in enumerate(EVecs):
        #        ret[i,j] = _np.abs(_np.vdot(E.todense(Escratch),final_state))**2
        #elif self.evotype == "densitymx":
        #    for j,E in enumerate(EVecs):
        #        ret[i,j] = _np.vdot(E.todense(Escratch),final_state)
        #        #OLD (slower): _np.dot(_np.conjugate(E.todense(Escratch)).T,final_state)
        #        # FUTURE: optionally pre-compute todense() results for speed if mem is available?
        #else: # evotype == "stabilizer" case
        #    #TODO: compute using tree-like fanout, only fanning when necessary. -- at least when there are O(d=2^nQ) effects
        #    for j,E in enumerate(EVecs):
        #        ret[i,j] = rho.measurement_probability(E.outcomes)

    #print("DEBUG TIME: pr_cache(dim=%d, cachesize=%d) in %gs" % (self.dim, cacheSize,_time.time()-tStart)) #DEBUG
    #return ret


#cdef struct Gate_crep:
#    int x
#    char * y
#    #put everything potentially necessary here
#ctypedef Gate_crep Gate_crep_t


#def fast_compute_pr_cache(rholabel, elabels, evalTree, comm, scratch=None)::
#    #needs to construct gate creps, etc...
#    
#    #calls propagate_state:
#    
#cdef propagate_state(Gate_crep_t* gate_creps, int* gids,
#                     State_crep* state_crep):
#    for gateid in gids:
#    	actonlib[gateid](gate_creps[gateid],state_crep) # act in-place / don't require copy?
	


## You can also typedef pointers too
#
#ctypedef int * int_ptr
#
#
#ctypedef int (* no_arg_c_func)(BaseThing)
#cdef no_arg_c_func funcs[1000]
#
#ctypedef void (*cfptr)(int)
#
## then we use the function pointer:
#cdef cfptr myfunctionptr = &myfunc



def dot(np.ndarray[double, ndim=1] f, np.ndarray[double, ndim=1] g):
    cdef long N = f.shape[0]
    cdef float ret = 0.0
    cdef int i
    for i in range(N):
        ret += f[i]*g[i]
    return ret

