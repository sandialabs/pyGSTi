# encoding: utf-8
# cython: profile=False
# cython: linetrace=False
# filename: fastactonlib.pyx

import sys
import numpy as np
from libcpp cimport bool
from libc.stdlib cimport malloc, free
cimport numpy as np
cimport cython


cdef extern from "fastreps.h" namespace "CReps":    
    cdef cppclass DMStateCRep:
        DMStateCRep() except +
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


def compute_pr_cache(ret, rhorep, ereps, gatereps, ievalTree, comm, scratch=None):
    #tStart = _time.time()
    #dim = self.dim
    #cacheSize = evalTree.cache_size()
    #rhoVec,EVecs = self._rhoEs_from_labels(rholabel, elabels)


    #Get rho & rhoCache
    if scratch is None:
        rho_cache = {}
    else:
        rho_cache = scratch
                   
    #comm is currently ignored
    #TODO: if evalTree is split, distribute among processors
    for i,iStart,remainder,iCache in ievalTree:
        if iStart is None:  init_state = rhorep
        else:               init_state = rho_cache[iStart] #[:,None]

        
        #Propagate state rep
        #print("init ",i,str(init_state))
        #print("applying")
        #for r in remainder:
        #    print("mx:")
        #    print(gatereps[r])
        final_state = propagate_staterep(init_state, [gatereps[r] for r in remainder] )
        #print("final ",i,str(final_state))
        if iCache is not None: rho_cache[iCache] = final_state # [:,0] #store this state in the cache

        for j,erep in enumerate(ereps):
            # p = erep.probability(final_state) #outcome probability
            p = erep.amplitude(final_state) #outcome probability
            #print("processing ",i,j,p)
            ret[i,j] = p

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

