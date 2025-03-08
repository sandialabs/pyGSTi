# encoding: utf-8
# cython: profile=False
# cython: linetrace=False

#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


from libc cimport time
from libcpp cimport bool
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref
from pygsti.evotypes.basereps_cython cimport PolynomialRep, PolynomialCRep, PolynomialVarsIndex
from pygsti.evotypes.statevec.statereps cimport StateRep, StateCRep
from pygsti.evotypes.statevec.opreps cimport OpRep, OpCRep, OpCRep_DenseUnitary
from pygsti.evotypes.statevec.effectreps cimport EffectRep, EffectCRep
from pygsti.evotypes.statevec.termreps cimport TermRep, TermCRep, TermDirectRep, TermDirectCRep

from libc.stdlib cimport malloc, free
from libc.math cimport log10, sqrt
from libcpp.unordered_map cimport unordered_map
#from libcpp.pair cimport pair
#from libcpp.algorithm cimport sort as stdsort
from cython.operator cimport dereference as deref  # , preincrement as inc
cimport numpy as np
cimport cython

import time as pytime
import numpy as np

#import itertools as _itertools
from pygsti.baseobjs.opcalc import fastopcalc as _fastopcalc
#from scipy.sparse.linalg import LinearOperator

cdef double SMALL = 1e-5
cdef double LOGSMALL = -5
# a number which is used in place of zero within the
# product of term magnitudes to keep a running path
# magnitude from being zero (and losing memory of terms).


#Use 64-bit integers
ctypedef long long INT
ctypedef unsigned long long UINT

ctypedef double complex DCOMPLEX
ctypedef StateCRep* StateCRep_ptr
ctypedef OpCRep* OpCRep_ptr
ctypedef EffectCRep* EffectCRep_ptr
ctypedef TermCRep* TermCRep_ptr
ctypedef TermDirectCRep* TermDirectCRep_ptr
ctypedef PolynomialCRep* PolynomialCRep_ptr
ctypedef vector[TermCRep_ptr]* vector_TermCRep_ptr_ptr
ctypedef vector[TermDirectCRep_ptr]* vector_TermDirectCRep_ptr_ptr
ctypedef vector[INT]* vector_INT_ptr


#Create a function pointer type for term-based calc inner loop
ctypedef void (*innerloopfn_ptr)(vector[vector_TermCRep_ptr_ptr],
                                 vector[INT]*, vector[PolynomialCRep*]*, INT)
ctypedef INT (*innerloopfn_direct_ptr)(vector[vector_TermDirectCRep_ptr_ptr],
                                       vector[INT]*, vector[DCOMPLEX]*, INT, vector[double]*, double)
ctypedef void (*addpathfn_ptr)(vector[PolynomialCRep*]*, vector[INT]&, INT, vector[vector_TermCRep_ptr_ptr]&,
                               StateCRep**, StateCRep**, vector[INT]*,
                               vector[StateCRep*]*, vector[StateCRep*]*, vector[PolynomialCRep]*)


cdef class RepCacheEl:
    cdef vector[TermCRep_ptr] reps
    cdef vector[INT] foat_indices
    cdef vector[INT] e_indices
    cdef public object pyterm_references

    def __cinit__(self):
        self.reps = vector[TermCRep_ptr](0)
        self.foat_indices = vector[INT](0)
        self.e_indices = vector[INT](0)
        self.pyterm_references = []


cdef class CircuitSetupCacheEl:
    cdef vector[INT] cgatestring
    cdef vector[TermCRep_ptr] rho_term_reps
    cdef unordered_map[INT, vector[TermCRep_ptr] ] op_term_reps
    cdef vector[TermCRep_ptr] E_term_reps
    cdef vector[INT] rho_foat_indices
    cdef unordered_map[INT, vector[INT] ] op_foat_indices
    cdef vector[INT] E_foat_indices
    cdef vector[INT] e_indices
    cdef object pyterm_references

    def __cinit__(self):
        self.cgatestring = vector[INT](0)
        self.rho_term_reps = vector[TermCRep_ptr](0)
        self.op_term_reps = unordered_map[INT, vector[TermCRep_ptr] ]()
        self.E_term_reps = vector[TermCRep_ptr](0)
        self.rho_foat_indices = vector[INT](0)
        self.op_foat_indices = unordered_map[INT, vector[INT] ]()
        self.E_foat_indices = vector[INT](0)
        self.e_indices = vector[INT](0)
        self.pyterm_references = []


# ------------------------------------- TERM CALC FUNCTIONS ------------------------------

# Helper functions
cdef PolynomialRep_from_allocd_PolynomialCRep(PolynomialCRep* crep):
    cdef PolynomialRep ret = PolynomialRep.__new__(PolynomialRep) # doesn't call __init__
    ret.c_polynomial = crep
    return ret


cdef vector[vector[TermCRep_ptr]] extract_cterms(python_termrep_lists, INT max_order):
    cdef vector[vector[TermCRep_ptr]] ret = vector[vector[TermCRep_ptr]](max_order+1)
    cdef vector[TermCRep*] vec_of_terms
    for order,termreps in enumerate(python_termrep_lists): # maxorder+1 lists
        vec_of_terms = vector[TermCRep_ptr](len(termreps))
        for i,termrep in enumerate(termreps):
            vec_of_terms[i] = (<TermRep?>termrep).c_term
        ret[order] = vec_of_terms
    return ret


def prs_as_polynomials(fwdsim, rholabel, elabels, circuit, polynomial_vindices_per_int,
                       comm=None, mem_limit=None, fastmode=True):

    # Create gatelable -> int mapping to be used throughout
    distinct_gateLabels = sorted(set(circuit))
    glmap = { gl: i for i,gl in enumerate(distinct_gateLabels) }

    # Convert circuit to a vector of ints
    cdef INT i
    cdef vector[INT] cgatestring
    for gl in circuit:
        cgatestring.push_back(<INT>glmap[gl])

    cdef INT mpv = fwdsim.model.num_params # max_polynomial_vars
    #cdef INT mpo = fwdsim.max_order*2 #max_polynomial_order
    cdef INT vpi = polynomial_vindices_per_int  #pass this in directly so fwdsim can compute once & use multiple times
    cdef INT order;
    cdef INT numEs = len(elabels)

    # Construct dict of gate term reps, then *convert* to c-reps, as this
    #  keeps alive the non-c-reps which keep the c-reps from being deallocated...
    op_term_reps = { glmap[glbl]: [ [t.torep() for t in fwdsim.model._circuit_layer_operator(glbl, 'op').taylor_order_terms(order, mpv)]
                                      for order in range(fwdsim.max_order+1) ]
                       for glbl in distinct_gateLabels }

    #Similar with rho_terms and E_terms
    rho_term_reps = [ [t.torep() for t in fwdsim.model._circuit_layer_operator(rholabel, 'prep').taylor_order_terms(order, mpv)]
                      for order in range(fwdsim.max_order+1) ]

    E_term_reps = []
    e_indices = []
    for order in range(fwdsim.max_order+1):
        cur_term_reps = [] # the term reps for *all* the effect vectors
        cur_indices = [] # the Evec-index corresponding to each term rep
        for i,elbl in enumerate(elabels):
            term_reps = [t.torep() for t in fwdsim.model._circuit_layer_operator(elbl, 'povm').taylor_order_terms(order, mpv) ]
            cur_term_reps.extend( term_reps )
            cur_indices.extend( [i]*len(term_reps) )
        E_term_reps.append( cur_term_reps )
        e_indices.append( cur_indices )

    #convert to c-reps
    cdef INT gi
    cdef vector[vector[TermCRep_ptr]] rho_term_creps = extract_cterms(rho_term_reps,fwdsim.max_order)
    cdef vector[vector[TermCRep_ptr]] E_term_creps = extract_cterms(E_term_reps,fwdsim.max_order)
    cdef unordered_map[INT, vector[vector[TermCRep_ptr]]] gate_term_creps
    for gi,termrep_lists in op_term_reps.items():
        gate_term_creps[gi] = extract_cterms(termrep_lists,fwdsim.max_order)

    E_cindices = vector[vector[INT]](<INT>len(e_indices))
    for ii,inds in enumerate(e_indices):
        E_cindices[ii] = vector[INT](<INT>len(inds))
        for jj,indx in enumerate(inds):
            E_cindices[ii][jj] = <INT>indx

    #Note: term calculator "dim" is the full density matrix dim
    stateDim = int(round(np.sqrt(fwdsim.model.dim)))

    #Call C-only function (which operates with C-representations only)
    cdef vector[PolynomialCRep*] polynomials = c_prs_as_polynomials(
        cgatestring, rho_term_creps, gate_term_creps, E_term_creps,
        E_cindices, numEs, fwdsim.max_order, mpv, vpi, stateDim, <bool>fastmode)

    return [ PolynomialRep_from_allocd_PolynomialCRep(polynomials[i]) for i in range(<INT>polynomials.size()) ]


cdef vector[PolynomialCRep*] c_prs_as_polynomials(
    vector[INT]& circuit, vector[vector[TermCRep_ptr]] rho_term_reps,
    unordered_map[INT, vector[vector[TermCRep_ptr]]] op_term_reps,
    vector[vector[TermCRep_ptr]] E_term_reps, vector[vector[INT]] E_term_indices,
    INT numEs, INT max_order, INT max_polynomial_vars, INT vindices_per_int, INT dim, bool fastmode):

    #NOTE: circuit and gate_terms use *integers* as operation labels, not Label objects, to speed
    # lookups and avoid weird string conversion stuff with Cython

    cdef INT N = len(circuit)
    cdef INT* p = <INT*>malloc((N+2) * sizeof(INT))
    cdef INT i,j,k,order,nTerms
    cdef INT gn

    cdef innerloopfn_ptr innerloop_fn;
    if fastmode:
        innerloop_fn = pr_as_polynomial_innerloop_savepartials
    else:
        innerloop_fn = pr_as_polynomial_innerloop

    #extract raw data from gate_terms dictionary-of-lists for faster lookup
    #gate_term_prefactors = np.empty( (nOperations,max_order+1,dim,dim)
    #cdef unordered_map[INT, vector[vector[unordered_map[INT, complex]]]] gate_term_coeffs
    #cdef vector[vector[unordered_map[INT, complex]]] rho_term_coeffs
    #cdef vector[vector[unordered_map[INT, complex]]] E_term_coeffs
    #cdef vector[vector[INT]] e_indices

    cdef vector[INT]* Einds
    cdef vector[vector_TermCRep_ptr_ptr] factor_lists

    assert(max_order <= 2) # only support this partitioning below (so far)

    cdef vector[PolynomialCRep_ptr] prps = vector[PolynomialCRep_ptr](numEs)
    for i in range(numEs):
        prps[i] = new PolynomialCRep(unordered_map[PolynomialVarsIndex,complex](), max_polynomial_vars, vindices_per_int)
        # create empty polynomials - maybe overload constructor for this?
        # these PolynomialCReps are alloc'd here and returned - it is the job of the caller to
        #  free them (or assign them to new PolynomialRep wrapper objs)

    for order in range(max_order+1):
        #for p in partition_into(order, N):
        for i in range(N+2): p[i] = 0 # clear p
        factor_lists = vector[vector_TermCRep_ptr_ptr](N+2)

        if order == 0:
            #inner loop(p)
            #factor_lists = [ gate_terms[glbl][pi] for glbl,pi in zip(circuit,p) ]
            factor_lists[0] = &rho_term_reps[p[0]]
            for k in range(N):
                gn = circuit[k]
                factor_lists[k+1] = &op_term_reps[circuit[k]][p[k+1]]
                #if factor_lists[k+1].size() == 0: continue # WHAT???
            factor_lists[N+1] = &E_term_reps[p[N+1]]
            Einds = &E_term_indices[p[N+1]]

            #print("Part0 ",p)
            innerloop_fn(factor_lists,Einds,&prps,dim) #, prps_chk)


        elif order == 1:
            for i in range(N+2):
                p[i] = 1
                #inner loop(p)
                factor_lists[0] = &rho_term_reps[p[0]]
                for k in range(N):
                    gn = circuit[k]
                    factor_lists[k+1] = &op_term_reps[gn][p[k+1]]
                    #if len(factor_lists[k+1]) == 0: continue #WHAT???
                factor_lists[N+1] = &E_term_reps[p[N+1]]
                Einds = &E_term_indices[p[N+1]]

                innerloop_fn(factor_lists,Einds,&prps,dim) #, prps_chk)
                p[i] = 0

        elif order == 2:
            for i in range(N+2):
                p[i] = 2
                #inner loop(p)
                factor_lists[0] = &rho_term_reps[p[0]]
                for k in range(N):
                    gn = circuit[k]
                    factor_lists[k+1] = &op_term_reps[circuit[k]][p[k+1]]
                    #if len(factor_lists[k+1]) == 0: continue # WHAT???
                factor_lists[N+1] = &E_term_reps[p[N+1]]
                Einds = &E_term_indices[p[N+1]]

                innerloop_fn(factor_lists,Einds,&prps,dim) #, prps_chk)
                p[i] = 0

            for i in range(N+2):
                p[i] = 1
                for j in range(i+1,N+2):
                    p[j] = 1
                    #inner loop(p)
                    factor_lists[0] = &rho_term_reps[p[0]]
                    for k in range(N):
                        gn = circuit[k]
                        factor_lists[k+1] = &op_term_reps[circuit[k]][p[k+1]]
                        #if len(factor_lists[k+1]) == 0: continue #WHAT???
                    factor_lists[N+1] = &E_term_reps[p[N+1]]
                    Einds = &E_term_indices[p[N+1]]

                    innerloop_fn(factor_lists,Einds,&prps,dim) #, prps_chk)
                    p[j] = 0
                p[i] = 0
        else:
            assert(False) # order > 2 not implemented yet...

    free(p)
    return prps



cdef void pr_as_polynomial_innerloop(vector[vector_TermCRep_ptr_ptr] factor_lists, vector[INT]* Einds,
                                     vector[PolynomialCRep*]* prps, INT dim): #, prps_chk):
    #print("DB partition = ","listlens = ",[len(fl) for fl in factor_lists])

    cdef INT i,j,Ei
    cdef double complex scale, val, newval, pLeft, pRight, p

    cdef TermCRep* factor

    cdef INT nFactorLists = factor_lists.size() # may need to recompute this after fast-mode
    cdef INT* factorListLens = <INT*>malloc(nFactorLists * sizeof(INT))
    cdef INT last_index = nFactorLists-1

    for i in range(nFactorLists):
        factorListLens[i] = factor_lists[i].size()
        if factorListLens[i] == 0:
            free(factorListLens)
            return # nothing to loop over! - (exit before we allocate more)

    cdef PolynomialCRep coeff
    cdef PolynomialCRep result

    cdef StateCRep *prop1 = new StateCRep(dim)
    cdef StateCRep *prop2 = new StateCRep(dim)
    cdef StateCRep *tprop
    cdef EffectCRep* EVec

    cdef INT* b = <INT*>malloc(nFactorLists * sizeof(INT))
    for i in range(nFactorLists): b[i] = 0

    assert(nFactorLists > 0), "Number of factor lists must be > 0!"

    #for factors in _itertools.product(*factor_lists):
    while(True):
        # In this loop, b holds "current" indices into factor_lists
        factor = deref(factor_lists[0])[b[0]]
        coeff = deref(factor._coeff) # an unordered_map (copies to new "coeff" variable)

        for i in range(1,nFactorLists):
            coeff = coeff.mult( deref(deref(factor_lists[i])[b[i]]._coeff) )

        #pLeft / "pre" sim
        factor = deref(factor_lists[0])[b[0]] # 0th-factor = rhoVec
        prop1.copy_from(factor._pre_state)
        for j in range(<INT>factor._pre_ops.size()):
            factor._pre_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop
        for i in range(1,last_index):
            factor = deref(factor_lists[i])[b[i]]
            for j in range(<INT>factor._pre_ops.size()):
                factor._pre_ops[j].acton(prop1,prop2)
                tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
        factor = deref(factor_lists[last_index])[b[last_index]] # the last factor (an Evec)

	# can't propagate effects, so effect's post_ops are constructed to act on *state*
        EVec = factor._post_effect
        for j in range(<INT>factor._post_ops.size()):
            rhoVec = factor._post_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
        pLeft = EVec.amplitude(prop1)

        #pRight / "post" sim
        factor = deref(factor_lists[0])[b[0]] # 0th-factor = rhoVec
        prop1.copy_from(factor._post_state)
        for j in range(<INT>factor._post_ops.size()):
            factor._post_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
        for i in range(1,last_index):
            factor = deref(factor_lists[i])[b[i]]
            for j in range(<INT>factor._post_ops.size()):
                factor._post_ops[j].acton(prop1,prop2)
                tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
        factor = deref(factor_lists[last_index])[b[last_index]] # the last factor (an Evec)

        EVec = factor._pre_effect
        for j in range(<INT>factor._pre_ops.size()):
            factor._pre_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
        pRight = EVec.amplitude(prop1).conjugate()

        #Add result to appropriate polynomial
        result = coeff  # use a reference?
        result.scale(pLeft * pRight)
        final_factor_indx = b[last_index]
        Ei = deref(Einds)[final_factor_indx] #final "factor" index == E-vector index
        deref(prps)[Ei].add_inplace(result)

        #increment b ~ itertools.product & update vec_index_noop = np.dot(self.multipliers, b)
        for i in range(nFactorLists-1,-1,-1):
            if b[i]+1 < factorListLens[i]:
                b[i] += 1
                break
            else:
                b[i] = 0
        else:
            break # can't increment anything - break while(True) loop

    #Clenaup: free allocated memory
    del prop1
    del prop2
    free(factorListLens)
    free(b)
    return


cdef void pr_as_polynomial_innerloop_savepartials(vector[vector_TermCRep_ptr_ptr] factor_lists,
                                                  vector[INT]* Einds, vector[PolynomialCRep*]* prps, INT dim): #, prps_chk):
    #print("DB partition = ","listlens = ",[len(fl) for fl in factor_lists])

    cdef INT i,j,Ei
    cdef double complex scale, val, newval, pLeft, pRight, p

    cdef INT incd
    cdef TermCRep* factor

    cdef INT nFactorLists = factor_lists.size() # may need to recompute this after fast-mode
    cdef INT* factorListLens = <INT*>malloc(nFactorLists * sizeof(INT))
    cdef INT last_index = nFactorLists-1

    for i in range(nFactorLists):
        factorListLens[i] = factor_lists[i].size()
        if factorListLens[i] == 0:
            free(factorListLens)
            return # nothing to loop over! (exit before we allocate anything else)

    cdef PolynomialCRep coeff
    cdef PolynomialCRep result

    #fast mode
    cdef vector[StateCRep*] leftSaved = vector[StateCRep_ptr](nFactorLists-1)  # saved[i] is state after i-th
    cdef vector[StateCRep*] rightSaved = vector[StateCRep_ptr](nFactorLists-1) # factor has been applied
    cdef vector[PolynomialCRep] coeffSaved = vector[PolynomialCRep](nFactorLists-1)
    cdef StateCRep *shelved = new StateCRep(dim)
    cdef StateCRep *prop2 = new StateCRep(dim) # prop2 is always a temporary allocated state not owned by anything else
    cdef StateCRep *prop1
    cdef StateCRep *tprop
    cdef EffectCRep* EVec

    cdef INT* b = <INT*>malloc(nFactorLists * sizeof(INT))
    for i in range(nFactorLists): b[i] = 0
    assert(nFactorLists > 0), "Number of factor lists must be > 0!"

    incd = 0

    #Fill saved arrays with allocated states
    for i in range(nFactorLists-1):
        leftSaved[i] = new StateCRep(dim)
        rightSaved[i] = new StateCRep(dim)

    #for factors in _itertools.product(*factor_lists):
    #for incd,fi in incd_product(*[range(len(l)) for l in factor_lists]):
    while(True):
        # In this loop, b holds "current" indices into factor_lists
        #print "DB: iter-product BEGIN"

        if incd == 0: # need to re-evaluate rho vector
            #print "DB: re-eval at incd=0"
            factor = deref(factor_lists[0])[b[0]]

            #print "DB: re-eval left"
            prop1 = leftSaved[0] # the final destination (prop2 is already alloc'd)
            prop1.copy_from(factor._pre_state)
            for j in range(<INT>factor._pre_ops.size()):
                #print "DB: re-eval left item"
                factor._pre_ops[j].acton(prop1,prop2)
                tprop = prop1; prop1 = prop2; prop2 = tprop # swap prop1 <-> prop2
            rhoVecL = prop1
            leftSaved[0] = prop1 # final state -> saved
            # (prop2 == the other allocated state)

            #print "DB: re-eval right"
            prop1 = rightSaved[0] # the final destination (prop2 is already alloc'd)
            prop1.copy_from(factor._post_state)
            for j in range(<INT>factor._post_ops.size()):
                #print "DB: re-eval right item"
                factor._post_ops[j].acton(prop1,prop2)
                tprop = prop1; prop1 = prop2; prop2 = tprop # swap prop1 <-> prop2
            rhoVecR = prop1
            rightSaved[0] = prop1 # final state -> saved
            # (prop2 == the other allocated state)

            #print "DB: re-eval coeff"
            coeff = deref(factor._coeff)
            coeffSaved[0] = coeff
            incd += 1
        else:
            #print "DB: init from incd " #,incd,last_index,nFactorLists,dim
            rhoVecL = leftSaved[incd-1]
            rhoVecR = rightSaved[incd-1]
            coeff = coeffSaved[incd-1]

        # propagate left and right states, saving as we go
        for i in range(incd,last_index):
            #print "DB: propagate left begin"
            factor = deref(factor_lists[i])[b[i]]
            prop1 = leftSaved[i] # destination
            prop1.copy_from(rhoVecL) #starting state
            for j in range(<INT>factor._pre_ops.size()):
                #print "DB: propagate left item"
                factor._pre_ops[j].acton(prop1,prop2)
                tprop = prop1; prop1 = prop2; prop2 = tprop
            rhoVecL = prop1
            leftSaved[i] = prop1

            # (prop2 == the other allocated state)

            #print "DB: propagate right begin"
            prop1 = rightSaved[i] # destination
            prop1.copy_from(rhoVecR) #starting state
            for j in range(<INT>factor._post_ops.size()):
                #print "DB: propagate right item"
                factor._post_ops[j].acton(prop1,prop2)
                tprop = prop1; prop1 = prop2; prop2 = tprop
            rhoVecR = prop1
            rightSaved[i] = prop1
            # (prop2 == the other allocated state)

            #print "DB: propagate coeff mult"
            coeff = coeff.mult(deref(factor._coeff)) # copy a PolynomialCRep
            coeffSaved[i] = coeff

        # for the last index, no need to save, and need to construct
        # and apply effect vector
        prop1 = shelved # so now prop1 (and prop2) are alloc'd states

        #print "DB: left ampl"
        factor = deref(factor_lists[last_index])[b[last_index]] # the last factor (an Evec)
        EVec = factor._post_effect
        prop1.copy_from(rhoVecL) # initial state (prop2 already alloc'd)
        for j in range(<INT>factor._post_ops.size()):
            factor._post_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop
        pLeft = EVec.amplitude(prop1) # output in prop1, so this is final amplitude

        #print "DB: right ampl"
        EVec = factor._pre_effect
        prop1.copy_from(rhoVecR)
        for j in range(<INT>factor._pre_ops.size()):
            factor._pre_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop
        pRight = EVec.amplitude(prop1).conjugate()

        shelved = prop1 # return prop1 to the "shelf" since we'll use prop1 for other things next

        #print "DB: final block"
        #print "DB running coeff = ",dict(coeff._coeffs)
        #print "DB factor coeff = ",dict(factor._coeff._coeffs)
        result = coeff.mult(deref(factor._coeff))
        #print "DB result = ",dict(result._coeffs)
        result.scale(pLeft * pRight)
        final_factor_indx = b[last_index]
        Ei = deref(Einds)[final_factor_indx] #final "factor" index == E-vector index
        deref(prps)[Ei].add_inplace(result)
        #print "DB prps[",INT(Ei),"] = ",dict(deref(prps)[Ei]._coeffs)

        #assert(debug < 100) #DEBUG
        #print "DB: end product loop"

        #increment b ~ itertools.product & update vec_index_noop = np.dot(self.multipliers, b)
        for i in range(nFactorLists-1,-1,-1):
            if b[i]+1 < factorListLens[i]:
                b[i] += 1; incd = i
                break
            else:
                b[i] = 0
        else:
            break # can't increment anything - break while(True) loop

    #Cleanup: free allocated memory
    for i in range(nFactorLists-1):
        del leftSaved[i]
        del rightSaved[i]
    del prop2
    del shelved
    free(factorListLens)
    free(b)
    return


# State-vector pruned-polynomial-term calcs -------------------------
def create_circuitsetup_cacheel(fwdsim, rholabel, elabels, circuit, repcache, min_term_mag, mpv):

    cdef INT i, j
    cdef vector[INT] cgatestring

    cdef RepCacheEl repcel;
    cdef vector[TermCRep_ptr] treps;
    cdef TermRep rep;
    cdef unordered_map[INT, vector[TermCRep_ptr] ] op_term_reps = unordered_map[INT, vector[TermCRep_ptr] ]();
    cdef unordered_map[INT, vector[INT] ] op_foat_indices = unordered_map[INT, vector[INT] ]();
    cdef vector[TermCRep_ptr] rho_term_reps;
    cdef vector[INT] rho_foat_indices;
    cdef vector[TermCRep_ptr] E_term_reps = vector[TermCRep_ptr](0);
    cdef vector[INT] E_foat_indices = vector[INT](0);
    cdef vector[INT] e_indices = vector[INT](0);
    cdef TermCRep_ptr cterm;
    cdef CircuitSetupCacheEl cscel = CircuitSetupCacheEl()

    # Create gatelable -> int mapping to be used throughout
    distinct_gateLabels = sorted(set(circuit))
    glmap = { gl: i for i,gl in enumerate(distinct_gateLabels) }

    # Convert circuit to a vector of ints
    for gl in circuit:
        cgatestring.push_back(<INT>glmap[gl])

    # Construct dict of gate term reps, then *convert* to c-reps, as this
    #  keeps alive the non-c-reps which keep the c-reps from being deallocated...
    for glbl in distinct_gateLabels:
        if glbl in repcache:
            repcel = <RepCacheEl>repcache[glbl]
        else:
            repcel = RepCacheEl()
            op = fwdsim.model._circuit_layer_operator(glbl, 'op')
            hmterms, foat_indices = op.highmagnitude_terms(
                min_term_mag, max_taylor_order=fwdsim.max_order, max_polynomial_vars=mpv)

            #DEBUG CHECK
            #if glbl in check_opcache:
            #    if np.linalg.norm( check_opcache[glbl].to_vector() - op.to_vector() ) > 1e-6:
            #        print("HERE!!!")
            #        raise ValueError("HERE!!!")
            #else:
            #    check_opcache[glbl] = op

            #DEBUG CHECK TERM MAGNITUDES make sense
            #chk_tot_mag = sum([t.magnitude for t in hmterms])
            #chk_tot_mag2 = op.total_term_magnitude()
            #if chk_tot_mag > chk_tot_mag2+1e-5: # give a tolerance here
            #    print "Warning: highmag terms for ",str(glbl),": ",len(hmterms)," have total mag = ",chk_tot_mag," but max should be ",chk_tot_mag2,"!!"
            #else:
            #    print "Highmag terms recomputed (OK) - made op = ", db_made_op

            for t in hmterms:
                rep = (<TermRep>t.torep())
                repcel.pyterm_references.append(rep)
                repcel.reps.push_back( rep.c_term )

            for i in foat_indices:
                repcel.foat_indices.push_back(<INT>i)
            repcache[glbl] = repcel

        op_term_reps[ glmap[glbl] ] = repcel.reps
        op_foat_indices[ glmap[glbl] ] = repcel.foat_indices

    #Similar with rho_terms and E_terms
    if rholabel in repcache:
        repcel = repcache[rholabel]
    else:
        repcel = RepCacheEl()
        rhoOp = fwdsim.model._circuit_layer_operator(rholabel, 'prep')
        hmterms, foat_indices = rhoOp.highmagnitude_terms(
            min_term_mag, max_taylor_order=fwdsim.max_order,
            max_polynomial_vars=mpv)

        for t in hmterms:
            rep = (<TermRep>t.torep())
            repcel.pyterm_references.append(rep)
            repcel.reps.push_back( rep.c_term )

        for i in foat_indices:
            repcel.foat_indices.push_back(<INT>i)
        repcache[rholabel] = repcel

    rho_term_reps = repcel.reps
    rho_foat_indices = repcel.foat_indices

    elabels = tuple(elabels) # so hashable
    if elabels in repcache:
        repcel = <RepCacheEl>repcache[elabels]
    else:
        repcel = RepCacheEl()
        E_term_indices_and_reps = []
        for i,elbl in enumerate(elabels):
            Evec = fwdsim.model._circuit_layer_operator(elbl, 'povm')
            hmterms, foat_indices = Evec.highmagnitude_terms(
                min_term_mag, max_taylor_order=fwdsim.max_order, max_polynomial_vars=mpv)
            E_term_indices_and_reps.extend(
                [ (i,t,t.magnitude,1 if (j in foat_indices) else 0) for j,t in enumerate(hmterms) ] )

        #Sort all terms by magnitude
        E_term_indices_and_reps.sort(key=lambda x: x[2], reverse=True)
        for j,(i,t,_,is_foat) in enumerate(E_term_indices_and_reps):
            rep = (<TermRep>t.torep())
            repcel.pyterm_references.append(rep)
            repcel.reps.push_back( rep.c_term )
            repcel.e_indices.push_back(<INT>i)
            if(is_foat): repcel.foat_indices.push_back(<INT>j)
        repcache[elabels] = repcel

    E_term_reps = repcel.reps
    e_indices = repcel.e_indices
    E_foat_indices = repcel.foat_indices

    cscel.cgatestring = cgatestring
    cscel.rho_term_reps = rho_term_reps
    cscel.op_term_reps = op_term_reps
    cscel.E_term_reps = E_term_reps
    cscel.rho_foat_indices = rho_foat_indices
    cscel.op_foat_indices = op_foat_indices
    cscel.E_foat_indices = E_foat_indices
    cscel.e_indices = e_indices
    return cscel


def refresh_magnitudes_in_repcache(repcache, paramvec):
    cdef RepCacheEl repcel
    cdef TermRep termrep
    cdef np.ndarray coeff_array

    for repcel in repcache.values():
        #repcel = <RepCacheEl?>repcel
        for termrep in repcel.pyterm_references:
            coeff_array = _fastopcalc.bulk_eval_compact_polynomials_complex(termrep.compact_coeff[0],termrep.compact_coeff[1],paramvec,(1,))
            termrep.set_magnitude_only(abs(coeff_array[0]))


def find_best_pathmagnitude_threshold(fwdsim, rholabel, elabels, circuit, polynomial_vindices_per_int,
                                      repcache, circuitsetup_cache, comm=None, mem_limit=None,
                                      pathmagnitude_gap=0.0, min_term_mag=0.01, max_paths=500, threshold_guess=0.0):

    cdef INT i
    cdef INT numEs = len(elabels)
    cdef INT mpv = fwdsim.model.num_params # max_polynomial_vars
    cdef INT vpi = polynomial_vindices_per_int  #pass this in directly so fwdsim can compute once & use multiple times
    cdef CircuitSetupCacheEl cscel;

    bHit = (circuit in circuitsetup_cache)
    if circuit in circuitsetup_cache:
        cscel = <CircuitSetupCacheEl?>circuitsetup_cache[circuit]
    else:
        cscel = <CircuitSetupCacheEl?>create_circuitsetup_cacheel(fwdsim, rholabel, elabels, circuit, repcache, min_term_mag, mpv)
        circuitsetup_cache[circuit] = cscel

    #MEM REMOVE above circuitsetup cache seems to use memory!
    #MEM REMOVE return 100, 1.0, 1.0, 1.0 #total_npaths, threshold, total_target_sopm, total_achieved_sopm

    cdef vector[double] target_sum_of_pathmags = vector[double](numEs)
    cdef vector[double] achieved_sum_of_pathmags = vector[double](numEs)
    cdef vector[INT] npaths = vector[INT](numEs)

    #Get MAX-SOPM for circuit outcomes and thereby the target SOPM (via MAX - gap)
    cdef double max_partial_sopm = fwdsim.model._circuit_layer_operator(rholabel, 'prep').total_term_magnitude
    for glbl in circuit:
        op = fwdsim.model._circuit_layer_operator(glbl, 'op')
        max_partial_sopm *= op.total_term_magnitude
    for i,elbl in enumerate(elabels):
        target_sum_of_pathmags[i] = max_partial_sopm * fwdsim.model._circuit_layer_operator(elbl, 'povm').total_term_magnitude - pathmagnitude_gap  # absolute gap
        #target_sum_of_pathmags[i] = max_partial_sopm * fwdsim.sos.get_effect(elbl).total_term_magnitude * (1.0 - pathmagnitude_gap)  # relative gap

    cdef double threshold = c_find_best_pathmagnitude_threshold(
        cscel.cgatestring, cscel.rho_term_reps, cscel.op_term_reps, cscel.E_term_reps,
        cscel.rho_foat_indices, cscel.op_foat_indices, cscel.E_foat_indices, cscel.e_indices,
        numEs, pathmagnitude_gap, min_term_mag, max_paths, threshold_guess, target_sum_of_pathmags,
        achieved_sum_of_pathmags, npaths)

    cdef INT total_npaths = 0
    cdef double total_target_sopm = 0.0
    cdef double total_achieved_sopm = 0.0
    for i in range(numEs):
        total_npaths += npaths[i]
        total_target_sopm += target_sum_of_pathmags[i]
        total_achieved_sopm += achieved_sum_of_pathmags[i]

    return total_npaths, threshold, total_target_sopm, total_achieved_sopm


cdef double c_find_best_pathmagnitude_threshold(
    vector[INT]& circuit, vector[TermCRep_ptr] rho_term_reps, unordered_map[INT, vector[TermCRep_ptr]] op_term_reps, vector[TermCRep_ptr] E_term_reps,
    vector[INT] rho_foat_indices, unordered_map[INT,vector[INT]] op_foat_indices, vector[INT] E_foat_indices, vector[INT] e_indices,
    INT numEs, double pathmagnitude_gap, double min_term_mag, INT max_paths, double threshold_guess,
    vector[double]& target_sum_of_pathmags, vector[double]& achieved_sum_of_pathmags, vector[INT]& npaths):

    #NOTE: circuit and gate_terms use *integers* as operation labels, not Label objects, to speed
    # lookups and avoid weird string conversion stuff with Cython

    cdef INT N = circuit.size()
    cdef INT nFactorLists = N+2
    #cdef INT n = N+2 # number of factor lists
    #cdef INT* p = <INT*>malloc((N+2) * sizeof(INT))
    cdef INT i #,j,k #,order,nTerms
    #cdef INT gn

    cdef INT t0 = time.clock()
    #cdef INT t, nPaths; #for below

    cdef vector[vector_TermCRep_ptr_ptr] factor_lists = vector[vector_TermCRep_ptr_ptr](nFactorLists)
    cdef vector[vector_INT_ptr] foat_indices_per_op = vector[vector_INT_ptr](nFactorLists)
    cdef vector[INT] nops = vector[INT](nFactorLists)
    cdef vector[INT] b = vector[INT](nFactorLists)

    factor_lists[0] = &rho_term_reps
    foat_indices_per_op[0] = &rho_foat_indices
    for i in range(N):
        factor_lists[i+1] = &op_term_reps[circuit[i]]
        foat_indices_per_op[i+1] = &op_foat_indices[circuit[i]]
    factor_lists[N+1] = &E_term_reps
    foat_indices_per_op[N+1] = &E_foat_indices

    cdef double threshold = pathmagnitude_threshold(factor_lists, e_indices, numEs, target_sum_of_pathmags, foat_indices_per_op,
                                              threshold_guess, pathmagnitude_gap / (3.0*max_paths), max_paths,
                                              achieved_sum_of_pathmags, npaths)  # 3.0 is heuristic

    #DEBUG CHECK that counting paths using this threshold gives the same results (can REMOVE)
    #cdef INT NO_LIMIT = 1000000000
    #cdef vector[double] check_mags = vector[double](numEs)
    #cdef vector[INT] check_npaths = vector[INT](numEs)
    #for i in range(numEs):
    #    check_mags[i] = 0.0; check_npaths[i] = 0
    #count_paths_upto_threshold(factor_lists, threshold, numEs,
    #                           foat_indices_per_op, e_indices, NO_LIMIT,
    #                           check_mags, check_npaths)
    #for i in range(numEs):
    #    assert(abs(achieved_sum_of_pathmags[i] - check_mags[i]) < 1e-8)
    #    assert(npaths[i] == check_npaths[i])

    #print("Threshold = ",threshold)
    #print("Mags = ",achieved_sum_of_pathmags)
    ##print("Check Mags = ",check_mags)
    #print("npaths = ",npaths)
    ##print("Check npaths = ",check_npaths)
    ##print("Target sopm = ",target_sum_of_pathmags)  # max - gap

    return threshold


def compute_pruned_path_polynomials_given_threshold(
        threshold, fwdsim, rholabel, elabels, circuit, polynomial_vindices_per_int,
        repcache, circuitsetup_cache, comm=None, mem_limit=None, fastmode=1):

    cdef INT i
    cdef INT numEs = len(elabels)
    cdef INT mpv = fwdsim.model.num_params # max_polynomial_vars
    cdef INT vpi = polynomial_vindices_per_int  #pass this in directly so fwdsim can compute once & use multiple times
    cdef INT stateDim = int(round(np.sqrt(fwdsim.model.dim)))
    cdef double min_term_mag = fwdsim.min_term_mag
    cdef CircuitSetupCacheEl cscel;

    bHit = (circuit in circuitsetup_cache)
    if circuit in circuitsetup_cache:
        cscel = <CircuitSetupCacheEl?>circuitsetup_cache[circuit]
    else:
        cscel = <CircuitSetupCacheEl?>create_circuitsetup_cacheel(fwdsim, rholabel, elabels, circuit, repcache, min_term_mag, mpv)
        circuitsetup_cache[circuit] = cscel

    cdef vector[PolynomialCRep*] polynomials = c_compute_pruned_polynomials_given_threshold(
        <double>threshold, cscel.cgatestring, cscel.rho_term_reps, cscel.op_term_reps, cscel.E_term_reps,
        cscel.rho_foat_indices, cscel.op_foat_indices, cscel.E_foat_indices, cscel.e_indices,
        numEs, stateDim, <INT>fastmode,  mpv, vpi)

    return [ PolynomialRep_from_allocd_PolynomialCRep(polynomials[i]) for i in range(<INT>polynomials.size()) ]


cdef vector[PolynomialCRep*] c_compute_pruned_polynomials_given_threshold(
    double threshold, vector[INT]& circuit,
    vector[TermCRep_ptr] rho_term_reps, unordered_map[INT, vector[TermCRep_ptr]] op_term_reps, vector[TermCRep_ptr] E_term_reps,
    vector[INT] rho_foat_indices, unordered_map[INT,vector[INT]] op_foat_indices, vector[INT] E_foat_indices, vector[INT] e_indices,
    INT numEs, INT dim, INT fastmode, INT max_polynomial_vars, INT vindices_per_int):

    cdef INT N = circuit.size()
    cdef INT nFactorLists = N+2
    cdef INT i

    cdef vector[vector_TermCRep_ptr_ptr] factor_lists = vector[vector_TermCRep_ptr_ptr](nFactorLists)
    cdef vector[vector_INT_ptr] foat_indices_per_op = vector[vector_INT_ptr](nFactorLists)
    cdef vector[INT] nops = vector[INT](nFactorLists)
    cdef vector[INT] b = vector[INT](nFactorLists)

    factor_lists[0] = &rho_term_reps
    foat_indices_per_op[0] = &rho_foat_indices
    for i in range(N):
        factor_lists[i+1] = &op_term_reps[circuit[i]]
        foat_indices_per_op[i+1] = &op_foat_indices[circuit[i]]
    factor_lists[N+1] = &E_term_reps
    foat_indices_per_op[N+1] = &E_foat_indices

    cdef vector[PolynomialCRep_ptr] prps = vector[PolynomialCRep_ptr](numEs)
    for i in range(numEs):
        prps[i] = new PolynomialCRep(unordered_map[PolynomialVarsIndex,complex](), max_polynomial_vars, vindices_per_int)
        # create empty polynomials - maybe overload constructor for this?
        # these PolynomialCReps are alloc'd here and returned - it is the job of the caller to
        #  free them (or assign them to new PolynomialRep wrapper objs)

    cdef double log_thres = log10(threshold)
    cdef double current_mag = 1.0
    cdef double current_logmag = 0.0
    for i in range(nFactorLists):
        nops[i] = factor_lists[i].size()
        b[i] = 0

    ## fn_visitpath(b, current_mag, 0) # visit root (all 0s) path
    cdef addpathfn_ptr addpath_fn;
    cdef vector[StateCRep*] leftSaved = vector[StateCRep_ptr](nFactorLists-1)  # saved[i] is state after i-th
    cdef vector[StateCRep*] rightSaved = vector[StateCRep_ptr](nFactorLists-1) # factor has been applied
    cdef vector[PolynomialCRep] coeffSaved = vector[PolynomialCRep](nFactorLists-1)

    #Fill saved arrays with allocated states
    if fastmode == 1: # fastmode
        #fast mode
        addpath_fn = add_path_savepartials
        for i in range(nFactorLists-1):
            leftSaved[i] = new StateCRep(dim)
            rightSaved[i] = new StateCRep(dim)

    elif fastmode == 2: #achieved-SOPM mode
        addpath_fn = add_path_achievedsopm
        for i in range(nFactorLists-1):
            leftSaved[i] = NULL
            rightSaved[i] = NULL

    else:
        addpath_fn = add_path
        for i in range(nFactorLists-1):
            leftSaved[i] = NULL
            rightSaved[i] = NULL

    cdef StateCRep *prop1 = new StateCRep(dim)
    cdef StateCRep *prop2 = new StateCRep(dim)
    addpath_fn(&prps, b, 0, factor_lists, &prop1, &prop2, &e_indices, &leftSaved, &rightSaved, &coeffSaved)
    ## -------------------------------

    add_paths(addpath_fn, b, factor_lists, foat_indices_per_op, numEs, nops, e_indices, 0, log_thres,
              current_mag, current_logmag, 0, &prps, &prop1, &prop2, &leftSaved, &rightSaved, &coeffSaved, 0)

    del prop1
    del prop2

    return prps


cdef void add_path(vector[PolynomialCRep*]* prps, vector[INT]& b, INT incd, vector[vector_TermCRep_ptr_ptr]& factor_lists,
                   StateCRep **pprop1, StateCRep **pprop2, vector[INT]* Einds,
                   vector[StateCRep*]* pleftSaved, vector[StateCRep*]* prightSaved, vector[PolynomialCRep]* pcoeffSaved):

    cdef PolynomialCRep coeff
    cdef PolynomialCRep result
    cdef double complex pLeft, pRight

    cdef INT i,j, Ei
    cdef TermCRep* factor
    cdef StateCRep *prop1 = deref(pprop1)
    cdef StateCRep *prop2 = deref(pprop2)
    cdef StateCRep *tprop
    cdef EffectCRep* EVec
    cdef StateCRep *rhoVec
    cdef INT nFactorLists = b.size()
    cdef INT last_index = nFactorLists-1
    # ** Assume prop1 and prop2 begin as allocated **

    # In this loop, b holds "current" indices into factor_lists
    factor = deref(factor_lists[0])[b[0]]
    coeff = deref(factor._coeff) # an unordered_map (copies to new "coeff" variable)

    for i in range(1,nFactorLists):
        coeff = coeff.mult( deref(deref(factor_lists[i])[b[i]]._coeff) )

    #pLeft / "pre" sim
    factor = deref(factor_lists[0])[b[0]] # 0th-factor = rhoVec
    prop1.copy_from(factor._pre_state)
    for j in range(<INT>factor._pre_ops.size()):
        factor._pre_ops[j].acton(prop1,prop2)
        tprop = prop1; prop1 = prop2; prop2 = tprop
    for i in range(1,last_index):
        factor = deref(factor_lists[i])[b[i]]
        for j in range(<INT>factor._pre_ops.size()):
            factor._pre_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
    factor = deref(factor_lists[last_index])[b[last_index]] # the last factor (an Evec)

	# can't propagate effects, so effect's post_ops are constructed to act on *state*
    EVec = factor._post_effect
    for j in range(<INT>factor._pre_ops.size()):
        rhoVec = factor._pre_ops[j].acton(prop1,prop2)
        tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
    pLeft = EVec.amplitude(prop1)

    #pRight / "post" sim
    factor = deref(factor_lists[0])[b[0]] # 0th-factor = rhoVec
    prop1.copy_from(factor._post_state)
    for j in range(<INT>factor._post_ops.size()):
        factor._post_ops[j].acton(prop1,prop2)
        tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
    for i in range(1,last_index):
        factor = deref(factor_lists[i])[b[i]]
        for j in range(<INT>factor._post_ops.size()):
            factor._post_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
    factor = deref(factor_lists[last_index])[b[last_index]] # the last factor (an Evec)

    EVec = factor._pre_effect
    for j in range(<INT>factor._post_ops.size()):
        factor._post_ops[j].acton(prop1,prop2)
        tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
    pRight = EVec.amplitude(prop1).conjugate()


    #Add result to appropriate polynomial
    result = coeff  # use a reference?
    result.scale(pLeft * pRight)
    Ei = deref(Einds)[ b[last_index] ] #final "factor" index == E-vector index
    #print("Ei = ",Ei," size = ",deref(prps).size())
    #print("result = ")
    #for x in result._coeffs:
    #    print x.first._parts, x.second
    #print("prps = ")
    #for x in deref(prps)[Ei]._coeffs:
    #    print x.first._parts, x.second
    deref(prps)[Ei].add_inplace(result)

    #Update the slots held by prop1 and prop2, which still have allocated states (though really no need?)
    pprop1[0] = prop1
    pprop2[0] = prop2


cdef void add_path_achievedsopm(vector[PolynomialCRep*]* prps, vector[INT]& b, INT incd, vector[vector_TermCRep_ptr_ptr]& factor_lists,
                                StateCRep **pprop1, StateCRep **pprop2, vector[INT]* Einds,
                                vector[StateCRep*]* pleftSaved, vector[StateCRep*]* prightSaved, vector[PolynomialCRep]* pcoeffSaved):

    cdef PolynomialCRep coeff
    cdef PolynomialCRep result

    cdef INT i,j, Ei
    cdef TermCRep* factor
    cdef INT nFactorLists = b.size()
    cdef INT last_index = nFactorLists-1

    # In this loop, b holds "current" indices into factor_lists
    factor = deref(factor_lists[0])[b[0]]
    coeff = deref(factor._coeff).abs() # an unordered_map (copies to new "coeff" variable)

    for i in range(1,nFactorLists):
        coeff = coeff.abs_mult( deref(deref(factor_lists[i])[b[i]]._coeff) )

    #Add result to appropriate polynomial
    result = coeff  # use a reference?
    Ei = deref(Einds)[ b[last_index] ] #final "factor" index == E-vector index
    deref(prps)[Ei].add_abs_inplace(result)


cdef void add_path_savepartials(vector[PolynomialCRep*]* prps, vector[INT]& b, INT incd, vector[vector_TermCRep_ptr_ptr]& factor_lists,
                                StateCRep** pprop1, StateCRep** pprop2, vector[INT]* Einds,
                                vector[StateCRep*]* pleftSaved, vector[StateCRep*]* prightSaved, vector[PolynomialCRep]* pcoeffSaved):

    cdef PolynomialCRep coeff
    cdef PolynomialCRep result
    cdef double complex pLeft, pRight

    cdef INT i,j, Ei
    cdef TermCRep* factor
    cdef StateCRep *prop1 = deref(pprop1)
    cdef StateCRep *prop2 = deref(pprop2)
    cdef StateCRep *tprop
    cdef StateCRep *shelved = prop1
    cdef EffectCRep* EVec
    cdef StateCRep *rhoVec
    cdef INT nFactorLists = b.size()
    cdef INT last_index = nFactorLists-1
    # ** Assume shelved and prop2 begin as allocated **

    if incd == 0: # need to re-evaluate rho vector
        #print "DB: re-eval at incd=0"
        factor = deref(factor_lists[0])[b[0]]

        #print "DB: re-eval left"
        prop1 = deref(pleftSaved)[0] # the final destination (prop2 is already alloc'd)
        prop1.copy_from(factor._pre_state)
        for j in range(<INT>factor._pre_ops.size()):
            #print "DB: re-eval left item"
            factor._pre_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop # swap prop1 <-> prop2
        rhoVecL = prop1
        deref(pleftSaved)[0] = prop1 # final state -> saved
        # (prop2 == the other allocated state)

        #print "DB: re-eval right"
        prop1 = deref(prightSaved)[0] # the final destination (prop2 is already alloc'd)
        prop1.copy_from(factor._post_state)
        for j in range(<INT>factor._post_ops.size()):
            #print "DB: re-eval right item"
            factor._post_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop # swap prop1 <-> prop2
        rhoVecR = prop1
        deref(prightSaved)[0] = prop1 # final state -> saved
        # (prop2 == the other allocated state)

        #print "DB: re-eval coeff"
        coeff = deref(factor._coeff)
        deref(pcoeffSaved)[0] = coeff
        incd += 1
    else:
        #print "DB: init from incd"
        rhoVecL = deref(pleftSaved)[incd-1]
        rhoVecR = deref(prightSaved)[incd-1]
        coeff = deref(pcoeffSaved)[incd-1]

    # propagate left and right states, saving as we go
    for i in range(incd,last_index):
        #print "DB: propagate left begin"
        factor = deref(factor_lists[i])[b[i]]
        prop1 = deref(pleftSaved)[i] # destination
        prop1.copy_from(rhoVecL) #starting state
        for j in range(<INT>factor._pre_ops.size()):
            #print "DB: propagate left item"
            factor._pre_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop
        rhoVecL = prop1
        deref(pleftSaved)[i] = prop1
        # (prop2 == the other allocated state)

        #print "DB: propagate right begin"
        prop1 = deref(prightSaved)[i] # destination
        prop1.copy_from(rhoVecR) #starting state
        for j in range(<INT>factor._post_ops.size()):
            #print "DB: propagate right item"
            factor._post_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop
        rhoVecR = prop1
        deref(prightSaved)[i] = prop1
        # (prop2 == the other allocated state)

        #print "DB: propagate coeff mult"
        coeff = coeff.mult(deref(factor._coeff)) # copy a PolynomialCRep
        deref(pcoeffSaved)[i] = coeff

    # for the last index, no need to save, and need to construct
    # and apply effect vector
    prop1 = shelved # so now prop1 (and prop2) are alloc'd states

    #print "DB: left ampl"
    factor = deref(factor_lists[last_index])[b[last_index]] # the last factor (an Evec)
    EVec = factor._post_effect
    prop1.copy_from(rhoVecL) # initial state (prop2 already alloc'd)
    for j in range(<INT>factor._pre_ops.size()):
        factor._pre_ops[j].acton(prop1,prop2)
        tprop = prop1; prop1 = prop2; prop2 = tprop
    pLeft = EVec.amplitude(prop1) # output in prop1, so this is final amplitude

    #print "DB: right ampl"
    EVec = factor._pre_effect
    prop1.copy_from(rhoVecR)
    for j in range(<INT>factor._post_ops.size()):
        factor._post_ops[j].acton(prop1,prop2)
        tprop = prop1; prop1 = prop2; prop2 = tprop
    pRight = EVec.amplitude(prop1).conjugate()

    shelved = prop1 # return prop1 to the "shelf" since we'll use prop1 for other things next

    #print "DB: final block"
    #print "DB running coeff = ",dict(coeff._coeffs)
    #print "DB factor coeff = ",dict(factor._coeff._coeffs)
    result = coeff.mult(deref(factor._coeff))
    #print "DB result = ",dict(result._coeffs)
    result.scale(pLeft * pRight)
    Ei = deref(Einds)[b[last_index]] #final "factor" index == E-vector index
    deref(prps)[Ei].add_inplace(result)

    #Update the slots held by prop1 and prop2, which still have allocated states
    pprop1[0] = prop1 # b/c can't deref(pprop1) = prop1 isn't allowed (?)
    pprop2[0] = prop2 # b/c can't deref(pprop2) = prop1 isn't allowed (?)
    #print "DB prps[",INT(Ei),"] = ",dict(deref(prps)[Ei]._coeffs)


cdef void add_paths(addpathfn_ptr addpath_fn, vector[INT]& b, vector[vector_TermCRep_ptr_ptr] oprep_lists,
                    vector[vector_INT_ptr] foat_indices_per_op, INT num_elabels,
                    vector[INT]& nops, vector[INT]& e_indices, INT incd, double log_thres,
                    double current_mag, double current_logmag, INT order,
                    vector[PolynomialCRep*]* prps, StateCRep **pprop1, StateCRep **pprop2,
                    vector[StateCRep*]* pleftSaved, vector[StateCRep*]* prightSaved, vector[PolynomialCRep]* pcoeffSaved,
                    INT current_nzeros):
    """ first_order means only one b[i] is incremented, e.g. b == [0 1 0] or [4 0 0] """
    cdef INT i, j, k, orig_bi, orig_bn
    cdef INT n = b.size()
    cdef INT sub_order
    cdef double mag, mag2

    for i in range(n-1, incd-1, -1):
        if b[i]+1 == nops[i]: continue
        b[i] += 1

        if order == 0: # then incd doesn't matter b/c can inc anything to become 1st order
            sub_order = 1 if (i != n-1 or b[i] >= num_elabels) else 0
        elif order == 1:
            # we started with a first order term where incd was incremented, and now
            # we're incrementing something else
            sub_order = 1 if i == incd else 2 # signifies anything over 1st order where >1 column has be inc'd
        else:
            sub_order = order

        logmag = current_logmag + (deref(oprep_lists[i])[b[i]]._logmagnitude - deref(oprep_lists[i])[b[i]-1]._logmagnitude)
        if logmag >= log_thres:
            numerator = deref(oprep_lists[i])[b[i]]._magnitude
            denom = deref(oprep_lists[i])[b[i]-1]._magnitude
            nzeros = current_nzeros
            if denom == 0:
                denom = SMALL; nzeros -= 1; logmag  = logmag - LOGSMALL
            if numerator == 0:
                numerator = SMALL; nzeros += 1; logmag = logmag + LOGSMALL
            mag = current_mag * (numerator / denom)

            ## fn_visitpath(b, mag, i) ##
            if nzeros == 0:
                addpath_fn(prps, b, i, oprep_lists, pprop1, pprop2, &e_indices, pleftSaved, prightSaved, pcoeffSaved)
            ## --------------------------

            #add any allowed paths beneath this one
            add_paths(addpath_fn, b, oprep_lists, foat_indices_per_op, num_elabels, nops, e_indices,
                      i, log_thres, mag, logmag, sub_order, prps, pprop1, pprop2,
                      pleftSaved, prightSaved, pcoeffSaved, nzeros)

        elif sub_order <= 1:
            #We've rejected term-index b[i] (in column i) because it's too small - the only reason
            # to accept b[i] or term indices higher than it is to include "foat" terms, so we now
            # iterate through any remaining foat indices for this column (we've accepted all lower
            # values of b[i], or we wouldn't be here).  Note that we just need to visit the path,
            # we don't need to traverse down, since we know the path magnitude is already too low.
            orig_bi = b[i]
            for j in deref(foat_indices_per_op[i]):
                if j >= orig_bi:
                    b[i] = j
                    nzeros = current_nzeros
                    numerator = deref(oprep_lists[i])[b[i]]._magnitude
                    denom = deref(oprep_lists[i])[orig_bi - 1]._magnitude
                    if denom == 0: denom = SMALL

                    ##mag = current_mag * (numerator / denom) if (nzeros == 0) else 0.0
                    ## fn_visitpath(b, mag, i) ##
                    if nzeros == 0:
                        addpath_fn(prps, b, i, oprep_lists, pprop1, pprop2, &e_indices, pleftSaved, prightSaved, pcoeffSaved)
                    ## --------------------------

                    if i != n-1:
                        # if we're not incrementing (from a zero-order term) the final index, then we
                        # need to to increment it until we hit num_elabels (*all* zero-th order paths)
                        orig_bn = b[n-1]
                        for k in range(1,num_elabels):
                            b[n-1] = k
                            #mag2 = mag * (deref(oprep_lists[n-1])[b[n-1]]._magnitude / deref(oprep_lists[i])[orig_bn]._magnitude)

                            ## fn_visitpath(b, mag2, n-1) ##
                            if nzeros == 0:
                                addpath_fn(prps, b, n-1, oprep_lists, pprop1, pprop2, &e_indices, pleftSaved, prightSaved, pcoeffSaved)
                            ## --------------------------
                        b[n-1] = orig_bn
            b[i] = orig_bi
        b[i] -= 1 # so we don't have to copy b


#HACK - need a way to add up magnitudes based on *current* coeffs (evaluated polynomials of terms at *current* paramvec) while
# using the locked in magnitudes to determine how many paths to actually include.  Currently, this is done by only
# "refreshing" the .magnitudes of the terms and leaving the .logmagnitudes (which are used to determing which paths to
# include) at their "locked in" values.  Thus, it is not always true that log10(term.magnitude) == term.logmagnitude.  This
# seems non-intuitive and could be problematic if it isn't at least clarified in the FUTURE.
cdef bool count_paths(vector[INT]& b, vector[vector_TermCRep_ptr_ptr]& oprep_lists,
                      vector[vector_INT_ptr]& foat_indices_per_op, INT num_elabels,
                      vector[INT]& nops, vector[INT]& e_indices, vector[double]& pathmags, vector[INT]& nPaths,
                      INT incd, double log_thres, double current_mag, double current_logmag, INT order, INT max_npaths,
                      INT current_nzeros):
    """ first_order means only one b[i] is incremented, e.g. b == [0 1 0] or [4 0 0] """
    cdef INT i, j, k, orig_bi, orig_bn
    cdef INT n = b.size()
    cdef INT sub_order
    cdef double mag, mag2
    cdef double numerator, denom

    for i in range(n-1, incd-1, -1):
        if b[i]+1 == nops[i]: continue
        b[i] += 1

        if order == 0: # then incd doesn't matter b/c can inc anything to become 1st order
            sub_order = 1 if (i != n-1 or b[i] >= num_elabels) else 0
        elif order == 1:
            # we started with a first order term where incd was incremented, and now
            # we're incrementing something else
            sub_order = 1 if i == incd else 2 # signifies anything over 1st order where >1 column has be inc'd
        else:
            sub_order = order

        logmag = current_logmag + (deref(oprep_lists[i])[b[i]]._logmagnitude - deref(oprep_lists[i])[b[i]-1]._logmagnitude)
        if logmag >= log_thres:
            numerator = deref(oprep_lists[i])[b[i]]._magnitude
            denom = deref(oprep_lists[i])[b[i]-1]._magnitude
            nzeros = current_nzeros
            if denom == 0:
                # Note: adjust logmag because when term's mag == 0, it's logmag == 0 also (convention)
                denom = SMALL; nzeros -= 1; logmag  = logmag - LOGSMALL
            if numerator == 0:
                numerator = SMALL; nzeros += 1; logmag = logmag + LOGSMALL
            mag = current_mag * (numerator / denom)

            ## fn_visitpath(b, mag, i) ##
            if nzeros == 0:
                pathmags[e_indices[b[n-1]]] += mag
            nPaths[e_indices[b[n-1]]] += 1
            if nPaths[e_indices[b[n-1]]] == max_npaths: return True
            #print("Adding ",b)
            ## --------------------------

            #add any allowed paths beneath this one
            if count_paths(b, oprep_lists, foat_indices_per_op, num_elabels, nops,
                           e_indices, pathmags, nPaths, i, log_thres, mag, logmag, sub_order, max_npaths, nzeros):
                return True

        elif sub_order <= 1:
            #We've rejected term-index b[i] (in column i) because it's too small - the only reason
            # to accept b[i] or term indices higher than it is to include "foat" terms, so we now
            # iterate through any remaining foat indices for this column (we've accepted all lower
            # values of b[i], or we wouldn't be here).  Note that we just need to visit the path,
            # we don't need to traverse down, since we know the path magnitude is already too low.
            orig_bi = b[i]
            for j in deref(foat_indices_per_op[i]):
                if j >= orig_bi:
                    b[i] = j
                    nzeros = current_nzeros
                    numerator = deref(oprep_lists[i])[b[i]]._magnitude
                    denom = deref(oprep_lists[i])[orig_bi-1]._magnitude
                    if denom == 0: denom = SMALL
                    #if numerator == 0: nzeros += 1  # not needed b/c we just leave numerator = 0
                    mag = current_mag * (numerator / denom)  # OK if mag == 0 as it's not passed to any recursive calls

                    ## fn_visitpath(b, mag, i) ##
                    if nzeros == 0:
                        pathmags[e_indices[b[n-1]]] += mag
                    nPaths[e_indices[b[n-1]]] += 1
                    if nPaths[e_indices[b[n-1]]] == max_npaths: return True
                    #print("FOAT Adding ",b)
                    ## --------------------------

                    if i != n-1:
                        # if we're not incrementing (from a zero-order term) the final index, then we
                        # need to to increment it until we hit num_elabels (*all* zero-th order paths)
                        orig_bn = b[n-1]
                        for k in range(1,num_elabels):
                            b[n-1] = k
                            numerator = deref(oprep_lists[n-1])[b[n-1]]._magnitude
                            denom = deref(oprep_lists[i])[orig_bn]._magnitude
                            if denom == 0: denom = SMALL

                            mag2 = mag * (numerator / denom)

                            ## fn_visitpath(b, mag2, n-1) ##
                            if nzeros == 0:  # if numerator was zero above, mag2 will be zero, so we still won't add anyting (good)
                                pathmags[e_indices[b[n-1]]] += mag2
                            nPaths[e_indices[b[n-1]]] += 1
                            if nPaths[e_indices[b[n-1]]] == max_npaths: return True
                            #print("FOAT Adding ",b)
                            ## --------------------------
                        b[n-1] = orig_bn
            b[i] = orig_bi
        b[i] -= 1 # so we don't have to copy b
    return False


cdef void count_paths_upto_threshold(vector[vector_TermCRep_ptr_ptr] oprep_lists, double pathmag_threshold, INT num_elabels,
                                     vector[vector_INT_ptr] foat_indices_per_op, vector[INT]& e_indices, INT max_npaths,
                                     vector[double]& pathmags, vector[INT]& nPaths):
    """ TODO: docstring """
    cdef INT i
    cdef INT n = oprep_lists.size()
    cdef vector[INT] nops = vector[INT](n)
    cdef vector[INT] b = vector[INT](n)
    cdef double log_thres = log10(pathmag_threshold)
    cdef double current_mag = 1.0
    cdef double current_logmag = 0.0

    for i in range(n):
        nops[i] = oprep_lists[i].size()
        b[i] = 0

    ## fn_visitpath(b, current_mag, 0) # visit root (all 0s) path
    pathmags[e_indices[0]] += current_mag
    nPaths[e_indices[0]] += 1
    #print("Adding ",b)
    ## -------------------------------
    count_paths(b, oprep_lists, foat_indices_per_op, num_elabels, nops, e_indices, pathmags, nPaths,
                0, log_thres, current_mag, current_logmag, 0, max_npaths, 0)
    return


cdef double pathmagnitude_threshold(vector[vector_TermCRep_ptr_ptr] oprep_lists, vector[INT]& e_indices,
                                    INT nEffects, vector[double] target_sum_of_pathmags,
                                    vector[vector_INT_ptr] foat_indices_per_op,
                                    double initial_threshold, double min_threshold, INT max_npaths,
                                    vector[double]& mags, vector[INT]& nPaths):
    """
    TODO: docstring - note: target_sum_of_pathmags is a *vector* that holds a separate value for each E-index
    """
    cdef INT nIters = 0
    cdef double threshold = initial_threshold if (initial_threshold >= 0) else 0.1 # default value
    #target_mag = target_sum_of_pathmags
    cdef double threshold_upper_bound = 1.0
    cdef double threshold_lower_bound = -1.0
    cdef INT i, j
    cdef INT try_larger_threshold

    while nIters < 100: # TODO: allow setting max_nIters as an arg?
        for i in range(nEffects):
            mags[i] = 0.0; nPaths[i] = 0
        count_paths_upto_threshold(oprep_lists, threshold, nEffects,
                                   foat_indices_per_op, e_indices, max_npaths,
                                   mags, nPaths)

        try_larger_threshold = 1 # True
        for i in range(nEffects):
            #if(mags[i] > target_sum_of_pathmags[i]): #DEBUG CHECK
            #    print "MAGS TOO LARGE!!! mags=",mags[i]," target_sum=",target_sum_of_pathmags[i]

            if(mags[i] < target_sum_of_pathmags[i]):
                try_larger_threshold = 0 # False

                #Check that max_npaths has not been reached - if so, *still* try a larger threshold
                for j in range(nEffects):
                    if nPaths[j] >= max_npaths:
                        try_larger_threshold = 1 # True
                        break
                break

        if try_larger_threshold:
            threshold_lower_bound = threshold
            if threshold_upper_bound >= 0: # ~(is not None)
                threshold = (threshold_upper_bound + threshold_lower_bound)/2
            else: threshold *= 2
        else: # try smaller threshold
            threshold_upper_bound = threshold
            if threshold_lower_bound >= 0: # ~(is not None)
                threshold = (threshold_upper_bound + threshold_lower_bound)/2
            else: threshold /= 2

        #print("  Interval: threshold in [%s,%s]: %s %s" % (str(threshold_upper_bound),str(threshold_lower_bound),mag,nPaths))
        if threshold_upper_bound >= 0 and threshold_lower_bound >= 0 and \
           (threshold_upper_bound - threshold_lower_bound)/threshold_upper_bound < 1e-3:
            #print("Converged after %d iters!" % nIters)
            break
        if threshold_upper_bound < min_threshold: # could also just set min_threshold to be the lower bound initially?
            threshold_upper_bound = threshold_lower_bound = min_threshold
            #print("Hit min threshold after %d iters!" % nIters)
            break

        nIters += 1

    #Run path traversal once more to count final number of paths
    for i in range(nEffects):
        mags[i] = 0.0; nPaths[i] = 0
    count_paths_upto_threshold(oprep_lists, threshold_lower_bound, nEffects,
                               foat_indices_per_op, e_indices, 1000000000, mags, nPaths) # sets mags and nPaths
    # 1000000000 == NO_LIMIT; we want to test that the threshold above limits the number of
    # paths to (approximately) max_npaths -- it's ok if the count is slightly higher since additional paths
    # may be needed to ensure all equal-weight paths are considered together (needed for the resulting prob to be *real*).

    return threshold_lower_bound

def circuit_achieved_and_max_sopm(fwdsim, rholabel, elabels, circuit, repcache, threshold, min_term_mag):
    """ TODO: docstring """

    #Same beginning as prs_as_pruned_polynomials -- should consolidate this setup code elsewhere

    #t0 = pytime.time()
    #if debug is not None:
    #    debug['tstartup'] += pytime.time()-t0
    #    t0 = pytime.time()

    cdef INT i, j
    cdef INT numEs = len(elabels)
    cdef INT mpv = fwdsim.model.num_params # max_polynomial_vars
    cdef CircuitSetupCacheEl cscel;
    circuitsetup_cache = {} # for now...

    if circuit in circuitsetup_cache:
        cscel = <CircuitSetupCacheEl?>circuitsetup_cache[circuit]
    else:
        cscel = <CircuitSetupCacheEl?>create_circuitsetup_cacheel(fwdsim, rholabel, elabels, circuit, repcache, min_term_mag, mpv)
        circuitsetup_cache[circuit] = cscel

    #Get MAX-SOPM for circuit outcomes and thereby the target SOPM (via MAX - gap)
    cdef double max_partial_sopm = fwdsim.model._circuit_layer_operator(rholabel, 'prep').total_term_magnitude
    cdef vector[double] max_sum_of_pathmags = vector[double](numEs)
    for glbl in circuit:
        op = fwdsim.model._circuit_layer_operator(glbl, 'op')
        max_partial_sopm *= op.total_term_magnitude
    for i,elbl in enumerate(elabels):
        max_sum_of_pathmags[i] = max_partial_sopm * fwdsim.model._circuit_layer_operator(elbl, 'povm').total_term_magnitude

    #Note: term calculator "dim" is the full density matrix dim
    stateDim = int(round(np.sqrt(fwdsim.model.dim)))

    #------ From prs_pruned ---- build up factor_lists and foat_indices_per_op
    cdef INT N = cscel.cgatestring.size()
    cdef INT nFactorLists = N+2
    cdef vector[vector_TermCRep_ptr_ptr] factor_lists = vector[vector_TermCRep_ptr_ptr](nFactorLists)
    cdef vector[vector_INT_ptr] foat_indices_per_op = vector[vector_INT_ptr](nFactorLists)

    factor_lists[0] = &cscel.rho_term_reps
    foat_indices_per_op[0] = &cscel.rho_foat_indices
    for i in range(N):
        factor_lists[i+1] = &cscel.op_term_reps[cscel.cgatestring[i]]
        foat_indices_per_op[i+1] = &cscel.op_foat_indices[cscel.cgatestring[i]]
    factor_lists[N+1] = &cscel.E_term_reps
    foat_indices_per_op[N+1] = &cscel.E_foat_indices
    # --------------------------------------------

    # Specific path magnitude summing (and we count paths, even though this isn't needed)
    cdef INT NO_LIMIT = 1000000000
    cdef vector[double] mags = vector[double](numEs)
    cdef vector[INT] npaths = vector[INT](numEs)

    for i in range(numEs):
        mags[i] = 0.0; npaths[i] = 0

    count_paths_upto_threshold(factor_lists, threshold, numEs,
                               foat_indices_per_op, cscel.e_indices, NO_LIMIT,
                               mags, npaths)

    ##DEBUG TODO REMOVE
    #print("Getting GAP for: ", circuit)
    #print("Threshold = ",threshold)
    #print("Mags = ",mags)
    #print("npaths = ",npaths)
    #print("MAX sopm = ",max_sum_of_pathmags)

    achieved_sopm = np.empty(numEs,'d')
    max_sopm = np.empty(numEs,'d')
    for i in range(numEs):
        achieved_sopm[i] = mags[i]
        max_sopm[i] = max_sum_of_pathmags[i]

    return achieved_sopm, max_sopm




# State-vector direct-term calcs -------------------------

#cdef vector[vector[TermDirectCRep_ptr]] extract_cterms_direct(python_termrep_lists, INT max_order):
#    cdef vector[vector[TermDirectCRep_ptr]] ret = vector[vector[TermDirectCRep_ptr]](max_order+1)
#    cdef vector[TermDirectCRep*] vec_of_terms
#    for order,termreps in enumerate(python_termrep_lists): # maxorder+1 lists
#        vec_of_terms = vector[TermDirectCRep_ptr](len(termreps))
#        for i,termrep in enumerate(termreps):
#            vec_of_terms[i] = (<TermDirectRep?>termrep).c_term
#        ret[order] = vec_of_terms
#    return ret

#def prs_directly(calc, rholabel, elabels, circuit, repcache, comm=None, mem_limit=None, fastmode=True, wt_tol=0.0, reset_term_weights=True, debug=None):
#
#    # Create gatelable -> int mapping to be used throughout
#    distinct_gateLabels = sorted(set(circuit))
#    glmap = { gl: i for i,gl in enumerate(distinct_gateLabels) }
#    t0 = pytime.time()
#
#    # Convert circuit to a vector of ints
#    cdef INT i, j
#    cdef vector[INT] cgatestring
#    for gl in circuit:
#        cgatestring.push_back(<INT>glmap[gl])
#
#    #TODO: maybe compute these weights elsewhere and pass in?
#    cdef double circuitWeight
#    cdef double remaingingWeightTol = <double?>wt_tol
#    cdef vector[double] remainingWeight = vector[double](<INT>len(elabels))
#    if 'circuitWeights' not in repcache:
#        repcache['circuitWeights'] = {}
#    if reset_term_weights or circuit not in repcache['circuitWeights']:
#        circuitWeight = calc.sos.get_prep(rholabel).total_term_weight()
#        for gl in circuit:
#            circuitWeight *= calc.sos.get_operation(gl).total_term_weight()
#        for i,elbl in enumerate(elabels):
#            remainingWeight[i] = circuitWeight * calc.sos.get_effect(elbl).total_term_weight()
#        repcache['circuitWeights'][circuit] = [ remainingWeight[i] for i in range(remainingWeight.size()) ]
#    else:
#        for i,wt in enumerate(repcache['circuitWeights'][circuit]):
#            assert(wt > 1.0)
#            remainingWeight[i] = wt
#
#    #if reset_term_weights:
#    #    print "Remaining weights: "
#    #    for i in range(remainingWeight.size()):
#    #        print remainingWeight[i]
#
#    cdef double order_base = 0.1 # default for now - TODO: make this a calc param like max_order?
#    cdef INT order
#    cdef INT numEs = len(elabels)
#
#    cdef RepCacheEl repcel;
#    cdef vector[TermDirectCRep_ptr] treps;
#    cdef DCOMPLEX* coeffs;
#    cdef vector[TermDirectCRep*] reps_at_order;
#    cdef np.ndarray coeffs_array;
#    cdef TermDirectRep rep;
#
#    # Construct dict of gate term reps, then *convert* to c-reps, as this
#    #  keeps alive the non-c-reps which keep the c-reps from being deallocated...
#    cdef unordered_map[INT, vector[vector[TermDirectCRep_ptr]] ] op_term_reps = unordered_map[INT, vector[vector[TermDirectCRep_ptr]] ](); # OLD = {}
#    for glbl in distinct_gateLabels:
#        if glbl in repcache:
#            repcel = <RepCacheEl?>repcache[glbl]
#            op_term_reps[ glmap[glbl] ] = repcel.reps
#            for order in range(calc.max_order+1):
#                treps = repcel.reps[order]
#                coeffs_array = calc.sos.operation(glbl).get_direct_order_coeffs(order,order_base)
#                coeffs = <DCOMPLEX*?>(coeffs_array.data)
#                for i in range(treps.size()):
#                    treps[i]._coeff = coeffs[i]
#                    if reset_term_weights: treps[i]._magnitude = abs(coeffs[i])
#            #for order,treps in enumerate(op_term_reps[ glmap[glbl] ]):
#            #    for coeff,trep in zip(calc.sos.operation(glbl).get_direct_order_coeffs(order,order_base), treps):
#            #        trep.set_coeff(coeff)
#        else:
#            repcel = RepCacheEl(calc.max_order)
#            for order in range(calc.max_order+1):
#                reps_at_order = vector[TermDirectCRep_ptr](0)
#                for t in calc.sos.operation(glbl).get_direct_order_terms(order,order_base):
#                    rep = (<TermDirectRep?>t.torep(None,None,"gate"))
#                    repcel.pyterm_references.append(rep)
#                    reps_at_order.push_back( rep.c_term )
#                repcel.reps[order] = reps_at_order
#            #OLD
#            #reps = [ [t.torep(None,None,"gate") for t in calc.sos.operation(glbl).get_direct_order_terms(order,order_base)]
#            #                                for order in range(calc.max_order+1) ]
#            op_term_reps[ glmap[glbl] ] = repcel.reps
#            repcache[glbl] = repcel
#
#    #OLD
#    #op_term_reps = { glmap[glbl]: [ [t.torep(None,None,"gate") for t in calc.sos.operation(glbl).get_direct_order_terms(order,order_base)]
#    #                                  for order in range(calc.max_order+1) ]
#    #                   for glbl in distinct_gateLabels }
#
#    #Similar with rho_terms and E_terms
#    cdef vector[vector[TermDirectCRep_ptr]] rho_term_reps;
#    if rholabel in repcache:
#        repcel = repcache[rholabel]
#        rho_term_reps = repcel.reps
#        for order in range(calc.max_order+1):
#            treps = rho_term_reps[order]
#            coeffs_array = calc.sos.prep(rholabel).get_direct_order_coeffs(order,order_base)
#            coeffs = <DCOMPLEX*?>(coeffs_array.data)
#            for i in range(treps.size()):
#                treps[i]._coeff = coeffs[i]
#                if reset_term_weights: treps[i]._magnitude = abs(coeffs[i])
#
#        #for order,treps in enumerate(rho_term_reps):
#        #    for coeff,trep in zip(calc.sos.prep(rholabel).get_direct_order_coeffs(order,order_base), treps):
#        #        trep.set_coeff(coeff)
#    else:
#        repcel = RepCacheEl(calc.max_order)
#        for order in range(calc.max_order+1):
#            reps_at_order = vector[TermDirectCRep_ptr](0)
#            for t in calc.sos.prep(rholabel).get_direct_order_terms(order,order_base):
#                rep = (<TermDirectRep?>t.torep(None,None,"prep"))
#                repcel.pyterm_references.append(rep)
#                reps_at_order.push_back( rep.c_term )
#            repcel.reps[order] = reps_at_order
#        rho_term_reps = repcel.reps
#        repcache[rholabel] = repcel
#
#        #OLD
#        #rho_term_reps = [ [t.torep(None,None,"prep") for t in calc.sos.prep(rholabel).get_direct_order_terms(order,order_base)]
#        #              for order in range(calc.max_order+1) ]
#        #repcache[rholabel] = rho_term_reps
#
#    #E_term_reps = []
#    cdef vector[vector[TermDirectCRep_ptr]] E_term_reps = vector[vector[TermDirectCRep_ptr]](0);
#    cdef TermDirectCRep_ptr cterm;
#    e_indices = [] # TODO: upgrade to C-type?
#    if all([ elbl in repcache for elbl in elabels]):
#        for order in range(calc.max_order+1):
#            reps_at_order = vector[TermDirectCRep_ptr](0) # the term reps for *all* the effect vectors
#            cur_indices = [] # the Evec-index corresponding to each term rep
#            for j,elbl in enumerate(elabels):
#                repcel = <RepCacheEl?>repcache[elbl]
#                #term_reps = [t.torep(None,None,"effect") for t in calc.sos.effect(elbl).get_direct_order_terms(order,order_base) ]
#
#                treps = repcel.reps[order]
#                coeffs_array = calc.sos.effect(elbl).get_direct_order_coeffs(order,order_base)
#                coeffs = <DCOMPLEX*?>(coeffs_array.data)
#                for i in range(treps.size()):
#                    treps[i]._coeff = coeffs[i]
#                    if reset_term_weights: treps[i]._magnitude = abs(coeffs[i])
#                    reps_at_order.push_back(treps[i])
#                cur_indices.extend( [j]*reps_at_order.size() )
#
#                #OLD
#                #term_reps = repcache[elbl][order]
#                #for coeff,trep in zip(calc.sos.effect(elbl).get_direct_order_coeffs(order,order_base), term_reps):
#                #    trep.set_coeff(coeff)
#                #cur_term_reps.extend( term_reps )
#                # cur_indices.extend( [j]*len(term_reps) )
#
#            E_term_reps.push_back(reps_at_order)
#            e_indices.append( cur_indices )
#            # E_term_reps.append( cur_term_reps )
#
#    else:
#        for elbl in elabels:
#            if elbl not in repcache: repcache[elbl] = RepCacheEl(calc.max_order) #[None]*(calc.max_order+1) # make sure there's room
#        for order in range(calc.max_order+1):
#            reps_at_order = vector[TermDirectCRep_ptr](0) # the term reps for *all* the effect vectors
#            cur_indices = [] # the Evec-index corresponding to each term rep
#            for j,elbl in enumerate(elabels):
#                repcel = <RepCacheEl?>repcache[elbl]
#                treps = vector[TermDirectCRep_ptr](0) # the term reps for *all* the effect vectors
#                for t in calc.sos.effect(elbl).get_direct_order_terms(order,order_base):
#                    rep = (<TermDirectRep?>t.torep(None,None,"effect"))
#                    repcel.pyterm_references.append(rep)
#                    treps.push_back( rep.c_term )
#                    reps_at_order.push_back( rep.c_term )
#                repcel.reps[order] = treps
#                cur_indices.extend( [j]*treps.size() )
#                #term_reps = [t.torep(None,None,"effect") for t in calc.sos.effect(elbl).get_direct_order_terms(order,order_base) ]
#                #repcache[elbl][order] = term_reps
#                #cur_term_reps.extend( term_reps )
#                #cur_indices.extend( [j]*len(term_reps) )
#            E_term_reps.push_back(reps_at_order)
#            e_indices.append( cur_indices )
#            #E_term_reps.append( cur_term_reps )
#
#    #convert to c-reps
#    cdef INT gi
#    #cdef vector[vector[TermDirectCRep_ptr]] rho_term_creps = rho_term_reps # already c-reps...
#    #cdef vector[vector[TermDirectCRep_ptr]] E_term_creps = E_term_reps # already c-reps...
#    #cdef unordered_map[INT, vector[vector[TermDirectCRep_ptr]]] gate_term_creps = op_term_reps # already c-reps...
#    #cdef vector[vector[TermDirectCRep_ptr]] rho_term_creps = extract_cterms_direct(rho_term_reps,calc.max_order)
#    #cdef vector[vector[TermDirectCRep_ptr]] E_term_creps = extract_cterms_direct(E_term_reps,calc.max_order)
#    #for gi,termrep_lists in op_term_reps.items():
#    #    gate_term_creps[gi] = extract_cterms_direct(termrep_lists,calc.max_order)
#
#    E_cindices = vector[vector[INT]](<INT>len(e_indices))
#    for ii,inds in enumerate(e_indices):
#        E_cindices[ii] = vector[INT](<INT>len(inds))
#        for jj,indx in enumerate(inds):
#            E_cindices[ii][jj] = <INT>indx
#
#    #Note: term calculator "dim" is the full density matrix dim
#    stateDim = int(round(np.sqrt(calc.dim)))
#    if debug is not None:
#        debug['tstartup'] += pytime.time()-t0
#        t0 = pytime.time()
#
#    #Call C-only function (which operates with C-representations only)
#    cdef vector[float] debugvec = vector[float](10)
#    debugvec[0] = 0.0
#    cdef vector[DCOMPLEX] prs = prs_directly(
#        cgatestring, rho_term_reps, op_term_reps, E_term_reps,
#        #cgatestring, rho_term_creps, gate_term_creps, E_term_creps,
#        E_cindices, numEs, calc.max_order, stateDim, <bool>fastmode, &remainingWeight, remaingingWeightTol, debugvec)
#
#    debug['total'] += debugvec[0]
#    debug['t1'] += debugvec[1]
#    debug['t2'] += debugvec[2]
#    debug['t3'] += debugvec[3]
#    debug['n1'] += debugvec[4]
#    debug['n2'] += debugvec[5]
#    debug['n3'] += debugvec[6]
#    debug['t4'] += debugvec[7]
#    debug['n4'] += debugvec[8]
#    #if not all([ abs(prs[i].imag) < 1e-4 for i in range(<INT>prs.size()) ]):
#    #    print("ERROR: prs = ",[ prs[i] for i in range(<INT>prs.size()) ])
#    #assert(all([ abs(prs[i].imag) < 1e-6 for i in range(<INT>prs.size()) ]))
#    return [ prs[i].real for i in range(<INT>prs.size()) ] # TODO: make this into a numpy array? - maybe pass array to fill to prs_directy above?
#
#
#cdef vector[DCOMPLEX] prs_directly(
#    vector[INT]& circuit, vector[vector[TermDirectCRep_ptr]] rho_term_reps,
#    unordered_map[INT, vector[vector[TermDirectCRep_ptr]]] op_term_reps,
#    vector[vector[TermDirectCRep_ptr]] E_term_reps, vector[vector[INT]] E_term_indices,
#    INT numEs, INT max_order, INT dim, bool fastmode, vector[double]* remainingWeight, double remTol, vector[float]& debugvec):
#
#    #NOTE: circuit and gate_terms use *integers* as operation labels, not Label objects, to speed
#    # lookups and avoid weird string conversion stuff with Cython
#
#    cdef INT N = len(circuit)
#    cdef INT* p = <INT*>malloc((N+2) * sizeof(INT))
#    cdef INT i,j,k,order,nTerms
#    cdef INT gn
#
#    cdef INT t0 = time.clock()
#    cdef INT t, n, nPaths; #for below
#
#    cdef innerloopfn_direct_ptr innerloop_fn;
#    if fastmode:
#        innerloop_fn = pr_directly_innerloop_savepartials
#    else:
#        innerloop_fn = pr_directly_innerloop
#
#    #extract raw data from gate_terms dictionary-of-lists for faster lookup
#    #gate_term_prefactors = np.empty( (nOperations,max_order+1,dim,dim)
#    #cdef unordered_map[INT, vector[vector[unordered_map[INT, complex]]]] gate_term_coeffs
#    #cdef vector[vector[unordered_map[INT, complex]]] rho_term_coeffs
#    #cdef vector[vector[unordered_map[INT, complex]]] E_term_coeffs
#    #cdef vector[vector[INT]] e_indices
#
#    cdef vector[INT]* Einds
#    cdef vector[vector_TermDirectCRep_ptr_ptr] factor_lists
#
#    assert(max_order <= 2) # only support this partitioning below (so far)
#
#    cdef vector[DCOMPLEX] prs = vector[DCOMPLEX](numEs)
#
#    for order in range(max_order+1):
#        #print("DB: pr_as_polynomial order=",order)
#
#        #for p in partition_into(order, N):
#        for i in range(N+2): p[i] = 0 # clear p
#        factor_lists = vector[vector_TermDirectCRep_ptr_ptr](N+2)
#
#        if order == 0:
#            #inner loop(p)
#            #factor_lists = [ gate_terms[glbl][pi] for glbl,pi in zip(circuit,p) ]
#            t = time.clock()
#            factor_lists[0] = &rho_term_reps[p[0]]
#            for k in range(N):
#                gn = circuit[k]
#                factor_lists[k+1] = &op_term_reps[circuit[k]][p[k+1]]
#                #if factor_lists[k+1].size() == 0: continue # WHAT???
#            factor_lists[N+1] = &E_term_reps[p[N+1]]
#            Einds = &E_term_indices[p[N+1]]
#
#            #print("Part0 ",p)
#            nPaths = innerloop_fn(factor_lists,Einds,&prs,dim,remainingWeight,0.0) #remTol) # force 0-order
#            debugvec[1] += float(time.clock() - t)/time.CLOCKS_PER_SEC
#            debugvec[4] += nPaths
#
#        elif order == 1:
#            t = time.clock(); n=0
#            for i in range(N+2):
#                p[i] = 1
#                #inner loop(p)
#                factor_lists[0] = &rho_term_reps[p[0]]
#                for k in range(N):
#                    gn = circuit[k]
#                    factor_lists[k+1] = &op_term_reps[gn][p[k+1]]
#                    #if len(factor_lists[k+1]) == 0: continue #WHAT???
#                factor_lists[N+1] = &E_term_reps[p[N+1]]
#                Einds = &E_term_indices[p[N+1]]
#
#                #print "DB: Order1 "
#                nPaths = innerloop_fn(factor_lists,Einds,&prs,dim,remainingWeight,0.0) #remTol) # force 1st-order
#                p[i] = 0
#                n += nPaths
#            debugvec[2] += float(time.clock() - t)/time.CLOCKS_PER_SEC
#            debugvec[5] += n
#
#        elif order == 2:
#            t = time.clock(); n=0
#            for i in range(N+2):
#                p[i] = 2
#                #inner loop(p)
#                factor_lists[0] = &rho_term_reps[p[0]]
#                for k in range(N):
#                    gn = circuit[k]
#                    factor_lists[k+1] = &op_term_reps[circuit[k]][p[k+1]]
#                    #if len(factor_lists[k+1]) == 0: continue # WHAT???
#                factor_lists[N+1] = &E_term_reps[p[N+1]]
#                Einds = &E_term_indices[p[N+1]]
#
#                nPaths = innerloop_fn(factor_lists,Einds,&prs,dim,remainingWeight,remTol)
#                p[i] = 0
#                n += nPaths
#
#            debugvec[3] += float(time.clock() - t)/time.CLOCKS_PER_SEC
#            debugvec[6] += n
#            t = time.clock(); n=0
#
#            for i in range(N+2):
#                p[i] = 1
#                for j in range(i+1,N+2):
#                    p[j] = 1
#                    #inner loop(p)
#                    factor_lists[0] = &rho_term_reps[p[0]]
#                    for k in range(N):
#                        gn = circuit[k]
#                        factor_lists[k+1] = &op_term_reps[circuit[k]][p[k+1]]
#                        #if len(factor_lists[k+1]) == 0: continue #WHAT???
#                    factor_lists[N+1] = &E_term_reps[p[N+1]]
#                    Einds = &E_term_indices[p[N+1]]
#
#                    nPaths = innerloop_fn(factor_lists,Einds,&prs,dim,remainingWeight,remTol)
#                    p[j] = 0
#                    n += nPaths
#                p[i] = 0
#            debugvec[7] += float(time.clock() - t)/time.CLOCKS_PER_SEC
#            debugvec[8] += n
#
#        else:
#            assert(False) # order > 2 not implemented yet...
#
#    free(p)
#
#    debugvec[0] += float(time.clock() - t0)/time.CLOCKS_PER_SEC
#    return prs
#
#
#
#cdef INT pr_directly_innerloop(vector[vector_TermDirectCRep_ptr_ptr] factor_lists, vector[INT]* Einds,
#                                   vector[DCOMPLEX]* prs, INT dim, vector[double]* remainingWeight, double remainingWeightTol):
#    #print("DB partition = ","listlens = ",[len(fl) for fl in factor_lists])
#
#    cdef INT i,j,Ei
#    cdef double complex scale, val, newval, pLeft, pRight, p
#    cdef double wt, cwt
#    cdef int nPaths = 0
#
#    cdef TermDirectCRep* factor
#
#    cdef INT nFactorLists = factor_lists.size() # may need to recompute this after fast-mode
#    cdef INT* factorListLens = <INT*>malloc(nFactorLists * sizeof(INT))
#    cdef INT last_index = nFactorLists-1
#
#    for i in range(nFactorLists):
#        factorListLens[i] = factor_lists[i].size()
#        if factorListLens[i] == 0:
#            free(factorListLens)
#            return 0 # nothing to loop over! - (exit before we allocate more)
#
#    cdef double complex coeff   # THESE are only real changes from "as_polynomial"
#    cdef double complex result  # version of this function (where they are PolynomialCRep type)
#
#    cdef StateCRep *prop1 = new StateCRep(dim)
#    cdef StateCRep *prop2 = new StateCRep(dim)
#    cdef StateCRep *tprop
#    cdef EffectCRep* EVec
#
#    cdef INT* b = <INT*>malloc(nFactorLists * sizeof(INT))
#    for i in range(nFactorLists): b[i] = 0
#
#    assert(nFactorLists > 0), "Number of factor lists must be > 0!"
#
#    #for factors in _itertools.product(*factor_lists):
#    while(True):
#        final_factor_indx = b[last_index]
#        Ei = deref(Einds)[final_factor_indx] #final "factor" index == E-vector index
#        wt = deref(remainingWeight)[Ei]
#        if remainingWeightTol == 0.0 or wt > remainingWeightTol: #if we need this "path"
#            # In this loop, b holds "current" indices into factor_lists
#            factor = deref(factor_lists[0])[b[0]] # the last factor (an Evec)
#            coeff = factor._coeff
#            cwt = factor._magnitude
#
#            for i in range(1,nFactorLists):
#                coeff *= deref(factor_lists[i])[b[i]]._coeff
#                cwt *= deref(factor_lists[i])[b[i]]._magnitude
#
#            #pLeft / "pre" sim
#            factor = deref(factor_lists[0])[b[0]] # 0th-factor = rhoVec
#            prop1.copy_from(factor._pre_state)
#            for j in range(<INT>factor._pre_ops.size()):
#                factor._pre_ops[j].acton(prop1,prop2)
#                tprop = prop1; prop1 = prop2; prop2 = tprop
#            for i in range(1,last_index):
#                factor = deref(factor_lists[i])[b[i]]
#                for j in range(<INT>factor._pre_ops.size()):
#                    factor._pre_ops[j].acton(prop1,prop2)
#                    tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
#            factor = deref(factor_lists[last_index])[b[last_index]] # the last factor (an Evec)
#
#        	# can't propagate effects, so effect's post_ops are constructed to act on *state*
#            EVec = factor._post_effect
#            for j in range(<INT>factor._post_ops.size()):
#                rhoVec = factor._post_ops[j].acton(prop1,prop2)
#                tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
#            pLeft = EVec.amplitude(prop1)
#
#            #pRight / "post" sim
#            factor = deref(factor_lists[0])[b[0]] # 0th-factor = rhoVec
#            prop1.copy_from(factor._post_state)
#            for j in range(<INT>factor._post_ops.size()):
#                factor._post_ops[j].acton(prop1,prop2)
#                tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
#            for i in range(1,last_index):
#                factor = deref(factor_lists[i])[b[i]]
#                for j in range(<INT>factor._post_ops.size()):
#                    factor._post_ops[j].acton(prop1,prop2)
#                    tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
#            factor = deref(factor_lists[last_index])[b[last_index]] # the last factor (an Evec)
#
#            EVec = factor._pre_effect
#            for j in range(<INT>factor._pre_ops.size()):
#                factor._pre_ops[j].acton(prop1,prop2)
#                tprop = prop1; prop1 = prop2; prop2 = tprop # final state in prop1
#            pRight = EVec.amplitude(prop1).conjugate()
#
#            #Add result to appropriate polynomial
#            result = coeff * pLeft * pRight
#            deref(prs)[Ei] = deref(prs)[Ei] + result #TODO - see why += doesn't work here
#            deref(remainingWeight)[Ei] = wt - cwt # "weight" of this path
#            nPaths += 1 # just for debuggins
#
#        #increment b ~ itertools.product & update vec_index_noop = np.dot(self.multipliers, b)
#        for i in range(nFactorLists-1,-1,-1):
#            if b[i]+1 < factorListLens[i]:
#                b[i] += 1
#                break
#            else:
#                b[i] = 0
#        else:
#            break # can't increment anything - break while(True) loop
#
#    #Clenaup: free allocated memory
#    del prop1
#    del prop2
#    free(factorListLens)
#    free(b)
#    return nPaths
#
#
#cdef INT pr_directly_innerloop_savepartials(vector[vector_TermDirectCRep_ptr_ptr] factor_lists,
#                                                vector[INT]* Einds, vector[DCOMPLEX]* prs, INT dim,
#                                                vector[double]* remainingWeight, double remainingWeightTol):
#    #print("DB partition = ","listlens = ",[len(fl) for fl in factor_lists])
#
#    cdef INT i,j,Ei
#    cdef double complex scale, val, newval, pLeft, pRight, p
#
#    cdef INT incd
#    cdef TermDirectCRep* factor
#
#    cdef INT nFactorLists = factor_lists.size() # may need to recompute this after fast-mode
#    cdef INT* factorListLens = <INT*>malloc(nFactorLists * sizeof(INT))
#    cdef INT last_index = nFactorLists-1
#
#    for i in range(nFactorLists):
#        factorListLens[i] = factor_lists[i].size()
#        if factorListLens[i] == 0:
#            free(factorListLens)
#            return 0 # nothing to loop over! (exit before we allocate anything else)
#
#    cdef double complex coeff
#    cdef double complex result
#
#    #fast mode
#    cdef vector[StateCRep*] leftSaved = vector[StateCRep_ptr](nFactorLists-1)  # saved[i] is state after i-th
#    cdef vector[StateCRep*] rightSaved = vector[StateCRep_ptr](nFactorLists-1) # factor has been applied
#    cdef vector[DCOMPLEX] coeffSaved = vector[DCOMPLEX](nFactorLists-1)
#    cdef StateCRep *shelved = new StateCRep(dim)
#    cdef StateCRep *prop2 = new StateCRep(dim) # prop2 is always a temporary allocated state not owned by anything else
#    cdef StateCRep *prop1
#    cdef StateCRep *tprop
#    cdef EffectCRep* EVec
#
#    cdef INT* b = <INT*>malloc(nFactorLists * sizeof(INT))
#    for i in range(nFactorLists): b[i] = 0
#    assert(nFactorLists > 0), "Number of factor lists must be > 0!"
#
#    incd = 0
#
#    #Fill saved arrays with allocated states
#    for i in range(nFactorLists-1):
#        leftSaved[i] = new StateCRep(dim)
#        rightSaved[i] = new StateCRep(dim)
#
#    #for factors in _itertools.product(*factor_lists):
#    #for incd,fi in incd_product(*[range(len(l)) for l in factor_lists]):
#    while(True):
#        # In this loop, b holds "current" indices into factor_lists
#        #print "DB: iter-product BEGIN"
#
#        if incd == 0: # need to re-evaluate rho vector
#            #print "DB: re-eval at incd=0"
#            factor = deref(factor_lists[0])[b[0]]
#
#            #print "DB: re-eval left"
#            prop1 = leftSaved[0] # the final destination (prop2 is already alloc'd)
#            prop1.copy_from(factor._pre_state)
#            for j in range(<INT>factor._pre_ops.size()):
#                #print "DB: re-eval left item"
#                factor._pre_ops[j].acton(prop1,prop2)
#                tprop = prop1; prop1 = prop2; prop2 = tprop # swap prop1 <-> prop2
#            rhoVecL = prop1
#            leftSaved[0] = prop1 # final state -> saved
#            # (prop2 == the other allocated state)
#
#            #print "DB: re-eval right"
#            prop1 = rightSaved[0] # the final destination (prop2 is already alloc'd)
#            prop1.copy_from(factor._post_state)
#            for j in range(<INT>factor._post_ops.size()):
#                #print "DB: re-eval right item"
#                factor._post_ops[j].acton(prop1,prop2)
#                tprop = prop1; prop1 = prop2; prop2 = tprop # swap prop1 <-> prop2
#            rhoVecR = prop1
#            rightSaved[0] = prop1 # final state -> saved
#            # (prop2 == the other allocated state)
#
#            #print "DB: re-eval coeff"
#            coeff = factor._coeff
#            coeffSaved[0] = coeff
#            incd += 1
#        else:
#            #print "DB: init from incd"
#            rhoVecL = leftSaved[incd-1]
#            rhoVecR = rightSaved[incd-1]
#            coeff = coeffSaved[incd-1]
#
#        # propagate left and right states, saving as we go
#        for i in range(incd,last_index):
#            #print "DB: propagate left begin"
#            factor = deref(factor_lists[i])[b[i]]
#            prop1 = leftSaved[i] # destination
#            prop1.copy_from(rhoVecL) #starting state
#            for j in range(<INT>factor._pre_ops.size()):
#                #print "DB: propagate left item"
#                factor._pre_ops[j].acton(prop1,prop2)
#                tprop = prop1; prop1 = prop2; prop2 = tprop
#            rhoVecL = prop1
#            leftSaved[i] = prop1
#            # (prop2 == the other allocated state)
#
#            #print "DB: propagate right begin"
#            prop1 = rightSaved[i] # destination
#            prop1.copy_from(rhoVecR) #starting state
#            for j in range(<INT>factor._post_ops.size()):
#                #print "DB: propagate right item"
#                factor._post_ops[j].acton(prop1,prop2)
#                tprop = prop1; prop1 = prop2; prop2 = tprop
#            rhoVecR = prop1
#            rightSaved[i] = prop1
#            # (prop2 == the other allocated state)
#
#            #print "DB: propagate coeff mult"
#            coeff *= factor._coeff
#            coeffSaved[i] = coeff
#
#        # for the last index, no need to save, and need to construct
#        # and apply effect vector
#        prop1 = shelved # so now prop1 (and prop2) are alloc'd states
#
#        #print "DB: left ampl"
#        factor = deref(factor_lists[last_index])[b[last_index]] # the last factor (an Evec)
#        EVec = factor._post_effect
#        prop1.copy_from(rhoVecL) # initial state (prop2 already alloc'd)
#        for j in range(<INT>factor._post_ops.size()):
#            factor._post_ops[j].acton(prop1,prop2)
#            tprop = prop1; prop1 = prop2; prop2 = tprop
#        pLeft = EVec.amplitude(prop1) # output in prop1, so this is final amplitude
#
#        #print "DB: right ampl"
#        EVec = factor._pre_effect
#        prop1.copy_from(rhoVecR)
#        for j in range(<INT>factor._pre_ops.size()):
#            factor._pre_ops[j].acton(prop1,prop2)
#            tprop = prop1; prop1 = prop2; prop2 = tprop
#        pRight = EVec.amplitude(prop1).conjugate()
#
#        shelved = prop1 # return prop1 to the "shelf" since we'll use prop1 for other things next
#
#        #print "DB: final block"
#        #print "DB running coeff = ",dict(coeff._coeffs)
#        #print "DB factor coeff = ",dict(factor._coeff._coeffs)
#        result = coeff * factor._coeff
#        #print "DB result = ",dict(result._coeffs)
#        result *= pLeft * pRight
#        final_factor_indx = b[last_index]
#        Ei = deref(Einds)[final_factor_indx] #final "factor" index == E-vector index
#        deref(prs)[Ei] += result
#        #print "DB prs[",INT(Ei),"] = ",dict(deref(prs)[Ei]._coeffs)
#
#        #assert(debug < 100) #DEBUG
#        #print "DB: end product loop"
#
#        #increment b ~ itertools.product & update vec_index_noop = np.dot(self.multipliers, b)
#        for i in range(nFactorLists-1,-1,-1):
#            if b[i]+1 < factorListLens[i]:
#                b[i] += 1; incd = i
#                break
#            else:
#                b[i] = 0
#        else:
#            break # can't increment anything - break while(True) loop
#
#    #Cleanup: free allocated memory
#    for i in range(nFactorLists-1):
#        del leftSaved[i]
#        del rightSaved[i]
#    del prop2
#    del shelved
#    free(factorListLens)
#    free(b)
#    return 0 #TODO: fix nPaths

