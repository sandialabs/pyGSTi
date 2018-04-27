# encoding: utf-8
# cython: profile=False
# cython: linetrace=False
# filename: fastcalc.pyx

import numpy as np
from libc.stdlib cimport malloc, free
from libcpp.algorithm cimport sort as stdsort
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp.string cimport string
from cython.operator cimport dereference as deref, preincrement as inc
from ..tools import symplectic
cimport numpy as np
cimport cython

def test_map(s):
    cdef string st = s.encode('UTF-8')
    cdef unordered_map[int, complex] my_map
    cdef unordered_map[int, complex] my_map2
    cdef unordered_map[int, complex].iterator it
    cdef vector[string] v
    v = vector[string](3)
    v[0] = st
    
    my_map[1]=3.0+2.0j
    my_map[2]=6.2
    my_map2 = my_map
    my_map2[2]=10.0
    my_map2[3]=20.0
    
    print my_map[1],my_map[2]
    print my_map2[1], my_map2[2],my_map2[3]
    print("HELLO!!!")

    #try to update map
    it = my_map.begin()
    while it != my_map.end():
        deref(it).second = 12.0+12.0j
        inc(it)

    #Print map
    it = my_map.begin()
    while it != my_map.end():
        print deref(it).first
        print deref(it).second
        inc(it)
#    for x in my_map:
#        print x.first
#        print my_map[x]

def fast_prs_as_polys(gatestring, rho_terms, gate_terms, E_terms, E_indices_py, int numEs, int max_order,
                      int stabilizer_evo):
    #NOTE: gatestring and gate_terms use *integers* as gate labels, not Label objects, to speed
    # lookups and avoid weird string conversion stuff with Cython
    
    #print("DB: pr_as_poly for ",str(tuple(map(str,gatestring))), " max_order=",self.max_order)
    

    #cdef double complex *pLeft = <double complex*>malloc(len(Es) * sizeof(double complex))
    #cdef double complex *pRight = <double complex*>malloc(len(Es) * sizeof(double complex))
    cdef int N = len(gatestring)
    cdef int* p = <int*>malloc((N+2) * sizeof(int))
    cdef int i,j,k,order,nTerms
    cdef int max_poly_order=-1, max_poly_vars=-1
    cdef int gn

    #extract raw data from gate_terms dictionary-of-lists for faster lookup
    #gate_term_prefactors = _np.empty( (nGates,max_order+1,dim,dim)
    cdef unordered_map[int, vector[vector[unordered_map[int, complex]]]] gate_term_coeffs
    cdef vector[vector[unordered_map[int, complex]]] rho_term_coeffs
    cdef vector[vector[unordered_map[int, complex]]] E_term_coeffs
    cdef vector[vector[int]] E_indices
    cdef vector[int] Einds

    for gl in gate_terms.keys():
        gn = gl
        gate_term_coeffs[gn] = extract_term_coeffs(gate_terms[gl], max_order, 
                                                   max_poly_vars, max_poly_order)
    rho_term_coeffs = extract_term_coeffs(rho_terms, max_order,
                                          max_poly_vars, max_poly_order)
    E_term_coeffs = extract_term_coeffs(E_terms, max_order,
                                        max_poly_vars, max_poly_order)

    k = len(E_indices_py)
    E_indices = vector[vector[int]](k)
    for ii,inds in enumerate(E_indices_py):
        k = len(inds)
        E_indices[ii] = vector[int](k)
        for jj,indx in enumerate(inds):
            k = indx; E_indices[ii][jj] = k

    #OLD
    #    for order in range(max_order+1):
    #        nTerms = len(gate_terms[gl][order])
    #        gate_term_coeffs[gn][order] = vector[unordered_map[int, complex]](nTerms)
    #        for i,term in enumerate(gate_terms[gl][order]):
    #            #gate_term_coeffs[gn][order][i] = fastpoly_to_unorderedmap(term.coeff)
    #            polymap = unordered_map[int, complex]()
    #            poly = term.coeff
    #            if max_poly_order == -1: max_poly_order = poly.max_order
    #            else: assert(max_poly_order == poly.max_order)
    #            if max_poly_vars == -1: max_poly_vars = poly.max_num_vars
    #            else: assert(max_poly_vars == poly.max_num_vars)
    #            for k,v in poly.items(): polymap[k] = v
    #            gate_term_coeffs[gn][order][i] = polymap

    #        gate_term_prefactors[igl][order] = vector( GateObj(term.pre_ops[0]).acton_fn for term in gate_terms[gl][order] ) # assume all terms collapsed?
    
    assert(max_order <= 2) # only support this partitioning below (so far)

    cdef vector[ unordered_map[int, complex] ] prps = vector[ unordered_map[int, complex] ](numEs)
    #prps_chk = [None]*numEs
    for order in range(max_order+1):
        #print("DB: pr_as_poly order=",order)
        
        #for p in partition_into(order, N):
        for i in range(N+2): p[i] = 0 # clear p
        factor_lists = [None]*(N+2)
        coeff_lists = vector[vector[unordered_map[int, complex]]](N+2)
        
        if order == 0:
            #inner loop(p)
            #factor_lists = [ gate_terms[glbl][pi] for glbl,pi in zip(gatestring,p) ]
            factor_lists[0] = rho_terms[p[0]]
            coeff_lists[0] = rho_term_coeffs[p[0]]
            for k in range(N):
                gn = gatestring[k]
                factor_lists[k+1] = gate_terms[gatestring[k]][p[k+1]]
                coeff_lists[k+1] = gate_term_coeffs[gn][p[k+1]]
                if len(factor_lists[k+1]) == 0: continue
            factor_lists[N+1] = E_terms[p[N+1]]
            coeff_lists[N+1] = E_term_coeffs[p[N+1]]
            Einds = E_indices[p[N+1]]
        
            #print("Part0 ",p)
            pr_as_poly_innerloop(factor_lists,coeff_lists,Einds,max_poly_vars,
                                 max_poly_order, stabilizer_evo, &prps) #, prps_chk)

            
        elif order == 1:
            for i in range(N+2):
                p[i] = 1
                #inner loop(p)
                factor_lists[0] = rho_terms[p[0]]
                coeff_lists[0] = rho_term_coeffs[p[0]]
                for k in range(N):
                    gn = gatestring[k]
                    factor_lists[k+1] = gate_terms[gatestring[k]][p[k+1]]
                    coeff_lists[k+1] = gate_term_coeffs[gn][p[k+1]]
                    if len(factor_lists[k+1]) == 0: continue
                factor_lists[N+1] = E_terms[p[N+1]]
                coeff_lists[N+1] = E_term_coeffs[p[N+1]]
                Einds = E_indices[p[N+1]]

                #print("Part1 ",p)
                pr_as_poly_innerloop(factor_lists,coeff_lists,Einds,
                                     max_poly_vars, max_poly_order,
                                     stabilizer_evo, &prps) #, prps_chk)
                p[i] = 0
            
        elif order == 2:
            for i in range(N+2):
                p[i] = 2
                #inner loop(p)
                factor_lists[0] = rho_terms[p[0]]
                coeff_lists[0] = rho_term_coeffs[p[0]]
                for k in range(N):
                    gn = gatestring[k]
                    factor_lists[k+1] = gate_terms[gatestring[k]][p[k+1]]
                    coeff_lists[k+1] = gate_term_coeffs[gn][p[k+1]]
                    if len(factor_lists[k+1]) == 0: continue
                factor_lists[N+1] = E_terms[p[N+1]]
                coeff_lists[N+1] = E_term_coeffs[p[N+1]]
                Einds = E_indices[p[N+1]]

                #print("Part2a ",p)
                pr_as_poly_innerloop(factor_lists, coeff_lists,Einds,
                                     max_poly_vars, max_poly_order,
                                     stabilizer_evo, &prps) #, prps_chk)
                p[i] = 0

            for i in range(N+2):
                p[i] = 1
                for j in range(i+1,N+2):
                    p[j] = 1
                    #inner loop(p)
                    factor_lists[0] = rho_terms[p[0]]
                    coeff_lists[0] = rho_term_coeffs[p[0]]
                    for k in range(N):
                        gn = gatestring[k]
                        factor_lists[k+1] = gate_terms[gatestring[k]][p[k+1]]
                        coeff_lists[k+1] = gate_term_coeffs[gn][p[k+1]]
                        if len(factor_lists[k+1]) == 0: continue
                    factor_lists[N+1] = E_terms[p[N+1]]
                    coeff_lists[N+1] = E_term_coeffs[p[N+1]]
                    Einds = E_indices[p[N+1]]

                    #print("Part2b ",p)
                    pr_as_poly_innerloop(factor_lists, coeff_lists, Einds,
                                         max_poly_vars, max_poly_order,
                                         stabilizer_evo, &prps) #, prps_chk)
                    p[j] = 0
                p[i] = 0
        else:
            assert(False) # order > 2 not implemented yet...

    return prps

                

cdef pr_as_poly_innerloop(factor_lists, factor_coeff_lists, vector[int]& Einds,
                          int max_poly_vars, int max_poly_order, int stabilizer_evo,
                          vector[ unordered_map[int, complex] ]* prps): #, prps_chk):
    #print("DB partition = ","listlens = ",[len(fl) for fl in factor_lists])

    cdef int i,j
    cdef int fastmode = 0 # HARDCODED - but it has been checked that non-fast-mode agrees w/fastmode
    cdef unordered_map[int, complex].iterator it1, it2, itk
    cdef unordered_map[int, complex] result, coeff, coeff2, curCoeff
    cdef double complex scale, val, newval, pLeft, pRight, p
    cdef vector[vector[unordered_map[int, complex]]] reduced_coeff_lists

    cdef int incd

    cdef int nFactorLists = len(factor_lists) # may need to recompute this after fast-mode
    cdef int* factorListLens = <int*>malloc(nFactorLists * sizeof(int))
    cdef int last_index = nFactorLists-1

    for i in range(nFactorLists):
        factorListLens[i] = len(factor_lists[i])
        if factorListLens[i] == 0: return # nothing to loop over!
    
    cdef int* b = <int*>malloc(nFactorLists * sizeof(int))
    for i in range(nFactorLists): b[i] = 0

    #DEBUG
    #if debug > 0:
    #    print "nLists = ", nFactorLists
    #    for i in range(nFactorLists):
    #        print factorListLens[i]
    assert(nFactorLists > 0), "Number of factor lists must be > 0!"
    
    if fastmode: # filter factor_lists to matrix-compose all length-1 lists

        leftSaved = [None]*(nFactorLists-1)  # saved[i] is state after i-th
        rightSaved = [None]*(nFactorLists-1) # factor has been applied
        coeffSaved = [None]*(nFactorLists-1)
        incd = 0

        #for factors in _itertools.product(*factor_lists):
        #for incd,fi in incd_product(*[range(len(l)) for l in factor_lists]):
        while(True):
            #if debug > 0: # DEBUG
            #    debug += 1
            #    print "DEBUG iter", debug, " b="
            #    for i in range(nFactorLists): print b[i]
            # In this loop, b holds "current" indices into factor_lists

            if incd == 0: # need to re-evaluate rho vector
                factor = factor_lists[0][b[0]]
                rhoVecL = factor.pre_ops[0].toarray()
                for j in range(1,len(factor.pre_ops)):
                    rhoVecL = factor.pre_ops[j].acton(rhoVecL)
                leftSaved[0] = rhoVecL

                rhoVecR = factor.post_ops[0].toarray()
                for j in range(1,len(factor.post_ops)):
                    rhoVecR = factor.post_ops[j].acton(rhoVecR)
                rightSaved[0] = rhoVecR
                
                coeff = factor_coeff_lists[0][b[0]]
                coeffSaved[0] = coeff
                incd += 1
            else:
                rhoVecL = leftSaved[incd-1]
                rhoVecR = rightSaved[incd-1]
                coeff = coeffSaved[incd-1]

            # propagate left and right states, saving as we go
            for i in range(incd,last_index):
                factor = factor_lists[i][b[i]]
                for j in range(len(factor.pre_ops)):
                    rhoVecL = factor.pre_ops[j].acton(rhoVecL)
                leftSaved[i] = rhoVecL
                            
                for j in range(len(factor.post_ops)):
                    rhoVecR = factor.post_ops[j].acton(rhoVecR)
                rightSaved[i] = rhoVecR

                coeff = mult_polys(coeff, factor_coeff_lists[i][b[i]],
                                   max_poly_vars, max_poly_order)
                coeffSaved[i] = coeff

            # for the last index, no need to save, and need to construct
            # and apply effect vector
            if stabilizer_evo == 0:
                factor = factor_lists[last_index][b[last_index]] # the last factor (an Evec)
                EVec = factor.post_ops[0].toarray() # TODO USE scratch here
                for j in range(1,len(factor.post_ops)): # evaluate effect term to arrive at final EVec
                    EVec = factor.post_ops[j].acton(EVec)
                pLeft = np.vdot(EVec,rhoVecL) # complex amplitudes, *not* real probabilities
    
                EVec = factor.pre_ops[0].toarray() # TODO USE scratch here
                for j in range(1,len(factor.pre_ops)): # evaluate effect term to arrive at final EVec
                    EVec = factor.pre_ops[j].acton(EVec)
                pRight = np.conjugate(np.vdot(EVec,rhoVecR)) # complex amplitudes, *not* real probabilities
            else: # CLIFFORD - can't propagate effects, but can act w/adjoint of post_ops in reverse order...
                factor = factor_lists[last_index][b[last_index]] # the last factor (an Evec)
                EVec = factor.post_ops[0]
                for j in range(len(factor.post_ops)-1,0,-1): # (reversed)
                    rhoVecL = factor.post_ops[j].adjoint_acton(rhoVecL)
                p = stabilizer_measurement_prob(rhoVecL, EVec.outcomes)
                pLeft = np.sqrt(p) # sqrt b/c pLeft is just *amplitude*

                EVec = factor.pre_ops[0]
                for j in range(len(factor.pre_ops)-1,0,-1): # (reversed)
                    rhoVecR = factor.pre_ops[j].adjoint_acton(rhoVecR)
                p = stabilizer_measurement_prob(rhoVecR, EVec.outcomes)
                pRight = np.sqrt(p) # sqrt b/c pRight is just *amplitude*

            result = mult_polys(coeff, factor_coeff_lists[last_index][b[last_index]],
                               max_poly_vars, max_poly_order)
            scale_poly(result, (pLeft * pRight) )
            final_factor_indx = b[last_index]
            Ei = Einds[final_factor_indx] #final "factor" index == E-vector index
            add_polys_inplace(deref(prps)[Ei], result)

            #assert(debug < 100) #DEBUG
            
            #increment b ~ itertools.product & update vec_index_noop = _np.dot(self.multipliers, b)
            for i in range(nFactorLists-1,-1,-1):
                if b[i]+1 < factorListLens[i]:
                    b[i] += 1; incd = i
                    break
                else:
                    b[i] = 0
            else:
                break # can't increment anything - break while(True) loop


    else: # "slow" mode
        #for factors in _itertools.product(*factor_lists):
        while(True):
            # In this loop, b holds "current" indices into factor_lists
            
    #        print "Inner loop", b

            #OLD - now that spams are factors to, nFactorLists should always be >= 2
            ##coeff = _functools.reduce(lambda x,y: x.mult_poly(y), [f.coeff for f in factors])
            #if nFactorLists == 0:
            #    #coeff = _FastPolynomial({(): 1.0}, max_poly_vars, max_poly_order)
            #    coeff = unordered_map[int,complex](); coeff[0] = 1.0
            #else:
            coeff = factor_coeff_lists[0][b[0]] # an unordered_map (copies to new "coeff" variable)
    
            # CHECK POLY MATH
            #print "\n----- PRE MULT ---------"
            #coeff_check = factor_lists[0][b[0]].coeff
            #checkpolys(coeff, coeff_check)
            
            for i in range(1,nFactorLists):
                coeff = mult_polys(coeff, factor_coeff_lists[i][b[i]],
                                   max_poly_vars, max_poly_order)
    
                #CHECK POLY MATH
                #print "\n----- MULT ---------"
                #coeff_check = coeff_check.mult_poly(factor_lists[i][b[i]].coeff) # DEBUG
                #checkpolys(coeff, coeff_check)
    
                
            #pLeft  = self.unitary_sim_pre(rhoLeft,Es, factors, comm, memLimit, pLeft)
            #pRight = self.unitary_sim_post(rhoRight,Es, factors, comm, memLimit, pRight) \
            #         if not self.unitary_evolution else 1
            #NOTE: no unitary_evolution == 1 support yet...

            #pLeft / "pre" sim
            factor = factor_lists[0][b[0]] # 0th-factor = rhoVec
            rhoVec = factor.pre_ops[0].toarray()
            for j in range(1,len(factor.pre_ops)):
                rhoVec = factor.pre_ops[j].acton(rhoVec)
            for i in range(1,last_index):
                factor = factor_lists[i][b[i]]
                for j in range(len(factor.pre_ops)):
                    rhoVec = factor.pre_ops[j].acton(rhoVec)
            factor = factor_lists[last_index][b[last_index]] # the last factor (an Evec)

            if stabilizer_evo == 0:
                EVec = factor.post_ops[0].toarray() # TODO USE scratch here
                for j in range(1,len(factor.post_ops)): # evaluate effect term to arrive at final EVec
                    EVec = factor.post_ops[j].acton(EVec)
                pLeft = np.vdot(EVec,rhoVec) # complex amplitudes, *not* real probabilities
            else: # CLIFFORD - can't propagate effects, but can act w/adjoint of post_ops in reverse order...
                EVec = factor.post_ops[0]
                for j in range(len(factor.post_ops)-1,0,-1): # (reversed)
                    rhoVec = factor.post_ops[j].adjoint_acton(rhoVec)
                p = stabilizer_measurement_prob(rhoVec, EVec.outcomes)
                pLeft = np.sqrt(p) # sqrt b/c pLeft is just *amplitude*

                
            #pRight / "post" sim
            factor = factor_lists[0][b[0]] # 0th-factor = rhoVec
            rhoVec = factor.post_ops[0].toarray()
            for j in range(1,len(factor.post_ops)):
                rhoVec = factor.post_ops[j].acton(rhoVec)
            for i in range(1,last_index):
                factor = factor_lists[i][b[i]]
                for j in range(len(factor.post_ops)):
                    rhoVec = factor.post_ops[j].acton(rhoVec)
            factor = factor_lists[last_index][b[last_index]] # the last factor (an Evec)

            if stabilizer_evo == 0:
                EVec = factor.pre_ops[0].toarray() # TODO USE scratch here
                for j in range(1,len(factor.pre_ops)): # evaluate effect term to arrive at final EVec
                    EVec = factor.pre_ops[j].acton(EVec)
                pRight = np.conjugate(np.vdot(EVec,rhoVec)) # complex amplitudes, *not* real probabilities
            else: # CLIFFORD - can't propagate effects, but can act w/adjoint of post_ops in reverse order...
                EVec = factor.pre_ops[0]
                for j in range(len(factor.pre_ops)-1,0,-1): # (reversed)
                    rhoVec = factor.pre_ops[j].adjoint_acton(rhoVec)
                p = stabilizer_measurement_prob(rhoVec, EVec.outcomes)
                pRight = np.sqrt(p) # sqrt b/c pRight is just *amplitude*

            #Add result to appropriate poly
            result = coeff  # use a reference?
            scale_poly(result, (pLeft * pRight) )
            final_factor_indx = b[last_index]
            Ei = Einds[final_factor_indx] #final "factor" index == E-vector index

            #CHECK POLY MATH
            #res = coeff_check.mult_scalar( (pLeft * pRight) ) #DEBUG
            #print "\n----- Post SCALE by ",(pLeft * pRight),"---------"
            #print "pLeft, pRight = ",pLeft,pRight
            #checkpolys(result, res)
            #if prps_chk[Ei] is None:  prps_chk[Ei] = res
            #else:                    prps_chk[Ei] += res 
            
            add_polys_inplace(deref(prps)[Ei], result)
            
            #CHECK POLY MATH
            #print "\n---------- PRPS Check ----------",Ei
            #checkpolys(deref(prps)[Ei], prps_chk[Ei])
    
            #print("DB: pr_as_poly     factor coeff=",coeff," pLeft=",pLeft," pRight=",pRight, "res=",res,str(type(res)))
    
            #increment b ~ itertools.product & update vec_index_noop = _np.dot(self.multipliers, b)
            for i in range(nFactorLists-1,-1,-1):
                if b[i]+1 < factorListLens[i]:
                    b[i] += 1
                    break
                else:
                    b[i] = 0
            else:
                break # can't increment anything - break while(True) loop

        #print("DB: pr_as_poly   partition=",p,"(cnt ",db_part_cnt," with ",db_nfactors," factors (cnt=",db_factor_cnt,")")

cdef unordered_map[int, complex] mult_polys(unordered_map[int, complex]& poly1,
                                            unordered_map[int, complex]& poly2,
                                            int max_poly_vars, int max_poly_order):
    cdef unordered_map[int, complex].iterator it1, it2
    cdef unordered_map[int, complex].iterator itk
    cdef unordered_map[int, complex] result = unordered_map[int,complex]()
    cdef double complex val

    it1 = poly1.begin()
    while it1 != poly1.end():
        it2 = poly2.begin()
        while it2 != poly2.end():
            k = mult_vinds_ints(deref(it1).first, deref(it2).first, max_poly_vars, max_poly_order) #key to add
            itk = result.find(k)
            val = deref(it1).second * deref(it2).second
            if itk != result.end():
                deref(itk).second = deref(itk).second + val
            else: result[k] = val
            inc(it2)
        inc(it1)
    return result


cdef void add_polys_inplace(unordered_map[int, complex]& poly1,
                            unordered_map[int, complex]& poly2):
    """ poly1 += poly2 """
    cdef unordered_map[int, complex].iterator it2, itk
    cdef double complex val, newval

    it2 = poly2.begin()
    while it2 != poly2.end():
        k = deref(it2).first # key
        val = deref(it2).second # value
        itk = poly1.find(k)
        if itk != poly1.end():
            newval = deref(itk).second + val
            if abs(newval) > 1e-12:
                deref(itk).second = newval # note: += doens't work here (complex Cython?)
            else: poly1.erase(itk)
        elif abs(val) > 1e-12:
            poly1[k] = val
        inc(it2)
    return


cdef void scale_poly(unordered_map[int, complex]& poly,
                     double complex scale):
    """" poly *= scale """
    cdef unordered_map[int, complex].iterator it
    it = poly.begin()
    while it != poly.end():
        deref(it).second = deref(it).second * scale # note: *= doesn't work here (complex Cython?)
        inc(it)
    return


cdef vinds_to_int(vector[int] vinds, int max_num_vars, int max_order):
    cdef int ret = 0
    cdef int i,m = 1
    for i in vinds: # last tuple index is most significant                                                                                                          
        ret += (i+1)*m
        m *= max_num_vars+1
    return ret

cdef int_to_vinds(int indx, int max_num_vars, int max_order):
    cdef vector[int] ret
    cdef int nxt, i
    while indx != 0:
        nxt = indx // (max_num_vars+1)
        i = indx - nxt*(max_num_vars+1)
        ret.push_back(i-1)
        indx = nxt
    stdsort(ret.begin(),ret.end())
    return ret

cdef mult_vinds_ints(int i1, int i2, int max_num_vars, int max_order):
    cdef vector[int] vinds1 = int_to_vinds(i1, max_num_vars, max_order)
    cdef vector[int] vinds2 = int_to_vinds(i2, max_num_vars, max_order)
    vinds1.insert( vinds1.end(), vinds2.begin(), vinds2.end() )
    stdsort(vinds1.begin(),vinds1.end())
    return vinds_to_int(vinds1, max_num_vars, max_order)

def checkpolys(unordered_map[int,complex] coeff, coeff_check):
    cdef int mismatch = 0
    cdef unordered_map[int,complex].iterator it = coeff.begin()
    while it != coeff.end():
        k = deref(it).first # key
        if k in coeff_check and abs(coeff_check[k]-deref(it).second) < 1e-6:
            inc(it)
        else:
            mismatch = 1; break

    print "MISMATCH = ", mismatch
    print"coeff="
    it = coeff.begin()
    while it != coeff.end():
        print deref(it); inc(it)
    print "coeff_check=",coeff_check
    #    assert(0),"Mismatch!"

cdef vector[vector[unordered_map[int, complex] ]] extract_term_coeffs(python_terms, int max_order, int& max_poly_vars, int& max_poly_order):
    ret = vector[vector[unordered_map[int, complex] ]](max_order+1)
    for order in range(max_order+1):
        nTerms = len(python_terms[order])
        ret[order] = vector[unordered_map[int, complex]](nTerms)
        for i,term in enumerate(python_terms[order]):
            #ret[order][i] = fastpoly_to_unorderedmap(term.coeff)
            polymap = unordered_map[int, complex]()
            poly = term.coeff
            if max_poly_order == -1: (&max_poly_order)[0] = poly.max_order #reference assignment workaround (known Cython bug)
            else: assert(max_poly_order == poly.max_order)
            if max_poly_vars == -1: (&max_poly_vars)[0] = poly.max_num_vars
            else: assert(max_poly_vars == poly.max_num_vars)
            for k,v in poly.items(): polymap[k] = v
            ret[order][i] = polymap
    return ret

cdef double stabilizer_measurement_prob(state_sp_tuple, moutcomes):
    #Note: an abridged version from what is in GateCalc... (no qubit_filter or return_state)
    #TODO: make this routine faster - port pauli_z_measurement to C?
    cdef float p = 1.0
    state_s, state_p = state_sp_tuple
    for i,outcm in enumerate(moutcomes): 
        p0,p1,ss0,ss1,sp0,sp1 = symplectic.pauli_z_measurement(state_s, state_p, i)

        if outcm == 0:
            p *= p0; state_s, state_p = ss0, sp0
        else:
            p *= p1; state_s, state_p = ss1, sp1
    return p



    
def dot(np.ndarray[double, ndim=1] f, np.ndarray[double, ndim=1] g):
    cdef long N = f.shape[0]
    cdef float ret = 0.0
    cdef int i
    for i in range(N):
        ret += f[i]*g[i]
    return ret

