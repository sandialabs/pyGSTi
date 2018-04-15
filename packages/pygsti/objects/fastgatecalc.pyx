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

def fast_prs_as_polys(gatestring, rho, Es, gate_terms, int max_order):
    #print("DB: pr_as_poly for ",str(tuple(map(str,gatestring))), " max_order=",self.max_order)

    cdef double complex *pLeft = <double complex*>malloc(len(Es) * sizeof(double complex))
    cdef double complex *pRight = <double complex*>malloc(len(Es) * sizeof(double complex))
    cdef int N = len(gatestring)
    cdef int numEs = len(Es)
    cdef int* p = <int*>malloc(N * sizeof(int))
    cdef int i,j,k,order,nTerms
    cdef int max_poly_order=-1, max_poly_vars=-1
    cdef string gn

    #extract raw data from gate_terms dictionary-of-lists for faster lookup
    #gate_term_prefactors = _np.empty( (nGates,max_order+1,dim,dim)
    cdef unordered_map[string, vector[vector[unordered_map[int, complex]]]] gate_term_coeffs
    for gl in gate_terms.keys():
        gn = gl.name.encode('UTF-8')
        gate_term_coeffs[gn] = vector[vector[unordered_map[int, complex] ]](max_order+1)
        for order in range(max_order+1):
            nTerms = len(gate_terms[gl][order])
            gate_term_coeffs[gn][order] = vector[unordered_map[int, complex]](nTerms)
            for i,term in enumerate(gate_terms[gl][order]):
                #gate_term_coeffs[gn][order][i] = fastpoly_to_unorderedmap(term.coeff)
                polymap = unordered_map[int, complex]()
                poly = term.coeff
                if max_poly_order == -1: max_poly_order = poly.max_order
                else: assert(max_poly_order == poly.max_order)
                if max_poly_vars == -1: max_poly_vars = poly.max_num_vars
                else: assert(max_poly_vars == poly.max_num_vars)
                for k,v in poly.items(): polymap[k] = v
                gate_term_coeffs[gn][order][i] = polymap

    #        gate_term_prefactors[igl][order] = vector( GateObj(term.pre_ops[0]).acton_fn for term in gate_terms[gl][order] ) # assume all terms collapsed?
    
    assert(max_order <= 2) # only support this partitioning below (so far)

    cdef vector[ unordered_map[int, complex] ] prps = vector[ unordered_map[int, complex] ](numEs)
    #prps_chk = [None]*len(Es)
    for order in range(max_order+1):
        #print("DB: pr_as_poly order=",order)
        
        #for p in partition_into(order, N):
        for i in range(N): p[i] = 0 # clear p
        factor_lists = [None]*N
        coeff_lists = vector[vector[unordered_map[int, complex]]](N)
        
        if order == 0:
            #inner loop(p)
            #factor_lists = [ gate_terms[glbl][pi] for glbl,pi in zip(gatestring,p) ]
            for k in range(N):
                gn = <string> gatestring[k].name.encode('UTF-8')
                factor_lists[k] = gate_terms[gatestring[k]][p[k]]
                coeff_lists[k] = gate_term_coeffs[gn][p[k]]
                if len(factor_lists[k]) == 0: continue
            #print("Part0 ",p)
            pr_as_poly_innerloop(rho,Es,factor_lists,coeff_lists,pLeft,pRight,max_poly_vars, max_poly_order, &prps) #, prps_chk)

            
        elif order == 1:
            for i in range(N):
                p[i] = 1
                #inner loop(p)
                for k in range(N):
                    gn =  <string> gatestring[k].name.encode('UTF-8')
                    factor_lists[k] = gate_terms[gatestring[k]][p[k]]
                    coeff_lists[k] = gate_term_coeffs[gn][p[k]]
                    if len(factor_lists[k]) == 0: continue
                #print("Part1 ",p)
                pr_as_poly_innerloop(rho,Es,factor_lists,coeff_lists,pLeft,pRight,max_poly_vars, max_poly_order, &prps) #, prps_chk)
                p[i] = 0
            
        elif order == 2:
            for i in range(N):
                p[i] = 2
                #inner loop(p)
                for k in range(N):
                    gn = <string> gatestring[k].name.encode('UTF-8')
                    factor_lists[k] = gate_terms[gatestring[k]][p[k]]
                    coeff_lists[k] = gate_term_coeffs[gn][p[k]]
                    if len(factor_lists[k]) == 0: continue
                #print("Part2a ",p)
                pr_as_poly_innerloop(rho,Es,factor_lists,coeff_lists,pLeft,pRight,max_poly_vars, max_poly_order, &prps) #, prps_chk)
                p[i] = 0

            for i in range(N):
                p[i] = 1
                for j in range(i+1,N):
                    p[j] = 1
                    #inner loop(p)
                    for k in range(N):
                        gn = <string> gatestring[k].name.encode('UTF-8')
                        factor_lists[k] = gate_terms[gatestring[k]][p[k]]
                        coeff_lists[k] = gate_term_coeffs[gn][p[k]]
                        if len(factor_lists[k]) == 0: continue
                    #print("Part2b ",p)
                    pr_as_poly_innerloop(rho,Es,factor_lists,coeff_lists,pLeft,pRight, max_poly_vars, max_poly_order, &prps) #, prps_chk)
                    p[j] = 0
                p[i] = 0
        else:
            assert(False) # order > 2 not implemented yet...

    return prps

                

cdef pr_as_poly_innerloop(rho, Es, factor_lists, factor_coeff_lists, double complex* pLeft, double complex* pRight,
                          int max_poly_vars, int max_poly_order, vector[ unordered_map[int, complex] ]* prps): #, prps_chk):
    #print("DB partition = ","listlens = ",[len(fl) for fl in factor_lists])

    cdef int i,j
    cdef int fastmode = 1 # False
    cdef unordered_map[int, complex].iterator it1, it2, itk
    cdef unordered_map[int, complex] result, coeff, coeff2, curCoeff
    cdef double complex scale, val, newval
    cdef vector[vector[unordered_map[int, complex]]] reduced_coeff_lists

    
    rhoLeft = rhoRight = rho
    if fastmode: # filter factor_lists to matrix-compose all length-1 lists

        reduced_coeff_lists = vector[vector[unordered_map[int, complex]]]()
        reduced_factor_lists = []; curTerm = None
        
        #for i,fl in enumerate(factor_lists):
        for i in range(len(factor_lists)):
            fl = factor_lists[i]
            if len(fl) == 1:
                if curTerm is None:
                    curTerm = fl[0].copy()
                    curCoeff = factor_coeff_lists[i][0] # an unordered_map
                else:
                    curTerm.compose(fl[0])
                    # curCoeff *= factor_coeff_lists[i][0]
                    result = unordered_map[int,complex]()
                    coeff2 = factor_coeff_lists[i][0]
                    it1 = curCoeff.begin()
                    while it1 != curCoeff.end():
                        it2 = coeff2.begin()
                        while it2 != coeff2.end():
                            k = mult_vinds_ints(deref(it1).first, deref(it2).first, max_poly_vars, max_poly_order) #key to add
                            itk = result.find(k)
                            val = deref(it1).second * deref(it2).second
                            if itk != result.end():
                                deref(itk).second = deref(itk).second + val
                            else: result[k] = val
                            inc(it2)
                        inc(it1)
                    curCoeff = result
                    
            else: # len(fl) > 1:
                if curTerm is not None:
                    reduced_factor_lists.append([curTerm.collapse()])
                    vec_curCoeff = vector[unordered_map[int,complex]](1)
                    vec_curCoeff[0] = curCoeff
                    reduced_coeff_lists.push_back(vec_curCoeff)
                reduced_factor_lists.append(fl); curTerm = None
                reduced_coeff_lists.push_back( factor_coeff_lists[i] )

        if curTerm is not None:
            reduced_factor_lists.append([curTerm.collapse()])
            vec_curCoeff = vector[unordered_map[int,complex]](1)
            vec_curCoeff[0] = curCoeff
            reduced_coeff_lists.push_back(vec_curCoeff)

        if reduced_factor_lists:
            if len(reduced_factor_lists[0]) == 1: # AND coeff == 1.0?
                t = reduced_factor_lists[0][0] # single term
                #rhoLeft = self.propagate_state(rho, t.pre_ops)
                for j in range(len(t.pre_ops)):
                    rhoLeft = t.pre_ops[j].acton(rhoLeft)
                #rhoRight = self.propagate_state(rho, t.post_ops)
                for j in range(len(t.post_ops)):
                    rhoRight = t.post_ops[j].acton(rhoRight)
                del reduced_factor_lists[0]
                reduced_coeff_lists.erase(reduced_coeff_lists.begin()) # ASSUMES coeff is 1.0 and can just be removed
    
        factor_lists = reduced_factor_lists
        factor_coeff_lists = reduced_coeff_lists
        #print("DB post fastmode listlens = ",[len(fl) for fl in factor_lists])

    cdef int nFactorLists = len(factor_lists) # may need to recompute this after fast-mode
    cdef int* factorListLens = <int*>malloc(nFactorLists * sizeof(int))

    for i in range(nFactorLists):
        factorListLens[i] = len(factor_lists[i])
    
    cdef int* b = <int*>malloc(nFactorLists * sizeof(int))
    for i in range(nFactorLists): b[i] = 0

    #for factors in _itertools.product(*factor_lists):
    while(True):
        # In this loop, b holds "current" indices into factor_lists
        
#        print "Inner loop", b

        ##coeff = _functools.reduce(lambda x,y: x.mult_poly(y), [f.coeff for f in factors])
        if nFactorLists == 0:
            #coeff = _FastPolynomial({(): 1.0}, max_poly_vars, max_poly_order)
            coeff = unordered_map[int,complex](); coeff[0] = 1.0
        else:
            coeff = factor_coeff_lists[0][b[0]] # an unordered_map (copies to new "coeff" variable)

        # CHECK POLY MATH
        #print "\n----- PRE MULT ---------"
        #coeff_check = factor_lists[0][b[0]].coeff
        #checkpolys(coeff, coeff_check)
        
        for i in range(1,nFactorLists):
            result = unordered_map[int,complex]()
            coeff2 = factor_coeff_lists[i][b[i]]

            # multiply coeff by current factor
            it1 = coeff.begin()
            while it1 != coeff.end():
                it2 = coeff2.begin()
                while it2 != coeff2.end():
                    k = mult_vinds_ints(deref(it1).first, deref(it2).first, max_poly_vars, max_poly_order) #key to add
                    
                    itk = result.find(k)
                    val = deref(it1).second * deref(it2).second
                    if itk != result.end():
                        deref(itk).second = deref(itk).second + val
                    else: result[k] = val
                    inc(it2)
                inc(it1)
            coeff = result

            #CHECK POLY MATH
            #print "\n----- MULT ---------"
            #coeff_check = coeff_check.mult_poly(factor_lists[i][b[i]].coeff) # DEBUG
            #checkpolys(coeff, coeff_check)

            
        #pLeft  = self.unitary_sim_pre(rhoLeft,Es, factors, comm, memLimit, pLeft)
        #pRight = self.unitary_sim_post(rhoRight,Es, factors, comm, memLimit, pRight) \
        #         if not self.unitary_evolution else 1
        #NOTE: no unitary_evolution == 1 support yet...
        rhoVec = rhoLeft
        for i in range(nFactorLists):
            factor = factor_lists[i][b[i]]
            for j in range(len(factor.pre_ops)):
                rhoVec = factor.pre_ops[j].acton(rhoVec)
        for i in range(len(Es)):
            pLeft[i] = np.dot(Es[i],rhoVec)

        rhoVec = rhoRight
        for i in range(nFactorLists):
            factor = factor_lists[i][b[i]]
            for j in range(len(factor.post_ops)):
                rhoVec = factor.post_ops[j].acton(rhoVec)
        for i in range(len(Es)):
            pRight[i] = np.conjugate(np.dot(Es[i],rhoVec))

        for i in range(len(Es)):
            scale = pLeft[i] * pRight[i]
            result = coeff # copies so different Es start with same coeff
            it1 = result.begin()
            while it1 != result.end():
                deref(it1).second = deref(it1).second * scale # note: *= doesn't work here (complex Cython?)
                inc(it1)

            #CHECK POLY MATH
            #res = coeff_check.mult_scalar( (pLeft[i] * pRight[i]) ) #DEBUG
            #print "\n----- Post SCALE by ",scale,"---------"
            #checkpolys(result, res)
            #if prps_chk[i] is None:  prps_chk[i] = res
            #else:                    prps_chk[i] += res #prps[i].addin(res) # assumes a Polynomial - use += for numeric types...
            
            it1 = result.begin()
            while it1 != result.end():
                k = deref(it1).first # key
                val = deref(it1).second # value
                itk = deref(prps)[i].find(k)
                if itk != deref(prps)[i].end():
                    newval = deref(itk).second + val
                    if abs(newval) > 1e-12:
                        deref(itk).second = newval # note: += doens't work here (complex Cython?)
                    else: deref(prps)[i].erase(itk)
                elif abs(val) > 1e-12:
                    deref(prps)[i][k] = val
                inc(it1)

            #CHECK POLY MATH
            #print "\n---------- PRPS Check ----------",i
            #checkpolys(deref(prps)[i], prps_chk[i])

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

    
def dot(np.ndarray[double, ndim=1] f, np.ndarray[double, ndim=1] g):
    cdef long N = f.shape[0]
    cdef float ret = 0.0
    cdef int i
    for i in range(N):
        ret += f[i]*g[i]
    return ret

