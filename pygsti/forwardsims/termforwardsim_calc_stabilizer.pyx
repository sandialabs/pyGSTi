

cdef class SBTermRep:
    cdef SBTermCRep* c_term

    #Hold references to other reps so we don't GC them
    cdef public PolynomialRep coeff
    cdef public SBStateRep pre_state
    cdef public SBStateRep post_state
    cdef public SBEffectRep pre_effect
    cdef public SBEffectRep post_effect
    cdef public object pre_ops
    cdef public object post_ops

    @classmethod
    def composed(cls, terms_to_compose, double magnitude):
        cdef double logmag = log10(magnitude) if magnitude > 0 else -LARGE
        cdef SBTermRep first = terms_to_compose[0]
        cdef PolynomialRep coeffrep = first.coeff
        pre_ops = first.pre_ops[:]
        post_ops = first.post_ops[:]
        for t in terms_to_compose[1:]:
            coeffrep = coeffrep.mult(t.coeff)
            pre_ops += t.pre_ops
            post_ops += t.post_ops
        return SBTermRep(coeffrep, magnitude, logmag, first.pre_state, first.post_state,
                         first.pre_effect, first.post_effect, pre_ops, post_ops)

    def __cinit__(self, PolynomialRep coeff, double mag, double logmag,
                  SBStateRep pre_state, SBStateRep post_state,
                  SBEffectRep pre_effect, SBEffectRep post_effect, pre_ops, post_ops):
        self.coeff = coeff
        self.pre_ops = pre_ops
        self.post_ops = post_ops

        cdef INT i
        cdef INT npre = len(pre_ops)
        cdef INT npost = len(post_ops)
        cdef vector[SBOpCRep*] c_pre_ops = vector[SBOpCRep_ptr](npre)
        cdef vector[SBOpCRep*] c_post_ops = vector[SBOpCRep_ptr](<INT>len(post_ops))
        for i in range(npre):
            c_pre_ops[i] = (<SBOpRep?>pre_ops[i]).c_op
        for i in range(npost):
            c_post_ops[i] = (<SBOpRep?>post_ops[i]).c_op

        if pre_state is not None or post_state is not None:
            assert(pre_state is not None and post_state is not None)
            self.pre_state = pre_state
            self.post_state = post_state
            self.pre_effect = self.post_effect = None
            self.c_term = new SBTermCRep(coeff.c_polynomial, mag, logmag,
                                         pre_state.c_state, post_state.c_state,
                                         c_pre_ops, c_post_ops);
        elif pre_effect is not None or post_effect is not None:
            assert(pre_effect is not None and post_effect is not None)
            self.pre_effect = pre_effect
            self.post_effect = post_effect
            self.pre_state = self.post_state = None
            self.c_term = new SBTermCRep(coeff.c_polynomial, mag, logmag,
                                         pre_effect.c_effect, post_effect.c_effect,
                                         c_pre_ops, c_post_ops);
        else:
            self.pre_state = self.post_state = None
            self.pre_effect = self.post_effect = None
            self.c_term = new SBTermCRep(coeff.c_polynomial, mag, logmag, c_pre_ops, c_post_ops);

    def __dealloc__(self):
        del self.c_term

    def __reduce__(self):
        return (SBTermRep, (self.coeff, self.magnitude, self.logmagnitude,
                self.pre_state, self.post_state, self.pre_effect, self.post_effect,
                self.pre_ops, self.post_ops))

    def set_magnitude(self, double mag):
        self.c_term._magnitude = mag
        self.c_term._logmagnitude = log10(mag) if mag > 0 else -LARGE

    def mapvec_indices_inplace(self, mapvec):
        self.coeff.mapvec_indices_inplace(mapvec)

    @property
    def magnitude(self):
        return self.c_term._magnitude

    @property
    def logmagnitude(self):
        return self.c_term._logmagnitude

    def scalar_mult(self, x):
        coeff = self.coeff.copy()
        coeff.scale(x)
        return SBTermRep(coeff, self.magnitude * x, self.logmagnitude + log10(x),
                         self.pre_state, self.post_state, self.pre_effect, self.post_effect,
                         self.pre_ops, self.post_ops)

    def copy(self):
        return SBTermRep(self.coeff.copy(), self.magnitude, self.logmagnitude,
                         self.pre_state, self.post_state, self.pre_effect, self.post_effect,
                         self.pre_ops, self.post_ops)

    #Not needed - and this implementation is quite right as it will need to change
    # the ordering of the pre/post ops also.
    #def conjugate(self):
    #    return SBTermRep(self.coeff.copy(), self.magnitude, self.logmagnitude,
    #                     self.post_state, self.pre_state, self.post_effect, self.pre_effect,
    #                     self.post_ops, self.pre_ops)





# Stabilizer-evolution version of polynomial term calcs -----------------------

cdef vector[vector[SBTermCRep_ptr]] sb_extract_cterms(python_termrep_lists, INT max_order):
    cdef vector[vector[SBTermCRep_ptr]] ret = vector[vector[SBTermCRep_ptr]](max_order+1)
    cdef vector[SBTermCRep*] vec_of_terms
    for order,termreps in enumerate(python_termrep_lists): # maxorder+1 lists
        vec_of_terms = vector[SBTermCRep_ptr](len(termreps))
        for i,termrep in enumerate(termreps):
            vec_of_terms[i] = (<SBTermRep?>termrep).c_term
        ret[order] = vec_of_terms
    return ret


def SB_prs_as_polynomials(fwdsim, rholabel, elabels, circuit, polynomial_vindices_per_int,
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
    cdef vector[vector[SBTermCRep_ptr]] rho_term_creps = sb_extract_cterms(rho_term_reps,fwdsim.max_order)
    cdef vector[vector[SBTermCRep_ptr]] E_term_creps = sb_extract_cterms(E_term_reps,fwdsim.max_order)
    cdef unordered_map[INT, vector[vector[SBTermCRep_ptr]]] gate_term_creps
    for gi,termrep_lists in op_term_reps.items():
        gate_term_creps[gi] = sb_extract_cterms(termrep_lists,fwdsim.max_order)

    E_cindices = vector[vector[INT]](<INT>len(e_indices))
    for ii,inds in enumerate(e_indices):
        E_cindices[ii] = vector[INT](<INT>len(inds))
        for jj,indx in enumerate(inds):
            E_cindices[ii][jj] = <INT>indx

    # Assume when we calculate terms, that "dimension" of Model is
    # a full vectorized-density-matrix dimension, so nqubits is:
    cdef INT nqubits = <INT>(np.log2(fwdsim.model.dim)//2)

    #Call C-only function (which operates with C-representations only)
    cdef vector[PolynomialCRep*] polynomials = sb_prs_as_polynomials(
        cgatestring, rho_term_creps, gate_term_creps, E_term_creps,
        E_cindices, numEs, fwdsim.max_order, mpv, vpi, nqubits, <bool>fastmode)

    return [ PolynomialRep_from_allocd_PolynomialCRep(polynomials[i]) for i in range(<INT>polynomials.size()) ]


cdef vector[PolynomialCRep*] sb_prs_as_polynomials(
    vector[INT]& circuit, vector[vector[SBTermCRep_ptr]] rho_term_reps,
    unordered_map[INT, vector[vector[SBTermCRep_ptr]]] op_term_reps,
    vector[vector[SBTermCRep_ptr]] E_term_reps, vector[vector[INT]] E_term_indices,
    INT numEs, INT max_order, INT max_polynomial_vars, INT vindices_per_int, INT nqubits, bool fastmode):

    #NOTE: circuit and gate_terms use *integers* as operation labels, not Label objects, to speed
    # lookups and avoid weird string conversion stuff with Cython

    cdef INT N = len(circuit)
    cdef INT* p = <INT*>malloc((N+2) * sizeof(INT))
    cdef INT i,j,k,order,nTerms
    cdef INT gn

    cdef sb_innerloopfn_ptr innerloop_fn;
    if fastmode:
        innerloop_fn = sb_pr_as_polynomial_innerloop_savepartials
    else:
        innerloop_fn = sb_pr_as_polynomial_innerloop

    #extract raw data from gate_terms dictionary-of-lists for faster lookup
    #gate_term_prefactors = np.empty( (nOperations,max_order+1,dim,dim)
    #cdef unordered_map[INT, vector[vector[unordered_map[INT, complex]]]] gate_term_coeffs
    #cdef vector[vector[unordered_map[INT, complex]]] rho_term_coeffs
    #cdef vector[vector[unordered_map[INT, complex]]] E_term_coeffs
    #cdef vector[vector[INT]] e_indices

    cdef vector[INT]* Einds
    cdef vector[vector_SBTermCRep_ptr_ptr] factor_lists

    assert(max_order <= 2) # only support this partitioning below (so far)

    cdef vector[PolynomialCRep_ptr] prps = vector[PolynomialCRep_ptr](numEs)
    for i in range(numEs):
        prps[i] = new PolynomialCRep(unordered_map[PolynomialVarsIndex,complex](), max_polynomial_vars, vindices_per_int)
        # create empty polynomials - maybe overload constructor for this?
        # these PolynomialCReps are alloc'd here and returned - it is the job of the caller to
        #  free them (or assign them to new PolynomialRep wrapper objs)

    for order in range(max_order+1):
        #print "DB CYTHON: pr_as_polynomial order=",INT(order)

        #for p in partition_into(order, N):
        for i in range(N+2): p[i] = 0 # clear p
        factor_lists = vector[vector_SBTermCRep_ptr_ptr](N+2)

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

            #print "DB CYTHON: Order0"
            innerloop_fn(factor_lists,Einds,&prps,nqubits) #, prps_chk)


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

                #print "DB CYTHON: Order1 "
                innerloop_fn(factor_lists,Einds,&prps,nqubits) #, prps_chk)
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

                innerloop_fn(factor_lists,Einds,&prps,nqubits) #, prps_chk)
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

                    innerloop_fn(factor_lists,Einds,&prps,nqubits) #, prps_chk)
                    p[j] = 0
                p[i] = 0
        else:
            assert(False) # order > 2 not implemented yet...

    free(p)
    return prps



cdef void sb_pr_as_polynomial_innerloop(vector[vector_SBTermCRep_ptr_ptr] factor_lists, vector[INT]* Einds,
                                  vector[PolynomialCRep*]* prps, INT n): #, prps_chk):
    #print("DB partition = ","listlens = ",[len(fl) for fl in factor_lists])

    cdef INT i,j,Ei
    cdef double complex scale, val, newval, pLeft, pRight, p

    cdef INT incd
    cdef SBTermCRep* factor

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

    cdef INT namps = 1 # HARDCODED namps for SB states - in future this may be just the *initial* number
    cdef SBStateCRep *prop1 = new SBStateCRep(namps, n)
    cdef SBStateCRep *prop2 = new SBStateCRep(namps, n)
    cdef SBStateCRep *tprop
    cdef SBEffectCRep* EVec

    cdef INT* b = <INT*>malloc(nFactorLists * sizeof(INT))
    for i in range(nFactorLists): b[i] = 0

    assert(nFactorLists > 0), "Number of factor lists must be > 0!"

    #for factors in _itertools.product(*factor_lists):
    while(True):
        # In this loop, b holds "current" indices into factor_lists
        factor = deref(factor_lists[0])[b[0]] # the last factor (an Evec)
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


cdef void sb_pr_as_polynomial_innerloop_savepartials(vector[vector_SBTermCRep_ptr_ptr] factor_lists,
                                               vector[INT]* Einds, vector[PolynomialCRep*]* prps, INT n): #, prps_chk):
    #print("DB partition = ","listlens = ",[len(fl) for fl in factor_lists])

    cdef INT i,j,Ei
    cdef double complex scale, val, newval, pLeft, pRight, p

    cdef INT incd
    cdef SBTermCRep* factor

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

    cdef INT namps = 1 # HARDCODED namps for SB states - in future this may be just the *initial* number
    cdef vector[SBStateCRep*] leftSaved = vector[SBStateCRep_ptr](nFactorLists-1)  # saved[i] is state after i-th
    cdef vector[SBStateCRep*] rightSaved = vector[SBStateCRep_ptr](nFactorLists-1) # factor has been applied
    cdef vector[PolynomialCRep] coeffSaved = vector[PolynomialCRep](nFactorLists-1)
    cdef SBStateCRep *shelved = new SBStateCRep(namps, n)
    cdef SBStateCRep *prop2 = new SBStateCRep(namps, n) # prop2 is always a temporary allocated state not owned by anything else
    cdef SBStateCRep *prop1
    cdef SBStateCRep *tprop
    cdef SBEffectCRep* EVec

    cdef INT* b = <INT*>malloc(nFactorLists * sizeof(INT))
    for i in range(nFactorLists): b[i] = 0
    assert(nFactorLists > 0), "Number of factor lists must be > 0!"

    incd = 0

    #Fill saved arrays with allocated states
    for i in range(nFactorLists-1):
        leftSaved[i] = new SBStateCRep(namps, n)
        rightSaved[i] = new SBStateCRep(namps, n)

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
            #print "DB: init from incd"
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
        for j in range(<INT>factor._pre_ops.size()):
            factor._pre_ops[j].acton(prop1,prop2)
            tprop = prop1; prop1 = prop2; prop2 = tprop
        pLeft = EVec.amplitude(prop1) # output in prop1, so this is final amplitude

        #print "DB: right ampl"
        EVec = factor._pre_effect
        prop1.copy_from(rhoVecR)
        pRight = EVec.amplitude(prop1)
        #DEBUG print "  - begin: ",complex(pRight)
        for j in range(<INT>factor._post_ops.size()):
            #DEBUG print " - state = ", [ prop1._smatrix[ii] for ii in range(2*2)]
            #DEBUG print "         = ", [ prop1._pvectors[ii] for ii in range(2)]
            #DEBUG print "         = ", [ prop1._amps[ii] for ii in range(1)]
            factor._post_ops[j].acton(prop1,prop2)
            #DEBUG print " - action with ", [ (<SBOpCRep_Clifford*>factor._pre_ops[j])._smatrix_inv[ii] for ii in range(2*2)]
            #DEBUG print " - action with ", [ (<SBOpCRep_Clifford*>factor._pre_ops[j])._svector_inv[ii] for ii in range(2)]
            #DEBUG print " - action with ", [ (<SBOpCRep_Clifford*>factor._pre_ops[j])._unitary_adj[ii] for ii in range(2*2)]
            tprop = prop1; prop1 = prop2; prop2 = tprop
            pRight = EVec.amplitude(prop1)
            #DEBUG print "  - prop ",INT(j)," = ",complex(pRight)
            #DEBUG print " - post state = ", [ prop1._smatrix[ii] for ii in range(2*2)]
            #DEBUG print "              = ", [ prop1._pvectors[ii] for ii in range(2)]
            #DEBUG print "              = ", [ prop1._amps[ii] for ii in range(1)]

        pRight = EVec.amplitude(prop1).conjugate()

        shelved = prop1 # return prop1 to the "shelf" since we'll use prop1 for other things next

        #print "DB: final block: pLeft=",complex(pLeft)," pRight=",complex(pRight)
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
