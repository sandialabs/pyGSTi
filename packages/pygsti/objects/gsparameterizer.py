#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation              
#    This Software is released under the GPL license detailed    
#    in the file "license.txt" in the top-level pyGSTi directory 
#*****************************************************************
""" Defines gate set parameterizer classes """

class GateSetParameterizer(object):
    """ 
    Base class specifying the "parameterizer" interface for GateSets.

    A parameterizer object is held by each GateSet, and the GateSet uses
    it to translate between "gateset parameters" and raw gate matrices & 
    SPAM vectors.  Thus, a parameterizer object abstracts away the details
    of particular ways of parameterizing a set of gates and SPAM operations.

    As such, a parameterizer object typically holds Gate and SPAMVec objects,
    to which it can defer some of the gate-set-parameterization details to.
    However, it is important to note that there can exist parameters "global"
    to the entire gate set that do not fall within any particular gate or
    SPAM operation (e.g. explicitly gauge-transforming parameters - which
    include gate-set basis changes), and these belong directly to the
    parameterizer object.
    """

    def __init__(self):
        """
        Create a new parameterizer object.
        """
        raise NotImplementedError("Should be implemented by derived class")

    def get_gate_matrix(self, gateLabel):
        """
        Build and return the raw gate matrix for the given gate label.
        
        Parameters
        ----------
        gateLabel : string
            The gate label
            
        Returns
        -------
        numpy array
            The 2-dimensional gate matrix.
        """
        raise NotImplementedError("Should be implemented by derived class")

    def set_gate_matrix(self, gateLabel, mx):
        """
        Attempts to modify gate set parameters so that the specified raw
        gate matrix becomes mx.  Will raise ValueError if this operation
        is not possible.
        
        Parameters
        ----------
        gateLabel : string
            The gate label.

        mx : numpy array
            Desired raw gate matrix.
            
        Returns
        -------
        numpy array
            The 2-dimensional gate matrix.
        """
        raise NotImplementedError("Should be implemented by derived class")


    def from_vector(self, v):
        """
        Set the gate set parameters using an array of values.

        Parameters
        ----------
        v : numpy array
            A 1D array of parameter values.

        Returns
        -------
        None
        """
        raise NotImplementedError("Should be implemented by derived class")


    def to_vector(self):
        """
        Get the gate set parameters as an array of value.

        Returns
        -------
        numpy array
            The gate set parameters as a 1D array.
        """
        raise NotImplementedError("Should be implemented by derived class")


    def deriv_wrt_params(self):
        """
        Construct a matrix of derivatives whose columns correspond
        to gate set parameters and whose rows correspond to elements
        of the raw gate matrices and raw SPAM vectors.

        Each column is the length of a vectorizing the raw gateset elements
        and there are num_params(...) columns.  If the gateset is fully 
        parameterized (i.e. gate-set-parameters <==> gate-set-elements) then
        the resulting matrix will be the (square) identity matrix.

        Returns
        -------
        numpy array
        """
        raise NotImplementedError("Should be implemented by derived class")



#Note: perhaps contain a "basis" string here too - to specify what 
# basis ("std", "gm", "pp") the gates should be interpreted as being in??
class StandardParameterizer(GateSetParameterizer):
    """ 
    Parameterizes gates and SPAM ops as generic Gate and SPAMOp objects,
    and contains addition of "global" gauge parameters.
    """

    def __init__(self, gates=True,G0=True,SPAM=True,SP0=True):
        """
        Create a new parameterizer object.
        """
        self.gate_dim = None  # dimension of gate matrices (each Gate 
                              # should be a gateDim x gateDim matrix)
        self.set_parameterization(gates, G0, SPAM, SP0)

    def set_parameterization(self, gates=True, G0=True, SPAM=True, SP0=True):
        self.default_gates = gates
        self.default_G0 = G0
        self.default_SPAM = SPAM
        self.default_SP0 = SP0

    def get_dimension(self):
        return self.gate_dim

    def set_vector(self, vectorType, vec, index=None):

        #Dimension check
        if self.gate_dim is None:
            self.gate_dim = len(vec)
        elif self.gate_dim != len(vec):
            raise ValueError("Cannot add vector with dimension %d to gateset of dimension %d" \
                                 % (len(identityVec),self.gate_dim))

        #Create the vector object
        if isinstance(vec, ParameterizedSPAMVec):
            vecObj = vec
        elif not self.default_SPAM:
            vecObj = NotParameterizedSPAMVec(vec)
        elif vectorType == "rho":
            if self.default_SP0:
                vecObj = FullyParameterizedSPAMVec(vec)
            else:
                vecObj = TPParameterizedSPAMVec(vec)
        elif vectorType == "E":
            vecObj = FullyParameterizedSPAMVec(vec)
        elif vectorType == "identity":
            vecObj = NotParameterizedSPAMVec(vec)
        else:
            raise ValueError("Invalid vector type: %s" % vectorType)


        #Store the vector object
        if vectorType == "rho":
            if index < len(self.rhoVecs):
                self.rhoVecs[index] = vecObj
            elif index == len(self.rhoVecs):
                self.rhoVecs.append(vecObj)
            else: raise ValueError("Cannot set rhoVec %d -- index must be <= %d" % (index, len(self.rhoVecs)))

        elif vectorType == "E":
            if index < len(self.EVecs):
                self.EVecs[index] = vecObj
            elif index == len(self.EVecs):
                self.EVecs.append(vecObj)
            else: raise ValueError("Cannot set EVec %d -- index must be <= %d" % (index, len(self.EVecs)))

        elif vectorType == "identity":
            assert(index is None) # shouldn't be setting index for identity vector
            self.identityVec = vectorObj
        
    def set_identityvec(self, identityVec):
        self.set_vector("identity", identityVec)

    def set_rhovec(self, rhoVec, index=0):
        self.set_vector("rho", rhoVec, index)

    def set_evec(self, eVec, index=0):
        self.set_vector("E", eVec, index)

    def set_gate(self, label, gate):
        """
        Set the Gate object associated with a given label.
    
        Parameters
        ----------
        label : string
            the gate label.
    
        gate : Gate
            the gate object, which must have the dimension of the GateSet.
        """

        def get_gate_dim(M):
            if isinstance(M, ParameterizedGate): return M.dim
            try:
                d1 = len(M)
                d2 = len(M[0])
            except:
                raise ValueError("%s doesn't look like a 2D array/list" % M)
            if any([len(row) != d1 for row in M]):
                raise ValueError("%s is not a *square* 2D array" % M)
            return d1

        #Dimension check
        gate_dim = get_gate_dim(gate)
        if self.gate_dim is None:
            self.gate_dim = gate_dim
        elif self.gate_dim != gate_dim:
            raise ValueError("Cannot add gate with dimension %d to gateset of dimension %d" \
                                 % (gate_dim,self.gate_dim))

        #Create the gate object
        if isinstance(gate, ParameterizedGate):
            gateObj = gate
        if not self.default_gates:
            gateObj = NotParameterizedGate(gate)
        elif self.default_G0:
            gateObj = FullyParameterizedGate(gate)
        else:
            gateObj = TPParameterizedGate(gate)

        #Store the gate object
        self.gates[label] = gateObj



    def num_params(self):
        """
        Parameters
        ----------
        gates : bool or list, optional
          Whether/which gate matrices should be vectorized (i.e. included
          as gateset parameters).

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of gate matrices should be vectorized
          (i.e. included as gateset parameters).

        SPAM : bool, optional
          Whether the rhoVecs and EVecs should be vectorized
          (i.e. included as gateset parameters).

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be vectorized (i.e. included as gateset parameters).

        """
        gates; bSPAM = SPAM; bG0 = G0; bSP0 = SP0

        m = 0 if bSP0 else 1
        rhoSize = [ len(rhoVec)-m for rhoVec in self.rhoVecs ]
        eSize   = [ len(EVec) for EVec in self.EVecs ]

        L = 0
        if gates == True:     gates = self.keys()
        elif gates == False:  gates = []
        for gateLabelToInclude in gates:
            L += self.get_gate(gateLabelToInclude).num_params(bG0)
        if bSPAM: L += sum(rhoSize) + sum(eSize)

        assert(L == len(self.to_vector(gates,G0,SPAM,SP0)) ) #Sanity check
        return L

    def num_nongauge_params(self):
        """
        Return the number of non-gauge parameters when vectorizing
        this gateset according to the optional parameters.

        Parameters
        ----------
        gates : bool or list, optional
          Whether/which gate matrices should be vectorized (i.e. included
          as gateset parameters).

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of gate matrices should be vectorized
          (i.e. included as gateset parameters).

        SPAM : bool, optional
          Whether the rhoVecs and EVecs should be vectorized
          (i.e. included as gateset parameters).

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be vectorized (i.e. included as gateset parameters).

        Returns
        -------
        int
            the number of non-gauge gateset parameters.
        """
        P = self.get_nongauge_projector(gates,G0,SPAM,SP0)
        return _np.linalg.matrix_rank(P, P_RANK_TOL)


    def num_gauge_params(self,gates=True,G0=True,SPAM=True,SP0=True):
        """
        Return the number of gauge parameters when vectorizing
        this gateset according to the optional parameters.

        Parameters
        ----------
        gates : bool or list, optional
          Whether/which gate matrices should be vectorized (i.e. included
          as gateset parameters).

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of gate matrices should be vectorized
          (i.e. included as gateset parameters).

        SPAM : bool, optional
          Whether the rhoVecs and EVecs should be vectorized
          (i.e. included as gateset parameters).

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be vectorized (i.e. included as gateset parameters).

        Returns
        -------
        int
            the number of gauge gateset parameters.
        """
        return self.num_params(gates,G0,SPAM,SP0) - self.num_nongauge_params(gates,G0,SPAM,SP0)



    def to_vector(self, gates=True,G0=True,SPAM=True,SP0=True):
        """
        Returns the gateset vectorized according to the optional parameters.

        Parameters
        ----------
        gates : bool or list, optional
          Whether/which gate matrices should be vectorized.

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of gate matrices should be vectorized.

        SPAM : bool, optional
          Whether the rhoVecs and EVecs should be vectorized.

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be vectorized.

        Returns
        -------
        numpy array
            The vectorized gateset parameters.
        """
        bSPAM = SPAM; bG0 = G0; bSP0 = SP0
        if len(self) == 0: return _np.array([])

        m = 0 if bSP0 else 1
        gsize = dict( [ (l,g.num_params(bG0)) for (l,g) in self.gates.iteritems() ])
        rhoSize = [ len(rhoVec)-m for rhoVec in self.rhoVecs ]
        eSize   = [ len(EVec) for EVec in self.EVecs ]

        L = 0
        if gates == True:     gates = self.keys()
        elif gates == False:  gates = []
        for gateLabelToInclude in gates:
            L += gsize[gateLabelToInclude]

        if bSPAM: L += sum(rhoSize) + sum(eSize)

        v = _np.empty( L ); off = 0

        if bSPAM:
            for (i,rhoVec) in enumerate(self.rhoVecs):
                v[off:off+rhoSize[i]] = rhoVec[m:,0]          
                off += rhoSize[i]
            for (i,EVec) in enumerate(self.EVecs):
                v[off:off+eSize[i]] = EVec[:,0]          
                off += eSize[i]

        for l in gates:
            v[off:off+gsize[l]] = self.get_gate(l).to_vector(bG0)
            off += gsize[l]

        return v

    def from_vector(self, v, gates=True,G0=True,SPAM=True,SP0=True):
        """
        The inverse of to_vector.  Loads values of gates and/or rho and E vecs from
        from a vector v according to the optional parameters. Note that neither v 
        nor the optional parameters specify what number of gates and their labels,
        and that this information must be contained in the gateset prior to calling
        from_vector.  In practice, this just means you should call the from_vector method
        of the gateset that was used to generate the vector v in the first place.

        Parameters
        ----------
        v : numpy array
           the vectorized gateset vector whose values are loaded into the present gateset.

        gates : bool or list, optional
          Whether/which gate matrices should be un-vectorized.

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of gate matrices should be un-vectorized.

        SPAM : bool, optional
          Whether the rhoVecs and EVecs should be un-vectorized.

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be un-vectorized.
        """

        bSPAM = SPAM; bG0 = G0; bSP0 = SP0

        m = 0 if bSP0 else 1
        gsize = dict( [ (l,g.num_params(bG0)) for (l,g) in self.gates.iteritems() ])
        rhoSize = [ len(rhoVec)-m for rhoVec in self.rhoVecs ]
        eSize   = [ len(EVec) for EVec in self.EVecs ]

        L = 0
        if gates == True:     gates = self.keys()
        elif gates == False:  gates = []
        for gateLabelToInclude in gates:
            L += gsize[ gateLabelToInclude ]

        if bSPAM: L += sum(rhoSize) + sum(eSize)

        assert( len(v) == L ); off = 0

        if bSPAM:
            for i in range(len(self.rhoVecs)):
                self.rhoVecs[i][m:,0] = v[off:off+rhoSize[i]]
                off += rhoSize[i]
            for i in range(len(self.EVecs)):
                self.EVecs[i][:,0] = v[off:off+eSize[i]]
                off += eSize[i]
        self.make_spams()

        for l in gates:
            gateObj = self.gates[l]
            gateObj.from_vector( v[off:off+gsize[l]], bG0)
            super(GateSet, self).__setitem__(l, gateObj.matrix)
            off += gsize[l]


    def get_vector_offsets(self, gates=True,G0=True,SPAM=True,SP0=True):
        """
        Returns the offsets of individual components in the vectorized 
        gateset according to the optional parameters.
    
        Parameters
        ----------
        gates : bool or list, optional
          Whether/which gate matrices should be vectorized.
    
          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.
    
        G0 : bool, optional
          Whether the first row of gate matrices should be vectorized.
    
        SPAM : bool, optional
          Whether the rhoVecs and EVecs should be vectorized.
    
        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be vectorized.
    
        Returns
        -------
        dict
            A dictionary whose keys are either "rho<number>", "E<number>",
            or a gate label and whose values are (start,next_start) tuples of
            integers indicating the start and end+1 indices of the component.
        """
        bSPAM = SPAM; bG0 = G0; bSP0 = SP0
    
        m = 0 if bSP0 else 1
        gsize = dict( [ (l,g.num_params(bG0)) for (l,g) in self.gates.iteritems() ])
        rhoSize = [ len(rhoVec)-m for rhoVec in self.rhoVecs ]
        eSize   = [ len(EVec) for EVec in self.EVecs ]
    
        if gates == True:     gates = self.keys()
        elif gates == False:  gates = []
    
        off = 0
        offsets = {}
    
        if bSPAM:
            for (i,rhoVec) in enumerate(self.rhoVecs):
                offsets["rho%d" % i] = (off,off+rhoSize[i])
                off += rhoSize[i]
    
            for (i,EVec) in enumerate(self.EVecs):
                offsets["E%d" % i] = (off,off+eSize[i])
                off += eSize[i]
    
        for l in gates:
            offsets[l] = (off,off+gsize[l])
            off += gsize[l]
    
        return offsets


    def deriv_wrt_params(self, gates=True,G0=True,SPAM=True,SP0=True):
        """
        Construct a matrix whose columns are the vectorized
        derivatives of the gateset when all gates are treated as
        fully parameterized with respect to a single parameter
        of the true vectorized gateset.

        Each column is the length of a vectorized gateset of
        fully parameterized gates and there are num_params(...) columns.
        If the gateset is fully parameterized (i.e. contains only
        fully parameterized gates) then the resulting matrix will be
        the (square) identity matrix.

        Parameters
        ----------
        gates : bool or list, optional
          Whether/which gate matrices should be vectorized.

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of gate matrices should be vectorized.

        SPAM : bool, optional
          Whether the rhoVecs and EVecs should be vectorized.

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be vectorized.

        Returns
        -------
        numpy array
        """

        bSPAM = SPAM; bG0 = G0; bSP0 = SP0
        if len(self) == 0: return _np.array([])

        m = 0 if bSP0 else 1
        gsize = dict( [ (l,g.num_params(bG0)) for (l,g) in self.gates.iteritems() ])
        rhoSize = [ len(rhoVec)-m for rhoVec in self.rhoVecs ]
        eSize   = [ len(EVec) for EVec in self.EVecs ]
        full_vsize = self.gate_dim
        full_gsize = self.gate_dim**2 # (flattened) size of a gate matrix

        if gates == True:     gates = self.keys()
        elif gates == False:  gates = []

        nParams = self.num_params(gates,G0,SPAM,SP0)
        nElements = self.num_elements() #total number of gate mx and spam vec elements
        deriv = _np.zeros( (nElements, nParams), 'd' )

        k = 0; foff= 0; off = 0 #independently track full-offset and (parameterized)-offset

        if bSPAM:
            for (i,rhoVec) in enumerate(self.rhoVecs):
                deriv[foff+m:foff+m+rhoSize[i],off:off+rhoSize[i]] = _np.identity( rhoSize[i], 'd' )
                off += rhoSize[i]; foff += full_vsize

            for (i,EVec) in enumerate(self.EVecs):
                deriv[foff:foff+eSize[i],off:off+eSize[i]] = _np.identity( eSize[i], 'd' )
                off += eSize[i]; foff += full_vsize
        else:
            foff += full_vsize * (len(self.rhoVecs) + len(self.EVecs))

        for l in gates:
            deriv[foff:foff+full_gsize,off:off+gsize[l]] = self.get_gate(l).deriv_wrt_params(bG0)
            off += gsize[l]; foff += full_gsize

        return deriv

    def get_nongauge_projector(self, gates=True,G0=True,SPAM=True,SP0=True):
        # We want to divide the GateSet-space H (a Hilbert space, 56-dimensional in the 1Q, 3-gate, 2-vec case)
        # into the direct sum of gauge and non-gauge spaces, and find projectors onto each
        # sub-space (the non-gauge space in particular).  
        #
        # Within the GateSet H-space lies a gauge-manifold of maximum chi2 (16-dimensional in 1Q case),
        #  where the gauge-invariant chi2 function is constant.  At a single point (GateSet) P on this manifold,
        #  chosen by (somewhat arbitrarily) fixing the gauge, we can talk about the tangent space
        #  at that point.  This tangent space is spanned by some basis (~16 elements for the 1Q case),
        #  which associate with the infinitesimal gauge transform ?generators? on the full space.
        #  The subspace of H spanned by the derivatives w.r.t gauge transformations at P (a GateSet) spans
        #  the gauge space, and the complement of this (in H) is the non-gauge space.
        #
        #  An element of the gauge group can be written gg = exp(-K), where K is a n x n matrix.  If K is
        #   hermitian then gg is unitary, but this need not be the case.  A gauge transform acts on a 
        #   gatset via Gateset => gg^-1 G gg, i.e. G => exp(K) G exp(-K).  We care about the action of
        #   infinitesimal gauge tranformations (b/c the *derivative* vectors span the tangent space), 
        #   which act as:
        #    G => (I+K) G (I-K) = G + [K,G] + ignored(K^2), where [K,G] := KG-GK
        #
        # To produce a projector onto the gauge-space, we compute the *column* vectors
        #  dG_ij = [K_ij,G], where K_ij is the i,j-th matrix unit (56x1 in the 1Q case, 16 such column vectors)
        #  and then form a projector in the standard way.
        #  (to project onto |v1>, |v2>, etc., form P = sum_i |v_i><v_i|)
        #
        #Typically nGateParams < len(dG_ij) and linear system is overconstrained
        #   and no solution is expected.  If no solution exists, simply ignore
        #
        # So we form P = sum_ij dG_ij * transpose(dG_ij) (56x56 in 1Q case)
        #              = dG * transpose(dG)              where dG_ij form the *columns* of dG (56x16 in 1Q case) 
        # But dG_ij are not orthonormal, so really we need a slight modification,
        #  otherwise P'P' != P' as must be the case for a projector:
        # 
        # P' = dG * (transpose(dG) * dG)^-1 * transpose(dG) (see wikipedia on projectors)
        #
        #    or equivalently (I think)
        #
        # P' = pseudo-inv(P)*P  
        #
        #  since the pseudo-inv is defined by P*pseudo-inv(P) = I and so P'P' == P'
        #  and P' is our gauge-projector!

        # Note: In the general case of parameterized gates (i.e. non-fully parameterized gates), there are fewer
        #   gate parameters than the size of dG_ij ( < 56 in 1Q case).  In this general case, we want to know
        #   what (if any) change in gate parameters produces the change dG_ij of the gate matrix elements.
        #   That is, we solve dG_ij = derivWRTParams * dParams_ij for dParams_ij, where derivWRTParams is 
        #   the derivative of the gate matrix elements with respect to the gate parameters (derivWRTParams 
        #   is 56x(nGateParams) and dParams_ij is (nGateParams)x1 in the 1Q case) -- then substitute dG_ij
        #   with dParams_ij above to arrive at a (nGateParams)x(nGateParams) projector (compatible with
        #   hessian computed by gateset).
        #
        #   Actually, we want to know if what changes gate parameters
        #   produce changes in the span of all the dG_ij, so really we want the intersection of the space
        #   defined by the columns of derivWRTParams (the "gate-parameter range" space) and the dG_ij vectors.
        #
        #   This intersection is determined by nullspace( derivWRTParams | dG ), where the pipe denotes
        #     concatenating the matrices together.  Then, if x is a basis vector of the nullspace
        #     x[0:nGateParams] is the basis vector for the intersection space within the gate parameter space,
        #     that is, the analogue of dParams_ij in the single-dG_ij introduction above.
        # 
        #   Still, we just substitue these dParams_ij vectors (as many as the nullspace dimension) for dG_ij
        #   above to get the general case projector.


        #Use a gateset object to hold & then vectorize the derivatives wrt each gauge transform basis element (each ij)
        #  **If G0 == False, then don't include gauge basis elements corresponding to the first row (since we assume
        #     gauge transforms will be TP constrained in this case)
        dim = self.gate_dim
        nParams = self.num_params(gates,G0,SPAM,SP0)

        #Note: gateset object (gsDeriv) must have all elements of gate mxs and spam vectors as parameters in order
        #  to match deriv_wrt_params calls, which give derivs wrt 
        gsDeriv = self.copy() 
        for gateLabel in self:
            gsDeriv.set_gate(gateLabel, _gate.FullyParameterizedGate(_np.zeros((dim,dim),'d')))
        for k,rhoVec in enumerate(gsDeriv.rhoVecs): gsDeriv.set_rhovec(_np.zeros((dim,1),'d'), k)
        for k,EVec in enumerate(gsDeriv.EVecs):     gsDeriv.set_evec(_np.zeros((dim,1),'d'), k)

        nElements = gsDeriv.num_elements()

        if gates == True: gatesToInclude = self.keys() #all gates
        elif gates == False: gatesToInclude = [] #no gates

        zMx = _np.zeros( (dim,dim), 'd')

        dG = _np.empty( (nElements, dim**2), 'd' )
        for i in xrange(dim): #Note: always range over all rows: this is *generator* mx, not gauge mx itself
            for j in xrange(dim):
                unitMx = _bt._mut(i,j,dim)
                #DEBUG: gsDeriv = self.copy() -- should delete this after debugging is done since doesn't work for parameterized gates
                if SPAM:
                    for k,rhoVec in enumerate(self.rhoVecs):
                        gsDeriv.set_rhovec( _np.dot(unitMx, rhoVec), k)
                    for k,EVec in enumerate(self.EVecs):
                        gsDeriv.set_evec( -_np.dot(EVec.T, unitMx).T, k)
                #else don't consider spam space as part of gateset-space => leave spam derivs zero
                    
                for gateLabel,gateMx in self.iteritems():
                    if gateLabel in gatesToInclude:
                        gsDeriv.get_gate(gateLabel).set_value( _np.dot(unitMx,gateMx)-_np.dot(gateMx,unitMx) )
                    #else leave gate as zeros

                #Note: vectorize *everything* in this gateset of FullyParameterizedGate
                #      objects to match the number of gateset *elements*.  (so all True's to to_vector)
                dG[:,i*dim+j] = gsDeriv.to_vector(True,True,True,True) 


        dP = self.deriv_wrt_params(gates,G0,SPAM,SP0)  #TODO maybe make this a hidden method if it's only used here...
        M = _np.concatenate( (dP,dG), axis=1 )
        
        def nullspace(m, tol=1e-7): #get the nullspace of a matrix
            u,s,vh = _np.linalg.svd(m)
            rank = (s > tol).sum()
            return vh[rank:].T.copy()

        nullsp = nullspace(M) #columns are nullspace basis vectors
        gen_dG = nullsp[0:nParams,:] #take upper (gate-param-segment) of vectors for basis
                                     # of subspace intersection in gate-parameter space
        #Note: gen_dG == "generalized dG", and is (nParams)x(nullSpaceDim==gaugeSpaceDim), so P 
        #  (below) is (nParams)x(nParams) as desired.

        #DEBUG
        #for iRow in range(nElements):
        #    pNorm = _np.linalg.norm(dP[iRow])
        #    if pNorm < 1e-6: 
        #        print "Row %d of dP is zero" % iRow
        #
        #print "------------------------------"
        #print "nParams = ",nParams
        #print "nElements = ",nElements
        #print " shape(M) = ",M.shape
        #print " shape(dG) = ",dG.shape
        #print " shape(dP) = ",dP.shape
        #print " shape(gen_dG) = ",gen_dG.shape
        #print " shape(nullsp) = ",nullsp.shape
        #print " rank(dG) = ",_np.linalg.matrix_rank(dG)
        #print " rank(dP) = ",_np.linalg.matrix_rank(dP)
        #print "------------------------------"
        #assert(_np.linalg.norm(_np.imag(gen_dG)) < 1e-9) #gen_dG is real


        # ORIG WAY: use psuedo-inverse to normalize projector.  Ran into problems where
        #  default rcond == 1e-15 didn't work for 2-qubit case, but still more stable than inv method below
        P = _np.dot(gen_dG, _np.transpose(gen_dG)) # almost a projector, but cols of dG are not orthonormal
        Pp = _np.dot( _np.linalg.pinv(P, rcond=1e-7), P ) # make P into a true projector (onto gauge space)

        # ALT WAY: use inverse of dG^T*dG to normalize projector (see wikipedia on projectors, dG => A)
        #  This *should* give the same thing as above, but numerical differences indicate the pinv method
        #  is prefereable (so long as rcond=1e-7 is ok in general...)
        #  Check: P'*P' = (dG (dGT dG)^1 dGT)(dG (dGT dG)^-1 dGT) = (dG (dGT dG)^1 dGT) = P'
        #invGG = _np.linalg.inv(_np.dot(_np.transpose(gen_dG), gen_dG))
        #Pp_alt = _np.dot(gen_dG, _np.dot(invGG, _np.transpose(gen_dG))) # a true projector (onto gauge space)
        #print "Pp - Pp_alt norm diff = ", _np.linalg.norm(Pp_alt - Pp)

        ret = _np.identity(nParams,'d') - Pp # projector onto the non-gauge space

        # Check ranks to make sure everything is consistent.  If either of these assertions fail,
        #  then pinv's rcond or some other numerical tolerances probably need adjustment.
        #print "Rank P = ",_np.linalg.matrix_rank(P)
        #print "Rank Pp = ",_np.linalg.matrix_rank(Pp, P_RANK_TOL)
        #print "Rank 1-Pp = ",_np.linalg.matrix_rank(_np.identity(nParams,'d') - Pp, P_RANK_TOL)
          #print " Evals(1-Pp) = \n","\n".join([ "%d: %g" % (i,ev) \
          #    for i,ev in enumerate(_np.sort(_np.linalg.eigvals(_np.identity(nParams,'d') - Pp))) ])

        rank_P = _np.linalg.matrix_rank(P, P_RANK_TOL) # original un-normalized projector onto gauge space
          # Note: use P_RANK_TOL here even though projector is *un-normalized* since sometimes P will
          #  have eigenvalues 1e-17 and one or two 1e-11 that should still be "zero" but aren't when
          #  no tolerance is given.  Perhaps a more custom tolerance based on the singular values of P
          #  but different from numpy's default tolerance would be appropriate here.

        assert( rank_P == _np.linalg.matrix_rank(Pp, P_RANK_TOL)) #rank shouldn't change with normalization
        assert( (nParams - rank_P) == _np.linalg.matrix_rank(ret, P_RANK_TOL) ) # dimension of orthogonal space
        return ret





    def get_nongauge_projector_ex(self, nonGaugeMixMx, gates=True,G0=True,SPAM=True,SP0=True):
        dim = self.gate_dim
        nParams = self.num_params(gates,G0,SPAM,SP0)

        #Note: gateset object (gsDeriv) must have all elements of gate mxs and spam vectors as parameters in order
        #  to match deriv_wrt_params calls, which give derivs wrt 
        gsDeriv = self.copy() 
        for gateLabel in self:
            gsDeriv.set_gate(gateLabel, _gate.FullyParameterizedGate(_np.zeros((dim,dim),'d')))
        for k,rhoVec in enumerate(gsDeriv.rhoVecs): gsDeriv.set_rhovec(_np.zeros((dim,1),'d'), k)
        for k,EVec in enumerate(gsDeriv.EVecs):     gsDeriv.set_evec(_np.zeros((dim,1),'d'), k)

        nElements = gsDeriv.num_elements()

        if gates == True: gatesToInclude = self.keys() #all gates
        elif gates == False: gatesToInclude = [] #no gates

        zMx = _np.zeros( (dim,dim), 'd')

        dG = _np.empty( (nElements, dim**2), 'd' )
        for i in xrange(dim): #Note: always range over all rows: this is *generator* mx, not gauge mx itself
            for j in xrange(dim):
                unitMx = _bt._mut(i,j,dim)
                gsDeriv = self.copy()
                if SPAM:
                    for k,rhoVec in enumerate(self.rhoVecs):
                        gsDeriv.set_rhovec( _np.dot(unitMx, rhoVec), k)
                    for k,EVec in enumerate(self.EVecs):
                        gsDeriv.set_evec( -_np.dot(EVec.T, unitMx).T, k)
                #else don't consider spam space as part of gateset-space => leave spam derivs zero
                    
                for gateLabel,gateMx in self.iteritems():
                    if gateLabel in gatesToInclude:
                        gsDeriv.get_gate(gateLabel).set_value( _np.dot(unitMx,gateMx)-_np.dot(gateMx,unitMx) )
                    #else leave gate as zeros

                #Note: vectorize *everything* in this gateset of FullyParameterizedGate
                #      objects to match the number of gateset *elements*.  (so all True's to to_vector)
                dG[:,i*dim+j] = gsDeriv.to_vector(True,True,True,True) 


        dP = self.deriv_wrt_params(gates,G0,SPAM,SP0)  #TODO maybe make this a hidden method if it's only used here...
        M = _np.concatenate( (dP,dG), axis=1 )
        
        def nullspace(m, tol=1e-7): #get the nullspace of a matrix
            u,s,vh = _np.linalg.svd(m)
            rank = (s > tol).sum()
            return vh[rank:].T.copy()

        nullsp = nullspace(M) #columns are nullspace basis vectors
        gen_dG = nullsp[0:nParams,:] #take upper (gate-param-segment) of vectors for basis
                                     # of subspace intersection in gate-parameter space
        #Note: gen_dG == "generalized dG", and is (nParams)x(nullSpaceDim==gaugeSpaceDim), so P 
        #  (below) is (nParams)x(nParams) as desired.


        #EXTRA ----------------------------------------

        # nullspace of gen_dG^T (mx with gauge direction vecs as rows) gives non-gauge directions
        nonGaugeDirections = nullspace(gen_dG.T) #columns are non-gauge directions

        #for each column of gen_dG, which is a gauge direction in gateset parameter space,
        # we add some amount of non-gauge direction, given by coefficients of the 
        # numNonGaugeParams non-gauge directions.
        gen_dG = gen_dG + _np.dot( nonGaugeDirections, nonGaugeMixMx) #add non-gauge direction components in
         # dims: (nParams,nGaugeParams) + (nParams,nNonGaugeParams) * (nNonGaugeParams,nGaugeParams)
         # nonGaugeMixMx is a (nNonGaugeParams,nGaugeParams) matrix whose i-th column specifies the 
         #  coefficents to multipy each of the non-gauge directions by before adding them to the i-th 
         #  direction to project out (i.e. what were the pure gauge directions).

        #DEBUG
        #print "gen_dG shape = ",gen_dG.shape
        #print "NGD shape = ",nonGaugeDirections.shape
        #print "NGD rank = ",_np.linalg.matrix_rank(nonGaugeDirections, P_RANK_TOL)


        #END EXTRA ----------------------------------------

        # ORIG WAY: use psuedo-inverse to normalize projector.  Ran into problems where
        #  default rcond == 1e-15 didn't work for 2-qubit case, but still more stable than inv method below
        P = _np.dot(gen_dG, _np.transpose(gen_dG)) # almost a projector, but cols of dG are not orthonormal
        Pp = _np.dot( _np.linalg.pinv(P, rcond=1e-7), P ) # make P into a true projector (onto gauge space)

        # ALT WAY: use inverse of dG^T*dG to normalize projector (see wikipedia on projectors, dG => A)
        #  This *should* give the same thing as above, but numerical differences indicate the pinv method
        #  is prefereable (so long as rcond=1e-7 is ok in general...)
        #  Check: P'*P' = (dG (dGT dG)^1 dGT)(dG (dGT dG)^-1 dGT) = (dG (dGT dG)^1 dGT) = P'
        #invGG = _np.linalg.inv(_np.dot(_np.transpose(gen_dG), gen_dG))
        #Pp_alt = _np.dot(gen_dG, _np.dot(invGG, _np.transpose(gen_dG))) # a true projector (onto gauge space)
        #print "Pp - Pp_alt norm diff = ", _np.linalg.norm(Pp_alt - Pp)

        ret = _np.identity(nParams,'d') - Pp # projector onto the non-gauge space

        # Check ranks to make sure everything is consistent.  If either of these assertions fail,
        #  then pinv's rcond or some other numerical tolerances probably need adjustment.
        #print "Rank P = ",_np.linalg.matrix_rank(P)
        #print "Rank Pp = ",_np.linalg.matrix_rank(Pp, P_RANK_TOL)
        #print "Rank 1-Pp = ",_np.linalg.matrix_rank(_np.identity(nParams,'d') - Pp, P_RANK_TOL)
          #print " Evals(1-Pp) = \n","\n".join([ "%d: %g" % (i,ev) \
          #    for i,ev in enumerate(_np.sort(_np.linalg.eigvals(_np.identity(nParams,'d') - Pp))) ])

        rank_P = _np.linalg.matrix_rank(P, P_RANK_TOL) # original un-normalized projector onto gauge space
          # Note: use P_RANK_TOL here even though projector is *un-normalized* since sometimes P will
          #  have eigenvalues 1e-17 and one or two 1e-11 that should still be "zero" but aren't when
          #  no tolerance is given.  Perhaps a more custom tolerance based on the singular values of P
          #  but different from numpy's default tolerance would be appropriate here.

        assert( rank_P == _np.linalg.matrix_rank(Pp, P_RANK_TOL)) #rank shouldn't change with normalization
        assert( (nParams - rank_P) == _np.linalg.matrix_rank(ret, P_RANK_TOL) ) # dimension of orthogonal space
        return ret


    def transform(self, S, Si=None):
        """
        Update each of the gate matrices G in this gateset with inv(S) * G * S,
        each rhoVec with inv(S) * rhoVec, and each EVec with EVec * S

        Parameters
        ----------
        S : numpy array
            Matrix to perform similarity transform.  
            Should be shape (gate_dim, gate_dim).
            
        Si : numpy array, optional
            Inverse of S.  If None, inverse of S is computed.  
            Should be shape (gate_dim, gate_dim).
        """
        if Si is None: Si = _nla.inv(S) 
        for (i,rhoVec) in enumerate(self.rhoVecs): 
            self.rhoVecs[i] = _np.dot(Si,rhoVec)
        for (i,EVec) in enumerate(self.EVecs): 
            self.EVecs[i] = _np.dot(_np.transpose(S),EVec)  # same as ( Evec^T * S )^T

        if self.identityVec is not None:
            self.identityVec = _np.dot(_np.transpose(S),self.identityVec) #same as for EVecs

        self.make_spams()

        for (l,gate) in self.gates.iteritems():
            gate.transform(Si,S)
            super(GateSet, self).__setitem__(l, gate.matrix)

    def compute_rhovec(self, index):
        pass

    def compute_evec(self, index):
        pass

    def compute_gate(self, gateLabel):
        pass

    def compute_identity_vec(self):


    def iter_rhovecs(self):
        pass

    def iter_evecs(self):
        pass

    def iter_gates(self):
        pass

    def compute_identityvec(self):
        if self.identityVec is None: return None
        return self.identityVec.construct_vector()

    
    




###########################################################################################################

    def get_gate_matrix(self, gateLabel):
        """
        Build and return the raw gate matrix for the given gate label.
        
        Parameters
        ----------
        gateLabel : string
            The gate label
            
        Returns
        -------
        numpy array
            The 2-dimensional gate matrix.
        """
        raise NotImplementedError("Should be implemented by derived class")

    def set_gate_matrix(self, gateLabel, mx):
        """
        Attempts to modify gate set parameters so that the specified raw
        gate matrix becomes mx.  Will raise ValueError if this operation
        is not possible.
        
        Parameters
        ----------
        gateLabel : string
            The gate label.

        mx : numpy array
            Desired raw gate matrix.
            
        Returns
        -------
        numpy array
            The 2-dimensional gate matrix.
        """
        raise NotImplementedError("Should be implemented by derived class")


    def from_vector(self, v):
        """
        Set the gate set parameters using an array of values.

        Parameters
        ----------
        v : numpy array
            A 1D array of parameter values.

        Returns
        -------
        None
        """
        raise NotImplementedError("Should be implemented by derived class")


    def to_vector(self):
        """
        Get the gate set parameters as an array of value.

        Returns
        -------
        numpy array
            The gate set parameters as a 1D array.
        """
        raise NotImplementedError("Should be implemented by derived class")


    def deriv_wrt_params(self):
        """
        Construct a matrix of derivatives whose columns correspond
        to gate set parameters and whose rows correspond to elements
        of the raw gate matrices and raw SPAM vectors.

        Each column is the length of a vectorizing the raw gateset elements
        and there are num_params(...) columns.  If the gateset is fully 
        parameterized (i.e. gate-set-parameters <==> gate-set-elements) then
        the resulting matrix will be the (square) identity matrix.

        Returns
        -------
        numpy array
        """
        raise NotImplementedError("Should be implemented by derived class")



class FullParameterizer(StandardParameterizer):
    """ 
    Special case of StandardParameterizer where each gate parameterizations and SPAM
    operation are fully parameterized.  Because of this, additional
    "global gauge" parameters are not needed.
    """

    def __init__(self):
        """
        Create a new parameterizer object.
        """
        raise NotImplementedError("Should be implemented by derived class")

    def get_gate_matrix(self, gateLabel):
        """
        Build and return the raw gate matrix for the given gate label.
        
        Parameters
        ----------
        gateLabel : string
            The gate label
            
        Returns
        -------
        numpy array
            The 2-dimensional gate matrix.
        """
        raise NotImplementedError("Should be implemented by derived class")

    def set_gate_matrix(self, gateLabel, mx):
        """
        Attempts to modify gate set parameters so that the specified raw
        gate matrix becomes mx.  Will raise ValueError if this operation
        is not possible.
        
        Parameters
        ----------
        gateLabel : string
            The gate label.

        mx : numpy array
            Desired raw gate matrix.
            
        Returns
        -------
        numpy array
            The 2-dimensional gate matrix.
        """
        raise NotImplementedError("Should be implemented by derived class")


    def from_vector(self, v):
        """
        Set the gate set parameters using an array of values.

        Parameters
        ----------
        v : numpy array
            A 1D array of parameter values.

        Returns
        -------
        None
        """
        raise NotImplementedError("Should be implemented by derived class")


    def to_vector(self):
        """
        Get the gate set parameters as an array of value.

        Returns
        -------
        numpy array
            The gate set parameters as a 1D array.
        """
        raise NotImplementedError("Should be implemented by derived class")


    def deriv_wrt_params(self):
        """
        Construct a matrix of derivatives whose columns correspond
        to gate set parameters and whose rows correspond to elements
        of the raw gate matrices and raw SPAM vectors.

        Each column is the length of a vectorizing the raw gateset elements
        and there are num_params(...) columns.  If the gateset is fully 
        parameterized (i.e. gate-set-parameters <==> gate-set-elements) then
        the resulting matrix will be the (square) identity matrix.

        Returns
        -------
        numpy array
        """
        raise NotImplementedError("Should be implemented by derived class")


class GaugeInvParameterizer(GateSetParameterizer):
    """ 
    Parametrizes a gate set using a minimal set of gauge invariant parameters.
    """

    def __init__(self):
        """
        Create a new parameterizer object.
        """
        raise NotImplementedError("Should be implemented by derived class")

    def get_gate_matrix(self, gateLabel):
        """
        Build and return the raw gate matrix for the given gate label.
        
        Parameters
        ----------
        gateLabel : string
            The gate label
            
        Returns
        -------
        numpy array
            The 2-dimensional gate matrix.
        """
        raise NotImplementedError("Should be implemented by derived class")

    def set_gate_matrix(self, gateLabel, mx):
        """
        Attempts to modify gate set parameters so that the specified raw
        gate matrix becomes mx.  Will raise ValueError if this operation
        is not possible.
        
        Parameters
        ----------
        gateLabel : string
            The gate label.

        mx : numpy array
            Desired raw gate matrix.
            
        Returns
        -------
        numpy array
            The 2-dimensional gate matrix.
        """
        raise NotImplementedError("Should be implemented by derived class")


    def from_vector(self, v):
        """
        Set the gate set parameters using an array of values.

        Parameters
        ----------
        v : numpy array
            A 1D array of parameter values.

        Returns
        -------
        None
        """
        raise NotImplementedError("Should be implemented by derived class")


    def to_vector(self):
        """
        Get the gate set parameters as an array of value.

        Returns
        -------
        numpy array
            The gate set parameters as a 1D array.
        """
        raise NotImplementedError("Should be implemented by derived class")


    def deriv_wrt_params(self):
        """
        Construct a matrix of derivatives whose columns correspond
        to gate set parameters and whose rows correspond to elements
        of the raw gate matrices and raw SPAM vectors.

        Each column is the length of a vectorizing the raw gateset elements
        and there are num_params(...) columns.  If the gateset is fully 
        parameterized (i.e. gate-set-parameters <==> gate-set-elements) then
        the resulting matrix will be the (square) identity matrix.

        Returns
        -------
        numpy array
        """
        raise NotImplementedError("Should be implemented by derived class")
