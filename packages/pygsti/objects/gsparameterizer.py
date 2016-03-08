#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation              
#    This Software is released under the GPL license detailed    
#    in the file "license.txt" in the top-level pyGSTi directory 
#*****************************************************************
""" Defines gate set parameterizer classes """

import collections as _collections
import numpy as _np
import spamvec as _sv
import gate as _g

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
        self.gate_dim = None  # dimension of gate matrices (each Gate 
                              # should be a gateDim x gateDim matrix)

    def get_dimension(self):
        """
        Get the dimension D of the gate matrices and SPAM vectors parameterized
        by this object.  Gate matrices have dimension D x D, and SPAM vectors
        have dimension D x 1.

        Returns
        -------
        int
        """
        return self.gate_dim



#Note: perhaps contain a "basis" string here too - to specify what 
# basis ("std", "gm", "pp") the gates should be interpreted as being in??
class StandardParameterizer(GateSetParameterizer):
    """ 
    Parameterizes gates and SPAM ops as independent Gate and SPAMVec objects,
    """

    def __init__(self, gates=True,G0=True,SPAM=True,SP0=True):
        """
        Create a new StandardParameterizer object.  The options specify
        "default" gate and SPAM vector creation behavior.

        Parameters
        ----------
        gates : bool or list, optional
          Whether/which gates should be parameterized by default.

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of the included gate matrices
          should be parameterized by default.

        SPAM : bool, optional
          Whether rhoVecs and EVecs should be parameterized by default.

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be parameterized by default.

        """
        self.set_default_parameterization(gates, G0, SPAM, SP0)
        self.EVecs = []
        self.rhoVecs = []
        self.identityVec = None
        self.gates = _collections.OrderedDict()
        super(StandardParameterizer, self).__init__()


    def set_default_parameterization(self, gates=True, G0=True,
                                     SPAM=True, SP0=True):
        """
        Specify "default" gate and SPAM vector creation behavior.

        Parameters
        ----------
        gates : bool or list, optional
          Whether/which gates should be parameterized by default.

          - True = all gates
          - False = no gates
          - list of gate labels = those particular gates.

        G0 : bool, optional
          Whether the first row of the included gate matrices
          should be parameterized by default.

        SPAM : bool, optional
          Whether rhoVecs and EVecs should be parameterized by default.

        SP0 : bool, optional
          Whether the first element of the state preparation (rho) vectors
          should be parameterized by default.

        """
        self.default_gates = gates
        self.default_G0 = G0
        self.default_SPAM = SPAM
        self.default_SP0 = SP0


    def _set_vector(self, vectorType, vec, index=None):

        #Dimension check
        if self.gate_dim is None:
            self.gate_dim = len(vec)
        elif self.gate_dim != len(vec):
            raise ValueError("Cannot add vector with dimension" +
                             "%d to gateset of dimension %d"
                             % (len(vec),self.gate_dim))

        if vectorType == "rho":
            vecArray = self.rhoVecs
        elif vectorType == "E":
            vecArray = self.EVecs
        elif vectorType == "identity":
            assert(index is None)
            self.identityVec = vec if isinstance(vec, _sv.StaticSPAMVec) \
                          else _sv.StaticSPAMVec(vec) # *always* static
            return
        else: raise ValueError("Invalid vectorType: %s" % vectorType)

        #Create the vector object if needed
        if isinstance(vec, _sv.SPAMVec):  #if we're given a SPAMVec object...
            vecArray[index] = vec     #just replace or create

        elif index < len(vecArray): #if a SPAMVec object already exists...
            vecArray[index].set_vector(vec) # try to set value

        else:
            #otherwise, we've been given a non-SPAMVec-object that doesn't 
            # exist yet, so use default creation flags to make one:

            if index != len(vecArray): #can only create next highest index
                raise ValueError("Cannot set %sVec %d -- index must be <= %d"
                                 % (vectorType, index, len(vecArray)))

            if not self.default_SPAM:
                vecObj = _sv.StaticSPAMVec(vec)
            elif vectorType == "rho" and not self.default_SP0:
                vecObj = _sv.TPParameterizedSPAMVec(vec)
            else:
                vecObj = _sv.FullyParameterizedSPAMVec(vec)                

            #Store the vector object
            vecArray.append(vecObj)


    def set_identityvec(self, identityVec):
        """
        Set the value of the identity vector, a non-parameterized object that
        is used to compute a "complement" POVM effect vector (by subtracting
        the sum of the other POVM effect vectors from this vector).

        Parameters
        ----------
        identityVec : array-like or StaticSPAMVec
            the value to set as the identity vector.

        Returns
        -------
        None
        """
        self._set_vector("identity", identityVec)


    def set_rhovec(self, rhoVec, index=0):
        """
        Set the value of the state preparation vector at a specified
        index.  A new vector is created by setting index to the lowest un-used
        integer.

        Parameters
        ----------
        rhoVec : array-like or SPAMVec
            the value to set.  If a SPAMVec, any existing SPAMVec object will
            be overwritten.  Otherwise, rhoVec will be used to set the vector-
            value of the existing SPAMVec when one is present.

        Returns
        -------
        None
        """
        self._set_vector("rho", rhoVec, index)


    def set_evec(self, eVec, index=0):
        """
        Set the value of the POVM effect vector at a specified index.  A new
        vector is created by setting index to the lowest un-used integer.

        Parameters
        ----------
        eVec : array-like or SPAMVec
            the value to set.  If a SPAMVec, any existing SPAMVec object will
            be overwritten.  Otherwise, rhoVec will be used to set the vector-
            value of the existing SPAMVec when one is present.

        Returns
        -------
        None
        """
        self._set_vector("E", eVec, index)


    def set_gate(self, label, gate):
        """
        Set the value of the gate associated with a given label.  A new gate
        is created by setting an un-used label.
    
        Parameters
        ----------
        label : string
            the gate label.
    
        gate : array-like or Gate
            the value to set.  If a Gate object, any existing Gate object will
            be overwritten.  Otherwise, gate will be used to set the matrix-
            value of the existing Gate object when one is present. 

        Returns
        -------
        None
        """

        def get_gate_dim(M):
            if isinstance(M, _g.Gate): return M.dim
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
            raise ValueError("Cannot add gate with dimension " +
                             "%d to gateset of dimension %d"
                             % (gate_dim,self.gate_dim))

        #Create the gate object if needed
        if isinstance(gate, _g.Gate): #if we're given a Gate object...
            self.gates[label] = gate  # just replace or create

        elif label in self.gates: #if a gate object already exists...
            self.gates[label].set_matrix(gate) # try to set value

        else: 
            #otherwise, we've been given a non-Gate-object that doesn't 
            # exist yet, so use default creation flags to make a Gate:
            if self.default_gates == False:
                gateObj = _g.StaticGate(gate)
            elif self.default_gates == True or label in self.default_gates:
                if self.default_G0:
                    gateObj = _g.FullyParameterizedGate(gate)
                else:
                    gateObj = _g.TPParameterizedGate(gate)
            else:
                gateObj = _g.StaticGate(gate)

            #Store the gate object
            self.gates[label]  = gateObj


    def compute_rhovec(self, index):
        """
        Build and return the raw state-preparation (column) vector for the
        given index.
        
        Parameters
        ----------
        index : string
            The rho-vector index.
            
        Returns
        -------
        numpy array
            The state preparation column vector.
        """
        return self.rhoVecs[index].construct_vector()

    def compute_evec(self, index):
        """
        Build and return the raw POVM effect (column) vector for the
        given index.
        
        Parameters
        ----------
        index : string
            The POVM-vector index.
            
        Returns
        -------
        numpy array
            The POVM effect column vector.
        """
        return self.EVecs[index].construct_vector()

    def compute_gate_matrix(self, gateLabel):
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
        return self.gates[gateLabel].construct_matrix()

    def compute_identity_vector(self):
        """
        Build and return the identity (column) vector for the
        given index.
                    
        Returns
        -------
        numpy array
            The identity column vector.
        """
        if self.identityVec is None: return None
        return self.identityVec.construct_vector()


    def iter_rho_vectors(self):
        """ Return an iterator over indexed rho-Vectors """
        return enumerate([rhoVec.construct_vector() for rhoVec in self.rhoVecs])

    def iter_e_vectors(self):
        return enumerate([EVec.construct_vector() for EVec in self.EVecs])

    def iter_gate_matrices(self):
        return { gl : gate.construct_matrix
                        for (gl,gate) in self.gates.iteritems() }.iteritems()
    
    def num_params(self):
        """
        Returns the number of parameters, that is, the total number of degrees
        of freedom used to construct the set of gates and SPAM vectors being
        parameterized.

        Returns
        -------
        int
            the total number of gateset parameters.
        """
        L = sum([ rhoVec.num_params() for rhoVec in self.rhoVecs ])
        L += sum([ EVec.num_params() for EVec in self.EVecs ])
        L += sum([ gate.num_params() for gate in self.gates.values()])
        assert(L == len(self.to_vector()) )
        return L


    def num_elements(self):
        """
        Returns the total number of elements in all of the gates matrices and
        SPAM vectors being parameterized.

        Returns
        -------
        int
            the total number of matrix and vector elements.
        """
        full_vsize = self.gate_dim
        full_gsize = self.gate_dim**2 # (flattened) size of a gate matrix
        return full_vsize * (len(self.rhoVecs) + len(self.EVecs)) \
                + full_gsize * len(self.gates)

    def num_nongauge_params(self):
        """
        Returns the number of non-gauge parameters.

        Returns
        -------
        int
            the number of non-gauge gateset parameters.
        """
        P = self.get_nongauge_projector()
        return _np.linalg.matrix_rank(P, P_RANK_TOL)


    def num_gauge_params(self):
        """
        Returns the number of gauge parameters.

        Returns
        -------
        int
            the number of gauge gateset parameters.
        """
        return self.num_params() - self.num_nongauge_params()


    def transform(self, S, Si=None, rhoVectors=None, EVectors=None,
                  identityVector=None, gateMxs=None):
        """
        Update the parameters such that each of the gate matrices G becomse
        inv(S) * G * S, each rhoVec becomes inv(S) * rhoVec, and each EVec
        becomes EVec * S.   If this is not possible because of parameterization
        constraints, ValueError is raised.

        Parameters
        ----------
        S : numpy array
            Matrix to perform similarity transform.  
            Should be shape (gate_dim, gate_dim).
            
        Si : numpy array, optional
            Inverse of S.  If None, inverse of S is computed.  
            Should be shape (gate_dim, gate_dim).

        rhoVectors, EVectors : lists of numpy arrays, optional
            Pre-computed (usually because cached by gateset) values of
            the raw SPAM vectors.

        identityVector : numpy array, optional
            Pre-computed (usually because cached by gateset) values of
            the identity vector.
            
        gateMxs : dict of numpy arrays, optional
            Pre-computed (usually because cached by gateset) values of
            the raw gate matrices.  Keys are gate labels.
        """

        if Si is None: Si = _nla.inv(S) 

        if rhoVectors is None or EVectors is None: 
            rhoVectors = [rhoVec.construct_vector() for rhoVec in self.rhoVecs]
            EVectors = [EVec.construct_vector() for EVec in self.EVecs]
        if gateMxs is None: 
            gateMxs = { gl : gate.construct_matrix
                        for (gl,gate) in self.gates.iteritems() }

        for i,(rhoObj,rhoVec) in enumerate(zip(self.rhoVecs,rhoVectors)):
            rhoObj.set_vector(_np.dot(Si,rhoVec))

        for i,(EObj,EVec) in enumerate(zip(self.EVecs,EVectors)): 
            EObj.set_vector(_np.dot(_np.transpose(S),EVec, i)) #( Evec^T * S )^T

        if self.identityVec is not None:
            if identityVector is None: 
                identityVector = self.identityVec.construct_vector()
            self.identityVec.set_vector( _np.dot(_np.transpose(S),
                                                 identityVector) ) # same as Es

        for (l,gateObj) in self.gates.iteritems():
            gateObj.set_matrix(_np.dot(Si,_np.dot(gateMxs[l], S)))


    def to_vector(self):
        """
        Returns the gateset parameters in a 1D vector.

        Returns
        -------
        numpy array
            The vectorized gateset parameters.
        """
        if len(self) == 0: return _np.array([])

        gsize =   [ g.num_params() for g in self.gates.values() ]
        rhoSize = [ rhoVec.num_params() for rhoVec in self.rhoVecs ]
        eSize   = [ EVec.num_params() for EVec in self.EVecs ]
        L = sum(gsize) + sum(rhoSize) + sum(eSize)

        v = _np.empty( L ); off = 0

        for (i,rhoVec) in enumerate(self.rhoVecs):
            v[off:off+rhoSize[i]] = rhoVec.to_vector()
            off += rhoSize[i]
        for (i,EVec) in enumerate(self.EVecs):
            v[off:off+eSize[i]] = EVec.to_vector()
            off += eSize[i]
        for (gate,sz) in zip(self.gates.values(),gsize):
            v[off:off+sz] = gate.to_vector()
            off += sz

        return v


    def from_vector(self, v):
        """
        The inverse of to_vector.  Loads values of gates and/or rho and E
        vecs from from a vector v.  Note that v does not specify the number
        of gates and their labels, and that this information must be contained
        in the parameterizaton object prior to calling from_vector.  In
        practice, this just means you should call the from_vector method
        of the object that was used to generate the vector v in the first place.

        Parameters
        ----------
        v : numpy array
           the vectorized gateset parameters whose values are loaded into the
           parameterizaton object.
        """

        gsize =   [ g.num_params() for g in self.gates.values() ]
        rhoSize = [ rhoVec.num_params() for rhoVec in self.rhoVecs ]
        eSize   = [ EVec.num_params() for EVec in self.EVecs ]
        L = sum(gsize) + sum(rhoSize) + sum(eSize)

        assert( len(v) == L ); off = 0

        for i in range(len(self.rhoVecs)):
            self.rhoVecs[i].from_vector( v[off:off+rhoSize[i]] )
            off += rhoSize[i]
        for i in range(len(self.EVecs)):
            self.EVecs[i].from_vector( v[off:off+eSize[i]] )
            off += eSize[i]
        for (gate,sz) in zip(self.gates.values(),gsize):
            gate.from_vector( v[off:off+sz] )
            off += sz


    def get_vector_offsets(self):
        """
        Returns the offsets of individual components in the vector of 
        parameters used by from_vector and to_vector.
        
        Returns
        -------
        dict
            A dictionary whose keys are either "rho<number>", "E<number>",
            or a gate label and whose values are (start,next_start) tuples of
            integers indicating the start and end+1 indices of the component.
        """

        gsize =   [ g.num_params() for g in self.gates.values() ]
        rhoSize = [ rhoVec.num_params() for rhoVec in self.rhoVecs ]
        eSize   = [ EVec.num_params() for EVec in self.EVecs ]
    
        off = 0
        offsets = {}
    
        for (i,rhoVec) in enumerate(self.rhoVecs):
            offsets["rho%d" % i] = (off,off+rhoSize[i])
            off += rhoSize[i]
        for (i,EVec) in enumerate(self.EVecs):
            offsets["E%d" % i] = (off,off+eSize[i])
            off += eSize[i]
        for (gate,sz) in zip(self.gates.values(),gsize):
            offsets[l] = (off,off+sz)
            off += sz
    
        return offsets


    def deriv_wrt_params(self):
        """
        Construct a matrix whose columns are the vectorized derivatives of all
        the gateset's raw matrix and vector *elements* (placed in a vector)
        with respect to each single gateset parameter.

        Thus, each column has length equal to the number of elements in the 
        gateset, and there are num_params() columns.  In the case of a "fully
        parameterized gateset" (i.e. all gate matrices and SPAM vectors are
        fully parameterized) then the resulting matrix will be the (square)
        identity matrix.

        Returns
        -------
        numpy array
            2D array of derivatives.
        """
        if len(self) == 0: return _np.array([])

        gsize =   [ g.num_params() for g in self.gates.values() ]
        rhoSize = [ rhoVec.num_params() for rhoVec in self.rhoVecs ]
        eSize   = [ EVec.num_params() for EVec in self.EVecs ]
        nParams = sum(gsize) + sum(rhoSize) + sum(eSize)
        assert(nParams == self.num_params())

        full_vsize = self.gate_dim
        full_gsize = self.gate_dim**2 # (flattened) size of a gate matrix
        nElements = full_vsize * (len(self.rhoVecs) + len(self.EVecs)) \
            + full_gsize * len(self.gates)

        deriv = _np.zeros( (nElements, nParams), 'd' )

        k = 0
        eloff= 0 # element offset
        off = 0   # parameter offset

        for (i,rhoVec) in enumerate(self.rhoVecs):
            deriv[eloff:eloff+full_vsize,off:off+rhoSize[i]] = \
                rhoVec.deriv_wrt_params()
            off += rhoSize[i]; eloff += full_vsize

        for (i,EVec) in enumerate(self.EVecs):
            deriv[eloff:eloff+full_vsize,off:off+eSize[i]] = \
                EVec.deriv_wrt_params()
            off += eSize[i]; foff += full_vsize

        for (gate,sz) in zip(self.gates.values(),gsize):
            deriv[eloff:eloff+full_gsize,off:off+sz] = \
                gate.deriv_wrt_params()
            off += sz; foff += full_gsize

        return deriv

    def get_nongauge_projector(self, nonGaugeMixMx=None):
        """
        Construct a projector onto the non-gauge parameter space, useful for
        isolating the gauge degrees of freedom from the non-gauge degrees of
        freedom.

        Parameters
        ----------
        nonGaugeMixMx : numpy array, optional
            A matrix specifying how to mix the non-gauge degrees of freedom
            into the gauge degrees of freedom that are projected out by the
            returned object.  This argument is for advanced usage and typically
            is left set to None.

        Returns
        -------
        numpy array
           The projection operator as a N x N matrix, where N is the number
           of parameters (obtained via num_params()).  This projector acts on
           parameter-space, and has rank equal to the number of non-gauge 
           degrees of freedom.
        """
        
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


        #Use a parameterizaton object to hold & then vectorize the derivatives wrt each gauge transform basis element (each ij)
        dim = self.gate_dim
        nParams = self.num_params()

        #Note: parameterizaton object (pmDeriv) must have all elements of gate
        # mxs and spam vectors as parameters (i.e. be "fully parameterized") in
        # order to match deriv_wrt_params call, which gives derivatives wrt
        # *all* elements of a gate set / parameterizaton.
        pmDeriv = StandardParameterizer(True,True,True,True)
        for gateLabel in self.gates:
            pmDeriv.set_gate(gateLabel, _np.zeros((dim,dim),'d'))
        for k,rhoVec in enumerate(self.rhoVecs): 
            pmDeriv.set_rhovec(_np.zeros((dim,1),'d'), k)
        for k,EVec in enumerate(self.EVecs):
            pmDeriv.set_evec(_np.zeros((dim,1),'d'), k)

        assert(pmDeriv.num_elements() == pmDeriv.num_params()) 

        nElements = pmDeriv.num_elements()
        zMx = _np.zeros( (dim,dim), 'd')
        gateMxTups = [ (gl,self.compute_gate(gl)) for gl in self.gates ]

        dG = _np.empty( (nElements, dim**2), 'd' )
        for i in xrange(dim):      # always range over all rows: this is the
            for j in xrange(dim):  # *generator* mx, not gauge mx itself
                unitMx = _bt._mut(i,j,dim)
                for k,rhoVec in enumerate(self.rhoVecs):
                    pmDeriv.set_rhovec( _np.dot(unitMx, rhoVec), k)
                for k,EVec in enumerate(self.EVecs):
                    pmDeriv.set_evec( -_np.dot(EVec.T, unitMx).T, k)
                for gl,gateMx in gateMxTups:
                    pmDeriv.set_gate( _np.dot(unitMx,gateMx) - 
                                      _np.dot(gateMx,unitMx) )

                #Note: vectorize all the parameters in this full-
                # parameterization object, which gives a vector of length
                # equal to the number of gateset *elements*.
                dG[:,i*dim+j] = pmDeriv.to_vector() 


        #TODO;  make deriv_wrt a hidden method if it's only used here...
        dP = self.deriv_wrt_params() 
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

        # BEGIN GAUGE MIX ----------------------------------------
        if nonGaugeMixMx is not None:
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
        #END GAUGE MIX ----------------------------------------



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




#EXTRA "DIRECT-SETTING" methods, which I don't think we really need and will
# just confuse things by their similarity to the normal "set" methods above.
    #def set_identity_vector(self, idvec):
    #    if self.identityVec is None:
    #        raise ValueError("No identity object present!")
    #    self.identityVec.set_vector(idvec)
    #
    #def set_rho_vector(self, rhovec, index=0):
    #    self.rhoVecs[index].set_vector(rhovec)
    #
    #def set_e_vector(self, evec, index=0):
    #    self.EVecs[index].set_vector(evec)
    #
    #def set_gate_matrix(self, gateLabel, mx):
    #    """
    #    Attempts to modify gate set parameters so that the specified raw
    #    gate matrix becomes mx.  Will raise ValueError if this operation
    #    is not possible.
    #    
    #    Parameters
    #    ----------
    #    gateLabel : string
    #        The gate label.
    #
    #    mx : numpy array
    #        Desired raw gate matrix.
    #        
    #    Returns
    #    -------
    #    numpy array
    #        The 2-dimensional gate matrix.
    #    """
    #    self.gates[gateLabel].set_matrix(mx)




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
        Get the parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array.
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
