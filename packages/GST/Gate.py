""" Defines classes which represent gates, as well as supporting functions """
import numpy as _np
import MatrixOps as _MOps
import Optimize as _OPT

def optimizeGate(gateToOptimize, targetGate, bG0 = True):
    """
    Optimize the parameters of gateToOptimize so that the 
      the resulting gate matrix is as close as possible to 
      targetGate's matrix.

    This is trivial for the case of FullyParameterizedGate
      instances, but for other types of parameterization
      this involves an iterative optimization over all the
      parameters of gateToOptimize.

    Parameters
    ----------
    gateToOptimize : Gate
      The gate to optimize.  This object gets altered.

    targetGate : Gate
      The gate whose matrix is used as the target.

    Returns
    -------
    None
    """

    #TODO: cleanup this code:
    if isinstance(gateToOptimize, FullyParameterizedGate):
        if(targetGate.dim != gateToOptimize.dim): #special case: gates can have different overall dimension
            gateToOptimize.dim = targetGate.dim   #  this is a HACK to allow model selection code to work correctly
        gateToOptimize.matrix = targetGate.matrix #just copy entire overall matrix since fully parameterized
        return

    assert(targetGate.dim == gateToOptimize.dim) #gates must have the same overall dimension
    targetMatrix = targetGate.matrix

    def objectiveFunc(param_vec):
        gateToOptimize.fromVector(param_vec,bG0)
        return _MOps.frobeniusNorm(gateToOptimize.matrix-targetMatrix)
        
    x0 = gateToOptimize.toVector(bG0)
    minSol = _OPT.minimize(objectiveFunc, x0, method='BFGS', maxiter=10000, maxfev=10000,
                           tol=1e-6, callback=None)

    gateToOptimize.fromVector(minSol.x)
    print "DEBUG: optimized gate to min frobenius distance %g" % _MOps.frobeniusNorm(gateToOptimize.matrix-targetMatrix)


def compose(gate1, gate2):
    """
    Returns a new Gate that is the composition of gate1 and gate2.

    The resulting gate's matrix == dot(gate1.matrix, gate2.matrix),
     (so gate1 acts *second* on an input) and the type of Gate instance
     returned will depend on how much of the parameterization in gate1 
     and gate2 can be preserved in the resulting gate.

    Parameters
    ----------
    gate1 : Gate
        Gate to compose as left term of matrix product.

    gate2 : Gate
        Gate to compose as right term of matrix product.

    Returns
    -------
    Gate
       The composed gate.
    """
    return gate1.compose(gate2)


class FullyParameterizedGate:
    """ 
    Encapsulates a gate matrix that is fully paramterized, that is,
      each element of the gate matrix is an independent parameter.
    """
    def __init__(self, matrix):
        """ 
        Initialize a FullyParameterizedGate object.

        Parameters
        ----------
        matrix : numpy array
            a square 2D numpy array representing the gate action.  The
            shape of this array sets the dimension of the gate.
        """
        assert(len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1])        
        self.matrix = matrix
        self.dim = matrix.shape[0]

    def setValue(self, value):
        """
        Sets the value of the gate.  In general, the "value" of a gate means a 
          floating point number for each paramter.  In this case when all gate
          matrix element are parameters, the value is the gate matrix itself.

        Parameters
        ----------
        value : numpy array
            A numpy array of shape (gate_dim, gate_dim) representing the gate action.

        Returns
        -------
        None
        """
        if(value.shape != (self.dim, self.dim)):
            raise ValueError("You can only assign a (%d,%d) matrix to this fully parameterized gate" % (self.dim,self.dim))
        self.matrix = _np.array(value)

    def valueDimension(self):
        """ 
        Get the dimensions of the parameterized "value" of
        this gate which can be set using setValue(...).

        Returns
        -------
        tuple of ints
            The dimension of the parameterized "value" of this gate.
        """
        return (self.dim,self.dim)

    def getNumParams(self, bG0=True):
        """
        Get the number of independent parameters which specify this gate.

        Parameters
        ----------
        bG0 : bool
            Whether or not the first row of the gate matrix should be 
            parameterized.  This is significant in that in the Pauli or 
            Gell-Mann bases a the first row determines whether the 
            gate is trace-preserving (TP).

        Returns
        -------
        int
           the number of independent parameters.
        """
        # if bG0 == True, need to subtract the number of parameters which *only*
        #  parameterize the first row of the final gate matrix
        if bG0:
            return self.dim**2
        else:
            #subtract params for the first row
            return self.dim**2 - self.dim 

    def toVector(self, bG0=True):
        """
        Extract a vector of the underlying gate parameters from this gate.

        Parameters
        ----------
        bG0 : bool
            Whether or not the first row of the gate matrix should be 
            parameterized.

        Returns
        -------
        numpy array
            a 1D numpy array with length == getNumParams(bG0).
        """
        if bG0:
            return self.matrix.flatten() #.real in case of complex matrices
        else:
            return self.matrix.flatten()[self.dim:] #.real in case of complex matrices

    def fromVector(self, v, bG0=True):
        """
        Initialize the gate using a vector of its gate parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length 
            must == getNumParams(bG0).

        bG0 : bool
            Whether or not the first row of the gate matrix 
            should be parameterized.

        Returns
        -------
        None
        """

        if bG0:
            self.matrix = v.reshape(self.matrix.shape)
        else:
            flattened = _np.empty(self.getNumParams())
            flattened[0:self.dim] = self.matrix[0,:]
            flattened[self.dim:] = v
            self.matrix = flattened.reshape(self.matrix.shape)

    def transform(self, Si, S):
        """ 
        Transform this gate, so that:
          gate matrix => Si * gate matrix * S

        Parameters
        ----------
        Si : numpy array
            The matrix which left-multiplies the gate matrix.

        S : numpy array
            The matrix which right-multiplies the gate matrix.

        Returns
        -------
        None
        """
        self.matrix = _np.dot(Si, _np.dot(self.matrix, S))

    def derivWRTparams(self, bG0=True):
        """ 
        Construct a matrix whose columns are the vectorized
        derivatives of the gate matrix with respect to a
        single param.  Thus, each column is of length gate_dim^2
        and there is one column per gate parameter.

        Parameters
        ----------
        bG0 : bool
            Whether or not the first row of the gate matrix 
            should be parameterized.

        Returns
        -------
        numpy array 
            Array of derivatives, shape == (gate_dim^2, nGateParams)
        """
        derivMx = _np.identity( self.dim**2, 'd' )
        if not bG0:
            #remove first gate_dim cols, which correspond to the first-row parameters
            derivMx = derivMx[:,self.dim:] 
        return derivMx

    def copy(self):
        """
        Copy this gate.
        
        Returns
        -------
        Gate
            A copy of this gate.
        """
        return FullyParameterizedGate(self.matrix)

    def __str__(self):
        s = "Fully Parameterized gate with shape %s\n" % str(self.matrix.shape)
        s += _MOps.mxToString(self.matrix, width=4, prec=2)
        return s

    def compose(self, otherGate):
        """
        Create and return a new gate that is the composition of this gate
        followed by otherGate. In this case the returned gate is exactly
        dot(this, otherGate).

        Parameters
        ----------
        otherGate : FullyParameterizedGate
            The gate to compose to the right of this one.

        Returns
        -------
        FullyParameterizedGate
        """
        return FullyParameterizedGate( _np.dot( self.matrix, otherGate.matrix ) )



class LinearlyParameterizedElementTerm(object):
    def __init__(self, coeff=1.0, paramIndices=[]):
        self.coeff = coeff
        self.paramIndices = paramIndices

    def copy(self):
        return LinearlyParameterizedElementTerm(self.coeff, self.paramIndices)


class LinearlyParameterizedGate(object):
    """ 
    Encapsulates a gate matrix that is paramterized such that each
    element of the gate matrix depends only linearly on any paramter.
    """
    def __init__(self, baseMatrix, parameterArray, parameterToBaseIndicesMap,
                 leftTransform=None, rightTransform=None, real=False):
        """ 
        Initialize a LinearlyParameterizedGate object.

        Parameters
        ----------
        basematrix : numpy array
            a square 2D numpy array that acts as the starting point when
            constructin the gate's matrix.  The shape of this array sets
            the dimension of the gate.

        parameterArray : numpy array
            a 1D numpy array that holds the all the parameters for this
            gate.  The shape of this array sets is what is returned by
            valueDimension(...).

        parameterToBaseIndicesMap : dict
            A dictionary with keys == index of a parameter
            (i.e. in parameterArray) and values == list of 2-tuples
            indexing potentially multiple gate matrix coordinates 
            which should be set equal to this parameter.  

        leftTransform : numpy array or None, optional
            A 2D array of the same shape as basematrix which left-multiplies
            the base matrix after parameters have been evaluated.  Defaults to 
            no tranform.

        rightTransform : numpy array or None, optional
            A 2D array of the same shape as basematrix which right-multiplies
            the base matrix after parameters have been evaluated.  Defaults to 
            no tranform.

        real : bool, optional
            Whether or not the resulting gate matrix, after all
            parameter evaluation and left & right transforms have
            been performed, should be real.  If True, ValueError will
            be raised if the matrix contains any complex or imaginary
            elements.
        """ 

        self.baseMatrix = baseMatrix
        self.parameterArray = parameterArray
        self.numParams = len(parameterArray)

        self.elementExpressions = {}
        for p,ij_tuples in parameterToBaseIndicesMap.iteritems():
            for i,j in ij_tuples:
                assert((i,j) not in self.elementExpressions) #only one parameter allowed per base index pair
                self.elementExpressions[(i,j)] = [ LinearlyParameterizedElementTerm(1.0, [p]) ]

        self.leftTrans = leftTransform
        self.rightTrans = rightTransform
        self.enforceReal = real

        assert(len(self.baseMatrix.shape) == 2)
        assert(self.baseMatrix.shape[0] == self.baseMatrix.shape[1])
        
        self._computeMatrix()
        self.dim = self.matrix.shape[0]
        self.derivMx = self._computeDerivs()

    def _computeMatrix(self):
        self.matrix = self.baseMatrix.copy()
        for (i,j),terms in self.elementExpressions.iteritems():
            for term in terms:
                param_prod = _np.prod( [ self.parameterArray[p] for p in term.paramIndices ] )
                self.matrix[i,j] += term.coeff * param_prod
        self.matrix = _np.dot(self.leftTrans, _np.dot(self.matrix, self.rightTrans))

        if self.enforceReal:
            if _np.linalg.norm(_np.imag(self.matrix)) > 1e-8:
                raise ValueError("Linearly parameterized matrix has non-zero imaginary part (%g)!" % 
                                 _np.linalg.norm(_np.imag(self.matrix)))
            self.matrix = _np.real(self.matrix)


    def _computeDerivs(self):
        k = 0
        derivMx = _np.zeros( (self.dim**2, self.numParams), 'd' )
        for (i,j),terms in self.elementExpressions.iteritems():
            vec_ij = i*self.dim + j
            for term in terms:
                params_to_mult = [ self.parameterArray[p] for p in term.paramIndices ]
                for i,p in enumerate(term.paramIndices):
                    param_partial_prod = _np.prod( params_to_mult[0:i] + params_to_mult[i+1:] ) # exclude i-th factor
                    derivMx[vec_ij, p] += term.coeff * param_partial_prod
        return derivMx


    def setValue(self, value):
        """
        Sets the value of the gate.  In general, the "value" of a gate means a 
          floating point number for each parameter. 

        Parameters
        ----------
        value : numpy array
            A numpy array of length equal to that of the parameterArray
            passed upon construction.

        Returns
        -------
        None
        """
        if( len(value) != self.numParams):
            raise ValueError("You cannot set the value of this linearly-parameterized gate with a length-%d object.  Length must be %d"  \
                                 % (len(value), self.numParams))
        self.parameterArray = _np.array(value)
        self._computeMatrix()

#    def valueDimension(self):
#        """ 
#        Get the dimensions of the parameterized "value" of
#        this gate which can be set using setValue(...).
#
#        Returns
#        -------
#        tuple of ints
#            The dimension of the parameterized "value" of this gate.
#        """
#        return self.numParams

    def getNumParams(self, bG0=True):
        """
        Get the number of independent parameters which specify this gate.

        Parameters
        ----------
        bG0 : bool
            Whether or not the first row of the gate matrix should be 
            parameterized.  This is significant in that in the Pauli or 
            Gell-Mann bases a the first row determines whether the 
            gate is trace-preserving (TP).

        Returns
        -------
        int
           the number of independent parameters.
        """
        # if bG0 == True, need to subtract the number of parameters which *only*
        #  parameterize the first row of the final gate matrix
        if bG0:
            return self.numParams
        else:
            raise ValueError("Linearly parameterized gate with first gate row *not* parameterized is not implemented yet!")

    def toVector(self, bG0=True):
        """
        Extract a vector of the underlying gate parameters from this gate.

        Parameters
        ----------
        bG0 : bool
            Whether or not the first row of the gate matrix should be 
            parameterized.

        Returns
        -------
        numpy array
            a 1D numpy array with length == getNumParams(bG0).
        """
        if bG0:
            return self.parameterArray #.real in case of complex matrices
        else:
            raise ValueError("Linearly parameterized gate with first gate row *not* parameterized is not implemented yet!")

    def fromVector(self, v, bG0=True):
        """
        Initialize the gate using a vector of its gate parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length 
            must == getNumParams(bG0).

        bG0 : bool
            Whether or not the first row of the gate matrix 
            should be parameterized.

        Returns
        -------
        None
        """
        if bG0:
            self.parameterArray = v
            self._computeMatrix()
        else:
            raise ValueError("Linearly parameterized gate with first gate row *not* parameterized is not implemented yet!")

    def transform(self, Si, S):
        """ 
        Transform this gate, so that:
          gate matrix => Si * gate matrix * S

        Parameters
        ----------
        Si : numpy array
            The matrix which left-multiplies the gate matrix.

        S : numpy array
            The matrix which right-multiplies the gate matrix.

        Returns
        -------
        None
        """
        self.leftTrans = _np.dot(Si, self.leftTrans)
        self.rightTrans = _np.dot(self.rightTrans,S)
        # update self.enforceReal to False if Si or S is imaginary?
        self._computeMatrix()
            
    def derivWRTparams(self, bG0=True):
        """ 
        Construct a matrix whose columns are the vectorized
        derivatives of the gate matrix with respect to a
        single param.  Thus, each column is of length gate_dim^2
        and there is one column per gate parameter.

        Parameters
        ----------
        bG0 : bool
            Whether or not the first row of the gate matrix 
            should be parameterized.

        Returns
        -------
        numpy array 
            Array of derivatives, shape == (gate_dim^2, nGateParams)
        """
        if bG0:
            return self.derivMx
        else:
            raise ValueError("Linearly parameterized gate with first gate row *not* parameterized is not implemented yet!")
        
    def copy(self):
        #Construct new gate with no intial elementExpressions
        newGate = LinearlyParameterizedGate(self.baseMatrix, self.parameterArray, {},
                                            self.leftTrans, self.rightTrans, self.enforceReal)

        #Deep copy elementExpressions into new gate
        for tup, termList in self.elementExpressions.iteritems():
            newGate.elementExpressions[tup] = [ term.copy() for term in termList ]

        return newGate

    def __str__(self):
        s = "Linearly Parameterized gate with shape %s, num params = %d\n" % (str(self.matrix.shape), self.numParams)
        s += _MOps.mxToString(self.matrix, width=5, prec=1)
        return s

    def compose(self, otherGate):
        """
        Create and return a new gate that is the composition of this gate
        followed by otherGate, so the returned gate ~= dot(this, otherGate).

        Parameters
        ----------
        otherGate : LinearlyParameterizedGate
            The gate to compose to the right of this one.

        Returns
        -------
        LinearlyParameterizedGate
        """
        
        ### Implementation Notes ###
        #
        # Let self == L1 * A * R1, other == L2 * B * R2
        #   where  [A]_ij = a_ij + sum_l c^(ij)_l T^(ij)_l  so that
        #      a_ij == base matrix of self, c's are term coefficients, and T's are parameter products
        #   and similarly [B]_ij = b_ij + sum_l d^(ij)_l R^(ij)_l.
        #
        # We want in the end a gate with matrix: 
        #   L1 * A * R1 * L2 * B * R2 == L1 * (A * W * B) * R2,  (where W := R1 * L2 )
        #   which is a linearly parameterized gate with leftTrans == L1, rightTrans == R2
        #   and a parameterized part == (A * W * B) which can be written with implied sum on k,n:
        #
        #  [A * W * B]_ij
        #   = (a_ik + sum_l c^(ik)_l T^(ik)_l) * W_kn * (b_nj + sum_m d^(nj)_m R^(nj)_m)
        #
        #   = a_ik * W_kn * b_nj + 
        #     a_ik * W_kn * sum_m d^(nj)_m R^(nj)_m + 
        #     sum_l c^(ik)_l T^(ik)_l * W_kn * b_nj +
        #     (sum_l c^(ik)_l T^(ik)_l) * W_kn * (sum_m d^(nj)_m R^(nj)_m)
        #
        #   = aWb_ij   # (new base matrix == a*W*b)
        #     aW_in * sum_m d^(nj)_m R^(nj)_m +   # coeffs w/params of otherGate
        #     sum_l c^(ik)_l T^(ik)_l * Wb_kj +   # coeffs w/params of this gate
        #     sum_m,l c^(ik)_l W_kn d^(nj)_m T^(ik)_l R^(nj)_m) # coeffs w/params of both gates
        #

        W = _np.dot(self.rightTrans, otherGate.leftTrans)
        baseMx = _np.dot(self.baseMatrix, _np.dot(W, otherGate.baseMatrix)) # aWb above
        paramArray = _np.concatenate( (self.parameterArray, otherGate.parameterArray), axis=0)
        composedGate = LinearlyParameterizedGate(baseMx, paramArray, {}, 
                                                 self.leftTrans, otherGate.rightTrans,
                                                 self.enforceReal and otherGate.enforceReal)

        # Precompute what we can before the compute loop
        aW = _np.dot(self.baseMatrix, W)
        Wb = _np.dot(W, otherGate.baseMatrix)
        kMax,nMax = W.shape
        offset = len(self.parameterArray) # amt to offset parameter indices of otherGate

        # Compute  [A * W * B]_ij element expression as described above
        for i in range(self.baseMatrix.shape[0]):
            for j in range(otherGate.baseMatrix.shape[1]):
                terms = []
                for n in range(nMax):
                    if (n,j) in otherGate.elementExpressions:
                        for term in otherGate.elementExpressions[(n,j)]:
                            coeff = aW[i,n] * term.coeff
                            paramIndices = [ p+offset for p in term.paramIndices ]
                            terms.append( LinearlyParameterizedElementTerm( coeff, paramIndices ) )
                    
                for k in range(kMax):
                    if (i,k) in self.elementExpressions:
                        for term in self.elementExpressions[(i,k)]:
                            coeff = term.coeff * Wb[k,j]
                            terms.append( LinearlyParameterizedElementTerm( coeff, term.paramIndices ) )

                            for n in range(nMax):
                                if (n,j) in otherGate.elementExpressions:
                                    for term2 in otherGate.elementExpressions[(n,j)]:
                                        coeff = term.coeff * W[k,n] * term2.coeff
                                        paramIndices = term.paramIndices + [ p+offset for p in term2.paramIndices ]
                                        terms.append( LinearlyParameterizedElementTerm( coeff, paramIndices ) )

                composedGate.elementExpressions[(i,j)] = terms

        composedGate._computeMatrix()
        return composedGate

        
#Currently unused, but perhaps useful later, so keep it around...
#class TensorProductGate:
#    def __init__(self, mxsInTensorProduct, iParameterizedMx):
#        """ Initialize a TensorProductGate object.  """
#        self.matricesToTensor = mxsInTensorProduct
#        self.iParameterized = iParameterizedMx
#
#        if iParameterizedMx < 0 or iParameterizedMx >= len(mxsInTensorProduct):
#            raise ValueError("Index of parametrized component must be >= 0 and < %d" % len(mxsInTensorProduct))
#
#        pre_mx = _np.identity(1,'d')
#        post_mx = _np.identity(1,'d')
#        param_mx = None
#        for i,m in enumerate(mxsInTensorProduct):
#            assert(len(m.shape) == 2 and m.shape[0] == m.shape[1]) #each component to tensor must be a square matrix
#            if   i < iParameterizedMx:  pre_mx = _np.kron(pre_mx,m)
#            elif i > iParameterizedMx:  post_mx = _np.kron(post_mx,m)
#            else: param_mx = m
#
#        self.pre = pre_mx
#        self.post = post_mx
#        self.param_matrix = param_mx
#        self.param_dim = param_mx.shape[0]
#        self.matrix = _np.kron(self.pre, _np.kron(self.param_matrix, self.post))
#        self.dim = self.matrix.shape[0]
#        self.derivMx = self._computeDerivs()
#
#    def setValue(self, value):
#        if(value.shape != (self.param_dim, self.param_dim)):
#            raise ValueError("You cannot set the value of this tensor-product gate with a %s matrix.  Shape must be (%d,%d)"  \
#                                 % (str(value.shape), self.param_dim, self.param_dim))
#        self.param_matrix = _np.array(value)
#        self.matrix = _np.kron(self.pre, _np.kron(self.param_matrix, self.post))
#
#    def valueDimension(self):
#        """ 
#        Returns the dimension of the parameterized "value" of
#        this gate which can be set using setValue(...)
#        """
#        return (self.param_dim,self.param_dim)
#
#    def getNumParams(self, bG0=True):
#        # if bG0 == True, need to subtract the number of parameters which *only*
#        #  parameterize the first row of the final gate matrix
#        if bG0:
#            return self.param_dim**2
#        else:
#            raise ValueError("Tensor product gate with first gate row *not* parameterized is not implemented yet!")
#
#    def toVector(self, bG0=True):
#        if bG0:
#            return self.param_matrix.flatten() #.real in case of complex matrices
#        else:
#            raise ValueError("Tensor product gate with first gate row *not* parameterized is not implemented yet!")
#
#    def fromVector(self, v, bG0=True):
#        if bG0:
#            self.param_matrix = v.reshape(self.matrix.shape)
#            self.matrix = _np.kron(self.pre, _np.kron(self.param_matrix, self.post))
#        else:
#            raise ValueError("Tensor product gate with first gate row *not* parameterized is not implemented yet!")
#
#    def transform(self, Si, S):
#        """ gate matrix => Si * gate matrix * S """
#        raise ValueError("The similarity transform of a tensor product gate is not implemented yet!")
#            
#    def derivWRTparams(self, bG0=True):
#        """ 
#        Return a matrix whose columns are the vectorized
#        derivatives of the gate matrix with respect to a
#        single param.  Thus, each column is of length gate_dim^2
#        and there is one column per gate parameter.
#
#        Returns: numpy array of dimension (gate_dim^2, nGateParams)
#        """
#        if bG0:
#            return self.derivMx
#        else:
#            raise ValueError("Tensor product gate with first gate row *not* parameterized is not implemented yet!")
#        
#    def _computeDerivs(self):
#        #There are faster ways to compute this using kronecker products, etc -- see http://www4.ncsu.edu/~pfackler/MatCalc.pdf
#        #  but for now we only need to call this once upon construction so don't worry about speed.
#        derivMx = _np.empty( (self.dim**2, self.param_dim**2), 'd' )
#        k = 0
#        for i in xrange(self.param_dim):
#            for j in xrange(self.param_dim):
#                elMx = _np.zeros( (self.param_dim,self.param_dim), 'd'); elMx[i,j] = 1.0
#                derivMx[:,k] = ( _np.kron(self.pre, _np.kron(elMx, self.post)) ).flatten() #flatten vectorizes mx => column vector
#                k += 1
#        return derivMx
#
#    def copy(self):
#        return TensorProductGate(self.matricesToTensor, self.iParameterized)
        




#def buildTensorProdGate(stateSpaceDims, stateSpaceLabels, gateExpr, basis="gm"):
##coherentStateSpaceBlockDims
#    """
#    Build a gate matrix from an expression
#
#      Parameters
#      ----------
#      stateSpaceDims : a list of integers specifying the dimension of each block of a block-diagonal the density matrix
#      stateSpaceLabels : a list of tuples, each one corresponding to a block of the density matrix.  Elements of the tuple are
#                       user-defined labels beginning with "L" (single level) or "Q" (two-level; qubit) which interpret the
#                       states within the block as a tensor product structure between the labelled constituent systems.
#
#      gateExpr : string containing an expression for the gate to build
#
#      basis : string
#        "std" = gate matrix operates on density mx expressed as sum of matrix units
#        "gm"  = gate matrix operates on dentity mx expressed as sum of normalized Gell-Mann matrices
#        "pp"  = gate matrix operates on density mx expresses as sum of tensor-prod of pauli matrices
#        
#    """
#    # gateExpr can contain single qubit ops: X(theta) ,Y(theta) ,Z(theta)
#    #                      two qubit ops: CNOT
#    #                      clevel qubit ops: Leak
#    #                      two clevel opts: Flip
#    #  each of which is given additional parameters specifying which indices it acts upon
#
#    
#    #Gate matrix will be in matrix unit basis, which we order by vectorizing
#    # (by concatenating rows) each block of coherent states in the order given.
#    dmiToVi, dmDim, vecDim = _BT._processStateSpaceBlocks(stateSpaceDims)
#    #fullOpDim = dmDim**2
#
#    # check that stateSpaceDims is a single tensor product space containing only qubits
#    assert( len(stateSpaceDims) == len(stateSpaceLabels) == 1 and dmDim == stateSpaceDims[0] )
#    qubitLabels = stateSpaceLabels[0]
#    for s in qubitLabels:
#        if not s.startswith('Q'): 
#            raise ValueError("Invalid qubit space specifier: %s -- must begin with 'Q'" % s)
#            
#    nQubits = len(qubitSpaceLabels)
#    unitaryOpTerms = []
#
#    nextGroup = 0
#    groups = [-1]*nQubits
#
#    exprTerms = gateExpr.split(':')
#    for exprTerm in exprTerms:
#
#        #gateTermInStdBasis = _np.identity( fullOpDim, 'complex' )
#        l = exprTerm.index('('); r = exprTerm.index(')')
#        gateName = exprTerm[0:l]
#        argsStr = exprTerm[l+1:r]
#        args = argsStr.split(',')
#
#        if gateName == "I":
#            pass
#
#        elif gateName in ('X','Y','Z'): #single-qubit gate names
#            assert(len(args) == 2) # theta, qubit-index
#            theta = eval( args[0], {"__builtins__":None}, {'pi': _np.pi})
#            label = args[1]; assert(label.startswith('Q'))
#
#            if gateName == 'X': ex = 1j * theta*_BT.sigmax/2
#            elif gateName == 'Y': ex = 1j * theta*_BT.sigmay/2
#            elif gateName == 'Z': ex = 1j * theta*_BT.sigmaz/2
#            Ugate = _spl.expm(ex) # 2x2 unitary matrix operating on single qubit in [0,1] basis
#            
#            K = qubitLabels.index(label)
#            oldGroup = groups[K]
#            for i in range(len(groups)):
#                if groups[i] == oldGroup: groups[i] = nextGroup
#            unitaryOpTerms.append( (Ugate, (label,)) )
#            nextGroup += 1
#
#
#        elif gateName in ('CX','CY','CZ'): #two-qubit gate names
#            assert(len(args) == 3) # theta, qubit-label1, qubit-label2
#            theta = eval( args[0], {"__builtins__":None}, {'pi': _np.pi})
#            label1 = args[1]; assert(label1.startswith('Q'))
#            label2 = args[2]; assert(label2.startswith('Q'))
#
#            if gateName == 'CX': ex = 1j * theta*_BT.sigmax/2
#            elif gateName == 'CY': ex = 1j * theta*_BT.sigmay/2
#            elif gateName == 'CZ': ex = 1j * theta*_BT.sigmaz/2
#            Utarget = _spl.expm(ex) # 2x2 unitary matrix operating on target qubit
#            Ugate = _np.identity(4, 'complex'); Ugate[2:,2:] = Utarget #4x4 unitary matrix operating on isolated two-qubit space
#
#            K1 = qubitLabels.index(label1)
#            K2 = qubitLabels.index(label2)
#            oldGroup1 = groups[K1]; oldGroup2 = groups[K2]
#            for i in range(len(groups)):
#                if groups[i] in (oldGroup1,oldGroup2): groups[i] = nextGroup
#            unitaryOpTerms.append( (Ugate, (label1,label2)) )
#            nextGroup += 1
#
#        else: raise ValueError("Invalid gate name: %s" % gateName)
#    
#    #just to test for non-adjacent qubits in groups as a temporary restriction
#    groupDone = [False]*nextGroup 
#
#    i = 0
#    while i < nQubits:
#        curGroupIndex = groups[i]
#        if curGroupIndex >= 0 and groupDone[curGroupIndex]:
#            #need to work out permutation matrices, etc? so that a TensorProductGate can act on non-adjacent qubits... TODO
#            raise ValueError("Tensor groups of non-consecutive qubits is not supported yet.") 
#        curGroupQubits = [ i ]
#        i += 1
#
#        while i < nQubits and groups[i] == curGroupIndex:
#            curGroupQubits.append( i ); i += 1
#
#        if curGroupIndex == -1: #signals unused qubits in this gate, so just the identity
#            mxsToTensor.append( _np.identity( 4**len(curGroupQubits), 'complex' ) )
#        else:
#            #Process current group of consecutive qubits => one term in final tensor product gate
#            
#            for (Uop, labels) in unitaryOpTerms:
#                
#            
#    for iGroup in range(nextGroup):
#
#
#        #gateInStdBasis = _np.dot(gateInStdBasis, gateTermInStdBasis)
#
#        #HERE -- create gate (in pauli-prod basis?) for each group, then create a tensor product gate (now requires that there's only one non -1 group)

