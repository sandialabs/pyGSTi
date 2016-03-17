#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation              
#    This Software is released under the GPL license detailed    
#    in the file "license.txt" in the top-level pyGSTi directory 
#*****************************************************************
""" Defines classes which represent gates, as well as supporting functions """

import numpy as _np
from .. import optimize as _opt
from ..tools import matrixtools as _mt

from protectedarray import ProtectedArray as _ProtectedArray


def optimize_gate(gateToOptimize, targetGate):
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
        gateToOptimize.set_matrix(targetGate)     #just copy entire overall matrix since fully parameterized
        return

    assert(targetGate.dim == gateToOptimize.dim) #gates must have the same overall dimension
    targetMatrix = _np.asarray(targetGate)
    import sys
    def objective_func(param_vec):
        gateToOptimize.from_vector(param_vec)
        return _mt.frobeniusnorm(gateToOptimize-targetMatrix)
        
    x0 = gateToOptimize.to_vector()
    minSol = _opt.minimize(objective_func, x0, method='BFGS', maxiter=10000, maxfev=10000,
                           tol=1e-6, callback=None)

    gateToOptimize.from_vector(minSol.x)
    print "DEBUG: optimized gate to min frobenius distance %g" % _mt.frobeniusnorm(gateToOptimize-targetMatrix)


def compose(gate1, gate2):
    """
    Returns a new Gate that is the composition of gate1 and gate2.

    The resulting gate's matrix == dot(gate1, gate2),
     (so gate1 acts *second* on an input) and the type of Gate instance
     returned will depend on how much of the parameterization in gate1 
     and gate2 can be preserved in the resulting gate.

    Parameters
    ----------
    gate1 : Gate
        Gate to compose as left term of matrix product (applied second).

    gate2 : Gate
        Gate to compose as right term of matrix product (applied first).

    Returns
    -------
    Gate
       The composed gate.
    """

    #Find the most restrictive common parameterization that both gate1
    # and gate2 can be cast/converted into. Utilized converstions are:
    #
    # Static => TP (sometimes)
    # Static => Linear
    # Static => Full
    # Linear => TP (sometimes)
    # Linear => Full    
    # TP => Full

    if any([isinstance(g, FullyParameterizedGate) for g in (gate1,gate2)]):
        paramType = "full"
    elif any([isinstance(g, TPParameterizedGate) for g in (gate1,gate2)]):
        paramType = "tp" #may update to "full" below if TP-conversion not possible
    elif any([isinstance(g, LinearlyParameterizedGate) for g in (gate1,gate2)]):
        paramType = "linear"
    else:
        assert( isinstance(gate1, StaticGate) and isinstance(gate2, StaticGate) )
        paramType = "static"

    #Convert to paramType as necessary
    cgate1 = convert(gate1, paramType)
    cgate2 = convert(gate2, paramType)
        
    # cgate1 and cgate2 are the same type, so can invoke the gate's compose method
    return cgate1.compose(cgate2) 


def convert(gate, toType):
    """
    Convert gate to a new type of parameterization, potentially creating
    a new Gate object.  Raises ValueError for invalid conversions.

    Parameters
    ----------
    gate : Gate
        Gate to convert

    toType : {"full","tp","linear","static"}
        The type of parameterizaton to convert to.

    Returns
    -------
    Gate
       The converted gate, usually a distinct
       object from the gate object passed as input.
    """
    if toType == "full":
        if isinstance(gate, FullyParameterizedGate):
            return gate #no conversion necessary
        else:
            ret = FullyParameterizedGate( gate )
            return ret

    elif toType == "tp":
        if isinstance(gate, TPParameterizedGate):
            return gate #no conversion necessary
        else:
            return TPParameterizedGate( gate )
              # above will raise ValueError if conversion cannot be done

    elif toType == "linear":
        if isinstance(gate, LinearlyParameterizedGate):
            return gate #no conversion necessary
        elif isinstance(gate, StaticGate):
            real = _np.isclose(_np.linalg.norm( gate.imag ),0)
            return LinearlyParameterizedGate(gate, _np.array([]), {}, real)
        else:
            raise ValueError("Cannot convert type %s to LinearlyParameterizedGate"
                             % type(gate))
    
    elif toType == "static":
        if isinstance(gate, StaticGate):
            return gate #no conversion necessary
        else:
            return StaticGate( gate )
        
    else:
        raise ValueError("Invalid toType argument: %s" % toType)



class Gate(object):  #LATER: make this into an ABC specifying the "Gate" interface??
    """
    Excapulates a parameterization of a gate matrix.  This class is the 
    common base class for all specific parameterizations of a gate.
    """

    #def __init__(self):
    #    """ Initialize a new Gate """
    #    pass

    #def get_dimension(self):
    #    """ Return the dimension of the gate matrix. """
    #    return self.dim

    @staticmethod
    def convert_to_matrix(M):
        """ 
        Static method that converts a matrix-like object to a 2D numpy array.

        Parameters
        ----------
        M : array_like
        
        Returns
        -------
        numpy array
        """
        if isinstance(M, Gate): 
            dim = M.dim
            matrix = _np.asarray(M).copy()
              # Gate objs should also derive from ndarray
        else:
            try:
                dim = len(M)
                d2 = len(M[0])
            except:
                raise ValueError("%s doesn't look like a 2D array/list" % M)
            if any([len(row) != dim for row in M]):
                raise ValueError("%s is not a *square* 2D array" % M)

            ar = _np.array(M)
            if _np.all(_np.isreal(ar)):
                matrix = _np.array(ar.real, 'd')
            else:
                matrix = _np.array(ar, 'complex')


        if len(matrix.shape) != 2:
            raise ValueError("%s has %d dimensions when 2 are expected"
                             % (M, len(matrix.shape)))

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("%s is not a *square* 2D array" % M)
            
        return matrix



class StaticGate(_np.ndarray, Gate):
    """ 
    Encapsulates a gate matrix that is completely fixed, or "static", meaning
      that is contains no parameters.
    """
    __array_priority__ = -2.0

    def __new__(cls, M):
        """ 
        Initialize a StaticGate object.

        Parameters
        ----------
        M : array_like or Gate
            a square 2D array-like or Gate object representing the gate action.
            The shape of M sets the dimension of the gate.
        """
        gateMx = Gate.convert_to_matrix(M)
        obj = _np.asarray(gateMx).view(cls)
        obj.dim = gateMx.shape[0]
        return obj

        #Note: no need to override __init__ as all initialization is performed
        # in __new__, and no need to override __array_finalize__ since we don't
        # need slices of Gates to be considered "Gates", nor do we need to 
        # view-cast other types of arrays to Gates

    def __getitem__( self, key ):
        #Items and slices are *not* gates - just numpy arrays:
        return _np.ndarray.__getitem__(self.view(_np.ndarray), key)

    def __getslice__(self, i,j):
        #For special cases when getslice is still called, e.g. A[:]
        return self.__getitem__(slice(i,j))

    def __array_finalize__(self, obj):
        if obj is None: return # let __new__ handle flags
        if len(self.shape) == 2 and self.shape[0] == self.shape[1]:
            self.dim = self.shape[0]
        else:
            raise ValueError("Gates must be square 2D arrays")
        #print "Finalize called with obj type: ", type(obj), " dim = ",self.dim


    def set_matrix(self, mx):
        """
        Attempts to modify gate parameters so that the specified raw
        gate matrix becomes mx.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        mx : array_like or Gate
            An array of shape (dim, dim) or Gate representing the gate action.

        Returns
        -------
        None
        """
        mx = Gate.convert_to_matrix(mx)
        if(mx.shape != (self.dim, self.dim)):
            raise ValueError("Argument must be a (%d,%d) matrix!"
                             % (self.dim,self.dim))
        self[:,:] = _np.array(mx)


    def num_params(self):
        """
        Get the number of independent parameters which specify this gate.  Zero
        in the case of a StaticGate.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return 0 # no parameters


    def to_vector(self):
        """
        Get the gate parameters as an array of values.  An empty array in the
        case of StaticGate.

        Returns
        -------
        numpy array
            The gate parameters as a 1D array with length num_params().
        """
        return _np.array([], 'd') # no parameters


    def from_vector(self, v):
        """
        Initialize the gate using a vector of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length 
            must == num_params()

        Returns
        -------
        None
        """
        assert(len(v) == 0) #should be no parameters, and nothing to do


    def deriv_wrt_params(self):
        """ 
        Construct a matrix whose columns are the vectorized
        derivatives of the flattened gate matrix with respect to a
        single gate parameter.  Thus, each column is of length
        gate_dim^2 and there is one column per gate parameter. An
        empty 2D array in the StaticGate case (num_params == 0).

        Returns
        -------
        numpy array 
            Array of derivatives with shape (dimension^2, num_params)
        """
        return _np.zeros((self.dim**2,0),'d')


    def copy(self):
        """
        Copy this gate.
        
        Returns
        -------
        Gate
            A copy of this gate.
        """
        return StaticGate(self)


    def transform(self, S, Si):
        """
        Update gate matrix G with inv(S) * G * S,

        Parameters
        ----------
        S : numpy array
            Matrix to perform similarity transform.  
            Should be shape (dim, dim).
            
        Si : numpy array
            Inverse of S.  If None, inverse of S is computed.  
            Should be shape (dim, dim).
        """
        self.set_matrix(_np.dot(Si,_np.dot(self, S)))


    def __str__(self):
        s = "Static gate with shape %s\n" % str(self.shape)
        s += _mt.mx_to_string(self, width=4, prec=2)
        return s


    def compose(self, otherGate):
        """
        Create and return a new gate that is the composition of this gate
        followed by otherGate, which *must be another StaticGate*.  (For more
        general compositions between different types of gates, use the module-
        level compose function.)  The returned gate's matrix is equal to
        dot(this, otherGate).

        Parameters
        ----------
        otherGate : StaticGate
            The gate to compose to the right of this one.

        Returns
        -------
        StaticGate
        """
        assert( isinstance(otherGate, StaticGate) )
        return StaticGate( _np.dot( self,otherGate ) )

    def __reduce__(self):
        """ Pickle plumbing. """
        createFn, args, state = _np.ndarray.__reduce__(self)
        new_state = state + (self.__dict__,)
        return (createFn, args, new_state)

    def __setstate__(self, state):
        """ Pickle plumbing. """
        _np.ndarray.__setstate__(self,state[0:-1])
        self.__dict__.update(state[-1])




class FullyParameterizedGate(_np.ndarray, Gate):
    """ 
    Encapsulates a gate matrix that is fully parameterized, that is,
      each element of the gate matrix is an independent parameter.
    """
    __array_priority__ = -2.0

    def __new__(cls, M):
        """ 
        Initialize a FullyParameterizedGate object.

        Parameters
        ----------
        M : array_like or Gate
            a square 2D array-like or Gate object representing the gate action.
            The shape of M sets the dimension of the gate.
        """
        gateMx = Gate.convert_to_matrix(M)
        obj = _np.asarray(gateMx).view(cls)
        obj.dim = gateMx.shape[0]
        return obj

        #Note: no need to override __init__ as all initialization is performed
        # in __new__, and no need to override __array_finalize__ since we don't
        # need slices of Gates to be considered "Gates", nor do we need to 
        # view-cast other types of arrays to Gates

    def __getitem__( self, key ):
        #Items and slices are *not* gates - just numpy arrays:
        return _np.ndarray.__getitem__(self.view(_np.ndarray), key)
    
    def __getslice__(self, i,j):
        #For special cases when getslice is still called, e.g. A[:]
        return self.__getitem__(slice(i,j))

    def __array_finalize__(self, obj):
        if obj is None: return # let __new__ handle flags
        if len(self.shape) == 2 and self.shape[0] == self.shape[1]:
            self.dim = self.shape[0]
        else:
            raise ValueError("Gates must be square 2D arrays")
        #print "Finalize called with obj type: ", type(obj), " dim = ",self.dim


    def set_matrix(self, mx):
        """
        Attempts to modify gate parameters so that the specified raw
        gate matrix becomes mx.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        mx : array_like or Gate
            An array of shape (dim, dim) or Gate representing the gate action.

        Returns
        -------
        None
        """
        mx = Gate.convert_to_matrix(mx)
        if(mx.shape != (self.dim, self.dim)):
            raise ValueError("Argument must be a (%d,%d) matrix!"
                             % (self.dim,self.dim))
        self[:,:] = _np.array(mx)


    def num_params(self):
        """
        Get the number of independent parameters which specify this gate.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.dim**2


    def to_vector(self):
        """
        Get the gate parameters as an array of values.

        Returns
        -------
        numpy array
            The gate parameters as a 1D array with length num_params().
        """
        return _np.asarray(self).flatten() #.real in case of complex matrices?


    def from_vector(self, v):
        """
        Initialize the gate using a vector of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length 
            must == num_params()

        Returns
        -------
        None
        """
        assert(self.shape == (self.dim, self.dim))
        self[:,:] = v.reshape( (self.dim,self.dim) )


    def deriv_wrt_params(self):
        """ 
        Construct a matrix whose columns are the vectorized
        derivatives of the flattened gate matrix with respect to a
        single gate parameter.  Thus, each column is of length
        gate_dim^2 and there is one column per gate parameter.

        Returns
        -------
        numpy array 
            Array of derivatives with shape (dimension^2, num_params)
        """
        return _np.identity( self.dim**2, 'd' )


    def copy(self):
        """
        Copy this gate.
        
        Returns
        -------
        Gate
            A copy of this gate.
        """
        return FullyParameterizedGate(self)


    def transform(self, S, Si):
        """
        Update gate matrix G with inv(S) * G * S,

        Parameters
        ----------
        S : numpy array
            Matrix to perform similarity transform.  
            Should be shape (dim, dim).
            
        Si : numpy array
            Inverse of S.  If None, inverse of S is computed.  
            Should be shape (dim, dim).
        """
        self.set_matrix(_np.dot(Si,_np.dot(self, S)))


    def __str__(self):
        s = "Fully Parameterized gate with shape %s\n" % str(self.shape)
        s += _mt.mx_to_string(self, width=4, prec=2)
        return s


    def compose(self, otherGate):
        """
        Create and return a new gate that is the composition of this gate
        followed by otherGate, which *must be another FullyParameterizedGate*.
        (For more general compositions between different types of gates, use
        the module-level compose function.)  The returned gate's matrix is
        equal to dot(this, otherGate).

        Parameters
        ----------
        otherGate : FullyParameterizedGate
            The gate to compose to the right of this one.

        Returns
        -------
        FullyParameterizedGate
        """
        assert( isinstance(otherGate, FullyParameterizedGate) )
        return FullyParameterizedGate( _np.dot( self,otherGate) )

    def __reduce__(self):
        """ Pickle plumbing. """
        createFn, args, state = _np.ndarray.__reduce__(self)
        new_state = state + (self.__dict__,)
        return (createFn, args, new_state)

    def __setstate__(self, state):
        """ Pickle plumbing. """
        _np.ndarray.__setstate__(self,state[0:-1])
        self.__dict__.update(state[-1])




class TPParameterizedGate(_ProtectedArray, Gate):
    """ 
    Encapsulates a gate matrix that is fully parameterized except for
    the first row, which is frozen to be [1 0 ... 0] so that the action
    of the gate, when interpreted in the Pauli or Gell-Mann basis, is 
    trace preserving (TP).
    """
    __array_priority__ = -2.0

    def __new__(cls, M):
        """ 
        Initialize a TPParameterizedGate object.

        Parameters
        ----------
        M : array_like or Gate
            a square 2D numpy array representing the gate action.  The
            shape of this array sets the dimension of the gate.
        """
        gateMx = Gate.convert_to_matrix(M)
        if not (_np.isclose(gateMx[0,0], 1.0) and \
                _np.allclose(gateMx[0,1:], 0.0)):
            raise ValueError("Cannot create TPParameterizedGate: " + 
                             "invalid form for 1st row!")

        obj = _ProtectedArray.__new__(cls, gateMx, 
                                      indicesToProtect=(0, slice(None,None,None)))
        obj.dim = gateMx.shape[0]
        return obj

        #Note: no need to override __init__ as all initialization is performed
        # in __new__.

    def __getitem__( self, key ):
        #Items and slices are *not* gates - just numpy arrays:
        return _ProtectedArray.__getitem__(self.view(_ProtectedArray), key)

    def __getslice__(self, i,j):
        #For special cases when getslice is still called, e.g. A[:]
        return self.__getitem__(slice(i,j))

    def __array_finalize__(self, obj):
        _ProtectedArray.__array_finalize__(self, obj) #call base class handler
        if obj is None: return # let __new__ handle flags
        #print "Finalize called with obj type: ", type(obj) #, " dim = ",self.dim
        if len(self.shape) == 2 and self.shape[0] == self.shape[1]:
            self.dim = self.shape[0]
        else:
            raise ValueError("Gates must be square 2D arrays")

#Taken care of by lower __array_priority__
#    def __array_wrap__(self, out_arr, context=None):
#        # The outputs of ufuncs are *not* in general gates (e.g. kron). Perhaps
#        # in the future we could preserve "gate-ness" when appropriate?
#        return _np.asarray( _np.ndarray.__array_wrap__(self, out_arr, context))

    
    def set_matrix(self, M):
        """
        Attempts to modify gate parameters so that the specified raw
        gate matrix becomes mx.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        M : array_like or Gate
            An array of shape (dim, dim) or Gate representing the gate action.

        Returns
        -------
        None
        """
        mx = Gate.convert_to_matrix(M)
        if(mx.shape != (self.dim, self.dim)):
            raise ValueError("Argument must be a (%d,%d) matrix!"
                             % (self.dim,self.dim))
        if not (_np.isclose(mx[0,0], 1.0) and _np.allclose(mx[0,1:], 0.0)):
            raise ValueError("Cannot set TPParameterizedGate: " +
                             "invalid form for 1st row!" )
        self[1:,:] = _np.array(mx)[1:,:]


    def num_params(self):
        """
        Get the number of independent parameters which specify this gate.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.dim**2 - self.dim 


    def to_vector(self):
        """
        Get the gate parameters as an array of values.

        Returns
        -------
        numpy array
            The gate parameters as a 1D array with length num_params().
        """
        return _np.asarray(self).flatten()[self.dim:] #.real in case of complex matrices?

    def from_vector(self, v):
        """
        Initialize the gate using a vector of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length 
            must == num_params()

        Returns
        -------
        None
        """
        assert(self.shape == (self.dim, self.dim))
        self[1:,:] = v.reshape((self.dim-1,self.dim))


    def deriv_wrt_params(self):
        """ 
        Construct a matrix whose columns are the vectorized
        derivatives of the flattened gate matrix with respect to a
        single gate parameter.  Thus, each column is of length
        gate_dim^2 and there is one column per gate parameter.

        Returns
        -------
        numpy array 
            Array of derivatives with shape (dimension^2, num_params)
        """
        derivMx = _np.identity( self.dim**2, 'd' )
        derivMx = derivMx[:,self.dim:] #remove first gate_dim cols ( <=> first-row parameters )
        return derivMx

    def copy(self):
        """
        Copy this gate.
        
        Returns
        -------
        Gate
            A copy of this gate.
        """
        return TPParameterizedGate(self)


    def transform(self, S, Si):
        """
        Update gate matrix G with inv(S) * G * S,

        Parameters
        ----------
        S : numpy array
            Matrix to perform similarity transform.  
            Should be shape (dim, dim).
            
        Si : numpy array
            Inverse of S.  If None, inverse of S is computed.  
            Should be shape (dim, dim).
        """
        self.set_matrix(_np.dot(Si,_np.dot(self, S)))


    def __str__(self):
        s = "TP Parameterized gate with shape %s\n" % str(self.shape)
        s += _mt.mx_to_string(self, width=4, prec=2)
        return s

    def compose(self, otherGate):
        """
        Create and return a new gate that is the composition of this gate
        followed by otherGate, which *must be another TPParameterizedGate*.
        (For more general compositions between different types of gates, use
        the module-level compose function.)  The returned gate's matrix is
        equal to dot(this, otherGate).

        Parameters
        ----------
        otherGate : TPParameterizedGate
            The gate to compose to the right of this one.

        Returns
        -------
        TPParameterizedGate
        """
        assert( isinstance(otherGate, TPParameterizedGate) )
        return TPParameterizedGate( _np.dot( self,otherGate ) )

    def __reduce__(self):
        """ Pickle plumbing. """
        createFn, args, state = _np.ndarray.__reduce__(self)
        new_state = state + (self.__dict__,)
        return (createFn, args, new_state)

    def __setstate__(self, state):
        """ Pickle plumbing. """
        _np.ndarray.__setstate__(self,state[0:-1])
        self.__dict__.update(state[-1])





class LinearlyParameterizedElementTerm(object):
    def __init__(self, coeff=1.0, paramIndices=[]):
        self.coeff = coeff
        self.paramIndices = paramIndices

    def copy(self):
        return LinearlyParameterizedElementTerm(self.coeff, self.paramIndices)


class LinearlyParameterizedGate(_np.ndarray, Gate):
    """ 
    Encapsulates a gate matrix that is parameterized such that each
    element of the gate matrix depends only linearly on any parameter.
    """
    __array_priority__ = -2.0

    def __new__(cls, baseMatrix, parameterArray, parameterToBaseIndicesMap,
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
            value_dimension(...).

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
                                             
        baseMatrix = _np.array( Gate.convert_to_matrix(baseMatrix), 'complex')
          #complex, even if passed all real base matrix

        elementExpressions = {}
        for p,ij_tuples in parameterToBaseIndicesMap.iteritems():
            for i,j in ij_tuples:
                assert((i,j) not in elementExpressions) #only one parameter allowed per base index pair
                elementExpressions[(i,j)] = [ LinearlyParameterizedElementTerm(1.0, [p]) ]

        typ = "d" if real else "complex"
        obj = _np.empty( baseMatrix.shape, typ ).view(cls)
        obj.baseMatrix = baseMatrix
        obj.parameterArray = parameterArray
        obj.numParams = len(parameterArray)
        obj.elementExpressions = elementExpressions
        obj.dim = baseMatrix.shape[0]

        I = _np.identity(obj.baseMatrix.shape[0],'d')
        obj.leftTrans = leftTransform if (leftTransform is not None) else I
        obj.rightTrans = rightTransform if (rightTransform is not None) else I
        obj.enforceReal = real
        obj.flags.writeable = False # only _construct_matrix can change array
        obj._construct_matrix() # construct the matrix from the parameters
        return obj


    def __getitem__( self, key ):
        #Items and slices are *not* gates - just numpy arrays:
        return _np.ndarray.__getitem__(self.view(_np.ndarray), key)

    def __getslice__(self, i,j):
        #For special cases when getslice is still called, e.g. A[:]
        return self.__getitem__(slice(i,j))

    def __array_finalize__(self, obj):
        if obj is None: return # let __new__ handle flags
        self.dim = None # mark object as invalid, since we don't have enough
                        # info to create one from obj

    def _check(self):
        if self.dim is None:
            raise ValueError("Gate object is invalid!")

    def _construct_matrix(self):
        """
        Build the internal gate matrix using the current parameters.
        """
        matrix = self.baseMatrix.copy()
        for (i,j),terms in self.elementExpressions.iteritems():
            for term in terms:
                param_prod = _np.prod( [ self.parameterArray[p] for p in term.paramIndices ] )
                matrix[i,j] += term.coeff * param_prod
        matrix = _np.dot(self.leftTrans, _np.dot(matrix, self.rightTrans))

        if self.enforceReal:
            if _np.linalg.norm(_np.imag(matrix)) > 1e-8:
                raise ValueError("Linearly parameterized matrix has non-zero" +
                        "imaginary part (%g)!" % _np.linalg.norm(_np.imag(matrix)))
            matrix = _np.real(matrix)

        assert(matrix.shape == (self.dim,self.dim))
        self.flags.writeable = True
        self[:,:] = matrix
        self.flags.writeable = False

    def set_matrix(self, M):
        """
        Attempts to modify gate parameters so that the specified raw
        gate matrix becomes mx.  Will raise ValueError if this operation
        is not possible.
    
        Parameters
        ----------
        M : numpy array or Gate
            A numpy array of shape (dim, dim) or Gate representing the gate action.
    
        Returns
        -------
        None
        """
        self._check()
        mx = Gate.convert_to_matrix(M)
        if(mx.shape != (self.dim, self.dim)):
            raise ValueError("Argument must be a (%d,%d) matrix!" % (self.dim,self.dim))
        raise ValueError("Currently, directly setting the matrix of a" +
                         " LinearlyParameterizedGate object is not" + 
                         " supported.  Please use the from_vector method.")


    def num_params(self):
        """
        Get the number of independent parameters which specify this gate.

        Returns
        -------
        int
           the number of independent parameters.
        """
        self._check()
        return self.numParams


    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this gate.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        self._check()
        return self.parameterArray #.real in case of complex matrices


    def from_vector(self, v):
        """
        Initialize the gate using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length 
            must == num_params().

        Returns
        -------
        None
        """
        self._check()
        self.parameterArray = v
        self._construct_matrix()

            
    def deriv_wrt_params(self):
        """ 
        Construct a matrix whose columns are the vectorized
        derivatives of the flattened gate matrix with respect to a
        single gate parameter.  Thus, each column is of length
        gate_dim^2 and there is one column per gate parameter.

        Returns
        -------
        numpy array 
            Array of derivatives, shape == (dimension^2, num_params)
        """
        self._check()

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

        
    def copy(self):
        self._check()

        #Construct new gate with no intial elementExpressions
        newGate = LinearlyParameterizedGate(self.baseMatrix, self.parameterArray,
                                            {}, self.leftTrans, self.rightTrans,
                                            self.enforceReal)

        #Deep copy elementExpressions into new gate
        for tup, termList in self.elementExpressions.iteritems():
            newGate.elementExpressions[tup] = [ term.copy() for term in termList ]
        newGate._construct_matrix()

        return newGate

    def transform(self, S, Si):
        """
        Update gate matrix G with inv(S) * G * S,

        Parameters
        ----------
        S : numpy array
            Matrix to perform similarity transform.  
            Should be shape (dim, dim).
            
        Si : numpy array
            Inverse of S.  If None, inverse of S is computed.  
            Should be shape (dim, dim).
        """
        self.leftTrans = _np.dot(Si,self.leftTrans)
        self.rightTrans = _np.dot(self.rightTrans,S)
        

    def __str__(self):
        self._check()
        s = "Linearly Parameterized gate with shape %s, num params = %d\n" % \
            (str(self.shape), self.numParams)
        s += _mt.mx_to_string(self, width=5, prec=1)
        return s

    def compose(self, otherGate):
        """
        Create and return a new gate that is the composition of this gate
        followed by otherGate, which *must be another LinearlyParameterizedGate*.
        (For more general compositions between different types of gates, use
        the module-level compose function.)  The returned gate's matrix is
        equal to dot(this, otherGate).

        Parameters
        ----------
        otherGate : LinearlyParameterizedGate
            The gate to compose to the right of this one.

        Returns
        -------
        LinearlyParameterizedGate
        """
        self._check()
        assert( isinstance(otherGate, LinearlyParameterizedGate) )
        
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

        kMax,nMax = (self.dim,self.dim) #W.shape
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

        composedGate._construct_matrix()
        return composedGate


    def __reduce__(self):
        """ Pickle plumbing. """
        self._check()
        createFn, args, state = _np.ndarray.__reduce__(self)
        new_state = state + (self.__dict__,)
        return (createFn, args, new_state)

    def __setstate__(self, state):
        """ Pickle plumbing. """
        self._check()
        _np.ndarray.__setstate__(self,state[0:-1])
        self.__dict__.update(state[-1])

