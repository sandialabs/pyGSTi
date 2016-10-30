from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines classes which represent gates, as well as supporting functions """

import numpy as _np
import functools as _functools
from ..      import optimize as _opt
from ..tools import matrixtools as _mt

from .protectedarray import ProtectedArray as _ProtectedArray


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
    if isinstance(gateToOptimize, StaticGate):
        return #nothing to optimize

    if isinstance(gateToOptimize, FullyParameterizedGate):
        if(targetGate.dim != gateToOptimize.dim): #special case: gates can have different overall dimension
            gateToOptimize.dim = targetGate.dim   #  this is a HACK to allow model selection code to work correctly
        gateToOptimize.set_matrix(targetGate)     #just copy entire overall matrix since fully parameterized
        return

    assert(targetGate.dim == gateToOptimize.dim) #gates must have the same overall dimension
    targetMatrix = _np.asarray(targetGate)
    def objective_func(param_vec):
        gateToOptimize.from_vector(param_vec)
        return _mt.frobeniusnorm(gateToOptimize-targetMatrix)

    x0 = gateToOptimize.to_vector()
    minSol = _opt.minimize(objective_func, x0, method='BFGS', maxiter=10000, maxfev=10000,
                           tol=1e-6, callback=None)

    gateToOptimize.from_vector(minSol.x)
    print("DEBUG: optimized gate to min frobenius distance %g" % _mt.frobeniusnorm(gateToOptimize-targetMatrix))


def compose(gate1, gate2, parameterization="auto"):
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

    parameterization : {"auto","full","TP","linear","static"}, optional
        The parameterization of the resulting gates.  The default, "auto",
        attempts to convert to the most restrictive common parameterization.

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

    if parameterization == "auto":
        if any([isinstance(g, FullyParameterizedGate) for g in (gate1,gate2)]):
            paramType = "full"
        elif any([isinstance(g, TPParameterizedGate) for g in (gate1,gate2)]):
            paramType = "TP" #update to "full" below if TP-conversion
                             #not possible?
        elif any([isinstance(g, LinearlyParameterizedGate)
                  for g in (gate1,gate2)]):
            paramType = "linear"
        else:
            assert( isinstance(gate1, StaticGate)
                    and isinstance(gate2, StaticGate) )
            paramType = "static"
    else:
        paramType = parameterization #user-specified final parameterization

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

    toType : {"full","TP","linear","static"}
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

    elif toType == "TP":
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


def check_deriv_wrt_params(gate, deriv_to_check=None, eps=1e-7):
    """
    Checks the `deriv_wrt_params` method of a Gate object.

    This routine is meant to be used as an aid in testing and debugging
    gate classes by computing by finite-difference the Jacobian that
    should be returned by `gate.deriv_wrt_params` and comparing the
    two results.  A ValueError is raised if the two do not match.

    Parameters
    ----------
    gate : Gate
        The gate object to test.

    deriv_to_check : numpy.ndarray or None, optional
        If not None, the Jacobian to compare against the finite difference
        result.  If None, `gate.deriv_wrt_parms()` is used.  Setting this
        argument can be useful when the function is called *within* a Gate
        class's `deriv_wrt_params()` method itself as a part of testing.
        
    eps : float, optional
        The finitite difference step to use.

    Returns
    -------
    None
    """
    dim = gate.get_dimension()
    gate2 = gate.copy()
    p = gate.to_vector()
    fd_deriv = _np.empty((dim,dim,gate.num_params()), 'd') #assume real (?)

    for i in range(gate.num_params()):
        p_plus_dp = p.copy()
        p_plus_dp[i] += eps
        gate2.from_vector(p_plus_dp)
        fd_deriv[:,:,i] = (gate2-gate)/eps

    fd_deriv.shape = [dim**2,gate.num_params()]

    if deriv_to_check is None:
        deriv_to_check = gate.deriv_wrt_params()

    #print("fd_deriv = \n",fd_deriv)
    #print("an_deriv = \n",deriv_to_check)

    if _np.linalg.norm(fd_deriv - deriv_to_check) > 5*eps:
        raise ValueError("Failed check of deriv_wrt_params:\n" +
                         " norm diff = %g" % _np.linalg.norm(fd_deriv - deriv_to_check))
    



class Gate(object):
    """
    Excapulates a parameterization of a gate matrix.  This class is the
    common base class for all specific parameterizations of a gate.
    """

    def __init__(self, mx=None):
        """ Initialize a new Gate """
        self.base = mx
        self.dim = self.base.shape[0]

    def get_dimension(self):
        """ Return the dimension of the gate matrix. """
        return self.dim

    #Handled by derived classes
    #def __str__(self):
    #    s = "Gate with shape %s\n" % str(self.base.shape)
    #    s += _mt.mx_to_string(self.base, width=4, prec=2)
    #    return s


    #Pickle plumbing
    def __setstate__(self, state):
        self.__dict__.update(state)


    #Access to underlying ndarray
    def __getitem__( self, key ):
        return self.base.__getitem__(key)

    def __getslice__(self, i,j):
        return self.__getitem__(slice(i,j)) #Called for A[:]

    def __setitem__(self, key, val):
        return self.base.__setitem__(key,val)

    def __getattr__(self, attr):
        #use __dict__ so no chance for recursive __getattr__
        ret = getattr(self.__dict__['base'],attr)
        if(self.base.shape != (self.dim,self.dim)):
            raise ValueError("Cannot change shape of Gate")
        return ret


    #Mimic array behavior
    def __pos__(self):         return self.base
    def __neg__(self):         return -self.base
    def __abs__(self):         return abs(self.base)
    def __add__(self,x):       return self.base + x
    def __radd__(self,x):      return x + self.base
    def __sub__(self,x):       return self.base - x
    def __rsub__(self,x):      return x - self.base
    def __mul__(self,x):       return self.base * x
    def __rmul__(self,x):      return x * self.base
    def __truediv__(self, x):  return self.base / x
    def __rtruediv__(self, x): return x / self.base
    def __floordiv__(self,x):  return self.base // x
    def __rfloordiv__(self,x): return x // self.base
    def __pow__(self,x):       return self.base ** x
    def __eq__(self,x):        return self.base == x
    def __len__(self):         return len(self.base)
    def __int__(self):         return int(self.base)
    def __long__(self):        return int(self.base)
    def __float__(self):       return float(self.base)
    def __complex__(self):     return complex(self.base)



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
                d2  = len(M[0]) #pylint : disable=unused-variable
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



class StaticGate(Gate):
    """
    Encapsulates a gate matrix that is completely fixed, or "static", meaning
      that is contains no parameters.
    """

    def __init__(self, M):
        """
        Initialize a StaticGate object.

        Parameters
        ----------
        M : array_like or Gate
            a square 2D array-like or Gate object representing the gate action.
            The shape of M sets the dimension of the gate.
        """
        Gate.__init__(self, Gate.convert_to_matrix(M))


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
        self.base[:,:] = _np.array(mx)


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


    def deriv_wrt_params(self, wrtFilter=None):
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
        return StaticGate(self.base)


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
        self.set_matrix(_np.dot(Si,_np.dot(self.base, S)))


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
        return StaticGate( _np.dot( self.base, otherGate.base ) )


    def __str__(self):
        s = "Static gate with shape %s\n" % str(self.base.shape)
        s += _mt.mx_to_string(self.base, width=4, prec=2)
        return s

    def __reduce__(self):
        return (StaticGate, (_np.identity(self.dim,'d'),), self.__dict__)





class FullyParameterizedGate(Gate):
    """
    Encapsulates a gate matrix that is fully parameterized, that is,
      each element of the gate matrix is an independent parameter.
    """

    def __init__(self, M):
        """
        Initialize a FullyParameterizedGate object.

        Parameters
        ----------
        M : array_like or Gate
            a square 2D array-like or Gate object representing the gate action.
            The shape of M sets the dimension of the gate.
        """
        M2 = Gate.convert_to_matrix(M)
        Gate.__init__(self,M2)

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
        self.base[:,:] = _np.array(mx)


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
        return self.base.flatten() #.real in case of complex matrices?


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
        assert(self.base.shape == (self.dim, self.dim))
        self.base[:,:] = v.reshape( (self.dim,self.dim) )


    def deriv_wrt_params(self, wrtFilter=None):
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
        if wrtFilter is None:
            return derivMx
        else:
            return _np.take( derivMx, wrtFilter, axis=1 )


    def copy(self):
        """
        Copy this gate.

        Returns
        -------
        Gate
            A copy of this gate.
        """
        return FullyParameterizedGate(self.base)


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
        self.set_matrix(_np.dot(Si,_np.dot(self.base, S)))


    def __str__(self):
        s = "Fully Parameterized gate with shape %s\n" % str(self.base.shape)
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
        return FullyParameterizedGate( _np.dot( self.base, otherGate.base) )


    def __reduce__(self):
        return (FullyParameterizedGate, (_np.identity(self.dim,'d'),), self.__dict__)



class TPParameterizedGate(Gate):
    """
    Encapsulates a gate matrix that is fully parameterized except for
    the first row, which is frozen to be [1 0 ... 0] so that the action
    of the gate, when interpreted in the Pauli or Gell-Mann basis, is
    trace preserving (TP).
    """

    def __init__(self, M):
        """
        Initialize a TPParameterizedGate object.

        Parameters
        ----------
        M : array_like or Gate
            a square 2D numpy array representing the gate action.  The
            shape of this array sets the dimension of the gate.
        """
        #Gate.__init__(self, Gate.convert_to_matrix(M))
        mx = Gate.convert_to_matrix(M)
        if not (_np.isclose(mx[0,0], 1.0) and \
                _np.allclose(mx[0,1:], 0.0)):
            raise ValueError("Cannot create TPParameterizedGate: " +
                             "invalid form for 1st row!")
        Gate.__init__(self, _ProtectedArray(mx,
                       indicesToProtect=(0, slice(None,None,None))))


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
        self.base[1:,:] = mx[1:,:]


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
        return self.base.flatten()[self.dim:] #.real in case of complex matrices?


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
        assert(self.base.shape == (self.dim, self.dim))
        self.base[1:,:] = v.reshape((self.dim-1,self.dim))


    def deriv_wrt_params(self, wrtFilter=None):
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

        if wrtFilter is None:
            return derivMx
        else:
            return _np.take( derivMx, wrtFilter, axis=1 )


    def copy(self):
        """
        Copy this gate.

        Returns
        -------
        Gate
            A copy of this gate.
        """
        return TPParameterizedGate(self.base)


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
        self.set_matrix(_np.dot(Si,_np.dot(self.base, S)))


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
        return TPParameterizedGate( _np.dot( self.base, otherGate ) )


    def __str__(self):
        s = "TP Parameterized gate with shape %s\n" % str(self.base.shape)
        s += _mt.mx_to_string(self.base, width=4, prec=2)
        return s

    def __reduce__(self):
        return (TPParameterizedGate, (_np.identity(self.dim,'d'),), self.__dict__)



class LinearlyParameterizedElementTerm(object):
    def __init__(self, coeff=1.0, paramIndices=[]):
        self.coeff = coeff
        self.paramIndices = paramIndices

    def copy(self):
        return LinearlyParameterizedElementTerm(self.coeff, self.paramIndices)


class LinearlyParameterizedGate(Gate):
    """
    Encapsulates a gate matrix that is parameterized such that each
    element of the gate matrix depends only linearly on any parameter.
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
        for p,ij_tuples in list(parameterToBaseIndicesMap.items()):
            for i,j in ij_tuples:
                assert((i,j) not in elementExpressions) #only one parameter allowed per base index pair
                elementExpressions[(i,j)] = [ LinearlyParameterizedElementTerm(1.0, [p]) ]

        typ = "d" if real else "complex"
        mx = _np.empty( baseMatrix.shape, typ )
        self.baseMatrix = baseMatrix
        self.parameterArray = parameterArray
        self.numParams = len(parameterArray)
        self.elementExpressions = elementExpressions

        I = _np.identity(self.baseMatrix.shape[0],'d')
        self.leftTrans = leftTransform if (leftTransform is not None) else I
        self.rightTrans = rightTransform if (rightTransform is not None) else I
        self.enforceReal = real
        mx.flags.writeable = False # only _construct_matrix can change array

        Gate.__init__(self, mx)
        self._construct_matrix() # construct base from the parameters


    def _construct_matrix(self):
        """
        Build the internal gate matrix using the current parameters.
        """
        matrix = self.baseMatrix.copy()
        for (i,j),terms in self.elementExpressions.items():
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
        self.base = matrix
        self.base.flags.writeable = False


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
        return self.numParams


    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this gate.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
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
        self.parameterArray = v
        self._construct_matrix()


    def deriv_wrt_params(self, wrtFilter=None):
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
        #DEBUG - print expressions
        #for (i,j),terms in self.elementExpressions.items():
        #    tStr = ' + '.join([ '*'.join(["p%d"%p for p in term.paramIndices])
        #                        for term in terms] )
        #    print("Gate[%d,%d] = " % (i,j), tStr)
        
        derivMx = _np.zeros( (self.numParams, self.dim, self.dim), 'complex' )
        for (i,j),terms in self.elementExpressions.items():
            for term in terms:
                params_to_mult = [ self.parameterArray[p] for p in term.paramIndices ]
                for k,p in enumerate(term.paramIndices):
                    param_partial_prod = _np.prod( params_to_mult[0:k] + params_to_mult[k+1:] ) # exclude k-th factor
                    derivMx[p,i,j] += term.coeff * param_partial_prod

        dbg = derivMx.copy()
        dbg = _np.rollaxis(dbg,0,3) # now (d,d,P)
        dbg = dbg.reshape([self.dim**2, self.numParams]) # (d^2,P) == final shape
        #print("DEBUG pre-trans: \n",_np.real(dbg))

        #print("DEBUG left-trans: \n",self.leftTrans)
        #print("DEBUG right-trans: \n",self.rightTrans)

        tmp = _np.dot(derivMx, self.rightTrans)
        #print("SHAPES = ",self.leftTrans.shape,derivMx.shape,self.rightTrans.shape,tmp.shape)
        
        #print("DEBUG pre0: \n",derivMx[0,:,:])
        derivMx = _np.dot(self.leftTrans, _np.dot(derivMx, self.rightTrans)) # (d,d) * (P,d,d) * (d,d) => (d,P,d)
        #print("DEBUG post0: \n",derivMx[0,:,:],"\nshape = ",derivMx.shape)
        derivMx = _np.rollaxis(derivMx,1,3) # now (d,d,P)
        derivMx = derivMx.reshape([self.dim**2, self.numParams]) # (d^2,P) == final shape

        #print("DEBUG post-trans: \n",_np.real(derivMx))

        if self.enforceReal:
            assert(_np.linalg.norm(_np.imag(derivMx)) < 1e-8)
            derivMx = _np.real(derivMx)


        #DEBUG
        #check_deriv_wrt_params(self, derivMx)

        if wrtFilter is None:
            return derivMx
        else:
            return _np.take( derivMx, wrtFilter, axis=1 )


    def copy(self):

        #Construct new gate with no intial elementExpressions
        newGate = LinearlyParameterizedGate(self.baseMatrix, self.parameterArray,
                                            {}, self.leftTrans, self.rightTrans,
                                            self.enforceReal)

        #Deep copy elementExpressions into new gate
        for tup, termList in list(self.elementExpressions.items()):
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
        self._construct_matrix()


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


    def __str__(self):
        s = "Linearly Parameterized gate with shape %s, num params = %d\n" % \
            (str(self.base.shape), self.numParams)
        s += _mt.mx_to_string(self.base, width=5, prec=1)
        return s

    def __reduce__(self):
        return (LinearlyParameterizedGate,
                (self.baseMatrix, _np.array([]), {}, None, None, self.enforceReal),
                self.__dict__)



#Or CommutantParameterizedGate(Gate): ?
class EigenvalueParameterizedGate(Gate):
    """
    Encapsulates a real gate matrix that is parameterized only by its
    eigenvalues, which are assumed to be either real or to occur in
    conjugate pairs.  Thus, the number of parameters is equal to the
    number of eigenvalues.
    """

    def __init__(self, matrix, includeOffDiagsInDegen2Blocks=False,
                 TPconstrainedAndUnital=False):
        """
        Initialize a LinearlyParameterizedGate object.

        Parameters
        ----------
        matrix : numpy array
            a square 2D numpy array that gives the raw gate matrix to
            paramterize.  The shape of this array sets the dimension
            of the gate.

        includeOffDiagsInDegen2Blocks : bool
            If True, include as parameters the (initially zero) 
            off-diagonal elements in degenerate 2x2 blocks of the 
            the diagonalized gate matrix (no off-diagonals are 
            included in blocks larger than 2x2).  This is an option
            specifically used in the intelligent fiducial pair
            reduction (IFPR) algorithm.

        TPconstrainedAndUnital : bool
            If True, assume the top row of the gate matrix is fixed
            to [1, 0, ... 0] and should not be parameterized, and verify
            that the matrix is unital.  In this case, "1" is always a
            fixed (not-paramterized0 eigenvalue with eigenvector
            [1,0,...0] and if includeOffDiagsInDegen2Blocks is True
            any off diagonal elements lying on the top row are *not*
            parameterized as implied by the TP constraint.
        """

        def cmplx_compare(ia,ib):
            """Helper for comparing complex numbers"""
            a,b = evals[ia], evals[ib]
            if a.real < b.real:   return -1
            elif a.real > b.real: return  1
            elif a.imag < b.imag: return -1
            elif a.imag > b.imag: return  1
            else: return 0
        cmplx_compare_key = _functools.cmp_to_key(cmplx_compare)

        def isreal(a): #b/c numpy's isreal test for equality w/0
            return _np.isclose(_np.imag(a),0.0)

        # Since matrix is real, eigenvalues must either be real or occur in
        #  conjugate pairs.  Find and sort by conjugate pairs.

        assert(_np.linalg.norm(_np.imag(matrix)) < 1e-8 ) #matrix should be real
        evals, B = _np.linalg.eig(matrix) # matrix == B * diag(evals) * Bi
        dim = len(evals)

        #Sort eigenvalues & eigenvectors by:
        # 1) unit eigenvalues first (with TP eigenvalue first of all)
        # 2) non-unit real eigenvalues in order of magnitude
        # 3) complex eigenvalues in order of real then imaginary part

        unitInds = []; realInds = []; complexInds = []
        for i,ev in enumerate(evals):
            if _np.isclose(ev,1.0): unitInds.append(i)
            elif isreal(ev): realInds.append(i)
            else: complexInds.append(i)

        if TPconstrainedAndUnital:
            #check matrix is TP and unital
            unitRow = _np.zeros( (len(evals)), 'd'); unitRow[0] = 1.0
            assert( _np.allclose(matrix[0,:],unitRow) )
            assert( _np.allclose(matrix[:,0],unitRow) )

            #find the eigenvector with largest first element and make sure
            # this is the first index in unitInds
            k = _np.argmax([B[0,i] for i in unitInds])
            if k != 0:  #swap indices 0 <-> k in unitInds
                t = unitInds[0]; unitInds[0] = unitInds[k]; unitInds[k] = t

            #Assume we can recombine unit-eval eigenvectors so that the first
            # one (actually the closest-to-unit-row one) == unitRow and the
            # rest do not have any 0th component.
            iClose = _np.argmax( [abs(B[0,ui]) for ui in unitInds] )
            B[:, unitInds[iClose] ] = unitRow
            for i,ui in enumerate(unitInds):
                if i == iClose: continue
                B[0, ui] = 0.0; B[:,ui] /= _np.linalg.norm(B[:,ui])

        realInds = sorted(realInds, key=lambda i: -abs(evals[i]))
        complexInds = sorted(complexInds, key=cmplx_compare_key)
        new_ordering = unitInds + realInds + complexInds

        #Re-order the eigenvalues & vectors
        sorted_evals = _np.zeros(evals.shape,'complex')
        sorted_B = _np.zeros(B.shape, 'complex')
        for i,indx in enumerate(new_ordering):
            sorted_evals[i] = evals[indx]
            sorted_B[:,i] = B[:,indx]

        #Save the final list of (sorted) eigenvalues & eigenvectors
        self.evals = sorted_evals
        self.B = sorted_B
        self.Bi = _np.linalg.inv(sorted_B)

        self.options = { 'includeOffDiags': includeOffDiagsInDegen2Blocks,
                         'TPandUnital': TPconstrainedAndUnital }

        #Check that nothing has gone horribly wrong
        assert(_np.allclose(_np.dot(
                    self.B,_np.dot(_np.diag(self.evals),self.Bi)), matrix))
                
        #Build a list of parameter descriptors.  Each element of self.params
        # is a list of (prefactor, (i,j)) tuples.
        self.params = []        
        i = 0; N = len(self.evals); processed = [False]*N
        while i < N:
            if processed[i]:
                i += 1; continue

            # Find block (i -> j) of degenerate eigenvalues
            j = i+1
            while j < N and _np.isclose(self.evals[i],self.evals[j]): j += 1
            blkSize = j-i

            #Add eigenvalues as parameters
            ev = self.evals[i] #current eigenvalue being processed
            if isreal(ev):
                
                # Side task: for a *real* block of degenerate evals, we want
                # to ensure the eigenvectors are real, which numpy doesn't
                # always guarantee (could be conj. pairs for instance).

                # Solve or Cmx: [v1,v2,v3,v4]Cmx = [v1',v2',v3',v4'] ,
                # where ' qtys == real, so Im([v1,v2,v3,v4]Cmx) = 0
                # Let Cmx = Cr + i*Ci, v1 = v1.r + i*v1.i, etc.,
                #  then solve [v1.r, ...]Ci + [v1.i, ...]Cr = 0
                #  which can be cast as [Vr,Vi]*[Ci] = 0
                #                               [Cr]      (nullspace of V)
                # Note: only involve complex evecs (don't disturb TP evec!)            
                evecIndsToMakeReal = []
                for k in range(i,j): 
                    if _np.linalg.norm(self.B[:,k].imag) >= 1e-8:
                        evecIndsToMakeReal.append(k)

                nToReal = len(evecIndsToMakeReal)
                if nToReal > 0:
                    vecs = _np.empty( (dim,nToReal),'complex')
                    for ik,k in enumerate(evecIndsToMakeReal): 
                        vecs[:,ik] = self.B[:,k]
                    V = _np.concatenate((vecs.real, vecs.imag), axis=1)
                    nullsp = _mt.nullspace(V); assert(nullsp.shape[1] == nToReal)
                      #assert we can find enough real linear combos!
    
                    Cmx = nullsp[nToReal:,:] + 1j*nullsp[0:nToReal,:] # Cr + i*Ci
                    new_vecs = _np.dot(vecs,Cmx)
                    assert(_np.linalg.norm(new_vecs.imag) < 1e-8)
                    for ik,k in enumerate(evecIndsToMakeReal): 
                        self.B[:,k] = new_vecs[:,ik]
                    self.Bi = _np.linalg.inv(self.B)                

                
                #Now, back to constructing parameter descriptors...
                for k in range(i,j):
                    if TPconstrainedAndUnital and k == 0: continue
                    prefactor = 1.0; mx_indx = (k,k)
                    self.params.append( [(prefactor, mx_indx)] )
                    processed[k] = True
            else:
                iConjugate = {}
                for k in range(i,j):
                    #Find conjugate eigenvalue to eval[k]
                    conj = _np.conj(self.evals[k]) # == conj(ev), indep of k
                    conjB = _np.conj(self.B[:,k])
                    for l in range(j,N):
                        if _np.isclose(conj,self.evals[l]) and \
                                _np.allclose(conjB, self.B[:,l]):
                            self.params.append( [  # real-part param
                                    (1.0, (k,k)),  # (prefactor, index)
                                    (1.0, (l,l)) ] ) 
                            self.params.append( [  # imag-part param
                                    (1j, (k,k)),  # (prefactor, index)
                                    (-1j, (l,l)) ] ) 
                            processed[k] = processed[l] = True
                            iConjugate[k] = l #save conj. pair index for below
                            break
                    else:
                        raise ValueError("Could not find conjugate pair " 
                                         + " for %s" % self.evals[k])
            
            
            if includeOffDiagsInDegen2Blocks and blkSize == 2:
                #Note: could remove blkSize == 2 condition or make this a 
                # separate option.  This is useful currently so that we don't
                # add lots of off-diag elements in accidentally-degenerate 
                # cases, but there's probabaly a better heuristic for this, such
                # as only including off-diag els for unit-eigenvalue blocks
                # of size 2 (?)
                for k1 in range(i,j-1):
                    for k2 in range(k1+1,j):
                        if isreal(ev):
                            # k1,k2 element
                            if not TPconstrainedAndUnital or k1 != 0:
                                self.params.append( [(1.0, (k1,k2))] )

                            # k2,k1 element
                            if not TPconstrainedAndUnital or k2 != 0:
                                self.params.append( [(1.0, (k2,k1))] )
                        else:
                            k1c, k2c = iConjugate[k1],iConjugate[k2]

                            # k1,k2 element
                            self.params.append( [  # real-part param
                                    (1.0, (k1,k2)),
                                    (1.0, (k1c,k2c)) ] ) 
                            self.params.append( [  # imag-part param
                                    (1j, (k1,k2)),
                                    (-1j, (k1c,k2c)) ] ) 

                            # k2,k1 element
                            self.params.append( [  # real-part param
                                    (1.0, (k2,k1)),
                                    (1.0, (k2c,k1c)) ] ) 
                            self.params.append( [  # imag-part param
                                    (1j, (k2,k1)),
                                    (-1j, (k2c,k1c)) ] )

            i = j #advance to next block

        #Allocate array of parameter values (all zero initially)
        self.paramvals = _np.zeros(len(self.params), 'd')

        #Finish Gate construction
        mx = _np.empty( matrix.shape, "d" )
        mx.flags.writeable = False # only _construct_matrix can change array
        Gate.__init__(self, mx)
        self._construct_matrix() # construct base from the parameters


    def _construct_matrix(self):
        """
        Build the internal gate matrix using the current parameters.
        """
        base_diag = _np.diag(self.evals)
        for pdesc,pval in zip(self.params, self.paramvals):
            for prefactor,(i,j) in pdesc:
                base_diag[i,j] += prefactor * pval
        matrix = _np.dot(self.B,_np.dot(base_diag,self.Bi))
        assert(_np.linalg.norm(matrix.imag) < 1e-8)
        assert(matrix.shape == (self.dim,self.dim))
        self.base = matrix.real
        self.base.flags.writeable = False

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
        mx = Gate.convert_to_matrix(M)
        if(mx.shape != (self.dim, self.dim)):
            raise ValueError("Argument must be a (%d,%d) matrix!" % (self.dim,self.dim))
        raise ValueError("Currently, directly setting the matrix of an" +
                         " EigenvalueParameterizedGate object is not" +
                         " supported.  Please use the from_vector method.")

    def num_params(self):
        """
        Get the number of independent parameters which specify this gate.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return len(self.paramvals)


    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this gate.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self.paramvals


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
        assert(len(v) == self.num_params())
        self.paramvals = v
        self._construct_matrix()


    def deriv_wrt_params(self, wrtFilter=None):
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
        
        # matrix = B * diag * Bi, and only diag depends on parameters
        #  (and only linearly), so:
        # d(matrix)/d(param) = B * d(diag)/d(param) * Bi

        derivMx = _np.zeros( (self.dim**2, self.num_params()), 'd' )

        # Compute d(diag)/d(param) for each params, then apply B & Bi
        for k,(pdesc,pval) in enumerate(zip(self.params, self.paramvals)):
            dMx = _np.zeros( (self.dim,self.dim), 'complex')
            for prefactor,(i,j) in pdesc:
                dMx[i,j] = prefactor
            tmp = _np.dot(self.B, _np.dot(dMx, self.Bi))
            assert(_np.linalg.norm(tmp.imag) < 1e-8)
            derivMx[:,k] = tmp.real.flatten()

        if wrtFilter is None:
            return derivMx
        else:
            return _np.take( derivMx, wrtFilter, axis=1 )


    def copy(self):

        #Construct new gate with dummy identity mx
        newGate = EigenvalueParameterizedGate(_np.identity(self.dim,'d'))
        
        #Deep copy data
        newGate.evals = self.evals.copy()
        newGate.B = self.B.copy()
        newGate.Bi = self.Bi.copy()
        newGate.params = self.params[:] #copies tuple elements
        newGate.paramvals = self.paramvals.copy()
        newGate.options = self.options.copy()
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
        if Si is None: Si = _np.linalg.inv(S)
        self.B = _np.dot(Si,self.B)
        self.Bi = _np.dot(self.Bi,S)
        self._construct_matrix()


    def compose(self, otherGate):
        """
        Create and return a new gate that is the composition of this gate
        followed by otherGate, which *must be another 
        EigenvalueParameterizedGate*.  (For more general compositions between
        different types of gates, use the module-level compose function.)  The
        returned gate's matrix is equal to dot(this, otherGate).

        Parameters
        ----------
        otherGate : EigenvalueParameterizedGate
            The gate to compose to the right of this one.

        Returns
        -------
        EigenvalueParameterizedGate
        """
        assert( isinstance(otherGate, EigenvalueParameterizedGate) )

        composed_mx = _np.dot(self, otherGate)
        return EigenvalueParameterizedGate(composed_mx,
                                           self.options['includeOffDiags'],
                                           self.options['TPandUnital'])

    def __str__(self):
        s = "Eigenvalue Parameterized gate with shape %s, num params = %d\n" % \
            (str(self.base.shape), self.num_params())
        s += _mt.mx_to_string(self.base, width=5, prec=1)
        return s

    def __reduce__(self):
        return (EigenvalueParameterizedGate, 
                (_np.identity(self.dim,'d'),), self.__dict__)




#SCRATCH: TO REMOVE



#    def __getattr__(self, attr):
#        print "GATE ATTR",attr
#        ret = getattr(self.__dict__['base'],attr) #use __dict__ so no chance for recursive __getattr__
#        if(self.base.shape != (self.dim,self.dim)):
#           raise ValueError("Cannot change shape of Gate")
#        if isinstance(ret, _ProtectedArray):
#            print "PA RET"
#            if ret.base is self.base:
#                print "SAME MEM.  Writeable = ",ret.flags.writeable
#                print " Dict = ",ret.__dict__
#                if ret.flags.writeable == True and ret.indicesToProtect is None:
#                    print "SETTING READ-ONLY"
#                    ret.flags.writeable = False
#        #if getattr(ret,'base',None) is not self.base: #if doesn't share memory with parent
#        #    if hasattr(ret,'flags'):
#        #        print "ALLOWING WRITE"
#        #        ret.flags.writeable = True           # don't preserve read-only
#        return ret



    #def __reduce__(self):
    #    """ Pickle plumbing. """
    #    createFn, args, state = _np.ndarray.__reduce__(self)
    #    new_state = state + (self.__dict__,)
    #    return (createFn, args, new_state)
    #
    #def __setstate__(self, state):
    #    """ Pickle plumbing. """
    #    _np.ndarray.__setstate__(self,state[0:-1])
    #    self.__dict__.update(state[-1])


#    def __lt__(self,x):
#
#
#    def __gt__(self,x):
#
#
#    def __hash__(self):
#
#
#    def __copy__(self):
#
#
#    def __deepcopy__(self):
