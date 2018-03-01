""" Defines classes which represent gates, as well as supporting functions """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import scipy.linalg as _spl
import scipy.sparse as _sps
import scipy.sparse.linalg as _spsl
import functools as _functools
import itertools as _itertools
import copy as _copy
import warnings as _warnings

from ..      import optimize as _opt
from ..tools import matrixtools as _mt
from ..tools import gatetools as _gt
from ..tools import jamiolkowski as _jt
from ..tools import basistools as _bt
from ..tools import slicetools as _slct
from ..tools import compattools as _compat
from . import gaugegroup as _gaugegroup
from . import gatesetmember as _gatesetmember
from ..baseobjs import ProtectedArray as _ProtectedArray
from ..baseobjs import Basis as _Basis

IMAG_TOL = 1e-7 #tolerance for imaginary part being considered zero

try:
    #import pyximport; pyximport.install(setup_args={'include_dirs': _np.get_include()}) # develop-mode
    from ..tools import fastcalc as _fastcalc
except ImportError:
    _fastcalc = None


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
        gateToOptimize.set_value(targetGate)     #just copy entire overall matrix since fully parameterized
        return

    assert(targetGate.dim == gateToOptimize.dim) #gates must have the same overall dimension
    targetMatrix = _np.asarray(targetGate)
    def _objective_func(param_vec):
        gateToOptimize.from_vector(param_vec)
        return _mt.frobeniusnorm(gateToOptimize - targetMatrix)

    x0 = gateToOptimize.to_vector()
    minSol = _opt.minimize(_objective_func, x0, method='BFGS', maxiter=10000, maxfev=10000,
                           tol=1e-6, callback=None)

    gateToOptimize.from_vector(minSol.x)
    #print("DEBUG: optimized gate to min frobenius distance %g" %
    #      _mt.frobeniusnorm(gateToOptimize-targetMatrix))


def compose(gate1, gate2, basis, parameterization="auto"):
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

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

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
    cgate1 = convert(gate1, paramType, basis)
    cgate2 = convert(gate2, paramType, basis)

    # cgate1 and cgate2 are the same type, so can invoke the gate's compose method
    return cgate1.compose(cgate2)



def convert(gate, toType, basis):
    """
    Convert gate to a new type of parameterization, potentially creating
    a new Gate object.  Raises ValueError for invalid conversions.

    Parameters
    ----------
    gate : Gate
        Gate to convert

    toType : {"full", "TP", "CPTP", "H+S", "S", "static", "GLND"}
        The type of parameterizaton to convert to.

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for `gate`.  Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
        (or a custom basis object).

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

    elif toType in ("CPTP","H+S","S","GLND"):
        RANK_TOL = 1e-6        
        J = _jt.fast_jamiolkowski_iso_std(gate, basis) #Choi mx basis doesn't matter
        if _np.linalg.matrix_rank(J, RANK_TOL) == 1: 
            unitary_post = gate # when 'gate' is unitary
        else: unitary_post = None

        nQubits = _np.log2(gate.dim)/2.0
        bQubits = bool(abs(nQubits-round(nQubits)) < 1e-10) #integer # of qubits?
        
        proj_basis = "pp" if (basis == "pp" or bQubits) else basis
        ham_basis = proj_basis if toType in ("CPTP","H+S") else None
        nonham_basis = proj_basis
        nonham_diagonal_only = bool(toType in ("H+S","S") )
        cptp = False if toType == "GLND" else True #only "General LiNDbladian" is non-cptp
        truncate=True

        def beq(b1,b2):
            """ Check if bases have equal names """
            b1 = b1.name if isinstance(b1,_Basis) else b1
            b2 = b2.name if isinstance(b2,_Basis) else b2
            return b1 == b2

        if isinstance(gate, LindbladParameterizedGate) and \
           _np.linalg.norm(gate.unitary_postfactor-unitary_post) < 1e-6 \
           and beq(ham_basis,gate.ham_basis) and beq(nonham_basis,gate.other_basis) \
           and cptp==gate.cptp and nonham_diagonal_only==gate.nonham_diagonal_only \
           and beq(basis,gate.matrix_basis):
            return gate #no conversion necessary
        else:
            return LindbladParameterizedGate(
                gate, unitary_post, ham_basis, nonham_basis, cptp,
                nonham_diagonal_only, truncate, basis)
        

    elif toType == "static":
        if isinstance(gate, StaticGate):
            return gate #no conversion necessary
        else:
            return StaticGate( gate )

    else:
        raise ValueError("Invalid toType argument: %s" % toType)


def finite_difference_deriv_wrt_params(gate, eps=1e-7):
    """
    Computes a finite-difference Jacobian for a Gate object.

    The returned value is a matrix whose columns are the vectorized
    derivatives of the flattened gate matrix with respect to a single
    gate parameter, matching the format expected from the gate's
    `deriv_wrt_params` method.

    Parameters
    ----------
    gate : Gate
        The gate object to compute a Jacobian for.
        
    eps : float, optional
        The finite difference step to use.

    Returns
    -------
    numpy.ndarray
        An M by N matrix where M is the number of gate elements and
        N is the number of gate parameters. 
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
    return fd_deriv


def check_deriv_wrt_params(gate, deriv_to_check=None, eps=1e-7):
    """
    Checks the `deriv_wrt_params` method of a Gate object.

    This routine is meant to be used as an aid in testing and debugging
    gate classes by comparing the finite-difference Jacobian that
    *should* be returned by `gate.deriv_wrt_params` with the one that
    actually is.  A ValueError is raised if the two do not match.

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
        The finite difference step to use.

    Returns
    -------
    None
    """
    fd_deriv = finite_difference_deriv_wrt_params(gate, eps)
    if deriv_to_check is None:
        deriv_to_check = gate.deriv_wrt_params()

    #print("Deriv shapes = %s and %s" % (str(fd_deriv.shape),
    #                                    str(deriv_to_check.shape)))
    #print("finite difference deriv = \n",fd_deriv)
    #print("deriv_wrt_params deriv = \n",deriv_to_check)
    #print("deriv_wrt_params - finite diff deriv = \n",
    #      deriv_to_check - fd_deriv)
    for i in range(deriv_to_check.shape[0]):
        for j in range(deriv_to_check.shape[1]):
            diff = abs(deriv_to_check[i,j] - fd_deriv[i,j])
            if diff > 10*eps:
                print("deriv_chk_mismatch: (%d,%d): %g (comp) - %g (fd) = %g" %
                      (i,j,deriv_to_check[i,j],fd_deriv[i,j],diff)) # pragma: no cover

    if _np.linalg.norm(fd_deriv - deriv_to_check)/fd_deriv.size > 10*eps:
        raise ValueError("Failed check of deriv_wrt_params:\n" +
                         " norm diff = %g" % 
                         _np.linalg.norm(fd_deriv - deriv_to_check)) # pragma: no cover


#Note on initialization sequence of Gates within a GateSet:
# 1) a GateSet is constructed (empty)
# 2) a Gate is constructed - apart from a GateSet if it's locally parameterized,
#    otherwise with explicit reference to an existing GateSet's labels/indices.
#    All gates (GateSetMember objs in general) have a "gpindices" member which
#    can either be initialized upon construction or set to None, which signals
#    that the GateSet must initialize it.
# 3) the Gate is assigned/added to a dict within the GateSet.  As a part of this
#    process, the Gate's 'gpindices' member is set, if it isn't already, and the
#    GateSet's "global" parameter vector (and number of params) is updated as
#    needed to accomodate new parameters.
#
# Note: gpindices may be None (before initialization) or any valid index
#  into a 1D numpy array (e.g. a slice or integer array).  It may NOT have
#  any repeated elements.
#
# When a Gate is removed from the GateSet, parameters only used by it can be
# removed from the GateSet, and the gpindices members of existing gates
# adjusted as needed.
#
# When derivatives are taken wrt. a gateset parameter (1 col of a jacobian)
# derivatives wrt each gate that includes that parameter in its gpindices
# must be processed.


class Gate(_gatesetmember.GateSetMember):
    """ Base class for all gate representations """
    
    def __init__(self, dim):
        """ Initialize a new Gate """
        super(Gate, self).__init__(dim)
        
    def acton(self, state):
        """ Act this gate map on an input state """
        raise NotImplementedError("acton(...) not implemented!")

    def transform(self, S):
        """ Update gate G with inv(S) * G * S."""
        raise NotImplementedError("This gate cannot be transform()'d")

    def depolarize(self, amount):
        """ Depolarize gate by the given amount. """
        raise NotImplementedError("This gate cannot be depolarize()'d")

    def rotate(self, amount, mxBasis="gm"):
        """ Rotate gate by the given amount. """
        raise NotImplementedError("This gate cannot be rotate()'d")

    def frobeniusdist2(self, otherGate, transform=None, inv_transform=None):
        """ 
        Return the squared frobenius distance between this gate
        and `otherGate`, optionally transforming this gate first
        using `transform` and `inv_transform`.
        """
        raise NotImplementedError("frobeniusdist2(...) not implemented!")

    def frobeniusdist(self, otherGate, transform=None, inv_transform=None):
        """ 
        Return the frobenius distance between this gate
        and `otherGate`, optionally transforming this gate first
        using `transform` and `inv_transform`.
        """
        return _np.sqrt(self.frobeniusdist2(otherGate, transform, inv_transform))

    def jtracedist(self, otherGate, transform=None, inv_transform=None):
        """ 
        Return the Jamiolkowski trace distance between this gate
        and `otherGate`, optionally transforming this gate first
        using `transform` and `inv_transform`.
        """
        raise NotImplementedError("jtracedist(...) not implemented!")

    def diamonddist(self, otherGate, transform=None, inv_transform=None):
        """ 
        Return the diamon distance between this gate
        and `otherGate`, optionally transforming this gate first
        using `transform` and `inv_transform`.
        """
        raise NotImplementedError("diamonddist(...) not implemented!")

    #Pickle plumbing
    def __setstate__(self, state):
        self.__dict__.update(state)


class GateMap(Gate):
    def __init__(self, dim):
        """ Initialize a new Gate """
        super(GateMap, self).__init__(dim)

    @property
    def size(self):
        return (self.dim)**2

    #Maybe add an as_sparse_mx function and compute
    # metrics using this?
    #And perhaps a finite-difference deriv_wrt_params?

    
class GateMatrix(Gate):
    """
    Excapulates a parameterization of a gate matrix.  This class is the
    common base class for all specific parameterizations of a gate.
    """

    def __init__(self, mx=None):
        """ Initialize a new Gate """
        self.base = mx
        super(GateMatrix, self).__init__(self.base.shape[0])
        
    def acton(self, state):
        """ Act this gate matrix on an input state (left-multiply w/matrix) """
        return _np.dot(self.base, state) #TODO: return a SPAMVec?

    def frobeniusdist2(self, otherGate, transform=None, inv_transform=None):
        """ 
        Return the squared frobenius difference between this gate and
        `otherGate`, optionally transforming this gate first using matrices
        `transform` and `inv_transform`.
        """
        if transform is None and inv_transform is None:
            return _gt.frobeniusdist2(self.base,otherGate)
        else:
            return _gt.frobeniusdist2(_np.dot(
                    inv_transform,_np.dot(self.base,transform)),
                    otherGate)

    def residuals(self, otherGate, transform=None, inv_transform=None):
        """
        The per-element difference between this `GateMatrix` and `otherGate`,
        possibly after transforming this gate as 
        `G => inv_transform * G * transform`.

        Parameters
        ----------
        otherGate : GateMatrix
            The gate to compare against.

        transform, inv_transform : numpy.ndarray, optional
            The transform and its inverse, respectively, to apply before
            taking the element-wise difference.

        Returns
        -------
        numpy.ndarray
            A 1D-array of size equal to that of the flattened gate matrix.
        """
        if transform is None and inv_transform is None:
            return _gt.residuals(self.base,otherGate)
        else:
            return _gt.residuals(_np.dot(
                    inv_transform,_np.dot(self.base,transform)),
                    otherGate)

    def jtracedist(self, otherGate, transform=None, inv_transform=None):
        """ 
        Return the Jamiolkowski trace distance between this gate
        and `otherGate`, optionally transforming this gate first
        using `transform` and `inv_transform`.
        """
        if transform is None and inv_transform is None:
            return _gt.jtracedist(self.base,otherGate)
        else:
            return _gt.jtracedist(_np.dot(
                    inv_transform,_np.dot(self.base,transform)),
                    otherGate)

    def diamonddist(self, otherGate, transform=None, inv_transform=None):
        """ 
        Return the diamon distance between this gate
        and `otherGate`, optionally transforming this gate first
        using `transform` and `inv_transform`.
        """
        if transform is None and inv_transform is None:
            return _gt.diamonddist(self.base,otherGate)
        else:
            return _gt.diamonddist(_np.dot(
                    inv_transform,_np.dot(self.base,transform)),
                    otherGate)

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
        raise NotImplementedError("deriv_wrt_params(...) is not implemented")

    def has_nonzero_hessian(self):
        """ 
        Returns whether this gate has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return True #conservative

    def hessian_wrt_params(self, wrtFilter1=None, wrtFilter2=None):
        """
        Construct the Hessian of this gate with respect to its parameters.

        This function returns a tensor whose first axis corresponds to the
        flattened gate matrix and whose 2nd and 3rd axes correspond to the
        parameters that are differentiated with respect to.

        Parameters
        ----------
        wrtFilter1, wrtFilter2 : list
            Lists of indices of the paramters to take first and second
            derivatives with respect to.  If None, then derivatives are
            taken with respect to all of the gate's parameters.

        Returns
        -------
        numpy array
            Hessian with shape (dimension^2, num_params1, num_params2)
        """
        raise NotImplementedError("hessian_wrt_params(...) is not implemented")


    #Handled by derived classes
    #def __str__(self):
    #    s = "Gate with shape %s\n" % str(self.base.shape)
    #    s += _mt.mx_to_string(self.base, width=4, prec=2)
    #    return s

    #Access to underlying ndarray
    def __getitem__( self, key ):
        self.dirty = True
        return self.base.__getitem__(key)

    def __getslice__(self, i,j):
        self.dirty = True
        return self.__getitem__(slice(i,j)) #Called for A[:]

    def __setitem__(self, key, val):
        self.dirty = True
        return self.base.__setitem__(key,val)

    def __getattr__(self, attr):
        #use __dict__ so no chance for recursive __getattr__
        ret = getattr(self.__dict__['base'],attr)
        self.dirty = True
        return ret

    def set_value(self, M):
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
        raise ValueError("Cannot set the value of this GateMatrix directly!")


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

        if matrix.shape[0] != matrix.shape[1]: # checked above, but just to be safe
            raise ValueError("%s is not a *square* 2D array" % M) # pragma: no cover

        return matrix



class StaticGate(GateMatrix):
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
        GateMatrix.__init__(self, GateMatrix.convert_to_matrix(M))


    def set_value(self, M):
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
        mx = GateMatrix.convert_to_matrix(M)
        if(mx.shape != (self.dim, self.dim)):
            raise ValueError("Argument must be a (%d,%d) matrix!"
                             % (self.dim,self.dim))
        self.base[:,:] = _np.array(mx)
        self.dirty = True


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
        derivMx = _np.zeros((self.dim**2,0),'d')
        if wrtFilter is None:
            return derivMx
        else:
            return _np.take( derivMx, wrtFilter, axis=1 )

        
    def has_nonzero_hessian(self):
        """ 
        Returns whether this gate has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return False


    def copy(self, parent=None):
        """
        Copy this gate.

        Returns
        -------
        Gate
            A copy of this gate.
        """
        return self._copy_gpindices( StaticGate(self.base), parent)


    def transform(self, S):
        """
        Update gate matrix G with inv(S) * G * S.

        Generally, the transform function updates the *parameters* of 
        the gate such that the resulting gate matrix is altered as 
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        In this particular case *no* transforms are possible since a
        StaticGate has no parameters to modify.  Thus, this function
        *always* raises a ValueError.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.
        """
        raise ValueError("Cannot transform a StaticGate - it has no parameters!")
        #self.set_value(_np.dot(Si,_np.dot(self.base, S)))

    def depolarize(self, amount):
        """
        Depolarize this gate by the given `amount`.

        Generally, the depolarize function updates the *parameters* of 
        the gate such that the resulting gate matrix is depolarized.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        In this particular case *no* alterations are possible since a
        StaticGate has no parameters to modify.  Thus, this function
        *always* raises a ValueError.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. In standard
            bases, depolarization corresponds to multiplying the gate matrix
            by a diagonal matrix whose first diagonal element (corresponding
            to the identity) equals 1.0 and whose subsequent elements 
            (corresponding to non-identity basis elements) equal
            `1.0 - amount[i]` (or just `1.0 - amount` if `amount` is a
            float).

        Returns
        -------
        None
        """
        raise ValueError("Cannot depolarize a StaticGate - it has no parameters!")


    def rotate(self, amount, mxBasis="gm"):
        """
        Rotate this gate by the given `amount`.

        Generally, the rotate function updates the *parameters* of 
        the gate such that the resulting gate matrix is rotated.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        In this particular case *no* alterations are possible since a
        StaticGate has no parameters to modify.  Thus, this function
        *always* raises a ValueError.

        Parameters
        ----------
        amount : tuple of floats, optional
            Specifies the rotation "coefficients" along each of the non-identity
            Pauli-product axes.  The gate's matrix `G` is composed with a
            rotation operation `R`  (so `G` -> `dot(R, G)` ) where `R` is the
            unitary superoperator corresponding to the unitary operator 
            `U = exp( sum_k( i * rotate[k] / 2.0 * Pauli_k ) )`.  Here `Pauli_k`
            ranges over all of the non-identity un-normalized Pauli operators.

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        Returns
        -------
        None
        """
        raise ValueError("Cannot rotate a StaticGate - it has no parameters!")


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



class FullyParameterizedGate(GateMatrix):
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
        M2 = GateMatrix.convert_to_matrix(M)
        GateMatrix.__init__(self,M2)

    def set_value(self, M):
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
        mx = GateMatrix.convert_to_matrix(M)
        if(mx.shape != (self.dim, self.dim)):
            raise ValueError("Argument must be a (%d,%d) matrix!"
                             % (self.dim,self.dim))
        self.base[:,:] = _np.array(mx)
        self.dirty = True


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
        self.dirty = True


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

        
    def has_nonzero_hessian(self):
        """ 
        Returns whether this gate has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return False


    def copy(self, parent=None):
        """
        Copy this gate.

        Returns
        -------
        Gate
            A copy of this gate.
        """
        return self._copy_gpindices( FullyParameterizedGate(self.base), parent )


    def transform(self, S):
        """
        Update gate matrix G with inv(S) * G * S,

        Generally, the transform function updates the *parameters* of 
        the gate such that the resulting gate matrix is altered as 
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        In this particular case *any* transform of the appropriate
        dimension is possible, since all gate matrix elements are parameters.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.
        """
        Smx = S.get_transform_matrix()
        Si  = S.get_transform_matrix_inverse()
        self.set_value(_np.dot(Si,_np.dot(self.base, Smx)))


    def depolarize(self, amount):
        """
        Depolarize this gate by the given `amount`.

        Generally, the depolarize function updates the *parameters* of 
        the gate such that the resulting gate matrix is depolarized.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. In standard
            bases, depolarization corresponds to multiplying the gate matrix
            by a diagonal matrix whose first diagonal element (corresponding
            to the identity) equals 1.0 and whose subsequent elements 
            (corresponding to non-identity basis elements) equal
            `1.0 - amount[i]` (or just `1.0 - amount` if `amount` is a
            float).

        Returns
        -------
        None
        """
        if isinstance(amount,float):
            D = _np.diag( [1]+[1-amount]*(self.dim-1) )
        else:
            assert(len(amount) == self.dim-1)
            D = _np.diag( [1]+list(1.0 - _np.array(amount,'d')) )
        self.set_value(_np.dot(D,self))


    def rotate(self, amount, mxBasis="gm"):
        """
        Rotate this gate by the given `amount`.

        Generally, the rotate function updates the *parameters* of 
        the gate such that the resulting gate matrix is rotated.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : tuple of floats, optional
            Specifies the rotation "coefficients" along each of the non-identity
            Pauli-product axes.  The gate's matrix `G` is composed with a
            rotation operation `R`  (so `G` -> `dot(R, G)` ) where `R` is the
            unitary superoperator corresponding to the unitary operator 
            `U = exp( sum_k( i * rotate[k] / 2.0 * Pauli_k ) )`.  Here `Pauli_k`
            ranges over all of the non-identity un-normalized Pauli operators.

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        Returns
        -------
        None
        """
        rotnMx = _gt.rotation_gate_mx(amount,mxBasis)
        self.set_value(_np.dot(rotnMx,self))


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
    

class TPParameterizedGate(GateMatrix):
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
        mx = GateMatrix.convert_to_matrix(M)
        if not (_np.isclose(mx[0,0], 1.0) and \
                _np.allclose(mx[0,1:], 0.0)):
            raise ValueError("Cannot create TPParameterizedGate: " +
                             "invalid form for 1st row!")
        GateMatrix.__init__(self, _ProtectedArray(mx,
                       indicesToProtect=(0, slice(None,None,None))))


    def set_value(self, M):
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
        mx = GateMatrix.convert_to_matrix(M)
        if(mx.shape != (self.dim, self.dim)):
            raise ValueError("Argument must be a (%d,%d) matrix!"
                             % (self.dim,self.dim))
        if not (_np.isclose(mx[0,0], 1.0) and _np.allclose(mx[0,1:], 0.0)):
            raise ValueError("Cannot set TPParameterizedGate: " +
                             "invalid form for 1st row!" )
            #For further debugging:  + "\n".join([str(e) for e in mx[0,:]])
        self.base[1:,:] = mx[1:,:]
        self.dirty = True


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
        self.dirty = True


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


    def has_nonzero_hessian(self):
        """ 
        Returns whether this gate has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return False


    def copy(self, parent=None):
        """
        Copy this gate.

        Returns
        -------
        Gate
            A copy of this gate.
        """
        return self._copy_gpindices( TPParameterizedGate(self.base), parent )


    def transform(self, S):
        """
        Update gate matrix G with inv(S) * G * S,

        Generally, the transform function updates the *parameters* of 
        the gate such that the resulting gate matrix is altered as 
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        In this particular case any TP gauge transformation is possible,
        i.e. when `S` is an instance of `TPGaugeGroupElement` or 
        corresponds to a TP-like transform matrix.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.
        """
        Smx = S.get_transform_matrix()
        Si  = S.get_transform_matrix_inverse()
        self.set_value(_np.dot(Si,_np.dot(self.base, Smx)))


    def depolarize(self, amount):
        """
        Depolarize this gate by the given `amount`.

        Generally, the depolarize function updates the *parameters* of 
        the gate such that the resulting gate matrix is depolarized.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. In standard
            bases, depolarization corresponds to multiplying the gate matrix
            by a diagonal matrix whose first diagonal element (corresponding
            to the identity) equals 1.0 and whose subsequent elements 
            (corresponding to non-identity basis elements) equal
            `1.0 - amount[i]` (or just `1.0 - amount` if `amount` is a
            float).

        Returns
        -------
        None
        """
        if isinstance(amount,float):
            D = _np.diag( [1]+[1-amount]*(self.dim-1) )
        else:
            assert(len(amount) == self.dim-1)
            D = _np.diag( [1]+list(1.0 - _np.array(amount,'d')) )
        self.set_value(_np.dot(D,self))


    def rotate(self, amount, mxBasis="gm"):
        """
        Rotate this gate by the given `amount`.

        Generally, the rotate function updates the *parameters* of 
        the gate such that the resulting gate matrix is rotated.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : tuple of floats, optional
            Specifies the rotation "coefficients" along each of the non-identity
            Pauli-product axes.  The gate's matrix `G` is composed with a
            rotation operation `R`  (so `G` -> `dot(R, G)` ) where `R` is the
            unitary superoperator corresponding to the unitary operator 
            `U = exp( sum_k( i * rotate[k] / 2.0 * Pauli_k ) )`.  Here `Pauli_k`
            ranges over all of the non-identity un-normalized Pauli operators.

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        Returns
        -------
        None
        """
        rotnMx = _gt.rotation_gate_mx(amount,mxBasis)
        self.set_value(_np.dot(rotnMx,self))


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
        """ Return string representation """
        s = "TP Parameterized gate with shape %s\n" % str(self.base.shape)
        s += _mt.mx_to_string(self.base, width=4, prec=2)
        return s


class LinearlyParameterizedElementTerm(object):
    """
    Encapsulates a single term within a LinearlyParameterizedGate.
    """
    def __init__(self, coeff=1.0, paramIndices=[]):
        """
        Create a new LinearlyParameterizedElementTerm

        Parameters
        ----------
        coeff : float
            The term's coefficient

        paramIndices : list
            A list of integers, specifying which parameters are muliplied
            together (and finally, with `coeff`) to form this term.
        """
        self.coeff = coeff
        self.paramIndices = paramIndices

    def copy(self, parent=None):
        """ Copy this term. """
        return LinearlyParameterizedElementTerm(self.coeff, self.paramIndices[:])


class LinearlyParameterizedGate(GateMatrix):
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
            no transform.

        rightTransform : numpy array or None, optional
            A 2D array of the same shape as basematrix which right-multiplies
            the base matrix after parameters have been evaluated.  Defaults to
            no transform.

        real : bool, optional
            Whether or not the resulting gate matrix, after all
            parameter evaluation and left & right transforms have
            been performed, should be real.  If True, ValueError will
            be raised if the matrix contains any complex or imaginary
            elements.
        """

        baseMatrix = _np.array( GateMatrix.convert_to_matrix(baseMatrix), 'complex')
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

        GateMatrix.__init__(self, mx)
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
            if _np.linalg.norm(_np.imag(matrix)) > IMAG_TOL:
                raise ValueError("Linearly parameterized matrix has non-zero" +
                        "imaginary part (%g)!" % _np.linalg.norm(_np.imag(matrix)))
            matrix = _np.real(matrix)

        assert(matrix.shape == (self.dim,self.dim))
        self.base = matrix
        self.base.flags.writeable = False


    def set_value(self, M):
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
        mx = GateMatrix.convert_to_matrix(M)
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
        self.dirty = True


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
        derivMx = _np.zeros( (self.numParams, self.dim, self.dim), 'complex' )
        for (i,j),terms in self.elementExpressions.items():
            for term in terms:
                params_to_mult = [ self.parameterArray[p] for p in term.paramIndices ]
                for k,p in enumerate(term.paramIndices):
                    param_partial_prod = _np.prod( params_to_mult[0:k] + params_to_mult[k+1:] ) # exclude k-th factor
                    derivMx[p,i,j] += term.coeff * param_partial_prod
        
        derivMx = _np.dot(self.leftTrans, _np.dot(derivMx, self.rightTrans)) # (d,d) * (P,d,d) * (d,d) => (d,P,d)
        derivMx = _np.rollaxis(derivMx,1,3) # now (d,d,P)
        derivMx = derivMx.reshape([self.dim**2, self.numParams]) # (d^2,P) == final shape

        if self.enforceReal:
            assert(_np.linalg.norm(_np.imag(derivMx)) < IMAG_TOL)
            derivMx = _np.real(derivMx)

        if wrtFilter is None:
            return derivMx
        else:
            return _np.take( derivMx, wrtFilter, axis=1 )

        
    def has_nonzero_hessian(self):
        """ 
        Returns whether this gate has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return False


    def copy(self, parent=None):
        """
        Copy this gate.

        Returns
        -------
        Gate
            A copy of this gate.
        """
        #Construct new gate with no intial elementExpressions
        newGate = LinearlyParameterizedGate(self.baseMatrix, self.parameterArray.copy(),
                                            {}, self.leftTrans.copy(), self.rightTrans.copy(),
                                            self.enforceReal)

        #Deep copy elementExpressions into new gate
        for tup, termList in list(self.elementExpressions.items()):
            newGate.elementExpressions[tup] = [ term.copy() for term in termList ]
        newGate._construct_matrix()

        return self._copy_gpindices( newGate, parent )


    def transform(self, S):
        """
        Update gate matrix G with inv(S) * G * S,

        Generally, the transform function updates the *parameters* of 
        the gate such that the resulting gate matrix is altered as 
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.
        """
        #Currently just raise error - in general there is no *unique* 
        # parameter alteration that would affect a similarity transform,
        # and even if one exists it may be difficult to find efficiently.
        raise ValueError("Invalid transform for this LinearlyParameterizedGate")
    

    def depolarize(self, amount):
        """
        Depolarize this gate by the given `amount`.

        Generally, the depolarize function updates the *parameters* of 
        the gate such that the resulting gate matrix is depolarized.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. In standard
            bases, depolarization corresponds to multiplying the gate matrix
            by a diagonal matrix whose first diagonal element (corresponding
            to the identity) equals 1.0 and whose subsequent elements 
            (corresponding to non-identity basis elements) equal
            `1.0 - amount[i]` (or just `1.0 - amount` if `amount` is a
            float).

        Returns
        -------
        None
        """
        raise ValueError("Cannot depolarize a LinearlyParameterizedGate (no general procedure)!")


    def rotate(self, amount, mxBasis="gm"):
        """
        Rotate this gate by the given `amount`.

        Generally, the rotate function updates the *parameters* of 
        the gate such that the resulting gate matrix is rotated.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : tuple of floats, optional
            Specifies the rotation "coefficients" along each of the non-identity
            Pauli-product axes.  The gate's matrix `G` is composed with a
            rotation operation `R`  (so `G` -> `dot(R, G)` ) where `R` is the
            unitary superoperator corresponding to the unitary operator 
            `U = exp( sum_k( i * rotate[k] / 2.0 * Pauli_k ) )`.  Here `Pauli_k`
            ranges over all of the non-identity un-normalized Pauli operators.

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        Returns
        -------
        None
        """
        raise ValueError("Cannot rotate a LinearlyParameterizedGate (no general procedure)!")



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
        s += "\nParameterization:"
        for (i,j),terms in self.elementExpressions.items():
            tStr = ' + '.join([ '*'.join(["p%d"%p for p in term.paramIndices])
                                for term in terms] )
            s += "Gate[%d,%d] = %s\n" % (i,j,tStr)
        return s


#Or CommutantParameterizedGate(Gate): ?
class EigenvalueParameterizedGate(GateMatrix):
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
            return _mt.complex_compare(evals[ia], evals[ib])
        cmplx_compare_key = _functools.cmp_to_key(cmplx_compare)

        def isreal(a):
            """ b/c numpy's isreal tests for strict equality w/0 """
            return _np.isclose(_np.imag(a),0.0)

        # Since matrix is real, eigenvalues must either be real or occur in
        #  conjugate pairs.  Find and sort by conjugate pairs.

        assert(_np.linalg.norm(_np.imag(matrix)) < IMAG_TOL ) #matrix should be real
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
                    if _np.linalg.norm(self.B[:,k].imag) >= IMAG_TOL:
                        evecIndsToMakeReal.append(k)

                nToReal = len(evecIndsToMakeReal)
                if nToReal > 0:
                    vecs = _np.empty( (dim,nToReal),'complex')
                    for ik,k in enumerate(evecIndsToMakeReal): 
                        vecs[:,ik] = self.B[:,k]
                    V = _np.concatenate((vecs.real, vecs.imag), axis=1)
                    nullsp = _mt.nullspace(V); 
                    #if nullsp.shape[1] < nToReal: # DEBUG
                    #    raise ValueError("Nullspace only has dimension %d when %d was expected! (i=%d, j=%d, blkSize=%d)\nevals = %s" \
                    #                     % (nullsp.shape[1],nToReal, i,j,blkSize,str(self.evals)) )
                    assert(nullsp.shape[1] >= nToReal),"Cannot find enough real linear combos!"
                    nullsp = nullsp[:,0:nToReal] #truncate #cols if there are more than we need
    
                    Cmx = nullsp[nToReal:,:] + 1j*nullsp[0:nToReal,:] # Cr + i*Ci
                    new_vecs = _np.dot(vecs,Cmx)
                    assert(_np.linalg.norm(new_vecs.imag) < IMAG_TOL), "Imaginary mag = %g!" % _np.linalg.norm(new_vecs.imag)
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
                                (_np.allclose(conjB, self.B[:,l]) or
                                 _np.allclose(conjB, 1j*self.B[:,l]) or
                                 _np.allclose(conjB, -1j*self.B[:,l]) or
                                 _np.allclose(conjB, -1*self.B[:,l])): #numpy normalizes but doesn't fix "phase" of evecs
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
                        # should be unreachable, since we ensure mx is real above - but
                        # this may fail when there are multiple degenerate complex evals
                        # since the evecs can get mixed (and we check for evec "match" above)
                        raise ValueError("Could not find conjugate pair " 
                                         + " for %s" % self.evals[k]) # pragma: no cover
            
            
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
        GateMatrix.__init__(self, mx)
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
        assert(_np.linalg.norm(matrix.imag) < IMAG_TOL)
        assert(matrix.shape == (self.dim,self.dim))
        self.base = matrix.real
        self.base.flags.writeable = False

    def set_value(self, M):
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
        mx = GateMatrix.convert_to_matrix(M)
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
        self.dirty = True


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
        for k,pdesc in enumerate(self.params):
            dMx = _np.zeros( (self.dim,self.dim), 'complex')
            for prefactor,(i,j) in pdesc:
                dMx[i,j] = prefactor
            tmp = _np.dot(self.B, _np.dot(dMx, self.Bi))
            if _np.linalg.norm(tmp.imag) >= IMAG_TOL: #just a warning until we figure this out.
                print("EigenvalueParameterizedGate deriv_wrt_params WARNING:" + 
                      " Imag part = ",_np.linalg.norm(tmp.imag)," pdesc = ",pdesc) # pragma: no cover
            #assert(_np.linalg.norm(tmp.imag) < IMAG_TOL), \
            #       "Imaginary mag = %g!" % _np.linalg.norm(tmp.imag)
            derivMx[:,k] = tmp.real.flatten()

        if wrtFilter is None:
            return derivMx
        else:
            return _np.take( derivMx, wrtFilter, axis=1 )

        
    def has_nonzero_hessian(self):
        """ 
        Returns whether this gate has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return False


    def copy(self, parent=None):
        """
        Copy this gate.

        Returns
        -------
        Gate
            A copy of this gate.
        """
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

        return self._copy_gpindices( newGate, parent )


    def transform(self, S):
        """
        Update gate matrix G with inv(S) * G * S,

        Generally, the transform function updates the *parameters* of 
        the gate such that the resulting gate matrix is altered as 
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.
        """
        #Currently just raise error, as a general similarity transformation
        # *will* affect the eigenvalues.  In the future, perhaps we could allow
        # *unitary* type similarity transformations.
        raise ValueError("Invalid transform for this EigenvalueParameterizedGate")


    def depolarize(self, amount):
        """
        Depolarize this gate by the given `amount`.

        Generally, the depolarize function updates the *parameters* of 
        the gate such that the resulting gate matrix is depolarized.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. In standard
            bases, depolarization corresponds to multiplying the gate matrix
            by a diagonal matrix whose first diagonal element (corresponding
            to the identity) equals 1.0 and whose subsequent elements 
            (corresponding to non-identity basis elements) equal
            `1.0 - amount[i]` (or just `1.0 - amount` if `amount` is a
            float).

        Returns
        -------
        None
        """
        #TODO: maybe could do this if we wanted to get clever: if one eigenvector is the
        # identity we could reduce all the eigenvalues corresponding to *other* evecs.
        raise ValueError("Cannot depolarize a EigenvalueParameterizedGate")


    def rotate(self, amount, mxBasis="gm"):
        """
        Rotate this gate by the given `amount`.

        Generally, the rotate function updates the *parameters* of 
        the gate such that the resulting gate matrix is rotated.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : tuple of floats, optional
            Specifies the rotation "coefficients" along each of the non-identity
            Pauli-product axes.  The gate's matrix `G` is composed with a
            rotation operation `R`  (so `G` -> `dot(R, G)` ) where `R` is the
            unitary superoperator corresponding to the unitary operator 
            `U = exp( sum_k( i * rotate[k] / 2.0 * Pauli_k ) )`.  Here `Pauli_k`
            ranges over all of the non-identity un-normalized Pauli operators.

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        Returns
        -------
        None
        """
        raise ValueError("Cannot rotate an EigenvalueParameterizedGate!")


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



#TODO: add "sparse" flag to LindbladParameterizedGate
#  -- maybe add "sparse" basis (where mxs are sparse) to facilitate this
#  -- use scipy.sparse.linalg.expm_multiply to perform sparse multiplies

class LindbladBase(object):
    """ Base class for Lindblad-parameterized gates """

    def __init__(self, errgen, ham_basis="pp", nonham_basis="pp", cptp=True,
                 nonham_diagonal_only=False, truncate=True, mxBasis="pp"):
        """
        Initialize a LindbladBase object.  Parameters are explained in 
        derived-class doc strings.
        """
        self.cptp = cptp
        self.nonham_diagonal_only = nonham_diagonal_only

        d2 = errgen.shape[0]
        d = int(round(_np.sqrt(d2)))
        assert(d*d == d2), "Gate dim must be a perfect square"

        #Determine whether we're using sparse bases or not
        self.sparse = None
        if ham_basis is not None:
            if isinstance(ham_basis, _Basis): self.sparse = ham_basis.sparse
            elif _compat.isstr(ham_basis): self.sparse = _sps.issparse(errgen)
            elif len(ham_basis) > 0: self.sparse = _sps.issparse(ham_basis[0])
        if self.sparse is None and nonham_basis is not None:
            if isinstance(nonham_basis, _Basis): self.sparse = nonham_basis.sparse
            elif _compat.isstr(nonham_basis): self.sparse = _sps.issparse(errgen)
            elif len(nonham_basis) > 0: self.sparse = _sps.issparse(nonham_basis[0])
        if self.sparse is None: self.sparse = False #the default

        self.dim = d2 #also set by GateMatrix and GateMap (but consistent, so OK)
        if isinstance(ham_basis, _Basis) or _compat.isstr(ham_basis):
            self.ham_basis = _Basis(ham_basis,d,sparse=self.sparse)
        else: # ham_basis is a list of matrices
            self.ham_basis = _Basis(matrices=ham_basis,dim=d,sparse=self.sparse)

        if isinstance(nonham_basis, _Basis) or _compat.isstr(nonham_basis):
            self.other_basis = _Basis(nonham_basis,d,sparse=self.sparse)
        else: # ham_basis is a list of matrices
            self.other_basis = _Basis(matrices=nonham_basis,dim=d,sparse=self.sparse)
        
        self.matrix_basis = _Basis(mxBasis,d,sparse=self.sparse)

        #Assumed gate was in the pauli-product basis initially.  I think
        # everything should work fine in the current general case with any other
        # basis with the same trace(BiBj) = delta_ij property (e.g. Gell-Mann)
        mxBasisToStd = _bt.transform_matrix(self.matrix_basis, "std", d)
        self.leftTrans  = _spsl.inv(mxBasisToStd.tocsc()).tocsr() if _sps.issparse(mxBasisToStd) \
                          else _np.linalg.inv(mxBasisToStd)
        self.rightTrans = mxBasisToStd            


        self._set_params_from_errgen(errgen, truncate)
          #also sets self.hamGens and self.otherGens as side effect (TODO FIX)
          # and self.ham_basis_size and self.other_basis_size.  Requires
          # self.leftTrans and self.rightTrans be set already

        #initialize intermediate storage for matrix and for deriv computation
        # (needed for _construct_errgen)
        bsO = self.other_basis_size
        if bsO > 0:
            self.Lmx = _np.zeros((bsO-1,bsO-1),'complex')
        else: self.Lmx = None
          
        #set self.err_gen (can't just use `errgen` b/c of truncation)
        self._construct_errgen() 



    def _set_params_from_errgen(self, errgen, truncate):
        d2 = errgen.shape[0]

        hamC, otherC, self.hamGens, self.otherGens = \
            _gt.lindblad_errgen_projections(
                errgen, self.ham_basis, self.other_basis, self.matrix_basis, normalize=False,
                return_generators=True, other_diagonal_only=self.nonham_diagonal_only,
                sparse=self.sparse) # in std basis
        # Note: lindblad_errgen_projections will return sparse generators when
        #  given a sparse basis (or basis matrices)

        if self.hamGens is not None:
            bsH = len(self.hamGens)+1 #projection-basis size (not nec. == d2)
            _gt._assert_shape(self.hamGens, (bsH-1,d2,d2), self.sparse)

            # apply basis change now, so we don't need to do so repeatedly later
            if self.sparse:
                self.hamGens = [ _mt.safereal(_mt.safedot(self.leftTrans, _mt.safedot(mx, self.rightTrans)),
                                              inplace=True, check=True) for mx in self.hamGens ]
                for mx in self.hamGens: mx.sort_indices()
                  # for faster addition ops in _construct_errgen
            else:
                self.hamGens = _np.einsum("ik,akl,lj->aij", self.leftTrans, self.hamGens, self.rightTrans)
            
            assert(_np.isclose(_np.linalg.norm(hamC.imag),0)), \
                "Hamiltoian coeffs are not all real!"
            hamParams = hamC.real
        else:
            bsH = 0
            hamParams = _np.empty(0,'d')
            

        if self.otherGens is not None:
            bsO = len(self.otherGens)+1 #projection-basis size (not nec. == d2)

            if self.nonham_diagonal_only:
                _gt._assert_shape(self.otherGens, (bsO-1,d2,d2), self.sparse)

                # apply basis change now, so we don't need to do so repeatedly later
                if self.sparse:
                    self.otherGens = [ _mt.safereal(_mt.safedot(self.leftTrans, _mt.safedot(mx, self.rightTrans)),
                                                    inplace=True, check=True) for mx in self.otherGens ]
                    for mx in self.hamGens: mx.sort_indices()
                      # for faster addition ops in _construct_errgen
                else:
                    self.otherGens = _np.einsum("ik,akl,lj->aij", self.leftTrans, self.otherGens, self.rightTrans)


                assert(_np.isclose(_np.linalg.norm(_np.imag(otherC)),0))
                #assert(_np.all(_np.isreal(otherC))) #sometimes fails even when imag to machine prec
    
                if self.cptp: #otherParams is a 1D vector of the sqrts of diagonal els
                    otherC = otherC.clip(1e-16,1e100) #must be positive
                    otherParams = _np.sqrt(otherC.real) # shape (bs-1,)
                else: #otherParams is a 1D vector of the real diagonal els of otherC
                    otherParams = otherC.real # shape (bs-1,)
    
            else:
                _gt._assert_shape(self.otherGens, (bsO-1,bsO-1,d2,d2), self.sparse)

                # apply basis change now, so we don't need to do so repeatedly later
                if self.sparse:
                    self.otherGens = [ [_mt.safedot(self.leftTrans, _mt.safedot(mx, self.rightTrans))
                                        for mx in mxRow ] for mxRow in self.otherGens ]
                    #Note: complex OK here, as only linear combos of otherGens (like (i,j) + (j,i)
                    # terms) need to be real

                    for mxRow in self.otherGens:
                        for mx in mxRow: mx.sort_indices()
                          # for faster addition ops in _construct_errgen:
                else:
                    preshape = self.otherGens.shape
                    self.otherGens = _np.einsum("ik,abkl,lj->abij", self.leftTrans,
                                                self.otherGens, self.rightTrans)

                assert(_np.isclose(_np.linalg.norm(otherC-otherC.T.conjugate())
                                   ,0)), "other coeff mx is not Hermitian!"
    
                otherParams = _np.empty((bsO-1,bsO-1),'d')
    
                if self.cptp: #otherParams mx stores Cholesky decomp
    
                    #push any slightly negative evals of otherC positive so that
                    # the Cholesky decomp will work.
                    evals,U = _np.linalg.eig(otherC)
                    Ui = _np.linalg.inv(U)
    
                    assert(truncate or all([ev >= -1e-12 for ev in evals])), \
                        "gateMatrix must be CPTP (truncate == False)!"
    
                    pos_evals = evals.clip(1e-16,1e100)
                    otherC = _np.dot(U,_np.dot(_np.diag(pos_evals),Ui))
                    try:
                        Lmx = _np.linalg.cholesky(otherC)

                    # if Lmx not postitive definite, try again with 1e-12 (same lines as above)
                    except _np.linalg.LinAlgError:                         # pragma: no cover
                        pos_evals = evals.clip(1e-12,1e100)                # pragma: no cover
                        otherC = _np.dot(U,_np.dot(_np.diag(pos_evals),Ui))# pragma: no cover
                        Lmx = _np.linalg.cholesky(otherC)                  # pragma: no cover
    
                    for i in range(bsO-1):
                        assert(_np.linalg.norm(_np.imag(Lmx[i,i])) < IMAG_TOL)
                        otherParams[i,i] = Lmx[i,i].real
                        for j in range(i):
                            otherParams[i,j] = Lmx[i,j].real
                            otherParams[j,i] = Lmx[i,j].imag
    
                else: #otherParams mx stores otherC (hermitian) directly
                    for i in range(bsO-1):
                        assert(_np.linalg.norm(_np.imag(otherC[i,i])) < IMAG_TOL)
                        otherParams[i,i] = otherC[i,i].real
                        for j in range(i):
                            otherParams[i,j] = otherC[i,j].real
                            otherParams[j,i] = otherC[i,j].imag
        else:
            bsO = 0
            otherParams = _np.empty(0,'d')

        if self.sparse:
            #Precompute for faster CSR sums in _construct_errgen
            all_csr_matrices = []; nHam = 0; nOther = 0;
            
            if self.hamGens is not None:
                all_csr_matrices.extend(self.hamGens)
                nHam = len(self.hamGens)
                
            if self.otherGens is not None:
                if self.nonham_diagonal_only:
                    oList = self.otherGens
                else:
                    oList = [ mx for mxRow in self.otherGens for mx in mxRow ]
                all_csr_matrices.extend(oList)

            csr_sum_array, indptr, indices, N = \
                    self.get_csr_sum_indices(all_csr_matrices)
            self.hamCSRSumIndices = csr_sum_array[0:nHam]
            self.otherCSRSumIndices = csr_sum_array[nHam:]
            self.sparse_err_gen_template = (indptr, indices, N)
        else:
            #N/A if not sparse
            self.hamCSRSumIndices = None
            self.otherCSRSumIndices = None
            self.sparse_err_gen_template = None            

        self.paramvals = _np.concatenate( (hamParams, otherParams.flat) )
        self.ham_basis_size = bsH
        self.other_basis_size = bsO

        #True unless bsH,bsO are zero
        #if nonham_diagonal_only:
        #    assert(self.paramvals.shape == ((bsH-1)+(bsO-1),))
        #else:
        #    assert(self.paramvals.shape == ((bsH-1)+(bsO-1)**2,))

    def get_csr_sum_indices(self, csr_matrices):
        """ 
        Precomputes the indices needed to sum a set of CSR sparse matrices.

        Computes the index-arrays needed for use in :method:`csr_sum`,
        along with the index pointer and column-indices arrays for constructing
        a "template" CSR matrix to be the destination of `csr_sum`.

        Parameters
        ----------
        csr_matrices : list
            The SciPy CSR matrices to be summed.

        Returns
        -------
        ind_arrays : list
            A list of numpy arrays giving the destination data-array indices
            of each element of `csr_matrices`.
        indptr, indices : numpy.ndarray
            The row-pointer and column-indices arrays specifying the sparsity 
            structure of a the destination CSR matrix.
        N : int
            The dimension of the destination matrix (and of each member of
            `csr_matrices`)
        """
        if len(csr_matrices) == 0: return []
        
        N = csr_matrices[0].shape[0]
        for mx in csr_matrices:
            assert(mx.shape == (N,N)), "Matrices must have the same square shape!"
            
        indptr = [0]
        indices = []
        csr_sum_array = [ list() for mx in csr_matrices ]

        #FUTURE sort column indices
        
        for iRow in range(N):
            dataInds = {} #keys = column indices, values = data indices (for data in current row)
                
            for iMx,mx in enumerate(csr_matrices):
                for i in range(mx.indptr[iRow],mx.indptr[iRow+1]):
                    iCol = mx.indices[i]
                    if iCol not in dataInds: #add a new element to final mx
                        indices.append(iCol)
                        dataInds[iCol] = len(indices)-1 #marks the final data index for this column
                    csr_sum_array[iMx].append(dataInds[iCol])
            indptr.append( len(indices) )

        #convert lists -> arrays
        csr_sum_array = [ _np.array(lst,'i') for lst in csr_sum_array ]
        indptr = _np.array( indptr )
        indices = _np.array( indices )

        return csr_sum_array, indptr, indices, N
    

        
    def csr_sum(self, data, coeffs, csr_mxs, csr_sum_indices):
        """ 
        Accelerated summation of several CSR-format sparse matrices.

        :method:`get_csr_sum_indices` precomputes the necessary indices for
        summing directly into the data-array of a destination CSR sparse matrix.
        If `data` is the data-array of matrix `D` (for "destination"), then this
        method performs:

        `D += sum_i( coeff[i] * csr_mxs[i] )`

        Note that `D` is not returned; the sum is done internally into D's
        data-array.

        Parameters
        ----------
        data : numpy.ndarray
            The data-array of the destination CSR-matrix.

        coeffs : iterable
            The weight coefficients which multiply each summed matrix.

        csr_mxs : iterable
            A list of CSR matrix objects whose data-array is given by
            `obj.data` (e.g. a SciPy CSR sparse matrix).

        csr_sum_indices : list
            A list of precomputed index arrays as returned by
            :method:`get_csr_sum_indices`.

        Returns
        -------
        None
        """
        for coeff,mx,inds in zip(coeffs, csr_mxs, csr_sum_indices):
            data[inds] += coeff*mx.data        

    def _construct_errgen(self):
        """
        Build the error generator matrix using the current parameters.
        """
        d2 = self.dim
        bsH = self.ham_basis_size
        bsO = self.other_basis_size

        # self.paramvals = [hamCoeffs] + [otherParams]
        #  where hamCoeffs are *real* and of length d2-1 (self.dim == d2)
        if bsH > 0:
            hamCoeffs = self.paramvals[0:bsH-1]
            nHam = bsH-1
        else:
            nHam = 0

        #built up otherCoeffs based on self.cptp and self.nonham_diagonal_only
        if bsO > 0:
            if self.nonham_diagonal_only:
                otherParams = self.paramvals[nHam:]
                assert(otherParams.shape == (bsO-1,))
    
                if self.cptp:
                    otherCoeffs = otherParams**2 #Analagous to L*L_dagger
                else:
                    otherCoeffs = otherParams
            else:
                otherParams = self.paramvals[nHam:].reshape((bsO-1,bsO-1))
    
                if self.cptp:
                    #  otherParams is an array of length (bs-1)*(bs-1) that
                    #  encodes a lower-triangular matrix "Lmx" via:
                    #  Lmx[i,i] = otherParams[i,i]
                    #  Lmx[i,j] = otherParams[i,j] + 1j*otherParams[j,i] (i > j)
                    for i in range(bsO-1):
                        self.Lmx[i,i] = otherParams[i,i]
                        for j in range(i):
                            self.Lmx[i,j] = otherParams[i,j] + 1j*otherParams[j,i]
            
                    #The matrix of (complex) "other"-coefficients is build by
                    # assuming Lmx is its Cholesky decomp; means otherCoeffs
                    # is pos-def.

                    # NOTE that the Cholesky decomp with all positive real diagonal
                    # elements is *unique* for a given positive-definite otherCoeffs
                    # matrix, but we don't care about this uniqueness criteria and so
                    # the diagonal els of Lmx can be negative and that's fine -
                    # otherCoeffs will still be posdef.
                    otherCoeffs = _np.dot(self.Lmx,self.Lmx.T.conjugate())
    
                    #DEBUG - test for pos-def
                    #evals = _np.linalg.eigvalsh(otherCoeffs)
                    #DEBUG_TOL = 1e-16; #print("EVALS DEBUG = ",evals)
                    #assert(all([ev >= -DEBUG_TOL for ev in evals]))
    
                else:
                    #otherParams holds otherCoeff real and imaginary parts directly
                    otherCoeffs = _np.empty((bsO-1,bsO-1),'complex')
                    for i in range(bsO-1):
                        otherCoeffs[i,i] = otherParams[i,i]
                        for j in range(i):
                            otherCoeffs[i,j] = otherParams[i,j] +1j*otherParams[j,i]
                            otherCoeffs[j,i] = otherParams[i,j] -1j*otherParams[j,i]
                            
        #Finally, build gate matrix from generators and coefficients:
        if self.sparse:
            #FUTURE: could try to optimize the sum-scalar-mults ops below, as these take the
            # bulk of from_vector time, which recurs frequently.
            indptr, indices, N = self.sparse_err_gen_template # the CSR arrays giving
               # the structure of a CSR matrix with 0-elements in all possible places
            data = _np.zeros(len(indices),'complex') # data starts at zero
            
            if bsH > 0:
                # lnd_error_gen = sum([c*gen for c,gen in zip(hamCoeffs, self.hamGens)])
                self.csr_sum(data,hamCoeffs, self.hamGens, self.hamCSRSumIndices)

            if bsO > 0:
                if self.nonham_diagonal_only:
                    # lnd_error_gen += sum([c*gen for c,gen in zip(otherCoeffs, self.otherGens)])
                    self.csr_sum(data, otherCoeffs, self.otherGens, self.otherCSRSumIndices)
                else:
                    # lnd_error_gen += sum([c*gen for cRow,genRow in zip(otherCoeffs, self.otherGens)
                    #                      for c,gen in zip(cRow,genRow)])
                    self.csr_sum(data, otherCoeffs.flat,
                                 [oGen for oGenRow in self.otherGens for oGen in oGenRow],
                                 self.otherCSRSumIndices)
            lnd_error_gen = _sps.csr_matrix( (data, indices.copy(), indptr.copy()), shape=(N,N) ) #copies needed (?)
            

        else: #dense matrices
            if bsH > 0:
                lnd_error_gen = _np.einsum('i,ijk', hamCoeffs, self.hamGens)
            else:
                lnd_error_gen = _np.zeros( (d2,d2), 'complex')

            if bsO > 0:
                if self.nonham_diagonal_only:
                    lnd_error_gen += _np.einsum('i,ikl', otherCoeffs, self.otherGens)
                else:
                    lnd_error_gen += _np.einsum('ij,ijkl', otherCoeffs,
                                                self.otherGens)

            #lnd_error_gen = _np.dot(self.leftTrans, _np.dot(   #REMOVE IF PRETRANS
            #    lnd_error_gen, self.rightTrans)) #basis chg    #REMOVE IF PRETRANS

        assert(_np.isclose( _mt.safenorm(lnd_error_gen,'imag'), 0))
        #print("errgen pre-real = \n"); _mt.print_mx(lnd_error_gen,width=4,prec=1)        
        self.err_gen = _mt.safereal(lnd_error_gen, inplace=True)


        
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
        self._construct_errgen()
        self.dirty = True



class LindbladParameterizedGateMap(LindbladBase, GateMap):

    def __init__(self, gateMatrix, unitaryPostfactor=None,
                 ham_basis="pp", nonham_basis="pp", cptp=True,
                 nonham_diagonal_only=False, truncate=True, mxBasis="pp"):
        """
        Initialize a LindbladParameterizedGateMap object.

        Parameters
        ----------
        gateMatrix : numpy array or SciPy sparse matrix
            a square 2D array that gives the raw gate matrix, assumed to
            be in the `mxBasis` basis, to parameterize.  The shape of this
            array sets the dimension of the gate. If None, then it is assumed
            equal to `unitaryPostfactor` (which cannot also be None). The
            quantity `gateMatrix inv(unitaryPostfactor)` is parameterized via
            projection onto the Lindblad terms.
            
        unitaryPostfactor : numpy array or SciPy sparse matrix, optional
            a square 2D array of the same size of `gateMatrix` (if
            not None).  This matrix specifies a part of the gate action 
            to remove before parameterization via Lindblad projections.
            Typically, this is a target (desired) gate operation such 
            that only the erroneous part of the gate (i.e. the gate 
            relative to the target), which should be close to the identity,
            is parameterized.  If none, the identity is used by default.

        ham_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis is used to construct the Stochastic-type lindblad error
            Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        other_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis is used to construct the Stochastic-type lindblad error
            Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        cptp : bool, optional
            Whether or not the new gate should be constrained to CPTP.
            (if True, see behavior or `truncate`).

        nonham_diagonal_only : boolean, optional
            If True, only *diagonal* Stochastic (non-Hamiltonain) terms are
            included in the parameterization.

        truncate : bool, optional
            Whether to truncate the projections onto the Lindblad terms in
            order to preserve CPTP (when necessary).  If False, then an 
            error is thrown when `cptp == True` and when Lindblad projections
            result in a non-positive-definite matrix of non-Hamiltonian term
            coefficients.

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).
        """
        if gateMatrix is None:
            assert(unitaryPostfactor is not None), "arguments cannot both be None"
            gateMatrix = unitaryPostfactor
            
        self.unitary_postfactor = unitaryPostfactor #can be None
        if unitaryPostfactor is None:
            if _sps.issparse(gateMatrix):
                unitaryPostfactor = _sps.identity(gateMatrix.shape[0],'d','csr')
            else: unitaryPostfactor = _np.identity(gateMatrix.shape[0],'d')
        
        #Init base from error generator: sets basis members and ultimately
        # the parameters in self.paramvals
        if _sps.issparse(gateMatrix):
            #Instead of making error_generator compatible with sparse matrices
            # we require sparse matrices to have trivial initial error generators
            # or we convert to dense:
            if(_mt.safenorm(gateMatrix-unitaryPostfactor) < 1e-8):
                errgen = _sps.csr_matrix( gateMatrix.shape, dtype='d' ) # all zeros
            else:
                errgen = _sps.csr_matrix(
                    _gt.error_generator(
                        gateMatrix.toarray(), unitaryPostfactor.toarray(),
                        mxBasis, "logGTi"), dtype='d')
        else:
            errgen = _gt.error_generator(gateMatrix, unitaryPostfactor, mxBasis, "logGTi")
            
        GateMap.__init__(self, errgen.shape[0]) #sets self.dim
        LindbladBase.__init__(self, errgen, ham_basis, nonham_basis, cptp,
                              nonham_diagonal_only, truncate, mxBasis)
          # initializes self.paramvals and self.err_gen among other things
        
        #Pre-compute the exponential of the error generator if dense matrices are used
        if self.sparse:
            self.err_gen_prep = _mt.expm_multiply_prep(self.err_gen)
            self.exp_err_gen = None
        else:
            self.err_gen_prep = None
            self.exp_err_gen = _spl.expm(self.err_gen)            
        
    def acton(self, state):
        """ Act this gate map on an input state """
        if self.unitary_postfactor is not None:
            state = self.unitary_postfactor.dot(state) #works for sparse or dense matrices
            
        if self.sparse:
            #state = _spsl.expm_multiply( self.err_gen, state) #SLOW
            state = _mt.expm_multiply_fast(self.err_gen_prep, state[:,0])[:,None] # (N,1) -> (N,1) shape mapping
        else:
            state = _np.dot(self.exp_err_gen, state)
        return state


    #FUTURE: maybe remove this function altogether, as it really shouldn't be called
    def as_sparse_matrix(self):
        """
        Return the gate as a sparse matrix.
        """
        _warnings.warn(("Constructing the sparse matrix of a LindbladParameterizedGate."
                        "  Usually this is *NOT* a sparse matrix (the exponential of a"
                        " sparse matrix isn't generally sparse)!"))
        if self.sparse:
            exp_err_gen = _spsl.expm(self.err_gen.tocsc()).tocsr()
            if self.unitary_postfactor is not None:
                return exp_err_gen.dot(self.unitary_postfactor)
            else:
                return exp_err_gen
        else:
            if self.unitary_postfactor is not None:
                dense = _np.dot(self.exp_err_gen, self.unitary_postfactor)
            else:
                dense = self.exp_err_gen
            return _sps.csr_matrix( dense )

    
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
        LindbladBase.from_vector(self, v)
        if self.sparse:
            self.err_gen_prep = _mt.expm_multiply_prep(self.err_gen)
        else:
            self.exp_err_gen = _spl.expm(self.err_gen)
            

    def copy(self, parent=None):
        """
        Copy this gate.

        Returns
        -------
        Gate
            A copy of this gate.
        """
        #Construct new gate with dummy identity mx
        if self.unitary_postfactor is not None:
            factor = self.unitary_postfactor.copy()
        elif self.sparse:
            factor = _sps.identity( self.err_gen.shape[0], 'd', 'csr' )
        else:
            factor = _np.identity( self.err_gen.shape[0], 'd' )
        newGate = LindbladParameterizedGateMap(factor,factor,
                                            _copy.deepcopy(self.ham_basis),
                                            _copy.deepcopy(self.other_basis),
                                            self.cptp,self.nonham_diagonal_only,
                                            True, _copy.deepcopy(self.matrix_basis))
        
        #Deep copy data
        newGate.paramvals = self.paramvals.copy()
        newGate._construct_errgen()
        if not newGate.sparse:
            newGate.exp_err_gen = _spl.expm(newGate.err_gen) 

        return self._copy_gpindices( newGate, parent )


    def transform(self, S):
        """
        Update gate matrix G with inv(S) * G * S,

        Generally, the transform function updates the *parameters* of 
        the gate such that the resulting gate matrix is altered as 
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.
        """
        if isinstance(S, _gaugegroup.UnitaryGaugeGroupElement) or \
           isinstance(S, _gaugegroup.TPSpamGaugeGroupElement):
            U = S.get_transform_matrix()
            Uinv = S.get_transform_matrix_inverse()

            #just conjugate postfactor and Lindbladian exponent by U:
            if self.unitary_postfactor is not None:
                self.unitary_postfactor = _mt.safedot(Uinv,_mt.safedot(self.unitary_postfactor, U))
            self.err_gen = _mt.safedot(Uinv,_mt.safedot(self.err_gen, U))
            self._set_params_from_errgen(self.err_gen, truncate=True)
            self._construct_errgen() # unnecessary? (TODO)
            if self.sparse:
                self.err_gen_prep = _mt.expm_multiply_prep(self.err_gen)
            else:
                self.exp_err_gen = _spl.expm(self.err_gen)
            self.dirty = True
            #Note: truncate=True above because some unitary transforms seem to
            ## modify eigenvalues to be negative beyond the tolerances
            ## checked when truncate == False.  I'm not sure why this occurs,
            ## since a true unitary should map CPTP -> CPTP...

            #CHECK WITH OLD (passes) TODO move to unit tests?
            #tMx = _np.dot(Uinv,_np.dot(self.base, U)) #Move above for checking
            #tGate = LindbladParameterizedGate(tMx,self.unitary_postfactor,
            #                                self.ham_basis, self.other_basis,
            #                                self.cptp,self.nonham_diagonal_only,
            #                                True, self.matrix_basis)
            #assert(_np.linalg.norm(tGate.paramvals - self.paramvals) < 1e-6)
        else:
            raise ValueError("Invalid transform for this LindbladParameterizedGate: type %s"
                             % str(type(S)))


    def depolarize(self, amount):
        """
        Depolarize this gate by the given `amount`.

        Generally, the depolarize function updates the *parameters* of 
        the gate such that the resulting gate matrix is depolarized.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. In standard
            bases, depolarization corresponds to multiplying the gate matrix
            by a diagonal matrix whose first diagonal element (corresponding
            to the identity) equals 1.0 and whose subsequent elements 
            (corresponding to non-identity basis elements) equal
            `1.0 - amount[i]` (or just `1.0 - amount` if `amount` is a
            float).

        Returns
        -------
        None
        """
        assert(not self.sparse), "depolarize() not implemented for sparse LindbladParameterizedGateMap objects"
        if isinstance(amount,float):
            D = _np.diag( [1]+[1-amount]*(self.dim-1) )
        else:
            assert(len(amount) == self.dim-1)
            D = _np.diag( [1]+list(1.0 - _np.array(amount,'d')) )

        if self.unitary_postfactor is not None:
            dense = _np.dot(self.exp_err_gen, self.unitary_postfactor)
        else:
            dense = self.exp_err_gen
        mx = _np.dot(D,dense)
        tGate = LindbladParameterizedGateMap(mx,self.unitary_postfactor,
                                          self.ham_basis, self.other_basis,
                                          self.cptp,self.nonham_diagonal_only,
                                          True, self.matrix_basis)

        #Note: truncate=True to be safe
        self.paramvals[:] = tGate.paramvals[:]
        self._construct_errgen()
        self.dirty = True


    def rotate(self, amount, mxBasis="gm"):
        """
        Rotate this gate by the given `amount`.

        Generally, the rotate function updates the *parameters* of 
        the gate such that the resulting gate matrix is rotated.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : tuple of floats, optional
            Specifies the rotation "coefficients" along each of the non-identity
            Pauli-product axes.  The gate's matrix `G` is composed with a
            rotation operation `R`  (so `G` -> `dot(R, G)` ) where `R` is the
            unitary superoperator corresponding to the unitary operator 
            `U = exp( sum_k( i * rotate[k] / 2.0 * Pauli_k ) )`.  Here `Pauli_k`
            ranges over all of the non-identity un-normalized Pauli operators.

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        Returns
        -------
        None
        """
        assert(not self.sparse), "rotate() not implemented for sparse LindbladParameterizedGateMap objects"
        rotnMx = _gt.rotation_gate_mx(amount,mxBasis)
        if self.unitary_postfactor is not None:
            dense = _np.dot(self.exp_err_gen, self.unitary_postfactor)
        else:
            dense = self.exp_err_gen
        mx = _np.dot(rotnMx,dense)
        tGate = LindbladParameterizedGateMap(mx,self.unitary_postfactor,
                                          self.ham_basis, self.other_basis,
                                          self.cptp,self.nonham_diagonal_only,
                                          True, self.matrix_basis)
        #Note: truncate=True to be safe
        self.paramvals[:] = tGate.paramvals[:]
        self._construct_errgen()
        self.dirty = True


    def compose(self, otherGate):
        """
        Create and return a new gate that is the composition of this gate
        followed by otherGate, which *must be another 
        LindbladParameterizedGate*.  (For more general compositions between
        different types of gates, use the module-level compose function.)  The
        returned gate's matrix is equal to dot(this, otherGate).

        Parameters
        ----------
        otherGate : LindbladParameterizedGateMap
            The gate to compose to the right of this one.

        Returns
        -------
        LindbladParameterizedGateMap
        """
        assert(not self.sparse), "compose() not implemented for sparse LindbladParameterizedGateMap objects"
        assert( isinstance(otherGate, LindbladParameterizedGateMap) )
        assert(self.cptp == otherGate.cptp)
        assert(self.nonham_diagonal_only == otherGate.nonham_diagonal_only)

        if self.unitary_postfactor is not None:
            dense = _np.dot(self.exp_err_gen, self.unitary_postfactor)
        else:
            dense = self.exp_err_gen

        if otherGate.unitary_postfactor is not None:
            other_dense = _np.dot(otherGate.exp_err_gen, otherGate.unitary_postfactor)
        else:
            other_dense = otherGate.exp_err_gen

        composed_mx = _np.dot(dense, other_dense)
        return LindbladParameterizedGateMap(composed_mx,self.unitary_postfactor,
                                          self.ham_basis, self.other_basis,
                                          self.cptp,self.nonham_diagonal_only,
                                          True, self.matrix_basis)

    def __str__(self):
        s = "Lindblad Parameterized gate map with dim = %d, num params = %d\n" % \
            (self.err_gen.shape[0], self.num_params())
        return s


class LindbladParameterizedGate(LindbladBase,GateMatrix):
    """
    Encapsulates a gate matrix that is parameterized by a Lindblad-form
    expression, such that each parameter multiplies a particular term in
    the Lindblad form that is exponentiated to give the gate matrix up
    to an optional unitary prefactor).  The basis used by the Lindblad
    form is referred to as the "projection basis".
    """
    
    def __init__(self, gateMatrix, unitaryPostfactor=None,
                 ham_basis="pp", nonham_basis="pp", cptp=True,
                 nonham_diagonal_only=False, truncate=True, mxBasis="pp"):
        """
        Initialize a LindbladParameterizedGate object.

        Parameters
        ----------
        gateMatrix : numpy array
            a square 2D numpy array that gives the raw gate matrix, assumed to
            be in the `mxBasis` basis, to parameterize.  The shape of this
            array sets the dimension of the gate. If None, then it is assumed
            equal to `unitaryPostfactor` (which cannot also be None). The
            quantity `gateMatrix inv(unitaryPostfactor)` is parameterized via
            projection onto the Lindblad terms.
            
        unitaryPostfactor : numpy array, optional
            a square 2D numpy array of the same size of `gateMatrix` (if
            not None).  This matrix specifies a part of the gate action 
            to remove before parameterization via Lindblad projections.
            Typically, this is a target (desired) gate operation such 
            that only the erroneous part of the gate (i.e. the gate 
            relative to the target), which should be close to the identity,
            is parameterized.  If none, the identity is used by default.

        ham_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis is used to construct the Stochastic-type lindblad error
            Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        other_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis is used to construct the Stochastic-type lindblad error
            Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        cptp : bool, optional
            Whether or not the new gate should be constrained to CPTP.
            (if True, see behavior or `truncate`).

        nonham_diagonal_only : boolean, optional
            If True, only *diagonal* Stochastic (non-Hamiltonain) terms are
            included in the parameterization.

        truncate : bool, optional
            Whether to truncate the projections onto the Lindblad terms in
            order to preserve CPTP (when necessary).  If False, then an 
            error is thrown when `cptp == True` and when Lindblad projections
            result in a non-positive-definite matrix of non-Hamiltonian term
            coefficients.

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).
        """
        if gateMatrix is None:
            assert(unitaryPostfactor is not None), "arguments cannot both be None"
            gateMatrix = unitaryPostfactor
        if unitaryPostfactor is None:
            unitaryPostfactor = _np.identity(gateMatrix.shape[0],'d')
            
        self.unitary_postfactor = unitaryPostfactor
        
        #Init base from error generator: sets basis members and ultimately
        # the parameters in self.paramvals
        errgen = _gt.error_generator(gateMatrix, unitaryPostfactor, mxBasis, "logGTi")
        GateMatrix.__init__(self, self.unitary_postfactor) #sets self.dim
        LindbladBase.__init__(self, errgen, ham_basis, nonham_basis, cptp,
                              nonham_diagonal_only, truncate, mxBasis)

        assert(not self.sparse), "LindbladParameterizedGate cannot be sparse - use LindbladParameterizedGateMap"
        assert(not _np.array_equal(gateMatrix,unitaryPostfactor) or
               _np.allclose(self.paramvals,0.0,atol=1e-6))

        self._construct_matrix() # construct base from the parameters

        if not truncate:
            assert(not _np.array_equal(gateMatrix,self.base)), \
                   "Gate matrix must be truncated and truncate == False!"

            
    def _construct_matrix(self):
        """
        Build the internal gate matrix using the current parameters.
        """
        self._construct_errgen() # constructs self.err_gen
        self.exp_err_gen = _spl.expm(self.err_gen)
        matrix = _np.dot(self.exp_err_gen, self.unitary_postfactor)

        #gen0_pp = hamGens_pp[0] #_np.dot(self.leftTrans, _np.dot(self.hamGens[0], self.rightTrans))
        #print("DEBUG CONSTRUCTED BASE:\n", self.unitary_postfactor)
        #print("DEBUG CONSTRUCTED GEN:\n", lnd_error_gen)
        #print("DEBUG CONSTRUCTED EXP(GEN):\n", _spl.expm(lnd_error_gen))
        #print("DEBUG CONSTRUCTED EXP(GEN)*GEN0:\n", _np.dot(_spl.expm(lnd_error_gen), gen0_pp))
        #print("DEBUG CONSTRUCTED MX:\n", matrix)
        #print("DEBUG GEN0:\n", gen0_pp)

        assert(_np.linalg.norm(matrix.imag) < IMAG_TOL)
        assert(matrix.shape == (self.dim,self.dim))

        self.base = matrix.real
        self.base.flags.writeable = False
        self.base_deriv = None
        self.base_hessian = None        

        ##TEST FOR CP: DEBUG!!!
        #from ..tools import jamiolkowski as _jt
        #evals = _np.linalg.eigvals(_jt.jamiolkowski_iso(matrix,"pp"))
        #if min(evals) < -1e-6:
        #    print("Choi eigenvalues of final mx = ",sorted(evals))
        #    print("Choi eigenvalues of final mx (pp Choi) = ",sorted(_np.linalg.eigvals(_jt.jamiolkowski_iso(matrix,"pp","pp"))))
        #    print("Choi evals of exp(gen) = ", sorted(_np.linalg.eigvals(_jt.jamiolkowski_iso(self.exp_err_gen,"pp"))))
        #
        #    ham_error_gen = _np.einsum('i,ijk', hamCoeffs, self.hamGens)
        #    other_error_gen = _np.einsum('ij,ijkl', otherCoeffs, self.otherGens)
        #    ham_error_gen = _np.dot(self.leftTrans, _np.dot(ham_error_gen, self.rightTrans))
        #    other_error_gen = _np.dot(self.leftTrans, _np.dot(other_error_gen, self.rightTrans))
        #    ham_exp_err_gen = _spl.expm(ham_error_gen)
        #    other_exp_err_gen = _spl.expm(other_error_gen)
        #    print("Choi evals of exp(hamgen) = ", sorted(_np.linalg.eigvals(_jt.jamiolkowski_iso(ham_exp_err_gen,"pp"))))
        #    print("Choi evals of exp(othergen) = ", sorted(_np.linalg.eigvals(_jt.jamiolkowski_iso(other_exp_err_gen,"pp"))))
        #    print("Evals of otherCoeffs = ",sorted(_np.linalg.eigvals(otherCoeffs)))
        #    assert(False)
        

    def set_value(self, M):
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
        mx = GateMatrix.convert_to_matrix(M)
        if(mx.shape != (self.dim, self.dim)):
            raise ValueError("Argument must be a (%d,%d) matrix!" % (self.dim,self.dim))
        raise ValueError("Currently, directly setting the matrix of an" +
                         " LindbladParameterizedGate object is not" +
                         " supported.  Please use the from_vector method.")

    
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
        self.dirty = True


#    def old_deriv_wrt_params(self, wrtFilter=None):
#        """
#        Construct a matrix whose columns are the vectorized
#        derivatives of the flattened gate matrix with respect to a
#        single gate parameter.  Thus, each column is of length
#        gate_dim^2 and there is one column per gate parameter.
#
#        Returns
#        -------
#        numpy array
#            Array of derivatives, shape == (dimension^2, num_params)
#        """
#        if self.base_deriv is None:
#            d2 = self.dim
#            bsH = self.ham_basis_size
#            bsO = self.other_basis_size
#            TERM_TOL = 1e-12
#    
#            #Deriv wrt hamiltonian params
#            if bsH > 0:
#                dHdp = _np.einsum("ik,akl,lj->ija", self.leftTrans, self.hamGens, self.rightTrans)
#                series = last_commutant = term = dHdp; i=2
#                while _np.amax(_np.abs(term)) > TERM_TOL: #_np.linalg.norm(term)
#                    commutant = _np.einsum("ik,kja->ija",self.err_gen,last_commutant) - \
#                        _np.einsum("ika,kj->ija",last_commutant,self.err_gen)
#                    term = 1/_np.math.factorial(i) * commutant
#                    series += term #1/_np.math.factorial(i) * commutant
#                    last_commutant = commutant; i += 1
#                dH = _np.einsum('ika,kl,lj->ija', series, self.exp_err_gen, self.unitary_postfactor)
#                dH = dH.reshape((d2**2,bsH-1)) # [iFlattenedGate,iHamParam]
#                nHam = bsH-1
#            else:
#                dH = _np.empty( (d2**2,0), 'd') #so concat works below
#                nHam = 0
#    
#            #Deriv wrt other params
#            if bsO > 0:
#                if self.nonham_diagonal_only:
#                    otherParams = self.paramvals[nHam:]
#                
#                    # Derivative of exponent wrt other param; shape == [d2,d2,bs-1]
#                    if self.cptp:
#                        dOdp  = _np.einsum('alj,a->lja', self.otherGens, 2*otherParams)
#                    else:
#                        dOdp  = _np.einsum('alj->lja', self.otherGens)
#                
#                    #apply basis transform
#                    dOdp  = _np.einsum('lk,kna,nj->lja', self.leftTrans, dOdp, self.rightTrans)
#                    assert(_np.linalg.norm(_np.imag(dOdp)) < IMAG_TOL)
#                
#                    #take d(matrix-exp) using series approximation
#                    series = last_commutant = term = dOdp; i=2
#                    while _np.amax(_np.abs(term)) > TERM_TOL: #_np.linalg.norm(term)
#                        commutant = _np.einsum("ik,kja->ija",self.err_gen,last_commutant) - \
#                            _np.einsum("ika,kj->ija",last_commutant,self.err_gen)
#                        term = 1/_np.math.factorial(i) * commutant
#                        series += term #1/_np.math.factorial(i) * commutant
#                        last_commutant = commutant; i += 1
#                    dO = _np.einsum('ika,kl,lj->ija', series, self.exp_err_gen, self.unitary_postfactor) # dim [d2,d2,bs-1]
#                    dO = dO.reshape(d2**2,bsO-1)
#                
#                
#                else: #all lindblad terms included                
#                    if self.cptp:
#                        L,Lbar = self.Lmx,self.Lmx.conjugate()
#                        F1 = _np.tril(_np.ones((bsO-1,bsO-1),'d'))
#                        F2 = _np.triu(_np.ones((bsO-1,bsO-1),'d'),1) * 1j
#                
#                          # Derivative of exponent wrt other param; shape == [d2,d2,bs-1,bs-1]
#                        dOdp  = _np.einsum('amlj,mb,ab->ljab', self.otherGens, Lbar, F1) #only a >= b nonzero (F1)
#                        dOdp += _np.einsum('malj,mb,ab->ljab', self.otherGens, L, F1)    # ditto
#                        dOdp += _np.einsum('bmlj,ma,ab->ljab', self.otherGens, Lbar, F2) #only b > a nonzero (F2)
#                        dOdp += _np.einsum('mblj,ma,ab->ljab', self.otherGens, L, F2.conjugate()) # ditto
#                    else:
#                        F0 = _np.identity(bsO-1,'d')
#                        F1 = _np.tril(_np.ones((bsO-1,bsO-1),'d'),-1)
#                        F2 = _np.triu(_np.ones((bsO-1,bsO-1),'d'),1) * 1j
#                
#                          # Derivative of exponent wrt other param; shape == [d2,d2,bs-1,bs-1]
#                        dOdp  = _np.einsum('ablj,ab->ljab', self.otherGens, F0)  # a == b case
#                        dOdp += _np.einsum('ablj,ab->ljab', self.otherGens, F1) + \
#                                _np.einsum('balj,ab->ljab', self.otherGens, F1) # a > b (F1)
#                        dOdp += _np.einsum('balj,ab->ljab', self.otherGens, F2) - \
#                                _np.einsum('ablj,ab->ljab', self.otherGens, F2) # a < b (F2)
#                
#                    #apply basis transform
#                    dOdp  = _np.einsum('lk,knab,nj->ljab', self.leftTrans, dOdp, self.rightTrans)
#                    assert(_np.linalg.norm(_np.imag(dOdp)) < IMAG_TOL)
#                
#                    #take d(matrix-exp) using series approximation
#                    series = last_commutant = term = dOdp; i=2
#                    while _np.amax(_np.abs(term)) > TERM_TOL: #_np.linalg.norm(term)
#                        commutant = _np.einsum("ik,kjab->ijab",self.err_gen,last_commutant) - \
#                            _np.einsum("ikab,kj->ijab",last_commutant,self.err_gen)
#                        term = 1/_np.math.factorial(i) * commutant
#                        series += term #1/_np.math.factorial(i) * commutant
#                        last_commutant = commutant; i += 1
#                    dO = _np.einsum('ikab,kl,lj->ijab', series, self.exp_err_gen, self.unitary_postfactor) # dim [d2,d2,bs-1,bs-1]
#                    dO = dO.reshape(d2**2,(bsO-1)**2)
#            else:
#                dO = _np.empty( (d2**2,0), 'd') #so concat works below
#            
#            derivMx = _np.concatenate((dH,dO), axis=1)
#            assert(_np.linalg.norm(_np.imag(derivMx)) < IMAG_TOL)
#            derivMx = _np.real(derivMx)
#            self.base_deriv = derivMx
#
#            #check_deriv_wrt_params(self, derivMx, eps=1e-7)
#            #fd_deriv = finite_difference_deriv_wrt_params(self, eps=1e-7)
#            #derivMx = fd_deriv
#
#        if wrtFilter is None:
#            return self.base_deriv
#        else:
#            return _np.take( self.base_deriv, wrtFilter, axis=1 )

    def _dHdp(self):
        return self.hamGens.transpose((1,2,0)) #PRETRANS
        #return _np.einsum("ik,akl,lj->ija", self.leftTrans, self.hamGens, self.rightTrans)

    def _dOdp(self):
        bsH = self.ham_basis_size
        bsO = self.other_basis_size
        nHam = bsH-1 if (bsH > 0) else 0
        
        assert(bsO > 0),"Cannot construct dOdp when other_basis_size == 0!"
        if self.nonham_diagonal_only:
            otherParams = self.paramvals[nHam:]
            
            # Derivative of exponent wrt other param; shape == [d2,d2,bs-1]
            if self.cptp:
                dOdp  = _np.einsum('alj,a->lja', self.otherGens, 2*otherParams)
            else:
                dOdp  = _np.einsum('alj->lja', self.otherGens)
                            
        else: #all lindblad terms included                
            if self.cptp:
                L,Lbar = self.Lmx,self.Lmx.conjugate()
                F1 = _np.tril(_np.ones((bsO-1,bsO-1),'d'))
                F2 = _np.triu(_np.ones((bsO-1,bsO-1),'d'),1) * 1j
                
                  # Derivative of exponent wrt other param; shape == [d2,d2,bs-1,bs-1]
                dOdp  = _np.einsum('amlj,mb,ab->ljab', self.otherGens, Lbar, F1) #only a >= b nonzero (F1)
                dOdp += _np.einsum('malj,mb,ab->ljab', self.otherGens, L, F1)    # ditto
                dOdp += _np.einsum('bmlj,ma,ab->ljab', self.otherGens, Lbar, F2) #only b > a nonzero (F2)
                dOdp += _np.einsum('mblj,ma,ab->ljab', self.otherGens, L, F2.conjugate()) # ditto
            else:
                F0 = _np.identity(bsO-1,'d')
                F1 = _np.tril(_np.ones((bsO-1,bsO-1),'d'),-1)
                F2 = _np.triu(_np.ones((bsO-1,bsO-1),'d'),1) * 1j
            
                # Derivative of exponent wrt other param; shape == [d2,d2,bs-1,bs-1]
                dOdp  = _np.einsum('ablj,ab->ljab', self.otherGens, F0)  # a == b case
                dOdp += _np.einsum('ablj,ab->ljab', self.otherGens, F1) + \
                        _np.einsum('balj,ab->ljab', self.otherGens, F1) # a > b (F1)
                dOdp += _np.einsum('balj,ab->ljab', self.otherGens, F2) - \
                        _np.einsum('ablj,ab->ljab', self.otherGens, F2) # a < b (F2)

        # apply basis transform
        tr = len(dOdp.shape) #tensor rank
        assert( (tr-2) in (1,2)), "Currently, dodp can only have 1 or 2 derivative dimensions"

        #REMOVE IF PRETRANS: before we changed basis right away (in _set_params_from_errgen)
        #if tr == 3:
        #    dOdp  = _np.einsum('lk,kna,nj->lja', self.leftTrans, dOdp, self.rightTrans)
        #elif tr == 4:
        #    dOdp  = _np.einsum('lk,knab,nj->ljab', self.leftTrans, dOdp, self.rightTrans)
        assert(_np.linalg.norm(_np.imag(dOdp)) < IMAG_TOL)
        return _np.real(dOdp)


    def _d2Odp2(self):
        bsH = self.ham_basis_size
        bsO = self.other_basis_size
        nHam = bsH-1 if (bsH > 0) else 0
        d2 = self.dim
        
        assert(bsO > 0),"Cannot construct dOdp when other_basis_size == 0!"
        if self.nonham_diagonal_only:
            otherParams = self.paramvals[nHam:]
            nP = len(otherParams); assert(nP == bsO-1)
            
            # Derivative of exponent wrt other param; shape == [d2,d2,bs-1]
            if self.cptp:
                d2Odp2  = _np.einsum('alj,aq->ljaq', self.otherGens, 2*_np.identity(nP,'d'))
            else:
                d2Odp2  = _np.zeros([d2,d2,nP,nP],'d')
                            
        else: #all lindblad terms included
            nP = bsO-1
            if self.cptp:
                d2Odp2  = _np.zeros([d2,d2,nP,nP,nP,nP],'complex') #yikes! maybe make this SPARSE in future?
                
                #Note: correspondence w/Erik's notes: a=alpha, b=beta, q=gamma, r=delta
                # indices of d2Odp2 are [i,j,a,b,q,r]
                
                def iter_base_ab_qr(ab_inc_eq, qr_inc_eq):
                    """ Generates (base,ab,qr) tuples such that `base` runs over
                        all possible 'other' params and 'ab' and 'qr' run over
                        parameter indices s.t. ab > base and qr > base.  If
                        ab_inc_eq == True then the > becomes a >=, and likewise
                        for qr_inc_eq.  Used for looping over nonzero hessian els. """
                    for _base in range(nP):
                        start_ab = _base if ab_inc_eq else _base+1
                        start_qr = _base if qr_inc_eq else _base+1
                        for _ab in range(start_ab,nP):
                            for _qr in range(start_qr,nP):
                                yield (_base,_ab,_qr)
                                
                for base,a,q in iter_base_ab_qr(True,True): # Case1: base=b=r, ab=a, qr=q
                    d2Odp2[:,:,a,base,q,base] = self.otherGens[a,q] + self.otherGens[q,a]
                for base,a,r in iter_base_ab_qr(True,False): # Case2: base=b=q, ab=a, qr=r
                    d2Odp2[:,:,a,base,base,r] = -1j*self.otherGens[a,r] + 1j*self.otherGens[r,a]
                for base,b,q in iter_base_ab_qr(False,True): # Case3: base=a=r, ab=b, qr=q
                    d2Odp2[:,:,base,b,q,base] = 1j*self.otherGens[b,q] - 1j*self.otherGens[q,b]
                for base,b,r in iter_base_ab_qr(False,False): # Case4: base=a=q, ab=b, qr=r
                    d2Odp2[:,:,base,b,base,r] = self.otherGens[b,r] + self.otherGens[r,b]
                
            else:
                d2Odp2  = _np.zeros([d2,d2,nP,nP,nP,nP],'d') #all params linear

        # apply basis transform
        tr = len(d2Odp2.shape) #tensor rank
        assert( (tr-2) in (2,4)), "Currently, d2Odp2 can only have 2 or 4 derivative dimensions"

        #REMOVE IF PRETRANS: before we changed basis right away (in _set_params_from_errgen)
        #if tr == 4:
        #    d2Odp2  = _np.einsum('lk,knaq,nj->ljaq', self.leftTrans, d2Odp2, self.rightTrans)
        #elif tr == 6:
        #    d2Odp2  = _np.einsum('lk,knabqr,nj->ljabqr', self.leftTrans, d2Odp2, self.rightTrans)
        assert(_np.linalg.norm(_np.imag(d2Odp2)) < IMAG_TOL)
        return _np.real(d2Odp2)


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
        if self.base_deriv is None:
            d2 = self.dim
            bsH = self.ham_basis_size
            bsO = self.other_basis_size
    
            #Deriv wrt hamiltonian params
            if bsH > 0:
                dexpL = _dexpX(self.err_gen, self._dHdp(), self.exp_err_gen,
                               self.unitary_postfactor)
                dH = dexpL.reshape((d2**2,bsH-1)) # [iFlattenedGate,iHamParam]
            else:
                dH = _np.empty( (d2**2,0), 'd') #so concat works below
    
            #Deriv wrt other params
            if bsO > 0:
                dexpL = _dexpX(self.err_gen, self._dOdp(), self.exp_err_gen,
                               self.unitary_postfactor)

                #Reshape so index as [iFlattenedGate,iOtherParam]
                # 2nd dim will be bsO-1 or (bsO-1)**2 depending on tensor rank
                dO = dexpL.reshape((d2**2,-1)) 
            else:
                dO = _np.empty( (d2**2,0), 'd') #so concat works below
            
            derivMx = _np.concatenate((dH,dO), axis=1)
            assert(_np.linalg.norm(_np.imag(derivMx)) < IMAG_TOL)
            derivMx = _np.real(derivMx)
            self.base_deriv = derivMx

            #check_deriv_wrt_params(self, derivMx, eps=1e-7)
            #fd_deriv = finite_difference_deriv_wrt_params(self, eps=1e-7)
            #derivMx = fd_deriv

        if wrtFilter is None:
            return self.base_deriv
        else:
            return _np.take( self.base_deriv, wrtFilter, axis=1 )

    def has_nonzero_hessian(self):
        """ 
        Returns whether this gate has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return True

    def hessian_wrt_params(self, wrtFilter1=None, wrtFilter2=None):
        """
        Construct the Hessian of this gate with respect to it's parameters.

        This function returns a tensor whose first axis corresponds to the
        flattened gate matrix and whose 2nd and 3rd axes correspond to the
        parameters that are differentiated with respect to.

        Parameters
        ----------
        wrtFilter1, wrtFilter2 : list
            Lists of indices of the paramters to take first and second
            derivatives with respect to.  If None, then derivatives are
            taken with respect to all of the gate's parameters.

        Returns
        -------
        numpy array
            Hessian with shape (dimension^2, num_params1, num_params2)
        """
        if self.base_hessian is None:
            d2 = self.dim
            bsH = self.ham_basis_size
            bsO = self.other_basis_size
            nHam = bsH-1 if (bsH > 0) else 0

            #Split hessian in 4 pieces:   d2H  |  dHdO
            #                             dHdO |  d2O
            # But only d2O is non-zero - and only when cptp == True

            nTotParams = self.num_params()
            hessianMx = _np.zeros( (d2**2, nTotParams, nTotParams), 'd' )
    
            #Deriv wrt other params
            if bsO > 0: #if there are any "other" params
                dOdp = self._dOdp()
                d2Odp2 = self._d2Odp2()
                tr = len(dOdp.shape)
                    
                series, series2 = _d2expSeries(self.err_gen, dOdp, d2Odp2)
                term1 = series2
                if tr == 3: #one deriv dimension "a"
                    term2 = _np.einsum("ija,jkq->ikaq",series,series)
                    d2expL = _np.einsum("ikaq,kl,lj->ijaq", term1+term2,
                                      self.exp_err_gen, self.unitary_postfactor)
                    dO = d2expL.reshape((d2**2,bsO-1,bsO-1))
                    #d2expL.shape = (d2**2,bsO-1,bsO-1); dO = d2expL

                elif tr == 4: #two deriv dimension "ab"
                    term2 = _np.einsum("ijab,jkqr->ikabqr",series,series)
                    d2expL = _np.einsum("ikabqr,kl,lj->ijabqr", term1+term2,
                                      self.exp_err_gen, self.unitary_postfactor)
                    dO = d2expL.reshape((d2**2, (bsO-1)**2, (bsO-1)**2 ))
                    #d2expL.shape = (d2**2, (bsO-1)**2, (bsO-1)**2); dO = d2expL

                #dO has been reshape so index as [iFlattenedGate,iDeriv1,iDeriv2]
                assert(_np.linalg.norm(_np.imag(dO)) < IMAG_TOL)
                hessianMx[:,nHam:,nHam:] = _np.real(dO) # d2O block of hessian

            self.base_hessian = hessianMx

            #TODO: check hessian with finite difference here?
            
        if wrtFilter1 is None:
            if wrtFilter2 is None:
                return self.base_hessian
            else:
                return _np.take(self.base_hessian, wrtFilter2, axis=2 )
        else:
            if wrtFilter2 is None:
                return _np.take(self.base_hessian, wrtFilter1, axis=1 )
            else:
                return _np.take( _np.take(self.base_hessian, wrtFilter1, axis=1),
                                 wrtFilter2, axis=2 )


    def copy(self, parent=None):
        """
        Copy this gate.

        Returns
        -------
        Gate
            A copy of this gate.
        """
        #Construct new gate with dummy identity mx
        newGate = LindbladParameterizedGate(None,self.unitary_postfactor.copy(),
                                            _copy.deepcopy(self.ham_basis),
                                            _copy.deepcopy(self.other_basis),
                                            self.cptp,self.nonham_diagonal_only,
                                            True, _copy.deepcopy(self.matrix_basis))
        
        #Deep copy data
        newGate.paramvals = self.paramvals.copy()
        newGate._construct_matrix()

        return self._copy_gpindices( newGate, parent )


    def transform(self, S):
        """
        Update gate matrix G with inv(S) * G * S,

        Generally, the transform function updates the *parameters* of 
        the gate such that the resulting gate matrix is altered as 
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.
        """
        if isinstance(S, _gaugegroup.UnitaryGaugeGroupElement) or \
           isinstance(S, _gaugegroup.TPSpamGaugeGroupElement):
            U = S.get_transform_matrix()
            Uinv = S.get_transform_matrix_inverse()

            #just conjugate postfactor and Lindbladian exponent by U:
            self.unitary_postfactor = _np.dot(Uinv,_np.dot(self.unitary_postfactor, U))
            self.err_gen = _np.dot(Uinv,_np.dot(self.err_gen, U))
            self._set_params_from_errgen(self.err_gen, truncate=True)
            self._construct_matrix()
            self.dirty = True
            #Note: truncate=True above because some unitary transforms seem to
            ## modify eigenvalues to be negative beyond the tolerances
            ## checked when truncate == False.  I'm not sure why this occurs,
            ## since a true unitary should map CPTP -> CPTP...

            #CHECK WITH OLD (passes) TODO move to unit tests?
            #tMx = _np.dot(Uinv,_np.dot(self.base, U)) #Move above for checking
            #tGate = LindbladParameterizedGate(tMx,self.unitary_postfactor,
            #                                self.ham_basis, self.other_basis,
            #                                self.cptp,self.nonham_diagonal_only,
            #                                True, self.matrix_basis)
            #assert(_np.linalg.norm(tGate.paramvals - self.paramvals) < 1e-6)
        else:
            raise ValueError("Invalid transform for this LindbladParameterizedGate: type %s"
                             % str(type(S)))


    def depolarize(self, amount):
        """
        Depolarize this gate by the given `amount`.

        Generally, the depolarize function updates the *parameters* of 
        the gate such that the resulting gate matrix is depolarized.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. In standard
            bases, depolarization corresponds to multiplying the gate matrix
            by a diagonal matrix whose first diagonal element (corresponding
            to the identity) equals 1.0 and whose subsequent elements 
            (corresponding to non-identity basis elements) equal
            `1.0 - amount[i]` (or just `1.0 - amount` if `amount` is a
            float).

        Returns
        -------
        None
        """
        if isinstance(amount,float):
            D = _np.diag( [1]+[1-amount]*(self.dim-1) )
        else:
            assert(len(amount) == self.dim-1)
            D = _np.diag( [1]+list(1.0 - _np.array(amount,'d')) )
        mx = _np.dot(D,self.base)
        tGate = LindbladParameterizedGate(mx,self.unitary_postfactor,
                                          self.ham_basis, self.other_basis,
                                          self.cptp,self.nonham_diagonal_only,
                                          True, self.matrix_basis)

        #Note: truncate=True to be safe
        self.paramvals[:] = tGate.paramvals[:]
        self._construct_matrix()
        self.dirty = True


    def rotate(self, amount, mxBasis="gm"):
        """
        Rotate this gate by the given `amount`.

        Generally, the rotate function updates the *parameters* of 
        the gate such that the resulting gate matrix is rotated.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : tuple of floats, optional
            Specifies the rotation "coefficients" along each of the non-identity
            Pauli-product axes.  The gate's matrix `G` is composed with a
            rotation operation `R`  (so `G` -> `dot(R, G)` ) where `R` is the
            unitary superoperator corresponding to the unitary operator 
            `U = exp( sum_k( i * rotate[k] / 2.0 * Pauli_k ) )`.  Here `Pauli_k`
            ranges over all of the non-identity un-normalized Pauli operators.

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        Returns
        -------
        None
        """
        rotnMx = _gt.rotation_gate_mx(amount,mxBasis)
        mx = _np.dot(rotnMx,self.base)
        tGate = LindbladParameterizedGate(mx,self.unitary_postfactor,
                                          self.ham_basis, self.other_basis,
                                          self.cptp,self.nonham_diagonal_only,
                                          True, self.matrix_basis)
        #Note: truncate=True to be safe
        self.paramvals[:] = tGate.paramvals[:]
        self._construct_matrix()
        self.dirty = True


    def compose(self, otherGate):
        """
        Create and return a new gate that is the composition of this gate
        followed by otherGate, which *must be another 
        LindbladParameterizedGate*.  (For more general compositions between
        different types of gates, use the module-level compose function.)  The
        returned gate's matrix is equal to dot(this, otherGate).

        Parameters
        ----------
        otherGate : LindbladParameterizedGate
            The gate to compose to the right of this one.

        Returns
        -------
        LindbladParameterizedGate
        """
        assert( isinstance(otherGate, LindbladParameterizedGate) )
        assert(self.cptp == otherGate.cptp)
        assert(self.nonham_diagonal_only == otherGate.nonham_diagonal_only)

        composed_mx = _np.dot(self, otherGate)
        return LindbladParameterizedGate(composed_mx,self.unitary_postfactor,
                                          self.ham_basis, self.other_basis,
                                          self.cptp,self.nonham_diagonal_only,
                                          True, self.matrix_basis)


    def __str__(self):
        s = "Lindblad Parameterized gate with shape %s, num params = %d\n" % \
            (str(self.base.shape), self.num_params())
        s += _mt.mx_to_string(self.base, width=5, prec=1)
        return s



def _dexpSeries(X, dX):
    TERM_TOL = 1e-12
    tr = len(dX.shape) #tensor rank of dX; tr-2 == # of derivative dimensions
    assert( (tr-2) in (1,2)), "Currently, dX can only have 1 or 2 derivative dimensions"
    series = last_commutant = term = dX; i=2

    #take d(matrix-exp) using series approximation
    while _np.amax(_np.abs(term)) > TERM_TOL: #_np.linalg.norm(term)
        if tr == 3:
            commutant = _np.einsum("ik,kja->ija",X,last_commutant) - \
                        _np.einsum("ika,kj->ija",last_commutant,X)
        elif tr == 4:
            commutant = _np.einsum("ik,kjab->ijab",X,last_commutant) - \
                    _np.einsum("ikab,kj->ijab",last_commutant,X)
        term = 1/_np.math.factorial(i) * commutant
        series += term #1/_np.math.factorial(i) * commutant
        last_commutant = commutant; i += 1
    return series

def _d2expSeries(X, dX, d2X):
    TERM_TOL = 1e-12
    tr = len(dX.shape) #tensor rank of dX; tr-2 == # of derivative dimensions
    tr2 = len(d2X.shape) #tensor rank of dX; tr-2 == # of derivative dimensions
    assert( (tr-2,tr2-2) in [(1,2),(2,4)]), "Current support for only 1 or 2 derivative dimensions"
    
    series = term = last_commutant = dX
    series2 = last_commutant2 = term2 = d2X
    i=2

    #take d(matrix-exp) using series approximation
    while _np.amax(_np.abs(term)) > TERM_TOL or _np.amax(_np.abs(term2)) > TERM_TOL:
        if tr == 3:
            commutant = _np.einsum("ik,kja->ija",X,last_commutant) - \
                        _np.einsum("ika,kj->ija",last_commutant,X)
            commutant2A = _np.einsum("ikq,kja->ijaq",dX,last_commutant) - \
                    _np.einsum("ika,kjq->ijaq",last_commutant,dX)
            commutant2B = _np.einsum("ik,kjaq->ijaq",X,last_commutant2) - \
                    _np.einsum("ikaq,kj->ijaq",last_commutant2,X)

        elif tr == 4:
            commutant = _np.einsum("ik,kjab->ijab",X,last_commutant) - \
                    _np.einsum("ikab,kj->ijab",last_commutant,X)
            commutant2A = _np.einsum("ikqr,kjab->ijabqr",dX,last_commutant) - \
                    _np.einsum("ikab,kjqr->ijabqr",last_commutant,dX)
            commutant2B = _np.einsum("ik,kjabqr->ijabqr",X,last_commutant2) - \
                    _np.einsum("ikabqr,kj->ijabqr",last_commutant2,X)

        term = 1/_np.math.factorial(i) * commutant
        term2 = 1/_np.math.factorial(i) * (commutant2A + commutant2B)
        series += term 
        series2 += term2
        last_commutant = commutant
        last_commutant2 = (commutant2A + commutant2B)
        i += 1        
    return series, series2

def _dexpX(X,dX,expX=None,postfactor=None):
    """ 
    Computes the derivative of the exponential of X(t) using
    the Haddamard lemma series expansion.

    Parameters
    ----------
    X : ndarray
        The 2-tensor being exponentiated

    dX : ndarray
        The derivative of X; can be either a 3- or 4-tensor where the
        3rd+ dimensions are for (multi-)indexing the parameters which
        are differentiated w.r.t.  For example, in the simplest case
        dX is a 3-tensor s.t. dX[i,j,p] == d(X[i,j])/dp.

    expX : ndarray, optional
        The value of `exp(X)`, which can be specified in order to save
        a call to `scipy.linalg.expm`.  If None, then the value is
        computed internally.

    postfactor : ndarray, optional
        A 2-tensor of the same shape as X that post-multiplies the
        result.

    Returns
    -------
    ndarray
        The derivative of `exp(X)*postfactor` given as a tensor with the
        same shape and axes as `dX`.
    """
    tr = len(dX.shape) #tensor rank of dX; tr-2 == # of derivative dimensions
    assert( (tr-2) in (1,2)), "Currently, dX can only have 1 or 2 derivative dimensions"

    series = _dexpSeries(X,dX)
    if expX is None: expX = _spl.expm(X)
    
    if tr == 3:
        dExpX = _np.einsum('ika,kj->ija', series, expX)
        if postfactor is not None:
            dExpX = _np.einsum('ila,lj->ija', dExpX, postfactor)
    elif tr == 4:
        dExpX = _np.einsum('ikab,kj->ijab', series, expX)
        if postfactor is not None:
            dExpX = _np.einsum('ilab,lj->ijab', dExpX, postfactor)
            
    return dExpX



class TPInstrumentGate(GateMatrix):
    """
    A partial implementation of :class:`Gate` which encapsulates an element of a
    :class:`TPInstrument`.  Instances rely on their parent being a 
    `TPInstrument`.
    """

    def __init__(self, param_gates, index):
        """
        Initialize a TPInstrumentGate object.

        Parameters
        ----------
        param_gates : list of Gate objects
            A list of the underlying gate objects which constitute a simple
            parameterization of a :class:`TPInstrument`.  Namely, this is
            the list of [MT,D1,D2,...Dn] gates which parameterize *all* of the
            `TPInstrument`'s elements.

        index : int
            The index indicating which element of the `TPInstrument` the
            constructed object is.  Must be in the range 
            `[0,len(param_gates)-1]`.
        """
        self.param_gates = param_gates
        self.index = index
        GateMatrix.__init__(self, _np.identity(param_gates[0].dim,'d')) #Note: sets self.gpindices
        self._construct_matrix()

        #Set our own parent and gpindices based on param_gates
        # (this breaks the usual paradigm of having the parent object set these,
        #  but the exception is justified b/c the parent has set these members
        #  of the underlying 'param_gates' gates)
        self.dependents = [0,index+1] if index < len(param_gates)-1 \
                          else list(range(len(param_gates)))
          #indices into self.param_gates of the gates this gate depends on
        self.set_gpindices(_slct.list_to_slice(
            _np.concatenate( [ param_gates[i].gpindices_as_array()
                               for i in self.dependents ], axis=0),True,False),
                           param_gates[0].parent) #use parent of first param gate
                                                  # (they should all be the same)


    def _construct_matrix(self):
        """
        Mi = Di + MT for i = 1...(n-1)
           = -(n-2)*MT-sum(Di) = -(n-2)*MT-[(MT-Mi)-n*MT] for i == (n-1)
        """
        nEls = len(self.param_gates)
        if self.index < nEls-1:
            self.base = _np.asarray( self.param_gates[self.index+1]
                                     + self.param_gates[0] )
        else:
            assert(self.index == nEls-1), \
                "Invalid index %d > %d" % (self.index,nEls-1)
            self.base = _np.asarray( -sum(self.param_gates)
                                     -(nEls-3)*self.param_gates[0] )
        
        assert(self.base.shape == (self.dim,self.dim))
        self.base.flags.writeable = False

        

    def set_value(self, M):
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
        raise ValueError("Cannot set the value of a TPInstrumentGate directly!")


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
        Np = len(self.gpindices_as_array())
        derivMx = _np.zeros((self.dim**2,Np),'d')
        Nels = len(self.param_gates)

        off = 0
        if self.index < Nels-1: # matrix = Di + MT = param_gates[index+1] + param_gates[0]
            for i in [self.index+1, 0]:
                Np = self.param_gates[i].num_params()
                derivMx[:,off:off+Np] = self.param_gates[i].deriv_wrt_params()
                off += Np

        else: # matrix = -(nEls-1)*MT-sum(Di)
            Np = self.param_gates[0].num_params()
            derivMx[:,off:off+Np] = -(Nels-1)*self.param_gates[0].deriv_wrt_params()
            off += Np

            for i in range(1,Nels):
                Np = self.param_gates[i].num_params()
                derivMx[:,off:off+Np] = -self.param_gates[i].deriv_wrt_params()
                off += Np
        
        if wrtFilter is None:
            return derivMx
        else:
            return _np.take( derivMx, wrtFilter, axis=1 )

        
    def has_nonzero_hessian(self):
        """ 
        Returns whether this gate has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return False



    def from_vector(self, v):
        """
        Initialize this partially-implemented gate using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of parameters.

        Returns
        -------
        None
        """
        #Rely on the Instrument ordering of it's elements: if we're being called
        # to init from v then this is within the context of a TPInstrument's gates
        # having been compiled and now being initialized from a vector (within a
        # calculator).  We rely on the Instrument elements having their
        # from_vector() methods called in self.index order.
        
        if self.index < len(self.param_gates)-1: #final element doesn't need to init any param gates
            for i in self.dependents: #re-init all my dependents (may be redundant)
                if i == 0 and self.index > 0: continue # 0th param-gate already init by index==0 element
                pgate_local_inds = _gatesetmember._decompose_gpindices(
                    self.gpindices, self.param_gates[i].gpindices)
                self.param_gates[i].from_vector( v[pgate_local_inds] )
                
        self._construct_matrix()


    def copy(self, parent=None):
        """
        Copy this gate.

        Returns
        -------
        Gate
            A copy of this gate.
        """
        assert(parent is None),"TPInstrumentGate set's it's own parent, so `parent` arg of copy must be None"
        return self._copy_gpindices( TPInstrumentGate(self.param_gates,self.index), parent)
          #Note: this doesn't deep-copy self.param_gates, since the TPInstrumentGate doesn't really own these.

    def __str__(self):
        s = "TPInstrumentGate with shape %s\n" % str(self.base.shape)
        s += _mt.mx_to_string(self.base, width=4, prec=2)
        return s


    
class ComposedGate(GateMatrix):
    """
    A gate that is the composition of a number of matrix factors (possibly other gates).
    """
    
    def __init__(self, gates_to_compose):
        """
        Creates a new ComposedGate.

        Parameters
        ----------
        gates_to_compose : list
            A list of 2D numpy arrays (matrices) and/or `GateMatrix`-derived
            objects that are composed to form this gate.  Elements are composed
            with vectors  in  *left-to-right* ordering, maintaining the same
            convention as gate sequences in pyGSTi.  Note that this is
            *opposite* from standard matrix multiplication order.
        """
        assert(len(gates_to_compose) > 0), "Must compose at least one gate!"
        self.factorgates = gates_to_compose
        
        dim = gates_to_compose[0].dim
        assert(all([dim == gate.dim for gate in gates_to_compose])), \
            "All gates must have the same dimension!"

        GateMatrix.__init__(self, _np.identity(dim,'d'))
        self._construct_matrix()

    def allocate_gpindices(self, startingIndex, parent):
        """
        Sets gpindices array for this object or any objects it
        contains (i.e. depends upon).  Indices may be obtained
        from contained objects which have already been initialized
        (e.g. if a contained object is shared with other
         top-level objects), or given new indices starting with
        `startingIndex`.

        Parameters
        ----------
        startingIndex : int
            The starting index for un-allocated parameters.

        parent : GateSet or GateSetMember
            The parent whose parameter array gpindices references.

        Returns
        -------
        num_new: int
            The number of *new* allocated parameters (so 
            the parent should mark as allocated parameter
            indices `startingIndex` to `startingIndex + new_new`).
        """
        #figure out how many parameters we need to allocate based on
        # which factorgates still need to be "allocated" within the
        # parent's parameter array:
        tot_new_params = 0
        all_gpindices = []
        #print(">>> DB COMPOSEDGATE %d allocating: " % id(self), [id(g) for g in self.factorgates]) #remove enumerate below too!
        for gate in self.factorgates:
            num_new_params = gate.allocate_gpindices( startingIndex, parent ) # *same* parent as this ComposedGate
            startingIndex += num_new_params
            tot_new_params += num_new_params
            all_gpindices.extend( gate.gpindices_as_array() )
            #print("  sub-gate%d: %d new params: indices = %s" % (i,num_new_params, str(gate.gpindices)))
        #print("<<< DB COMPOSEDGATE done allocating")

        _gatesetmember.GateSetMember.set_gpindices(
            self, _slct.list_to_slice(all_gpindices, array_ok=True), parent)
        return tot_new_params

    
    def set_gpindices(self, gpindices, parent, memo=None):
        """
        Set the parent and indices into the parent's parameter vector that
        are used by this GateSetMember object.

        Parameters
        ----------
        gpindices : slice or integer ndarray
            The indices of this objects parameters in its parent's array.

        parent : GateSet or GateSetMember
            The parent whose parameter array gpindices references.

        Returns
        -------
        None
        """
        if memo is None: memo = set()
        elif id(self) in memo: return
        memo.add(id(self))
        
        #must set the gpindices of self.factorgates based on new
        my_old_gpindices = self.gpindices
        for gate in self.factorgates:
            if id(gate) in memo: continue # already processed
            rel_gate_gpindices = _gatesetmember._decompose_gpindices(
                my_old_gpindices, gate.gpindices)
            new_gate_gpindices = _gatesetmember._compose_gpindices(
                gpindices, rel_gate_gpindices)
            gate.set_gpindices(new_gate_gpindices, parent, memo)
            
        _gatesetmember.GateSetMember.set_gpindices(self, gpindices, parent)
    

    def _construct_matrix(self):
        mx = self.factorgates[0]
        for gate in self.factorgates[1:]:
            mx = _np.dot(gate,mx)
        self.base = mx
        assert(self.base.shape == (self.dim,self.dim))
        self.base.flags.writeable = False
        
    
    def num_params(self):
        """
        Get the number of independent parameters which specify this gate.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return len(self.gpindices_as_array())


    def to_vector(self):
        """
        Get the gate parameters as an array of values.

        Returns
        -------
        numpy array
            The gate parameters as a 1D array with length num_params().
        """
        v = _np.empty(self.num_params(), 'd')
        for gate in self.factorgates:
            factorgate_local_inds = _gatesetmember._decompose_gpindices(
                    self.gpindices, gate.gpindices)
            v[factorgate_local_inds] = gate.to_vector()
        return v


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
        assert(len(v) == self.num_params())
        for gate in self.factorgates:
            factorgate_local_inds = _gatesetmember._decompose_gpindices(
                    self.gpindices, gate.gpindices)
            gate.from_vector( v[factorgate_local_inds] )
        self.dirty = True


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
        derivMx = _np.zeros( (self.dim,self.dim, self.num_params()), 'd')
        
        #Product rule to compute jacobian
        for i,gate in enumerate(self.factorgates): # loop over the gate we differentiate wrt
            if gate.num_params() == 0: continue #no contribution
            deriv = gate.deriv_wrt_params(None) #TODO: use filter?? / make relative to this gate...
            deriv.shape = (self.dim,self.dim,gate.num_params())

            if i > 0: # factors before ith
                pre = self.factorgates[0]
                for gateA in self.factorgates[1:i]:
                    pre = _np.dot(gateA,pre)
                deriv = _np.einsum("ija,jk->ika", deriv, pre )

            if i+1 < len(self.factorgates): # factors after ith
                post = self.factorgates[i+1]
                for gateA in self.factorgates[i+2:]:
                    post = _np.dot(gateA,post)
                deriv = _np.einsum("ij,jka->ika", post, deriv )

            derivMx[:,:,gate.gpindices] += deriv

        derivMx.shape = (self.dim**2, self.num_params())
        if wrtFilter is None:
            return derivMx
        else:
            return _np.take( derivMx, wrtFilter, axis=1 )


    def has_nonzero_hessian(self):
        """ 
        Returns whether this gate has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return any([gate.has_nonzero_hessian() for gate in self.factorgates])


    def copy(self, parent=None):
        """
        Copy this gate.

        Returns
        -------
        Gate
            A copy of this gate.
        """
        return self._copy_gpindices( ComposedGate([gate.copy(parent) for gate in self.factorgates]), parent )


    def transform(self, S):
        """
        Update gate matrix G with inv(S) * G * S,

        Generally, the transform function updates the *parameters* of 
        the gate such that the resulting gate matrix is altered as 
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        In this particular case any TP gauge transformation is possible,
        i.e. when `S` is an instance of `TPGaugeGroupElement` or 
        corresponds to a TP-like transform matrix.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.
        """
        for gate in self.factorgates:
            gate.transform(S)

            
    def depolarize(self, amount):
        """
        Depolarize this gate by the given `amount`.

        Generally, the depolarize function updates the *parameters* of 
        the gate such that the resulting gate matrix is depolarized.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. In standard
            bases, depolarization corresponds to multiplying the gate matrix
            by a diagonal matrix whose first diagonal element (corresponding
            to the identity) equals 1.0 and whose subsequent elements 
            (corresponding to non-identity basis elements) equal
            `1.0 - amount[i]` (or just `1.0 - amount` if `amount` is a
            float).

        Returns
        -------
        None
        """
        raise ValueError("Can't depolarize a ComposedGate")


    def rotate(self, amount, mxBasis="gm"):
        """
        Rotate this gate by the given `amount`.

        Generally, the rotate function updates the *parameters* of 
        the gate such that the resulting gate matrix is rotated.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : tuple of floats, optional
            Specifies the rotation "coefficients" along each of the non-identity
            Pauli-product axes.  The gate's matrix `G` is composed with a
            rotation operation `R`  (so `G` -> `dot(R, G)` ) where `R` is the
            unitary superoperator corresponding to the unitary operator 
            `U = exp( sum_k( i * rotate[k] / 2.0 * Pauli_k ) )`.  Here `Pauli_k`
            ranges over all of the non-identity un-normalized Pauli operators.

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        Returns
        -------
        None
        """
        raise ValueError("Can't rotate a ComposedGate")


    def compose(self, otherGate):
        """
        Create and return a new gate that is the composition of this gate
        followed by otherGate, which *must be another EmbeddedGate*.
        (For more general compositions between different types of gates, use
        the module-level compose function.)  The returned gate's matrix is
        equal to dot(this, otherGate).

        Parameters
        ----------
        otherGate : EmbeddedGate
            The gate to compose to the right of this one.

        Returns
        -------
        EmbeddedGate
        """
        raise NotImplementedError("TODO Later")


    def __str__(self):
        """ Return string representation """
        s = "Composed gate:\n"
        s += _mt.mx_to_string(self.base, width=4, prec=2)
        return s

    
class ComposedGateMap(GateMap):
    """
    A gate map that is the composition of a number of map-like factors (possibly
    other `GateMap`s)
    """
    
    def __init__(self, gates_to_compose):
        """
        Creates a new ComposedGateMap.

        Parameters
        ----------
        gates_to_compose : list
            List of `Gate`-derived objects (`GateMatrix`- or `GateMap`-derived)
            that are composed to form this gate map.  Elements are composed
            with vectors  in  *left-to-right* ordering, maintaining the same
            convention as gate sequences in pyGSTi.  Note that this is
            *opposite* from standard matrix multiplication order.
        """
        assert(len(gates_to_compose) > 0), "Must compose at least one gate!"
        self.factorgates = gates_to_compose
        
        dim = gates_to_compose[0].dim
        assert(all([dim == gate.dim for gate in gates_to_compose])), \
            "All gates must have the same dimension!"
        
        GateMap.__init__(self, dim)


    def allocate_gpindices(self, startingIndex, parent):
        """
        Sets gpindices array for this object or any objects it
        contains (i.e. depends upon).  Indices may be obtained
        from contained objects which have already been initialized
        (e.g. if a contained object is shared with other
         top-level objects), or given new indices starting with
        `startingIndex`.

        Parameters
        ----------
        startingIndex : int
            The starting index for un-allocated parameters.

        parent : GateSet or GateSetMember
            The parent whose parameter array gpindices references.

        Returns
        -------
        num_new: int
            The number of *new* allocated parameters (so 
            the parent should mark as allocated parameter
            indices `startingIndex` to `startingIndex + new_new`).
        """
        #figure out how many parameters we need to allocate based on
        # which factorgates still need to be "allocated" within the
        # parent's parameter array:
        tot_new_params = 0
        all_gpindices = []
        for gate in self.factorgates:
            num_new_params = gate.allocate_gpindices( startingIndex, parent ) # *same* parent as this ComposedGate
            startingIndex += num_new_params
            tot_new_params += num_new_params
            all_gpindices.extend( gate.gpindices_as_array() )

        _gatesetmember.GateSetMember.set_gpindices(
            self, _slct.list_to_slice(all_gpindices, array_ok=True), parent)
        return tot_new_params

    
    def set_gpindices(self, gpindices, parent, memo=None):
        """
        Set the parent and indices into the parent's parameter vector that
        are used by this GateSetMember object.

        Parameters
        ----------
        gpindices : slice or integer ndarray
            The indices of this objects parameters in its parent's array.

        parent : GateSet or GateSetMember
            The parent whose parameter array gpindices references.

        Returns
        -------
        None
        """
        if memo is None: memo = set()
        elif id(self) in memo: return
        memo.add(id(self))

        #must set the gpindices of self.factorgates based on new
        my_old_gpindices = self.gpindices
        for gate in self.factorgates:
            if id(gate) in memo: continue #already processed
            rel_gate_gpindices = _gatesetmember._decompose_gpindices(
                my_old_gpindices, gate.gpindices)
            new_gate_gpindices = _gatesetmember._compose_gpindices(
                gpindices, rel_gate_gpindices)
            gate.set_gpindices(new_gate_gpindices, parent, memo)
            
        _gatesetmember.GateSetMember.set_gpindices(self, gpindices, parent)


    def acton(self, state):
        """ Act this gate map on an input state """
        for gate in self.factorgates:
            state = gate.acton(state)
        return state


    def as_sparse_matrix(self):
        """ Return the gate as a sparse matrix """
        mx = self.factorgates[0].as_sparse_matrix()
        for gate in self.factorgates[1:]:
            mx = gate.as_sparse_matrix().dot(mx)
        return mx

    
    #def as_matrix(self):
    #    """ Return the gate as a sparse matrix """
    #    return self.as_sparse_matrix().toarray()

    
    def num_params(self):
        """
        Get the number of independent parameters which specify this gate.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return len(self.gpindices_as_array())


    def to_vector(self):
        """
        Get the gate parameters as an array of values.

        Returns
        -------
        numpy array
            The gate parameters as a 1D array with length num_params().
        """
        v = _np.empty(self.num_params(), 'd')
        for gate in self.factorgates:
            factorgate_local_inds = _gatesetmember._decompose_gpindices(
                    self.gpindices, gate.gpindices)
            v[factorgate_local_inds] = gate.to_vector()
        return v


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
        for gate in self.factorgates:
            factorgate_local_inds = _gatesetmember._decompose_gpindices(
                    self.gpindices, gate.gpindices)
            gate.from_vector( v[factorgate_local_inds] )
        self.dirty = True


    def copy(self, parent=None):
        """
        Copy this gate.

        Returns
        -------
        Gate
            A copy of this gate.
        """
        return self._copy_gpindices( ComposedGateMap([gate.copy() for gate in self.factorgates]), parent )


    def transform(self, S):
        """
        Update gate matrix G with inv(S) * G * S,

        Generally, the transform function updates the *parameters* of 
        the gate such that the resulting gate matrix is altered as 
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        In this particular case any TP gauge transformation is possible,
        i.e. when `S` is an instance of `TPGaugeGroupElement` or 
        corresponds to a TP-like transform matrix.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.
        """
        for gate in self.factorgates:
            gate.transform(S)

            
    def depolarize(self, amount):
        """
        Depolarize this gate by the given `amount`.

        Generally, the depolarize function updates the *parameters* of 
        the gate such that the resulting gate matrix is depolarized.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. In standard
            bases, depolarization corresponds to multiplying the gate matrix
            by a diagonal matrix whose first diagonal element (corresponding
            to the identity) equals 1.0 and whose subsequent elements 
            (corresponding to non-identity basis elements) equal
            `1.0 - amount[i]` (or just `1.0 - amount` if `amount` is a
            float).

        Returns
        -------
        None
        """
        raise ValueError("Can't depolarize a ComposedGateMap")


    def rotate(self, amount, mxBasis="gm"):
        """
        Rotate this gate by the given `amount`.

        Generally, the rotate function updates the *parameters* of 
        the gate such that the resulting gate matrix is rotated.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : tuple of floats, optional
            Specifies the rotation "coefficients" along each of the non-identity
            Pauli-product axes.  The gate's matrix `G` is composed with a
            rotation operation `R`  (so `G` -> `dot(R, G)` ) where `R` is the
            unitary superoperator corresponding to the unitary operator 
            `U = exp( sum_k( i * rotate[k] / 2.0 * Pauli_k ) )`.  Here `Pauli_k`
            ranges over all of the non-identity un-normalized Pauli operators.

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        Returns
        -------
        None
        """
        raise ValueError("Can't rotate a ComposedGateMap")


    def compose(self, otherGate):
        """
        Create and return a new gate that is the composition of this gate
        followed by otherGate, which *must be another EmbeddedGate*.
        (For more general compositions between different types of gates, use
        the module-level compose function.)  The returned gate's matrix is
        equal to dot(this, otherGate).

        Parameters
        ----------
        otherGate : EmbeddedGate
            The gate to compose to the right of this one.

        Returns
        -------
        EmbeddedGate
        """
        raise NotImplementedError("TODO Later")


    def __str__(self):
        """ Return string representation """
        s = "Composed gate map:\n"
        s += str(self.as_sparse_matrix())
        return s



class EmbeddedGateMap(GateMap):
    """
    A gate map containing a single lower (or equal) dimensional gate within it.
    An EmbeddedGateMap acts as the identity on all of its domain except the 
    subspace of its contained gate, where it acts as the contained gate does.
    """
    
    def __init__(self, stateSpaceLabels, targetLabels, gate_to_embed, basis):
        """
        Initialize an EmbeddedGateMap object.

        Parameters
        ----------
        stateSpaceLabels : a list of tuples
            This argument specifies the density matrix space upon which this
            gate acts.  Each tuple corresponds to a block of a density matrix
            in the standard basis (and therefore a component of the direct-sum
            density matrix space). Elements of a tuple are user-defined labels
            beginning with "L" (single Level) or "Q" (two-level; Qubit) which
            interpret the d-dimensional state space corresponding to a d x d
            block as a tensor product between qubit and single level systems.
            (E.g. a 2-qubit space might be labelled `[('Q0','Q1')]`).

        targetLabels : list of strs
            The labels contained in `stateSpaceLabels` which demarcate the
            portions of the state space acted on by `gate_to_embed` (the
            "contained" gate).

        gate_to_embed : Gate
            The gate object that is to be contained within this gate, and
            that specifies the only non-trivial action of the EmbeddedGateMap.

        basis : Basis
            Specifies the basis to be used for the *entire* density-matrix
            space described by `stateSpaceLabels`.  Thus, this basis must
            have the same dimension and direct-sum structure given by
            the `stateSpaceLabels`.
        """
        dmDim, gateDim, blockDims = basis.dim #cast to basis first?
        self.stateSpaceLabels = stateSpaceLabels
        self.targetLabels = targetLabels
        self.embedded_gate = gate_to_embed
        self.basis = basis

        labels = targetLabels
        gatemx = gate_to_embed

        dmDim, gateDim, blockDims = basis.dim
        #fullOpDim = dmDim**2
        #Store each tensor product blocks start index (within the density matrix), which tensor product block
        #  each label is in, and check to make sure dimensions match stateSpaceDims
        tensorBlkIndices = {}; startIndex = []; M = 0
        assert( len(blockDims) == len(stateSpaceLabels) )
        for k, blockDim in enumerate(blockDims):
            startIndex.append(M); M += blockDim
    
            #Make sure tensor-product interpretation agrees with given dimension
            tensorBlkDim = 1 #dimension of this coherent block of the *density matrix*
            for s in stateSpaceLabels[k]:
                tensorBlkIndices[s] = k
                if s.startswith('Q'): tensorBlkDim *= 2
                elif s.startswith('L'): tensorBlkDim *= 1
                else: raise ValueError("Invalid state space specifier: %s" % s)
            if tensorBlkDim != blockDim:
                raise ValueError("State labels %s for tensor product block %d have dimension %d != given dimension %d" \
                                     % (stateSpaceLabels[k], k, tensorBlkDim, blockDim))
    
            
        iTensorProdBlks = [ tensorBlkIndices[label] for label in labels ] # index of tensor product block (of state space) a bit label is part of
        if len(set(iTensorProdBlks)) != 1:
            raise ValueError("All qubit labels of a multi-qubit gate must correspond to the" + \
                             " same tensor-product-block of the state space -- checked previously") # pragma: no cover
    
        iTensorProdBlk = iTensorProdBlks[0] #because they're all the same (tested above)
        tensorProdBlkLabels = stateSpaceLabels[iTensorProdBlk]
        basisInds = [] # list of possible *density-matrix-space* indices of each component of the tensor product block
        for l in tensorProdBlkLabels:
            assert(l[0] in ('L','Q')) #should have already been checked
            if l.startswith('L'): basisInds.append([0]) # I
            elif l.startswith('Q'): basisInds.append([0,1,2,3])  # I, X, Y, Z

        # Separate the components of the tensor product that are not operated on, i.e. that our final map just acts as identity w.r.t.
        basisInds_noop = basisInds[:]
        basisInds_noop_blankaction = basisInds[:]
        labelIndices = [ tensorProdBlkLabels.index(label) for label in labels ]
        for labelIndex in sorted(labelIndices,reverse=True):
            del basisInds_noop[labelIndex]
            basisInds_noop_blankaction[labelIndex] = [0]
                    
        #offset into "active" tensor product block
        self.offset = sum( [ blockDims[i]**2 for i in range(0,iTensorProdBlk) ] ) #number of basis elements preceding our block's elements

        divisor = 1
        self.divisors = []
        for l in labels:
            self.divisors.append(divisor)
            if l.startswith('Q'): divisor *= 4
            elif l.startswith('L'): divisor *= 1

        # multipliers to go from per-label indices to tensor-product-block index
        # e.g. if map(len,basisInds) == [1,4,4] then multipliers == [ 16 4 1 ]
        self.multipliers = _np.array( _np.flipud( _np.cumprod([1] + list(
            reversed(list(map(len,basisInds[:-1]))))) ), 'i')

        self.sorted_bili = sorted( list(enumerate(labelIndices)), key=lambda x: x[1])
          # for inserting target-qubit basis indices into list of noop-qubit indices

        self.iTensorProdBlk = iTensorProdBlk #save which block is "active" one

        #Members for speeding up time-critical functions (e.g. acton)
        self.nBasisBlocks = len(blockDims) #store this separately for faster
                                           #shortcuts in common 1-block case
        #tensorBlkEls_noop = list(_itertools.product(*basisInds_noop)) #dm-space basis for noop-indices only (unneeded)
        self.basisInds_noop = basisInds_noop
        self.basisInds_noop_blankaction = basisInds_noop_blankaction
        self.actionInds = _np.array(labelIndices,'i')
        self.basisInds_action = [ basisInds[i] for i in labelIndices ]
        self.numBasisEls_noop_blankaction = _np.array([len(inds) for inds in self.basisInds_noop_blankaction], 'i')
        self.numBasisEls_action = _np.array([len(inds) for inds in self.basisInds_action], 'i')

          # self.noop_incrementers[i] specifies how much the overall vector index
          #  is incremented when the i-th "component" digit is advanced
        nParts = len(basisInds) # number of components in "active" tensor block
        self.noop_incrementers = _np.empty(nParts,'i'); dec = 0
        for i in range(nParts-1,-1,-1):
            self.noop_incrementers[i] = self.multipliers[i] - dec
            dec += (self.numBasisEls_noop_blankaction[i]-1)*self.multipliers[i]

          # self.baseinds specifies the contribution from the "active
          #  component" digits to the overall vector index.
        inds = []
        for gate_b in _itertools.product(*self.basisInds_action):
            vec_index = self.offset
            for i,bInd in zip(self.actionInds,gate_b):
                #b[i] = bInd #don't need to do this; just update vec_index:
                vec_index += self.multipliers[i]*bInd
            inds.append(vec_index)
        self.baseinds = _np.array(inds,'i')


        #DEBUG TO REMOVE
        #print("self.numBasisEls_noop_blankaction = ",self.numBasisEls_noop_blankaction)
        #print("numBasisEls_action = ",self.numBasisEls_action)
        #print("multipliers = ",self.multipliers)
        #print("noop incrementers = ",self.noop_incrementers," dim=",gateDim)
        #print("baseinds = ",self.baseinds)
        self._set_acton()
            
        GateMap.__init__(self, gateDim)

    def _set_acton(self):
        """ Set the self.acton() method """
        #Set self.acton appropriately:
        if _fastcalc is None:
            self.acton = self._slow_acton
        elif isinstance(self.embedded_gate, LindbladParameterizedGateMap) and \
             self.embedded_gate.sparse and self.embedded_gate.unitary_postfactor is None:
            self.acton = self._fast_acton_sp1 # "special case #1"
        elif isinstance(self.embedded_gate, GateMatrix):
            self.acton = self._fast_acton_sp2 # "special case #2"
        else:
            #DEBUG TO REMOVE
            #if not isinstance(self.embedded_gate, LindbladParameterizedGateMap):
            #    print("Can't use special case b/c embedded gate = ",type(self.embedded_gate))
            #elif not self.embedded_gate.sparse:
            #    print("Can't use special case b/c embedded Lind. gate is not sparse")
            #elif self.embedded_gate.unitary_postfactor is not None:
            #    print("Can't use special case b/c embedded Lind. gate has postfactor")
            #else:
            #    print("Can't use special case for some other reason (????)")
            self.acton = self._fast_acton

    def __getstate__(self):
        # Don't pickle 'instancemethod' or parent (see gatesetmember implementation)
        d = _gatesetmember.GateSetMember.__getstate__(self)
        del d['acton']
        return d
    
    def __setstate__(self, d):
        self.__dict__.update(d)
        self._set_acton()
        

    def allocate_gpindices(self, startingIndex, parent):
        """
        Sets gpindices array for this object or any objects it
        contains (i.e. depends upon).  Indices may be obtained
        from contained objects which have already been initialized
        (e.g. if a contained object is shared with other
         top-level objects), or given new indices starting with
        `startingIndex`.

        Parameters
        ----------
        startingIndex : int
            The starting index for un-allocated parameters.

        parent : GateSet or GateSetMember
            The parent whose parameter array gpindices references.

        Returns
        -------
        num_new: int
            The number of *new* allocated parameters (so 
            the parent should mark as allocated parameter
            indices `startingIndex` to `startingIndex + new_new`).
        """
        num_new_params = self.embedded_gate.allocate_gpindices(startingIndex, parent)
        _gatesetmember.GateSetMember.set_gpindices(
            self, self.embedded_gate.gpindices, parent)
        return num_new_params

    
    def set_gpindices(self, gpindices, parent, memo=None):
        """
        Set the parent and indices into the parent's parameter vector that
        are used by this GateSetMember object.

        Parameters
        ----------
        gpindices : slice or integer ndarray
            The indices of this objects parameters in its parent's array.

        parent : GateSet or GateSetMember
            The parent whose parameter array gpindices references.

        Returns
        -------
        None
        """
        if memo is None: memo = set()
        elif id(self) in memo: return
        memo.add(id(self))

        #must set the gpindices of self.embedded_gate
        self.embedded_gate.set_gpindices(gpindices, parent, memo)
        _gatesetmember.GateSetMember.set_gpindices(
            self, gpindices, parent) #could have used self.embedded_gate.gpindices (same)
        
        
        
    def _decomp_gate_index(self, indx):
        """ Decompose index of a Pauli-product matrix into indices of each
            Pauli in the product """
        ret = []
        for d in reversed(self.divisors):
            ret.append( indx // d )
            indx = indx % d
        return ret

    
    def _merge_gate_and_noop_bases(self, gate_b, noop_b):
        """
        Merge the Pauli basis indices for the "gate"-parts of the total
        basis contained in gate_b (i.e. of the components of the tensor
        product space that are operated on) and the "noop"-parts contained
        in noop_b.  Thus, len(gate_b) + len(noop_b) == len(basisInds), and
        this function merges together basis indices for the operated-on and
        not-operated-on tensor product components.
        Note: return value always have length == len(basisInds) == number
        of components
        """
        ret = list(noop_b[:])    #start with noop part...
        for bi,li in self.sorted_bili:
            ret.insert(li, gate_b[bi]) #... and insert gate parts at proper points
        return ret

    
    def _iter_matrix_elements(self, relToBlock=False):
        """ Iterates of (gate_i,gate_j,embedded_gate_i,embedded_gate_j) tuples giving mapping
            between nonzero elements of gate matrix and elements of the embedded gate matrx """
        offset = 0 if relToBlock else self.offset
        for gate_i in range(self.embedded_gate.dim):     # rows ~ "output" of the gate map
            for gate_j in range(self.embedded_gate.dim): # cols ~ "input"  of the gate map
                gate_b1 = self._decomp_gate_index(gate_i) # gate_b? are lists of dm basis indices, one index per
                gate_b2 = self._decomp_gate_index(gate_j) #  tensor product component that the gate operates on (2 components for a 2-qubit gate)
    
                for b_noop in _itertools.product(*self.basisInds_noop): #loop over all state configurations we don't operate on
                                                                   # - so really a loop over diagonal dm elements
                    b_out = self._merge_gate_and_noop_bases(gate_b1, b_noop)  # using same b_noop for in and out says we're acting
                    b_in  = self._merge_gate_and_noop_bases(gate_b2, b_noop)  #  as the identity on the no-op state space
                    out_vec_index = _np.dot(self.multipliers, tuple(b_out)) # index of output dm basis el within vec(tensor block basis)
                    in_vec_index  = _np.dot(self.multipliers, tuple(b_in))  # index of input dm basis el within vec(tensor block basis)

                    yield (out_vec_index+offset, in_vec_index+offset, gate_i, gate_j)


    def acton(self, state):
        """ Act this gate map on an input state """
        raise NotImplementedError(
            ("EmbeddedGateMap not intialized properly, as this acton(...) "
             "method *should* have been replaced by an appropriate "
             "implementation based on Cython availability")) # pragma: no cover

    
    def _slow_acton(self, state):
        """ Act this gate map on an input state """
        output_state = _np.zeros( state.shape, 'd')
        offset = 0 #if relToBlock else self.offset (relToBlock == False here)
        
        #LATER: Maybe a faster way for dense mxs based on this, which is very slow:
        #if isinstance(self.embedded_gate, GateMatrix):
        #    embedded_base = self.embedded_gate.base
        #
        #    #SLOWEST
        #    #act on active block as: w[i] = sum_j offset_gateBlk[i,j] * v[j]
        #    for i,j,gi,gj in self._iter_matrix_elements():
        #        output_state[i] += embedded_base[gi,gj] * state[j]
        #
        #else:

        # SLOW (but in pure python)
        for b in _itertools.product(*self.basisInds_noop_blankaction): #zeros in all action-index locations
            vec_index_noop = _np.dot(self.multipliers, tuple(b))
            inds = []
            #for gate_i in range(self.embedded_gate.dim):
            #    gate_b = self._decomp_gate_index(gate_i) # gate_b? are lists of dm basis indices, one index per
            for gate_b in _itertools.product(*self.basisInds_action):
                vec_index = vec_index_noop
                for i,bInd in zip(self.actionInds,gate_b):
                    #b[i] = bInd #don't need to do this; just update vec_index:
                    vec_index += self.multipliers[i]*bInd
                inds.append(offset + vec_index)
            output_state[ inds ] += self.embedded_gate.acton( state[inds] )

        #act on other blocks trivially:
        self._acton_other_blocks_trivially(output_state,state)

        assert(output_state.shape == state.shape)
        return output_state

    
    def _acton_other_blocks_trivially(self, output_state,state):
        offset = 0
        _, _, blockDims = self.basis.dim # dmDim, gateDim, blockDims
        for i,blockDim in enumerate(blockDims):
            blockSize = blockDim**2
            if i != self.iTensorProdBlk:
                output_state[offset:offset+blockSize] = state[offset:offset+blockSize] #identity op
            offset += blockSize

            
    def _fast_acton_sp1(self, state):
        """ Act this gate map on an input state """
        output_state = _np.zeros( state.shape, 'd')
                
        #FAST Special Case #1 - a sparse LindbladParameterizedGateMap with no unitary postfactor
        A, mu, m_star, s, eta = self.embedded_gate.err_gen_prep
        #More of a performance hit than a gain:
        #  if A.nnz == 0: return state # embedded gate is identity, so entire gate is
        _fastcalc.embedded_fast_acton_sparse_spc1(A.data, A.indptr, A.indices,
                                                  mu, m_star, s, _mt.EXPM_DEFAULT_TOL, eta,
                                                  output_state[:,0], state[:,0],
                                                  self.noop_incrementers,
                                                  self.numBasisEls_noop_blankaction,
                                                  self.baseinds)

        #act on other blocks trivially:
        if self.nBasisBlocks > 1:
            self._acton_other_blocks_trivially(output_state,state)
        return output_state

    
    def _fast_acton_sp2(self, state):
        """ Act this gate map on an input state """
        output_state = _np.zeros( state.shape, 'd')
                
        #FAST Special Case #2 - a GateMatrix-derived gate, so one with a dense representation
        _fastcalc.embedded_fast_acton_sparse_spc2(self.embedded_gate.base,
                                                  output_state[:,0], state[:,0],
                                                  self.noop_incrementers,
                                                  self.numBasisEls_noop_blankaction,
                                                  self.baseinds)

        #act on other blocks trivially:
        if self.nBasisBlocks > 1:
            self._acton_other_blocks_trivially(output_state,state)
        return output_state


    def _fast_acton(self, state):
        """ Act this gate map on an input state """
        output_state = _np.zeros( state.shape, 'd')
        _fastcalc.embedded_fast_acton_sparse(self.embedded_gate.acton,
                                             output_state[:,0], state[:,0],
                                             self.noop_incrementers,
                                             self.numBasisEls_noop_blankaction,
                                             self.baseinds)
                                 
        #act on other blocks trivially:
        if self.nBasisBlocks > 1:
            self._acton_other_blocks_trivially(output_state,state)
        return output_state


    def as_sparse_matrix(self):
        """ Return the gate as a sparse matrix """
        dmDim, gateDim, blockDims = self.basis.dim
        finalGate = _sps.identity( gateDim, 'd', format='lil' )

        #fill in embedded_gate contributions (always overwrites the diagonal
        # of finalGate where appropriate, so OK it starts as identity)
        embedded_sparse = self.embedded_gate.as_sparse_matrix().tolil()
        for i,j,gi,gj in self._iter_matrix_elements():
            finalGate[i,j] = embedded_sparse[gi,gj]

        return finalGate.tocsr()

    
    def as_matrix(self):
        """ Return the gate as a sparse matrix """
        dmDim, gateDim, blockDims = self.basis.dim
        finalGate = _np.identity( gateDim, 'd' ) # operates on entire state space (direct sum of tensor prod. blocks)

        #fill in embedded_gate contributions (always overwrites the diagonal
        # of finalGate where appropriate, so OK it starts as identity)
        for i,j,gi,gj in self._iter_matrix_elements():
            finalGate[i,j] = self.embedded_gate[gi,gj]

        return finalGate

    
    def num_params(self):
        """
        Get the number of independent parameters which specify this gate.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.embedded_gate.num_params()


    def to_vector(self):
        """
        Get the gate parameters as an array of values.

        Returns
        -------
        numpy array
            The gate parameters as a 1D array with length num_params().
        """
        return self.embedded_gate.to_vector()


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
        assert(len(v) == self.num_params())
        self.embedded_gate.from_vector(v)
        self.dirty = True



    def copy(self, parent=None):
        """
        Copy this gate.

        Returns
        -------
        Gate
            A copy of this gate.
        """
        return self._copy_gpindices( EmbeddedGateMap(self.stateSpaceLabels,
                                                     self.targetLabels,
                                                     self.embedded_gate, self.basis), parent )



    def transform(self, S):
        """
        Update gate matrix G with inv(S) * G * S,

        Generally, the transform function updates the *parameters* of 
        the gate such that the resulting gate matrix is altered as 
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        In this particular case any TP gauge transformation is possible,
        i.e. when `S` is an instance of `TPGaugeGroupElement` or 
        corresponds to a TP-like transform matrix.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.
        """
        raise ValueError("Cannot transform an EmbeddedGate!")

    def depolarize(self, amount):
        """
        Depolarize this gate by the given `amount`.

        Generally, the depolarize function updates the *parameters* of 
        the gate such that the resulting gate matrix is depolarized.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. In standard
            bases, depolarization corresponds to multiplying the gate matrix
            by a diagonal matrix whose first diagonal element (corresponding
            to the identity) equals 1.0 and whose subsequent elements 
            (corresponding to non-identity basis elements) equal
            `1.0 - amount[i]` (or just `1.0 - amount` if `amount` is a
            float).

        Returns
        -------
        None
        """
        self.embedded_gate.depolarize(amount)


    def rotate(self, amount, mxBasis="gm"):
        """
        Rotate this gate by the given `amount`.

        Generally, the rotate function updates the *parameters* of 
        the gate such that the resulting gate matrix is rotated.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : tuple of floats, optional
            Specifies the rotation "coefficients" along each of the non-identity
            Pauli-product axes.  The gate's matrix `G` is composed with a
            rotation operation `R`  (so `G` -> `dot(R, G)` ) where `R` is the
            unitary superoperator corresponding to the unitary operator 
            `U = exp( sum_k( i * rotate[k] / 2.0 * Pauli_k ) )`.  Here `Pauli_k`
            ranges over all of the non-identity un-normalized Pauli operators.

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        Returns
        -------
        None
        """
        self.embedded_gate.rotate(amount, mxBasis)


    def compose(self, otherGate):
        """
        Create and return a new gate that is the composition of this gate
        followed by otherGate, which *must be another EmbeddedGate*.
        (For more general compositions between different types of gates, use
        the module-level compose function.)  The returned gate's matrix is
        equal to dot(this, otherGate).

        Parameters
        ----------
        otherGate : EmbeddedGate
            The gate to compose to the right of this one.

        Returns
        -------
        EmbeddedGate
        """
        raise NotImplementedError("TODO Later")


    def __str__(self):
        """ Return string representation """
        _, gateDim, _ = self.basis.dim
        s = "Embedded gate map with dimension %d, embedding:\n" % gateDim
        s += str(self.embedded_gate)
        return s

    

class EmbeddedGate(GateMatrix):
    """
    A gate containing a single lower (or equal) dimensional gate within it.
    An EmbeddedGate acts as the identity on all of its domain except the 
    subspace of its contained gate, where it acts as the contained gate does.
    """
    def __init__(self, stateSpaceLabels, targetLabels, gate_to_embed, basis):
        """
        Initialize a EmbeddedGate object.

        Parameters
        ----------
        stateSpaceLabels : a list of tuples
            This argument specifies the density matrix space upon which this
            gate acts.  Each tuple corresponds to a block of a density matrix
            in the standard basis (and therefore a component of the direct-sum
            density matrix space). Elements of a tuple are user-defined labels
            beginning with "L" (single Level) or "Q" (two-level; Qubit) which
            interpret the d-dimensional state space corresponding to a d x d
            block as a tensor product between qubit and single level systems.
            (E.g. a 2-qubit space might be labelled `[('Q0','Q1')]`).

        targetLabels : list of strs
            The labels contained in `stateSpaceLabels` which demarcate the
            portions of the state space acted on by `gate_to_embed` (the
            "contained" gate).

        gate_to_embed : GateMatrix
            The gate object that is to be contained within this gate, and
            that specifies the only non-trivial action of the EmbeddedGate.

        basis : Basis
            Specifies the basis to be used for the *entire* density-matrix
            space described by `stateSpaceLabels`.  Thus, this basis must
            have the same dimension and direct-sum structure given by
            the `stateSpaceLabels`.
        """
        _,gateDim,_ = basis.dim #cast to basis first?
        self.embedded_map = EmbeddedGateMap(stateSpaceLabels, targetLabels, gate_to_embed, basis)
        GateMatrix.__init__(self, _np.identity(gateDim,'d'))
        self._construct_matrix()

    def _construct_matrix(self):
        self.base = self.embedded_map.as_matrix()
        assert(self.base.shape == (self.dim,self.dim))
        self.base.flags.writeable = False

        
    def allocate_gpindices(self, startingIndex, parent):
        """
        Sets gpindices array for this object or any objects it
        contains (i.e. depends upon).  Indices may be obtained
        from contained objects which have already been initialized
        (e.g. if a contained object is shared with other
         top-level objects), or given new indices starting with
        `startingIndex`.

        Parameters
        ----------
        startingIndex : int
            The starting index for un-allocated parameters.

        parent : GateSet or GateSetMember
            The parent whose parameter array gpindices references.

        Returns
        -------
        num_new: int
            The number of *new* allocated parameters (so 
            the parent should mark as allocated parameter
            indices `startingIndex` to `startingIndex + new_new`).
        """
        #print(">>> DB EmbeddedGate %d allocating: " % id(self), id(self.embedded_map.embedded_gate))
        num_new_params = self.embedded_map.allocate_gpindices(startingIndex, parent)
        _gatesetmember.GateSetMember.set_gpindices(
            self, self.embedded_map.gpindices, parent)
        return num_new_params

    
    def set_gpindices(self, gpindices, parent, memo=None):
        """
        Set the parent and indices into the parent's parameter vector that
        are used by this GateSetMember object.

        Parameters
        ----------
        gpindices : slice or integer ndarray
            The indices of this objects parameters in its parent's array.

        parent : GateSet or GateSetMember
            The parent whose parameter array gpindices references.

        Returns
        -------
        None
        """
        if memo is None: memo = set()
        elif id(self) in memo: return
        memo.add(id(self))

        #must set the gpindices of self.embedded_gate
        self.embedded_map.set_gpindices(gpindices, parent, memo)
        _gatesetmember.GateSetMember.set_gpindices(
            self, gpindices, parent) #could have used self.embedded_gate.gpindices (same)


    def num_params(self):
        """
        Get the number of independent parameters which specify this gate.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.embedded_map.num_params()


    def to_vector(self):
        """
        Get the gate parameters as an array of values.

        Returns
        -------
        numpy array
            The gate parameters as a 1D array with length num_params().
        """
        return self.embedded_map.to_vector()


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
        assert(len(v) == self.num_params())
        self.embedded_map.from_vector(v)
        self._construct_matrix()
        assert(self.base.shape == (self.dim, self.dim))
        self.dirty = True


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
        # Note: this function exploits knowledge of EmbeddedGateMap internals!!
        dmDim, gateDim, blockDims = self.embedded_map.basis.dim

        derivMx = _np.zeros((self.dim**2,self.num_params()),'d')
        embedded_deriv = self.embedded_map.embedded_gate.deriv_wrt_params(wrtFilter)
        M = self.embedded_map.embedded_gate.dim

        #fill in embedded_gate contributions (always overwrites the diagonal
        # of finalGate where appropriate, so OK it starts as identity)
        for i,j,gi,gj in self.embedded_map._iter_matrix_elements():
            derivMx[i*self.dim+j,:] = embedded_deriv[gi*M+gj,:] #fill row of jacobian

        if wrtFilter is None:
            return derivMx
        else:
            return _np.take( derivMx, wrtFilter, axis=1 )

        
    def has_nonzero_hessian(self):
        """ 
        Returns whether this gate has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return self.embedded_map.embedded_gate.has_nonzero_hessian()


    def copy(self, parent=None):
        """
        Copy this gate.

        Returns
        -------
        Gate
            A copy of this gate.
        """
        return self._copy_gpindices( EmbeddedGate(
            self.embedded_map.stateSpaceLabels,
            self.embedded_map.targetLabels,
            self.embedded_map.embedded_gate, self.embedded_map.basis), parent )
    

    def transform(self, S):
        """
        Update gate matrix G with inv(S) * G * S,

        Generally, the transform function updates the *parameters* of 
        the gate such that the resulting gate matrix is altered as 
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        In this particular case any TP gauge transformation is possible,
        i.e. when `S` is an instance of `TPGaugeGroupElement` or 
        corresponds to a TP-like transform matrix.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.
        """
        raise ValueError("Cannot transform an EmbeddedGate!")

    def depolarize(self, amount):
        """
        Depolarize this gate by the given `amount`.

        Generally, the depolarize function updates the *parameters* of 
        the gate such that the resulting gate matrix is depolarized.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. In standard
            bases, depolarization corresponds to multiplying the gate matrix
            by a diagonal matrix whose first diagonal element (corresponding
            to the identity) equals 1.0 and whose subsequent elements 
            (corresponding to non-identity basis elements) equal
            `1.0 - amount[i]` (or just `1.0 - amount` if `amount` is a
            float).

        Returns
        -------
        None
        """
        self.embedded_map.depolarize(amount)
        self._construct_matrix()


    def rotate(self, amount, mxBasis="gm"):
        """
        Rotate this gate by the given `amount`.

        Generally, the rotate function updates the *parameters* of 
        the gate such that the resulting gate matrix is rotated.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : tuple of floats, optional
            Specifies the rotation "coefficients" along each of the non-identity
            Pauli-product axes.  The gate's matrix `G` is composed with a
            rotation operation `R`  (so `G` -> `dot(R, G)` ) where `R` is the
            unitary superoperator corresponding to the unitary operator 
            `U = exp( sum_k( i * rotate[k] / 2.0 * Pauli_k ) )`.  Here `Pauli_k`
            ranges over all of the non-identity un-normalized Pauli operators.

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        Returns
        -------
        None
        """
        self.embedded_map.rotate(amount, mxBasis)
        self._construct_matrix()


    def compose(self, otherGate):
        """
        Create and return a new gate that is the composition of this gate
        followed by otherGate, which *must be another EmbeddedGate*.
        (For more general compositions between different types of gates, use
        the module-level compose function.)  The returned gate's matrix is
        equal to dot(this, otherGate).

        Parameters
        ----------
        otherGate : EmbeddedGate
            The gate to compose to the right of this one.

        Returns
        -------
        EmbeddedGate
        """
        raise NotImplementedError("TODO Later")


    def __str__(self):
        """ Return string representation """
        s = "Embedded gate with shape %s\n" % str(self.base.shape)
        s += _mt.mx_to_string(self.base, width=4, prec=2)
        return s


