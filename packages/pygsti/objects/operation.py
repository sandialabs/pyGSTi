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
import collections as _collections
import numbers as _numbers

from ..      import optimize as _opt
from ..tools import matrixtools as _mt
from ..tools import optools as _gt
from ..tools import jamiolkowski as _jt
from ..tools import basistools as _bt
from ..tools import listtools as _lt
from ..tools import slicetools as _slct
from ..tools import compattools as _compat
from ..tools import symplectic as _symp
from . import gaugegroup as _gaugegroup
from . import modelmember as _modelmember
from ..baseobjs import ProtectedArray as _ProtectedArray
from ..baseobjs import Basis as _Basis
from ..baseobjs import BuiltinBasis as _BuiltinBasis
from ..baseobjs.basis import basis_matrices as _basis_matrices

from . import term as _term
from .polynomial import Polynomial as _Polynomial
from . import replib

TOL = 1e-12
IMAG_TOL = 1e-7 #tolerance for imaginary part being considered zero


def optimize_operation(opToOptimize, targetOp):
    """
    Optimize the parameters of opToOptimize so that the
      the resulting operation matrix is as close as possible to
      targetOp's matrix.

    This is trivial for the case of FullDenseOp
      instances, but for other types of parameterization
      this involves an iterative optimization over all the
      parameters of opToOptimize.

    Parameters
    ----------
    opToOptimize : LinearOperator
      The gate to optimize.  This object gets altered.

    targetOp : LinearOperator
      The gate whose matrix is used as the target.

    Returns
    -------
    None
    """

    #TODO: cleanup this code:
    if isinstance(opToOptimize, StaticDenseOp):
        return #nothing to optimize

    if isinstance(opToOptimize, FullDenseOp):
        if(targetOp.dim != opToOptimize.dim): #special case: gates can have different overall dimension
            opToOptimize.dim = targetOp.dim   #  this is a HACK to allow model selection code to work correctly
        opToOptimize.set_value(targetOp)     #just copy entire overall matrix since fully parameterized
        return

    assert(targetOp.dim == opToOptimize.dim) #gates must have the same overall dimension
    targetMatrix = _np.asarray(targetOp)
    def _objective_func(param_vec):
        opToOptimize.from_vector(param_vec)
        return _mt.frobeniusnorm(opToOptimize - targetMatrix)

    x0 = opToOptimize.to_vector()
    minSol = _opt.minimize(_objective_func, x0, method='BFGS', maxiter=10000, maxfev=10000,
                           tol=1e-6, callback=None)

    opToOptimize.from_vector(minSol.x)
    #print("DEBUG: optimized gate to min frobenius distance %g" %
    #      _mt.frobeniusnorm(opToOptimize-targetMatrix))


def compose(op1, op2, basis, parameterization="auto"):
    """
    Returns a new LinearOperator that is the composition of op1 and op2.

    The resulting gate's matrix == dot(op1, op2),
     (so op1 acts *second* on an input) and the type of LinearOperator instance
     returned will depend on how much of the parameterization in op1
     and op2 can be preserved in the resulting gate.

    Parameters
    ----------
    op1 : LinearOperator
        LinearOperator to compose as left term of matrix product (applied second).

    op2 : LinearOperator
        LinearOperator to compose as right term of matrix product (applied first).

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    parameterization : {"auto","full","TP","linear","static"}, optional
        The parameterization of the resulting gates.  The default, "auto",
        attempts to convert to the most restrictive common parameterization.

    Returns
    -------
    LinearOperator
       The composed gate.
    """

    #Find the most restrictive common parameterization that both op1
    # and op2 can be cast/converted into. Utilized converstions are:
    #
    # Static => TP (sometimes)
    # Static => Linear
    # Static => Full
    # Linear => TP (sometimes)
    # Linear => Full
    # TP => Full

    if parameterization == "auto":
        if any([isinstance(g, FullDenseOp) for g in (op1,op2)]):
            paramType = "full"
        elif any([isinstance(g, TPDenseOp) for g in (op1,op2)]):
            paramType = "TP" #update to "full" below if TP-conversion
                             #not possible?
        elif any([isinstance(g, LinearlyParamDenseOp)
                  for g in (op1,op2)]):
            paramType = "linear"
        else:
            assert( isinstance(op1, StaticDenseOp)
                    and isinstance(op2, StaticDenseOp) )
            paramType = "static"
    else:
        paramType = parameterization #user-specified final parameterization

    #Convert to paramType as necessary
    cop1 = convert(op1, paramType, basis)
    cop2 = convert(op2, paramType, basis)

    # cop1 and cop2 are the same type, so can invoke the gate's compose method
    return cop1.compose(cop2)


def convert(gate, toType, basis, extra=None):
    """
    Convert gate to a new type of parameterization, potentially creating
    a new LinearOperator object.  Raises ValueError for invalid conversions.

    Parameters
    ----------
    gate : LinearOperator
        LinearOperator to convert

    toType : {"full","TP","static","static unitary","clifford",LINDBLAD}
        The type of parameterizaton to convert to.  "LINDBLAD" is a placeholder
        for the various Lindblad parameterization types.  See
        :method:`Model.set_all_parameterizations` for more details.

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for `gate`.  Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
        (or a custom basis object).

    extra : object, optional
        Additional information for conversion.

    Returns
    -------
    LinearOperator
       The converted gate, usually a distinct
       object from the gate object passed as input.
    """
    if toType == "full":
        if isinstance(gate, FullDenseOp):
            return gate #no conversion necessary
        else:
            ret = FullDenseOp( gate.todense() )
            return ret

    elif toType == "TP":
        if isinstance(gate, TPDenseOp):
            return gate #no conversion necessary
        else:
            return TPDenseOp( gate.todense() )
              # above will raise ValueError if conversion cannot be done

    elif toType == "linear":
        if isinstance(gate, LinearlyParamDenseOp):
            return gate #no conversion necessary
        elif isinstance(gate, StaticDenseOp):
            real = _np.isclose(_np.linalg.norm( gate.imag ),0)
            return LinearlyParamDenseOp(gate.todense(), _np.array([]), {}, real)
        else:
            raise ValueError("Cannot convert type %s to LinearlyParamDenseOp"
                             % type(gate))
        
    elif toType == "static":
        if isinstance(gate, StaticDenseOp):
            return gate #no conversion necessary
        else:
            return StaticDenseOp( gate )

    elif toType == "static unitary":
        op_std = _bt.change_basis(gate, basis, 'std')
        unitary = _gt.process_mx_to_unitary(op_std)
        return StaticDenseOp(unitary, "statevec")

    elif _gt.is_valid_lindblad_paramtype(toType):
        # e.g. "H+S terms","H+S clifford terms"

        _,evotype = _gt.split_lindblad_paramtype(toType)
        LindbladOpType = LindbladOp \
                           if evotype in ("svterm","cterm") else \
                              LindbladDenseOp

        nQubits = _np.log2(gate.dim)/2.0
        bQubits = bool(abs(nQubits-round(nQubits)) < 1e-10) #integer # of qubits?
        proj_basis = "pp" if (basis == "pp" or bQubits) else basis

        return LindbladOpType.from_operation_obj(gate, toType, None, proj_basis,
                                              basis, truncate=True, lazy=True)
    
    elif toType == "clifford":
        if isinstance(gate, CliffordOp):
            return gate #no conversion necessary
        
        # assume gate represents a unitary op (otherwise
        #  would need to change Model dim, which isn't allowed)
        return CliffordOp(gate)

    else:
        raise ValueError("Invalid toType argument: %s" % toType)


def finite_difference_deriv_wrt_params(gate, eps=1e-7):
    """
    Computes a finite-difference Jacobian for a LinearOperator object.

    The returned value is a matrix whose columns are the vectorized
    derivatives of the flattened operation matrix with respect to a single
    gate parameter, matching the format expected from the gate's
    `deriv_wrt_params` method.

    Parameters
    ----------
    gate : LinearOperator
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
    op2 = gate.copy()
    p = gate.to_vector()
    fd_deriv = _np.empty((dim,dim,gate.num_params()), gate.dtype)

    for i in range(gate.num_params()):
        p_plus_dp = p.copy()
        p_plus_dp[i] += eps
        op2.from_vector(p_plus_dp)
        fd_deriv[:,:,i] = (op2-gate)/eps

    fd_deriv.shape = [dim**2,gate.num_params()]
    return fd_deriv


def check_deriv_wrt_params(gate, deriv_to_check=None, eps=1e-7):
    """
    Checks the `deriv_wrt_params` method of a LinearOperator object.

    This routine is meant to be used as an aid in testing and debugging
    gate classes by comparing the finite-difference Jacobian that
    *should* be returned by `gate.deriv_wrt_params` with the one that
    actually is.  A ValueError is raised if the two do not match.

    Parameters
    ----------
    gate : LinearOperator
        The gate object to test.

    deriv_to_check : numpy.ndarray or None, optional
        If not None, the Jacobian to compare against the finite difference
        result.  If None, `gate.deriv_wrt_parms()` is used.  Setting this
        argument can be useful when the function is called *within* a LinearOperator
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


#Note on initialization sequence of Gates within a Model:
# 1) a Model is constructed (empty)
# 2) a LinearOperator is constructed - apart from a Model if it's locally parameterized,
#    otherwise with explicit reference to an existing Model's labels/indices.
#    All gates (ModelMember objs in general) have a "gpindices" member which
#    can either be initialized upon construction or set to None, which signals
#    that the Model must initialize it.
# 3) the LinearOperator is assigned/added to a dict within the Model.  As a part of this
#    process, the LinearOperator's 'gpindices' member is set, if it isn't already, and the
#    Model's "global" parameter vector (and number of params) is updated as
#    needed to accomodate new parameters.
#
# Note: gpindices may be None (before initialization) or any valid index
#  into a 1D numpy array (e.g. a slice or integer array).  It may NOT have
#  any repeated elements.
#
# When a LinearOperator is removed from the Model, parameters only used by it can be
# removed from the Model, and the gpindices members of existing gates
# adjusted as needed.
#
# When derivatives are taken wrt. a model parameter (1 col of a jacobian)
# derivatives wrt each gate that includes that parameter in its gpindices
# must be processed.


class LinearOperator(_modelmember.ModelMember):
    """ Base class for all gate representations """
    cache_reps = False
    
    def __init__(self, dim, evotype):
        """ Initialize a new LinearOperator """
        super(LinearOperator, self).__init__(dim, evotype)
        self._cachedrep = None

    @property
    def size(self):
        """
        Return the number of independent elements in this gate (when viewed as a dense array)
        """
        return (self.dim)**2

    def set_value(self, M):
        """
        Attempts to modify gate parameters so that the specified raw
        operation matrix becomes mx.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        M : array_like or LinearOperator
            An array of shape (dim, dim) or LinearOperator representing the gate action.

        Returns
        -------
        None
        """
        raise ValueError("Cannot set the value of a %s directly!" % self.__class__.__name__)

    def todense(self):
        """
        Return this operation as a dense matrix.
        """
        raise NotImplementedError("todense(...) not implemented for %s objects!" % self.__class__.__name__)

    def torep(self):
        """
        Return a "representation" object for this gate.

        Such objects are primarily used internally by pyGSTi to compute
        things like probabilities more efficiently.

        Returns
        -------
        OpRep
        """
        if self._evotype == "statevec":
            return replib.SVOpRep_Dense(_np.ascontiguousarray(self.todense(),complex) )
        elif self._evotype == "densitymx":
            if LinearOperator.cache_reps: # cache reps to avoid recomputation
                if self._cachedrep is None:
                    self._cachedrep = replib.DMOpRep_Dense(_np.ascontiguousarray(self.todense(),'d'))
                return self._cachedrep
            else:
                return replib.DMOpRep_Dense(_np.ascontiguousarray(self.todense(),'d'))
        else:
            raise NotImplementedError("torep(%s) not implemented for %s objects!" %
                                      (self._evotype, self.__class__.__name__))            

    @property
    def dirty(self):
        return _modelmember.ModelMember.dirty.fget(self) # call base class

    @dirty.setter
    def dirty(self, value):
        if value == True: self._cachedrep = None # clear cached rep
        _modelmember.ModelMember.dirty.fset(self, value) # call base class setter

    def __getstate__(self):
        st = super(LinearOperator, self).__getstate__()
        st['_cachedrep'] = None # can't pickle this!
        return st

    def copy(self, parent=None):
        self._cachedrep = None # deepcopy in ModelMember.copy can't copy CReps!
        return _modelmember.ModelMember.copy(self,parent)


    def tosparse(self):
        """
        Return this operation as a sparse matrix.
        """
        raise NotImplementedError("tosparse(...) not implemented for %s objects!" % self.__class__.__name__)

    
    def get_order_terms(self, order):
        """ 
        Get the `order`-th order Taylor-expansion terms of this operation.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that its action on a
        density matrix `rho` can be written:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the gate's
        parameters, where the polynomial's variable indices index the *global*
        parameters of the gate's parent (usually a :class:`Model`), not the 
        gate's local parameter array (i.e. that returned from `to_vector`).


        Parameters
        ----------
        order : int
            The order of terms to get.

        Returns
        -------
        list
            A list of :class:`RankOneTerm` objects.
        """
        raise NotImplementedError("get_order_terms(...) not implemented for %s objects!" % self.__class__.__name__)

    def frobeniusdist2(self, otherOp, transform=None, inv_transform=None):
        """ 
        Return the squared frobenius difference between this gate and
        `otherOp`, optionally transforming this gate first using matrices
        `transform` and `inv_transform`.
        """
        if transform is None and inv_transform is None:
            return _gt.frobeniusdist2(self.todense(),otherOp.todense())
        else:
            return _gt.frobeniusdist2(_np.dot(
                    inv_transform,_np.dot(self.todense(),transform)),
                                      otherOp.todense())

    def frobeniusdist(self, otherOp, transform=None, inv_transform=None):
        """ 
        Return the frobenius distance between this gate
        and `otherOp`, optionally transforming this gate first
        using `transform` and `inv_transform`.
        """
        return _np.sqrt(self.frobeniusdist2(otherOp, transform, inv_transform))


    def residuals(self, otherOp, transform=None, inv_transform=None):
        """
        The per-element difference between this `DenseOperator` and `otherOp`,
        possibly after transforming this operation as 
        `G => inv_transform * G * transform`.

        Parameters
        ----------
        otherOp : DenseOperator
            The gate to compare against.

        transform, inv_transform : numpy.ndarray, optional
            The transform and its inverse, respectively, to apply before
            taking the element-wise difference.

        Returns
        -------
        numpy.ndarray
            A 1D-array of size equal to that of the flattened operation matrix.
        """
        if transform is None and inv_transform is None:
            return _gt.residuals(self.todense(),otherOp.todense())
        else:
            return _gt.residuals(_np.dot(
                    inv_transform,_np.dot(self.todense(),transform)),
                                 otherOp.todense())

    def jtracedist(self, otherOp, transform=None, inv_transform=None):
        """ 
        Return the Jamiolkowski trace distance between this gate
        and `otherOp`, optionally transforming this gate first
        using `transform` and `inv_transform`.
        """
        if transform is None and inv_transform is None:
            return _gt.jtracedist(self.todense(),otherOp.todense())
        else:
            return _gt.jtracedist(_np.dot(
                    inv_transform,_np.dot(self.todense(),transform)),
                                  otherOp.todense())

    def diamonddist(self, otherOp, transform=None, inv_transform=None):
        """ 
        Return the diamon distance between this gate
        and `otherOp`, optionally transforming this gate first
        using `transform` and `inv_transform`.
        """
        if transform is None and inv_transform is None:
            return _gt.diamonddist(self.todense(),otherOp.todense())
        else:
            return _gt.diamonddist(_np.dot(
                    inv_transform,_np.dot(self.todense(),transform)),
                                   otherOp.todense())

    def transform(self, S):
        """
        Update operation matrix G with inv(S) * G * S,

        Generally, the transform function updates the *parameters* of 
        the gate such that the resulting operation matrix is altered as 
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        In this particular case *any* transform of the appropriate
        dimension is possible, since all operation matrix elements are parameters.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.
        """
        Smx = S.get_transform_matrix()
        Si  = S.get_transform_matrix_inverse()
        self.set_value(_np.dot(Si,_np.dot(self.todense(), Smx)))


    def depolarize(self, amount):
        """
        Depolarize this gate by the given `amount`.

        Generally, the depolarize function updates the *parameters* of 
        the gate such that the resulting operation matrix is depolarized.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. In standard
            bases, depolarization corresponds to multiplying the operation matrix
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
        self.set_value(_np.dot(D,self.todense()))


    def rotate(self, amount, mxBasis="gm"):
        """
        Rotate this gate by the given `amount`.

        Generally, the rotate function updates the *parameters* of 
        the gate such that the resulting operation matrix is rotated.  If
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
        self.set_value(_np.dot(rotnMx,self.todense()))

        
    def compose(self, otherOp):
        """
        Create and return a new gate that is the composition of this operation
        followed by otherOp of the same type.  (For more general compositions
        between different types of gates, use the module-level compose function.
        )  The returned gate's matrix is equal to dot(this, otherOp).

        Parameters
        ----------
        otherOp : DenseOperator
            The gate to compose to the right of this one.

        Returns
        -------
        DenseOperator
        """
        cpy = self.copy()
        cpy.set_value( _np.dot( self.todense(), otherOp.todense()) )
        return cpy

    
    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single gate parameter.  Thus, each column is of length
        op_dim^2 and there is one column per gate parameter. An
        empty 2D array in the StaticDenseOp case (num_params == 0).

        Returns
        -------
        numpy array
            Array of derivatives with shape (dimension^2, num_params)
        """
        if(self.num_params() != 0):
            raise NotImplementedError("Default deriv_wrt_params is only for 0-parameter (default) case (%s)"
                                      % str(self.__class__.__name__))

        dtype = complex if self._evotype == 'statevec' else 'd'
        derivMx = _np.zeros((self.size,0),dtype)
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
        #Default: assume Hessian can be nonzero if there are any parameters
        return self.num_params() > 0

    
    def hessian_wrt_params(self, wrtFilter1=None, wrtFilter2=None):
        """
        Construct the Hessian of this operation with respect to its parameters.

        This function returns a tensor whose first axis corresponds to the
        flattened operation matrix and whose 2nd and 3rd axes correspond to the
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
        if not self.has_nonzero_hessian():
            return _np.zeros((self.size, self.num_params(), self.num_params()),'d')

        # FUTURE: create a finite differencing hessian method?
        raise NotImplementedError("hessian_wrt_params(...) is not implemented for %s objects" % self.__class__.__name__)


    ##Pickle plumbing
    def __setstate__(self, state):
        self.__dict__.update(state)

    #Note: no __str__ fn

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
        if isinstance(M, LinearOperator):
            dim = M.dim
            matrix = _np.asarray(M).copy()
              # LinearOperator objs should also derive from ndarray
        elif isinstance(M, _np.ndarray):
            matrix = M.copy()
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


        
#class MapOperator(LinearOperator):
#    def __init__(self, dim, evotype):
#        """ Initialize a new LinearOperator """
#        super(MapOperator, self).__init__(dim, evotype)
#
#    #Maybe add an as_sparse_mx function and compute
#    # metrics using this?
#    #And perhaps a sparse-mode finite-difference deriv_wrt_params?

    
class DenseOperator(LinearOperator):
    """
    Excapulates a parameterization of a operation matrix.  This class is the
    common base class for all specific parameterizations of a gate.
    """

    def __init__(self, mx, evotype):
        """ Initialize a new LinearOperator """
        self.base = mx
        super(DenseOperator, self).__init__(self.base.shape[0], evotype)
        assert(evotype in ("densitymx","statevec")), \
            "Invalid evotype for a DenseOperator: %s" % evotype
        
    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single gate parameter.  Thus, each column is of length
        op_dim^2 and there is one column per gate parameter. An
        empty 2D array in the StaticDenseOp case (num_params == 0).

        Returns
        -------
        numpy array
            Array of derivatives with shape (dimension^2, num_params)
        """
        return finite_difference_deriv_wrt_params(self, eps=1e-7)

    def todense(self):
        """
        Return this operation as a dense matrix.
        """
        return _np.asarray(self.base)
          # *must* be a numpy array for Cython arg conversion

    def tosparse(self):
        """
        Return the operation as a sparse matrix.
        """
        return _sps.csr_matrix(self.todense())

    def __copy__(self):
        # We need to implement __copy__ because we defer all non-existing
        # attributes to self.base (a numpy array) which *has* a __copy__
        # implementation that we don't want to use, as it results in just a
        # copy of the numpy array.
        cls = self.__class__
        cpy = cls.__new__(cls)
        cpy.__dict__.update(self.__dict__)
        return cpy

    def __deepcopy__(self, memo):
        # We need to implement __deepcopy__ because we defer all non-existing
        # attributes to self.base (a numpy array) which *has* a __deepcopy__
        # implementation that we don't want to use, as it results in just a
        # copy of the numpy array.
        cls = self.__class__
        cpy = cls.__new__(cls)
        memo[id(self)] = cpy
        for k, v in self.__dict__.items():
            setattr(cpy, k, _copy.deepcopy(v, memo))
        return cpy
    
    def __str__(self):
        s = "%s with shape %s\n" % (self.__class__.__name__, str(self.base.shape))
        s += _mt.mx_to_string(self.base, width=4, prec=2)
        return s

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



class StaticDenseOp(DenseOperator):
    """
    Encapsulates a operation matrix that is completely fixed, or "static", meaning
      that is contains no parameters.
    """

    def __init__(self, M, evotype="auto"):
        """
        Initialize a StaticDenseOp object.

        Parameters
        ----------
        M : array_like or LinearOperator
            a square 2D array-like or LinearOperator object representing the gate action.
            The shape of M sets the dimension of the gate.
        """
        M = LinearOperator.convert_to_matrix(M)
        if evotype == "auto":
            evotype = "statevec" if _np.iscomplexobj(M) else "densitymx"
        assert(evotype in ("statevec","densitymx")), \
            "Invalid evolution type '%s' for %s" % (evotype,self.__class__.__name__)
        DenseOperator.__init__(self, M, evotype)
        #(default DenseOperator/LinearOperator methods implement an object with no parameters)

        #if self._evotype == "svterm": # then we need to extract unitary
        #    op_std = _bt.change_basis(gate, basis, 'std')
        #    U = _gt.process_mx_to_unitary(self)

    def compose(self, otherOp):
        """
        Create and return a new gate that is the composition of this operation
        followed by otherOp, which *must be another StaticDenseOp*.
        (For more general compositions between different types of gates, use
        the module-level compose function.)  The returned gate's matrix is
        equal to dot(this, otherOp).

        Parameters
        ----------
        otherOp : StaticDenseOp
            The gate to compose to the right of this one.

        Returns
        -------
        StaticDenseOp
        """
        return StaticDenseOp(_np.dot( self.base, otherOp.base), self._evotype)
        

class FullDenseOp(DenseOperator):
    """
    Encapsulates a operation matrix that is fully parameterized, that is,
      each element of the operation matrix is an independent parameter.
    """

    def __init__(self, M, evotype="auto"):
        """
        Initialize a FullDenseOp object.

        Parameters
        ----------
        M : array_like or LinearOperator
            a square 2D array-like or LinearOperator object representing the gate action.
            The shape of M sets the dimension of the gate.
        """
        M = LinearOperator.convert_to_matrix(M)
        if evotype == "auto":
            evotype = "statevec" if _np.iscomplexobj(M) else "densitymx"
        assert(evotype in ("statevec","densitymx")), \
            "Invalid evolution type '%s' for %s" % (evotype,self.__class__.__name__)
        DenseOperator.__init__(self,M,evotype)

        
    def set_value(self, M):
        """
        Attempts to modify gate parameters so that the specified raw
        operation matrix becomes mx.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        M : array_like or LinearOperator
            An array of shape (dim, dim) or LinearOperator representing the gate action.

        Returns
        -------
        None
        """
        mx = LinearOperator.convert_to_matrix(M)
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
        return 2*self.size if self._evotype == "statevec" else self.size


    def to_vector(self):
        """
        Get the gate parameters as an array of values.

        Returns
        -------
        numpy array
            The gate parameters as a 1D array with length num_params().
        """
        if self._evotype == "statevec":
            return _np.concatenate( (self.base.real.flatten(), self.base.imag.flatten()), axis=0)
        else:
            return self.base.flatten()


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
        if self._evotype == "statevec":
            self.base[:,:] = v[0:self.dim**2].reshape( (self.dim,self.dim) ) + \
                             1j*v[self.dim**2:].reshape( (self.dim,self.dim) )
        else:
            self.base[:,:] = v.reshape( (self.dim,self.dim) )
        self.dirty = True


    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single gate parameter.  Thus, each column is of length
        op_dim^2 and there is one column per gate parameter.

        Returns
        -------
        numpy array
            Array of derivatives with shape (dimension^2, num_params)
        """
        if self._evotype == "statevec":
            derivMx = _np.concatenate( (_np.identity( self.dim**2, 'complex' ),
                                        1j*_np.identity( self.dim**2,'complex')),
                                        axis=1 )
        else:
            derivMx = _np.identity( self.dim**2, self.base.dtype )
            
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


class TPDenseOp(DenseOperator):
    """
    Encapsulates a operation matrix that is fully parameterized except for
    the first row, which is frozen to be [1 0 ... 0] so that the action
    of the gate, when interpreted in the Pauli or Gell-Mann basis, is
    trace preserving (TP).
    """

    def __init__(self, M):
        """
        Initialize a TPDenseOp object.

        Parameters
        ----------
        M : array_like or LinearOperator
            a square 2D numpy array representing the gate action.  The
            shape of this array sets the dimension of the gate.
        """
        #LinearOperator.__init__(self, LinearOperator.convert_to_matrix(M))
        mx = LinearOperator.convert_to_matrix(M)
        assert(_np.isrealobj(mx)),"TPDenseOp must have *real* values!"
        if not (_np.isclose(mx[0,0], 1.0) and \
                _np.allclose(mx[0,1:], 0.0)):
            raise ValueError("Cannot create TPDenseOp: " +
                             "invalid form for 1st row!")
        DenseOperator.__init__(self, _ProtectedArray(
            mx, indicesToProtect=(0, slice(None,None,None))), "densitymx")
                


    def set_value(self, M):
        """
        Attempts to modify gate parameters so that the specified raw
        operation matrix becomes mx.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        M : array_like or LinearOperator
            An array of shape (dim, dim) or LinearOperator representing the gate action.

        Returns
        -------
        None
        """
        mx = LinearOperator.convert_to_matrix(M)
        if(mx.shape != (self.dim, self.dim)):
            raise ValueError("Argument must be a (%d,%d) matrix!"
                             % (self.dim,self.dim))
        if not (_np.isclose(mx[0,0], 1.0) and _np.allclose(mx[0,1:], 0.0)):
            raise ValueError("Cannot set TPDenseOp: " +
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
        derivatives of the flattened operation matrix with respect to a
        single gate parameter.  Thus, each column is of length
        op_dim^2 and there is one column per gate parameter.

        Returns
        -------
        numpy array
            Array of derivatives with shape (dimension^2, num_params)
        """
        derivMx = _np.identity( self.dim**2, 'd' ) # TP gates are assumed to be real
        derivMx = derivMx[:,self.dim:] #remove first op_dim cols ( <=> first-row parameters )

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

    

class LinearlyParameterizedElementTerm(object):
    """
    Encapsulates a single term within a LinearlyParamDenseOp.
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


class LinearlyParamDenseOp(DenseOperator):
    """
    Encapsulates a operation matrix that is parameterized such that each
    element of the operation matrix depends only linearly on any parameter.
    """

    def __init__(self, baseMatrix, parameterArray, parameterToBaseIndicesMap,
                 leftTransform=None, rightTransform=None, real=False, evotype="auto"):
        """
        Initialize a LinearlyParamDenseOp object.

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
            indexing potentially multiple operation matrix coordinates
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
            Whether or not the resulting operation matrix, after all
            parameter evaluation and left & right transforms have
            been performed, should be real.  If True, ValueError will
            be raised if the matrix contains any complex or imaginary
            elements.
        """

        baseMatrix = _np.array( LinearOperator.convert_to_matrix(baseMatrix), 'complex')
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
        assert(_np.isrealobj(self.parameterArray)), "Parameter array must be real-valued!"

        I = _np.identity(self.baseMatrix.shape[0],'d') # LinearlyParameterizedGates are currently assumed to be real
        self.leftTrans = leftTransform if (leftTransform is not None) else I
        self.rightTrans = rightTransform if (rightTransform is not None) else I
        self.enforceReal = real
        mx.flags.writeable = False # only _construct_matrix can change array

        if evotype == "auto": evotype = "densitymx" if real else "statevec"
        assert(evotype in ("densitymx","statevec")), \
            "Invalid evolution type '%s' for %s" % (evotype,self.__class__.__name__)

        DenseOperator.__init__(self, mx, evotype)
        self._construct_matrix() # construct base from the parameters

        
    def _construct_matrix(self):
        """
        Build the internal operation matrix using the current parameters.
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
        return self.parameterArray


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
        self.parameterArray[:] = v
        self._construct_matrix()
        self.dirty = True


    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single gate parameter.  Thus, each column is of length
        op_dim^2 and there is one column per gate parameter.

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


    def compose(self, otherOp):
        """
        Create and return a new gate that is the composition of this operation
        followed by otherOp, which *must be another LinearlyParamDenseOp*.
        (For more general compositions between different types of gates, use
        the module-level compose function.)  The returned gate's matrix is
        equal to dot(this, otherOp).

        Parameters
        ----------
        otherOp : LinearlyParamDenseOp
            The gate to compose to the right of this one.

        Returns
        -------
        LinearlyParamDenseOp
        """
        assert( isinstance(otherOp, LinearlyParamDenseOp) )

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
        #     aW_in * sum_m d^(nj)_m R^(nj)_m +   # coeffs w/params of otherOp
        #     sum_l c^(ik)_l T^(ik)_l * Wb_kj +   # coeffs w/params of this gate
        #     sum_m,l c^(ik)_l W_kn d^(nj)_m T^(ik)_l R^(nj)_m) # coeffs w/params of both gates
        #

        W = _np.dot(self.rightTrans, otherOp.leftTrans)
        baseMx = _np.dot(self.baseMatrix, _np.dot(W, otherOp.baseMatrix)) # aWb above
        paramArray = _np.concatenate( (self.parameterArray, otherOp.parameterArray), axis=0)
        composedOp = LinearlyParamDenseOp(baseMx, paramArray, {},
                                                 self.leftTrans, otherOp.rightTrans,
                                                 self.enforceReal and otherOp.enforceReal,
                                                 self._evotype)

        # Precompute what we can before the compute loop
        aW = _np.dot(self.baseMatrix, W)
        Wb = _np.dot(W, otherOp.baseMatrix)

        kMax,nMax = (self.dim,self.dim) #W.shape
        offset = len(self.parameterArray) # amt to offset parameter indices of otherOp

        # Compute  [A * W * B]_ij element expression as described above
        for i in range(self.baseMatrix.shape[0]):
            for j in range(otherOp.baseMatrix.shape[1]):
                terms = []
                for n in range(nMax):
                    if (n,j) in otherOp.elementExpressions:
                        for term in otherOp.elementExpressions[(n,j)]:
                            coeff = aW[i,n] * term.coeff
                            paramIndices = [ p+offset for p in term.paramIndices ]
                            terms.append( LinearlyParameterizedElementTerm( coeff, paramIndices ) )

                for k in range(kMax):
                    if (i,k) in self.elementExpressions:
                        for term in self.elementExpressions[(i,k)]:
                            coeff = term.coeff * Wb[k,j]
                            terms.append( LinearlyParameterizedElementTerm( coeff, term.paramIndices ) )

                            for n in range(nMax):
                                if (n,j) in otherOp.elementExpressions:
                                    for term2 in otherOp.elementExpressions[(n,j)]:
                                        coeff = term.coeff * W[k,n] * term2.coeff
                                        paramIndices = term.paramIndices + [ p+offset for p in term2.paramIndices ]
                                        terms.append( LinearlyParameterizedElementTerm( coeff, paramIndices ) )

                composedOp.elementExpressions[(i,j)] = terms

        composedOp._construct_matrix()
        return composedOp


    def __str__(self):
        s = "Linearly Parameterized gate with shape %s, num params = %d\n" % \
            (str(self.base.shape), self.numParams)
        s += _mt.mx_to_string(self.base, width=5, prec=1)
        s += "\nParameterization:"
        for (i,j),terms in self.elementExpressions.items():
            tStr = ' + '.join([ '*'.join(["p%d"%p for p in term.paramIndices])
                                for term in terms] )
            s += "LinearOperator[%d,%d] = %s\n" % (i,j,tStr)
        return s



class EigenvalueParamDenseOp(DenseOperator):
    """
    Encapsulates a real operation matrix that is parameterized only by its
    eigenvalues, which are assumed to be either real or to occur in
    conjugate pairs.  Thus, the number of parameters is equal to the
    number of eigenvalues.
    """

    def __init__(self, matrix, includeOffDiagsInDegen2Blocks=False,
                 TPconstrainedAndUnital=False):
        """
        Initialize an EigenvalueParamDenseOp object.

        Parameters
        ----------
        matrix : numpy array
            a square 2D numpy array that gives the raw operation matrix to
            paramterize.  The shape of this array sets the dimension
            of the gate.

        includeOffDiagsInDegen2Blocks : bool
            If True, include as parameters the (initially zero) 
            off-diagonal elements in degenerate 2x2 blocks of the 
            the diagonalized operation matrix (no off-diagonals are 
            included in blocks larger than 2x2).  This is an option
            specifically used in the intelligent fiducial pair
            reduction (IFPR) algorithm.

        TPconstrainedAndUnital : bool
            If True, assume the top row of the operation matrix is fixed
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

        #Finish LinearOperator construction
        mx = _np.empty( matrix.shape, "d" )
        mx.flags.writeable = False # only _construct_matrix can change array
        DenseOperator.__init__(self, mx, "densitymx")
        self._construct_matrix() # construct base from the parameters


    def _construct_matrix(self):
        """
        Build the internal operation matrix using the current parameters.
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
        derivatives of the flattened operation matrix with respect to a
        single gate parameter.  Thus, each column is of length
        op_dim^2 and there is one column per gate parameter.

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension^2, num_params)
        """
        
        # matrix = B * diag * Bi, and only diag depends on parameters
        #  (and only linearly), so:
        # d(matrix)/d(param) = B * d(diag)/d(param) * Bi

        derivMx = _np.zeros( (self.dim**2, self.num_params()), 'd' ) # EigenvalueParameterizedGates are assumed to be real

        # Compute d(diag)/d(param) for each params, then apply B & Bi
        for k,pdesc in enumerate(self.params):
            dMx = _np.zeros( (self.dim,self.dim), 'complex')
            for prefactor,(i,j) in pdesc:
                dMx[i,j] = prefactor
            tmp = _np.dot(self.B, _np.dot(dMx, self.Bi))
            if _np.linalg.norm(tmp.imag) >= IMAG_TOL: #just a warning until we figure this out.
                print("EigenvalueParamDenseOp deriv_wrt_params WARNING:" + 
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
        

class LindbladOp(LinearOperator):
    """
    A gate parameterized by the coefficients of Lindblad-like terms, which are
    exponentiated to give the gate action.
    """

    @classmethod 
    def decomp_paramtype(cls, paramType):
        """
        A utility method for creating LindbladOp objects.

        Decomposes a high-level parameter-type `paramType` (e.g. `"H+S terms"`
        into a "base" type (specifies parameterization without evolution type,
        e.g. "H+S"), an evolution type (i.e. one of "densitymx", "svterm",
        "cterm", or "statevec").  Furthermore, from the base type two "modes"
        - one describing the number (and structure) of the non-Hamiltonian 
        Lindblad coefficients and one describing how the Lindblad coefficients
        are converted to/from parameters - are derived.

        The "non-Hamiltonian mode" describes which non-Hamiltonian Lindblad
        coefficients are stored in a LindbladOp, and is one
        of `"diagonal"` (only the diagonal elements of the full coefficient 
        matrix as a 1D array), `"diag_affine"` (a 2-by-d array of the diagonal
        coefficients on top of the affine projections), or `"all"` (the entire
        coefficient matrix).

        The "parameter mode" describes how the Lindblad coefficients/projections
        are converted into parameter values.  This can be:
        `"unconstrained"` (coefficients are independent unconstrained parameters),
        `"cptp"` (independent parameters but constrained so map is CPTP),
        `"depol"` (all non-Ham. diagonal coeffs are the *same, positive* value), or
        `"reldepol"` (same as `"depol"` but no positivity constraint).

        Parameters
        ----------
        paramType : str
            The high-level Lindblad parameter type to decompose.  E.g "H+S", 
            "H+S+A terms", "CPTP clifford terms".

        Returns
        -------
        basetype : str
        evotype : str
        nonham_mode : str
        param_mode : str
        """
        bTyp, evotype = _gt.split_lindblad_paramtype(paramType)
        
        if bTyp == "CPTP":
            nonham_mode = "all"; param_mode = "cptp" 
        elif bTyp in ("H+S","S"):
            nonham_mode = "diagonal"; param_mode = "cptp" 
        elif bTyp in ("H+s","s"):
            nonham_mode = "diagonal"; param_mode = "unconstrained" 
        elif bTyp in ("H+S+A","S+A"):
            nonham_mode = "diag_affine"; param_mode = "cptp" 
        elif bTyp in ("H+s+A","s+A"):
            nonham_mode = "diag_affine"; param_mode = "unconstrained" 
        elif bTyp in ("H+D","D"):
            nonham_mode = "diagonal"; param_mode = "depol" 
        elif bTyp in ("H+d","d"):
            nonham_mode = "diagonal"; param_mode = "reldepol" 
        elif bTyp in ("H+D+A","D+A"):
            nonham_mode = "diag_affine"; param_mode = "depol" 
        elif bTyp in ("H+d+A","d+A"):
            nonham_mode = "diag_affine"; param_mode = "reldepol" 

        elif bTyp == "GLND":
            nonham_mode = "all"; param_mode = "unconstrained" 
        else:
            raise ValueError("Unrecognized base type in `paramType`=%s" % paramType)

        return bTyp, evotype, nonham_mode, param_mode


    @classmethod
    def from_operation_obj(cls, gate, paramType="GLND", unitary_postfactor=None,
                      proj_basis="pp", mxBasis="pp", truncate=True, lazy=False):
        """
        Creates a LindbladOp from an existing LinearOperator object and
        some additional information.  

        This function is different from `from_operation_matrix` in that it assumes
        that `gate` is a :class:`LinearOperator`-derived object, and if `lazy=True` and 
        if `gate` is already a matching LindbladOp, it is
        returned directly.  This routine is primarily used in gate conversion
        functions, where conversion is desired only when necessary.

        Parameters
        ----------
        gate : LinearOperator
            The gate object to "convert" to a `LindbladOp`.

        paramType : str
            The high-level "parameter type" of the gate to create.  This 
            specifies both which Lindblad parameters are included and what
            type of evolution is used.  Examples of valid values are
            `"CPTP"`, `"H+S"`, `"S terms"`, and `"GLND clifford terms"`.
            
        unitaryPostfactor : numpy array or SciPy sparse matrix, optional
            a square 2D array of the same dimension of `gate`.  This specifies
            a part of the gate action to remove before parameterization via
            Lindblad projections.  Typically, this is a target (desired) gate
            operation such that only the erroneous part of the gate (i.e. the
            gate relative to the target), which should be close to the identity,
            is parameterized.  If none, the identity is used by default.

        proj_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis used to construct the Lindblad-term error generators onto 
            which the gate's error generator is projected.  Allowed values are
            Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        truncate : bool, optional
            Whether to truncate the projections onto the Lindblad terms in
            order to meet constraints (e.g. to preserve CPTP) when necessary.
            If False, then an error is thrown when the given `gate` cannot 
            be realized by the specified set of Lindblad projections.
            
        lazy : bool, optional
            If True, then if `gate` is already a LindbladOp
            with the requested details (given by the other arguments), then
            `gate` is returned directly and no conversion/copying is performed.
            If False, then a new gate object is always created and returned.

        Returns
        -------
        LindbladOp
        """
        RANK_TOL = 1e-6

        if unitary_postfactor is None:
            #Try to obtain unitary_post by getting the closest unitary
            if isinstance(gate, LindbladDenseOp):
                unitary_postfactor = gate.unitary_postfactor
            elif isinstance(gate, LinearOperator) and gate._evotype == "densitymx": 
                J = _jt.fast_jamiolkowski_iso_std(gate.todense(), mxBasis) #Choi mx basis doesn't matter
                if _np.linalg.matrix_rank(J, RANK_TOL) == 1: 
                    unitary_postfactor = gate # when 'gate' is unitary
            # FUTURE: support other gate._evotypes?
            else:
                unitary_postfactor = None

        #Break paramType in to a "base" type and an evotype
        bTyp, evotype, nonham_mode, param_mode = cls.decomp_paramtype(paramType)

        ham_basis = proj_basis if (("H+" in bTyp) or bTyp in ("CPTP","GLND")) else None
        nonham_basis = proj_basis
        
        def beq(b1,b2):
            """ Check if bases have equal names """
            b1 = b1.name if isinstance(b1,_Basis) else b1
            b2 = b2.name if isinstance(b2,_Basis) else b2
            return b1 == b2

        def normeq(a,b):
            if a is None and b is None: return True
            if a is None or b is None: return False
            return _mt.safenorm(a-b) < 1e-6 # what about possibility of Clifford gates?
        
        if isinstance(gate, LindbladOp) and \
           normeq(gate.unitary_postfactor,unitary_postfactor) and \
           isinstance(gate.errorgen, LindbladErrorgen) \
           and beq(ham_basis,gate.errorgen.ham_basis) and beq(nonham_basis,gate.errorgen.other_basis) \
           and param_mode==gate.errorgen.param_mode and nonham_mode==gate.errorgen.nonham_mode \
           and beq(mxBasis,gate.errorgen.matrix_basis) and gate._evotype == evotype and lazy:
            return gate #no creation necessary!
        else:
            return cls.from_operation_matrix(
                gate, unitary_postfactor, ham_basis, nonham_basis, param_mode,
                nonham_mode, truncate, mxBasis, evotype)


    @classmethod
    def from_operation_matrix(cls, opMatrix, unitaryPostfactor=None,
                         ham_basis="pp", nonham_basis="pp", param_mode="cptp",
                         nonham_mode="all", truncate=True, mxBasis="pp",
                         evotype="densitymx"):
        """
        Creates a Lindblad-parameterized gate from a matrix and a basis which
        specifies how to decompose (project) the gate's error generator.

        Parameters
        ----------
        opMatrix : numpy array or SciPy sparse matrix
            a square 2D array that gives the raw operation matrix, assumed to
            be in the `mxBasis` basis, to parameterize.  The shape of this
            array sets the dimension of the gate. If None, then it is assumed
            equal to `unitaryPostfactor` (which cannot also be None). The
            quantity `opMatrix inv(unitaryPostfactor)` is parameterized via
            projection onto the Lindblad terms.
            
        unitaryPostfactor : numpy array or SciPy sparse matrix, optional
            a square 2D array of the same size of `opMatrix` (if
            not None).  This matrix specifies a part of the gate action 
            to remove before parameterization via Lindblad projections.
            Typically, this is a target (desired) gate operation such 
            that only the erroneous part of the gate (i.e. the gate 
            relative to the target), which should be close to the identity,
            is parameterized.  If none, the identity is used by default.

        ham_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis is used to construct the Hamiltonian-type lindblad error
            Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        other_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis is used to construct the Stochastic-type lindblad error
            Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        param_mode : {"unconstrained", "cptp", "depol", "reldepol"}
            Describes how the Lindblad coefficients/projections relate to the
            gate's parameter values.  Allowed values are:
            `"unconstrained"` (coeffs are independent unconstrained parameters),
            `"cptp"` (independent parameters but constrained so map is CPTP),
            `"reldepol"` (all non-Ham. diagonal coeffs take the *same* value),
            `"depol"` (same as `"reldepol"` but coeffs must be *positive*)

        nonham_mode : {"diagonal", "diag_affine", "all"}
            Which non-Hamiltonian Lindblad projections are potentially non-zero.
            Allowed values are: `"diagonal"` (only the diagonal Lind. coeffs.),
            `"diag_affine"` (diagonal coefficients + affine projections), and
            `"all"` (the entire matrix of coefficients is allowed).

        truncate : bool, optional
            Whether to truncate the projections onto the Lindblad terms in
            order to meet constraints (e.g. to preserve CPTP) when necessary.
            If False, then an error is thrown when the given `gate` cannot 
            be realized by the specified set of Lindblad projections.

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        evotype : {"densitymx","svterm","cterm"}
            The evolution type of the gate being constructed.  `"densitymx"` is
            usual Lioville density-matrix-vector propagation via matrix-vector
            products.  `"svterm"` denotes state-vector term-based evolution
            (action of gate is obtained by evaluating the rank-1 terms up to
            some order).  `"cterm"` is similar but uses Clifford gate action
            on stabilizer states.

        Returns
        -------
        LindbladOp        
        """
        
        #Compute a (errgen, unitaryPostfactor) pair from the given
        # (opMatrix, unitaryPostfactor) pair.  Works with both
        # dense and sparse matrices.

        if opMatrix is None:
            assert(unitaryPostfactor is not None), "arguments cannot both be None"
            opMatrix = unitaryPostfactor
            
        sparseOp = _sps.issparse(opMatrix)
        if unitaryPostfactor is None:
            if sparseOp:
                upost = _sps.identity(opMatrix.shape[0],'d','csr')
            else: upost = _np.identity(opMatrix.shape[0],'d')
        else: upost = unitaryPostfactor

        #Init base from error generator: sets basis members and ultimately
        # the parameters in self.paramvals
        if sparseOp:
            #Instead of making error_generator(...) compatible with sparse matrices
            # we require sparse matrices to have trivial initial error generators
            # or we convert to dense:
            if(_mt.safenorm(opMatrix-upost) < 1e-8):
                errgenMx = _sps.csr_matrix( opMatrix.shape, dtype='d' ) # all zeros
            else:
                errgenMx = _sps.csr_matrix(
                    _gt.error_generator(opMatrix.toarray(), upost.toarray(),
                                        mxBasis, "logGTi"), dtype='d')
        else:
            #DB: assert(_np.linalg.norm(opMatrix.imag) < 1e-8)
            #DB: assert(_np.linalg.norm(upost.imag) < 1e-8)
            errgenMx = _gt.error_generator(opMatrix, upost, mxBasis, "logGTi")

        errgen = LindbladErrorgen.from_error_generator(errgenMx, ham_basis,
                                        nonham_basis, param_mode, nonham_mode,
                                        mxBasis, truncate, evotype)

        #Use "sparse" matrix exponentiation when given operation matrix was sparse.
        return cls(unitaryPostfactor, errgen, sparse_expm=sparseOp)


    def __init__(self, unitaryPostfactor, errorgen, sparse_expm=False):     
        """
        Create a new `LinbladOp` based on an error generator and postfactor.

        Note that if you want to construct a `LinbladOp` from an operation
        matrix, you can use the :method:`from_operation_matrix` class
        method and save youself some time and effort.

        Parameters
        ----------
        unitaryPostfactor : numpy array or SciPy sparse matrix or int
            a square 2D array which specifies a part of the gate action 
            to remove before parameterization via Lindblad projections.
            While this is termed a "post-factor" because it occurs to the
            right of the exponentiated Lindblad terms, this means it is applied
            to a state *before* the Lindblad terms (which usually represent
            gate errors).  Typically, this is a target (desired) gate operation.
            If this post-factor is just the identity you can simply pass the
            integer dimension as `unitaryPostfactor` instead of a matrix, or 
            you can pass `None` and the dimension will be inferred from
            `errorgen`.

        errorgen : LinearOperator
            The error generator for this operator.  That is, the `L` if this
            operator is `exp(L)*unitaryPostfactor`.

        sparse_expm : bool, optional
            Whether to implement exponentiation in an approximate way that
            treats the error generator as a sparse matrix.  Namely, it only
            uses the action of `errorgen` and its adjoint on a state.  Setting
            `sparse_expm=True` is typically more efficient when `errorgen` has
            a large dimension, say greater than 100.
        """

        # Extract superop dimension from 'errorgen'
        d2 = errorgen.dim
        d = int(round(_np.sqrt(d2)))
        assert(d*d == d2), "LinearOperator dim must be a perfect square"

        self.errorgen = errorgen # don't copy (allow object reuse)
        
        # make unitary postfactor sparse when sparse_expm == True and vice versa.
        # (This doens't have to be the case, but we link these two "sparseness" noitions:
        #  when we perform matrix exponentiation in a "sparse" way we assume the matrices
        #  are large and so the unitary postfactor (if present) should be sparse).
        # FUTURE: warn if there is a sparsity mismatch btwn basis and postfactor?
        self.sparse_expm = sparse_expm
        if unitaryPostfactor is not None:
            if self.sparse_expm == False and _sps.issparse(unitaryPostfactor):
                unitaryPostfactor = unitaryPostfactor.toarray() # sparse -> dense
            elif self.sparse_expm == True and not _sps.issparse(unitaryPostfactor):
                unitaryPostfactor = _sps.csr_matrix( _np.asarray(unitaryPostfactor) ) # dense -> sparse

        evotype = self.errorgen._evotype
        LinearOperator.__init__(self, d2, evotype) #sets self.dim


        #Finish initialization based on evolution type
        if evotype == "densitymx":
            self.unitary_postfactor = unitaryPostfactor #can be None
            self.exp_err_gen = None
            self.err_gen_prep = None
            self._prepare_for_torep() #sets one of the above two members
                                      # depending on self.errorgen.sparse
            self.terms = None # Unused
            
        else: # Term-based evolution
            
            assert(not self.sparse_expm), "Sparse unitary postfactors are not supported for term-based evolution"
              #TODO: make terms init-able from sparse elements, and below code work with a *sparse* unitaryPostfactor
            termtype = "dense" if evotype == "svterm" else "clifford"

            # Store *unitary* as self.unitary_postfactor - NOT a superop
            if unitaryPostfactor is not None: #can be None
                op_std = _bt.change_basis(unitaryPostfactor, self.errorgen.matrix_basis, 'std')
                self.unitary_postfactor = _gt.process_mx_to_unitary(op_std)

                # automatically "up-convert" gate to CliffordOp if needed
                if termtype == "clifford":
                    self.unitary_postfactor = CliffordOp(self.unitary_postfactor) 
            else:
                self.unitary_postfactor = None
            
            self.terms = {}
            
            # Unused
            self.err_gen_prep = self.exp_err_gen = None

        #Done with __init__(...)


    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.
        
        Returns
        -------
        list
        """
        return [self.errorgen]


    def copy(self, parent=None):
        """
        Copy this object.

        Returns
        -------
        LinearOperator
            A copy of this object.
        """
        # We need to override this method so that error map has its
        # parent reset correctly.
        if self.unitary_postfactor is None:
            upost = None
        elif self._evotype == "densitymx":
            upost = self.unitary_postfactor
        else:
            #self.unitary_postfactor is actually the *unitary* not the postfactor
            termtype = "dense" if self._evotype == "svterm" else "clifford"

            # automatically "up-convert" gate to CliffordOp if needed
            if termtype == "clifford":
                assert(isinstance(self.unitary_postfactor, CliffordOp)) # see __init__
                U = self.unitary_postfactor.unitary
            else: U = self.unitary_postfactor
            op_std = _gt.unitary_to_process_mx(U)
            upost = _bt.change_basis(op_std, 'std', self.errorgen.matrix_basis)

        cls = self.__class__ # so that this method works for derived classes too
        copyOfMe = cls(upost, self.errorgen.copy(parent), self.sparse_expm)
        return self._copy_gpindices(copyOfMe, parent)

        
    def _prepare_for_torep(self):
        """ 
        Prepares needed intermediate values for calls to `torep()`.
        (sets `self.err_gen_prep` or `self.exp_err_gen`).
        """                
        #Pre-compute the exponential of the error generator if dense matrices
        # are used, otherwise cache prepwork for sparse expm calls
        if self.sparse_expm: # "sparse mode" => don't ever compute matrix-exponential explicitly
            self.err_gen_prep = _mt.expop_multiply_prep(
                self.errorgen.torep().aslinearoperator())
        else:
            self.exp_err_gen = _spl.expm(self.errorgen.todense())


        
    def set_gpindices(self, gpindices, parent, memo=None):
        """
        Set the parent and indices into the parent's parameter vector that
        are used by this ModelMember object.

        Parameters
        ----------
        gpindices : slice or integer ndarray
            The indices of this objects parameters in its parent's array.

        parent : Model or ModelMember
            The parent whose parameter array gpindices references.

        Returns
        -------
        None
        """
        self.terms = {} # clear terms cache since param indices have changed now
        _modelmember.ModelMember.set_gpindices(self, gpindices, parent, memo)


    def todense(self):
        """
        Return this operation as a dense matrix.
        """
        if self.sparse_expm: raise NotImplementedError("todense() not implemented for sparse-expm-mode LindbladOp objects")
        if self._evotype in ("svterm","cterm"): 
            raise NotImplementedError("todense() not implemented for term-based LindbladOp objects")

        if self.unitary_postfactor is not None:
            dense = _np.dot(self.exp_err_gen, self.unitary_postfactor)
        else:
            dense = self.exp_err_gen
        return dense

    #FUTURE: maybe remove this function altogether, as it really shouldn't be called
    def tosparse(self):
        """
        Return the operation as a sparse matrix.
        """
        _warnings.warn(("Constructing the sparse matrix of a LindbladDenseOp."
                        "  Usually this is *NOT* a sparse matrix (the exponential of a"
                        " sparse matrix isn't generally sparse)!"))
        if self.sparse_expm:
            exp_err_gen = _spsl.expm(self.errorgen.tosparse().tocsc()).tocsr()
            if self.unitary_postfactor is not None:
                return exp_err_gen.dot(self.unitary_postfactor)
            else:
                return exp_err_gen
        else:
            return _sps.csr_matrix(self.todense())

        
    def torep(self):
        """
        Return a "representation" object for this gate.

        Such objects are primarily used internally by pyGSTi to compute
        things like probabilities more efficiently.

        Returns
        -------
        OpRep
        """
        if self._evotype == "densitymx":
            if self.sparse_expm:
                if self.unitary_postfactor is None:
                    Udata = _np.empty(0,'d')
                    Uindices = Uindptr = _np.empty(0,_np.int64)
                else:
                    assert(_sps.isspmatrix_csr(self.unitary_postfactor)), \
                        "Internal error! Unitary postfactor should be a *sparse* CSR matrix!"
                    Udata = self.unitary_postfactor.data
                    Uindptr = _np.ascontiguousarray(self.unitary_postfactor.indptr, _np.int64)
                    Uindices = _np.ascontiguousarray(self.unitary_postfactor.indices, _np.int64)
                           
                mu, m_star, s, eta = self.err_gen_prep
                errorgen_rep = self.errorgen.torep()
                return replib.DMOpRep_Lindblad(errorgen_rep,
                                                 mu, eta, m_star, s,
                                                 Udata, Uindices, Uindptr)
            else:
                if self.unitary_postfactor is not None:
                    dense = _np.dot(self.exp_err_gen, self.unitary_postfactor)
                else: dense = self.exp_err_gen
                return replib.DMOpRep_Dense(_np.ascontiguousarray(dense,'d'))
        else:
            raise ValueError("Invalid evotype '%s' for %s.torep(...)" %
                             (self._evotype, self.__class__.__name__))
        
        
    def get_order_terms(self, order):
        """ 
        Get the `order`-th order Taylor-expansion terms of this operation.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that its action on a
        density matrix `rho` can be written:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the gate's
        parameters, where the polynomial's variable indices index the *global*
        parameters of the gate's parent (usually a :class:`Model`), not the 
        gate's local parameter array (i.e. that returned from `to_vector`).


        Parameters
        ----------
        order : int
            The order of terms to get.

        Returns
        -------
        list
            A list of :class:`RankOneTerm` objects.
        """

        def _compose_poly_indices(terms):
            for term in terms:
                term.map_indices(lambda x: tuple(_modelmember._compose_gpindices(
                    self.gpindices, _np.array(x,_np.int64))) )
            return terms
        
        if order not in self.terms:
            if self._evotype == "svterm": tt = "dense"
            elif self._evotype == "cterm": tt = "clifford"
            else: raise ValueError("Invalid evolution type %s for calling `get_order_terms`" % self._evotype)

            assert(self.gpindices is not None),"LindbladOp must be added to a Model before use!"
            assert(not _sps.issparse(self.unitary_postfactor)), "Unitary post-factor needs to be dense for term-based evotypes"
              # for now - until StaticDenseOp and CliffordOp can init themselves from a *sparse* matrix
            postTerm = _term.RankOneTerm(_Polynomial({(): 1.0}), self.unitary_postfactor,
                                         self.unitary_postfactor, tt) 
            #Note: for now, *all* of an error generator's terms are considered 0-th order,
            # so the below call to get_order_terms just gets all of them.  In the FUTURE
            # we might want to allow a distinction among the error generator terms, in which
            # case this term-exponentiation step will need to become more complicated...
            loc_terms = _term.exp_terms(self.errorgen.get_order_terms(0), [order], postTerm)[order]
            #OLD: loc_terms = [ t.collapse() for t in loc_terms ] # collapse terms for speed
            self.terms[order] = _compose_poly_indices(loc_terms)
        return self.terms[order]


    def num_params(self):
        """
        Get the number of independent parameters which specify this gate.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.errorgen.num_params()


    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this gate.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self.errorgen.to_vector()


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
        self.errorgen.from_vector(v)
        if self._evotype == "densitymx":
            self._prepare_for_torep()
        self.dirty = True

    def get_errgen_coeffs(self):
        """
        Constructs a dictionary of the Lindblad-error-generator coefficients 
        (i.e. the "error rates") of this operation.  Note that these are not
        necessarily the parameter values, as these coefficients are generally
        functions of the parameters (so as to keep the coefficients positive,
        for instance).

        Returns
        -------
        Ltermdict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Basis labels are integers starting at 0.  Values are complex
            coefficients (error rates).
    
        basisdict : dict
            A dictionary mapping the integer basis labels used in the
            keys of `Ltermdict` to basis matrices..
        """
        return self.errorgen.get_coeffs()

        
    def set_value(self, M):
        """
        Attempts to modify gate parameters so that the specified raw
        operation matrix becomes mx.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        M : array_like or LinearOperator
            An array of shape (dim, dim) or LinearOperator representing the gate action.

        Returns
        -------
        None
        """

        #TODO: move this function to errorgen?
        if not isinstance(self.errorgen, LindbladErrorgen):
            raise NotImplementedError(("Can only set the value of a LindbladDenseOp that "
                                       "contains a single LindbladErrorgen error generator"))

        tOp = LindbladOp.from_operation_matrix(
            M,self.unitary_postfactor,
            self.errorgen.ham_basis, self.errorgen.other_basis,
            self.errorgen.param_mode,self.errorgen.nonham_mode,
            True, self.errorgen.matrix_basis, self._evotype)

        #Note: truncate=True to be safe
        self.errorgen.from_vector(tOp.errorgen.to_vector())
        if self._evotype == "densitymx":
            self._prepare_for_torep()
        self.dirty = True

    
    def transform(self, S):
        """
        Update operation matrix G with inv(S) * G * S,

        Generally, the transform function updates the *parameters* of 
        the gate such that the resulting operation matrix is altered as 
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
            self.errorgen.transform(S)
            self._prepare_for_torep() # needed to rebuild exponentiated error gen
            self.dirty = True
            #Note: truncate=True above because some unitary transforms seem to
            ## modify eigenvalues to be negative beyond the tolerances
            ## checked when truncate == False.  I'm not sure why this occurs,
            ## since a true unitary should map CPTP -> CPTP...

            #CHECK WITH OLD (passes) TODO move to unit tests?
            #tMx = _np.dot(Uinv,_np.dot(self.base, U)) #Move above for checking
            #tOp = LindbladDenseOp(tMx,self.unitary_postfactor,
            #                                self.ham_basis, self.other_basis,
            #                                self.cptp,self.nonham_diagonal_only,
            #                                True, self.matrix_basis)
            #assert(_np.linalg.norm(tOp.paramvals - self.paramvals) < 1e-6)
        else:
            raise ValueError("Invalid transform for this LindbladDenseOp: type %s"
                             % str(type(S)))

    def spam_transform(self, S, typ):
        """
        Update operation matrix G with inv(S) * G OR G * S,
        depending on the value of `typ`.

        This functions as `transform(...)` but is used when this
        Lindblad-parameterized gate is used as a part of a SPAM
        vector.  When `typ == "prep"`, the spam vector is assumed
        to be `rho = dot(self, <spamvec>)`, which transforms as
        `rho -> inv(S) * rho`, so `self -> inv(S) * self`. When
        `typ == "effect"`, `e.dag = dot(e.dag, self)` (not that
        `self` is NOT `self.dag` here), and `e.dag -> e.dag * S`
        so that `self -> self * S`.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.

        typ : { 'prep', 'effect' }
            Which type of SPAM vector is being transformed (see above).
        """
        assert(typ in ('prep','effect')), "Invalid `typ` argument: %s" % typ
        
        if isinstance(S, _gaugegroup.UnitaryGaugeGroupElement) or \
           isinstance(S, _gaugegroup.TPSpamGaugeGroupElement):
            U = S.get_transform_matrix()
            Uinv = S.get_transform_matrix_inverse()

            #just act on postfactor and Lindbladian exponent:
            if typ == "prep":
                if self.unitary_postfactor is not None:
                    self.unitary_postfactor = _mt.safedot(Uinv,self.unitary_postfactor)
            else:
                if self.unitary_postfactor is not None:
                    self.unitary_postfactor = _mt.safedot(self.unitary_postfactor, U)
                
            self.errorgen.spam_transform(S,typ)
            self._prepare_for_torep() # needed to rebuild exponentiated error gen
            self.dirty = True
            #Note: truncate=True above because some unitary transforms seem to
            ## modify eigenvalues to be negative beyond the tolerances
            ## checked when truncate == False.  I'm not sure why this occurs,
            ## since a true unitary should map CPTP -> CPTP...
        else:
            raise ValueError("Invalid transform for this LindbladDenseOp: type %s"
                             % str(type(S)))

        
    def __str__(self):
        s = "Lindblad Parameterized gate map with dim = %d, num params = %d\n" % \
            (self.dim, self.num_params())
        return s


class LindbladDenseOp(LindbladOp,DenseOperator):
    """
    Encapsulates a operation matrix that is parameterized by a Lindblad-form
    expression, such that each parameter multiplies a particular term in
    the Lindblad form that is exponentiated to give the operation matrix up
    to an optional unitary prefactor).  The basis used by the Lindblad
    form is referred to as the "projection basis".
    """

    def __init__(self, unitaryPostfactor, errorgen, sparse_expm=False):     
        """
        Create a new LinbladDenseOp based on an error generator and postfactor.

        Note that if you want to construct a `LinbladDenseOp` from an operation
        matrix, you can use the :method:`from_operation_matrix` class method
        and save youself some time and effort.

        Parameters
        ----------
        unitaryPostfactor : numpy array or SciPy sparse matrix or int
            a square 2D array which specifies a part of the gate action 
            to remove before parameterization via Lindblad projections.
            While this is termed a "post-factor" because it occurs to the
            right of the exponentiated Lindblad terms, this means it is applied
            to a state *before* the Lindblad terms (which usually represent
            gate errors).  Typically, this is a target (desired) gate operation.
            If this post-factor is just the identity you can simply pass the
            integer dimension as `unitaryPostfactor` instead of a matrix, or 
            you can pass `None` and the dimension will be inferred from
            `errorgen`.

        errorgen : LinearOperator
            The error generator for this operator.  That is, the `L` if this
            operator is `exp(L)*unitaryPostfactor`.

        sparse_expm : bool, optional
            Whether to implement exponentiation in an approximate way that
            treats the error generator as a sparse matrix.  Namely, it only
            uses the action of `errorgen` and its adjoint on a state.  Setting
            `sparse_expm=True` is typically more efficient when `errorgen` has
            a large dimension, say greater than 100.
        """
        assert(errorgen._evotype == "densitymx"), \
            "LindbladDenseOp objects can only be used for the 'densitymx' evolution type"
            #Note: cannot remove the evotype argument b/c we need to maintain the same __init__
            # signature as LindbladOp so its @classmethods will work on us.

        # sparse_expm mode must be an arguement to mirror the call
        # signature of LindbladOp
        assert(not sparse_expm), \
            "LindbladDenseOp objects must have `sparse_expm=False`!"
        
        #Start with base class construction
        LindbladOp.__init__(
            self, unitaryPostfactor, errorgen, sparse_expm=False) # (sets self.dim and self.base)

        DenseOperator.__init__(self, self.base, "densitymx")
            

    def _prepare_for_torep(self): 
        """
        Build the internal operation matrix using the current parameters.
        """
        # Formerly a separate "construct_matrix" function, this extends
        # LindbladParmaeterizedGateMap's version, which just constructs
        # self.err_gen & self.exp_err_gen, to constructing the entire
        # final matrix.
        LindbladOp._prepare_for_torep(self) #constructs self.exp_err_gen b/c *not sparse*
        matrix = self.todense()  # dot(exp_err_gen, unitary_postfactor)

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

    def torep(self):
        """
        Return a "representation" object for this gate.

        Such objects are primarily used internally by pyGSTi to compute
        things like probabilities more efficiently.

        Returns
        -------
        OpRep
        """
        #Implement this b/c some ambiguity since both LindbladDenseOp
        # and DenseOperator implement torep() - and we want to use the DenseOperator one.
        if self._evotype == "densitymx":
            return DenseOperator.torep(self)
        else:
            raise ValueError("Invalid evotype '%s' for %s.torep(...)" %
                             (self._evotype, self.__class__.__name__))

        
    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single gate parameter.  Thus, each column is of length
        op_dim^2 and there is one column per gate parameter.

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension^2, num_params)
        """    
        if self.base_deriv is None:
            d2 = self.dim

            #Deriv wrt hamiltonian params
            derrgen = self.errorgen.deriv_wrt_params(None) #apply filter below; cache *full* deriv
            derrgen.shape = (d2,d2,-1) # separate 1st d2**2 dim to (d2,d2)
            dexpL = _dexpX(self.errorgen.todense(), derrgen, self.exp_err_gen,
                           self.unitary_postfactor)
            derivMx = dexpL.reshape(d2**2,self.num_params()) # [iFlattenedOp,iParam]

            assert(_np.linalg.norm(_np.imag(derivMx)) < IMAG_TOL), \
                ("Deriv matrix has imaginary part = %s.  This can result from "
                 "evaluating a Model derivative at a 'bad' point where the "
                 "error generator is large.  This often occurs when GST's "
                 "starting Model has *no* stochastic error and all such "
                 "parameters affect error rates at 2nd order.  Try "
                 "depolarizing the seed Model.") % str(_np.linalg.norm(_np.imag(derivMx)))
                 # if this fails, uncomment around "DB COMMUTANT NORM" for further debugging.
            derivMx = _np.real(derivMx)
            self.base_deriv = derivMx

            #check_deriv_wrt_params(self, derivMx, eps=1e-7)
            #fd_deriv = finite_difference_deriv_wrt_params(self, eps=1e-7)
            #derivMx = fd_deriv

        if wrtFilter is None:
            return self.base_deriv.view()
                #view because later setting of .shape by caller can mess with self.base_deriv!
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
        Construct the Hessian of this operation with respect to its parameters.

        This function returns a tensor whose first axis corresponds to the
        flattened operation matrix and whose 2nd and 3rd axes correspond to the
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
            nP = self.num_params()
            hessianMx = _np.zeros( (d2**2, nP, nP), 'd' )
    
            #Deriv wrt other params
            dEdp = self.errorgen.deriv_wrt_params(None) # filter later, cache *full*
            d2Edp2 = self.errorgen.hessian_wrt_params(None,None) # hessian
            dEdp.shape = (d2,d2,nP) # separate 1st d2**2 dim to (d2,d2)
            d2Edp2.shape = (d2,d2,nP,nP) # ditto

            series, series2 = _d2expSeries(self.errorgen.todense(), dEdp, d2Edp2)
            term1 = series2
            term2 = _np.einsum("ija,jkq->ikaq",series,series)
            if self.unitary_postfactor is None:
                d2expL = _np.einsum("ikaq,kj->ijaq", term1+term2,
                                    self.exp_err_gen)
            else:
                d2expL = _np.einsum("ikaq,kl,lj->ijaq", term1+term2,
                                    self.exp_err_gen, self.unitary_postfactor)
            hessianMx = d2expL.reshape((d2**2,nP,nP))

            #hessian has been made so index as [iFlattenedOp,iDeriv1,iDeriv2]
            assert(_np.linalg.norm(_np.imag(hessianMx)) < IMAG_TOL)
            hessianMx = _np.real(hessianMx) # d2O block of hessian

            self.base_hessian = hessianMx

            #TODO: check hessian with finite difference here?
            
        if wrtFilter1 is None:
            if wrtFilter2 is None:
                return self.base_hessian.view()
                  #view because later setting of .shape by caller can mess with self.base_hessian!
            else:
                return _np.take(self.base_hessian, wrtFilter2, axis=2 )
        else:
            if wrtFilter2 is None:
                return _np.take(self.base_hessian, wrtFilter1, axis=1 )
            else:
                return _np.take( _np.take(self.base_hessian, wrtFilter1, axis=1),
                                 wrtFilter2, axis=2 )

        
def _dexpSeries(X, dX):
    TERM_TOL = 1e-12
    tr = len(dX.shape) #tensor rank of dX; tr-2 == # of derivative dimensions
    assert( (tr-2) in (1,2)), "Currently, dX can only have 1 or 2 derivative dimensions"
    #assert( len( (_np.isnan(dX)).nonzero()[0] ) == 0 ) # NaN debugging
    #assert( len( (_np.isnan(X)).nonzero()[0] ) == 0 ) # NaN debugging
    series = dX.copy() # accumulates results, so *need* a separate copy
    last_commutant = term = dX; i=2

    #take d(matrix-exp) using series approximation
    while _np.amax(_np.abs(term)) > TERM_TOL: #_np.linalg.norm(term)
        if tr == 3:
            #commutant = _np.einsum("ik,kja->ija",X,last_commutant) - \
            #            _np.einsum("ika,kj->ija",last_commutant,X)
            commutant = _np.tensordot(X,last_commutant,(1,0)) - \
                        _np.transpose(_np.tensordot(last_commutant,X,(1,0)),(0,2,1))
        elif tr == 4:
            #commutant = _np.einsum("ik,kjab->ijab",X,last_commutant) - \
            #        _np.einsum("ikab,kj->ijab",last_commutant,X)
            commutant = _np.tensordot(X,last_commutant,(1,0)) - \
                _np.transpose(_np.tensordot(last_commutant,X,(1,0)),(0,3,1,2))

        term = 1/_np.math.factorial(i) * commutant

        #Uncomment some/all of this when you suspect an overflow due to X having large norm.
        #print("DB COMMUTANT NORM = ",_np.linalg.norm(commutant)) # sometimes this increases w/iter -> divergence => NaN
        #assert(not _np.isnan(_np.linalg.norm(term))), \
        #    ("Haddamard series = NaN! Probably due to trying to differentiate "
        #     "exp(X) where X has a large norm!")

        #DEBUG
        #if not _np.isfinite(_np.linalg.norm(term)): break # DEBUG high values -> overflow for nqubit gates
        #if len( (_np.isnan(term)).nonzero()[0] ) > 0: # NaN debugging
        #    #WARNING: stopping early b/c of NaNs!!! - usually caused by infs
        #    break
        
        series += term #1/_np.math.factorial(i) * commutant
        last_commutant = commutant; i += 1
    return series

def _d2expSeries(X, dX, d2X):
    TERM_TOL = 1e-12
    tr = len(dX.shape) #tensor rank of dX; tr-2 == # of derivative dimensions
    tr2 = len(d2X.shape) #tensor rank of dX; tr-2 == # of derivative dimensions
    assert( (tr-2,tr2-2) in [(1,2),(2,4)]), "Current support for only 1 or 2 derivative dimensions"

    series = dX.copy() # accumulates results, so *need* a separate copy
    series2 = d2X.copy() # accumulates results, so *need* a separate copy
    term = last_commutant = dX
    last_commutant2 = term2 = d2X
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
        #dExpX = _np.einsum('ika,kj->ija', series, expX)
        dExpX = _np.transpose(_np.tensordot(series, expX, (1,0)),(0,2,1))
        if postfactor is not None:
            #dExpX = _np.einsum('ila,lj->ija', dExpX, postfactor)
            dExpX = _np.transpose(_np.tensordot(dExpX, postfactor, (1,0)),(0,2,1))
    elif tr == 4:
        #dExpX = _np.einsum('ikab,kj->ijab', series, expX)
        dExpX = _np.transpose(_np.tensordot(series, expX, (1,0)),(0,3,1,2))
        if postfactor is not None:
            #dExpX = _np.einsum('ilab,lj->ijab', dExpX, postfactor)
            dExpX = _np.transpose(_np.tensordot(dExpX, postfactor, (1,0)),(0,3,1,2))
            
    return dExpX



class TPInstrumentOp(DenseOperator):
    """
    A partial implementation of :class:`LinearOperator` which encapsulates an element of a
    :class:`TPInstrument`.  Instances rely on their parent being a 
    `TPInstrument`.
    """

    def __init__(self, param_ops, index):
        """
        Initialize a TPInstrumentOp object.

        Parameters
        ----------
        param_ops : list of LinearOperator objects
            A list of the underlying gate objects which constitute a simple
            parameterization of a :class:`TPInstrument`.  Namely, this is
            the list of [MT,D1,D2,...Dn] gates which parameterize *all* of the
            `TPInstrument`'s elements.

        index : int
            The index indicating which element of the `TPInstrument` the
            constructed object is.  Must be in the range 
            `[0,len(param_ops)-1]`.
        """
        self.param_ops = param_ops
        self.index = index
        DenseOperator.__init__(self, _np.identity(param_ops[0].dim,'d'),
                            "densitymx") #Note: sets self.gpindices; TP assumed real
        self._construct_matrix()

        #Set our own parent and gpindices based on param_ops
        # (this breaks the usual paradigm of having the parent object set these,
        #  but the exception is justified b/c the parent has set these members
        #  of the underlying 'param_ops' gates)
        self.dependents = [0,index+1] if index < len(param_ops)-1 \
                          else list(range(len(param_ops)))
          #indices into self.param_ops of the gates this gate depends on
        self.set_gpindices(_slct.list_to_slice(
            _np.concatenate( [ param_ops[i].gpindices_as_array()
                               for i in self.dependents ], axis=0),True,False),
                           param_ops[0].parent) #use parent of first param gate
                                                  # (they should all be the same)


    def _construct_matrix(self):
        """
        Mi = Di + MT for i = 1...(n-1)
           = -(n-2)*MT-sum(Di) = -(n-2)*MT-[(MT-Mi)-n*MT] for i == (n-1)
        """
        nEls = len(self.param_ops)
        if self.index < nEls-1:
            self.base = _np.asarray( self.param_ops[self.index+1]
                                     + self.param_ops[0] )
        else:
            assert(self.index == nEls-1), \
                "Invalid index %d > %d" % (self.index,nEls-1)
            self.base = _np.asarray( -sum(self.param_ops)
                                     -(nEls-3)*self.param_ops[0] )
        
        assert(self.base.shape == (self.dim,self.dim))
        self.base.flags.writeable = False

        
    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single gate parameter.  Thus, each column is of length
        op_dim^2 and there is one column per gate parameter. An
        empty 2D array in the StaticDenseOp case (num_params == 0).

        Returns
        -------
        numpy array
            Array of derivatives with shape (dimension^2, num_params)
        """
        Np = self.num_params()
        derivMx = _np.zeros((self.dim**2,Np),'d')
        Nels = len(self.param_ops)

        off = 0
        if self.index < Nels-1: # matrix = Di + MT = param_ops[index+1] + param_ops[0]
            for i in [self.index+1, 0]:
                Np = self.param_ops[i].num_params()
                derivMx[:,off:off+Np] = self.param_ops[i].deriv_wrt_params()
                off += Np

        else: # matrix = -(nEls-1)*MT-sum(Di)
            Np = self.param_ops[0].num_params()
            derivMx[:,off:off+Np] = -(Nels-1)*self.param_ops[0].deriv_wrt_params()
            off += Np

            for i in range(1,Nels):
                Np = self.param_ops[i].num_params()
                derivMx[:,off:off+Np] = -self.param_ops[i].deriv_wrt_params()
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
        raise ValueError(("TPInstrumentOp.to_vector() should never be called"
                          " - use TPInstrument.to_vector() instead"))

    
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
        # having been simplified and now being initialized from a vector (within a
        # calculator).  We rely on the Instrument elements having their
        # from_vector() methods called in self.index order.
        
        if self.index < len(self.param_ops)-1: #final element doesn't need to init any param gates
            for i in self.dependents: #re-init all my dependents (may be redundant)
                if i == 0 and self.index > 0: continue # 0th param-gate already init by index==0 element
                paramop_local_inds = _modelmember._decompose_gpindices(
                    self.gpindices, self.param_ops[i].gpindices)
                self.param_ops[i].from_vector( v[paramop_local_inds] )
                
        self._construct_matrix()


class ComposedOp(LinearOperator):
    """
    A gate map that is the composition of a number of map-like factors (possibly
    other `LinearOperator`s)
    """
    
    def __init__(self, ops_to_compose, dim="auto", evotype="auto"):
        """
        Creates a new ComposedOp.

        Parameters
        ----------
        ops_to_compose : list
            List of `LinearOperator`-derived objects
            that are composed to form this gate map.  Elements are composed
            with vectors  in  *left-to-right* ordering, maintaining the same
            convention as operation sequences in pyGSTi.  Note that this is
            *opposite* from standard matrix multiplication order.
            
        dim : int or "auto"
            Dimension of this operation.  Can be set to `"auto"` to take dimension
            from `ops_to_compose[0]` *if* there's at least one gate being
            composed.

        evotype : {"densitymx","statevec","stabilizer","svterm","cterm","auto"}
            The evolution type of this operation.  Can be set to `"auto"` to take
            the evolution type of `ops_to_compose[0]` *if* there's at least
            one gate being composed.
        """
        assert(len(ops_to_compose) > 0 or dim != "auto"), \
            "Must compose at least one gate when dim='auto'!"
        self.factorops = list(ops_to_compose)
        
        if dim == "auto":
            dim = ops_to_compose[0].dim
        assert(all([dim == gate.dim for gate in ops_to_compose])), \
            "All gates must have the same dimension (%d expected)!" % dim

        if evotype == "auto":
            evotype = ops_to_compose[0]._evotype
        assert(all([evotype == gate._evotype for gate in ops_to_compose])), \
            "All gates must have the same evolution type (%s expected)!" % evotype
        
        LinearOperator.__init__(self, dim, evotype)

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.
        
        Returns
        -------
        list
        """
        return self.factorops

    def append(*factorops_to_add):
        """
        Add one or more factors to this operator.

        Parameters
        ----------
        *factors_to_add : LinearOperator
            One or multiple factor operators to add on at the *end* (evaluated
            last) of this operator.

        Returns
        -------
        None
        """
        self.factorops.extend(factorops_to_add)
        if self.parent: #need to alert parent that *number* (not just value)
            parent._mark_for_rebuild(self) #  of our params may have changed

    def remove(*factorop_indices):
        """
        Remove one or more factors from this operator.

        Parameters
        ----------
        *factorop_indices : int
            One or multiple factor indices to remove from this operator.

        Returns
        -------
        None
        """
        for i in sorted(factorop_indices, reverse=True):
            del self.factorops[i]
        if self.parent: #need to alert parent that *number* (not just value)
            parent._mark_for_rebuild(self) #  of our params may have changed

    def copy(self, parent=None):
        """
        Copy this object.

        Returns
        -------
        LinearOperator
            A copy of this object.
        """
        # We need to override this method so that factor gates have their 
        # parent reset correctly.
        cls = self.__class__ # so that this method works for derived classes too
        copyOfMe = cls([ g.copy(parent) for g in self.factorops ], self.dim, self._evotype)
        return self._copy_gpindices(copyOfMe, parent)


    def tosparse(self):
        """ Return the operation as a sparse matrix """
        mx = self.factorops[0].tosparse()
        for gate in self.factorops[1:]:
            mx = gate.tosparse().dot(mx)
        return mx

    def todense(self):
        """
        Return this operation as a dense matrix.
        """
        mx = self.factorops[0].todense()
        for gate in self.factorops[1:]:
            mx = _np.dot(gate.todense(),mx)
        return mx

    def torep(self):
        """
        Return a "representation" object for this gate.

        Such objects are primarily used internally by pyGSTi to compute
        things like probabilities more efficiently.

        Returns
        -------
        OpRep
        """
        factor_op_reps = [ gate.torep() for gate in self.factorops ]
        #FUTURE? factor_op_reps = [ repmemo.get(id(gate), gate.torep(debug_time_dict)) for gate in self.factorops ] #something like this?

        if self._evotype == "densitymx":
            return replib.DMOpRep_Composed(factor_op_reps, self.dim)
        elif self._evotype == "statevec":
            return replib.SVOpRep_Composed(factor_op_reps, self.dim)
        elif self._evotype == "stabilizer":
            nQubits = int(round(_np.log2(self.dim))) # "stabilizer" is a unitary-evolution type mode
            return replib.SBOpRep_Composed(factor_op_reps, nQubits)
        
        assert(False), "Invalid internal _evotype: %s" % self._evotype


    def get_order_terms(self, order):
        """ 
        Get the `order`-th order Taylor-expansion terms of this operation.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that its action on a
        density matrix `rho` can be written:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the gate's
        parameters, where the polynomial's variable indices index the *global*
        parameters of the gate's parent (usually a :class:`Model`), not the 
        gate's local parameter array (i.e. that returned from `to_vector`).


        Parameters
        ----------
        order : int
            The order of terms to get.

        Returns
        -------
        list
            A list of :class:`RankOneTerm` objects.
        """
        terms = []
        for p in _lt.partition_into(order, len(self.factorops)):
            factor_lists = [ self.factorops[i].get_order_terms(pi) for i,pi in enumerate(p) ]
            for factors in _itertools.product(*factor_lists):
                terms.append( _term.compose_terms(factors) )
        return terms # Cache terms in FUTURE?

    
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
        assert(self.gpindices is not None),"Must set a ComposedOp's .gpindices before calling to_vector"
        v = _np.empty(self.num_params(), 'd')
        for gate in self.factorops:
            factorgate_local_inds = _modelmember._decompose_gpindices(
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
        assert(self.gpindices is not None),"Must set a ComposedOp's .gpindices before calling from_vector"
        for gate in self.factorops:
            factorgate_local_inds = _modelmember._decompose_gpindices(
                    self.gpindices, gate.gpindices)
            gate.from_vector( v[factorgate_local_inds] )
        self.dirty = True


    def transform(self, S):
        """
        Update operation matrix G with inv(S) * G * S,

        Generally, the transform function updates the *parameters* of 
        the gate such that the resulting operation matrix is altered as 
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
        for gate in self.factorops:
            gate.transform(S)


    def __str__(self):
        """ Return string representation """
        s = "Composed gate of %d factors:\n" % len(self.factorops)
        for i,gate in enumerate(self.factorops):
            s += "Factor %d:\n" % i
            s += str(gate)
        return s


class ExponentiatedOp(ComposedOp):
    """
    A gate map that is the composition of a number of map-like factors (possibly
    other `LinearOperator`s)
    """
    
    def __init__(self, op_to_exponentiate, power, evotype="auto"):
        """
        TODO: docstring
        Creates a new ComposedOp.

        Parameters
        ----------
        ops_to_compose : list
            List of `LinearOperator`-derived objects
            that are composed to form this gate map.  Elements are composed
            with vectors  in  *left-to-right* ordering, maintaining the same
            convention as operation sequences in pyGSTi.  Note that this is
            *opposite* from standard matrix multiplication order.
            
        dim : int or "auto"
            Dimension of this operation.  Can be set to `"auto"` to take dimension
            from `ops_to_compose[0]` *if* there's at least one gate being
            composed.

        evotype : {"densitymx","statevec","stabilizer","svterm","cterm","auto"}
            The evolution type of this operation.  Can be set to `"auto"` to take
            the evolution type of `ops_to_compose[0]` *if* there's at least
            one gate being composed.
        """
        #We may not actually need to save these, since they can be inferred easily
        self.exponentiated_op = op_to_exponentiate
        self.power = power

        dim = op_to_exponentiate.dim

        if evotype == "auto":
            evotype = op_to_exponentiate._evotype
        
        ComposedOp.__init__(self, [self.exponentiated_op]*power, dim, evotype)

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.
        
        Returns
        -------
        list
        """
        return [ self.exponentiated_op ]

    def copy(self, parent=None):
        """
        Copy this object.

        Returns
        -------
        LinearOperator
            A copy of this object.
        """
        # We need to override this method so that factor gates have their 
        # parent reset correctly.
        cls = self.__class__ # so that this method works for derived classes too
        copyOfMe = cls(self.exponentiated_op.copy(parent), self.power, self._evotype)
        return self._copy_gpindices(copyOfMe, parent)


    def torep(self):
        """
        Return a "representation" object for this gate.

        Such objects are primarily used internally by pyGSTi to compute
        things like probabilities more efficiently.

        Returns
        -------
        OpRep
        """
        if self._evotype == "densitymx":
            return replib.DMOpRep_Exponentiated(self.exponentiated_op.torep(), self.power, self.dim)
        elif self._evotype == "statevec":
            return replib.SVOpRep_Exponentiated(self.exponentiated_op.torep(), self.power, self.dim)
        elif self._evotype == "stabilizer":
            nQubits = int(round(_np.log2(self.dim))) # "stabilizer" is a unitary-evolution type mode
            return replib.SVOpRep_Exponentiated(self.exponentiated_op.torep(), self.power, nQubits)
        assert(False), "Invalid internal _evotype: %s" % self._evotype


    def to_vector(self):
        """
        Get the gate parameters as an array of values.

        Returns
        -------
        numpy array
            The gate parameters as a 1D array with length num_params().
        """
        return self.exponentiated_op.to_vector()


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
        self.exponentiated_op.from_vector(v)
        self.dirty = True


    def __str__(self):
        """ Return string representation """
        s = "Exponentiated gate that raise the below op to the %d power\n" % self.power
        s += str(self.exponentiated_op)
        return s

    
class ComposedDenseOp(ComposedOp,DenseOperator):
    """
    A gate that is the composition of a number of matrix factors (possibly other gates).
    """
    
    def __init__(self, ops_to_compose, dim="auto", evotype="auto"):
        """
        Creates a new ComposedDenseOp.

        Parameters
        ----------
        ops_to_compose : list
            A list of 2D numpy arrays (matrices) and/or `DenseOperator`-derived
            objects that are composed to form this gate.  Elements are composed
            with vectors  in  *left-to-right* ordering, maintaining the same
            convention as operation sequences in pyGSTi.  Note that this is
            *opposite* from standard matrix multiplication order.

        dim : int or "auto"
            Dimension of this operation.  Can be set to `"auto"` to take dimension
            from `ops_to_compose[0]` *if* there's at least one gate being
            composed.

        evotype : {"densitymx","statevec","stabilizer","svterm","cterm","auto"}
            The evolution type of this operation.  Can be set to `"auto"` to take
            the evolution type of `ops_to_compose[0]` *if* there's at least
            one gate being composed.
        """
        ComposedOp.__init__(self, ops_to_compose, dim, evotype) #sets self.dim & self._evotype
        DenseOperator.__init__(self, _np.identity(self.dim), self._evotype) #type doesn't matter here - just a dummy
        self._construct_matrix()


    def _construct_matrix(self):
        self.base = self.todense()
        assert(self.base.shape == (self.dim,self.dim))
        self.base.flags.writeable = False

    def torep(self):
        """
        Return a "representation" object for this gate.

        Such objects are primarily used internally by pyGSTi to compute
        things like probabilities more efficiently.

        Returns
        -------
        OpRep
        """
        # implement this so we're sure to use DenseOperator version
        return DenseOperator.torep(self)

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
        ComposedOp.from_vector(self, v)
        self._construct_matrix()


    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single gate parameter.  Thus, each column is of length
        op_dim^2 and there is one column per gate parameter.

        Returns
        -------
        numpy array
            Array of derivatives with shape (dimension^2, num_params)
        """
        typ = complex if any([_np.iscomplexobj(gate) for gate in self.factorops]) else 'd'
        derivMx = _np.zeros( (self.dim,self.dim, self.num_params()), typ)
        
        #Product rule to compute jacobian
        for i,gate in enumerate(self.factorops): # loop over the gate we differentiate wrt
            if gate.num_params() == 0: continue #no contribution
            deriv = gate.deriv_wrt_params(None) #TODO: use filter?? / make relative to this gate...
            deriv.shape = (self.dim,self.dim,gate.num_params())

            if i > 0: # factors before ith
                pre = self.factorops[0]
                for gateA in self.factorops[1:i]:
                    pre = _np.dot(gateA,pre)
                #deriv = _np.einsum("ija,jk->ika", deriv, pre )
                deriv = _np.transpose(_np.tensordot(deriv, pre, (1,0)),(0,2,1))

            if i+1 < len(self.factorops): # factors after ith
                post = self.factorops[i+1]
                for gateA in self.factorops[i+2:]:
                    post = _np.dot(gateA,post)
                #deriv = _np.einsum("ij,jka->ika", post, deriv )
                deriv = _np.tensordot(post, deriv, (1,0))

            factorgate_local_inds = _modelmember._decompose_gpindices(
                self.gpindices, gate.gpindices)
            derivMx[:,:,factorgate_local_inds] += deriv

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
        return any([gate.has_nonzero_hessian() for gate in self.factorops])
            


class EmbeddedOp(LinearOperator):
    """
    A gate map containing a single lower (or equal) dimensional gate within it.
    An EmbeddedOp acts as the identity on all of its domain except the 
    subspace of its contained gate, where it acts as the contained gate does.
    """
    
    def __init__(self, stateSpaceLabels, targetLabels, gate_to_embed):
        """
        Initialize an EmbeddedOp object.

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

        gate_to_embed : LinearOperator
            The gate object that is to be contained within this gate, and
            that specifies the only non-trivial action of the EmbeddedOp.
        """
        from .labeldicts import StateSpaceLabels as _StateSpaceLabels
        self.state_space_labels = _StateSpaceLabels(stateSpaceLabels,
                                                    evotype=gate_to_embed._evotype)
        self.targetLabels = targetLabels
        self.embedded_op = gate_to_embed

        labels = targetLabels

        evotype = gate_to_embed._evotype
        if evotype in ("densitymx","statevec"):
            iTensorProdBlks = [ self.state_space_labels.tpb_index[label] for label in labels ]
              # index of tensor product block (of state space) a bit label is part of
            if len(set(iTensorProdBlks)) != 1:
                raise ValueError("All qubit labels of a multi-qubit gate must correspond to the" + \
                                 " same tensor-product-block of the state space -- checked previously") # pragma: no cover
        
            iTensorProdBlk = iTensorProdBlks[0] #because they're all the same (tested above)
            tensorProdBlkLabels = self.state_space_labels.labels[iTensorProdBlk]
            basisInds = [] # list of possible *density-matrix-space* indices of each component of the tensor product block
            for l in tensorProdBlkLabels:
                basisInds.append( list(range(self.state_space_labels.labeldims[l])) )
                  # e.g. [0,1,2,3] for densitymx qubits (I, X, Y, Z) OR [0,1] for statevec qubits (std *complex* basis)

            self.numBasisEls = _np.array(list(map(len,basisInds)),_np.int64)
            self.iTensorProdBlk = iTensorProdBlk #save which block is "active" one

            #offset into "active" tensor product block
            blockDims = self.state_space_labels.tpb_dims
            self.offset = sum( [ blockDims[i] for i in range(0,iTensorProdBlk) ] ) #number of basis elements preceding our block's elements
    
            divisor = 1
            self.divisors = []
            for l in labels:
                self.divisors.append(divisor)
                divisor *= self.state_space_labels.labeldims[l] # e.g. 4 or 2 for qubits (depending on evotype)
    
            # multipliers to go from per-label indices to tensor-product-block index
            # e.g. if map(len,basisInds) == [1,4,4] then multipliers == [ 16 4 1 ]
            self.multipliers = _np.array( _np.flipud( _np.cumprod([1] + list(
                reversed(list(map(len,basisInds[1:]))))) ), _np.int64)
    
            # Separate the components of the tensor product that are not operated on, i.e. that our final map just acts as identity w.r.t.
            labelIndices = [ tensorProdBlkLabels.index(label) for label in labels ]
            self.actionInds = _np.array(labelIndices,_np.int64)
            assert(_np.product([self.numBasisEls[i] for i in self.actionInds]) == self.embedded_op.dim), \
                "Embedded gate has dimension (%d) inconsistent with the given target labels (%s)" % (self.embedded_op.dim, str(labels))

            basisInds_noop = basisInds[:]
            basisInds_noop_blankaction = basisInds[:]
            for labelIndex in sorted(labelIndices,reverse=True):
                del basisInds_noop[labelIndex]
                basisInds_noop_blankaction[labelIndex] = [0]
            self.basisInds_noop = basisInds_noop

            self.sorted_bili = sorted( list(enumerate(labelIndices)), key=lambda x: x[1])
              # for inserting target-qubit basis indices into list of noop-qubit indices
            
        else: # evotype in ("stabilizer","svterm","cterm"):
            # term-mode doesn't need any of the following members
            self.offset = None
            self.multipliers = None
            self.divisors = None
            
            self.sorted_bili = None
            self.iTensorProdBlk = None
            self.numBasisEls = None
            self.basisInds_noop = None

        if evotype == "stabilizer":
            # assert that all state space labels == qubits, since we only know
            # how to embed cliffords on qubits...
            assert(len(self.state_space_labels.labels) == 1 and
                   all([ld == 2 for ld in self.state_space_labels.labeldims.values()])), \
                   "All state space labels must correspond to *qubits*"
            if isinstance(self.embedded_op, CliffordOp):
                assert(len(targetLabels) == len(self.embedded_op.svector) // 2), \
                    "Inconsistent number of qubits in `targetLabels` and Clifford `embedded_op`"

            #Cache info to speedup representation's acton(...) methods:
            # Note: ...labels[0] is the *only* tensor-prod-block, asserted above
            qubitLabels = self.state_space_labels.labels[0] 
            self.qubit_indices =  _np.array([ qubitLabels.index(targetLbl)
                                              for targetLbl in self.targetLabels ], _np.int64)
        else:
            self.qubit_indices = None # (unused)

        opDim = self.state_space_labels.dim
        LinearOperator.__init__(self, opDim, evotype)
        

    def __getstate__(self):
        # Don't pickle 'instancemethod' or parent (see modelmember implementation)
        return _modelmember.ModelMember.__getstate__(self)
    
    def __setstate__(self, d):
        if "dirty" in d: # backward compat: .dirty was replaced with ._dirty in ModelMember
            d['_dirty'] = d['dirty']; del d['dirty']
        self.__dict__.update(d)


    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.
        
        Returns
        -------
        list
        """
        return [self.embedded_op]


    def copy(self, parent=None):
        """
        Copy this object.

        Returns
        -------
        LinearOperator
            A copy of this object.
        """
        # We need to override this method so that embedded gate has its
        # parent reset correctly.
        cls = self.__class__ # so that this method works for derived classes too
        copyOfMe = cls(self.state_space_labels, self.targetLabels,
                       self.embedded_op.copy(parent))
        return self._copy_gpindices(copyOfMe, parent)
        
        
    def _decomp_op_index(self, indx):
        """ Decompose index of a Pauli-product matrix into indices of each
            Pauli in the product """
        ret = []
        for d in reversed(self.divisors):
            ret.append( indx // d )
            indx = indx % d
        return ret

    
    def _merge_op_and_noop_bases(self, op_b, noop_b):
        """
        Merge the Pauli basis indices for the "gate"-parts of the total
        basis contained in op_b (i.e. of the components of the tensor
        product space that are operated on) and the "noop"-parts contained
        in noop_b.  Thus, len(op_b) + len(noop_b) == len(basisInds), and
        this function merges together basis indices for the operated-on and
        not-operated-on tensor product components.
        Note: return value always have length == len(basisInds) == number
        of components
        """
        ret = list(noop_b[:])    #start with noop part...
        for bi,li in self.sorted_bili:
            ret.insert(li, op_b[bi]) #... and insert gate parts at proper points
        return ret

    
    def _iter_matrix_elements(self, relToBlock=False):
        """ Iterates of (op_i,op_j,embedded_op_i,embedded_op_j) tuples giving mapping
            between nonzero elements of operation matrix and elements of the embedded gate matrx """

        #DEPRECATED REP - move some __init__ constructed vars to here?

        offset = 0 if relToBlock else self.offset
        for op_i in range(self.embedded_op.dim):     # rows ~ "output" of the gate map
            for op_j in range(self.embedded_op.dim): # cols ~ "input"  of the gate map
                op_b1 = self._decomp_op_index(op_i) # op_b? are lists of dm basis indices, one index per
                op_b2 = self._decomp_op_index(op_j) #  tensor product component that the gate operates on (2 components for a 2-qubit gate)
    
                for b_noop in _itertools.product(*self.basisInds_noop): #loop over all state configurations we don't operate on
                                                                   # - so really a loop over diagonal dm elements
                    b_out = self._merge_op_and_noop_bases(op_b1, b_noop)  # using same b_noop for in and out says we're acting
                    b_in  = self._merge_op_and_noop_bases(op_b2, b_noop)  #  as the identity on the no-op state space
                    out_vec_index = _np.dot(self.multipliers, tuple(b_out)) # index of output dm basis el within vec(tensor block basis)
                    in_vec_index  = _np.dot(self.multipliers, tuple(b_in))  # index of input dm basis el within vec(tensor block basis)

                    yield (out_vec_index+offset, in_vec_index+offset, op_i, op_j)

    def torep(self):
        """
        Return a "representation" object for this gate.

        Such objects are primarily used internally by pyGSTi to compute
        things like probabilities more efficiently.

        Returns
        -------
        OpRep
        """
        if self._evotype == "stabilizer":
            nQubits = int(round(_np.log2(self.dim)))
            return replib.SBOpRep_Embedded(self.embedded_op.torep(),
                                             nQubits, self.qubit_indices)

        if self._evotype not in ("statevec","densitymx"):
            raise ValueError("Invalid evotype '%s' for %s.torep(...)" %
                             (self._evotype, self.__class__.__name__))

        nBlocks = self.state_space_labels.num_tensor_prod_blocks()
        iActiveBlock = self.iTensorProdBlk
        nComponents = len(self.state_space_labels.labels[iActiveBlock])
        embeddedDim = self.embedded_op.dim
        blocksizes = _np.array([ _np.product(self.state_space_labels.tensor_product_block_dims(k)) for k in range(nBlocks)], _np.int64)
        
        if self._evotype == "statevec":
            return replib.SVOpRep_Embedded(self.embedded_op.torep(),
                                           self.numBasisEls, self.actionInds, blocksizes,
                                           embeddedDim, nComponents, iActiveBlock, nBlocks,
                                           self.dim)
        else:
            return replib.DMOpRep_Embedded(self.embedded_op.torep(),
                                           self.numBasisEls, self.actionInds, blocksizes,
                                           embeddedDim, nComponents, iActiveBlock, nBlocks,
                                           self.dim)

    def tosparse(self):
        """ Return the operation as a sparse matrix """
        embedded_sparse = self.embedded_op.tosparse().tolil()
        finalOp = _sps.identity( self.dim, embedded_sparse.dtype, format='lil' )

        #fill in embedded_op contributions (always overwrites the diagonal
        # of finalOp where appropriate, so OK it starts as identity)
        for i,j,gi,gj in self._iter_matrix_elements():
            finalOp[i,j] = embedded_sparse[gi,gj]
        return finalOp.tocsr()

    
    def todense(self):
        """ Return the operation as a dense matrix """
        
        #FUTURE: maybe here or in a new "tosymplectic" method, could
        # create an embeded clifford symplectic rep as follows (when
        # evotype == "stabilizer"):
        #def tosymplectic(self):
        #    #Embed gate's symplectic rep in larger "full" symplectic rep
        #    #Note: (qubit) labels are in first (and only) tensor-product-block
        #    qubitLabels = self.state_space_labels.labels[0]
        #    smatrix, svector = _symp.embed_clifford(self.embedded_op.smatrix,
        #                                            self.embedded_op.svector,
        #                                            self.qubit_indices,len(qubitLabels))        
        
        embedded_dense = self.embedded_op.todense()
        finalOp = _np.identity( self.dim, embedded_dense.dtype ) # operates on entire state space (direct sum of tensor prod. blocks)

        #fill in embedded_op contributions (always overwrites the diagonal
        # of finalOp where appropriate, so OK it starts as identity)
        for i,j,gi,gj in self._iter_matrix_elements():
            finalOp[i,j] = embedded_dense[gi,gj]
        return finalOp

    
    def get_order_terms(self, order):
        """ 
        Get the `order`-th order Taylor-expansion terms of this operation.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that its action on a
        density matrix `rho` can be written:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the gate's
        parameters, where the polynomial's variable indices index the *global*
        parameters of the gate's parent (usually a :class:`Model`), not the 
        gate's local parameter array (i.e. that returned from `to_vector`).


        Parameters
        ----------
        order : int
            The order of terms to get.

        Returns
        -------
        list
            A list of :class:`RankOneTerm` objects.
        """
        #Reduce labeldims b/c now working on *state-space* instead of density mx:
        sslbls = self.state_space_labels.copy()
        sslbls.reduce_dims_densitymx_to_state()
        return [ _term.embed_term(t, sslbls, self.targetLabels)
                 for t in self.embedded_op.get_order_terms(order) ]

    
    def num_params(self):
        """
        Get the number of independent parameters which specify this gate.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.embedded_op.num_params()


    def to_vector(self):
        """
        Get the gate parameters as an array of values.

        Returns
        -------
        numpy array
            The gate parameters as a 1D array with length num_params().
        """
        return self.embedded_op.to_vector()


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
        self.embedded_op.from_vector(v)
        self.dirty = True


    def transform(self, S):
        """
        Update operation matrix G with inv(S) * G * S,

        Generally, the transform function updates the *parameters* of 
        the gate such that the resulting operation matrix is altered as 
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
        # I think we could do this but extracting the approprate parts of the
        # S and Sinv matrices... but haven't needed it yet.
        raise NotImplementedError("Cannot transform an EmbeddedDenseOp yet...")

    
    def depolarize(self, amount):
        """
        Depolarize this gate by the given `amount`.

        Generally, the depolarize function updates the *parameters* of 
        the gate such that the resulting operation matrix is depolarized.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. In standard
            bases, depolarization corresponds to multiplying the operation matrix
            by a diagonal matrix whose first diagonal element (corresponding
            to the identity) equals 1.0 and whose subsequent elements 
            (corresponding to non-identity basis elements) equal
            `1.0 - amount[i]` (or just `1.0 - amount` if `amount` is a
            float).

        Returns
        -------
        None
        """
        self.embedded_op.depolarize(amount)


    def rotate(self, amount, mxBasis="gm"):
        """
        Rotate this gate by the given `amount`.

        Generally, the rotate function updates the *parameters* of 
        the gate such that the resulting operation matrix is rotated.  If
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
        self.embedded_op.rotate(amount, mxBasis)


    #def compose(self, otherOp):
    #    """
    #    Create and return a new gate that is the composition of this gate
    #    followed by otherOp, which *must be another EmbeddedDenseOp*.
    #    (For more general compositions between different types of gates, use
    #    the module-level compose function.)  The returned gate's matrix is
    #    equal to dot(this, otherOp).
    #
    #    Parameters
    #    ----------
    #    otherOp : EmbeddedDenseOp
    #        The gate to compose to the right of this one.
    #
    #    Returns
    #    -------
    #    EmbeddedDenseOp
    #    """
    #    raise NotImplementedError("Can't compose an EmbeddedDenseOp yet")

    def has_nonzero_hessian(self):
        """ 
        Returns whether this gate has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return self.embedded_op.has_nonzero_hessian()


    def __str__(self):
        """ Return string representation """
        s = "Embedded gate with full dimension %d and state space %s\n" % (self.dim,self.state_space_labels)
        s += " that embeds the following %d-dimensional gate into acting on the %s space\n" \
             % (self.embedded_op.dim, str(self.targetLabels))
        s += str(self.embedded_op)
        return s

    

class EmbeddedDenseOp(EmbeddedOp, DenseOperator):
    """
    A gate containing a single lower (or equal) dimensional gate within it.
    An EmbeddedDenseOp acts as the identity on all of its domain except the 
    subspace of its contained gate, where it acts as the contained gate does.
    """
    def __init__(self, stateSpaceLabels, targetLabels, gate_to_embed):
        """
        Initialize a EmbeddedDenseOp object.

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

        gate_to_embed : DenseOperator
            The gate object that is to be contained within this gate, and
            that specifies the only non-trivial action of the EmbeddedDenseOp.
        """
        EmbeddedOp.__init__(self, stateSpaceLabels, targetLabels,
                                 gate_to_embed) # sets self.dim & self._evotype
        DenseOperator.__init__(self, _np.identity(self.dim), self._evotype) # type irrelevant - just a dummy
        self._construct_matrix()

    def _construct_matrix(self):
        self.base = self.todense()
        assert(self.base.shape == (self.dim,self.dim))
        self.base.flags.writeable = False

    def torep(self):
        """
        Return a "representation" object for this gate.

        Such objects are primarily used internally by pyGSTi to compute
        things like probabilities more efficiently.

        Returns
        -------
        OpRep
        """
        # implement this so we're sure to use DenseOperator version
        return DenseOperator.torep(self)


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
        EmbeddedOp.from_vector(self, v)
        self._construct_matrix()
        self.dirty = True


    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single gate parameter.  Thus, each column is of length
        op_dim^2 and there is one column per gate parameter.

        Returns
        -------
        numpy array
            Array of derivatives with shape (dimension^2, num_params)
        """
        # Note: this function exploits knowledge of EmbeddedOp internals!!
        embedded_deriv = self.embedded_op.deriv_wrt_params(wrtFilter)
        derivMx = _np.zeros((self.dim**2,self.num_params()),embedded_deriv.dtype)
        M = self.embedded_op.dim

        #fill in embedded_op contributions (always overwrites the diagonal
        # of finalOp where appropriate, so OK it starts as identity)
        for i,j,gi,gj in self._iter_matrix_elements():
            derivMx[i*self.dim+j,:] = embedded_deriv[gi*M+gj,:] #fill row of jacobian

        if wrtFilter is None:
            return derivMx
        else:
            return _np.take( derivMx, wrtFilter, axis=1 )


    def depolarize(self, amount):
        """
        Depolarize this gate by the given `amount`.

        See :method:`EmbeddedOp.depolarize`.
        """
        EmbeddedOp.depolarize(self, amount)
        self._construct_matrix()


    def rotate(self, amount, mxBasis="gm"):
        """
        Rotate this gate by the given `amount`.

        See :method:`EmbeddedOp.rotate`.
        """
        EmbeddedOp.rotate(self, amount, mxBasis)
        self._construct_matrix()


class CliffordOp(LinearOperator):
    """
    A Clifford gate, represented via a symplectic
    """
    
    def __init__(self, unitary, symplecticrep=None):
        """
        Creates a new CliffordOp from a unitary operation.

        Note: while the clifford gate is held internally in a symplectic
        representation, it is also be stored as a unitary (so the `unitary`
        argument is required) for keeping track of global phases when updating
        stabilizer frames.

        If a non-Clifford unitary is specified, then a ValueError is raised.

        Parameters
        ----------
        unitary : numpy.ndarray
            The unitary action of the clifford gate.

        symplecticrep : tuple, optional
            A (symplectic matrix, phase vector) 2-tuple specifying the pre-
            computed symplectic representation of `unitary`.  If None, then
            this representation is computed automatically from `unitary`.

        """
        #self.superop = superop
        self.unitary = unitary
        assert(self.unitary is not None),"Must supply `unitary` argument!"
        
        #if self.superop is not None:
        #    assert(unitary is None and symplecticrep is None),"Only supply one argument to __init__"
        #    raise NotImplementedError("Superop -> Unitary calc not implemented yet")

        if symplecticrep is not None:
            self.smatrix, self.svector = symplecticrep
        else:
            # compute symplectic rep from unitary
            self.smatrix, self.svector = _symp.unitary_to_symplectic(self.unitary, flagnonclifford=True)

        #Cached upon first usage
        self.inv_smatrix = None
        self.inv_svector = None

        nQubits = len(self.svector) // 2
        dim = 2**nQubits # "stabilizer" is a "unitary evolution"-type mode
        LinearOperator.__init__(self, dim, "stabilizer")

        
    #NOTE: if this gate had parameters, we'd need to clear inv_smatrix & inv_svector
    # whenever the smatrix or svector changed, respectively (probably in from_vector?)

    def torep(self):
        """
        Return a "representation" object for this gate.

        Such objects are primarily used internally by pyGSTi to compute
        things like probabilities more efficiently.

        Returns
        -------
        OpRep
        """
        if self.inv_smatrix is None or self.inv_svector is None:
            self.inv_smatrix, self.inv_svector = _symp.inverse_clifford(
                self.smatrix, self.svector) #cache inverse since it's expensive

        invs, invp = self.inv_smatrix, self.inv_svector
        U = self.unitary.todense() if isinstance(self.unitary, LinearOperator) else self.unitary
        return replib.SBOpRep_Clifford(_np.ascontiguousarray(self.smatrix,_np.int64),
                                         _np.ascontiguousarray(self.svector,_np.int64),
                                         _np.ascontiguousarray(invs,_np.int64),
                                         _np.ascontiguousarray(invp,_np.int64),
                                         _np.ascontiguousarray(U,complex))

    def __str__(self):
        """ Return string representation """
        s = "Clifford gate with matrix:\n"
        s += _mt.mx_to_string(self.smatrix, width=2, prec=0)
        s += " and vector " + _mt.mx_to_string(self.svector, width=2, prec=0)
        return s


# STRATEGY:
# - maybe create an abstract base TermGate class w/get_order_terms(...) function?
# - Note: if/when terms return a *polynomial* coefficient the poly's 'variables' should
#    reference the *global* Model-level parameters, not just the local gate ones.
# - create an EmbeddedTermGate class to handle embeddings, which holds a
#    LindbladDenseOp (or other in the future?) and essentially wraps it's
#    terms in EmbeddedOp or EmbeddedClifford objects.
# - similarly create an ComposedTermGate class...
# - so LindbladDenseOp doesn't need to deal w/"kite-structure" bases of terms;
#    leave this to some higher level constructor which can create compositions
#    of multiple LindbladOps based on kite structure (one per kite block).



class ComposedErrorgen(LinearOperator):
    """ 
    A composition (sum!) of several Lindbladian exponent operators, that is, a
    *sum* (not product) of other error generators.
    """
    
    def __init__(self, errgens_to_compose, dim="auto", evotype="auto"):
        """
        Creates a new ComposedErrorgen.

        Parameters
        ----------
        errgens_to_compose : list
            List of `LinearOperator`-derived objects that are summed together (composed)
            to form this error generator. 
            
        dim : int or "auto"
            Dimension of this error generator.  Can be set to `"auto"` to take
            the dimension from `errgens_to_compose[0]` *if* there's at least one
            error generator being composed.

        evotype : {"densitymx","statevec","stabilizer","svterm","cterm","auto"}
            The evolution type of this error generator.  Can be set to `"auto"`
            to take the evolution type of `errgens_to_compose[0]` *if* there's
            at least one error generator being composed.
        """
        assert(len(errgens_to_compose) > 0 or dim != "auto"), \
            "Must compose at least one error generator when dim='auto'!"
        self.factors = errgens_to_compose

        if dim == "auto":
            dim = errgens_to_compose[0].dim
        assert(all([dim == eg.dim for eg in errgens_to_compose])), \
            "All error generators must have the same dimension (%d expected)!" % dim

        if evotype == "auto":
            evotype = errgens_to_compose[0]._evotype
        assert(all([evotype == eg._evotype for eg in errgens_to_compose])), \
            "All error generators must have the same evolution type (%s expected)!" % evotype

        # set "API" error-generator members (to interface properly w/other objects)        
        # FUTURE: create a base class that defines this interface (maybe w/properties?)
        #self.sparse = errgens_to_compose[0].sparse \
        #    if len(errgens_to_compose) > 0 else False
        #assert(all([self.sparse == eg.sparse for eg in errgens_to_compose])), \
        #    "All error generators must have the same sparsity (%s expected)!" % self.sparse

        self.matrix_basis = errgens_to_compose[0].matrix_basis \
            if len(errgens_to_compose) > 0 else None
        assert(all([self.matrix_basis == eg.matrix_basis for eg in errgens_to_compose])), \
            "All error generators must have the same matrix basis (%s expected)!" % self.matrix_basis

        #OLD: when a generators "rep" was self.err_gen_mx
        #if evotype == "densitymx":
        #    self._construct_errgen_matrix()        
        #else:
        #    self.err_gen_mx = None
        
        LinearOperator.__init__(self, dim, evotype)


    def get_coeffs(self):
        """
        Constructs a dictionary of the Lindblad-error-generator coefficients 
        (i.e. the "error rates") of this error generator.  Note that these are
        not necessarily the parameter values, as these coefficients are
        generally functions of the parameters (so as to keep the coefficients
        positive, for instance).

        Returns
        -------
        Ltermdict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Basis labels are integers starting at 0.  Values are complex
            coefficients (error rates).
    
        basisdict : dict
            A dictionary mapping the integer basis labels used in the
            keys of `Ltermdict` to basis matrices..
        """
        Ltermdict = {}; basisdict = {}; next_available = 0
        for eg in self.factors:
            ltdict, bdict = eg.get_coeffs()

            # see if we need to update basisdict to avoid collisions
            # and avoid duplicating basis elements
            final_basisLbls = {}
            for lbl,basisEl in bdict.items():
                lblsEqual = bool(lbl == existingLbl)
                for existing_lbl,existing_basisEl in basisdict.values():
                    if _mt.safenorm(basisEl-existing_basisEl) < 1e-6:
                        final_basisLbls[lbl] = existingLbl
                        break
                else: # no existing basis element found - need a new element
                    if lbl in basisdict: # then can't keep current label
                        final_basisLbls[lbl] = next_available
                        next_available += 1
                    else:
                        #fine to keep lbl since it's unused
                        final_basisLbls[lbl] = lbl
                        if isinstance(lbl,int):
                            next_available = max(next_available,lbl+1)

                    #Add new basis element
                    basisdict[ final_basisLbls[lbl] ] = basisEl
                    
            for key, coeff in ltdict:
                new_key = tuple( [key[0]] + [ final_basisLbls[l] for l in key[1:] ] )
                if new_key in Ltermdict:
                    Ltermdict[new_key] += coeff
                else:
                    Ltermdict[new_key] = coeff

        return Ltermdict, basisdict


    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the vectorized derivatives of the
        flattened error generator matrix with respect to a single operator
        parameter.  Thus, each column is of length op_dim^2 and there is one
        column per gate parameter.

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension^2, num_params)
        """
        #TODO: in the furture could do this more cleverly so 
        # each factor gets an appropriate wrtFilter instead of
        # doing all filtering at the end
        
        d2 = self.dim
        derivMx = _np.zeros( (d2**2,self.num_params()), 'd')
        for eg in self.factors:
            factor_deriv = eg.deriv_wrt_params(None) # do filtering at end
            rel_gpindices = _modelmember._decompose_gpindices(
                self.gpindices, eg.gpindices)
            derivMx[:,rel_gpindices] += factor_deriv[:,:]

        if wrtFilter is None:
            return derivMx
        else:
            return _np.take( derivMx, wrtFilter, axis=1 )

        return derivMx
        

    def hessian_wrt_params(self, wrtFilter1=None, wrtFilter2=None):
        """
        Construct the Hessian of this error generator with respect to
        its parameters.

        This function returns a tensor whose first axis corresponds to the
        flattened operation matrix and whose 2nd and 3rd axes correspond to the
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
        #TODO: in the furture could do this more cleverly so 
        # each factor gets an appropriate wrtFilter instead of
        # doing all filtering at the end
        
        d2 = self.dim
        nP = self.num_params()
        hessianMx = _np.zeros( (d2**2,nP,nP), 'd')
        for eg in self.factors:
            factor_hessian = eg.hessian_wrt_params(None,None) # do filtering at end
            rel_gpindices = _modelmember._decompose_gpindices(
                self.gpindices, eg.gpindices)
            hessianMx[:,rel_gpindices,rel_gpindices] += factor_hessian[:,:,:]

        if wrtFilter1 is None:
            if wrtFilter2 is None:
                return hessianMx
            else:
                return _np.take(hessianMx, wrtFilter2, axis=2 )
        else:
            if wrtFilter2 is None:
                return _np.take(hessianMx, wrtFilter1, axis=1 )
            else:
                return _np.take( _np.take(hessianMx, wrtFilter1, axis=1),
                                 wrtFilter2, axis=2 )


    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.
        
        Returns
        -------
        list
        """
        return self.factors

    def append(*factors_to_add):
        """
        Add one or more factors to this operator.

        Parameters
        ----------
        *factors_to_add : LinearOperator
            One or multiple factor operators to add on at the *end* (summed
            last) of this operator.

        Returns
        -------
        None
        """
        self.factors.extend(factors_to_add)
        if self.parent: #need to alert parent that *number* (not just value)
            parent._mark_for_rebuild(self) #  of our params may have changed

    def remove(*factor_indices):
        """
        Remove one or more factors from this operator.

        Parameters
        ----------
        *factorop_indices : int
            One or multiple factor indices to remove from this operator.

        Returns
        -------
        None
        """
        for i in sorted(factor_indices, reverse=True):
            del self.factors[i]
        if self.parent: #need to alert parent that *number* (not just value)
            parent._mark_for_rebuild(self) #  of our params may have changed

    def copy(self, parent=None):
        """
        Copy this object.

        Returns
        -------
        LinearOperator
            A copy of this object.
        """
        # We need to override this method so that factors have their 
        # parent reset correctly.
        cls = self.__class__ # so that this method works for derived classes too
        copyOfMe = cls([ f.copy(parent) for f in self.factors ], self.dim, self._evotype)
        return self._copy_gpindices(copyOfMe, parent)

    def tosparse(self):
        """ Return this error generator as a sparse matrix """
        mx = self.factors[0].tosparse()
        for eg in self.factors[1:]:
            mx += eg.tosparse()
        return mx

    def todense(self):
        """ Return this error generator as a dense matrix """
        mx = self.factors[0].todense()
        for eg in self.factors[1:]:
            mx += eg.todense()
        return mx

    #OLD: UNUSED - now use tosparse/todense
    #def _construct_errgen_matrix(self):
    #    self.factors[0]._construct_errgen_matrix()
    #    mx = self.factors[0].err_gen_mx
    #    for eg in self.factors[1:]:
    #        eg._construct_errgen_matrix()
    #        mx += eg.err_gen_mx
    #    self.err_gen_mx = mx


    def torep(self):
        """
        Return a "representation" object for this error generator.

        Such objects are primarily used internally by pyGSTi to compute
        things like probabilities more efficiently.

        Returns
        -------
        OpRep
        """
        factor_reps = [ factor.torep() for factor in self.factors ]
        if self._evotype == "densitymx":
            return replib.DMOpRep_Sum(factor_reps, self.dim)
        elif self._evotype == "statevec":
            return replib.SVOpRep_Sum(factor_reps, self.dim)
        elif self._evotype == "stabilizer":
            nQubits = int(round(_np.log2(self.dim))) # "stabilizer" is a unitary-evolution type mode
            return replib.SBOpRep_Sum(factor_reps, nQubits)
        
        assert(False), "Invalid internal _evotype: %s" % self._evotype


    def get_order_terms(self, order):
        """ 
        Get the `order`-th order Taylor-expansion terms of this error generator..

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that its action on a
        density matrix `rho` can be written:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the gate's
        parameters, where the polynomial's variable indices index the *global*
        parameters of the gate's parent (usually a :class:`Model`), not the 
        gate's local parameter array (i.e. that returned from `to_vector`).


        Parameters
        ----------
        order : int
            The order of terms to get.

        Returns
        -------
        list
            A list of :class:`RankOneTerm` objects.
        """
        assert(order == 0), "Error generators currently treat all terms as 0-th order; nothing else should be requested!"
        return list(_itertools.chain(*[eg.get_order_terms(order) for eg in self.factors]))

    def num_params(self):
        """
        Get the number of independent parameters which specify this error generator.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return len(self.gpindices_as_array())


    def to_vector(self):
        """
        Get the error generator parameters as an array of values.

        Returns
        -------
        numpy array
            The gate parameters as a 1D array with length num_params().
        """
        assert(self.gpindices is not None),"Must set a ComposedErrorgen's .gpindices before calling to_vector"
        v = _np.empty(self.num_params(), 'd')
        for eg in self.factors:
            factor_local_inds = _modelmember._decompose_gpindices(
                    self.gpindices, eg.gpindices)
            v[factor_local_inds] = eg.to_vector()
        return v


    def from_vector(self, v):
        """
        Initialize the error generator using a vector of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params()

        Returns
        -------
        None
        """
        assert(self.gpindices is not None),"Must set a ComposedErrorgen's .gpindices before calling from_vector"
        for eg in self.factors:
            factor_local_inds = _modelmember._decompose_gpindices(
                    self.gpindices, eg.gpindices)
            eg.from_vector( v[factor_local_inds] )
        self.dirty = True


    def transform(self, S):
        """
        Update operation matrix G with inv(S) * G * S,

        Generally, the transform function updates the *parameters* of 
        the gate such that the resulting operation matrix is altered as 
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
        for eg in self.factors:
            eg.transform(S)


    def __str__(self):
        """ Return string representation """
        s = "Composed error generator of %d factors:\n" % len(self.factors)
        for i,eg in enumerate(self.factors):
            s += "Factor %d:\n" % i
            s += str(eg)
        return s


# Idea: 
# Op = exp(Errgen); Errgen is an error just on 2nd qubit (and say we have 3 qubits)
# so Op = I x (I+eps*A) x I (small eps limit); eps*A is 1-qubit error generator
# also Op ~= I+Errgen in small eps limit, so
# Errgen = I x (I+eps*A) x I - I x I x I 
#        = I x I x I + eps I x A x I - I x I x I
#        = eps I x A x I = I x eps*A x I
# --> we embed error generators by tensoring with I's on non-target sectors.
#  (identical to how be embed gates)

class EmbeddedErrorgen(EmbeddedOp):
    """
    An error generator containing a single lower (or equal) dimensional gate within it.
    An EmbeddedErrorGen acts as the null map (zero) on all of its domain except the 
    subspace of its contained error generator, where it acts as the contained item does.
    """
    
    def __init__(self, stateSpaceLabels, targetLabels, errgen_to_embed):
        """
        Initialize an EmbeddedErrorgen object.

        Parameters
        ----------
        stateSpaceLabels : a list of tuples
            This argument specifies the density matrix space upon which this
            generator acts.  Each tuple corresponds to a block of a density matrix
            in the standard basis (and therefore a component of the direct-sum
            density matrix space). Elements of a tuple are user-defined labels
            beginning with "L" (single Level) or "Q" (two-level; Qubit) which
            interpret the d-dimensional state space corresponding to a d x d
            block as a tensor product between qubit and single level systems.
            (E.g. a 2-qubit space might be labelled `[('Q0','Q1')]`).

        targetLabels : list of strs
            The labels contained in `stateSpaceLabels` which demarcate the
            portions of the state space acted on by `errgen_to_embed` (the
            "contained" error generator).

        errgen_to_embed : LinearOperator
            The error generator object that is to be contained within this
            error generator, and that specifies the only non-trivial action
            of the EmbeddedErrorgen.
        """
        EmbeddedOp.__init__(self, stateSpaceLabels, targetLabels, errgen_to_embed)

        # set "API" error-generator members (to interface properly w/other objects)
        # FUTURE: create a base class that defines this interface (maybe w/properties?)
        #self.sparse = True # Embedded error generators are *always* sparse (pointless to
        #                   # have dense versions of these)
        
        embedded_matrix_basis = errgen_to_embed.matrix_basis
        if _compat.isstr(embedded_matrix_basis):
            self.matrix_basis = embedded_matrix_basis
        else: # assume a Basis object
            my_basis_dim = self.state_space_labels.dim
            self.matrix_basis = _Basis.cast(embedded_matrix_basis.name, my_basis_dim, sparse=True)

            #OLD: constructs a subset of this errorgen's full mxbasis, but not the whole thing:
            #self.matrix_basis = _Basis(
            #    name="embedded_" + embedded_matrix_basis.name,
            #    matrices=[self._embed_basis_mx(mx) for mx in 
            #              embedded_matrix_basis.get_composite_matrices()],
            #    sparse=True)

        #OLD: when a generators "rep" was self.err_gen_mx
        #if errgen_to_embed._evotype == "densitymx":
        #    self._construct_errgen_matrix()        
        #else:
        #    self.err_gen_mx = None

    #def _construct_errgen_matrix(self):
    #    #Always construct a sparse errgen matrix, so just use
    #    # base class's .tosparse() (which calls embedded errorgen's
    #    # .tosparse(), which will convert a dense->sparse embedded
    #    # error generator, but this is fine).
    #    self.err_gen_mx = self.tosparse()

    def _embed_basis_mx(self, mx):
        """ Take a dense or sparse basis matrix and embed it. """
        mxAsGate = StaticDenseOp(mx) if isinstance(mx,_np.ndarray) \
            else StaticDenseOp(mx.todense()) #assume mx is a sparse matrix
        return EmbeddedOp(self.state_space_labels, self.targetLabels,
                               mxAsGate).tosparse() # always convert to *sparse* basis els


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
        EmbeddedOp.from_vector(self, v) 
        self.dirty = True


    def get_coeffs(self):
        """
        Constructs a dictionary of the Lindblad-error-generator coefficients 
        (i.e. the "error rates") of this operation.  Note that these are
        not necessarily the parameter values, as these coefficients are
        generally functions of the parameters (so as to keep the coefficients
        positive, for instance).

        Returns
        -------
        Ltermdict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Basis labels are integers starting at 0.  Values are complex
            coefficients (error rates).
    
        basisdict : dict
            A dictionary mapping the integer basis labels used in the
            keys of `Ltermdict` to basis matrices..
        """
        Ltermdict, basisdict = self.embedded_op.get_coeffs()
        
        #go through basis and embed basis matrices
        new_basisdict = {}
        for lbl, basisEl in basisdict.items():
            new_basisdict[lbl] = self._embed_basis_mx(basisEl)
        
        return Ltermdict, new_basisdict

    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the vectorized derivatives of the
        flattened error generator matrix with respect to a single operator
        parameter.  Thus, each column is of length op_dim^2 and there is one
        column per gate parameter.

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension^2, num_params)
        """
        raise NotImplementedError("deriv_wrt_params is not implemented for EmbeddedErrorGen objects")

    def hessian_wrt_params(self, wrtFilter1=None, wrtFilter2=None):
        """
        Construct the Hessian of this error generator with respect to
        its parameters.

        This function returns a tensor whose first axis corresponds to the
        flattened operation matrix and whose 2nd and 3rd axes correspond to the
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
        raise NotImplementedError("hessian_wrt_params is not implemented for EmbeddedErrorGen objects")

    def __str__(self):
        """ Return string representation """
        s = "Embedded error generator with full dimension %d and state space %s\n" % (self.dim,self.state_space_labels)
        s += " that embeds the following %d-dimensional gate into acting on the %s space\n" \
             % (self.embedded_op.dim, str(self.targetLabels))
        s += str(self.embedded_op)
        return s


class LindbladErrorgen(LinearOperator):
    """
    An Lindblad-form error generator consisting of terms that, with appropriate
    constraints ensurse that the resulting (after exponentiation) gate/layer
    operation is CPTP.  These terms can be divided into "Hamiltonian"-type
    terms, which map rho -> i[H,rho] and "non-Hamiltonian"/"other"-type terms,
    which map rho -> A rho B + 0.5*(ABrho + rhoAB).
    """

    @classmethod
    def from_error_generator(cls, errgen, ham_basis="pp", nonham_basis="pp",
                             param_mode="cptp", nonham_mode="all",
                             mxBasis="pp", truncate=True, evotype="densitymx"):
        """
        Create a Lindblad-form error generator from an error generator matrix
        and a basis which specifies how to decompose (project) the error
        generator.
            
        errgen : numpy array or SciPy sparse matrix
            a square 2D array that gives the full error generator. The shape of
            this array sets the dimension of the operator. The projections of
            this quantity onto the `ham_basis` and `nonham_basis` are closely
            related to the parameters of the error generator (they may not be
            exactly equal if, e.g `cptp=True`).

        ham_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis is used to construct the Hamiltonian-type lindblad error
            Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        other_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis is used to construct the non-Hamiltonian-type lindblad error
            Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        param_mode : {"unconstrained", "cptp", "depol", "reldepol"}
            Describes how the Lindblad coefficients/projections relate to the
            gate's parameter values.  Allowed values are:
            `"unconstrained"` (coeffs are independent unconstrained parameters),
            `"cptp"` (independent parameters but constrained so map is CPTP),
            `"reldepol"` (all non-Ham. diagonal coeffs take the *same* value),
            `"depol"` (same as `"reldepol"` but coeffs must be *positive*)

        nonham_mode : {"diagonal", "diag_affine", "all"}
            Which non-Hamiltonian Lindblad projections are potentially non-zero.
            Allowed values are: `"diagonal"` (only the diagonal Lind. coeffs.),
            `"diag_affine"` (diagonal coefficients + affine projections), and
            `"all"` (the entire matrix of coefficients is allowed).

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        truncate : bool, optional
            Whether to truncate the projections onto the Lindblad terms in
            order to meet constraints (e.g. to preserve CPTP) when necessary.
            If False, then an error is thrown when the given `errgen` cannot 
            be realized by the specified set of Lindblad projections.

        evotype : {"densitymx","svterm","cterm"}
            The evolution type of the error generator being constructed.  
            `"densitymx"` means usual Lioville density-matrix-vector propagation
            via matrix-vector products.  `"svterm"` denotes state-vector term-
            based evolution (action of gate is obtained by evaluating the rank-1
            terms up to some order).  `"cterm"` is similar but uses Clifford gate
            action on stabilizer states.

        Returns
        -------
        LindbladErrorgen
        """

        d2 = errgen.shape[0]
        #d = int(round(_np.sqrt(d2)))  # OLD TODO REMOVE
        #if d*d != d2: raise ValueError("Error generator dim must be a perfect square")
        
        #Determine whether we're using sparse bases or not
        sparse = None
        if ham_basis is not None:
            if isinstance(ham_basis, _Basis): sparse = ham_basis.sparse
            elif _compat.isstr(ham_basis): sparse = _sps.issparse(errgen)
            elif len(ham_basis) > 0: sparse = _sps.issparse(ham_basis[0])
        if sparse is None and nonham_basis is not None:
            if isinstance(nonham_basis, _Basis): sparse = nonham_basis.sparse
            elif _compat.isstr(nonham_basis): sparse = _sps.issparse(errgen)
            elif len(nonham_basis) > 0: sparse = _sps.issparse(nonham_basis[0])
        if sparse is None: sparse = False #the default

        #Create or convert bases to appropriate sparsity
        if not isinstance(ham_basis, _Basis): # needed b/c ham_basis could be a Basis w/dim=0 which can't be cast as dim=d2
            ham_basis = _Basis.cast(ham_basis,d2,sparse=sparse)
        if not isinstance(nonham_basis, _Basis):
            nonham_basis = _Basis.cast(nonham_basis,d2,sparse=sparse)
        if not isinstance(mxBasis, _Basis):
            matrix_basis = _Basis.cast(mxBasis,d2,sparse=sparse)
        else: matrix_basis = mxBasis

        # errgen + bases => coeffs
        hamC, otherC = \
            _gt.lindblad_errgen_projections(
                errgen, ham_basis, nonham_basis, matrix_basis, normalize=False,
                return_generators=False, other_mode=nonham_mode, sparse=sparse)

        # coeffs + bases => Ltermdict, basisdict
        Ltermdict, basisdict = _gt.projections_to_lindblad_terms(
            hamC, otherC, ham_basis, nonham_basis, nonham_mode)
        
        return cls(d2, Ltermdict, basisdict,
                   param_mode, nonham_mode, truncate,
                   matrix_basis, evotype )

        
    def __init__(self, dim, Ltermdict, basisdict=None,
                 param_mode="cptp", nonham_mode="all", truncate=True,
                 mxBasis="pp", evotype="densitymx"): 
        """
        Create a new LinbladErrorgen based on a set of Lindblad terms.

        Note that if you want to construct a LinbladErrorgen from a
        error generator matrix, you can use the :method:`from_error_generator`
        class method.

        Parameters
        ----------
        dim : int
            The Hilbert-Schmidt (superoperator) dimension, which will be the
            dimension of the created operator.

        Ltermdict : dict
            A dictionary specifying which Linblad terms are present in the
            parameteriztion.  Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` can be `"H"` (Hamiltonian), `"S"`
            (Stochastic), or `"A"` (Affine).  Hamiltonian and Affine terms always
            have a single basis label (so key is a 2-tuple) whereas Stochastic
            tuples with 1 basis label indicate a *diagonal* term, and are the
            only types of terms allowed when `nonham_mode != "all"`.  Otherwise,
            Stochastic term tuples can include 2 basis labels to specify
            "off-diagonal" non-Hamiltonian Lindblad terms.  Basis labels can be
            strings or integers.  Values are complex coefficients (error rates).

        basisdict : dict, optional
            A dictionary mapping the basis labels (strings or ints) used in the
            keys of `Ltermdict` to basis matrices (numpy arrays or Scipy sparse
            matrices).

        param_mode : {"unconstrained", "cptp", "depol", "reldepol"}
            Describes how the Lindblad coefficients/projections relate to the
            error generator's parameter values.  Allowed values are:
            `"unconstrained"` (coeffs are independent unconstrained parameters),
            `"cptp"` (independent parameters but constrained so map is CPTP),
            `"reldepol"` (all non-Ham. diagonal coeffs take the *same* value),
            `"depol"` (same as `"reldepol"` but coeffs must be *positive*)

        nonham_mode : {"diagonal", "diag_affine", "all"}
            Which non-Hamiltonian Lindblad projections are potentially non-zero.
            Allowed values are: `"diagonal"` (only the diagonal Lind. coeffs.),
            `"diag_affine"` (diagonal coefficients + affine projections), and
            `"all"` (the entire matrix of coefficients is allowed).

        truncate : bool, optional
            Whether to truncate the projections onto the Lindblad terms in
            order to meet constraints (e.g. to preserve CPTP) when necessary.
            If False, then an error is thrown when the given dictionary of
            Lindblad terms doesn't conform to the constrains.

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The basis for this error generator's linear mapping. Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        evotype : {"densitymx","svterm","cterm"}
            The evolution type of the error generator being constructed.
            `"densitymx"` means the usual Lioville density-matrix-vector
            propagation via matrix-vector products.  `"svterm"` denotes
            state-vector term-based evolution (action of gate is obtained by
            evaluating the rank-1 terms up to some order).  `"cterm"` is similar
            but uses Clifford gate action on stabilizer states.
        """

        #FUTURE:
        # - maybe allow basisdict values to specify an "embedded matrix" w/a tuple like
        #  e.g. a *list* of (matrix, state_space_label) elements -- e.g. [(sigmaX,'Q1'), (sigmaY,'Q4')]
        # - maybe let keys be tuples of (basisname, state_space_label) e.g. (('X','Q1'),('Y','Q4')) -- and
        # maybe allow ('XY','Q1','Q4')? format when can assume single-letter labels.
        # - could add standard basis dict items so labels like "X", "XY", etc. are understood?

        # Store superop dimension
        d2 = dim
        #d = int(round(_np.sqrt(d2))) #OLD TODO REMOVE
        #assert(d*d == d2), "Dimension must be a perfect square"

        self.nonham_mode = nonham_mode
        self.param_mode = param_mode
        
        # Ltermdict, basisdict => bases + parameter values
        # but maybe we want Ltermdict, basisdict => basis + projections/coeffs, then projections/coeffs => paramvals?
        # since the latter is what set_errgen needs
        hamC, otherC, self.ham_basis, self.other_basis, hamBInds, otherBInds = \
            _gt.lindblad_terms_to_projections(Ltermdict, basisdict, d2, self.nonham_mode)

        self.ham_basis_size = len(self.ham_basis)
        self.other_basis_size = len(self.other_basis)

        if self.ham_basis_size > 0: self.sparse = _sps.issparse(self.ham_basis[0])
        elif self.other_basis_size > 0: self.sparse = _sps.issparse(self.other_basis[0])
        else: self.sparse = False

        self.matrix_basis = _Basis.cast(mxBasis,d2,sparse=self.sparse)

        self.paramvals = _gt.lindblad_projections_to_paramvals(
            hamC, otherC, self.param_mode, self.nonham_mode, truncate)

        LinearOperator.__init__(self, d2, evotype) #sets self.dim

        #Finish initialization based on evolution type
        assert(evotype in ("densitymx","svterm","cterm")), \
            "Invalid evotype: %s for %s" % (evotype, self.__class__.__name__)

        #Fast CSR-matrix summing variables: N/A if not sparse or using terms
        self.hamCSRSumIndices = None
        self.otherCSRSumIndices = None
        self.sparse_err_gen_template = None            
        
        if evotype == "densitymx":
            self.hamGens, self.otherGens = self._init_generators()

            if self.sparse:
                #Precompute for faster CSR sums in _construct_errgen
                all_csr_matrices = []
                if self.hamGens is not None:
                    all_csr_matrices.extend(self.hamGens)

                if self.otherGens is not None:
                    if self.nonham_mode == "diagonal":
                        oList = self.otherGens
                    else: # nonham_mode in ("diag_affine", "all")
                        oList = [ mx for mxRow in self.otherGens for mx in mxRow ]                        
                    all_csr_matrices.extend(oList)
    
                csr_sum_array, indptr, indices, N = \
                        _mt.get_csr_sum_indices(all_csr_matrices)
                self.hamCSRSumIndices = csr_sum_array[0:len(self.hamGens)]
                self.otherCSRSumIndices = csr_sum_array[len(self.hamGens):]
                self.sparse_err_gen_template = (indptr, indices, N)

            #initialize intermediate storage for matrix and for deriv computation
            # (needed for _construct_errgen)
            bsO = self.other_basis_size
            self.Lmx = _np.zeros((bsO-1,bsO-1),'complex') if bsO > 0 else None

            self._construct_errgen_matrix() # sets self.err_gen_mx
            self.Lterms = None # Unused
            
        else: # Term-based evolution

            assert(not self.sparse), "Sparse bases are not supported for term-based evolution"
              #TODO: make terms init-able from sparse elements, and below code  work with a *sparse* unitaryPostfactor
            termtype = "dense" if evotype == "svterm" else "clifford"
            
            self.Lterms = self._init_terms(Ltermdict, basisdict, hamBInds,
                                           otherBInds, termtype)
            # Unused
            self.hamGens = self.other = self.Lmx = None
            self.err_gen_mx = None

        #Done with __init__(...)


    def _init_generators(self):
        #assumes self.dim, self.ham_basis, self.other_basis, and self.matrix_basis are setup...
        
        d2 = self.dim
        d = int(round(_np.sqrt(d2)))
        assert(d*d == d2), "Errorgen dim must be a perfect square"

        # Get basis transfer matrix
        mxBasisToStd = self.matrix_basis.transform_matrix(_BuiltinBasis("std",self.matrix_basis.dim, self.sparse))
          # use BuiltinBasis("std") instead of just "std" in case matrix_basis is a TensorProdBasis
        leftTrans  = _spsl.inv(mxBasisToStd.tocsc()).tocsr() if _sps.issparse(mxBasisToStd) \
                          else _np.linalg.inv(mxBasisToStd)
        rightTrans = mxBasisToStd

        hamBasisMxs = self.ham_basis.elements 
        otherBasisMxs = self.other_basis.elements
        #OLD: these don't work if basis is empty (dim=0)
        # OLD REMOVE: _basis_matrices(self.ham_basis, d2, sparse=self.sparse)
        # OLD REMOVE: _basis_matrices(self.other_basis, d2, sparse=self.sparse)
        
        hamGens, otherGens = _gt.lindblad_error_generators(
            hamBasisMxs,otherBasisMxs,normalize=False,
            other_mode=self.nonham_mode) # in std basis

        # Note: lindblad_error_generators will return sparse generators when
        #  given a sparse basis (or basis matrices)

        if hamGens is not None:
            bsH = len(hamGens)+1 #projection-basis size (not nec. == d2)
            _gt._assert_shape(hamGens, (bsH-1,d2,d2), self.sparse)

            # apply basis change now, so we don't need to do so repeatedly later
            if self.sparse:
                hamGens = [ _mt.safereal(_mt.safedot(leftTrans, _mt.safedot(mx, rightTrans)),
                                              inplace=True, check=True) for mx in hamGens ]
                for mx in hamGens: mx.sort_indices()
                  # for faster addition ops in _construct_errgen_matrix
            else:
                #hamGens = _np.einsum("ik,akl,lj->aij", leftTrans, hamGens, rightTrans)
                hamGens = _np.transpose( _np.tensordot( 
                        _np.tensordot(leftTrans, hamGens, (1,1)), rightTrans, (2,0)), (1,0,2))
        else:
            bsH = 0
        assert(bsH == self.ham_basis_size)
            
        if otherGens is not None:

            if self.nonham_mode == "diagonal":
                bsO = len(otherGens)+1 #projection-basis size (not nec. == d2)
                _gt._assert_shape(otherGens, (bsO-1,d2,d2), self.sparse)

                # apply basis change now, so we don't need to do so repeatedly later
                if self.sparse:
                    otherGens = [ _mt.safereal(_mt.safedot(leftTrans, _mt.safedot(mx, rightTrans)),
                                                    inplace=True, check=True) for mx in otherGens ]
                    for mx in hamGens: mx.sort_indices()
                      # for faster addition ops in _construct_errgen_matrix
                else:
                    #otherGens = _np.einsum("ik,akl,lj->aij", leftTrans, otherGens, rightTrans)
                    otherGens = _np.transpose( _np.tensordot( 
                            _np.tensordot(leftTrans, otherGens, (1,1)), rightTrans, (2,0)), (1,0,2))

            elif self.nonham_mode == "diag_affine":
                bsO = len(otherGens[0])+1 # projection-basis size (not nec. == d2) [~shape[1] but works for lists too]
                _gt._assert_shape(otherGens, (2,bsO-1,d2,d2), self.sparse)

                # apply basis change now, so we don't need to do so repeatedly later
                if self.sparse:
                    otherGens = [ [_mt.safedot(leftTrans, _mt.safedot(mx, rightTrans))
                                        for mx in mxRow ] for mxRow in otherGens ]

                    for mxRow in otherGens:
                        for mx in mxRow: mx.sort_indices()
                          # for faster addition ops in _construct_errgen_matrix
                else:
                    #otherGens = _np.einsum("ik,abkl,lj->abij", leftTrans,
                    #                          otherGens, rightTrans)
                    otherGens = _np.transpose( _np.tensordot( 
                            _np.tensordot(leftTrans, otherGens, (1,2)), rightTrans, (3,0)), (1,2,0,3))
                    
            else:
                bsO = len(otherGens)+1 #projection-basis size (not nec. == d2)
                _gt._assert_shape(otherGens, (bsO-1,bsO-1,d2,d2), self.sparse)

                # apply basis change now, so we don't need to do so repeatedly later
                if self.sparse:
                    otherGens = [ [_mt.safedot(leftTrans, _mt.safedot(mx, rightTrans))
                                        for mx in mxRow ] for mxRow in otherGens ]
                    #Note: complex OK here, as only linear combos of otherGens (like (i,j) + (j,i)
                    # terms) need to be real

                    for mxRow in otherGens:
                        for mx in mxRow: mx.sort_indices()
                          # for faster addition ops in _construct_errgen_matrix
                else:
                    #otherGens = _np.einsum("ik,abkl,lj->abij", leftTrans,
                    #                            otherGens, rightTrans)
                    otherGens = _np.transpose( _np.tensordot( 
                            _np.tensordot(leftTrans, otherGens, (1,2)), rightTrans, (3,0)), (1,2,0,3))

        else:
            bsO = 0
        assert(bsO == self.other_basis_size)
        return hamGens, otherGens

    
    def _init_terms(self, Ltermdict, basisdict, hamBasisLabels, otherBasisLabels, termtype):

        d2 = self.dim
        d = int(round(_np.sqrt(d2)))
        tt = termtype # shorthand - used to construct RankOneTerm objects below,
                      # as we expect `basisdict` will contain *dense* basis
                      # matrices (maybe change in FUTURE?)
        numHamParams = len(hamBasisLabels)
        numOtherBasisEls = len(otherBasisLabels)
                      
        # Create Lindbladian terms - rank1 terms in the *exponent* with polynomial
        # coeffs (w/ *local* variable indices) that get converted to per-order
        # terms later.
        IDENT = None # sentinel for the do-nothing identity op
        Lterms = []
        for termLbl in Ltermdict:
            termType = termLbl[0]
            if termType == "H": # Hamiltonian
                k = hamBasisLabels[termLbl[1]] #index of parameter
                Lterms.append( _term.RankOneTerm(_Polynomial({(k,): -1j} ), basisdict[termLbl[1]], IDENT, tt) )
                Lterms.append( _term.RankOneTerm(_Polynomial({(k,): +1j} ), IDENT, basisdict[termLbl[1]].conjugate().T, tt) )

            elif termType == "S": # Stochastic
                if self.nonham_mode in ("diagonal","diag_affine"):
                    if self.param_mode in ("depol","reldepol"): # => same single param for all stochastic terms
                        k = numHamParams + 0 #index of parameter
                    else:
                        k = numHamParams + otherBasisLabels[termLbl[1]] #index of parameter
                    Lm = Ln = basisdict[termLbl[1]]
                    pw = 2 if self.param_mode in ("cptp","depol") else 1 # power to raise parameter to in order to get coeff

                    Lm_dag = Lm.conjugate().T # assumes basis is dense (TODO: make sure works for sparse case too - and np.dots below!)
                    Ln_dag = Ln.conjugate().T
                    Lterms.append( _term.RankOneTerm(_Polynomial({(k,)*pw:  1.0} ), Ln, Lm_dag, tt) )
                    Lterms.append( _term.RankOneTerm(_Polynomial({(k,)*pw: -0.5} ), IDENT, _np.dot(Ln_dag,Lm), tt) )
                    Lterms.append( _term.RankOneTerm(_Polynomial({(k,)*pw: -0.5} ), _np.dot(Lm_dag,Ln), IDENT, tt) )
                        
                else:
                    i = otherBasisLabels[termLbl[1]] #index of row in "other" coefficient matrix
                    j = otherBasisLabels[termLbl[2]] #index of col in "other" coefficient matrix
                    Lm, Ln = basisdict[termLbl[1]],basisdict[termLbl[2]]

                    # TODO: create these polys and place below...
                    polyTerms = {}
                    assert(self.param_mode != "depol"), "`depol` mode not supported when nonham_mode=='all'"
                    assert(self.param_mode != "reldepol"), "`reldepol` mode not supported when nonham_mode=='all'"
                    if self.param_mode == "cptp":
                        # otherCoeffs = _np.dot(self.Lmx,self.Lmx.T.conjugate())
                        # coeff_ij = sum_k Lik * Ladj_kj = sum_k Lik * conjugate(L_jk)
                        #          = sum_k (Re(Lik) + 1j*Im(Lik)) * (Re(L_jk) - 1j*Im(Ljk))
                        def iRe(a,b): return numHamParams + (a*numOtherBasisEls + b)
                        def iIm(a,b): return numHamParams + (b*numOtherBasisEls + a)
                        for k in range(0,min(i,j)+1):
                            if k <= i and k <= j:
                                polyTerms[ (iRe(i,k),iRe(j,k)) ] = 1.0
                            if k <= i and k < j:
                                polyTerms[ (iRe(i,k),iIm(j,k)) ] = -1.0j
                            if k < i and k <= j:
                                polyTerms[ (iIm(i,k),iRe(j,k)) ] = 1.0j
                            if k < i and k < j:
                                polyTerms[ (iIm(i,k),iIm(j,k)) ] = 1.0
                    else: # param_mode == "unconstrained"
                        # coeff_ij = otherParam[i,j] + 1j*otherParam[j,i] (otherCoeffs is Hermitian)
                        ijIndx = numHamParams + (i*numOtherBasisEls + j)
                        jiIndx = numHamParams + (j*numOtherBasisEls + i)
                        polyTerms = { (ijIndx,): 1.0, (jiIndx,): 1.0j }

                    base_poly = _Polynomial(polyTerms)
                    Lm_dag = Lm.conjugate().T; Ln_dag = Ln.conjugate().T
                    Lterms.append( _term.RankOneTerm(1.0*base_poly, Ln, Lm, tt) )
                    Lterms.append( _term.RankOneTerm(-0.5*base_poly, IDENT, _np.dot(Ln_dag,Lm), tt) ) # adjoint(_np.dot(Lm_dag,Ln))
                    Lterms.append( _term.RankOneTerm(-0.5*base_poly, _np.dot(Lm_dag,Ln), IDENT, tt ) )

            elif termType == "A": # Affine
                assert(self.nonham_mode == "diag_affine")
                if self.param_mode in ("depol","reldepol"): # => same single param for all stochastic terms
                    k = numHamParams + 1 + otherBasisLabels[termLbl[1]] #index of parameter
                else:
                    k = numHamParams + numOtherBasisEls + otherBasisLabels[termLbl[1]] #index of parameter

                # rho -> basisdict[termLbl[1]] * I = basisdict[termLbl[1]] * sum{ P_i rho P_i } where Pi's
                #  are the normalized paulis (including the identity), and rho has trace == 1
                #  (all but "I/d" component of rho are annihilated by pauli sum; for the I/d component, all
                #   d^2 of the terms in the sum is P/sqrt(d) * I/d * P/sqrt(d) == I/d^2, so the result is just "I")
                L = basisdict[termLbl[1]]
                Bmxs = _bt.basis_matrices("pp",d2, sparse=False) #Note: only works when `d` corresponds to integral # of qubits!

                for B in Bmxs: # Note: *include* identity! (see pauli scratch notebook for details)
                    Lterms.append( _term.RankOneTerm(_Polynomial({(k,): 1.0} ), _np.dot(L,B), B, tt) ) # /(d2-1.)
                    
                #TODO: check normalization of these terms vs those used in projections.

        #DEBUG
        #print("DB: params = ", list(enumerate(self.paramvals)))
        #print("DB: Lterms = ")
        #for i,lt in enumerate(Lterms):
        #    print("Term %d:" % i)
        #    print("  coeff: ", str(lt.coeff)) # list(lt.coeff.keys()) )
        #    print("  pre:\n", lt.pre_ops[0] if len(lt.pre_ops) else "IDENT")
        #    print("  post:\n",lt.post_ops[0] if len(lt.post_ops) else "IDENT")

        return Lterms

    
    def _set_params_from_matrix(self, errgen, truncate):
        """ Sets self.paramvals based on `errgen` """
        hamC, otherC  = \
            _gt.lindblad_errgen_projections(
                errgen, self.ham_basis, self.other_basis, self.matrix_basis, normalize=False,
                return_generators=False, other_mode=self.nonham_mode,
                sparse=self.sparse) # in std basis

        self.paramvals = _gt.lindblad_projections_to_paramvals(
            hamC, otherC, self.param_mode, self.nonham_mode, truncate)

            
    def _construct_errgen_matrix(self):
        """
        Build the error generator matrix using the current parameters.
        """
        d2 = self.dim
        hamCoeffs, otherCoeffs = _gt.paramvals_to_lindblad_projections(
            self.paramvals, self.ham_basis_size, self.other_basis_size,
            self.param_mode, self.nonham_mode, self.Lmx)
                            
        #Finally, build operation matrix from generators and coefficients:
        if self.sparse:
            #FUTURE: could try to optimize the sum-scalar-mults ops below, as these take the
            # bulk of from_vector time, which recurs frequently.
            indptr, indices, N = self.sparse_err_gen_template # the CSR arrays giving
               # the structure of a CSR matrix with 0-elements in all possible places
            data = _np.zeros(len(indices),'complex') # data starts at zero
            
            if hamCoeffs is not None:
                # lnd_error_gen = sum([c*gen for c,gen in zip(hamCoeffs, self.hamGens)])
                _mt.csr_sum(data,hamCoeffs, self.hamGens, self.hamCSRSumIndices)

            if otherCoeffs is not None:
                if self.nonham_mode == "diagonal":
                    # lnd_error_gen += sum([c*gen for c,gen in zip(otherCoeffs, self.otherGens)])
                    _mt.csr_sum(data, otherCoeffs, self.otherGens, self.otherCSRSumIndices)
                    
                else: # nonham_mode in ("diag_affine", "all")
                    # lnd_error_gen += sum([c*gen for cRow,genRow in zip(otherCoeffs, self.otherGens)
                    #                      for c,gen in zip(cRow,genRow)])
                    _mt.csr_sum(data, otherCoeffs.flat,
                                [oGen for oGenRow in self.otherGens for oGen in oGenRow],
                                self.otherCSRSumIndices)
            lnd_error_gen = _sps.csr_matrix( (data, indices.copy(), indptr.copy()), shape=(N,N) ) #copies needed (?)
            
        else: #dense matrices
            if hamCoeffs is not None:
                #lnd_error_gen = _np.einsum('i,ijk', hamCoeffs, self.hamGens)
                lnd_error_gen = _np.tensordot(hamCoeffs, self.hamGens, (0,0))
            else:
                lnd_error_gen = _np.zeros( (d2,d2), 'complex')

            if otherCoeffs is not None:
                if self.nonham_mode == "diagonal":
                    #lnd_error_gen += _np.einsum('i,ikl', otherCoeffs, self.otherGens)
                    lnd_error_gen += _np.tensordot(otherCoeffs, self.otherGens, (0,0))

                else: # nonham_mode in ("diag_affine", "all")
                    #lnd_error_gen += _np.einsum('ij,ijkl', otherCoeffs,
                    #                            self.otherGens)
                    lnd_error_gen += _np.tensordot(otherCoeffs, self.otherGens, ((0,1),(0,1)))


        assert(_np.isclose( _mt.safenorm(lnd_error_gen,'imag'), 0)), \
            "Imaginary error gen norm: %g" % _mt.safenorm(lnd_error_gen,'imag')
        #print("errgen pre-real = \n"); _mt.print_mx(lnd_error_gen,width=4,prec=1)        
        self.err_gen_mx = _mt.safereal(lnd_error_gen, inplace=True)


    def todense(self):
        """
        Return this error generator as a dense matrix.
        """
        if self.sparse: raise NotImplementedError("todense() not implemented for sparse LindbladErrorgen objects")
        if self._evotype in ("svterm","cterm"): 
            raise NotImplementedError("todense() not implemented for term-based LindbladErrorgen objects")
        return self.err_gen_mx

    #FUTURE: maybe remove this function altogether, as it really shouldn't be called
    def tosparse(self):
        """
        Return the error generator as a sparse matrix.
        """
        _warnings.warn(("Constructing the sparse matrix of a LindbladDenseOp."
                        "  Usually this is *NOT* a sparse matrix (the exponential of a"
                        " sparse matrix isn't generally sparse)!"))
        if self.sparse:
            return self.err_gen_mx
        else:
            return _sps.csr_matrix(self.todense())


    def torep(self):
        """
        Return a "representation" object for this error generator.

        Such objects are primarily used internally by pyGSTi to compute
        things like probabilities more efficiently.

        Returns
        -------
        OpRep
        """
        if self._evotype == "densitymx":
            if self.sparse:
                A = self.err_gen_mx
                return replib.DMOpRep_Sparse(
                    _np.ascontiguousarray(A.data),
                    _np.ascontiguousarray(A.indices, _np.int64),
                    _np.ascontiguousarray(A.indptr, _np.int64) )
            else:
                return replib.DMOpRep_Dense(_np.ascontiguousarray(self.err_gen_mx,'d'))
        else:
            raise NotImplementedError("torep(%s) not implemented for %s objects!" %
                                      (self._evotype, self.__class__.__name__))            


    def get_order_terms(self, order):
        """ 
        Get the `order`-th order Taylor-expansion terms of this operation.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that its action on a
        density matrix `rho` can be written:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the gate's
        parameters, where the polynomial's variable indices index the *global*
        parameters of the gate's parent (usually a :class:`Model`), not the 
        gate's local parameter array (i.e. that returned from `to_vector`).


        Parameters
        ----------
        order : int
            The order of terms to get.

        Returns
        -------
        list
            A list of :class:`RankOneTerm` objects.
        """
        assert(order == 0), "Error generators currently treat all terms as 0-th order; nothing else should be requested!"
        return self.Lterms


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
        if self._evotype == "densitymx":
            self._construct_errgen_matrix()
        self.dirty = True


    def get_coeffs(self):
        """
        Constructs a dictionary of the Lindblad-error-generator coefficients 
        (i.e. the "error rates") of this error generator.  Note that these are
        not necessarily the parameter values, as these coefficients are
        generally functions of the parameters (so as to keep the coefficients
        positive, for instance).

        Returns
        -------
        Ltermdict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Basis labels are integers starting at 0.  Values are complex
            coefficients (error rates).
    
        basisdict : dict
            A dictionary mapping the integer basis labels used in the
            keys of `Ltermdict` to basis matrices..
        """
        hamC, otherC = _gt.paramvals_to_lindblad_projections(
            self.paramvals, self.ham_basis_size, self.other_basis_size,
            self.param_mode, self.nonham_mode, self.Lmx)

        Ltermdict, basisdict = _gt.projections_to_lindblad_terms(
            hamC, otherC, self.ham_basis, self.other_basis, self.nonham_mode)
        return Ltermdict, basisdict


    def transform(self, S):
        """
        Update error generator E with inv(S) * E * S,

        Generally, the transform function updates the *parameters* of 
        the gate such that the resulting operation matrix is altered as 
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

            #conjugate Lindbladian exponent by U:
            self.err_gen_mx = _mt.safedot(Uinv,_mt.safedot(self.err_gen_mx, U))
            self._set_params_from_matrix(self.err_gen_mx, truncate=True)
            self._construct_errgen_matrix() # unnecessary? (TODO)
            self.dirty = True
            #Note: truncate=True above because some unitary transforms seem to
            ## modify eigenvalues to be negative beyond the tolerances
            ## checked when truncate == False.  I'm not sure why this occurs,
            ## since a true unitary should map CPTP -> CPTP...

        else:
            raise ValueError("Invalid transform for this LindbladErrorgen: type %s"
                             % str(type(S)))

    def spam_transform(self, S, typ):
        """
        Update operation matrix G with inv(S) * G OR G * S,
        depending on the value of `typ`.

        This functions as `transform(...)` but is used when this
        Lindblad-parameterized gate is used as a part of a SPAM
        vector.  When `typ == "prep"`, the spam vector is assumed
        to be `rho = dot(self, <spamvec>)`, which transforms as
        `rho -> inv(S) * rho`, so `self -> inv(S) * self`. When
        `typ == "effect"`, `e.dag = dot(e.dag, self)` (not that
        `self` is NOT `self.dag` here), and `e.dag -> e.dag * S`
        so that `self -> self * S`.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.

        typ : { 'prep', 'effect' }
            Which type of SPAM vector is being transformed (see above).
        """
        assert(typ in ('prep','effect')), "Invalid `typ` argument: %s" % typ
        
        if isinstance(S, _gaugegroup.UnitaryGaugeGroupElement) or \
           isinstance(S, _gaugegroup.TPSpamGaugeGroupElement):
            U = S.get_transform_matrix()
            Uinv = S.get_transform_matrix_inverse()

            #just act on postfactor and Lindbladian exponent:
            if typ == "prep":
                self.err_gen_mx = _mt.safedot(Uinv,self.err_gen_mx)
            else:
                self.err_gen_mx = _mt.safedot(self.err_gen_mx, U)
                
            self._set_params_from_matrix(self.err_gen_mx, truncate=True)
            self._construct_errgen_matrix() # unnecessary? (TODO)
            self.dirty = True
            #Note: truncate=True above because some unitary transforms seem to
            ## modify eigenvalues to be negative beyond the tolerances
            ## checked when truncate == False.  I'm not sure why this occurs,
            ## since a true unitary should map CPTP -> CPTP...
        else:
            raise ValueError("Invalid transform for this LindbladDenseOp: type %s"
                             % str(type(S)))


    def _dHdp(self):
        return self.hamGens.transpose((1,2,0)) #PRETRANS
        #return _np.einsum("ik,akl,lj->ija", self.leftTrans, self.hamGens, self.rightTrans)

    def _dOdp(self):
        bsH = self.ham_basis_size
        bsO = self.other_basis_size
        nHam = bsH-1 if (bsH > 0) else 0
        d2 = self.dim
        
        assert(bsO > 0),"Cannot construct dOdp when other_basis_size == 0!"
        if self.nonham_mode == "diagonal":
            otherParams = self.paramvals[nHam:]
            
            # Derivative of exponent wrt other param; shape == [d2,d2,bs-1]
            #  except "depol" & "reldepol" cases, when shape == [d2,d2,1]
            if self.param_mode == "depol": # all coeffs same & == param^2
                assert(len(otherParams) == 1), "Should only have 1 non-ham parameter in 'depol' case!"
                #dOdp  = _np.einsum('alj->lj', self.otherGens)[:,:,None] * 2*otherParams[0]
                dOdp  = _np.transpose(self.otherGens, (1,2,0)) * 2*otherParams[0]
            elif self.param_mode == "reldepol": # all coeffs same & == param
                assert(len(otherParams) == 1), "Should only have 1 non-ham parameter in 'reldepol' case!"
                #dOdp  = _np.einsum('alj->lj', self.otherGens)[:,:,None]
                dOdp  = _np.transpose(self.otherGens, (1,2,0)) * 2*otherParams[0]
            elif self.param_mode == "cptp": # (coeffs = params^2)
                #dOdp  = _np.einsum('alj,a->lja', self.otherGens, 2*otherParams) 
                dOdp = _np.transpose(self.otherGens,(1,2,0)) * 2*otherParams # just a broadcast
            else: # "unconstrained" (coeff == params)
                #dOdp  = _np.einsum('alj->lja', self.otherGens)
                dOdp  = _np.transpose(self.otherGens, (1,2,0))


        elif self.nonham_mode == "diag_affine":
            otherParams = self.paramvals[nHam:]
            # Note: otherGens has shape (2,bsO-1,d2,d2) with diag-term generators
            # in first "row" and affine generators in second row.
            
            # Derivative of exponent wrt other param; shape == [d2,d2,2,bs-1]
            #  except "depol" & "reldepol" cases, when shape == [d2,d2,bs]
            if self.param_mode == "depol": # all coeffs same & == param^2
                diag_params, affine_params = otherParams[0:1], otherParams[1:]
                dOdp  = _np.empty((d2,d2,bsO),'complex')
                #dOdp[:,:,0]  = _np.einsum('alj->lj', self.otherGens[0]) * 2*diag_params[0] # single diagonal term
                #dOdp[:,:,1:] = _np.einsum('alj->lja', self.otherGens[1]) # no need for affine_params
                dOdp[:,:,0]  = _np.squeeze(self.otherGens[0],0) * 2*diag_params[0] # single diagonal term
                dOdp[:,:,1:] = _np.transpose(self.otherGens[1],(1,2,0)) # no need for affine_params
            elif self.param_mode == "reldepol": # all coeffs same & == param^2
                dOdp  = _np.empty((d2,d2,bsO),'complex')
                #dOdp[:,:,0]  = _np.einsum('alj->lj', self.otherGens[0]) # single diagonal term
                #dOdp[:,:,1:] = _np.einsum('alj->lja', self.otherGens[1]) # affine part: each gen has own param
                dOdp[:,:,0]  = _np.squeeze(self.otherGens[0],0) # single diagonal term
                dOdp[:,:,1:] = _np.transpose(self.otherGens[1],(1,2,0)) # affine part: each gen has own param
            elif self.param_mode == "cptp": # (coeffs = params^2)
                diag_params, affine_params = otherParams[0:bsO-1], otherParams[bsO-1:]
                dOdp  = _np.empty((d2,d2,2,bsO-1),'complex')
                #dOdp[:,:,0,:] = _np.einsum('alj,a->lja', self.otherGens[0], 2*diag_params)
                #dOdp[:,:,1,:] = _np.einsum('alj->lja', self.otherGens[1]) # no need for affine_params
                dOdp[:,:,0,:] = _np.transpose(self.otherGens[0],(1,2,0)) * 2*diag_params # broadcast works
                dOdp[:,:,1,:] = _np.transpose(self.otherGens[1],(1,2,0)) # no need for affine_params
            else: # "unconstrained" (coeff == params)
                #dOdp  = _np.einsum('ablj->ljab', self.otherGens) # -> shape (d2,d2,2,bsO-1)
                dOdp  = _np.transpose(self.otherGens, (2,3,0,1) ) # -> shape (d2,d2,2,bsO-1)

        else: # nonham_mode == "all" ; all lindblad terms included
            assert(self.param_mode in ("cptp","unconstrained"))
            
            if self.param_mode == "cptp":
                L,Lbar = self.Lmx,self.Lmx.conjugate()
                F1 = _np.tril(_np.ones((bsO-1,bsO-1),'d'))
                F2 = _np.triu(_np.ones((bsO-1,bsO-1),'d'),1) * 1j
                
                  # Derivative of exponent wrt other param; shape == [d2,d2,bs-1,bs-1]
                  # Note: replacing einsums here results in at least 3 numpy calls (probably slower?)
                dOdp  = _np.einsum('amlj,mb,ab->ljab', self.otherGens, Lbar, F1) #only a >= b nonzero (F1)
                dOdp += _np.einsum('malj,mb,ab->ljab', self.otherGens, L, F1)    # ditto
                dOdp += _np.einsum('bmlj,ma,ab->ljab', self.otherGens, Lbar, F2) #only b > a nonzero (F2)
                dOdp += _np.einsum('mblj,ma,ab->ljab', self.otherGens, L, F2.conjugate()) # ditto
            else: # "unconstrained"
                F0 = _np.identity(bsO-1,'d')
                F1 = _np.tril(_np.ones((bsO-1,bsO-1),'d'),-1)
                F2 = _np.triu(_np.ones((bsO-1,bsO-1),'d'),1) * 1j
            
                # Derivative of exponent wrt other param; shape == [d2,d2,bs-1,bs-1]
                #dOdp  = _np.einsum('ablj,ab->ljab', self.otherGens, F0)  # a == b case
                #dOdp += _np.einsum('ablj,ab->ljab', self.otherGens, F1) + \
                #           _np.einsum('balj,ab->ljab', self.otherGens, F1) # a > b (F1)
                #dOdp += _np.einsum('balj,ab->ljab', self.otherGens, F2) - \
                #           _np.einsum('ablj,ab->ljab', self.otherGens, F2) # a < b (F2)
                tmp_ablj = _np.transpose(self.otherGens, (2,3,0,1)) # ablj -> ljab
                tmp_balj = _np.transpose(self.otherGens, (2,3,1,0)) # balj -> ljab
                dOdp  = tmp_ablj * F0  # a == b case
                dOdp += tmp_ablj * F1 + tmp_balj * F1 # a > b (F1)
                dOdp += tmp_balj * F2 - tmp_ablj * F2 # a < b (F2)

        # apply basis transform
        tr = len(dOdp.shape) #tensor rank
        assert( (tr-2) in (1,2)), "Currently, dodp can only have 1 or 2 derivative dimensions"

        assert(_np.linalg.norm(_np.imag(dOdp)) < IMAG_TOL)
        return _np.real(dOdp)


    def _d2Odp2(self):
        bsH = self.ham_basis_size
        bsO = self.other_basis_size
        nHam = bsH-1 if (bsH > 0) else 0
        d2 = self.dim
        
        assert(bsO > 0),"Cannot construct dOdp when other_basis_size == 0!"
        if self.nonham_mode == "diagonal":
            otherParams = self.paramvals[nHam:]
            nP = len(otherParams); 
            
            # Derivative of exponent wrt other param; shape == [d2,d2,nP,nP]
            if self.param_mode == "depol":
                assert(nP == 1)
                #d2Odp2  = _np.einsum('alj->lj', self.otherGens)[:,:,None,None] * 2
                d2Odp2  = _np.squeeze(self.otherGens,0)[:,:,None,None] * 2
            elif self.param_mode == "cptp":
                assert(nP == bsO-1)
                #d2Odp2  = _np.einsum('alj,aq->ljaq', self.otherGens, 2*_np.identity(nP,'d'))
                d2Odp2  = _np.transpose(self.otherGens, (1,2,0))[:,:,:,None] * 2*_np.identity(nP,'d')
            else: # param_mode == "unconstrained" or "reldepol"
                assert(nP == bsO-1)
                d2Odp2  = _np.zeros([d2,d2,nP,nP],'d')

        elif self.nonham_mode == "diag_affine":
            otherParams = self.paramvals[nHam:]
            nP = len(otherParams); 
            
            # Derivative of exponent wrt other param; shape == [d2,d2,nP,nP]
            if self.param_mode == "depol":
                assert(nP == bsO) # 1 diag param + (bsO-1) affine params
                d2Odp2  = _np.empty((d2,d2,nP,nP),'complex')
                #d2Odp2[:,:,0,0]  = _np.einsum('alj->lj', self.otherGens[0]) * 2 # single diagonal term
                d2Odp2[:,:,0,0]  = _np.squeeze(self.otherGens[0],0) * 2 # single diagonal term
                d2Odp2[:,:,1:,1:]  = 0 # 2nd deriv wrt. all affine params == 0
            elif self.param_mode == "cptp":
                assert(nP == 2*(bsO-1)); hnP = bsO-1 # half nP
                d2Odp2  = _np.empty((d2,d2,nP,nP),'complex')
                #d2Odp2[:,:,0:hnP,0:hnp] = _np.einsum('alj,aq->ljaq', self.otherGens[0], 2*_np.identity(nP,'d'))
                d2Odp2[:,:,0:hnP,0:hnp] = _np.transpose(self.otherGens[0],(1,2,0))[:,:,:,None] * 2*_np.identity(nP,'d')
                d2Odp2[:,:,hnP:,hnp:]   = 0 # 2nd deriv wrt. all affine params == 0
            else: # param_mode == "unconstrained" or "reldepol"
                assert(nP == 2*(bsO-1))
                d2Odp2  = _np.zeros([d2,d2,nP,nP],'d')

        else: # nonham_mode == "all" : all lindblad terms included
            nP = bsO-1
            if self.param_mode == "cptp":
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
                
            else: # param_mode == "unconstrained"
                d2Odp2  = _np.zeros([d2,d2,nP,nP,nP,nP],'d') #all params linear

        # apply basis transform
        tr = len(d2Odp2.shape) #tensor rank
        assert( (tr-2) in (2,4)), "Currently, d2Odp2 can only have 2 or 4 derivative dimensions"

        assert(_np.linalg.norm(_np.imag(d2Odp2)) < IMAG_TOL)
        return _np.real(d2Odp2)


    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the vectorized derivatives of the
        flattened error generator matrix with respect to a single operator
        parameter.  Thus, each column is of length op_dim^2 and there is one
        column per gate parameter.

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension^2, num_params)
        """    
        assert(not self.sparse), \
            "LindbladErrorgen.deriv_wrt_params(...) can only be called when using *dense* basis elements!"

        d2 = self.dim
        bsH = self.ham_basis_size
        bsO = self.other_basis_size
    
        #Deriv wrt hamiltonian params
        if bsH > 0:
            dH = self._dHdp()
            dH = dH.reshape((d2**2,bsH-1)) # [iFlattenedOp,iHamParam]
        else: 
            dH = _np.empty( (d2**2,0), 'd') #so concat works below

        #Deriv wrt other params
        if bsO > 0:
            dO = self._dOdp()
            dO = dO.reshape((d2**2,-1)) # [iFlattenedOp,iOtherParam]
        else:
            dO = _np.empty( (d2**2,0), 'd') #so concat works below

        derivMx = _np.concatenate((dH,dO), axis=1)
        assert(_np.linalg.norm(_np.imag(derivMx)) < IMAG_TOL) #allowed to be complex?
        derivMx = _np.real(derivMx)

        if wrtFilter is None:
            return derivMx
        else:
            return _np.take( derivMx, wrtFilter, axis=1 )


    def hessian_wrt_params(self, wrtFilter1=None, wrtFilter2=None):
        """
        Construct the Hessian of this error generator with respect to
        its parameters.

        This function returns a tensor whose first axis corresponds to the
        flattened operation matrix and whose 2nd and 3rd axes correspond to the
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
        assert(not self.sparse), \
            "LindbladErrorgen.hessian_wrt_params(...) can only be called when using *dense* basis elements!"

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
            nP = nTotParams-nHam #num "other" params, e.g. (bsO-1) or (bsO-1)**2
            d2Odp2 = self._d2Odp2()
            d2Odp2 = d2Odp2.reshape((d2**2, nP, nP))

            #d2Odp2 has been reshape so index as [iFlattenedOp,iDeriv1,iDeriv2]
            assert(_np.linalg.norm(_np.imag(d2Odp2)) < IMAG_TOL)
            hessianMx[:,nHam:,nHam:] = _np.real(d2Odp2) # d2O block of hessian
            
        if wrtFilter1 is None:
            if wrtFilter2 is None:
                return hessianMx
            else:
                return _np.take(hessianMx, wrtFilter2, axis=2 )
        else:
            if wrtFilter2 is None:
                return _np.take(hessianMx, wrtFilter1, axis=1 )
            else:
                return _np.take( _np.take(hessianMx, wrtFilter1, axis=1),
                                 wrtFilter2, axis=2 )
        
    def __str__(self):
        s = "Lindblad error generator with dim = %d, num params = %d\n" % \
            (self.dim, self.num_params())
        return s
