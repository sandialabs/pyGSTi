"""
Defines classes with represent SPAM operations, along with supporting
functionality.
"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import scipy.sparse as _sps
import collections as _collections
import numbers as _numbers
import itertools as _itertools
import functools as _functools
import copy as _copy

from ..      import optimize as _opt
from ..tools import matrixtools as _mt
from ..tools import gatetools as _gt
from ..tools import basistools as _bt
from ..tools import listtools as _lt
from ..tools import slicetools as _slct
from ..tools import compattools as _compat
from ..tools import symplectic as _symp
from ..baseobjs import Basis as _Basis
from ..baseobjs import ProtectedArray as _ProtectedArray
from . import gatesetmember as _gatesetmember

from . import term as _term
from . import stabilizer as _stabilizer
from .polynomial import Polynomial as _Polynomial
from . import replib    

IMAG_TOL = 1e-8 #tolerance for imaginary part being considered zero

def optimize_spamvec(vecToOptimize, targetVec):
    """
    Optimize the parameters of vecToOptimize so that the
      the resulting SPAM vector is as close as possible to
      targetVec.

    This is trivial for the case of FullyParameterizedSPAMVec
      instances, but for other types of parameterization
      this involves an iterative optimization over all the
      parameters of vecToOptimize.

    Parameters
    ----------
    vecToOptimize : SPAMVec
      The vector to optimize. This object gets altered.

    targetVec : SPAMVec
      The SPAM vector used as the target.

    Returns
    -------
    None
    """

    #TODO: cleanup this code:
    if isinstance(vecToOptimize, StaticSPAMVec):
        return #nothing to optimize

    if isinstance(vecToOptimize, FullyParameterizedSPAMVec):
        if(targetVec.dim != vecToOptimize.dim): #special case: gates can have different overall dimension
            vecToOptimize.dim = targetVec.dim   #  this is a HACK to allow model selection code to work correctly
        vecToOptimize.set_value(targetVec)     #just copy entire overall matrix since fully parameterized
        return

    assert(targetVec.dim == vecToOptimize.dim) #vectors must have the same overall dimension
    targetVector = _np.asarray(targetVec)

    def _objective_func(param_vec):
        vecToOptimize.from_vector(param_vec)
        return _mt.frobeniusnorm(vecToOptimize-targetVector)

    x0 = vecToOptimize.to_vector()
    minSol = _opt.minimize(_objective_func, x0, method='BFGS', maxiter=10000, maxfev=10000,
                           tol=1e-6, callback=None)

    vecToOptimize.from_vector(minSol.x)
    #print("DEBUG: optimized vector to min frobenius distance %g" % _mt.frobeniusnorm(vecToOptimize-targetVector))


def convert(spamvec, toType, basis, extra=None):
    """
    Convert SPAM vector to a new type of parameterization, potentially
    creating a new SPAMVec object.  Raises ValueError for invalid conversions.

    Parameters
    ----------
    spamvec : SPAMVec
        SPAM vector to convert

    toType : {"full","TP","static","static unitary","clifford",LINDBLAD}
        The type of parameterizaton to convert to.  "LINDBLAD" is a placeholder
        for the various Lindblad parameterization types.  See
        :method:`GateSet.set_all_parameterizations` for more details.

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for `spamvec`.  Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
        (or a custom basis object).

    extra : object, optional
        Additional information for conversion.


    Returns
    -------
    SPAMVec
       The converted SPAM vector, usually a distinct
       object from the object passed as input.
    """
    if toType == "full":
        if isinstance(spamvec, FullyParameterizedSPAMVec):
            return spamvec #no conversion necessary
        else:
            return FullyParameterizedSPAMVec( spamvec.todense() )

    elif toType == "TP":
        if isinstance(spamvec, TPParameterizedSPAMVec):
            return spamvec #no conversion necessary
        else:
            return TPParameterizedSPAMVec( spamvec.todense() )
              # above will raise ValueError if conversion cannot be done

    elif toType == "TrueCPTP": # a non-lindbladian CPTP spamvec that hasn't worked well...
        if isinstance(spamvec, CPTPParameterizedSPAMVec):
            return spamvec #no conversion necessary
        else:
            return CPTPParameterizedSPAMVec( spamvec, basis )
              # above will raise ValueError if conversion cannot be done

    elif toType == "static":
        if isinstance(spamvec, StaticSPAMVec):
            return spamvec #no conversion necessary
        else:
            return StaticSPAMVec( spamvec )

    elif toType == "static unitary":
        dmvec = _bt.change_basis(spamvec.todense(),basis,'std')
        purevec = _gt.dmvec_to_state(dmvec)
        return StaticSPAMVec(purevec, "statevec")

    elif _gt.is_valid_lindblad_paramtype(toType):

        if extra is None:
            purevec = spamvec # right now, we don't try to extract a "closest pure vec"
                              # to spamvec - below will fail if spamvec isn't pure.
        else:
            purevec = extra # assume extra info is a pure vector

        nQubits = _np.log2(spamvec.dim)/2.0
        bQubits = bool(abs(nQubits-round(nQubits)) < 1e-10) #integer # of qubits?
        proj_basis = "pp" if (basis == "pp" or bQubits) else basis

        return LindbladParameterizedSPAMVec.from_spamvec_obj(
            spamvec, "prep", toType, None, proj_basis, basis,
            truncate=True, lazy=True)

    elif toType == "clifford":
        if isinstance(spamvec, StabilizerSPAMVec):
            return spamvec #no conversion necessary

        purevec = spamvec.flatten() # assume a pure state (otherwise would
                                    # need to change GateSet dim)
        return StabilizerSPAMVec.from_dense_purevec(purevec)

    else:
        raise ValueError("Invalid toType argument: %s" % toType)

def _convert_to_lindblad_base(vec, typ, new_evotype, mxBasis="pp"):
    """ 
    Attempts to convert `vec` to a static (0 params) SPAMVec with 
    evoution type `new_evotype`.  Used to convert spam vecs to
    being LindbladParameterizedSPAMVec objects.
    """
    if vec._evotype == new_evotype and vec.num_params() == 0:
        return vec #no conversion necessary
    if new_evotype == "densitymx":
        return StaticSPAMVec(vec.todense(),"densitymx")
    if new_evotype in ("svterm","cterm"):
        if isinstance(vec, ComputationalSPAMVec): #special case when conversion is easy
            return ComputationalSPAMVec(vec._zvals, new_evotype)
        elif vec._evotype == "densitymx":
            # then try to extract a (static) pure state from vec wth
            # evotype 'statevec' or 'stabilizer' <=> 'svterm', 'cterm'
            if isinstance(vec, DenseSPAMVec):
                dmvec = _bt.change_basis(vec,mxBasis,'std')
                purestate = StaticSPAMVec(_gt.dmvec_to_state(dmvec),'statevec')
            elif isinstance(vec, PureStateSPAMVec):
                purestate = vec.pure_state_vec # evotype 'statevec'
            else:
                raise ValueError("Unable to obtain pure state from density matrix type %s!" % type(vec))

            if new_evotype == "cterm": # then purestate 'statevec' => 'stabilizer' (if possible)
                if typ =="prep":
                    purestate = StabilizerSPAMVec.from_dense_purevec(purestate.todense())
                else: # type == "effect"
                    purestate = StabilizerEffectVec.from_dense_purevec(purestate.todense())
                
            return PureStateSPAMVec(purestate, new_evotype, mxBasis)
        
    raise ValueError("Could not convert %s (evotype %s) to %s w/0 params!" %
                     (str(type(vec)), vec._evotype, new_evotype))



def finite_difference_deriv_wrt_params(spamvec, eps=1e-7):
    """
    Computes a finite-difference Jacobian for a SPAMVec object.

    The returned value is a matrix whose columns are the vectorized
    derivatives of the spam vector with respect to a single
    parameter, matching the format expected from the spam vectors's
    `deriv_wrt_params` method.

    Parameters
    ----------
    spamvec : SPAMVec
        The spam vector object to compute a Jacobian for.
        
    eps : float, optional
        The finite difference step to use.

    Returns
    -------
    numpy.ndarray
        An M by N matrix where M is the number of gate elements and
        N is the number of gate parameters. 
    """
    dim = spamvec.get_dimension()
    spamvec2 = spamvec.copy()
    p = spamvec.to_vector()
    fd_deriv = _np.empty((dim,spamvec.num_params()), 'd') #assume real (?)

    for i in range(spamvec.num_params()):
        p_plus_dp = p.copy()
        p_plus_dp[i] += eps
        spamvec2.from_vector(p_plus_dp)
        fd_deriv[:,i:i+1] = (spamvec2-spamvec)/eps

    fd_deriv.shape = [dim,spamvec.num_params()]
    return fd_deriv


def check_deriv_wrt_params(spamvec, deriv_to_check=None, eps=1e-7):
    """
    Checks the `deriv_wrt_params` method of a SPAMVec object.

    This routine is meant to be used as an aid in testing and debugging
    SPAMVec classes by comparing the finite-difference Jacobian that
    *should* be returned by `spamvec.deriv_wrt_params` with the one that
    actually is.  A ValueError is raised if the two do not match.

    Parameters
    ----------
    spamvec : SPAMVec
        The gate object to test.

    deriv_to_check : numpy.ndarray or None, optional
        If not None, the Jacobian to compare against the finite difference
        result.  If None, `spamvec.deriv_wrt_parms()` is used.  Setting this
        argument can be useful when the function is called *within* a Gate
        class's `deriv_wrt_params()` method itself as a part of testing.
        
    eps : float, optional
        The finite difference step to use.

    Returns
    -------
    None
    """
    fd_deriv = finite_difference_deriv_wrt_params(spamvec, eps)
    if deriv_to_check is None:
        deriv_to_check = spamvec.deriv_wrt_params()

    #print("Deriv shapes = %s and %s" % (str(fd_deriv.shape),
    #                                    str(deriv_to_check.shape)))
    #print("finite difference deriv = \n",fd_deriv)
    #print("deriv_wrt_params deriv = \n",deriv_to_check)
    #print("deriv_wrt_params - finite diff deriv = \n",
    #      deriv_to_check - fd_deriv)
    
    for i in range(deriv_to_check.shape[0]):
        for j in range(deriv_to_check.shape[1]):
            diff = abs(deriv_to_check[i,j] - fd_deriv[i,j])
            if diff > 5*eps:
                print("deriv_chk_mismatch: (%d,%d): %g (comp) - %g (fd) = %g" %
                      (i,j,deriv_to_check[i,j],fd_deriv[i,j],diff))

    if _np.linalg.norm(fd_deriv - deriv_to_check) > 100*eps:
        raise ValueError("Failed check of deriv_wrt_params:\n" +
                         " norm diff = %g" % 
                         _np.linalg.norm(fd_deriv - deriv_to_check))


class SPAMVec(_gatesetmember.GateSetMember):
    """
    Excapulates a parameterization of a state preparation OR POVM effect
    vector. This class is the  common base class for all specific
    parameterizations of a SPAM vector.
    """

    def __init__(self, dim, evotype):
        """ Initialize a new SPAM Vector """
        super(SPAMVec, self).__init__( dim, evotype )

    @property
    def size(self):
        """
        Return the number of independent elements in this gate (when viewed as a dense array)
        """
        return self.dim

    @property
    def outcomes(self):
        """
        Return the z-value outcomes corresponding to this effect SPAM vector
        in the context of a stabilizer-state simulation.
        """
        raise NotImplementedError("'outcomes' property is not implemented for %s objects" % self.__class__.__name__)


    def set_value(self, vec):
        """
        Attempts to modify SPAMVec parameters so that the specified raw
        SPAM vector becomes vec.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        vec : array_like or SPAMVec
            A numpy array representing a SPAM vector, or a SPAMVec object.

        Returns
        -------
        None
        """
        raise ValueError("Cannot set the value of a %s directly!" % self.__class__.__name__)

    
    def todense(self, scratch=None):      
        """
        Return this SPAM vector as a (dense) numpy array.  The memory
        in `scratch` maybe used when it is not-None.
        """
        raise NotImplementedError("todense(...) not implemented for %s objects!" % self.__class__.__name__)

    def torep(self, typ, outrep=None):
        """
        Return a "representation" object for this SPAM vector.

        Such objects are primarily used internally by pyGSTi to compute
        things like probabilities more efficiently.

        Parameters
        ----------
        typ : {'prep','effect'}
            The type of representation (for cases when the vector type is
            not already defined).

        outrep : StateRep
            If not None, an existing state representation appropriate to this
            SPAM vector that may be used instead of allocating a new one.

        Returns
        -------
        StateRep
        """
        if typ == "prep":
            if self._evotype == "statevec":
                return replib.SVStateRep(self.todense())
            elif self._evotype == "densitymx":
                return replib.DMStateRep(self.todense())
            raise NotImplementedError("torep(%s) not implemented for %s objects!" %
                                      (self._evotype, self.__class__.__name__))
        elif typ == "effect":
            if self._evotype == "statevec":
                return replib.SVEffectRep_Dense(self.todense())
            elif self._evotype == "densitymx":
                return replib.DMEffectRep_Dense(self.todense())
            raise NotImplementedError("torep(%s) not implemented for %s objects!" %
                                      (self._evotype, self.__class__.__name__))
        else:
            raise ValueError("Invalid `typ` argument for torep(): %s" % typ)

            
    def get_order_terms(self, order):
        """ 
        Get the `order`-th order Taylor-expansion terms of this SPAM vector.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that it is a state
        preparation followed by or POVM effect preceded by actions on a
        density matrix `rho` of the form:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the 
        SPAMVec's parameters, where the polynomial's variable indices index the
        *global* parameters of the SPAMVec's parent (usually a :class:`GateSet`)
        , not the SPAMVec's local parameter array (i.e. that returned from
        `to_vector`).


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

    
    def frobeniusdist2(self, otherSpamVec, typ, transform=None,
                       inv_transform=None):
        """ 
        Return the squared frobenius difference between this spam vector and
        `otherSpamVec`, optionally transforming this vector first using
        `transform` and `inv_transform` (depending on the value of `typ`).

        Parameters
        ----------
        otherSpamVec : SPAMVec
            The other spam vector

        typ : { 'prep', 'effect' }
            Which type of SPAM vector is being transformed.

        transform, inv_transform : numpy.ndarray
            The transformation (if not None) to be performed.
        
        Returns
        -------
        float
        """
        vec = self.todense()
        if typ == 'prep':
            if inv_transform is None:
                return _gt.frobeniusdist2(vec,otherSpamVec.todense())
            else:
                return _gt.frobeniusdist2(_np.dot(inv_transform,vec),
                                          otherSpamVec.todense())
        elif typ == "effect":
            if transform is None:
                return _gt.frobeniusdist2(vec,otherSpamVec.todense())
            else:
                return _gt.frobeniusdist2(_np.dot(_np.transpose(transform),
                                                  vec), otherSpamVec.todense())
        else: raise ValueError("Invalid 'typ' argument: %s" % typ)
        

    def residuals(self, otherSpamVec, typ, transform=None, inv_transform=None):
        """ 
        Return a vector of residuals between this spam vector and
        `otherSpamVec`, optionally transforming this vector first using
        `transform` and `inv_transform` (depending on the value of `typ`).

        Parameters
        ----------
        otherSpamVec : SPAMVec
            The other spam vector

        typ : { 'prep', 'effect' }
            Which type of SPAM vector is being transformed.

        transform, inv_transform : numpy.ndarray
            The transformation (if not None) to be performed.
        
        Returns
        -------
        float
        """
        vec = self.todense()
        if typ == 'prep':
            if inv_transform is None:
                return _gt.residuals(vec,otherSpamVec.todense())
            else:
                return _gt.residuals(_np.dot(inv_transform,vec),
                                          otherSpamVec.todense())
        elif typ == "effect":
            if transform is None:
                return _gt.residuals(vec,otherSpamVec.todense())
            else:
                return _gt.residuals(_np.dot(_np.transpose(transform),
                                             vec), otherSpamVec.todense())

            
    def transform(self, S, typ):
        """
        Update SPAM (column) vector V as inv(S) * V or S^T * V for preparation
        or  effect SPAM vectors, respectively.

        Note that this is equivalent to state preparation vectors getting 
        mapped: `rho -> inv(S) * rho` and the *transpose* of effect vectors
        being mapped as `E^T -> E^T * S`.

        Generally, the transform function updates the *parameters* of 
        the SPAM vector such that the resulting vector is altered as 
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.
            
        typ : { 'prep', 'effect' }
            Which type of SPAM vector is being transformed (see above).
        """
        if typ == 'prep':
            Si  = S.get_transform_matrix_inverse()
            self.set_value(_np.dot(Si, self.todense()))
        elif typ == 'effect':
            Smx = S.get_transform_matrix()
            self.set_value(_np.dot(_np.transpose(Smx),self.todense()))
              #Evec^T --> ( Evec^T * S )^T
        else:
            raise ValueError("Invalid typ argument: %s" % typ)


    def depolarize(self, amount):
        """
        Depolarize this SPAM vector by the given `amount`.

        Generally, the depolarize function updates the *parameters* of 
        the SPAMVec such that the resulting vector is depolarized.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. All but the
            first element of the spam vector (often corresponding to the 
            identity element) are multiplied by `amount` (if a float) or
            the corresponding `amount[i]` (if a tuple).

        Returns
        -------
        None
        """
        if isinstance(amount,float) or _compat.isint(amount):
            D = _np.diag( [1]+[1-amount]*(self.dim-1) )
        else:
            assert(len(amount) == self.dim-1)
            D = _np.diag( [1]+list(1.0 - _np.array(amount,'d')) )
        self.set_value(_np.dot(D,self.todense()))

        
    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return 0 #no parameters


    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return _np.array([], 'd') #no parameters


    def from_vector(self, v):
        """
        Initialize the SPAM vector using a 1D array of parameters.

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
        Construct a matrix whose columns are the derivatives of the SPAM vector
        with respect to a single param.  Thus, each column is of length
        get_dimension and there is one column per SPAM vector parameter.
        An empty 2D array in the StaticSPAMVec case (num_params == 0).

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension, num_params)
        """
        dtype = complex if self._evotype == 'statevec' else 'd'
        derivMx = _np.zeros((self.dim,0),dtype)
        if wrtFilter is None:
            return derivMx
        else:
            return _np.take( derivMx, wrtFilter, axis=1 )


    def has_nonzero_hessian(self):
        """ 
        Returns whether this SPAM vector has a non-zero Hessian with
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
        Construct the Hessian of this SPAM vector with respect to its parameters.

        This function returns a tensor whose first axis corresponds to the
        flattened gate matrix and whose 2nd and 3rd axes correspond to the
        parameters that are differentiated with respect to.

        Parameters
        ----------
        wrtFilter1, wrtFilter2 : list
            Lists of indices of the paramters to take first and second
            derivatives with respect to.  If None, then derivatives are
            taken with respect to all of the vectors's parameters.

        Returns
        -------
        numpy array
            Hessian with shape (dimension, num_params1, num_params2)
        """
        if not self.has_nonzero_hessian():
            return _np.zeros(self.size, self.num_params(), self.num_params())

        # FUTURE: create a finite differencing hessian method?
        raise NotImplementedError("hessian_wrt_params(...) is not implemented for %s objects" % self.__class__.__name__)


    #Pickle plumbing
    def __setstate__(self, state):
        self.__dict__.update(state)

    #Note: no __str__ fn

    @staticmethod
    def convert_to_vector(V):
        """
        Static method that converts a vector-like object to a 2D numpy
        dim x 1 column array.

        Parameters
        ----------
        V : array_like

        Returns
        -------
        numpy array
        """
        if isinstance(V, SPAMVec):
            vector = V.todense().copy()
            vector.shape = (vector.size,1)
        elif isinstance(V, _np.ndarray):
            vector = V.copy()
            if len(vector.shape) == 1: # convert (N,) shape vecs to (N,1)
                vector.shape = (vector.size,1)
        else:
            try:
                dim = len(V) #pylint: disable=unused-variable
            except:
                raise ValueError("%s doesn't look like an array/list" % V)
            try:
                d2s = [ len(row) for row in V ]
            except TypeError: # thrown if len(row) fails because no 2nd dim
                d2s = None

            if d2s is not None:
                if any([len(row) != 1 for row in V]):
                    raise ValueError("%s is 2-dimensional but 2nd dim != 1" % V)

                typ = 'd' if _np.all(_np.isreal(V)) else 'complex'
                try:
                    vector = _np.array(V, typ) #vec is already a 2-D column vector
                except TypeError:
                    raise ValueError("%s doesn't look like an array/list" % V)
            else:
                typ = 'd' if _np.all(_np.isreal(V)) else 'complex'
                vector = _np.array(V, typ)[:,None] # make into a 2-D column vec

        assert(len(vector.shape) == 2 and vector.shape[1] == 1)
        return vector

    
class DenseSPAMVec(SPAMVec):
    """
    Excapulates a parameterization of a state preparation OR POVM effect
    vector. This class is the  common base class for all specific
    parameterizations of a SPAM vector.
    """

    def __init__(self, vec, evotype):
        """ Initialize a new SPAM Vector """
        self.base = vec
        super(DenseSPAMVec, self).__init__(len(vec), evotype)

    def todense(self, scratch=None):
        """ 
        Return this SPAM vector as a (dense) numpy array.  The memory
        in `scratch` maybe used when it is not-None.
        """
        #don't use scratch since we already have memory allocated
        return _np.asarray(self.base[:,0])
          # *must* be a numpy array for Cython arg conversion

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
    
    #Access to underlying array
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


    #Mimic array
    def __pos__(self):        return self.base
    def __neg__(self):        return -self.base
    def __abs__(self):        return abs(self.base)
    def __add__(self,x):      return self.base + x
    def __radd__(self,x):     return x + self.base
    def __sub__(self,x):      return self.base - x
    def __rsub__(self,x):     return x - self.base
    def __mul__(self,x):      return self.base * x
    def __rmul__(self,x):     return x * self.base
    def __truediv__(self, x):  return self.base / x
    def __rtruediv__(self, x): return x / self.base
    def __floordiv__(self,x):  return self.base // x
    def __rfloordiv__(self,x): return x // self.base
    def __pow__(self,x):      return self.base ** x
    def __eq__(self,x):       return self.base == x
    def __len__(self):        return len(self.base)
    def __int__(self):        return int(self.base)
    def __long__(self):       return int(self.base)
    def __float__(self):      return float(self.base)
    def __complex__(self):    return complex(self.base)

    def __str__(self):
        s = "%s with dimension %d\n" % (self.__class__.__name__,self.dim)
        s += _mt.mx_to_string(self.todense(), width=4, prec=2)
        return s



class StaticSPAMVec(DenseSPAMVec):
    """
    Encapsulates a SPAM vector that is completely fixed, or "static", meaning
      that is contains no parameters.
    """

    def __init__(self, vec, evotype="auto"):
        """
        Initialize a StaticSPAMVec object.

        Parameters
        ----------
        vec : array_like or SPAMVec
            a 1D numpy array representing the SPAM operation.  The
            shape of this array sets the dimension of the SPAM op.
        """
        vec = SPAMVec.convert_to_vector(vec)
        if evotype == "auto":
            evotype = "statevec" if _np.iscomplexobj(vec) else "densitymx"
        elif evotype == "statevec":
            vec = _np.asarray(vec,complex) # ensure all statevec vecs are complex (densitymx could be either?)
            
        assert(evotype in ("statevec","densitymx")), \
            "Invalid evolution type '%s' for %s" % (evotype,self.__class__.__name__)
        DenseSPAMVec.__init__(self, vec, evotype)



class FullyParameterizedSPAMVec(DenseSPAMVec):
    """
    Encapsulates a SPAM vector that is fully parameterized, that is,
      each element of the SPAM vector is an independent parameter.
    """

    def __init__(self, vec, evotype="auto"):
        """
        Initialize a FullyParameterizedSPAMVec object.

        Parameters
        ----------
        vec : array_like or SPAMVec
            a 1D numpy array representing the SPAM operation.  The
            shape of this array sets the dimension of the SPAM op.
        """
        vec = SPAMVec.convert_to_vector(vec)
        if evotype == "auto":
            evotype = "statevec" if _np.iscomplexobj(vec) else "densitymx"
        assert(evotype in ("statevec","densitymx")), \
            "Invalid evolution type '%s' for %s" % (evotype,self.__class__.__name__)
        DenseSPAMVec.__init__(self, vec, evotype)


    def set_value(self, vec):
        """
        Attempts to modify SPAMVec parameters so that the specified raw
        SPAM vector becomes vec.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        vec : array_like or SPAMVec
            A numpy array representing a SPAM vector, or a SPAMVec object.

        Returns
        -------
        None
        """
        vec = SPAMVec.convert_to_vector(vec)
        if(vec.size != self.dim):
            raise ValueError("Argument must be length %d" % self.dim)
        self.base[:,:] = vec
        self.dirty = True


    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return 2*self.size if self._evotype == "statevec" else self.size


    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        if self._evotype == "statevec":
            return _np.concatenate( (self.base.real.flatten(), self.base.imag.flatten()), axis=0)
        else:
            return self.base.flatten()


    def from_vector(self, v):
        """
        Initialize the SPAM vector using a 1D array of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params()

        Returns
        -------
        None
        """
        if self._evotype == "statevec":
            self.base[:,0] = v[0:self.dim] + 1j*v[self.dim:]
        else:
            self.base[:,0] = v
        self.dirty = True


    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the derivatives of the SPAM vector
        with respect to a single param.  Thus, each column is of length
        get_dimension and there is one column per SPAM vector parameter.

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension, num_params)
        """
        if self._evotype == "statevec":
            derivMx = _np.concatenate( (_np.identity( self.dim, complex ),
                                        1j*_np.identity( self.dim, complex )), axis=1)
        else:
            derivMx = _np.identity( self.dim, 'd' )
            
        if wrtFilter is None:
            return derivMx
        else:
            return _np.take( derivMx, wrtFilter, axis=1 )


    def has_nonzero_hessian(self):
        """ 
        Returns whether this SPAM vector has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return False

    

class TPParameterizedSPAMVec(DenseSPAMVec):
    """
    Encapsulates a SPAM vector that is fully parameterized except for the first
    element, which is frozen to be 1/(d**0.25).  This is so that, when the SPAM
    vector is interpreted in the Pauli or Gell-Mann basis, the represented
    density matrix has trace == 1.  This restriction is frequently used in
    conjuction with trace-preserving (TP) gates.
    """

    #Note: here we assume that the first basis element is (1/sqrt(x) * I),
    # where I the d-dimensional identity (where len(vector) == d**2). So
    # if Tr(basisEl*basisEl) == Tr(1/x*I) == d/x must == 1, then we must
    # have x == d.  Thus, we multiply this first basis element by
    # alpha = 1/sqrt(d) to obtain a trace-1 matrix, i.e., finding alpha
    # s.t. Tr(alpha*[1/sqrt(d)*I]) == 1 => alpha*d/sqrt(d) == 1 =>
    # alpha = 1/sqrt(d) = 1/(len(vec)**0.25).
    def __init__(self, vec):
        """
        Initialize a TPParameterizedSPAMVec object.

        Parameters
        ----------
        vec : array_like or SPAMVec
            a 1D numpy array representing the SPAM operation.  The
            shape of this array sets the dimension of the SPAM op.
        """
        vector = SPAMVec.convert_to_vector(vec)
        firstEl =  len(vector)**-0.25
        if not _np.isclose(vector[0,0], firstEl):
            raise ValueError("Cannot create TPParameterizedSPAMVec: " +
                             "first element must equal %g!" % firstEl)
        DenseSPAMVec.__init__(self, _ProtectedArray(
            vector, indicesToProtect=(0,0)), "densitymx")


    def set_value(self, vec):
        """
        Attempts to modify SPAMVec parameters so that the specified raw
        SPAM vector becomes vec.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        vec : array_like or SPAMVec
            A numpy array representing a SPAM vector, or a SPAMVec object.

        Returns
        -------
        None
        """
        vec = SPAMVec.convert_to_vector(vec)
        firstEl =  (self.dim)**-0.25
        if(vec.size != self.dim):
            raise ValueError("Argument must be length %d" % self.dim)
        if not _np.isclose(vec[0,0], firstEl):
            raise ValueError("Cannot create TPParameterizedSPAMVec: " +
                             "first element must equal %g!" % firstEl)
        self.base[1:,:] = vec[1:,:]
        self.dirty = True

        
    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.dim-1


    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return self.base.flatten()[1:] #.real in case of complex matrices?


    def from_vector(self, v):
        """
        Initialize the SPAM vector using a 1D array of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params()

        Returns
        -------
        None
        """
        assert(_np.isclose(self.base[0,0], (self.dim)**-0.25))
        self.base[1:,0] = v
        self.dirty = True


    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the derivatives of the SPAM vector
        with respect to a single param.  Thus, each column is of length
        get_dimension and there is one column per SPAM vector parameter.

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension, num_params)
        """
        derivMx = _np.identity( self.dim, 'd' ) # TP vecs assumed real
        derivMx = derivMx[:,1:] #remove first col ( <=> first-el parameters )
        if wrtFilter is None:
            return derivMx
        else:
            return _np.take( derivMx, wrtFilter, axis=1 )


    def has_nonzero_hessian(self):
        """ 
        Returns whether this SPAM vector has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return False


class ComplementSPAMVec(DenseSPAMVec):
    """
    Encapsulates a SPAM vector that is parameterized by 
    `I - sum(other_spam_vecs)` where `I` is a (static) identity element 
    and `other_param_vecs` is a list of other spam vectors in the same parent
    :class:`POVM`.  This only *partially* implements the SPAMVec interface 
    (some methods such as `to_vector` and `from_vector` will thunk down to base
    class versions which raise `NotImplementedError`), as instances are meant to
    be contained within a :class:`POVM` which takes care of vectorization.
    """

    def __init__(self, identity, other_spamvecs):
        """
        Initialize a ComplementSPAMVec object.

        Parameters
        ----------
        identity : array_like or SPAMVec
            a 1D numpy array representing the static identity operation from
            which the sum of the other vectors is subtracted.

        other_spamvecs : list of SPAMVecs
            A list of the "other" parameterized SPAM vectors which are
            subtracted from `identity` to compute the final value of this 
            "complement" SPAM vector.
        """
        self.identity = FullyParameterizedSPAMVec(
            SPAMVec.convert_to_vector(identity) ) #so easy to transform
                                         # or depolarize by parent POVM
        
        self.other_vecs = other_spamvecs
          #Note: we assume that our parent will do the following:
          # 1) set our gpindices to indicate how many parameters we have
          # 2) set the gpindices of the elements of other_spamvecs so
          #    that they index into our local parameter vector.
            
        DenseSPAMVec.__init__(self, self.identity, "densitymx") # dummy
        self._construct_vector() #reset's self.base
        
    def _construct_vector(self):
        self.base = self.identity - sum(self.other_vecs)
        self.base.flags.writeable = False


    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return len(self.gpindices_as_array())

    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        raise ValueError(("ComplementSPAMVec.to_vector() should never be called"
                          " - use TPPOVM.to_vector() instead"))

    def from_vector(self, v):
        """
        Initialize this SPAM vector using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of parameters.

        Returns
        -------
        None
        """
        #Rely on prior .from_vector initialization of self.other_vecs, so
        # we just construct our vector based on them.
        #Note: this is needed for finite-differencing in map-based calculator
        self._construct_vector()

    
    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the derivatives of the SPAM vector
        with respect to a single param.  Thus, each column is of length
        get_dimension and there is one column per SPAM vector parameter.

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension, num_params)
        """
        if len(self.other_vecs) == 0: return _np.zeros((self.dim,0), 'd') # Complement vecs assumed real
        Np = len(self.gpindices_as_array())
        neg_deriv = _np.zeros( (self.dim, Np), 'd')
        for ovec in self.other_vecs:
            local_inds = _gatesetmember._decompose_gpindices(
                self.gpindices, ovec.gpindices)
              #Note: other_vecs are not copies but other *sibling* effect vecs
              # so their gpindices index the same space as this complement vec's
              # does - so we need to "_decompose_gpindices"
            neg_deriv[:,local_inds] += ovec.deriv_wrt_params()
        derivMx = -neg_deriv
        
        if wrtFilter is None:
            return derivMx
        else:
            return _np.take( derivMx, wrtFilter, axis=1 )
    
    def has_nonzero_hessian(self):
        """ 
        Returns whether this SPAM vector has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return False



class CPTPParameterizedSPAMVec(DenseSPAMVec):
    """
    Encapsulates a SPAM vector that is parameterized through the Cholesky
    decomposition of it's standard-basis representation as a density matrix
    (not a Liouville vector).  The resulting SPAM vector thus represents a
    positive density matrix, and additional constraints on the parameters
    also guarantee that the trace == 1.  This SPAM vector is meant for
    use with CPTP processes, hence the name.
    """

    def __init__(self, vec, basis, truncate=False):
        """
        Initialize a CPTPParameterizedSPAMVec object.

        Parameters
        ----------
        vec : array_like or SPAMVec
            a 1D numpy array representing the SPAM operation.  The
            shape of this array sets the dimension of the SPAM op.

        basis : {"std", "gm", "pp", "qt"} or Basis
            The basis `vec` is in.  Needed because this parameterization
            requires we construct the density matrix corresponding to
            the Lioville vector `vec`.

        trunctate : bool, optional
            Whether or not a non-positive, trace=1 `vec` should
            be truncated to force a successful construction.
        """
        vector = SPAMVec.convert_to_vector(vec)
        basis = _Basis(basis, int(round(_np.sqrt(len(vector)))))

        self.basis = basis
        self.basis_mxs = basis.get_composite_matrices()   #shape (len(vec), dmDim, dmDim)
        self.basis_mxs = _np.rollaxis(self.basis_mxs,0,3) #shape (dmDim, dmDim, len(vec))
        assert( self.basis_mxs.shape[-1] == len(vector) )

        # set self.params and self.dmDim
        self._set_params_from_vector(vector, truncate) 

        #scratch space
        self.Lmx = _np.zeros((self.dmDim,self.dmDim),'complex')

        DenseSPAMVec.__init__(self, vector, "densitymx")

    def _set_params_from_vector(self, vector, truncate):
        density_mx = _np.dot( self.basis_mxs, vector )
        density_mx = density_mx.squeeze() 
        dmDim = density_mx.shape[0]
        assert(dmDim == density_mx.shape[1]), "Density matrix must be square!"

        trc = _np.trace(density_mx)
        assert(truncate or _np.isclose(trc, 1.0)), \
            "`vec` must correspond to a trace-1 density matrix (truncate == False)!"
            
        if not _np.isclose(trc, 1.0): #truncate to trace == 1
            density_mx -= _np.identity(dmDim, 'd') / dmDim * (trc - 1.0)

        #push any slightly negative evals of density_mx positive
        # so that the Cholesky decomp will work.
        evals,U = _np.linalg.eig(density_mx)
        Ui = _np.linalg.inv(U)
    
        assert(truncate or all([ev >= -1e-12 for ev in evals])), \
            "`vec` must correspond to a positive density matrix (truncate == False)!"
    
        pos_evals = evals.clip(1e-16,1e100)
        density_mx = _np.dot(U,_np.dot(_np.diag(pos_evals),Ui))
        try:
            Lmx = _np.linalg.cholesky(density_mx)
        except _np.linalg.LinAlgError: #Lmx not postitive definite?
            pos_evals = evals.clip(1e-12,1e100) #try again with 1e-12
            density_mx = _np.dot(U,_np.dot(_np.diag(pos_evals),Ui))
            Lmx = _np.linalg.cholesky(density_mx)

        #check TP condition: that diagonal els of Lmx squared add to 1.0
        Lmx_norm = _np.trace(_np.dot(Lmx.T.conjugate(), Lmx)) # sum of magnitude^2 of all els
        assert(_np.isclose( Lmx_norm, 1.0 )), \
            "Cholesky decomp didn't preserve trace=1!"        
        
        self.dmDim = dmDim
        self.params = _np.empty( dmDim**2, 'd')
        for i in range(dmDim):
            assert(_np.linalg.norm(_np.imag(Lmx[i,i])) < IMAG_TOL)
            self.params[i*dmDim+i] = Lmx[i,i].real # / paramNorm == 1 as asserted above
            for j in range(i):
                self.params[i*dmDim+j] = Lmx[i,j].real
                self.params[j*dmDim+i] = Lmx[i,j].imag
        
        

    def _construct_vector(self):
        dmDim = self.dmDim
        
        #  params is an array of length dmDim^2-1 that
        #  encodes a lower-triangular matrix "Lmx" via:
        #  Lmx[i,i] = params[i*dmDim + i] / param-norm  # i = 0...dmDim-2
        #     *last diagonal el is given by sqrt(1.0 - sum(L[i,j]**2))
        #  Lmx[i,j] = params[i*dmDim + j] + 1j*params[j*dmDim+i] (i > j)

        param2Sum = _np.vdot(self.params, self.params) #or "dot" would work, since params are real
        paramNorm = _np.sqrt( param2Sum ) #also the norm of *all* Lmx els

        for i in range(dmDim):
            self.Lmx[i,i] = self.params[i*dmDim + i] / paramNorm
            for j in range(i):
                self.Lmx[i,j] = (self.params[i*dmDim+j] + 1j*self.params[j*dmDim+i]) / paramNorm

        Lmx_norm = _np.trace(_np.dot(self.Lmx.T.conjugate(), self.Lmx)) # sum of magnitude^2 of all els
        assert(_np.isclose(Lmx_norm, 1.0)), "Violated trace=1 condition!"        
                
        #The (complex, Hermitian) density matrix is build by
        # assuming Lmx is its Cholesky decomp, which makes
        # the density matrix is pos-def.
        density_mx = _np.dot(self.Lmx,self.Lmx.T.conjugate())
        assert( _np.isclose(_np.trace(density_mx), 1.0 )), "density matrix must be trace == 1"

        # write density matrix in given basis: = sum_i alpha_i B_i
        # ASSUME that basis is orthogonal, i.e. Tr(Bi^dag*Bj) = delta_ij
        basis_mxs = _np.rollaxis(self.basis_mxs,2) #shape (dmDim, dmDim, len(vec))
        vec = _np.array( [ _np.trace(_np.dot(M.T.conjugate(), density_mx)) for M in basis_mxs ] )

        #for now, assume Liouville vector should always be real (TODO: add 'real' flag later?)
        assert(_np.linalg.norm(_np.imag(vec)) < IMAG_TOL)
        vec = _np.real(vec)
        
        self.base = vec[:,None] # so shape is (dim,1) - the convention for spam vectors
        self.base.flags.writeable = False


    def set_value(self, vec):
        """
        Attempts to modify SPAMVec parameters so that the specified raw
        SPAM vector becomes vec.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        vec : array_like or SPAMVec
            A numpy array representing a SPAM vector, or a SPAMVec object.

        Returns
        -------
        None
        """
        try:
            self._set_params_from_vector(vec, truncate=False)
            self.dirty = True
        except AssertionError as e:
            raise ValueError("Error initializing the parameters of this " +
                         " CPTPParameterizedSPAMVec object: " + str(e))
        
    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        assert(self.dmDim**2 == self.dim) #should at least be true without composite bases...
        return self.dmDim**2


    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return self.params


    def from_vector(self, v):
        """
        Initialize the SPAM vector using a 1D array of parameters.

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
        self.params[:] = v[:]
        self._construct_vector()
        self.dirty = True

    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the derivatives of the SPAM vector
        with respect to a single param.  Thus, each column is of length
        get_dimension and there is one column per SPAM vector parameter.

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension, num_params)
        """
        dmDim = self.dmDim
        nP = len(self.params)
        assert(nP == dmDim**2) #number of parameters
        
        # v_i = trace( B_i^dag * Lmx * Lmx^dag )
        # d(v_i) = trace( B_i^dag * (dLmx * Lmx^dag + Lmx * (dLmx)^dag) )  #trace = linear so commutes w/deriv
        #               / 
        # where dLmx/d[ab] = {
        #               \
        L,Lbar = self.Lmx,self.Lmx.conjugate()
        F1 = _np.tril(_np.ones((dmDim,dmDim),'d'))
        F2 = _np.triu(_np.ones((dmDim,dmDim),'d'),1) * 1j
        conj_basis_mxs = self.basis_mxs.conjugate()
        
          # Derivative of vector wrt params; shape == [vecLen,dmDim,dmDim] *not dealing with TP condition yet*
          # (first get derivative assuming last diagonal el of Lmx *is* a parameter, then use chain rule)
        dVdp  = _np.einsum('aml,mb,ab->lab', conj_basis_mxs, Lbar, F1) #only a >= b nonzero (F1)
        dVdp += _np.einsum('mal,mb,ab->lab', conj_basis_mxs, L, F1)    # ditto
        dVdp += _np.einsum('bml,ma,ab->lab', conj_basis_mxs, Lbar, F2) #only b > a nonzero (F2)
        dVdp += _np.einsum('mbl,ma,ab->lab', conj_basis_mxs, L, F2.conjugate()) # ditto

        dVdp.shape = [ dVdp.shape[0], nP ] # jacobian with respect to "p" params,
                            # which don't include normalization for TP-constraint

        #Now get jacobian of actual params wrt the params used above. Denote the actual
        # params "P" in variable names, so p_ij = P_ij / sqrt(sum(P_xy**2))
        param2Sum = _np.vdot(self.params, self.params)
        paramNorm = _np.sqrt( param2Sum ) #norm of *all* Lmx els (note lastDiagEl
        dpdP = _np.identity(nP, 'd' )
        
        # all p_ij params ==  P_ij / paramNorm = P_ij / sqrt(sum(P_xy**2))
        # and so have derivs wrt *all* Pxy elements.
        for ij in range(nP):
            for kl in range(nP):
                if ij == kl:  # dp_ij / dP_ij = 1.0 / (sum(P_xy**2))^(1/2) - 0.5 * P_ij / (sum(P_xy**2))^(3/2) * 2*P_ij
                              #               = 1.0 / (sum(P_xy**2))^(1/2) - P_ij^2 / (sum(P_xy**2))^(3/2)
                    dpdP[ij,ij] = 1.0/paramNorm - self.params[ij]**2 / paramNorm**3
                else:   # dp_ij / dP_kl = -0.5 * P_ij / (sum(P_xy**2))^(3/2) * 2*P_kl
                        #               = - P_ij * P_kl / (sum(P_xy**2))^(3/2)
                    dpdP[ij,kl] = - self.params[ij] * self.params[kl] / paramNorm**3

        #Apply the chain rule to get dVdP:
        dVdP = _np.dot(dVdp, dpdP) #shape (vecLen, nP) - the jacobian!
        dVdp = dpdP = None # free memory!

        assert(_np.linalg.norm(_np.imag(dVdP)) < IMAG_TOL)
        derivMx = _np.real(dVdP)
        
        if wrtFilter is None:
            return derivMx
        else:
            return _np.take( derivMx, wrtFilter, axis=1 )


    def has_nonzero_hessian(self):
        """ 
        Returns whether this SPAM vector has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return True

    def hessian_wrt_params(self, wrtFilter1=None, wrtFilter2=None):
        """
        Construct the Hessian of this SPAM vector with respect to its parameters.

        This function returns a tensor whose first axis corresponds to the
        flattened gate matrix and whose 2nd and 3rd axes correspond to the
        parameters that are differentiated with respect to.

        Parameters
        ----------
        wrtFilter1, wrtFilter2 : list
            Lists of indices of the paramters to take first and second
            derivatives with respect to.  If None, then derivatives are
            taken with respect to all of the vectors's parameters.

        Returns
        -------
        numpy array
            Hessian with shape (dimension, num_params1, num_params2)
        """
        raise NotImplementedError("TODO: add hessian computation for CPTPParameterizedSPAMVec")
    
    

class TensorProdSPAMVec(SPAMVec):
    """
    Encapsulates a SPAM vector that is a tensor-product of other SPAM vectors.
    """

    def __init__(self, typ, factors, povmEffectLbls=None):
        """
        Initialize a TensorProdSPAMVec object.

        Parameters
        ----------
        typ : {"prep","effect"}
            The type of spam vector - either a product of preparation
            vectors ("prep") or of POVM effect vectors ("effect") 

        factors : list of SPAMVecs or POVMs
            if `typ == "prep"`, a list of the component SPAMVecs; if 
            `typ == "effect"`, a list of "reference" POVMs into which 
            `povmEffectLbls` indexes.

        povmEffectLbls : array-like
            Only non-None when `typ == "effect"`.  The effect label of each
            factor POVM which is tensored together to form this SPAM vector.
        """
        assert(len(factors) > 0), "Must have at least one factor!"
        
        self.typ = typ
        self.factors = factors #do *not* copy - needs to reference common objects
        self.Np = sum([fct.num_params() for fct in factors])
        if typ == "effect":
            self.effectLbls = _np.array(povmEffectLbls)                
        elif typ == "prep":
            assert(povmEffectLbls is None), '`povmEffectLbls` must be None when `typ != "effects"`'
            self.effectLbls = None                        
        else: raise ValueError("Invalid `typ` argument: %s" % typ)

        SPAMVec.__init__(self, _np.product([fct.dim for fct in factors]),
                         self.factors[0]._evotype)
          #sets gpindices, so do before stuff below

        if typ == "effect":
            #Set our parent and gpindices based on those of factor-POVMs, which
            # should all be owned by a TensorProdPOVM object.
            # (for now say we depend on *all* the POVMs parameters (even though
            #  we really only depend on one element of each POVM, which may allow
            #  using just a subset of each factor POVMs indices - but this is tricky).
            self.set_gpindices(_slct.list_to_slice(
                _np.concatenate( [ fct.gpindices_as_array()
                                   for fct in factors ], axis=0),True,False),
                               factors[0].parent) #use parent of first factor
                                                  # (they should all be the same)
        else:
            # don't init our own gpindices (prep case), since our parent
            # is likely to be a GateSet and it will init them correctly.
            #But do set the indices of self.factors, since they're now 
            # considered "owned" by this product-prep-vec (different from
            # the "effect" case when the factors are shared).
            off = 0
            for fct in factors:
                assert(isinstance(fct,SPAMVec)),"Factors must be SPAMVec objects!"
                N = fct.num_params()
                fct.set_gpindices( slice(off,off+N), self ); off += N
            assert(off == self.Np)

        #Memory for speeding up kron product in todense()
        if self._evotype in ("statevec","densitymx"): #types that require fast kronecker prods
            max_factor_dim = max(fct.dim for fct in factors)
            self._fast_kron_array = _np.empty( (len(factors), max_factor_dim), complex if self._evotype == "statevec" else 'd')
            self._fast_kron_factordims = _np.array([fct.dim for fct in factors],_np.int64)
            try:
                self._fill_fast_kron()
            except NotImplementedError: # if todense() or any other prereq isn't implemented (
                self._fast_kron_array = None   # e.g. if factors are LindbladTermSPAMVecs
                self._fast_kron_factordims = None
                
        else:
            self._fast_kron_array = None
            self._fast_kron_factordims = None

        
    def _fill_fast_kron(self):
        """ Fills in self._fast_kron_array based on current self.factors """
        if self.typ == "prep":
            for i,factor_dim in enumerate(self._fast_kron_factordims):
                self._fast_kron_array[i][0:factor_dim] = self.factors[i].todense()
        else:
            factorPOVMs = self.factors
            for i,(factor_dim,Elbl) in enumerate(zip(self._fast_kron_factordims,self.effectLbls)):
                self._fast_kron_array[i][0:factor_dim] = factorPOVMs[i][Elbl].todense()


    def todense(self):
        """
        Return this SPAM vector as a (dense) numpy array.  The memory
        in `scratch` maybe used when it is not-None.
        """
        if self._evotype in ("statevec","densitymx"):
            if len(self.factors) == 0: return _np.empty(0,complex if self._evotype == "statevec" else 'd')
            #NOTE: moved a fast version of todense to replib - could use that if need a fast todense call...
                
            if self.typ == "prep":
                ret = self.factors[0].todense() # factors are just other SPAMVecs
                for i in range(1,len(self.factors)):
                    ret = _np.kron(ret, self.factors[i].todense())
            else:
                factorPOVMs = self.factors
                ret = factorPOVMs[0][self.effectLbls[0]].todense()
                for i in range(1,len(factorPOVMs)):
                    ret = _np.kron(ret, factorPOVMs[i][self.effectLbls[i]].todense())
    
            return ret            
        elif self._evotype == "stabilizer":
            
            if self.typ == "prep":
                # => self.factors should all be StabilizerSPAMVec objs
                #Return stabilizer-rep tuple, just like StabilizerSPAMVec
                sframe_factors = [ f.todense() for f in self.factors ]
                return _stabilizer.sframe_kronecker(sframe_factors)

            else: #self.typ == "effect", so each factor is a StabilizerEffectVec
                raise ValueError("Cannot convert Stabilizer tensor product effect to an array!")
                # should be using effect.outcomes property...
        else: # self._evotype in ("svterm","cterm")
            raise NotImplementedError("todense() not implemented for %s evolution type" % self._evotype)


    def torep(self, typ, outrep=None):
        """
        Return a "representation" object for this SPAM vector.

        Such objects are primarily used internally by pyGSTi to compute
        things like probabilities more efficiently.

        Parameters
        ----------
        typ : {'prep','effect'}
            The type of representation (for cases when the vector type is
            not already defined).

        outrep : StateRep
            If not None, an existing state representation appropriate to this
            SPAM vector that may be used instead of allocating a new one.

        Returns
        -------
        StateRep
        """
        assert(len(self.factors) > 0), "Cannot get representation of a TensorProdSPAMVec with no factors!"
        assert(self.typ in ('prep','effect')), "Invalid internal type: %s!" % self.typ
        
        #FUTURE: use outrep as scratch for rep constructor?
        if self._evotype == "statevec":
            if self.typ == "prep": # prep-type vectors can be represented as dense effects too
                if typ == "prep":
                    return replib.SVStateRep( self.todense() )
                else:
                    return replib.SVEffectRep_Dense( self.todense() )
            else:
                return replib.SVEffectRep_TensorProd(self._fast_kron_array, self._fast_kron_factordims,
                                                     len(self.factors), self._fast_kron_array.shape[1], self.dim)
        elif self._evotype == "densitymx":
            if self.typ == "prep":
                if typ == "prep":
                    return replib.DMStateRep( self.todense() )
                else:
                    return replib.DMEffectRep_Dense( self.todense() )

            else:
                return replib.DMEffectRep_TensorProd(self._fast_kron_array, self._fast_kron_factordims,
                                                     len(self.factors), self._fast_kron_array.shape[1], self.dim)
        
        elif self._evotype == "stabilizer":
            if self.typ == "prep": # prep-type vectors can be represented as dense effects too; this just
                                   # means that self.factors 
                if typ == "prep":
                    # => self.factors should all be StabilizerSPAMVec objs
                    #Return stabilizer-rep tuple, just like StabilizerSPAMVec
                    sframe_factors = [ f.todense() for f in self.factors ] # StabilizerFrame for each factor
                    return _stabilizer.sframe_kronecker(sframe_factors).torep()
                else: #self.typ == "effect", so each factor is a StabilizerEffectVec
                    outcomes = _np.array( list(_itertools.chain(*[f.outcomes for f in self.factors])), _np.int64)
                    return replib.SBEffectRep(outcomes)

            else: #self.typ == "effect", so each factor is a StabilizerZPOVM
                # like above, but get a StabilizerEffectVec from each StabilizerZPOVM factor
                factorPOVMs = self.factors
                factorVecs = [ factorPOVMs[i][self.effectLbls[i]] for i in range(1,len(factorPOVMs)) ]
                outcomes = _np.array( list(_itertools.chain(*[f.outcomes for f in factorVecs])), _np.int64)
                return replib.SBEffectRep(outcomes)
                
                #OLD - now can remove outcomes prop?
                #raise ValueError("Cannot convert Stabilizer tensor product effect to a representation!")
                # should be using effect.outcomes property...
                
        else: # self._evotype in ("svterm","cterm")
            raise NotImplementedError("torep() not implemented for %s evolution type" % self._evotype)
        

    def get_order_terms(self, order):
        """ 
        Get the `order`-th order Taylor-expansion terms of this SPAM vector.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that it is a state
        preparation followed by or POVM effect preceded by actions on a
        density matrix `rho` of the form:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the 
        SPAMVec's parameters, where the polynomial's variable indices index the
        *global* parameters of the SPAMVec's parent (usually a :class:`GateSet`)
        , not the SPAMVec's local parameter array (i.e. that returned from
        `to_vector`).


        Parameters
        ----------
        order : int
            The order of terms to get.

        Returns
        -------
        list
            A list of :class:`RankOneTerm` objects.
        """
        if self._evotype == "svterm": tt = "dense"
        elif self._evotype == "cterm": tt = "clifford"
        else: raise ValueError("Invalid evolution type %s for calling `get_order_terms`" % self._evotype)
        
        from .gate import EmbeddedGateMap as _EmbeddedGateMap
        terms = []
        fnq = [ int(round(_np.log2(f.dim)))//2 for f in self.factors ] # num of qubits per factor
          # assumes density matrix evolution
        total_nQ = sum(fnq) # total number of qubits
        
        for p in _lt.partition_into(order, len(self.factors)):
            if self.typ == "prep":
                factor_lists = [self.factors[i].get_order_terms(pi) for i,pi in enumerate(p)]
            else:
                factorPOVMs = self.factors
                factor_lists = [ factorPOVMs[i][Elbl].get_order_terms(pi)
                                 for i,(pi,Elbl) in enumerate(zip(p,self.effectLbls)) ]

            # When possible, create COLLAPSED factor_lists so each factor has just a single
            # (SPAMVec) pre & post op, which can be formed into the new terms'
            # TensorProdSPAMVec ops.
            # - DON'T collapse stabilizer states & clifford ops - can't for POVMs
            collapsible = False # bool(self._evotype =="svterm") # need to use reps for collapsing now... TODO?
            
            if collapsible:
                factor_lists = [ [t.collapse_vec() for t in fterms] for fterms in factor_lists]
                
            for factors in _itertools.product(*factor_lists):
                # create a term with a TensorProdSPAMVec - Note we always create
                # "prep"-mode vectors, since even when self.typ == "effect" these
                # vectors are created with factor (prep- or effect-type) SPAMVecs not factor POVMs
                # we workaround this by still allowing such "prep"-mode
                # TensorProdSPAMVecs to be represented as effects (i.e. in torep('effect'...) works)
                coeff = _functools.reduce(lambda x,y: x.mult(y), [f.coeff for f in factors])
                pre_op = TensorProdSPAMVec("prep", [f.pre_ops[0] for f in factors
                                                      if (f.pre_ops[0] is not None)])
                post_op = TensorProdSPAMVec("prep", [f.post_ops[0] for f in factors
                                                       if (f.post_ops[0] is not None)])
                term = _term.RankOneTerm(coeff, pre_op, post_op, tt)

                if not collapsible: # then may need to add more ops.  Assume factor ops are clifford gates
                    # Embed each factors ops according to their target qubit(s) and just daisy chain them
                    stateSpaceLabels = tuple(range(total_nQ)); curQ=0
                    for f,nq in zip(factors,fnq):
                        targetLabels = tuple(range(curQ,curQ+nq)); curQ += nq
                        term.pre_ops.extend( [ _EmbeddedGateMap(stateSpaceLabels,targetLabels,op)
                                               for op in f.pre_ops[1:] ] ) # embed and add ops
                        term.post_ops.extend( [ _EmbeddedGateMap(stateSpaceLabels,targetLabels,op)
                                               for op in f.post_ops[1:] ] ) # embed and add ops
                
                terms.append(term)
                    
        return terms # Cache terms in FUTURE?

    
    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.Np


    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        if self.typ == "prep":
            return _np.concatenate( [fct.to_vector() for fct in self.factors], axis=0)
        else:
            raise ValueError(("'`to_vector` should not be called on effect-like"
                              " TensorProdSPAMVecs (instead it should be called"
                              " on the POVM)"))

    def from_vector(self, v):
        """
        Initialize the SPAM vector using a 1D array of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params()

        Returns
        -------
        None
        """
        if self.typ == "prep":
            for sv in self.factors:
                sv.from_vector( v[sv.gpindices] ) # factors hold local indices
                
        elif all([self.effectLbls[i] == list(povm.keys())[0]
                  for i,povm in enumerate(self.factors)]):
            #then this is the *first* vector in the larger TensorProdPOVM
            # and we should initialize all of the factorPOVMs
            for povm in self.factors:
                local_inds = _gatesetmember._decompose_gpindices(
                    self.gpindices, povm.gpindices)
                povm.from_vector( v[local_inds] )

        #No need to construct anything except fast-kron array, as
        # no dense matrices are stored
        if self._fast_kron_array is not None: # if it's in use...
            self._fill_fast_kron()


    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the derivatives of the SPAM vector
        with respect to a single param.  Thus, each column is of length
        get_dimension and there is one column per SPAM vector parameter.
        An empty 2D array in the StaticSPAMVec case (num_params == 0).

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension, num_params)
        """
        assert(self._evotype in ("statevec","densitymx"))
        typ = complex if self._evotype == "statevec" else 'd'
        derivMx = _np.zeros( (self.dim, self.num_params()), typ)
        
        #Product rule to compute jacobian
        for i,fct in enumerate(self.factors): # loop over the spamvec/povm we differentiate wrt
            vec = fct if (self.typ == "prep") else fct[self.effectLbls[i]]

            if vec.num_params() == 0: continue #no contribution
            deriv = vec.deriv_wrt_params(None) #TODO: use filter?? / make relative to this gate...
            deriv.shape = (vec.dim,vec.num_params())

            if i > 0: # factors before ith
                if self.typ == "prep":
                    pre = self.factors[0].todense()
                    for vecA in self.factors[1:i]:
                        pre = _np.kron(pre,vecA.todense())
                else:
                    pre = self.factors[0][self.effectLbls[0]].todense()
                    for j,fctA in enumerate(self.factors[1:i],start=1):
                        pre = _np.kron(pre,fctA[self.effectLbls[j]].todense())
                deriv = _np.kron(pre[:,None], deriv) # add a dummy 1-dim to 'pre' and do kron properly...

            if i+1 < len(self.factors): # factors after ith
                if self.typ == "prep":
                    post = self.factors[i+1].todense()
                    for vecA in self.factors[i+2:]:
                        post = _np.kron(post,vecA.todense())
                else:
                    post = self.factors[i+1][self.effectLbls[i+1]].todense()
                    for j,fctA in enumerate(self.factors[i+2:],start=i+2):
                        post = _np.kron(post,fctA[self.effectLbls[j]].todense())
                deriv = _np.kron(deriv, post[:,None]) # add a dummy 1-dim to 'post' and do kron properly...

            if self.typ == "prep":
                local_inds = fct.gpindices # factor vectors hold local indices
            else: # in effect case, POVM-factors hold global indices (b/c they're meant to be shareable)
                local_inds = _gatesetmember._decompose_gpindices(
                    self.gpindices, fct.gpindices)

            assert(local_inds is not None), \
                "Error: gpindices has not been initialized for factor %d - cannot compute derivative!" % i
            derivMx[:,local_inds] += deriv

        derivMx.shape = (self.dim, self.num_params()) # necessary?
        if wrtFilter is None:
            return derivMx
        else:
            return _np.take( derivMx, wrtFilter, axis=1 )


    def has_nonzero_hessian(self):
        """ 
        Returns whether this SPAM vector has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return False


    def __str__(self):
        s = "Tensor product %s vector with length %d\n" % (self.typ,self.dim)
        #ar = self.todense()
        #s += _mt.mx_to_string(ar, width=4, prec=2)

        if self.typ == "prep":
            # factors are just other SPAMVecs
            s += " x ".join([_mt.mx_to_string(fct.todense(), width=4, prec=2) for fct in self.factors])
        else:
            # factors are POVMs
            s += " x ".join([_mt.mx_to_string(fct[self.effectLbls[i]].todense(), width=4, prec=2)
                             for i,fct in enumerate(self.factors)])
        return s



class PureStateSPAMVec(SPAMVec):
    """
    Encapsulates a SPAM vector that is a pure state but evolves according to
    one of the density matrix evolution types ("denstiymx", "svterm", and
    "cterm").  It is parameterized by a contained pure-state SPAMVec which
    evolves according to a state vector evolution type ("statevec" or
    "stabilizer").
    """

    def __init__(self, pure_state_vec, evotype='densitymx', dm_basis='pp'):
        """
        Initialize a PureStateSPAMVec object.

        Parameters
        ----------
        pure_state_vec : array_like or SPAMVec
            a 1D numpy array or object representing the pure state.  This object
            sets the parameterization and dimension of this SPAM vector (if 
            `pure_state_vec`'s dimension is `d`, then this SPAM vector's
            dimension is `d^2`).  Assumed to be a complex vector in the 
            standard computational basis.

        evotype : {'densitymx', 'svterm', 'cterm'}
            The evolution type of this SPAMVec.  Note that the evotype of
            `pure_state_vec` must be compatible with this value.  In particular,
            `pure_state_vec` must have an evotype of `"statevec"` (then allowed
            values are `"densitymx"` and `"svterm"`) or `"stabilizer"` (then 
            the only allowed value is `"cterm"`).

        dm_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The basis for this SPAM vector - that is, for the *density matrix*
            corresponding to `pure_state_vec`.  Allowed values are Matrix-unit
            (std),  Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
            (or a custom basis object).
        """
        if not isinstance(pure_state_vec,SPAMVec):
            pure_state_vec = StaticSPAMVec(SPAMVec.convert_to_vector(pure_state_vec),'statevec')
        self.pure_state_vec = pure_state_vec
        self.basis = dm_basis # only used for dense conversion

        pure_evo = pure_state_vec._evotype
        if pure_evo == "statevec":
            if evotype not in ("densitymx","svterm"):
                raise ValueError(("`evotype` arg must be 'densitymx' or 'svterm'"
                                  " when `pure_state_vec` evotype is 'statevec'"))
        elif pure_evo == "stabilizer":
            if evotype not in ("cterm",):
                raise ValueError(("`evotype` arg must be 'densitymx' or 'svterm'"
                                  " when `pure_state_vec` evotype is 'statevec'"))
        else:
            raise ValueError("`pure_state_vec` evotype must be 'statevec' or 'stabilizer' (not '%s')" % pure_evo)

        SPAMVec.__init__(self, self.pure_state_vec.dim**2, evotype)

        
    def todense(self, scratch=None):
        """
        Return this SPAM vector as a (dense) numpy array.  The memory
        in `scratch` maybe used when it is not-None.
        """
        assert(self._evotype == "densitymx"), \
            "todense(...) is only implemented for the 'densitymx' evotype"
        dmVec_std = _gt.state_to_dmvec( self.pure_state_vec.todense() )
        return _bt.change_basis(dmVec_std, 'std', self.basis)


    def get_order_terms(self, order):
        """ 
        Get the `order`-th order Taylor-expansion terms of this SPAM vector.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that it is a state
        preparation followed by or POVM effect preceded by actions on a
        density matrix `rho` of the form:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the 
        SPAMVec's parameters, where the polynomial's variable indices index the
        *global* parameters of the SPAMVec's parent (usually a :class:`GateSet`)
        , not the SPAMVec's local parameter array (i.e. that returned from
        `to_vector`).


        Parameters
        ----------
        order : int
            The order of terms to get.

        Returns
        -------
        list
            A list of :class:`RankOneTerm` objects.
        """
        if self.num_params() > 0:
            raise ValueError(("PureStateSPAMVec.get_order_terms(...) is only "
                              "implemented for the case when its underlying "
                              "pure state vector has 0 parameters (is static)"))
        
        if order == 0: # only 0-th order term exists (assumes static pure_state_vec)
            if self._evotype == "svterm":  tt = "dense"
            elif self._evotype == "cterm": tt = "clifford"
            else: raise ValueError("Invalid evolution type %s for calling `get_order_terms`" % self._evotype)

            purevec = self.pure_state_vec
            terms = [ _term.RankOneTerm(_Polynomial({(): 1.0}), purevec, purevec, tt) ]
        else:
            terms = []
        return terms

        
    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.pure_state_vec.num_params()


    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return self.pure_state_vec.to_vector()


    def from_vector(self, v):
        """
        Initialize the SPAM vector using a 1D array of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params()

        Returns
        -------
        None
        """
        self.pure_state_vec.from_vector(v)


    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the derivatives of the SPAM vector
        with respect to a single param.  Thus, each column is of length
        get_dimension and there is one column per SPAM vector parameter.
        An empty 2D array in the StaticSPAMVec case (num_params == 0).

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension, num_params)
        """
        raise NotImplementedError("Still need to work out derivative calculation of PureStateSPAMVec")


    def has_nonzero_hessian(self):
        """ 
        Returns whether this SPAM vector has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return self.pure_state_vec.has_nonzero_hessian()


    def __str__(self):
        s = "Pure-state spam vector with length %d holding:\n" % self.dim
        s += "  " + str(self.pure_state_vec)
        return s


class LindbladParameterizedSPAMVec(SPAMVec):
    """ A Lindblad-parameterized SPAMVec (that is also expandable into terms)"""

    @classmethod
    def from_spamvec_obj(cls, spamvec, typ, paramType="GLND", purevec=None,
                         proj_basis="pp", mxBasis="pp", truncate=True,
                         lazy=False):
        """
        Creates a LindbladParameterizedSPAMVec from an existing SPAMVec object
        and some additional information.  

        This function is different from `from_spam_vector` in that it assumes
        that `spamvec` is a :class:`SPAMVec`-derived object, and if `lazy=True`
        and if `spamvec` is already a matching LindbladParameterizedSPAMVec, it
        is returned directly.  This routine is primarily used in spam vector
        conversion functions, where conversion is desired only when necessary.

        Parameters
        ----------
        spamvec : SPAMVec
            The spam vector object to "convert" to a 
            `LindbladParameterizedSPAMVec`.

        typ : {"prep","effect"}
            Whether this is a state preparation or POVM effect vector.

        paramType : str, optional
            The high-level "parameter type" of the gate to create.  This 
            specifies both which Lindblad parameters are included and what
            type of evolution is used.  Examples of valid values are
            `"CPTP"`, `"H+S"`, `"S terms"`, and `"GLND clifford terms"`.

        purevec : numpy array or SPAMVec object, optional
            A SPAM vector which represents a pure-state, taken as the "ideal"
            reference state when constructing the error generator of the
            returned `LindbladParameterizedSPAMVec`.  Note that this vector
            still acts on density matrices (if it's a SPAMVec it should have 
            a "densitymx", "svterm", or "cterm" evolution type, and if it's
            a numpy array it should have the same dimension as `spamvec`).
            If None, then it is taken to be `spamvec`, and so `spamvec` must 
            represent a pure state in this case.
            
        proj_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis used to construct the Lindblad-term error generators onto 
            which the SPAM vector's error generator is projected.  Allowed values
            are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        truncate : bool, optional
            Whether to truncate the projections onto the Lindblad terms in
            order to meet constraints (e.g. to preserve CPTP) when necessary.
            If False, then an error is thrown when the given `spamvec` cannot
            be realized by the specified set of Lindblad projections.
            
        lazy : bool, optional
            If True, then if `spamvec` is already a LindbladParameterizedSPAMVec
            with the requested details (given by the other arguments), then
            `spamvec` is returned directly and no conversion/copying is
            performed. If False, then a new object is always returned.

        Returns
        -------
        LindbladParameterizedSPAMVec
        """

        if not isinstance(spamvec, SPAMVec):
            spamvec = StaticSPAMVec(spamvec) #assume spamvec is just a vector

        if purevec is None:
            purevec = spamvec # right now, we don't try to extract a "closest pure vec"
                              # to spamvec - below will fail if spamvec isn't pure.
        elif not isinstance(purevec, SPAMVec):
            purevec = StaticSPAMVec(purevec) #assume spamvec is just a vector

        #Break paramType in to a "base" type and an evotype
        from .gate import LindbladParameterizedGateMap as _LPGMap
        bTyp, evotype, nonham_mode, param_mode = _LPGMap.decomp_paramtype(paramType)

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

        if isinstance(spamvec, LindbladParameterizedSPAMVec) \
           and spamvec._evotype == evotype and spamvec.typ == typ \
           and beq(ham_basis,spamvec.error_map.ham_basis) and beq(nonham_basis,spamvec.error_map.other_basis) \
           and param_mode==spamvec.error_map.param_mode and nonham_mode==spamvec.error_map.nonham_mode \
           and beq(mxBasis,spamvec.error_map.matrix_basis) and lazy:
           #normeq(gate.pure_state_vec,purevec) \ # TODO: more checks for equality?!
            return spamvec #no creation necessary!
        else:
            #Convert vectors (if possible) to SPAMVecs
            # of the appropriate evotype and 0 params.
            bDiff = spamvec is not purevec
            spamvec = _convert_to_lindblad_base(spamvec, typ, evotype, mxBasis) 
            purevec = _convert_to_lindblad_base(purevec, typ, evotype, mxBasis) if bDiff else spamvec
            assert(spamvec._evotype == evotype)
            assert(purevec._evotype == evotype)
            
            return cls.from_spam_vector(
                spamvec, purevec, typ, ham_basis, nonham_basis,
                param_mode, nonham_mode, truncate, mxBasis, evotype)

    
    @classmethod
    def from_spam_vector(cls, spamVec, pureVec, typ,
                         ham_basis="pp", nonham_basis="pp", param_mode="cptp",
                         nonham_mode="all", truncate=True, mxBasis="pp",
                         evotype="densitymx"):
        """ 
        Creates a Lindblad-parameterized spamvec from a state vector and a basis
        which specifies how to decompose (project) the vector's error generator.

        spamVec : SPAMVec
            the SPAM vector to initialize from.  The error generator that
            tranforms `pureVec` into `spamVec` forms the parameterization
            of the returned LindbladParameterizedSPAMVec.
            
        pureVec : numpy array or SPAMVec
            An array or SPAMVec in the *full* density-matrix space (this
            vector will have the same dimension as `spamVec` - 4 in the case
            of a single qubit) which represents a pure-state preparation or
            projection.  This is used as the "base" preparation/projection
            when computing the error generator that will be parameterized.
            Note that this argument must be specified, as there is no natural
            default value (like the identity in the case of gates).

        typ : {"prep","effect"}
            Whether this is a state preparation or POVM effect vector.

        ham_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis is used to construct the Hamiltonian-type lindblad error
            Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        nonham_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis is used to construct the Stochastic-type lindblad error
            Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        param_mode : {"unconstrained", "cptp", "depol", "reldepol"}
            Describes how the Lindblad coefficients/projections relate to the
            SPAM vector's parameter values.  Allowed values are:
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
            The evolution type of the spamvec being constructed.  `"densitymx"` is
            usual Lioville density-matrix-vector propagation via matrix-vector
            products.  `"svterm"` denotes state-vector term-based evolution
            (spamvec is obtained by evaluating the rank-1 terms up to
            some order).  `"cterm"` is similar but stabilizer states.

        Returns
        -------
        LindbladParameterizedSPAMVec
        """
        #Compute a (errgen, pureVec) pair from the given
        # (spamVec, pureVec) pair.

        assert(pureVec is not None), "Must supply `pureVec`!" # since there's no good default?
        
        if not isinstance(spamVec, SPAMVec):
            spamVec = StaticSPAMVec(spamVec, evotype) #assume spamvec is just a vector
        if not isinstance(pureVec, SPAMVec):
            pureVec = StaticSPAMVec(pureVec, evotype) #assume spamvec is just a vector
        d2 = pureVec.dim

        #Determine whether we're using sparse bases or not
        sparse = None
        if ham_basis is not None:
            if isinstance(ham_basis, _Basis): sparse = ham_basis.sparse
            elif not _compat.isstr(ham_basis) and len(ham_basis) > 0:
                sparse = _sps.issparse(ham_basis[0])
        if sparse is None and nonham_basis is not None:
            if isinstance(nonham_basis, _Basis): sparse = nonham_basis.sparse
            elif not _compat.isstr(nonham_basis) and len(nonham_basis) > 0:
                sparse = _sps.issparse(nonham_basis[0])
        if sparse is None: sparse = False #the default

        if spamVec is None or spamVec is pureVec:
            if sparse: errgen = _sps.csr_matrix((d2,d2), dtype='d')
            else:      errgen = _np.zeros((d2,d2),'d')
        else:
            #Construct "spam error generator" by comparing *dense* vectors
            pvdense = pureVec.todense()
            svdense = spamVec.todense()
            errgen = _gt.spam_error_generator(svdense, pvdense, mxBasis)
            if sparse: errgen = _sps.csr_matrix( errgen )


        assert(pureVec._evotype == evotype), "`pureVec` must have evotype == '%s'" % evotype
        return cls.from_error_generator(pureVec, errgen, typ, ham_basis,
                                        nonham_basis, param_mode, nonham_mode,
                                        truncate, mxBasis, evotype)


    @classmethod
    def from_error_generator(cls, pureVec, errgen, typ, ham_basis="pp",
                             nonham_basis="pp", param_mode="cptp",
                             nonham_mode="all", truncate=True, mxBasis="pp",
                             evotype="densitymx"):
        """
        Create a Lindblad-parameterized spamvec from an error generator and a
        basis which specifies how to decompose (project) the error generator.

        Parameters
        ----------
        pureVec : numpy array or SPAMVec
            An array or SPAMVec in the *full* density-matrix space (this
            vector will have dimension 4 in the case of a single qubit) which
            represents a pure-state preparation or projection.  This is used as
            the "base" preparation or projection that is followed or preceded
            by, respectively, the action of `errgen`.
            
        errgen : numpy array or SciPy sparse matrix
            a square 2D array that gives the full error generator `L` such 
            that the spamvec is `exp(L)*pureVec` in the case of state
            preparations and `pureVec*exp(L)` in the case of POVM effects. The
            projections of this quantity onto the `ham_basis` and `nonham_basis`
            are closely related to the parameters of the spamvec (they may not
            be exactly equal if, e.g `cptp=True`).

        typ : {"prep","effect"}
            Whether this is a state preparation or POVM effect vector.

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
            SPAM vector's parameter values.  Allowed values are:
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
            The evolution type of the spamvec being constructed.  `"densitymx"` is
            usual Lioville density-matrix-vector propagation via matrix-vector
            products.  `"svterm"` denotes state-vector term-based evolution
            (spamvec is obtained by evaluating the rank-1 terms up to
            some order).  `"cterm"` is similar but stabilizer states.

        Returns
        -------
        LindbladParameterizedSPAMVec
        """
        from .gate import LindbladParameterizedGateMap as _LPGMap
        from .gate import LindbladParameterizedGate as _LPGate
        errcls = _LPGate if (pureVec.dim <= 64 and evotype == "densitymx") else _LPGMap
        errmap = errcls.from_error_generator(
            None, errgen, ham_basis, nonham_basis, param_mode,
            nonham_mode, truncate, mxBasis, evotype)

        return cls(pureVec, errmap, typ)

        
    @classmethod
    def from_lindblad_terms(cls, pureVec, Ltermdict, typ, basisdict=None,
                            param_mode="cptp", nonham_mode="all", truncate=True,
                            mxBasis="pp", evotype="densitymx"):
        """
        Create a Lindblad-parameterized spamvec with a given set of Lindblad terms.

        Parameters
        ----------
        pureVec : numpy array or SPAMVec
            An array or SPAMVec in the *full* density-matrix space (this
            vector will have dimension 4 in the case of a single qubit) which
            represents a pure-state preparation or projection.  This is used as
            the "base" preparation or projection that is followed or preceded
            by, respectively, the parameterized Lindblad-form error generator.

        Ltermdict : dict
            A dictionary specifying which Linblad terms are present in the gate
            parameteriztion.  Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` can be `"H"` (Hamiltonian), `"S"`
            (Stochastic), or `"A"` (Affine).  Hamiltonian and Affine terms always
            have a single basis label (so key is a 2-tuple) whereas Stochastic
            tuples with 1 basis label indicate a *diagonal* term, and are the
            only types of terms allowed when `nonham_mode != "all"`.  Otherwise,
            Stochastic term tuples can include 2 basis labels to specify
            "off-diagonal" non-Hamiltonian Lindblad terms.  Basis labels can be
            strings or integers.  Values are complex coefficients (error rates).

        typ : {"prep","effect"}
            Whether this is a state preparation or POVM effect vector.

        basisdict : dict, optional
            A dictionary mapping the basis labels (strings or ints) used in the
            keys of `Ltermdict` to basis matrices (numpy arrays or Scipy sparse
            matrices).

        param_mode : {"unconstrained", "cptp", "depol", "reldepol"}
            Describes how the Lindblad coefficients/projections relate to the
            SPAM vector's parameter values.  Allowed values are:
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
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        evotype : {"densitymx","svterm","cterm"}
            The evolution type of the spamvec being constructed.  `"densitymx"` is
            usual Lioville density-matrix-vector propagation via matrix-vector
            products.  `"svterm"` denotes state-vector term-based evolution
            (spamvec is obtained by evaluating the rank-1 terms up to
            some order).  `"cterm"` is similar but stabilizer states.

        Returns
        -------
        LindbladParameterizedGateMap                
        """
        #Need a dimension for error map construction (basisdict could be completely empty)
        if not isinstance(pureVec, SPAMVec):
            pureVec = StaticSPAMVec(pureVec, evotype) #assume spamvec is just a vector
        d2 = pureVec.dim

        from .gate import LindbladParameterizedGateMap as _LPGMap
        errmap = _LPGMap(d2, Ltermdict, basisdict, param_mode, nonham_mode,
                         truncate, mxBasis, evotype)
        return cls(pureVec, errmap, typ)

    
    def __init__(self, pureVec, errormap, typ):
        """
        Initialize a LindbladParameterizedSPAMVec object.

        Essentially a pure state preparation or projection that is followed
        or preceded by, respectively, the action of LindbladParameterizedGate.

        Parameters
        ----------
        pureVec : numpy array or SPAMVec
            An array or SPAMVec in the *full* density-matrix space (this
            vector will have dimension 4 in the case of a single qubit) which
            represents a pure-state preparation or projection.  This is used as
            the "base" preparation or projection that is followed or preceded
            by, respectively, the parameterized Lindblad-form error generator.
            (This argument is *not* copied if it is a SPAMVec.  A numpy array
             is converted to a new StaticSPAMVec.)

        errormap : GateMap
            The error generator action and parameterization, encapsulated in
            a gate object.  Usually a :class:`LindbladParameterizedGateMap`
            or :class:`ComposedGateMap` object.  (This argument is *not* copied,
            to allow LindbladParameterizedSPAMVecs to share error generator
            parameters with other gates and spam vectors.)

        typ : {"prep","effect"}
            Whether this is a state preparation or POVM effect vector.
        """
        from .gate import LindbladParameterizedGateMap as _LPGMap
        evotype = errormap._evotype
        assert(evotype in ("densitymx","svterm","cterm")), \
            "Invalid evotype: %s for %s" % (evotype, self.__class__.__name__)

        if not isinstance(pureVec, SPAMVec):
            pureVec = StaticSPAMVec(pureVec, evotype) #assume spamvec is just a vector
        
        assert(pureVec._evotype == evotype), \
            "`pureVec` evotype must match `errormap` ('%s' != '%s')" % (pureVec._evotype,evotype)
        assert(pureVec.num_params() == 0), "`pureVec` 'reference' must have *zero* parameters!"

        d2 = pureVec.dim
        self.typ = typ
        self.state_vec = pureVec
        self.error_map = errormap
        self.terms = {} if evotype in ("svterm","cterm") else None
        SPAMVec.__init__(self, d2, evotype) #sets self.dim


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
        num_new_params = self.error_map.allocate_gpindices( startingIndex, parent ) # *same* parent as this SPAMVec
        _gatesetmember.GateSetMember.set_gpindices(
            self, self.error_map.gpindices, parent)
        return num_new_params


    def relink_parent(self, parent):
        """ 
        Sets the parent of this object *without* altering its gpindices.

        In addition to setting the parent of this object, this method 
        sets the parent of any objects this object contains (i.e.
        depends upon) - much like allocate_gpindices.  To ensure a valid
        parent is not overwritten, the existing parent *must be None*
        prior to this call.
        """
        self.error_map.relink_parent(parent)
        _gatesetmember.GateSetMember.relink_parent(self, parent)

        
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

        self.error_map.set_gpindices(gpindices, parent, memo)
        self.terms = {} # clear terms cache since param indices have changed now
        _gatesetmember.GateSetMember.set_gpindices(self, gpindices, parent)

        
    def todense(self, scratch=None):
        """
        Return this SPAM vector as a (dense) numpy array.  The memory
        in `scratch` maybe used when it is not-None.
        """
        if self.typ == "prep":
            #error map acts on dmVec
            return _np.dot(self.error_map.todense(), self.state_vec.todense())
        else:
            #Note: if this is an effect vector, self.error_map is the
            # map that acts on the *state* vector before dmVec acts
            # as an effect:  E.T -> dot(E.T,errmap) ==> E -> dot(errmap.T,E)
            return _np.dot(self.error_map.todense().conjugate().T, self.state_vec.todense())

    def torep(self, typ, outvec=None):
        """
        Return a "representation" object for this SPAMVec.

        Such objects are primarily used internally by pyGSTi to compute
        things like probabilities more efficiently.

        Returns
        -------
        StateRep
        """
        if self._evotype == "densitymx":

            if typ == "prep":
                dmRep = self.state_vec.torep(typ)
                errmapRep = self.error_map.torep()
                return errmapRep.acton(dmRep) # FUTURE: do this acton in place somehow? (like C-reps do)
                  #maybe make a special _Errgen *state* rep?

            else: # effect
                dmEffectRep = self.state_vec.torep(typ)
                errmapRep = self.error_map.torep()
                return replib.DMEffectRep_Errgen(errmapRep, dmEffectRep, id(self.error_map))
                  # an effect that applies a *named* errormap before computing with dmEffectRep

        else:
            #framework should not be calling "torep" on states w/a term-based evotype...
            # they should call torep on the *terms* given by get_order_terms(...)
            raise ValueError("Invalid evotype '%s' for %s.torep(...)" %
                             (self._evotype, self.__class__.__name__))

    
    def get_order_terms(self, order):
        """ 
        Get the `order`-th order Taylor-expansion terms of this SPAM vector.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that it is a state
        preparation followed by or POVM effect preceded by actions on a
        density matrix `rho` of the form:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the 
        SPAMVec's parameters, where the polynomial's variable indices index the
        *global* parameters of the SPAMVec's parent (usually a :class:`GateSet`)
        , not the SPAMVec's local parameter array (i.e. that returned from
        `to_vector`).


        Parameters
        ----------
        order : int
            The order of terms to get.

        Returns
        -------
        list
            A list of :class:`RankOneTerm` objects.
        """
        if order not in self.terms:
            if self._evotype == "svterm": tt = "dense"
            elif self._evotype == "cterm": tt = "clifford"
            else: raise ValueError("Invalid evolution type %s for calling `get_order_terms`" % self._evotype)
            assert(self.gpindices is not None),"LindbladParameterizedSPAMVec must be added to a GateSet before use!"

            state_terms = self.state_vec.get_order_terms(0); assert(len(state_terms) == 1)
            stateTerm = state_terms[0]
            err_terms = self.error_map.get_order_terms(order)
            if self.typ == "prep":
                terms = [ _term.compose_terms((stateTerm,t)) for t in err_terms] # t ops occur *after* stateTerm's
            else: # "effect"
                #Effect terms are special in that all their pre/post ops act in order on the *state* before
                # the final effect is used to compute a probability.  Thus, constructing the same "terms"
                # as above works here too - the difference comes when this SPAMVec is used as an effect rather than a prep.
                terms = [ _term.compose_terms((stateTerm,t)) for t in err_terms] # t ops occur *after* stateTerm's

            #OLD: now this is done within calculator when possible b/c not all terms can be collapsed
            #terms = [ t.collapse() for t in terms ] # collapse terms for speed
            # - resulting in terms with just a single pre/post op, each == a pure state
            self.terms[order] = terms

        return self.terms[order]

    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the derivatives of the SPAM vector
        with respect to a single param.  Thus, each column is of length
        get_dimension and there is one column per SPAM vector parameter.
    
        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension, num_params)
        """
        dmVec = self.state_vec.todense()

        derrgen = self.error_map.deriv_wrt_params(wrtFilter) # shape (dim*dim, nParams)
        derrgen.shape = (self.dim, self.dim, derrgen.shape[1]) # => (dim,dim,nParams)

        if self.typ == "prep":
            #derror map acts on dmVec
            #return _np.einsum("ijk,j->ik", derrgen, dmVec) # return shape = (dim,nParams)
            return _np.tensordot(derrgen, dmVec, (1,0)) # return shape = (dim,nParams)
        else:
            # self.error_map acts on the *state* vector before dmVec acts
            # as an effect:  E.dag -> dot(E.dag,errmap) ==> E -> dot(errmap.dag,E)
            #return _np.einsum("jik,j->ik", derrgen.conjugate(), dmVec) # return shape = (dim,nParams)
            return _np.tensordot(derrgen.conjugate(), dmVec, (0,0)) # return shape = (dim,nParams)


    def hessian_wrt_params(self, wrtFilter1=None, wrtFilter2=None):
        """
        Construct the Hessian of this SPAM vector with respect to its parameters.

        This function returns a tensor whose first axis corresponds to the
        flattened gate matrix and whose 2nd and 3rd axes correspond to the
        parameters that are differentiated with respect to.

        Parameters
        ----------
        wrtFilter1, wrtFilter2 : list
            Lists of indices of the paramters to take first and second
            derivatives with respect to.  If None, then derivatives are
            taken with respect to all of the vectors's parameters.

        Returns
        -------
        numpy array
            Hessian with shape (dimension, num_params1, num_params2)
        """
        dmVec = self.state_vec.todense()

        herrgen = self.error_map.hessian_wrt_params(wrtFilter1, wrtFilter2) # shape (dim*dim, nParams1, nParams2)
        herrgen.shape = (self.dim, self.dim, herrgen.shape[1], herrgen.shape[2]) # => (dim,dim,nParams1, nParams2)

        if self.typ == "prep":
            #derror map acts on dmVec
            #return _np.einsum("ijkl,j->ikl", herrgen, dmVec) # return shape = (dim,nParams)
            return _np.tensordot(herrgen, dmVec, (1,0)) # return shape = (dim,nParams)
        else:
            # self.error_map acts on the *state* vector before dmVec acts
            # as an effect:  E.dag -> dot(E.dag,errmap) ==> E -> dot(errmap.dag,E)
            #return _np.einsum("jikl,j->ikl", herrgen.conjugate(), dmVec) # return shape = (dim,nParams)
            return _np.tensordot(herrgen.conjugate(), dmVec, (0,0)) # return shape = (dim,nParams)


    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.error_map.num_params()

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this gate.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self.error_map.to_vector()


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
        self.error_map.from_vector(v)
        self.dirty = True


    def transform(self, S, typ):
        """
        Update SPAM (column) vector V as inv(S) * V or S^T * V for preparation
        or  effect SPAM vectors, respectively.

        Note that this is equivalent to state preparation vectors getting 
        mapped: `rho -> inv(S) * rho` and the *transpose* of effect vectors
        being mapped as `E^T -> E^T * S`.

        Generally, the transform function updates the *parameters* of 
        the SPAM vector such that the resulting vector is altered as 
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.
            
        typ : { 'prep', 'effect' }
            Which type of SPAM vector is being transformed (see above).
        """
        #Defer everything to LindbladParameterizedGateMap's
        # `spam_tranform` function, which applies either 
        # error_map -> inv(S) * error_map ("prep" case) OR
        # error_map -> error_map * S      ("effect" case)
        self.error_map.spam_transform(S,typ)

        
    def depolarize(self, amount):
        """
        Depolarize this SPAM vector by the given `amount`.

        Generally, the depolarize function updates the *parameters* of 
        the SPAMVec such that the resulting vector is depolarized.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the spam vector. All but
            the first element of the spam vector (often corresponding to the 
            identity element) are multiplied by `amount` (if a float) or
            the corresponding `amount[i]` (if a tuple).

        Returns
        -------
        None
        """
        self.error_map.depolarize(amount)
            


class StabilizerSPAMVec(SPAMVec):
    """
    A stabilizer state preparation represented internally using a compact
    representation of its stabilizer group.
    """

    @classmethod
    def from_dense_purevec(cls, purevec):
        """
        Create a new StabilizerSPAMVec from a pure-state vector.

        Currently, purevec must be a single computational basis state (it 
        cannot be a superpostion of multiple of them).

        Parameters
        ----------
        purevec : numpy.ndarray
            A complex-valued state vector specifying a pure state in the
            standard computational basis.  This vector has length 2^n for 
            n qubits.

        Returns
        -------
        StabilizerSPAMVec
        """
        nqubits = int(round(_np.log2(len(purevec))))
        v = (_np.array([1,0],'d'), _np.array([0,1],'d')) # (v0,v1)
        for zvals in _itertools.product(*([(0,1)]*nqubits)):
            testvec = _functools.reduce(_np.kron, [v[i] for i in zvals])
            if _np.allclose(testvec, purevec.flat):
                return cls(nqubits, zvals)
        raise ValueError(("Given `purevec` must be a z-basis product state - "
                          "cannot construct StabilizerSPAMVec"))


    def __init__(self, nqubits, zvals=None):
        """
        Initialize a StabilizerSPAMVec object.

        Parameters
        ----------
        nqubits : int 
            Number of qubits

        zvals : iterable, optional
            An iterable over anything that can be cast as True/False
            to indicate the 0/1 value of each qubit in the Z basis.
            If None, the all-zeros state is created.
        """
        self.sframe = _stabilizer.StabilizerFrame.from_zvals(nqubits,zvals)
        dim = 2**nqubits # assume "unitary evolution"-type mode?
        SPAMVec.__init__(self, dim, "stabilizer")


    def todense(self, scratch=None):
        """
        Return this SPAM vector as a (dense) numpy array of shape
        (2^(nqubits), 1).  The memory in `scratch` maybe used when
        it is not-None.
        """
        statevec = self.sframe.to_statevec()
        statevec.shape = (statevec.size,1)
        return statevec

    
    def torep(self, typ, outvec=None):
        """
        Return a "representation" object for this SPAMVec.

        Such objects are primarily used internally by pyGSTi to compute
        things like probabilities more efficiently.

        Returns
        -------
        SBStateRep
        """
        return self.sframe.torep()


    def __str__(self):
        s = "Stabilizer spam vector for %d qubits with rep:\n" % (self.sframe.nqubits)
        s += str(self.sframe)
        return s


class StabilizerEffectVec(SPAMVec): # FUTURE: merge this with ComptationalSPAMVec (w/evotype == "stabilizer")?
    """
    A dummy SPAM vector that points to a set/product of 1-qubit POVM 
    outcomes from stabilizer-state measurements.
    """

    @classmethod
    def from_dense_purevec(cls, purevec):
        """
        Create a new StabilizerEffectVec from a pure-state vector.

        Currently, purevec must be a single computational basis state (it 
        cannot be a superpostion of multiple of them).

        Parameters
        ----------
        purevec : numpy.ndarray
            A complex-valued state vector specifying a pure state in the
            standard computational basis.  This vector has length 2^n for 
            n qubits.

        Returns
        -------
        StabilizerSPAMVec
        """
        nqubits = int(round(_np.log2(len(purevec))))
        v = (_np.array([1,0],'d'), _np.array([0,1],'d')) # (v0,v1)
        for zvals in _itertools.product(*([(0,1)]*nqubits)):
            testvec = _functools.reduce(_np.kron, [v[i] for i in zvals])
            if _np.allclose(testvec, purevec.flat):
                return cls(zvals)
        raise ValueError(("Given `purevec` must be a z-basis product state - "
                          "cannot construct StabilizerEffectVec"))
    
    def __init__(self, outcomes):
        """
        Initialize a StabilizerEffectVec object.

        Parameters
        ----------
        outcomes : iterable
            A list or other iterable of integer 0 or 1 outcomes specifying
            which POVM effect vector this object represents within the 
            full `stabilizerPOVM`
        """
        self._outcomes = _np.array(outcomes,int)
        nqubits = len(outcomes)
        dim = 2**nqubits # assume "unitary evolution"-type mode?
        SPAMVec.__init__(self, dim, "stabilizer")

        
    def torep(self, typ, outvec=None):
        # changes to_statevec/to_dmvec -> todense, and have
        # torep create an effect rep object...
        return replib.SBEffectRep(_np.ascontiguousarray(self._outcomes,_np.int64))
          #Note: dtype='i' => int in Cython, whereas dtype=int/np.int64 => long in Cython

    def todense(self):
        """
        Return this SPAM vector as a dense state vector of shape
        (2^(nqubits), 1)

        Returns
        -------
        numpy array
        """
        v = (_np.array([1,0],'d'), _np.array([0,1],'d')) # (v0,v1) - eigenstates of sigma_z
        statevec = _functools.reduce(_np.kron, [v[i] for i in self.outcomes])
        statevec.shape = (statevec.size,1)
        return statevec

        
    @property
    def outcomes(self):
        """ The 0/1 outcomes identifying this effect within its StabilizerZPOVM """
        return self._outcomes

    def __str__(self):
        nQubits = len(self.outcomes)
        s = "Stabilizer effect vector for %d qubits with outcome %s" % (nQubits, str(self.outcomes))
        return s



class ComputationalSPAMVec(SPAMVec):
    """
    A static SPAM vector that is tensor product of 1-qubit Z-eigenstates.

    This is called a "computational basis state" in many contexts.
    """

    @classmethod
    def from_dense_vec(cls, vec, evotype):
        """
        Create a new ComputationalSPAMVec from a dense vector.

        Parameters
        ----------
        vec : numpy.ndarray
            A state vector specifying a computational basis state in the
            standard basis.  This vector has length 2^n or 4^n for 
            n qubits depending on `evotype`.

        evotype : {"densitymx", "statevec", "stabilizer", "svterm", "cterm"}
            The evolution type of the resulting SPAM vector.  This value
            must be consistent with `len(vec)`, in that `"statevec"` and
            `"stabilizer"` expect 2^n whereas the rest expect 4^n.

        Returns
        -------
        ComputationalSPAMVec
        """
        if evotype in ('stabilizer','statevec'):
            nqubits = int(round(_np.log2(len(vec))))
            v0 = _np.array((1,0),complex) # '0' qubit state as complex state vec
            v1 = _np.array((0,1),complex) # '1' qubit state as complex state vec
        else:
            nqubits = int(round(_np.log2(len(vec))/2))
            v0 = 1.0/_np.sqrt(2) * _np.array((1,0,0,1),'d') # '0' qubit state as Pauli dmvec
            v1 = 1.0/_np.sqrt(2) * _np.array((1,0,0,-1),'d')# '1' qubit state as Pauli dmvec
            
        v = (v0,v1)
        for zvals in _itertools.product(*([(0,1)]*nqubits)):
            testvec = _functools.reduce(_np.kron, [v[i] for i in zvals])
            if _np.allclose(testvec, vec.flat):
                return cls(zvals, evotype)
        raise ValueError(("Given `vec` is not a z-basis product state - "
                          "cannot construct ComputatinoalSPAMVec"))


    def __init__(self, zvals, evotype):
        """
        Initialize a ComputationalSPAMVec object.

        Parameters
        ----------
        zvals : iterable
            A list or other iterable of integer 0 or 1 outcomes specifying
            which computational basis element this object represents.  The
            length of `zvals` gives the total number of qubits.

        evotype : {"densitymx", "statevec", "stabilizer", "svterm", "cterm"}
            The type of evolution being performed.
        """
        self._zvals = _np.array(zvals,_np.int64)

        nqubits = len(self._zvals)
        if evotype in ("densitymx","svterm","cterm"):
            dim = 4**nqubits
        elif evotype in ("statevec","stabilizer"):
            dim = 2**nqubits
        else: raise ValueError("Invalid `evotype`: %s" % evotype)

        SPAMVec.__init__(self, dim, evotype)

    def todense(self, scratch=None):
        """
        Return this SPAM vector as a (dense) numpy array.  The memory
        in `scratch` maybe used when it is not-None.
        """
        if self._evotype == "densitymx":
            v0 = 1.0/_np.sqrt(2) * _np.array((1,0,0,1),'d') # '0' qubit state as Pauli dmvec
            v1 = 1.0/_np.sqrt(2) * _np.array((1,0,0,-1),'d')# '1' qubit state as Pauli dmvec
        elif self._evotype in ("statevec","stabilizer"):
            v0 = _np.array((1,0),complex) # '0' qubit state as complex state vec
            v1 = _np.array((0,1),complex) # '1' qubit state as complex state vec
        elif self._evotype in ("svterm","cterm"):
            raise NotImplementedError("todense() is not implemented for evotype %s!" %
                                      self._evotype)
        else: raise ValueError("Invalid `evotype`: %s" % evotype)

        v = (v0,v1)
        return _functools.reduce(_np.kron, [v[i] for i in self._zvals])


    def torep(self, typ, outvec=None):
        if typ == "prep":
            if self._evotype == "statevec":
                return replib.SVStateRep(self.todense())
            elif self._evotype == "densitymx":
                return replib.DMStateRep(self.todense())
            elif self._evotype == "stabilizer":
                sframe = _stabilizer.StabilizerFrame.from_zvals(len(self._zvals),self._zvals)
                return sframe.torep()
            raise NotImplementedError("torep(%s) not implemented for %s objects!" %
                                      (self._evotype, self.__class__.__name__))
        elif typ == "effect":
            if self._evotype == "statevec":
                return replib.SVEffectRep_Computational(self._zvals, self.dim)
            elif self._evotype == "densitymx":
                return replib.DMEffectRep_Computational(self._zvals, self.dim)
            elif self._evotype == "stabilizer":
                return replib.SBEffectRep(_np.ascontiguousarray(self._zvals,_np.int64))
            raise NotImplementedError("torep(%s) not implemented for %s objects!" %
                                      (self._evotype, self.__class__.__name__))
        else:
            raise ValueError("Invalid `typ` argument for torep(): %s" % typ)


    def get_order_terms(self, order):
        """ 
        Get the `order`-th order Taylor-expansion terms of this SPAM vector.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that it is a state
        preparation followed by or POVM effect preceded by actions on a
        density matrix `rho` of the form:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the 
        SPAMVec's parameters, where the polynomial's variable indices index the
        *global* parameters of the SPAMVec's parent (usually a :class:`GateSet`)
        , not the SPAMVec's local parameter array (i.e. that returned from
        `to_vector`).


        Parameters
        ----------
        order : int
            The order of terms to get.

        Returns
        -------
        list
            A list of :class:`RankOneTerm` objects.
        """
        if order == 0: # only 0-th order term exists
            if self._evotype == "svterm":
                tt = "dense"
                purevec = ComputationalSPAMVec(self._zvals, "statevec")
            elif self._evotype == "cterm":
                tt = "clifford"
                purevec = ComputationalSPAMVec(self._zvals, "stabilizer")
            else: raise ValueError("Invalid evolution type %s for calling `get_order_terms`" % self._evotype)

            terms = [ _term.RankOneTerm(_Polynomial({(): 1.0}), purevec, purevec, tt) ]
        else:
            terms = []

        return terms # Cache terms in FUTURE?


    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return 0 #no parameters


    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return _np.array([], 'd') #no parameters


    def from_vector(self, v):
        """
        Initialize the SPAM vector using a 1D array of parameters.

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


    def __str__(self):
        nQubits = len(self._zvals)
        s = "Computational Z-basis SPAM vec for %d qubits w/z-values: %s" % (nQubits, str(self._zvals))
        return s
