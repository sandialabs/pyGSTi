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
from .polynomial import Polynomial as _Polynomial


try:
    #import pyximport; pyximport.install(setup_args={'include_dirs': _np.get_include()}) # develop-mode
    from ..tools import fastcalc as _fastcalc
except ImportError:
    _fastcalc = None

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

    toType : {"full","TP","static","H+S terms","clifford"}
        The type of parameterizaton to convert to.

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for `spamvec`.  Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
        (or a custom basis object).

    extra : object, optional
        Additional information for conversion.  An ideal pure state
        in "H+S terms" case.


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
            return FullyParameterizedSPAMVec( spamvec )

    elif toType == "TP":
        if isinstance(spamvec, TPParameterizedSPAMVec):
            return spamvec #no conversion necessary
        else:
            return TPParameterizedSPAMVec( spamvec )
              # above will raise ValueError if conversion cannot be done

    elif toType == "CPTP":
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

    elif toType in ("H+S terms","H+S clifford terms"):
        typ = "dense" if toType == "H+S terms" else "clifford prep"
        if isinstance(spamvec, LindbladTermSPAMVec) and spamvec.termtype == typ:
            return spamvec

        if extra is None:
            dmvec = _bt.change_basis(spamvec.toarray(),basis,'std')
            ideal_purevec = _gt.dmvec_to_state(dmvec)
        else:
            ideal_purevec = extra

        # Compute error generator for rho
        d2 = spamvec.dim ; d = len(ideal_purevec)
        assert(d**2 == d2) # what about for unitary evolution?? purevec could be size "d2"
        ideal_purevec = _np.array(ideal_purevec); ideal_purevec.shape = (d,1) # expect this is a dense vector
        ideal_spamvec = _bt.change_basis(_np.kron( ideal_purevec, _np.conjugate(ideal_purevec.T)).flatten(), 'std', basis)
        errgen = _gt.spam_error_generator(spamvec, ideal_spamvec, basis)
        return LindbladTermSPAMVec.from_error_generator(ideal_purevec[:,0], errgen,
                                                        nonham_diagonal_only=True, termtype=typ)

    elif toType == "clifford":
        if isinstance(spamvec, StabilizerSPAMVec):
            return spamvec #no conversion necessary

        purevec = spamvec.flatten() # assume a pure state (otherwise would
                                    # need to change GateSet dim)
        return StabilizerSPAMVec.from_dense_purevec(purevec)

    else:
        raise ValueError("Invalid toType argument: %s" % toType)


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

    def __init__(self, dim):
        """ Initialize a new SPAM Vector """
        super(SPAMVec, self).__init__( dim )

    def toarray(self):
        """ Return this SPAM vector as a (dense) numpy array """
        raise NotImplementedError("Derived classes must implement toarray()!")

    def transform(self, S, typ):
        """
        Update SPAM (column) vector V as inv(S) * V or S^T * V for prep and
        effect SPAM vectors, respectively (depending on the value of `typ`). 
        """
        raise NotImplementedError("This SPAM vector cannot be tranform()'d")

    def depolarize(self, amount):
        """ Depolarize spam vector by the given amount. """
        raise NotImplementedError("This SPAM vector cannot be depolarize()'d")

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
        vec = self.toarray()
        if typ == 'prep':
            if inv_transform is None:
                return _gt.frobeniusdist2(vec,otherSpamVec.toarray())
            else:
                return _gt.frobeniusdist2(_np.dot(inv_transform,vec),
                                          otherSpamVec.toarray())
        elif typ == "effect":
            if transform is None:
                return _gt.frobeniusdist2(vec,otherSpamVec.toarray())
            else:
                return _gt.frobeniusdist2(_np.dot(_np.transpose(transform),
                                                  vec), otherSpamVec.toarray())
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
        vec = self.toarray()
        if typ == 'prep':
            if inv_transform is None:
                return _gt.residuals(vec,otherSpamVec.toarray())
            else:
                return _gt.residuals(_np.dot(inv_transform,vec),
                                          otherSpamVec.toarray())
        elif typ == "effect":
            if transform is None:
                return _gt.residuals(vec,otherSpamVec.toarray())
            else:
                return _gt.residuals(_np.dot(_np.transpose(transform),
                                             vec), otherSpamVec.toarray())

    #Handled by derived classes
    #def __str__(self):
    #    s = "Spam vector with length %d\n" % len(self.base)
    #    s += _mt.mx_to_string(self.base, width=4, prec=2)
    #    return s


    #Pickle plumbing
    def __setstate__(self, state):
        self.__dict__.update(state)

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
            vector = _np.asarray(V).copy()
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

    def __init__(self, vec):
        """ Initialize a new SPAM Vector """
        self.base = vec
        super(DenseSPAMVec, self).__init__( len(vec) )

    def toarray(self, scratch=None):
        """ 
        Return this SPAM vector as a (dense) numpy array.  The memory
        in `scratch` maybe used when it is not-None.
        """
        #don't use scratch since we already have memory allocated
        return self.base[:,0]
                                   
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




class StaticSPAMVec(DenseSPAMVec):
    """
    Encapsulates a SPAM vector that is completely fixed, or "static", meaning
      that is contains no parameters.
    """

    def __init__(self, vec):
        """
        Initialize a StaticSPAMVec object.

        Parameters
        ----------
        vec : array_like or SPAMVec
            a 1D numpy array representing the SPAM operation.  The
            shape of this array sets the dimension of the SPAM op.
        """
        DenseSPAMVec.__init__(self, SPAMVec.convert_to_vector(vec))


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
        vec  = SPAMVec.convert_to_vector(vec)
        if(vec.size != self.dim):
            raise ValueError("Argument must be length %d" % self.dim)
        self.base[:,:] = vec
        self.dirty = True

    def transform(self, S, typ):
        """
        Update SPAM (column) vector V as inv(S) * V or S^T * V for prep and
        effect SPAM vectors, respectively (depending on the value of `typ`). 

        Note that this is equivalent to state preparation vectors getting 
        mapped: `rho -> inv(S) * rho` and the *transpose* of effect vectors
        being mapped as `E^T -> E^T * S`.

        Generally, the transform function updates the *parameters* of 
        the SPAM vector such that the resulting vector is altered as 
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        In this case, a ValueError is *always* raised, since a 
        StaticSPAMVec has no parameters.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.
            
        typ : { 'prep', 'effect' }
            Which type of SPAM vector is being transformed (see above).
        """
        raise ValueError("Invalid transform for StaticSPAMVec - no parameters")


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
        raise ValueError("Cannot depolarize a StaticSPAMVec - no parameters")


    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.
        Zero in the case of StaticSPAMVec.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return 0 #no parameters


    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.  An empty
        array in the case of StaticSPAMVec.

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
        derivMx = _np.zeros((self.dim,0),self.base.dtype)
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


    def copy(self, parent=None):
        """
        Copy this SPAM vector.

        Returns
        -------
        StaticSPAMVec
            A copy of this SPAM operation.
        """
        return self._copy_gpindices( StaticSPAMVec(self.base), parent )


    def __str__(self):
        s = "Static spam vector with length %d\n" % \
            len(self.base)
        s += _mt.mx_to_string(self.base, width=4, prec=2)
        return s





class FullyParameterizedSPAMVec(DenseSPAMVec):
    """
    Encapsulates a SPAM vector that is fully parameterized, that is,
      each element of the SPAM vector is an independent parameter.
    """

    def __init__(self, vec):
        """
        Initialize a FullyParameterizedSPAMVec object.

        Parameters
        ----------
        vec : array_like or SPAMVec
            a 1D numpy array representing the SPAM operation.  The
            shape of this array sets the dimension of the SPAM op.
        """
        DenseSPAMVec.__init__(self, SPAMVec.convert_to_vector(vec))


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


    def transform(self, S, typ):
        """
        Update SPAM (column) vector V as inv(S) * V or S^T * V for prep and
        effect SPAM vectors, respectively (depending on the value of `typ`). 

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
            self.set_value(_np.dot(Si, self))
        elif typ == 'effect':
            Smx = S.get_transform_matrix()
            self.set_value(_np.dot(_np.transpose(Smx),self)) 
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
        self.set_value(_np.dot(D,self)) 


    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return 2*self.dim if _np.iscomplexobj(self.base) else self.dim


    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        if _np.iscomplexobj(self.base):
            return _np.concatenate((self.base.real.flatten(),self.base.imag.flatten()), axis=0)
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
        if _np.iscomplexobj(self.base):
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
        if _np.iscomplexobj(self.base):
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


    def copy(self, parent=None):
        """
        Copy this SPAM vector.

        Returns
        -------
        FullyParameterizedSPAMVec
            A copy of this SPAM operation.
        """
        return self._copy_gpindices( FullyParameterizedSPAMVec(self.base), parent )

    def __str__(self):
        s = "Fully Parameterized spam vector with length %d\n" % len(self.base)
        s += _mt.mx_to_string(self.base, width=4, prec=2)
        return s    



#Helpful for deriv_wrt_params??
#            for (i,rhoVec) in enumerate(self.preps):
#            deriv[foff+m:foff+m+rhoSize[i],off:off+rhoSize[i]] = _np.identity( rhoSize[i], 'd' )
#                off += rhoSize[i]; foff += full_vsize
#
#            for (i,EVec) in enumerate(self.effects):
#                deriv[foff:foff+eSize[i],off:off+eSize[i]] = _np.identity( eSize[i], 'd' )
#                off += eSize[i]; foff += full_vsize


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
    def __setstate__(self, state):
        self.__dict__.update(state)
        
    def __getstate__(self):
        d = self.__dict__.copy()
        d['_parent'] = None
        return d

    
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
        DenseSPAMVec.__init__(self, _ProtectedArray(vector,
                                                    indicesToProtect=(0,0)))


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

        
    def transform(self, S, typ):
        """
        Update SPAM (column) vector V as inv(S) * V or S^T * V for prep and
        effect SPAM vectors, respectively (depending on the value of `typ`). 

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
            self.set_value(_np.dot(Si, self))
        elif typ == 'effect':
            Smx = S.get_transform_matrix()
            self.set_value(_np.dot(_np.transpose(Smx),self)) 
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
        self.set_value(_np.dot(D,self)) 


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


    def copy(self, parent=None):
        """
        Copy this SPAM vector.

        Returns
        -------
        TPParameterizedSPAMVec
            A copy of this SPAM operation.
        """
        return self._copy_gpindices( TPParameterizedSPAMVec(self.base), parent )

    def __str__(self):
        s = "TP-Parameterized spam vector with length %d\n" % self.dim
        s += _mt.mx_to_string(self.base, width=4, prec=2)
        return s


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
            
        DenseSPAMVec.__init__(self, self.identity) # dummy
        self._construct_vector() #reset's self.base
        
    def _construct_vector(self):
        self.base = self.identity - sum(self.other_vecs)
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
        raise ValueError(("Cannot set the value of a ComplementSPAMVector "
                          "directly, as its elements depend on *other* SPAM "
                          "vectors"))

    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return len(self.gpindices_as_array())
    
    
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

    def from_vector(self, v):
        """
        Initialize this partially-implemented spam-vec using a vector of its parameters.

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


    def copy(self, parent=None):
        """
        Copy this SPAM vector.

        Returns
        -------
        ComplementSPAMVec
            A copy of this SPAM operation.
        """
        # don't copy other vecs - leave this to parent
        return self._copy_gpindices( ComplementSPAMVec(self.identity, []), parent ) 

    def __str__(self):
        s = "Complement spam vector with length %d\n" % len(self.base)
        s += _mt.mx_to_string(self.base, width=4, prec=2)
        return s


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

        DenseSPAMVec.__init__(self, vector)

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
        
    def transform(self, S, typ):
        """
        Update SPAM (column) vector V as inv(S) * V or S^T * V for prep and
        effect SPAM vectors, respectively (depending on the value of `typ`). 

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
            self.set_value(_np.dot(Si, self))
        elif typ == 'effect':
            Smx = S.get_transform_matrix()
            self.set_value(_np.dot(_np.transpose(Smx),self)) 
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
        self.set_value(_np.dot(D,self)) 


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
    
    
    def copy(self, parent=None):
        """
        Copy this SPAM vector.

        Returns
        -------
        TPParameterizedSPAMVec
            A copy of this SPAM operation.
        """
        copyOfMe = CPTPParameterizedSPAMVec(self.base, self.basis.copy())
        copyOfMe.params = self.params.copy() #ensure params are exactly the same
        return self._copy_gpindices( copyOfMe, parent )

    def __str__(self):
        s = "CPTP-Parameterized spam vector with length %d\n" % self.dim
        s += _mt.mx_to_string(self.base, width=4, prec=2)
        return s



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
        self.typ = typ
        self.factors = factors #do *not* copy - needs to reference common objects
        self.Np = sum([fct.num_params() for fct in factors])
        if typ == "effect":
            self.effectLbls = _np.array(povmEffectLbls)
            if any([ any([_np.iscomplexobj(Evec) for Evec in fct.values()]) for fct in factors]):
                self._evotype = "statevec"
            else: self._evotype = "densitymx"
                
        elif typ == "prep":
            assert(povmEffectLbls is None), '`povmEffectLbls` must be None when `typ != "effects"`'
            self.effectLbls = None
            if any([_np.iscomplexobj(v) for v in factors]):
                self._evotype = "statevec"
            elif any([isinstance(v, (StabilizerSPAMVec,StabilizerEffectVec)) for v in factors]):
                self._evotype = "stabilizer"
            else: self._evotype = "densitymx"
                        
        else: raise ValueError("Invalid `typ` argument: %s" % typ)
        
        SPAMVec.__init__(self, _np.product([fct.dim for fct in factors]))
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

        #Memory for speeding up kron product in toarray()
        if self._evotype in ("statevec","densitymx"): #types that require fast kronecker prods
            max_factor_dim = max(fct.dim for fct in factors)
            self._fast_kron_array = _np.empty( (len(factors), max_factor_dim), complex if self._evotype == "statevec" else 'd')
            self._fast_kron_factordims = _np.array([fct.dim for fct in factors],'i')
            try:
                self._fill_fast_kron()
            except NotImplementedError: # if toarray() or any other prereq isn't implemented (
                self._fast_kron_array = None   # e.g. if factors are LindbladTermSPAMVecs
                self._fast_kron_factordims = None
                
        else:
            self._fast_kron_array = None
            self._fast_kron_factordims = None

        
    def _fill_fast_kron(self):
        """ Fills in self._fast_kron_array based on current self.factors """
        if self.typ == "prep":
            for i,factor_dim in enumerate(self._fast_kron_factordims):
                self._fast_kron_array[i][0:factor_dim] = self.factors[i].toarray()
        else:
            factorPOVMs = self.factors
            for i,(factor_dim,Elbl) in enumerate(zip(self._fast_kron_factordims,self.effectLbls)):
                self._fast_kron_array[i][0:factor_dim] = factorPOVMs[i][Elbl].toarray()


    def toarray(self, scratch=None):
        """
        Return this SPAM vector as a (dense) numpy array.  The memory
        in `scratch` maybe used when it is not-None.
        """
        if self._evotype in ("statevec","densitymx"):
            if len(self.factors) == 0: return _np.empty(0,complex if self._evotype == "statevec" else 'd')
            if scratch is not None and _fastcalc is not None:
                assert(scratch.shape[0] == self.dim)
                # use faster kron that avoids memory allocation.
                # Note: this uses more memory b/c all self.factors.toarray() results
                #  are present in memory at the *same* time - could add a flag to
                #  disable fast-kron-array when memory is extra tight(?).
                if self._evotype == "statevec":
                    _fastcalc.fast_kron_complex(scratch, self._fast_kron_array, self._fast_kron_factordims)
                else:
                    _fastcalc.fast_kron(scratch, self._fast_kron_array, self._fast_kron_factordims)
                return scratch if scratch.ndim == 1 else scratch
                
            if self.typ == "prep":
                ret = self.factors[0].toarray() # factors are just other SPAMVecs
                for i in range(1,len(self.factors)):
                    ret = _np.kron(ret, self.factors[i].toarray())
            else:
                factorPOVMs = self.factors
                ret = factorPOVMs[0][self.effectLbls[0]].toarray()
                for i in range(1,len(factorPOVMs)):
                    ret = _np.kron(ret, factorPOVMs[i][self.effectLbls[i]].toarray())
    
            # DEBUG: initial test that fast_kron works...
            #if scratch is not None: assert(_np.linalg.norm(ret - scratch[:,None]) < 1e-6)
            return ret            
        else: # self._evotype == "stabilizer"
            
            if self.typ == "prep":
                # => self.factors should all be StabilizerSPAMVec objs
                #Return stabilizer-rep tuple, just like StabilizerSPAMVec
                sp_factors = [ f.toarray() for f in self.factors ]
                return _symp.symplectic_kronecker(sp_factors)

            else: #self.typ == "effect", so each factor is a StabilizerEffectVec
                raise ValueError("Cannot convert Stabilizer tensor product effect to an array!")
                # should be using effect.outcomes property...


    @property
    def size(self):
        return self.dim

    @property
    def outcomes(self):
        """ TODO: docstring - to mimic StabilizerEffectVec """
        out = list(_itertools.chain(*[f.outcomes for f in self.factors]))
        return _np.array(out, int)
          #Note: may need to a qubit filter property here and to StabilizerEffectVec...
        
    
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
        raise ValueError("Cannot set the value of a TensorProdSPAMVec directly!")
        #self.dirty = True

    def transform(self, S, typ):
        """
        Update SPAM (column) vector V as inv(S) * V or S^T * V for prep and
        effect SPAM vectors, respectively (depending on the value of `typ`). 

        Note that this is equivalent to state preparation vectors getting 
        mapped: `rho -> inv(S) * rho` and the *transpose* of effect vectors
        being mapped as `E^T -> E^T * S`.

        Generally, the transform function updates the *parameters* of 
        the SPAM vector such that the resulting vector is altered as 
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        In this case, a ValueError is *always* raised, since a 
        StaticSPAMVec has no parameters.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.
            
        typ : { 'prep', 'effect' }
            Which type of SPAM vector is being transformed (see above).
        """
        raise ValueError("Invalid transform for TensorProdSPAMVec")


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
        raise ValueError("Cannot depolarize a TensorProdSPAMVec")

    def get_order_terms(self, order):
        """ TODO: docstring """
        from .gate import CliffordGate as _CliffordGate
        terms = []
        fnq = [ int(round(_np.log2(f.dim)))//2 for f in self.factors ] # num of qubits per factor
          # assumes density matrix evolution
        
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
            collapsible = bool(len(factor_lists) == 0 or len(factor_lists[0]) == 0
                               or factor_lists[0][0].typ == "dense") # assume all terms are the same
            if collapsible:
                factor_lists = [ [t.collapse_vec() for t in fterms] for fterms in factor_lists]
                
            for factors in _itertools.product(*factor_lists):
                # create a term with a TensorProdSPAMVec - Note we always create
                # "prep"-mode vectors, since even when self.typ == "effect" these
                # vectors are created with factor SPAMVecs not factor POVMs
                coeff = _functools.reduce(lambda x,y: x.mult_poly(y), [f.coeff for f in factors])
                pre_op = TensorProdSPAMVec("prep", [f.pre_ops[0] for f in factors
                                                      if (f.pre_ops[0] is not None)])
                post_op = TensorProdSPAMVec("prep", [f.post_ops[0] for f in factors
                                                       if (f.post_ops[0] is not None)])
                term = _term.RankOneTerm(coeff, pre_op, post_op)

                if not collapsible: # then may need to add more ops.  Assume factor ops are clifford gates
                    def iop(nq): # identity symplectic op for missing factors
                        return _np.identity(2*nq,int), _np.zeros(2*nq,int)
                    
                    mx = max([len(f.pre_ops) for f in factors])
                    for i in range(1,mx): # for each "layer" of additional terms
                        ops = [ (f.pre_ops[i].smatrix,f.pre_ops[i].svector)
                                if (i < len(f.pre_ops)) else iop(nq) for f,nq in zip(factors,fnq)]
                        term.pre_ops.append( _CliffordGate(
                            symplecticrep = _symp.symplectic_kronecker(ops)))

                    mx = max([len(f.post_ops) for f in factors])
                    for i in range(1,mx): # for each "layer" of additional terms
                        ops = [ (f.post_ops[i].smatrix,f.post_ops[i].svector)
                                if (i < len(f.post_ops)) else iop(nq) for f,nq in zip(factors,fnq)]
                        term.post_ops.append( _CliffordGate(
                            symplecticrep = _symp.symplectic_kronecker(ops)))
                
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
                    pre = self.factors[0].toarray()
                    for vecA in self.factors[1:i]:
                        pre = _np.kron(pre,vecA.toarray())
                else:
                    pre = self.factors[0][self.effectLbls[0]].toarray()
                    for j,fctA in enumerate(self.factors[1:i],start=1):
                        pre = _np.kron(pre,fctA[self.effectLbls[j]].toarray())
                deriv = _np.kron(pre[:,None], deriv) # add a dummy 1-dim to 'pre' and do kron properly...

            if i+1 < len(self.factors): # factors after ith
                if self.typ == "prep":
                    post = self.factors[i+1].toarray()
                    for vecA in self.factors[i+2:]:
                        post = _np.kron(post,vecA.toarray())
                else:
                    post = self.factors[i+1][self.effectLbls[i+1]].toarray()
                    for j,fctA in enumerate(self.factors[i+2:],start=i+2):
                        post = _np.kron(post,fctA[self.effectLbls[j]].toarray())
                deriv = _np.kron(deriv, post[:,None]) # add a dummy 1-dim to 'post' and do kron properly...

            if self.typ == "prep":
                local_inds = fct.gpindices # factor vectors hold local indices
            else: # in effect case, POVM-factors hold global indices (b/c they're meant to be shareable)
                local_inds = _gatesetmember._decompose_gpindices(
                    self.gpindices, fct.gpindices)
                
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


    def copy(self, parent=None):
        """
        Copy this SPAM vector.

        Returns
        -------
        StaticSPAMVec
            A copy of this SPAM operation.
        """
        return self._copy_gpindices( TensorProdSPAMVec(self.typ, self.factors, self.effectLbls), parent )


    def __str__(self):
        s = "Tensor product %s vector with length %d\n" % (self.typ,self.dim)
        #ar = self.toarray()
        #s += _mt.mx_to_string(ar, width=4, prec=2)

        if self.typ == "prep":
            # factors are just other SPAMVecs
            s += " x ".join([_mt.mx_to_string(fct.toarray(), width=4, prec=2) for fct in self.factors])
        else:
            # factors are POVMs
            s += " x ".join([_mt.mx_to_string(fct[self.effectLbls[i]].toarray(), width=4, prec=2)
                             for i,fct in enumerate(self.factors)])
        return s



class PureStateSPAMVec(SPAMVec):
    """
    Encapsulates a SPAM vector that is completely fixed, or "static", meaning
      that is contains no parameters.
    """

    def __init__(self, pure_state_vec, dm_basis='pp'):
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

        dm_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The basis for this SPAM vector - that is, for the *density matrix*
            corresponding to `pure_state_vec`.  Allowed values are Matrix-unit
            (std),  Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
            (or a custom basis object).
        """
        if not isinstance(pure_state_vec,SPAMVec):
            pure_state_vec = StaticSPAMVec(SPAMVec.convert_to_vector(pure_state_vec))
        elif not isinstance(pure_state_vec,DenseSPAMVec):
            raise ValueError("Currently can only construct PureStateSPAMVecs from *dense* state vectors")
        self.pure_state_vec = pure_state_vec
        self.basis = dm_basis
        self._construct_vector()
        
        SPAMVec.__init__(self, self.pure_state_vec.dim**2)

    def _construct_vector(self):
        dmMx_std = _np.dot( self.pure_state_vec, self.pure_state_vec.conjugate().T )
        dmVec = _bt.change_basis(dmMx_std.flatten(), 'std', self.basis)

    @property
    def size(self):
        return self.dim
    
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
        self.pure_state_vec.set_value(vec)
        self.dirty = True

    def transform(self, S, typ):
        """
        Update SPAM (column) vector V as inv(S) * V or S^T * V for prep and
        effect SPAM vectors, respectively (depending on the value of `typ`). 

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
        raise ValueError("Cannot transform a PureStateSPAMVec")


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
        raise ValueError("Cannot depolarize a PureStateSPAMVec")


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


    def copy(self, parent=None):
        """
        Copy this SPAM vector.

        Returns
        -------
        StaticSPAMVec
            A copy of this SPAM operation.
        """
        return self._copy_gpindices( PureStateSPAMVec(self.pure_state_vec, self.basis), parent )


    def __str__(self):
        s = "Pure-state spam vector with length %d holding:\n" % self.dim
        s += "  " + str(self.pure_state_vec)
        return s


class TermSPAMVec(SPAMVec):

    @property
    def size(self):
        return self.dim


class LindbladTermSPAMVec(TermSPAMVec):
    """ A Lindblad-parameterized SPAMVec that is expandable into terms """

    # TODO: init & basis specification; maybe breakout an OrderedTermGate base class?

    #NOTE: These methods share a lot in common with those of LindbladTermGate - maybe can consolidate in FUTURE?
    # - mostly just change baseunitary => basepurestate
    #  (maybe add a termtools.py or add more helper functions to term.py?)
    @classmethod
    def from_error_generator(cls, basepurestate, errgen, ham_basis="pp", nonham_basis="pp", cptp=True,
                             nonham_diagonal_only=False, truncate=True, mxBasis="pp", termtype="dense"):
        # This looks like a from_errgen constructor; we want a from_rates constructor that
        # takes a dict(?) of rate coeffs (0 by default) associated with lindblad terms (ham_i terms and other_ij terms...)
        # --> add same construction methods to other lindblad gates, which currently just have "from_errgen"-type __init__ functions?
        """
        TODO: docstring (similar to other Lindblad gates above)
        """

        d2 = errgen.shape[0]
        d = int(round(_np.sqrt(d2)))
        assert(d*d == d2), "Gate dim must be a perfect square"
        
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

        #Get matrices (possibly sparse ones) from the bases
        if isinstance(ham_basis, _Basis) or _compat.isstr(ham_basis):
            ham_basis = _Basis(ham_basis,d,sparse=sparse)
        else: # ham_basis is a list of matrices
            ham_basis = _Basis(matrices=ham_basis,dim=d,sparse=sparse)
        
        if isinstance(nonham_basis, _Basis) or _compat.isstr(nonham_basis):
            other_basis = _Basis(nonham_basis,d,sparse=sparse)
        else: # ham_basis is a list of matrices
            other_basis = _Basis(matrices=nonham_basis,dim=d,sparse=sparse)
        
        matrix_basis = _Basis(mxBasis,d,sparse=sparse)

        
        hamC, otherC = \
            _gt.lindblad_errgen_projections(
                errgen, ham_basis, other_basis, matrix_basis, normalize=False,
                return_generators=False, other_diagonal_only=nonham_diagonal_only,
                sparse=sparse)
        print("DB: ham projections = ",hamC)
        print("DB: sto projections = ",otherC)

        # Make None => length-0 arrays so iteration code works below (when basis is None)
        if hamC is None: hamC = _np.empty(0,'d') 
        if otherC is None:
            otherC = _np.empty(0,'d') if nonham_diagonal_only \
                     else _np.empty((0,0),'d')

        Ltermdict = _collections.OrderedDict()
        basisdict = _collections.OrderedDict(); nextLbl = 0
        def get_basislbl(bmx):
            for l,b in basisdict.items():
                if _np.allclose(b,bmx): return l
            blbl = nextLbl
            basisdict[blbl] = bmx
            nextLbl += 1
            return blbl
        
        ham_mxs = ham_basis.get_composite_matrices()
        assert(len(ham_mxs[1:]) == len(hamC))

        for coeff, bmx in zip(hamC,ham_mxs[1:]): # skip identity
            Ltermdict[('H',nextLbl)] = coeff
            basisdict[nextLbl] = bmx
            nextLbl += 1
            
        other_mxs = other_basis.get_composite_matrices()
        if nonham_diagonal_only:
            assert(len(other_mxs[1:]) == len(otherC))
            for coeff, bmx in zip(otherC,other_mxs[1:]): # skip identity
                blbl = get_basislbl(bmx)
                Ltermdict[('S',blbl)] = coeff
        else:
            assert((len(other_mxs[1:]),len(other_mxs[1:])) == otherC.shape)
            for i, bmx1 in enumerate(other_mxs[1:]): # skip identity
                blbl1 = get_basislbl(bmx1)
                for j, bmx2 in enumerate(other_mxs[1:]): # skip identity
                    blbl2 = get_basislbl(bmx2)
                    Ltermdict[('S',blbl1,blbl2)] = otherC[i,j]

        print("DB: Ltermdict = ",Ltermdict)
        print("DB: basisdict = ")
        for k,v in basisdict.items():
            print(k,":")
            print(v)
        return cls.from_lindblad_terms(basepurestate, Ltermdict, basisdict,
                                       cptp, nonham_diagonal_only, truncate,
                                       termtype)



        
    @classmethod
    def from_lindblad_terms(cls, basepurestate, Ltermdict, basisdict=None,
                            cptp=True, nonham_diagonal_only=False, truncate=True,
                            termtype="dense"):
        """
        TODO: docstring
        Ltermdict keys are (termType, basisLabel(s)); values are floating point coeffs (error rates)
        basisdict keys are string/int basis element "names"; values are numpy matrices or an "embedded matrix",
                i.e. a *list* of (matrix, state_space_label) elements -- e.g. [(sigmaX,'Q1'), (sigmaY,'Q4')]
         -- maybe let keys be tuples of (basisname, state_space_label) e.g. (('X','Q1'),('Y','Q4')) -- and maybe allow ('XY','Q1','Q4')
                 format when can assume single-letter labels.
        """
        tt = "clifford" if (termtype in ("clifford prep","clifford effect")) else termtype
          # class distinguishes between prep/effect clifford vecs, but in this function we
          # just create gate-like terms which are just of type "clifford"
        
        #Enumerate the basis elements used for Hamiltonian and Stochasitic
        # error terms (separately)
        hamBasisLabels = _collections.OrderedDict()  # holds index of each basis element
        otherBasisLabels = _collections.OrderedDict()
        for termLbl,coeff in Ltermdict.items():
            termType = termLbl[0]
            if termType == "H": # Hamiltonian
                assert(len(termLbl) == 2),"Hamiltonian term labels should have form ('H',<basis element label>)"
                if termLbl[1] not in hamBasisLabels:
                    hamBasisLabels[ termLbl[1] ] = len(hamBasisLabels)
                    
            elif termType == "S": # Stochastic
                if nonham_diagonal_only == 'auto':
                    nonham_diagonal_only = bool(len(termLbl) == 2)
                    
                if nonham_diagonal_only:
                    assert(len(termLbl) == 2),"Stochastic term labels should have form ('S',<basis element label>)"
                    if termLbl[1] not in otherBasisLabels:
                        otherBasisLabels[ termLbl[1] ] = len(otherBasisLabels)
                else:
                    assert(len(termLbl) == 3),"Stochastic term labels should have form ('S',<bel1>, <bel2>)"
                    if termLbl[1] not in otherBasisLabels:
                        otherBasisLabels[ termLbl[1] ] = len(otherBasisLabels)
                    if termLbl[2] not in otherBasisLabels:
                        otherBasisLabels[ termLbl[2] ] = len(otherBasisLabels)

        #Get parameter counts based on # of basis elements
        numHamParams = len(hamBasisLabels)
        if nonham_diagonal_only:  # OK if this runs for 'auto' too since then len(otherBasisLabels) == 0
            numOtherParams = len(otherBasisLabels)
        else:
            numOtherBasisEls = len(otherBasisLabels)
            numOtherParams = numOtherBasisEls**2
            otherCoeffs = _np.zeros((numOtherParams,numOtherParams),'complex')
        nTotalParams = numHamParams + numOtherParams
        print("DB: hamBasisLabels = ",hamBasisLabels)
        print("DB: otherBasisLabels = ",otherBasisLabels)


        #Create & Fill parameter array
        params = _np.zeros(nTotalParams, 'd') # ham params, then "other"
        for termLbl,coeff in Ltermdict.items():
            termType = termLbl[0]
            if termType == "H": # Hamiltonian
                k = hamBasisLabels[termLbl[1]] #index of parameter
                assert(abs(_np.imag(coeff)) < IMAG_TOL)
                params[k] = _np.real_if_close(coeff)
            elif termType == "S": # Stochastic
                if nonham_diagonal_only:
                    k = numHamParams + otherBasisLabels[termLbl[1]] #index of parameter
                    assert(abs(_np.imag(coeff)) < IMAG_TOL)
                    if cptp:
                        if -1e-12 < coeff < 0: coeff = 0.0 # avoid sqrt warning due to small negative numbers
                        params[k] = _np.sqrt(_np.real_if_close(coeff))
                    else:
                        params[k] = _np.real_if_close(coeff)
                else:
                    k = otherBasisLabels[termLbl[1]] #index of row in "other" coefficient matrix
                    j = otherBasisLabels[termLbl[2]] #index of col in "other" coefficient matrix
                    otherCoeffs[k,j] = coeff
                    
        if not nonham_diagonal_only:
            #Finish up filling parameters for "other" terms - need to take
            # cholesky decomp of otherCoeffs matrix:
            otherParams = _np.empty((numOtherBasisEls,numOtherBasisEls),'d')

            #ROBIN: is this necessary? -- or just need otherCoeffs to be positive semidefinite?
            # Assume for now this must be true, as code for parameterizing otherCoeffs
            # depends upon it (and parameter counting indicates this probably is a valid condition)
            assert(_np.isclose(_np.linalg.norm(otherCoeffs-otherCoeffs.T.conjugate())
                               ,0)), "other coeff mx is not Hermitian!"

            if cptp: #otherParams mx stores Cholesky decomp
    
                #push any slightly negative evals of otherC positive so that
                # the Cholesky decomp will work.
                evals,U = _np.linalg.eig(otherCoeffs)
                Ui = _np.linalg.inv(U)
    
                assert(truncate or all([ev >= -1e-12 for ev in evals])), \
                    "Given error germs are not CPTP (truncate == False)!"
    
                pos_evals = evals.clip(1e-16,1e100)
                otherCoeffs = _np.dot(U,_np.dot(_np.diag(pos_evals),Ui))
                try:
                    Lmx = _np.linalg.cholesky(otherCoeffs)

                # if Lmx not postitive definite, try again with 1e-12 (same lines as above)
                except _np.linalg.LinAlgError:                              # pragma: no cover
                    pos_evals = evals.clip(1e-12,1e100)                     # pragma: no cover
                    otherCoeffs = _np.dot(U,_np.dot(_np.diag(pos_evals),Ui))# pragma: no cover
                    Lmx = _np.linalg.cholesky(otherCoeffs)                  # pragma: no cover
    
                for i in range(numOtherBasisEls):
                    assert(_np.linalg.norm(_np.imag(Lmx[i,i])) < IMAG_TOL)
                    otherParams[i,i] = Lmx[i,i].real
                    for j in range(i):
                        otherParams[i,j] = Lmx[i,j].real
                        otherParams[j,i] = Lmx[i,j].imag
    
            else: #otherParams mx stores otherC (hermitian) directly
                for i in range(numOtherBasisEls):
                    assert(_np.linalg.norm(_np.imag(otherC[i,i])) < IMAG_TOL)
                    otherParams[i,i] = otherC[i,i].real
                    for j in range(i):
                        otherParams[i,j] = otherC[i,j].real
                        otherParams[j,i] = otherC[i,j].imag

            params[numHamParams:] = otherParams.flatten()

        
        # Create Lindbladian terms - rank1 terms in the *exponent* with polynomial
        # coeffs (w/ *local* variable indices) that get converted to per-order
        # terms later.
        IDENT = None # sentinel for the do-nothing identity op
        Lterms = []
        for termLbl in Ltermdict:
            print("DB: processing ",termLbl)
            termType = termLbl[0]
            if termType == "H": # Hamiltonian
                k = hamBasisLabels[termLbl[1]] #index of parameter
                Lterms.append( _term.RankOneTerm(_Polynomial({(k,): -1j} ), basisdict[termLbl[1]], IDENT, tt) )
                Lterms.append( _term.RankOneTerm(_Polynomial({(k,): +1j} ), IDENT, basisdict[termLbl[1]].conjugate().T, tt) )
                print("DB: H term w/index %d= " % k, " len=",len(Lterms))
                #print("  coeff: ", list(Lterms[-1].coeff.keys()) )
                #print("  coeff: ", list(Lterms[-2].coeff.keys()) )
                #print("  coeff: ", list(Lterms[-1].coeff.inds) )
                #print("  coeff: ", list(Lterms[-2].coeff.inds) )

            elif termType == "S": # Stochastic
                if nonham_diagonal_only:
                    k = numHamParams + otherBasisLabels[termLbl[1]] #index of parameter
                    Lm = Ln = basisdict[termLbl[1]]
                    pw = 2 if cptp else 1 # power to raise parameter to in order to get coeff

                    Lm_dag = Lm.conjugate().T # assumes basis is dense (TODO: make sure works for sparse case too - and np.dots below!)
                    Ln_dag = Ln.conjugate().T
                    Lterms.append( _term.RankOneTerm(_Polynomial({(k,)*pw:  1.0} ), Ln, Lm_dag, tt) )
                    Lterms.append( _term.RankOneTerm(_Polynomial({(k,)*pw: -0.5} ), IDENT, _np.dot(Ln_dag,Lm), tt) )
                    Lterms.append( _term.RankOneTerm(_Polynomial({(k,)*pw: -0.5} ), _np.dot(Lm_dag,Ln), IDENT, tt) )
                else:
                    i = otherBasisLabels[termLbl[1]] #index of row in "other" coefficient matrix
                    j = otherBasisLabels[termLbl[2]] #index of col in "other" coefficient matrix
                    Lm, Ln = basisdict[termLbl[1]],basisdict[termLbl[2]]
                    print("DB: S indices = ",i,j)

                    # TODO: create these polys and place below...
                    polyTerms = {}
                    if cptp:
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
                    else: # coeff_ij = otherParam[i,j] + 1j*otherParam[j,i] (otherCoeffs is Hermitian)
                        ijIndx = numHamParams + (i*numOtherBasisEls + j)
                        jiIndx = numHamParams + (j*numOtherBasisEls + i)
                        polyTerms = { (ijIndx,): 1.0, (jiIndx,): 1.0j }

                    base_poly = _Polynomial(polyTerms)
                    Lm_dag = Lm.conjugate().T; Ln_dag = Ln.conjugate().T
                    Lterms.append( _term.RankOneTerm(1.0*base_poly, Ln, Lm, tt) )
                    Lterms.append( _term.RankOneTerm(-0.5*base_poly, IDENT, _np.dot(Ln_dag,Lm), tt) ) # adjoint(_np.dot(Lm_dag,Ln))
                    Lterms.append( _term.RankOneTerm(-0.5*base_poly, _np.dot(Lm_dag,Ln), IDENT, tt) )
                    print("DB: S term w/index terms= ", polyTerms, " len=",len(Lterms))
                    print("  coeff: ", list(Lterms[-1].coeff.keys()) )
                    print("  coeff: ", list(Lterms[-2].coeff.keys()) )
                    print("  coeff: ", list(Lterms[-3].coeff.keys()) )


                #TODO: check normalization of these terms vs those used in projections.

        print("DB: params = ", list(enumerate(params)))
        print("DB: Lterms = ")
        for i,lt in enumerate(Lterms):
            print("Term %d:" % i)
            print("  coeff: ", str(lt.coeff)) # list(lt.coeff.keys()) )
            print("  pre:\n", lt.pre_ops[0] if len(lt.pre_ops) else "IDENT")
            print("  post:\n",lt.post_ops[0] if len(lt.post_ops) else "IDENT")
        return cls(basepurestate, params, Lterms, cptp, nonham_diagonal_only, termtype)

    
    def __init__(self, basepurestate=None, initial_paramvals=None, Lterms=None,
                 cptp=True, nonham_diagonal_only=False, termtype="dense"):
        """
        Initialize a LindbladTermGateBase object.
        TODO: docstring
        """
        # 'basepurestate' can just be a dimension
        if isinstance(basepurestate,_numbers.Integral): 
            dim = basepurestate
            basepurestate = None
        else:
            try:
                dim = basepurestate.dim # if a SPAMVec
            except:
                # otherwise try to treat as array-like
                basepurestate = SPAMVec.convert_to_vector(basepurestate)
                dim = basepurestate.shape[0]

            # automatically "up-convert" gate to StabilizerSPAMVec if needed
            if termtype == "clifford prep" and not isinstance(
                    basepurestate, StabilizerSPAMVec):
                basepurestate = StabilizerSPAMVec.from_dense_purevec(basepurestate)
            elif termtype == "clifford effect" and not isinstance(
                    basepurestate, StabilizerEffectVec):
                #basepurestate = StabilizerEffectVec(outcomes, stabilizerZPOVM)
                raise ValueError(("Must supply a base StabilizerEffectVec when "
                                  " creating an 'clifford effect' LindbladTermSPAMVec"))
                
        self.termtype = termtype
        self.cptp = cptp
        self.nonham_diagonal_only = nonham_diagonal_only
        self.paramvals = _np.array(initial_paramvals,'d')
        self.Lterms = Lterms
        self.basepurestate = basepurestate
        self.terms = {}
        TermSPAMVec.__init__(self, dim**2) #sets self.dim

    #def get_max_order(self):
    #    if len(self.Lterms) > 0: return 2**32 # ~inf
    #    else: return 1

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

        self.terms = {} # clear terms cache since param indices have changed now
        _gatesetmember.GateSetMember.set_gpindices(self, gpindices, parent)

    def _compose_poly_indices(self, terms):
        for term in terms:
            term.map_indices(lambda x: tuple(_gatesetmember._compose_gpindices(
                self.gpindices, _np.array(x,'i'))) )
        return terms
        

    def get_order_terms(self, order):
        if order not in self.terms:
            assert(self.gpindices is not None),"LindbladTermGate must be added to a GateSet before use!"
            postTerm = _term.RankOneTerm(_Polynomial({(): 1.0}), self.basepurestate, self.basepurestate, self.termtype)
            loc_terms = _term.exp_terms(self.Lterms, [order], postTerm)[order]
            #loc_terms = [ t.collapse() for t in loc_terms ] # collapse terms for speed - resulting in terms with just a single pre/post op, each == a pure state
            self.terms[order] = self._compose_poly_indices(loc_terms)
        return self.terms[order]
    
    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return len(self.paramvals)

    #TODO: Are these to/from vector fns needed??? -- maybe calc only needs to get terms & construct polys...
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
        self.dirty = True

        
    def copy(self, parent=None):
        """
        Copy this gate.

        Returns
        -------
        Gate
            A copy of this gate.
        """
        return self._copy_gpindices( LindbladTermSPAMVec(
            self.basepurestate, self.paramvals, self.Lterms, self.cptp,
            self.nonham_diagonal_only, self.termtype), parent)


class StabilizerSPAMVec(SPAMVec):
    """
    TODO: docstring
    """

    @classmethod
    def from_dense_purevec(cls, purevec):
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
        self.smatrix, self.phasevec = _symp.prep_stabilizer_state(nqubits, zvals)
        dim = 2**nqubits # assume "unitary evolution"-type mode?
        SPAMVec.__init__(self, dim)

    @property
    def size(self):
        return self.dim
    
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
        raise ValueError("Cannot set the value of a StabilizerSPAMVec")

    def transform(self, S, typ):
        """
        Update SPAM (column) vector V as inv(S) * V or S^T * V for prep and
        effect SPAM vectors, respectively (depending on the value of `typ`). 

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
        raise ValueError("Cannot transform a StabilizerSPAMVec")


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
        raise ValueError("Cannot depolarize a StabilizerSPAMVec")


    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return 0


    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return _np.array([], 'd') # no parameters


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
        raise ValueError("Cannot take derivative of StabilizerSPAMVec")


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


    def copy(self, parent=None):
        """
        Copy this SPAM vector.

        Returns
        -------
        StaticSPAMVec
            A copy of this SPAM operation.
        """
        nQubits = len(self.phasevec) // 2
        ret = self._copy_gpindices( StabilizerSPAMVec(nQubits, None), parent )
        ret.smatrix = self.smatrix.copy()
        ret.phasevec = self.phasevec.copy()
        return ret

    def toarray(self, scratch=None):
        """
        Return this SPAM vector as a (dense) numpy array.  The memory
        in `scratch` maybe used when it is not-None.
        """
        return (self.smatrix,self.phasevec)  # need to concat?


    def __str__(self):
        nQubits = len(self.phasevec) // 2
        s = "Stabilizer spam vector for %d qubits with rep:\n" % nQubits
        s += "  " + _mt.mx_to_string(self.smatrix) + ", " + _mt.mx_to_string(self.phasevec) + "\n"
        return s


class StabilizerEffectVec(SPAMVec):
    """
    A dummy SPAM vector that points to a set/product of 1-qubit POVM 
    outcomes from stabilizer-state measurements.
    """

    def __init__(self, outcomes, stabilizerPOVM):
        """
        Initialize a StabilizerEffectVec object.

        Parameters
        ----------
        outcomes : iterable
            A list or other iterable of integer 0 or 1 outcomes specifying
            which POVM effect vector this object represents within the 
            full `stabilizerPOVM`

        stabilizerPOVM : StabilizerZPOVM
            The parent POVM object, used to cache 1Q-factor-POVM outcomes
            to avoid repeated calculations.
        """
        self.outcomes = _np.array(outcomes,int)
        self.parent_povm = stabilizerPOVM
        nqubits = len(outcomes)
        dim = 2**nqubits # assume "unitary evolution"-type mode?
        SPAMVec.__init__(self, dim)

    @property
    def size(self):
        return self.dim
    
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
        raise ValueError("Cannot set the value of a StabilizerEffectVec")

    def transform(self, S, typ):
        """
        Update SPAM (column) vector V as inv(S) * V or S^T * V for prep and
        effect SPAM vectors, respectively (depending on the value of `typ`). 

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
        raise ValueError("Cannot transform a StabilizerEffectVec")


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
        raise ValueError("Cannot depolarize a StabilizerEffectVec")


    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return 0


    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return _np.array([], 'd') # no parameters


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
        raise ValueError("Cannot take derivative of StabilizerEffectVec")


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


    def copy(self, parent=None):
        """
        Copy this SPAM vector.

        Returns
        -------
        StaticSPAMVec
            A copy of this SPAM operation.
        """
        return self._copy_gpindices( StabilizerEffectVec(self.outcomes, self.parent_povm), parent )
    

    def __str__(self):
        nQubits = len(self.outcomes)
        s = "Stabilizer effect vector for %d qubits with outcome %s" % (nQubits, str(self.outcomes))
        return s
