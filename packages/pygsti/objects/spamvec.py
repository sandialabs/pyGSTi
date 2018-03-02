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
from ..      import optimize as _opt
from ..tools import matrixtools as _mt
from ..tools import gatetools as _gt
from ..tools import slicetools as _slct
from ..tools import compattools as _compat
from ..baseobjs import Basis as _Basis
from ..baseobjs import ProtectedArray as _ProtectedArray
from . import gatesetmember as _gatesetmember

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


def convert(spamvec, toType, basis):
    """
    Convert SPAM vector to a new type of parameterization, potentially
    creating a new SPAMVec object.  Raises ValueError for invalid conversions.

    Parameters
    ----------
    spamvec : SPAMVec
        SPAM vector to convert

    toType : {"full","TP","static"}
        The type of parameterizaton to convert to.

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for `spamvec`.  Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
        (or a custom basis object).

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
        return self.base
                                   
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
        derivMx = _np.zeros((self.dim,0),'d')
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
        return self.dim


    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return self.base.flatten() #.real in case of complex matrices


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
        derivMx = _np.identity( self.dim, 'd' )
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
        if len(self.other_vecs) == 0: return _np.zeros((self.dim,0), 'd')
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
        elif typ == "prep":
            assert(povmEffectLbls is None), '`povmEffectLbls` must be None when `typ != "effects"`'
            self.effectLbls = None
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
        max_factor_dim = max(fct.dim for fct in factors)
        self._fast_kron_array = _np.empty( (len(factors), max_factor_dim), 'd')
        self._fast_kron_factordims = _np.array([fct.dim for fct in factors],'i')
        self._fill_fast_kron()

        
    def _fill_fast_kron(self):
        """ Fills in self._fast_kron_array based on current self.factors """
        if self.typ == "prep":
            for i,factor_dim in enumerate(self._fast_kron_factordims):
                self._fast_kron_array[i][0:factor_dim] = self.factors[i].toarray()[:,0]
        else:
            factorPOVMs = self.factors
            for i,(factor_dim,Elbl) in enumerate(zip(self._fast_kron_factordims,self.effectLbls)):
                self._fast_kron_array[i][0:factor_dim] = factorPOVMs[i][Elbl].toarray()[:,0]


    def toarray(self, scratch=None):
        """
        Return this SPAM vector as a (dense) numpy array.  The memory
        in `scratch` maybe used when it is not-None.
        """
        if len(self.factors) == 0: return _np.empty(0,'d')
        if scratch is not None and _fastcalc is not None:
            assert(scratch.shape[0] == self.dim)
            # use faster kron that avoids memory allocation.
            # Note: this uses more memory b/c all self.factors.toarray() results
            #  are present in memory at the *same* time - could add a flag to
            #  disable fast-kron-array when memory is extra tight(?).
            _fastcalc.fast_kron(scratch, self._fast_kron_array, self._fast_kron_factordims)
            return scratch[:,None] if scratch.ndim == 1 else scratch
            
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
        # - no dense matrices are stored
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
        raise NotImplementedError("Analytic derivatives of a TensorProdSPAMVec are not implemented yet")


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
        ar = self.toarray()
        s = "Tensor product spam vector with length %d\n" % len(ar)
        s += _mt.mx_to_string(ar, width=4, prec=2)
        return s
