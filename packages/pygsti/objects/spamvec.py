from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""
Defines classes with represent SPAM operations, along with supporting
functionality.
"""

import numpy as _np
from ..      import optimize as _opt
from ..tools import matrixtools as _mt
from .protectedarray import ProtectedArray as _ProtectedArray


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
        vecToOptimize.set_vector(targetVec)     #just copy entire overall matrix since fully parameterized
        return

    assert(targetVec.dim == vecToOptimize.dim) #vectors must have the same overall dimension
    targetVector = _np.asarray(targetVec)
    import sys
    def objective_func(param_vec):
        vecToOptimize.from_vector(param_vec)
        return _mt.frobeniusnorm(vecToOptimize-targetVector)

    x0 = vecToOptimize.to_vector()
    minSol = _opt.minimize(objective_func, x0, method='BFGS', maxiter=10000, maxfev=10000,
                           tol=1e-6, callback=None)

    vecToOptimize.from_vector(minSol.x)
    #print("DEBUG: optimized vector to min frobenius distance %g" % _mt.frobeniusnorm(vecToOptimize-targetVector))


def convert(spamvec, toType):
    """
    Convert SPAM vector to a new type of parameterization, potentially
    creating a new SPAMVec object.  Raises ValueError for invalid conversions.

    Parameters
    ----------
    spamvec : SPAMVec
        SPAM vector to convert

    toType : {"full","TP","static"}
        The type of parameterizaton to convert to.

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

    elif toType == "static":
        if isinstance(spamvec, StaticSPAMVec):
            return spamvec #no conversion necessary
        else:
            return StaticSPAMVec( spamvec )

    else:
        raise ValueError("Invalid toType argument: %s" % toType)



class SPAMVec(object):
    """
    Excapulates a parameterization of a state preparation OR POVM effect
    vector. This class is the  common base class for all specific
    parameterizations of a SPAM vector.
    """

    def __init__(self, vec):
        """ Initialize a new SPAM Vector """
        self.base = vec
        self.dim = len(vec)

    def get_dimension(self):
        """ Return the dimension of the gate matrix. """
        return self.dim

    def transform(self, S):
        """
        Update SPAM (column) vector V as inv(S) * V or S^T * V for prep and
        effect SPAM vectors, respectively (depending on the value of `typ`). 
        """
        raise NotImplementedError("This SPAM vector cannot be tranform()'d")

    def depolarize(self, amount):
        """ Depolarize spam vector by the given amount. """
        raise NotImplementedError("This SPAM vector cannot be depolarize()'d")

    #Handled by derived classes
    #def __str__(self):
    #    s = "Spam vector with length %d\n" % len(self.base)
    #    s += _mt.mx_to_string(self.base, width=4, prec=2)
    #    return s


    #Pickle plumbing
    def __setstate__(self, state):
        self.__dict__.update(state)

    #Access to underlying array
    def __getitem__( self, key ):
        return self.base.__getitem__(key)

    def __getslice__(self, i,j):
        return self.__getitem__(slice(i,j)) #Called for A[:]

    def __setitem__(self, key, val):
        return self.base.__setitem__(key,val)

    def __getattr__(self, attr):
        #use __dict__ so no chance for recursive __getattr__
        ret = getattr(self.__dict__['base'],attr)
        if(self.base.shape != (self.dim,1)):
            raise ValueError("Cannot change dimension of Vector")
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
                vector = _np.array(V, typ) #vec is already a 2-D column vector
            else:
                typ = 'd' if _np.all(_np.isreal(V)) else 'complex'
                vector = _np.array(V, typ)[:,None] # make into a 2-D column vec

        return vector




class StaticSPAMVec(SPAMVec):
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
        SPAMVec.__init__(self, SPAMVec.convert_to_vector(vec))


    def set_vector(self, vec):
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
        S : GaugeGroup.element
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
        amount : float
            The amount to depolarize by.  All but the first element
            of vector are multiplied by `1.0 - amount`.

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


    def deriv_wrt_params(self):
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
        return _np.zeros((self.dim,0),'d')


    def copy(self):
        """
        Copy this SPAM vector.

        Returns
        -------
        StaticSPAMVec
            A copy of this SPAM operation.
        """
        return StaticSPAMVec(self.base)


    def __str__(self):
        s = "Static spam vector with length %d\n" % \
            len(self.base)
        s += _mt.mx_to_string(self.base, width=4, prec=2)
        return s

    def __reduce__(self):
        return (StaticSPAMVec, (_np.empty((self.dim,1),'d'),), self.__dict__)





class FullyParameterizedSPAMVec(SPAMVec):
    """
    Encapsulates a SPAM vector that is fully parameterized, that is,
      each element of the SPAM vector is an independent parameter.
    """

    def __init__(self, vec):
        """
        Initialize a FullyParameterizedSPAMOp object.

        Parameters
        ----------
        vec : array_like or SPAMVec
            a 1D numpy array representing the SPAM operation.  The
            shape of this array sets the dimension of the SPAM op.
        """
        SPAMVec.__init__(self, SPAMVec.convert_to_vector(vec))


    def set_vector(self, vec):
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
        S : GaugeGroup.element
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.
            
        typ : { 'prep', 'effect' }
            Which type of SPAM vector is being transformed (see above).
        """
        if typ == 'prep':
            Si  = S.get_transform_matrix_inverse()
            self.set_vector(_np.dot(Si, self))
        elif typ == 'effect':
            Smx = S.get_transform_matrix()
            self.set_vector(_np.dot(_np.transpose(Smx),self)) 
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
        amount : float
            The amount to depolarize by.  All but the first element
            of vector are multiplied by `1.0 - amount`.

        Returns
        -------
        None
        """
        D = _np.diag( [1]+[1-amount]*(self.dim-1) )
        self.set_vector(_np.dot(D,self)) 


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


    def deriv_wrt_params(self):
        """
        Construct a matrix whose columns are the derivatives of the SPAM vector
        with respect to a single param.  Thus, each column is of length
        get_dimension and there is one column per SPAM vector parameter.

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension, num_params)
        """
        return _np.identity( self.dim, 'd' )


    def copy(self):
        """
        Copy this SPAM vector.

        Returns
        -------
        FullyParameterizedSPAMVec
            A copy of this SPAM operation.
        """
        return FullyParameterizedSPAMVec(self.base)

    def __str__(self):
        s = "Fully Parameterized spam vector with length %d\n" % len(self.base)
        s += _mt.mx_to_string(self.base, width=4, prec=2)
        return s

    def __reduce__(self):
        return (FullyParameterizedSPAMVec, (_np.empty((self.dim,1),'d'),), self.__dict__)



#Helpful for deriv_wrt_params??
#            for (i,rhoVec) in enumerate(self.preps):
#            deriv[foff+m:foff+m+rhoSize[i],off:off+rhoSize[i]] = _np.identity( rhoSize[i], 'd' )
#                off += rhoSize[i]; foff += full_vsize
#
#            for (i,EVec) in enumerate(self.effects):
#                deriv[foff:foff+eSize[i],off:off+eSize[i]] = _np.identity( eSize[i], 'd' )
#                off += eSize[i]; foff += full_vsize


class TPParameterizedSPAMVec(SPAMVec):
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
        Initialize a TPParameterizedSPAMOp object.

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
        SPAMVec.__init__(self, _ProtectedArray(vector,
                                indicesToProtect=(0,0)))


    def set_vector(self, vec):
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
        S : GaugeGroup.element
            A gauge group element which specifies the "S" matrix 
            (and it's inverse) used in the above similarity transform.
            
        typ : { 'prep', 'effect' }
            Which type of SPAM vector is being transformed (see above).
        """
        if typ == 'prep':
            Si  = S.get_transform_matrix_inverse()
            self.set_vector(_np.dot(Si, self))
        elif typ == 'effect':
            Smx = S.get_transform_matrix()
            self.set_vector(_np.dot(_np.transpose(Smx),self)) 
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
        amount : float
            The amount to depolarize by.  All but the first element
            of vector are multiplied by `1.0 - amount`.

        Returns
        -------
        None
        """
        D = _np.diag( [1]+[1-amount]*(self.dim-1) )
        self.set_vector(_np.dot(D,self)) 


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


    def deriv_wrt_params(self):
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
        return derivMx

    def copy(self):
        """
        Copy this SPAM vector.

        Returns
        -------
        TPParameterizedSPAMVec
            A copy of this SPAM operation.
        """
        return TPParameterizedSPAMVec(self.base)

    def __str__(self):
        s = "TP-Parameterized spam vector with length %d\n" % self.dim
        s += _mt.mx_to_string(self.base, width=4, prec=2)
        return s

    def __reduce__(self):
        return (TPParameterizedSPAMVec, (self.base.copy(),), self.__dict__)





#SCRATCH: TO REMOVE
#    def __len__(self):
#        return len(self.base)
#
#    def __add__(self,x):
#        if isinstance(x, SPAMVec):
#            return self.base + x.base
#        else:
#            return self.base + x
#
#    def __radd__(self,x):
#        if isinstance(x, SPAMVec):
#            return x.base + self.base
#        else:
#            return x + self.base
#
#    def __sub__(self,x):
#        if isinstance(x, SPAMVec):
#            return self.base - x.base
#        else:
#            return self.base - x
#
#    def __rsub__(self,x):
#        if isinstance(x, SPAMVec):
#            return x.base - self.base
#        else:
#            return x - self.base
#
#    def __mul__(self,x):
#        if isinstance(x, SPAMVec):
#            return self.base * x.base
#        else:
#            return self.base * x
#
#    def __rmul__(self,x):
#        if isinstance(x, SPAMVec):
#            return x.base * self.base
#        else:
#            return x * self.base
#
#    def __pow__(self,x): #same as __mul__()
#        return self.base ** x
#
#    def __eq__(self,x):
#        if isinstance(x, SPAMVec):
#            return self.base == x.base
#        else:
#            return self.base == x
