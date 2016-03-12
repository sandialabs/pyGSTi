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
from ..tools import matrixtools as _mt
from protectedarray import ProtectedArray as _ProtectedArray

def convert(spamvec, toType):
    """
    Convert SPAM vector to a new type of parameterization, potentially
    creating a new SPAMVec object.  Raises ValueError for invalid conversions.

    Parameters
    ----------
    spamvec : SPAMVec
        SPAM vector to convert

    toType : {"full","tp","static"}
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

    elif toType == "tp":
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



class SPAMVec(object): #LATER: make this into an ABC specifying the "SPAMVec" interface??
    """
    Excapulates a parameterization of a state preparation OR POVM effect
    vector. This class is the  common base class for all specific
    parameterizations of a SPAM vector.
    """

    #def __init__(self, dim):
    #    """ Initialize a new SPAM Vector """
    #    self.dim = dim
    #
    #def get_dimension(self):
    #    """ Return the dimension of the gate matrix. """
    #    return self.dim

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
                dim = len(V)
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

        transposed = False # column vector
        return vector, transposed



class StaticSPAMVec(_np.ndarray, SPAMVec):
    """ 
    Encapsulates a SPAM vector that is completely fixed, or "static", meaning
      that is contains no parameters.
    """
    __array_priority__ = -1.0

    def __new__(cls, vec):
        """ 
        Initialize a StaticSPAMVec object.

        Parameters
        ----------
        vec : array_like or SPAMVec
            a 1D numpy array representing the SPAM operation.  The
            shape of this array sets the dimension of the SPAM op.
        """
        vector, T = SPAMVec.convert_to_vector(vec)
        obj = _np.asarray(vector).view(cls)
        obj.dim = len(vector)
        obj.transposed = T
        return obj

    def __getitem__( self, key ):
        #Items and slices are *not* gates - just numpy arrays:
        return _np.ndarray.__getitem__(self.view(_np.ndarray), key)

    def __getslice__(self, i,j):
        #For special cases when getslice is still called, e.g. A[:]
        return self.__getitem__(slice(i,j))

    def __array_finalize__(self, obj):
        if obj is None: return # let __new__ handle flags
        if len(self.shape) == 2:
            if self.shape[1] == 1:   # column vector
                self.dim,self.transposed = self.shape[0],False
            elif self.shape[0] == 1: # row vector
                self.dim,self.transposed = self.shape[1],True
            else:
                raise ValueError("SPAMVecs must have a unit dimension")
        else: raise ValueError("SPAMVecs must be 2D arrays")


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
        vec, T = SPAMVec.convert_to_vector(vec)
        if T != self.transposed:
            raise ValueError("Transpose mismatch: must set vector " +
                             "using the same shape")
        if(vec.size != self.dim):
            raise ValueError("Argument must be length %d" % self.dim) 
        self[:,:] = vec



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
        return StaticSPAMVec(self)

    def __str__(self):
        s = "Static spam vector with length %d\n" % \
            len(self)
        s += _mt.mx_to_string(self, width=4, prec=2)
        return s 

    def __reduce__(self):
        createFn, args, state = _np.ndarray.__reduce__(self)
        new_state = state + (self.__dict__,)
        return (createFn, args, new_state)

    def __setstate__(self, state):
        _np.ndarray.__setstate__(self,state[0:-1])
        self.__dict__.update(state[-1])





class FullyParameterizedSPAMVec(_np.ndarray, SPAMVec):
    """
    Encapsulates a SPAM vector that is fully parameterized, that is,
      each element of the SPAM vector is an independent parameter.
    """
    __array_priority__ = -1.0

    def __new__(cls, vec):
        """ 
        Initialize a FullyParameterizedSPAMOp object.

        Parameters
        ----------
        vec : array_like or SPAMVec
            a 1D numpy array representing the SPAM operation.  The
            shape of this array sets the dimension of the SPAM op.
        """
        vector,T = SPAMVec.convert_to_vector(vec)
        obj = _np.asarray(vector).view(cls)
        obj.dim = len(vector)
        obj.transposed = T
        return obj

        #Note: alternative to asarray line above is to call:
        # obj=_np.ndarray.__new__(cls,vector.shape)
        # obj[:,:] = vector.copy()

    def __getitem__( self, key ):
        #Items and slices are *not* gates - just numpy arrays:
        return _np.ndarray.__getitem__(self.view(_np.ndarray), key)

    def __getslice__(self, i,j):
        #For special cases when getslice is still called, e.g. A[:]
        return self.__getitem__(slice(i,j))

    def __array_finalize__(self, obj):
        if obj is None: return # let __new__ handle flags
        if len(self.shape) == 2:
            if self.shape[1] == 1:   # column vector
                self.dim,self.transposed = self.shape[0],False
            elif self.shape[0] == 1: # row vector
                self.dim,self.transposed = self.shape[1],True
            else:
                raise ValueError("SPAMVecs must have a unit dimension")
        else: raise ValueError("SPAMVecs must be 2D arrays")


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
        vec,T = SPAMVec.convert_to_vector(vec)
        if T != self.transposed:
            raise ValueError("Transpose mismatch: must set vector " +
                             "using the same shape")
        if(vec.size != self.dim):
            raise ValueError("Argument must be length %d" % self.dim) 
        self[:,:] = vec


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
        return self.flatten() #.real in case of complex matrices


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
        self[:,0] = v


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
        return FullyParameterizedSPAMVec(self)

    def __str__(self):
        s = "Fully Parameterized spam vector with length %d\n" % len(self)
        s += _mt.mx_to_string(self, width=4, prec=2)
        return s 

    def __reduce__(self):
        createFn, args, state = _np.ndarray.__reduce__(self)
        new_state = state + (self.__dict__,)
        return (createFn, args, new_state)

    def __setstate__(self, state):
        _np.ndarray.__setstate__(self,state[0:-1])
        self.__dict__.update(state[-1])



#Helpful for deriv_wrt_params??
#            for (i,rhoVec) in enumerate(self.rhoVecs):
#            deriv[foff+m:foff+m+rhoSize[i],off:off+rhoSize[i]] = _np.identity( rhoSize[i], 'd' )
#                off += rhoSize[i]; foff += full_vsize
#
#            for (i,EVec) in enumerate(self.EVecs):
#                deriv[foff:foff+eSize[i],off:off+eSize[i]] = _np.identity( eSize[i], 'd' )
#                off += eSize[i]; foff += full_vsize


class TPParameterizedSPAMVec(_ProtectedArray, SPAMVec):
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
    __array_priority__ = -1.0

    def __new__(cls, vec):
        """ 
        Initialize a TPParameterizedSPAMOp object.

        Parameters
        ----------
        vec : array_like or SPAMVec
            a 1D numpy array representing the SPAM operation.  The
            shape of this array sets the dimension of the SPAM op.
        """
        vector,T = SPAMVec.convert_to_vector(vec)
        firstEl =  len(vector)**-0.25
        if not _np.isclose(vector[0,0], firstEl):
            raise ValueError("Cannot create TPParameterizedSPAMVec: " + 
                             "first element must equal %g!" % firstEl)
        obj = _ProtectedArray.__new__(cls, vector, indicesToProtect=(0,0))
        obj.dim = len(vector)
        obj.transposed = T
        return obj

    def __getitem__( self, key ):
        #Items and slices are *not* gates - just numpy arrays:
        return _ProtectedArray.__getitem__(self.view(_ProtectedArray), key)

    def __getslice__(self, i,j):
        #For special cases when getslice is still called, e.g. A[:]
        return self.__getitem__(slice(i,j))

    def __array_finalize__(self, obj):
        _ProtectedArray.__array_finalize__(self, obj) #call base class handler
        if obj is None: return # let __new__ handle flags
        if len(self.shape) == 2:
            if self.shape[1] == 1:   # column vector
                self.dim,self.transposed = self.shape[0],False
            elif self.shape[0] == 1: # row vector
                self.dim,self.transposed = self.shape[1],True
            else:
                raise ValueError("SPAMVecs must have a unit dimension")
        else: raise ValueError("SPAMVecs must be 2D arrays")


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
        vec,T = SPAMVec.convert_to_vector(vec)
        firstEl =  (self.dim)**-0.25
        if T != self.transposed:
            raise ValueError("Transpose mismatch: must set vector " +
                             "using the same shape")
        if(vec.size != self.dim):
            raise ValueError("Argument must be length %d" % self.dim) 

        if not _np.isclose(vec[0,0], firstEl):
            raise ValueError("Cannot create TPParameterizedSPAMVec: " + 
                             "first element must equal %g!" % firstEl)
        if self.transposed:
            self[:,1:] = vec[:,1:]
        else:
            self[1:,:] = vec[1:,:]


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
        return self.flatten()[1:] #.real in case of complex matrices?


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
        self[0,0] = (self.dim)**-0.25 #necessary? (should be set already)
        self[1:,0] = v


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
        return TPParameterizedSPAMVec(self)

    def __str__(self):
        s = "TP-Parameterized spam vector with length %d\n" % len(self)
        s += _mt.mx_to_string(self, width=4, prec=2)
        return s 

    def __reduce__(self):
        createFn, args, state = _np.ndarray.__reduce__(self)
        new_state = state + (self.__dict__,)
        return (createFn, args, new_state)

    def __setstate__(self, state):
        _np.ndarray.__setstate__(self,state[0:-1])
        self.__dict__.update(state[-1])


    
