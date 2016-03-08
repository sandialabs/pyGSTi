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

class SPAMVec(object):
    """
    Excapulates a parameterization of a state preparation OR POVM effect
    vector. This class is the  common base class for all specific
    parameterizations of a SPAM vector.
    """

    def __init__(self, dim):
        """ Initialize a new SPAM Vector """
        self.dim = dim

    def get_dimension(self):
        """ Return the dimension of the gate matrix. """
        return self.dim

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
            vector = V.construct_vector()
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
                vector = _np.array(V) #vec is already a 2-D column vector
            else:
                vector = _np.array(V)[:,None] # make into a 2-D column vec
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
        self.vector = SPAMVec.convert_to_vector(vec)
        super(StaticSPAMVec, self).__init__( len(self.vector) )

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
        raise ValueError("Cannot set the vector of a StaticSPAMVec (no parameters!)")


    def construct_vector(self):
        """
        Build and return the SPAM vector using the current parameters.

        Returns
        -------
        numpy array
            The SPAM column vector with shape (dim,1)
        """
        return self.vector


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
            len(self.vector)
        s += _mt.mx_to_string(self.vector, width=4, prec=2)
        return s 




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
        self.vector = SPAMVec.convert_to_vector(vec)
        super(FullyParameterizedSPAMVec, self).__init__( len(self.vector) )


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
        if(vec.shape != (self.dim, 1)):
            raise ValueError("Argument must be a vector of dim %d" % self.dim)
        self.vector = vec


    def construct_vector(self):
        """
        Build and return the SPAM vector using the current parameters.

        Returns
        -------
        numpy array
            The SPAM column vector with shape (dim,1)
        """
        return self.vector


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
        return self.vector.flatten() #.real in case of complex matrices


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

        self.vector = v.reshape(self.vector.shape)
        # OR JUST: self.vector[:,0] = v


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
        return  _np.identity( self.dim, 'd' )


#Helpful??
#            for (i,rhoVec) in enumerate(self.rhoVecs):
#            deriv[foff+m:foff+m+rhoSize[i],off:off+rhoSize[i]] = _np.identity( rhoSize[i], 'd' )
#                off += rhoSize[i]; foff += full_vsize
#
#            for (i,EVec) in enumerate(self.EVecs):
#                deriv[foff:foff+eSize[i],off:off+eSize[i]] = _np.identity( eSize[i], 'd' )
#                off += eSize[i]; foff += full_vsize


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
        s = "Fully Parameterized spam vector with length %d\n" % \
            len(self.vector)
        s += _mt.mx_to_string(self.vector, width=4, prec=2)
        return s 


#TODO: "TPRhoOp" - perhaps just add bSP0 to FullyParameterizedSPAMOp,
# similar to how FullyParameterizedGate is now (which should then
# be divided into separate classes for "TP" and "non-TP" versions...)


class TPParameterizedSPAMVec(SPAMVec):
    """
    Encapsulates a SPAM vector that is fully parameterized except for the first
    element, which is frozen to be "1".  This is so that, when the SPAM vector is
    interpreted in the Pauli or Gell-Mann basis, the represented density matrix
    has trace == 1.  This restriction is frequently used in conjuction with 
    trace-preserving (TP) gates.
    """

    def __init__(self, vec):
        """ 
        Initialize a TPParameterizedSPAMOp object.

        Parameters
        ----------
        vec : array_like or SPAMVec
            a 1D numpy array representing the SPAM operation.  The
            shape of this array sets the dimension of the SPAM op.
        """
        self.vector = SPAMVec.convert_to_vector(vec)
        if not _np.isclose(self.vector[0,0], 1.0):
            raise ValueError("Cannot create TPParameterizedSPAMVec: " + 
                             "first element must equal 1.0!")

        super(TPParameterizedSPAMVec, self).__init__( len(self.vector) )


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
        if(vec.shape != (self.dim, 1)):
            raise ValueError("Argument must be a vector of dim %d" % self.dim)
        if not _np.isclose(vec[0,0], 1.0):
            raise ValueError("Cannot create TPParameterizedSPAMVec: " + 
                             "first element must equal 1.0!")
        self.vector = vec



    def construct_vector(self):
        """
        Build and return the SPAM vector using the current parameters.

        Returns
        -------
        numpy array
            The SPAM column vector with shape (dim,1)
        """
        return self.vector


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
        return self.vector.flatten()[1:] #.real in case of complex matrices


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
        self.vector[0,0] = 1.0
        self.vector[1:,0] = v


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
        s = "TP-Parameterized spam vector with length %d\n" % \
            len(self.vector)
        s += _mt.mx_to_string(self.vector, width=4, prec=2)
        return s 
