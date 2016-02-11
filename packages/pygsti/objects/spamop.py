#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation              
#    This Software is released under the GPL license detailed    
#    in the file "license.txt" in the top-level pyGSTi directory 
#*****************************************************************
""" 
Defines classes with represent SPAM operations, along with supporting
functionality.
"""


class FullyParameterizedSPAMOp(object):
    """
    Encapsulates a SPAM vector that is fully parameterized, that is,
      each element of the SPAM vector is an independent parameter.
    """

    def __init__(self, vec):
        """ 
        Initialize a FullyParameterizedSPAMOp object.

        Parameters
        ----------
        vec : numpy array
            a 1D numpy array representing the SPAM operation.  The
            shape of this array sets the dimension of the SPAM op.
        """
        self.vec = vec
        self.dim = len(vec)

    def set_value(self, value):
        """
        Sets the value of the SPAM op.  In general, the "value" means one
          floating point number for each parameter.  In this case when all
          vector elements are parameters, the value is the SPAM vector itself.

        Parameters
        ----------
        value : numpy array
            A 1D numpy array representing the SPAM operation.

        Returns
        -------
        None
        """
        if len(value) != self.dim:
            raise ValueError("You can only assign a length-%d object" % self.dim
                             + "to this fully parameterized SPAM op")
        self.vec[:,0] = value[:]

    def value_dimension(self):
        """ 
        Get the dimensions of the parameterized "value" of
        this SPAMOp which can be set using set_value(...).

        Returns
        -------
        tuple of ints
            The dimension of the parameterized "value" of this gate.
        """
        return (self.dim,)

    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM op.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.dim

    def to_vector(self):
        """
        Get a vector of the underlying SPAM op parameters.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self.vec.flatten() #.real in case of complex vecs??

    def from_vector(self, v):
        """
        Initialize the SPAM op using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length 
            must == num_params().

        Returns
        -------
        None
        """
        self.vector[:,0] = v

    def transform(self, Si, S):
        """ 
        Transform this gate, so that:
          gate matrix => Si * gate matrix * S

        Parameters
        ----------
        Si : numpy array
            The matrix which left-multiplies gate matricies.

        S : numpy array
            The matrix which right-multiplies gate matricies.

        Returns
        -------
        None
        """
        raise NotImplementedError("Should be implemented by derived class")

    def deriv_wrt_params(self):
        """ 
        Construct a matrix whose columns are the
        derivatives of the SPAM vector with respect to a
        single param.  Thus, each column is of length gate_dim
        and there is one column per SPAM op parameter.

        Returns
        -------
        numpy array 
            Array of derivatives, shape == (gate_dim, nSPAMOpParams)
        """
        derivMx = _np.identity( self.dim, 'd' )
        return derivMx

    def copy(self):
        """
        Copy this SPAM operation.
        
        Returns
        -------
        SPAMOp
            A copy of this SPAM operation.
        """
        return FullyParameterizedSpamOp(self.vector)

    def __str__(self):
        s = "Fully Parameterized spam op with length %d\n" % \
            len(self.vector)
        s += _mt.mx_to_string(self.vector, width=4, prec=2)
        return s 


class FullyParameterizedRhoOp(FullyParameterizedSPAMOp):

    def __init__(self, vec):
        """ See FullyParameterizedSPAMOp """
        FullyParameterizedSPAMOp.__init__(self, vec)

    def transform(self, Si, S):
        """ See FullyParameterizedSPAMOp """
        self.vector = _np.dot(Si, self.vec)


class FullyParameterizedEOp(FullyParameterizedSPAMOp):

    def __init__(self, vec):
        """ See FullyParameterizedSPAMOp """
        FullyParameterizedSPAMOp.__init__(self, vec)

    def transform(self, Si, S):
        """ See FullyParameterizedSPAMOp """
        self.vector = _np.dot(_np.transpose(S), self.vec)
        # same as ( vec^T * S )^T


#TODO: "TPRhoOp" - perhaps just add bSP0 to FullyParameterizedSPAMOp,
# similar to how FullyParameterizedGate is now (which should then
# be divided into separate classes for "TP" and "non-TP" versions...)
