
from .effect import POVMEffect as _POVMEffect
from ...tools import matrixtools as _mt


class ConjugatedStatePOVMEffect(_POVMEffect):
    """
    TODO: update docstring
    A POVM effect vector that behaves like a numpy array.

    This class is the common base class for parameterizations of an effect vector
    that have a dense representation and can be accessed like a numpy array.

    Parameters
    ----------
    vec : numpy.ndarray
        The SPAM vector as a dense numpy array.

    evotype : {"statevec", "densitymx"}
        The evolution type.

    Attributes
    ----------
    _base_1d : numpy.ndarray
        Direct access to the underlying 1D array.

    base : numpy.ndarray
        Direct access the the underlying data as column vector,
        i.e, a (dim,1)-shaped array.
    """

    def __init__(self, state):
        self.state = state
        evotype = state._evotype
        rep = evotype.create_conjugatedstate_effect_rep(state._rep)
        super(ConjugatedStatePOVMEffect, self).__init__(rep, evotype)

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        return self.state.parameter_labels

    def to_dense(self, scratch=None):
        """
        Return this SPAM vector as a (dense) numpy array.

        The memory in `scratch` maybe used when it is not-None.

        Parameters
        ----------
        scratch : numpy.ndarray, optional
            scratch space available for use.

        Returns
        -------
        numpy.ndarray
        """
        #don't use scratch since we already have memory allocated
        return self._rep.to_dense()  # conjugate?

    @property
    def size(self):
        return self.state.size

    def __str__(self):
        s = "%s with dimension %d\n" % (self.__class__.__name__, self.dim)
        s += _mt.mx_to_string(self.to_dense(), width=4, prec=2)
        return s

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.state]

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return self.state.num_params

    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return self.state.to_vector()

    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize the SPAM vector using a 1D array of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of SPAM vector parameters.  Length
            must == num_params()

        close : bool, optional
            Whether `v` is close to this SPAM vector's current
            set of parameters.  Under some circumstances, when this
            is true this call can be completed more quickly.

        dirty_value : bool, optional
            The value to set this object's "dirty flag" to before exiting this
            call.  This is passed as an argument so it can be updated *recursively*.
            Leave this set to `True` unless you know what you're doing.

        Returns
        -------
        None
        """
        self.state.from_vector(v, close, dirty_value)
        self.dirty = dirty_value

    def deriv_wrt_params(self, wrt_filter=None):
        """
        The element-wise derivative this SPAM vector.

        Construct a matrix whose columns are the derivatives of the SPAM vector
        with respect to a single param.  Thus, each column is of length
        dimension and there is one column per SPAM vector parameter.
        An empty 2D array in the StaticSPAMVec case (num_params == 0).

        Parameters
        ----------
        wrt_filter : list or numpy.ndarray
            List of parameter indices to take derivative with respect to.
            (None means to use all the this operation's parameters.)

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension, num_params)
        """
        return self.state.deriv_wrt_params(wrt_filter)

    def has_nonzero_hessian(self):
        """
        Whether this SPAM vector has a non-zero Hessian with respect to its parameters.

        Returns
        -------
        bool
        """
        #Default: assume Hessian can be nonzero if there are any parameters
        return self.state.has_nonzero_hessian()

    def hessian_wrt_params(self, wrt_filter1=None, wrt_filter2=None):
        """
        Construct the Hessian of this SPAM vector with respect to its parameters.

        This function returns a tensor whose first axis corresponds to the
        flattened operation matrix and whose 2nd and 3rd axes correspond to the
        parameters that are differentiated with respect to.

        Parameters
        ----------
        wrt_filter1 : list or numpy.ndarray
            List of parameter indices to take 1st derivatives with respect to.
            (None means to use all the this operation's parameters.)

        wrt_filter2 : list or numpy.ndarray
            List of parameter indices to take 2nd derivatives with respect to.
            (None means to use all the this operation's parameters.)

        Returns
        -------
        numpy array
            Hessian with shape (dimension, num_params1, num_params2)
        """
        return self.state.hessian_wrt_params(wrt_filter1, wrt_filter2)
