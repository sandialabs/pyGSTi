import numpy as _np
from .state import State as _State
from .densestate import DenseState as _DenseState
from ...evotypes import Evotype as _Evotype

from ...objects.protectedarray import ProtectedArray as _ProtectedArray


class TPState(_DenseState):
    """
    A fixed-unit-trace state vector.

    This SPAM vector is fully parameterized except for the first element, which
    is frozen to be 1/(d**0.25).  This is so that, when the SPAM vector is
    interpreted in the Pauli or Gell-Mann basis, the represented density matrix
    has trace == 1.  This restriction is frequently used in conjuction with
    trace-preserving (TP) gates, hence its name.

    Parameters
    ----------
    vec : array_like or SPAMVec
        a 1D numpy array representing the SPAM operation.  The
        shape of this array sets the dimension of the SPAM op.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.
    """

    #Note: here we assume that the first basis element is (1/sqrt(x) * I),
    # where I the d-dimensional identity (where len(vector) == d**2). So
    # if Tr(basisEl*basisEl) == Tr(1/x*I) == d/x must == 1, then we must
    # have x == d.  Thus, we multiply this first basis element by
    # alpha = 1/sqrt(d) to obtain a trace-1 matrix, i.e., finding alpha
    # s.t. Tr(alpha*[1/sqrt(d)*I]) == 1 => alpha*d/sqrt(d) == 1 =>
    # alpha = 1/sqrt(d) = 1/(len(vec)**0.25).
    def __init__(self, vec, evotype="default"):
        vector = _State._to_vector(vec)
        firstEl = len(vector)**-0.25
        if not _np.isclose(vector[0], firstEl):
            raise ValueError("Cannot create TPSPAMVec: "
                             "first element must equal %g!" % firstEl)

        evotype = _Evotype.cast(evotype)
        rep = evotype.create_dense_state_rep(vector)
        _DenseState.__init__(self, rep, evotype, rep.base)
        assert(isinstance(self.base, _ProtectedArray))
        self._paramlbls = _np.array(["VecElement %d" % i for i in range(1, self.dim)], dtype=object)

    def _base_1d_has_changed(self):
        self._rep.base_has_changed()

    @property
    def base(self):
        """
        Direct access the the underlying data as column vector, i.e, a (dim,1)-shaped array.
        """
        bv = self._base_1d.view()
        bv.shape = (bv.size, 1)
        return _ProtectedArray(bv, indices_to_protect=(0, 0))

    def set_dense(self, vec):
        """
        Set the dense-vector value of this SPAM vector.

        Attempts to modify this SPAM vector's parameters so that the raw
        SPAM vector becomes `vec`.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        vec : array_like or SPAMVec
            A numpy array representing a SPAM vector, or a SPAMVec object.

        Returns
        -------
        None
        """
        vec = _State._to_vector(vec)
        firstEl = (self.dim)**-0.25
        if(vec.size != self.dim):
            raise ValueError("Argument must be length %d" % self.dim)
        if not _np.isclose(vec[0], firstEl):
            raise ValueError("Cannot create TPSPAMVec: "
                             "first element must equal %g!" % firstEl)
        self._base_1d[1:] = vec[1:]
        self._base_1d_has_changed()
        self.dirty = True

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return self.dim - 1

    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return self._base_1d[1:]  # .real in case of complex matrices?

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
        #assert(_np.isclose(self._base_1d[0], (self.dim)**-0.25))  # takes too much time!
        self._base_1d[1:] = v
        self._base_1d_has_changed()
        self.dirty = dirty_value

    def deriv_wrt_params(self, wrt_filter=None):
        """
        The element-wise derivative this SPAM vector.

        Construct a matrix whose columns are the derivatives of the SPAM vector
        with respect to a single param.  Thus, each column is of length
        dimension and there is one column per SPAM vector parameter.

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
        derivMx = _np.identity(self.dim, 'd')  # TP vecs assumed real
        derivMx = derivMx[:, 1:]  # remove first col ( <=> first-el parameters )
        if wrt_filter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrt_filter, axis=1)

    def has_nonzero_hessian(self):
        """
        Whether this SPAM vector has a non-zero Hessian with respect to its parameters.

        Returns
        -------
        bool
        """
        return False
