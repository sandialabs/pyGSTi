import numpy as _np
from .state import State as _State
from .densestate import DenseState as _DenseState
from ...evotypes import Evotype as _Evotype
from ...models import statespace as _statespace
from ...objects.basis import Basis as _Basis


class FullPureState(_DenseState):
    """
    A "fully parameterized" state vector where each element is an independent parameter.

    Parameters
    ----------
    vec : array_like or SPAMVec
        a 1D numpy array representing the SPAM operation.  The
        shape of this array sets the dimension of the SPAM op.

    basis : Basis or {'pp','gm','std'}, optional
        The basis used to construct the Hilbert-Schmidt space representation
        of this state as a super-ket.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    state_space : StateSpace, optional
        The state space for this operation.  If `None` a default state space
        with the appropriate number of qubits is used.
    """

    def __init__(self, purevec, basis="pp", evotype="default", state_space=None):
        purevec = _State._to_vector(purevec)

        state_space = _statespace.default_space_for_udim(purevec.shape[0]) if (state_space is None) \
            else _statespace.StateSpace.cast(state_space)

        evotype = _Evotype.cast(evotype)
        basis = _Basis.cast(basis, state_space.dim)  # basis for Hilbert-Schmidt (superop) space
        rep = evotype.create_pure_state_rep(purevec, basis, state_space)
        _DenseState.__init__(self, rep, evotype, rep.purebase)
        self._paramlbls = _np.array(["VecElement Re(%d)" % i for i in range(self.dim)]
                                    + ["VecElement Im(%d)" % i for i in range(self.dim)], dtype=object)

    def _base_1d_has_changed(self):
        self._rep.purebase_has_changed()

    #Cannot set to arbitrary vector
    #def set_dense(self, vec):
    #    """
    #    Set the dense-vector value of this SPAM vector.
    #
    #    Attempts to modify this SPAM vector's parameters so that the raw
    #    SPAM vector becomes `vec`.  Will raise ValueError if this operation
    #    is not possible.
    #
    #    Parameters
    #    ----------
    #    vec : array_like or SPAMVec
    #        A numpy array representing a SPAM vector, or a SPAMVec object.
    #
    #    Returns
    #    -------
    #    None
    #    """
    #    vec = State._to_vector(vec)
    #    if(vec.size != self.dim):
    #        raise ValueError("Argument must be length %d" % self.dim)
    #    self._base_1d[:] = vec
    #    self.dirty = True

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return 2 * self.size

    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        #TODO: what if _base_1d isn't implemented - use init_from_dense_purevec?
        return _np.concatenate((self._base_1d.real, self._base_1d.imag), axis=0)

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
        self._base_1d[:] = v[0:self.dim] + 1j * v[self.dim:]
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
        derivMx = _np.concatenate((_np.identity(self.dim, complex),
                                   1j * _np.identity(self.dim, complex)), axis=1)
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
