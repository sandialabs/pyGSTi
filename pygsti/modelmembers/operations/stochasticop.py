class StochasticNoiseOp(LinearOperator):
    """
    A stochastic noise operation.

    Implements the stochastic noise map:
    `rho -> (1-sum(p_i))rho + sum_(i>0) p_i * B_i * rho * B_i^dagger`
    where `p_i > 0` and `sum(p_i) < 1`, and `B_i` is basis where `B_0` is the identity.

    In the case of the 'chp' evotype, the `B_i` element is returned with
    probability `p_i`, such that the outcome distribution matches the aforementioned
    stochastic noise map when considered over many samples.

    Parameters
    ----------
    dim : int
        The basis dimension of this operator (4 for a single qubit).

    basis : Basis or {'pp','gm','qt'}, optional
        The basis to use, defining the "principle axes"
        along which there is stochastic noise.  We assume that
        the first element of `basis` is the identity.

    evotype : {"densitymx", "cterm", "svterm"}
        the evolution type being used.

    initial_rates : list or array
        if not None, a list of `basis.size-1` initial error rates along each of
        the directions corresponding to each basis element.  If None,
        then all initial rates are zero.

    seed_or_state : float or RandomState, optional
        Random seed for RandomState (or directly provided RandomState)
        for sampling stochastic superoperators with the 'chp' evotype.
    """
    # Difficult to parameterize and maintain the p_i conditions - Initially just store positive p_i's
    # and don't bother restricting their sum to be < 1?

    def __init__(self, dim, basis="pp", evotype="densitymx", initial_rates=None, seed_or_state=None):
        """
        Create a new StochasticNoiseOp, representing a stochastic noise
        channel with possibly asymmetric noise but only noise that is
        "diagonal" in a particular basis (e.g. Pauli-stochastic noise).

        Parameters
        ----------
        dim : int
            The basis dimension of this operator (4 for a single qubit).

        basis : Basis or {'pp','gm','qt'}, optional
            The basis to use, defining the "principle axes"
            along which there is stochastic noise.  We assume that
            the first element of `basis` is the identity.
            This must be 'pp' for the 'chp' evotype.

        evotype : {"densitymx", "cterm", "svterm", "chp"}
            the evolution type being used.

        initial_rates : list or array
            if not None, a list of `dim-1` initial error rates along each of
            the directions corresponding to each basis element.  If None,
            then all initial rates are zero.

        seed_or_state : float or RandomState, optional
            Random seed for RandomState (or directly provided RandomState)
            for sampling stochastic superoperators with the 'chp' evotype.
        """
        self.basis = _Basis.cast(basis, dim, sparse=False)
        assert(dim == self.basis.dim), "Dimension of `basis` must match the dimension (`dim`) of this op."

        evotype = _Evotype.cast(evotype)

        #Setup initial parameters
        self.params = _np.zeros(self.basis.size - 1, 'd')  # note that basis.dim can be < self.dim (OK)
        if initial_rates is not None:
            assert(len(initial_rates) == self.basis.size - 1), \
                "Expected %d initial rates but got %d!" % (self.basis.size - 1, len(initial_rates))
            self.params[:] = self._rates_to_params(initial_rates)
            rates = _np.array(initial_rates)
        else:
            rates = _np.zeros(len(initial_rates), 'd')

        rep = evotype.create_stochastic_rep(basis, self._get_rate_poly_dicts(), rates, seed_or_state)
        LinearOperator.__init__(self, rep, evotype)
        self._update_rep()  # initialize self._rep
        self._paramlbls = _np.array(['sqrt(%s error rate)' % bl for bl in self.basis.labels[1:]], dtype=object)

    def _update_rep(self):
        # Create dense error superoperator from paramvec
        self._rep.update_rates(self._params_to_rates(self.params))

    def _rates_to_params(self, rates):
        return _np.sqrt(_np.array(rates))

    def _params_to_rates(self, params):
        return params**2

    def _get_rate_poly_dicts(self):
        """ Return a list of dicts, one per rate, expressing the
            rate as a polynomial of the local parameters (tuple
            keys of dicts <=> poly terms, e.g. (1,1) <=> x1^2) """
        return [{(i, i): 1.0} for i in range(self.basis.size - 1)]  # rates are just parameters squared

    def copy(self, parent=None, memo=None):
        """
        Copy this object.

        Parameters
        ----------
        parent : Model, optional
            The parent model to set for the copy.

        Returns
        -------
        StochasticNoiseOp
            A copy of this object.
        """
        if memo is not None and id(self) in memo: return memo[id(self)]
        copyOfMe = StochasticNoiseOp(self.dim, self.basis, self._evotype, self._params_to_rates(self.to_vector()))
        return self._copy_gpindices(copyOfMe, parent, memo)

    #to_dense / to_sparse?
    def to_dense(self):
        """
        Return this operation as a dense matrix.

        Returns
        -------
        numpy.ndarray
        """
        return self._rep.to_dense()

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this operation.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return len(self.to_vector())

    def to_vector(self):
        """
        Extract a vector of the underlying operation parameters from this operation.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self.params

    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize the operation using a vector of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of operation parameters.  Length
            must == num_params()

        close : bool, optional
            Whether `v` is close to this operation's current
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
        self.params[:] = v
        self._update_rep()
        self.dirty = dirty_value

    #Transform functions? (for gauge opt)

    def __str__(self):
        s = "Stochastic noise operation map with dim = %d, num params = %d\n" % \
            (self.dim, self.num_params)
        s += 'Rates: %s\n' % self._params_to_rates(self.to_vector())
        return s
