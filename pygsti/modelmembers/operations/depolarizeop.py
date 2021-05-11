class DepolarizeOp(StochasticNoiseOp):
    """
    A depolarizing channel.

    Parameters
    ----------
    dim : int
        The dimension of this operator (4 for a single qubit).

    basis : Basis or {'pp','gm','qt'}, optional
        The basis to use, defining the "principle axes"
        along which there is stochastic noise.  While strictly unnecessary
        since all complete bases yield the same operator, this affects the
        underlying :class:`StochasticNoiseOp` and so is given as an option
        to the user.

    evotype : {"densitymx", "cterm", "svterm"}
        the evolution type being used.

    initial_rate : float, optional
        the initial error rate.

    seed_or_state : float or RandomState, optional
            Random seed for RandomState (or directly provided RandomState)
            for sampling stochastic superoperators with the 'chp' evotype.
    """
    def __init__(self, dim, basis="pp", evotype="densitymx", initial_rate=0, seed_or_state=None):
        """
        Create a new DepolarizeOp, representing a depolarizing channel.

        Parameters
        ----------
        dim : int
            The dimension of this operator (4 for a single qubit).

        basis : Basis or {'pp','gm','qt'}, optional
            The basis to use, defining the "principle axes"
            along which there is stochastic noise.  While strictly unnecessary
            since all complete bases yield the same operator, this affects the
            underlying :class:`StochasticNoiseOp` and so is given as an option
            to the user.

        evotype : {"densitymx", "cterm", "svterm"}
            the evolution type being used.

        initial_rate : float, optional
            the initial error rate.

        seed_or_state : float or RandomState, optional
            Random seed for RandomState (or directly provided RandomState)
            for sampling stochastic superoperators with the 'chp' evotype.
        """

        #TODO - need to fix CHP basis dimension issue (dim ~= statevec but acts as density mx)
        #if evotype == 'chp':
        #    assert (basis == 'pp'), "Only Pauli basis is allowed for 'chp' evotype"
        #    # For chp (and statevec, etc), want full superoperator basis
        #    basis = _Basis.cast(basis, 2**dim, sparse=False)
        #else:
        basis = _Basis.cast(basis, dim, sparse=False)
        
        num_rates = basis.size - 1
        initial_sto_rates = [initial_rate / num_rates] * num_rates
        StochasticNoiseOp.__init__(self, dim, basis, evotype, initial_sto_rates, seed_or_state)

        # For DepolarizeOp, set params to only first element
        self.params = _np.array([self.params[0]])
        self._paramlbls = _np.array(["common stochastic error rate for depolarization"], dtype=object)

    def _rates_to_params(self, rates):
        """Note: requires rates to all be the same"""
        assert(all([rates[0] == r for r in rates[1:]]))
        return _np.array([_np.sqrt(rates[0])], 'd')

    def _params_to_rates(self, params):
        return _np.array([params[0]**2] * (self.basis.size - 1), 'd')

    def _get_rate_poly_dicts(self):
        """ Return a list of dicts, one per rate, expressing the
            rate as a polynomial of the local parameters (tuple
            keys of dicts <=> poly terms, e.g. (1,1) <=> x1^2) """
        return [{(0, 0): 1.0} for i in range(self.basis.size - 1)]  # rates are all just param0 squared

    def copy(self, parent=None, memo=None):
        """
        Copy this object.

        Parameters
        ----------
        parent : Model, optional
            The parent model to set for the copy.

        Returns
        -------
        DepolarizeOp
            A copy of this object.
        """
        if memo is not None and id(self) in memo: return memo[id(self)]
        copyOfMe = DepolarizeOp(self.dim, self.basis, self._evotype, self._params_to_rates(self.to_vector())[0])
        return self._copy_gpindices(copyOfMe, parent, memo)

    def __str__(self):
        s = "Depolarize noise operation map with dim = %d, num params = %d\n" % \
            (self.dim, self.num_params)
        s += 'Strength: %s\n' % (self.params**2 * (self.basis.size - 1))
        return s
