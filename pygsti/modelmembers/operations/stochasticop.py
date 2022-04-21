"""
The StochasticNoiseOp class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np

from pygsti.modelmembers.operations.linearop import LinearOperator as _LinearOperator
from pygsti.modelmembers.operations.krausop import KrausOperatorInterface as _KrausOperatorInterface
from pygsti.modelmembers import modelmember as _modelmember, term as _term
from pygsti.evotypes import Evotype as _Evotype
from pygsti.baseobjs import statespace as _statespace
from pygsti.baseobjs.basis import Basis as _Basis
from pygsti.baseobjs.polynomial import Polynomial as _Polynomial


class StochasticNoiseOp(_LinearOperator, _KrausOperatorInterface):
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
    state_space : StateSpace, optional
        The state space for this operation.

    basis : Basis or {'pp','gm','qt'}, optional
        The basis to use, defining the "principle axes"
        along which there is stochastic noise.  We assume that
        the first element of `basis` is the identity.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

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

    def __init__(self, state_space, stochastic_basis='PP', basis="pp", evotype="default",
                 initial_rates=None, seed_or_state=None):
        state_space = _statespace.StateSpace.cast(state_space)
        self.basis = _Basis.cast(basis, state_space.dim, sparse=False)
        assert(state_space.dim == self.basis.dim), "Dimension of `basis` must match the dimension (`dim`) of this op."

        evotype = _Evotype.cast(evotype)

        #Setup initial parameters
        self.params = _np.zeros(self.basis.size - 1, 'd')  # note that basis.dim can be < self.dim (OK)
        if initial_rates is not None:
            assert(len(initial_rates) == self.basis.size - 1), \
                "Expected %d initial rates but got %d!" % (self.basis.size - 1, len(initial_rates))
            self.params[:] = self._rates_to_params(initial_rates)
            rates = _np.array(initial_rates)
        else:
            rates = _np.zeros(len(self.params), 'd')

        #self._get_rate_poly_dicts()
        self.stochastic_basis = _Basis.cast(stochastic_basis, state_space.dim, sparse=False)
        rep = evotype.create_stochastic_rep(self.stochastic_basis, self.basis, rates, seed_or_state, state_space)
        _LinearOperator.__init__(self, rep, evotype)
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

    def to_dense(self, on_space='minimal'):
        """
        Return this operation as a dense matrix.

        Parameters
        ----------
        on_space : {'minimal', 'Hilbert', 'HilbertSchmidt'}
            The space that the returned dense operation acts upon.  For unitary matrices and bra/ket vectors,
            use `'Hilbert'`.  For superoperator matrices and super-bra/super-ket vectors use `'HilbertSchmidt'`.
            `'minimal'` means that `'Hilbert'` is used if possible given this operator's evolution type, and
            otherwise `'HilbertSchmidt'` is used.

        Returns
        -------
        numpy.ndarray
        """
        return self._rep.to_dense(on_space)

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

    def taylor_order_terms(self, order, max_polynomial_vars=100, return_coeff_polys=False):
        """
        Get the `order`-th order Taylor-expansion terms of this operation.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that its action on a
        density matrix `rho` can be written:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the operation's
        parameters, where the polynomial's variable indices index the *global*
        parameters of the operation's parent (usually a :class:`Model`), not the
        operation's local parameter array (i.e. that returned from `to_vector`).

        Parameters
        ----------
        order : int
            Which order terms (in a Taylor expansion of this :class:`LindbladOp`)
            to retrieve.

        max_polynomial_vars : int, optional
            maximum number of variables the created polynomials can have.

        return_coeff_polys : bool
            Whether a parallel list of locally-indexed (using variable indices
            corresponding to *this* object's parameters rather than its parent's)
            polynomial coefficients should be returned as well.

        Returns
        -------
        terms : list
            A list of :class:`RankOneTerm` objects.
        coefficients : list
            Only present when `return_coeff_polys == True`.
            A list of *compact* polynomial objects, meaning that each element
            is a `(vtape,ctape)` 2-tuple formed by concatenating together the
            output of :method:`Polynomial.compact`.
        """

        def _compose_poly_indices(terms):
            for term in terms:
                term.map_indices_inplace(lambda x: tuple(_modelmember._compose_gpindices(
                    self.gpindices, _np.array(x, _np.int64))))
            return terms

        IDENT = None  # sentinel for the do-nothing identity op
        mpv = max_polynomial_vars
        if order == 0:
            polydict = {(): 1.0}
            for pd in self._get_rate_poly_dicts():
                polydict.update({k: -v for k, v in pd.items()})  # subtracts the "rate" `pd` from `polydict`
            loc_terms = [_term.RankOnePolynomialOpTerm.create_from(_Polynomial(polydict, mpv),
                                                                   IDENT, IDENT, self._evotype, self.state_space)]

        elif order == 1:
            loc_terms = [_term.RankOnePolynomialOpTerm.create_from(_Polynomial(pd, mpv), bel, bel,
                                                                   self._evotype, self.state_space)
                         for i, (pd, bel) in enumerate(zip(self.rate_poly_dicts, self.basis.elements[1:]))]
        else:
            loc_terms = []  # only first order "taylor terms"

        poly_coeffs = [t.coeff for t in loc_terms]
        tapes = [poly.compact(complex_coeff_tape=True) for poly in poly_coeffs]
        if len(tapes) > 0:
            vtape = _np.concatenate([t[0] for t in tapes])
            ctape = _np.concatenate([t[1] for t in tapes])
        else:
            vtape = _np.empty(0, _np.int64)
            ctape = _np.empty(0, complex)
        coeffs_as_compact_polys = (vtape, ctape)

        local_term_poly_coeffs = coeffs_as_compact_polys
        global_param_terms = _compose_poly_indices(loc_terms)

        if return_coeff_polys:
            return global_param_terms, local_term_poly_coeffs
        else:
            return global_param_terms

    @property
    def total_term_magnitude(self):
        """
        Get the total (sum) of the magnitudes of all this operator's terms.

        The magnitude of a term is the absolute value of its coefficient, so
        this function returns the number you'd get from summing up the
        absolute-coefficients of all the Taylor terms (at all orders!) you
        get from expanding this operator in a Taylor series.

        Returns
        -------
        float
        """
        # return exp( mag of errorgen ) = exp( sum of absvals of errgen term coeffs )
        # (unitary postfactor has weight == 1.0 so doesn't enter)
        rates = self._params_to_rates(self.to_vector())
        return _np.sum(_np.abs(rates))

    @property
    def total_term_magnitude_deriv(self):
        """
        The derivative of the sum of *all* this operator's terms.

        Computes the derivative of the total (sum) of the magnitudes of all this
        operator's terms with respect to the operators (local) parameters.

        Returns
        -------
        numpy array
            An array of length self.num_params
        """
        # abs(rates) = rates = params**2
        # so d( sum(abs(rates)) )/dparam_i = 2*param_i
        return 2 * self.to_vector()

    #Transform functions? (for gauge opt)

    def to_memoized_dict(self, mmg_memo):
        """Create a serializable dict with references to other objects in the memo.

        Parameters
        ----------
        mmg_memo: dict
            Memo dict from a ModelMemberGraph, i.e. keys are object ids and values
            are ModelMemberGraphNodes (which contain the serialize_id). This is NOT
            the same as other memos in ModelMember (e.g. copy, allocate_gpindices, etc.).

        Returns
        -------
        mm_dict: dict
            A dict representation of this ModelMember ready for serialization
            This must have at least the following fields:
                module, class, submembers, params, state_space, evotype
            Additional fields may be added by derived classes.
        """
        mm_dict = super().to_memoized_dict(mmg_memo)

        mm_dict['basis'] = self.basis.to_nice_serialization()
        mm_dict['stochastic_basis'] = self.stochastic_basis.to_nice_serialization()
        mm_dict['rates'] = self._params_to_rates(self.to_vector()).tolist()

        return mm_dict

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        state_space = _statespace.StateSpace.from_nice_serialization(mm_dict['state_space'])
        basis = _Basis.from_nice_serialization(mm_dict['basis'])
        stochastic_basis = _Basis.from_nice_serialization(mm_dict['stochastic_basis'])
        return cls(state_space, stochastic_basis, basis, mm_dict['evotype'], mm_dict['rates'], seed_or_state=None)
        # Note: we currently don't serialize random seed/state - that gets reset w/serialization

    def _is_similar(self, other, rtol, atol):
        """ Returns True if `other` model member (which it guaranteed to be the same type as self) has
            the same local structure, i.e., not considering parameter values or submembers """
        return (self.basis == other.basis and self.state_space == other.state_space)

    def __str__(self):
        s = "Stochastic noise operation map with state space = %s, num params = %d\n" % \
            (self.state_space, self.num_params)
        s += 'Rates: %s\n' % self._params_to_rates(self.to_vector())
        return s

    def _oneline_contents(self):
        """ Summarizes the contents of this object in a single line.  Does not summarize submembers. """
        return 'rates: %s' % self._params_to_rates(self.to_vector())
