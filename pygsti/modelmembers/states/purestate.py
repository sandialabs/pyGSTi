"""
The EmbeddedPureState class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


import numpy as _np

from pygsti.modelmembers.states.state import State as _State
from pygsti.modelmembers.states.staticstate import StaticState as _StaticState
from pygsti.modelmembers import term as _term
from pygsti.evotypes import Evotype as _Evotype
from pygsti.baseobjs.polynomial import Polynomial as _Polynomial
from pygsti.tools import basistools as _bt
from pygsti.tools import optools as _ot


#TODO: figure out what to do with this class when we wire up term calcs??
# may need this to be an effect class too?
class EmbeddedPureState(_State):
    """
    TODO: update docstring
    A state vector that is a rank-1 density matrix.

    This is essentially a pure state that evolves according to one of the
    density matrix evolution types ("denstiymx", "svterm", and "cterm").  It is
    parameterized by a contained pure-state State which evolves according to a
    state vector evolution type ("statevec" or "stabilizer").

    Parameters
    ----------
    pure_state_vec : array_like or State
        a 1D numpy array or object representing the pure state.  This object
        sets the parameterization and dimension of this state vector (if
        `pure_state_vec`'s dimension is `d`, then this state vector's
        dimension is `d^2`).  Assumed to be a complex vector in the
        standard computational basis.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.
        Note that the evotype of `pure_state_vec` must be compatible with this value.
        For example, if `pure_state_vec` has an evotype of `"statevec"` then allowed
        values are `"densitymx"` and `"svterm"`,  or  if `"stabilizer"` then the only
        allowed value is `"cterm"`.

    dm_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for this state vector - that is, for the *density matrix*
        corresponding to `pure_state_vec`.  Allowed values are Matrix-unit
        (std),  Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
        (or a custom basis object).
    """

    def __init__(self, pure_state, evotype='default', dm_basis='pp'):
        if not isinstance(pure_state, _State):
            pure_state = _StaticState(_State._to_vector(pure_state), dm_basis, 'statevec')
        self.pure_state = pure_state
        self.basis = dm_basis  # only used for dense conversion

        evotype = _Evotype.cast(evotype, state_space=self.pure_state.state_space)
        #rep = evotype.create_state_rep()
        #rep.init_from_dense_purevec(pure_state)
        raise NotImplementedError("Maybe this class isn't even needed, or need to create a static pure state class?")

        #TODO: remove
        #pure_evo = pure_state._evotype
        #if pure_evo == "statevec":
        #    if evotype not in ("densitymx", "svterm"):
        #        raise ValueError(("`evotype` arg must be 'densitymx' or 'svterm'"
        #                          " when `pure_state_vec` evotype is 'statevec'"))
        #elif pure_evo == "stabilizer":
        #    if evotype not in ("cterm",):
        #        raise ValueError(("`evotype` arg must be 'densitymx' or 'svterm'"
        #                          " when `pure_state_vec` evotype is 'statevec'"))
        #else:
        #    raise ValueError("`pure_state_vec` evotype must be 'statevec' or 'stabilizer' (not '%s')" % pure_evo)

        #Create representation
        #_State.__init__(self, rep, evotype)
        #self.init_gpindices()  # initialize our gpindices based on sub-members

    def to_dense(self, on_space='minimal', scratch=None):
        """
        Return this state vector as a (dense) numpy array.

        The memory in `scratch` maybe used when it is not-None.

        Parameters
        ----------
        on_space : {'minimal', 'Hilbert', 'HilbertSchmidt'}
            The space that the returned dense operation acts upon.  For unitary matrices and bra/ket vectors,
            use `'Hilbert'`.  For superoperator matrices and super-bra/super-ket vectors use `'HilbertSchmidt'`.
            `'minimal'` means that `'Hilbert'` is used if possible given this operator's evolution type, and
            otherwise `'HilbertSchmidt'` is used.

        scratch : numpy.ndarray, optional
            scratch space available for use.

        Returns
        -------
        numpy.ndarray
        """
        assert(on_space in ('minimal', 'HilbertSchmidt'))
        dmVec_std = _ot.state_to_dmvec(self.pure_state.to_dense(on_space='Hilbert'))
        return _bt.change_basis(dmVec_std, 'std', self.basis)

    def taylor_order_terms(self, order, max_polynomial_vars=100, return_coeff_polys=False):
        """
        Get the `order`-th order Taylor-expansion terms of this state vector.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that it is a state
        preparation followed by or POVM effect preceded by actions on a
        density matrix `rho` of the form:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the
        State's parameters, where the polynomial's variable indices index the
        *global* parameters of the State's parent (usually a :class:`Model`)
        , not the State's local parameter array (i.e. that returned from
        `to_vector`).

        Parameters
        ----------
        order : int
            The order of terms to get.

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
            output of :meth:`Polynomial.compact`.
        """
        if self.num_params > 0:
            raise ValueError(("EmbeddedPureState.taylor_order_terms(...) is only "
                              "implemented for the case when its underlying "
                              "pure state vector has 0 parameters (is static)"))

        if order == 0:  # only 0-th order term exists (assumes static pure_state_vec)
            purevec = self.pure_state
            coeff = _Polynomial({(): 1.0}, max_polynomial_vars)
            #if self._prep_or_effect == "prep":
            terms = [_term.RankOnePolynomialPrepTerm.create_from(coeff, purevec, purevec,
                                                                 self._evotype, self.state_space)]
            #else:
            #    terms = [_term.RankOnePolynomialEffectTerm.create_from(coeff, purevec, purevec,
            #                                                           self._evotype, self.state_space)]

            if return_coeff_polys:
                coeffs_as_compact_polys = coeff.compact(complex_coeff_tape=True)
                return terms, coeffs_as_compact_polys
            else:
                return terms
        else:
            if return_coeff_polys:
                vtape = _np.empty(0, _np.int64)
                ctape = _np.empty(0, complex)
                return [], (vtape, ctape)
            else:
                return []

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        return self.pure_state.parameter_labels

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this state vector.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return self.pure_state.num_params

    def to_vector(self):
        """
        Get the state vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return self.pure_state.to_vector()

    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize the state vector using a 1D array of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of state vector parameters.  Length
            must == num_params()

        close : bool, optional
            Whether `v` is close to this state vector's current
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
        self.pure_state.from_vector(v, close, dirty_value)
        #Update dense rep if one is created (TODO)

    def deriv_wrt_params(self, wrt_filter=None):
        """
        The element-wise derivative this state vector.

        Construct a matrix whose columns are the derivatives of the state vector
        with respect to a single param.  Thus, each column is of length
        dimension and there is one column per state vector parameter.

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
        raise NotImplementedError("Still need to work out derivative calculation of EmbeddedPureState")

    def has_nonzero_hessian(self):
        """
        Whether this state vector has a non-zero Hessian with respect to its parameters.

        Returns
        -------
        bool
        """
        return self.pure_state.has_nonzero_hessian()

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.pure_state]

    def __str__(self):
        s = "Pure-state spam vector with length %d holding:\n" % self.dim
        s += "  " + str(self.pure_state)
        return s
