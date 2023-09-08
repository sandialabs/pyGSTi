"""
The StaticPureState class and supporting functionality.
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
from pygsti.modelmembers import term as _term
from pygsti.modelmembers.errorgencontainer import NoErrorGeneratorInterface as _NoErrorGeneratorInterface
from pygsti.modelmembers.states.densestate import DensePureState as _DensePureState
from pygsti.baseobjs.polynomial import Polynomial as _Polynomial


class StaticPureState(_DensePureState, _NoErrorGeneratorInterface):
    """
    A pure state vector that is completely fixed, or "static" (i.e. that posesses no parameters).

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

    def __init__(self, purevec, basis='pp', evotype="default", state_space=None):
        _DensePureState.__init__(self, purevec, basis, evotype, state_space)

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
        if order == 0:  # only 0-th order term exists (assumes static pure_state_vec)
            coeff = _Polynomial({(): 1.0}, max_polynomial_vars)
            #if self._prep_or_effect == "prep":
            terms = [_term.RankOnePolynomialPrepTerm.create_from(coeff, self, self,
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

    def _is_similar(self, other, rtol, atol):
        """ Returns True if `other` model member (which it guaranteed to be the same type as self) has
            the same local structure, i.e., not considering parameter values or submembers """
        # static objects must also test their values in is_similar, since these aren't parameters.
        return (super()._is_similar(other, rtol, atol)
                and _np.allclose(self.to_dense(), other.to_dense(), rtol=rtol, atol=atol))
