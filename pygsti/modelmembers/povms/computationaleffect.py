"""
The ComputationalBasisPOVMEffect class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


import functools as _functools
import itertools as _itertools

import numpy as _np

from pygsti.modelmembers.povms.effect import POVMEffect as _POVMEffect
from pygsti.modelmembers import term as _term
from pygsti.evotypes import Evotype as _Evotype
from pygsti.baseobjs import statespace as _statespace
from pygsti.baseobjs.basis import Basis as _Basis
from pygsti.baseobjs.polynomial import Polynomial as _Polynomial

try:
    from pygsti.tools import fastcalc as _fastcalc
except ImportError:
    _fastcalc = None


class ComputationalBasisPOVMEffect(_POVMEffect):
    """
    A static POVM effect that is tensor product of 1-qubit Z-eigenstates.

    This is called a "computational basis state" in many contexts.

    Parameters
    ----------
    zvals : iterable
        A list or other iterable of integer 0 or 1 outcomes specifying
        which computational basis element this object represents.  The
        length of `zvals` gives the total number of qubits.

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

    @classmethod
    def from_state_vector(cls, vec, basis='pp', evotype="default", state_space=None):
        """
        Create a new ComputationalBasisPOVMEffect from a dense vector.

        Parameters
        ----------
        vec : numpy.ndarray
            A state vector specifying a computational basis state in the
            standard basis.  This vector has length 4^n for n qubits.

        basis : Basis or {'pp','gm','std'}, optional
            The basis of `vec` as a super-ket.

        evotype : Evotype or str, optional
            The evolution type of the resulting effect vector.  The special
            value `"default"` is equivalent to specifying the value of
            `pygsti.evotypes.Evotype.default_evotype`.

        state_space : StateSpace, optional
            The state space for this operation.  If `None` a default state space
            with the appropriate number of qubits is used.

        Returns
        -------
        ComputationalBasisPOVMEffect
        """
        #if evotype in ('stabilizer', 'statevec'):
        #    nqubits = int(round(_np.log2(len(vec))))
        #    v0 = _np.array((1, 0), complex)  # '0' qubit state as complex state vec
        #    v1 = _np.array((0, 1), complex)  # '1' qubit state as complex state vec
        #else:
        nqubits = int(round(_np.log2(len(vec)) / 2))
        v0 = 1.0 / _np.sqrt(2) * _np.array((1, 0, 0, 1), 'd')  # '0' qubit state as Pauli dmvec
        v1 = 1.0 / _np.sqrt(2) * _np.array((1, 0, 0, -1), 'd')  # '1' qubit state as Pauli dmvec

        v = (v0, v1)
        for zvals in _itertools.product(*([(0, 1)] * nqubits)):
            testvec = _functools.reduce(_np.kron, [v[i] for i in zvals])
            if _np.allclose(testvec, vec.flat):
                return cls(zvals, basis, evotype, state_space)
        raise ValueError(("Given `vec` is not a z-basis product state - "
                          "cannot construct ComputationalBasisPOVMEffect"))

    @classmethod
    def from_pure_vector(cls, purevec, basis='pp', evotype="default", state_space=None):
        """
        TODO: update docstring
        Create a new StabilizerEffectVec from a pure-state vector.

        Currently, purevec must be a single computational basis state (it
        cannot be a superpostion of multiple of them).

        Parameters
        ----------
        purevec : numpy.ndarray
            A complex-valued state vector specifying a pure state in the
            standard computational basis.  This vector has length 2^n for
            n qubits.

        basis : Basis or {'pp','gm','std'}, optional
            The basis of `vec` as a super-ket.

        evotype : Evotype or str, optional
            The evolution type of the resulting effect vector.  The special
            value `"default"` is equivalent to specifying the value of
            `pygsti.evotypes.Evotype.default_evotype`.

        state_space : StateSpace, optional
            The state space for this operation.  If `None` a default state space
            with the appropriate number of qubits is used.

        Returns
        -------
        ComputationalBasisPOVMEffect
        """
        nqubits = int(round(_np.log2(len(purevec))))
        v = (_np.array([1, 0], 'd'), _np.array([0, 1], 'd'))  # (v0,v1)
        for zvals in _itertools.product(*([(0, 1)] * nqubits)):
            testvec = _functools.reduce(_np.kron, [v[i] for i in zvals])
            if _np.allclose(testvec, purevec.flat):
                return cls(zvals, basis, evotype, state_space)
        raise ValueError(("Given `purevec` must be a z-basis product state - "
                          "cannot construct StabilizerEffectVec"))

    def __init__(self, zvals, basis='pp', evotype="default", state_space=None):
        zvals = _np.ascontiguousarray(_np.array(zvals, _np.int64))

        state_space = _statespace.default_space_for_num_qubits(len(zvals)) if (state_space is None) \
            else _statespace.StateSpace.cast(state_space)
        basis = _Basis.cast(basis, state_space.dim)  # basis for Hilbert-Schmidt (superop) space

        evotype = _Evotype.cast(evotype)
        self._evotype = evotype  # set this before call to _State.__init__ so self.to_dense() can work...
        rep = evotype.create_computational_effect_rep(zvals, basis, state_space)
        _POVMEffect.__init__(self, rep, evotype)

    def to_dense(self, on_space='minimal', scratch=None):
        """
        Return this POVM effect vector as a (dense) numpy array.

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
        return self._rep.to_dense(on_space)

    def taylor_order_terms(self, order, max_polynomial_vars=100, return_coeff_polys=False):
        """
        Get the `order`-th order Taylor-expansion terms of this POVM effect vector.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that it is a state
        preparation followed by or POVM effect preceded by actions on a
        density matrix `rho` of the form:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the
        POVMEffect's parameters, where the polynomial's variable indices index the
        *global* parameters of the POVMEffect's parent (usually a :class:`Model`)
        , not the POVMEffect's local parameter array (i.e. that returned from
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
            output of :method:`Polynomial.compact`.
        """
        if order == 0:  # only 0-th order term exists
            coeff = _Polynomial({(): 1.0}, max_polynomial_vars)
            terms = [_term.RankOnePolynomialEffectTerm.create_from(coeff, self, self,
                                                                   self._evotype, self.state_space)]

            if return_coeff_polys:
                coeffs_as_compact_polys = coeff.compact(complex_coeff_tape=True)
                return terms, coeffs_as_compact_polys
            else:
                return terms  # Cache terms in FUTURE?
        else:
            if return_coeff_polys:
                vtape = _np.empty(0, _np.int64)
                ctape = _np.empty(0, complex)
                return [], (vtape, ctape)
            else:
                return []

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this POVM effect vector.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return 0  # no parameters

    def to_vector(self):
        """
        Get the POVM effect vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return _np.array([], 'd')  # no parameters

    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize the POVM effect vector using a 1D array of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of POVM effect vector parameters.  Length
            must == num_params()

        close : bool, optional
            Whether `v` is close to this POVM effect vector's current
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
        assert(len(v) == 0)  # should be no parameters, and nothing to do

    def __str__(self):
        nQubits = len(self._rep.zvals)
        s = "Computational Z-basis POVM effect vec for %d qubits w/z-values: %s" % (nQubits, str(self._rep.zvals))
        return s
