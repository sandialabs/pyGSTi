import functools as _functools
import itertools as _itertools

import numpy as _np

from .state import State as _State
from .. import term as _term
from ...evotypes import Evotype as _Evotype
from ...baseobjs import statespace as _statespace
from ...baseobjs.basis import Basis as _Basis
from ...baseobjs.polynomial import Polynomial as _Polynomial

try:
    from ...tools import fastcalc as _fastcalc
except ImportError:
    _fastcalc = None


class ComputationalBasisState(_State):
    """
    A static state vector that is tensor product of 1-qubit Z-eigenstates.

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
    def from_dense_vec(cls, vec, basis='pp', evotype='default', state_space=None):
        """
        Create a new ComputationalBasisState from a dense vector.

        Parameters
        ----------
        vec : numpy.ndarray
            A state vector specifying a computational basis state in the
            standard basis.  This vector has length 4^n for n qubits.

        basis : Basis or {'pp','gm','std'}, optional
            The basis of `vec` as a super-ket.

        evotype : Evotype or str
            The evolution type.  The special value `"default"` is equivalent
            to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

        state_space : StateSpace, optional
            The state space for this operation.  If `None` a default state space
            with the appropriate number of qubits is used.

        Returns
        -------
        ComputationalBasisState
        """
        if evotype in ('stabilizer', 'statevec'):
            nqubits = int(round(_np.log2(len(vec))))
            v0 = _np.array((1, 0), complex)  # '0' qubit state as complex state vec
            v1 = _np.array((0, 1), complex)  # '1' qubit state as complex state vec
        else:
            nqubits = int(round(_np.log2(len(vec)) / 2))
            v0 = 1.0 / _np.sqrt(2) * _np.array((1, 0, 0, 1), 'd')  # '0' qubit state as Pauli dmvec
            v1 = 1.0 / _np.sqrt(2) * _np.array((1, 0, 0, -1), 'd')  # '1' qubit state as Pauli dmvec

        v = (v0, v1)
        for zvals in _itertools.product(*([(0, 1)] * nqubits)):
            testvec = _functools.reduce(_np.kron, [v[i] for i in zvals])
            if _np.allclose(testvec, vec.flat):
                return cls(zvals, basis, evotype, state_space)
        raise ValueError(("Given `vec` is not a z-basis product state - "
                          "cannot construct ComputationalBasisState"))

    @classmethod
    def from_dense_purevec(cls, purevec, basis='pp', evotype="default", state_space=None):
        """
        Create a new ComputationalBasisState from a pure-state vector.

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
        ComputationalBasisState
        """
        nqubits = int(round(_np.log2(len(purevec))))
        v = (_np.array([1, 0], 'd'), _np.array([0, 1], 'd'))  # (v0,v1)
        for zvals in _itertools.product(*([(0, 1)] * nqubits)):
            testvec = _functools.reduce(_np.kron, [v[i] for i in zvals])
            if _np.allclose(testvec, purevec.flat):
                return cls(zvals, basis, evotype, state_space)
        raise ValueError(("Given `purevec` must be a z-basis product state - "
                          "cannot construct ComputationalBasisState"))

    def __init__(self, zvals, basis='pp', evotype="default", state_space=None):
        self._zvals = _np.ascontiguousarray(_np.array(zvals, _np.int64))

        state_space = _statespace.default_space_for_num_qubits(len(self._zvals)) if (state_space is None) \
            else _statespace.StateSpace.cast(state_space)
        basis = _Basis.cast(basis, state_space.dim)  # basis for Hilbert-Schmidt (superop) space

        evotype = _Evotype.cast(evotype)
        self._evotype = evotype  # set this before call to _State.__init__ so self.to_dense() can work...
        rep = evotype.create_computational_state_rep(zvals, basis, state_space)
        _State.__init__(self, rep, evotype)

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
        from .staticpurestate import StaticPureState as _StaticPureState
        v0 = _StaticPureState(_np.array((1, 0), complex), basis='pp', evotype=self._evotype).to_dense('minimal')
        v1 = _StaticPureState(_np.array((0, 1), complex), basis='pp', evotype=self._evotype).to_dense('minimal')
        factor_dim = len(v0)

        #OLD REMOVE
        #if self._evotype == "densitymx":
        #    factor_dim = 4
        #    v0 = 1.0 / _np.sqrt(2) * _np.array((1, 0, 0, 1), 'd')  # '0' qubit state as Pauli dmvec
        #    v1 = 1.0 / _np.sqrt(2) * _np.array((1, 0, 0, -1), 'd')  # '1' qubit state as Pauli dmvec
        #elif self._evotype in ("statevec", "stabilizer", "chp"):
        #    factor_dim = 2
        #    v0 = _np.array((1, 0), complex)  # '0' qubit state as complex state vec
        #    v1 = _np.array((0, 1), complex)  # '1' qubit state as complex state vec
        #elif self._evotype in ("svterm", "cterm"):
        #    raise NotImplementedError("to_dense() is not implemented for evotype %s!" %
        #                              self._evotype)
        #else: raise ValueError("Invalid `evotype`: %s" % self._evotype)

        v = (v0, v1)

        if _fastcalc is None:  # do it the slow way using numpy
            return _functools.reduce(_np.kron, [v[i] for i in self._zvals])
        else:
            typ = 'd' if self._evotype == "densitymx" else complex
            fast_kron_array = _np.ascontiguousarray(
                _np.empty((len(self._zvals), factor_dim), typ))
            fast_kron_factordims = _np.ascontiguousarray(_np.array([factor_dim] * len(self._zvals), _np.int64))
            for i, zi in enumerate(self._zvals):
                fast_kron_array[i, :] = v[zi]
            ret = _np.ascontiguousarray(_np.empty(factor_dim**len(self._zvals), typ))
            if self._evotype == "densitymx":
                _fastcalc.fast_kron(ret, fast_kron_array, fast_kron_factordims)
            else:
                _fastcalc.fast_kron_complex(ret, fast_kron_array, fast_kron_factordims)
            return ret

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
            output of :method:`Polynomial.compact`.
        """
        if order == 0:  # only 0-th order term exists

            #TODO - there should be a "term" evotype containing a sub-evotype used to create the state here
            #REMOVE term_evotype = self._evotype.term_evotype
            #REMOVE purevec = ComputationalBasisState(self._zvals, term_evotype)
            coeff = _Polynomial({(): 1.0}, max_polynomial_vars)
            terms = [_term.RankOnePolynomialPrepTerm.create_from(coeff, self, self,
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
        Get the number of independent parameters which specify this state vector.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return 0  # no parameters

    def to_vector(self):
        """
        Get the state vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return _np.array([], 'd')  # no parameters

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
        assert(len(v) == 0)  # should be no parameters, and nothing to do

    def __str__(self):
        nQubits = len(self._zvals)
        s = "Computational Z-basis state vec for %d qubits w/z-values: %s" % (nQubits, str(self._zvals))
        return s


#REMOVE
#class StabilizerState(_State):
#    """
#    A stabilizer state preparation.
#
#    This is represented internally using a compact representation of its stabilizer group.
#
#    Parameters
#    ----------
#    nqubits : int
#        Number of qubits
#
#    zvals : iterable, optional
#        An iterable over anything that can be cast as True/False
#        to indicate the 0/1 value of each qubit in the Z basis.
#        If None, the all-zeros state is created.
#
#    sframe : StabilizerFrame, optional
#        A complete stabilizer frame to initialize this state from.
#        If this is not None, then `nqubits` and `zvals` must be None.
#    """
#
#    @classmethod
#    def from_dense_purevec(cls, purevec):
#        """
#        Create a new ComputationalBasisState from a pure-state vector.
#
#        Currently, purevec must be a single computational basis state (it
#        cannot be a superpostion of multiple of them).
#
#        Parameters
#        ----------
#        purevec : numpy.ndarray
#            A complex-valued state vector specifying a pure state in the
#            standard computational basis.  This vector has length 2^n for
#            n qubits.
#
#        Returns
#        -------
#        ComputationalBasisState
#        """
#        nqubits = int(round(_np.log2(len(purevec))))
#        v = (_np.array([1, 0], 'd'), _np.array([0, 1], 'd'))  # (v0,v1)
#        for zvals in _itertools.product(*([(0, 1)] * nqubits)):
#            testvec = _functools.reduce(_np.kron, [v[i] for i in zvals])
#            if _np.allclose(testvec, purevec.flat):
#                return cls(nqubits, zvals)
#        raise ValueError(("Given `purevec` must be a z-basis product state - "
#                          "cannot construct ComputationalBasisState"))
#
#    def __init__(self, nqubits, zvals=None, sframe=None):
#        """
#        Initialize a ComputationalBasisState object.
#
#        Parameters
#        ----------
#        nqubits : int
#            Number of qubits
#
#        zvals : iterable, optional
#            An iterable over anything that can be cast as True/False
#            to indicate the 0/1 value of each qubit in the Z basis.
#            If None, the all-zeros state is created.
#
#        sframe : StabilizerFrame, optional
#            A complete stabilizer frame to initialize this state from.
#            If this is not None, then `nqubits` and `zvals` must be None.
#        """
#        if sframe is not None:
#            assert(nqubits is None and zvals is None), "`nqubits` and `zvals` must be None when `sframe` isn't!"
#            self.sframe = sframe
#        else:
#            self.sframe = _stabilizer.StabilizerFrame.from_zvals(nqubits, zvals)
#        rep = self.sframe.to_rep()  # dim == 2**nqubits
#        _State.__init__(self, rep, "stabilizer")
#
#    def to_dense(self, on_space='minimal', scratch=None):
#        """
#        Return this state vector as a (dense) numpy array.
#
#        The memory in `scratch` maybe used when it is not-None.
#
#        Parameters
#        ----------
#        scratch : numpy.ndarray, optional
#            scratch space available for use.
#
#        Returns
#        -------
#        numpy.ndarray
#        """
#        assert(on_space in ('minimal', 'Hilbert'))
#        statevec = self.sframe.to_statevec()
#        statevec.shape = (statevec.size, 1)
#        return statevec
#
#    def __str__(self):
#        s = "Stabilizer spam vector for %d qubits with rep:\n" % (self.sframe.nqubits)
#        s += str(self.sframe)
#        return s
