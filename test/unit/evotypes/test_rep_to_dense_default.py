"""Regression tests for issue #806: rep-level to_dense() default on_space.

Several evotype representation classes defined ``to_dense(self, on_space)``
with no default for ``on_space``.  Callers occasionally invoke ``to_dense()``
with no argument, which raised ``TypeError`` for those reps.  These tests check
that the reps now accept a no-argument call and that it agrees with the
explicit ``'minimal'`` call (the default used throughout the modelmember API).
"""
import numpy as np

from pygsti.modelmembers.states import FullState
from pygsti.modelmembers.operations import FullArbitraryOp
from pygsti.modelmembers.povms import ComputationalBasisPOVM
from pygsti.baseobjs.statespace import QubitSpace
from pygsti.baseobjs.basis import Basis
from ..util import BaseCase


class RepToDenseDefaultTester(BaseCase):
    """to_dense() (no argument) should match to_dense('minimal')."""

    def _check(self, rep):
        no_arg = rep.to_dense()
        explicit = rep.to_dense('minimal')
        self.assertArraysAlmostEqual(np.asarray(no_arg), np.asarray(explicit))

    def test_densitymx_state(self):
        for evo in ('densitymx', 'densitymx_slow'):
            with self.subTest(evotype=evo):
                st = FullState(np.array([1, 0, 0, 0], 'd'), evotype=evo)
                self._check(st._rep)

    def test_densitymx_op(self):
        for evo in ('densitymx', 'densitymx_slow'):
            with self.subTest(evotype=evo):
                op = FullArbitraryOp(np.eye(4, dtype='d'), evotype=evo)
                self._check(op._rep)

    def test_densitymx_effect(self):
        for evo in ('densitymx', 'densitymx_slow'):
            with self.subTest(evotype=evo):
                povm = ComputationalBasisPOVM(1, evotype=evo)
                self._check(povm['0']._rep)

    def test_statevec_effect(self):
        povm = ComputationalBasisPOVM(1, evotype='statevec')
        self._check(povm['0']._rep)

    def test_statevec_op(self):
        # OpRepDenseUnitary is one of the reps called out in issue #806
        # (pygsti/evotypes/statevec/opreps.pyx).
        from pygsti.evotypes.statevec import opreps as _svop
        op = _svop.OpRepDenseUnitary(np.eye(2, dtype=complex),
                                     Basis.cast('pp', 4), QubitSpace(1))
        self._check(op)


class ComputationalEffectProbabilityTester(BaseCase):
    """
    Cover ``EffectRepComputational.probability`` for the pure-Python
    ``densitymx_slow`` evotype.

    Ensure that the O(2**nfactors) sparse algorithm computes the same results
    as the O(4**nfactors) dense algorithm.
    """

    ZVAL_STRINGS = ['0', '1', '00', '01', '10', '11', '011', '101', '111']

    @staticmethod
    def _effect_and_state(evotype, zvals_string, seed):
        n_qubits = len(zvals_string)
        povm = ComputationalBasisPOVM(n_qubits, evotype=evotype)
        state = FullState(
            np.random.default_rng(seed).standard_normal(4 ** n_qubits), evotype=evotype)
        return povm[zvals_string]._rep, state._rep

    def test_probability_matches_dense_dot_product(self):
        for zvals_string in self.ZVAL_STRINGS:
            with self.subTest(zvals=zvals_string):
                effect, state = self._effect_and_state('densitymx_slow', zvals_string, 0)
                expected = np.dot(np.asarray(effect.to_dense()), state.data)
                self.assertAlmostEqual(effect.probability(state), expected)

    def test_probability_matches_compiled_evotype(self):
        for zvals_string in self.ZVAL_STRINGS:
            with self.subTest(zvals=zvals_string):
                slow_effect, slow_state = self._effect_and_state(
                    'densitymx_slow', zvals_string, 1)
                fast_effect, fast_state = self._effect_and_state(
                    'densitymx', zvals_string, 1)
                self.assertAlmostEqual(slow_effect.probability(slow_state),
                                       fast_effect.probability(fast_state))

    def test_probabilities_over_a_computational_povm_sum_to_the_state_trace(self):
        # Physical sanity check that does not go through to_dense at all: the
        # outcome probabilities of a complete computational-basis POVM must sum
        # to the trace of the state, which in the PP super-ket convention is
        # sqrt(2)**n times the first element.
        n_qubits = 2
        povm = ComputationalBasisPOVM(n_qubits, evotype='densitymx_slow')
        state = FullState(np.random.default_rng(3).standard_normal(4 ** n_qubits),
                          evotype='densitymx_slow')
        total = sum(povm[lbl]._rep.probability(state._rep) for lbl in povm.keys())
        self.assertAlmostEqual(total, (np.sqrt(2) ** n_qubits) * state._rep.data[0])
