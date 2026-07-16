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
