import numpy as np
import pytest

from pygsti.baseobjs import Basis
from pygsti.baseobjs.statespace import ExplicitStateSpace, QubitSpace
from pygsti.evotypes.statevec_slow.opreps import (
    OpRepComposed, OpRepDenseUnitary, OpRepEmbedded, OpRepRandomUnitary)
from pygsti.evotypes.statevec_slow.statereps import StateRepDensePure
from pygsti.modelmembers.operations import EmbeddedOp, StaticUnitaryOp


@pytest.mark.parametrize('ss, targets', [
    (QubitSpace(4), [3, 1]),
    (ExplicitStateSpace([('Q0', 'T1', 'Q2')], [(2, 3, 2)]), ['Q2', 'T1']),
    (ExplicitStateSpace([('Qleft',), ('Q0', 'T1', 'Q2'), ('Qright',)],
                        [(2,), (2, 3, 2), (2,)]), ['T1', 'Q2']),
])
@pytest.mark.parametrize('composed', [False, True])
def test_statevec_embedded_actions(ss, targets, composed):
    sub = ss.create_subspace(targets)
    rng = np.random.default_rng(90)
    mx = np.linalg.qr(rng.normal(size=(sub.udim, sub.udim))
                      + 1j * rng.normal(size=(sub.udim, sub.udim)))[0]
    op = StaticUnitaryOp(mx, evotype='statevec_slow', state_space=sub)
    inner = OpRepComposed([op._rep, op._rep], sub) if composed else op._rep
    rep = OpRepEmbedded(ss, targets, inner)
    vector = rng.normal(size=ss.udim) + 1j * rng.normal(size=ss.udim)
    state = StateRepDensePure(vector.copy(), ss, None)

    # Repeat after changing the contained matrix, so cached indices cannot conceal a stale action.
    for matrix in (mx, mx.conj()):
        op._rep.base[:] = matrix
        local = matrix @ matrix if composed else matrix
        ref = EmbeddedOp(ss, targets, StaticUnitaryOp(
            local, evotype='statevec_slow', state_space=sub)).to_dense('Hilbert')
        np.testing.assert_allclose(rep.acton(state).data, ref @ vector, atol=1e-12)
        np.testing.assert_allclose(rep.adjoint_acton(state).data, ref.conj().T @ vector, atol=1e-12)
        np.testing.assert_allclose(rep.acton_random(state, None).data, ref @ vector, atol=1e-12)
        np.testing.assert_allclose(rep.adjoint_acton_random(state, None).data, ref.conj().T @ vector, atol=1e-12)
        np.testing.assert_array_equal(state.data, vector)


@pytest.mark.parametrize('adjoint', [False, True])
@pytest.mark.parametrize('internal_rng', [False, True])
def test_statevec_embedded_random_draws_per_block(adjoint, internal_rng):
    ss = QubitSpace(4)
    targets = [3, 1]
    sub = ss.create_subspace(targets)
    rng = np.random.default_rng(42)
    matrices = [np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
                for _ in range(2)]
    inner = OpRepRandomUnitary(Basis.cast('pp', 16), np.array([0.5, 0.5]),
                              [OpRepDenseUnitary(mx, None, sub) for mx in matrices], 123, sub)
    rep = OpRepEmbedded(ss, targets, inner)
    vector = rng.normal(size=ss.udim) + 1j * rng.normal(size=ss.udim)
    state = StateRepDensePure(vector.copy(), ss, None)
    expected = vector.copy().reshape((2,) * 4)
    expected_rng = np.random.RandomState(123)
    # Untouched factors are 0 and 2; transpose each slice into target order (3, 1).
    for q0, q2 in np.ndindex(2, 2):
        matrix = matrices[expected_rng.choice(2, p=[0.5, 0.5])]
        if adjoint:
            matrix = matrix.conj().T
        block = expected[q0, :, q2, :].T.reshape(4)
        expected[q0, :, q2, :] = (matrix @ block).reshape(2, 2).T
    random_state = None if internal_rng else np.random.RandomState(123)
    action = rep.adjoint_acton_random if adjoint else rep.acton_random
    np.testing.assert_allclose(action(state, random_state).data, expected.ravel(), atol=1e-12)
    used_rng = inner.rand_state if internal_rng else random_state
    assert used_rng.randint(100000) == expected_rng.randint(100000)
    np.testing.assert_array_equal(state.data, vector)
