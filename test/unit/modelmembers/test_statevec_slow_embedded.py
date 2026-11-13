import numpy as np
import pytest
from scipy.linalg import block_diag

from pygsti.baseobjs import Basis
from pygsti.baseobjs.statespace import ExplicitStateSpace, QubitSpace, StateSpace
from pygsti.evotypes.statevec_slow.opreps import (
    OpRepComposed, OpRepDenseUnitary, OpRepEmbedded, OpRepRandomUnitary)
from pygsti.evotypes.statevec_slow.statereps import StateRepDensePure
from pygsti.modelmembers.operations import StaticUnitaryOp


def _hilbert_embedding_reference(ss: StateSpace, targets: list[str | int], matrix: np.ndarray) -> np.ndarray:
    """Form I ⊗ matrix, then permute its tensor axes into the physical label order."""
    block_index = ss.label_tensor_product_block_index(targets[0])
    labels = list(ss.tensor_product_block_labels(block_index))
    spectators = [label for label in labels if label not in targets]
    factor_order = spectators + targets
    dims = [ss.label_udimension(label) for label in factor_order]
    spectator_dim = int(np.prod([ss.label_udimension(label) for label in spectators]))
    active = np.kron(np.eye(spectator_dim), matrix)
    row_axes = [factor_order.index(label) for label in labels]
    column_axes = [axis + len(labels) for axis in row_axes]
    active = active.reshape(dims + dims).transpose(row_axes + column_axes).reshape(active.shape)

    # Other direct-sum sectors are independent identity blocks.
    blocks = [np.eye(int(np.prod(dims))) for dims in ss.tensor_product_blocks_udimensions]
    blocks[block_index] = active
    return block_diag(*blocks)


@pytest.mark.parametrize('ss, targets', [
    (QubitSpace(1), [0]),
    (QubitSpace(2), [1, 0]),
    (QubitSpace(4), [3, 1]),
    (QubitSpace(['q2', 'q16', 'q5']), ['q5', 'q2']),
    (ExplicitStateSpace([('Q0', 'T1', 'Q2')], [(2, 3, 2)]), ['T1', 'Q2']),
    (ExplicitStateSpace([('Q0', 'T1', 'Q2')], [(2, 3, 2)]), ['Q2', 'T1']),
])
@pytest.mark.parametrize('composed', [False, True])
def test_statevec_embedded_actions(ss: StateSpace, targets: list[str | int], composed: bool) -> None:
    sub = ss.create_subspace(targets)
    rng = np.random.default_rng(90)
    mx = np.linalg.qr(rng.normal(size=(sub.udim, sub.udim))
                      + 1j * rng.normal(size=(sub.udim, sub.udim)))[0]
    op = StaticUnitaryOp(mx, evotype='statevec_slow', state_space=sub)
    inner = OpRepComposed([op._rep, op._rep], sub) if composed else op._rep
    rep = OpRepEmbedded(ss, targets, inner)
    vector = rng.normal(size=ss.udim) + 1j * rng.normal(size=ss.udim)
    state = StateRepDensePure(vector.copy(), ss, None)

    # Use a tensor reference rather than EmbeddedOp.to_dense: the two implementations
    # have different known bugs and must be checked independently.
    for matrix in (mx, mx.conj()):
        op._rep.base[:] = matrix
        local = matrix @ matrix if composed else matrix
        ref = _hilbert_embedding_reference(ss, targets, local)
        np.testing.assert_allclose(rep.acton(state).data, ref @ vector, atol=1e-12)
        np.testing.assert_allclose(rep.adjoint_acton(state).data, ref.conj().T @ vector, atol=1e-12)
        np.testing.assert_allclose(rep.acton_random(state, None).data, ref @ vector, atol=1e-12)
        np.testing.assert_allclose(rep.adjoint_acton_random(state, None).data, ref.conj().T @ vector, atol=1e-12)
        np.testing.assert_array_equal(state.data, vector)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='statevec_slow uses operator-space direct-sum offsets')
@pytest.mark.parametrize('action', ['acton', 'adjoint_acton', 'acton_random', 'adjoint_acton_random'])
def test_statevec_embedded_direct_sum_actions(action: str) -> None:
    ss = ExplicitStateSpace([('Qleft',), ('Q0', 'T1', 'Q2'), ('Qright',)], [(2,), (2, 3, 2), (2,)])
    targets = ['T1', 'Q2']
    sub = ss.create_subspace(targets)
    rng = np.random.default_rng(91)
    mx = np.linalg.qr(rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))[0]
    rep = OpRepEmbedded(ss, targets, OpRepDenseUnitary(mx, None, sub))
    vector = rng.normal(size=ss.udim) + 1j * rng.normal(size=ss.udim)
    state = StateRepDensePure(vector.copy(), ss, None)
    ref = _hilbert_embedding_reference(ss, targets, mx)
    if action.startswith('adjoint'):
        ref = ref.conj().T
    args = (state, None) if action.endswith('random') else (state,)
    actual = getattr(rep, action)(*args).data
    np.testing.assert_allclose(actual, ref @ vector, atol=1e-12)
    np.testing.assert_array_equal(state.data, vector)


@pytest.mark.parametrize('adjoint', [False, True])
@pytest.mark.parametrize('internal_rng', [False, True])
def test_statevec_embedded_random_draws_per_block(adjoint: bool, internal_rng: bool) -> None:
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
