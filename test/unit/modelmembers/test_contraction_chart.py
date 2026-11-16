"""The contraction chart K = C (I + H) C for sparse S/C/A blocks on any C/A mask."""

from itertools import combinations

import numpy as np
import pytest

from pygsti.baseobjs import Basis, QubitSpace
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL
from pygsti.modelmembers.operations import LindbladErrorgen, ExpErrorgenOp
from pygsti.modelmembers.operations.lindbladcoefficients import LindbladCoefficientBlock as LCB


def make_block(num_qubits, sectors, edges=None, flip=(), mode='contraction'):
    """S on every direction, plus C/A labels on `edges` chosen per edge by `sectors(edge_index)`."""
    basis = Basis.cast('PP', 4**num_qubits)
    bels = basis.labels[1:]
    labels = [LEEL('S', (b,)) for b in bels]
    pairs = list(combinations(range(len(bels)), 2)) if edges is None else edges
    for e, (i, j) in enumerate(pairs):
        for sector in sectors(e):
            labels.append(LEEL(sector, (bels[j], bels[i]) if e in flip else (bels[i], bels[j])))
    return LCB('other_sparse', basis, param_mode=mode, error_generator_labels=labels)


PATTERNS = {
    'S+C one qubit': lambda: make_block(1, lambda e: 'C'),
    'S+A one qubit': lambda: make_block(1, lambda e: 'A', flip=(1,)),
    'S+C+A one qubit': lambda: make_block(1, lambda e: 'CA', flip=(2,)),
    'mixed two qubits': lambda: make_block(2, lambda e: ['C', 'A', 'CA', ''][e % 4], flip=(1, 6, 11)),
    'two components and isolated directions': lambda: make_block(
        2, lambda e: ['CA', 'A', 'C'][e % 3], edges=[(0, 1), (1, 2), (5, 9), (9, 14), (5, 14)]),
}


def fd_jac(function, vector, eps=1e-6):
    return np.stack([(function(vector + eps * e) - function(vector - eps * e)) / (2 * eps)
                     for e in np.eye(len(vector))], axis=-1)


def allowed_mask(blk):
    """Which real and imaginary Kossakowski components the labels allow."""
    n = len(blk._bel_labels)
    real, imag = np.eye(n, dtype=bool), np.zeros((n, n), bool)
    for lbl, (i, j) in zip(blk._eeg_labels, blk._coefficient_pairs):
        target = imag if lbl.errorgen_type == 'A' else real
        target[i, j] = target[j, i] = True
    return real, imag


def set_kossakowski(blk, K):
    blk.block_data[:] = [K[i, i].real if lbl.errorgen_type == 'S' else
                         (K[i, j].real if lbl.errorgen_type == 'C' else -K[i, j].imag)
                         for lbl, (i, j) in zip(blk._eeg_labels, blk._coefficient_pairs)]


@pytest.mark.parametrize('pattern', list(PATTERNS))
def test_positivity_support_and_roundtrip(pattern):
    blk = PATTERNS[pattern]()
    assert blk.num_params == len(blk._eeg_labels)
    real, imag = allowed_mask(blk)
    rng = np.random.default_rng(3)
    for scale in (1e-3, .3, 1., 20.):
        vector = rng.normal(scale=scale, size=blk.num_params)
        blk.from_vector(vector)
        K = blk.kossakowski_matrix()
        assert np.linalg.eigvalsh(K)[0] >= -1e-14 * np.abs(K).max()
        np.testing.assert_array_equal(K.real[~real], 0)
        np.testing.assert_array_equal(K.imag[~imag], 0)
        data = blk.block_data.copy()
        blk.from_vector(blk.to_vector())
        np.testing.assert_allclose(blk.block_data, data, rtol=1e-12, atol=1e-14 * np.abs(data).max())


@pytest.mark.parametrize('pattern', list(PATTERNS))
def test_analytic_jacobian_and_hessian(pattern):
    blk = PATTERNS[pattern]()
    chart = blk._parameterization
    rng = np.random.default_rng(11)
    for scale in (.05, 1., 4.):
        vector = rng.normal(scale=scale, size=blk.num_params)
        _, jacobian, hessian = chart._evaluate(blk, vector, order=2)
        np.testing.assert_allclose(jacobian, fd_jac(lambda v: chart._evaluate(blk, v)[0], vector),
                                   atol=1e-8 * max(1, np.abs(jacobian).max()))
        np.testing.assert_allclose(hessian, fd_jac(lambda v: chart._evaluate(blk, v, order=1)[1], vector),
                                   atol=1e-7 * max(1, np.abs(hessian).max()))
        np.testing.assert_array_equal(blk.deriv_wrt_params(vector), jacobian)


def test_zero_noise_is_in_the_chart():
    blk = PATTERNS['mixed two qubits']()
    np.testing.assert_array_equal(blk.to_vector(), np.zeros(blk.num_params))
    np.testing.assert_array_equal(blk.block_data, 0)
    data, jacobian, hessian = blk._parameterization._evaluate(blk, np.zeros(blk.num_params), order=2)
    np.testing.assert_array_equal(data, 0)
    assert np.isfinite(jacobian).all() and np.isfinite(hessian).all()


@pytest.mark.parametrize('name, vector', [
    ('dephasing about a tilted axis', np.ones(3) / np.sqrt(3)),
    ('amplitude damping', np.array([1, 1j, 0]) / np.sqrt(2)),
    ('rank two', None),
])
def test_rank_deficient_boundary_is_reached_exactly(name, vector):
    blk = PATTERNS['S+C+A one qubit']()
    if vector is None:
        u = np.random.default_rng(1).normal(size=(3, 2)) + 1j * np.random.default_rng(2).normal(size=(3, 2))
        K = .02 * u @ u.conj().T
    else:
        K = .01 * np.outer(vector, vector.conj())
    set_kossakowski(blk, K)
    parameters = blk.to_vector()
    assert np.all(np.abs(parameters) <= 1)  # the boundary is at t = 1, not at infinity
    blk.from_vector(parameters)
    np.testing.assert_allclose(blk.kossakowski_matrix(), K, atol=1e-16)


def test_tilted_dephasing_is_outside_the_spectral_norm_ball():
    # Real off-diagonals have an asymmetric spectrum, which is why the gauge is -lambda_min.
    blk = PATTERNS['S+C one qubit']()
    K = .01 * np.ones((3, 3))
    set_kossakowski(blk, K)
    H = K / .01 - np.eye(3)
    assert np.linalg.norm(H, 2) == pytest.approx(2)
    blk.from_vector(blk.to_vector())
    np.testing.assert_allclose(blk.kossakowski_matrix(), K, atol=1e-16)


def test_inversion_rejects_or_truncates_infeasible_coefficients():
    blk = PATTERNS['S+C one qubit']()
    K = .01 * (np.eye(3) + 1.2 * (np.ones((3, 3)) - np.eye(3)) / 2)  # lambda_min of H is -0.6: feasible
    set_kossakowski(blk, K)
    blk.from_vector(blk.to_vector())
    np.testing.assert_allclose(blk.kossakowski_matrix(), K, atol=1e-16)
    bad = .01 * np.array([[1, -.9, -.9], [-.9, 1, -.9], [-.9, -.9, 1]])  # lambda_min = -0.8
    set_kossakowski(blk, bad)
    with pytest.raises(ValueError, match='not positive semidefinite'):
        blk.to_vector()
    truncated = blk._block_data_to_params(truncate=True)
    blk.from_vector(truncated)
    assert np.linalg.eigvalsh(blk.kossakowski_matrix())[0] == pytest.approx(0, abs=1e-15)
    set_kossakowski(blk, np.diag([.01, 0, .01]) + 1e-3 * (np.ones((3, 3)) - np.eye(3)))
    with pytest.raises(ValueError, match='S coefficients is zero'):
        blk.to_vector()
    set_kossakowski(blk, np.diag([.01, -1e-3, .01]))
    with pytest.raises(ValueError, match='nonnegative S'):
        blk.to_vector()


@pytest.mark.parametrize('pattern', list(PATTERNS))
def test_torch_reconstruction_matches(pattern):
    torch = pytest.importorskip('torch')
    blk = PATTERNS[pattern]()
    chart = blk._parameterization
    vector = np.random.default_rng(5).normal(scale=.7, size=blk.num_params)
    data, jacobian, _ = chart._evaluate(blk, vector, order=1)
    sd = chart.torch_stateless_data(blk, torch.float64, 'cpu')
    t_vector = torch.tensor(vector)
    np.testing.assert_allclose(chart.block_data_torch(sd, t_vector).numpy(), data, atol=1e-15)
    t_jacobian = torch.autograd.functional.jacobian(lambda x: chart.block_data_torch(sd, x), t_vector)
    np.testing.assert_allclose(t_jacobian.numpy(), jacobian, atol=1e-13)
    zero = torch.zeros(blk.num_params, dtype=torch.float64, requires_grad=True)
    chart.block_data_torch(sd, zero).sum().backward()
    assert torch.isfinite(zero.grad).all()


def test_auto_selection():
    space = QubitSpace(2)
    assert LindbladErrorgen.from_weight_limited(space).coefficient_blocks[-1]._param_mode == 'cholesky'
    assert LindbladErrorgen.from_weight_limited(
        space, errorgen_types=('S', 'C')).coefficient_blocks[-1]._param_mode == 'cholesky'
    for kwargs in (dict(errorgen_types=('S', 'A')), dict(max_weights=dict(C=2, A=1))):
        errgen = LindbladErrorgen.from_weight_limited(space, **kwargs)
        block = errgen.coefficient_blocks[-1]
        assert block._param_mode == 'contraction'
        np.testing.assert_array_equal(errgen.to_vector(), np.zeros(errgen.num_params))
    nonchordal = [('ZIII', 'IZII'), ('IZII', 'IIZI'), ('IIZI', 'IIIZ'), ('IIIZ', 'ZIII')]
    dephasing = LindbladErrorgen.from_correlated_dephasing(QubitSpace(4), correlation_edges=nonchordal)
    assert dephasing.coefficient_blocks[0]._param_mode == 'contraction'
    np.testing.assert_array_equal(dephasing.to_vector(), np.zeros(4 + 4))


def test_polynomial_terms_are_unsupported():
    blk = PATTERNS['S+A one qubit']()
    with pytest.raises(NotImplementedError, match='nonpolynomial'):
        blk._parameterization._coefficient_polynomial(blk, 3, 0, 10)


def test_model_derivatives_serialization_and_cptp():
    from pygsti.models import ExplicitOpModel
    from pygsti.modelmembers.operations import ComposedOp, StaticArbitraryOp
    from pygsti.modelmembers.povms import ComposedPOVM, ComputationalBasisPOVM
    from pygsti.modelmembers.states import ComposedState, ComputationalBasisState
    from pygsti.tools import jamiolkowski_iso
    space = QubitSpace(2)
    errgen = LindbladErrorgen.from_weight_limited(space, dict(H=1, S=2, C=1, A=2))
    blk = errgen.coefficient_blocks[-1]
    assert blk._param_mode == 'contraction'
    vector = np.random.default_rng(9).normal(scale=.1, size=errgen.num_params)
    errgen.from_vector(vector)
    channel = ExpErrorgenOp(errgen)
    assert np.linalg.eigvalsh(jamiolkowski_iso(channel.to_dense(), op_mx_basis='pp'))[0] >= -1e-12

    def dense(v):
        channel.from_vector(v)
        return channel.to_dense().copy().ravel()
    channel.from_vector(vector)
    np.testing.assert_allclose(channel.deriv_wrt_params(), fd_jac(dense, vector), atol=1e-9)
    errgen.from_vector(vector)
    expected_hessian = fd_jac(lambda v: (errgen.from_vector(v), errgen.deriv_wrt_params())[1], vector)
    errgen.from_vector(vector)
    np.testing.assert_allclose(errgen.hessian_wrt_params(), expected_hessian, atol=1e-8)

    model = ExplicitOpModel(space, basis='pp')
    model.preps['rho0'] = ComposedState(ComputationalBasisState([0, 0], state_space=space),
                                        ExpErrorgenOp(errgen.copy()))
    model.povms['Mdefault'] = ComposedPOVM(ExpErrorgenOp(errgen.copy()), ComputationalBasisPOVM(2, state_space=space),
                                           mx_basis='pp')
    model.operations['Gx'] = ComposedOp([StaticArbitraryOp(np.eye(16), state_space=space), channel])
    restored = ExplicitOpModel.from_nice_serialization(model.to_nice_serialization())
    np.testing.assert_allclose(restored.to_vector(), model.to_vector())
    shifted = model.to_vector() + .01
    model.from_vector(shifted)
    restored.from_vector(shifted)
    assert restored.probabilities(('Gx', 'Gx')) == pytest.approx(model.probabilities(('Gx', 'Gx')))
    assert restored.operations['Gx'].factorops[1].errorgen.coefficient_blocks[-1]._param_mode == 'contraction'
