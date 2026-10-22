"""Covariance graph constraints, physical dephasing, and factory contracts."""

import numpy as np
import pytest

from pygsti.baseobjs import Basis, ExplicitBasis, ExplicitStateSpace, QubitSpace
from pygsti.baseobjs.statespace import QuditSpace
from pygsti.modelmembers.operations import LindbladErrorgen, ExpErrorgenOp
from pygsti.tools.basistools import change_basis


def construct(n=2, **kwargs):
    return LindbladErrorgen.from_correlated_dephasing(QubitSpace(n), **kwargs)


def covariance(errorgen):
    return errorgen.coefficient_blocks[0].kossakowski_matrix().real


def test_default_direction_order_and_independent_dense_counts():
    dense = construct(3)
    block = dense.coefficient_blocks[0]
    assert block.basis_element_labels == ('ZII', 'IZI', 'IIZ')
    assert dense.num_params == 6
    np.testing.assert_array_equal(dense.to_dense(), np.zeros((64, 64)))
    independent = construct(3, correlation_edges=(), initial_stochastic_rate=.02)
    assert independent.num_params == 3
    np.testing.assert_allclose(covariance(independent), .02 * np.eye(3))
    assert {label.errorgen_type for label in block.elementary_errorgens} == {'S', 'C'}


@pytest.mark.parametrize('normalization', ['row', 'column'])
def test_nonchordal_real_completion_keeps_covariance_zeros(normalization):
    bels = ('ZIII', 'IZII', 'IIZI', 'IIIZ')
    edges = [(bels[i], bels[(i + 1) % 4]) for i in range(4)]
    errorgen = construct(4, correlation_edges=edges, completion_normalization=normalization)
    block = errorgen.coefficient_blocks[0]
    assert errorgen.num_params == 8
    assert block._param_mode == 'cholesky completion'
    np.testing.assert_allclose(covariance(errorgen), 1e-6 * np.eye(4))
    vector = errorgen.to_vector()
    vector += np.random.default_rng(8).normal(scale=.08, size=8)
    errorgen.from_vector(vector)
    result = covariance(errorgen)
    assert result[0, 2] == result[1, 3] == 0
    assert np.linalg.eigvalsh(result)[0] > 0
    standard = change_basis(errorgen.to_dense(), 'pp', 'std')
    # Matrix units are eigenvectors and every population is fixed.
    np.testing.assert_allclose(standard - np.diag(standard.diagonal()), 0, atol=1e-14)
    np.testing.assert_allclose(standard.diagonal()[np.arange(16) * 17], 0, atol=1e-14)


@pytest.mark.parametrize('directions', [None, ('IZ', 'ZZ', 'ZI')])
def test_coherence_decay_matches_diagonal_eigenvalue_formula(directions):
    n = 2 if directions is None else 3
    lower = np.arange(1, n * n + 1).reshape(n, n) / 50
    target = lower @ lower.T + .01 * np.eye(n)
    errorgen = construct(basis_element_labels=directions, initial_covariance=target)
    np.testing.assert_allclose(covariance(errorgen), target, atol=1e-14)
    block = errorgen.coefficient_blocks[0]
    basis = Basis.cast('PP', 16)
    eigenvalues = np.array([basis[b].diagonal().real for b in block.basis_element_labels]).T
    difference = eigenvalues[:, None, :] - eigenvalues[None, :, :]
    rates = .5 * np.einsum('xya,ab,xyb->xy', difference, target, difference)
    generator = change_basis(errorgen.to_dense(), 'pp', 'std')
    np.testing.assert_allclose(generator, np.diag(-rates.ravel()), atol=1e-13)
    channel = change_basis(ExpErrorgenOp(errorgen).to_dense(), 'pp', 'std')
    np.testing.assert_allclose(channel, np.diag(np.exp(-rates).ravel()), atol=1e-13)
    # The Schur multiplier is PSD, equivalently the channel is CP.
    assert np.linalg.eigvalsh(np.exp(-rates))[0] >= -1e-12


@pytest.mark.parametrize('space,dimension', [(QuditSpace(('T0',), 3), 2),
                                          (ExplicitStateSpace(('Q0', 'T1'), (2, 3)), 3)])
def test_qudit_and_mixed_diagonal_directions(space, dimension):
    errorgen = LindbladErrorgen.from_correlated_dephasing(space, mx_basis='gm', initial_stochastic_rate=.02)
    assert len(errorgen.coefficient_blocks[0].basis_element_labels) == dimension
    assert errorgen.num_params == dimension * (dimension + 1) // 2
    standard = change_basis(errorgen.to_dense(), 'gm', 'std')
    np.testing.assert_allclose(standard - np.diag(standard.diagonal()), 0., atol=1e-13)


def test_custom_labels_need_explicit_selection():
    basis = ExplicitBasis(Basis.cast('PP', 4).elements, ('identity', 'a', 'b', 'phase'))
    with pytest.raises(ValueError, match='subsystem support'):
        construct(1, elementary_errorgen_basis=basis)
    errorgen = construct(1, elementary_errorgen_basis=basis, basis_element_labels=('phase',),
                         initial_covariance=[[.1]])
    assert errorgen.num_params == 1


def test_singular_common_mode_and_zero_supported_without_ridge():
    target = .02 * np.ones((3, 3))
    errorgen = construct(3, initial_covariance=target)
    np.testing.assert_allclose(covariance(errorgen), target, atol=1e-14)
    errorgen.from_vector(errorgen.to_vector())
    np.testing.assert_allclose(covariance(errorgen), target, atol=1e-14)
    assert np.linalg.matrix_rank(covariance(errorgen), tol=1e-12) == 1
    np.testing.assert_array_equal(covariance(construct()), np.zeros((2, 2)))


def test_completion_rejects_boundary_and_explicit_zero():
    bels = ('ZIII', 'IZII', 'IIZI', 'IIIZ')
    edges = [(bels[i], bels[(i + 1) % 4]) for i in range(4)]
    with pytest.raises(ValueError, match='positive.definite|positive pivots'):
        construct(4, correlation_edges=edges, initial_stochastic_rate=0.)
    laplacian = 2 * np.eye(4)
    for i in range(4):
        laplacian[i, (i + 1) % 4] = laplacian[(i + 1) % 4, i] = -1
    with pytest.raises(ValueError, match='positive.definite|positive pivots'):
        construct(4, correlation_edges=edges, initial_covariance=laplacian)


def test_covariance_derivatives_and_persistence():
    from pygsti.models import ExplicitOpModel
    from pygsti.modelmembers.states import StaticState, ComposedState
    from pygsti.modelmembers.povms import StaticPOVMEffect, UnconstrainedPOVM, ComposedPOVM
    from pygsti.tools.basistools import stdmx_to_vec

    errorgen = construct(initial_covariance=[[.04, -.01], [-.01, .03]])
    block = errorgen.coefficient_blocks[0]
    vector = block.to_vector()
    eps = 1e-6

    def data(v):
        block.from_vector(v)
        return block.block_data.copy()

    numeric = np.stack([(data(vector + eps * e) - data(vector - eps * e)) / (2 * eps)
                        for e in np.eye(len(vector))], axis=-1)
    np.testing.assert_allclose(block.deriv_wrt_params(vector), numeric, atol=1e-10)
    block.from_vector(vector)
    model = ExplicitOpModel(QubitSpace(2), basis='pp')
    reference_state = StaticState(stdmx_to_vec(np.ones((4, 4)) / 4, 'pp'),
                                  basis='pp', state_space=model.state_space)
    xx = Basis.cast('PP', 16)['XX']
    reference_povm = UnconstrainedPOVM([
        (label, StaticPOVMEffect(stdmx_to_vec((np.eye(4) + sign * xx) / 2, 'pp'),
                                basis='pp', state_space=model.state_space))
        for label, sign in (('+', 1), ('-', -1))
    ])
    model.preps['rho0'] = ComposedState(reference_state, ExpErrorgenOp(
        construct(initial_covariance=.25 * covariance(errorgen))))
    model.povms['Mdefault'] = ComposedPOVM(ExpErrorgenOp(
        construct(initial_covariance=.4 * covariance(errorgen))), reference_povm, mx_basis='pp')
    model.operations['Gi'] = ExpErrorgenOp(errorgen)
    assert model.num_params == 3 * errorgen.num_params
    assert model.probabilities(())[('+',)] < 1  # SPAM is physically noisy at zero idle duration.
    restored = ExplicitOpModel.from_nice_serialization(model.to_nice_serialization())
    for vector in (model.to_vector(), np.random.default_rng(4).normal(scale=.1, size=model.num_params)):
        model.from_vector(vector)
        restored.from_vector(vector)
        np.testing.assert_allclose(model.operations['Gi'].to_dense(), restored.operations['Gi'].to_dense())
        assert restored.probabilities(('Gi',)) == pytest.approx(model.probabilities(('Gi',)))
        restored_block = restored.operations['Gi'].errorgen.coefficient_blocks[0]
        assert all(label.errorgen_type in ('S', 'C') for label in restored_block.elementary_errorgens)


@pytest.mark.parametrize('kwargs,match', [
    ({'basis_element_labels': ()}, 'distinct nonidentity'),
    ({'basis_element_labels': ('ZI', 'ZI')}, 'distinct nonidentity'),
    ({'basis_element_labels': ('II',)}, 'distinct nonidentity'),
    ({'basis_element_labels': ('XI',)}, 'diagonal Hermitian'),
    ({'correlation_edges': [('ZI', 'XI')]}, 'distinct selected'),
    ({'correlation_edges': [('ZI', 'ZI')]}, 'distinct selected'),
    ({'correlation_edges': [('ZI', 'IZ'), ('IZ', 'ZI')]}, 'Duplicate'),
    ({'initial_covariance': np.eye(3)}, 'finite square'),
    ({'initial_covariance': [[1., np.nan], [0., 1.]]}, 'finite square'),
    ({'initial_covariance': [[1., 1j], [-1j, 1.]]}, 'must be real'),
    ({'initial_covariance': [[1., .2], [.1, 1.]]}, 'symmetric'),
    ({'initial_covariance': [[1., 2.], [2., 1.]]}, 'positive semidefinite'),
    ({'initial_covariance': [[1., .1], [.1, 1.]], 'correlation_edges': ()}, 'forbidden correlation'),
    ({'initial_covariance': np.eye(2), 'initial_stochastic_rate': .1}, 'not both'),
    ({'initial_stochastic_rate': -1.}, 'finite nonnegative'),
    ({'initial_stochastic_rate': np.nan}, 'finite nonnegative'),
    ({'param_mode': 'elements'}, 'param_mode'),
    ({'completion_normalization': 'invalid'}, 'completion_normalization'),
])
def test_invalid_requests(kwargs, match):
    with pytest.raises(ValueError, match=match):
        construct(**kwargs)
