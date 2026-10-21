"""Exact nonchordal/component-constrained Cholesky completion and its derivatives."""

from itertools import combinations

import numpy as np
import pytest

from pygsti.baseobjs import Basis
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL
from pygsti.modelmembers.operations.lindbladcoefficients import LindbladCoefficientBlock as LCB


def cycle_block(components='complex', ordering=(0, 1, 2, 3), mode='cholesky completion'):
    basis = Basis.cast('PP', 16)
    bels = ('IX', 'IY', 'IZ', 'XI')
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    labels = [LEEL('S', (b,)) for b in bels]
    for e, (i, j) in enumerate(edges):
        for sector in ('C', 'A'):
            if (components == 'complex' or (components == 'real' and sector == 'C')
                    or (components == 'imag' and sector == 'A')
                    or (components == 'mixed' and (sector == 'C' or e % 2 == 0))):
                labels.append(LEEL(sector, (bels[i], bels[j])))
    return LCB('other_sparse', basis, param_mode=mode, error_generator_labels=labels,
               cholesky_ordering=ordering)


def fd_jac(function, vector, eps=1e-6):
    return np.stack([(function(vector + eps * e) - function(vector - eps * e)) / (2 * eps)
                     for e in np.eye(len(vector))], axis=-1)


def full_gram(blk, vector):
    values = blk._parameterization._factor_jet(blk, vector)[0]
    st = blk.cholesky_structure
    factor = np.zeros((st.n, st.n), complex)
    factor[blk._completion_rows, blk._completion_cols] = values
    return (factor @ factor.conj().T)[np.ix_(st.inv, st.inv)]


@pytest.mark.parametrize('components', ['complex', 'real', 'imag', 'mixed'])
@pytest.mark.parametrize('ordering', [(0, 1, 2, 3), (2, 0, 3, 1)])
def test_support_positivity_and_coordinate_roundtrip(components, ordering):
    blk = cycle_block(components, ordering)
    assert blk.num_params == len(blk._eeg_labels)
    np.testing.assert_array_equal(blk.to_vector(), np.zeros(blk.num_params))
    rng = np.random.default_rng(10)
    for _ in range(3):
        vector = rng.normal(scale=.2, size=blk.num_params)
        blk.from_vector(vector)
        gram = full_gram(blk, vector)
        np.testing.assert_allclose(gram, blk.kossakowski_matrix(), atol=1e-14)
        assert np.linalg.eigvalsh(gram)[0] > 0
        np.testing.assert_allclose(blk.to_vector(), vector, atol=1e-13)


@pytest.mark.parametrize('components', ['complex', 'real', 'imag', 'mixed'])
def test_analytic_jacobian_hessian_and_explicit_vector(components):
    blk = cycle_block(components)
    vector = np.random.default_rng(7).normal(scale=.2, size=blk.num_params)
    data = lambda v: blk._parameterization._evaluate(blk, v)[0]
    np.testing.assert_allclose(blk.deriv_wrt_params(vector), fd_jac(data, vector), atol=2e-9)
    superops = blk.create_lindblad_term_superoperators('pp', sparse=False, flat=True)
    # The block's stored state differs deliberately: completed Hessians depend on v.
    assert np.linalg.norm(blk.to_vector() - vector) > .1
    actual = blk.superop_hessian_wrt_params(superops, vector, True)
    expected = fd_jac(lambda v: blk.superop_deriv_wrt_params(superops, v, True), vector)
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    _, jac, hess = blk._parameterization._evaluate(blk, vector, order=2)
    np.testing.assert_allclose(jac, blk.deriv_wrt_params(vector))
    np.testing.assert_allclose(hess, fd_jac(lambda v: blk.deriv_wrt_params(v), vector), atol=2e-9)


@pytest.mark.parametrize('components', ['complex', 'real', 'imag', 'mixed'])
def test_independent_supported_pd_target_inversion(components):
    blk = cycle_block(components)
    rng = np.random.default_rng(2)
    for scale in (1., 1e-12, 1e-24):
        # Construct rates directly, independently of the forward map.
        rates = rng.normal(scale=.02, size=len(blk.block_data))
        for k, label in enumerate(blk._eeg_labels):
            if label.errorgen_type == 'S':
                rates[k] = 1.
        blk.block_data[:] = scale * rates
        expected = blk.kossakowski_matrix()
        vector = blk.to_vector()
        blk.from_vector(vector)
        np.testing.assert_allclose(blk.kossakowski_matrix() / scale, expected / scale, atol=1e-13)


@pytest.mark.parametrize('components', ['complex', 'mixed'])
def test_torch_values_derivatives_hessian_and_vmap(components):
    torch = pytest.importorskip('torch')
    blk = cycle_block(components)
    vector = np.random.default_rng(9).normal(scale=.15, size=blk.num_params)
    blk.from_vector(vector)
    sd = blk.stateless_data(torch.float64, 'cpu')
    func = lambda v: LCB.torch_base(sd, v)
    v = torch.tensor(vector, dtype=torch.float64)
    np.testing.assert_allclose(func(v).numpy(), blk.block_data, atol=1e-13)
    np.testing.assert_allclose(torch.func.jacfwd(func)(v).numpy(), blk.deriv_wrt_params(vector), atol=1e-12)
    np.testing.assert_allclose(torch.func.hessian(func)(v).numpy(),
                               blk._parameterization._evaluate(blk, vector, order=2)[2], atol=1e-12)
    batch = torch.stack((v, v + .03))
    expected = np.stack([blk._parameterization._evaluate(blk, x.numpy())[0] for x in batch])
    np.testing.assert_allclose(torch.vmap(func)(batch).numpy(), expected, atol=1e-13)


def test_serialization_preserves_order_and_coordinates():
    from unittest.mock import patch
    blk = cycle_block(ordering=(2, 0, 3, 1))
    vector = np.random.default_rng(8).normal(scale=.1, size=blk.num_params)
    blk.from_vector(vector)
    saved = blk.to_nice_serialization()
    assert saved['cholesky_ordering'] == [2, 0, 3, 1]
    # Loading must not silently choose a different chart on another installation.
    with patch('pygsti.tools.graphs.sparsechol.fill_reducing_ordering', side_effect=AssertionError):
        restored = LCB.from_nice_serialization(saved)
    assert restored.is_similar(blk)
    np.testing.assert_allclose(restored.to_vector(), vector, atol=1e-13)
    for mode in ('static', 'elements', 'cholesky completion'):
        converted = blk.convert(mode)
        np.testing.assert_allclose(converted.block_data, blk.block_data)


def test_polynomial_and_singular_inputs_are_rejected_explicitly():
    blk = cycle_block()
    with pytest.raises(NotImplementedError, match='nonpolynomial'):
        blk._parameterization._coefficient_polynomial(blk, 0, 0, 100)
    for rates in (np.zeros(blk.num_params), np.r_[-1., np.zeros(blk.num_params - 1)]):
        blk.block_data[:] = rates
        with pytest.raises(ValueError, match='positive-definite'):
            blk.to_vector()
    for vector in (np.full(blk.num_params, np.nan), np.full(blk.num_params, 1000.)):
        with pytest.raises(ValueError):
            blk.from_vector(vector)


def test_four_qubit_weight_three_full_block_and_directional_derivative():
    basis = Basis.cast('PP', 256)
    bels = [b for b in basis.labels[1:] if sum(c != 'I' for c in b) <= 3]
    labels = [LEEL('S', (b,)) for b in bels]
    for a, b in combinations(bels, 2):
        if sum(x != 'I' or y != 'I' for x, y in zip(a, b)) <= 3:
            labels.extend([LEEL('C', (a, b)), LEEL('A', (a, b))])
    ordering = sorted(range(len(bels)), key=lambda i: -sum(c != 'I' for c in bels[i]))
    blk = LCB('other_sparse', basis, param_mode='cholesky completion',
               error_generator_labels=labels, cholesky_ordering=ordering)
    assert len(bels) == 174
    assert blk.num_params == 14562
    rng = np.random.default_rng(4)
    vector = rng.normal(scale=.02, size=blk.num_params)
    vector[:len(bels)] -= 6.
    blk.from_vector(vector)
    np.testing.assert_allclose(full_gram(blk, vector), blk.kossakowski_matrix(), atol=1e-19)
    np.testing.assert_allclose(blk.to_vector(), vector, atol=1e-12)
    direction = rng.normal(size=blk.num_params)
    _, derivative, _ = blk._parameterization._evaluate(blk, vector, order=1, directions=direction[:, None])
    step = 1e-6
    numerical = (blk._parameterization._evaluate(blk, vector + step * direction)[0]
                 - blk._parameterization._evaluate(blk, vector - step * direction)[0]) / (2 * step)
    np.testing.assert_allclose(derivative[:, 0], numerical, atol=1e-14, rtol=1e-7)
