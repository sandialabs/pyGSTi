"""Exact sparse CPTPLND support, including derivative and persistence contracts."""

import numpy as np
import pytest

from pygsti.baseobjs import Basis
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL
from pygsti.modelmembers.operations.lindbladcoefficients import LindbladCoefficientBlock as LCB


def sparse_block(complex_factor=True, edges=((0, 1), (1, 2), (1, 3))):
    basis = Basis.cast('PP', 16)
    bels = ('IX', 'IY', 'IZ', 'XI')
    labels = [LEEL('S', (b,)) for b in bels]
    for i, j in edges:
        labels.append(LEEL('C', (bels[i], bels[j])))
        if complex_factor:
            labels.append(LEEL('A', (bels[i], bels[j])))
    return LCB('other_sparse', basis, param_mode='cholesky', error_generator_labels=labels)


def fd_jac(function, vector, eps=1e-6):
    return np.stack([(function(vector + eps * e) - function(vector - eps * e)) / (2 * eps)
                     for e in np.eye(len(vector))], axis=-1)


@pytest.mark.parametrize('complex_factor', [False, True])
def test_exact_gram_support_and_roundtrip(complex_factor):
    blk = sparse_block(complex_factor)
    st = blk.cholesky_structure
    assert st.factor_pattern.nnz == st.n + st.pattern.nnz // 2
    assert blk.num_params == st.n + len(st.rows) * (2 if complex_factor else 1)
    for seed in range(5):
        vector = np.random.default_rng(seed).normal(size=blk.num_params)
        blk.from_vector(vector)
        K = blk.kossakowski_matrix()
        assert np.linalg.eigvalsh(K)[0] >= -1e-12
        forbidden = ~(st.pattern.toarray() | np.eye(st.n, dtype=bool))
        assert np.all(K[forbidden] == 0)
        # Independent dense Gram construction checks signs and the permutation convention.
        n, m = st.n, len(st.rows)
        L = np.diag(vector[:n]).astype(complex)
        L[st.rows, st.cols] = vector[n:n + m]
        if complex_factor:
            L[st.rows, st.cols] += 1j * vector[n + m:]
        expected = (L @ L.conj().T)[np.ix_(st.inv, st.inv)]
        np.testing.assert_allclose(K, expected, atol=1e-13)
        blk.from_vector(blk.to_vector())
        np.testing.assert_allclose(blk.kossakowski_matrix(), K, atol=1e-12)


@pytest.mark.parametrize('kind', ['zero', 'diagonal', 'rank_one', 'tiny'])
def test_singular_and_small_psd_inversion(kind):
    blk = sparse_block(edges=((0, 1), (0, 2), (1, 2)))
    if kind == 'zero':
        K = np.zeros((4, 4), complex)
    elif kind == 'diagonal':
        K = np.diag([0., 2., 0., 3.]).astype(complex)
    else:
        v = np.array([1., 2j, 3., 0.]) * (1e-12 if kind == 'tiny' else 1.)
        K = np.outer(v, v.conj())
    for k, (lbl, (i, j)) in enumerate(zip(blk._eeg_labels, blk._coefficient_pairs)):
        blk.block_data[k] = -K[i, j].imag if lbl.errorgen_type == 'A' else K[i, j].real
    blk.from_vector(blk.to_vector())
    np.testing.assert_allclose(blk.kossakowski_matrix(), K, rtol=1e-11, atol=1e-35 if kind == 'tiny' else 1e-11)


@pytest.mark.parametrize('complex_factor', [False, True])
def test_derivatives_hessians_polynomials_and_dense_reference(complex_factor):
    blk = sparse_block(complex_factor)
    v = np.random.default_rng(9).normal(size=blk.num_params)

    def data(x):
        blk.from_vector(x)
        return blk.block_data.copy()

    np.testing.assert_allclose(blk.deriv_wrt_params(v), fd_jac(data, v), atol=2e-9)
    superops = blk.create_lindblad_term_superoperators('pp', sparse=False, flat=True)
    np.testing.assert_allclose(blk.superop_hessian_wrt_params(superops, v, True),
                               fd_jac(lambda x: blk.superop_deriv_wrt_params(superops, x, True), v), atol=2e-9)
    blk.from_vector(v)
    reference = LCB('other', blk._basis, blk.basis_element_labels,
                    blk.kossakowski_matrix(), param_mode='static')
    expected = np.einsum('i,ijk->jk', reference.block_data.ravel(),
                         reference.create_lindblad_term_superoperators('pp', sparse=False, flat=True))
    actual = np.einsum('i,ijk->jk', blk.block_data, superops)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    for k, coefficient in enumerate(blk.block_data):
        poly = blk._parameterization._coefficient_polynomial(blk, k, 3, blk.num_params + 3)
        assert poly.evaluate(np.r_[np.zeros(3), v]) == pytest.approx(coefficient)


def test_serialization_conversion_and_invalid_coefficients():
    blk = sparse_block()
    blk.from_vector(np.random.default_rng(4).normal(size=blk.num_params))
    restored = LCB.from_nice_serialization(blk.to_nice_serialization())
    assert blk.is_similar(restored)
    np.testing.assert_allclose(restored.block_data, blk.block_data)
    np.testing.assert_array_equal(restored.cholesky_structure.perm, blk.cholesky_structure.perm)
    for mode in ('static', 'elements', 'cholesky'):
        converted = blk.convert(mode)
        np.testing.assert_allclose(converted.block_data, blk.block_data)
    blk.block_data[0] = -1.
    with pytest.raises(ValueError, match='positive semidefinite'):
        blk.to_vector()
    # Explicit truncation uses a diagonal shift and keeps the structural zeros.
    blk._truncate_block_data(True)
    assert np.linalg.eigvalsh(blk.kossakowski_matrix())[0] >= -1e-12


def test_reject_nonchordal_and_unmatched_component_support():
    with pytest.raises(ValueError, match='chordal'):
        sparse_block(edges=((0, 1), (1, 2), (2, 3), (3, 0)))
    blk = sparse_block()
    with pytest.raises(ValueError, match='matching C/A'):
        LCB('other_sparse', blk._basis, param_mode='cholesky',
            error_generator_labels=blk._eeg_labels[:-1])
    with pytest.raises(ValueError, match='endpoint'):
        LCB('other_sparse', blk._basis, param_mode='cholesky',
            error_generator_labels=blk._eeg_labels[1:])


def test_torch_matches_analytic_derivatives():
    pytest.importorskip('torch')
    from pygsti.tools._static_tls import import_torch
    torch = import_torch()
    blk = sparse_block()
    v = np.random.default_rng(6).normal(size=blk.num_params)
    blk.from_vector(v)
    sd = blk.stateless_data(torch.float64, 'cpu')
    func = lambda x: LCB.torch_base(sd, x)
    t = torch.tensor(v, dtype=torch.float64)
    np.testing.assert_allclose(func(t).numpy(), blk.block_data)
    np.testing.assert_allclose(torch.func.jacfwd(func)(t).numpy(), blk.deriv_wrt_params(v))
    batch = torch.stack((t, 2 * t))
    np.testing.assert_allclose(torch.vmap(func)(batch).numpy(), np.array([blk.block_data, 4 * blk.block_data]))

