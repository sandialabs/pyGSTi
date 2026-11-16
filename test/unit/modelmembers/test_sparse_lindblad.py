"""Exact sparse CPTPLND support, including derivative and persistence contracts."""

import numpy as np
import pytest

from pygsti.baseobjs import Basis, QubitSpace
from pygsti.baseobjs.statespace import QuditSpace
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL
from pygsti.modelmembers.operations.lindbladcoefficients import LindbladCoefficientBlock as LCB
from pygsti.modelmembers.operations import LindbladErrorgen, ExpErrorgenOp, ComposedOp, StaticUnitaryOp


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


@pytest.mark.parametrize('nqubits', [1, 2, 3])
def test_weight_two_graph_and_parameter_count(nqubits):
    errgen = LindbladErrorgen.from_weight_limited(QubitSpace(nqubits), dict(H=1, S=2, C=2, A=2))
    blk = errgen.coefficient_blocks[-1]
    n = 3 * nqubits + 9 * nqubits * (nqubits - 1) // 2
    edges = (3 * nqubits) * (3 * nqubits - 1) // 2 + 90 * nqubits * (nqubits - 1) // 2
    assert blk.num_params == n + 2 * edges
    assert errgen.num_params == 3 * nqubits + blk.num_params
    assert blk.cholesky_structure.factor_pattern.nnz == n + edges
    assert np.all(errgen.to_vector() == 0)
    errgen.from_vector(np.random.default_rng(1).normal(scale=.01, size=errgen.num_params))
    assert np.linalg.eigvalsh(blk.kossakowski_matrix())[0] >= -1e-12
    assert all(len(lbl.support_indices()) <= (1 if lbl.errorgen_type == 'H' else 2)
               for lbl in errgen.coefficients(label_type='local'))


@pytest.mark.parametrize('sectors,limits,expected', [
    (('S',), dict(S=1), 6),
    (('H',), dict(H=1), 6),
    (('S', 'C'), dict(S=2, C=1), 21),
    (('S', 'C', 'A'), dict(S=2, C=1, A=1), 27),
])
def test_sector_selection(sectors, limits, expected):
    errgen = LindbladErrorgen.from_weight_limited(QubitSpace(2), limits, sectors)
    assert errgen.num_params == expected
    errgen.from_vector(np.random.default_rng(3).normal(scale=.01, size=expected))
    for lbl in errgen.coefficients(label_type='local'):
        assert lbl.errorgen_type in sectors
        assert len(lbl.support_indices()) <= limits.get(lbl.errorgen_type, 2)


def test_post_gate_operation_model_and_serialization():
    from pygsti.models import ExplicitOpModel
    from pygsti.modelmembers.states import ComputationalBasisState, ComposedState
    from pygsti.modelmembers.povms import ComputationalBasisPOVM, ComposedPOVM
    from pygsti.tools import jamiolkowski_iso

    space = QubitSpace(('left', 'right'))
    errgen = LindbladErrorgen.from_weight_limited(space, dict(H=1, S=2, C=1, A=1))
    errgen.from_vector(np.random.default_rng(19).normal(scale=.03, size=errgen.num_params))
    channel = ExpErrorgenOp(errgen)
    ideal = StaticUnitaryOp(np.kron([[0., 1.], [1., 0.]], np.eye(2)), state_space=space)
    gate = ComposedOp([ideal, channel])
    np.testing.assert_allclose(gate.to_dense(), channel.to_dense() @ ideal.to_dense(), atol=1e-12)
    np.testing.assert_allclose(gate.to_dense()[0], np.eye(16)[0], atol=1e-12)
    choi = jamiolkowski_iso(gate.to_dense(), op_mx_basis='pp')
    assert np.linalg.eigvalsh(choi)[0] >= -1e-12
    # Noise is not covariant with the ideal gate, so this also detects wrong composition order.
    assert np.linalg.norm(gate.to_dense() - ideal.to_dense() @ channel.to_dense()) > 1e-6
    model = ExplicitOpModel(space, basis='pp')
    prep_errorgen, meas_errorgen = errgen.copy(), errgen.copy()
    prep_errorgen.from_vector(.6 * errgen.to_vector())
    meas_errorgen.from_vector(.4 * errgen.to_vector())
    reference_state = ComputationalBasisState([0, 0], state_space=space)
    reference_povm = ComputationalBasisPOVM(2, state_space=space)
    model.preps['rho0'] = ComposedState(reference_state, ExpErrorgenOp(prep_errorgen))
    model.povms['Mdefault'] = ComposedPOVM(ExpErrorgenOp(meas_errorgen), reference_povm, mx_basis='pp')
    model.operations['Gx'] = gate
    assert model.num_params == 3 * errgen.num_params
    restored = ExplicitOpModel.from_nice_serialization(model.to_nice_serialization())
    assert restored.num_params == model.num_params
    for vector in (model.to_vector(), np.random.default_rng(8).normal(scale=.02, size=model.num_params)):
        model.from_vector(vector)
        restored.from_vector(vector)
        np.testing.assert_allclose(restored.operations['Gx'].to_dense(), model.operations['Gx'].to_dense())
        probabilities = model.probabilities(('Gx', 'Gx'))
        assert sum(probabilities.values()) == pytest.approx(1.)
        assert min(probabilities.values()) >= -1e-12
        assert restored.probabilities(('Gx', 'Gx')) == pytest.approx(probabilities)
        prep = model.preps['rho0']
        povm = model.povms['Mdefault']
        np.testing.assert_allclose(prep.to_dense(), prep.error_map.to_dense() @ reference_state.to_dense())
        for label in povm:
            np.testing.assert_allclose(povm[label].to_dense(),
                                       povm.error_map.to_dense().T @ reference_povm[label].to_dense())
        restored_blk = restored.operations['Gx'].factorops[1].errorgen.coefficient_blocks[-1]
        assert restored_blk.block_type == 'other_sparse'
        assert np.linalg.eigvalsh(restored_blk.kossakowski_matrix())[0] >= -1e-12


def test_errorgen_and_operation_analytic_derivatives():
    errgen = LindbladErrorgen.from_weight_limited(QubitSpace(1), dict(H=1, S=1, C=1, A=1))
    vector = np.random.default_rng(4).normal(scale=.03, size=errgen.num_params)

    def dense(v):
        errgen.from_vector(v)
        return errgen.to_dense().copy().ravel()

    np.testing.assert_allclose(errgen.deriv_wrt_params(), fd_jac(dense, np.zeros_like(vector)), atol=1e-9)
    errgen.from_vector(vector)
    np.testing.assert_allclose(errgen.deriv_wrt_params(), fd_jac(dense, vector), atol=1e-9)
    np.testing.assert_allclose(errgen.hessian_wrt_params(),
                               fd_jac(lambda v: (errgen.from_vector(v), errgen.deriv_wrt_params())[1], vector), atol=1e-9)
    op = ExpErrorgenOp(errgen)

    def op_dense(v):
        op.from_vector(v)
        return op.to_dense().copy().ravel()

    op.from_vector(vector)
    np.testing.assert_allclose(op.deriv_wrt_params(), fd_jac(op_dense, vector), atol=1e-8)


def test_qudit_weight_limits_and_positivity():
    space = QuditSpace(('qutrit',), (3,))
    errgen = LindbladErrorgen.from_weight_limited(space, dict(S=1, C=1, A=1),
                                                errorgen_types=('S', 'C', 'A'), mx_basis='gm')
    assert errgen.num_params == 64
    errgen.from_vector(np.random.default_rng(4).normal(scale=.01, size=errgen.num_params))
    blk = errgen.coefficient_blocks[0]
    assert np.linalg.eigvalsh(blk.kossakowski_matrix())[0] >= -1e-12
    np.testing.assert_allclose(ExpErrorgenOp(errgen).to_dense()[0], np.eye(9)[0], atol=1e-12)


def test_sparse_operation_torch_derivatives():
    torch = pytest.importorskip('torch')
    errgen = LindbladErrorgen.from_weight_limited(QubitSpace(1), dict(H=1, S=1, C=1, A=1))
    vector = np.random.default_rng(17).normal(scale=.02, size=errgen.num_params)
    errgen.from_vector(vector)
    for obj in (errgen, ExpErrorgenOp(errgen)):
        sd = obj.stateless_data(torch.float64, 'cpu')
        func = lambda v: type(obj).torch_base(sd, v).ravel()
        tv = torch.tensor(vector, dtype=torch.float64)
        np.testing.assert_allclose(func(tv).numpy(), obj.to_dense().ravel(), atol=1e-12)
        np.testing.assert_allclose(torch.func.jacrev(func)(tv).numpy(), obj.deriv_wrt_params(), atol=1e-9)
    sd = errgen.stateless_data(torch.float64, 'cpu')
    func = lambda v: LindbladErrorgen.torch_base(sd, v).ravel()
    np.testing.assert_allclose(torch.func.hessian(func)(tv).numpy(), errgen.hessian_wrt_params(), atol=1e-9)


@pytest.mark.parametrize('kwargs,match', [
    (dict(max_weights=dict(C=1, A=2), param_mode='cholesky'), 'matching C/A'),
    (dict(max_weights=dict(S=2, C=3, A=3), state_space=QubitSpace(4), param_mode='cholesky'), 'chordal'),
    (dict(errorgen_types=('A',)), 'require S'),
    (dict(errorgen_types=('S', 'A'), param_mode='cholesky'), 'matching C/A'),
    (dict(errorgen_types=('Q',)), 'only H'),
    (dict(max_weights=dict(S=-1)), 'nonnegative integer'),
    (dict(max_weights=dict(S=1.5)), 'nonnegative integer'),
    (dict(max_weights=dict(S=True)), 'nonnegative integer'),
    (dict(max_weights=dict(H=0, S=0)), 'at least one'),
])
def test_weight_limited_rejects_unsupported_constraints(kwargs, match):
    kwargs = dict(state_space=QubitSpace(2)) | kwargs
    with pytest.raises(ValueError, match=match):
        LindbladErrorgen.from_weight_limited(**kwargs)
