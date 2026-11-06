"""Exact nonchordal/component-constrained Cholesky completion and its derivatives."""

from itertools import combinations

import numpy as np
import pytest

from pygsti.baseobjs import Basis
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL
from pygsti.modelmembers.operations.lindbladcoefficients import LindbladCoefficientBlock as LCB


def cycle_block(components='complex', ordering=(0, 1, 2, 3), mode='cholesky completion', normalization='row'):
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
               cholesky_ordering=ordering, completion_normalization=normalization)


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
    assert saved['completion_normalization'] == 'row'
    # Loading must not silently choose a different chart on another installation.
    with patch('pygsti.tools.graphs.sparsechol.fill_reducing_ordering', side_effect=AssertionError):
        restored = LCB.from_nice_serialization(saved)
    assert restored.is_similar(blk)
    np.testing.assert_allclose(restored.to_vector(), vector, atol=1e-13)
    for mode in ('static', 'elements', 'cholesky completion'):
        converted = blk.convert(mode)
        np.testing.assert_allclose(converted.block_data, blk.block_data)


def test_earlier_column_normalization_serialization_and_derivatives():
    blk = cycle_block(normalization='column')
    vector = np.random.default_rng(3).normal(scale=.1, size=blk.num_params)
    blk.from_vector(vector)
    saved = blk.to_nice_serialization()
    del saved['completion_normalization']  # Earlier development artifacts omitted it.
    restored = LCB.from_nice_serialization(saved)
    assert restored._completion_normalization == 'column'
    assert restored.is_similar(blk)
    np.testing.assert_allclose(restored.to_vector(), vector, atol=1e-13)
    np.testing.assert_allclose(blk.deriv_wrt_params(vector),
                               fd_jac(lambda v: blk._parameterization._evaluate(blk, v)[0], vector), atol=1e-9)
    _, _, hess = blk._parameterization._evaluate(blk, vector, order=2)
    np.testing.assert_allclose(hess, fd_jac(lambda v: blk.deriv_wrt_params(v), vector), atol=1e-9)
    torch = pytest.importorskip('torch')
    sd = blk.stateless_data(torch.float64, 'cpu')
    np.testing.assert_allclose(LCB.torch_base(sd, torch.tensor(vector)).numpy(), blk.block_data)


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


@pytest.mark.parametrize('scale', [1., 1e-24])
@pytest.mark.parametrize('row_scale', [1., 1e-12])
def test_singular_cycle_rejection_is_invariant_under_diagonal_scaling(scale, row_scale):
    blk = cycle_block('real')
    laplacian = 2 * np.eye(4)
    for i in range(4):
        laplacian[i, (i + 1) % 4] = laplacian[(i + 1) % 4, i] = -1
    diagonal = np.array([1., row_scale, 1., 1.])
    for epsilon in (0., 1e-6):
        target = scale * np.outer(diagonal, diagonal) * (laplacian + epsilon * np.eye(4))
        for k, (i, j) in enumerate(blk._coefficient_pairs):
            blk.block_data[k] = target[i, j]
        if epsilon == 0:
            with pytest.raises(ValueError, match='positive-definite'):
                blk.to_vector()
        else:
            vector = blk.to_vector()
            blk.from_vector(vector)
            np.testing.assert_allclose(blk.kossakowski_matrix(), target, rtol=1e-10, atol=0.)


def test_four_qubit_weight_three_full_block_and_directional_derivative():
    basis = Basis.cast('PP', 256)
    bels = [b for b in basis.labels[1:] if sum(c != 'I' for c in b) <= 3]
    labels = [LEEL('S', (b,)) for b in bels]
    for a, b in combinations(bels, 2):
        if sum(x != 'I' or y != 'I' for x, y in zip(a, b)) <= 3:
            labels.extend([LEEL('C', (a, b)), LEEL('A', (a, b))])
    ordering = sorted(range(len(bels)), key=lambda i: -sum(c != 'I' for c in bels[i]))
    blk = LCB('other_sparse', basis, param_mode='auto',
               error_generator_labels=labels, cholesky_ordering=ordering)
    assert blk._param_mode == 'cholesky completion'
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


@pytest.mark.parametrize('normalization', ['row', 'column'])
@pytest.mark.parametrize('mode', ['cholesky', 'cholesky completion'])
def test_factory_initialization_and_legacy_zero(mode, normalization):
    from pygsti.baseobjs import QubitSpace
    from pygsti.modelmembers.operations import LindbladErrorgen
    space = QubitSpace(1)
    errgen = LindbladErrorgen.from_weight_limited(space, param_mode=mode, initial_stochastic_rate=.002,
                                                 completion_normalization=normalization)
    block = errgen.coefficient_blocks[-1]
    np.testing.assert_allclose(block.kossakowski_matrix(), .002 * np.eye(3))
    np.testing.assert_allclose(errgen.to_dense()[0], np.zeros(4), atol=1e-15)
    automatic = LindbladErrorgen.from_weight_limited(space)
    assert automatic.coefficient_blocks[-1]._param_mode == 'cholesky'
    np.testing.assert_array_equal(automatic.to_vector(), np.zeros(automatic.num_params))


def test_factory_independent_weights_and_physical_operation():
    from pygsti.baseobjs import QubitSpace
    from pygsti.modelmembers.operations import LindbladErrorgen, ExpErrorgenOp
    from pygsti.tools import jamiolkowski_iso
    errgen = LindbladErrorgen.from_weight_limited(
        QubitSpace(3), dict(H=1, S=2, C=2, A=3), initial_stochastic_rate=1e-4)
    block = errgen.coefficient_blocks[-1]
    assert block._param_mode == 'cholesky completion'
    assert block.num_params == 972
    np.testing.assert_allclose(block.kossakowski_matrix(), 1e-4 * np.eye(36))
    vector = errgen.to_vector()
    vector += np.random.default_rng(5).normal(scale=.01, size=errgen.num_params)
    errgen.from_vector(vector)
    for label in errgen.coefficients(label_type='local'):
        assert len(label.support_indices()) <= dict(H=1, S=2, C=2, A=3)[label.errorgen_type]
    channel = ExpErrorgenOp(errgen).to_dense()
    np.testing.assert_allclose(channel[0], np.eye(64)[0], atol=1e-13)
    assert np.linalg.eigvalsh(jamiolkowski_iso(channel, op_mx_basis='pp'))[0] >= -1e-12


def test_factory_active_only_offdiagonals():
    from pygsti.baseobjs import QubitSpace
    from pygsti.modelmembers.operations import LindbladErrorgen
    errgen = LindbladErrorgen.from_weight_limited(QubitSpace(2), dict(S=2, A=2), ('S', 'A'))
    block = errgen.coefficient_blocks[0]
    assert block._param_mode == 'cholesky completion'
    vector = errgen.to_vector() + np.random.default_rng(2).normal(scale=.02, size=errgen.num_params)
    errgen.from_vector(vector)
    K = block.kossakowski_matrix()
    np.testing.assert_array_equal(K.real - np.diag(K.diagonal().real), np.zeros(K.shape))
    assert np.linalg.eigvalsh(K)[0] > 0


@pytest.mark.parametrize('kwargs', [
    dict(param_mode='bad'), dict(initial_stochastic_rate=-1), dict(initial_stochastic_rate=np.nan),
    dict(initial_stochastic_rate=np.inf), dict(initial_stochastic_rate=1j),
    dict(initial_stochastic_rate=[1.]), dict(initial_stochastic_rate='bad'),
    dict(completion_normalization='bad'),
    dict(param_mode='cholesky completion', initial_stochastic_rate=0),
])
def test_factory_rejects_invalid_chart_initialization(kwargs):
    from pygsti.baseobjs import QubitSpace
    from pygsti.modelmembers.operations import LindbladErrorgen
    with pytest.raises(ValueError):
        LindbladErrorgen.from_weight_limited(QubitSpace(1), **kwargs)


def test_model_serialization_operation_derivatives_and_torch():
    from pygsti.baseobjs import QubitSpace
    from pygsti.models import ExplicitOpModel
    from pygsti.modelmembers.operations import LindbladErrorgen, ExpErrorgenOp, ComposedOp, StaticUnitaryOp
    from pygsti.modelmembers.states import ComputationalBasisState, ComposedState
    from pygsti.modelmembers.povms import ComputationalBasisPOVM, ComposedPOVM
    blk = cycle_block('mixed')
    vector = np.random.default_rng(3).normal(scale=.1, size=blk.num_params)
    vector[:4] -= 4.
    blk.from_vector(vector)
    space = QubitSpace(2)
    errgen = LindbladErrorgen([blk], mx_basis='pp', state_space=space)
    channel = ExpErrorgenOp(errgen)
    ideal = StaticUnitaryOp(np.kron([[0., 1.], [1., 0.]], np.eye(2)), state_space=space)
    model = ExplicitOpModel(space, basis='pp')
    spam_channels = []
    for scale in (.6, .4):
        spam = errgen.copy()
        spam_vector = vector.copy()
        spam_vector[:4] += .5 * np.log(scale)
        spam.from_vector(spam_vector)
        spam_channels.append(ExpErrorgenOp(spam))
    model.preps['rho0'] = ComposedState(ComputationalBasisState([0, 0], state_space=space), spam_channels[0])
    model.povms['Mdefault'] = ComposedPOVM(
        spam_channels[1], ComputationalBasisPOVM(2, state_space=space), mx_basis='pp')
    model.operations['Gx'] = ComposedOp([ideal, channel])
    assert model.num_params == 3 * errgen.num_params
    restored = ExplicitOpModel.from_nice_serialization(model.to_nice_serialization())
    model_vector = model.to_vector()
    np.testing.assert_allclose(restored.to_vector(), model_vector)
    for v in (model_vector, model_vector + .01):
        model.from_vector(v)
        restored.from_vector(v)
        np.testing.assert_allclose(restored.operations['Gx'].to_dense(), model.operations['Gx'].to_dense())
        assert sum(model.probabilities(('Gx', 'Gx')).values()) == pytest.approx(1.)
        assert restored.probabilities(('Gx', 'Gx')) == pytest.approx(model.probabilities(('Gx', 'Gx')))
        for member in (restored.preps['rho0'], restored.povms['Mdefault']):
            restored_block = member.error_map.errorgen.coefficient_blocks[0]
            assert restored_block._param_mode == 'cholesky completion'
            assert restored_block._eeg_labels == blk._eeg_labels
    def dense(v):
        channel.from_vector(v)
        return channel.to_dense().copy().ravel()
    channel.from_vector(vector)
    np.testing.assert_allclose(channel.deriv_wrt_params(), fd_jac(dense, vector), atol=1e-9)
    errgen.from_vector(vector)
    expected_hessian = fd_jac(lambda v: (errgen.from_vector(v), errgen.deriv_wrt_params())[1], vector)
    errgen.from_vector(vector)
    np.testing.assert_allclose(errgen.hessian_wrt_params(), expected_hessian, atol=1e-9)
    torch = pytest.importorskip('torch')
    for obj in (errgen, channel):
        obj.from_vector(vector)
        sd = obj.stateless_data(torch.float64, 'cpu')
        func = lambda v: type(obj).torch_base(sd, v).ravel()
        t = torch.tensor(vector, dtype=torch.float64)
        np.testing.assert_allclose(func(t).numpy(), obj.to_dense().ravel(), atol=1e-12)
        np.testing.assert_allclose(torch.func.jacrev(func)(t).numpy(), obj.deriv_wrt_params(), atol=1e-10)


def test_small_noise_scale_invariance_and_least_squares_fit():
    from scipy.optimize import least_squares
    blk = cycle_block()
    p = blk._parameterization
    vector = np.random.default_rng(12).normal(scale=.1, size=blk.num_params)
    base, jacobian, _ = p._evaluate(blk, vector, order=1)
    for rate in (1e-6, 1e-12, 1e-24):
        scaled = vector.copy()
        scaled[:4] += .5 * np.log(rate)
        data, jac, _ = p._evaluate(blk, scaled, order=1)
        np.testing.assert_allclose(data / rate, base, atol=1e-13)
        np.testing.assert_allclose(jac / rate, jacobian, atol=1e-13)
    blk.from_vector(vector)
    K = blk.kossakowski_matrix()
    diagonal = np.array([1., 1e-4, 1., 1.])
    congruent = vector.copy()
    congruent[:4] += np.log(diagonal)
    blk.from_vector(congruent)
    np.testing.assert_allclose(blk.kossakowski_matrix(), diagonal[:, None] * K * diagonal[None, :], atol=1e-14)
    np.testing.assert_allclose(blk.to_vector(), congruent, atol=1e-12)
    target = vector.copy()
    target[:4] += .5 * np.log(1e-3)
    rates = p._evaluate(blk, target)[0]
    initial = np.zeros(blk.num_params)
    initial[:4] = .5 * np.log(1e-3)
    result = least_squares(lambda v: (p._evaluate(blk, v)[0] - rates) / 1e-3, initial,
                           jac=lambda v: blk.deriv_wrt_params(v) / 1e-3,
                           max_nfev=50, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    assert result.success
    assert np.linalg.norm(result.fun) < 1e-9
    np.testing.assert_allclose(result.x, target, atol=1e-9)


@pytest.mark.parametrize('basis_name,dimension,mx_basis', [('PP', 16, 'pp'), ('pp', 16, 'pp'), ('gm', 9, 'gm')])
def test_direct_sparse_superoperators_match_dense_reference(basis_name, dimension, mx_basis):
    basis = Basis.cast(basis_name, dimension)
    bels = basis.labels[1:5]
    labels = [LEEL('S', (b,)) for b in bels]
    for a, b in combinations(bels, 2):
        labels.extend(LEEL(typ, (a, b)) for typ in ('C', 'A'))
    blk = LCB('other_sparse', basis, param_mode='cholesky completion', error_generator_labels=labels)
    expected = blk.create_lindblad_term_superoperators(mx_basis, sparse=False, flat=True)
    actual = blk.create_lindblad_term_superoperators(mx_basis, sparse=True, flat=True)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a.toarray(), b, atol=1e-14)


def test_sparse_representation_matches_dense_operation_and_derivative():
    from pygsti.baseobjs import QubitSpace
    from pygsti.evotypes import Evotype
    from pygsti.modelmembers.operations import LindbladErrorgen, ExpErrorgenOp
    vector = np.random.default_rng(11).normal(scale=.1, size=12)
    vector[:4] -= 4.
    space = QubitSpace(2)
    dense = LindbladErrorgen([cycle_block()], state_space=space, evotype=Evotype('densitymx', True))
    sparse = LindbladErrorgen([cycle_block()], state_space=space, evotype=Evotype('densitymx', False))
    dense.from_vector(vector)
    sparse.from_vector(vector)
    assert sparse._rep_type == 'sparse superop'
    np.testing.assert_allclose(sparse.to_sparse().toarray(), dense.to_dense(), atol=1e-14)
    np.testing.assert_allclose(ExpErrorgenOp(sparse).to_dense(), ExpErrorgenOp(dense).to_dense(), atol=1e-13)
    with pytest.warns(UserWarning, match='finite differencing'):
        np.testing.assert_allclose(sparse.deriv_wrt_params(), dense.deriv_wrt_params(), atol=1e-8)
