"""Sparse S/C/A blocks beyond the fill-free chart: factories, persistence, Torch and sparse representations."""

from itertools import combinations

import numpy as np
import pytest

from pygsti.baseobjs import Basis
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL
from pygsti.modelmembers.operations.lindbladcoefficients import LindbladCoefficientBlock as LCB


def cycle_block(components='complex', mode='contraction'):
    """S on four directions of two qubits, with C and/or A pair terms on a (nonchordal) four-cycle."""
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
    return LCB('other_sparse', basis, param_mode=mode, error_generator_labels=labels)


def fd_jac(function, vector, eps=1e-6):
    return np.stack([(function(vector + eps * e) - function(vector - eps * e)) / (2 * eps)
                     for e in np.eye(len(vector))], axis=-1)


def test_auto_selects_contraction_for_a_cycle_and_completion_is_gone():
    basis = Basis.cast('PP', 16)
    labels = cycle_block()._eeg_labels
    assert LCB('other_sparse', basis, param_mode='auto', error_generator_labels=labels)._param_mode == 'contraction'
    with pytest.raises(ValueError, match='needs fill'):
        LCB('other_sparse', basis, param_mode='cholesky', error_generator_labels=labels)
    with pytest.raises(ValueError):
        LCB('other_sparse', basis, param_mode='cholesky completion', error_generator_labels=labels)


@pytest.mark.parametrize('components', ['complex', 'mixed'])
def test_torch_values_derivatives_and_hessian(components):
    torch = pytest.importorskip('torch')
    blk = cycle_block(components)
    vector = np.random.default_rng(9).normal(scale=.15, size=blk.num_params)
    blk.from_vector(vector)
    sd = blk.stateless_data(torch.float64, 'cpu')
    func = lambda v: LCB.torch_base(sd, v)
    v = torch.tensor(vector, dtype=torch.float64)
    np.testing.assert_allclose(func(v).numpy(), blk.block_data, atol=1e-13)
    np.testing.assert_allclose(torch.func.jacrev(func)(v).numpy(), blk.deriv_wrt_params(vector), atol=1e-12)
    np.testing.assert_allclose(torch.func.hessian(func)(v).numpy(),
                               blk._parameterization._evaluate(blk, vector, order=2)[2], atol=1e-10)


def test_four_qubit_weight_three_full_block():
    basis = Basis.cast('PP', 256)
    bels = [b for b in basis.labels[1:] if sum(c != 'I' for c in b) <= 3]
    labels = [LEEL('S', (b,)) for b in bels]
    for a, b in combinations(bels, 2):
        if sum(x != 'I' or y != 'I' for x, y in zip(a, b)) <= 3:
            labels.extend([LEEL('C', (a, b)), LEEL('A', (a, b))])
    blk = LCB('other_sparse', basis, param_mode='auto', error_generator_labels=labels)
    assert blk._param_mode == 'contraction'
    assert len(bels) == 174
    assert blk.num_params == 14562
    vector = np.random.default_rng(4).normal(scale=.02, size=blk.num_params)
    blk.from_vector(vector)
    K = blk.kossakowski_matrix()
    assert np.linalg.eigvalsh(K)[0] >= -1e-14 * np.abs(K).max()
    data = blk.block_data.copy()
    blk.from_vector(blk.to_vector())
    np.testing.assert_allclose(blk.block_data, data, rtol=1e-10, atol=1e-16)
    allowed = np.eye(len(bels), dtype=bool)
    for i, j in blk._coefficient_pairs:
        allowed[i, j] = allowed[j, i] = True
    np.testing.assert_array_equal(K[~allowed], 0)


@pytest.mark.parametrize('mode', ['cholesky', 'contraction'])
def test_factory_initialization_and_zero_default(mode):
    from pygsti.baseobjs import QubitSpace
    from pygsti.modelmembers.operations import LindbladErrorgen
    space = QubitSpace(1)
    errgen = LindbladErrorgen.from_weight_limited(space, param_mode=mode, initial_stochastic_rate=.002)
    block = errgen.coefficient_blocks[-1]
    assert block._param_mode == mode
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
    assert block._param_mode == 'contraction'
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
    assert block._param_mode == 'contraction'
    np.testing.assert_array_equal(errgen.to_vector(), np.zeros(errgen.num_params))
    vector = errgen.to_vector() + np.random.default_rng(2).normal(scale=.02, size=errgen.num_params)
    errgen.from_vector(vector)
    K = block.kossakowski_matrix()
    np.testing.assert_array_equal(K.real - np.diag(K.diagonal().real), np.zeros(K.shape))
    assert np.linalg.eigvalsh(K)[0] >= -1e-15 * np.abs(K).max()


@pytest.mark.parametrize('kwargs', [
    dict(param_mode='bad'), dict(param_mode='cholesky completion'), dict(initial_stochastic_rate=-1),
    dict(initial_stochastic_rate=np.nan), dict(initial_stochastic_rate=np.inf), dict(initial_stochastic_rate=1j),
    dict(initial_stochastic_rate=[1.]), dict(initial_stochastic_rate='bad'),
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
    vector = np.random.default_rng(3).normal(scale=.3, size=blk.num_params)
    vector[:4] *= .3  # S rates of order 0.01
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
        spam_vector[:4] *= np.sqrt(scale)
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
            assert restored_block._param_mode == 'contraction'
            assert restored_block._eeg_labels == blk._eeg_labels

    def dense(v):
        channel.from_vector(v)
        return channel.to_dense().copy().ravel()
    channel.from_vector(vector)
    np.testing.assert_allclose(channel.deriv_wrt_params(), fd_jac(dense, vector), atol=1e-9)
    errgen.from_vector(vector)
    expected_hessian = fd_jac(lambda v: (errgen.from_vector(v), errgen.deriv_wrt_params())[1], vector)
    errgen.from_vector(vector)
    np.testing.assert_allclose(errgen.hessian_wrt_params(), expected_hessian, atol=1e-8)
    torch = pytest.importorskip('torch')
    for obj in (errgen, channel):
        obj.from_vector(vector)
        sd = obj.stateless_data(torch.float64, 'cpu')
        func = lambda v: type(obj).torch_base(sd, v).ravel()
        t = torch.tensor(vector, dtype=torch.float64)
        np.testing.assert_allclose(func(t).numpy(), obj.to_dense().ravel(), atol=1e-12)
        np.testing.assert_allclose(torch.func.jacrev(func)(t).numpy(), obj.deriv_wrt_params(), atol=1e-10)


@pytest.mark.parametrize('basis_name,dimension,mx_basis', [('PP', 16, 'pp'), ('pp', 16, 'pp'), ('gm', 9, 'gm')])
def test_direct_sparse_superoperators_match_dense_reference(basis_name, dimension, mx_basis):
    basis = Basis.cast(basis_name, dimension)
    bels = basis.labels[1:5]
    labels = [LEEL('S', (b,)) for b in bels]
    for a, b in combinations(bels, 2):
        labels.extend(LEEL(typ, (a, b)) for typ in ('C', 'A'))
    blk = LCB('other_sparse', basis, param_mode='contraction', error_generator_labels=labels)
    expected = blk.create_lindblad_term_superoperators(mx_basis, sparse=False, flat=True)
    actual = blk.create_lindblad_term_superoperators(mx_basis, sparse=True, flat=True)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a.toarray(), b, atol=1e-14)


def test_sparse_representation_matches_dense_operation_and_derivative():
    from pygsti.baseobjs import QubitSpace
    from pygsti.evotypes import Evotype
    from pygsti.modelmembers.operations import LindbladErrorgen, ExpErrorgenOp
    vector = np.random.default_rng(11).normal(scale=.3, size=12)
    vector[:4] *= .3
    space = QubitSpace(2)
    dense = LindbladErrorgen([cycle_block()], state_space=space, evotype=Evotype('densitymx', True))
    sparse = LindbladErrorgen([cycle_block()], state_space=space, evotype=Evotype('densitymx', False))
    dense.from_vector(vector)
    sparse.from_vector(vector)
    assert sparse._rep_type == 'sparse superop'
    np.testing.assert_allclose(sparse.to_sparse().toarray(), dense.to_dense(), atol=1e-14)
    np.testing.assert_allclose(ExpErrorgenOp(sparse).to_dense(), ExpErrorgenOp(dense).to_dense(), atol=1e-13)
    with pytest.warns(UserWarning, match='finite differencing'):
        # The sparse representation differentiates by finite differences; its truncation error
        # here is about 2e-7 against the analytic dense derivative.
        np.testing.assert_allclose(sparse.deriv_wrt_params(), dense.deriv_wrt_params(), atol=1e-6)
