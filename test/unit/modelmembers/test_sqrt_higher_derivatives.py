"""Implicit square-root AD: repeated spectra, complex adjoints, and boundary policy."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from pygsti.circuits import Circuit
from pygsti.modelmembers.instruments import Instrument
from pygsti.modelmembers.operations.cptrop import _EffectSqrt, _HermitianSylvesterSolve
from pygsti.modelpacks import smq1Q_XYI, smq2Q_XYICNOT


def _hermitian(matrix):
    return (matrix + matrix.mH) / 2


def _sqrt(matrix, tolerance=0.):
    return _EffectSqrt.apply(matrix, tolerance)[0]


@pytest.mark.parametrize('dimension', [2, 4])
@pytest.mark.parametrize('complex_case', [False, True])
@pytest.mark.parametrize('repeated', [False, True])
def test_interior_square_root_first_and_second_derivatives(dimension, complex_case, repeated):
    dtype = torch.complex128 if complex_case else torch.float64
    if repeated:
        matrix = .7 * torch.eye(dimension, dtype=dtype)
    else:
        generator = torch.Generator().manual_seed(207 + dimension)
        raw = torch.randn((dimension, dimension), dtype=dtype, generator=generator)
        gram = raw @ raw.mH
        matrix = .4 * gram / torch.linalg.norm(gram) + .3 * torch.eye(dimension, dtype=dtype)
    matrix.requires_grad_()
    assert torch.autograd.gradcheck(_sqrt, (matrix,), check_forward_ad=True, atol=2e-5, rtol=2e-4)
    assert torch.autograd.gradgradcheck(_sqrt, (matrix,), atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize('complex_case', [False, True])
def test_uncached_sylvester_coefficient_and_rhs_derivatives(complex_case):
    dtype = torch.complex128 if complex_case else torch.float64
    generator = torch.Generator().manual_seed(123)
    root = (.7 * torch.eye(4, dtype=dtype)).requires_grad_()
    rhs = _hermitian(torch.randn((4, 4), dtype=dtype, generator=generator)).requires_grad_()
    solve = lambda r, h: _HermitianSylvesterSolve.apply(r, h, None, None)
    assert torch.autograd.gradcheck(solve, (root, rhs), check_forward_ad=True, atol=2e-5, rtol=2e-4)
    assert torch.autograd.gradgradcheck(solve, (root, rhs), atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize('spectrum,tolerance', [([0., 0., 0., 0.], 0.),
                                               ([0., .4, .4, 1.], 0.),
                                               ([.01, .02, .3, .8], .05)])
@pytest.mark.parametrize('complex_case', [False, True])
def test_minimum_norm_first_derivatives_preserve_zero_and_cutoff_policy(spectrum, tolerance, complex_case):
    dtype = torch.complex128 if complex_case else torch.float64
    matrix = torch.diag(torch.tensor(spectrum, dtype=dtype)).requires_grad_()
    generator = torch.Generator().manual_seed(99)
    direction = _hermitian(torch.randn((4, 4), dtype=dtype, generator=generator))
    cotangent = _hermitian(torch.randn((4, 4), dtype=dtype, generator=generator))
    function = lambda e: _sqrt(e, tolerance)
    _, vectors, values = _EffectSqrt.apply(matrix, tolerance)
    # This is the established first-order divided-difference convention. A
    # positive cutoff can deliberately differ from the true interior derivative.
    expected_jvp = _EffectSqrt._frechet(vectors, values, tolerance, direction)
    expected_vjp = _EffectSqrt._frechet(vectors, values, tolerance, cotangent)
    torch.testing.assert_close(torch.func.jvp(function, (matrix,), (direction,))[1], expected_jvp)
    # This uses reverse-over-reverse: the inner Sylvester backward must preserve
    # the same cached truncation to keep even this FIRST derivative unchanged.
    torch.testing.assert_close(torch.autograd.functional.jvp(function, matrix, direction)[1], expected_jvp)
    gradient, = torch.autograd.grad((function(matrix).conj() * cotangent).sum().real, matrix)
    torch.testing.assert_close(gradient, expected_vjp)


@pytest.mark.parametrize('complex_case', [False, True])
def test_repeated_spectrum_batched_values_gradients_and_hvps(complex_case):
    dtype = torch.complex128 if complex_case else torch.float64
    generator = torch.Generator().manual_seed(2087)
    base = .7 * torch.eye(4, dtype=dtype)
    matrices = torch.stack([base, base * 1.2, base * .8])
    directions = _hermitian(torch.randn((3, 4, 4), dtype=dtype, generator=generator))
    weight = _hermitian(torch.randn((4, 4), dtype=dtype, generator=generator))
    scalar = lambda e: (weight.conj() * _sqrt(e)).sum().real
    gradient = torch.func.grad(scalar)
    hvp = lambda e, d: torch.func.jvp(gradient, (e,), (d,))[1]
    torch.testing.assert_close(_sqrt(matrices), torch.vmap(_sqrt)(matrices))
    torch.testing.assert_close(torch.vmap(gradient)(matrices),
                               torch.stack([gradient(e) for e in matrices]))
    torch.testing.assert_close(torch.vmap(hvp)(matrices, directions),
                               torch.stack([hvp(e, d) for e, d in zip(matrices, directions)]),
                               atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('qubits', [1, 2])
def test_instrument_probability_hvp_matches_gradient_differences(qubits):
    pack = smq1Q_XYI if qubits == 1 else smq2Q_XYICNOT
    model = pack.target_model('static', simulator='matrix')
    parity = np.array([1.])
    for _ in range(qubits):
        parity = np.kron(parity, [1., -1.])
    effect = np.diag((1. + .6 * parity) / 2).astype(complex)
    label = ('Iz',) + tuple(range(qubits))
    model.instruments[label] = Instrument.from_effects(
        {'p0': effect, 'p1': np.eye(2**qubits) - effect}, model.basis)
    rng = np.random.default_rng(20261015 + qubits)
    model.from_vector(.02 * rng.standard_normal(model.num_params))
    circuits = [Circuit([('Gxpi2', 0), label, ('Gypi2', 0)], line_labels=tuple(range(qubits))),
                Circuit([('Gypi2', 0), label, ('Gxpi2', 0), label], line_labels=tuple(range(qubits)))]
    layout = model.sim.create_layout(circuits, array_types=('e', 'ep'))
    function = model.sim.probability_function(layout, backend='torch')
    x = torch.tensor(model.to_vector(), dtype=torch.float64, requires_grad=True)
    direction = torch.tensor(rng.standard_normal(x.numel()), dtype=x.dtype)
    direction /= torch.linalg.norm(direction)
    cotangent = torch.tensor(rng.standard_normal(function.num_outcomes), dtype=x.dtype)
    gradient, = torch.autograd.grad(function(x) @ cotangent, x, create_graph=True)
    actual, = torch.autograd.grad(gradient @ direction, x)
    epsilon = 1e-5
    expected = (function.vjp(x + epsilon*direction, cotangent)
                - function.vjp(x - epsilon*direction, cotangent)) / (2*epsilon)
    relative = torch.linalg.norm(actual-expected) / torch.linalg.norm(expected)
    assert relative.detach().item() < 1e-6
