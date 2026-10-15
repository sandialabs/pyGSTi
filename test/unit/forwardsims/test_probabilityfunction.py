"""Probability-map contracts: ordering, state ownership, and native derivatives."""
import numpy as np
import pytest

from pygsti.circuits import Circuit
from pygsti.forwardsims.probabilityfunction import probability_function
from pygsti.modelmembers.instruments import TPInstrument
from pygsti.modelpacks import smq1Q_XYI


def fixture():
    model = smq1Q_XYI.target_model('static')
    model.sim = 'matrix'  # compare analytic derivatives, not Map's forward differences
    model.instruments[('Iz', 0)] = TPInstrument({'p0': .4 * np.eye(4), 'p1': .6 * np.eye(4)})
    circuits = [Circuit([('Iz', 0)], line_labels=(0,)),
                Circuit([('Gxpi2', 0), ('Iz', 0), ('Gypi2', 0), ('Iz', 0)], line_labels=(0,))]
    layout = model.sim.create_layout(circuits, array_types=('e', 'ep'))
    return model, circuits, layout


def test_numpy_order_products_and_state_restoration():
    model, circuits, layout = fixture()
    function = probability_function(model, layout)
    entry = model.to_vector().copy()
    x = entry + np.random.default_rng(4).normal(scale=1e-4, size=entry.size)
    direction = np.random.default_rng(5).normal(size=x.size)
    probabilities = function(x)
    jacobian = function.jacobian(x)
    np.testing.assert_array_equal(model.to_vector(), entry)
    eps = 1e-6
    np.testing.assert_allclose(function.jvp(x, direction),
                               (function(x + eps * direction) - function(x - eps * direction)) / (2 * eps),
                               atol=1e-8, rtol=1e-5)
    cotangent = np.arange(function.num_outcomes, dtype=float)
    np.testing.assert_allclose(function.vjp(x, cotangent), jacobian.T @ cotangent)
    np.testing.assert_allclose(cotangent @ function.jvp(x, direction),
                               function.vjp(x, cotangent) @ direction)
    model.from_vector(x)
    try:
        for row in function.outcomes:
            expected = model.probabilities(row.circuit)
            np.testing.assert_allclose(probabilities[list(row.indices)],
                                       [expected[o] for o in row.outcomes])
    finally:
        model.from_vector(entry)
    saved = probabilities.copy()
    function(entry)
    np.testing.assert_array_equal(probabilities, saved)
    assert isinstance(function.outcomes, tuple)
    assert len(function.outcomes[1].outcomes) == 8  # two binary instruments and binary POVM


def test_numpy_restores_after_simulation_failure(monkeypatch):
    model, circuits, layout = fixture()
    function = probability_function(model, layout)
    entry = model.to_vector().copy()
    def fail(*args, **kwargs):
        raise RuntimeError('deliberate simulator failure')
    monkeypatch.setattr(model.sim, 'bulk_fill_probs', fail)
    with pytest.raises(RuntimeError, match='deliberate'):
        function(entry + 1e-4)
    np.testing.assert_array_equal(model.to_vector(), entry)


def test_derivatives_require_preplanned_storage():
    model, circuits, _ = fixture()
    layout = model.sim.create_layout(circuits, array_types=('e',))
    function = probability_function(model, layout)
    with pytest.raises(ValueError, match='layout created'):
        function.jacobian(model.to_vector())


def test_torch_native_graph_and_products():
    torch = pytest.importorskip('torch')
    model, _, layout = fixture()
    entry = model.to_vector().copy()
    numpy_function = probability_function(model, layout)
    function = probability_function(model, layout, backend='torch')
    x = torch.tensor(entry + 1e-4, dtype=torch.float64, requires_grad=True)
    direction = torch.linspace(-.2, .3, x.numel(), dtype=x.dtype)
    cotangent = torch.linspace(.1, .9, function.num_outcomes, dtype=x.dtype)
    probabilities = function(x)
    assert probabilities.requires_grad
    np.testing.assert_allclose(probabilities.detach().numpy(), numpy_function(x.detach().numpy()), atol=1e-12)
    jacobian = function.jacobian(x)
    np.testing.assert_allclose(jacobian.detach().numpy(), numpy_function.jacobian(x.detach().numpy()), atol=1e-10)
    torch.testing.assert_close(function.jvp(x, direction), jacobian @ direction)
    torch.testing.assert_close(function.vjp(x, cotangent), jacobian.T @ cotangent)
    gradient, = torch.autograd.grad(probabilities @ cotangent, x)
    torch.testing.assert_close(gradient, function.vjp(x, cotangent))
    np.testing.assert_array_equal(model.to_vector(), entry)


def test_torch_uses_actual_parameter_index_maps(monkeypatch):
    torch = pytest.importorskip('torch')
    model, _, layout = fixture()
    original_function = probability_function(model, layout, backend='torch')
    member = next(obj for _, obj in model._iter_parameterized_objs() if obj.num_params)
    actual = member.gpindices_as_array().copy()
    reverse = actual[::-1].copy()
    monkeypatch.setattr(member, 'gpindices_as_array', lambda: reverse)
    function = probability_function(model, layout, backend='torch')
    x = torch.arange(model.num_params, dtype=torch.float64) * 1e-4
    # Reversing the member map is equivalent to permuting those flat coordinates
    # before evaluating the unmodified function. This detects contiguous slicing.
    permuted = x.clone()
    permuted[torch.as_tensor(actual)] = x[torch.as_tensor(reverse)]
    torch.testing.assert_close(function(x), original_function(permuted))


def test_backend_validation():
    model, _, layout = fixture()
    with pytest.raises(ValueError, match='backend'):
        probability_function(model, layout, backend='unknown')
    with pytest.raises(ValueError, match='real floating'):
        probability_function(model, layout, dtype=complex)


def test_torch_accepts_non_torchable_static_reference_frame():
    torch = pytest.importorskip('torch')
    from pygsti.modelmembers.torchable import Torchable
    model, _, layout = fixture()
    constants = [member for _, member in model._iter_parameterized_objs()
                 if member.num_params == 0 and not isinstance(member, Torchable)]
    assert constants  # StaticState and/or the fixed POVM exercise the adapter.
    function = probability_function(model, layout, backend='torch')
    reference = probability_function(model, layout)
    x = torch.tensor(model.to_vector(), dtype=torch.float64, requires_grad=True)
    np.testing.assert_allclose(function(x).detach().numpy(), reference(x.detach().numpy()), atol=1e-12)
    gradient, = torch.autograd.grad(function(x).square().sum(), x)
    assert torch.isfinite(gradient).all()


def test_torch_preserves_reversed_observed_subset_order():
    torch = pytest.importorskip('torch')
    from pygsti.layouts.copalayout import CircuitOutcomeProbabilityArrayLayout
    model, circuits, _ = fixture()
    # Model the arbitrary observed order supplied by a dataset, including omitted
    # outcomes in the repeated-instrument circuit. No implicit completion occurs.
    elements = {}
    offset = 0
    for i, circuit in enumerate(circuits):
        observed = tuple(model.circuit_outcomes(circuit))[::-1][::2]
        elements[i] = tuple((offset + j, outcome) for j, outcome in enumerate(observed))
        offset += len(observed)
    layout = CircuitOutcomeProbabilityArrayLayout(circuits, circuits, {0: 0, 1: 1}, elements,
                                                  [model.complete_circuit(c) for c in circuits])
    function = probability_function(model, layout, backend='torch')
    x = torch.tensor(model.to_vector(), dtype=torch.float64, requires_grad=True)
    probabilities = function(x)
    for row in function.outcomes:
        expected = model.probabilities(row.circuit)
        np.testing.assert_allclose(probabilities.detach().numpy()[list(row.indices)],
                                   [expected[outcome] for outcome in row.outcomes], atol=1e-12)
    direction = torch.linspace(-.2, .3, x.numel(), dtype=x.dtype)
    eps = 1e-6
    torch.testing.assert_close(function.jvp(x, direction),
                               (function(x + eps * direction) - function(x - eps * direction)) / (2 * eps),
                               atol=1e-8, rtol=1e-5)
