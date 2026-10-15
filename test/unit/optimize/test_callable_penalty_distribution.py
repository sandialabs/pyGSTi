"""Callable penalty derivatives respect distributed columns and model ownership."""
from types import SimpleNamespace
import numpy as np
import pytest

from pygsti.layouts.distlayout import DistributableCOPALayout
from pygsti.objectivefns.objectivefns import TimeIndependentMDCObjectiveFunction


class AliasingModel:
    """Match the important model behavior: to_vector may alias mutable state."""
    def __init__(self, point):
        self.point = np.array(point, copy=True)

    def to_vector(self):
        return self.point

    def from_vector(self, point):
        self.point[:] = point


def context(model, penalty, parameter_slice):
    if parameter_slice is None:
        layout = object()
    else:
        # Only this real layout type's public ownership slice is used by the
        # method under test. No MPI communicator or numerical layout is needed.
        layout = DistributableCOPALayout.__new__(DistributableCOPALayout)
        layout.global_param_slice = parameter_slice
    return SimpleNamespace(model=model, layout=layout, forcefn_grad=None,
                           regularize_factor=0., cptp_penalty_factor=0.,
                           spam_penalty_factor=0., errorgen_penalty_factor=0.,
                           callable_penalty=penalty, callable_penalty_factor=3.5,
                           local_ex=1)


@pytest.mark.parametrize('parameter_slice', [None, slice(0, 6), slice(2, 5)])
def test_scaled_quadratic_penalty_has_only_owned_derivative_columns(parameter_slice):
    entry = np.array([.17, -.3, .5, -.8, 1.1, -1.4])
    weights = np.arange(1., 7.)
    model = AliasingModel(entry)
    seen = []

    def penalty(m):
        seen.append(m.to_vector().copy())
        return float(weights @ m.to_vector() ** 2)

    objective = context(model, penalty, parameter_slice)
    owned = np.arange(entry.size)[slice(None) if parameter_slice is None else parameter_slice]
    jacobian = np.full((1, owned.size), np.nan)
    TimeIndependentMDCObjectiveFunction._dterms_fill_penalty(objective, entry.copy(), jacobian)
    # Forward differences have known O(eps) truncation error for this quadratic.
    np.testing.assert_allclose(jacobian[0], 3.5 * 2 * weights[owned] * entry[owned],
                               atol=3e-6, rtol=1e-7)
    np.testing.assert_array_equal(model.to_vector(), entry)
    np.testing.assert_array_equal(seen[0], entry)
    assert len(seen) == 1 + owned.size
    for point, coordinate in zip(seen[1:], owned):
        changed = np.flatnonzero(point != entry)
        np.testing.assert_array_equal(changed, [coordinate])
        assert point[coordinate] > entry[coordinate]


def test_penalty_exception_restores_exact_entry_vector():
    entry = np.array([.17, -.3, .5, -.8, 1.1, -1.4])
    model = AliasingModel(entry)
    calls = 0

    def penalty(m):
        nonlocal calls
        calls += 1
        if calls == 3:  # Fail after one successful derivative and another perturbation.
            raise RuntimeError('penalty failed at trial point')
        return float(m.to_vector() @ m.to_vector())

    objective = context(model, penalty, slice(2, 5))
    with pytest.raises(RuntimeError, match='penalty failed at trial point'):
        TimeIndependentMDCObjectiveFunction._dterms_fill_penalty(objective, entry.copy(), np.zeros((1, 3)))
    np.testing.assert_array_equal(model.to_vector(), entry)


def test_zero_penalty_factor_skips_evaluation_and_leaves_zero_local_columns():
    entry = np.arange(6., dtype=float)
    model = AliasingModel(entry)

    def forbidden(m):
        raise AssertionError('A disabled callable penalty must not run')

    objective = context(model, forbidden, slice(2, 5))
    objective.callable_penalty_factor = 0.
    jacobian = np.full((1, 3), np.nan)
    TimeIndependentMDCObjectiveFunction._dterms_fill_penalty(objective, entry.copy(), jacobian)
    np.testing.assert_array_equal(jacobian, np.zeros((1, 3)))
    np.testing.assert_array_equal(model.to_vector(), entry)
