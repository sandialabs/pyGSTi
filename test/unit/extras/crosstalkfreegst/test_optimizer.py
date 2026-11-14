"""The Levenberg-Marquardt routine on small least-squares problems with known answers."""
import numpy as np
import pytest
import scipy.sparse as sps

from pygsti.extras.crosstalkfreegst.fit import _solve, levenberg_marquardt


class LeastSquares(object):
    """``f(x) = |r(x)|^2`` through the interface levenberg_marquardt minimizes (gradient returned as ``J^T r``)."""

    def __init__(self, residuals, jacobian):
        self.residuals, self.jacobian = residuals, jacobian
        self.num_fn = self.num_derivatives = 0

    def fn(self, x):
        self.num_fn += 1
        r = self.residuals(x)
        return float(r @ r)

    def fn_grad_gn(self, x):
        self.num_derivatives += 1
        r, J = self.residuals(x), self.jacobian(x)
        return float(r @ r), J.T @ r, sps.csc_matrix(J.T @ J)


def rosenbrock():
    return LeastSquares(lambda x: np.array([10 * (x[1] - x[0]**2), 1 - x[0]]),
                        lambda x: np.array([[-20 * x[0], 10.0], [-1.0, 0.0]]))


def exponential_fit():
    """Residuals of a * exp(-b t) against data from a = 2, b = 0.7 plus fixed offsets: a nonzero minimum."""
    t = np.linspace(0, 4, 9)
    data = 2 * np.exp(-0.7 * t) + 0.05 * np.cos(5 * t)
    return LeastSquares(lambda x: x[0] * np.exp(-x[1] * t) - data,
                        lambda x: np.stack([np.exp(-x[1] * t), -x[0] * t * np.exp(-x[1] * t)], axis=1))


@pytest.mark.parametrize('damping', ['identity', 'marquardt'])
def test_rosenbrock_is_minimized(damping):
    objective = rosenbrock()
    result = levenberg_marquardt(objective, [-1.2, 1.0], damping=damping, rel_ftol=1e-12)
    assert result.converged
    assert np.allclose(result.x, [1, 1], atol=1e-6)
    assert result.fn == pytest.approx(objective.fn(result.x), abs=1e-20) and result.fn < 1e-12
    assert 1 < result.num_iterations < 60


@pytest.mark.parametrize('damping', ['identity', 'marquardt'])
def test_minimum_with_nonzero_residuals_is_found(damping):
    objective = exponential_fit()
    result = levenberg_marquardt(objective, [1.0, 0.2], damping=damping, rel_ftol=1e-10)
    assert result.converged and result.message == "Predicted remaining decrease below tolerance"
    _, g, _ = objective.fn_grad_gn(result.x)
    assert np.max(np.abs(g)) < 1e-5           # a stationary point ...
    assert 0 < result.fn < objective.fn(np.array([2.0, 0.7]))   # ... below the value at the generating parameters


@pytest.mark.parametrize('damping', ['identity', 'marquardt'])
def test_flat_directions_are_left_alone(damping):
    """A linear problem whose second and fourth parameters do not enter (like gauge directions)."""
    A = np.array([[1.0, 0, 2, 0], [0, 0, 1, 0], [3, 0, 1, 0]])
    b = np.array([1.0, 2, 3])
    objective = LeastSquares(lambda x: A @ x - b, lambda x: A)
    x0 = np.array([0.5, 7.0, -0.5, -3.0])
    result = levenberg_marquardt(objective, x0, damping=damping, rel_ftol=1e-12)
    best = np.linalg.lstsq(A[:, [0, 2]], b, rcond=None)[0]
    assert result.converged
    assert np.allclose(result.x[[0, 2]], best, atol=1e-5)
    assert np.allclose(result.x[[1, 3]], x0[[1, 3]], atol=1e-12)   # no gradient, so no movement
    assert result.fn == pytest.approx(float(np.sum((A[:, [0, 2]] @ best - b)**2)), rel=1e-8)


def test_objective_never_increases_and_rejected_steps_raise_the_damping():
    """From a start where the first Gauss-Newton steps overshoot, steps are rejected and then shortened."""
    objective = rosenbrock()
    values = []
    fn = objective.fn
    objective.fn = lambda x: values.append(fn(x)) or values[-1]
    result = levenberg_marquardt(objective, [-1.2, 1.0], damping='identity', init_damping=1e-12, rel_ftol=1e-12)
    start = fn(np.array([-1.2, 1.0]))
    assert max(values) > start                      # some trial steps were worse than the start ...
    assert result.converged and result.fn < 1e-12   # ... and were not taken
    # an accepted step is followed by one derivative evaluation: rejected steps cost none
    accepted = objective.num_derivatives - 1
    assert accepted < len(values)
    assert result.num_iterations >= len(values)


def test_iteration_limit_is_reported():
    objective = rosenbrock()
    result = levenberg_marquardt(objective, [-1.2, 1.0], max_iterations=3)
    assert not result.converged and result.num_iterations == 3
    assert result.message == "Maximum number of iterations (3) reached"
    assert result.fn == pytest.approx(objective.fn(result.x)) and result.fn <= objective.fn(np.array([-1.2, 1.0]))
    nothing = levenberg_marquardt(objective, [-1.2, 1.0], max_iterations=0)
    assert not nothing.converged and nothing.num_iterations == 0 and np.array_equal(nothing.x, [-1.2, 1.0])


def test_stopping_at_a_minimum_takes_no_step():
    objective = rosenbrock()
    result = levenberg_marquardt(objective, [1.0, 1.0])
    assert result.converged and result.message == "Gradient below tolerance"
    assert np.array_equal(result.x, [1.0, 1.0]) and objective.num_fn == 0


def test_stalled_objective_stops_the_run():
    """A valley floor that falls slowly: the quadratic model keeps predicting gains that do not arrive."""
    objective = LeastSquares(lambda x: np.array([1.0 + x[0]**4]), lambda x: np.array([[4 * x[0]**3]]))
    result = levenberg_marquardt(objective, [0.3], rel_ftol=1e-4, stat_tol=0.0)
    assert result.converged and result.message == "Objective stalled over 10 iterations"
    result = levenberg_marquardt(objective, [0.3], rel_ftol=1e-30, abs_gtol=0, rel_xtol=0, stat_tol=0.01)
    assert result.converged and result.message == "Objective change over 10 iterations statistically negligible"


def test_unknown_damping_is_rejected():
    with pytest.raises(ValueError, match="Unknown damping 'levenberg'"):
        levenberg_marquardt(rosenbrock(), [-1.2, 1.0], damping='levenberg')


def test_solve_dense_and_sparse_agree():
    rng = np.random.default_rng(0)
    J = rng.normal(size=(30, 12))
    J[:, 5] = 0                                    # a flat direction: singular without damping
    jtj, rhs = sps.csc_matrix(J.T @ J), rng.normal(size=12)
    damping = np.full(12, 1e-3)
    expected = np.linalg.solve(J.T @ J + 1e-3 * np.identity(12), rhs)
    assert np.allclose(_solve(jtj, damping, rhs), expected, rtol=1e-10)
    assert np.allclose(_solve(jtj, damping, rhs, dense_threshold=0), expected, rtol=1e-10)
    varying = np.linspace(1e-3, 2e-3, 12)          # the damping is a diagonal, not a multiple of the identity
    expected = np.linalg.solve(J.T @ J + np.diag(varying), rhs)
    assert np.allclose(_solve(jtj, varying, rhs), expected, rtol=1e-10)
    assert np.allclose(_solve(jtj, varying, rhs, dense_threshold=0), expected, rtol=1e-10)
