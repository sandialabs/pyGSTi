"""Mathematical and ownership checks for the optimizer problem boundary."""
import unittest
import numpy as np

from pygsti.optimize.problem import OptimizationProblem


class ProblemTester(unittest.TestCase):
    def test_nonlinear_curvature_and_normalization(self):
        # Exact Hessian differs from Gauss–Newton away from a zero residual.
        problem = OptimizationProblem(
            [2.0], lambda x: (x[0] ** 2 - 1) ** 2,
            gradient=lambda x: np.array([4 * x[0] * (x[0] ** 2 - 1)]),
            residual=lambda x: np.array([x[0] ** 2 - 1]),
            residual_jacobian=lambda x: np.array([[2 * x[0]]]),
            hessian_vector_product=lambda x, v: (12 * x[0] ** 2 - 4) * v)
        x, v = np.array([2.0]), np.array([0.7])
        step = 1e-6
        finite_difference = (problem.gradient(x + step * v) - problem.gradient(x - step * v)) / (2 * step)
        np.testing.assert_allclose(problem.hessian_vector_product(x, v), finite_difference)
        np.testing.assert_allclose(problem.gauss_newton_vector_product(x, v), 32 * v)
        self.assertNotEqual(problem.hessian_vector_product(x, v)[0], 32 * v[0])
        self.assertEqual(problem.value(x), np.dot(problem.residual(x), problem.residual(x)))

    def test_product_only_adjoint_and_no_dense_fallback(self):
        matrix = np.array([[1., 2.], [-3., 4.], [5., -6.]])
        problem = OptimizationProblem(
            [0., 0.], lambda x: np.linalg.norm(matrix @ x) ** 2,
            residual=lambda x: matrix @ x,
            residual_jvp=lambda x, v: matrix @ v,
            residual_vjp=lambda x, w: matrix.T @ w)
        linearization = problem.linearize([1., 2.])
        v, w = np.array([0.3, -0.2]), np.array([1., -2., 3.])
        self.assertAlmostEqual(np.dot(w, linearization.jvp(v)), np.dot(linearization.vjp(w), v))
        self.assertEqual(linearization.storage, 'operator')
        with self.assertRaises(NotImplementedError):
            linearization.to_dense()
        with self.assertRaises(NotImplementedError):
            problem.gradient([1., 2.])
        with self.assertRaises(NotImplementedError):
            problem.hessian_vector_product([1., 2.], v)

    def test_captured_linearization_survives_reused_buffers_and_trials(self):
        residual_buffer, jacobian_buffer = np.zeros(1), np.zeros((1, 1))
        def residual(x):
            residual_buffer[:] = x ** 2 - 1
            return residual_buffer
        def jacobian(x):
            jacobian_buffer[:] = 2 * x
            return jacobian_buffer
        initial = np.array([2.])
        problem = OptimizationProblem(initial, lambda x: (x[0] ** 2 - 1) ** 2,
                                      residual=residual, residual_jacobian=jacobian)
        captured = problem.linearize(initial)
        initial[:] = 9
        problem.residual([10.])
        problem.residual_jacobian([10.])
        np.testing.assert_array_equal(captured.residual, [3.])
        np.testing.assert_array_equal(captured.jvp([1.]), [4.])
        point = captured.point
        point[:] = 20
        np.testing.assert_array_equal(captured.point, [2.])
        np.testing.assert_array_equal(problem.initial_point, [2.])

    def test_scalar_only_loss_and_explicit_missing_structure(self):
        problem = OptimizationProblem([1.], lambda x: -np.log(x[0]))
        self.assertAlmostEqual(problem.value([2.]), -np.log(2.))
        with self.assertRaises(NotImplementedError):
            problem.linearize([2.])
        with self.assertRaises(ValueError):
            problem.value([1., 2.])

    def test_bounds_owned_and_validated(self):
        lower, upper = np.array([0.]), np.array([3.])
        problem = OptimizationProblem([1.], lambda x: x[0], bounds=(lower, upper))
        lower[:] = -10
        returned = problem.bounds
        returned[0][:] = -20
        np.testing.assert_array_equal(problem.bounds[0], [0.])
        # The oracle is evaluated without silently clipping a solver's trial.
        self.assertEqual(problem.value([-2.]), -2.)
        with self.assertRaises(ValueError):
            OptimizationProblem([1.], lambda x: x[0], bounds=([3.], [0.]))
