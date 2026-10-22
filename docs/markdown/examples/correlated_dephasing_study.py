"""Known-SPAM identification of Markovian dephasing covariance graphs.

Run with the source checkout on PYTHONPATH. All fits in each scenario share
one binomial dataset; this is noise identification, not self-consistent GST.
The common-mode alternative assumes a known uniform spatial profile.
"""

import itertools
import json

import numpy as np
from scipy.optimize import least_squares
from scipy.special import rel_entr

from pygsti.baseobjs import Basis, QubitSpace
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL
from pygsti.modelmembers.operations import LindbladErrorgen, ExpErrorgenOp
from pygsti.modelmembers.operations.lindbladcoefficients import LindbladCoefficientBlock as LCB
from pygsti.tools.basistools import change_basis


DIRECTIONS = ('ZIII', 'IZII', 'IIZI', 'IIIZ')
EDGES = ((0, 1), (1, 2), (2, 3), (0, 3))


def product_design(times):
    """Independent experiments measuring X_i, X_i X_j, or Y_i Y_j on |+>^4.

    Each signal is a weighted sum of at most two exp(-2 t v^T K v).
    No entangled preparations are needed for this training design.
    """
    vectors, weights, durations, labels = [], [], [], []
    eye = np.eye(4)
    for t in times:
        for i in range(4):
            vectors.append([eye[i], np.zeros(4)])
            weights.append([1., 0.])
            durations.append(t)
            labels.append(('X', i))
        for i, j in itertools.combinations(range(4), 2):
            for observable, sign in (('XX', 1), ('YY', -1)):
                vectors.append([eye[i] + eye[j], eye[i] - eye[j]])
                weights.append([sign * .5, .5])
                durations.append(t)
                labels.append((observable, i, j))
    return np.array(vectors), np.array(weights), np.array(durations), labels


def probabilities(covariance, design, covariance_jacobian=None):
    """Positive-parity probabilities and optional parameter Jacobian."""
    vectors, weights, times, _ = design
    rates = 2 * np.einsum('msi,ij,msj->ms', vectors, covariance, vectors)
    contrasts = np.exp(-times[:, None] * rates)
    p = .5 * (1 + np.sum(weights * contrasts, axis=1))
    if covariance_jacobian is None:
        return p
    rate_jacobian = 2 * np.einsum('msi,ijp,msj->msp', vectors, covariance_jacobian, vectors)
    jacobian = -.5 * times[:, None] * np.einsum('ms,ms,msp->mp', weights, contrasts, rate_jacobian)
    return p, jacobian


class CovarianceChart:
    """Use pyGSTi's coefficient charts without exponentiating global matrices in a fit."""

    def __init__(self, family):
        self.family = family
        edges = () if family == 'independent' else (
            None if family == 'dense' else [(DIRECTIONS[i], DIRECTIONS[j]) for i, j in EDGES])
        errorgen = LindbladErrorgen.from_correlated_dephasing(
            QubitSpace(4), edges, initial_stochastic_rate=.03
        )
        self.block = errorgen.coefficient_blocks[0]
        self.initial = self.block.to_vector()
        self.nblock = self.block.num_params
        if family == 'graph+common':
            self.initial = np.r_[self.initial, .05]  # Common covariance is amplitude squared.
        self.nparams = len(self.initial)
        pos = {b: i for i, b in enumerate(self.block.basis_element_labels)}
        self.pairs = [(pos[label.basis_element_labels[0]], pos[label.basis_element_labels[-1]])
                      for label in self.block.elementary_errorgens]

    def evaluate(self, vector):
        self.block.from_vector(vector[:self.nblock])
        covariance = self.block.kossakowski_matrix().real
        coefficient_jacobian = self.block.deriv_wrt_params(vector[:self.nblock])
        jacobian = np.zeros((4, 4, self.nparams))
        for (i, j), derivative in zip(self.pairs, coefficient_jacobian):
            jacobian[i, j, :self.nblock] = derivative
            jacobian[j, i, :self.nblock] = derivative
        if self.family == 'graph+common':
            covariance += vector[-1] ** 2 * np.ones((4, 4))
            jacobian[:, :, -1] = 2 * vector[-1]
        return covariance, jacobian


def binomial_residuals(p, positives, shots):
    """Signed sqrt likelihood-deviance residuals and derivatives with respect to p."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    q = positives / shots
    deviance = 2 * shots * (rel_entr(q, p) + rel_entr(1 - q, 1 - p))
    residual = np.sign(q - p) * np.sqrt(np.maximum(deviance, 0.))
    near = (q > 0) & (q < 1) & (np.abs(q - p) < 1e-7)
    far = ~near
    derivative = np.empty_like(p)
    derivative[far] = shots * (p[far] - q[far]) / (p[far] * (1 - p[far]) * residual[far])
    # Local expansion avoids cancellation when predicted and observed proportions coincide.
    variance = p[near] * (1 - p[near])
    residual[near] = np.sqrt(shots) * (q[near] - p[near]) / np.sqrt(variance)
    derivative[near] = (-np.sqrt(shots) / np.sqrt(variance)
                        - .5 * np.sqrt(shots) * (q[near] - p[near]) * (1 - 2 * p[near]) / variance ** 1.5)
    return residual, derivative


def fit(family, design, positives, shots):
    chart = CovarianceChart(family)

    def evaluate(vector, with_jacobian=False):
        covariance, derivative = chart.evaluate(vector)
        p, jacobian = probabilities(covariance, design, derivative)
        residual, slope = binomial_residuals(p, positives, shots)
        return slope[:, None] * jacobian if with_jacobian else residual

    optimum = least_squares(evaluate, chart.initial, jac=lambda v: evaluate(v, True),
                            max_nfev=300, ftol=1e-10, xtol=1e-10, gtol=1e-10, x_scale='jac')
    if not optimum.success:
        raise RuntimeError(f'{family} failed to converge: {optimum.message}')
    covariance = chart.evaluate(optimum.x)[0]
    return covariance, {
        'parameters': chart.nparams,
        'deviance': float(np.sum(optimum.fun ** 2)),
        'converged': bool(optimum.success),
        'status': int(optimum.status),
        'function_evaluations': int(optimum.nfev),
        'optimality': float(optimum.optimality),
    }


def heldout_design():
    # Contrasts between basis states differing on 3 or 4 sites, absent from training.
    vectors = [v for v in itertools.product((-1., 0., 1.), repeat=4)
               if sum(x != 0 for x in v) >= 3 and next(x for x in v if x != 0) > 0]
    rows = [(v, t) for t in (.5, 1.5, 3., 6.) for v in vectors]
    return (np.array([[v, np.zeros(4)] for v, _ in rows]),
            np.tile([1., 0.], (len(rows), 1)), np.array([t for _, t in rows]),
            [('coherence',) for _ in rows])


def verify_physical_signals(covariance, design):
    """Independent full-channel check of the product-state parity formulas."""
    p_formula = probabilities(covariance, design)
    plus = np.ones((16, 16)) / 16
    one = Basis.cast('PP', 4)
    expected = []
    for t in sorted(set(design[2])):
        errorgen = LindbladErrorgen.from_correlated_dephasing(QubitSpace(4), initial_covariance=t * covariance)
        channel = change_basis(ExpErrorgenOp(errorgen).to_dense(), 'pp', 'std')
        rho = (channel @ plus.ravel()).reshape(16, 16)
        for label in [label for duration, label in zip(design[2], design[3]) if duration == t]:
            factors = [one['I']] * 4
            for i in label[1:]:
                factors[i] = one['Y' if label[0] == 'YY' else 'X']
            observable = factors[0]
            for factor in factors[1:]:
                observable = np.kron(observable, factor)
            expected.append(.5 * (1 + np.trace(observable @ rho).real))
    np.testing.assert_allclose(expected, p_formula, atol=2e-13)
    return float(np.max(np.abs(expected - p_formula)))


def quantum_phase_cycle():
    """A phase cycle separates Gaussian covariance plus H from general dephasing."""
    basis = Basis.cast('PP', 16)
    labels = [LEEL('S', ('ZI',)), LEEL('S', ('IZ',)),
              LEEL('C', ('ZI', 'IZ')), LEEL('A', ('ZI', 'IZ'))]
    block = LCB('other_sparse', basis, param_mode='cholesky', error_generator_labels=labels,
                initial_block_data=np.array([.03, .02, 0., .01]))
    errorgen = LindbladErrorgen([block], basis, 'pp', state_space=QubitSpace(2))
    generator = change_basis(errorgen.to_dense(), 'pp', 'std')
    np.testing.assert_allclose(generator - np.diag(generator.diagonal()), 0, atol=1e-14)
    phases = generator.diagonal().reshape(4, 4).imag
    cycle = phases[0, 1] + phases[1, 3] + phases[3, 0]
    assert abs(cycle) > .01  # A Hamiltonian phase h_y-h_x has zero sum on every cycle.
    # A classical Poisson process applying a fixed diagonal unitary also has a
    # nonzero phase cycle. This is not, by itself, a witness of a quantum bath.
    unitary_diagonal = np.exp(1j * np.array([0., np.pi / 2, 0., np.pi]))
    poisson_phases = .01 * np.outer(unitary_diagonal, unitary_diagonal.conj()).imag
    poisson_cycle = poisson_phases[0, 1] + poisson_phases[1, 3] + poisson_phases[3, 0]
    assert abs(poisson_cycle) > .01
    return float(cycle), float(poisson_cycle)


def run_study():
    design = product_design((1., 2., 4., 8.))
    heldout = heldout_design()
    true_graph = np.diag([.028, .033, .025, .03])
    for (i, j), value in zip(EDGES, (.007, -.006, .005, .004)):
        true_graph[i, j] = true_graph[j, i] = value

    # Identify coefficient directions directly, independently of factor-chart geometry.
    coordinates = np.zeros((4, 4, 10))
    pairs = [(i, i) for i in range(4)] + list(itertools.combinations(range(4), 2))
    for k, (i, j) in enumerate(pairs):
        coordinates[i, j, k] = coordinates[j, i, k] = 1
    _, jacobian = probabilities(true_graph, design, coordinates)
    singles = np.array([label[0] == 'X' for label in design[3]])
    ranks = {'single_ramsey_means': int(np.linalg.matrix_rank(jacobian[singles])),
             'all_product_parities': int(np.linalg.matrix_rank(jacobian))}
    assert ranks == {'single_ramsey_means': 4, 'all_product_parities': 10}

    # Verify the complete analytic probability Jacobian before using it in fitting.
    chart = CovarianceChart('graph+common')
    vector = chart.initial.copy()
    covariance, derivative = chart.evaluate(vector)
    _, analytic = probabilities(covariance, design, derivative)
    eps = 1e-6
    numeric = np.stack([(probabilities(chart.evaluate(vector + eps * e)[0], design)
                         - probabilities(chart.evaluate(vector - eps * e)[0], design)) / (2 * eps)
                        for e in np.eye(len(vector))], axis=-1)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-9)

    observed = np.random.default_rng(90).binomial(50000, probabilities(covariance, design))
    p_at_start = probabilities(covariance, design)
    residual_slope = binomial_residuals(p_at_start, observed, 50000)[1]
    residual_jacobian = residual_slope[:, None] * analytic
    numeric_residual = np.stack([
        (binomial_residuals(probabilities(chart.evaluate(vector + eps * e)[0], design), observed, 50000)[0]
         - binomial_residuals(probabilities(chart.evaluate(vector - eps * e)[0], design), observed, 50000)[0])
        / (2 * eps) for e in np.eye(len(vector))], axis=-1)
    np.testing.assert_allclose(residual_jacobian, numeric_residual, rtol=2e-5, atol=2e-4)

    # A noise-free recovery check distinguishes optimizer behavior from sample fluctuations.
    exact_p = probabilities(true_graph, design)
    exact_fit, exact_info = fit('graph', design, 50000 * exact_p, 50000)
    np.testing.assert_allclose(exact_fit, true_graph, atol=2e-9)

    scenarios = {}
    for seed, (name, truth) in enumerate((('graph', true_graph),
                                        ('graph+common', true_graph + .006 * np.ones((4, 4)))), 810):
        shots = 50000
        true_p = probabilities(truth, design)
        positives = np.random.default_rng(seed).binomial(shots, true_p)
        true_deviance = float(np.sum(binomial_residuals(true_p, positives, shots)[0] ** 2))
        true_heldout = probabilities(truth, heldout)
        results = {}
        for family in ('independent', 'graph', 'dense', 'graph+common'):
            fitted, info = fit(family, design, positives, shots)
            predicted = probabilities(fitted, heldout)
            info.update({
                'deviance_minus_truth': info['deviance'] - true_deviance,
                'heldout_mean_tvd': float(np.mean(np.abs(predicted - true_heldout))),
                'heldout_max_tvd': float(np.max(np.abs(predicted - true_heldout))),
                'relative_covariance_error': float(np.linalg.norm(fitted - truth) / np.linalg.norm(truth)),
                'fitted_covariance': fitted.tolist(),
            })
            assert np.linalg.eigvalsh(fitted)[0] > 0
            if family == name or family == 'dense':
                assert info['deviance_minus_truth'] <= 1e-6
            # A freshly constructed pyGSTi generator reproduces the fitted covariance.
            restored = LindbladErrorgen.from_correlated_dephasing(QubitSpace(4), initial_covariance=fitted)
            np.testing.assert_allclose(restored.coefficient_blocks[0].kossakowski_matrix(), fitted, atol=1e-13)
            results[family] = info
        scenarios[name] = {
            'true_covariance': truth.tolist(), 'true_deviance': true_deviance,
            'noiseless_model_heldout_mean_tvd': float(np.mean(1 - true_heldout)),
            'fits': results,
        }

    dense_covariance = np.ones((3, 3)) + .1 * np.eye(3)
    thresholded = dense_covariance.copy()
    thresholded[0, 2] = thresholded[2, 0] = 0
    assert np.linalg.eigvalsh(dense_covariance)[0] > 0
    assert np.linalg.eigvalsh(thresholded)[0] < 0
    complex_cycle, poisson_cycle = quantum_phase_cycle()
    result = {
        'shots_per_setting': shots, 'training_settings': len(design[2]),
        'heldout_coherences_and_times': len(heldout[2]), 'identifiability_ranks': ranks,
        'max_analytic_probability_jacobian_error': float(np.max(np.abs(analytic - numeric))),
        'max_binomial_residual_jacobian_error': float(np.max(np.abs(residual_jacobian - numeric_residual))),
        'expected_data_recovery_error': float(np.max(np.abs(exact_fit - true_graph))),
        'expected_data_fit_converged': exact_info['converged'],
        'max_full_channel_signal_error': verify_physical_signals(true_graph, product_design((1., 4.))),
        'entrywise_zeroing_negative_eigenvalue': float(np.linalg.eigvalsh(thresholded)[0]),
        'imaginary_correlation_phase_cycle': complex_cycle,
        'classical_poisson_phase_cycle': poisson_cycle,
        'scenarios': scenarios,
    }
    return result


if __name__ == '__main__':
    print(json.dumps(run_study(), indent=2))
