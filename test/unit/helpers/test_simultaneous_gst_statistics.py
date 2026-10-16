"""Validate the harness statistics against independent numerical references.

Checks Fisher information using a finite-difference expected-likelihood Hessian,
identifiable dimension using matrix rank and a rank-one example, and linearized
TVD quantiles using nonlinear Monte Carlo predictions. Controlled identical
models and recorded Fisher calls check score assembly and scenario selection.
"""

import numpy as np
import pytest
from pygsti.circuits import Circuit
from pygsti.data import simulate_data
from pygsti.tools import tvd
from test.helpers.simultaneous_gst_validation import ValidationProfile, build_component_model
from test.helpers.simultaneous_gst_statistics import (
    expected_fisher_information, identifiable_parameter_count,
    _identifiable_eigenpairs, _shot_noise_quantiles,
)


def _noisy_oneq_model_and_circuits(shots):
    """Prepare training/held-out inputs for the harness statistics under test.

    Use a fixed seeded nonzero parameter vector, explicit fiducial/germ products,
    training powers 1/2 and distinct held-out power 4. Each test constructs its
    own reference (Hessian, rank or nonlinear Monte Carlo), rather than treating
    these generated inputs as golden expected answers.
    """
    model = build_component_model(ValidationProfile.inclass(nqubits=3, max_lengths=(1,)), 1)
    model.from_vector(np.random.default_rng(7).uniform(0.002, 0.02, model.num_params))
    fiducials = [(), ('Gxpi2',), ('Gypi2',), ('Gxpi2',) * 2, ('Gxpi2',) * 3, ('Gypi2',) * 3]
    germs = [('Gi',), ('Gxpi2',), ('Gypi2',), ('Gxpi2', 'Gypi2')]

    def circuits(powers):
        """Build the statistics fixture's explicit fiducial-germ circuit products.

        The reference calculations use these same inputs; this helper generates no
        expected Fisher matrix or prediction-error quantile.
        """
        return list(dict.fromkeys(
            Circuit([(gate, 0) for gate in prep + germ * power + meas], line_labels=(0,))
            for prep in fiducials for germ in germs for power in powers for meas in fiducials))

    training = circuits((1, 2))
    held_out = [circuit for circuit in circuits((4,)) if circuit not in set(training)]
    dataset = simulate_data(model, training, num_samples=shots, seed=0)
    return model, dataset, held_out


def test_expected_fisher_information_is_the_expected_negative_loglikelihood_hessian():
    # Brute-force reference from probabilities alone: central differences of
    # sum_c N_c sum_o p_o(theta) log p_o(theta') in theta', at theta' = theta.
    """Check ``expected_fisher_information`` against an independent Hessian.

    Reference: central differences of the inline expected log likelihood, with
    fixed generating probabilities, give its negative parameter Hessian. The
    comparison allows the stated finite-difference truncation/numerical tolerance.
    """
    model, dataset, _ = _noisy_oneq_model_and_circuits(shots=1000)
    circuits = list(dataset.keys())
    theta = model.to_vector()
    truth = model.sim.bulk_probs(circuits)
    trial = model.copy()

    def expected_loglikelihood(vector):
        """Evaluate the inline reference for ``expected_fisher_information``.

        Hold generating probabilities fixed while varying trial parameters in the
        multinomial expected log likelihood; central differences supply the Hessian.
        """
        trial.from_vector(vector)
        probabilities = trial.sim.bulk_probs(circuits)
        return sum(dataset[circuit].total * truth[circuit][outcome] * np.log(probabilities[circuit][outcome])
                   for circuit in circuits for outcome in truth[circuit])

    step = 1e-4
    basis = np.eye(len(theta)) * step
    hessian = np.zeros((len(theta), len(theta)))
    for i in range(len(theta)):
        for j in range(i, len(theta)):
            hessian[i, j] = hessian[j, i] = (
                expected_loglikelihood(theta + basis[i] + basis[j])
                - expected_loglikelihood(theta + basis[i] - basis[j])
                - expected_loglikelihood(theta - basis[i] + basis[j])
                + expected_loglikelihood(theta - basis[i] - basis[j])) / (4 * step ** 2)

    fisher = expected_fisher_information(model, dataset)
    np.testing.assert_allclose(fisher, -hessian, rtol=1e-3, atol=1e-6 * np.abs(fisher).max())


def test_identifiable_parameter_count_is_the_fisher_information_rank():
    """Check ``identifiable_parameter_count`` against matrix rank and a rank-one case.

    Reference: NumPy matrix rank supplies the first answer using the computed
    Fisher matrix. In the second example only the idle circuit contributes an
    informative binary-outcome direction, whose opposite Jacobian rows imply rank one.
    """
    model, dataset, _ = _noisy_oneq_model_and_circuits(shots=1000)
    fisher = expected_fisher_information(model, dataset)
    assert identifiable_parameter_count(model, dataset) == np.linalg.matrix_rank(fisher)

    # One informative circuit identifies one direction: its two outcome rows of the Jacobian
    # are r and -r, because the probabilities sum to one, so F is a multiple of r r^T.
    single_circuit = simulate_data(
        model, [Circuit([], line_labels=(0,)), Circuit([('Gi', 0)], line_labels=(0,))],
        num_samples=1000, seed=0)
    assert identifiable_parameter_count(model, single_circuit) == 1


def test_shot_noise_quantiles_match_monte_carlo_of_exact_held_out_predictions():
    # Reference: draw theta_hat ~ N(theta, F^-1) and evaluate the exact (not linearized)
    # held-out TVDs.  Enough shots keep theta_hat close enough to theta for the
    # linearization to hold to within the Monte Carlo error.
    """Check ``_shot_noise_quantiles`` against nonlinear Monte Carlo predictions.

    Reference: invert the training Fisher matrix, draw Gaussian parameter vectors,
    and evaluate exact model probabilities and TVDs inline, rather than linearized
    responses. Compare empirical 90th percentiles with 10% relative tolerance for
    sampling variation and linearization error at the chosen large shot count.
    """
    model, dataset, held_out = _noisy_oneq_model_and_circuits(shots=100000)
    theta = model.to_vector()
    fisher = expected_fisher_information(model, dataset)
    covariance = np.linalg.inv(fisher)
    truth = model.sim.bulk_probs(held_out)
    trial = model.copy()
    rng = np.random.default_rng(1)
    means, maxima = [], []
    for vector in rng.multivariate_normal(theta, covariance, size=1000):
        trial.from_vector(vector)
        predictions = trial.sim.bulk_probs(held_out)
        tvds = [float(tvd(predictions[circuit], truth[circuit])) for circuit in held_out]
        means.append(np.mean(tvds))
        maxima.append(np.max(tvds))

    mean_quantile, max_quantile = _shot_noise_quantiles(
        model, held_out, _identifiable_eigenpairs(fisher), quantile=0.9, samples=20000)
    assert mean_quantile == pytest.approx(np.quantile(means, 0.9), rel=0.1)
    assert max_quantile == pytest.approx(np.quantile(maxima, 0.9), rel=0.1)


@pytest.mark.parametrize('scenario', ['inclass', 'outclass'])
def test_score_assembly_and_scenario_specific_references(scenario, monkeypatch):
    """Check ``score_validation_fit`` values and scenario-specific reference fields.

    Reference: identical generator and estimate imply equal likelihood deficits
    and zero predictive TVDs; a zero-noise baseline differs. Explicit field sets
    and recorded calls require one shared Fisher calculation in class and none
    out of class. This checks assembly, not an independent likelihood implementation.
    """
    import test.helpers.simultaneous_gst_statistics as statistics
    from test.helpers.simultaneous_gst_validation import HeldOutCircuits

    model, dataset, circuits = _noisy_oneq_model_and_circuits(shots=1000)
    ideal_model = model.copy()
    ideal_model.from_vector(np.zeros(model.num_params))
    fisher_calls = []
    def fisher(model, dataset):
        """Record Fisher calls made by ``score_validation_fit`` and delegate values.

        The expected call count is fixed by the scenario; numerical Fisher correctness
        is checked separately against the finite-difference reference.
        """
        fisher_calls.append(True)
        return expected_fisher_information(model, dataset)
    monkeypatch.setattr(statistics, 'expected_fisher_information', fisher)
    scores = statistics.score_validation_fit(
        ideal_model, model, model.copy(), dataset, HeldOutCircuits(tuple(circuits), tuple(circuits[-2:])), scenario)
    assert scores['two_delta_logl'] == pytest.approx(scores['datagen_two_delta_logl'])
    assert scores['held_out_mean_tvd'] == scores['held_out_max_tvd'] == 0.0
    assert scores['held_out_deep_mean_tvd'] == scores['held_out_deep_max_tvd'] == 0.0
    assert scores['target_held_out_mean_tvd'] > 0.0
    reference_keys = {'identifiable_params', 'shot_noise_held_out_mean_tvd_q999',
                      'shot_noise_held_out_max_tvd_q999'}
    assert reference_keys.intersection(scores) == (reference_keys if scenario == 'inclass' else set())
    assert len(fisher_calls) == (1 if scenario == 'inclass' else 0)
