"""Independent numerical references for simultaneous-GST validation scores."""

import numpy as np
import pytest
from pygsti.circuits import Circuit
from pygsti.data import simulate_data
from pygsti.tools import tvd
from test.helpers.simultaneous_gst_validation import ValidationProfile, build_component_model
from test.helpers.simultaneous_gst_statistics import (
    expected_fisher_information, identifiable_parameter_count,
    shot_noise_held_out_tvd_quantiles,
)


def _noisy_oneq_model_and_circuits(shots):
    """A noisy one-qubit `somehs` model, training data at powers 1 and 2, held-out circuits at 4."""
    model = build_component_model(ValidationProfile.inclass(nqubits=3, max_lengths=(1,)), 1)
    model.from_vector(np.random.default_rng(7).uniform(0.002, 0.02, model.num_params))
    fiducials = [(), ('Gxpi2',), ('Gypi2',), ('Gxpi2',) * 2, ('Gxpi2',) * 3, ('Gypi2',) * 3]
    germs = [('Gi',), ('Gxpi2',), ('Gypi2',), ('Gxpi2', 'Gypi2')]

    def circuits(powers):
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
    model, dataset, _ = _noisy_oneq_model_and_circuits(shots=1000)
    circuits = list(dataset.keys())
    theta = model.to_vector()
    truth = model.sim.bulk_probs(circuits)
    trial = model.copy()

    def expected_loglikelihood(vector):
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
    model, dataset, held_out = _noisy_oneq_model_and_circuits(shots=100000)
    theta = model.to_vector()
    covariance = np.linalg.inv(expected_fisher_information(model, dataset))
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

    mean_quantile, max_quantile = shot_noise_held_out_tvd_quantiles(
        model, dataset, held_out, quantile=0.9, samples=20000)
    assert mean_quantile == pytest.approx(np.quantile(means, 0.9), rel=0.1)
    assert max_quantile == pytest.approx(np.quantile(maxima, 0.9), rel=0.1)


@pytest.mark.parametrize('scenario', ['inclass', 'outclass'])
def test_score_assembly_and_scenario_specific_references(scenario, monkeypatch):
    import test.helpers.simultaneous_gst_statistics as statistics
    from test.helpers.simultaneous_gst_validation import HeldOutCircuits

    model, dataset, circuits = _noisy_oneq_model_and_circuits(shots=1000)
    ideal_model = model.copy()
    ideal_model.from_vector(np.zeros(model.num_params))
    fisher_calls = []
    def fisher(model, dataset):
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
