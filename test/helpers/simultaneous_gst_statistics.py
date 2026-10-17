"""Compute likelihood, predictive-error and shot-noise references for validation.

Builds expected multinomial Fisher information and its identifiable eigenspace,
propagates Gaussian parameter fluctuations to held-out TVD quantiles, and scores
fitted models against the data-generating model and ideal baseline. These are test-support
calculations, not production statistical APIs or acceptance assertions.
"""

import numpy as np
from pygsti.tools import two_delta_logl, tvd

FISHER_RANK_RTOL = 1e-10


def _probabilities_and_jacobian(model, circuits):
    """Stack outcome probabilities and their parameter derivatives, one row per (circuit, outcome).

    Also returns, for each row, the index in `circuits` of the circuit it belongs to.
    """
    probabilities = model.sim.bulk_probs(circuits)
    derivatives = model.sim.bulk_dprobs(circuits)
    rows = [(index, probabilities[circuit][outcome], derivatives[circuit][outcome])
            for index, circuit in enumerate(circuits) for outcome in probabilities[circuit]]
    owner, values, jacobian = zip(*rows)
    return np.array(values), np.array(jacobian), np.array(owner)


def expected_fisher_information(model, dataset) -> np.ndarray:
    """Expected multinomial Fisher information of `dataset`'s circuits and shot counts at `model`."""
    circuits = list(dataset.keys())
    values, jacobian, owner = _probabilities_and_jacobian(model, circuits)
    shots = np.array([dataset[circuit].total for circuit in circuits])[owner]
    return (jacobian.T * (shots / np.clip(values, 1e-12, None))) @ jacobian


def _identifiable_eigenpairs(fisher: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenpairs of `fisher` above a relative threshold: the directions the data identify."""
    eigenvalues, eigenvectors = np.linalg.eigh(fisher)
    identifiable = eigenvalues > FISHER_RANK_RTOL * eigenvalues.max()
    return eigenvalues[identifiable], eigenvectors[:, identifiable]


def identifiable_parameter_count(model, dataset) -> int:
    """Return the rank of the expected Fisher information of `dataset` at `model`.

    This is the number of parameter directions the data identify, and the degrees of
    freedom in Wilks' theorem; gauge and other unidentifiable directions do not count.
    """
    eigenvalues, _ = _identifiable_eigenpairs(expected_fisher_information(model, dataset))
    return len(eigenvalues)


def _shot_noise_quantiles(model, held_out_circuits, eigenpairs, quantile=0.999,
                          samples=20000, seed=0):
    """Propagate a precomputed identifiable eigensystem to held-out predictions."""
    eigenvalues, eigenvectors = eigenpairs
    covariance_root = eigenvectors / np.sqrt(eigenvalues)
    _, held_out_jacobian, owner = _probabilities_and_jacobian(model, list(held_out_circuits))
    response = held_out_jacobian @ covariance_root
    circuit_starts = np.flatnonzero(np.r_[True, np.diff(owner) != 0])
    rng = np.random.default_rng(seed)
    means, maxima = [], []
    for chunk in np.array_split(np.arange(samples), max(1, samples // 500)):
        errors = np.abs(response @ rng.standard_normal((response.shape[1], len(chunk))))
        tvds = 0.5 * np.add.reduceat(errors, circuit_starts, axis=0)
        means.append(tvds.mean(axis=0))
        maxima.append(tvds.max(axis=0))
    return (float(np.quantile(np.concatenate(means), quantile)),
            float(np.quantile(np.concatenate(maxima), quantile)))


def score_validation_fit(ideal_model, datagen_model, fitted_model, dataset, held_out, scenario):
    """Score an existing fit against same-run references without executing or saving it.

    Returns two_delta_logl for the fitted and data-generating models and
    held_out_{mean,max}_tvd summaries for the fitted and ideal models. In-class scores also include identifiable_params and
    shot_noise_held_out_{mean,max}_tvd_q999. The nsigma and held_out_deep_* scores
    are diagnostics, not acceptance criteria. Statistical bounds are asymptotic.
    """
    two_delta_logl_value, nsigma, _ = two_delta_logl(
        fitted_model, dataset, dof_calc_method='modeltest')
    tvd_by_circuit = {
        circuit: float(tvd(fitted_model.probabilities(circuit),
                           datagen_model.probabilities(circuit)))
        for circuit in held_out.circuits
    }
    held_out_tvds = list(tvd_by_circuit.values())
    deep_tvds = [tvd_by_circuit[circuit] for circuit in held_out.deepest]
    # References from the same run.  The data-generating model's own 2*deltaLogL is what the
    # true model scores on exactly this dataset, and the ideal model's held-out TVD (the
    # GST's noiseless starting point) is the prediction error of a fit that learned nothing.
    datagen_two_delta_logl = two_delta_logl(datagen_model, dataset)
    ideal_tvds = [float(tvd(ideal_model.probabilities(circuit), datagen_model.probabilities(circuit)))
                  for circuit in held_out.circuits]
    metrics: dict[str, float | int | str] = {
        'two_delta_logl': float(two_delta_logl_value),
        'datagen_two_delta_logl': float(datagen_two_delta_logl),
        'nsigma': float(nsigma),
        'held_out_mean_tvd': float(np.mean(held_out_tvds)),
        'held_out_max_tvd': float(np.max(held_out_tvds)),
        'held_out_deep_mean_tvd': float(np.mean(deep_tvds)),
        'held_out_deep_max_tvd': float(np.max(deep_tvds)),
        'ideal_held_out_mean_tvd': float(np.mean(ideal_tvds)),
        'ideal_held_out_max_tvd': float(np.max(ideal_tvds)),
    }
    if scenario == 'inclass':
        eigenpairs = _identifiable_eigenpairs(expected_fisher_information(datagen_model, dataset))
        metrics['identifiable_params'] = len(eigenpairs[0])
        mean_threshold, max_threshold = _shot_noise_quantiles(
            datagen_model, held_out.circuits, eigenpairs)
        metrics['shot_noise_held_out_mean_tvd_q999'] = mean_threshold
        metrics['shot_noise_held_out_max_tvd_q999'] = max_threshold
    return metrics
