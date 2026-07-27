import numpy as np
import scipy as sp
from scipy.stats import norm
from scipy.linalg import hadamard
import stim 

from .utils import build_masked_hadamard
from .io import dem_from_event_probabilities


def dense_dem_estimation(syndrome_counts: dict) -> np.ndarray:
    """
    Estimate a dense DEM from syndrome counts using log and Hadamard transforms.

    Parameters:
        syndrome_counts: dict
            Mapping bitstrings (e.g., '0011') to counts

    Returns:
        event_probabilities: ndarray
            Array of estimated event probabilities
    """
    n = len(next(iter(syndrome_counts)))  # number of bits in syndrome
    n_runs = sum(syndrome_counts.values())
    size = 2 ** n

    # Create vector of observed bitstring probabilities
    probabilities = np.zeros(size)
    for i in range(size):
        bitstring = format(i, f"0{n}b")
        probabilities[i] = syndrome_counts.get(bitstring, 0) / n_runs

    # Convert to polarizations
    polarizations = hadamard(size) @ probabilities

    # Compute depolarizations
    depolarizations = -np.log(polarizations)

    # Convert to attenuations
    attenuations = 2 * hadamard(size) @ depolarizations / size
    attenuations[0] = 0

    # Recover probabilities
    event_probabilities = 0.5 - 0.5 * np.exp(attenuations)
    return event_probabilities


def estimate_dem_and_covariance(syndrome_counts: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate a dense DEM from syndrome counts using log and Hadamard transforms.
    Also computes covariance matrix from standard error formula using Jacobian.

    Parameters:
        syndrome_counts: dict
            Mapping bitstrings (e.g., '0011') to counts

    Returns:
        event_probabilities: ndarray
            Estimated probabilities
        covariance_matrix: ndarray
            Covariance matrix of estimated probabilities
    """
    n = len(next(iter(syndrome_counts)))
    n_runs = sum(syndrome_counts.values())
    size = 2 ** n

    # Estimate bitstring probabilities from input bitstrings
    probabilities = np.zeros(size)
    for i in range(size):
        bitstring = format(i, f"0{n}b")
        probabilities[i] = syndrome_counts.get(bitstring, 0) / n_runs

    # Transform to event probabilities
    H = hadamard(size)
    polarizations = H @ probabilities
    depolarizations = -np.log(polarizations)
    attenuations = 2 * H @ depolarizations / size
    attenuations[0] = 0
    event_probabilities = 0.5 - 0.5 * np.exp(attenuations)

    # Compute the Jacobian of event_probabilities w.r.t. input probabilities
    # Chain rule: d(event_probs)/d(probabilities)
    # J = d(event_probs)/d(attenuations) * d(attenuations)/d(depolarizations)
    #     * d(depolarizations)/d(polarizations) * d(polarizations)/d(probabilities)

    d_event_d_att = 0.5 * np.exp(attenuations)
    d_att_d_dep = 2 * H / size
    d_dep_d_pol = -np.diag(1 / polarizations)
    d_pol_d_prob = H

    # Compute covariance of input probabilities (multinomial)
    cov_input = np.diag(probabilities) - np.outer(probabilities, probabilities)
    cov_input /= n_runs

    J = np.diag(d_event_d_att) @ d_att_d_dep @ d_dep_d_pol @ d_pol_d_prob
    covariance_matrix = J @ cov_input @ J.T

    return event_probabilities, covariance_matrix


def threshold_probabilities(
    estimated_probabilities: np.ndarray,
    covariance_matrix: np.ndarray,
    alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """
    Zero out elements of estimated_probabilities that are statistically consistent with zero,
    using a Bonferroni-corrected z-test.

    Parameters:
        estimated_probabilities: np.ndarray of shape (n,)
        covariance_matrix: np.ndarray of shape (n, n)
        alpha: float
            Family-wise error rate (default 0.05)

    Returns:
        thresholded: np.ndarray
            Thresholded probabilities
        mask: np.ndarray
            Boolean array where True means "statistically significant"
    """
    n = len(estimated_probabilities)
    std_devs = np.sqrt(np.diag(covariance_matrix))

    # Bonferroni correction with one-sided test
    corrected_alpha = alpha / n
    z_threshold = norm.ppf(1 - corrected_alpha)  # ppf is the inverse CDF of the normal dist

    # Compute z-scores and check statistical significance
    z_scores = estimated_probabilities / std_devs
    significant = z_scores > z_threshold

    thresholded = estimated_probabilities.copy()
    thresholded[~significant] = 0.0

    return thresholded, significant


def compute_outcome_distribution_from_dem(dem: stim.DetectorErrorModel) -> np.ndarray:
    """
    Compute the outcome distribution from a DEM using log and Hadamard transform.

    Parameters:
        dem: stim.DetectorErrorModel

    Returns:
        prob_estimate: np.ndarray
            Array of 2^n probabilities, in increasing binary order
    """
    # Convert DEM to attenuations
    attenuations = np.zeros(2 ** dem.num_detectors, dtype=float)
    all_bitstrings = np.array([
        [int(bit) for bit in format(n, f"0{dem.num_detectors}b")]
        for n in range(2 ** dem.num_detectors)
    ])

    for event in dem:
        prob = event.args_copy()[0]
        targets = [target.val for target in event.targets_copy()]
        event_vec = [1 if (dem.num_detectors - idx - 1) in targets else 0 for idx in range(dem.num_detectors)]
        attenuation = -np.log(1 - 2 * prob)
        attenuations += attenuation * (1 - (-1) ** np.dot(all_bitstrings, event_vec)) / 2

    # Compute polarizations from attenuations
    polarizations = np.exp(-attenuations)
    polarizations[0] = 1

    # Compute probabilities from polarizations
    probabilities = hadamard(2 ** dem.num_detectors) @ polarizations / (2 ** dem.num_detectors)
    return probabilities

def fit_specified_dem(
    syndrome_counts,
    masks,
    atol=1e-4,
    return_probs=False,
    return_covariance=False,
):
    """
    Given a set of DEM events (as integer bitmasks), find the best-fit error rates.
    Uses polarizations and a submatrix of the Hadamard matrix to invert.

    Parameters:
        syndrome_counts: dict
            Observed syndrome bitstrings.
        masks: list[int]
            Integer bitmasks describing DEM events.
        atol: float
            Threshold for zeroing small probabilities.
        return_probs: bool
            Return event probabilities instead of a DEM.
        return_covariance: bool
            Return covariance matrix for event probabilities.

    Returns:
        stim.DetectorErrorModel or np.ndarray
    """
    if isinstance(masks, set):
        masks = sorted(list(masks))
    masks = np.array(masks)
    counts = np.fromiter(syndrome_counts.values(), dtype=float)
    n_runs = sum(counts)
    n_bits = len(next(iter(syndrome_counts)))
    probabilities = counts/n_runs

    # Compute polarizations
    dem_masks = masks
    pol_masks = set(masks)
    pol_masks.update([1 << i for i in range(n_bits)])
    pol_masks.update([1 << i | 1 << j for i in range(1, n_bits) for j in range(i)])
    pol_masks = np.array(list(pol_masks))

    # Apply transformations
    syndrome_masks = [int(synd, 2) for synd in syndrome_counts.keys()]
    H_sub = build_masked_hadamard(pol_masks, syndrome_masks)
    polarizations = H_sub @ counts / n_runs
    depolarizations = -np.log(polarizations)
    W = (np.ones((len(pol_masks), len(dem_masks))) - build_masked_hadamard(pol_masks, dem_masks)) / 2
    Winv = np.linalg.pinv(W)
    attenuations = Winv @ depolarizations
    event_probs = 0.5 - 0.5 * np.exp(-attenuations)

    if return_covariance:
        # Compute the Jacobian of event_probabilities w.r.t. input probabilities
        # # Chain rule: d(event_probs)/d(probabilities)
        # # J = d(event_probs)/d(attenuations) * d(attenuations)/d(depolarizations)
        # #     * d(depolarizations)/d(polarizations) * d(polarizations)/d(probabilities)       
        d_event_d_att = 0.5 * np.exp(-attenuations)
        d_att_d_dep = Winv
        d_dep_d_pol = -np.diag(1 / polarizations)
        d_pol_d_prob = H_sub

        # Compute covariance of input probabilities (multinomial)
        cov_input = np.diag(probabilities) - np.outer(probabilities, probabilities)
        cov_input /= n_runs

        J = np.diag(d_event_d_att) @ d_att_d_dep @ d_dep_d_pol @ d_pol_d_prob
        covariance_matrix = J @ cov_input @ J.T

    if return_probs:
        if return_covariance:
            return dem_masks, event_probs, covariance_matrix
        return dem_masks, event_probs
    else:
        dem = dem_from_event_probabilities(event_probs, dem_masks, atol=atol)
        if return_covariance:
            return dem, dem_masks, event_probs, covariance_matrix
        return dem
