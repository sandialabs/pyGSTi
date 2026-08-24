import itertools
import warnings

import numpy as np
import scipy as sp
from scipy.stats import norm
from scipy.linalg import hadamard
import stim

from .utils import (
    build_masked_hadamard,
    counts_to_detector_arrays,
    masked_hadamard_dot,
    masks_to_packed,
    pack_detector_samples,
    packed_parity_matrix,
    weighted_odd_counts,
)
from .io import dem_from_event_probabilities

#: fit_specified_dem includes every detector pair among its polarization
#: masks by default; above this many detectors that set is quadratically
#: large and a restricted mask set is used instead (with a warning if the
#: caller did not supply one explicitly).
_ALL_PAIRS_POL_LIMIT = 48


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


def estimate_dem_and_covariance_from_probabilities(
    probabilities: np.ndarray,
    n_runs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dense DEM estimate and covariance from a full outcome-probability vector.

    This is the computational core of `estimate_dem_and_covariance`, exposed
    separately so callers that already hold the (marginal) outcome
    distribution as an array -- e.g. the vectorized lattice-pruning checks --
    can skip the bitstring-dict round trip.

    Parameters:
        probabilities: np.ndarray
            2**n outcome probabilities in increasing binary order.
        n_runs: int
            Number of shots behind the probabilities (sets the multinomial
            covariance scale).

    Returns:
        event_probabilities: ndarray
            Estimated probabilities.
        covariance_matrix: ndarray
            Covariance matrix of estimated probabilities.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    size = probabilities.size

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

    return estimate_dem_and_covariance_from_probabilities(probabilities, n_runs)


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

def default_polarization_masks(dem_masks, n_bits: int) -> list:
    """
    A restricted polarization-mask set for fitting a sparse DEM at scale.

    Contains every single-detector mask, every pair of detectors that
    co-occur within some DEM event, and the event masks themselves. This
    grows linearly with the number of detectors and events (unlike the
    all-pairs default of `fit_specified_dem`, which is quadratic in the
    detector count) while still constraining every event locally.

    Parameters:
        dem_masks: Iterable[int]
            Integer bitmasks of the DEM events being fit.
        n_bits: int
            Number of detectors.

    Returns:
        pol_masks: list[int]
            Sorted, deduplicated polarization masks.
    """
    pol = set(int(m) for m in dem_masks if int(m) != 0)
    pol.update(1 << i for i in range(n_bits))
    for m in dem_masks:
        mm = int(m)
        bits = []
        while mm:
            low = mm & -mm
            bits.append(low.bit_length() - 1)
            mm ^= low
        for a, b in itertools.combinations(bits, 2):
            pol.add((1 << a) | (1 << b))
    return sorted(pol)


def _fit_specified_dem_packed(
    syndrome_counts,
    dem_masks,
    pol_masks,
    n_bits,
    return_covariance,
    cov_chunk=4096,
):
    """
    Bit-packed implementation of the `fit_specified_dem` math.

    Identical statistics to the legacy dense path: exact integer
    polarizations, the same (1 - H)/2 design matrix, pseudoinverse solve and
    Jacobian-propagated multinomial covariance -- but never materializes the
    (num_pol_masks x num_syndromes) Hadamard submatrix, so it scales to
    hundreds of detectors.
    """
    samples, weights = counts_to_detector_arrays(syndrome_counts)
    n_runs = int(weights.sum())
    packed = pack_detector_samples(samples)

    pol_list = sorted(set(int(m) for m in pol_masks))
    pol_packed = masks_to_packed(pol_list, n_bits)
    dem_packed = masks_to_packed(dem_masks, n_bits)

    # Exact integer polarizations: pol = (N - 2 * weighted odd count) / N.
    odd = weighted_odd_counts(packed, weights, pol_packed)
    polarizations = (n_runs - 2 * odd) / n_runs

    usable = polarizations > 0
    if not usable.all():
        warnings.warn(
            f"fit_specified_dem: dropping {int((~usable).sum())} polarization "
            "mask(s) with non-positive observed polarization (their "
            "log-polarizations are undefined)."
        )
        pol_list = [m for m, u in zip(pol_list, usable) if u]
        pol_packed = pol_packed[usable]
        polarizations = polarizations[usable]

    depolarizations = -np.log(polarizations)
    W = packed_parity_matrix(pol_packed, dem_packed).astype(float)
    Winv = np.linalg.pinv(W)
    attenuations = Winv @ depolarizations
    event_probs = 0.5 - 0.5 * np.exp(-attenuations)

    if not return_covariance:
        return event_probs, None

    # J = diag(0.5 e^{-att}) @ Winv @ (-diag(1/pol)) @ H_sub with
    # cov_input = (diag(p) - p p^T) / n; accumulate J diag(p) J^T and J p in
    # syndrome chunks so H_sub never exists in full.
    A = (0.5 * np.exp(-attenuations))[:, None] * Winv
    A = A * (-1.0 / polarizations)[None, :]

    n_events = len(dem_masks)
    probabilities = weights / n_runs
    pol_bits = np.zeros((len(pol_list), n_bits), dtype=np.float32)
    for i, m in enumerate(pol_list):
        mm = int(m)
        while mm:
            low = mm & -mm
            pol_bits[i, low.bit_length() - 1] = 1.0
            mm ^= low

    S1 = np.zeros((n_events, n_events), dtype=float)
    v = np.zeros(n_events, dtype=float)
    sample_bits = samples.astype(np.float32)
    for i0 in range(0, samples.shape[0], cov_chunk):
        chunk_bits = sample_bits[i0:i0 + cov_chunk]
        # Overlap counts are exact small integers in float32.
        parity = (pol_bits @ chunk_bits.T).astype(np.int64) & 1
        H_c = 1.0 - 2.0 * parity
        K_c = A @ H_c
        p_c = probabilities[i0:i0 + cov_chunk]
        S1 += (K_c * p_c) @ K_c.T
        v += K_c @ p_c
    covariance_matrix = (S1 - np.outer(v, v)) / n_runs
    return event_probs, covariance_matrix


def fit_specified_dem(
    syndrome_counts,
    masks,
    atol=1e-4,
    return_probs=False,
    return_covariance=False,
    pol_masks=None,
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
        pol_masks: Optional[Sequence[int]]
            Polarization masks used to constrain the fit. Default (None)
            uses the event masks plus every single and every pair of
            detectors when the detector count is at most
            `_ALL_PAIRS_POL_LIMIT`; above that the quadratic all-pairs set is
            replaced by `default_polarization_masks(masks, n_bits)` (with a
            warning). Pass an explicit sequence to control the trade-off
            yourself.

    Returns:
        stim.DetectorErrorModel or np.ndarray
    """
    if isinstance(masks, set):
        masks = sorted(list(masks))
    masks = np.array(masks)
    n_bits = len(next(iter(syndrome_counts)))
    dem_masks = masks

    if pol_masks is None and n_bits <= _ALL_PAIRS_POL_LIMIT:
        # Legacy dense path: all singles and pairs.
        counts = np.fromiter(syndrome_counts.values(), dtype=float)
        n_runs = sum(counts)
        probabilities = counts / n_runs

        pol_mask_set = set(masks)
        pol_mask_set.update([1 << i for i in range(n_bits)])
        pol_mask_set.update([1 << i | 1 << j for i in range(1, n_bits)
                             for j in range(i)])
        pol_mask_arr = np.array(list(pol_mask_set))

        syndrome_masks = [int(synd, 2) for synd in syndrome_counts.keys()]
        # Streamed product: the dense (pol_masks x syndromes) Hadamard
        # submatrix is only materialized when the covariance (which needs
        # the full Jacobian) is requested.
        polarizations = masked_hadamard_dot(pol_mask_arr, syndrome_masks,
                                            counts) / n_runs
        depolarizations = -np.log(polarizations)
        W = (np.ones((len(pol_mask_arr), len(dem_masks)))
             - build_masked_hadamard(pol_mask_arr, dem_masks)) / 2
        Winv = np.linalg.pinv(W)
        attenuations = Winv @ depolarizations
        event_probs = 0.5 - 0.5 * np.exp(-attenuations)

        covariance_matrix = None
        if return_covariance:
            # Compute the Jacobian of event_probabilities w.r.t. input probabilities
            # # Chain rule: d(event_probs)/d(probabilities)
            # # J = d(event_probs)/d(attenuations) * d(attenuations)/d(depolarizations)
            # #     * d(depolarizations)/d(polarizations) * d(polarizations)/d(probabilities)
            d_event_d_att = 0.5 * np.exp(-attenuations)
            d_att_d_dep = Winv
            d_dep_d_pol = -np.diag(1 / polarizations)
            d_pol_d_prob = build_masked_hadamard(pol_mask_arr, syndrome_masks)

            # Compute covariance of input probabilities (multinomial)
            cov_input = np.diag(probabilities) - np.outer(probabilities, probabilities)
            cov_input /= n_runs

            J = np.diag(d_event_d_att) @ d_att_d_dep @ d_dep_d_pol @ d_pol_d_prob
            covariance_matrix = J @ cov_input @ J.T
    else:
        if pol_masks is None:
            warnings.warn(
                f"fit_specified_dem: {n_bits} detectors exceed "
                f"_ALL_PAIRS_POL_LIMIT={_ALL_PAIRS_POL_LIMIT}; using the "
                "restricted default_polarization_masks set instead of all "
                "detector pairs. Pass pol_masks explicitly to control this."
            )
            pol_masks = default_polarization_masks(dem_masks, n_bits)
        event_probs, covariance_matrix = _fit_specified_dem_packed(
            syndrome_counts, dem_masks, pol_masks, n_bits, return_covariance,
        )

    if return_probs:
        if return_covariance:
            return dem_masks, event_probs, covariance_matrix
        return dem_masks, event_probs
    else:
        dem = dem_from_event_probabilities(event_probs, dem_masks, atol=atol)
        if return_covariance:
            return dem, dem_masks, event_probs, covariance_matrix
        return dem
