import numpy as np
import scipy as sp
from scipy.stats import norm
from scipy.linalg import hadamard
import stim 

from .utils import build_masked_hadamard, masked_hadamard_dot
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

def _as_exclusive_blocks(dem, num_detectors=None):
    """
    Normalize a DEM-with-exclusive-events description to a list of blocks.

    Each block is a list of (probability, detector_indices) branches; a block
    with a single branch is an ordinary independent event. Accepts a
    stim.DetectorErrorModel (all events independent), an object with
    ``.instructions`` / ``.num_detectors`` attributes (e.g. a
    HighRankDetectorErrorModel, whose instructions are events with
    ``.probability`` / ``.detectors`` or exclusive blocks with ``.events``),
    or a raw iterable of blocks of (probability, detector_indices) pairs.

    Returns:
        blocks: list[list[tuple[float, tuple[int, ...]]]]
        num_detectors: int
    """
    if isinstance(dem, stim.DetectorErrorModel):
        blocks = []
        for inst in dem.flattened():
            if inst.type == "error":
                dets = tuple(t.val for t in inst.targets_copy() if t.is_relative_detector_id())
                blocks.append([(inst.args_copy()[0], dets)])
        num_detectors = num_detectors or dem.num_detectors
    elif hasattr(dem, "instructions"):
        blocks = []
        for inst in dem.instructions:
            branches = inst.events if hasattr(inst, "events") else (inst,)
            blocks.append([(ev.probability, tuple(ev.detectors)) for ev in branches])
        num_detectors = num_detectors or dem.num_detectors
    else:
        blocks = [[(float(p), tuple(dets)) for p, dets in block] for block in dem]
        if num_detectors is None:
            num_detectors = 1 + max(
                (d for block in blocks for _, dets in block for d in dets), default=-1
            )

    for block in blocks:
        total = sum(p for p, _ in block)
        if any(p < 0 for p, _ in block) or total > 1 + 1e-9:
            raise ValueError(
                f"branch probabilities of an exclusive block must be nonnegative "
                f"and sum to <= 1 (got sum {total})"
            )
    return blocks, num_detectors


def compute_outcome_distribution_from_high_rank_dem(dem, num_detectors=None) -> np.ndarray:
    """
    Compute the outcome distribution of a DEM that may contain exclusive
    (high-rank) events, using the Hadamard/Fourier transform.

    An exclusive block with branch probabilities p_1..p_k applies at most one
    of its branches per shot (branch i with probability p_i, nothing with
    probability 1 - sum p_i). Distinct blocks and independent events act
    independently, so the syndrome distribution is their XOR-convolution and
    its Walsh-Hadamard transform factorizes into per-block polarizations:

        pol(s) = prod_blocks (1 - 2 * sum_{branches i with <s, f_i> odd} p_i)

    where f_i is the branch's detector-flip vector. For a single-branch block
    this reduces to the familiar independent-event factor (1 - 2 p)^{<s, f>},
    i.e. exp(-attenuation) as used by compute_outcome_distribution_from_dem.
    The product is taken directly rather than via -log, so blocks with
    zero or negative polarization (odd-overlap mass >= 1/2) are handled too.

    Logical observable targets, if present, are ignored; the returned
    distribution is over detector outcomes only.

    Parameters:
        dem: stim.DetectorErrorModel, or HighRankDetectorErrorModel-like
            object, or iterable of blocks of (probability, detector_indices)
            branches (see _as_exclusive_blocks).
        num_detectors: int, optional
            Overrides / provides the number of detectors.

    Returns:
        probabilities: np.ndarray
            Array of 2^n probabilities, in increasing binary order (D0 is the
            least significant bit of the outcome index).
    """
    blocks, n = _as_exclusive_blocks(dem, num_detectors)
    size = 2 ** n
    all_bitstrings = np.array([
        [int(bit) for bit in format(i, f"0{n}b")]
        for i in range(size)
    ])

    polarizations = np.ones(size, dtype=float)
    for block in blocks:
        odd_overlap_mass = np.zeros(size, dtype=float)
        for prob, dets in block:
            event_vec = [1 if (n - idx - 1) in dets else 0 for idx in range(n)]
            odd_overlap_mass += prob * (1 - (-1) ** np.dot(all_bitstrings, event_vec)) / 2
        polarizations *= 1 - 2 * odd_overlap_mass

    probabilities = hadamard(size) @ polarizations / size
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
    # Streamed product: the dense (pol_masks x syndromes) Hadamard submatrix
    # is only materialized when the covariance (which needs the full
    # Jacobian) is requested.
    polarizations = masked_hadamard_dot(pol_masks, syndrome_masks, counts) / n_runs
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
        d_pol_d_prob = build_masked_hadamard(pol_masks, syndrome_masks)

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
