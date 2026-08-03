import numpy as np
from collections import Counter
from scipy.stats import norm

from .estimation import estimate_dem_and_covariance, fit_specified_dem


def marginalize_syndrome_counts(syndrome_counts, bitmask):
    """
    Take dictionary of observed syndromes and marginalize over a subset of bits.

    Parameters:
        syndrome_counts: dict
            Observed n-bit syndrome data.
        bitmask: str
            n-bit string, indicates bits to keep.

    Returns:
        marginalized_syndrome_counts: dict
            Syndrome data marginalized over bitmask.
    """
    marginalized_counts = {}
    indices_to_keep = [i for i, b in enumerate(bitmask) if b == '1']
    for syndrome, count in syndrome_counts.items():
        marginalized_syndrome = ''.join(syndrome[i] for i in indices_to_keep)
        marginalized_counts[marginalized_syndrome] = marginalized_counts.get(marginalized_syndrome, 0) + count
    return marginalized_counts


def check_event_mask(bitmask, syndrome_counts, alpha=0.05):
    """
    Check if there is evidence of DEM events that flip all bits in the bitmask simultaneously.
    Uses a one-sided z-test. Not currently propagating any confidence through the bitmask trie.

    Parameters:
        bitmask: str or int
            Bitmask indicating the bits to keep.
        syndrome_counts: dict
            Observed n-bit syndrome data.
        alpha: float
            Statistical confidence level.

    Returns:
        event_present: bool
            True if statistically significant evidence of event.
    """
    if isinstance(bitmask, int):
        n_bits = len(next(iter(syndrome_counts)))
        bitmask = f"{bitmask:0{n_bits}b}"

    subset_counts = marginalize_syndrome_counts(syndrome_counts, bitmask)
    event_probs, cov_matrix = estimate_dem_and_covariance(subset_counts)
    p_hat = event_probs[-1]
    var = cov_matrix[-1, -1]
    if not np.isfinite(var) or var <= 0:
        return False
    z = p_hat / np.sqrt(var)
    z_threshold = norm.ppf(1 - alpha)
    return z > z_threshold


def bitmask_trie_search(n, check_flip):
    """
    Discover valid flip events using a bitmask trie.
    # TODO: add significance passing logic here. 
    # Will need to make check_flip accept a significance parameter, and will need
    # to keep a dictionary of significances. Look up "Closed hypothesis testing."
    #  
    
    Parameters:
        n: int
            Number of bits in each observation.
        check_flip: Callable[[int], bool]
            Function that returns True if the bits in the mask flip together.

    Returns:
        Set[int]: Valid event bitmasks.
    """
    valid_events = set()

    def dfs(current_mask, bit_index):
        if bit_index == n:
            return
        next_mask = current_mask | (1 << bit_index)
        if check_flip(next_mask):
            valid_events.add(next_mask)
            dfs(next_mask, bit_index + 1)
        dfs(current_mask, bit_index + 1)

    dfs(0, 0)
    return valid_events


def lattice_pruning_dem_estimation(syndrome_counts, confidence=0.95, return_covariance = False):
    """
    Estimate a sparse DEM from syndrome counts using the lattice pruning algorithm.

    Parameters:
        syndrome_counts: dict
            Mapping bitstrings (e.g., '0011') to counts.
        confidence: float
            Statistical confidence level for event detection.

    Returns:
        stim.DetectorErrorModel
    """
    n_bits = len(next(iter(syndrome_counts)))
    check_flip = lambda m: check_event_mask(m, syndrome_counts, alpha=1 - confidence)

    masks = bitmask_trie_search(n_bits, check_flip)

    return fit_specified_dem(syndrome_counts, masks, return_covariance=return_covariance)
    