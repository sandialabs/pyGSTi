"""
I/O utilities for converting between external representations and stim DEMs.

This module focuses on parsing/serializing DEMs and related formats (strings,
dicts, matrices, probability vectors). Pure in-memory helpers live in
`sparsedem.utils`.
"""

import stim
import io
import numpy as np
from .utils import bits_to_binary_number


def dem_from_str(dem_str: str) -> stim.DetectorErrorModel:
    """
    Convert dem string to stim DetectorErrorModel.

    Parameters:
        dem_str: str
            Properly formatted DEM string.

    Returns:
        stim.DetectorErrorModel
    """
    dem_file = io.StringIO(dem_str)
    dem = stim.DetectorErrorModel.from_file(dem_file)
    return dem


def dem_to_dict(dem: stim.DetectorErrorModel) -> dict:
    """
    Convert stim dem object to python dictionary. Keys are integers corresponding
    to DEM event labels. Eg., D0 D4 = 2^0 + 2^4 = 10001b = 17.

    Parameters:
        dem: stim DetectorErrorModel
    Returns:
        dem_dict: dict representation of dem
    """
    dem_dict = {}
    for event in dem:
        p = event.args_copy()[0]
        targets = event.targets_copy()
        label = sum([1 << targ.val for targ in targets])
        if label in dem_dict:
            p0 = dem_dict[label]
            dem_dict[label] = 0.5 - 0.5 * np.exp(np.log(1 - 2 * p0) + np.log(1 - 2 * p))
        else:
            dem_dict[label] = p
    return dem_dict

def dem_from_matrix(B: np.ndarray, probs: np.ndarray) -> stim.DetectorErrorModel:
    """
    Convert a bitmatrix and list of probabilities to a stim DetectorErrorModel.
    
    Parameters:
        B: np.ndarray
            MxN matrix of binary event vectors (MSB-first ordering per column).
        probs: np.ndarray
            N-vector of corresponding probabilities

    Returns:
        stim.DetectorErrorModel
    """

    dem_dict = {bits_to_binary_number(b): p for b, p in zip(B.T, probs)}
    return dem_from_dict(dem_dict)

def dem_to_matrix(dem: stim.DetectorErrorModel) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a stim DetectorErrorModel to a bitmatrix and list of probabilities.

    Parameters:
        dem: stim.DetectorErrorModel

    Returns:
        B: np.ndarray
            MxN matrix of binary event vectors (MSB-first ordering per column).
        probs: np.ndarray
            N-vector of corresponding probabilities.
    """
    dem_dict = dem_to_dict(dem)
    if not dem_dict:
        return np.zeros((dem.num_detectors, 0), dtype=np.uint8), np.array([], dtype=float)
    n_bits = dem.num_detectors
    items = sorted(dem_dict.items())
    probs = np.array([p for _, p in items], dtype=float)
    B = np.zeros((n_bits, len(items)), dtype=np.uint8)
    for j, (mask, _) in enumerate(items):
        for i in range(n_bits):
            B[i, j] = (mask >> (n_bits-i-1)) & 1
    return B, probs

def dem_from_dict(dem_dict: dict) -> stim.DetectorErrorModel:
    """
    Convert a properly formatted dictionary to a stim DetectorErrorModel.

    Parameters:
        dem_dict: dict
            A formatted dem_dict (as is returned from dem_to_dict())

    Returns:
        stim.DetectorErrorModel
    """
    assert 0 not in dem_dict, "Dictionary contains event labeled 0, which flips no bits"
    dem_dict = dict(sorted(dem_dict.items()))
    n_bits = len(f"{max(dem_dict):0b}")
    dem_str = ""
    for event, p in dem_dict.items():
        event_in_binary = f"{event:0{n_bits}b}"
        event_label = " ".join([f"D{ind}" for ind in range(n_bits) if event_in_binary[-ind-1] == '1'])
        dem_str += f"error({p}) {event_label}\n"
    return dem_from_str(dem_str)


def dem_to_event_probabilities(dem: stim.DetectorErrorModel) -> np.ndarray:
    """
    Convert a stim DEM object into a 2^n dimensional list of event probabilities.

    Parameters:
        dem: stim DetectorErrorModel

    Returns:
        dem_event_probabilities: np.ndarray of shape (2^n,)
    """
    n = dem.num_detectors
    attenuations = np.zeros(2**n, dtype=float)
    for event in dem:
        probability = event.args_copy()[0]
        targets = [target.val for target in event.targets_copy()]
        idxs = [1 << target for target in targets]
        idx = sum(idxs)
        attenuations[idx] += -np.log(1 - 2 * probability)
    probabilities = 0.5 - 0.5 * np.exp(-attenuations)
    return probabilities


def dem_from_event_probabilities(
    event_probabilities: np.ndarray,
    masks=None,
    atol=1e-4,
) -> stim.DetectorErrorModel:
    """
    Convert array of detector event probabilities to a stim DetectorErrorModel.

    Parameters:
        event_probabilities: np.ndarray
            Array of event probabilities.
        masks: Optional[np.ndarray]
            Array of integers labeling event probabilities.
        atol: float
            Threshold below which probabilities are set to zero.

    Returns:
        stim.DetectorErrorModel
    """
    n_entries = len(event_probabilities)
    if masks is None:
        assert np.isclose(event_probabilities[0], 0, atol=atol), "Null event has nonzero probability."
        assert n_entries > 0 and (n_entries & (n_entries - 1)) == 0, "Length of event_probability vector must be a power of 2"
        n_bits = int(np.log2(n_entries))
    else:
        assert 0 not in masks, "0 is an invalid mask"
        assert n_entries == len(masks), "Length of masks must equal length of event_probabilities"
        n_bits = len(f"{max(masks):0b}")

    dem_str = ""
    for ind, probability in enumerate(event_probabilities):
        if probability > 0 and not np.isclose(probability, 0, atol=atol):
            if masks is None:
                targets = f"{ind:0{n_bits}b}"
            else:
                targets = f"{masks[ind]:0{n_bits}b}"
            # Note reversed rows:
            dem_str += f"error({probability}) " + " ".join([f"D{idx}" for idx, val in enumerate(reversed(targets)) if int(val)]) + "\n"

    return dem_from_str(dem_str)
