import stim
import numpy as np
from typing import Dict, Tuple, Optional

from .lattice import lattice_pruning_dem_estimation
from .estimation import (
    dense_dem_estimation,
    estimate_dem_and_covariance,
    threshold_probabilities,
    fit_specified_dem,
)
from .io import (
    dem_from_event_probabilities,
    dem_to_event_probabilities,
)


class SparseDEMEstimator:
    def __init__(self, syndrome_counts: Dict[str, int]):
        """
        Initialize the estimator with syndrome count data.

        Args:
            syndrome_counts (dict): Mapping of bitstrings (e.g., '0011') to counts.
        """
        self.syndrome_counts = syndrome_counts
        self._dense_probs = None
        self._covariance = None
        self._thresholded_probs = None
        self._mask = None
        
    def estimate_dense(self) -> stim.DetectorErrorModel:
        """
        Estimate a dense DEM using Hadamard-based inversion.

        Returns:
            stim.DetectorErrorModel: Estimated DEM.
        """
        self._dense_probs = dense_dem_estimation(self.syndrome_counts)
        return dem_from_event_probabilities(self._dense_probs)

    def estimate_with_covariance(self) -> Tuple[stim.DetectorErrorModel, np.ndarray]:
        """
        Estimate a DEM and its covariance matrix.

        Returns:
            Tuple[stim.DetectorErrorModel, np.ndarray]: DEM and covariance matrix.
        """
        self._dense_probs, self._covariance = estimate_dem_and_covariance(self.syndrome_counts)
        dem = dem_from_event_probabilities(self._dense_probs)
        return dem, self._covariance

    def threshold(self, alpha: float = 0.05) -> stim.DetectorErrorModel:
        """
        Threshold estimated probabilities using Bonferroni-corrected z-test.

        Args:
            alpha (float): Family-wise error rate.

        Returns:
            stim.DetectorErrorModel: Thresholded DEM.
        """
        if self._dense_probs is None or self._covariance is None:
            self.estimate_with_covariance()

        self._thresholded_probs, self._mask = threshold_probabilities(
            self._dense_probs, self._covariance, alpha=alpha
        )
        return dem_from_event_probabilities(self._thresholded_probs)

    def estimate_lattice_pruned(self, confidence: float = 0.95) -> stim.DetectorErrorModel:
        """
        Estimate a DEM using the lattice pruning algorithm.

        Args:
            confidence (float): Confidence level for event detection.

        Returns:
            stim.DetectorErrorModel: Estimated DEM.
        """
        return lattice_pruning_dem_estimation(self.syndrome_counts, confidence=confidence)

    def fit_custom_masks(self, masks: list[int], atol: float = 1e-4) -> stim.DetectorErrorModel:
        """
        Fit a DEM using a specified set of event masks.

        Args:
            masks (list[int]): List of integer bitmasks.
            atol (float): Tolerance for zeroing small probabilities.

        Returns:
            stim.DetectorErrorModel: Fitted DEM.
        """
        return fit_specified_dem(self.syndrome_counts, masks, atol=atol)

    def get_dense_probabilities(self) -> Optional[np.ndarray]:
        return self._dense_probs

    def get_covariance_matrix(self) -> Optional[np.ndarray]:
        return self._covariance

    def get_threshold_mask(self) -> Optional[np.ndarray]:
        return self._mask
