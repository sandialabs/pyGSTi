import stim
import numpy as np
from typing import Dict, Optional, Tuple

from .lattice import lattice_pruning_dem_estimation
from .cp_decomposition import cp_dem_estimation, CPConfig
from .tensor_contraction import log_likelihood as _tn_log_likelihood
from .estimation import (
    dense_dem_estimation,
    estimate_dem_and_covariance,
    threshold_probabilities,
    fit_specified_dem,
)
from .model_selection import lasso_select_events
from .io import dem_from_event_probabilities


class SparseDEMEstimator:
    def __init__(self, syndrome_counts: Dict[str, int]):
        """
        Initialize the estimator with syndrome count data.

        Args:
            syndrome_counts (dict): Mapping of bitstrings (e.g., '0011') to counts.
        """
        self.syndrome_counts = syndrome_counts
        self._dense_probs = None
        self._dense_covariance = None
        self._thresholded_probs = None
        self._threshold_mask = None
        self._lasso_info = None
        self._last_dem = None
        self._last_masks = None
        self._last_event_probs = None
        self._last_covariance = None
        self._last_method = None

    def _cache_results(
        self,
        method: str,
        dem: Optional[stim.DetectorErrorModel],
        masks: Optional[np.ndarray],
        event_probs: Optional[np.ndarray],
        covariance: Optional[np.ndarray],
    ) -> None:
        self._last_method = method
        self._last_dem = dem
        self._last_masks = masks
        self._last_event_probs = event_probs
        self._last_covariance = covariance

    def estimate_dense(self) -> stim.DetectorErrorModel:
        """
        Estimate a dense DEM using Hadamard-based inversion.

        Returns:
            stim.DetectorErrorModel: Estimated DEM.
        """
        self._dense_probs = dense_dem_estimation(self.syndrome_counts)
        self._dense_covariance = None
        dem = dem_from_event_probabilities(self._dense_probs)
        self._cache_results("dense", dem, None, self._dense_probs, None)
        return dem

    def estimate_dense_covariance(self) -> Tuple[stim.DetectorErrorModel, np.ndarray]:
        """
        Estimate a dense DEM and its covariance matrix.

        Returns:
            Tuple[stim.DetectorErrorModel, np.ndarray]: DEM and covariance matrix.
        """
        self._dense_probs, self._dense_covariance = estimate_dem_and_covariance(self.syndrome_counts)
        dem = dem_from_event_probabilities(self._dense_probs)
        self._cache_results("dense_covariance", dem, None, self._dense_probs, self._dense_covariance)
        return dem, self._dense_covariance

    def estimate_with_covariance(self) -> Tuple[stim.DetectorErrorModel, np.ndarray]:
        """
        Estimate a DEM and its covariance matrix.

        Returns:
            Tuple[stim.DetectorErrorModel, np.ndarray]: DEM and covariance matrix.
        """
        return self.estimate_dense_covariance()

    def threshold(self, alpha: float = 0.05) -> stim.DetectorErrorModel:
        """
        Threshold estimated probabilities using Bonferroni-corrected z-test.

        Args:
            alpha (float): Family-wise error rate.

        Returns:
            stim.DetectorErrorModel: Thresholded DEM.
        """
        if self._dense_probs is None or self._dense_covariance is None:
            self.estimate_dense_covariance()

        self._thresholded_probs, self._threshold_mask = threshold_probabilities(
            self._dense_probs, self._dense_covariance, alpha=alpha
        )
        dem = dem_from_event_probabilities(self._thresholded_probs)
        self._cache_results("threshold", dem, None, self._thresholded_probs, self._dense_covariance)
        return dem

    def estimate_lasso(
        self,
        lam: float = None,
        n_lambdas: int = 50,
        lambda_min_ratio: float = 1e-3,
        rcond: float = 1e-8,
        atol: float = 1e-4,
    ) -> stim.DetectorErrorModel:
        """
        Select a sparse DEM via non-negative whitened lasso with BIC, then
        refit the selected masks unpenalized for debiased probabilities.

        Args:
            lam (float, optional): Fixed penalty strength; skips the BIC path.
            n_lambdas (int): Number of points on the lambda path.
            lambda_min_ratio (float): Smallest lambda as a fraction of lambda_max.
            rcond (float): Relative eigenvalue cutoff for covariance whitening.
            atol (float): Tolerance for zeroing small probabilities.

        Returns:
            stim.DetectorErrorModel: Selected and refit DEM.
        """
        if self._dense_probs is None or self._dense_covariance is None:
            self.estimate_dense_covariance()

        n_shots = sum(self.syndrome_counts.values())
        masks, info = lasso_select_events(
            self._dense_probs,
            self._dense_covariance,
            n_shots,
            lam=lam,
            n_lambdas=n_lambdas,
            lambda_min_ratio=lambda_min_ratio,
            rcond=rcond,
        )
        self._lasso_info = info

        if len(masks) == 0:
            dem = stim.DetectorErrorModel()
            self._cache_results("lasso", dem, np.array([], dtype=int), np.zeros(0), np.zeros((0, 0)))
            return dem

        dem, dem_masks, event_probs, covariance = fit_specified_dem(
            self.syndrome_counts, masks, atol=atol, return_covariance=True
        )
        self._cache_results("lasso", dem, dem_masks, event_probs, covariance)
        return dem

    def estimate_lattice_pruned(
        self,
        confidence: float = 0.95,
        return_covariance: bool = False,
    ):
        """
        Estimate a DEM using the lattice pruning algorithm.

        Args:
            confidence (float): Confidence level for event detection.
            return_covariance (bool): Return masks, event probabilities, and covariance.

        Returns:
            stim.DetectorErrorModel or tuple: Estimated DEM and optional metadata.
        """
        if return_covariance:
            dem, dem_masks, event_probs, covariance = lattice_pruning_dem_estimation(
                self.syndrome_counts,
                confidence=confidence,
                return_covariance=True,
            )
            self._cache_results("lattice_pruned", dem, dem_masks, event_probs, covariance)
            return dem, dem_masks, event_probs, covariance

        dem = lattice_pruning_dem_estimation(self.syndrome_counts, confidence=confidence)
        self._cache_results("lattice_pruned", dem, None, None, None)
        return dem

    def estimate_cp(
        self,
        order: int = 3,
        rank: Optional[int] = None,
        config: Optional[CPConfig] = None,
        return_info: bool = False,
    ):
        """
        Estimate a DEM by symmetric CP decomposition of the joint cumulant tensor.

        Support discovery from the order-`order` joint cumulant tensor of the
        detector indicators (leading-order symmetric CP structure, see
        `cp_decomposition`), followed by an exact refit of the recovered masks
        with `fit_specified_dem`.

        Args:
            order (int): Cumulant order (3 recommended; 2 is the non-unique
                covariance / "p_ij" setting).
            rank (Optional[int]): Fixed CP rank; None selects it automatically.
            config (Optional[CPConfig]): Solver / rank-selection configuration.
            return_info (bool): Also return the diagnostic info dict.

        Returns:
            stim.DetectorErrorModel or tuple: Estimated DEM and optional info dict.
        """
        result = cp_dem_estimation(
            self.syndrome_counts,
            order=order,
            rank=rank,
            config=config,
            return_info=return_info,
        )
        if return_info:
            dem, info = result
            self._cache_results("cp", dem, info.get("masks"), None, None)
            return dem, info
        self._cache_results("cp", result, None, None, None)
        return result

    def log_likelihood(
        self,
        dem: Optional[stim.DetectorErrorModel] = None,
        backend: str = "auto",
    ) -> float:
        """
        Log-likelihood of the syndrome counts under a DEM, by tensor-network contraction.

        Uses `tensor_contraction.log_likelihood`, which evaluates exact outcome
        probabilities without the 2^n dense distribution, so it works for DEMs
        far beyond the reach of `compute_outcome_distribution_from_dem`.

        Args:
            dem (Optional[stim.DetectorErrorModel]): DEM to score; defaults to
                the most recently estimated DEM.
            backend (str): 'auto', 'numpy', or 'quimb'.

        Returns:
            float: sum over observed syndromes of count * log P(syndrome | dem).
        """
        if dem is None:
            dem = self._last_dem
        if dem is None:
            raise ValueError("No DEM given and no DEM has been estimated yet.")
        return float(_tn_log_likelihood(dem, self.syndrome_counts, backend=backend))

    def fit_custom_masks(
        self,
        masks: list[int],
        atol: float = 1e-4,
        return_probs: bool = False,
        return_covariance: bool = False,
    ):
        """
        Fit a DEM using a specified set of event masks.

        Args:
            masks (list[int]): List of integer bitmasks.
            atol (float): Tolerance for zeroing small probabilities.
            return_probs (bool): Return event probabilities.
            return_covariance (bool): Return covariance matrix.

        Returns:
            stim.DetectorErrorModel or tuple: Fitted DEM and optional metadata.
        """
        result = fit_specified_dem(
            self.syndrome_counts,
            masks,
            atol=atol,
            return_probs=return_probs,
            return_covariance=return_covariance,
        )

        if return_probs and return_covariance:
            dem_masks, event_probs, covariance = result
            self._cache_results("fit_custom_masks", None, dem_masks, event_probs, covariance)
            return dem_masks, event_probs, covariance
        if return_probs and not return_covariance:
            dem_masks, event_probs = result
            self._cache_results("fit_custom_masks", None, dem_masks, event_probs, None)
            return dem_masks, event_probs
        if not return_probs and return_covariance:
            dem, dem_masks, event_probs, covariance = result
            self._cache_results("fit_custom_masks", dem, dem_masks, event_probs, covariance)
            return dem, dem_masks, event_probs, covariance

        dem = result
        self._cache_results("fit_custom_masks", dem, None, None, None)
        return dem

    def get_dense_probabilities(self) -> Optional[np.ndarray]:
        return self._dense_probs

    def get_covariance_matrix(self) -> Optional[np.ndarray]:
        return self._dense_covariance

    def get_threshold_mask(self) -> Optional[np.ndarray]:
        return self._threshold_mask

    def get_lasso_info(self) -> Optional[dict]:
        return self._lasso_info

    def get_last_dem(self) -> Optional[stim.DetectorErrorModel]:
        return self._last_dem

    def get_last_masks(self) -> Optional[np.ndarray]:
        return self._last_masks

    def get_last_event_probabilities(self) -> Optional[np.ndarray]:
        return self._last_event_probs

    def get_last_covariance(self) -> Optional[np.ndarray]:
        return self._last_covariance

    def get_last_method(self) -> Optional[str]:
        return self._last_method
