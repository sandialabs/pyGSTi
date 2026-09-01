import itertools
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from .interfaces import ErrorModel, Estimator, ForwardSimulator


class MalignantSetEstimator(Estimator):
    """Estimator that enumerates all error configurations up to a maximum weight.
    
    This provides a strict lower bound on the logical failure rate, which is 
    extremely tight at very low physical error rates p. It works by brute-force 
    iterating through all combinations of error mechanisms of size <= max_weight
    and summing the exact probabilities of the configurations that cause a 
    logical failure.
    """

    def estimate(
        self,
        error_model: ErrorModel,
        simulator: ForwardSimulator,
        p_scales: Sequence[float] | None = None,
        max_weight: int = 3,
        num_mechanisms: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Estimate failure rates by enumerating malignant sets.
        
        Args:
            error_model: The error model providing mechanism probabilities.
            simulator: The forward simulator and decoder.
            p_scales: Sequence of physical error rates to evaluate.
            max_weight: The maximum weight (number of simultaneous mechanisms) to explore.
            num_mechanisms: Total number of mechanisms in the error model. Must be 
                provided so the estimator knows how many mechanisms to combine.
        """
        if p_scales is None:
            raise ValueError("p_scales must be provided to MalignantSetEstimator")
        if num_mechanisms is None:
            raise ValueError("num_mechanisms must be provided to MalignantSetEstimator")

        malignant_sets: list[tuple[int, ...]] = []

        print(f"Enumerating malignant sets up to weight {max_weight} from {num_mechanisms} mechanisms...")
        
        # Enumerate and filter malignant sets
        for w in range(1, max_weight + 1):
            count_w = 0
            for combo in itertools.combinations(range(num_mechanisms), w):
                active = set(combo)
                if simulator.fails(active):
                    malignant_sets.append(combo)
                    count_w += 1
            print(f"Weight {w}: found {count_w} malignant sets.")

        # Evaluate probabilities for each p
        failure_estimates = []
        log_failure_estimates = []

        for p in p_scales:
            probs = error_model.probabilities(p)
            
            for combo in malignant_sets:
                # Calculate the exact probability of this specific configuration:
                # P(combo) = (prod_{i in combo} p_i) * (prod_{i not in combo} (1 - p_i))
                
                # To prevent underflow, we can compute this carefully, but at very low p,
                # the product of (1-p_i) is roughly exp(-sum p_i).
                
                # A robust way:
                log_p = 0.0
                for i in combo:
                    log_p += math.log(probs[i]) if probs[i] > 0 else -float('inf')
                
                # We also need to multiply by (1-p_i) for i NOT in combo.
                # It's faster to precalculate sum(log(1-p_i)) for ALL i, and then 
                # subtract the (1-p_i) for the items in combo.
                
                # Wait, doing this per combo inside the loop is slow if done naively.
                # Let's optimize the evaluation:
                pass
            
            # Optimization for evaluation:
            # log_prob_all_healthy = sum(log(1 - p_i))
            log_prob_all_healthy = np.sum(np.log1p(-probs))
            
            p_fail_p = 0.0
            for combo in malignant_sets:
                # P(combo) = P(all healthy) * prod_{i in combo} (p_i / (1 - p_i))
                weight = 1.0
                for i in combo:
                    if probs[i] == 1.0:
                        weight = float('inf') # Needs careful handling, but rare for low p
                    else:
                        weight *= probs[i] / (1.0 - probs[i])
                
                p_fail_p += math.exp(log_prob_all_healthy) * weight
                
            failure_estimates.append(p_fail_p)
            log_failure_estimates.append(math.log(p_fail_p) if p_fail_p > 0 else -float('inf'))
            
        return {
            "p_scales": list(p_scales),
            "failure_estimates": failure_estimates,
            "log_failure_estimates": log_failure_estimates,
            "malignant_sets": malignant_sets,
        }
