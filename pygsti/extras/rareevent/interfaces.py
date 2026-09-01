from typing import Any, Protocol

import numpy as np


class ErrorModel(Protocol):
    """Protocol for a parameterized error model.
    
    Provides mechanism probabilities or parameters at a given physical error rate.
    """
    def probabilities(self, p: float) -> np.ndarray:
        """Return the probabilities of the independent error mechanisms at physical rate p."""
        ...

class ForwardSimulator(Protocol):
    """Protocol for a forward simulator and decoder.
    
    Translates an active set of error mechanisms into syndromes and logical observables,
    and determines if a decoding failure occurs. This encapsulates both the physical
    circuit simulation (like stim or clifft) and the decoding step.
    """
    def fails(self, active: set[int]) -> bool:
        """Return True if the active set of error mechanisms results in a logical failure."""
        ...

class Estimator(Protocol):
    """Protocol for an error rate estimator."""
    def estimate(
        self,
        error_model: ErrorModel,
        simulator: ForwardSimulator,
        **kwargs: Any
    ) -> Any:
        """Estimate the logical failure rate."""
        ...
