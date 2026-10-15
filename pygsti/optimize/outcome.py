"""Algorithm-independent termination information."""

from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class OptimizationStatus(str, Enum):
    """Why a solve terminated; success is not inferred from a message string."""

    CONVERGED = 'converged'
    ITERATION_LIMIT = 'iteration_limit'
    EVALUATION_LIMIT = 'evaluation_limit'
    DOMAIN_LIMIT = 'domain_limit'
    STALLED = 'stalled'
    INTERRUPTED = 'interrupted'
    FAILED = 'failed'


@dataclass
class OptimizerOutcome:
    """An optimizer's proposed final point and its termination information.

    Final objective statistics are evaluated by the GST integration layer.
    ``diagnostics`` and ``restart_state`` belong to the algorithm and are never
    interpreted as mathematical convergence tests by orchestration.
    """

    x: object
    status: OptimizationStatus
    message: str = ''
    diagnostics: dict = field(default_factory=dict)
    restart_state: object = None

    def __post_init__(self):
        if np.iscomplexobj(self.x):
            raise ValueError('The returned optimization point must be real')
        self.x = np.array(self.x, dtype=float, copy=True)
        if self.x.ndim != 1 or not np.all(np.isfinite(self.x)):
            raise ValueError('The returned optimization point must be a finite vector')
        self.status = OptimizationStatus(self.status)
        self.diagnostics = dict(self.diagnostics)
