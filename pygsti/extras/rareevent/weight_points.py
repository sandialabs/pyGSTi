"""Common output contract for low-weight failure-structure estimators.

The four malignant-set counting routines (fixed-weight gap-splitting,
connected-cluster enumeration, Knuth tree-size estimation, core-planting
importance sampling) all report their results as ``WeightPoint`` values so
that downstream consumers (the failure-spectrum fit, the splitting
estimator's diagnostics, benchmark plots) can treat them uniformly.

Two kinds of point are supported:

- ``kind="f_w"``: ``estimate`` is f(w) = P(fail | W = w), the failure
  fraction of weight-``w`` fault sets under the exact conditional
  distribution P(E | |E| = w) at the reference physical rate.
- ``kind="m_v"``: ``estimate`` is m(v), the number of minimal malignant
  clusters of weight ``v`` (connected in the shared-detector graph; no
  proper nonempty subset fails).
"""

from __future__ import annotations

import dataclasses
from typing import Any

KINDS = ("f_w", "m_v")


@dataclasses.dataclass(frozen=True)
class WeightPoint:
    """One estimated point of the low-weight failure structure.

    Attributes:
        method: Short method slug, e.g. "gap_splitting" or "core_planting".
        kind: "f_w" (failure fraction) or "m_v" (malignant cluster count).
        weight: The fault-set weight w (for f_w) or cluster weight v (for m_v).
        estimate: The estimated value; may be 0.0 (e.g. exact count of zero).
        rel_err: Relative standard error of ``estimate``; 0.0 for exact values.
        exact: True when the value is an exact count/probability, not sampled.
        lower_bound: True when the value is a certified lower bound (e.g. an
            importance-sampling estimate over an incomplete core list).
        meta: Free-form method-specific diagnostics (trial counts, level
            thresholds, acceptance rates, ...). Must be JSON-serializable.
    """

    method: str
    kind: str
    weight: int
    estimate: float
    rel_err: float
    exact: bool = False
    lower_bound: bool = False
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}; got {self.kind!r}.")
        if self.weight < 0:
            raise ValueError(f"weight must be nonnegative; got {self.weight}.")
        if self.estimate < 0:
            raise ValueError(f"estimate must be nonnegative; got {self.estimate}.")
        if self.rel_err < 0:
            raise ValueError(f"rel_err must be nonnegative; got {self.rel_err}.")


def weight_point_record(
    point: WeightPoint,
    *,
    d: int,
    p_ref: float,
    seed: int | None = None,
    time_s: float | None = None,
) -> dict[str, Any]:
    """Serialize a WeightPoint into the shared JSONL record shape.

    Every record carries the code distance and the reference physical rate at
    which the point was measured, so records from different methods and runs
    can be pooled from a single results file.
    """
    record: dict[str, Any] = {
        "method": point.method,
        "kind": point.kind,
        "weight": point.weight,
        "estimate": point.estimate,
        "rel_err": point.rel_err,
        "exact": point.exact,
        "lower_bound": point.lower_bound,
        "d": d,
        "p_ref": p_ref,
        "meta": point.meta,
    }
    if seed is not None:
        record["seed"] = seed
    if time_s is not None:
        record["time_s"] = time_s
    return record
