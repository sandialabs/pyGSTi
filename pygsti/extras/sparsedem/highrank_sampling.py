"""Samplers for high-rank detector error models.

Two implementations:

* :class:`CompiledHighRankSampler` — the fast path.  Compiles the model into
  a stim ``Circuit`` built from ``CORRELATED_ERROR`` / ``ELSE_CORRELATED_ERROR``
  chains and samples it with stim's compiled frame simulator.  Stim's
  ``ELSE_CORRELATED_ERROR`` applies its error *only if no earlier error in the
  chain fired*, with the stated probability conditioned on that — exactly the
  semantics of an ``exclusive(k)`` block.  A block with branch probabilities
  ``p_1..p_k`` therefore compiles to a chain with conditional probabilities

      q_i = p_i / (1 - p_1 - ... - p_{i-1})

  which reproduces the joint distribution exactly (not approximately).  All
  per-shot work happens inside stim's C++ core, so throughput matches stim's
  native DEM sampler.

* :class:`NumpyReferenceSampler` — a slow, transparent numpy implementation
  used to validate the compiled sampler.
"""

from __future__ import annotations

import numpy as np
import stim

from .highrank import ErrorEvent, ExclusiveBlock, HighRankDetectorErrorModel

__all__ = [
    "to_stim_sampling_circuit",
    "CompiledHighRankSampler",
    "NumpyReferenceSampler",
]


def _chain_conditional_probs(block: ExclusiveBlock) -> list[float]:
    """Converts branch probabilities to ELSE-chain conditional probabilities."""
    qs = []
    remaining = 1.0
    for ev in block.events:
        if remaining <= 0.0:
            qs.append(0.0)
            continue
        q = ev.probability / remaining
        qs.append(min(q, 1.0))
        remaining -= ev.probability
    return qs


def to_stim_sampling_circuit(model: HighRankDetectorErrorModel) -> stim.Circuit:
    """Compiles a high-rank DEM into an equivalent stim sampling circuit.

    Detector ``d`` maps to qubit ``d`` and observable ``o`` to qubit
    ``num_detectors + o``.  Every event becomes a correlated X error on the
    qubits it flips; exclusive blocks become E / ELSE_CORRELATED_ERROR chains.
    Sampling the circuit's detectors/observables is distributed identically
    to sampling the model.
    """
    n_det = model.num_detectors
    n_obs = model.num_observables
    n_meas = n_det + n_obs
    # Events with no targets still need a target for stim's parser; they get
    # a scratch qubit that is never measured, so they only consume
    # probability mass in their chain.
    junk_qubit = n_meas

    def x_targets(ev: ErrorEvent) -> str:
        qubits = list(ev.detectors) + [n_det + o for o in ev.observables]
        if not qubits:
            qubits = [junk_qubit]
        return " ".join(f"X{q}" for q in qubits)

    lines: list[str] = []
    for inst in model.instructions:
        if isinstance(inst, ExclusiveBlock):
            qs = _chain_conditional_probs(inst)
            for i, (ev, q) in enumerate(zip(inst.events, qs)):
                gate = "E" if i == 0 else "ELSE_CORRELATED_ERROR"
                lines.append(f"{gate}({q!r}) {x_targets(ev)}")
        else:
            lines.append(f"E({inst.probability!r}) {x_targets(inst)}")

    if n_meas:
        lines.append("M " + " ".join(str(q) for q in range(n_meas)))
    for d in range(n_det):
        coords = model.detector_coords.get(d, ())
        args = f"({', '.join(repr(c) for c in coords)})" if coords else ""
        lines.append(f"DETECTOR{args} rec[{d - n_meas}]")
    for o in range(n_obs):
        lines.append(f"OBSERVABLE_INCLUDE({o}) rec[{n_det + o - n_meas}]")

    return stim.Circuit("\n".join(lines))


class CompiledHighRankSampler:
    """Fast sampler backed by stim's compiled frame simulator.

    Usage::

        sampler = CompiledHighRankSampler(model, seed=0)
        dets, obs = sampler.sample(10_000)
    """

    def __init__(self, model: HighRankDetectorErrorModel, *, seed: int | None = None):
        self.model = model
        self.circuit = to_stim_sampling_circuit(model)
        self._sampler = self.circuit.compile_detector_sampler(seed=seed)

    def sample(
        self, shots: int, *, bit_packed: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Samples ``shots`` shots.

        Returns ``(detectors, observables)`` with shapes
        ``(shots, num_detectors)`` and ``(shots, num_observables)`` (bit-packed
        along the last axis if ``bit_packed=True``), matching the conventions
        of ``stim.CompiledDetectorSampler.sample(separate_observables=True)``.
        """
        return self._sampler.sample(
            shots, separate_observables=True, bit_packed=bit_packed
        )


class NumpyReferenceSampler:
    """Slow but transparent sampler used to cross-validate the compiled one.

    Independent errors are sampled as Bernoulli draws; each exclusive block
    draws a single uniform variate and picks the branch via its cumulative
    distribution, guaranteeing at most one branch fires.
    """

    def __init__(self, model: HighRankDetectorErrorModel, *, seed: int | None = None):
        self.model = model
        self._rng = np.random.default_rng(seed)
        n_det = model.num_detectors
        n_obs = model.num_observables
        width = n_det + n_obs

        def row(ev: ErrorEvent) -> np.ndarray:
            r = np.zeros(width, dtype=np.uint8)
            r[list(ev.detectors)] = 1
            r[[n_det + o for o in ev.observables]] = 1
            return r

        ind = model.independent_errors
        self._ind_probs = np.array([e.probability for e in ind], dtype=np.float64)
        self._ind_rows = (
            np.array([row(e) for e in ind], dtype=np.uint8)
            if ind
            else np.zeros((0, width), dtype=np.uint8)
        )

        self._blocks: list[tuple[np.ndarray, np.ndarray]] = []
        for blk in model.exclusive_blocks:
            cum = np.cumsum([e.probability for e in blk.events])
            # Row k (one past the last branch) is the "nothing happened" row.
            rows = np.vstack(
                [row(e) for e in blk.events] + [np.zeros(width, dtype=np.uint8)]
            )
            self._blocks.append((cum, rows))

    def sample(self, shots: int) -> tuple[np.ndarray, np.ndarray]:
        n_det = self.model.num_detectors
        width = n_det + self.model.num_observables
        flips = np.zeros((shots, width), dtype=np.int64)

        if len(self._ind_probs):
            fired = self._rng.random((shots, len(self._ind_probs))) < self._ind_probs
            flips += fired.astype(np.int64) @ self._ind_rows.astype(np.int64)

        for cum, rows in self._blocks:
            u = self._rng.random(shots)
            idx = np.searchsorted(cum, u, side="right")  # == len(cum) -> nothing
            flips += rows[idx].astype(np.int64)

        bits = (flips & 1).astype(bool)
        return bits[:, :n_det], bits[:, n_det:]
