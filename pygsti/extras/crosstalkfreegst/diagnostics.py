#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""
Joint (many-qubit) goodness of fit and prediction for crosstalk-free models.

The fit uses lane-marginal counts, so its own objective cannot see errors that correlate the
outcomes of different lanes -- e.g. crosstalk between qubits in different lanes of the same
circuit.  The joint likelihood can, and for a crosstalk-free model it is still cheap: each joint
outcome's probability is a product of lane probabilities.
"""
import itertools as _itertools
from collections import defaultdict as _defaultdict
from typing import Dict, Sequence

import numpy as _np

from .lanemodels import LaneModels
from .lanes import LaneDecomposition, split_circuit_into_lanes


def _lane_tables(lane_models: LaneModels, circuits, instrument_names):
    """Per-circuit (lanes, instrument qubits), and ``(support, lane circuit) -> {lane outcome: p}``."""
    qubits = lane_models.qubit_labels
    decomposer = LaneDecomposition(qubits, instrument_names=instrument_names)
    splits, by_support = [], _defaultdict(dict)
    for circuit in circuits:
        lanes = split_circuit_into_lanes(circuit, qubits)
        splits.append((lanes, decomposer._instrument_qubits(circuit)))
        for support, lane_circuit in lanes:
            by_support[support][lane_circuit] = None
    probs = {}
    for support, lane_circuits in by_support.items():
        lane_model, _ = lane_models[support]
        for lc, p in lane_model.sim.bulk_probs(list(lane_circuits)).items():
            probs[support, lc] = p
    return splits, probs


def _lane_outcome_readers(circuit, lanes, inst_qubits):
    """
    For each lane of `circuit`: which instrument outcomes and which final bits of a joint outcome
    label make up the lane's outcome label, as ``(instrument positions, bit positions)``.
    """
    lpos = {q: i for i, q in enumerate(circuit.line_labels)}
    return [([j for j, q in enumerate(inst_qubits) if q in support], [lpos[q] for q in support])
            for support, _ in lanes]


def _lane_outcome(outcome, reader):
    slots, bits = reader
    return tuple(outcome[j] for j in slots) + (''.join(outcome[-1][i] for i in bits),)


def joint_loglikelihood(lane_models: LaneModels, dataset, circuits: Sequence = None, min_prob: float = 1e-12):
    """
    The exact joint log-likelihood ``sum_c sum_x N_c(x) log p_c(x)`` of `dataset` under `lane_models`.

    Each joint outcome probability is the product of its lanes' probabilities, so only the outcomes
    that occur in the data are ever evaluated, and nothing larger than a lane is simulated.
    Probabilities are clipped below at `min_prob`.

    Parameters
    ----------
    lane_models : LaneModels
        Set to the parameters to evaluate.

    dataset : DataSet
        Joint counts.  Outcome labels follow pyGSTi's convention (instrument outcomes, then the
        final bits over each circuit's lines in ``line_labels`` order).

    circuits : sequence of Circuit, optional
        Defaults to every circuit in `dataset`.

    Returns
    -------
    float
    """
    circuits = list(dataset.keys()) if circuits is None else list(circuits)
    names = lane_models.processor_spec.instrument_names
    splits, probs = _lane_tables(lane_models, circuits, names)
    total = 0.0
    for circuit, (lanes, inst_qubits) in zip(circuits, splits):
        tables = [probs[lane] for lane in lanes]
        readers = _lane_outcome_readers(circuit, lanes, inst_qubits)
        for outcome, n in dataset[circuit].counts.items():
            if n == 0:
                continue
            outcome = outcome if isinstance(outcome, tuple) else (outcome,)
            p = 1.0
            for table, reader in zip(tables, readers):
                p *= table[_lane_outcome(outcome, reader)]
            total += n * _np.log(max(p, min_prob))
    return total


def saturated_loglikelihood(dataset, circuits: Sequence = None) -> float:
    """``sum_c sum_x N_c(x) log(N_c(x) / N_c)``: the log-likelihood of the observed frequencies."""
    circuits = list(dataset.keys()) if circuits is None else list(circuits)
    total = 0.0
    for circuit in circuits:
        counts = _np.array([n for n in dataset[circuit].counts.values() if n > 0], dtype=float)
        total += float(_np.sum(counts * _np.log(counts / counts.sum())))
    return total


def joint_goodness_of_fit(lane_models: LaneModels, dataset, circuits: Sequence = None) -> Dict:
    """
    The joint log-likelihood-ratio statistic ``2 (log L_saturated - log L_model)``.

    Unlike the fit's lane-marginal objective, this sees errors that correlate outcomes across lanes.
    With many qubits most joint outcomes are rare, so its asymptotic chi^2 distribution is a poor
    guide; calibrate it instead against its distribution over data simulated from in-class models.

    Returns
    -------
    dict
        ``'two_delta_logl'``, ``'logl_model'``, ``'logl_saturated'``, and ``'num_observed_outcomes'``
        (the number of distinct (circuit, outcome) pairs with nonzero counts).
    """
    circuits = list(dataset.keys()) if circuits is None else list(circuits)
    model = joint_loglikelihood(lane_models, dataset, circuits)
    saturated = saturated_loglikelihood(dataset, circuits)
    observed = sum(sum(1 for n in dataset[c].counts.values() if n > 0) for c in circuits)
    return {'two_delta_logl': 2 * (saturated - model), 'logl_model': model, 'logl_saturated': saturated,
            'num_observed_outcomes': observed}


def predicted_distributions(lane_models: LaneModels, circuits: Sequence) -> Dict:
    """
    The full outcome distribution of each circuit, as products of lane distributions.

    Enumerates every joint outcome, so it is for circuits on modest numbers of qubits (e.g. held-out
    prediction tests against full simulation).  Labels follow pyGSTi's convention.

    Returns
    -------
    dict
        Circuit -> {outcome label: probability}.
    """
    circuits = list(circuits)
    names = lane_models.processor_spec.instrument_names
    splits, probs = _lane_tables(lane_models, circuits, names)
    out = {}
    for circuit, (lanes, inst_qubits) in zip(circuits, splits):
        lpos = {q: i for i, q in enumerate(circuit.line_labels)}
        lane_items = [list(probs[support, lc].items()) for support, lc in lanes]
        dist = {}
        for combo in _itertools.product(*lane_items):
            p = 1.0
            insts = [None] * len(inst_qubits)
            bits = ['?'] * circuit.num_lines
            for (support, _), (lane_outcome, lane_p) in zip(lanes, combo):
                p *= lane_p
                slots = [j for j, q in enumerate(inst_qubits) if q in support]
                for j, o in zip(slots, lane_outcome[:-1]):
                    insts[j] = o
                for q, b in zip(support, lane_outcome[-1]):
                    bits[lpos[q]] = b
            dist[tuple(insts) + (''.join(bits),)] = p
        out[circuit] = dist
    return out


def total_variation_distance(p: Dict, q: Dict) -> float:
    """``0.5 * sum_x |p(x) - q(x)|`` over the union of the two distributions' outcomes."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def _binomial_deviance_moments(lam: _np.ndarray, N: float, width: float = 12.0):
    """
    Mean and variance of ``t(n) = lam - n + n log(n / lam)`` for ``n ~ Binomial(N, lam / N)``, per outcome.

    Sums the binomial pmf over ``n`` within `width` standard deviations of the mean (plus a margin for
    small means), which captures all but a negligible part of the distribution.
    """
    from scipy.stats import binom
    p = _np.clip(lam / N, 0.0, 1.0)
    sd = _np.sqrt(N * p * (1 - p))
    lo = _np.maximum(0, _np.floor(lam - width * sd - 10)).astype(int)
    hi = _np.minimum(int(N), _np.ceil(lam + width * sd + 10)).astype(int)
    means, variances = _np.zeros(len(lam)), _np.zeros(len(lam))
    for i in range(len(lam)):
        if lam[i] <= 0:
            continue
        n = _np.arange(lo[i], hi[i] + 1, dtype=float)
        w = binom.pmf(n, int(N), p[i])
        with _np.errstate(divide='ignore', invalid='ignore'):
            t = lam[i] - n + _np.where(n > 0, n * _np.log(n / lam[i]), 0.0)
        means[i] = _np.dot(w, t)
        variances[i] = _np.dot(w, (t - means[i])**2)
    return means, variances


def logl_statistic_moments(lane_models: LaneModels, lane_data: Dict) -> Dict:
    """
    Finite-sample expectation and variance of pyGSTi's log-likelihood statistic (Delta logL, Poisson
    picture, summed over every possible outcome of every lane circuit) for data drawn from `lane_models`.

    For a correct model with many counts per outcome the statistic has mean and variance ~ k/2,
    k = (independent outcome frequencies) - (parameters).  When many outcomes have small expected
    counts -- e.g. lane circuits with mid-circuit measurements, whose outcome spaces grow with the
    number of measurements -- each outcome contributes less than the asymptotic 1/2, and the
    statistic falls well below k/2.  This computes, per outcome, the exact mean and variance of its
    term under the outcome's binomial marginal at the model's probabilities.  The mean is exact (by
    linearity); the variance sums the per-outcome variances, rescaled per lane circuit so that the
    large-count limit is the exact (m - 1)/2 for m outcomes (outcomes of one circuit are correlated).
    The parameter-count correction is left to the caller.

    Parameters
    ----------
    lane_models : LaneModels
        Set to the parameters the data are tested against (e.g. the estimate).

    lane_data : dict
        Support -> ``(dataset, lane_circuits)``, as from :meth:`LaneDecomposition.lane_datasets`.

    Returns
    -------
    dict
        ``'mean'``, ``'variance'``, ``'num_outcomes'`` (possible outcomes, summed over lane circuits,
        as :attr:`FactorizedObjective.num_elements` counts them; those of zero probability never occur
        and add nothing to the mean or variance), ``'num_lane_circuits'``, and ``'asymptotic_mean'`` =
        (num_outcomes - num_lane_circuits) / 2.
    """
    mean = variance = 0.0
    outcomes = circuits = 0
    for support, (dataset, lane_circuits) in lane_data.items():
        lane_model, _ = lane_models[support]
        for lc, p in lane_model.sim.bulk_probs(list(lane_circuits)).items():
            N = float(dataset[lc].total)
            outcomes += len(p)
            probs = _np.clip(_np.array(list(p.values()), dtype=float), 0.0, None)
            probs = probs[probs > 0]
            probs /= probs.sum()
            m, v = _binomial_deviance_moments(N * probs, N)
            large_count_variance = 0.5 * _np.sum((1 - probs)**2)
            scale = 0.5 * (len(probs) - 1) / large_count_variance if large_count_variance > 0 else 1.0
            mean += float(m.sum())
            variance += float(v.sum()) * scale
            circuits += 1
    return {'mean': mean, 'variance': variance, 'num_outcomes': outcomes, 'num_lane_circuits': circuits,
            'asymptotic_mean': (outcomes - circuits) / 2}
