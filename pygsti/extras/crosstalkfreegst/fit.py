#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""
Fitting a many-qubit crosstalk-free model to tensor-product (e.g. simultaneous GST) data.

The objective -- chi^2 or Delta(log-likelihood) -- is a sum of per-lane terms, each of
which depends on only the parameters of the gates and SPAM on that lane's qubits.
:class:`FactorizedObjective` evaluates each term with an ordinary pyGSTi objective function
on a lane-sized model, and assembles the gradient and Gauss-Newton matrix of the whole
objective from the lane-sized pieces.  The Gauss-Newton matrix couples only parameters
that share a lane, so it is sparse; :func:`levenberg_marquardt` minimizes the objective using
it directly, never forming the full Jacobian.
"""
import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as _np
import scipy.linalg as _sla
import scipy.sparse as _sps
import scipy.sparse.linalg as _spsl

from pygsti.modelmembers.operations import DenseOperator as _DenseOperator
from pygsti.objectivefns import ObjectiveFunctionBuilder as _ObjectiveFunctionBuilder

from .lanes import LaneDecomposition
from .lanemodels import LaneModels, member_objects
from .parallel import LanePool, build_lane_objective, lane_pieces, num_outcomes

Support = Tuple


class FactorizedObjective(object):
    """
    A sum of lane objective functions sharing one global parameter vector.

    Evaluating the objective at a vector does not set `lane_models` to it.  Terms evaluated in
    this process leave their lane models at the last vector evaluated, while
    ``lane_models.current_vector()`` is unchanged; terms evaluated in a pool leave them
    untouched.  Call ``lane_models.set_vector(x)`` before using the lane models again, as
    :func:`run_crosstalk_free_gst` does.

    Parameters
    ----------
    lane_models : LaneModels
        Supplies the model and global parameter indices for each support.

    lane_data : dict
        Maps support -> ``(dataset, lane_circuits)``, as returned by
        :meth:`LaneDecomposition.lane_datasets`.  Every lane circuit must have counts.

    objective : str or ObjectiveFunctionBuilder, optional
        Anything ``ObjectiveFunctionBuilder.cast`` accepts, e.g. ``'logl'`` or ``'chi2'``.
        Each lane's least-squares vector is computed by the objective it builds.  Objectives
        with penalty terms are not supported: a penalty would be added once per lane rather
        than once.

    pool : LanePool, optional
        Worker processes to evaluate the lane terms in parallel.  The terms are loaded into it,
        replacing any it held before, so only the latest objective given a pool can be
        evaluated.  If None, terms are evaluated in this process.

    Attributes
    ----------
    num_elements : int
        The number of possible outcomes of the lane circuits, summed over them (whether or not
        each outcome was observed).
    """

    _PENALTY_TERMS = ('regularize_factor', 'cptp_penalty_factor', 'spam_penalty_factor', 'errorgen_penalty_factor',
                      'forcefn_grad')

    def __init__(self, lane_models: LaneModels, lane_data: Dict, objective='logl', pool: LanePool = None):
        builder = _ObjectiveFunctionBuilder.cast(objective)
        penalties = [name for name in self._PENALTY_TERMS if _np.any((builder.penalties or {}).get(name, 0))]
        if builder.additional_args.get('callable_penalty') is not None:
            penalties.append('callable_penalty')
        if penalties:
            raise NotImplementedError(f"Objectives with penalty terms ({', '.join(penalties)}) are not supported: "
                                      "the penalty would be added once per lane.")

        self.supports = sorted(lane_data, key=lambda s: (len(s), str(s)))
        lanes = []  # (support, lane model, global indices, dataset, circuits, largest total count)
        for support in self.supports:
            dataset, circuits = lane_data[support]
            lane_model, gidx = lane_models[support]
            totals = [dataset[c].total for c in circuits]
            if min(totals, default=1) <= 0:
                empty = circuits[int(_np.argmin(totals))]
                raise ValueError(f"Lane circuit {empty.str} on {support} has no counts; "
                                 "leave circuits without data out of the fit.")
            lanes.append((support, lane_model, gidx, dataset, list(circuits), max(totals, default=1.0)))
        self.num_params = lane_models.num_params  # after building every lane model it needs
        self.pool = pool

        # A lane's objective is a sum over its circuits, so in parallel, lanes much costlier
        # than the rest are split into chunks of circuits to balance the workers' loads.
        terms = []  # as `lanes`, with a chunk of the lane's circuits
        if pool is not None:
            costs = [len(cs) * max(len(gidx), 1) for _, _, gidx, _, cs, _ in lanes]
            target = sum(costs) / (2 * max(pool.processes, 1))
            for lane, cost in zip(lanes, costs):
                support, model, gidx, dataset, circuits, max_total = lane
                num_chunks = max(1, min(len(circuits), int(_np.ceil(cost / target))))
                for chunk in _np.array_split(_np.arange(len(circuits)), num_chunks):
                    terms.append((support, model, gidx, dataset, [circuits[i] for i in chunk], max_total))
        else:
            terms = lanes
        self.term_supports = [t[0] for t in terms]
        self.global_indices = [t[2] for t in terms]
        if pool is None:
            self._objfns = [build_lane_objective(m, ds, cs, builder, n) for _, m, _, ds, cs, n in terms]
            self.num_elements = sum(num_outcomes(m, cs) for _, m, _, _, cs, _ in terms)
        else:
            self.num_elements = sum(pool.load([t[1:] for t in terms], builder))
            self._load = pool.num_loads

    def _pieces(self, x, derivatives):
        if self.pool is not None:
            if self.pool.num_loads != self._load:
                raise RuntimeError("This objective's terms are no longer loaded in its LanePool: "
                                   "another objective was given the same pool.")
            return self.pool.evaluate(x, derivatives)
        return [lane_pieces(objfn, x[gidx], derivatives) for objfn, gidx in zip(self._objfns, self.global_indices)]

    def fn(self, x: _np.ndarray) -> float:
        """The objective at global parameter vector `x`: the sum of squares of every lane's lsvec."""
        return sum(self._pieces(x, False))

    def lane_fns(self, x: _np.ndarray) -> Dict[Support, float]:
        """Each lane's contribution to :meth:`fn`."""
        out = dict.fromkeys(self.supports, 0.0)
        for support, rr in zip(self.term_supports, self._pieces(x, False)):
            out[support] += rr
        return out

    def fn_grad_gn(self, x: _np.ndarray):
        """
        The objective, ``J^T r``, and the Gauss-Newton matrix ``J^T J`` (sparse) at `x`.

        ``J`` and ``r`` are the global Jacobian and least-squares vector; neither is formed.
        """
        f = 0.0
        jtr = _np.zeros(self.num_params)
        rows, cols, vals = [], [], []
        for (rr, lane_jtr, lane_jtj), gidx in zip(self._pieces(x, True), self.global_indices):
            f += rr
            jtr[gidx] += lane_jtr
            r_idx, c_idx = _np.meshgrid(gidx, gidx, indexing='ij')
            rows.append(r_idx.ravel())
            cols.append(c_idx.ravel())
            vals.append(lane_jtj.ravel())
        jtj = _sps.coo_matrix((_np.concatenate(vals), (_np.concatenate(rows), _np.concatenate(cols))),
                              shape=(self.num_params, self.num_params)).tocsc()  # sums duplicates
        return f, jtr, jtj


def _solve(jtj, damping, rhs, dense_threshold=3000):
    """Solve ``(jtj + diag(damping)) dx = rhs``."""
    n = jtj.shape[0]
    if n <= dense_threshold:
        A = jtj.toarray()
        A[_np.diag_indices(n)] += damping
        try:
            return _sla.solve(A, rhs, assume_a='pos')
        except (_sla.LinAlgError, ValueError):
            return _sla.lstsq(A, rhs)[0]
    return _spsl.spsolve((jtj + _sps.diags(damping, format='csc')).tocsc(), rhs)


@dataclass
class LMResult:
    """
    The output of :func:`levenberg_marquardt`.

    `converged` is True when any of its stopping tests was met -- including the two that detect
    a stalled objective -- and False only when the run ended at the iteration limit; `message`
    says which.
    """
    x: _np.ndarray
    fn: float
    num_iterations: int
    converged: bool
    message: str


def levenberg_marquardt(objective: FactorizedObjective, x0: _np.ndarray, max_iterations: int = 100,
                        rel_ftol: float = 1e-6, abs_gtol: float = 1e-8, rel_xtol: float = 1e-8,
                        init_damping: float = 1e-3, damping: str = 'identity', stall_window: int = 10,
                        stat_tol: float = 0.01, verbosity: int = 0) -> LMResult:
    """
    Minimize a :class:`FactorizedObjective` by Levenberg-Marquardt on its sparse normal equations.

    With ``damping='identity'`` (the default, as in pyGSTi's own optimizer) the step solves
    ``(J^T J + mu I) dx = -J^T r``, starting from ``mu = init_damping * max(diag(J^T J))``;
    ``'marquardt'`` uses ``mu * diag(J^T J)`` instead, floored so that directions with no
    curvature (e.g. gauge directions) stay regularized.  Marquardt damping stalled well short
    of the optimum on a 3-qubit fit with sqrt-parameterized stochastic rates ('H+S'), where
    identity damping converged quickly; both converge on linear rates ('H+s').  `mu` follows
    Nielsen's update rule.

    The run ends when any of these holds:

    - the *undamped* Gauss-Newton model predicts that at most ``rel_ftol`` times the
      objective's value remains to be gained (``g^T (J^T J)^-1 g``, with a tiny ridge for
      flat directions);
    - the objective has stalled: over the last `stall_window` iterations it fell by less than
      ``rel_ftol`` times its value per iteration on average (this catches slow crawls along
      weakly constrained directions, where the quadratic prediction is unreliable);
    - the objective's change is statistically negligible: over the last `stall_window` iterations
      it fell by less than `stat_tol` times ``sqrt(f)``.  For a correct model the log-likelihood
      statistic (and half the chi^2) has standard deviation of order ``sqrt(f)``, so smaller
      changes cannot matter for any inference;
    - the gradient falls below ``abs_gtol``, or a step falls below ``rel_xtol`` relative to the
      parameter vector.

    None of the first three tests looks at a single damped step, whose decrease heavy
    damping can make small long before convergence.
    """
    def remaining(f, g, jtj):  # decrease the undamped Gauss-Newton model predicts, relative to f
        diag = jtj.diagonal()
        return float(g @ _solve(jtj, _np.full(len(g), 1e-10 * max(diag.max(), 1e-12)), g)) / max(f, 1e-300)

    x = _np.array(x0, dtype=float)
    f, g, jtj = objective.fn_grad_gn(x)
    if damping not in ('marquardt', 'identity'):
        raise ValueError(f"Unknown damping {damping!r}")
    mu = init_damping if damping == 'marquardt' else init_damping * max(float(jtj.diagonal().max()), 1e-12)
    nu = 2.0
    message = f"Maximum number of iterations ({max_iterations}) reached"
    converged = False
    history = [f]
    it = 0
    for it in range(1, max_iterations + 1):
        if _np.max(_np.abs(g)) < abs_gtol:
            message, converged = "Gradient below tolerance", True
            break
        if remaining(f, g, jtj) <= rel_ftol:
            message, converged = "Predicted remaining decrease below tolerance", True
            break
        if damping == 'marquardt':
            diag = jtj.diagonal()
            dx = -_solve(jtj, mu * _np.maximum(diag, 1e-8 * max(diag.max(), 1e-12)), g)
        else:
            dx = -_solve(jtj, _np.full(len(x), mu), g)
        if _np.linalg.norm(dx) < rel_xtol * (_np.linalg.norm(x) + rel_xtol):
            message, converged = "Step below tolerance", True
            break
        x_new = x + dx
        f_new = objective.fn(x_new)
        predicted = -(2 * dx @ g + dx @ (jtj @ dx))
        rho = (f - f_new) / predicted if predicted > 0 else -1.0
        if verbosity > 1:
            print(f"    LM iter {it}: f={f_new:.6g} (was {f:.6g}), mu={mu:.3g}, rho={rho:.3g}")
        if f_new < f and rho > 0:
            x = x_new
            mu *= max(1 / 3, 1 - (2 * rho - 1)**3)
            nu = 2.0
            f, g, jtj = objective.fn_grad_gn(x)
        else:
            mu *= nu
            nu *= 2
        history.append(f)
        if len(history) > stall_window and history[-1 - stall_window] - f <= stall_window * rel_ftol * f:
            message, converged = f"Objective stalled over {stall_window} iterations", True
            break
        if len(history) > stall_window and history[-1 - stall_window] - f <= stat_tol * _np.sqrt(max(f, 0.0)):
            message, converged = f"Objective change over {stall_window} iterations statistically negligible", True
            break
    if verbosity > 0:
        print(f"    LM: {message} after {it} iterations; objective = {f:.8g}")
    return LMResult(x, f, it, converged, message)


def _resolve_damping(damping: str, lane_models: LaneModels) -> str:
    """``'marquardt'`` for densely parameterized lane models, ``'identity'`` otherwise; see run_crosstalk_free_gst."""
    if damping != 'auto':
        return damping

    def is_dense(key, member):
        if key[0] == 'instrument':  # an Instrument of embedded single-qubit operations
            return all(isinstance(op.embedded_op, _DenseOperator) for op in member.values())
        return isinstance(member, _DenseOperator)
    dense = all(is_dense(key, member)
                for _, (lane_model, _) in lane_models.items()
                for key, (_, member) in member_objects(lane_model).items())
    return 'marquardt' if dense else 'identity'


@dataclass
class IterationRecord:
    """One stage of :func:`run_crosstalk_free_gst`."""
    circuit_list_index: int
    objective: str
    fn: float
    num_circuits: int
    num_lane_circuits: int
    num_elements: int  # possible outcomes of the lane circuits, summed over them
    seconds: float
    lm: LMResult = field(repr=False)


@dataclass
class CrosstalkFreeGSTResults:
    """
    The output of :func:`run_crosstalk_free_gst`.

    Attributes
    ----------
    lane_models : LaneModels
        Lane models set to the final estimate.  ``lane_models[support][0]`` is the estimate
        restricted to one lane's qubits; ``lane_models.parameters()`` gives every member's
        estimated parameters by member key.

    x : numpy.ndarray
        The final global parameter vector (indexed by ``lane_models.keys``).

    model : LocalNoiseModel or None
        When a many-qubit model was given, a copy of it set to the final estimate.

    lane_fns : dict
        Each support's contribution to the final objective, evaluated on the final circuit
        list.  Objective values follow pyGSTi's objective functions: chi^2 for ``'chi2'``,
        and Delta(log-likelihood) -- *not* twice it -- for ``'logl'``.  They are computed
        from lane-marginal data, so the saturated (constant) part differs from the full
        many-qubit objective's; differences between parameter values agree exactly.

    iterations : list of IterationRecord
    """
    lane_models: LaneModels
    x: _np.ndarray
    model: Optional[object]
    lane_fns: Dict[Support, float]
    iterations: List[IterationRecord]
    decomposition: LaneDecomposition = field(repr=False)

    @property
    def final_fn(self) -> float:
        return self.iterations[-1].fn


def run_crosstalk_free_gst(dataset, model, circuit_lists: Sequence[Sequence], qubit_labels: Sequence = None,
                           iteration_objective='chi2', final_objective='logl', max_iterations: int = 100,
                           rel_ftol: float = 1e-6, simulator='auto', processes: int = 1, damping: str = 'auto',
                           verbosity: int = 1) -> CrosstalkFreeGSTResults:
    """
    Fit a crosstalk-free model to data from tensor-product circuits, one lane at a time.

    This mirrors pyGSTi's standard iterative GST fit -- each germ-power circuit list in turn,
    seeded by the previous list's estimate, using `iteration_objective`, then a final fit of
    the last list using `final_objective` -- but evaluates every objective as a sum over lanes
    (see :mod:`pygsti.extras.crosstalkfreegst.lanes`), so its cost grows with the number of
    distinct lane circuits rather than exponentially with the number of qubits.  For a
    crosstalk-free model the result is the same estimate a full many-qubit fit would find.

    Parameters
    ----------
    dataset : DataSet or LaneDecomposition
        Counts for every circuit in `circuit_lists`; outcome labels are bit strings over each
        circuit's lines, in its ``line_labels`` order.  At tens of qubits, pass a
        :class:`LaneDecomposition` instead, filled circuit by circuit: it keeps only lane
        marginals, while a joint DataSet can hold millions of distinct outcome strings.

    model : LocalNoiseModel or LaneModels
        The crosstalk-free model to fit -- either a many-qubit model, e.g. from
        ``create_crosstalk_free_model``, or, when that is too large to build (beyond ~12
        qubits), a :class:`LaneModels` that builds lane models directly.  Its current
        parameters are the starting point.  A model is not modified; a LaneModels is left set
        to the estimate.

    circuit_lists : list of lists of Circuit
        Typically ``design.circuit_lists`` of a :class:`SimultaneousGSTDesign`, but any
        circuits whose idles are explicit will do; each circuit is split into lanes on its own.

    qubit_labels : sequence, optional
        Canonical qubit order.  Defaults to the processor spec's.

    iteration_objective, final_objective : str, optional
        Objectives for the per-list fits and the final fit (``'chi2'``, ``'logl'``, ...).  Set
        `final_objective` to None to skip the final fit.

    max_iterations, rel_ftol : optional
        Passed to :func:`levenberg_marquardt` for each fit.

    simulator : str or ForwardSimulator, optional
        Forward simulator for the lane models, when `model` is a many-qubit model.

    damping : {'auto', 'marquardt', 'identity'}, optional
        Levenberg-Marquardt damping (see :func:`levenberg_marquardt`).  ``'auto'`` uses
        ``'marquardt'`` when every gate and SPAM member is densely parameterized (e.g. the
        general crosstalk-free model) and ``'identity'`` otherwise (e.g. Lindblad
        parameterizations).  Each failed where the other worked: on the 4-qubit general
        model, Marquardt damping reached in 10 iterations a lower objective than identity
        damping did in 600; on a 3-qubit 'H+S' model, Marquardt damping stalled short of the
        optimum that identity damping reached in about 20 iterations.

    processes : int, optional
        Number of worker processes evaluating lane terms in parallel (see :class:`LanePool`).
        1 (the default) evaluates everything in this process.  Worker start-up costs a few
        seconds, so parallelism pays off for fits that take more than a minute or so.

    verbosity : int, optional

    Returns
    -------
    CrosstalkFreeGSTResults
    """
    full_model = None
    if isinstance(model, LaneModels):
        lane_models = model
    else:
        full_model = model
        lane_models = LaneModels.from_model(model, simulator)
    qubit_labels = tuple(lane_models.qubit_labels if qubit_labels is None else qubit_labels)

    t0 = _time.time()
    all_circuits = list({c: None for lst in circuit_lists for c in lst})
    if isinstance(dataset, LaneDecomposition):
        decomposition = dataset
        missing = [c for c in all_circuits if c not in decomposition.marginals]
        if missing:
            raise ValueError(f"{len(missing)} circuits have no data, e.g. {missing[0].str}")
    else:
        decomposition = LaneDecomposition(qubit_labels, dataset, all_circuits,
                                          instrument_names=lane_models.processor_spec.instrument_names)
    if verbosity > 0:
        print(f"Split {len(all_circuits)} circuits into lanes in {_time.time() - t0:.1f}s")

    if len(circuit_lists) == 0:
        raise ValueError("circuit_lists is empty")
    stages = [(i, iteration_objective) for i in range(len(circuit_lists))]
    if final_objective is not None:
        stages.append((len(circuit_lists) - 1, final_objective))

    x = lane_models.current_vector()
    iterations = []
    lane_data_cache = {}
    pool = LanePool(processes) if processes > 1 else None
    try:
        for list_index, objective in stages:
            t0 = _time.time()
            if list_index not in lane_data_cache:
                lane_data_cache = {list_index: decomposition.lane_datasets(circuit_lists[list_index])}
            lane_data = lane_data_cache[list_index]
            objfn = FactorizedObjective(lane_models, lane_data, objective, pool)
            x = _np.concatenate([x, lane_models.initial_vector()[len(x):]])  # new lanes, new parameters
            num_lane_circuits = sum(len(c) for _, c in lane_data.values())
            if verbosity > 0:
                print(f"Circuit list {list_index} ({len(circuit_lists[list_index])} circuits, "
                      f"{num_lane_circuits} lane circuits on {len(lane_data)} supports), objective '{objective}':")
            lm = levenberg_marquardt(objfn, x, max_iterations=max_iterations, rel_ftol=rel_ftol,
                                     damping=_resolve_damping(damping, lane_models), verbosity=verbosity)
            x = lm.x
            iterations.append(IterationRecord(list_index, str(objective), lm.fn, len(circuit_lists[list_index]),
                                              num_lane_circuits, objfn.num_elements, _time.time() - t0, lm))
        lane_fns = objfn.lane_fns(x)
    finally:
        if pool is not None:
            pool.close()
        # Evaluating the objective leaves lane models at the last vector tried.  Set them all to
        # the estimate -- or, if the fit was interrupted, to the last completed stage's.
        lane_models.set_vector(_np.concatenate([x, lane_models.initial_vector()[len(x):]]))

    estimate = None if full_model is None else lane_models.to_model(full_model, x)
    return CrosstalkFreeGSTResults(lane_models, x, estimate, lane_fns, iterations, decomposition)
