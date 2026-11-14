#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""
Lane-sized crosstalk-free models, and the bookkeeping that ties their parameters together.

Every parameter of a crosstalk-free model belongs to a *member* identified by a key that
does not depend on which (or how many) qubits the model covers:

- ``('gate', label)`` for an entry of ``operation_blks['gates']``, e.g. ``Label('Gxpi2', 3)``
  -- or ``Label('Gxpi2')`` for a gate shared by every placement;
- ``('prep', prep_label, qubit, i)`` and ``('povm', povm_label, qubit, i)`` for the ``i``-th
  single-qubit error factor acting on `qubit` in a state preparation or POVM;
- ``('instrument', label)`` for a single-qubit instrument (mid-circuit measurement) in
  ``instrument_blks['layers']``, e.g. ``Label('Iz', 3)``.

So a model on qubits ``(3, 4)`` and a model on qubit ``(3,)`` both hold the parameters of
``('gate', Label('Gxpi2', 3))``, and :class:`LaneModels` keeps a single global copy of them.
The same keys identify parameters of a full many-qubit model, when one is small enough to
build (:meth:`LaneModels.to_model`) -- but nothing here requires one.
"""
from collections import OrderedDict as _OrderedDict
from typing import Callable, Dict, Sequence, Tuple

import numpy as _np

from pygsti.baseobjs import statespace as _statespace
from pygsti.models.localnoisemodel import LocalNoiseModel as _LocalNoiseModel
from pygsti.modelmembers import instruments as _inst
from pygsti.modelmembers import operations as _op
from pygsti.modelmembers import povms as _povm
from pygsti.modelmembers import states as _state

Support = Tuple


def _spam_error_factors(error_map, what):
    """The per-qubit factors of a SPAM error map, as a list of ``(qubit, op)`` pairs, in order."""
    factors = error_map.factorops if isinstance(error_map, _op.ComposedOp) else [error_map]
    out = []
    for f in factors:
        if not isinstance(f, _op.EmbeddedOp) or len(f.target_labels) != 1:
            raise NotImplementedError(
                f"The {what} error map has a factor of type {type(f).__name__} acting on "
                f"{getattr(f, 'target_labels', 'all qubits')}.  Only products of single-qubit "
                "errors factorize over lanes.")
        out.append((f.target_labels[0], f.embedded_op))
    return out


def _decompose_prep(prep, what):
    """``(zvals, [(qubit, error op), ...])`` for a product state preparation."""
    if isinstance(prep, _state.ComposedState):
        ideal, factors = prep.state_vec, _spam_error_factors(prep.error_map, what)
    else:
        ideal, factors = prep, []
    if not isinstance(ideal, _state.ComputationalBasisState):
        raise NotImplementedError(f"The ideal part of {what} is a {type(ideal).__name__}; only a "
                                  "ComputationalBasisState (optionally composed with product "
                                  "errors) factorizes over lanes.")
    return ideal._zvals, factors


def _decompose_povm(povm, what):
    """``[(qubit, error op), ...]`` for a product (computational-basis) POVM."""
    if isinstance(povm, _povm.ComposedPOVM):
        ideal, factors = povm.base_povm, _spam_error_factors(povm.error_map, what)
    else:
        ideal, factors = povm, []
    if not isinstance(ideal, _povm.ComputationalBasisPOVM):
        raise NotImplementedError(f"The ideal part of {what} is a {type(ideal).__name__}; only a "
                                  "ComputationalBasisPOVM (optionally composed with product "
                                  "errors) factorizes over lanes.")
    return factors


def _keyed_factors(kind, lbl, factors):
    """``[(key, qubit, op), ...]`` for SPAM error factors; see the module docstring for keys."""
    seen = {}
    out = []
    for q, op in factors:
        i = seen.get(q, 0)
        seen[q] = i + 1
        out.append(((kind, lbl, q, i), q, op))
    return out


def check_factorizable(model: _LocalNoiseModel) -> None:
    """
    Raise an error unless `model`'s outcome probabilities factorize over circuit lanes.

    That holds for the models built by ``pygsti.models.create_crosstalk_free_model`` with
    their default computational-basis SPAM: every gate acts only on its target qubits, the
    state preparation is a computational basis state followed by single-qubit errors, and
    the measurement is single-qubit errors followed by a computational-basis POVM.  Instruments
    (mid-circuit measurements) are allowed if each acts on a single qubit, i.e. its label names
    one qubit and its members are embedded on that qubit (see
    :func:`~pygsti.extras.crosstalkfreegst.add_single_qubit_instruments`).
    """
    if not isinstance(model, _LocalNoiseModel):
        raise ValueError(f"Expected a LocalNoiseModel (e.g. from create_crosstalk_free_model), not {type(model)}")
    if len(model.factories['gates']) or len(model.factories['layers']):
        raise NotImplementedError("Models with operation factories are not supported.")
    if model._layer_rules.implicit_idle_mode != 'none':
        raise NotImplementedError("Only implicit_idle_mode='none' is supported.")
    for lbl, prep in model.prep_blks['layers'].items():
        _decompose_prep(prep, f"prep {lbl}")
    for lbl, povm in model.povm_blks['layers'].items():
        _decompose_povm(povm, f"POVM {lbl}")
    for lbl, inst in model.instrument_blks['layers'].items():
        _instrument_qubit(lbl, inst)


def _instrument_qubit(lbl, inst):
    """The qubit a single-qubit instrument acts on, or an error if it isn't one."""
    sslbls = getattr(lbl, 'sslbls', None)
    if sslbls is None or len(sslbls) != 1:
        raise NotImplementedError(f"Instrument {lbl} does not name a single qubit; only single-qubit instruments "
                                  "with qubit-qualified labels (e.g. Label('Iz', 0)) are supported.")
    for k, member in inst.items():
        if not isinstance(member, _op.EmbeddedOp) or tuple(member.target_labels) != tuple(sslbls):
            raise NotImplementedError(f"Member {k} of instrument {lbl} is not embedded on qubit {sslbls[0]}.")
    return sslbls[0]


def member_objects(model: _LocalNoiseModel) -> Dict[tuple, Tuple[tuple, object]]:
    """
    Map each member key of a crosstalk-free `model` to ``(qubits, member)``.

    `qubits` are the qubits the member acts on -- a gate's target qubits (None for a gate
    shared by every placement), or the single qubit of a SPAM error factor or instrument.
    `member` is the gate, the SPAM factor's single-qubit error map (applied after the ideal
    preparation, or before the ideal measurement), or the instrument (whose members are
    embedded single-qubit operations).
    """
    check_factorizable(model)
    out = _OrderedDict((('gate', key), (key.sslbls, gate)) for key, gate in model.operation_blks['gates'].items())
    for lbl, prep in model.prep_blks['layers'].items():
        out.update((k, ((q,), op)) for k, q, op in _keyed_factors('prep', lbl, _decompose_prep(prep, lbl)[1]))
    for lbl, povm in model.povm_blks['layers'].items():
        out.update((k, ((q,), op)) for k, q, op in _keyed_factors('povm', lbl, _decompose_povm(povm, lbl)))
    for lbl, inst in model.instrument_blks['layers'].items():
        out[('instrument', lbl)] = ((_instrument_qubit(lbl, inst),), inst)
    return out


def member_parameter_indices(model: _LocalNoiseModel) -> Dict[tuple, _np.ndarray]:
    """
    Map each member key of a crosstalk-free `model` to the indices of its parameters.

    Raises an error if the members found do not account for every parameter of `model`
    exactly once, since then keys could not tie its parameters to other models'.
    """
    out = _OrderedDict()
    covered = _np.zeros(model.num_params, dtype=int)
    for key, (_, member) in member_objects(model).items():
        idx = member.gpindices_as_array()
        out[key] = idx
        covered[idx] += 1
    if not _np.all(covered == 1):
        raise ValueError(f"{_np.count_nonzero(covered != 1)} of the model's {model.num_params} parameters "
                         "do not belong to exactly one gate or single-qubit SPAM error member.  (pyGSTi leaves "
                         "the members' parameter indices stale in a model restored by pickle or copy.deepcopy; "
                         "if this is one, use its .copy().)")
    return out


def restrict_model(model: _LocalNoiseModel, support: Sequence, simulator='matrix') -> _LocalNoiseModel:
    """
    Restrict a crosstalk-free model to the qubits in `support`.

    The result is a :class:`LocalNoiseModel` on just `support`, holding copies of the gates
    of `model` that act within `support` and of the single-qubit SPAM errors on `support`'s
    qubits.  It predicts exactly the marginal outcome distribution of `model` on
    `support` for any circuit whose gates on `support` act only within `support`, and its
    parameters have the same member keys (see :func:`member_parameter_indices`) as the
    corresponding parameters of `model`.

    Parameters
    ----------
    model : LocalNoiseModel
        A crosstalk-free model; see :func:`check_factorizable`.

    support : sequence
        The qubits to restrict to, in the order outcome bits should refer to them.

    simulator : ForwardSimulator or str, optional
        The forward simulator for the restricted model.

    Returns
    -------
    LocalNoiseModel
    """
    check_factorizable(model)
    support = tuple(support)
    sset = set(support)
    lane_space = _statespace.QubitSpace(support)
    evotype = model.evotype

    # Name-only keys are shared by every placement, so they are kept as-is.
    gatedict = {key: gate.copy() for key, gate in model.operation_blks['gates'].items()
                if key.sslbls is None or sset.issuperset(key.sslbls)}

    def lane_error_map(factors):
        ops = [_op.EmbeddedOp(lane_space, (q,), op.copy()) for q, op in factors if q in sset]
        return None if len(ops) == 0 else (ops[0] if len(ops) == 1 else _op.ComposedOp(ops))

    qpos = {q: i for i, q in enumerate(model.state_space.qubit_labels)}
    prep_layers = {}
    for lbl, prep in model.prep_blks['layers'].items():
        zvals, factors = _decompose_prep(prep, f"prep {lbl}")
        ideal = _state.ComputationalBasisState([zvals[qpos[q]] for q in support], 'pp', evotype, lane_space)
        error_map = lane_error_map(factors)
        prep_layers[lbl] = ideal if error_map is None else _state.ComposedState(ideal, error_map)

    povm_layers = {}
    for lbl, povm in model.povm_blks['layers'].items():
        ideal = _povm.ComputationalBasisPOVM(len(support), evotype, state_space=lane_space)
        error_map = lane_error_map(_decompose_povm(povm, f"POVM {lbl}"))
        povm_layers[lbl] = ideal if error_map is None else _povm.ComposedPOVM(error_map, ideal, 'pp')

    lane_model = _LocalNoiseModel(model.processor_spec.subset('all', support), gatedict, prep_layers, povm_layers,
                                  evotype=evotype, simulator=simulator, independent_gates=False,
                                  implicit_idle_mode='none')

    # Instruments on the support's qubits.  The whole instrument is copied at once, so that
    # parameters its members share stay shared, and its members are re-embedded on the lane.
    instruments = [(lbl, inst) for lbl, inst in model.instrument_blks['layers'].items()
                   if _instrument_qubit(lbl, inst) in sset]
    for lbl, inst in instruments:
        copied = inst.copy()
        lane_model.instrument_blks['layers'][lbl] = _inst.Instrument(
            {k: _op.EmbeddedOp(lane_space, tuple(member.target_labels), member.embedded_op)
             for k, member in copied.items()})
    if instruments:
        lane_model._clean_paramvec()
        lane_model._reinit_opcaches()
    return lane_model


class LaneModels(object):
    """
    Lane-sized crosstalk-free models whose shared parameters live in one global vector.

    Lane models are built on demand, one per support (set of qubits), by `lane_model_factory`.
    Each new member key a lane model brings (see the module docstring) is appended to the
    global parameter vector, initialized from that lane model's value; a key already seen
    must have the same number of parameters.  Use :meth:`from_model` to restrict an existing
    many-qubit model, or pass a factory directly when that model would be too big to build.

    Parameters
    ----------
    processor_spec : QubitProcessorSpec
        The whole device.

    lane_model_factory : callable
        Called as ``lane_model_factory(processor_spec.subset('all', support))``; returns a
        crosstalk-free :class:`LocalNoiseModel` on exactly those qubits.  For example
        ``lambda ps: create_crosstalk_free_model(ps, lindblad_error_coeffs=..., independent_gates=True)``.
        It must parameterize members the same way for every support.

    simulator : str or ForwardSimulator, optional
        Forward simulator assigned to every lane model.  ``'auto'`` picks pyGSTi's matrix
        simulator for lane models with at most 200 parameters and its map simulator above
        that.  The matrix simulator caches derivatives of circuit prefixes, which is fast for
        few parameters but costs memory and time that grow with their number: for the general
        model's ~600-parameter 2-qubit lanes, map was 10x faster and used ~7x less memory,
        while for 40-parameter lanes matrix was 6x faster.
    """

    def __init__(self, processor_spec, lane_model_factory: Callable, simulator='auto'):
        self.processor_spec = processor_spec
        self.qubit_labels = tuple(processor_spec.qubit_labels)
        self.lane_model_factory = lane_model_factory
        self.simulator = simulator
        self.keys: Dict[tuple, _np.ndarray] = _OrderedDict()  # member key -> global indices
        self._initial = []  # initial global values, in index order
        self._x = None  # current global vector, if set
        self._cache: Dict[Support, Tuple[_LocalNoiseModel, _np.ndarray]] = {}

    @classmethod
    def from_model(cls, model: _LocalNoiseModel, simulator='auto') -> 'LaneModels':
        """
        Lane models restricted from the many-qubit crosstalk-free `model`.

        Each lane model is restricted from `model` when it is first requested.  Members it shares
        with lane models built earlier keep the values held here, so changing `model` afterwards
        does not make lane models disagree; only members not met before take `model`'s values.
        """
        check_factorizable(model)
        return cls(model.processor_spec, lambda pspec: restrict_model(model, pspec.qubit_labels), simulator)

    @property
    def num_params(self) -> int:
        return len(self._initial)

    def items(self):
        """``(support, (lane model, global parameter indices))`` for every lane model built so far."""
        return self._cache.items()

    def __getitem__(self, support: Support) -> Tuple[_LocalNoiseModel, _np.ndarray]:
        """
        ``(lane_model, global_indices)`` for `support`: ``lane_model``'s ``i``-th parameter is
        the global vector's ``global_indices[i]``-th.
        """
        support = tuple(support)
        if support not in self._cache:
            lane_model = self.lane_model_factory(self.processor_spec.subset('all', support))
            if self.simulator == 'auto':
                lane_model.sim = 'matrix' if lane_model.num_params <= 200 else 'map'
            else:
                lane_model.sim = self.simulator
            values = lane_model.to_vector()
            gidx = _np.full(lane_model.num_params, -1, dtype=_np.int64)
            for key, lidx in member_parameter_indices(lane_model).items():
                if key not in self.keys:
                    self.keys[key] = _np.arange(self.num_params, self.num_params + len(lidx))
                    self._initial.extend(values[lidx])
                elif len(self.keys[key]) != len(lidx):
                    raise ValueError(f"Member {key} has {len(lidx)} parameters in the lane model for "
                                     f"{support} but {len(self.keys[key])} in another lane model.")
                gidx[lidx] = self.keys[key]
            # Members met before take their values from the global vector, not from this lane model.
            if self._x is not None:
                self._x = _np.concatenate([self._x, self._initial[len(self._x):]])
            lane_model.from_vector(self.current_vector()[gidx])
            self._cache[support] = (lane_model, gidx)
        return self._cache[support]

    def initial_vector(self) -> _np.ndarray:
        """The global parameter vector as initialized from the lane models built so far."""
        return _np.array(self._initial, dtype=float)

    def current_vector(self) -> _np.ndarray:
        """The global vector last given to :meth:`set_vector`, extended by any newer parameters."""
        if self._x is None:
            return self.initial_vector()
        return _np.concatenate([self._x, self._initial[len(self._x):]])

    def set_vector(self, x: _np.ndarray) -> None:
        """Set every lane model's parameters from the global vector `x`."""
        x = _np.array(x, dtype=float)
        if x.shape != (self.num_params,):
            raise ValueError(f"Expected a vector of {self.num_params} parameters, got shape {x.shape}")
        self._x = x
        for lane_model, gidx in self._cache.values():
            lane_model.from_vector(self._x[gidx])

    def parameters(self, x: _np.ndarray = None) -> Dict[tuple, _np.ndarray]:
        """Member key -> parameter values, for global vector `x` (default: :meth:`current_vector`)."""
        x = self.current_vector() if x is None else x
        return _OrderedDict((key, x[idx]) for key, idx in self.keys.items())

    def errorgen_rates(self) -> Dict[tuple, float]:
        """
        The elementary error-generator rates of the lane models built so far, at their current
        parameters, read from their gate and SPAM members.

        Requires every member to have a Lindblad error generator, such as those of
        ``pygsti.models.create_crosstalk_free_model(..., lindblad_error_coeffs=...)`` in any of
        its Lindblad parameterizations.  Other models -- the general model of
        :func:`~pygsti.extras.crosstalkfreegst.create_general_crosstalk_free_model`, whose gates
        are arbitrary trace-preserving maps, and models with instruments -- raise an error.

        Returns
        -------
        dict
            ``{(operation, generator): rate}``.  ``operation`` is ``(gate name, q1[, q2])`` for a
            gate placement (just ``(gate name,)`` for a gate shared by every placement) and
            ``('prep', q)`` or ``('povm', q)`` for a qubit's state preparation or measurement.
            ``generator`` is ``(type, pauli)`` for Hamiltonian ('H') and Pauli-stochastic ('S')
            generators and ``(type, pauli, pauli)`` for Pauli-correlation ('C') and active ('A')
            ones, each ``pauli`` a Pauli string on the operation's qubits, in order.
        """
        rates, seen = {}, set()
        for lane_model, _ in self._cache.values():
            for key, (_, member) in member_objects(lane_model).items():
                if key in seen:
                    continue
                seen.add(key)
                if key[0] == 'gate':
                    operation = (key[1].name,) + tuple(key[1].sslbls or ())
                elif key[0] in ('prep', 'povm'):
                    operation = (key[0], key[2])
                else:
                    raise NotImplementedError(f"Error-generator rates of {key[0]} {key[1]} are not supported.")
                if not hasattr(member, 'errorgen_coefficients'):
                    raise ValueError(f"Member {key} is a {type(member).__name__}, which has no error-generator "
                                     "rates; this needs Lindblad-parameterized members.")
                for lbl, rate in member.errorgen_coefficients(label_type='local').items():
                    generator = (lbl.errorgen_type,) + tuple(lbl.basis_element_labels)
                    rates[operation, generator] = float(_np.real(rate))
        return rates

    def to_model(self, model: _LocalNoiseModel, x: _np.ndarray = None) -> _LocalNoiseModel:
        """
        A copy of the many-qubit `model` with the parameters of every member key known here set
        from `x` (default: :meth:`current_vector`); other parameters keep `model`'s values.
        """
        x = self.current_vector() if x is None else x
        model = model.copy()
        v = model.to_vector()
        for key, idx in member_parameter_indices(model).items():
            if key in self.keys:
                v[idx] = x[self.keys[key]]
        model.from_vector(v)
        return model
