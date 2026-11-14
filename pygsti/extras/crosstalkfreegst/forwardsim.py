#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""
A forward simulator for crosstalk-free models that simulates each circuit lane by lane.
"""
from collections import defaultdict as _defaultdict

import numpy as _np

from pygsti.forwardsims.forwardsim import ForwardSimulator as _ForwardSimulator
from pygsti.tools import slicetools as _slct

from .diagnostics import _lane_outcome, _lane_outcome_readers
from .lanemodels import LaneModels, member_objects, member_parameter_indices
from .lanes import LaneDecomposition, split_circuit_into_lanes


class CrosstalkFreeForwardSimulator(_ForwardSimulator):
    """
    Simulates circuits on a crosstalk-free model by factorizing them into lanes.

    For a crosstalk-free model (see :func:`~pygsti.extras.crosstalkfreegst.check_factorizable`),
    the probability of a circuit outcome is the product, over the circuit's lanes, of the
    probability of the outcome's restriction to the lane.  This simulator computes each lane's
    outcome distribution -- and its derivatives -- with a lane-sized model, and assembles the
    requested many-qubit outcome probabilities (and derivatives, by the product rule) from
    them.  Distinct lane circuits are simulated once per call, however many circuits share them.

    Assign it like any other simulator, ``model.sim = CrosstalkFreeForwardSimulator()``; then
    ``model.probabilities``, ``pygsti.data.simulate_data``, pyGSTi's objective functions and
    ``GateSetTomography`` all use it.  The work per circuit outcome is that of simulating its
    lanes, instead of a ``4**n``-dimensional superoperator product, so it is much faster than
    pyGSTi's general simulators for crosstalk-free models -- though pyGSTi still enumerates all
    ``2**n`` outcomes of each circuit when building layouts, and builds the ``n``-qubit model
    itself, which limits it to roughly a dozen qubits.  (Beyond that, use the lane-level tools
    of this package directly.)  It also handles circuits whose line order differs from the
    model's correctly, unlike pyGSTi's general simulators, which currently use a static cached
    marginalized POVM for them.

    The lane models are built when first needed and then kept: each call only sets their
    parameters from the model's.  They are rebuilt if a gate, SPAM error or instrument of the model
    is replaced or added.  The model needs exactly one state preparation and one measurement
    (circuits may name them or not).  First derivatives are exact; second derivatives are the
    base class's finite differences of them.

    Parameters
    ----------
    model : LocalNoiseModel, optional
        Usually set by assigning the simulator to ``model.sim``.

    lane_simulator : str or ForwardSimulator, optional
        Simulator for the lane models (see :class:`LaneModels`).
    """

    def __init__(self, model=None, lane_simulator='auto'):
        self.lane_simulator = lane_simulator
        super().__init__(model)  # (assigns self.model, which resets the lane models)

    @_ForwardSimulator.model.setter
    def model(self, val):
        _ForwardSimulator.model.fset(self, val)
        self._lane_models = None  # lane models are rebuilt for the new model on demand
        self._members = None
        self._lane_to_model = None

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        sim = self.lane_simulator
        state['lane_simulator'] = sim if isinstance(sim, str) else sim.to_nice_serialization()
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        sim = state.get('lane_simulator', 'auto')
        return cls(None, sim if isinstance(sim, str) else _ForwardSimulator.from_nice_serialization(sim))

    def __getstate__(self):
        state = super().__getstate__()
        state.update(_lane_models=None, _members=None, _lane_to_model=None)  # rebuilt on demand
        return state

    def _current_lane_models(self):
        """
        The lane models of the model, rebuilt if this is their first use or if any of the model's
        gates, SPAM error factors or instruments has been replaced or added since they were built.
        """
        model = self.model
        members = {key: member for key, (_, member) in member_objects(model).items()}  # (checks the model too)
        if len(model.prep_blks['layers']) != 1 or len(model.povm_blks['layers']) != 1:
            raise NotImplementedError("Only models with one state preparation and one measurement are supported.")
        if self._lane_models is None or members.keys() != self._members.keys() \
                or any(member is not self._members[key] for key, member in members.items()):
            self._lane_models = LaneModels.from_model(model, self.lane_simulator)
            self._members = members
        return self._lane_models

    def _sync(self, supports):
        """Build lane models for `supports` and set every lane model from the model's current parameters."""
        lane_models = self._current_lane_models()
        for support in supports:
            lane_models[support]
        model_indices = member_parameter_indices(self.model)
        lane_to_model = _np.empty(lane_models.num_params, dtype=_np.int64)
        for key, idx in lane_models.keys.items():
            lane_to_model[idx] = model_indices[key]
        self._lane_to_model = lane_to_model
        lane_models.set_vector(self.model.to_vector()[lane_to_model])

    def _lane_tables(self, layout, derivatives):
        """
        Split `layout`'s circuits into lanes and simulate every distinct lane circuit.

        Returns, for each circuit, its lanes and how to read each lane's outcome out of a circuit
        outcome, and dicts ``(support, lane circuit) -> {lane outcome: p}`` and, if `derivatives`,
        ``-> {lane outcome: dp (lane params)}``.
        """
        self._current_lane_models()  # first, so that an unsupported model gets that function's error
        qubits = tuple(self.model.state_space.qubit_labels)
        decomposer = LaneDecomposition(qubits, instrument_names=self.model.processor_spec.instrument_names)
        splits, by_support = [], _defaultdict(dict)
        circuits = [circuit for _, circuit, _ in layout.iter_unique_circuits()]
        # A circuit may name the state preparation and measurement explicitly; lanes are split without them.
        for _, circuit, _ in self.model.split_circuits(circuits):
            lanes = split_circuit_into_lanes(circuit, qubits)
            splits.append((lanes, _lane_outcome_readers(circuit, lanes, decomposer._instrument_qubits(circuit))))
            for support, lane_circuit in lanes:
                by_support[support][lane_circuit] = None
        self._sync(by_support)
        probs, dprobs = {}, {}
        for support, lane_circuits in by_support.items():
            lane_model, _ = self._lane_models[support]
            lane_circuits = list(lane_circuits)
            if derivatives:
                for lc, d in lane_model.sim.bulk_dprobs(lane_circuits).items():
                    dprobs[support, lc] = d
            for lc, p in lane_model.sim.bulk_probs(lane_circuits).items():
                probs[support, lc] = p
        return splits, probs, dprobs

    def _bulk_fill_probs_block(self, array_to_fill, layout):
        splits, probs, _ = self._lane_tables(layout, derivatives=False)
        for (element_indices, _, outcomes), (lanes, readers) in zip(layout.iter_unique_circuits(), splits):
            outcomes = [ol if isinstance(ol, tuple) else (ol,) for ol in outcomes]
            values = _np.ones(len(outcomes))
            for lane, reader in zip(lanes, readers):
                p = probs[lane]
                values *= [p[_lane_outcome(ol, reader)] for ol in outcomes]
            array_to_fill[element_indices] = values

    def _bulk_fill_dprobs_block(self, array_to_fill, dest_param_slice, layout, param_slice):
        splits, probs, dprobs = self._lane_tables(layout, derivatives=True)
        num_params = self.model.num_params
        wanted = _slct.to_array(param_slice) if param_slice is not None else _np.arange(num_params)
        dest = dest_param_slice if dest_param_slice is not None else slice(0, len(wanted))
        for (element_indices, _, outcomes), (lanes, readers) in zip(layout.iter_unique_circuits(), splits):
            outcomes = [ol if isinstance(ol, tuple) else (ol,) for ol in outcomes]
            lane_outcomes = [[_lane_outcome(ol, reader) for ol in outcomes] for reader in readers]
            lane_p = _np.array([[probs[lane][lo] for lo in los] for lane, los in zip(lanes, lane_outcomes)])
            full = _np.zeros((len(outcomes), num_params))
            for i, ((support, lane_circuit), los) in enumerate(zip(lanes, lane_outcomes)):
                others = _np.prod(_np.delete(lane_p, i, axis=0), axis=0) if len(lanes) > 1 else _np.ones(len(los))
                d = dprobs[support, lane_circuit]
                lane_d = _np.array([d[lo] for lo in los])            # (outcomes, lane params)
                gidx = self._lane_to_model[self._lane_models[support][1]]
                _np.add.at(full, (slice(None), gidx), lane_d * others[:, None])
            array_to_fill[element_indices, dest] = full[:, wanted]

    def _compute_circuit_outcome_probabilities(self, array_to_fill, circuit, outcomes, resource_alloc, time=None):
        # (the base class's hook for one circuit; `time` is ignored, as the lane models do not depend on it)
        layout = _SingleCircuitLayout(circuit, outcomes)
        self._bulk_fill_probs_block(array_to_fill, layout)


class _SingleCircuitLayout(object):
    """Just enough of a layout to evaluate one circuit with the block methods."""

    def __init__(self, circuit, outcomes):
        self._item = (slice(0, len(outcomes)), circuit, outcomes)

    def iter_unique_circuits(self):
        yield self._item
