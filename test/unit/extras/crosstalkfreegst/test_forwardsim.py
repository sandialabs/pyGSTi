import pickle

import numpy as np
import pytest

import pygsti
from pygsti.forwardsims import MapForwardSimulator
from pygsti.modelpacks import smq2Q_XYCNOT
from pygsti.models import create_crosstalk_free_model
from pygsti.protocols import GateSetTomography, ProtocolData, make_simultaneous_gst_design

from pygsti.extras.crosstalkfreegst import (CrosstalkFreeForwardSimulator, add_single_qubit_instruments,
                                            create_simple_mcm_oneq_design, create_simple_mcm_twoq_design)

from .helpers import ERRORS, line_pspec
from .helpers import model as model_without_instruments


@pytest.fixture(scope='module')
def design():
    return make_simultaneous_gst_design(line_pspec(3, ('Iz',)), create_simple_mcm_oneq_design(1),
                                        create_simple_mcm_twoq_design(1), seed=1)


def noisy_model(seed=0, independent_gates=True):
    m = create_crosstalk_free_model(line_pspec(3, ('Iz',)), lindblad_error_coeffs=ERRORS,
                                    lindblad_parameterization='H+S', independent_gates=independent_gates)
    add_single_qubit_instruments(m, {'Iz': 'Z'})
    m.from_vector(np.random.default_rng(seed).uniform(-0.03, 0.03, m.num_params))
    return m


def max_difference(p, q):
    assert all(set(p[c]) == set(q[c]) for c in p)
    return max(abs(p[c][o] - q[c][o]) for c in p for o in p[c])


def test_probabilities_match_map_simulator(design):
    circuits = design.all_circuits_needing_data[::10]
    assert any(c.line_labels != (0, 1, 2) for c in circuits) and any('Iz' in c.str for c in circuits)
    ref, fast = noisy_model(), noisy_model()   # fresh models: pyGSTi's map simulator is right before caching
    ref.sim = 'map'
    fast.sim = CrosstalkFreeForwardSimulator()
    assert max_difference(ref.sim.bulk_probs(circuits), fast.sim.bulk_probs(circuits)) < 1e-12


def test_parameter_updates_are_seen(design):
    circuits = design.all_circuits_needing_data[::40]
    fast = noisy_model()
    fast.sim = CrosstalkFreeForwardSimulator()
    fast.sim.bulk_probs(circuits)
    x = fast.to_vector() + 0.01
    fast.from_vector(x)
    ref = noisy_model()
    ref.from_vector(x)
    ref.sim = 'map'
    assert max_difference(ref.sim.bulk_probs(circuits), fast.sim.bulk_probs(circuits)) < 1e-12


def test_explicit_state_preparation_and_measurement_labels(design):
    """A circuit may name the model's state preparation and measurement; that changes nothing."""
    circuits = design.all_circuits_needing_data[5::60]
    assert any(c.line_labels != (0, 1, 2) for c in circuits) and any('Iz' in c.str for c in circuits)
    m = noisy_model()
    m.sim = CrosstalkFreeForwardSimulator()
    for c in circuits:
        explicit = m.complete_circuit(c)
        assert explicit[0].name == 'rho0' and explicit[-1].name == 'Mdefault'
        assert max_difference({c: m.probabilities(c)}, {c: m.probabilities(explicit)}) < 1e-14


def test_changes_to_the_models_members_are_seen(design):
    """Lane models are kept between calls, and rebuilt when the model's gates or instruments change."""
    circuits = design.all_circuits_needing_data[::40]
    plain = [c for c in circuits if 'Iz' not in c.str]
    m = create_crosstalk_free_model(line_pspec(3, ('Iz',)), lindblad_error_coeffs=ERRORS,
                                    lindblad_parameterization='H+S', independent_gates=True)
    m.sim = CrosstalkFreeForwardSimulator()
    m.sim.bulk_probs(plain)
    lane_models = m.sim._lane_models
    m.from_vector(m.to_vector() + 0.01)
    m.sim.bulk_probs(plain)
    assert m.sim._lane_models is lane_models            # a parameter change needs no new lane models

    add_single_qubit_instruments(m, {'Iz': 'Z'})         # instruments added after the first use
    ref = noisy_model()
    m.from_vector(ref.to_vector())
    ref.sim = 'map'
    assert max_difference(ref.sim.bulk_probs(circuits), m.sim.bulk_probs(circuits)) < 1e-12

    for model in (m, ref):                              # an instrument replaced by another object
        add_single_qubit_instruments(model, {'Iz': 'Z'}, qubits=[1], parameterization='full TP')
    x = np.random.default_rng(4).uniform(-0.03, 0.03, ref.num_params)
    fresh = ref.copy()                                  # (pyGSTi's map simulator is right before caching)
    for model in (m, fresh):
        model.from_vector(model.to_vector() + x)
    assert max_difference(fresh.sim.bulk_probs(circuits), m.sim.bulk_probs(circuits)) < 1e-12


def test_copies_and_saved_models_keep_the_simulator(design, tmp_path):
    circuits = [c for c in design.all_circuits_needing_data[::40] if 'Iz' not in c.str]
    m = model_without_instruments(3)
    m.sim = CrosstalkFreeForwardSimulator(lane_simulator='map')
    p = m.sim.bulk_probs(circuits)
    assert all(isinstance(lane_model.sim, MapForwardSimulator) for _, (lane_model, _) in m.sim._lane_models.items())
    m.write(tmp_path / 'model.json')
    # (.copy() of the unpickled model, because pyGSTi leaves a pickled model's parameter indices stale)
    copies = {'model copy': m.copy(), 'pickled': pickle.loads(pickle.dumps(m)).copy(),
              'written and read': pygsti.models.Model.read(tmp_path / 'model.json')}
    assert m.sim.copy().lane_simulator == 'map'
    for how, copy in copies.items():
        assert isinstance(copy.sim, CrosstalkFreeForwardSimulator) and copy.sim.lane_simulator == 'map', how
        assert max_difference(p, copy.sim.bulk_probs(circuits)) < 1e-14, how
        copy.from_vector(copy.to_vector() + 0.01)       # the copy is tied to its own model, not the original's
        assert max_difference(p, copy.sim.bulk_probs(circuits)) > 1e-4, how
    assert max_difference(p, m.sim.bulk_probs(circuits)) == 0

    with_object = CrosstalkFreeForwardSimulator(lane_simulator=MapForwardSimulator())
    assert isinstance(with_object.copy().lane_simulator, MapForwardSimulator)


def test_unsupported_models_are_rejected():
    m = smq2Q_XYCNOT.target_model()
    m.sim = CrosstalkFreeForwardSimulator()
    with pytest.raises(ValueError, match='LocalNoiseModel'):
        m.probabilities(pygsti.circuits.Circuit('Gxpi2:0@(0,1)'))

    m = model_without_instruments(2)                    # two measurements: lanes use the default one only
    m.povm_blks['layers']['Mother'] = m.povm_blks['layers']['Mdefault'].copy()
    m.sim = CrosstalkFreeForwardSimulator()
    with pytest.raises(NotImplementedError, match='one measurement'):
        m.probabilities(pygsti.circuits.Circuit(['rho0', ('Gxpi2', 0), 'Mother'], line_labels=(0, 1)))


@pytest.mark.parametrize('independent_gates', [True, False])
def test_derivatives_match_finite_differences(design, independent_gates):
    """With shared gates (independent_gates=False) a parameter gets contributions from several lanes."""
    circuits = [c for c in design.all_circuits_needing_data if 'Iz' in c.str][:6]
    m = noisy_model(independent_gates=independent_gates)
    m.sim = CrosstalkFreeForwardSimulator()
    dp = m.sim.bulk_dprobs(circuits)
    x, eps = m.to_vector(), 1e-6
    fd = {c: {o: np.zeros(len(x)) for o in dp[c]} for c in circuits}
    for i in range(len(x)):
        for sign in (1, -1):
            xx = x.copy()
            xx[i] += sign * eps
            m.from_vector(xx)
            p = m.sim.bulk_probs(circuits)
            for c in circuits:
                for o in dp[c]:
                    fd[c][o][i] += sign * p[c][o] / (2 * eps)
    m.from_vector(x)
    assert max(np.max(np.abs(dp[c][o] - fd[c][o])) for c in circuits for o in dp[c]) < 1e-7


def test_array_filling_interfaces(design):
    """The base class's other entry points: probabilities with derivatives, parameter blocks, one circuit."""
    circuits = design.all_circuits_needing_data[::60]
    m = noisy_model()
    m.sim = CrosstalkFreeForwardSimulator()
    layout = m.sim.create_layout(circuits, array_types=('ep',))
    probs, dprobs = np.empty(len(layout)), np.empty((len(layout), m.num_params))
    m.sim.bulk_fill_probs(probs, layout)
    m.sim.bulk_fill_dprobs(dprobs, layout)
    assert np.max(np.abs(dprobs)) > 0.1

    probs2, dprobs2 = np.zeros(len(layout)), np.zeros((len(layout), m.num_params))
    m.sim.bulk_fill_dprobs(dprobs2, layout, pr_array_to_fill=probs2)
    assert np.array_equal(probs2, probs) and np.array_equal(dprobs2, dprobs)

    block = np.zeros((len(layout), 12))                  # (used by the base class's Hessians)
    m.sim._bulk_fill_dprobs_block(block, slice(2, 9), layout, slice(40, 47))
    assert np.array_equal(block[:, 2:9], dprobs[:, 40:47])
    assert not block[:, :2].any() and not block[:, 9:].any()

    for indices, circuit, outcomes in list(layout.iter_unique_circuits())[:3]:
        one = np.empty(len(outcomes))
        m.sim._compute_circuit_outcome_probabilities(one, circuit, outcomes, None)
        assert np.allclose(one, probs[indices], rtol=0, atol=1e-14)


def test_standard_pygsti_workflow(design):
    """simulate_data and the GateSetTomography protocol run with the simulator."""
    truth = noisy_model()
    truth.sim = CrosstalkFreeForwardSimulator()
    lists = [lst[::6] for lst in design.circuit_lists]
    small = pygsti.protocols.GateSetTomographyDesign(design.processor_spec, lists, qubit_labels=(0, 1, 2), nested=True)
    ds = pygsti.data.simulate_data(truth, small.all_circuits_needing_data, 1000, seed=2)
    start = noisy_model()
    start.from_vector(np.zeros(start.num_params) + 1e-3)
    start.sim = CrosstalkFreeForwardSimulator()
    result = GateSetTomography(start, gaugeopt_suite=None, verbosity=0).run(ProtocolData(small, ds))
    est = result.estimates['GateSetTomography'].models['final iteration estimate']
    assert isinstance(est.sim, CrosstalkFreeForwardSimulator)
    assert pygsti.tools.logl(est, ds) >= pygsti.tools.logl(truth, ds) - 1.0
