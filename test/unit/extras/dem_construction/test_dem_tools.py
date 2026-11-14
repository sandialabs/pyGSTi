import numpy as np
import pytest
import stim

import pygsti
from pygsti.baseobjs import Label, QubitSpace
from pygsti.extras.dem_construction import dem_tools
from pygsti.extras.dem_construction.dem_tools import build_dem_end2end
from pygsti.extras.dem_construction.pygsti_object_builders import build_model, stim_to_pygsti_circuit
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel
from pygsti.modelmembers.operations import ExpErrorgenOp, LindbladErrorgen
from pygsti.modelmembers.states import ComposedState, ComputationalBasisState

# Mid-circuit MR (a virtual qubit), two detectors with ideal value -1, an inverted measurement
# and two observables.
CIRC_C = stim.Circuit("""
R 0 1 2
X 0
CX 0 1
H 2
CX 1 2
H 2
MR 1
DETECTOR rec[-1]
CX 0 1
M !1
DETECTOR rec[-1] rec[-2]
M 0 2
OBSERVABLE_INCLUDE(0) rec[-2]
OBSERVABLE_INCLUDE(1) rec[-1]
""")
CIRC_T = stim.Circuit("""
R 0 1
CX 0 1
M 0 1
DETECTOR rec[-1] rec[-2]
OBSERVABLE_INCLUDE(0) rec[-1]
""")
# the converter puts the two CNOTs in one layer
CIRC_P = stim.Circuit("""
R 0 1 2 3
CX 0 1 2 3
M 0 1 2 3
DETECTOR rec[-1] rec[-2]
DETECTOR rec[-3] rec[-4]
OBSERVABLE_INCLUDE(0) rec[-1]
""")

S_ONLY = {('Gcnot', 0, 1): {('S', 'XI'): 0.01, ('S', 'ZY'): 0.004},
          ('Gcnot', 1, 2): {('S', 'IX'): 0.02, ('S', 'YY'): 0.003},
          ('Gh', 2): {('S', 'X'): 0.015},
          ('Gxpi', 0): {('S', 'Y'): 0.025},
          'prep': {('S', 'X'): 0.02},
          'povm': {('S', 'X'): 0.03}}

MIXED = {('Gcnot', 0, 1): {('H', 'XI'): 0.03, ('H', 'IX'): 0.02, ('S', 'XI'): 0.01, ('S', 'YI'): 0.01,
                           ('C', 'XI', 'YI'): 0.004, ('A', 'XI', 'YI'): 0.006},
         ('Gcnot', 1, 2): {('H', 'IX'): 0.02, ('H', 'IY'): 0.01, ('S', 'IY'): 0.01, ('S', 'ZY'): 0.01,
                           ('C', 'IY', 'ZY'): 0.005, ('A', 'IY', 'ZY'): -0.003},
         ('Gh', 2): {('H', 'X'): 0.02, ('S', 'Z'): 0.01},
         ('Gxpi', 0): {('H', 'Y'): 0.02}}


def _scaled(coeffs, scale):
    return {k: {t: r * scale for t, r in v.items()} for k, v in coeffs.items()}


def _pspec(n_qubits, names, connections):
    availability = {name: sorted(connections[name]) for name in names if name in connections}
    return pygsti.processors.QubitProcessorSpec(n_qubits, names, qubit_labels=list(range(n_qubits)),
                                                availability=availability)


def _components(pyg_circuit):
    return [c for i in range(pyg_circuit.depth) for c in pyg_circuit.layer_label(i).components]


def _spec_for(circuit, physical):
    """ A processor spec on the stim qubits (`physical`) or on the virtual qubits. """
    pyg, _, _, v2p = stim_to_pygsti_circuit(circuit)
    comps = _components(pyg)
    names = sorted({c.name for c in comps})
    to_qubit = (lambda v: v2p[v]) if physical else (lambda v: v)
    connections = {name: {tuple(to_qubit(q) for q in c.qubits) for c in comps if c.name == name and len(c.qubits) == 2}
                   for name in names}
    connections = {name: conns for name, conns in connections.items() if conns}
    return _pspec(circuit.num_qubits if physical else len(v2p), names, connections)


def _physical_model(circuit, coeffs):
    return pygsti.models.create_crosstalk_free_model(_spec_for(circuit, True), evotype='stabilizer',
                                                     lindblad_error_coeffs=coeffs)


def _set_prep(model, prep_coeffs, evotype):
    """ Replaces the model's preparation with one carrying `prep_coeffs` on all of its qubits. """
    n = model.state_space.num_qubits
    errorgen = LindbladErrorgen.from_elementary_errorgens(prep_coeffs, state_space=QubitSpace(n), evotype=evotype)
    model.prep_blks['layers']['rho0'] = ComposedState(ComputationalBasisState([0] * n, evotype=evotype),
                                                      ExpErrorgenOp(errorgen))
    return model


def _dense_distribution(circuit, coeffs, prep_coeffs=None):
    """
    Distribution over detector then observable bits, by dense simulation on the virtual qubits.
    `prep_coeffs` replaces the preparation, for a circuit without virtual qubits.
    """
    pyg, dets, obs, v2p = stim_to_pygsti_circuit(circuit)
    comps = _components(pyg)
    virtual_coeffs = {}
    for key, terms in coeffs.items():
        if isinstance(key, tuple):  # every virtual copy of the physical gate
            for c in comps:
                if c.name == key[0] and tuple(v2p[q] for q in c.qubits) == tuple(key[1:]):
                    virtual_coeffs[(c.name,) + tuple(c.qubits)] = terms
        else:
            virtual_coeffs[key] = terms
    dense = pygsti.models.create_crosstalk_free_model(_spec_for(circuit, False), evotype='densitymx',
                                                      lindblad_error_coeffs=virtual_coeffs)
    if prep_coeffs is not None:
        _set_prep(dense, prep_coeffs, 'densitymx')
    probs = dense.probabilities(pyg)
    n = len(v2p)
    sim = stim.TableauSimulator()
    sim.do_tableau(pyg.convert_to_stim_tableau(), list(range(n)))
    paulis = [p * sim.peek_observable_expectation(p) for p in dets + obs]
    dist = {}
    for (bits,), prob in probs.items():
        key = tuple((sum(int(bits[q]) for q in range(n) if p[q] == 3) + (p.sign.real < 0)) % 2 for p in paulis)
        dist[key] = dist.get(key, 0) + prob
    return dist


def _dem_distribution(dem):
    """ XOR-convolution of the independent errors of a DEM. """
    n = dem.num_detectors + dem.num_observables
    dist = {(0,) * n: 1.0}
    for inst in dem.flattened():
        if inst.type != 'error':
            continue
        p = inst.args_copy()[0]
        flip = [0] * n
        for t in inst.targets_copy():
            flip[t.val if t.is_relative_detector_id() else dem.num_detectors + t.val] ^= 1
        new = {}
        for bits, q in dist.items():
            moved = tuple(a ^ b for a, b in zip(bits, flip))
            new[bits] = new.get(bits, 0) + q * (1 - p)
            new[moved] = new.get(moved, 0) + q * p
        dist = new
    return dist


def _tv(a, b):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


def _tv_to_dense(circuit, coeffs, **kwargs):
    dem = build_dem_end2end(circuit, _physical_model(circuit, coeffs), **kwargs)
    assert dem.num_detectors == circuit.num_detectors
    assert dem.num_observables == circuit.num_observables
    return _tv(_dem_distribution(dem), _dense_distribution(circuit, coeffs))


def test_s_only_noise_on_every_gate_placement_matches_dense():
    assert _tv_to_dense(CIRC_C, S_ONLY) < 1e-9


def test_build_model_one_qubit_povm_applies_to_every_qubit():
    gate_noise = {'Gcnot': {('S', 'XI'): 0.01}, 'Gh': {('S', 'X'): 0.015}, 'Gxpi': {('S', 'Y'): 0.02}}
    pspec = _pspec(3, ['Gcnot', 'Gh', 'Gxpi'], {'Gcnot': [(0, 1), (1, 2)]})
    model = build_model(gate_noise, pspec, ['Gh', 'Gxpi'], ['Gcnot'], meas_err=0.03, prep_err=0.02)
    dense_coeffs = {'Gcnot': gate_noise['Gcnot'], 'Gh': gate_noise['Gh'], 'Gxpi': gate_noise['Gxpi'],
                    'prep': {('S', 'X'): 0.02}, 'povm': {('S', 'X'): 0.03}}
    dem = build_dem_end2end(CIRC_C, model)
    assert _tv(_dem_distribution(dem), _dense_distribution(CIRC_C, dense_coeffs)) < 1e-9


def test_two_qubit_preparation_noise_stays_on_each_qubit():
    gate_noise = {'Gcnot': {('S', 'XI'): 0.01}}
    prep = {('S', 'XI'): 0.01, ('S', 'IX'): 0.03}
    model = _set_prep(_physical_model(CIRC_T, gate_noise), prep, 'stabilizer')
    dem = build_dem_end2end(CIRC_T, model)
    assert _tv(_dem_distribution(dem), _dense_distribution(CIRC_T, gate_noise, prep)) < 1e-9


def test_single_h_term_matches_dense():
    assert _tv_to_dense(CIRC_T, {'Gcnot': {('H', 'IX'): 0.05}}) < 1e-9


def test_two_h_terms_on_one_event_match_dense():
    coeffs = {'Gcnot': {('H', 'XI'): 0.04, ('H', 'YI'): 0.02}}
    assert _tv_to_dense(CIRC_T, coeffs) < 1e-5


@pytest.mark.parametrize('order', [1, 2])
def test_mixed_noise_converges_at_second_order(order):
    pspec_model = _physical_model(CIRC_C, MIXED)
    # the built layers carry the requested coefficients
    for key, terms in MIXED.items():
        coeffs = pspec_model.circuit_layer_operator(Label(key[0], key[1:])).errorgen_coefficients(label_type='local')
        coeffs = {k: v for k, v in coeffs.items() if v != 0}
        offset = key[1]
        width = 3
        expected = {}
        for term, rate in terms.items():
            padded = tuple('I' * offset + p + 'I' * (width - offset - len(p)) for p in term[1:])
            expected[(term[0], padded)] = rate
        assert {(k.errorgen_type, k.basis_element_labels): v for k, v in coeffs.items()} == pytest.approx(expected)

    tvs = [_tv_to_dense(CIRC_C, _scaled(MIXED, scale), zassenhaus_order=order) for scale in (0.5, 0.25)]
    assert tvs[0] < 1e-3
    assert tvs[0] >= 3 * tvs[1]


def test_parallel_layer_places_noise_on_each_gate():
    coeffs = {('Gcnot', 0, 1): {('S', 'XI'): 0.01, ('S', 'ZY'): 0.004}, ('Gcnot', 2, 3): {('S', 'IX'): 0.02}}
    pyg, _, _, _ = stim_to_pygsti_circuit(CIRC_P)
    assert [len(pyg.layer_label(i).components) for i in range(pyg.depth)] == [2]
    assert _tv_to_dense(CIRC_P, coeffs) < 1e-9


def _refusal_cases():
    cases = {}
    noise = {'Gh': {('S', 'X'): 0.01}}
    one_qubit = _pspec(1, ['Gh'], {})
    cases['nondeterministic detector'] = (
        stim.Circuit('R 0\nH 0\nM 0\nDETECTOR rec[-1]'),
        pygsti.models.create_crosstalk_free_model(one_qubit, evotype='stabilizer', lindblad_error_coeffs=noise))
    cases['nondeterministic observable'] = (
        stim.Circuit('R 0\nH 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]'),
        pygsti.models.create_crosstalk_free_model(one_qubit, evotype='stabilizer', lindblad_error_coeffs=noise))
    cases['missing gate'] = (
        CIRC_T, pygsti.models.create_crosstalk_free_model(_pspec(2, ['Gh'], {}), evotype='stabilizer',
                                                          lindblad_error_coeffs=noise))
    # qubit 2 is only reset and measured
    cases['too few qubits'] = (
        stim.Circuit('R 0 1 2\nCX 0 1\nM 0 1 2\nDETECTOR rec[-1]\nDETECTOR rec[-2]'),
        pygsti.models.create_crosstalk_free_model(
            _pspec(2, ['Gcnot'], {'Gcnot': [(0, 1)]}), evotype='stabilizer',
            lindblad_error_coeffs={'Gcnot': {('S', 'XI'): 0.01}}))
    spectator = stim.Circuit('R 0 1 2\nH 2\nM 0 1 2\nDETECTOR rec[-1] rec[-2]')
    cases['spectator noise'] = (
        spectator,
        pygsti.models.create_cloud_crosstalk_model(
            _pspec(3, ['Gh'], {}), evotype='stabilizer',
            lindblad_error_coeffs={('Gh', 2): {('S', 'X:2'): 0.01, ('S', 'X:0'): 0.01}}))
    cases['depolarizing densitymx'] = (
        CIRC_T, pygsti.models.create_crosstalk_free_model(
            _pspec(2, ['Gcnot'], {'Gcnot': [(0, 1)]}), evotype='densitymx', depolarization_strengths={'Gcnot': 0.01}))

    two_qubit_prep = pygsti.models.create_crosstalk_free_model(
        _pspec(2, ['Gcnot'], {'Gcnot': [(0, 1)]}), evotype='stabilizer',
        lindblad_error_coeffs={'Gcnot': {('S', 'XI'): 0.01}})
    cases['two-qubit preparation term'] = (CIRC_T, _set_prep(two_qubit_prep, {('S', 'XX'): 0.01}, 'stabilizer'))

    pspec = _pspec(2, ['Gcnot'], {'Gcnot': [(0, 1)]})
    cases['reuse without reset'] = (
        stim.Circuit('R 0 1\nCX 0 1\nM 1\nCX 0 1\nM 0 1\nDETECTOR rec[-1]'),
        build_model({'Gcnot': {('S', 'XI'): 0.01}}, pspec, [], ['Gcnot'], meas_err=0.01))
    return cases


_REFUSALS = _refusal_cases()


@pytest.mark.parametrize('name', list(_REFUSALS))
def test_unsupported_input_raises(name):
    circuit, model = _REFUSALS[name]
    with pytest.raises(ValueError):
        build_dem_end2end(circuit, model)


def test_noise_free_model_gives_empty_dem_with_circuit_counts():
    model = _physical_model(CIRC_T, {})
    dem = build_dem_end2end(CIRC_T, model)
    assert not any(inst.type == 'error' for inst in dem.flattened())
    assert dem.num_detectors == CIRC_T.num_detectors
    assert dem.num_observables == CIRC_T.num_observables


def test_gate_free_circuit_with_measurement_noise_gives_one_error_per_detector():
    circuit = stim.Circuit('R 0 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]')
    model = _physical_model(circuit, {'povm': {('S', 'X'): 0.03}})
    dem = build_dem_end2end(circuit, model)
    errors = [inst for inst in dem.flattened() if inst.type == 'error']
    assert len(errors) == 2
    assert dem.num_detectors == 2


def test_add_to_dem_converts_rates_to_real_floats():
    sim = stim.TableauSimulator()
    sim.do_tableau(stim.Tableau(1), [0])
    label = LocalStimErrorgenLabel('S', (stim.PauliString('X'),))
    dem = dem_tools.add_to_dem({'1': 0.0}, ['1'], [{label: complex(0.01, 0)}], [stim.PauliString('Z')], sim)
    assert isinstance(dem['1'], float)
    assert dem['1'] == pytest.approx((1 - np.exp(-0.02)) / 2)
    with pytest.raises(ValueError):
        dem_tools.add_to_dem({'1': 0.0}, ['1'], [{label: complex(0.01, 1e-3)}], [stim.PauliString('Z')], sim)
