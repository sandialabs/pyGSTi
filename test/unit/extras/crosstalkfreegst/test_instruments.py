"""Single-qubit instruments (mid-circuit measurements) in crosstalk-free models."""
import numpy as np
import pytest

import pygsti
from pygsti.baseobjs import Label
from pygsti.circuits import Circuit
from pygsti.modelpacks import smq2Q_XYCPHASE, smq2Q_XYICNOT
from pygsti.models import create_crosstalk_free_model
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols import GateSetTomographyDesign, make_simultaneous_gst_design

from pygsti.extras.crosstalkfreegst import (FactorizedObjective, LaneDecomposition, LaneModels, LanePool,
                                            add_single_qubit_instruments, apply_product_gauge, check_factorizable,
                                            create_general_crosstalk_free_model, create_simple_mcm_oneq_design,
                                            create_simple_mcm_twoq_design, error_summary, format_error_summary,
                                            gauge_optimize, ideal_z_measurement, lane_outcome_distributions,
                                            run_crosstalk_free_gst, simulate_lanewise_data,
                                            simulate_lanewise_marginals, split_circuit_into_lanes)
from pygsti.extras.crosstalkfreegst.diagnostics import predicted_distributions
from pygsti.extras.crosstalkfreegst.gaugeopt import tp_gauge

from .helpers import ERRORS, joint_logl, line_pspec

QUBITS = (0, 1, 2)
SUPPORTS = [(0,), (1,), (2,), (0, 1), (1, 2)]


@pytest.fixture(scope='module')
def mcm_design():
    """A 3-qubit simultaneous design with mid-circuit measurements, circuits in model line order."""
    d = make_simultaneous_gst_design(line_pspec(3, ('Iz',)), create_simple_mcm_oneq_design(1),
                                     create_simple_mcm_twoq_design(1), seed=1)
    # pyGSTi's n-qubit models mishandle circuits whose line order differs from theirs (a static,
    # cached marginalized POVM); reorder lines so the full-model comparisons below are valid.
    lists = [[c.reorder_lines(QUBITS) for c in lst] for lst in d.circuit_lists]
    return GateSetTomographyDesign(line_pspec(3, ('Iz',)), lists, qubit_labels=QUBITS, nested=True)


def lindblad_model_with_instruments(seed=0, scale=0.03):
    """An H+S crosstalk-free model with CPTPLND Z-measurements on every qubit, at random parameters."""
    m = create_crosstalk_free_model(line_pspec(3, ('Iz',)), lindblad_error_coeffs=ERRORS,
                                    lindblad_parameterization='H+S', independent_gates=True)
    add_single_qubit_instruments(m, {'Iz': 'Z'})
    m.from_vector(np.random.default_rng(seed).uniform(-scale, scale, m.num_params))
    return m


def mcm_circuits(design):
    return [c for c in design.all_circuits_needing_data if 'Iz' in c.str]


def lane_models_of(m):
    lm = LaneModels.from_model(m)
    for s in SUPPORTS:
        lm[s]
    return lm


def test_undeclared_instruments_are_rejected():
    m = create_general_crosstalk_free_model(line_pspec(2))   # a spec without instrument names
    with pytest.raises(ValueError, match='instrument_names'):
        add_single_qubit_instruments(m, {'Iz': 'Z'})


def test_instrument_outcomes_are_marginalized_in_pygsti_order():
    # pyGSTi stores the layer [Iz:1 Iz:0] as [Iz:0 Iz:1] and records outcomes in stored order.
    c = Circuit([[('Iz', 2)], [('Iz', 1), ('Iz', 0)]], line_labels=QUBITS)
    ds = pygsti.data.DataSet()
    ds.add_count_dict(c, {('1', '0', '1', '010'): 5})  # Iz:2 -> '1', Iz:0 -> '0', Iz:1 -> '1'
    ds.done_adding_data()
    marg = {s: m for s, _, m in LaneDecomposition(QUBITS, ds, [c], instrument_names=('Iz',)).marginals[c]}
    assert marg == {(0,): {('0', '0'): 5.0}, (1,): {('1', '1'): 5.0}, (2,): {('1', '0'): 5.0}}


def test_factorized_likelihood_with_instruments_equals_joint_likelihood(mcm_design):
    m = lindblad_model_with_instruments()
    circuits = mcm_design.all_circuits_needing_data
    assert len(mcm_circuits(mcm_design)) > 100
    assert any(sum(comp.name == 'Iz' for comp in layer.components) > 1
               for c in circuits for layer in c.layertup)  # simultaneous measurements on several qubits
    ds = pygsti.data.simulate_data(m, circuits, 100, seed=1)
    lane_models = LaneModels.from_model(m)
    total = 0.0
    for support, (lds, lcs) in LaneDecomposition(QUBITS, ds, circuits, ('Iz',)).lane_datasets().items():
        total += joint_logl(lane_models[support][0], lds, lcs)
    assert total == pytest.approx(joint_logl(m, ds, circuits), rel=1e-12)


def test_instruments_on_chosen_qubits():
    pspec = line_pspec(3, ('Iz', 'Ireset'))
    reset = {'0': ideal_z_measurement()['0'], '1': ideal_z_measurement()['0'] @ np.diag([1, 1, -1, -1.])}  # X on 1
    m = create_general_crosstalk_free_model(pspec)
    before = m.num_params
    add_single_qubit_instruments(m, {'Iz': 'Z', 'Ireset': reset}, qubits={'Iz': (0, 2), 'Ireset': (1,)})
    assert set(m.instrument_blks['layers']) == {Label('Iz', 0), Label('Iz', 2), Label('Ireset', 1)}
    assert m.num_params == before + 3 * 28   # a full TP instrument with two outcomes has 16 + 12 parameters
    c = Circuit([[Label('Gxpi2', 0), Label('Gxpi2', 1), Label('Gi', 2)],
                 [Label('Iz', 0), Label('Ireset', 1), Label('Gi', 2)]], line_labels=QUBITS)
    p = m.probabilities(c)
    assert sum(p.values()) == pytest.approx(1) and p[('0', '1', '000')] == pytest.approx(0.25)
    assert sum(v for ol, v in p.items() if ol[-1][1] == '1') == pytest.approx(0, abs=1e-12)   # qubit 1 was reset
    check_factorizable(m)


def test_simulating_with_undeclared_instruments_is_rejected():
    lane_models = LaneModels.from_model(lindblad_model_with_instruments())
    lane_models.processor_spec = line_pspec(3)   # as if the instrument names had not been declared
    c = Circuit([[Label('Gxpi2', 0), Label('Gxpi2', 1)], [Label('Iz', 0), Label('Gi', 1)]], line_labels=(0, 1))
    with pytest.raises(ValueError, match='mid-circuit outcomes'):   # not data without the mid-circuit outcomes
        simulate_lanewise_data(lane_models, [c], 10, seed=0)


def test_simulated_data_follows_each_circuits_line_order():
    lane_models = lane_models_of(lindblad_model_with_instruments())
    layers = [[Label('Gxpi2', 0), Label('Gypi2', 1), Label('Gxpi2', 2)],
              [Label('Gcnot', (0, 1)), Label('Iz', 2)],
              [Label('Iz', 1), Label('Gi', 0), Label('Gypi2', 2)],
              [Label('Gxpi2', 0), Label('Gxpi2', 1), Label('Gi', 2)]]
    c = Circuit(layers, line_labels=(0, 1, 2))
    permuted = Circuit(layers, line_labels=(2, 0, 1))
    # the same draws, so the same counts, with the final bits in each circuit's own line order
    counts = simulate_lanewise_data(lane_models, [c], 500, seed=7)[c].counts
    permuted_counts = simulate_lanewise_data(lane_models, [permuted], 500, seed=7)[permuted].counts
    assert len(counts) > 10 and sum(counts.values()) == 500
    assert all(len(ol) == 3 for ol in counts)   # two mid-circuit outcomes, then the final bits
    assert dict(permuted_counts) == {ol[:2] + (ol[2][2] + ol[2][0] + ol[2][1],): n for ol, n in counts.items()}
    with pytest.raises(NotImplementedError, match='instruments'):
        lane_outcome_distributions(lane_models, [split_circuit_into_lanes(c, QUBITS)])


def test_twoq_mcm_design_adds_the_oneq_measurement_circuits_on_each_qubit():
    for modelpack, twoq_gate in [(None, 'Gcnot'), (smq2Q_XYCPHASE, 'Gcphase')]:
        design = create_simple_mcm_twoq_design(2, modelpack=modelpack)
        assert twoq_gate in design.processor_spec.gate_names and design.processor_spec.instrument_names == ('Iz',)
        base = (modelpack or smq2Q_XYICNOT).create_gst_experiment_design(max_max_length=2, qubit_labels=(0, 1))
        added = set(design.all_circuits_needing_data) - set(base.all_circuits_needing_data)
        oneq = [c for c in create_simple_mcm_oneq_design(2).all_circuits_needing_data if 'Iz' in c.str]
        assert len(oneq) == 72   # fiducial pairs around Iz and around Iz Iz
        expected = {Circuit(c.map_state_space_labels({0: q}).layertup, line_labels=(0, 1))
                    for q in (0, 1) for c in oneq}
        assert added == expected and len(added) == 144
        assert [len(lst) for lst in design.circuit_lists] == [len(lst) + 72 * (i + 1)
                                                              for i, lst in enumerate(base.circuit_lists)]


def test_oneq_mcm_design_determines_every_gauge_invariant_parameter():
    pspec = QubitProcessorSpec(1, ['Gi', 'Gxpi2', 'Gypi2'], instrument_names=('Iz',), qubit_labels=(0,))
    design = create_simple_mcm_oneq_design(1)
    assert 'Iz' in design.processor_spec.instrument_names and '{idle}' in design.processor_spec.gate_names
    # the design writes idles as empty layers; make them explicit
    circuits = [Circuit([layer if layer.components else Label('Gi', 0) for layer in c.layertup], line_labels=(0,))
                for c in design.all_circuits_needing_data]
    m = create_general_crosstalk_free_model(pspec, instruments={'Iz': 'Z'}, simulator='matrix')
    m.from_vector(m.to_vector() + 0.02 * np.random.default_rng(1).standard_normal(m.num_params))
    jacobian = np.vstack([np.array(list(m.sim.dprobs(c).values())) for c in circuits])
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    # 3 gates x 12 + instrument 28 + state 3 + POVM 4, less the 12 gauge directions; the model's other
    # 9 + 8 SPAM error-map parameters are redundant
    assert m.num_params == 88
    assert int((singular_values > 1e-8 * singular_values[0]).sum()) == 36 + 28 + 3 + 4 - 12


def test_twoq_mcm_design_determines_every_gauge_invariant_parameter():
    def with_explicit_idles(c):
        layers = []
        for layer in c.layertup:
            idle = [q for q in (0, 1) if all(q not in comp.sslbls for comp in layer.components)]
            layers.append([Label('Gii', (0, 1))] if len(idle) == 2
                          else list(layer.components) + [Label('Gi', q) for q in idle])
        return Circuit(layers, line_labels=(0, 1))

    circuits = [with_explicit_idles(c) for c in create_simple_mcm_twoq_design(1).all_circuits_needing_data]
    m = create_general_crosstalk_free_model(line_pspec(2, ('Iz',)), instruments={'Iz': 'Z'}, simulator='matrix')
    m.from_vector(m.to_vector() + 0.02 * np.random.default_rng(1).standard_normal(m.num_params))

    jacobian = {c: np.array(list(m.sim.dprobs(c).values())) for c in circuits}

    def rank(circuits):
        singular_values = np.linalg.svd(np.vstack([jacobian[c] for c in circuits]), compute_uv=False)
        return int((singular_values > 1e-8 * singular_values[0]).sum())

    # per qubit: 3 gates x 12 + instrument 28 + state 3 + POVM 4, less 12 gauge directions; Gcnot and Gii: 240 each
    identifiable = 2 * (36 + 28 + 3 + 4 - 12) + 2 * 240
    assert m.num_params == identifiable + 2 * 12 + 2 * (9 + 8)   # plus gauge and redundant SPAM parameters
    assert rank(circuits) == identifiable
    # without the measurement circuits nothing is learned about either instrument
    assert rank([c for c in circuits if 'Iz' not in c.str]) == identifiable - 2 * 28


def test_lanewise_simulation_with_instruments(mcm_design):
    m = lindblad_model_with_instruments(scale=0.05)
    c = max(mcm_circuits(mcm_design), key=lambda c: c.str.count('Iz'))
    ds = simulate_lanewise_data(m, [c], 200000, seed=3)
    p = m.probabilities(c)
    assert set(ds[c].counts.keys()) <= set(p.keys())
    for ol, n in ds[c].counts.items():
        assert n / 200000 == pytest.approx(p[ol], abs=5e-3)
    dec = simulate_lanewise_marginals(m, [c], 200000, seed=4)
    lane_models = LaneModels.from_model(m)
    for support, lane_circuit, counts in dec.marginals[c]:
        lp = lane_models[support][0].probabilities(lane_circuit)
        for ol, n in (counts.items() if isinstance(counts, dict) else []):
            assert n / 200000 == pytest.approx(lp[ol], abs=5e-3)


@pytest.mark.parametrize('parameterization', ['full TP', 'full'])
def test_product_gauge_leaves_instrument_probabilities_unchanged(mcm_design, parameterization):
    rng = np.random.default_rng(5)
    est = create_general_crosstalk_free_model(line_pspec(3, ('Iz',)))
    add_single_qubit_instruments(est, {'Iz': 'Z'}, parameterization=parameterization)
    assert est.num_params == 3 * (28 if parameterization == 'full TP' else 32) + 1140
    est.from_vector(est.to_vector() + rng.normal(0, 1e-3, est.num_params))
    circuits = mcm_circuits(mcm_design)[::5]
    before = est.sim.bulk_probs(circuits)
    lm = lane_models_of(est)
    apply_product_gauge(lm, {q: tp_gauge(rng.normal(0, 0.05, 12)) for q in QUBITS})
    after = lm.to_model(est).sim.bulk_probs(circuits)  # a fresh copy, so no stale caches
    assert max(abs(before[c][o] - after[c][o]) for c in circuits for o in before[c]) < 1e-12
    assert not np.allclose(lm.current_vector(), lane_models_of(est).current_vector())


def test_gauge_optimization_with_instruments_is_gauge_invariant():
    pspec = line_pspec(3, ('Iz',))
    rng = np.random.default_rng(6)
    est = create_general_crosstalk_free_model(pspec, instruments={'Iz': 'Z'})
    est.from_vector(est.to_vector() + rng.normal(0, 1e-3, est.num_params))
    target = lane_models_of(create_general_crosstalk_free_model(pspec, instruments={'Iz': 'Z'}))
    a, b = lane_models_of(est), lane_models_of(est)
    apply_product_gauge(b, {q: tp_gauge(rng.normal(0, 0.05, 12)) for q in QUBITS})
    ra, rb = gauge_optimize(a, target), gauge_optimize(b, target)
    assert ra.fn_after == pytest.approx(rb.fn_after, rel=1e-6)
    assert np.allclose(a.current_vector(), b.current_vector(), atol=1e-6)


def test_measurement_error_rates_in_summary():
    pspec = line_pspec(3, ('Iz',))
    flip = 0.02  # outcome misreported with probability `flip`; post-measurement state is the true one
    P = ideal_z_measurement()
    noisy = {'0': (1 - flip) * P['0'] + flip * P['1'], '1': flip * P['0'] + (1 - flip) * P['1']}
    est = create_general_crosstalk_free_model(pspec, instruments={'Iz': noisy})
    target = lane_models_of(create_general_crosstalk_free_model(pspec, instruments={'Iz': 'Z'}))
    lm = lane_models_of(est)
    rows = [r for r in error_summary(lm, target, diamond=False) if r['kind'] == 'instrument']
    assert len(rows) == 3
    for r in rows:
        assert r['mcm_p1_given_0'] == pytest.approx(flip) and r['mcm_p0_given_1'] == pytest.approx(flip)
        assert r['instrument_infidelity'] > 0
    p = lm[(1,)][0].probabilities(Circuit([('Iz', 1)], line_labels=(1,)))
    assert p[('1', '0')] + p[('1', '1')] == pytest.approx(flip)
    ideal_rows = [r for r in error_summary(target, target, diamond=False) if r['kind'] == 'instrument']
    assert all(abs(r['instrument_infidelity']) < 1e-12 for r in ideal_rows)
    # gates, then instruments, then SPAM, each in qubit order
    all_rows = error_summary(lm, target, diamond=False)
    kinds = [r['kind'] for r in all_rows]
    assert kinds == sorted(kinds) and set(kinds) == {'gate', 'instrument', 'spam'}
    assert [r['qubits'] for r in rows] == [(0,), (1,), (2,)]
    assert 'Iz          1' in format_error_summary(all_rows)
    # Only the instruments differ from the target: each of an instrument's two elements is off by
    # flip * (P0 - P1), of squared norm 2 flip^2, and `instrument_weight` multiplies that.
    for weight in (1.0, 3.0):
        fresh = lane_models_of(est)
        distance = gauge_optimize(fresh, target, instrument_weight=weight, max_iterations=0).fn_before
        assert distance == pytest.approx(weight * 3 * 2 * 2 * flip**2)


def test_dense_instruments_keep_the_marquardt_damping():
    from pygsti.extras.crosstalkfreegst.fit import _resolve_damping
    pspec = line_pspec(2, ('Iz',))
    for parameterization in ('full TP', 'full'):
        general = create_general_crosstalk_free_model(pspec)
        add_single_qubit_instruments(general, {'Iz': 'Z'}, parameterization=parameterization)
        lane_models = LaneModels.from_model(general)
        lane_models[(0, 1)]
        assert _resolve_damping('auto', lane_models) == 'marquardt'
        assert _resolve_damping('identity', lane_models) == 'identity'
    lindblad = create_crosstalk_free_model(pspec, lindblad_error_coeffs=ERRORS, lindblad_parameterization='H+S')
    add_single_qubit_instruments(lindblad, {'Iz': 'Z'})
    lane_models = LaneModels.from_model(lindblad)
    lane_models[(0, 1)]
    assert _resolve_damping('auto', lane_models) == 'identity'
    assert _resolve_damping('marquardt', lane_models) == 'marquardt'


def test_num_elements_counts_unobserved_outcomes():
    """An ideal measurement of |0> gives one outcome, but the lane circuit has four possible ones."""
    m = create_general_crosstalk_free_model(line_pspec(2, ('Iz',)), instruments={'Iz': 'Z'})
    c = Circuit([('Iz', 0), ('Gi', 1)], line_labels=(0, 1))
    ds = simulate_lanewise_data(m, [c], 100, seed=0)
    assert dict(ds[c].counts) == {('0', '00'): 100}
    lane_models = LaneModels.from_model(m)
    lane_data = LaneDecomposition((0, 1), ds, [c], instrument_names=('Iz',)).lane_datasets()
    for objective in ('logl', 'chi2'):
        serial = FactorizedObjective(lane_models, lane_data, objective)
        assert serial.num_elements == 4 + 2
        assert 0 <= serial.fn(lane_models.current_vector()) < 0.01   # not exactly 0: pyGSTi's regularization


def test_fit_with_instruments(mcm_design):
    def factory(ps):
        m = create_crosstalk_free_model(ps, lindblad_error_coeffs=ERRORS, lindblad_parameterization='H+s',
                                        independent_gates=True)
        return add_single_qubit_instruments(m, {'Iz': 'Z'})
    pspec = line_pspec(3, ('Iz',))
    truth, start = LaneModels(pspec, factory), LaneModels(pspec, factory)
    for s in SUPPORTS:
        truth[s]
    x = truth.current_vector()
    rng = np.random.default_rng(7)
    truth.set_vector(x + rng.uniform(0, 0.01, len(x)))  # nonnegative, so stochastic rates stay physical
    data = simulate_lanewise_marginals(truth, mcm_design.all_circuits_needing_data, 1000, seed=8)
    res = run_crosstalk_free_gst(data, start, mcm_design.circuit_lists, processes=2, verbosity=0)
    lane_data = data.lane_datasets(mcm_design.circuit_lists[-1])
    final = FactorizedObjective(start, lane_data, 'logl')
    at_truth = FactorizedObjective(truth, lane_data, 'logl')
    assert final.fn(res.x) <= at_truth.fn(truth.current_vector())
    assert res.iterations[-1].num_elements == final.num_elements   # counted by the workers and in this process


def test_wide_lanes_with_instruments_match_the_whole_register():
    # Random circuits on a 5-qubit line, with lanes of 1 to 5 qubits and mid-circuit measurements anywhere in
    # them, against pyGSTi's simulation of the whole register (all the other tests have lanes of 1 or 2 qubits).
    n, rng = 5, np.random.default_rng(0)
    qubits = tuple(range(n))
    m = create_crosstalk_free_model(line_pspec(n, ('Iz',)), lindblad_error_coeffs=ERRORS,
                                    lindblad_parameterization='H+S', independent_gates=True)
    add_single_qubit_instruments(m, {'Iz': 'Z'})
    m.from_vector(rng.uniform(-0.2, 0.2, m.num_params))   # large errors, different on every qubit
    lane_models = LaneModels.from_model(m)

    def random_circuit():
        edges = [(q, q + 1) for q in range(n - 1) if rng.random() < 0.55]
        layers, num_instruments = [], 0
        for _ in range(rng.integers(3, 6)):
            comps, used = [], set()
            for a, b in rng.permutation(edges):
                if a not in used and b not in used and rng.random() < 0.5:
                    comps.append(Label('Gcnot', (int(a), int(b))))
                    used |= {int(a), int(b)}
            for q in sorted(set(qubits) - used):
                name = str(rng.choice(['Gxpi2', 'Gypi2', 'Iz', 'Gi', ''], p=[0.3, 0.2, 0.25, 0.1, 0.15]))
                if name == 'Iz' and num_instruments == 5:   # (keeps the number of joint outcomes down)
                    name = 'Gxpi2'
                if name:
                    comps.append(Label(name, (q,)))
                    num_instruments += name == 'Iz'
            comps = comps or [Label('Gxpi2', (0,))]
            layers.append([comps[i] for i in rng.permutation(len(comps))])   # any order within the layer
        return Circuit(layers, line_labels=qubits)

    worst, seen = 0.0, set()
    for _ in range(30):
        c = random_circuit()
        joint = {(ol if isinstance(ol, tuple) else (ol,)): p for ol, p in m.probabilities(c).items()}
        decomposition = LaneDecomposition(qubits, instrument_names=('Iz',))
        decomposition.add(c, joint)
        # each lane's marginal of the joint distribution is the lane model's distribution of the lane circuit
        for support, lane_circuit, marginal in decomposition.marginals[c]:
            k = len(support)
            p = lane_models[support][0].probabilities(lane_circuit)
            num_instruments = sum(comp.name == 'Iz' for layer in lane_circuit.layertup for comp in layer.components)
            seen.add((k, num_instruments > 0))
            assert isinstance(marginal, dict) == (num_instruments > 0)
            if not isinstance(marginal, dict):
                marginal = {(format(i, f'0{k}b'),): v for i, v in enumerate(marginal)}
            p = {(ol if isinstance(ol, tuple) else (ol,)): v for ol, v in p.items()}
            worst = max([worst] + [abs(p.get(ol, 0.0) - marginal.get(ol, 0.0)) for ol in set(p) | set(marginal)])
        # the joint distribution is the product of the lanes' distributions, outcome by outcome
        predicted = predicted_distributions(lane_models, [c])[c]
        worst = max([worst] + [abs(predicted.get(ol, 0.0) - joint.get(ol, 0.0)) for ol in set(predicted) | set(joint)])
        # the same circuit and outcomes with the lines in another order give the same marginals
        order = tuple(int(q) for q in rng.permutation(n))
        reordered = c.reorder_lines(order)
        other = LaneDecomposition(qubits, instrument_names=('Iz',))
        other.add(reordered, {ol[:-1] + (''.join(ol[-1][q] for q in order),): p for ol, p in joint.items()})
        for (s1, lc1, m1), (s2, lc2, m2) in zip(decomposition.marginals[c], other.marginals[reordered]):
            assert s1 == s2 and lc1 == lc2
            if isinstance(m1, dict):
                worst = max([worst] + [abs(m1.get(ol, 0.0) - m2.get(ol, 0.0)) for ol in set(m1) | set(m2)])
            else:
                worst = max(worst, np.abs(m1 - m2).max())
    assert worst < 1e-12
    assert seen >= {(k, instruments) for k in (1, 2, 3, 4) for instruments in (False, True)} | {(5, True)}
