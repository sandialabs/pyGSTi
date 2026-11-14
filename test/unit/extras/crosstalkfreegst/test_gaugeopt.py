import numpy as np
import pytest

import pygsti
from pygsti.baseobjs import Label
from pygsti.modelpacks import smq1Q_XYI, smq2Q_XYICNOT
from pygsti.models import create_crosstalk_free_model
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols import make_simultaneous_gst_design

from pygsti.extras.crosstalkfreegst import (LaneDecomposition, LaneModels, apply_product_gauge,
                                            create_general_crosstalk_free_model, error_summary,
                                            format_error_summary, gauge_optimize, member_objects)
from pygsti.extras.crosstalkfreegst.gaugeopt import GaugeObjective, _collect_items, tp_gauge

from .helpers import ERRORS, line_pspec, model


SUPPORTS = [(0,), (1,), (2,), (0, 1), (1, 2)]


def general_copy_of(m):
    """The general crosstalk-free model holding the same channels as crosstalk-free model `m`."""
    g = create_general_crosstalk_free_model(m.processor_spec)
    source = member_objects(m)
    for key, (_, member) in member_objects(g).items():
        member.set_dense(source[key][1].to_dense())
    g.from_vector(g.to_vector())
    return g


def lane_models_of(m):
    lm = LaneModels.from_model(m)
    for s in SUPPORTS:
        lm[s]
    return lm


def random_gauges(seed, scale=0.05):
    rng = np.random.default_rng(seed)
    return {q: tp_gauge(rng.normal(0, scale, 12)) for q in (0, 1, 2)}


def set_member(lane_models, key, matrix):
    """Set the member `key` of `lane_models` to the dense `matrix`, in every lane model holding it."""
    member = next(members[key][1] for _, (lane_model, _) in lane_models.items()
                  for members in [member_objects(lane_model)] if key in members)
    member.set_dense(matrix)
    x = lane_models.current_vector()
    x[lane_models.keys[key]] = member.to_vector()
    lane_models.set_vector(x)


def dense(lane_models, support, key):
    return member_objects(lane_models[support][0])[key][1].to_dense()


@pytest.fixture(scope='module')
def circuits():
    d = make_simultaneous_gst_design(
        line_pspec(3), smq1Q_XYI.create_gst_experiment_design(max_max_length=2, qubit_labels=(0,)),
        smq2Q_XYICNOT.create_gst_experiment_design(max_max_length=2, qubit_labels=(0, 1)), seed=1)
    return d.all_circuits_needing_data[::15]


def lane_probs(lm, circuits):
    ds = pygsti.data.DataSet(outcome_labels=['000'])  # dummy counts; only the lane circuits are used
    for c in circuits:
        ds.add_count_dict(c, {'000': 1})
    ds.done_adding_data()
    out = {}
    for support, (_, lcs) in LaneDecomposition((0, 1, 2), ds, circuits).lane_datasets().items():
        for c, p in lm[support][0].sim.bulk_probs(lcs).items():
            out[support, c] = np.array(list(p.values()))
    return out


def test_product_gauge_leaves_probabilities_unchanged(circuits):
    est = general_copy_of(model(3))
    lm = lane_models_of(est)
    before_lanes = lane_probs(lm, circuits)
    before_full = est.sim.bulk_probs(circuits)
    apply_product_gauge(lm, random_gauges(1))
    after_lanes = lane_probs(lm, circuits)
    after_full = lm.to_model(est).sim.bulk_probs(circuits)   # a fresh copy, so no stale caches
    assert max(np.max(np.abs(before_lanes[k] - after_lanes[k])) for k in before_lanes) < 1e-12
    assert max(abs(before_full[c][o] - after_full[c][o]) for c in circuits for o in before_full[c]) < 1e-12
    assert not np.allclose(lm.current_vector(), lane_models_of(est).current_vector())  # it did change


def test_gauge_optimization_is_gauge_invariant():
    target = lane_models_of(create_general_crosstalk_free_model(line_pspec(3)))
    a = lane_models_of(general_copy_of(model(3)))
    b = lane_models_of(general_copy_of(model(3)))
    apply_product_gauge(b, random_gauges(2))
    ra, rb = gauge_optimize(a, target), gauge_optimize(b, target)
    assert ra.fn_after == pytest.approx(rb.fn_after, rel=1e-6)
    assert np.allclose(a.current_vector(), b.current_vector(), atol=1e-6)
    assert np.array_equal(ra.x, a.current_vector())
    # The model was transformed by the gauges found: optimizing again starts where the first run ended
    # (the distance reported is that of the transformed model) and finds nothing to do.
    again = gauge_optimize(a, target)
    assert again.fn_before == pytest.approx(ra.fn_after, rel=1e-10)
    assert again.fn_after == pytest.approx(ra.fn_after, rel=1e-6)
    assert all(np.allclose(g, np.identity(4), atol=1e-4) for g in again.gauges.values())


def test_gauge_transformed_target_is_restored():
    ideal = lane_models_of(create_general_crosstalk_free_model(line_pspec(3)))
    # the target may have any parameterization: here, Lindblad error generators at zero
    target = LaneModels(line_pspec(3), lambda ps: create_crosstalk_free_model(
        ps, lindblad_error_coeffs=ERRORS, lindblad_parameterization='H+S', independent_gates=True))
    est = lane_models_of(create_general_crosstalk_free_model(line_pspec(3)))
    gauges = random_gauges(3, scale=0.1)
    apply_product_gauge(est, gauges)
    res = gauge_optimize(est, target)
    assert res.fn_before > 1e-2
    assert res.fn_after < 1e-16
    assert np.allclose(est.current_vector(), ideal.current_vector(), atol=1e-8)
    assert all(np.allclose(res.gauges[q], np.linalg.inv(gauges[q]), atol=1e-6) for q in gauges)   # g -> g^-1


def test_gauge_objective_derivatives_match_finite_differences():
    """Gates on one and two qubits, states, effects and instruments; and J^T J where the residuals vanish."""
    pspec = line_pspec(3, ('Iz',))
    target = lane_models_of(create_general_crosstalk_free_model(pspec, instruments={'Iz': 'Z'}))
    est = lane_models_of(create_general_crosstalk_free_model(pspec, instruments={'Iz': 'Z'}))
    rng = np.random.default_rng(4)
    est.set_vector(est.current_vector() + rng.normal(0, 0.02, est.num_params))
    objective = GaugeObjective(_collect_items(est, target, 1.0, 0.3, 2.0), (0, 1, 2))
    assert {it.kind for it in objective.items} == {'gate', 'prep', 'povm', 'instrument'}
    assert {len(it.qubits) for it in objective.items} == {1, 2}
    x = rng.normal(0, 0.1, 36)
    f, g, _ = objective.fn_grad_gn(x)
    assert f == pytest.approx(objective.fn(x), rel=1e-12)
    eps = 1e-6
    fd = np.array([(objective.fn(x + eps * e) - objective.fn(x - eps * e)) / (2 * eps) for e in np.identity(36)])
    assert np.allclose(2 * g, fd, rtol=1e-6, atol=1e-7 * np.abs(fd).max())

    # A gauge-transformed target is restored exactly by the inverse gauges, where the Hessian is 2 J^T J.
    gauges = random_gauges(5, scale=0.1)
    moved = lane_models_of(create_general_crosstalk_free_model(pspec, instruments={'Iz': 'Z'}))
    apply_product_gauge(moved, gauges)
    objective = GaugeObjective(_collect_items(moved, target, 1.0, 0.3, 2.0), (0, 1, 2))
    x = np.concatenate([(np.linalg.inv(gauges[q]) - np.identity(4))[1:].ravel() for q in (0, 1, 2)])
    f, g, jtj = objective.fn_grad_gn(x)
    assert f < 1e-24 and np.max(np.abs(g)) < 1e-12
    hessian = np.array([(2 * objective.fn_grad_gn(x + eps * e)[1] - 2 * objective.fn_grad_gn(x - eps * e)[1])
                        / (2 * eps) for e in np.identity(36)])
    assert np.allclose(2 * jtj.toarray(), hessian, atol=1e-6 * np.abs(hessian).max())


def test_distance_is_the_weighted_sum_of_squared_frobenius_distances():
    target = lane_models_of(create_general_crosstalk_free_model(line_pspec(3)))
    est = lane_models_of(create_general_crosstalk_free_model(line_pspec(3)))
    gate = ('gate', Label('Gcnot', (1, 2)))
    prep, povm = ('prep', Label('rho0'), 0, 0), ('povm', Label('Mdefault'), 2, 0)
    G = dense(est, (1, 2), gate)
    G[5, 7] += 0.1                     # squared distance 0.01
    set_member(est, gate, G)
    E = np.identity(4)
    E[3, 0] = 0.2                      # the state E|0> moves by 0.2 / sqrt(2) in one entry: squared distance 0.02
    set_member(est, prep, E)
    E = np.identity(4)
    E[3, 3] = 0.9                      # each of the two effects moves by 0.1 / sqrt(2): squared distance 0.01
    set_member(est, povm, E)

    def distance(**weights):
        items = _collect_items(est, target, weights.get('gate', 1.0), weights.get('spam', 1.0), 1.0)
        return GaugeObjective(items, (0, 1, 2)).fn(np.zeros(36))
    assert distance() == pytest.approx(0.01 + 0.02 + 0.01)
    assert distance(gate=3.0) == pytest.approx(3 * 0.01 + 0.02 + 0.01)
    assert distance(spam=0.5) == pytest.approx(0.01 + 0.5 * (0.02 + 0.01))
    assert gauge_optimize(est, target, gate_weight=3.0, max_iterations=0).fn_before == pytest.approx(0.06)


def test_models_that_cannot_be_transformed_are_rejected_before_anything_is_done(monkeypatch):
    from pygsti.extras.crosstalkfreegst import gaugeopt
    monkeypatch.setattr(gaugeopt, 'levenberg_marquardt', None)   # so that reaching the optimization fails
    target = LaneModels(line_pspec(3), create_general_crosstalk_free_model)

    lindblad = lane_models_of(model(3))
    x = lindblad.current_vector()
    with pytest.raises(NotImplementedError, match='cannot be set from a dense matrix'):
        gauge_optimize(lindblad, target)
    with pytest.raises(NotImplementedError, match='cannot be set from a dense matrix'):
        apply_product_gauge(lindblad, random_gauges(1))
    assert np.array_equal(lindblad.current_vector(), x)

    with pytest.raises(ValueError, match='No lane models have been built'):
        gauge_optimize(LaneModels(line_pspec(3), create_general_crosstalk_free_model), target)

    general = lane_models_of(create_general_crosstalk_free_model(line_pspec(3)))
    x = general.current_vector()
    with pytest.raises(ValueError, match=r'No gauge was given for qubits \[2\]'):
        apply_product_gauge(general, {q: g for q, g in random_gauges(1).items() if q != 2})
    assert np.array_equal(general.current_vector(), x)
    gate = ('gate', Label('Gxpi2', 0))
    assert np.array_equal(dense(general, (0,), gate), dense(target, (0,), gate))

    # a second preparation, of |1>: prepared states are read off as (error map)|0>
    pspec = QubitProcessorSpec(1, gate_names=['Gxpi2'], qubit_labels=(0,), prep_names=('rho0', 'rho1'))
    two_preps = LaneModels(pspec, create_general_crosstalk_free_model)
    two_preps[(0,)]
    for function in (lambda: gauge_optimize(two_preps, LaneModels(pspec, create_general_crosstalk_free_model)),
                     lambda: apply_product_gauge(two_preps, {0: np.identity(4)}),
                     lambda: error_summary(two_preps, two_preps)):
        with pytest.raises(NotImplementedError, match=r'is not \|0> on every qubit'):
            function()

    # two POVMs can be gauge-optimized (each is an item), but the summary has one readout row per qubit
    pspec = QubitProcessorSpec(1, gate_names=['Gxpi2'], qubit_labels=(0,), povm_names=('Mdefault', 'Mz'))
    two_povms, ideal = (LaneModels(pspec, create_general_crosstalk_free_model) for _ in range(2))
    two_povms[(0,)]
    kinds = [it.key[:2] for it in _collect_items(two_povms, ideal, 1.0, 1.0, 1.0) if it.kind == 'povm']
    assert kinds == [('povm', Label('Mdefault')), ('povm', Label('Mz'))]
    with pytest.raises(NotImplementedError, match='several state preparations or POVMs'):
        error_summary(two_povms, ideal)


@pytest.mark.parametrize('spam_weight, method', [(1.0, 'ls'), (0.01, 'L-BFGS-B')])
def test_single_qubit_case_matches_pygsti_gaugeopt(spam_weight, method):
    """
    On one qubit the product group is pyGSTi's TPGaugeGroup; compare with gaugeopt_to_target.

    Weights multiply the squared distances, as in pyGSTi's objective (`frobeniusdist`).  With weights
    other than one the comparison is with pyGSTi's 'L-BFGS-B' method: the Jacobian of its 'ls' method
    carries the weights where its residuals carry their square roots, so 'ls' stops short of the minimum.
    """
    from pygsti.models.gaugegroup import TPGaugeGroup
    from pygsti.algorithms.gaugeopt import gaugeopt_to_target

    pspec = QubitProcessorSpec(1, gate_names=['Gxpi2', 'Gypi2'], qubit_labels=(0,))
    noisy = create_crosstalk_free_model(pspec, lindblad_error_coeffs={
        'Gxpi2': {('H', 'X'): 0.03, ('H', 'Z'): 0.02, ('S', 'Y'): 0.01},
        'Gypi2': {('H', 'Y'): -0.02, ('S', 'X'): 0.02},
        'prep': {('H', 'Y'): 0.05, ('S', 'X'): 0.02}, 'povm': {('H', 'X'): 0.04, ('S', 'Y'): 0.03}},
        independent_gates=True)
    est = general_copy_of(noisy)
    ideal = create_general_crosstalk_free_model(pspec)
    lm, target = LaneModels.from_model(est), LaneModels.from_model(ideal)
    lm[(0,)], target[(0,)]
    ours = gauge_optimize(lm, target, spam_weight=spam_weight)

    explicit, explicit_target = est.to_explicit_model(), ideal.to_explicit_model()
    for m in (explicit, explicit_target):  # dense SPAM: pyGSTi cannot gauge-transform a ComposedState
        m.set_all_parameterizations('full TP')
    pyg = gaugeopt_to_target(explicit, explicit_target, gauge_group=TPGaugeGroup(explicit.state_space, 'pp'),
                             item_weights={'gates': 1.0, 'spam': spam_weight}, gates_metric='frobenius',
                             spam_metric='frobenius', method=method, tol=1e-12)
    lane = lm[(0,)][0]
    for lbl in ['Gxpi2', 'Gypi2']:
        ours_gate = lane.operation_blks['gates'][pygsti.baseobjs.Label(lbl, 0)].to_dense()
        assert np.allclose(ours_gate, pyg.operations[pygsti.baseobjs.Label(lbl, 0)].to_dense(), atol=1e-6)
    assert ours.fn_after < 0.9 * ours.fn_before

    def weighted_distance(m):
        gates = sum(np.sum((m.operations[k].to_dense() - explicit_target.operations[k].to_dense())**2)
                    for k in explicit_target.operations)
        spam = np.sum((m.preps['rho0'].to_dense() - explicit_target.preps['rho0'].to_dense())**2)
        spam += sum(np.sum((m.povms['Mdefault'][k].to_dense() - explicit_target.povms['Mdefault'][k].to_dense())**2)
                    for k in ('0', '1'))
        return gates + spam_weight * spam
    assert ours.fn_before == pytest.approx(weighted_distance(explicit), rel=1e-10)
    assert ours.fn_after == pytest.approx(weighted_distance(pyg), rel=1e-8)


def test_error_summary():
    from pygsti.circuits import Circuit
    target = lane_models_of(create_general_crosstalk_free_model(line_pspec(3)))
    rows = error_summary(target, target, diamond=False)
    assert all(abs(v) < 1e-12 for r in rows for k, v in r.items() if k.startswith(('ent', 'avg', 'prep', 'readout')))
    assert len([r for r in rows if r['kind'] == 'spam']) == 3
    # Ideal prep, noisy readout on qubit 1: P(1|0) is the lane model's P('1') for an empty circuit.
    est = lane_models_of(create_general_crosstalk_free_model(line_pspec(3)))
    lane_model = est[(1,)][0]
    readout = member_objects(lane_model)[('povm', pygsti.baseobjs.Label('Mdefault'), 1, 0)][1]
    E = np.identity(4)
    E[3, 0], E[3, 3] = 0.02, 0.95          # a non-unital readout error: P(1|0) = 0.015, P(0|1) = 0.035
    readout.set_dense(E)
    x = est.current_vector()
    x[est.keys[('povm', pygsti.baseobjs.Label('Mdefault'), 1, 0)]] = readout.to_vector()
    est.set_vector(x)
    spam1 = [r for r in error_summary(est, target, diamond=False) if r['kind'] == 'spam' and r['qubits'] == (1,)][0]
    assert spam1['readout_p1_given_0'] == pytest.approx(lane_model.probabilities(Circuit([], line_labels=(1,)))[('1',)])
    assert spam1['readout_p1_given_0'] == pytest.approx(0.015)
    assert spam1['readout_p0_given_1'] == pytest.approx(0.035)
    assert 'P(1|0)' in format_error_summary(rows)


def test_error_summary_of_known_errors():
    """Pauli errors of known probability on a one- and a two-qubit gate, and a bit flip of the prepared state."""
    target = lane_models_of(create_general_crosstalk_free_model(line_pspec(3)))
    est = lane_models_of(create_general_crosstalk_free_model(line_pspec(3)))
    anticommutes_with_x = np.diag([1.0, 1, -1, -1])
    flip = 0.99 * np.identity(4) + 0.01 * anticommutes_with_x                       # X with probability 0.01
    gate = ('gate', Label('Gxpi2', 1))
    set_member(est, gate, flip @ dense(est, (1,), gate))
    flip2 = 0.98 * np.identity(16) + 0.02 * np.kron(anticommutes_with_x, anticommutes_with_x)   # XX, 0.02
    gate2 = ('gate', Label('Gcnot', (0, 1)))
    set_member(est, gate2, flip2 @ dense(est, (0, 1), gate2))
    set_member(est, ('prep', Label('rho0'), 2, 0), 0.97 * np.identity(4) + 0.03 * anticommutes_with_x)

    rows = error_summary(est, target)
    gates = {(r['name'], r['qubits']): r for r in rows if r['kind'] == 'gate'}
    assert len(gates) == 3 * 3 + 2 * 2 and len(rows) == 13 + 3
    assert [r['kind'] for r in rows] == ['gate'] * 13 + ['spam'] * 3
    assert [len(r['qubits']) for r in rows[:13]] == [1] * 9 + [2] * 4
    one, two = gates['Gxpi2', (1,)], gates['Gcnot', (0, 1)]
    assert one['entanglement_infidelity'] == pytest.approx(0.01)
    assert one['average_gate_infidelity'] == pytest.approx(0.01 * 2 / 3)
    assert one['half_diamond_distance'] == pytest.approx(0.01, abs=1e-5)
    assert two['entanglement_infidelity'] == pytest.approx(0.02)
    assert two['average_gate_infidelity'] == pytest.approx(0.02 * 4 / 5)
    assert two['half_diamond_distance'] == pytest.approx(0.02, abs=1e-5)
    others = [r for key, r in gates.items() if key not in (('Gxpi2', (1,)), ('Gcnot', (0, 1)))]
    assert all(abs(r['entanglement_infidelity']) < 1e-12 and abs(r['half_diamond_distance']) < 1e-5 for r in others)
    spam = {r['qubits'][0]: r for r in rows if r['kind'] == 'spam'}
    assert spam[2]['prep_infidelity'] == pytest.approx(0.03) and abs(spam[0]['prep_infidelity']) < 1e-12
    assert all(abs(r['readout_p1_given_0']) < 1e-12 and abs(r['readout_p0_given_1']) < 1e-12 for r in spam.values())

    without = error_summary(est, target, diamond=False)
    assert np.isnan(without[0]['half_diamond_distance'])
    text = format_error_summary(without).splitlines()     # no diamond column when none was computed
    assert text[0].split() == ['gate', 'qubits', 'ent.', 'infid.', 'avg.', 'infid.'] and 'nan' not in ''.join(text)
    assert next(t for t in text if t.startswith('Gcnot') and '(0, 1)' in t).split()[-2:] == ['2.00e-02', '1.60e-02']
    text = format_error_summary(rows).splitlines()
    assert len(text) == 1 + 13 + 2 + 3
    assert text[0].split() == ['gate', 'qubits', 'ent.', 'infid.', 'avg.', 'infid.', '1/2', 'diamond']
    line = next(t for t in text if t.startswith('Gcnot') and '(0, 1)' in t)
    assert line.split()[-3:] == ['2.00e-02', '1.60e-02', '2.00e-02']
    assert [float(v) for v in text[-1].split()] == pytest.approx([2, 0.03, 0, 0], abs=1e-12)
