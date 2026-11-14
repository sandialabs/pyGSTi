import numpy as np
import pytest

import pygsti
from pygsti.circuits import Circuit
from pygsti.models import create_crosstalk_free_model
from pygsti.protocols import GateSetTomographyDesign, make_simultaneous_gst_design

from pygsti.extras.crosstalkfreegst import (LaneModels, add_single_qubit_instruments, create_simple_mcm_oneq_design,
                                            create_simple_mcm_twoq_design)
from pygsti.extras.crosstalkfreegst.diagnostics import (joint_goodness_of_fit, joint_loglikelihood,
                                                        predicted_distributions, saturated_loglikelihood,
                                                        total_variation_distance)

from .helpers import ERRORS, joint_logl, line_pspec

QUBITS = (0, 1, 2)


@pytest.fixture(scope='module')
def setup():
    pspec = line_pspec(3, ('Iz',))
    d = make_simultaneous_gst_design(pspec, create_simple_mcm_oneq_design(1), create_simple_mcm_twoq_design(1),
                                     seed=1)
    # Reorder lines so pyGSTi's full-model simulation is valid (see test_instruments.py).
    lists = [[c.reorder_lines(QUBITS) for c in lst] for lst in d.circuit_lists]
    circuits = GateSetTomographyDesign(pspec, lists, qubit_labels=QUBITS, nested=True).all_circuits_needing_data[::6]
    m = create_crosstalk_free_model(pspec, lindblad_error_coeffs=ERRORS, lindblad_parameterization='H+S',
                                    independent_gates=True)
    add_single_qubit_instruments(m, {'Iz': 'Z'})
    m.from_vector(np.random.default_rng(0).uniform(-0.03, 0.03, m.num_params))
    ds = pygsti.data.simulate_data(m, circuits, 200, seed=1)
    return m, circuits, ds


def test_joint_loglikelihood_matches_full_model(setup):
    m, circuits, ds = setup
    assert joint_loglikelihood(LaneModels.from_model(m), ds, circuits) == pytest.approx(
        joint_logl(m, ds, circuits), rel=1e-12)


def test_predicted_distributions_match_full_model(setup):
    m, circuits, _ = setup
    some = [c for c in circuits if 'Iz' in c.str][:5] + circuits[:5]
    # two measurements in one lane with different distributions (the first gives '0', the second is 50:50),
    # the second in the same layer as another lane's
    some.append(Circuit([[('Iz', 0), ('Gi', 1), ('Gi', 2)], [('Gxpi2', 0), ('Gi', 1), ('Gypi2', 2)],
                         [('Iz', 0), ('Gi', 1), ('Iz', 2)]], line_labels=QUBITS))
    predicted = predicted_distributions(LaneModels.from_model(m), some)
    for c in some:
        p = m.probabilities(c)
        assert set(predicted[c]) == set(p.keys())
        assert total_variation_distance(predicted[c], dict(p)) < 1e-12


def test_goodness_of_fit_statistic(setup):
    m, circuits, ds = setup
    gof = joint_goodness_of_fit(LaneModels.from_model(m), ds, circuits)
    assert gof['logl_model'] == pytest.approx(joint_logl(m, ds, circuits), rel=1e-12)
    assert gof['logl_saturated'] == pytest.approx(saturated_loglikelihood(ds, circuits), rel=1e-12)
    assert gof['two_delta_logl'] == pytest.approx(2 * (gof['logl_saturated'] - gof['logl_model']))
    assert gof['two_delta_logl'] > 0
    assert gof['num_observed_outcomes'] == sum(1 for c in circuits for n in ds[c].counts.values() if n > 0)
    # only the circuits asked for are counted
    part = joint_goodness_of_fit(LaneModels.from_model(m), ds, circuits[:7])
    assert part['logl_model'] == pytest.approx(joint_logl(m, ds, circuits[:7]), rel=1e-12)
    assert part['logl_saturated'] == pytest.approx(saturated_loglikelihood(ds, circuits[:7]), rel=1e-12)


def test_likelihoods_of_a_small_example():
    pspec = line_pspec(2)
    ideal = LaneModels(pspec, lambda ps: create_crosstalk_free_model(ps, independent_gates=True))
    c = Circuit([[('Gxpi2', 0), ('Gi', 1)], [('Gxpi2', 0), ('Gi', 1)]], line_labels=(0, 1))   # ideally always '10'
    d = Circuit([[('Gxpi2', 0), ('Gxpi2', 1)]], line_labels=(0, 1))                           # ideally uniform
    ds = pygsti.data.DataSet()
    ds.add_count_dict(c, {'10': 3, '00': 1})
    ds.add_count_dict(d, {'00': 4, '11': 4, '01': 0})
    ds.done_adding_data()
    assert saturated_loglikelihood(ds) == pytest.approx(3 * np.log(0.75) + np.log(0.25) + 8 * np.log(0.5))
    assert saturated_loglikelihood(ds, [d]) == pytest.approx(8 * np.log(0.5))
    # the impossible outcome '00' of c is given probability `min_prob`
    assert joint_loglikelihood(ideal, ds, min_prob=1e-6) == pytest.approx(np.log(1e-6) + 8 * np.log(0.25))
    assert joint_loglikelihood(ideal, ds, [d]) == pytest.approx(8 * np.log(0.25))
    gof = joint_goodness_of_fit(ideal, ds, [d])
    assert gof['num_observed_outcomes'] == 2
    assert gof['two_delta_logl'] == pytest.approx(2 * 8 * np.log(2))
    assert total_variation_distance({'a': 0.5, 'b': 0.5}, {'b': 0.25, 'c': 0.75}) == pytest.approx(0.75)


def test_outcome_labels_follow_each_circuits_line_order(setup):
    """The same circuit with its lines reordered, and its data relabeled to match, has the same likelihood."""
    m, circuits, ds = setup
    lane_models = LaneModels.from_model(m)
    some = list(dict.fromkeys([c for c in circuits if 'Iz' in c.str][:6] + circuits[:6]))
    order = (2, 0, 1)
    moved = pygsti.data.DataSet()
    for c in some:
        moved.add_count_dict(c.reorder_lines(order), {ol[:-1] + (''.join(ol[-1][q] for q in order),): n
                                                      for ol, n in ds[c].counts.items()})
    moved.done_adding_data()
    reordered = [c.reorder_lines(order) for c in some]
    assert any(len(set(ol[-1] for ol in ds[c].counts)) > 1 for c in some)   # the bits do differ between outcomes
    assert joint_loglikelihood(lane_models, moved, reordered) == pytest.approx(
        joint_loglikelihood(lane_models, ds, some), rel=1e-12)
    predicted = predicted_distributions(lane_models, iter(reordered))
    original = predicted_distributions(lane_models, some)
    for c, r in zip(some, reordered):
        assert predicted[r] == pytest.approx({ol[:-1] + (''.join(ol[-1][q] for q in order),): v
                                              for ol, v in original[c].items()})


def test_logl_statistic_moments_match_monte_carlo():
    """At the true parameters, the statistic's mean and variance over simulated datasets match the computed ones."""
    from pygsti.extras.crosstalkfreegst import FactorizedObjective, simulate_lanewise_marginals
    from pygsti.extras.crosstalkfreegst.diagnostics import logl_statistic_moments
    pspec = line_pspec(2, ('Iz',))
    truth = LaneModels(pspec, lambda ps: add_single_qubit_instruments(
        create_crosstalk_free_model(ps, lindblad_error_coeffs=ERRORS, lindblad_parameterization='H+s',
                                    independent_gates=True), {'Iz': 'Z'}))
    truth[(0,)]
    rng = np.random.default_rng(3)
    truth.set_vector(truth.current_vector() + rng.uniform(0, 0.02, truth.num_params))
    # Sparse outcome spaces: several mid-circuit measurements per circuit, few shots.
    circuits = [Circuit([('Gxpi2', 0)] * a + [('Iz', 0)] * b + [('Gypi2', 0)], line_labels=(0,))
                for a in range(3) for b in range(1, 5)]
    stats = []
    for seed in range(300):
        data = simulate_lanewise_marginals(truth, circuits, 50, seed=seed).lane_datasets()
        stats.append(FactorizedObjective(truth, data, 'logl').fn(truth.current_vector()))
    moments = logl_statistic_moments(truth, data)
    assert moments['num_outcomes'] > 3 * moments['num_lane_circuits']  # genuinely sparse
    assert moments['num_lane_circuits'] == len(circuits)
    assert moments['num_outcomes'] == sum(2**(1 + c.str.count('Iz')) for c in circuits)
    assert moments['asymptotic_mean'] == (moments['num_outcomes'] - len(circuits)) / 2
    assert moments['mean'] < 0.8 * moments['asymptotic_mean']          # the asymptotic value is well off here
    assert np.mean(stats) == pytest.approx(moments['mean'], abs=4 * np.std(stats) / np.sqrt(len(stats)))
    assert np.var(stats) == pytest.approx(moments['variance'], rel=0.35)


def test_logl_statistic_moments_at_many_counts():
    """With many counts per outcome the statistic is chi^2 / 2: mean and variance (outcomes - circuits) / 2."""
    from pygsti.extras.crosstalkfreegst import simulate_lanewise_marginals
    from pygsti.extras.crosstalkfreegst.diagnostics import _binomial_deviance_moments, logl_statistic_moments
    truth = LaneModels(line_pspec(2), lambda ps: create_crosstalk_free_model(ps, independent_gates=True))
    circuits = [Circuit([[('Gxpi2', 0), ('Gxpi2', 1)]], line_labels=(0, 1)),     # two lanes, each 50:50
                Circuit([[('Gcnot', 0, 1)], [('Gxpi2', 0), ('Gypi2', 1)]], line_labels=(0, 1)),   # one lane, uniform
                Circuit([[('Gxpi2', 0), ('Gi', 1)]] * 2, line_labels=(0, 1))]    # two lanes, each one outcome
    data = simulate_lanewise_marginals(truth, circuits, 100000, seed=0).lane_datasets()
    moments = logl_statistic_moments(truth, data)
    # every possible outcome is counted, but the two deterministic lane circuits add nothing to the moments
    assert (moments['num_outcomes'], moments['num_lane_circuits'], moments['asymptotic_mean']) == (12, 5, 3.5)
    from pygsti.extras.crosstalkfreegst import FactorizedObjective
    assert moments['num_outcomes'] == FactorizedObjective(truth, data, 'logl').num_elements
    assert moments['mean'] == pytest.approx(2.5, rel=1e-3)
    assert moments['variance'] == pytest.approx(2.5, rel=1e-3)
    # one outcome with 3 expected counts in 5 shots, summed by hand over n = 0..5
    from math import comb, log
    terms = [3 - n + (n * log(n / 3) if n else 0) for n in range(6)]
    weights = [comb(5, n) * 0.6**n * 0.4**(5 - n) for n in range(6)]
    mean = sum(w * t for w, t in zip(weights, terms))
    m, v = _binomial_deviance_moments(np.array([3.0, 0.0]), 5.0)
    assert m[0] == pytest.approx(mean) and v[0] == pytest.approx(sum(w * (t - mean)**2 for w, t in zip(weights, terms)))
    assert m[1] == 0 and v[1] == 0
    # each lane circuit's own number of shots is used
    a, b = Circuit([('Gxpi2', 0)], line_labels=(0,)), Circuit([('Gypi2', 0)], line_labels=(0,))
    ds = pygsti.data.DataSet()
    ds.add_count_dict(a, {'0': 2, '1': 3})
    ds.add_count_dict(b, {'0': 30, '1': 20})
    ds.done_adding_data()
    few, many = (_binomial_deviance_moments(np.array([N / 2, N / 2]), N)[0].sum() for N in (5.0, 50.0))
    assert abs(few - many) > 0.01
    assert logl_statistic_moments(truth, {(0,): (ds, [a, b])})['mean'] == pytest.approx(few + many)
