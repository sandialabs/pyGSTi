import numpy as np
import pytest

import pygsti
from pygsti.baseobjs import Label
from pygsti.circuits import Circuit
from pygsti.modelpacks import smq1Q_XYI, smq2Q_XYICNOT
from pygsti.models import create_cloud_crosstalk_model, create_crosstalk_free_model
from pygsti.objectivefns import ObjectiveFunctionBuilder
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols import make_simultaneous_gst_design
from pygsti.tools.exceptions import ProbabilityClippingWarning

from pygsti.extras.crosstalkfreegst import (FactorizedObjective, LaneDecomposition, LaneModels, LanePool,
                                            check_factorizable, create_general_crosstalk_free_model,
                                            lane_outcome_distributions, member_objects, member_parameter_indices,
                                            restrict_model, run_crosstalk_free_gst, simulate_lanewise_data,
                                            simulate_lanewise_marginals, split_circuit_into_lanes)

from pygsti.extras.crosstalkfreegst import fit as xf_fit
from pygsti.extras.crosstalkfreegst.parallel import build_lane_objective, lane_pieces

from .helpers import ERRORS, joint_logl, line_pspec, model


@pytest.fixture(scope='module')
def design3():
    return make_simultaneous_gst_design(
        line_pspec(3), smq1Q_XYI.create_gst_experiment_design(max_max_length=1, qubit_labels=(0,)),
        smq2Q_XYICNOT.create_gst_experiment_design(max_max_length=1, qubit_labels=(0, 1)), seed=1)


def test_split_uses_canonical_lane_order():
    c = Circuit([[('Gxpi2', 2), ('Gcnot', 0, 1)], [('Gii', 0, 1), ('Gi', 2)]], line_labels=(2, 0, 1))
    lanes = split_circuit_into_lanes(c, (0, 1, 2))
    assert [s for s, _ in lanes] == [(0, 1), (2,)]
    assert lanes[0][1] == Circuit([('Gcnot', 0, 1), ('Gii', 0, 1)], line_labels=(0, 1))
    assert lanes[1][1] == Circuit([('Gxpi2', 2), ('Gi', 2)], line_labels=(2,))


def test_split_merges_lanes_joined_by_a_later_gate():
    # (0, 1) and (2, 3) are lanes of their own until the gate on (1, 3) joins them into one of four qubits
    c = Circuit([[('Gcnot', 0, 1), ('Gcnot', 2, 3)], [('Gcnot', 1, 3), ('Gxpi2', 4)], [('Gxpi2', 2)]],
                line_labels=(0, 1, 2, 3, 4))
    wide = Circuit([[('Gcnot', 0, 1), ('Gcnot', 2, 3)], [('Gcnot', 1, 3)], [('Gxpi2', 2)]], line_labels=(0, 1, 2, 3))
    alone = Circuit([('Gxpi2', 4)], line_labels=(4,))
    assert split_circuit_into_lanes(c, (0, 1, 2, 3, 4)) == [((0, 1, 2, 3), wide), ((4,), alone)]


def test_split_drops_layers_idle_on_a_lane_and_keeps_untouched_lines():
    c = Circuit([[('Gxpi2', 0)], [('Gxpi2', 0)]], line_labels=(0, 1))
    lanes = split_circuit_into_lanes(c, (0, 1))
    assert lanes[1] == ((1,), Circuit([], line_labels=(1,)))


def test_split_rejects_empty_layers():
    with pytest.raises(ValueError, match='empty layer'):
        split_circuit_into_lanes(Circuit([Label(()), ('Gxpi2', 0)], line_labels=(0, 1)), (0, 1))


def test_marginals_follow_each_circuits_line_order():
    c = Circuit([('Gxpi2', 2)], line_labels=(1, 2, 0))
    ds = pygsti.data.DataSet(outcome_labels=['010', '000'])
    ds.add_count_dict(c, {'010': 7, '000': 3})  # bits are qubits (1, 2, 0)
    ds.done_adding_data()
    marg = {s: m for s, _, m in LaneDecomposition((0, 1, 2), ds, [c]).marginals[c]}
    assert np.array_equal(marg[(2,)], [3, 7])
    assert np.array_equal(marg[(1,)], [10, 0])


def test_adding_a_circuit_twice_is_rejected():
    c = Circuit([('Gxpi2', 0)], line_labels=(0, 1))
    dec = LaneDecomposition((0, 1))
    dec.add(c, {'00': 6, '10': 4})
    with pytest.raises(ValueError, match='already been added'):
        dec.add(c, {'00': 1})
    with pytest.raises(ValueError, match='already been added'):
        dec.add_lane_counts(c, {(0,): [1, 0], (1,): [1, 0]})
    assert {s: m.tolist() for s, _, m in dec.marginals[c]} == {(0,): [6, 4], (1,): [10, 0]}


@pytest.mark.parametrize('independent_gates', [True, False])
def test_restricted_model_reproduces_marginals(independent_gates):
    m = model(3, independent_gates)
    c = Circuit([[('Gxpi2', 0), ('Gcnot', 1, 2)], [('Gypi2', 0), ('Gii', 1, 2)]], line_labels=(0, 1, 2))
    full = m.probabilities(c)
    lane = restrict_model(m, (1, 2))
    lane_keys, full_keys = member_parameter_indices(lane), member_parameter_indices(m)
    for key, idx in lane_keys.items():
        assert np.array_equal(lane.to_vector()[idx], m.to_vector()[full_keys[key]])
    p = lane.probabilities(Circuit([('Gcnot', 1, 2), ('Gii', 1, 2)], line_labels=(1, 2)))
    for bits, prob in p.items():
        marginal = sum(v for k, v in full.items() if k[0][1:] == bits[0])
        assert prob == pytest.approx(marginal, abs=1e-12)


@pytest.mark.parametrize('independent_gates', [True, False])
def test_factory_lane_models_match_restricted_ones(independent_gates):
    """Lane models built directly from sub-processor-specs tie parameters as restriction does."""
    m = model(4, independent_gates)
    factory = LaneModels(line_pspec(4), lambda ps: create_crosstalk_free_model(
        ps, lindblad_error_coeffs=ERRORS, lindblad_parameterization='H+S', independent_gates=independent_gates))
    restricted = LaneModels.from_model(m)
    for support in [(0,), (1, 2), (2,), (2, 3), (3,), (0, 1), (1,)]:
        factory[support], restricted[support]
    assert factory.keys.keys() == restricted.keys.keys()
    x = factory.to_model(m, restricted.current_vector()).to_vector()  # restricted values via keys
    assert np.allclose(x, m.to_vector())
    factory.set_vector(np.concatenate([restricted.parameters()[k] for k in factory.keys]))
    c = Circuit([('Gcnot', 1, 2), ('Gxpi2', 1), ('Gii', 1, 2)], line_labels=(1, 2))
    p1, p2 = factory[(1, 2)][0].probabilities(c), restricted[(1, 2)][0].probabilities(c)
    assert all(p1[ol] == pytest.approx(p2[ol], abs=1e-12) for ol in p1)


def test_factorized_likelihood_equals_joint_likelihood(design3):
    m = model(3)
    circuits = design3.all_circuits_needing_data
    assert any(c.line_labels != (0, 1, 2) for c in circuits)  # exercise permuted line orders
    ds = pygsti.data.simulate_data(m, circuits, 100, seed=1)
    lane_models = LaneModels.from_model(m)
    total = 0.0
    for support, (lds, lcs) in LaneDecomposition((0, 1, 2), ds, circuits).lane_datasets().items():
        total += joint_logl(lane_models[support][0], lds, lcs)
    assert total == pytest.approx(joint_logl(m, ds, circuits), rel=1e-12)


def test_objective_differences_match_full_model(design3):
    """The factorized objective differs from -logL of the full model by a constant (saturated terms)."""
    m = model(3)
    circuits = design3.all_circuits_needing_data
    ds = pygsti.data.simulate_data(m, circuits, 100, seed=1)
    lane_models = LaneModels.from_model(m)
    obj = FactorizedObjective(lane_models, LaneDecomposition((0, 1, 2), ds, circuits).lane_datasets(), 'logl')
    x1 = lane_models.current_vector()
    x2 = x1 + 0.01
    m2 = lane_models.to_model(m, x2)
    # pyGSTi's 'logl' objective is Delta(logL), whose differences are differences of -logL.
    # (Not compared with the full model's 'logl' objective: that regularizes probabilities below
    # ~1e-4, which joint probabilities reach far more often than lane marginals do.)
    expected = joint_logl(m, ds, circuits) - joint_logl(m2, ds, circuits)
    assert obj.fn(x2) - obj.fn(x1) == pytest.approx(expected, rel=1e-4)  # lane objectives regularize too


def test_gradient_matches_finite_differences(design3):
    m = model(3)
    circuits = design3.circuit_lists[0][::5]
    ds = pygsti.data.simulate_data(m, circuits, 100, seed=1)
    lane_models = LaneModels.from_model(m)
    obj = FactorizedObjective(lane_models, LaneDecomposition((0, 1, 2), ds, circuits).lane_datasets(), 'logl')
    x = lane_models.current_vector()
    _, g, _ = obj.fn_grad_gn(x)
    eps = 1e-6
    fd = np.array([(obj.fn(x + eps * e) - obj.fn(x - eps * e)) / (2 * eps) for e in np.eye(len(x))])
    assert np.allclose(2 * g, fd, rtol=1e-5, atol=1e-5 * np.abs(fd).max())


def test_lanewise_simulation_matches_model(design3):
    m = model(3)
    c = design3.circuit_lists[0][-1]
    ds = simulate_lanewise_data(m, [c], 200000, seed=3)
    p = m.probabilities(c)
    for ol, n in ds[c].counts.items():
        assert n / 200000 == pytest.approx(p[ol], abs=5e-3)


def test_simulating_an_unphysical_model_warns():
    m = create_crosstalk_free_model(line_pspec(2), lindblad_error_coeffs={'Gi': {('S', 'X'): -0.01}},
                                    lindblad_parameterization='H+s', independent_gates=True)   # negative bit-flip rate
    c = Circuit([[Label('Gi', 0), Label('Gi', 1)]], line_labels=(0, 1))
    assert min(m.probabilities(c).values()) < -1e-3
    with pytest.warns(ProbabilityClippingWarning):
        ds = simulate_lanewise_data(m, [c], 100, seed=0)
    assert dict(ds[c].counts) == {('00',): 100}
    with pytest.warns(ProbabilityClippingWarning):
        (first, second) = lane_outcome_distributions(LaneModels.from_model(m), [split_circuit_into_lanes(c, (0, 1))])
    np.testing.assert_array_equal(first, [1, 0])


def test_lanewise_marginals_match_model(design3):
    m = model(3)
    c = design3.circuit_lists[0][-1]
    dec = simulate_lanewise_marginals(m, [c], 200000, seed=3)
    lane_models = LaneModels.from_model(m)
    for support, lane_circuit, counts in dec.marginals[c]:
        p = lane_models[support][0].probabilities(lane_circuit)
        k = len(support)
        assert np.allclose(counts / 200000, [p[(format(i, f'0{k}b'),)] for i in range(2**k)], atol=5e-3)


def test_lane_outcome_distributions_multiply_to_model_probabilities(design3):
    m = model(3)
    circuits = design3.circuit_lists[0][-10:]
    splits = [split_circuit_into_lanes(c, (0, 1, 2)) for c in circuits]
    distributions = iter(lane_outcome_distributions(LaneModels.from_model(m), splits))
    for c, lanes in zip(circuits, splits):
        lane_probs = [(support, next(distributions)) for support, _ in lanes]
        assert all(len(dist) == 2**len(support) and dist.sum() == pytest.approx(1) for support, dist in lane_probs)
        position = {q: i for i, q in enumerate(c.line_labels)}
        for (bits,), p in m.probabilities(c).items():
            product = 1.0
            for support, dist in lane_probs:
                product *= dist[int(''.join(bits[position[q]] for q in support), 2)]  # first lane qubit = MSB
            assert product == pytest.approx(p, abs=1e-10)
    assert next(distributions, None) is None


@pytest.mark.parametrize('parameterization', ['CPTPLND', 'GLND'])
def test_lane_models_from_a_factory_match_the_many_qubit_model(parameterization):
    # every type of elementary error generator: Hamiltonian, stochastic, correlation and active
    coeffs = {'Gxpi2': {('H', 'X'): 0.02, ('S', 'X'): 0.01, ('S', 'Y'): 0.01, ('C', 'X', 'Y'): 0.004,
                        ('A', 'X', 'Y'): 0.003},
              'Gcnot': {('H', 'ZZ'): 0.02, ('S', 'XI'): 0.01, ('S', 'IX'): 0.01, ('C', 'IX', 'XI'): 0.004},
              'prep': {('S', 'X'): 0.01}, 'povm': {('H', 'Y'): 0.02}}

    def make(pspec):
        return create_crosstalk_free_model(pspec, lindblad_error_coeffs=coeffs, independent_gates=True,
                                           lindblad_parameterization=parameterization)
    lanes = LaneModels(line_pspec(3), make)   # the 3-qubit model is not needed to build these
    for support in [(0,), (1,), (2,), (0, 1), (1, 2)]:
        lanes[support]
    full = make(line_pspec(3))
    assert lanes.num_params == full.num_params

    c = Circuit([[Label('Gxpi2', 0), Label('Gxpi2', 1), Label('Gxpi2', 2)], [Label('Gcnot', (0, 1)), Label('Gxpi2', 2)],
                 [Label('Gxpi2', 0), Label('Gcnot', (1, 2))]], line_labels=(0, 1, 2))
    (dist,) = lane_outcome_distributions(lanes, [split_circuit_into_lanes(c, (0, 1, 2))])
    for (bits,), p in full.probabilities(c).items():
        assert dist[int(bits, 2)] == pytest.approx(p, abs=1e-12)

    rates = lanes.errorgen_rates()
    expected = {((name, *qubits), generator): rate
                for name, placements in [('Gxpi2', [(0,), (1,), (2,)]), ('Gcnot', [(0, 1), (1, 2)]),
                                         ('prep', [(0,), (1,), (2,)]), ('povm', [(0,), (1,), (2,)])]
                for qubits in placements for generator, rate in coeffs[name].items()}
    assert set(expected) <= set(rates)   # pyGSTi also lists the active partner of a correlation generator, at zero
    assert all(rate == pytest.approx(expected.get(key, 0.0), abs=1e-9) for key, rate in rates.items())


def test_errorgen_rates_of_other_models():
    def rates(m):
        lanes = LaneModels.from_model(m)
        lanes[(0, 1)]
        return lanes.errorgen_rates()

    shared = rates(model(2, independent_gates=False))   # one copy of each gate: no qubits in its key
    assert shared[('Gxpi2',), ('H', 'X')] != 0 and shared[('Gcnot',), ('S', 'XX')] != 0
    assert ('prep', 1) in {operation for operation, _ in shared}
    with pytest.raises(ValueError, match='no error-generator rates'):
        rates(create_general_crosstalk_free_model(line_pspec(2)))


def test_lane_models_built_later_agree_with_earlier_ones():
    source = model(3)
    lanes = LaneModels.from_model(source)
    one, _ = lanes[(0,)]
    source.from_vector(source.to_vector() + 0.01)   # the source model changes before the next lane is built
    two, indices = lanes[(0, 1)]
    np.testing.assert_array_equal(two.to_vector(), lanes.current_vector()[indices])
    key = ('gate', Label('Gxpi2', 0))
    np.testing.assert_array_equal(two.to_vector()[member_parameter_indices(two)[key]],
                                  one.to_vector()[member_parameter_indices(one)[key]])
    with pytest.raises(ValueError, match='parameters'):
        lanes.set_vector(np.zeros(3))


def test_general_model_gives_every_declared_prep_and_povm_its_errors():
    pspec = QubitProcessorSpec(2, ['Gxpi2', 'Gcnot'], geometry='line', prep_names=('rho0', 'rho1'),
                               povm_names=('Mz',))
    general = create_general_crosstalk_free_model(pspec)
    check_factorizable(general)
    assert set(map(str, general.prep_blks['layers'])) == {'rho0', 'rho1'}
    assert set(map(str, general.povm_blks['layers'])) == {'Mz'}
    keys = member_parameter_indices(general)   # also checks that every parameter belongs to one member
    for kind, name in [('prep', 'rho0'), ('prep', 'rho1'), ('povm', 'Mz')]:
        for q in (0, 1):
            assert len(keys[kind, Label(name), q, 0]) == 12
    assert general.num_params == 2 * 12 + 2 * 240 + 3 * 2 * 12   # Gxpi2s, Gcnot both ways, SPAM errors
    plain = create_crosstalk_free_model(pspec)   # the general model is ideal to start with
    c = Circuit([Label('rho1'), Label('Gxpi2', 0), Label('Gcnot', (0, 1)), Label('Mz')], line_labels=(0, 1))
    p, q = general.probabilities(c), plain.probabilities(c)
    assert all(p[ol] == pytest.approx(q[ol], abs=1e-12) for ol in q)


def test_non_factorizable_models_are_rejected():
    cloud = create_cloud_crosstalk_model(line_pspec(3), lindblad_error_coeffs={'Gxpi2': {('H', 'X'): 0.01}})
    with pytest.raises(ValueError):
        check_factorizable(cloud)
    nonlocal_spam = create_crosstalk_free_model(line_pspec(2), ideal_spam_type='full pure')
    with pytest.raises(NotImplementedError):
        check_factorizable(nonlocal_spam)


def test_fit_reaches_at_least_the_true_likelihood(design3):
    # Linear stochastic rates ('H+s'): the sqrt-rate 'H+S' parameterization has a saddle at zero
    # rate for every stochastic term, which can stall any Gauss-Newton optimizer.
    start = create_crosstalk_free_model(line_pspec(3), lindblad_error_coeffs=ERRORS, lindblad_parameterization='H+s')
    truth = start.copy()
    rng = np.random.default_rng(0)
    stochastic = np.array(['stochastic' in lbl[1] for lbl in truth.parameter_labels])
    truth.from_vector(np.where(stochastic, rng.uniform(0, 2e-3, truth.num_params),
                               rng.uniform(-0.02, 0.02, truth.num_params)))
    circuits = design3.all_circuits_needing_data
    ds = simulate_lanewise_data(truth, circuits, 1000, seed=5)
    res = run_crosstalk_free_gst(ds, start, design3.circuit_lists, verbosity=0)
    assert joint_logl(res.model, ds, circuits) >= joint_logl(truth, ds, circuits)


def test_fit_of_the_general_model_finds_the_maximum_likelihood(design3):
    truth = model(3)
    start = create_general_crosstalk_free_model(line_pspec(3))
    circuits = design3.all_circuits_needing_data
    ds = simulate_lanewise_data(truth, circuits, 1000, seed=5)
    res = run_crosstalk_free_gst(ds, start, design3.circuit_lists, verbosity=0)
    assert xf_fit._resolve_damping('auto', res.lane_models) == 'marquardt'
    assert all(stage.lm.converged for stage in res.iterations)

    # The truth lies inside the general model, so by Wilks' theorem twice the log-likelihood the fit
    # gains over the truth is chi^2 distributed, with one degree of freedom per parameter that is
    # neither gauge (12 per qubit) nor redundant in the SPAM (9 + 8 per qubit).  The likelihoods are
    # those of the joint outcomes, computed from the 3-qubit models.
    k = start.num_params - 3 * 12 - 3 * (9 + 8)
    gain = 2 * (joint_logl(res.model, ds, circuits) - joint_logl(truth, ds, circuits))
    assert abs(gain - k) < 5 * np.sqrt(2 * k)

    # Those k parameters are the ones the data determine: the rank of the Gauss-Newton matrix.
    objective = FactorizedObjective(res.lane_models, res.decomposition.lane_datasets(circuits), 'logl')
    f, g, jtj = objective.fn_grad_gn(res.x)
    assert f == pytest.approx(res.final_fn, rel=1e-12)
    assert np.linalg.matrix_rank(jtj.toarray()) == k
    # The sparse solver, used above 3000 parameters, agrees with the dense one on this singular
    # matrix with the small ridge of the convergence test, and with the damping of a step.
    scale = jtj.diagonal().max()
    for damping in (1e-10 * scale, 1e-3 * scale):
        dense = xf_fit._solve(jtj, np.full(len(g), damping), g)
        sparse = xf_fit._solve(jtj, np.full(len(g), damping), g, dense_threshold=0)
        assert g @ sparse == pytest.approx(g @ dense, rel=1e-6)


def test_fit_leaves_the_lane_models_at_the_estimate(design3, monkeypatch):
    lane_models = LaneModels.from_model(model(3))
    ds = simulate_lanewise_data(model(3, seed=1), design3.all_circuits_needing_data, 100, seed=5)

    def lane_vectors_match(x):
        return all(np.allclose(lane_model.to_vector(), x[gidx], atol=1e-14)
                   for _, (lane_model, gidx) in lane_models.items())

    # No iterations: the objectives are evaluated at the starting point only.
    res = run_crosstalk_free_gst(ds, lane_models, design3.circuit_lists, max_iterations=0, verbosity=0)
    start = lane_models.initial_vector()
    assert np.array_equal(res.x, start) and [stage.lm.num_iterations for stage in res.iterations] == [0, 0]
    assert not res.iterations[-1].lm.converged
    objective = FactorizedObjective(lane_models, res.decomposition.lane_datasets(), 'logl')
    assert res.final_fn == pytest.approx(objective.fn(start), rel=1e-12)

    # Evaluating an objective moves its lane models but not the LaneModels' vector ...
    objective.fn(start + 0.01)
    assert np.array_equal(lane_models.current_vector(), start) and not lane_vectors_match(start)
    # ... and a fit leaves both at its estimate,
    res = run_crosstalk_free_gst(ds, lane_models, design3.circuit_lists, max_iterations=2, verbosity=0)
    assert not np.allclose(res.x, start)
    assert np.array_equal(lane_models.current_vector(), res.x) and lane_vectors_match(res.x)

    # or, when interrupted, where the last completed stage left them.
    def interrupted(objective, x0, **kwargs):
        objective.fn(x0 + 0.01)
        raise KeyboardInterrupt
    monkeypatch.setattr(xf_fit, 'levenberg_marquardt', interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_crosstalk_free_gst(ds, lane_models, design3.circuit_lists, verbosity=0)
    assert np.array_equal(lane_models.current_vector(), res.x) and lane_vectors_match(res.x)


def test_fit_rejects_what_it_cannot_handle():
    m = model(2)
    lane_models = LaneModels.from_model(m)
    circuits = [Circuit([('Gxpi2', 0), ('Gi', 1)], line_labels=(0, 1)),
                Circuit([('Gypi2', 0), ('Gxpi2', 1)], line_labels=(0, 1))]
    ds = pygsti.data.DataSet(outcome_labels=['00', '01', '10', '11'])
    ds.add_count_dict(circuits[0], {'00': 50, '10': 50})
    ds.add_count_dict(circuits[1], {'00': 0})
    ds.done_adding_data()
    # A lane circuit with no counts at all (pyGSTi's chi^2 is NaN for one).
    for objective in ('chi2', 'logl'):
        with pytest.raises(ValueError, match='has no counts'):
            run_crosstalk_free_gst(ds, m, [circuits], iteration_objective=objective, final_objective=None,
                                   verbosity=0)
    with pytest.raises(ValueError, match='circuit_lists is empty'):
        run_crosstalk_free_gst(ds, m, [], verbosity=0)
    # Penalty terms, which would be added once per lane.
    lane_data = LaneDecomposition((0, 1), ds, circuits[:1]).lane_datasets()
    with_penalty = ObjectiveFunctionBuilder.create_from('logl')
    with_penalty.penalties = {'cptp_penalty_factor': 1.0}
    with pytest.raises(NotImplementedError, match='cptp_penalty_factor'):
        FactorizedObjective(lane_models, lane_data, with_penalty)
    callable_penalty = ObjectiveFunctionBuilder.create_from(('logl', lambda mdl: 0.0))
    with pytest.raises(NotImplementedError, match='callable_penalty'):
        FactorizedObjective(lane_models, lane_data, callable_penalty)
    FactorizedObjective(lane_models, lane_data, ObjectiveFunctionBuilder.create_from('logl'))  # zero penalties


def test_lane_regularization_does_not_depend_on_chunking():
    m = model(2)
    circuits = [Circuit([('Gxpi2', 0), ('Gi', 1)], line_labels=(0, 1)),
                Circuit([('Gypi2', 0), ('Gi', 1)], line_labels=(0, 1))]
    ds = pygsti.data.DataSet(outcome_labels=['00', '01', '10', '11'])
    ds.add_count_dict(circuits[0], {'00': 10**6})
    ds.add_count_dict(circuits[1], {'00': 10})
    ds.done_adding_data()
    dataset, lane_circuits = LaneDecomposition((0, 1), ds, circuits).lane_datasets()[(0,)]
    lane_model = LaneModels.from_model(m)[(0,)][0]
    whole = build_lane_objective(lane_model, dataset, lane_circuits, 'logl')
    assert whole.raw_objfn.min_p == pytest.approx(0.5e-6)
    for chunk in ([lane_circuits[0]], [lane_circuits[1]]):
        assert build_lane_objective(lane_model, dataset, chunk, 'logl', 10**6).raw_objfn.min_p == whole.raw_objfn.min_p


def test_pooled_chunks_use_the_lanes_regularization():
    """A lane's circuits split between two workers, one holding a circuit whose term depends on the clip."""
    m = model(2)
    m.from_vector(0.01 * m.to_vector())   # nearly ideal, so that the outcome observed below is nearly impossible
    circuits = [Circuit([('Gxpi2', 0), ('Gi', 1)], line_labels=(0, 1)),
                Circuit([[('Gxpi2', 0), ('Gi', 1)]] * 2, line_labels=(0, 1))]
    ds = pygsti.data.DataSet(outcome_labels=['00', '01', '10', '11'])
    ds.add_count_dict(circuits[0], {'00': 10**6})
    ds.add_count_dict(circuits[1], {'00': 10})
    ds.done_adding_data()
    lane_models = LaneModels.from_model(m)
    lane_data = LaneDecomposition((0, 1), ds, circuits).lane_datasets()
    serial = FactorizedObjective(lane_models, lane_data, 'logl')
    x = lane_models.current_vector()
    # the few-count circuit's term with the clip of its own counts differs from that with the lane's clip
    (lane_model, gidx), (dataset, lane_circuits) = lane_models[(0,)], lane_data[(0,)]
    own, lanes = (lane_pieces(build_lane_objective(lane_model, dataset, lane_circuits[1:], 'logl', *total), x[gidx],
                              False) for total in ((), (10**6,)))
    assert abs(own - lanes) > 0.1 * lanes
    with LanePool(2) as pool:
        parallel = FactorizedObjective(lane_models, lane_data, 'logl', pool)
        assert parallel.term_supports == [(0,), (0,), (1,), (1,)]
        assert parallel.fn(x) == pytest.approx(serial.fn(x), rel=1e-12)


def test_parallel_objective_matches_serial(design3):
    m = model(3)
    circuits = design3.circuit_lists[0]
    ds = pygsti.data.simulate_data(m, circuits, 100, seed=1)
    lane_models = LaneModels.from_model(m)
    lane_data = LaneDecomposition((0, 1, 2), ds, circuits).lane_datasets()
    serial = FactorizedObjective(lane_models, lane_data, 'logl')
    x = lane_models.current_vector() + 0.003
    with LanePool(2) as pool:
        parallel = FactorizedObjective(lane_models, lane_data, 'logl', pool)
        assert parallel.num_elements == serial.num_elements
        assert parallel.fn(x) == pytest.approx(serial.fn(x), rel=1e-12)
        assert parallel.lane_fns(x) == pytest.approx(serial.lane_fns(x), rel=1e-12)
        (fs, gs, Hs), (fp, gp, Hp) = serial.fn_grad_gn(x), parallel.fn_grad_gn(x)
        assert fp == pytest.approx(fs, rel=1e-12)
        # Heavy lanes are split into circuit chunks in parallel, so sums differ only by rounding.
        assert np.allclose(gp, gs, rtol=1e-10, atol=1e-12 * np.abs(gs).max())
        Hp, Hs = Hp.toarray(), Hs.toarray()
        assert np.allclose(Hp, Hs, rtol=1e-10, atol=1e-12 * np.abs(Hs).max())


def test_parallel_fit_matches_serial(design3):
    start = create_crosstalk_free_model(line_pspec(3), lindblad_error_coeffs=ERRORS, lindblad_parameterization='H+s')
    ds = simulate_lanewise_data(model(3), design3.all_circuits_needing_data, 1000, seed=5)
    serial = run_crosstalk_free_gst(ds, start, design3.circuit_lists, verbosity=0)
    parallel = run_crosstalk_free_gst(ds, start, design3.circuit_lists, processes=2, verbosity=0)
    assert parallel.final_fn == pytest.approx(serial.final_fn, rel=1e-8)
    assert np.allclose(parallel.x, serial.x, atol=1e-8)


def test_worker_errors_are_raised_and_leave_the_pool_usable(design3):
    m = model(3)
    circuits = design3.circuit_lists[0]
    ds = pygsti.data.simulate_data(m, circuits, 100, seed=1)
    lane_models = LaneModels.from_model(m)
    lane_data = LaneDecomposition((0, 1, 2), ds, circuits).lane_datasets()
    serial = FactorizedObjective(lane_models, lane_data, 'logl')
    x = lane_models.current_vector() + 0.003
    lane_model, gidx = lane_models[(0,)]
    good = (lane_model, gidx) + lane_data[(0,)] + (100.0,)
    with LanePool(2) as pool:
        with pytest.raises(RuntimeError, match='No lane terms are loaded'):
            pool.evaluate(x, False)
        # One worker fails to build its term while the other succeeds: nothing is left loaded.
        with pytest.raises(RuntimeError, match='Error in a crosstalk-free GST worker'):
            pool.load([good, (None, np.arange(1), None, [], 1.0)], 'logl')
        with pytest.raises(RuntimeError, match='No lane terms are loaded'):
            pool.evaluate(x, False)
        # The other worker's reply to the failed load is not taken for the answer to a later call.
        first = FactorizedObjective(lane_models, lane_data, 'logl', pool)
        assert first.fn(x) == pytest.approx(serial.fn(x), rel=1e-12)
        # An error while evaluating (a vector that is too short) leaves the terms loaded.
        with pytest.raises(RuntimeError, match='Error in a crosstalk-free GST worker'):
            first.fn(x[:5])
        assert first.fn(x) == pytest.approx(serial.fn(x), rel=1e-12)
        # A second objective on the same pool replaces the first one's terms.
        second = FactorizedObjective(lane_models, {(0,): lane_data[(0,)]}, 'logl', pool)
        with pytest.raises(RuntimeError, match='no longer loaded'):
            first.fn(x)
        assert second.fn(x) == pytest.approx(serial.lane_fns(x)[(0,)], rel=1e-12)
        # A worker that dies closes the pool.
        pool._procs[0].kill()
        with pytest.raises(RuntimeError, match='died unexpectedly'):
            second.fn(x)
        with pytest.raises(RuntimeError, match='closed'):
            second.fn(x)
        with pytest.raises(RuntimeError, match='closed'):
            FactorizedObjective(lane_models, lane_data, 'logl', pool)


def test_worker_start_failure_is_raised(monkeypatch):
    import pygsti.extras.crosstalkfreegst.parallel as par
    monkeypatch.setattr(par._sys, 'executable', 'false')   # every worker exits at once
    with pytest.raises(RuntimeError, match='failed to start'):
        LanePool(2, timeout=30)


def test_non_adjacent_lanes_and_reversed_gate_orientation():
    """Lane (0, 2) with a CNOT written as Gcnot:2:0 reproduces the full model's marginals."""
    from pygsti.processors import QubitProcessorSpec
    qubits = (0, 1, 2)
    one = [(q,) for q in qubits]
    pspec = QubitProcessorSpec(3, gate_names=['Gxpi2', 'Gypi2', 'Gcnot'],
                               availability={'Gxpi2': one, 'Gypi2': one, 'Gcnot': [(2, 0)]}, qubit_labels=qubits)
    errors = {'Gxpi2': {('H', 'X'): 0, ('S', 'Z'): 0}, 'Gypi2': {('H', 'Y'): 0},
              'Gcnot': {('H', 'ZX'): 0, ('H', 'XI'): 0, ('S', 'ZZ'): 0},
              'prep': {('H', 'X'): 0}, 'povm': {('S', 'X'): 0}}
    m = create_crosstalk_free_model(pspec, lindblad_error_coeffs=errors, lindblad_parameterization='H+S',
                                    independent_gates=True)
    m.from_vector(np.random.default_rng(4).uniform(-0.05, 0.05, m.num_params))
    c = Circuit([[('Gxpi2', 0), ('Gypi2', 1)], [('Gcnot', 2, 0), ('Gxpi2', 1)], [('Gypi2', 2)]], line_labels=qubits)
    lanes = dict(split_circuit_into_lanes(c, qubits))
    assert set(lanes) == {(0, 2), (1,)}
    full = m.probabilities(c)
    lane_models = LaneModels.from_model(m)
    for support, lane_circuit in lanes.items():
        pos = [qubits.index(q) for q in support]
        for ol, p in lane_models[support][0].probabilities(lane_circuit).items():
            marginal = sum(v for k, v in full.items() if ''.join(k[0][i] for i in pos) == ol[0])
            assert p == pytest.approx(marginal, abs=1e-12)


def exact_lane_data(lane_models, circuits, shots, instrument_names=()):
    """Lane counts equal to `shots` times each lane circuit's probabilities under `lane_models`."""
    qubits = lane_models.qubit_labels
    data = LaneDecomposition(qubits, instrument_names=instrument_names)
    for c in circuits:
        counts = {}
        for support, lane_circuit in split_circuit_into_lanes(c, qubits):
            p = lane_models[support][0].probabilities(lane_circuit)
            if all(len(ol) == 1 for ol in p):
                counts[support] = [shots * p[(format(i, f'0{len(support)}b'),)] for i in range(2**len(support))]
            else:
                counts[support] = {ol: shots * v for ol, v in p.items()}
        data.add_lane_counts(c, counts)
    return data


def general_lane_models(instruments):
    """A physical 3-qubit general model (optionally with instruments) as lane models, and circuits with 5 supports."""
    pspec = line_pspec(3, ('Iz',) if instruments else ())
    general = create_general_crosstalk_free_model(pspec, **({'instruments': {'Iz': 'Z'}} if instruments else {}))
    noisy = member_objects(model(3))                      # physical errors, so that every probability is positive
    for key, (_, member) in member_objects(general).items():
        if key[0] != 'instrument':
            member.set_dense(noisy[key][1].to_dense())
    general.from_vector(general.to_vector())
    lane_models = LaneModels.from_model(general)
    first = [Label('Iz', 1)] if instruments else [Label('Gi', 1)]
    circuits = [Circuit(layers, line_labels=(0, 1, 2)) for layers in (
        [[Label('Gxpi2', 0), Label('Gypi2', 1), Label('Gi', 2)], [Label('Gi', 0)] + first + [Label('Gxpi2', 2)]],
        [[Label('Gcnot', (0, 1)), Label('Gypi2', 2)], [Label('Gxpi2', 0)] + first + [Label('Gi', 2)]],
        [[Label('Gypi2', 0), Label('Gcnot', (1, 2))], [Label('Gxpi2', 0), Label('Gii', (1, 2))]],
        [[Label('Gxpi2', 0), Label('Gxpi2', 1), Label('Gypi2', 2)]])]
    for c in circuits:
        for support, _ in split_circuit_into_lanes(c, (0, 1, 2)):
            lane_models[support]
    return lane_models, circuits


@pytest.mark.parametrize('objective', ['logl', 'chi2'])
@pytest.mark.parametrize('instruments', [False, True])
def test_gradient_of_the_general_model_matches_finite_differences(objective, instruments):
    """
    The gradient is 2 J^T r.  Every gate and SPAM factor of qubit 1 is in three lane models, whose
    contributions must be summed.
    """
    lane_models, circuits = general_lane_models(instruments)
    data = simulate_lanewise_marginals(lane_models, circuits, 1000, seed=1)
    obj = FactorizedObjective(lane_models, data.lane_datasets(), objective)
    assert len(obj.supports) == 5
    rng = np.random.default_rng(11)
    x = lane_models.current_vector() + 0.002 * rng.standard_normal(lane_models.num_params)
    f, g, _ = obj.fn_grad_gn(x)
    assert f == pytest.approx(obj.fn(x), rel=1e-12) and f > 1
    eps = 1e-6
    for _ in range(4):
        v = rng.standard_normal(len(x))
        slope = (obj.fn(x + eps * v) - obj.fn(x - eps * v)) / (2 * eps)
        assert 2 * g @ v == pytest.approx(slope, rel=1e-5)


@pytest.mark.parametrize('instruments', [False, True])
def test_gauss_newton_matrix_is_half_the_hessian_at_a_perfect_fit(instruments):
    """
    Where the data equal the model's predictions the residuals vanish, so the Hessian of chi^2 is
    exactly 2 J^T J.  (Not checked for 'logl': pyGSTi's residuals for it are square roots of terms
    that vanish quadratically there, and their derivatives are 0/0.)
    """
    lane_models, circuits = general_lane_models(instruments)
    x = lane_models.current_vector()
    data = exact_lane_data(lane_models, circuits, 1000, lane_models.processor_spec.instrument_names)
    obj = FactorizedObjective(lane_models, data.lane_datasets(), 'chi2')
    f, g, jtj = obj.fn_grad_gn(x)
    assert abs(f) < 1e-6 and np.max(np.abs(g)) < 1e-3   # zero up to the single precision of the stored counts
    assert np.allclose(jtj.toarray(), jtj.toarray().T)
    rng = np.random.default_rng(12)
    eps = 1e-6
    for _ in range(3):
        v = rng.standard_normal(len(x))
        hessian_v = (2 * obj.fn_grad_gn(x + eps * v)[1] - 2 * obj.fn_grad_gn(x - eps * v)[1]) / (2 * eps)
        assert np.linalg.norm(2 * jtj @ v - hessian_v) < 1e-4 * np.linalg.norm(hessian_v)


@pytest.mark.parametrize('processes', [1, 2])
def test_stages_follow_the_circuit_lists(monkeypatch, processes):
    """
    Each circuit list is fitted in turn using its own circuits' data only, starting from the previous
    stage's estimate; parameters of lanes that first appear in a later list start at their initial values.
    """
    pspec = line_pspec(2)
    oneq = [Circuit([[Label(a, 0), Label(b, 1)]] * n, line_labels=(0, 1))
            for a, b, n in [('Gxpi2', 'Gypi2', 1), ('Gi', 'Gxpi2', 1), ('Gypi2', 'Gi', 2), ('Gxpi2', 'Gxpi2', 2)]]
    twoq = [Circuit(layers, line_labels=(0, 1)) for layers in (
        [Label('Gcnot', (0, 1))], [[Label('Gxpi2', 0), Label('Gi', 1)], Label('Gcnot', (0, 1))],
        [Label('Gii', (0, 1)), Label('Gcnot', (0, 1)), [Label('Gypi2', 0), Label('Gypi2', 1)]])]
    lists = [oneq[:3], oneq, oneq + twoq]
    truth = model(2)
    ds = simulate_lanewise_data(truth, oneq + twoq, 1000, seed=2)

    calls = []
    real = xf_fit.levenberg_marquardt

    def recording(objective, x0, **kwargs):
        calls.append((objective, np.array(x0), objective.fn(x0), kwargs['damping']))
        return real(objective, x0, **kwargs)
    monkeypatch.setattr(xf_fit, 'levenberg_marquardt', recording)

    lane_models = LaneModels(pspec, create_general_crosstalk_free_model)
    res = run_crosstalk_free_gst(ds, lane_models, lists, max_iterations=2, processes=processes, verbosity=0)

    assert [(r.circuit_list_index, r.objective) for r in res.iterations] == [(0, 'chi2'), (1, 'chi2'), (2, 'chi2'),
                                                                             (2, 'logl')]
    assert [r.num_circuits for r in res.iterations] == [3, 4, 7, 7]
    assert [r.num_lane_circuits for r in res.iterations] == [6, 8, 11, 11]   # distinct lane circuits per support
    assert [r.num_elements for r in res.iterations] == [12, 16, 28, 28]      # 2 or 4 outcomes each
    assert all(damping == 'marquardt' for _, _, _, damping in calls)

    # The starting points: nothing but single-qubit lanes until the last list, whose new lane adds Gcnot and Gii.
    initial = lane_models.initial_vector()
    sizes = [len(x0) for _, x0, _, _ in calls]
    assert sizes == [120, 120, 600, 600] and len(initial) == 600
    assert np.array_equal(calls[0][1], initial[:120])
    for (_, x0, _, _), previous in zip(calls[1:], res.iterations):
        assert np.array_equal(x0[:len(previous.lm.x)], previous.lm.x)
        assert np.array_equal(x0[len(previous.lm.x):], initial[len(previous.lm.x):len(x0)])
    assert np.array_equal(res.x, res.iterations[-1].lm.x)
    assert not np.allclose(res.iterations[0].lm.x, initial[:120])           # the fit did move

    # Each stage's objective is the named objective on that list's circuits only.
    decomposition = LaneDecomposition((0, 1), ds, oneq + twoq)
    fresh = LaneModels(pspec, create_general_crosstalk_free_model)
    for (_, x0, value, _), record in zip(calls, res.iterations):
        lane_data = decomposition.lane_datasets(lists[record.circuit_list_index])
        expected = FactorizedObjective(fresh, lane_data, record.objective)
        assert value == pytest.approx(expected.fn(x0), rel=1e-10)
        assert record.fn == pytest.approx(expected.fn(record.lm.x), rel=1e-10)
        assert record.fn <= value

    assert set(res.lane_fns) == {(0,), (1,), (0, 1)}
    assert sum(res.lane_fns.values()) == pytest.approx(res.final_fn, rel=1e-10)
    assert res.model is None and res.lane_models is lane_models


def test_fit_options(design3):
    """`final_objective=None`, an explicit damping, a many-qubit model, and lane data with circuits missing."""
    circuits = design3.circuit_lists[0][::40]
    truth = model(3)
    start = create_crosstalk_free_model(line_pspec(3), lindblad_error_coeffs=ERRORS, lindblad_parameterization='H+s',
                                        independent_gates=True)
    before = start.to_vector()
    ds = simulate_lanewise_data(truth, circuits, 1000, seed=4)
    lane_data = LaneDecomposition((0, 1, 2), ds, circuits).lane_datasets()

    def chi2_of(many_qubit_model):
        lane_models = LaneModels.from_model(many_qubit_model)
        objective = FactorizedObjective(lane_models, lane_data, 'chi2')
        return objective.fn(lane_models.current_vector())

    res = run_crosstalk_free_gst(ds, start, [circuits], final_objective=None, max_iterations=3, verbosity=0)
    assert [(r.circuit_list_index, r.objective) for r in res.iterations] == [(0, 'chi2')]
    assert np.array_equal(start.to_vector(), before)                    # the model given is not modified ...
    assert res.final_fn == pytest.approx(chi2_of(res.model), rel=1e-10)  # ... a copy of it holds the estimate
    assert res.final_fn < chi2_of(start)
    assert sum(res.lane_fns.values()) == pytest.approx(res.final_fn, rel=1e-10)

    identity = run_crosstalk_free_gst(ds, start, [circuits], final_objective=None, max_iterations=3,
                                      damping='identity', verbosity=0)
    marquardt = run_crosstalk_free_gst(ds, start, [circuits], final_objective=None, max_iterations=3,
                                       damping='marquardt', verbosity=0)
    assert np.array_equal(identity.x, res.x)                            # 'auto' is 'identity' for Lindblad models
    assert not np.allclose(marquardt.x, res.x)

    partial = simulate_lanewise_marginals(truth, circuits[:-2], 1000, seed=4)
    with pytest.raises(ValueError, match='2 circuits have no data'):
        run_crosstalk_free_gst(partial, start, [circuits], verbosity=0)
