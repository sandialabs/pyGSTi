"""Tests for simultaneous-GST validation execution helpers."""

import dataclasses
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
from pygsti.circuits import Circuit
from pygsti.data import simulate_data
from pygsti.tools import tvd

import test.helpers.simultaneous_gst_validation as validation
from test.integration.test_simultaneous_gst import profile_with_overrides
from test.helpers.simultaneous_gst_validation import (
    LINE_3Q_SPARSE_SPECTATOR,
    ValidationArtifacts,
    ValidationProfile,
    build_component_designs,
    build_fit_and_datagen_models,
    build_training_design_and_held_out_circuits,
    create_run_root,
    expected_fisher_information,
    generate_finite_shot_data,
    identifiable_parameter_count,
    shot_noise_held_out_tvd_quantiles,
    spectator_support,
    spectator_term_text,
)


def test_long_running_marker_is_registered(pytestconfig):
    assert any('long_running:' in line for line in pytestconfig.getini('markers'))


def test_artifact_root_is_unique_and_has_manifest(tmp_path):
    root = create_run_root(tmp_path, profile_name='line_3q_sparse')
    assert root.is_dir()
    assert (root / 'run-manifest.json').is_file()


def test_artifact_roots_are_unique_for_colliding_profile_names(tmp_path):
    first = create_run_root(tmp_path, profile_name='line_3q_sparse')
    second = create_run_root(tmp_path, profile_name='line_3q_sparse')
    assert first != second
    assert first.is_dir()
    assert second.is_dir()


def test_sparse_component_models_have_expected_parameter_counts():
    profile = ValidationProfile.sparse(nqubits=4, max_lengths=(1, 2))
    oneq_design, twoq_design = build_component_designs(profile)
    assert profile.oneq_model.num_params == 8
    assert profile.twoq_model.num_params == 17
    assert oneq_design.qubit_labels == (0,)
    assert twoq_design.qubit_labels == (0, 1)


def test_line_4q_full_hs_fit_model_has_independent_placement_parameters():
    profile = ValidationProfile.full_hs(nqubits=4, max_lengths=(1, 2))
    fit_model, _ = build_fit_and_datagen_models(profile)
    assert fit_model.num_params == 252


def test_in_class_datagen_model_is_a_seeded_point_in_the_hypothesis_class():
    profile = ValidationProfile.sparse(nqubits=3, max_lengths=(1, 2))
    fit_model, datagen_model = build_fit_and_datagen_models(profile)
    assert fit_model.num_params == datagen_model.num_params
    assert np.any(datagen_model.to_vector() != 0.0)
    _, repeated_datagen_model = build_fit_and_datagen_models(profile)
    assert np.array_equal(datagen_model.to_vector(), repeated_datagen_model.to_vector())


def test_spectator_profiles_default_to_calibrated_effect():
    sparse_profile = ValidationProfile.sparse_spectator(
        nqubits=3, max_lengths=(1, 2))

    for profile in (sparse_profile, LINE_3Q_SPARSE_SPECTATOR, validation.LINE_4Q_SPARSE_SPECTATOR):
        assert (profile.spectator_term, profile.spectator_error) == (('S', 'Z:2'), 0.01)


def test_profile_overrides_are_immutable_and_optional(monkeypatch):
    profile = ValidationProfile.sparse_spectator(nqubits=3, max_lengths=(1, 2), seed=20260901)
    for variable in ('PYGSTI_SGST_PROFILE_SEED', 'PYGSTI_SGST_SPECTATOR_ERROR',
                     'PYGSTI_SGST_SPECTATOR_TERM'):
        monkeypatch.delenv(variable, raising=False)

    assert profile_with_overrides(profile) is profile

    monkeypatch.setenv('PYGSTI_SGST_PROFILE_SEED', '606')
    monkeypatch.setenv('PYGSTI_SGST_SPECTATOR_ERROR', '0.02')
    monkeypatch.setenv('PYGSTI_SGST_SPECTATOR_TERM', 'H|ZZ:1,2')
    overridden = profile_with_overrides(profile)

    assert overridden == ValidationProfile.sparse_spectator(
        nqubits=3, max_lengths=(1, 2), seed=606, spectator_error=0.02,
        spectator_term=('H', 'ZZ:1,2'))
    assert overridden is not profile
    # The canonical module-level profile must not be disturbed by a swept batch.
    assert (profile.seed, profile.spectator_error, profile.spectator_term) == (
        20260901, 0.01, ('S', 'Z:2'))


def test_spectator_datagen_model_lies_outside_the_hypothesis_class():
    profile = ValidationProfile.sparse_spectator(
        nqubits=3, max_lengths=(1, 2), spectator_error=0.001)
    fit_model, datagen_model = build_fit_and_datagen_models(profile)
    assert fit_model.num_params < datagen_model.num_params
    cnot_coefficients = datagen_model.errorgen_coefficients()[('Gcnot', 0, 1)]
    spectator_coefficients = {
        label: value for label, value in cnot_coefficients.items()
        if label.support == (2,)
    }
    assert len(spectator_coefficients) == 1
    spectator_label, spectator_value = next(iter(spectator_coefficients.items()))
    assert spectator_label.errorgen_type == 'S'
    assert spectator_label.basis_element_labels == ('Z',)
    assert spectator_value == pytest.approx(0.001)


def test_spectator_support_reads_qubits_from_coefficient_keys():
    assert spectator_support(('H', 'Z:2')) == (2,)
    assert spectator_support(('H', 'ZZ:1,2')) == (1, 2)
    assert spectator_support(('S', 'ZZZ:0,1,2')) == (0, 1, 2)
    assert spectator_support(('H', 'Z')) == ()
    assert spectator_term_text(('H', 'ZZ:1,2')) == 'H(ZZ:1,2)'


def test_weight_two_spectator_term_survives_the_seeded_local_noise():
    """A weight-2 key has support (1, 2), so the (2,)-only guard would not protect it."""
    profile = ValidationProfile.sparse_spectator(
        nqubits=3, max_lengths=(1, 2), spectator_error=0.02, spectator_term=('H', 'ZZ:1,2'))
    _, datagen_model = build_fit_and_datagen_models(profile)

    weight_two = {
        label: value
        for label, value in datagen_model.errorgen_coefficients()[('Gcnot', 0, 1)].items()
        if label.support == (1, 2)
    }
    assert len(weight_two) == 1
    label, value = next(iter(weight_two.items()))
    assert label.errorgen_type == 'H'
    assert value == pytest.approx(0.02)


def test_correlation_spectator_terms_are_built_rather_than_silently_discarded():
    """`H+S` drops `C` coefficients without warning, which would fake a crosstalk run."""
    profile = ValidationProfile.sparse_spectator(
        nqubits=3, max_lengths=(1, 2), spectator_error=0.01,
        spectator_term=('C', 'Z:2', 'X:2'))
    _, datagen_model = build_fit_and_datagen_models(profile)

    injected = {
        label: value
        for label, value in datagen_model.errorgen_coefficients()[('Gcnot', 0, 1)].items()
        if label.errorgen_type == 'C'
    }
    assert len(injected) == 1
    label, value = next(iter(injected.items()))
    assert label.support == (2,)
    assert tuple(label.basis_element_labels) == ('Z', 'X')
    assert value == pytest.approx(0.01)


def test_hamiltonian_and_stochastic_terms_keep_the_original_parameterization():
    """Promoting every data-generating model to `GLNDU` would change runs that do not need it."""
    assert validation._spectator_parameterization(('H', 'Z:2')) == 'H+S'
    assert validation._spectator_parameterization(('S', 'Z:2')) == 'H+S'
    assert validation._spectator_parameterization(('C', 'Z:2', 'X:2')) == 'GLNDU'
    assert validation._spectator_parameterization(('A', 'Z:2', 'X:2')) == 'GLNDU'


def test_held_out_circuits_are_distinct_so_the_mean_is_not_reweighted():
    """A circuit reached by two germ powers is listed twice; averaging would double it."""
    profile = ValidationProfile.sparse(nqubits=3, max_lengths=(1, 2, 4), seed=20260902)
    _, held_out = build_training_design_and_held_out_circuits(profile)

    assert len(set(held_out.circuits)) == len(held_out.circuits)
    assert len(set(held_out.deepest)) == len(held_out.deepest)


def test_spectator_term_must_reach_beyond_the_gate_it_is_attached_to():
    """A term confined to the gate's own qubits is local error, not crosstalk."""
    profile = ValidationProfile.sparse_spectator(
        nqubits=3, max_lengths=(1, 2), spectator_term=('H', 'ZZ:0,1'))

    with pytest.raises(ValueError, match='does not reach beyond'):
        build_fit_and_datagen_models(profile)


def test_deepest_held_out_circuits_come_from_the_largest_max_length():
    profile = ValidationProfile.sparse(nqubits=3, max_lengths=(1, 2, 4), seed=1234)
    _, held_out = build_training_design_and_held_out_circuits(profile)

    assert held_out.deepest
    assert set(held_out.deepest) <= set(held_out.circuits)
    assert len(held_out.deepest) < len(held_out.circuits)
    # The point of the subset is that it concentrates the deep circuits.
    assert (max(circuit.depth for circuit in held_out.deepest)
            == max(circuit.depth for circuit in held_out.circuits))
    assert (min(circuit.depth for circuit in held_out.deepest)
            > min(circuit.depth for circuit in held_out.circuits))


def test_line_4q_sparse_fit_model_has_59_parameters():
    sparse_profile = ValidationProfile.sparse(nqubits=4, max_lengths=(1, 2))
    sparse_fit_model, _ = build_fit_and_datagen_models(sparse_profile)
    assert sparse_fit_model.num_params == 59


def test_ring_processor_spec_adds_the_wraparound_edge_to_the_line():
    line = validation._processor_spec(4)
    ring = validation._processor_spec(4, 'ring')
    line_edges = [(0, 1), (1, 2), (2, 3)]
    for gate_name in ('Gcnot', 'Gii'):
        assert list(line.resolved_availability(gate_name)) == line_edges
        assert list(ring.resolved_availability(gate_name)) == line_edges + [(0, 3)]
    with pytest.raises(ValueError, match='at least three qubits'):
        validation._processor_spec(2, 'ring')


def test_ring_fit_models_add_one_edge_of_parameters():
    # Each edge carries 3 Gcnot and 6 Gii parameters, so the ring adds 9 to the line.
    for nqubits, num_params in ((3, 51), (4, 68)):
        profile = ValidationProfile.sparse(nqubits=nqubits, max_lengths=(1,), topology='ring')
        fit_model, _ = build_fit_and_datagen_models(profile)
        assert profile.name == f'ring_{nqubits}q_sparse'
        assert fit_model.num_params == num_params


def test_ring_3q_design_colors_and_uses_the_wraparound_edge():
    profile = ValidationProfile.sparse(nqubits=3, max_lengths=(1,), seed=1234, topology='ring')
    training_design, _ = build_training_design_and_held_out_circuits(profile)
    # An odd cycle has no proper 2-edge-coloring.
    assert training_design.color_patches == {0: [(0, 1)], 1: [(1, 2)], 2: [(0, 2)]}
    cnot_supports = {
        component.sslbls
        for circuit in training_design.all_circuits_needing_data
        for layer in circuit
        for component in layer.components
        if component.name == 'Gcnot'
    }
    assert cnot_supports == {(0, 1), (1, 2), (0, 2)}


def test_canonical_ring_profiles_match_their_line_counterparts():
    pairs = (
        (validation.LINE_3Q_SPARSE_IN_CLASS, validation.RING_3Q_SPARSE_IN_CLASS),
        (validation.LINE_3Q_SPARSE_SPECTATOR, validation.RING_3Q_SPARSE_SPECTATOR),
        (validation.LINE_4Q_SPARSE_IN_CLASS, validation.RING_4Q_SPARSE_IN_CLASS),
        (validation.LINE_4Q_SPARSE_SPECTATOR, validation.RING_4Q_SPARSE_SPECTATOR),
    )
    for line, ring in pairs:
        assert (line.topology, ring.topology) == ('line', 'ring')
        assert ring.name == line.name.replace('line_', 'ring_', 1)
        assert ring == dataclasses.replace(line, name=ring.name, topology='ring')


def test_ring_spectator_datagen_model_keeps_the_injected_term():
    _, datagen_model = build_fit_and_datagen_models(validation.RING_4Q_SPARSE_SPECTATOR)
    cnot_coefficients = datagen_model.errorgen_coefficients()[('Gcnot', 0, 1)]
    spectator_coefficients = {
        label: value for label, value in cnot_coefficients.items() if label.support == (2,)
    }
    assert len(spectator_coefficients) == 1
    assert next(iter(spectator_coefficients.values())) == pytest.approx(0.01)


def test_canonical_line_4q_sparse_profile_has_max_length_four():
    assert validation.LINE_4Q_SPARSE_IN_CLASS.max_lengths == (1, 2, 4)


def test_canonical_line_4q_spectator_profile_adds_s_z2_to_the_in_class_setup():
    profile = validation.LINE_4Q_SPARSE_SPECTATOR
    fit_model, datagen_model = build_fit_and_datagen_models(profile)

    assert profile.max_lengths == (1, 2, 4)
    assert profile.model_terms == 'sparse'
    assert profile.scenario == 'spectator_crosstalk'
    assert profile.spectator_error == pytest.approx(0.01)
    assert fit_model.num_params == 59
    cnot_coefficients = datagen_model.errorgen_coefficients()[('Gcnot', 0, 1)]
    spectator_coefficients = {
        label: value for label, value in cnot_coefficients.items()
        if label.support == (2,)
    }
    assert len(spectator_coefficients) == 1
    spectator_label, spectator_value = next(iter(spectator_coefficients.items()))
    assert spectator_label.errorgen_type == 'S'
    assert spectator_label.basis_element_labels == ('Z',)
    assert spectator_value == pytest.approx(0.01)


def test_long_running_collection_includes_line_4q_spectator_node():
    checkout = pathlib.Path(__file__).resolve().parents[3]
    node_id = (
        'test/integration/test_simultaneous_gst.py::SimultaneousGSTValidationTester::'
        'test_line_4q_spectator_crosstalk'
    )
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '--collect-only', '-q', '-m', 'long_running',
         'test/integration/test_simultaneous_gst.py'],
        cwd=checkout, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert node_id in result.stdout


def test_full_hs_component_models_have_expected_parameter_counts():
    profile = ValidationProfile.full_hs(nqubits=4, max_lengths=(1, 2))
    assert profile.oneq_model.num_params == 18
    assert profile.twoq_model.num_params == 78


def test_component_design_selection_is_reproducible():
    profile = ValidationProfile.sparse(nqubits=4, max_lengths=(1, 2), seed=1234)
    first_designs = build_component_designs(profile)
    second_designs = build_component_designs(profile)

    def serialize(design):
        return {
            'prep_fiducials': tuple(circuit.str for circuit in design.prep_fiducials),
            'meas_fiducials': tuple(circuit.str for circuit in design.meas_fiducials),
            'germs': tuple(circuit.str for circuit in design.germs),
            'circuit_lists': tuple(
                tuple(circuit.str for circuit in circuit_list)
                for circuit_list in design.circuit_lists
            ),
        }

    assert tuple(map(serialize, first_designs)) == tuple(map(serialize, second_designs))


def test_held_out_circuits_are_reproducible_and_disjoint_from_training():
    profile = ValidationProfile.sparse(nqubits=3, max_lengths=(1, 2), seed=1234)
    training_design, held_out = build_training_design_and_held_out_circuits(profile)
    _, repeated_held_out = build_training_design_and_held_out_circuits(profile)
    training_circuits = set(training_design.all_circuits_needing_data)
    assert held_out.circuits
    assert training_circuits.isdisjoint(held_out.circuits)
    assert held_out.circuits == repeated_held_out.circuits
    assert held_out.deepest == repeated_held_out.deepest


def test_finite_shot_data_are_seeded_multinomial_counts():
    profile = ValidationProfile.sparse(
        nqubits=3, max_lengths=(1,), shots=37, seed=1234)
    _, datagen_model = build_fit_and_datagen_models(profile)
    circuits = (
        Circuit([('Gxpi2', 0)], line_labels=(0, 1, 2)),
        Circuit([('Gcnot', 0, 1)], line_labels=(0, 1, 2)),
    )
    first_dataset = generate_finite_shot_data(profile, datagen_model, circuits)
    second_dataset = generate_finite_shot_data(profile, datagen_model, circuits)
    for circuit in circuits:
        assert first_dataset[circuit].total == 37
        assert all(float(count).is_integer() for count in first_dataset[circuit].counts.values())
        assert first_dataset[circuit].counts == second_dataset[circuit].counts


def _noisy_oneq_model_and_circuits(shots):
    """A noisy one-qubit sparse model, training data at powers 1 and 2, held-out circuits at 4."""
    model = ValidationProfile.sparse(nqubits=3, max_lengths=(1,)).oneq_model.copy()
    model.from_vector(np.random.default_rng(7).uniform(0.002, 0.02, model.num_params))
    fiducials = [(), ('Gxpi2',), ('Gypi2',), ('Gxpi2',) * 2, ('Gxpi2',) * 3, ('Gypi2',) * 3]
    germs = [('Gi',), ('Gxpi2',), ('Gypi2',), ('Gxpi2', 'Gypi2')]

    def circuits(powers):
        return list(dict.fromkeys(
            Circuit([(gate, 0) for gate in prep + germ * power + meas], line_labels=(0,))
            for prep in fiducials for germ in germs for power in powers for meas in fiducials))

    training = circuits((1, 2))
    held_out = [circuit for circuit in circuits((4,)) if circuit not in set(training)]
    dataset = simulate_data(model, training, num_samples=shots, seed=0)
    return model, dataset, held_out


def test_expected_fisher_information_is_the_expected_negative_loglikelihood_hessian():
    # Brute-force reference from probabilities alone: central differences of
    # sum_c N_c sum_o p_o(theta) log p_o(theta') in theta', at theta' = theta.
    model, dataset, _ = _noisy_oneq_model_and_circuits(shots=1000)
    circuits = list(dataset.keys())
    theta = model.to_vector()
    truth = model.sim.bulk_probs(circuits)
    trial = model.copy()

    def expected_loglikelihood(vector):
        trial.from_vector(vector)
        probabilities = trial.sim.bulk_probs(circuits)
        return sum(dataset[circuit].total * truth[circuit][outcome] * np.log(probabilities[circuit][outcome])
                   for circuit in circuits for outcome in truth[circuit])

    step = 1e-4
    basis = np.eye(len(theta)) * step
    hessian = np.zeros((len(theta), len(theta)))
    for i in range(len(theta)):
        for j in range(i, len(theta)):
            hessian[i, j] = hessian[j, i] = (
                expected_loglikelihood(theta + basis[i] + basis[j])
                - expected_loglikelihood(theta + basis[i] - basis[j])
                - expected_loglikelihood(theta - basis[i] + basis[j])
                + expected_loglikelihood(theta - basis[i] - basis[j])) / (4 * step ** 2)

    fisher = expected_fisher_information(model, dataset)
    np.testing.assert_allclose(fisher, -hessian, rtol=1e-3, atol=1e-6 * np.abs(fisher).max())


def test_identifiable_parameter_count_is_the_fisher_information_rank():
    model, dataset, _ = _noisy_oneq_model_and_circuits(shots=1000)
    fisher = expected_fisher_information(model, dataset)
    assert identifiable_parameter_count(model, dataset) == np.linalg.matrix_rank(fisher)

    # One informative circuit identifies one direction: its two outcome rows of the Jacobian
    # are r and -r, because the probabilities sum to one, so F is a multiple of r r^T.
    single_circuit = simulate_data(
        model, [Circuit([], line_labels=(0,)), Circuit([('Gi', 0)], line_labels=(0,))],
        num_samples=1000, seed=0)
    assert identifiable_parameter_count(model, single_circuit) == 1


def test_shot_noise_quantiles_match_monte_carlo_of_exact_held_out_predictions():
    # Reference: draw theta_hat ~ N(theta, F^-1) and evaluate the exact (not linearized)
    # held-out TVDs.  Enough shots keep theta_hat close enough to theta for the
    # linearization to hold to within the Monte Carlo error.
    model, dataset, held_out = _noisy_oneq_model_and_circuits(shots=100000)
    theta = model.to_vector()
    covariance = np.linalg.inv(expected_fisher_information(model, dataset))
    truth = model.sim.bulk_probs(held_out)
    trial = model.copy()
    rng = np.random.default_rng(1)
    means, maxima = [], []
    for vector in rng.multivariate_normal(theta, covariance, size=1000):
        trial.from_vector(vector)
        predictions = trial.sim.bulk_probs(held_out)
        tvds = [float(tvd(predictions[circuit], truth[circuit])) for circuit in held_out]
        means.append(np.mean(tvds))
        maxima.append(np.max(tvds))

    mean_quantile, max_quantile = shot_noise_held_out_tvd_quantiles(
        model, dataset, held_out, quantile=0.9, samples=20000)
    assert mean_quantile == pytest.approx(np.quantile(means, 0.9), rel=0.1)
    assert max_quantile == pytest.approx(np.quantile(maxima, 0.9), rel=0.1)


def test_serial_validation_fit_uses_protocol_run_and_persists_results(tmp_path):
    calls = []

    class Results:
        def write(self, path):
            calls.append(('write', path))

    class Protocol:
        def run(self, pdata):
            calls.append(('run', pdata))
            return Results()

    artifacts = ValidationArtifacts(
        tmp_path, tmp_path / 'dataset', tmp_path / 'fit', tmp_path / 'profile.json')
    pdata = object()
    results, elapsed_seconds = validation._run_protocol_persistently(
        Protocol(), pdata, artifacts, mpi_ranks=1)
    assert isinstance(results, Results)
    assert calls == [('run', pdata), ('write', artifacts.results_dir)]
    assert elapsed_seconds >= 0.0


def test_mpi_validation_fit_uses_persistent_directory_and_single_blas_thread(
        tmp_path, monkeypatch):
    calls = []
    expected_results = object()

    class Protocol:
        def run(self, pdata):
            raise AssertionError('serial run must not be used for multiple ranks')

        def run_mpi(self, pdata, **kwargs):
            calls.append((pdata, kwargs))
            return expected_results

    artifacts = ValidationArtifacts(
        tmp_path, tmp_path / 'dataset', tmp_path / 'fit', tmp_path / 'profile.json')
    pdata = object()
    monkeypatch.setattr(validation, 'openmpi_extra_args', lambda: ['--oversubscribe'])
    monkeypatch.setattr(
        validation, '_mpi_worker_environment', lambda: {'PYTHONPATH': '/active:/existing'})
    results, elapsed_seconds = validation._run_protocol_persistently(
        Protocol(), pdata, artifacts, mpi_ranks=2)
    assert results is expected_results
    assert calls == [(pdata, {
        'num_ranks': 2,
        'persistent_dir': artifacts.results_dir,
        'blas_threads_per_rank': 1,
        'extra_mpi_args': ['--oversubscribe'],
        'env': {'PYTHONPATH': '/active:/existing'},
    })]
    assert elapsed_seconds >= 0.0


def test_mpi_worker_environment_imports_active_repo_before_existing_pythonpath(tmp_path):
    shadow_root = tmp_path / 'shadow-checkout'
    shadow_package = shadow_root / 'pygsti'
    shadow_package.mkdir(parents=True)
    (shadow_package / '__init__.py').write_text("SOURCE = 'shadow'\n")
    retained_entry = tmp_path / 'retained-entry'
    existing_pythonpath = os.pathsep.join((str(shadow_root), str(retained_entry)))

    worker_overrides = validation._mpi_worker_environment({
        'PYTHONPATH': existing_pythonpath,
    })

    assert worker_overrides['PYTHONPATH'].split(os.pathsep) == [
        str(validation.REPO), str(shadow_root), str(retained_entry),
    ]
    completed = subprocess.run(
        [
            sys.executable,
            '-c',
            'import pathlib, pygsti, pygsti.protocols.simultaneous_gst; '
            'print(pathlib.Path(pygsti.__file__).resolve())',
        ],
        cwd=tmp_path,
        env={**os.environ, **worker_overrides},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert pathlib.Path(completed.stdout.splitlines()[-1]).is_relative_to(validation.REPO)
