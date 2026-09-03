"""Tests for simultaneous-GST validation execution helpers."""

import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
from pygsti.circuits import Circuit

import test.helpers.simultaneous_gst_validation as validation
from test.integration.test_simultaneous_gst import profile_with_overrides
from test.helpers.simultaneous_gst_validation import (
    RuntimeEstimate,
    THREE_QUBIT_SPARSE_SPECTATOR,
    ValidationArtifacts,
    ValidationProfile,
    build_component_designs,
    build_fit_and_datagen_models,
    build_training_design_and_validation_circuits,
    create_run_root,
    eligible_to_launch,
    estimate_runtime,
    generate_finite_shot_data,
    spectator_support,
    spectator_term_text,
    static_work_units,
)


def test_long_running_marker_is_registered(pytestconfig):
    assert any('long_running:' in line for line in pytestconfig.getini('markers'))


def test_artifact_root_is_unique_and_has_manifest(tmp_path):
    root = create_run_root(tmp_path, profile_name='threeq_sparse_l4')
    assert root.is_dir()
    assert (root / 'run-manifest.json').is_file()


def test_artifact_roots_are_unique_for_colliding_profile_names(tmp_path):
    first = create_run_root(tmp_path, profile_name='threeq_sparse_l4')
    second = create_run_root(tmp_path, profile_name='threeq_sparse_l4')
    assert first != second
    assert first.is_dir()
    assert second.is_dir()


def test_sparse_component_models_have_expected_parameter_counts():
    profile = ValidationProfile.sparse_line(nqubits=4, max_lengths=(1, 2))
    oneq_design, twoq_design = build_component_designs(profile)
    assert profile.oneq_model.num_params == 8
    assert profile.twoq_model.num_params == 17
    assert oneq_design.qubit_labels == (0,)
    assert twoq_design.qubit_labels == (0, 1)


def test_full_hs_fourqubit_model_has_independent_placement_parameters():
    profile = ValidationProfile.full_hs_line(nqubits=4, max_lengths=(1, 2))
    fit_model, _ = build_fit_and_datagen_models(profile)
    assert fit_model.num_params == 252


def test_markovian_generator_matches_the_fit_hypothesis():
    profile = ValidationProfile.sparse_line(nqubits=3, max_lengths=(1, 2))
    fit_model, datagen_model = build_fit_and_datagen_models(profile)
    assert fit_model.num_params == datagen_model.num_params
    assert np.any(datagen_model.to_vector() != 0.0)
    _, repeated_datagen_model = build_fit_and_datagen_models(profile)
    assert np.array_equal(datagen_model.to_vector(), repeated_datagen_model.to_vector())


def test_spectator_profiles_default_to_calibrated_effect():
    sparse_profile = ValidationProfile.sparse_spectator_line(
        nqubits=3, max_lengths=(1, 2))
    coherent_profile = ValidationProfile.coherent_spectator_line(
        nqubits=4, max_lengths=(1, 2))

    assert sparse_profile.spectator_error == 0.005
    assert coherent_profile.spectator_error == 0.005
    assert THREE_QUBIT_SPARSE_SPECTATOR.spectator_error == 0.005


def test_profile_overrides_are_immutable_and_optional(monkeypatch):
    profile = ValidationProfile.sparse_spectator_line(nqubits=3, max_lengths=(1, 2), seed=20260901)
    for variable in ('PYGSTI_SGST_PROFILE_SEED', 'PYGSTI_SGST_SPECTATOR_ERROR',
                     'PYGSTI_SGST_SPECTATOR_TERM'):
        monkeypatch.delenv(variable, raising=False)

    assert profile_with_overrides(profile) is profile

    monkeypatch.setenv('PYGSTI_SGST_PROFILE_SEED', '606')
    monkeypatch.setenv('PYGSTI_SGST_SPECTATOR_ERROR', '0.02')
    monkeypatch.setenv('PYGSTI_SGST_SPECTATOR_TERM', 'H|ZZ:1,2')
    overridden = profile_with_overrides(profile)

    assert overridden == ValidationProfile.sparse_spectator_line(
        nqubits=3, max_lengths=(1, 2), seed=606, spectator_error=0.02,
        spectator_term=('H', 'ZZ:1,2'))
    assert overridden is not profile
    # The canonical module-level profile must not be disturbed by a swept batch.
    assert (profile.seed, profile.spectator_error, profile.spectator_term) == (
        20260901, 0.005, ('H', 'Z:2'))


def test_spectator_generator_differs_from_the_local_fit_hypothesis():
    profile = ValidationProfile.sparse_spectator_line(
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
    assert spectator_label.errorgen_type == 'H'
    assert spectator_label.basis_element_labels == ('Z',)
    assert spectator_value == pytest.approx(0.001)


def test_spectator_support_parses_local_and_correlated_coefficient_keys():
    assert spectator_support(('H', 'Z:2')) == (2,)
    assert spectator_support(('H', 'ZZ:1,2')) == (1, 2)
    assert spectator_support(('S', 'ZZZ:0,1,2')) == (0, 1, 2)
    assert spectator_support(('H', 'Z')) == ()
    assert spectator_term_text(('H', 'ZZ:1,2')) == 'H(ZZ:1,2)'


def test_correlated_spectator_term_survives_the_deterministic_local_noise():
    """A correlated key has support (1, 2), so the (2,)-only guard would not protect it."""
    profile = ValidationProfile.sparse_spectator_line(
        nqubits=3, max_lengths=(1, 2), spectator_error=0.02, spectator_term=('H', 'ZZ:1,2'))
    _, datagen_model = build_fit_and_datagen_models(profile)

    correlated = {
        label: value
        for label, value in datagen_model.errorgen_coefficients()[('Gcnot', 0, 1)].items()
        if label.support == (1, 2)
    }
    assert len(correlated) == 1
    label, value = next(iter(correlated.items()))
    assert label.errorgen_type == 'H'
    assert value == pytest.approx(0.02)


def test_correlation_spectator_terms_are_built_rather_than_silently_discarded():
    """`H+S` drops `C` coefficients without warning, which would fake a crosstalk run."""
    profile = ValidationProfile.sparse_spectator_line(
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
    """Promoting every data generator to `GLNDU` would change runs that do not need it."""
    assert validation._spectator_parameterization(('H', 'Z:2')) == 'H+S'
    assert validation._spectator_parameterization(('S', 'Z:2')) == 'H+S'
    assert validation._spectator_parameterization(('C', 'Z:2', 'X:2')) == 'GLNDU'
    assert validation._spectator_parameterization(('A', 'Z:2', 'X:2')) == 'GLNDU'


def test_held_out_circuits_are_distinct_so_the_mean_is_not_reweighted():
    """A circuit reached by two germ powers is listed twice; averaging would double it."""
    profile = ValidationProfile.sparse_line(nqubits=3, max_lengths=(1, 2, 4), seed=20260902)
    _, held_out = build_training_design_and_validation_circuits(profile)

    assert len(set(held_out.circuits)) == len(held_out.circuits)
    assert len(set(held_out.deepest)) == len(held_out.deepest)


def test_spectator_term_must_reach_beyond_the_gate_it_is_attached_to():
    """A term confined to the gate's own qubits is local error, not crosstalk."""
    profile = ValidationProfile.sparse_spectator_line(
        nqubits=3, max_lengths=(1, 2), spectator_term=('H', 'ZZ:0,1'))

    with pytest.raises(ValueError, match='does not reach beyond'):
        build_fit_and_datagen_models(profile)


def test_deepest_held_out_circuits_come_from_the_largest_germ_power():
    profile = ValidationProfile.sparse_line(nqubits=3, max_lengths=(1, 2, 4), seed=1234)
    _, held_out = build_training_design_and_validation_circuits(profile)

    assert held_out.deepest
    assert set(held_out.deepest) <= set(held_out.circuits)
    assert len(held_out.deepest) < len(held_out.circuits)
    # The point of the subset is that it concentrates the deep circuits.
    assert (max(circuit.depth for circuit in held_out.deepest)
            == max(circuit.depth for circuit in held_out.circuits))
    assert (min(circuit.depth for circuit in held_out.deepest)
            > min(circuit.depth for circuit in held_out.circuits))


def test_four_qubit_fit_profiles_have_bridge_parameter_counts():
    sparse_profile = ValidationProfile.sparse_line(nqubits=4, max_lengths=(1, 2))
    coherent_profile = ValidationProfile.coherent_line(nqubits=4, max_lengths=(1, 2))
    sparse_fit_model, _ = build_fit_and_datagen_models(sparse_profile)
    coherent_fit_model, _ = build_fit_and_datagen_models(coherent_profile)
    assert sparse_fit_model.num_params == 59
    assert coherent_fit_model.num_params == 21


def test_canonical_four_qubit_sparse_profile_uses_routine_l4_depth():
    assert validation.FOUR_QUBIT_SPARSE_MARKOVIAN.max_lengths == (1, 2, 4)


def test_canonical_four_qubit_sparse_spectator_profile_matches_the_l4_bridge():
    profile = validation.FOUR_QUBIT_SPARSE_SPECTATOR
    fit_model, datagen_model = build_fit_and_datagen_models(profile)

    assert profile.max_lengths == (1, 2, 4)
    assert profile.model_terms == 'sparse'
    assert profile.scenario == 'spectator_crosstalk'
    assert profile.spectator_error == pytest.approx(0.005)
    assert fit_model.num_params == 59
    cnot_coefficients = datagen_model.errorgen_coefficients()[('Gcnot', 0, 1)]
    spectator_coefficients = {
        label: value for label, value in cnot_coefficients.items()
        if label.support == (2,)
    }
    assert len(spectator_coefficients) == 1
    spectator_label, spectator_value = next(iter(spectator_coefficients.items()))
    assert spectator_label.errorgen_type == 'H'
    assert spectator_label.basis_element_labels == ('Z',)
    assert spectator_value == pytest.approx(0.005)


def test_long_running_collection_includes_four_qubit_sparse_spectator_node():
    checkout = pathlib.Path(__file__).resolve().parents[3]
    node_id = (
        'test/integration/test_simultaneous_gst.py::SimultaneousGSTValidationTester::'
        'test_four_qubit_sparse_spectator_crosstalk'
    )
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '--collect-only', '-q', '-m', 'long_running',
         'test/integration/test_simultaneous_gst.py'],
        cwd=checkout, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert node_id in result.stdout


def test_full_hs_component_models_have_expected_parameter_counts():
    profile = ValidationProfile.full_hs_line(nqubits=4, max_lengths=(1, 2))
    assert profile.oneq_model.num_params == 18
    assert profile.twoq_model.num_params == 78


def test_component_design_selection_is_reproducible():
    profile = ValidationProfile.sparse_line(nqubits=4, max_lengths=(1, 2), seed=1234)
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


def test_validation_circuits_are_deterministic_and_disjoint_from_training():
    profile = ValidationProfile.sparse_line(nqubits=3, max_lengths=(1, 2), seed=1234)
    training_design, held_out = build_training_design_and_validation_circuits(profile)
    _, repeated_held_out = build_training_design_and_validation_circuits(profile)
    training_circuits = set(training_design.all_circuits_needing_data)
    assert held_out.circuits
    assert training_circuits.isdisjoint(held_out.circuits)
    assert held_out.circuits == repeated_held_out.circuits
    assert held_out.deepest == repeated_held_out.deepest


def test_finite_shot_data_are_seeded_multinomial_counts():
    profile = ValidationProfile.sparse_line(
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


def test_work_units_count_all_nested_gst_iterations():
    profile = ValidationProfile.sparse_line(nqubits=4, max_lengths=(1, 2, 4))
    work = static_work_units(profile)
    assert work['fit_circuits'] == sum(work['circuits_per_iteration'])
    assert work['jacobian_rows'] == work['fit_circuits'] * 16 * 59


def test_scheduler_rejects_a_profile_above_remaining_budget():
    estimate = RuntimeEstimate(median_seconds=3600, upper_seconds=7200, factors={})
    assert not eligible_to_launch(estimate, remaining_seconds=7199)
    assert eligible_to_launch(estimate, remaining_seconds=7200)


def test_runtime_predictor_fits_nonnegative_coefficients_and_conservative_upper_bound():
    observations = [
        {
            'fit_circuits': index,
            'normal_equation_work': index ** 2,
            'dprob_seconds_per_circuit': 2.0,
            'elapsed_seconds': 6.0 * index + 0.5 * index ** 2 + 7.0,
        }
        for index in range(1, 19)
    ]
    profiles = {
        'candidate': {'fit_circuits': 10, 'normal_equation_work': 100},
    }
    microbenchmarks = {
        'candidate': {'dprob_seconds_per_circuit': 2.0},
    }

    estimates = estimate_runtime(profiles, observations, microbenchmarks)

    assert estimates['candidate'].median_seconds == pytest.approx(117.0)
    assert estimates['candidate'].upper_seconds == pytest.approx(175.5)
    assert estimates['candidate'].factors['coefficients'] == pytest.approx({
        'fit_circuit_dprob': 3.0,
        'normal_equation_work': 0.5,
        'intercept': 7.0,
    })
