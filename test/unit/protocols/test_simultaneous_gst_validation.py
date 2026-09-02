"""Tests for simultaneous-GST validation execution helpers."""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
from pygsti.circuits import Circuit

import test.helpers.simultaneous_gst_validation as validation
from test.integration.test_simultaneous_gst import profile_with_seed_override
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
    static_work_units,
)


def _load_estimator_module():
    checkout = pathlib.Path(__file__).resolve().parents[3]
    script = checkout.parent / 'projects/simultaneous-gst-correctness/scripts/estimate_runtime.py'
    spec = importlib.util.spec_from_file_location('simultaneous_gst_runtime_estimator', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_overnight_runner_finds_the_validation_checkout_from_workspace():
    checkout = pathlib.Path(__file__).resolve().parents[3]
    workspace = checkout.parent
    runner = workspace / 'projects/simultaneous-gst-correctness/scripts/run_overnight.py'
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, str(runner), '--help'], cwd=workspace,
                            env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


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


def test_profile_seed_override_is_immutable_and_optional(monkeypatch):
    profile = ValidationProfile.sparse_line(nqubits=3, max_lengths=(1, 2), seed=20260901)

    monkeypatch.delenv('PYGSTI_SGST_PROFILE_SEED', raising=False)
    assert profile_with_seed_override(profile) is profile

    monkeypatch.setenv('PYGSTI_SGST_PROFILE_SEED', '606')
    overridden = profile_with_seed_override(profile)
    assert overridden == ValidationProfile.sparse_line(nqubits=3, max_lengths=(1, 2), seed=606)
    assert overridden is not profile
    assert profile.seed == 20260901


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
    training_design, validation_circuits = build_training_design_and_validation_circuits(profile)
    _, repeated_validation_circuits = build_training_design_and_validation_circuits(profile)
    training_circuits = set(training_design.all_circuits_needing_data)
    assert validation_circuits
    assert training_circuits.isdisjoint(validation_circuits)
    assert validation_circuits == repeated_validation_circuits


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


def test_estimator_constrains_loaded_blas_libraries_to_one_thread():
    checkout = pathlib.Path(__file__).resolve().parents[3]
    script = checkout.parent / 'projects/simultaneous-gst-correctness/scripts/estimate_runtime.py'
    environment = os.environ.copy()
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
        environment[variable] = '8'
    code = (
        'import builtins, json, os, runpy; '
        "variables = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', "
        "'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'); "
        'observed = {}; original_import = builtins.__import__; '
        "builtins.__import__ = lambda name, *args, **kwargs: (observed.setdefault("
        "'numpy', {key: os.environ[key] for key in variables}) if name == 'numpy' "
        "and 'numpy' not in observed else None) or original_import(name, *args, **kwargs); "
        f'runpy.run_path({str(script)!r}); '
        "print(json.dumps({'at_numpy_import': observed['numpy'], "
        "'after_script': {key: os.environ[key] for key in variables}}))"
    )
    result = subprocess.run(
        [sys.executable, '-c', code], cwd=checkout, env=environment,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    configured = json.loads(result.stdout.splitlines()[-1])
    assert set(configured['at_numpy_import'].values()) == {'1'}
    assert set(configured['after_script'].values()) == {'1'}


def test_resume_rejects_an_incompatible_mpi_fingerprint():
    estimator = _load_estimator_module()
    checkpoint = {
        'configuration_fingerprint': {
            'sha256': 'old',
            'inputs': {'mpi_ranks': 1},
        },
    }
    expected = {
        'sha256': 'new',
        'inputs': {'mpi_ranks': 2},
    }

    with pytest.raises(ValueError, match='mpi_ranks'):
        estimator._validated_checkpoint(checkpoint, expected, legacy_configuration=None)


def test_recovery_refuses_completed_results_without_prelaunch_fingerprint(tmp_path):
    estimator = _load_estimator_module()
    results_dir = tmp_path / 'twoq-fit-results' / 'sparse_hs' / 'l1'
    marker = results_dir / 'results/twoq_runtime_probe/meta.json'
    marker.parent.mkdir(parents=True)
    marker.write_text('{}')

    with pytest.raises(ValueError, match='prelaunch fingerprint'):
        estimator._recover_elapsed_seconds(
            results_dir, {'sha256': 'expected', 'inputs': {'mpi_ranks': 2}}
        )


def test_recovery_refuses_completed_results_with_mismatched_prelaunch_fingerprint(tmp_path):
    estimator = _load_estimator_module()
    results_dir = tmp_path / 'twoq-fit-results' / 'sparse_hs' / 'l1'
    marker = results_dir / 'results/twoq_runtime_probe/meta.json'
    marker.parent.mkdir(parents=True)
    marker.write_text('{}')
    estimator._persist_prelaunch_fingerprint(
        results_dir, {'sha256': 'recorded', 'inputs': {'mpi_ranks': 1}}
    )

    with pytest.raises(ValueError, match='incompatible prelaunch fingerprint'):
        estimator._recover_elapsed_seconds(
            results_dir, {'sha256': 'expected', 'inputs': {'mpi_ranks': 2}}
        )


def test_legacy_serial_checkpoint_rejects_declarative_blas_thread_setting():
    estimator = _load_estimator_module()
    expected = {
        'sha256': 'current',
        'inputs': {
            'mpi_ranks': 1,
            'shots': 1000,
            'blas_threads_per_rank': 1,
            'optimizer': 'GateSetTomography default SimplerLMOptimizer',
            'family': 'sparse_hs',
            'max_lengths': [1],
            'parameter_count': 17,
            'design_selection_sha256': 'selection',
        },
    }
    checkpoint = {
        'family': 'sparse_hs',
        'max_lengths': [1],
        'parameter_count': 17,
    }
    legacy_configuration = {
        'mpi_ranks': 1,
        'shots': 1000,
        'blas_threads_per_rank': 1,
        'optimizer': 'GateSetTomography default SimplerLMOptimizer',
    }

    with pytest.raises(ValueError, match='serial legacy checkpoint'):
        estimator._validated_checkpoint(
            checkpoint, expected, legacy_configuration=legacy_configuration,
            legacy_design_selection_sha256='selection',
        )


def test_matching_legacy_checkpoint_is_backfilled_with_current_fingerprint():
    estimator = _load_estimator_module()
    expected = {
        'sha256': 'current',
        'inputs': {
            'mpi_ranks': 2,
            'shots': 1000,
            'blas_threads_per_rank': 1,
            'optimizer': 'GateSetTomography default SimplerLMOptimizer',
            'family': 'sparse_hs',
            'max_lengths': [1, 2],
            'parameter_count': 17,
            'design_selection_sha256': 'selection',
        },
    }
    checkpoint = {
        'family': 'sparse_hs',
        'max_lengths': [1, 2],
        'parameter_count': 17,
    }
    legacy_configuration = {
        'mpi_ranks': 2,
        'shots': 1000,
        'blas_threads_per_rank': 1,
        'optimizer': 'GateSetTomography default SimplerLMOptimizer',
    }

    upgraded = estimator._validated_checkpoint(
        checkpoint, expected, legacy_configuration=legacy_configuration,
        legacy_design_selection_sha256='selection',
    )

    assert upgraded['configuration_fingerprint'] == expected


def test_legacy_checkpoint_uses_the_recorded_full_design_selection_for_migration():
    """Legacy records selected once on the full ladder, then fit its prefixes."""
    estimator = _load_estimator_module()
    expected = {
        'sha256': 'current-prefix',
        'inputs': {
            'mpi_ranks': 2,
            'shots': 1000,
            'blas_threads_per_rank': 1,
            'optimizer': 'GateSetTomography default SimplerLMOptimizer',
            'family': 'sparse_hs',
            'max_lengths': [1],
            'parameter_count': 17,
            'design_selection_sha256': 'prefix-selection',
        },
    }
    checkpoint = {
        'family': 'sparse_hs',
        'max_lengths': [1],
        'parameter_count': 17,
    }
    legacy_configuration = {
        'mpi_ranks': 2,
        'shots': 1000,
        'blas_threads_per_rank': 1,
        'optimizer': 'GateSetTomography default SimplerLMOptimizer',
    }

    upgraded = estimator._validated_checkpoint(
        checkpoint, expected, legacy_configuration=legacy_configuration,
        legacy_design_selection_sha256='full-selection',
        expected_legacy_design_selection_sha256='full-selection',
    )

    assert upgraded['configuration_fingerprint'] == expected
