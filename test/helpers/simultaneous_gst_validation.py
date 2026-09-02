"""Shared helpers for simultaneous-GST validation runs."""

import dataclasses
import datetime
import functools
import itertools
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Literal, Mapping, Sequence

import numpy as np
from scipy.optimize import nnls

from pygsti.algorithms.fiducialselection import find_fiducials
from pygsti.algorithms.germselection import find_germs
from pygsti.data import simulate_data
from pygsti.models import create_cloud_crosstalk_model, create_crosstalk_free_model
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols.gst import GateSetTomography, StandardGSTDesign
from pygsti.protocols.protocol import ProtocolData
from pygsti.protocols.simultaneous_gst import make_simultaneous_gst_design
from pygsti.tools import two_delta_logl, tvd


REPO = pathlib.Path(__file__).resolve().parents[2]


SPARSE_TERMS = {
    'Gi': (('H', 'Z'), ('S', 'X'), ('S', 'Y'), ('S', 'Z')),
    'Gxpi2': (('H', 'Z'), ('S', 'Z')),
    'Gypi2': (('H', 'X'), ('S', 'Z')),
    'Gcnot': (('H', 'ZZ'), ('S', 'XX'), ('S', 'ZZ')),
    'Gii': (('H', 'ZI'), ('H', 'IZ'), ('S', 'XI'), ('S', 'IX'), ('S', 'ZI'), ('S', 'IZ')),
}


COHERENT_TERMS = {
    'Gi': (('H', 'Z'),),
    'Gxpi2': (('H', 'Z'),),
    'Gypi2': (('H', 'X'),),
    'Gcnot': (('H', 'ZZ'),),
    'Gii': (('H', 'ZI'), ('H', 'IZ')),
}


def _full_hs_terms(num_qubits: int) -> tuple[tuple[str, str], ...]:
    pauli_labels = (
        ''.join(label)
        for label in itertools.product('IXYZ', repeat=num_qubits)
        if any(pauli != 'I' for pauli in label)
    )
    labels = tuple(pauli_labels)
    return tuple((term_type, label) for term_type in ('H', 'S') for label in labels)


FULL_HS_TERMS = {
    gate_name: _full_hs_terms(2 if gate_name in ('Gcnot', 'Gii') else 1)
    for gate_name in SPARSE_TERMS
}


@dataclasses.dataclass(frozen=True)
class ValidationProfile:
    """Configuration and matching reduced models for one validation run."""

    name: str
    nqubits: int
    max_lengths: tuple[int, ...]
    shots: int
    model_terms: Literal['coherent', 'sparse', 'full_hs']
    scenario: Literal['markovian', 'spectator_crosstalk']
    seed: int
    spectator_error: float = 0.0

    @classmethod
    def sparse_line(cls, nqubits: int, max_lengths: tuple[int, ...], shots: int = 1000,
                    scenario: Literal['markovian', 'spectator_crosstalk'] = 'markovian',
                    seed: int = 20260901, spectator_error: float = 0.0) -> 'ValidationProfile':
        return cls(f'{nqubits}q_sparse_line', nqubits, tuple(max_lengths), shots,
                   'sparse', scenario, seed, spectator_error)

    @classmethod
    def sparse_spectator_line(cls, nqubits: int, max_lengths: tuple[int, ...], shots: int = 1000,
                              seed: int = 20260901,
                              spectator_error: float = 0.005) -> 'ValidationProfile':
        return cls(f'{nqubits}q_sparse_spectator_line', nqubits, tuple(max_lengths), shots,
                   'sparse', 'spectator_crosstalk', seed, spectator_error)

    @classmethod
    def coherent_line(cls, nqubits: int, max_lengths: tuple[int, ...], shots: int = 1000,
                      scenario: Literal['markovian', 'spectator_crosstalk'] = 'markovian',
                      seed: int = 20260901,
                      spectator_error: float = 0.0) -> 'ValidationProfile':
        return cls(f'{nqubits}q_coherent_line', nqubits, tuple(max_lengths), shots,
                   'coherent', scenario, seed, spectator_error)

    @classmethod
    def coherent_spectator_line(cls, nqubits: int, max_lengths: tuple[int, ...], shots: int = 1000,
                                seed: int = 20260901,
                                spectator_error: float = 0.005) -> 'ValidationProfile':
        return cls(f'{nqubits}q_coherent_spectator_line', nqubits, tuple(max_lengths), shots,
                   'coherent', 'spectator_crosstalk', seed, spectator_error)

    @classmethod
    def full_hs_line(cls, nqubits: int, max_lengths: tuple[int, ...], shots: int = 1000,
                     scenario: Literal['markovian', 'spectator_crosstalk'] = 'markovian',
                     seed: int = 20260901, spectator_error: float = 0.0) -> 'ValidationProfile':
        return cls(f'{nqubits}q_full_hs_line', nqubits, tuple(max_lengths), shots,
                   'full_hs', scenario, seed, spectator_error)

    @functools.cached_property
    def oneq_model(self):
        return _build_model(_oneq_processor_spec(), self.model_terms, independent_gates=False)

    @functools.cached_property
    def twoq_model(self):
        return _build_model(_line_processor_spec(2), self.model_terms, independent_gates=False)


@dataclasses.dataclass(frozen=True)
class ValidationArtifacts:
    root: pathlib.Path
    dataset_dir: pathlib.Path
    results_dir: pathlib.Path
    manifest_path: pathlib.Path


@dataclasses.dataclass(frozen=True)
class RuntimeEstimate:
    """Predicted wall time and the auditable inputs used to predict it."""

    median_seconds: float
    upper_seconds: float
    factors: dict


THREE_QUBIT_SPARSE_MARKOVIAN = ValidationProfile.sparse_line(
    nqubits=3, max_lengths=(1, 2, 4))
THREE_QUBIT_SPARSE_SPECTATOR = ValidationProfile.sparse_spectator_line(
    nqubits=3, max_lengths=(1, 2, 4))
FOUR_QUBIT_SPARSE_MARKOVIAN = ValidationProfile.sparse_line(
    nqubits=4, max_lengths=(1, 2, 4))
FOUR_QUBIT_SPARSE_SPECTATOR = ValidationProfile.sparse_spectator_line(
    nqubits=4, max_lengths=(1, 2, 4))
FOUR_QUBIT_COHERENT_MARKOVIAN = ValidationProfile.coherent_line(
    nqubits=4, max_lengths=(1, 2))
FOUR_QUBIT_COHERENT_SPECTATOR = ValidationProfile.coherent_spectator_line(
    nqubits=4, max_lengths=(1, 2))


def eligible_to_launch(estimate: RuntimeEstimate, remaining_seconds: float) -> bool:
    """Return whether the conservative estimate fits within the remaining budget."""
    return estimate.upper_seconds <= remaining_seconds


def estimate_runtime(
        profiles: Mapping[str, Mapping[str, int | list[int]]],
        twoq_observations: Sequence[Mapping[str, float | int]],
        microbenchmarks: Mapping[str, Mapping[str, float]],
) -> dict[str, RuntimeEstimate]:
    """Fit the specified nonnegative predictor to exactly 18 two-qubit observations."""
    if len(twoq_observations) != 18:
        raise ValueError(f'Expected 18 two-qubit observations, got {len(twoq_observations)}')

    design_matrix = np.asarray([
        (
            float(observation['fit_circuits'])
            * float(observation['dprob_seconds_per_circuit']),
            float(observation['normal_equation_work']),
            1.0,
        )
        for observation in twoq_observations
    ], dtype=float)
    elapsed = np.asarray([
        float(observation['elapsed_seconds']) for observation in twoq_observations
    ], dtype=float)

    # The normal-equation proxy is many orders of magnitude larger than the
    # timed-Jacobian feature.  Column scaling preserves the model while keeping
    # NNLS well-conditioned.
    scales = np.maximum(np.max(np.abs(design_matrix), axis=0), 1.0)
    scaled_coefficients, _ = nnls(design_matrix / scales, elapsed)
    coefficients = scaled_coefficients / scales
    fitted = design_matrix @ coefficients
    residuals = elapsed - fitted
    largest_abs_residual = float(np.max(np.abs(residuals)))
    coefficient_record = {
        'fit_circuit_dprob': float(coefficients[0]),
        'normal_equation_work': float(coefficients[1]),
        'intercept': float(coefficients[2]),
    }

    estimates = {}
    for name, work in profiles.items():
        dprob_rate = float(microbenchmarks[name]['dprob_seconds_per_circuit'])
        median_seconds = float(
            coefficients[0] * int(work['fit_circuits']) * dprob_rate
            + coefficients[1] * int(work['normal_equation_work'])
            + coefficients[2]
        )
        upper_seconds = max(
            1.5 * median_seconds,
            median_seconds + 2.0 * largest_abs_residual,
        )
        estimates[name] = RuntimeEstimate(
            median_seconds=median_seconds,
            upper_seconds=upper_seconds,
            factors={
                **work,
                'dprob_seconds_per_circuit': dprob_rate,
                'coefficients': coefficient_record,
                'largest_absolute_residual': largest_abs_residual,
            },
        )
    return estimates


def _oneq_processor_spec() -> QubitProcessorSpec:
    return QubitProcessorSpec(
        1, gate_names=['Gi', 'Gxpi2', 'Gypi2'], qubit_labels=(0,)
    )


def _line_processor_spec(nqubits: int) -> QubitProcessorSpec:
    qubits = tuple(range(nqubits))
    oneq_locations = [(qubit,) for qubit in qubits]
    line_edges = [(qubit, qubit + 1) for qubit in range(nqubits - 1)]
    availability = {
        'Gi': oneq_locations,
        'Gxpi2': oneq_locations,
        'Gypi2': oneq_locations,
        'Gcnot': line_edges,
        'Gii': line_edges,
    }
    return QubitProcessorSpec(
        nqubits,
        gate_names=['Gi', 'Gxpi2', 'Gypi2', 'Gcnot', 'Gii'],
        nonstd_gate_unitaries={'Gii': np.eye(4)},
        availability=availability,
        qubit_labels=qubits,
    )


def _terms_by_gate(model_terms: Literal['coherent', 'sparse', 'full_hs']) -> dict:
    if model_terms == 'coherent':
        return COHERENT_TERMS
    if model_terms == 'sparse':
        return SPARSE_TERMS
    return FULL_HS_TERMS


def _zero_term_dict(model_terms: Literal['coherent', 'sparse', 'full_hs'], gate_names) -> dict:
    terms_by_gate = _terms_by_gate(model_terms)
    return {
        gate_name: {term: 0.0 for term in terms_by_gate[gate_name]}
        for gate_name in gate_names
    }


def _build_model(processor_spec: QubitProcessorSpec,
                 model_terms: Literal['coherent', 'sparse', 'full_hs'], independent_gates: bool):
    return create_crosstalk_free_model(
        processor_spec,
        lindblad_error_coeffs=_zero_term_dict(model_terms, processor_spec.gate_names),
        lindblad_parameterization='H+S',
        independent_gates=independent_gates,
    )


def _set_deterministic_error_coefficients(model, seed: int, spectator_gate=None) -> None:
    """Set every local coefficient to a small deterministic nonzero value."""
    rng = np.random.default_rng(seed)
    for op_label, coefficients in sorted(model.errorgen_coefficients().items(), key=lambda item: str(item[0])):
        if op_label not in model.operation_blks['layers']:
            continue
        updates = {}
        for coefficient_label in sorted(coefficients, key=str):
            is_spectator_gate = (
                spectator_gate is not None
                and (op_label.name,) + tuple(op_label.sslbls) == spectator_gate
            )
            if (is_spectator_gate
                    and coefficient_label.support == (2,)):
                continue
            if coefficient_label.errorgen_type == 'H':
                value = rng.uniform(-1e-3, 1e-3)
                if value == 0.0:
                    value = 1e-3
            else:
                value = rng.uniform(2.5e-4, 1e-3)
            updates[coefficient_label] = value
        model._op_decomposition(op_label)[1].set_errorgen_coefficients(updates, truncate=False)


def _build_spectator_datagen_model(profile: ValidationProfile):
    if profile.nqubits < 3:
        raise ValueError('Spectator-crosstalk profiles require at least three qubits')
    processor_spec = _line_processor_spec(profile.nqubits)
    terms_by_gate = _terms_by_gate(profile.model_terms)
    coefficients = {
        (gate_name,) + tuple(placement): {
            term: 0.0 for term in terms_by_gate[gate_name]
        }
        for gate_name in processor_spec.gate_names
        for placement in processor_spec.resolved_availability(gate_name)
    }
    spectator_gate = ('Gcnot', 0, 1)
    coefficients[spectator_gate][('H', 'Z:2')] = profile.spectator_error
    model = create_cloud_crosstalk_model(
        processor_spec,
        lindblad_error_coeffs=coefficients,
        lindblad_parameterization='H+S',
        independent_gates=True,
    )
    _set_deterministic_error_coefficients(model, profile.seed, spectator_gate=spectator_gate)
    return model


def build_fit_and_datagen_models(profile: ValidationProfile):
    """Build the local fit hypothesis and deterministic scenario generator."""
    fit_model = _build_model(_line_processor_spec(profile.nqubits), profile.model_terms,
                             independent_gates=True)
    if profile.scenario == 'spectator_crosstalk':
        datagen_model = _build_spectator_datagen_model(profile)
    else:
        datagen_model = fit_model.copy()
        _set_deterministic_error_coefficients(datagen_model, profile.seed)
    return fit_model, datagen_model


def _perturb_hamiltonian_terms(model, seed: int) -> None:
    """Move a selector model off symmetry points without changing its parameterization."""
    rng = np.random.default_rng(seed)
    for op_label, coefficients in model.errorgen_coefficients().items():
        if op_label not in model.operations:
            continue
        perturbation = {
            coefficient_label: 1e-3 * rng.random()
            for coefficient_label in coefficients
            if coefficient_label.errorgen_type == 'H'
        }
        model.operations[op_label].set_errorgen_coefficients(perturbation, truncate=False)


def _build_component_design(processor_spec: QubitProcessorSpec, model, max_lengths: tuple[int, ...],
                            seed: int) -> StandardGSTDesign:
    # Fiducial selection currently requires ExplicitOpModel.operations.  This
    # conversion preserves the reduced H+S parameterization and the component
    # model's tied parameters; it does not substitute a full model or model pack.
    explicit_model = model.to_explicit_model()
    prep_fiducials, meas_fiducials = find_fiducials(
        explicit_model,
        algorithm='greedy',
        candidate_fid_counts=3,
        candidate_seed=seed,
        verbosity=0,
    )

    # find_germs(randomize=True) converts each randomized operation to a full
    # parameterization.  Perturb only the allowed Hamiltonian coefficients and
    # disable that conversion so selection remains on the matching support.
    germ_selection_model = explicit_model.copy()
    _perturb_hamiltonian_terms(germ_selection_model, seed)
    germs = find_germs(
        germ_selection_model,
        randomize=False,
        seed=seed,
        candidate_germ_counts={3: 'all upto'},
        force=None,
        mode='all-Jac',
        verbosity=0,
    )
    if not germs:
        raise RuntimeError('Unable to select an amplificationally complete germ set')

    return StandardGSTDesign(
        processor_spec,
        prep_fiducials,
        meas_fiducials,
        germs,
        list(max_lengths),
        qubit_labels=tuple(processor_spec.qubit_labels),
    )


def build_component_designs(profile: ValidationProfile) -> tuple[StandardGSTDesign, StandardGSTDesign]:
    """Select reproducible 1Q and tied-support 2Q GST component designs."""
    oneq_processor_spec = _oneq_processor_spec()
    twoq_processor_spec = _line_processor_spec(2)
    oneq_design = _build_component_design(
        oneq_processor_spec, profile.oneq_model, profile.max_lengths, profile.seed
    )
    twoq_design = _build_component_design(
        twoq_processor_spec, profile.twoq_model, profile.max_lengths, profile.seed + 1
    )
    return oneq_design, twoq_design


def build_training_design_and_validation_circuits(profile: ValidationProfile):
    """Build one training design and a deterministic disjoint held-out circuit list."""
    processor_spec = _line_processor_spec(profile.nqubits)
    oneq_design, twoq_design = build_component_designs(profile)
    training_design = make_simultaneous_gst_design(
        processor_spec, oneq_design, twoq_design, seed=profile.seed)

    validation_profile = dataclasses.replace(profile, seed=profile.seed + 2)
    validation_oneq_design, validation_twoq_design = build_component_designs(validation_profile)
    validation_design = make_simultaneous_gst_design(
        processor_spec,
        validation_oneq_design,
        validation_twoq_design,
        seed=validation_profile.seed,
    )
    training_circuits = set(training_design.all_circuits_needing_data)
    validation_circuits = tuple(
        circuit for circuit in validation_design.all_circuits_needing_data
        if circuit not in training_circuits
    )
    if not validation_circuits:
        raise RuntimeError('The second seeded design produced no held-out validation circuits')
    return training_design, validation_circuits


def openmpi_extra_args() -> list[str]:
    """Return launcher arguments supported specifically by Open MPI."""
    launcher = shutil.which('mpiexec') or shutil.which('mpirun')
    if launcher is None:
        return []
    try:
        completed = subprocess.run(
            [launcher, '--version'], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    output = completed.stdout + completed.stderr
    return ['--oversubscribe'] if 'Open MPI' in output or 'OpenRTE' in output else []


def generate_finite_shot_data(profile: ValidationProfile, datagen_model, circuits):
    """Sample reproducible multinomial counts for a validation profile."""
    return simulate_data(
        datagen_model,
        circuits,
        num_samples=profile.shots,
        sample_error='multinomial',
        seed=profile.seed,
    )


def _mpi_worker_environment(existing_environment=None) -> dict[str, str]:
    """Build worker overrides that import pyGSTi from this active checkout."""
    environment = os.environ if existing_environment is None else existing_environment
    existing_pythonpath = environment.get('PYTHONPATH')
    pythonpath_entries = [str(REPO)]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    return {'PYTHONPATH': os.pathsep.join(pythonpath_entries)}


def _run_protocol_persistently(protocol, pdata, artifacts: ValidationArtifacts, mpi_ranks: int):
    """Run the fit through its serial or persistent MPI execution path."""
    started = time.perf_counter()
    if mpi_ranks == 1:
        results = protocol.run(pdata)
        results.write(artifacts.results_dir)
    else:
        results = protocol.run_mpi(
            pdata,
            num_ranks=mpi_ranks,
            persistent_dir=artifacts.results_dir,
            blas_threads_per_rank=1,
            extra_mpi_args=openmpi_extra_args(),
            env=_mpi_worker_environment(),
        )
    return results, time.perf_counter() - started


def run_validation_profile(
        profile: ValidationProfile,
        artifacts: ValidationArtifacts,
        mpi_ranks: int,
) -> dict[str, float | int | str]:
    """Run one finite-shot validation profile and persist its complete evidence."""
    if mpi_ranks < 1:
        raise ValueError('mpi_ranks must be at least one')
    artifacts.root.mkdir(parents=True, exist_ok=True)
    artifacts.manifest_path.write_text(
        json.dumps(dataclasses.asdict(profile), indent=2) + '\n')

    fit_model, datagen_model = build_fit_and_datagen_models(profile)
    training_design, validation_circuits = build_training_design_and_validation_circuits(profile)
    training_circuits = training_design.all_circuits_needing_data
    dataset = generate_finite_shot_data(profile, datagen_model, training_circuits)
    pdata = ProtocolData(training_design, dataset)
    pdata.write(artifacts.dataset_dir)

    protocol = GateSetTomography(
        fit_model,
        gaugeopt_suite='none',
        name='simultaneous_gst_validation',
        verbosity=1,
    )
    results, elapsed_seconds = _run_protocol_persistently(
        protocol, pdata, artifacts, mpi_ranks)

    estimate = results.estimates[protocol.name]
    fitted_model = estimate.models['final iteration estimate']
    two_delta_logl_value, nsigma, _ = two_delta_logl(
        fitted_model, dataset, dof_calc_method='modeltest')
    validation_tvds = [
        float(tvd(fitted_model.probabilities(circuit), datagen_model.probabilities(circuit)))
        for circuit in validation_circuits
    ]
    metrics: dict[str, float | int | str] = {
        'profile': profile.name,
        'scenario': profile.scenario,
        'two_delta_logl': float(two_delta_logl_value),
        'nsigma': float(nsigma),
        'validation_mean_tvd': float(np.mean(validation_tvds)),
        'validation_max_tvd': float(np.max(validation_tvds)),
        'elapsed_seconds': float(elapsed_seconds),
        'fit_model_params': int(fit_model.num_params),
        'training_circuits': int(len(training_circuits)),
        'validation_circuits': int(len(validation_circuits)),
        'shots': int(profile.shots),
        'mpi_ranks': int(mpi_ranks),
    }
    (artifacts.root / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    return metrics


def static_work_units(profile: ValidationProfile) -> dict[str, int | list[int]]:
    """Build the exact stitched design and count its fitting work without fitting."""
    oneq_design, twoq_design = build_component_designs(profile)
    simultaneous_design = make_simultaneous_gst_design(
        _line_processor_spec(profile.nqubits), oneq_design, twoq_design,
        seed=profile.seed,
    )
    fit_model, _ = build_fit_and_datagen_models(profile)
    circuits_per_iteration = [len(circuit_list) for circuit_list in simultaneous_design.circuit_lists]
    fit_circuits = sum(circuits_per_iteration)
    outcomes = 2 ** profile.nqubits
    parameter_count = fit_model.num_params
    return {
        'circuits_per_iteration': circuits_per_iteration,
        'fit_circuits': fit_circuits,
        'parameter_count': parameter_count,
        'outcomes': outcomes,
        'jacobian_rows': fit_circuits * outcomes * parameter_count,
        'normal_equation_work': len(simultaneous_design.circuit_lists) * parameter_count ** 3,
    }


def create_run_root(artifact_root: pathlib.Path, profile_name: str) -> pathlib.Path:
    """Create an artifact root and record the execution context."""
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    suffix = 0
    while True:
        collision_suffix = '' if suffix == 0 else f'-{suffix}'
        root = artifact_root / f'{stamp}-{profile_name}{collision_suffix}'
        try:
            root.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            suffix += 1
    (root / 'run-manifest.json').write_text(json.dumps({
        'profile': profile_name,
        'started_at_utc': stamp,
        'git_sha': subprocess.check_output(['git', '-C', REPO, 'rev-parse', 'HEAD'], text=True).strip(),
        'python': sys.executable,
    }, indent=2) + '\n')
    return root
