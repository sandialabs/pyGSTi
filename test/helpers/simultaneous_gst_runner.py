"""Execute simultaneous-GST validation profiles and persist their evidence.

Applies environment overrides, creates per-run manifests and paths, samples data,
delegates fitting to serial or persistent MPI execution, and writes fit metrics.
Scientific acceptance assertions live in the integration tests, not this runner.
"""

import dataclasses
import datetime
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

from pygsti.data import simulate_data
from pygsti.protocols.gst import GateSetTomography
from pygsti.protocols.protocol import ProtocolData
from test.helpers.simultaneous_gst_validation import (
    ValidationProfile, build_ideal_and_datagen_models,
    build_training_design_and_held_out_circuits,
    spectator_term_text,
)
from test.helpers.simultaneous_gst_statistics import score_validation_fit

REPO = pathlib.Path(__file__).resolve().parents[2]


@dataclasses.dataclass(frozen=True)
class ValidationArtifacts:
    root: pathlib.Path

    @property
    def dataset_dir(self):
        return self.root / 'dataset'

    @property
    def results_dir(self):
        return self.root / 'fit'

    @property
    def manifest_path(self):
        return self.root / 'profile.json'


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

    ideal_model, datagen_model = build_ideal_and_datagen_models(profile)
    training_design, held_out = build_training_design_and_held_out_circuits(profile)
    training_circuits = training_design.all_circuits_needing_data
    dataset = simulate_data(
        datagen_model, training_circuits, num_samples=profile.shots,
        sample_error='multinomial', seed=profile.seed)
    pdata = ProtocolData(training_design, dataset)
    pdata.write(artifacts.dataset_dir)

    protocol = GateSetTomography(
        ideal_model,
        gaugeopt_suite='none',
        name='simultaneous_gst_validation',
        verbosity=1,
    )
    results, elapsed_seconds = _run_protocol_persistently(
        protocol, pdata, artifacts, mpi_ranks)

    estimate = results.estimates[protocol.name]
    fitted_model = estimate.models['final iteration estimate']
    metrics = score_validation_fit(
        ideal_model, datagen_model, fitted_model, dataset, held_out, profile.scenario)
    metrics.update({
        'profile': profile.name,
        'scenario': profile.scenario,
        'spectator_term': spectator_term_text(profile.spectator_term),
        'spectator_error': float(profile.spectator_error),
        'elapsed_seconds': float(elapsed_seconds),
        'hypothesis_class_num_params': int(ideal_model.num_params),
        'training_circuits': int(len(training_circuits)),
        'held_out_circuits': int(len(held_out.circuits)),
        'held_out_deep_circuits': int(len(held_out.deepest)),
        'shots': int(profile.shots),
        'mpi_ranks': int(mpi_ranks),
    })
    (artifacts.root / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    return metrics


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


def run_profile(profile):
    """Read runner settings before creating artifacts, then execute one profile."""
    profile = profile_with_overrides(profile)
    ranks = int(os.environ.get('PYGSTI_SGST_MPI_RANKS', '1'))
    if ranks < 1:
        raise ValueError('mpi_ranks must be at least one')
    artifact_root = os.environ.get('PYGSTI_SGST_ARTIFACT_DIR')
    if not artifact_root:
        raise ValueError('PYGSTI_SGST_ARTIFACT_DIR must name an artifact directory')
    artifacts = ValidationArtifacts(create_run_root(pathlib.Path(artifact_root), profile.name))
    return run_validation_profile(profile, artifacts, ranks)


def profile_with_overrides(profile):
    """Return ``profile`` or an immutable per-run replacement selected by the runner.

    The spectator overrides let one queue sweep crosstalk strength and character
    without editing the canonical profiles. ``PYGSTI_SGST_SPECTATOR_TERM`` is
    pipe-separated because a weight-2 coefficient key already contains commas,
    e.g. ``H|ZZ:1,2``.
    """
    changes = {}
    seed = os.environ.get('PYGSTI_SGST_PROFILE_SEED')
    if seed is not None:
        changes['seed'] = int(seed)
    spectator_error = os.environ.get('PYGSTI_SGST_SPECTATOR_ERROR')
    if spectator_error is not None:
        changes['spectator_error'] = float(spectator_error)
    spectator_term = os.environ.get('PYGSTI_SGST_SPECTATOR_TERM')
    if spectator_term is not None:
        changes['spectator_term'] = tuple(spectator_term.split('|'))
    if profile.scenario == 'inclass' and changes.keys() & {'spectator_error', 'spectator_term'}:
        raise ValueError('Spectator overrides require an out-of-class profile')
    return dataclasses.replace(profile, **changes) if changes else profile
