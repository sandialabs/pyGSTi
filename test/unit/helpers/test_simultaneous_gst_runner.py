"""Test configuration, persistence delegation and import isolation of the GST runner.

Uses known environment/path contracts and recorded calls to stand-in protocols;
a subprocess checks actual checkout selection. These tests do not run GST fits
or validate MPI numerical results. The long-running marker is checked separately
as a pytest configuration contract.
"""

import os
import pathlib
import subprocess
import sys
import pytest
import test.helpers.simultaneous_gst_runner as runner
from test.helpers.simultaneous_gst_runner import (
    ValidationArtifacts, create_run_root, profile_with_overrides,
)
from test.helpers.simultaneous_gst_validation import ValidationProfile


def test_long_running_marker_is_registered(pytestconfig):
    """Check pytest marker registration against the required ``long_running`` name.

    Reference: the marker must appear in pytest's configured marker declarations.
    """
    assert any('long_running:' in line for line in pytestconfig.getini('markers'))


def test_artifact_root_is_unique_and_has_manifest(tmp_path):
    """Check ``create_run_root`` creates a run directory and manifest.

    Reference: the filesystem contract requires an existing root containing the
    literal ``run-manifest.json`` filename; this does not inspect manifest contents.
    """
    root = create_run_root(tmp_path, profile_name='line_3q_inclass')
    assert root.is_dir()
    assert (root / 'run-manifest.json').is_file()


def test_artifact_roots_are_unique_for_colliding_profile_names(tmp_path):
    """Check ``create_run_root`` keeps repeated profile names in distinct directories.

    Reference: two calls with identical inputs must return unequal, existing paths;
    no particular generated directory name is prescribed.
    """
    first = create_run_root(tmp_path, profile_name='line_3q_inclass')
    second = create_run_root(tmp_path, profile_name='line_3q_inclass')
    assert first != second
    assert first.is_dir()
    assert second.is_dir()


def test_profile_overrides_are_immutable_and_optional(monkeypatch):
    """Check ``profile_with_overrides`` parses overrides without mutating its input.

    Reference: an explicitly constructed profile contains the supplied seed,
    strength and term; no overrides must preserve object identity, while the
    original fields must retain their known initial values after replacement.
    """
    profile = ValidationProfile.outclass(nqubits=3, max_lengths=(1, 2), seed=20260901)
    for variable in ('PYGSTI_SGST_PROFILE_SEED', 'PYGSTI_SGST_SPECTATOR_ERROR',
                     'PYGSTI_SGST_SPECTATOR_TERM'):
        monkeypatch.delenv(variable, raising=False)

    assert profile_with_overrides(profile) is profile

    monkeypatch.setenv('PYGSTI_SGST_PROFILE_SEED', '606')
    monkeypatch.setenv('PYGSTI_SGST_SPECTATOR_ERROR', '0.02')
    monkeypatch.setenv('PYGSTI_SGST_SPECTATOR_TERM', 'H|ZZ:1,2')
    overridden = profile_with_overrides(profile)

    assert overridden == ValidationProfile.outclass(
        nqubits=3, max_lengths=(1, 2), seed=606, spectator_error=0.02,
        spectator_term=('H', 'ZZ:1,2'))
    assert overridden is not profile
    # The canonical module-level profile must not be disturbed by a swept batch.
    assert (profile.seed, profile.spectator_error, profile.spectator_term) == (
        20260901, 0.01, ('S', 'Z:2'))


def test_serial_validation_fit_uses_protocol_run_and_persists_results(tmp_path):
    """Check ``_run_protocol_persistently`` delegates serial execution and persistence.

    Reference: stand-ins record the expected ``run`` then ``write`` calls and return
    a recognizable result; elapsed time need only be nonnegative. No fit is run.
    """
    calls = []

    class Results:
        def write(self, path):
            """Record the persistence call expected from ``_run_protocol_persistently``.

            The enclosing test compares this call log with an explicitly ordered list.
            """
            calls.append(('write', path))

    class Protocol:
        def run(self, pdata):
            """Stand in for serial protocol execution in ``_run_protocol_persistently``.

            Record and return a stub result in the serial test, or fail if the MPI test
            unexpectedly takes this path. No numerical reference model is fitted.
            """
            calls.append(('run', pdata))
            return Results()

    artifacts = ValidationArtifacts(tmp_path)
    pdata = object()
    results, elapsed_seconds = runner._run_protocol_persistently(
        Protocol(), pdata, artifacts, mpi_ranks=1)
    assert isinstance(results, Results)
    assert calls == [('run', pdata), ('write', artifacts.results_dir)]
    assert elapsed_seconds >= 0.0


def test_mpi_validation_fit_uses_persistent_directory_and_single_blas_thread(
        tmp_path, monkeypatch):
    """Check ``_run_protocol_persistently`` selects the MPI delegation path.

    Reference: a stand-in records the exact supplied rank/path/thread/environment
    arguments and returns a sentinel. A serial call fails deliberately; this checks
    launcher configuration, not a real MPI fit.
    """
    calls = []
    expected_results = object()

    class Protocol:
        def run(self, pdata):
            """Stand in for serial protocol execution in ``_run_protocol_persistently``.

            Record and return a stub result in the serial test, or fail if the MPI test
            unexpectedly takes this path. No numerical reference model is fitted.
            """
            raise AssertionError('serial run must not be used for multiple ranks')

        def run_mpi(self, pdata, **kwargs):
            """Record MPI arguments passed by ``_run_protocol_persistently``.

            Return a sentinel so the enclosing test can check identity and exact known
            launcher arguments without running MPI or producing a fitted model.
            """
            calls.append((pdata, kwargs))
            return expected_results

    artifacts = ValidationArtifacts(tmp_path)
    pdata = object()
    monkeypatch.setattr(runner, 'openmpi_extra_args', lambda: ['--oversubscribe'])
    monkeypatch.setattr(
        runner, '_mpi_worker_environment', lambda: {'PYTHONPATH': '/active:/existing'})
    results, elapsed_seconds = runner._run_protocol_persistently(
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
    """Check ``_mpi_worker_environment`` puts the active checkout first.

    Reference: the expected path list is assembled inline from the checkout and
    supplied entries. A subprocess facing a fake competing package must actually
    import pyGSTi from the active checkout, as verified by its resolved file path.
    """
    shadow_root = tmp_path / 'shadow-checkout'
    shadow_package = shadow_root / 'pygsti'
    shadow_package.mkdir(parents=True)
    (shadow_package / '__init__.py').write_text("SOURCE = 'shadow'\n")
    retained_entry = tmp_path / 'retained-entry'
    existing_pythonpath = os.pathsep.join((str(shadow_root), str(retained_entry)))

    worker_overrides = runner._mpi_worker_environment({
        'PYTHONPATH': existing_pythonpath,
    })

    assert worker_overrides['PYTHONPATH'].split(os.pathsep) == [
        str(runner.REPO), str(shadow_root), str(retained_entry),
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
    assert pathlib.Path(completed.stdout.splitlines()[-1]).is_relative_to(runner.REPO)


@pytest.mark.parametrize('variable, value', [
    ('PYGSTI_SGST_MPI_RANKS', '0'), ('PYGSTI_SGST_MPI_RANKS', 'bad'),
    ('PYGSTI_SGST_ARTIFACT_DIR', ''),
    ('PYGSTI_SGST_SPECTATOR_ERROR', '0.0'), ('PYGSTI_SGST_SPECTATOR_TERM', 'S|Z:2'),
])
def test_invalid_runner_settings_create_no_artifacts(tmp_path, monkeypatch, variable, value):
    """Check ``run_profile`` rejects invalid settings before creating artifacts.

    Reference: the listed malformed/inconsistent values must raise ``ValueError``
    and leave the explicitly chosen artifact root absent; no numerical fit is used.
    """
    for name in ('PROFILE_SEED', 'SPECTATOR_ERROR', 'SPECTATOR_TERM'):
        monkeypatch.delenv(f'PYGSTI_SGST_{name}', raising=False)
    root = tmp_path / 'runs'
    monkeypatch.setenv('PYGSTI_SGST_ARTIFACT_DIR', str(root))
    monkeypatch.setenv('PYGSTI_SGST_MPI_RANKS', '1')
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError):
        runner.run_profile(ValidationProfile.inclass(3, (1,)))
    assert not root.exists()


def test_local_spectator_override_is_rejected_before_creating_artifacts(tmp_path, monkeypatch):
    """Check ``run_profile`` rejects a spectator override confined to the CNOT.

    Reference: support (0, 1) is local to that gate, so the known invalid input must
    raise the matching ``ValueError`` and leave the requested artifact root absent.
    """
    monkeypatch.setenv('PYGSTI_SGST_ARTIFACT_DIR', str(tmp_path / 'runs'))
    monkeypatch.setenv('PYGSTI_SGST_SPECTATOR_TERM', 'H|ZZ:0,1')
    with pytest.raises(ValueError, match='does not reach beyond'):
        runner.run_profile(ValidationProfile.outclass(3, (1,)))
    assert not (tmp_path / 'runs').exists()
