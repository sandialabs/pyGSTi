"""Unit tests for run_mpi / stage_slurm helpers.

These tests require neither mpi4py nor an MPI launcher on PATH.
"""
import builtins
import pathlib
import pickle
import sys
from unittest.mock import MagicMock, patch

import pytest

import pygsti
from pygsti.protocols.protocol import SlurmSettings
from pygsti.tools.mpitools import (
    build_slurm_script,
    compute_blas_threads,
    resolve_mpiexec,
    write_mpi_runner_artifacts,
)


# ---------------------------------------------------------------------------
# Minimal Protocol subclass usable without a real model / experiment design.
# ---------------------------------------------------------------------------

class _MinimalProtocol(pygsti.protocols.Protocol):
    def run(self, data, **kwargs):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# resolve_mpiexec
# ---------------------------------------------------------------------------

class ResolveMpiexecTester:
    def test_auto_finds_mpiexec(self):
        with patch('shutil.which', side_effect=lambda x: '/usr/bin/mpiexec' if x == 'mpiexec' else None):
            assert resolve_mpiexec('auto') == '/usr/bin/mpiexec'

    def test_auto_skips_to_mpirun(self):
        with patch('shutil.which', side_effect=lambda x: '/usr/bin/mpirun' if x == 'mpirun' else None):
            assert resolve_mpiexec('auto') == '/usr/bin/mpirun'

    def test_auto_skips_to_hydra(self):
        def _which(x):
            return '/usr/bin/mpiexec.hydra' if x == 'mpiexec.hydra' else None
        with patch('shutil.which', side_effect=_which):
            assert resolve_mpiexec('auto') == '/usr/bin/mpiexec.hydra'

    def test_auto_nothing_found_raises(self):
        with patch('shutil.which', return_value=None):
            with pytest.raises(FileNotFoundError, match="MPI launcher"):
                resolve_mpiexec('auto')

    def test_explicit_valid(self):
        with patch('shutil.which', side_effect=lambda x: f'/opt/{x}'):
            assert resolve_mpiexec('myexec') == '/opt/myexec'

    def test_explicit_invalid_raises(self):
        with patch('shutil.which', return_value=None):
            with pytest.raises(FileNotFoundError, match='badexec'):
                resolve_mpiexec('badexec')


# ---------------------------------------------------------------------------
# compute_blas_threads
# ---------------------------------------------------------------------------

class ComputeBlasThreadsTester:
    def test_explicit_value_passthrough(self):
        assert compute_blas_threads(4, 3) == 3

    def test_explicit_value_one(self):
        assert compute_blas_threads(8, 1) == 1

    def test_auto_with_psutil(self):
        psutil = pytest.importorskip('psutil')
        with patch.object(psutil, 'cpu_count', return_value=8):
            result = compute_blas_threads(4, 0)
        assert result == 2

    def test_auto_floors_to_one(self):
        psutil = pytest.importorskip('psutil')
        with patch.object(psutil, 'cpu_count', return_value=4):
            result = compute_blas_threads(100, 0)
        assert result == 1

    def test_auto_without_psutil(self):
        """Falls back to os.cpu_count when psutil is not installed."""
        sentinel = object()
        saved = sys.modules.pop('psutil', sentinel)
        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name == 'psutil':
                raise ImportError("psutil blocked for test")
            return real_import(name, *args, **kwargs)

        try:
            with patch.object(builtins, '__import__', side_effect=_blocked):
                with patch('os.cpu_count', return_value=8):
                    result = compute_blas_threads(4, 0)
        finally:
            if saved is not sentinel:
                sys.modules['psutil'] = saved
        assert result == 2


# ---------------------------------------------------------------------------
# build_slurm_script
# ---------------------------------------------------------------------------

class BuildSlurmScriptTester:
    """build_slurm_script is a pure function; no patching needed."""

    def _call(self, **overrides):
        kwargs = dict(
            job_name='TestJob', nodes=2, ntasks_per_node=4,
            cpus_per_task=4, time='1:00:00', partition='debug',
            output=None, error=None, ranks_per_host=None,
            num_ranks=8, runner_path='/work/mpi_runner.py',
            script_path='/work/submit.sh',
        )
        kwargs.update(overrides)
        return build_slurm_script(**kwargs)

    def test_partition_directive(self):
        assert '#SBATCH --partition=debug' in self._call()

    def test_time_directive(self):
        assert '#SBATCH --time=1:00:00' in self._call()

    def test_srun_launch_line(self):
        assert 'srun python /work/mpi_runner.py' in self._call()

    def test_blas_env_exports(self):
        script = self._call(cpus_per_task=4)
        assert 'export OMP_NUM_THREADS=4' in script
        assert 'export MKL_NUM_THREADS=4' in script

    def test_nodes_and_ntasks(self):
        script = self._call()
        assert '#SBATCH --nodes=2' in script
        assert '#SBATCH --ntasks-per-node=4' in script

    def test_fill_in_partition(self):
        assert '--partition=FILL_IN' in self._call(partition=None)

    def test_fill_in_time(self):
        assert '--time=FILL_IN' in self._call(time=None)

    def test_default_output_path(self):
        script = self._call()
        assert 'slurm-%j.out' in script

    def test_default_error_path(self):
        script = self._call()
        assert 'slurm-%j.err' in script

    def test_explicit_output_path(self):
        script = self._call(output='/logs/out.txt')
        assert '#SBATCH --output=/logs/out.txt' in script

    def test_ranks_per_host_comment(self):
        script = self._call(ranks_per_host=4)
        assert '# export PYGSTI_MAX_HOST_PROCS=4' in script

    def test_default_ranks_per_host_suggestion(self):
        # When ranks_per_host is None, suggest num_ranks // nodes.
        script = self._call(ranks_per_host=None, num_ranks=8, nodes=2)
        assert '# export PYGSTI_MAX_HOST_PROCS=4' in script


# ---------------------------------------------------------------------------
# write_mpi_runner_artifacts
# ---------------------------------------------------------------------------

class WriteMpiRunnerArtifactsTester:
    def test_files_created(self, tmp_path):
        mock_proto = MagicMock()
        runner_path = write_mpi_runner_artifacts(mock_proto, '/some/data', {}, tmp_path)
        assert (tmp_path / 'mpi_runner.py').exists()
        assert (tmp_path / 'run_kwargs.pkl').exists()
        assert runner_path == str(tmp_path / 'mpi_runner.py')

    def test_protocol_write_called(self, tmp_path):
        mock_proto = MagicMock()
        write_mpi_runner_artifacts(mock_proto, '/some/data', {}, tmp_path)
        mock_proto.write.assert_called_once_with(str(tmp_path / 'protocol'))

    def test_runner_script_has_mpi4py(self, tmp_path):
        mock_proto = MagicMock()
        write_mpi_runner_artifacts(mock_proto, '/data/path', {}, tmp_path)
        content = (tmp_path / 'mpi_runner.py').read_text()
        assert 'from mpi4py import MPI' in content

    def test_runner_script_embeds_data_dir(self, tmp_path):
        mock_proto = MagicMock()
        write_mpi_runner_artifacts(mock_proto, '/my/data/dir', {}, tmp_path)
        content = (tmp_path / 'mpi_runner.py').read_text()
        assert '/my/data/dir' in content

    def test_disable_checkpointing_default(self, tmp_path):
        mock_proto = MagicMock()
        kwargs = {}
        write_mpi_runner_artifacts(mock_proto, '/data', kwargs, tmp_path)
        with open(tmp_path / 'run_kwargs.pkl', 'rb') as f:
            loaded = pickle.load(f)
        assert loaded['disable_checkpointing'] is True

    def test_disable_checkpointing_not_overwritten(self, tmp_path):
        mock_proto = MagicMock()
        kwargs = {'disable_checkpointing': False}
        write_mpi_runner_artifacts(mock_proto, '/data', kwargs, tmp_path)
        with open(tmp_path / 'run_kwargs.pkl', 'rb') as f:
            loaded = pickle.load(f)
        assert loaded['disable_checkpointing'] is False

    def test_with_real_protocol(self, tmp_path):
        """Smoke-test that a real Protocol instance serializes without error."""
        proto = _MinimalProtocol()
        runner_path = write_mpi_runner_artifacts(proto, str(tmp_path / 'data'), {}, tmp_path)
        assert pathlib.Path(runner_path).exists()
        assert (tmp_path / 'protocol').is_dir()


# ---------------------------------------------------------------------------
# run_mpi — error paths and dry_run (no subprocess spawned)
# ---------------------------------------------------------------------------

class RunMpiErrorPathsTester:
    def test_dry_run_without_persistent_dir_raises(self):
        proto = _MinimalProtocol()
        mock_data = MagicMock()
        with pytest.raises(ValueError, match="persistent_dir"):
            proto.run_mpi(mock_data, num_ranks=4, dry_run=True)

    def test_num_ranks_1_with_dry_run_raises(self):
        proto = _MinimalProtocol()
        mock_data = MagicMock()
        with pytest.raises(AssertionError):
            proto.run_mpi(mock_data, num_ranks=1, dry_run=True)


class RunMpiDryRunTester:
    """dry_run=True writes artifacts and returns None; no subprocess is spawned."""

    def test_returns_none(self, tmp_path):
        proto = _MinimalProtocol()
        mock_data = MagicMock()
        with patch('pygsti.tools.mpitools.resolve_mpiexec', return_value='/fake/mpiexec'):
            with patch('pygsti.tools.mpitools.compute_blas_threads', return_value=2):
                result = proto.run_mpi(
                    mock_data, num_ranks=4, dry_run=True,
                    persistent_dir=str(tmp_path),
                )
        assert result is None

    def test_artifacts_written_to_persistent_dir(self, tmp_path):
        proto = _MinimalProtocol()
        mock_data = MagicMock()
        with patch('pygsti.tools.mpitools.resolve_mpiexec', return_value='/fake/mpiexec'):
            with patch('pygsti.tools.mpitools.compute_blas_threads', return_value=2):
                proto.run_mpi(
                    mock_data, num_ranks=4, dry_run=True,
                    persistent_dir=str(tmp_path),
                )
        assert (tmp_path / 'mpi_runner.py').exists()
        assert (tmp_path / 'run_kwargs.pkl').exists()
        assert (tmp_path / 'protocol').is_dir()

    def test_printed_command_includes_launcher(self, tmp_path, capsys):
        proto = _MinimalProtocol()
        mock_data = MagicMock()
        with patch('pygsti.tools.mpitools.resolve_mpiexec', return_value='/fake/mpiexec'):
            with patch('pygsti.tools.mpitools.compute_blas_threads', return_value=2):
                proto.run_mpi(
                    mock_data, num_ranks=4, dry_run=True,
                    persistent_dir=str(tmp_path),
                )
        out = capsys.readouterr().out
        assert '/fake/mpiexec' in out


# ---------------------------------------------------------------------------
# stage_slurm — no MPI launcher needed
# ---------------------------------------------------------------------------

class StageSlurmMethodTester:

    def test_num_ranks_1_raises(self, tmp_path):
        proto = _MinimalProtocol()
        mock_data = MagicMock()
        slurm = SlurmSettings(str(tmp_path / 'submit.sh'))
        with pytest.raises(ValueError):
            proto.stage_slurm(mock_data, 1, slurm, str(tmp_path))

    def test_uneven_divmod_warns(self, tmp_path):
        proto = _MinimalProtocol()
        mock_data = MagicMock()
        slurm = SlurmSettings(str(tmp_path / 'submit.sh'), nodes=2)
        with patch('pygsti.tools.mpitools.compute_blas_threads', return_value=2):
            with pytest.warns(UserWarning, match="not evenly divisible"):
                proto.stage_slurm(mock_data, 5, slurm, str(tmp_path))

    def test_script_file_written(self, tmp_path):
        proto = _MinimalProtocol()
        mock_data = MagicMock()
        script_path = str(tmp_path / 'submit.sh')
        slurm = SlurmSettings(script_path, partition='debug', time='1:00:00', nodes=2)
        with patch('pygsti.tools.mpitools.compute_blas_threads', return_value=4):
            proto.stage_slurm(mock_data, 8, slurm, str(tmp_path))
        assert pathlib.Path(script_path).exists()

    def test_script_content_spot_checks(self, tmp_path):
        proto = _MinimalProtocol()
        mock_data = MagicMock()
        script_path = str(tmp_path / 'submit.sh')
        slurm = SlurmSettings(script_path, partition='debug', time='2:00:00', nodes=2, job_name='MyJob')
        with patch('pygsti.tools.mpitools.compute_blas_threads', return_value=4):
            proto.stage_slurm(mock_data, 8, slurm, str(tmp_path))
        content = pathlib.Path(script_path).read_text()
        assert 'srun python' in content
        assert '#SBATCH --nodes=2' in content
        assert 'export OMP_NUM_THREADS=4' in content
        assert 'mpi_runner.py' in content
        assert '#SBATCH --partition=debug' in content

    def test_job_name_fallback_to_class_name(self, tmp_path):
        proto = _MinimalProtocol()
        mock_data = MagicMock()
        script_path = str(tmp_path / 'submit.sh')
        slurm = SlurmSettings(script_path, nodes=1)  # job_name=None
        with patch('pygsti.tools.mpitools.compute_blas_threads', return_value=2):
            proto.stage_slurm(mock_data, 4, slurm, str(tmp_path))
        content = pathlib.Path(script_path).read_text()
        assert f'#SBATCH --job-name={type(proto).__name__}' in content

    def test_artifacts_written_to_work_dir(self, tmp_path):
        proto = _MinimalProtocol()
        mock_data = MagicMock()
        script_path = str(tmp_path / 'submit.sh')
        slurm = SlurmSettings(script_path, nodes=1)
        with patch('pygsti.tools.mpitools.compute_blas_threads', return_value=2):
            proto.stage_slurm(mock_data, 4, slurm, str(tmp_path))
        assert (tmp_path / 'mpi_runner.py').exists()
        assert (tmp_path / 'run_kwargs.pkl').exists()
        assert (tmp_path / 'protocol').is_dir()
