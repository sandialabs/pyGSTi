import os
from pathlib import Path
import pytest
import subprocess
import shutil

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    MPI = None


@pytest.mark.skipif(MPI is None, reason="mpi4py could not be imported")
class TestParallelApply:

    def _extra_mpi_args(self):
        launcher = shutil.which('mpiexec') or shutil.which('mpirun')
        if launcher is not None:
            try:
                result = subprocess.run(
                    [launcher, '--version'],
                    capture_output=True, text=True, timeout=10,
                )
                output = result.stdout + result.stderr
                if 'Open MPI' in output or 'OpenRTE' in output:
                    return ['--oversubscribe']
            except Exception:
                pass
        return []

    @pytest.mark.skipif(MPI is None, reason="mpi4py could not be imported")
    def test_all(self, capfd: pytest.LogCaptureFixture):
        current_filepath = Path(os.path.abspath(__file__))
        to_run = current_filepath.parent / 'run_parallel_apply_with_mpiexec.py'

        launcher = shutil.which('mpiexec') or shutil.which('mpirun')
        if launcher is None:
            raise RuntimeError(
                "mpi4py is installed, but standard MPI launchers (mpiexec and mpirun) are "
                "unavailable. If one should be available in this shell, perhaps you need to "
                "run `module load mpi` or `spack load mpi` first."
            )

        subprocess_args = (
            [launcher, '-np', '4'] + self._extra_mpi_args()
            + ['python', '-W', 'ignore', str(to_run)]
        )
        subprocess.run(subprocess_args, capture_output=False, text=True)
        out, err = capfd.readouterr()

        if len(out.replace('\n', '').replace('\r', '')) + len(err.replace('\n', '').replace('\r', '')) > 0:
            msg = out + '\n' + 80 * '-' + '\n' + err
            raise RuntimeError(msg)
