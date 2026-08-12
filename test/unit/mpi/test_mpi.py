import os
from pathlib import Path
import pytest
import subprocess
import sys
import shutil

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    MPI = None


@pytest.mark.skipif(MPI is None, reason="mpi4py could not be imported")
class MPITester:

    def _extra_mpi_args(self):
        # CI runners tend to have fewer than four cores.
        # --oversubscribe is an Open MPI flag; Intel MPI (Hydra) and MPICH
        # don't recognise it, so only add it when Open MPI is detected.
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
        to_run = current_filepath.parents[0] / Path('run_me_with_mpiexec.py')

        launcher = shutil.which('mpiexec') or shutil.which('mpirun')
        if launcher is None:
            msg = \
            """
            mpi4py is installed, but standard MPI launchers (mpiexec and mpirun) are unavailable.
            We're exitng this test with an error.

            If you think an MPI launcher should be available in this shell, perhaps you need to run
            `module load mpi` or `spack load mpi` (or something similar) first.
            """
            raise RuntimeError(msg)

        launcher = shutil.which('mpiexec') or shutil.which('mpirun')
        subprocess_args = [launcher, '-np', '4'] + self._extra_mpi_args() + ['python', '-W', 'ignore', str(to_run)]

        result = subprocess.run(subprocess_args, capture_output=False, text=True)
        out, err = capfd.readouterr()

        #strip new lines/carriage returns before checking length.
        if len(out.replace('\n', '').replace('\r', '')) + len(err.replace('\n', '').replace('\r', '')) > 0:
            msg = out + '\n'+ 80*'-' + err
            raise RuntimeError(msg)
        return
    
