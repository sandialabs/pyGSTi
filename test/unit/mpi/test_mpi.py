import os
from pathlib import Path
import pytest
import subprocess
import sys
import shutil

from test.helpers.mpi_launch import openmpi_extra_args

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    MPI = None


@pytest.mark.skipif(MPI is None, reason="mpi4py could not be imported")
class MPITester:

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
        subprocess_args = (
            [launcher, '-np', '4'] + openmpi_extra_args(launcher) + [sys.executable, '-W', 'ignore', str(to_run)]
        )
        env = {
            **os.environ,
            'OMP_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1',
            'BLIS_NUM_THREADS': '1',
        }

        result = subprocess.run(subprocess_args, capture_output=False, text=True, env=env)
        out, err = capfd.readouterr()

        #strip new lines/carriage returns before checking length.
        if result.returncode != 0 or len(out.replace('\n', '').replace('\r', '')) + len(err.replace('\n', '').replace('\r', '')) > 0:
            msg = out + '\n'+ 80*'-' + err
            raise RuntimeError(msg)
        return
    
