import subprocess
import pytest
import os
from pathlib import Path

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

class MPITester:

    @pytest.mark.skipif(MPI is None, reason="mpi4py is not installed.")
    def test_all(self, capfd: pytest.LogCaptureFixture):
        current_filepath = Path(os.path.abspath(__file__))
        to_run = current_filepath.parents[0] / Path('run_me_with_mpiexec.py')
        subprocess_args = (f"mpiexec -np 4 python -W ignore {str(to_run)}").split(' ')
        
        result = subprocess.run(subprocess_args, capture_output=False, text=True)
        out, err = capfd.readouterr()
        if len(out) + len(err) > 0:
            msg = out + '\n'+ 80*'-' + err
            raise RuntimeError(msg)
        return

