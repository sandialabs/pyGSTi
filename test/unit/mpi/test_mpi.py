import os
from pathlib import Path
import pytest
import subprocess
import sys

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    MPI = None


class MPITester:

    @pytest.mark.skipif(MPI is None, reason="mpi4py could not be imported")
    def test_all(self, capfd: pytest.LogCaptureFixture):
        current_filepath = Path(os.path.abspath(__file__))
        to_run = current_filepath.parents[0] / Path('run_me_with_mpiexec.py')
        subprocess_args = (f"mpiexec -np 4 python -W ignore {str(to_run)}").split(' ')

        # Oversubscribe is needed because latest Mac runners have only 3 cores
        # Cannot have in general though because then Windows breaks (not right arg name)
        if sys.platform == "darwin":
            subprocess_args.insert(3, "-oversubscribe")
        
        result = subprocess.run(subprocess_args, capture_output=False, text=True)
        out, err = capfd.readouterr()
        if len(out) + len(err) > 0:
            msg = out + '\n'+ 80*'-' + err
            raise RuntimeError(msg)
        return
    
