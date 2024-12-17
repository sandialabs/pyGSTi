import subprocess
import pytest
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class MPITester:

    @pytest.mark.skipif(MPI is None, reason="mpi4py is not installed.")
    def test_all(self, capfd: pytest.LogCaptureFixture):
        result = subprocess.run("mpiexec -np 4 python -W ignore core_test_mpi.py".split(' '), capture_output=True, text=True)
        out, err = capfd.readouterr()
        if len(out) + len(err) > 0:
            msg = out + '\n'+ 80*'-' + err
            raise RuntimeError(msg)
        return

