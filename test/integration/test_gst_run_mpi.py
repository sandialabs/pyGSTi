import shutil
import sys

import numpy as np
import pytest

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    MPI = None

import pygsti
from pygsti.modelpacks import smq1Q_XYI as std


@pytest.mark.skipif(MPI is None, reason="mpi4py could not be imported")
class RunMpiTester:

    @pytest.fixture(autouse=True)
    def check_mpiexec(self):
        if shutil.which('mpiexec') is None and shutil.which('mpirun') is None:
            pytest.skip("No MPI launcher (mpiexec/mpirun) found on PATH")

    def _extra_mpi_args(self):
        # CI runners tend to have fewer than four cores.
        # --oversubscribe is an Open MPI flag; Intel MPI (Hydra) and MPICH
        # don't recognise it, so only add it when Open MPI is detected.
        import subprocess, shutil
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

    def test_run_mpi_matches_serial(self):
        """run_mpi with num_workers>1 should produce the same model as serial run."""
        exp_design = std.create_gst_experiment_design(4)
        mdl_datagen = std.target_model().depolarize(op_noise=0.1, spam_noise=0.01)
        ds = pygsti.data.simulate_data(mdl_datagen, exp_design, 1000, seed=1234)
        data = pygsti.protocols.ProtocolData(exp_design, ds)

        initial_model = std.target_model("full TP")
        proto = pygsti.protocols.GateSetTomography(
            initial_model, verbosity=0,
            optimizer={'maxiter': 100, 'serial_solve_proc_threshold': 100},
        )

        results_serial = proto.run(data)
        results_parallel = proto.run_mpi(
            data, num_ranks=4, mpiexec='auto',
            extra_mpi_args=self._extra_mpi_args(),
        )

        serial_params = results_serial.estimates["GateSetTomography"].models['stdgaugeopt'].to_vector()
        parallel_params = results_parallel.estimates["GateSetTomography"].models['stdgaugeopt'].to_vector()
        assert np.allclose(serial_params, parallel_params)

    def test_run_mpi_num_workers_1(self):
        """num_workers=1 short-circuits to self.run() — no subprocess spawned."""
        exp_design = std.create_gst_experiment_design(4)
        mdl_datagen = std.target_model().depolarize(op_noise=0.1, spam_noise=0.01)
        ds = pygsti.data.simulate_data(mdl_datagen, exp_design, 1000, seed=1234)
        data = pygsti.protocols.ProtocolData(exp_design, ds)

        initial_model = std.target_model("full TP")
        proto = pygsti.protocols.GateSetTomography(initial_model, verbosity=0)
        results = proto.run_mpi(data, num_ranks=1)
        assert "GateSetTomography" in results.estimates

    def test_run_mpi_repeated_call_inmemory_data(self):
        """Calling run_mpi twice on in-memory data should work on both calls.

        First call writes data to a temp dir (setting edesign._loaded_from).
        After the context manager exits that temp dir is deleted.
        Second call must detect the stale path and re-write, not crash.
        """
        exp_design = std.create_gst_experiment_design(4)
        mdl_datagen = std.target_model().depolarize(op_noise=0.1, spam_noise=0.01)
        ds = pygsti.data.simulate_data(mdl_datagen, exp_design, 1000, seed=1234)
        data = pygsti.protocols.ProtocolData(exp_design, ds)

        initial_model = std.target_model("full TP")
        proto = pygsti.protocols.GateSetTomography(
            initial_model, verbosity=0,
            optimizer={'maxiter': 10},
        )

        extra = self._extra_mpi_args()
        results1 = proto.run_mpi(data, num_ranks=2, mpiexec='auto', extra_mpi_args=extra)
        results2 = proto.run_mpi(data, num_ranks=2, mpiexec='auto', extra_mpi_args=extra)
        assert "GateSetTomography" in results1.estimates
        assert "GateSetTomography" in results2.estimates
