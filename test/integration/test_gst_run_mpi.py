import shutil
import sys

import numpy as np
import pytest

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    MPI = None

import pygsti
from pygsti.forwardsims import MatrixForwardSimulator
from pygsti.modelpacks import smq1Q_XY as smq1Q_XY
from pygsti.modelpacks import smq1Q_XYI as std
from pygsti.protocols.protocol import SlurmSettings


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

    def test_run_mpi_cptplnd_matrix_matches_serial(self):
        """run_mpi on a CPTPLND MatrixForwardSimulator model escapes the degenerate target start and matches serial."""
        exp_design = smq1Q_XY.create_gst_experiment_design(2)
        mdl_datagen = smq1Q_XY.target_model().depolarize(op_noise=0.01, spam_noise=0.01).rotate((0.01, 0, 0))
        ds = pygsti.data.simulate_data(mdl_datagen, exp_design, 1000, seed=2026)
        data = pygsti.protocols.ProtocolData(exp_design, ds)

        target_ser = smq1Q_XY.target_model("CPTPLND")
        target_ser.sim = MatrixForwardSimulator()
        proto_ser = pygsti.protocols.GateSetTomography(
            pygsti.protocols.GSTInitialModel(model=target_ser, target_model=target_ser),
            gaugeopt_suite=None, verbosity=0, optimizer={'maxiter': 100},
        )
        results_serial = proto_ser.run(data, disable_checkpointing=True)

        target_mpi = smq1Q_XY.target_model("CPTPLND")
        target_mpi.sim = MatrixForwardSimulator()
        proto_mpi = pygsti.protocols.GateSetTomography(
            pygsti.protocols.GSTInitialModel(model=target_mpi, target_model=target_mpi),
            gaugeopt_suite=None, verbosity=0, optimizer={'maxiter': 100},
        )
        results_parallel = proto_mpi.run_mpi(
            data, num_ranks=4, blas_threads_per_rank=1, mpiexec='auto',
            extra_mpi_args=self._extra_mpi_args(),
        )

        fit_ser = results_serial.estimates["GateSetTomography"].models['final iteration estimate']
        fit_mpi = results_parallel.estimates["GateSetTomography"].models['final iteration estimate']
        circuits = exp_design.all_circuits_needing_data
        tdl_ser = pygsti.tools.two_delta_logl(fit_ser, ds, circuits)
        tdl_mpi = pygsti.tools.two_delta_logl(fit_mpi, ds, circuits)
        assert np.isclose(tdl_ser, tdl_mpi, atol=1e-5)
        probs_ser = fit_ser.sim.bulk_probs(circuits)
        probs_mpi = fit_mpi.sim.bulk_probs(circuits)
        for c in circuits:
            for outcome, p_ser in probs_ser[c].items():
                assert abs(probs_mpi[c][outcome] - p_ser) < 1e-6

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

    def test_run_mpi_with_persistent_dir(self, tmp_path):
        """run_mpi with persistent_dir leaves results on disk and returns them."""
        exp_design = std.create_gst_experiment_design(4)
        mdl_datagen = std.target_model().depolarize(op_noise=0.1, spam_noise=0.01)
        ds = pygsti.data.simulate_data(mdl_datagen, exp_design, 1000, seed=1234)
        data = pygsti.protocols.ProtocolData(exp_design, ds)

        initial_model = std.target_model("full TP")
        proto = pygsti.protocols.GateSetTomography(
            initial_model, verbosity=0,
            optimizer={'maxiter': 10},
        )

        results = proto.run_mpi(
            data, num_ranks=2, mpiexec='auto',
            extra_mpi_args=self._extra_mpi_args(),
            persistent_dir=str(tmp_path),
        )
        assert "GateSetTomography" in results.estimates
        # Directory persists and contains written data.
        assert tmp_path.exists()
        assert (tmp_path / 'edesign').is_dir()

    def test_stage_slurm_writes_valid_script(self, tmp_path):
        """stage_slurm produces a script with correct content (no subprocess launched)."""
        exp_design = std.create_gst_experiment_design(4)
        mdl_datagen = std.target_model().depolarize(op_noise=0.1, spam_noise=0.01)
        ds = pygsti.data.simulate_data(mdl_datagen, exp_design, 1000, seed=1234)
        data = pygsti.protocols.ProtocolData(exp_design, ds)

        initial_model = std.target_model("full TP")
        proto = pygsti.protocols.GateSetTomography(initial_model, verbosity=0)

        script_path = str(tmp_path / 'submit.sh')
        slurm = SlurmSettings(script_path, partition='debug', time='1:00:00', nodes=2)
        proto.stage_slurm(
            data, num_ranks=4, slurm=slurm, work_dir=str(tmp_path),
            blas_threads_per_rank=2,
        )

        import ast
        content = (tmp_path / 'submit.sh').read_text()
        assert 'srun python' in content
        assert '#SBATCH --nodes=2' in content
        assert '#SBATCH --partition=debug' in content
        runner_path = tmp_path / 'mpi_runner.py'
        assert runner_path.exists()
        ast.parse(runner_path.read_text())  # syntactically valid Python
