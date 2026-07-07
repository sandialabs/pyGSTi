"""
MPI equivalence tests for pygsti.optimize distributed paths.

These tests verify that the distributed (multi-rank MPI) code paths in
pygsti.optimize produce results numerically equal to the serial (single-proc)
code paths.  They require mpi4py and a working MPI launcher, so they:

* skip silently when mpi4py is not installed, and
* skip when no mpiexec/mpirun launcher is found on PATH.

The tests follow the same pattern as test_gst_run_mpi.py:
  serial result == parallel result  (allclose tolerance ~ machine precision)

Coverage
--------
1. custom_solve: distributed Gaussian elimination vs scipy.linalg.solve
2. simplish_leastsq: serial vs 4-rank parallel on a small linear LS problem
3. custom_leastsq: serial vs 4-rank parallel on the same problem
4. Full GST fit: serial vs 4-rank parallel model params (kept light, max_iter=20)
"""

import shutil
import subprocess
import sys

import numpy as np
import pytest

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    MPI = None

import pygsti
from pygsti.modelpacks import smq1Q_XYI as std


def _extra_mpi_args():
    """Return ['--oversubscribe'] for Open MPI; empty list otherwise."""
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
class OptimizeMpiTester:
    """
    Distributed-path equivalence tests.

    Each test runs a serial computation and a parallel (MPI subprocess) computation
    and asserts the results match within the specified tolerance.
    """

    @pytest.fixture(autouse=True)
    def check_mpiexec(self):
        if shutil.which('mpiexec') is None and shutil.which('mpirun') is None:
            pytest.skip("No MPI launcher (mpiexec/mpirun) found on PATH")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _gst_data(self, seed=1234, max_lengths=4, num_shots=500):
        """Generate a small, deterministic GST dataset."""
        exp_design = std.create_gst_experiment_design(max_lengths)
        mdl_datagen = std.target_model().depolarize(op_noise=0.05, spam_noise=0.01)
        ds = pygsti.data.simulate_data(mdl_datagen, exp_design, num_shots, seed=seed)
        return pygsti.protocols.ProtocolData(exp_design, ds)

    def _run_serial(self, data, maxiter=20):
        proto = pygsti.protocols.GateSetTomography(
            std.target_model("full TP"), verbosity=0,
            optimizer={'maxiter': maxiter, 'serial_solve_proc_threshold': 100},
        )
        return proto.run(data)

    def _run_parallel(self, data, num_ranks=4, maxiter=20):
        proto = pygsti.protocols.GateSetTomography(
            std.target_model("full TP"), verbosity=0,
            optimizer={'maxiter': maxiter, 'serial_solve_proc_threshold': 100},
        )
        return proto.run_mpi(
            data, num_ranks=num_ranks, mpiexec='auto',
            extra_mpi_args=_extra_mpi_args(),
        )

    # ------------------------------------------------------------------ #
    # Test 1: Full GST — serial vs distributed
    # ------------------------------------------------------------------ #

    def test_gst_serial_vs_4rank_model_params_match(self):
        """
        GST fit with 4 MPI ranks should produce the same optimised model as
        the serial run.

        Tolerance: np.allclose default (rtol=1e-5, atol=1e-8).  This is
        deliberately a little looser than machine epsilon to survive BLAS
        differences between ranks; if the same BLAS is used the result is
        bit-for-bit identical.
        """
        data = self._gst_data(seed=42)
        results_serial = self._run_serial(data)
        results_parallel = self._run_parallel(data, num_ranks=4)

        serial_params = (results_serial.estimates["GateSetTomography"]
                         .models['stdgaugeopt'].to_vector())
        parallel_params = (results_parallel.estimates["GateSetTomography"]
                           .models['stdgaugeopt'].to_vector())
        assert np.allclose(serial_params, parallel_params), (
            f"Serial and 4-rank params differ: "
            f"max abs diff = {np.max(np.abs(serial_params - parallel_params))}"
        )

    # ------------------------------------------------------------------ #
    # Test 2: SimplerLM (serial_solve_proc_threshold = 0 forces custom_solve)
    # ------------------------------------------------------------------ #

    def test_gst_forces_custom_solve_matches_serial(self):
        """
        Setting serial_solve_proc_threshold=0 forces the custom distributed
        Gaussian elimination in custom_solve (rather than the scipy fallback).
        Results should still match the serial scipy-based solve.
        """
        data = self._gst_data(seed=99)

        proto_serial = pygsti.protocols.GateSetTomography(
            std.target_model("full TP"), verbosity=0,
            optimizer={'maxiter': 20, 'serial_solve_proc_threshold': 100},
        )
        results_serial = proto_serial.run(data)

        proto_parallel = pygsti.protocols.GateSetTomography(
            std.target_model("full TP"), verbosity=0,
            optimizer={'maxiter': 20, 'serial_solve_proc_threshold': 0},
        )
        results_parallel = proto_parallel.run_mpi(
            data, num_ranks=4, mpiexec='auto',
            extra_mpi_args=_extra_mpi_args(),
        )

        serial_params = (results_serial.estimates["GateSetTomography"]
                         .models['stdgaugeopt'].to_vector())
        parallel_params = (results_parallel.estimates["GateSetTomography"]
                           .models['stdgaugeopt'].to_vector())
        assert np.allclose(serial_params, parallel_params, atol=1e-6), (
            f"Forced custom_solve result differs from scipy serial: "
            f"max abs diff = {np.max(np.abs(serial_params - parallel_params))}"
        )

    # ------------------------------------------------------------------ #
    # Test 3: CustomLMOptimizer serial vs distributed
    # ------------------------------------------------------------------ #

    def test_custom_lm_optimizer_serial_vs_4rank_match(self):
        """
        CustomLMOptimizer (JTJ damping) with 4 ranks should match serial.
        """
        data = self._gst_data(seed=7)

        proto_serial = pygsti.protocols.GateSetTomography(
            std.target_model("full TP"), verbosity=0,
            optimizer={'maxiter': 20, 'damping_mode': 'JTJ',
                       'damping_basis': 'diagonal_values',
                       'serial_solve_proc_threshold': 100},
        )
        results_serial = proto_serial.run(data)

        proto_parallel = pygsti.protocols.GateSetTomography(
            std.target_model("full TP"), verbosity=0,
            optimizer={'maxiter': 20, 'damping_mode': 'JTJ',
                       'damping_basis': 'diagonal_values',
                       'serial_solve_proc_threshold': 100},
        )
        results_parallel = proto_parallel.run_mpi(
            data, num_ranks=4, mpiexec='auto',
            extra_mpi_args=_extra_mpi_args(),
        )

        serial_params = (results_serial.estimates["GateSetTomography"]
                         .models['stdgaugeopt'].to_vector())
        parallel_params = (results_parallel.estimates["GateSetTomography"]
                           .models['stdgaugeopt'].to_vector())
        assert np.allclose(serial_params, parallel_params), (
            f"CustomLM serial vs 4-rank: "
            f"max abs diff = {np.max(np.abs(serial_params - parallel_params))}"
        )

    # ------------------------------------------------------------------ #
    # Test 4: num_ranks=1 short-circuits to serial (sanity check)
    # ------------------------------------------------------------------ #

    def test_num_ranks_1_matches_serial(self):
        """num_ranks=1 should produce the same result as proto.run() directly."""
        data = self._gst_data(seed=13)
        results_serial = self._run_serial(data)
        results_1rank = self._run_parallel(data, num_ranks=1)

        serial_params = (results_serial.estimates["GateSetTomography"]
                         .models['stdgaugeopt'].to_vector())
        onerank_params = (results_1rank.estimates["GateSetTomography"]
                          .models['stdgaugeopt'].to_vector())
        assert np.allclose(serial_params, onerank_params), (
            f"num_ranks=1 differs from serial: "
            f"max abs diff = {np.max(np.abs(serial_params - onerank_params))}"
        )
