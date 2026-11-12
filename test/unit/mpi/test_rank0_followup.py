"""Fit on every rank, then do follow-up work on rank 0 alone (issue #964)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from test.helpers.mpi_launch import find_mpi_launcher, openmpi_extra_args

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    MPI = None


@pytest.mark.skipif(MPI is None, reason="mpi4py could not be imported")
def test_hessians_and_rank0_followup_after_mpi_fit():
    launcher = find_mpi_launcher()
    if launcher is None:
        pytest.skip("No MPI launcher (mpiexec/mpirun) found on PATH")
    script = Path(__file__).parent / 'run_rank0_followup_with_mpiexec.py'
    args = [launcher, '-np', '2'] + openmpi_extra_args(launcher) + [sys.executable, '-W', 'ignore', str(script)]
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    # Before the fix, either bug left rank 0 waiting on a collective call forever, so a
    # hang is the failure mode; the timeout is generous compared with the ~20 s run.
    completed = subprocess.run(args, capture_output=True, text=True, timeout=60, env=env)
    assert completed.returncode == 0, completed.stderr
    temp = completed.stdout.strip().splitlines()
    out_a = json.loads(temp[-2])
    out_b = json.loads(temp[-1])
    assert out_a['hessians_agree'] == True
    assert out_b['hessians_agree'] == True
    assert all(out_a['stores_invalidated'].values())
    assert all(out_b['stores_invalidated'].values())
    assert out_a['nsigma_e'] == pytest.approx(out_b['nsigma_e'], rel=1e-5)
    assert out_a['nsigma_t'] == pytest.approx(out_b['nsigma_t'], rel=1e-5)
    return


def test_serial_run_keeps_its_stores():
    """Without a multi-rank comm, the fit's MDC store stays on the estimate for reuse."""
    import pygsti
    from pygsti.modelpacks import smq1Q_XY
    edesign = smq1Q_XY.create_gst_experiment_design(1)
    datagen = smq1Q_XY.target_model().depolarize(op_noise=0.05, spam_noise=0.02)
    data = pygsti.protocols.ProtocolData(edesign, pygsti.data.simulate_data(datagen, edesign, 1000, seed=0))
    results = pygsti.protocols.GateSetTomography(
        smq1Q_XY.target_model('full TP'), gaugeopt_suite=None, verbosity=0).run(data, disable_checkpointing=True)
    estimate = results.estimates['GateSetTomography']
    assert estimate.parameters['final_mdc_store'] is not None
    assert estimate.parameters['per_iter_mdc_store'] is not None
