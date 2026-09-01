"""Shared inputs for the documentation notebooks.

Several tutorial notebooks under ``docs/markdown`` analyze data that *other*
tutorials generate and save under ``docs/tutorial_files``:

* ``guides/workflow/DataSets`` writes ``Example_Dataset.txt``,
  ``Example_Dataset_LowCnts.txt`` and the ``Example_GST_Data`` protocol-data
  directory.  ``advanced/extending/LowLevelGST`` reads both datasets,
  ``guides/analysis/Reports`` reads ``Example_Dataset.txt``, and
  ``guides/gst/RunningGST`` reads ``Example_GST_Data``.
* ``guides/gst/RunningGST`` runs GST and writes the ``Example_GST_Data/results``
  tree, which ``guides/analysis/Results`` reads back.

Nothing guarantees a generating notebook runs before its consumers, under either
runner, so these shared inputs are materialized exactly once up front.  There are
two entry points:

* Under pytest, the autouse session fixture below does it, guarded by a lock so
  that ``-n auto`` workers cooperate.
* ``docs/execute-notebooks.py`` -- which is not pytest, so no fixture fires --
  calls ``_generate_shared_fixtures()`` directly.  CI does this before executing
  the notebooks for the hosted docs.

Everything is written into the gitignored ``docs/tutorial_files`` dir, so nothing
here is ever committed.

Note that this covers only *cross-notebook* inputs.  Files a notebook writes and
then reads a few cells later are its own business and need nothing here.
"""
import os
import time
import pathlib

_DOCS_DIR = pathlib.Path(__file__).resolve().parent
_TUT = _DOCS_DIR / "tutorial_files"
_READY = _TUT / ".fixtures_ready"      # written only after generation fully succeeds
_LOCK = _TUT / ".fixtures.lock"        # cross-worker mutex (pytest-xdist)


def _generate_shared_fixtures():
    """Reproduce the data the DataSet/Protocols tutorials save to disk.

    Uses the same model pack and seed as ``objects/DataSet`` so the generated
    files match what a reader running the tutorials in order would produce.
    """
    import pygsti
    from pygsti.modelpacks import smq1Q_XYI

    gst_data_dir = _TUT / "Example_GST_Data"

    # --- data saved by the DataSet tutorial -------------------------------------
    depol = smq1Q_XYI.target_model().depolarize(op_noise=0.1)
    edesign = smq1Q_XYI.create_gst_experiment_design(max_max_length=128)
    circuits = edesign.all_circuits_needing_data
    ds = pygsti.data.simulate_data(depol, circuits, num_samples=1000,
                                   sample_error='binomial', seed=100)
    ds_low = pygsti.data.simulate_data(depol, circuits, num_samples=50,
                                       sample_error='binomial', seed=100)
    pygsti.protocols.ProtocolData(edesign, ds).write(str(gst_data_dir))
    pygsti.io.write_dataset(str(_TUT / "Example_Dataset.txt"), ds,
                            outcome_label_order=['0', '1'])
    pygsti.io.write_dataset(str(_TUT / "Example_Dataset_LowCnts.txt"), ds_low)

    # --- GateSetTomography results read back by the Results tutorial -------------
    # ``disable_checkpointing`` because this runs from a sessionstart hook, so the
    # default checkpoint dir would be resolved against pytest's invocation directory
    # (the repo root) rather than anything under docs/.  We only want the final
    # results here; nothing resumes this run, so the checkpoints are pure litter.
    data = pygsti.io.read_data_from_dir(str(gst_data_dir))
    results = pygsti.protocols.GateSetTomography(
        smq1Q_XYI.target_model("full TP"), verbosity=0).run(
            data, disable_checkpointing=True)
    results.write()  # -> Example_GST_Data/results/GateSetTomography


def pytest_sessionstart(session):
    """Ensure the cross-notebook input fixtures exist before any notebook runs.

    Implemented as a ``pytest_sessionstart`` hook (rather than an autouse
    fixture) because nbval's cell items don't reliably trigger autouse fixtures.
    xdist-safe: the first worker to grab the lock generates the fixtures; the
    others wait for the ``.fixtures_ready`` sentinel. Skipped for ``--collect-only``.
    """
    if getattr(session.config.option, "collectonly", False):
        return
    _TUT.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 600  # generation itself takes only a few seconds
    while not _READY.exists():
        try:
            fd = os.open(str(_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() > deadline:
                raise RuntimeError("Timed out waiting for shared tutorial fixtures.")
            time.sleep(0.2)
            continue
        try:
            if not _READY.exists():
                _generate_shared_fixtures()
                _READY.write_text("ok\n")
        finally:
            os.close(fd)
            os.unlink(_LOCK)
