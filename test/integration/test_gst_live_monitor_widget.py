"""
Integration tests for `pygsti.report.livemonitor.LiveGSTMonitor` (Phase 3):
the live-updating notebook widget/poller built on top of the checkpoint
goodness-of-fit augmentation (Phase 1) and gauge-invariant metric extraction
(Phase 2, `pygsti.report.livemetrics`).

These tests actually run small, real `GateSetTomography`/`StandardGST` fits
(some as child subprocesses, to exercise the "spawn-and-watch" code path),
so they are integration- rather than unit-level tests.
"""
import json
import os
import tempfile
import time
import unittest
import unittest.mock

import pytest

from pygsti.data import simulate_data
from pygsti.modelpacks import smq1Q_XYI
from pygsti.protocols import gst, modeltest
from pygsti.protocols.protocol import ProtocolData

try:
    from pygsti.report.livemonitor import LiveGSTMonitor, LiveGSTMonitorError, _build_figure_widget
    _build_figure_widget()
    _WIDGET_BACKEND_AVAILABLE = True
except Exception:
    _WIDGET_BACKEND_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not _WIDGET_BACKEND_AVAILABLE,
    reason="LiveGSTMonitor requires a plotly FigureWidget backend "
          "('anywidget' for plotly>=6, or 'ipywidgets' for older plotly).")


class BaseLiveMonitorData(object):

    @classmethod
    def setUpClass(cls):
        cls.gst_design = smq1Q_XYI.create_gst_experiment_design(max_max_length=2)
        cls.mdl_target = smq1Q_XYI.target_model()
        cls.mdl_datagen = cls.mdl_target.depolarize(op_noise=0.05, spam_noise=0.025)
        ds = simulate_data(cls.mdl_datagen, cls.gst_design.all_circuits_needing_data,
                           1000, sample_error='none')
        cls.gst_data = ProtocolData(cls.gst_design, ds)
        cls.n_iters = len(cls.gst_design.circuit_lists)


class LiveGSTMonitorWatchTester(BaseLiveMonitorData, unittest.TestCase):
    """Tests for LiveGSTMonitor.watch(), attaching to already-written checkpoints."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.checkpoint_prefix = os.path.join(self.tmpdir.name, 'testGST')

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_watch_against_completed_run(self):
        proto = gst.GateSetTomography(self.mdl_target.copy(), 'stdgaugeopt', name="testGST")
        proto.run(self.gst_data, checkpoint_path=self.checkpoint_prefix)

        monitor = LiveGSTMonitor(self.checkpoint_prefix, poll_interval=0.05,
                                 n_iterations=self.n_iters)
        last_iter = monitor.watch(timeout=5)

        self.assertEqual(last_iter, self.n_iters - 1)
        self.assertIsNotNone(monitor._fig)

        gof_trace = monitor._fig.data[0]
        self.assertEqual(list(gof_trace.x), list(range(self.n_iters)))
        self.assertEqual(len(gof_trace.y), self.n_iters)

        # First two traces are the p-value trace and the unit-circle
        # reference; everything after that is a per-gate eigenvalue trace.
        eigenvalue_trace_names = {t.name for t in monitor._fig.data[2:]}
        self.assertEqual(eigenvalue_trace_names, {str(lbl) for lbl in self.mdl_target.operations.keys()})

        self.assertIn('PROVISIONAL', monitor._fig.layout.title.text)
        self.assertIn(f'iteration {self.n_iters - 1}', monitor._fig.layout.title.text)

    def test_watch_returns_negative_one_with_no_checkpoints(self):
        # No checkpoint files exist at all under this prefix.
        monitor = LiveGSTMonitor(self.checkpoint_prefix, poll_interval=0.05)
        last_iter = monitor.watch(timeout=0.2)
        self.assertEqual(last_iter, -1)

    def test_watch_stops_early_via_timeout_without_n_iterations(self):
        # Only iteration 0's checkpoint exists; without n_iterations given,
        # watch() should poll until the timeout rather than hanging forever.
        proto = gst.GateSetTomography(self.mdl_target.copy(), 'stdgaugeopt', name="testGST")
        proto.run(self.gst_data, checkpoint_path=self.checkpoint_prefix)

        # Delete the last iteration's checkpoint to simulate a run that's
        # "still going" after iteration 0.
        last_path = f'{self.checkpoint_prefix}_iteration_{self.n_iters - 1}.json'
        os.remove(last_path)

        monitor = LiveGSTMonitor(self.checkpoint_prefix, poll_interval=0.05)  # no n_iterations
        import time
        start = time.time()
        last_iter = monitor.watch(timeout=0.3)
        elapsed = time.time() - start

        self.assertEqual(last_iter, 0)
        self.assertGreaterEqual(elapsed, 0.3)

    def test_safe_read_falls_back_when_highest_checkpoint_is_corrupted(self):
        proto = gst.GateSetTomography(self.mdl_target.copy(), 'stdgaugeopt', name="testGST")
        proto.run(self.gst_data, checkpoint_path=self.checkpoint_prefix)

        # Simulate observing the highest-index checkpoint file mid-write
        # (i.e. containing invalid/truncated JSON).
        last_path = f'{self.checkpoint_prefix}_iteration_{self.n_iters - 1}.json'
        with open(last_path, 'w') as f:
            f.write('{"mdl_list": [invalid json truncated mid-writ')

        monitor = LiveGSTMonitor(self.checkpoint_prefix, poll_interval=0.05)
        index, state = monitor._read_latest_safe_state()

        # Should fall back to the previous (guaranteed-complete) iteration
        # rather than raising or returning nothing.
        self.assertEqual(index, self.n_iters - 2)
        self.assertIsNotNone(state)

    def test_standard_gst_mode_disambiguation(self):
        proto = gst.StandardGST(modes=['full TP', 'CPTPLND'], name='StdGST')
        proto.run(self.gst_data, checkpoint_path=self.checkpoint_prefix)

        prefix_for_mode = f'{self.checkpoint_prefix}_full_TP'
        monitor = LiveGSTMonitor(prefix_for_mode, mode='full TP', poll_interval=0.05,
                                 n_iterations=self.n_iters)
        last_iter = monitor.watch(timeout=5)
        self.assertEqual(last_iter, self.n_iters - 1)


class LiveGSTMonitorRunTester(BaseLiveMonitorData, unittest.TestCase):
    """Tests for LiveGSTMonitor.run(), the spawn-and-watch convenience path."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.checkpoint_prefix = os.path.join(self.tmpdir.name, 'testGST')

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run_spawns_fit_and_returns_final_results(self):
        proto = gst.GateSetTomography(self.mdl_target.copy(), 'stdgaugeopt', name="testGST")
        monitor = LiveGSTMonitor(self.checkpoint_prefix, poll_interval=0.1, n_iterations=self.n_iters)

        results = monitor.run(proto, self.gst_data, timeout=90)

        self.assertIsNotNone(results)
        self.assertIn('testGST', results.estimates)
        final_model = results.estimates['testGST'].models['stdgaugeopt']
        self.assertIsNotNone(final_model)

        # The monitor should have observed the run to completion.
        self.assertEqual(monitor._last_rendered_iter, self.n_iters - 1)

        # Checkpoint files should persist at the requested prefix even though
        # the run happened in a temporary, now-deleted, working directory.
        for i in range(self.n_iters):
            self.assertTrue(os.path.isfile(f'{self.checkpoint_prefix}_iteration_{i}.json'))

    def test_run_rejects_reserved_run_kwargs(self):
        proto = gst.GateSetTomography(self.mdl_target.copy(), 'stdgaugeopt', name="testGST")
        monitor = LiveGSTMonitor(self.checkpoint_prefix)

        for bad_kwarg in ('checkpoint_path', 'checkpoint', 'disable_checkpointing'):
            with self.assertRaises(ValueError):
                monitor.run(proto, self.gst_data, run_kwargs={bad_kwarg: 'anything'})

    def test_run_terminates_child_process_on_timeout(self):
        # Regression test: LiveGSTMonitor.run() must not orphan its child
        # process when a timeout is hit - the child depends on a temporary
        # working directory that gets deleted as soon as run() returns, so
        # leaving the child running would just cause it to fail later with
        # no way for the caller to know. See the "Notes" section of
        # LiveGSTMonitor.run's docstring.
        import pygsti.report.livemonitor as livemonitor_mod

        proto = gst.GateSetTomography(self.mdl_target.copy(), 'stdgaugeopt', name="testGST")
        monitor = LiveGSTMonitor(self.checkpoint_prefix, poll_interval=0.05, n_iterations=self.n_iters)

        captured = {}
        real_popen = livemonitor_mod._subprocess.Popen

        def spying_popen(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            captured['proc'] = proc
            return proc

        with unittest.mock.patch.object(livemonitor_mod._subprocess, 'Popen', spying_popen):
            with self.assertWarns(UserWarning):
                result = monitor.run(proto, self.gst_data, timeout=0.001)

        self.assertIsNone(result)
        self.assertIn('proc', captured)
        # Give the OS a brief moment to finish reaping; poll() returning
        # non-None means the process has actually exited (not left running).
        time.sleep(1.0)
        self.assertIsNotNone(
            captured['proc'].poll(),
            "child process should have been terminated on timeout, not left running")

    def test_run_terminates_child_process_on_keyboard_interrupt(self):
        # Same regression test as above, but for the KeyboardInterrupt path
        # (e.g. a user hitting the notebook "stop" button) rather than the
        # timeout path.
        import pygsti.report.livemonitor as livemonitor_mod

        proto = gst.GateSetTomography(self.mdl_target.copy(), 'stdgaugeopt', name="testGST")
        monitor = LiveGSTMonitor(self.checkpoint_prefix, poll_interval=0.05, n_iterations=self.n_iters)

        captured = {}
        real_popen = livemonitor_mod._subprocess.Popen

        def spying_popen(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            captured['proc'] = proc
            return proc

        real_sleep = livemonitor_mod._time.sleep
        call_count = {'n': 0}

        def interrupting_sleep(seconds):
            call_count['n'] += 1
            if call_count['n'] == 1:
                raise KeyboardInterrupt()
            real_sleep(seconds)

        with unittest.mock.patch.object(livemonitor_mod._subprocess, 'Popen', spying_popen), \
            unittest.mock.patch.object(livemonitor_mod._time, 'sleep', interrupting_sleep):
            with self.assertWarns(UserWarning):
                with self.assertRaises(KeyboardInterrupt):
                    monitor.run(proto, self.gst_data, timeout=None)

        self.assertIn('proc', captured)
        time.sleep(1.0)
        self.assertIsNotNone(
            captured['proc'].poll(),
            "child process should have been terminated on interrupt, not left running")


class ProtocolRunLiveTester(BaseLiveMonitorData, unittest.TestCase):
    """Tests for the Protocol.run_live(...) convenience wrapper (Phase 4)."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.checkpoint_prefix = os.path.join(self.tmpdir.name, 'testGST')

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run_live_gate_set_tomography(self):
        proto = gst.GateSetTomography(self.mdl_target.copy(), 'stdgaugeopt', name="testGST")
        results = proto.run_live(self.gst_data, checkpoint_path=self.checkpoint_prefix,
                                 poll_interval=0.1, timeout=90)

        self.assertIsNotNone(results)
        self.assertIn('testGST', results.estimates)
        for i in range(self.n_iters):
            self.assertTrue(os.path.isfile(f'{self.checkpoint_prefix}_iteration_{i}.json'))

    def test_run_live_standard_gst_with_mode(self):
        proto = gst.StandardGST(modes=['full TP'], name='StdGST')
        results = proto.run_live(self.gst_data, checkpoint_path=self.checkpoint_prefix,
                                 mode='full TP', poll_interval=0.1, timeout=90)

        self.assertIsNotNone(results)
        self.assertIn('full TP', results.estimates)

    def test_run_live_rejects_linear_gst(self):
        # LinearGateSetTomography is a single non-iterative fit - it accepts
        # checkpoint-related run() kwargs syntactically, but has no
        # meaningful per-depth progression, so run_live should refuse it
        # with a clear error rather than doing something silently useless.
        proto = gst.LinearGateSetTomography(self.mdl_target.copy(), 'stdgaugeopt', name="testLGST")
        with self.assertRaises(TypeError):
            proto.run_live(self.gst_data)

    def test_run_live_rejects_model_test(self):
        # ModelTest also accepts checkpoint_path/disable_checkpointing kwargs,
        # but writes a checkpoint with an incompatible schema (objfn_vals /
        # chi2k_distributed_vals, no mdl_list) - run_live must not silently
        # produce an empty/misleading dashboard for it.
        proto = modeltest.ModelTest(self.mdl_target.copy(), name="testMT")
        with self.assertRaises(TypeError):
            proto.run_live(self.gst_data)

    def test_run_live_default_checkpoint_path(self):
        proto = gst.GateSetTomography(self.mdl_target.copy(), 'stdgaugeopt', name="defaultPathTestXYZ")
        cwd = os.getcwd()
        os.chdir(self.tmpdir.name)
        try:
            results = proto.run_live(self.gst_data, poll_interval=0.1, timeout=90)
        finally:
            os.chdir(cwd)

        self.assertIsNotNone(results)
        default_dir = os.path.join(self.tmpdir.name, 'gst_checkpoints')
        self.assertTrue(os.path.isdir(default_dir))
        self.assertTrue(any('defaultPathTestXYZ_iteration' in f for f in os.listdir(default_dir)))
