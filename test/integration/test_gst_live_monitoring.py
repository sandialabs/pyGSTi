"""
Integration tests for the goodness-of-fit checkpoint augmentation (Phase 1)
and the gauge-invariant metric extraction helpers in
`pygsti.report.livemetrics` (Phase 2) that together support a "live" GST
monitoring tool.

These are integration (rather than unit) tests: they actually run small,
real `GateSetTomography`/`StandardGST` fits to disk and then check the
checkpoint files those fits produce, rather than exercising the new code
against hand-built fixtures in isolation.
"""
import json
import os
import tempfile
import unittest

import numpy as np
import scipy.stats as stats

from pygsti.data import simulate_data
from pygsti.modelpacks import smq1Q_XYI
from pygsti.protocols import gst
from pygsti.protocols.protocol import ProtocolData
from pygsti.report import livemetrics
from pygsti.tools import matrixtools


def _sorted_by_matching(evals_a, evals_b):
    """
    Sort two eigenvalue arrays so they can be compared element-wise, robust to
    the numerically-unstable ordering `numpy.sort_complex` can produce for
    near-degenerate complex-conjugate pairs (where a tiny numerical
    perturbation to the real part can flip which of the pair sorts first).
    Uses a greedy nearest-neighbor matching instead of a naive sort.
    """
    evals_a = list(evals_a)
    remaining = list(evals_b)
    matched_b = []
    for a in evals_a:
        idx = int(np.argmin([abs(a - b) for b in remaining]))
        matched_b.append(remaining.pop(idx))
    return np.array(evals_a), np.array(matched_b)


class GSTCheckpointGoodnessOfFitTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.gst_design = smq1Q_XYI.create_gst_experiment_design(max_max_length=2)
        cls.mdl_target = smq1Q_XYI.target_model()
        cls.mdl_datagen = cls.mdl_target.depolarize(op_noise=0.05, spam_noise=0.025)
        ds = simulate_data(cls.mdl_datagen, cls.gst_design.all_circuits_needing_data,
                            1000, sample_error='none')
        cls.gst_data = ProtocolData(cls.gst_design, ds)

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.checkpoint_prefix = os.path.join(self.tmpdir.name, 'testGST')

    def tearDown(self):
        self.tmpdir.cleanup()

    def _run_and_get_checkpoint_paths(self):
        proto = gst.GateSetTomography(self.mdl_target.copy(), 'stdgaugeopt', name="testGST")
        proto.run(self.gst_data, checkpoint_path=self.checkpoint_prefix)
        n_iters = len(self.gst_data.edesign.circuit_lists)
        paths = [f'{self.checkpoint_prefix}_iteration_{i}.json' for i in range(n_iters)]
        for p in paths:
            self.assertTrue(os.path.isfile(p))
        return paths

    def test_checkpoint_contains_per_iter_gof(self):
        paths = self._run_and_get_checkpoint_paths()

        # Each successive checkpoint file should have one more per_iter_gof
        # entry than the last (one per completed iteration).
        for i, p in enumerate(paths):
            with open(p, 'r') as f:
                state = json.load(f)
            self.assertIn('per_iter_gof', state)
            self.assertEqual(len(state['per_iter_gof']), i + 1)

            # The scalars for *this* iteration should all be populated (not None)
            # for an ordinary logl-based GateSetTomography run.
            latest_entry = state['per_iter_gof'][-1]
            self.assertIsNotNone(latest_entry['chi2k_distributed_qty'])
            self.assertIsNotNone(latest_entry['n_data_params'])
            self.assertIsNotNone(latest_entry['n_model_params'])
            self.assertGreater(latest_entry['n_data_params'], 0)
            self.assertGreater(latest_entry['n_model_params'], 0)

    def test_checkpoint_roundtrip_preserves_per_iter_gof(self):
        paths = self._run_and_get_checkpoint_paths()
        last_checkpoint = gst.GateSetTomographyCheckpoint.read(paths[-1])
        self.assertEqual(len(last_checkpoint.per_iter_gof), len(paths))

        with open(paths[-1], 'r') as f:
            raw_state = json.load(f)
        self.assertEqual(last_checkpoint.per_iter_gof, raw_state['per_iter_gof'])

    def test_old_style_checkpoint_without_per_iter_gof_still_loads(self):
        # Simulate a checkpoint file written before this feature existed
        # (i.e. with no 'per_iter_gof' key at all) and confirm it still
        # loads, with per_iter_gof defaulting to an empty list.
        paths = self._run_and_get_checkpoint_paths()
        with open(paths[0], 'r') as f:
            state = json.load(f)
        del state['per_iter_gof']

        old_style_path = os.path.join(self.tmpdir.name, 'old_style.json')
        with open(old_style_path, 'w') as f:
            json.dump(state, f)

        loaded = gst.GateSetTomographyCheckpoint.read(old_style_path)
        self.assertEqual(loaded.per_iter_gof, [])

    def test_livemetrics_goodness_of_fit_history(self):
        paths = self._run_and_get_checkpoint_paths()
        state = livemetrics.load_checkpoint_state(paths[-1])
        history = livemetrics.goodness_of_fit_history(state)

        self.assertEqual(len(history), len(paths))
        for i, entry in enumerate(history):
            self.assertEqual(entry['iteration'], i)
            self.assertIsNotNone(entry['degrees_of_freedom'])
            self.assertIsNotNone(entry['pvalue'])
            # Cross-check the p-value computation directly against scipy.
            expected_pvalue = 1.0 - stats.chi2.cdf(
                entry['chi2k_distributed_qty'], entry['degrees_of_freedom'])
            self.assertAlmostEqual(entry['pvalue'], expected_pvalue)

    def test_livemetrics_handles_missing_per_iter_gof(self):
        # An empty/absent 'per_iter_gof' key should yield an empty history,
        # not raise.
        history = livemetrics.goodness_of_fit_history({})
        self.assertEqual(history, [])

    def test_livemetrics_latest_model_and_eigenvalues(self):
        paths = self._run_and_get_checkpoint_paths()
        state = livemetrics.load_checkpoint_state(paths[-1])

        mdl = livemetrics.latest_model_from_state(state)
        self.assertIsNotNone(mdl)

        evals = livemetrics.gate_eigenvalues(mdl)
        self.assertEqual(set(evals.keys()), set(mdl.operations.keys()))

        # Cross check against directly computing eigenvalues off the model's
        # dense operation matrices.
        for lbl, op in mdl.operations.items():
            expected = matrixtools.eigenvalues(np.asarray(op.to_dense("HilbertSchmidt")))
            got, exp = _sorted_by_matching(evals[lbl], expected)
            np.testing.assert_allclose(got, exp)

    def test_livemetrics_eigenvalues_are_gauge_invariant(self):
        # Sanity check the core mathematical claim underpinning this feature:
        # gate eigenvalues are unchanged by a gauge (similarity) transform.
        paths = self._run_and_get_checkpoint_paths()
        state = livemetrics.load_checkpoint_state(paths[-1])
        mdl = livemetrics.latest_model_from_state(state)
        evals_before = livemetrics.gate_eigenvalues(mdl)

        # Apply an arbitrary invertible "gauge transform" directly to the
        # dense operation matrices and confirm eigenvalues are unaffected.
        rng = np.random.RandomState(1234)
        dim = mdl.operations[list(mdl.operations.keys())[0]].to_dense("HilbertSchmidt").shape[0]
        m = rng.randn(dim, dim) + 1j * 0  # real invertible matrix
        m_inv = np.linalg.inv(m)

        for lbl, op in mdl.operations.items():
            dense = np.asarray(op.to_dense("HilbertSchmidt"))
            transformed = m @ dense @ m_inv
            evals_after = matrixtools.eigenvalues(transformed)
            before, after = _sorted_by_matching(evals_before[lbl], evals_after)
            np.testing.assert_allclose(before, after, atol=1e-6)

    def test_livemetrics_extract_live_metrics_convenience(self):
        paths = self._run_and_get_checkpoint_paths()
        state = livemetrics.load_checkpoint_state(paths[-1])
        result = livemetrics.extract_live_metrics(state)

        self.assertEqual(result['last_completed_iter'], state['last_completed_iter'])
        self.assertEqual(len(result['gof_history']), len(paths))
        self.assertEqual(set(result['eigenvalues'].keys()),
                         set(livemetrics.latest_model_from_state(state).operations.keys()))

    def test_standard_gst_checkpoint_unwrapping(self):
        # StandardGST fits multiple "modes" as child protocols. Because
        # ProtocolCheckpoint.write() delegates to a checkpoint's parent when
        # one is set, every per-mode checkpoint file written in the course of
        # a StandardGST run actually contains the *entire* StandardGSTCheckpoint
        # (all modes), with each mode's own GateSetTomographyCheckpoint nested
        # under state['children'][mode]. Confirm livemetrics transparently
        # unwraps this.
        proto = gst.StandardGST(modes=['full TP'], name='StdGST')
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, 'StandardGST')
            proto.run(self.gst_data, checkpoint_path=prefix)

            n_iters = len(self.gst_data.edesign.circuit_lists)
            last_path = os.path.join(tmpdir, f'StandardGST_full_TP_iteration_{n_iters - 1}.json')
            self.assertTrue(os.path.isfile(last_path))

            state = livemetrics.load_checkpoint_state(last_path)
            # Confirm this is indeed the "wrapped" StandardGSTCheckpoint shape,
            # not a bare GateSetTomographyCheckpoint - otherwise this test isn't
            # actually exercising the unwrapping logic.
            self.assertIn('children', state)
            self.assertIn('full TP', state['children'])

            # No mode specified: should auto-select the sole GST mode.
            history = livemetrics.goodness_of_fit_history(state)
            self.assertEqual(len(history), n_iters)

            result = livemetrics.extract_live_metrics(state)
            self.assertEqual(result['last_completed_iter'], n_iters - 1)
            self.assertGreater(len(result['eigenvalues']), 0)

            # Explicit mode argument should also work.
            history_explicit = livemetrics.goodness_of_fit_history(state, mode='full TP')
            self.assertEqual(history, history_explicit)

            # A nonexistent/incorrect mode should raise a clear error rather
            # than silently returning wrong data.
            with self.assertRaises(ValueError):
                livemetrics.goodness_of_fit_history(state, mode='not a real mode')

    def test_standard_gst_checkpoint_multi_mode_requires_explicit_mode(self):
        proto = gst.StandardGST(modes=['full TP', 'CPTPLND'], name='StdGST')
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, 'StandardGST')
            proto.run(self.gst_data, checkpoint_path=prefix)

            n_iters = len(self.gst_data.edesign.circuit_lists)
            last_path = os.path.join(tmpdir, f'StandardGST_full_TP_iteration_{n_iters - 1}.json')
            state = livemetrics.load_checkpoint_state(last_path)

            # Ambiguous: two GST modes present, none specified.
            with self.assertRaises(ValueError):
                livemetrics.goodness_of_fit_history(state)

            # Disambiguating with an explicit mode should work fine.
            history = livemetrics.goodness_of_fit_history(state, mode='full TP')
            self.assertEqual(len(history), n_iters)

    def test_livemetrics_only_deserializes_latest_model(self):
        # Performance-sensitive contract: latest_model_from_state should not
        # need to deserialize every model in mdl_list. We can't easily spy on
        # internal calls without heavier mocking machinery, but we can at
        # least confirm it returns the *correct* (i.e. last) model, and that
        # calling it against a state with a deliberately-corrupted earlier
        # entry in mdl_list still succeeds (proving the earlier entries are
        # never touched).
        paths = self._run_and_get_checkpoint_paths()
        state = livemetrics.load_checkpoint_state(paths[-1])
        if len(state['mdl_list']) > 1:
            state['mdl_list'][0] = {'this is': 'not a valid model serialization'}
        mdl = livemetrics.latest_model_from_state(state)
        self.assertIsNotNone(mdl)
