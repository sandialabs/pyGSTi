"""
Tests for per-cell, basis-driven leakage-metric column selection in reports.

These cover the ``basis_aware_display`` helper (which chooses subspace vs. full-space
gates-vs-target columns per switchboard cell from each model's basis and the interactive
"Metrics" switch) and the defense-in-depth hardening that keeps a leakage display from
aborting a report when applied to a non-leakage model.
"""
import numpy as np
import pytest

from pygsti.modelpacks import smq1Q_XYI
from pygsti.leakage import leaky_qubit_model_from_pspec
from pygsti.report.workspace import Workspace, NotApplicable
from pygsti.report.factory import basis_aware_display
from pygsti.report import reportables as _reportables
from pygsti.baseobjs.basis import Basis
from ..util import BaseCase


_ORDINARY = ('inf', 'agi', 'geni', 'trace', 'diamond', 'nuinf', 'nuagi')
_LEAKAGE = ('sub-inf', 'sub-trace', 'sub-diamond', 'plf-sub-diamond',
            'leak-rate-max', 'leak-rate-min', 'seep-rate')


def _leaky_model():
    return leaky_qubit_model_from_pspec(smq1Q_XYI.processor_spec())


def _plain_model():
    return smq1Q_XYI.target_model()


def _mixed_switchboard():
    """A master-switchboard-like object: 2 estimates (leaky l2p1, plain pp), 1 gauge-opt."""
    ws = Workspace()
    sb = ws.Switchboard(
        ["Dataset", "Estimate", "Gauge-Opt", "max(L)", "Metrics"],
        [["ds0"], ["est_leaky", "est_plain"], ["go0"], ["0"], ["Subspace", "Full-space"]],
        ["dropdown", "dropdown", "buttons", "slider", "buttons"],
        [0, 0, 0, 0, 0],
        show=[False, True, False, False, True],
    )
    sb.metric_space_switch_index = 4
    sb.add("mdl_final", (0, 1, 2))
    sb.mdl_final[0, 0, 0] = _leaky_model()   # leaky estimate -> l2p1 basis
    sb.mdl_final[0, 1, 0] = _plain_model()   # plain estimate -> pp basis
    return sb


class BasisAwareDisplayTester(BaseCase):

    def test_model_bases_as_expected(self):
        self.assertTrue(_leaky_model().basis.implies_leakage_modeling)
        self.assertFalse(_plain_model().basis.implies_leakage_modeling)

    def test_per_cell_selection(self):
        sb = _mixed_switchboard()
        sv = basis_aware_display(sb, 'gv', _ORDINARY, _LEAKAGE)

        # deps == (Dataset, Estimate, Gauge-Opt, Metrics) -> shape (1, 2, 1, 2)
        self.assertEqual(sv.base.shape, (1, 2, 1, 2))

        # ms==0 is "Subspace", ms==1 is "Full-space".
        self.assertEqual(sv.base[0, 0, 0, 0], _LEAKAGE)    # leaky + Subspace  -> leakage cols
        self.assertEqual(sv.base[0, 0, 0, 1], _ORDINARY)   # leaky + Full-space -> full-space
        self.assertEqual(sv.base[0, 1, 0, 0], _ORDINARY)   # plain + Subspace  -> full-space
        self.assertEqual(sv.base[0, 1, 0, 1], _ORDINARY)   # plain + Full-space -> full-space

    def test_idempotent_by_name(self):
        sb = _mixed_switchboard()
        first = basis_aware_display(sb, 'gv', _ORDINARY, _LEAKAGE)
        second = basis_aware_display(sb, 'gv', _ORDINARY, _LEAKAGE)
        self.assertIs(first, second)

    def test_not_applicable_cell_gets_ordinary(self):
        ws = Workspace()
        sb = ws.Switchboard(
            ["Dataset", "Estimate", "Gauge-Opt", "max(L)", "Metrics"],
            [["ds0"], ["est0"], ["go0"], ["0"], ["Subspace", "Full-space"]],
            ["dropdown", "dropdown", "buttons", "slider", "buttons"],
            [0, 0, 0, 0, 0], show=[False, False, False, False, True],
        )
        sb.metric_space_switch_index = 4
        sb.add("mdl_final", (0, 1, 2))
        sb.mdl_final[0, 0, 0] = NotApplicable(ws)
        sv = basis_aware_display(sb, 'gv', _ORDINARY, _LEAKAGE)
        self.assertEqual(sv.base[0, 0, 0, 0], _ORDINARY)
        self.assertEqual(sv.base[0, 0, 0, 1], _ORDINARY)


class LeakSeepHardeningTester(BaseCase):
    """Leak/seep reductions must degrade to NaN (not raise) on a non-leakage basis."""

    def setUp(self):
        self.pp = Basis.cast('pp', 4)
        self.op = np.eye(4)  # identity superop in pp basis (no leakage profile)

    def test_leakrate_max_nan_on_pp(self):
        val = _reportables.pergate_leakrate_max(self.op, None, self.pp)
        self.assertTrue(np.isnan(val))

    def test_leakrate_min_nan_on_pp(self):
        val = _reportables.pergate_leakrate_min(self.op, None, self.pp)
        self.assertTrue(np.isnan(val))

    def test_seeprate_nan_on_pp(self):
        val = _reportables.pergate_seeprate(self.op, None, self.pp)
        self.assertTrue(np.isnan(val))


class GatesVsTargetTableFallbackTester(BaseCase):
    """A leakage display forced onto a non-leakage model yields NaN cells, not a crash."""

    def test_leakage_display_on_plain_model_does_not_raise(self):
        ws = Workspace()
        model = _plain_model()
        target = _plain_model()
        # Building the table triggers per-cell evaluation via switched_compute; the
        # plf-sub-diamond / leak / seep columns would otherwise raise on a pp basis.
        tbl = ws.GatesVsTargetTable(model, target, None, display=_LEAKAGE)
        self.assertEqual(len(tbl.tables), 1)
        # Rendering exercises the built cells end-to-end.
        tbl.render('html')


class MixedReportTester(BaseCase):
    """
    End-to-end: a report spanning a leakage (l2p1) dataset and a plain (pp) dataset must
    build (this is the case that previously crashed under leakage_modeling=True) and expose
    the interactive Subspace/Full-space "Metrics" switch. Uses ModelTest (no GST fit) to
    keep the fixture fast.
    """

    def _mixed_results(self):
        from pygsti.data import simulate_data
        from pygsti.protocols import ModelTest, ProtocolData
        mp = smq1Q_XYI
        # max_max_length=2 gives >1 max-length so the goodness section renders its
        # max(L) switchboard *view* -- the path that regressed when the master
        # switchboard grew a 5th ("Metrics") switch (fixed-length view mask).
        ed = mp.create_gst_experiment_design(max_max_length=2)
        tm_leaky = leaky_qubit_model_from_pspec(mp.processor_spec())
        ds_leaky = simulate_data(tm_leaky, ed.all_circuits_needing_data, num_samples=1000, seed=1)
        res_leaky = ModelTest(tm_leaky, target_model=tm_leaky).run(ProtocolData(ed, ds_leaky))
        tm_plain = mp.target_model()
        ds_plain = simulate_data(tm_plain, ed.all_circuits_needing_data, num_samples=1000, seed=2)
        res_plain = ModelTest(tm_plain, target_model=tm_plain).run(ProtocolData(ed, ds_plain))
        return {'leaky': res_leaky, 'plain': res_plain}

    @pytest.mark.filterwarnings("ignore::pygsti.tools.exceptions.OverparameterizationWarning")
    def test_mixed_basis_report_builds_with_metrics_switch(self):
        import os
        import glob
        from tempfile import TemporaryDirectory
        from pygsti.report import construct_standard_report

        results = self._mixed_results()
        adv = {'skip_sections': ['colorbox', 'input', 'meta', 'help']}
        report = construct_standard_report(results, title='mixed', advanced_options=adv, verbosity=0)
        with TemporaryDirectory() as d:
            report.write_html(d)
            html = ''
            for fn in glob.glob(os.path.join(d, '**', '*.html'), recursive=True):
                with open(fn, encoding='utf-8', errors='ignore') as fh:
                    html += fh.read()
        # The interactive Metrics switch and both positions are rendered.
        self.assertIn('Metrics', html)
        self.assertIn('Subspace', html)
        self.assertIn('Full-space', html)
