import numpy as np
import pytest

import pygsti
import pygsti.tools.basistools as bt
import pygsti.tools.optools as ot
from pygsti.baseobjs import Basis
from pygsti.circuits import Circuit
from pygsti.modelmembers.instruments import Instrument
from pygsti.modelpacks import smq1Q_XZ
from pygsti.protocols import (
    GateSetTomography, GSTInitialModel, ModelTest, ProtocolData, StandardGSTDesign,
    refit_instruments_cptplnd
)
from pygsti.tools.jamiolkowski import fast_jamiolkowski_iso_std
from ..util import BaseCase

QL = (0,)
INST_LBL = ('Iz', 0)


def _sk(mat, basis):
    return bt.stdmx_to_vec(mat, basis).ravel().real


def _ideal_projective_members(basis):
    members = {}
    for k in range(2):
        P = np.zeros((2, 2))
        P[k, k] = 1.0
        members[f'p{k}'] = bt.change_basis(np.kron(P, P.conj()), 'std', basis).real
    return members


def _build_target(with_instrument=True):
    mdl = smq1Q_XZ.target_model(qubit_labels=QL).copy()
    mdl[[]] = np.eye(4)   # global idle, needed by the empty germ
    if with_instrument:
        mdl[INST_LBL] = Instrument(_ideal_projective_members(mdl.basis))
    return mdl


def _build_truth(target):
    """A noisy version of the target whose instrument members are FULL RANK:
    slightly depolarizing gates, and a weak (interior-effect) measurement
    followed by a depolarizing kick.  This is the regime where a target-seeded
    Lindblad instrument fit fails (rank cap) but the projection-seeded refit
    must reach TP-fit quality."""
    truth = target.copy()
    basis = truth.basis
    D = np.eye(4)
    D[1:, 1:] *= 0.98
    for lbl in list(truth.operations.keys()):
        truth[lbl] = D @ np.asarray(truth[lbl].to_dense())
    E0 = np.diag([0.92, 0.08]).astype(complex)
    members = {'p0': D @ ot.rootconj_superop(_sk(E0, basis), basis),
               'p1': D @ ot.rootconj_superop(_sk(np.eye(2) - E0, basis), basis)}
    truth[INST_LBL] = Instrument(members)
    return truth


def _build_edesign(target):
    germs = smq1Q_XZ.germs(qubit_labels=QL)[:4]
    germs.append(Circuit([[]], line_labels=QL))
    inst_germ = Circuit([INST_LBL])
    germs.append(inst_germ)
    prep_fids = smq1Q_XZ.prep_fiducials(qubit_labels=QL)
    meas_fids = smq1Q_XZ.meas_fiducials(qubit_labels=QL)
    prep_fids[4] = Circuit('Gxpi2:0Gxpi2:0Gxpi2:0@(0)')
    prep_fids[5] = Circuit('Gxpi2:0Gzpi2:0Gxpi2:0Gxpi2:0@(0)')
    meas_fids[4] = Circuit('Gxpi2:0Gxpi2:0Gxpi2:0@(0)')
    meas_fids[5] = Circuit('Gxpi2:0Gxpi2:0Gzpi2:0Gxpi2:0@(0)')
    return StandardGSTDesign(target, prep_fids, meas_fids, germs, [1],
                             germ_length_limits={inst_germ: 1})


def _min_member_choi_eig(model):
    worst = np.inf
    for inst in model.instruments.values():
        for m in inst.values():
            choi = fast_jamiolkowski_iso_std(m.to_dense(), model.basis)
            worst = min(worst, np.linalg.eigvalsh(choi).min())
    return worst


class RefitGuardTester(BaseCase):
    """The pre-flight guards, exercised without running any optimization
    (ModelTest evaluates a fixed model, so building results is cheap)."""

    def setUp(self):
        self.target = _build_target()
        self.edesign = _build_edesign(self.target)
        self.data = ProtocolData(
            self.edesign,
            pygsti.data.simulate_data(_build_truth(self.target),
                                      self.edesign.all_circuits_needing_data,
                                      num_samples=200, seed=2026))
        mdl_tp = self.target.copy()
        mdl_tp.set_all_parameterizations('full TP')
        self.results = ModelTest(mdl_tp, self.target, verbosity=0).run(self.data)
        self.base_label = list(self.results.estimates.keys())[0]

    def test_missing_base_label_raises(self):
        with self.assertRaises(ValueError) as ctx:
            refit_instruments_cptplnd(self.results, 'no-such-estimate')
        self.assertIn('available', str(ctx.exception))

    def test_label_collision_raises(self):
        with self.assertRaises(ValueError):
            refit_instruments_cptplnd(self.results, self.base_label,
                                      new_estimate_label=self.base_label)

    def test_lindblad_source_raises(self):
        mdl_lnd = self.target.copy()
        mdl_lnd.set_all_parameterizations('CPTPLND')
        results = ModelTest(mdl_lnd, self.target, verbosity=0).run(self.data)
        lbl = list(results.estimates.keys())[0]
        with self.assertRaises(ValueError) as ctx:
            refit_instruments_cptplnd(results, lbl)
        self.assertIn('effect-then-gate', str(ctx.exception))

    def test_no_instruments_raises(self):
        target = _build_target(with_instrument=False)
        # circuits without the instrument label so the instrument-free model fits
        edesign = _build_edesign(target)
        circuits = [c for c in edesign.all_circuits_needing_data
                    if 'Iz' not in str(c)]
        from pygsti.protocols import CircuitListsDesign
        edesign2 = CircuitListsDesign([circuits])
        data = ProtocolData(
            edesign2, pygsti.data.simulate_data(target, circuits,
                                                num_samples=200, seed=2026))
        mdl_tp = target.copy()
        mdl_tp.set_all_parameterizations('full TP')
        results = ModelTest(mdl_tp, target, verbosity=0).run(data)
        lbl = list(results.estimates.keys())[0]
        with self.assertRaises(ValueError) as ctx:
            refit_instruments_cptplnd(results, lbl)
        self.assertIn('no instruments', str(ctx.exception))


@pytest.mark.slow
class RefitRegressionTester(BaseCase):
    """The whole bug as a test: data generated from a noisy full-rank
    instrument model, TP fit, then the projection-seeded CPTPLND refit must
    reach TP-fit quality with all members CP.  (A target-seeded CPTPLND fit
    fails this by a large margin -- the frozen singular base caps the member
    rank; see findings encoded in the diagnostics module.)"""

    OPT = {'maxiter': 150}

    def test_projection_seeded_refit_reaches_tp_quality(self):
        target = _build_target()
        edesign = _build_edesign(target)
        truth = _build_truth(target)
        circuits = edesign.all_circuits_needing_data
        ds = pygsti.data.simulate_data(truth, circuits, num_samples=5000, seed=2026)
        data = ProtocolData(edesign, ds)

        mdl_tp = target.copy()
        mdl_tp.set_all_parameterizations('full TP')
        tp_proto = GateSetTomography(
            initial_model=GSTInitialModel(model=mdl_tp, target_model=target),
            gaugeopt_suite=None, optimizer=self.OPT, verbosity=1, name='full TP')
        results = tp_proto.run(data, disable_checkpointing=True)

        tp_mdl = results.estimates['full TP'].models['final iteration estimate']
        tp_2dll = pygsti.tools.two_delta_logl(tp_mdl, ds, circuits)

        new_label = refit_instruments_cptplnd(results, 'full TP',
                                              optimizer=self.OPT, verbosity=1)
        self.assertEqual(new_label, 'full TP.CPTPLND')
        self.assertIn(new_label, results.estimates)

        est = results.estimates[new_label]
        lnd_mdl = est.models['final iteration estimate']
        lnd_2dll = pygsti.tools.two_delta_logl(lnd_mdl, ds, circuits)

        # CP-constrained fit quality must be statistically comparable to TP's
        self.assertLessEqual(lnd_2dll, 1.2 * tp_2dll + 20.0,
                             f"CPTPLND refit 2dlogl {lnd_2dll:.1f} vs TP {tp_2dll:.1f}")
        # ... with genuinely CP instrument members
        self.assertGreater(_min_member_choi_eig(lnd_mdl), -1e-6)

        # parameterization-preserving gauge opt added its models: structure is
        # retained (members still carry the effect-then-gate chart rather than
        # being coerced to a dense/TP form) and predictions are gauge-invariant.
        # (num_params is NOT compared: transform_composed_model currently
        # duplicates the shared POVM-errormap parameters -- see
        # issues/transform-composed-model-duplicates-shared-instrument-params.md)
        self.assertIn('stdgaugeopt', est.models)
        go_mdl = est.models['stdgaugeopt']
        from pygsti.modelmembers.operations import ComposedOp
        for inst in go_mdl.instruments.values():
            for m in inst.values():
                self.assertIsInstance(m, ComposedOp)
        go_2dll = pygsti.tools.two_delta_logl(go_mdl, ds, circuits)
        self.assertAlmostEqual(go_2dll, lnd_2dll, delta=1e-4 * max(lnd_2dll, 1.0))
