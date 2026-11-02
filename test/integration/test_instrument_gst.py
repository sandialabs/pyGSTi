"""
End-to-end GST with a mid-circuit measurement: a CPTPLND fit started cold from an ideal
projective instrument, whose Lindblad charts are built around unitary post-measurement gates.
"""
import unittest

import numpy as np

import pygsti
import pygsti.tools.basistools as bt
import pygsti.tools.optools as ot
from pygsti.circuits import Circuit
from pygsti.modelmembers import operations as op
from pygsti.modelmembers.instruments import Instrument
from pygsti.modelpacks import smq1Q_XZ
from pygsti.protocols import GateSetTomography, GSTInitialModel, ProtocolData, StandardGSTDesign
from pygsti.tools.jamiolkowski import fast_jamiolkowski_iso_std

QL = (0,)
INST_LBL = ('Iz', 0)


def _sk(mat, basis):
    return bt.stdmx_to_vec(mat, basis).ravel().real


def _build_target():
    mdl = smq1Q_XZ.target_model(qubit_labels=QL).copy()
    mdl[[]] = np.eye(4)   # global idle, needed by the empty germ
    members = {}
    for k in range(2):
        P = np.zeros((2, 2))
        P[k, k] = 1.0
        members[f'p{k}'] = bt.change_basis(np.kron(P, P.conj()), 'std', mdl.basis).real
    mdl[INST_LBL] = Instrument(members)
    mdl.sim = 'matrix'
    return mdl


def _build_truth(target):
    """Slightly depolarizing gates, and an instrument whose members are full rank: a weak
    measurement followed by a depolarizing kick, so its members have several Kraus operators."""
    truth = target.copy()
    basis = truth.basis
    D = np.eye(4)
    D[1:, 1:] *= 0.98
    for lbl in list(truth.operations.keys()):
        truth[lbl] = D @ np.asarray(truth[lbl].to_dense())
    E0 = np.diag([0.92, 0.08]).astype(complex)
    truth[INST_LBL] = Instrument({'p0': D @ ot.rootconj_superop(_sk(E0, basis), basis),
                                  'p1': D @ ot.rootconj_superop(_sk(np.eye(2) - E0, basis), basis)})
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


class ColdInstrumentFitTester(unittest.TestCase):

    OPT = {'maxiter': 150}

    def _fit(self, data, target, parameterization):
        mdl = target.copy()
        mdl.set_all_parameterizations(parameterization)
        proto = GateSetTomography(initial_model=GSTInitialModel(model=mdl, target_model=target),
                                  gaugeopt_suite=None, optimizer=self.OPT, verbosity=0, name=parameterization)
        results = proto.run(data, disable_checkpointing=True)
        return mdl, results.estimates[parameterization].models['final iteration estimate']

    def test_cold_cptplnd_fit_reaches_tp_quality(self):
        target = _build_target()
        edesign = _build_edesign(target)
        circuits = edesign.all_circuits_needing_data
        ds = pygsti.data.simulate_data(_build_truth(target), circuits, num_samples=5000, seed=2026)
        data = ProtocolData(edesign, ds)

        _, tp_mdl = self._fit(data, target, 'full TP')
        seed, lnd_mdl = self._fit(data, target, 'CPTPLND')

        # The CPTPLND seed is the ideal instrument, built around identity post-measurement unitaries.
        for member in seed.instruments[INST_LBL].values():
            static = member.factorops[1].factorops[0]
            self.assertIsInstance(static, op.StaticUnitaryOp)
            np.testing.assert_allclose(static.to_dense('HilbertSchmidt'), np.eye(4), atol=1e-12)

        tp_2dll = pygsti.tools.two_delta_logl(tp_mdl, ds, circuits)
        lnd_2dll = pygsti.tools.two_delta_logl(lnd_mdl, ds, circuits)
        self.assertLessEqual(lnd_2dll, 1.2 * tp_2dll + 20.0,
                             f"CPTPLND 2dlogl {lnd_2dll:.1f} vs full TP {tp_2dll:.1f}")
        self.assertGreater(_min_member_choi_eig(lnd_mdl), -1e-6)
