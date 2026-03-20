import numpy as np

from pygsti.baseobjs import Label
from pygsti.tools.basistools import stdmx_to_vec, vec_to_stdmx
from pygsti.leakage.models import leaky_qubit_model_from_pspec, promote_bb_to_bt
from ..util import BaseCase


class LeakyQubitModelTester(BaseCase):

    @classmethod
    def setUpClass(cls):
        from pygsti.processors import QubitProcessorSpec
        pspec = QubitProcessorSpec(1, ['Gxpi2', 'Gypi2'], qubit_labels=['Q0'])
        cls.model    = leaky_qubit_model_from_pspec(pspec)
        cls.mx_basis = cls.model.basis

    def test_state_space_udim(self):
        self.assertEqual(self.model.state_space.udim, 3)

    def test_gauge_group_is_direct_sum(self):
        from pygsti.models.gaugegroup import DirectSumUnitaryGroup
        self.assertIsInstance(self.model.default_gauge_group, DirectSumUnitaryGroup)

    def test_povm_has_two_effects(self):
        povm = self.model.povms['Mdefault']
        self.assertIn('0', povm)
        self.assertIn('1', povm)

    def test_rho0_trace_one(self):
        rho_vec = self.model.preps['rho0'].to_dense().ravel()
        rho_mx  = vec_to_stdmx(rho_vec, self.mx_basis)
        self.assertAlmostEqual(np.trace(rho_mx).real, 1.0, places=10)

    def test_povm_effects_sum_to_identity(self):
        povm = self.model.povms['Mdefault']
        eff0 = povm['0'].to_dense().ravel()
        eff1 = povm['1'].to_dense().ravel()
        expected = stdmx_to_vec(np.eye(3, dtype=complex), self.mx_basis).ravel()
        self.assertArraysAlmostEqual(eff0 + eff1, expected)

    def test_gate_superop_dimensions(self):
        for op in self.model.operations.values():
            mx = op.to_dense()
            self.assertEqual(mx.shape, (9, 9))


class PromoteBBToBTTester(BaseCase):

    @classmethod
    def setUpClass(cls):
        from pygsti.processors import QubitProcessorSpec
        from pygsti.models import create_explicit_model
        pspec_2q  = QubitProcessorSpec(2, ['Gcnot', 'Gxpi2', 'Gypi2'], qubit_labels=['Q0', 'Q1'], geometry='line')
        cls.model_2q = create_explicit_model(pspec_2q)
        cls.model_6  = promote_bb_to_bt(cls.model_2q)

    def test_state_space_udim(self):
        self.assertEqual(self.model_6.state_space.udim, 6)

    def test_gauge_group_is_unitary(self):
        from pygsti.models.gaugegroup import UnitaryGaugeGroup
        self.assertIsInstance(self.model_6.default_gauge_group, UnitaryGaugeGroup)

    def test_povm_has_4_effects(self):
        povm = self.model_6.povms['Mdefault']
        for key in ('00', '01', '10', '11'):
            self.assertIn(key, povm)

    def test_povm_effects_sum_to_identity(self):
        mx_basis = self.model_6.basis
        total = sum(
            eff.to_dense().ravel() for eff in self.model_6.povms['Mdefault'].values()
        )
        expected = stdmx_to_vec(np.eye(6, dtype=complex), mx_basis).ravel()
        self.assertArraysAlmostEqual(total, expected)

    def test_rho0_trace_one(self):
        mx_basis = self.model_6.basis
        rho_vec  = self.model_6.preps['rho0'].to_dense().ravel()
        rho_mx   = vec_to_stdmx(rho_vec, mx_basis)
        self.assertAlmostEqual(np.trace(rho_mx).real, 1.0, places=10)

    def test_operations_match_source(self):
        # Every non-idle gate in the 2-qubit model should be present in the 6-level model.
        non_idle = [k for k in self.model_2q.operations.keys() if k != Label(())]
        for op_lbl in non_idle:
            self.assertIn(op_lbl, self.model_6.operations)

    def test_gate_superop_dimensions(self):
        for op in self.model_6.operations.values():
            mx = op.to_dense()
            self.assertEqual(mx.shape, (36, 36))
