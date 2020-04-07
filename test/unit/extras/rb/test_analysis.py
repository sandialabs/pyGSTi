import numpy as np

from ...util import BaseCase

from pygsti.modelpacks.legacy import std1Q_Cliffords as stdC
from pygsti.tools import average_gate_infidelity, entanglement_infidelity

#from pygsti.extras.rb import analysis #SKIP until RB gets fixed


class RBAnalysisTester(BaseCase):
    @classmethod
    def setUpClass(cls):
        super(RBAnalysisTester, cls).setUpClass()
        cls.target_model = stdC.target_model()
        cls.depol_strength = 1e-3
        cls.mdl = cls.target_model.depolarize(op_noise=cls.depol_strength)

    def test_p_to_r_AGI(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        AGI = average_gate_infidelity(self.mdl.operations['Gc0'], self.target_model.operations['Gc0'])
        r_AGI = analysis.p_to_r(1 - self.depol_strength, d=2, rtype='AGI')
        # TODO assert correctness without comparing to optools AGI
        self.assertAlmostEqual(AGI, r_AGI, places=10)

    def test_p_to_r_EI(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        EI = entanglement_infidelity(self.mdl.operations['Gc0'], self.target_model.operations['Gc0'])
        r_EI = analysis.p_to_r(1 - self.depol_strength, d=2, rtype='EI')
        # TODO assert correctness without comparing to optools EI
        self.assertAlmostEqual(EI, r_EI, places=10)
