import pygsti.extras.interpygate as interp
from pygsti.tools.basistools import change_basis
import numpy as np
from ...util import BaseCase

sigI = np.array([[1.,0],[0, 1]], dtype='complex')
sigX = np.array([[0, 1],[1, 0]], dtype='complex')
sigY = np.array([[0,-1],[1, 0]], dtype='complex') * 1.j
sigZ = np.array([[1, 0],[0,-1]], dtype='complex')
sigM = (sigX - 1.j*sigY)/2.
sigP = (sigX + 1.j*sigY)/2.

class SingleQubitTargetOp(pygsti.modelmembers.operations.OpFactory):
    def __init__(self):
        self.process = self.create_target_gate
        pygsti.modelmembers.operations.OpFactory.__init__(self, 4, evotype="densitymx")
        self.dim = 4
    
    def create_target_gate(self, v):
        phi, theta = v
        target_unitary = (np.cos(theta/2) * sigI + 
                          1.j * np.sin(theta/2) * (np.cos(phi) * sigX + np.sin(phi) * sigY))
        superop = change_basis(np.kron(target_unitary.conj(), target_unitary), 'col', 'pp')
        return superop
    
    def create_object(self, args=None, sslbls=None):
        assert(sslbls is None)
        mx = self.process([*args])
        return pygsti.modelmembers.operations.StaticArbitraryOp(mx)






class InterpygateConstructionTester(BaseCase):
    @classmethod
    def setUpClass(cls):
        super(InterpygateConstructionTester, cls).setUpClass()

        cls.target_model = stdC.target_model()
        cls.depol_strength = 1e-3
        cls.mdl = cls.target_model.depolarize(op_noise=cls.depol_strength)

    def test_create_target(self):
        target = np.bmat([[np.eye(2),np.zeros([2,2])],[np.zeros([2,2]),np.sqrt(2)/2*(sigI++1.j*sigY)]])
        target_op = SingleQubitTargetOp()
        target_op.create_target_gate([0,np.pi/2])
        self.assertAlmostEqual(AGI, r_AGI, places=10)

    def test_p_to_r_EI(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        EI = entanglement_infidelity(self.mdl.operations['Gc0'], self.target_model.operations['Gc0'])
        r_EI = analysis.p_to_r(1 - self.depol_strength, d=2, rtype='EI')
        # TODO assert correctness without comparing to optools EI
        self.assertAlmostEqual(EI, r_EI, places=10)
