import unittest
import pygsti
import numpy as np

class JamiolkowskiTestCase(unittest.TestCase):

    def setUp(self):
        #Set GateSet objects to "strict" mode for testing
        pygsti.objects.GateSet._strict = True


        # density matrix == 3x3 block diagonal matrix: a 2x2 block followed by a 1x1 block 
        self.stateSpaceDims = [2,1] 

        #labels which give a tensor product interp. for the states within each density matrix block
        self.stateSpaceLabels = [('Qhappy',),('Lsad',)] 
                                            
        #Build a test gate   -- old # X(pi,Qhappy)*LX(pi,0,2)
        self.testGate = pygsti.construction.build_gate( self.stateSpaceDims, self.stateSpaceLabels, "LX(pi,0,2)","std") 
        self.testGateGM_mx = pygsti.std_to_gm(self.testGate, self.stateSpaceDims)
        self.expTestGate_mx = pygsti.expand_from_std_direct_sum_mx(self.testGate, self.stateSpaceDims)
        self.expTestGateGM_mx = pygsti.std_to_gm(self.expTestGate_mx)

    def checkBasis(self, cmb):
        #Op with Jamio map on gate in std and gm bases
        Jmx1 = pygsti.jamiolkowski_iso(self.testGate, gateMxBasis='std',
                                       choiMxBasis=cmb, dimOrStateSpaceDims=self.stateSpaceDims)
        Jmx2 = pygsti.jamiolkowski_iso(self.testGateGM_mx, gateMxBasis='gm',
                                       choiMxBasis=cmb, dimOrStateSpaceDims=self.stateSpaceDims)

        #Make sure these yield the same trace == 1 matrix
        self.assertArraysAlmostEqual(Jmx1,Jmx2)
        self.assertAlmostEqual(np.trace(Jmx1), 1.0)
        
        #Op on expanded gate in std and gm bases
        JmxExp1 = pygsti.jamiolkowski_iso(self.expTestGate_mx,gateMxBasis='std',choiMxBasis=cmb)
        JmxExp2 = pygsti.jamiolkowski_iso(self.expTestGateGM_mx,gateMxBasis='gm',choiMxBasis=cmb)

        #Make sure these are the same as operating on the contracted basis
        self.assertArraysAlmostEqual(Jmx1,JmxExp1)
        self.assertArraysAlmostEqual(Jmx1,JmxExp2)
                
        #Reverse transform should yield back the gate matrix
        revTestGate_mx = pygsti.jamiolkowski_iso_inv(Jmx1,choiMxBasis=cmb,
                                                                   dimOrStateSpaceDims=self.stateSpaceDims,
                                                                   gateMxBasis='gm')
        self.assertArraysAlmostEqual(revTestGate_mx, self.testGateGM_mx)
    
        #Reverse transform without specifying stateSpaceDims, then contraction, should yield same result
        revExpTestGate_mx = pygsti.jamiolkowski_iso_inv(Jmx1,choiMxBasis=cmb,
                                                                      gateMxBasis='std')
        self.assertArraysAlmostEqual( pygsti.contract_to_std_direct_sum_mx(revExpTestGate_mx,self.stateSpaceDims),
                                      self.testGate)


    def assertArraysAlmostEqual(self,a,b):
        self.assertAlmostEqual( np.linalg.norm(a-b), 0 )

class TestJamiolkowskiMethods(JamiolkowskiTestCase):

    def test_std_basis(self):
        cmb = 'std' #choi matrix basis
        self.checkBasis(cmb)

    def test_gm_basis(self):
        cmb = 'gm' #choi matrix basis
        self.checkBasis(cmb)
      
if __name__ == "__main__":
    unittest.main(verbosity=2)
