import unittest
import GST
import numpy as np

class JamiolkowskiTestCase(unittest.TestCase):

    def setUp(self):
        # density matrix == 3x3 block diagonal matrix: a 2x2 block followed by a 1x1 block 
        self.stateSpaceDims = [2,1] 

        #labels which give a tensor product interp. for the states within each density matrix block
        self.stateSpaceLabels = [('Qhappy',),('Lsad',)] 
                                            
        #Build a test gate   -- old # X(pi,Qhappy)*LX(pi,0,2)
        self.testGate = GST.buildGate( self.stateSpaceDims, self.stateSpaceLabels, "LX(pi,0,2)","std") 
        self.testGateGM_mx = GST.BT.basisChg_StdToGellMann(self.testGate.matrix, self.stateSpaceDims)
        self.expTestGate_mx = GST.BT.expandFromStdDirectSumMx(self.testGate.matrix, self.stateSpaceDims)
        self.expTestGateGM_mx = GST.BT.basisChg_StdToGellMann(self.expTestGate_mx)

    def checkBasis(self, cmb):
        #Op with Jamio map on gate in std and gm bases
        Jmx1 = GST.JOps.opWithJamiolkowskiIsomorphism(self.testGate.matrix, gateMxBasis='std',
                                                      choiMxBasis=cmb, dimOrStateSpaceDims=self.stateSpaceDims)
        Jmx2 = GST.JOps.opWithJamiolkowskiIsomorphism(self.testGateGM_mx, gateMxBasis='gm',
                                                      choiMxBasis=cmb, dimOrStateSpaceDims=self.stateSpaceDims)

        #Make sure these yield the same trace == 1 matrix
        self.assertArraysAlmostEqual(Jmx1,Jmx2)
        self.assertAlmostEqual(np.trace(Jmx1), 1.0)
        
        #Op on expanded gate in std and gm bases
        JmxExp1 = GST.JOps.opWithJamiolkowskiIsomorphism(self.expTestGate_mx,gateMxBasis='std',choiMxBasis=cmb)
        JmxExp2 = GST.JOps.opWithJamiolkowskiIsomorphism(self.expTestGateGM_mx,gateMxBasis='gm',choiMxBasis=cmb)

        #Make sure these are the same as operating on the contracted basis
        self.assertArraysAlmostEqual(Jmx1,JmxExp1)
        self.assertArraysAlmostEqual(Jmx1,JmxExp2)
                
        #Reverse transform should yield back the gate matrix
        revTestGate_mx = GST.JOps.opWithInvJamiolkowskiIsomorphism(Jmx1,choiMxBasis=cmb,
                                                                   dimOrStateSpaceDims=self.stateSpaceDims,
                                                                   gateMxBasis='gm')
        self.assertArraysAlmostEqual(revTestGate_mx, self.testGateGM_mx)
    
        #Reverse transform without specifying stateSpaceDims, then contraction, should yield same result
        revExpTestGate_mx = GST.JOps.opWithInvJamiolkowskiIsomorphism(Jmx1,choiMxBasis=cmb,
                                                                      gateMxBasis='std')
        self.assertArraysAlmostEqual( GST.BT.contractToStdDirectSumMx(revExpTestGate_mx,self.stateSpaceDims),
                                      self.testGate.matrix)


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
