import unittest
import GST
import numpy as np
import warnings

class GateSetConstructionTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def assertArraysAlmostEqual(self,a,b):
        self.assertAlmostEqual( np.linalg.norm(a-b), 0 )

    def assertWarns(self, callable, *args, **kwds):
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter('always')
            result = callable(*args, **kwds)
            self.assertTrue(len(warning_list) > 0)
        return result



class TestGateSetConstructionMethods(GateSetConstructionTestCase):
    
    def test_constructGates(self):
        b = "gm" #basis -- "gm" (Gell-Mann) or "std" (Standard)
        prm = "full" #parameterization "full" or "linear"
        ue = True #unitary embedding

        leakA_old   = GST.GateSetConstruction._oldBuildGate( [1,1,1], [('L0',),('L1',),('L2',)], "LX(pi,0,1)",b).matrix
        rotXa_old   = GST.GateSetConstruction._oldBuildGate( [2],[('Q0',)], "X(pi/2,Q0)",b).matrix
        rotX2_old   = GST.GateSetConstruction._oldBuildGate( [2],[('Q0',)], "X(pi,Q0)",b).matrix
        rotLeak_old = GST.GateSetConstruction._oldBuildGate( [2,1],[('Q0',),('L0',)], "X(pi,Q0):LX(pi,0,2)",b).matrix
        leakB_old   = GST.GateSetConstruction._oldBuildGate( [2,1],[('Q0',),('L0',)], "LX(pi,0,2)",b).matrix
        rotXb_old   = GST.GateSetConstruction._oldBuildGate( [2,1,1],[('Q0',),('L0',),('L1',)], "X(pi,Q0)",b).matrix
        CnotA_old   = GST.GateSetConstruction._oldBuildGate( [4],[('Q0','Q1')], "CX(pi,Q0,Q1)",b).matrix
        CnotB_old   = GST.GateSetConstruction._oldBuildGate( [4,1],[('Q0','Q1'),('L0',)], "CX(pi,Q0,Q1)",b).matrix
        

        leakA   = GST.build_gate( [1,1,1], [('L0',),('L1',),('L2',)], "LX(pi,0,1)",b,prm,ue).matrix
        rotXa   = GST.build_gate( [2],[('Q0',)], "X(pi/2,Q0)",b,prm,ue).matrix
        rotX2   = GST.build_gate( [2],[('Q0',)], "X(pi,Q0)",b,prm,ue).matrix
        rotLeak = GST.build_gate( [2,1],[('Q0',),('L0',)], "X(pi,Q0):LX(pi,0,2)",b,prm,ue).matrix
        leakB   = GST.build_gate( [2,1],[('Q0',),('L0',)], "LX(pi,0,2)",b,prm,ue).matrix
        rotXb   = GST.build_gate( [2,1,1],[('Q0',),('L0',),('L1',)], "X(pi,Q0)",b,prm,ue).matrix
        CnotA   = GST.build_gate( [4],[('Q0','Q1')], "CX(pi,Q0,Q1)",b,prm,ue).matrix
        CnotB   = GST.build_gate( [4,1],[('Q0','Q1'),('L0',)], "CX(pi,Q0,Q1)",b,prm,ue).matrix

        self.assertArraysAlmostEqual(leakA  , leakA_old  )
        self.assertArraysAlmostEqual(rotXa  , rotXa_old  )
        self.assertArraysAlmostEqual(rotX2  , rotX2_old  )
        self.assertArraysAlmostEqual(rotLeak, rotLeak_old)
        self.assertArraysAlmostEqual(leakB  , leakB_old  )
        self.assertArraysAlmostEqual(rotXb  , rotXb_old  )
        self.assertArraysAlmostEqual(CnotA  , CnotA_old  )
        self.assertArraysAlmostEqual(CnotB  , CnotB_old  )

        leakA_ans = np.array( [[ 0.,  1.,  0.],
                               [ 1.,  0.,  0.],
                               [ 0.,  0.,  1.]], 'd')
        self.assertArraysAlmostEqual(leakA, leakA_ans)

        rotXa_ans = np.array([[ 1.,  0.,  0.,  0.],
                              [ 0.,  1.,  0.,  0.],
                              [ 0.,  0.,  0,  -1.],
                              [ 0.,  0.,  1.,  0]], 'd')
        self.assertArraysAlmostEqual(rotXa, rotXa_ans)

        rotX2_ans = np.array([[ 1.,  0.,  0.,  0.],
                              [ 0.,  1.,  0.,  0.],
                              [ 0.,  0., -1.,  0.],
                              [ 0.,  0.,  0., -1.]], 'd')
        self.assertArraysAlmostEqual(rotX2, rotX2_ans)
        
        rotLeak_ans = np.array([[ 0.5,         0.,          0.,         -0.5,         0.70710678],
                                [ 0.,          0.,          0.,          0.,          0.        ],
                                [ 0.,          0.,          0.,          0.,          0.        ],
                                [ 0.5,         0.,          0.,         -0.5,        -0.70710678],
                                [ 0.70710678,  0.,          0.,          0.70710678,  0.        ]], 'd')
        self.assertArraysAlmostEqual(rotLeak, rotLeak_ans)

        leakB_ans = np.array(  [[ 0.5,         0.,          0.,         -0.5,         0.70710678],
                                [ 0.,          0.,          0.,          0.,          0.        ],
                                [ 0.,          0.,          0.,          0.,          0.        ],
                                [-0.5,         0.,          0.,          0.5,         0.70710678],
                                [ 0.70710678,  0.,          0.,          0.70710678,  0.        ]], 'd')
        self.assertArraysAlmostEqual(leakB, leakB_ans)

        rotXb_ans = np.array( [[ 1.,  0.,  0.,  0.,  0.,  0.],
                               [ 0.,  1.,  0.,  0.,  0.,  0.],
                               [ 0.,  0., -1.,  0.,  0.,  0.],
                               [ 0.,  0.,  0., -1.,  0.,  0.],
                               [ 0.,  0.,  0.,  0.,  1.,  0.],
                               [ 0.,  0.,  0.,  0.,  0.,  1.]], 'd')
        self.assertArraysAlmostEqual(rotXb, rotXb_ans)

        A = 9.42809042e-01
        B = 3.33333333e-01
        CnotA_ans = np.array( [[ 1.0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                               [   0,  1.0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                               [   0,    0,    0,    0,    0,    0,    0,    0,    0,  1.0,    0,    0,    0,    0,    0,    0 ],
                               [   0,    0,    0,    0,    0,    0,    0,    0,  1.0,    0,    0,    0,    0,    0,    0,    0 ],
                               [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  1.0,    0,    0,    0,    0 ],
                               [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  1.0,    0,    0,    0,    0,    0 ],
                               [   0,    0,    0,    0,    0,    0,  1.0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                               [   0,    0,    0,    0,    0,    0,    0,  1.0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                               [   0,    0,    0, -1.0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                               [   0,    0, -1.0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                               [   0,    0,    0,    0,    0, -1.0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                               [   0,    0,    0,    0, -1.0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                               [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, -1.0,    0,    0,    0 ],
                               [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  1.0,    0,    0 ],
                               [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    B,    A ],
                               [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    A,   -B ]], 'd')
        self.assertArraysAlmostEqual(CnotA, CnotA_ans)

        CnotB_ans = np.array( [[ 1.0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                               [   0,  1.0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                               [   0,    0,    0,    0,    0,    0,    0,    0,    0,  1.0,    0,    0,    0,    0,    0,    0,    0 ],
                               [   0,    0,    0,    0,    0,    0,    0,    0,  1.0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                               [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  1.0,    0,    0,    0,    0,    0 ],
                               [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  1.0,    0,    0,    0,    0,    0,    0 ],
                               [   0,    0,    0,    0,    0,    0,  1.0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                               [   0,    0,    0,    0,    0,    0,    0,  1.0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                               [   0,    0,    0, -1.0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                               [   0,    0, -1.0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                               [   0,    0,    0,    0,    0, -1.0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                               [   0,    0,    0,    0, -1.0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                               [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, -1.0,    0,    0,    0,    0 ],
                               [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  1.0,    0,    0,    0 ],
                               [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    B,    A,    0 ],
                               [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    A,   -B,    0 ],
                               [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1 ]], 'd')
        self.assertArraysAlmostEqual(CnotB, CnotB_ans)

    def test_parameterized_gate(self):

        # A single-qubit gate on a 2-qubit space, parameterizes so that there are only 16 gate parameters
        gate = GST.build_gate( [4],[('Q0','Q1')], "X(pi,Q0)","gm", parameterization="linear")

        A = -5.77350269e-01
        B = 8.16496581e-01
        C = -6.66666667e-01
        D = -4.71404521e-01
        E = -3.33333333e-01
        gate_ans = np.array( [[ 1.0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                              [   0,    0,    0,    0,    0,    0,  1.0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                              [   0,    0,  1.0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                              [   0,    0,    0,    0,  1.0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                              [   0,    0,    0,  1.0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                              [   0,    0,    0,    0,    0,  1.0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                              [   0,  1.0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                              [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  1.0,    0,    0,    0 ],
                              [   0,    0,    0,    0,    0,    0,    0,    0, -1.0,    0,    0,    0,    0,    0,    0,    0 ],
                              [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0, -1.0,    0,    0,    0,    0,    0 ],
                              [   0,    0,    0,    0,    0,    0,    0,    0,    0, -1.0,    0,    0,    0,    0,    0,    0 ],
                              [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, -1.0,    0,    0,    0,    0 ],
                              [   0,    0,    0,    0,    0,    0,    0,  1.0,    0,    0,    0,    0,    0,    0,    0,    0 ],
                              [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    A,    B ],
                              [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    A,    C,    D ],
                              [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    B,    D,    E ]], 'd')
        self.assertArraysAlmostEqual(gate.matrix, gate_ans)

        vec = gate.to_vector()
        self.assertEqual(vec.shape, (16,)) #should only have 16 parameters
        self.assertEqual(gate.matrix.dtype, np.float64)  #should be real-valued

        vec_ans = np.array([ 1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0., -1.], 'd')
        self.assertArraysAlmostEqual(vec, vec_ans)


    def test_build_gatesets(self):

        stateSpace = [2] #density matrix is a 2x2 matrix
        spaceLabels = [('Q0',)] #interpret the 2x2 density matrix as a single qubit named 'Q0'
        gateset1 = GST.GateSet()
        gateset1.set_rhovec( GST.build_vector(stateSpace,spaceLabels,"0") )
        gateset1.set_evec(   GST.build_vector(stateSpace,spaceLabels,"1") )
        gateset1.set_gate('Gi', GST.build_gate(stateSpace,spaceLabels,"I(Q0)"))
        gateset1.set_gate('Gx', GST.build_gate(stateSpace,spaceLabels,"X(pi/2,Q0)"))
        gateset1.set_gate('Gy', GST.build_gate(stateSpace,spaceLabels,"Y(pi/2,Q0)"))
        gateset1.set_identity_vec( GST.build_identity_vec(stateSpace) )
        gateset1.add_spam_label(0,0,'plus')
        gateset1.add_spam_label(0,-1,'minus')

        gateset2 = GST.build_gateset( [2], [('Q0',)],['Gi','Gx','Gy'], 
                                     [ "I(Q0)","X(pi/2,Q0)", "Y(pi/2,Q0)"],
                                     rhoExpressions=["0"], EExpressions=["1"], 
                                     spamLabelDict={'plus': (0,0), 'minus': (0,-1) })

        gateset3 = self.assertWarns(GST.build_gateset, [2], [('Q0',)],['Gi','Gx','Gy'], 
                                     [ "I(Q0)","X(pi/2,Q0)", "Y(pi/2,Q0)"],
                                     rhoExpressions=["0"], EExpressions=["1"], 
                                     spamLabelDict={'plus': (0,0), 'minus': (-1,-1) })

        gateset4_txt = \
"""
# Test text file describing a gateset

# State prepared, specified as a state in the Pauli basis (I,X,Y,Z)
rho
PauliVec
1/sqrt(2) 0 0 1/sqrt(2)

# State measured as yes outcome, also specified as a state in the Pauli basis
E
PauliVec
1/sqrt(2) 0 0 -1/sqrt(2)

Gi
PauliMx
1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1

Gx
PauliMx
1 0 0 0
0 1 0 0
0 0 0 1
0 0 -1 0

Gy
PauliMx
1 0 0 0
0 0 0 -1
0 0 1 0
0 1 0 0

IDENTITYVEC sqrt(2) 0 0 0
SPAMLABEL plus = rho E
SPAMLABEL minus = rho remainder
"""
        open("temp_test_files/Test_Gateset.txt","w").write(gateset4_txt)
        gateset4 = GST.load_gateset("temp_test_files/Test_Gateset.txt")

    def test_two_qubit_gate(self):
        gate = GST.GateSetConstruction.two_qubit_gate(xx=0.5, xy=0.5, xz=0.5, yy=0.5, yz=0.5, zz=0.5)

        
    def test_gateset_tools(self):

        gateset = GST.build_gateset( [2], [('Q0',)],['Gi','Gx','Gy'], 
                                    [ "I(Q0)","X(pi/2,Q0)", "Y(pi/2,Q0)"],
                                    rhoExpressions=["0"], EExpressions=["1"], 
                                    spamLabelDict={'plus': (0,0), 'minus': (0,-1) })

        gateset_rot = GST.GateSetTools.rotate_gateset(gateset, (np.pi/2,0,0) ) #rotate all gates by pi/2 about X axis
        #print gateset_rot

        rotXPi   = GST.build_gate( [2],[('Q0',)], "X(pi,Q0)").matrix
        rotXPiOv2   = GST.build_gate( [2],[('Q0',)], "X(pi/2,Q0)").matrix        
        rotYPiOv2   = GST.build_gate( [2],[('Q0',)], "Y(pi/2,Q0)").matrix        

        self.assertArraysAlmostEqual(gateset_rot['Gi'], rotXPiOv2)
        self.assertArraysAlmostEqual(gateset_rot['Gx'], rotXPi)
        self.assertArraysAlmostEqual(gateset_rot['Gx'], np.dot(rotXPiOv2,rotXPiOv2))
        self.assertArraysAlmostEqual(gateset_rot['Gy'], np.dot(rotXPiOv2,rotYPiOv2))


        gateset_dep = GST.GateSetTools.depolarize_gateset(gateset,noise=0.1)
        #print gateset_dep

        Gi_dep = np.array([[ 1,   0,   0,   0 ],
                           [ 0, 0.9,   0,   0 ],
                           [ 0,   0, 0.9,   0 ],
                           [ 0,   0,   0, 0.9 ]], 'd')
        Gx_dep = np.array([[ 1,   0,   0,   0 ],
                           [ 0, 0.9,   0,   0 ],
                           [ 0,   0,   0,-0.9 ],
                           [ 0,   0, 0.9,   0 ]], 'd')
        Gy_dep = np.array([[ 1,   0,   0,   0 ],
                           [ 0,   0,   0, 0.9 ],
                           [ 0,   0, 0.9,   0 ],
                           [ 0,-0.9,   0,   0 ]], 'd')

        self.assertArraysAlmostEqual(gateset_dep['Gi'], Gi_dep)
        self.assertArraysAlmostEqual(gateset_dep['Gx'], Gx_dep)
        self.assertArraysAlmostEqual(gateset_dep['Gy'], Gy_dep)


        gateset_spam = GST.GateSetTools.depolarize_spam(gateset,noise=0.1)
        #print gateset_spam
        self.assertAlmostEqual(np.dot(gateset.EVecs[0].T,gateset.rhoVecs[0]),0)
        self.assertAlmostEqual(np.dot(gateset_spam.EVecs[0].T,gateset_spam.rhoVecs[0]),0.095)
        self.assertArraysAlmostEqual(gateset_spam.rhoVecs[0], 1/np.sqrt(2)*np.array([1,0,0,0.9]).reshape(-1,1) )
        self.assertArraysAlmostEqual(gateset_spam.EVecs[0], 1/np.sqrt(2)*np.array([1,0,0,-0.9]).reshape(-1,1) )
        


      
if __name__ == "__main__":
    unittest.main(verbosity=2)
