import unittest
import pygsti
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

        old_build_gate = pygsti.construction.gatesetconstruction._oldBuildGate
        leakA_old   = old_build_gate( [1,1,1], [('L0',),('L1',),('L2',)], "LX(pi,0,1)",b).matrix
        ident_old   = old_build_gate( [2],[('Q0',)], "I(Q0)",b).matrix
        rotXa_old   = old_build_gate( [2],[('Q0',)], "X(pi/2,Q0)",b).matrix
        rotX2_old   = old_build_gate( [2],[('Q0',)], "X(pi,Q0)",b).matrix
        rotYa_old   = old_build_gate( [2],[('Q0',)], "Y(pi/2,Q0)",b).matrix
        rotZa_old   = old_build_gate( [2],[('Q0',)], "Z(pi/2,Q0)",b).matrix
        rotLeak_old = old_build_gate( [2,1],[('Q0',),('L0',)], "X(pi,Q0):LX(pi,0,2)",b).matrix
        leakB_old   = old_build_gate( [2,1],[('Q0',),('L0',)], "LX(pi,0,2)",b).matrix
        rotXb_old   = old_build_gate( [2,1,1],[('Q0',),('L0',),('L1',)], "X(pi,Q0)",b).matrix
        CnotA_old   = old_build_gate( [4],[('Q0','Q1')], "CX(pi,Q0,Q1)",b).matrix
        CnotB_old   = old_build_gate( [4,1],[('Q0','Q1'),('L0',)], "CX(pi,Q0,Q1)",b).matrix
        CY_old      = old_build_gate( [4],[('Q0','Q1')], "CY(pi,Q0,Q1)",b).matrix
        CZ_old      = old_build_gate( [4],[('Q0','Q1')], "CZ(pi,Q0,Q1)",b).matrix
        rotXstd_old = old_build_gate( [2],[('Q0',)], "X(pi/2,Q0)","std").matrix
        rotXpp_old  = old_build_gate( [2],[('Q0',)], "X(pi/2,Q0)","pp").matrix

        with self.assertRaises(ValueError):
            old_build_gate( [2],[('Q0',)], "X(pi/2,Q0)","FooBar") #bad basis specifier
        with self.assertRaises(ValueError):
            old_build_gate( [2],[('Q0',)], "FooBar(Q0)",b) #bad gate name
        with self.assertRaises(ValueError):
            old_build_gate( [2],[('A0',)], "I(Q0)",b) #bad state specifier (A0)
        with self.assertRaises(ValueError):
            old_build_gate( [4],[('Q0',)], "I(Q0)",b) #state space dim mismatch


        build_gate = pygsti.construction.build_gate
        leakA   = build_gate( [1,1,1], [('L0',),('L1',),('L2',)], "LX(pi,0,1)",b,prm,ue).matrix
        ident   = build_gate( [2],[('Q0',)], "I(Q0)",b,prm,ue).matrix
        rotXa   = build_gate( [2],[('Q0',)], "X(pi/2,Q0)",b,prm,ue).matrix
        rotX2   = build_gate( [2],[('Q0',)], "X(pi,Q0)",b,prm,ue).matrix
        rotYa   = build_gate( [2],[('Q0',)], "Y(pi/2,Q0)",b,prm,ue).matrix
        rotZa   = build_gate( [2],[('Q0',)], "Z(pi/2,Q0)",b,prm,ue).matrix
        rotNa   = build_gate( [2],[('Q0',)], "N(pi/2,1.0,0.5,0,Q0)",b,prm,ue).matrix
        rotLeak = build_gate( [2,1],[('Q0',),('L0',)], "X(pi,Q0):LX(pi,0,2)",b,prm,ue).matrix
        leakB   = build_gate( [2,1],[('Q0',),('L0',)], "LX(pi,0,2)",b,prm,ue).matrix
        rotXb   = build_gate( [2,1,1],[('Q0',),('L0',),('L1',)], "X(pi,Q0)",b,prm,ue).matrix
        CnotA   = build_gate( [4],[('Q0','Q1')], "CX(pi,Q0,Q1)",b,prm,ue).matrix
        CnotB   = build_gate( [4,1],[('Q0','Q1'),('L0',)], "CX(pi,Q0,Q1)",b,prm,ue).matrix
        CY      = build_gate( [4],[('Q0','Q1')], "CY(pi,Q0,Q1)",b,prm,ue).matrix
        CZ      = build_gate( [4],[('Q0','Q1')], "CZ(pi,Q0,Q1)",b,prm,ue).matrix
        rotXstd = build_gate( [2],[('Q0',)], "X(pi/2,Q0)","std",prm,ue).matrix
        rotXpp  = build_gate( [2],[('Q0',)], "X(pi/2,Q0)","pp",prm,ue).matrix

        with self.assertRaises(ValueError):
            build_gate( [2],[('Q0',)], "X(pi/2,Q0)","FooBar",prm,ue) #bad basis specifier
        with self.assertRaises(ValueError):
            build_gate( [2],[('Q0',)], "FooBar(Q0)",b,prm,ue) #bad gate name
        with self.assertRaises(ValueError):
            build_gate( [2],[('A0',)], "I(Q0)",b,prm,ue) #bad state specifier (A0)
        with self.assertRaises(ValueError):
            build_gate( [4],[('Q0',)], "I(Q0)",b,prm,ue) #state space dim mismatch
        with self.assertRaises(ValueError):
            build_gate( [2,2],[('Q0',),('Q1',)], "CZ(pi,Q0,Q1)",b,prm,ue) # Q0 & Q1 must be in same tensor-prod block of state space

        self.assertArraysAlmostEqual(leakA  , leakA_old  )
        self.assertArraysAlmostEqual(ident  , ident_old  )
        self.assertArraysAlmostEqual(rotXa  , rotXa_old  )
        self.assertArraysAlmostEqual(rotX2  , rotX2_old  )
        self.assertArraysAlmostEqual(rotYa  , rotYa_old  )
        self.assertArraysAlmostEqual(rotZa  , rotZa_old  )
        self.assertArraysAlmostEqual(rotLeak, rotLeak_old)
        self.assertArraysAlmostEqual(leakB  , leakB_old  )
        self.assertArraysAlmostEqual(rotXb  , rotXb_old  )
        self.assertArraysAlmostEqual(CnotA  , CnotA_old  )
        self.assertArraysAlmostEqual(CnotB  , CnotB_old  )
        self.assertArraysAlmostEqual(CY     , CY_old     )
        self.assertArraysAlmostEqual(CZ     , CZ_old     )


        #Do it all again with unitary embedding == False
        ue = False #unitary embedding
        leakA   = build_gate( [1,1,1], [('L0',),('L1',),('L2',)], "LX(pi,0,1)",b,prm,ue).matrix
        ident   = build_gate( [2],[('Q0',)], "I(Q0)",b,prm,ue).matrix
        rotXa   = build_gate( [2],[('Q0',)], "X(pi/2,Q0)",b,prm,ue).matrix
        rotX2   = build_gate( [2],[('Q0',)], "X(pi,Q0)",b,prm,ue).matrix
        rotYa   = build_gate( [2],[('Q0',)], "Y(pi/2,Q0)",b,prm,ue).matrix
        rotZa   = build_gate( [2],[('Q0',)], "Z(pi/2,Q0)",b,prm,ue).matrix
        rotNa   = build_gate( [2],[('Q0',)], "N(pi/2,1.0,0.5,0,Q0)",b,prm,ue).matrix
        rotLeak = build_gate( [2,1],[('Q0',),('L0',)], "X(pi,Q0):LX(pi,0,2)",b,prm,ue).matrix
        leakB   = build_gate( [2,1],[('Q0',),('L0',)], "LX(pi,0,2)",b,prm,ue).matrix
        rotXb   = build_gate( [2,1,1],[('Q0',),('L0',),('L1',)], "X(pi,Q0)",b,prm,ue).matrix
        CnotA   = build_gate( [4],[('Q0','Q1')], "CX(pi,Q0,Q1)",b,prm,ue).matrix
        CnotB   = build_gate( [4,1],[('Q0','Q1'),('L0',)], "CX(pi,Q0,Q1)",b,prm,ue).matrix
        CY      = build_gate( [4],[('Q0','Q1')], "CY(pi,Q0,Q1)",b,prm,ue).matrix
        CZ      = build_gate( [4],[('Q0','Q1')], "CZ(pi,Q0,Q1)",b,prm,ue).matrix
        rotXstd = build_gate( [2],[('Q0',)], "X(pi/2,Q0)","std",prm,ue).matrix
        rotXpp  = build_gate( [2],[('Q0',)], "X(pi/2,Q0)","pp",prm,ue).matrix

        self.assertArraysAlmostEqual(leakA  , leakA_old  )
        self.assertArraysAlmostEqual(ident  , ident_old  )
        self.assertArraysAlmostEqual(rotXa  , rotXa_old  )
        self.assertArraysAlmostEqual(rotX2  , rotX2_old  )
        self.assertArraysAlmostEqual(rotYa  , rotYa_old  )
        self.assertArraysAlmostEqual(rotZa  , rotZa_old  )
        self.assertArraysAlmostEqual(rotLeak, rotLeak_old)
        self.assertArraysAlmostEqual(leakB  , leakB_old  )
        self.assertArraysAlmostEqual(rotXb  , rotXb_old  )
        self.assertArraysAlmostEqual(CnotA  , CnotA_old  )
        self.assertArraysAlmostEqual(CnotB  , CnotB_old  )
        self.assertArraysAlmostEqual(CY     , CY_old     )
        self.assertArraysAlmostEqual(CZ     , CZ_old     )

        


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
        gate = pygsti.construction.build_gate( [4],[('Q0','Q1')], "X(pi,Q0)","gm",
                                               parameterization="linear")
        gate2 = pygsti.construction.build_gate( [2],[('Q0',)], "D(Q0)","gm", parameterization="linear")
        gate3 = pygsti.construction.build_gate( [4],[('Q0','Q1')], "X(pi,Q0):D(Q1)","gm",
                                               parameterization="linear") #composes parameterized gates


        
        with self.assertRaises(ValueError):
            gate = pygsti.construction.build_gate( [4],[('Q0','Q1')], "X(pi,Q0)","gm",
                                                   parameterization="linear",unitaryEmbedding=True) 
            #no unitary embedding support for "linear"

        with self.assertRaises(ValueError):
            gate = pygsti.construction.build_gate( [4],[('Q0','Q1')], "X(pi,Q0)","gm", 
                                                   parameterization="FooBar") # bad parameterization



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

        #Note: answer is all zeros b/c parameters give *deviation* from base matrix, and this case has no deviation
        vec_ans = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0., 0.], 'd')
        self.assertArraysAlmostEqual(vec, vec_ans)


    def test_build_gatesets(self):

        stateSpace = [2] #density matrix is a 2x2 matrix
        spaceLabels = [('Q0',)] #interpret the 2x2 density matrix as a single qubit named 'Q0'
        gateset1 = pygsti.objects.GateSet()
        gateset1.set_rhovec( pygsti.construction.build_vector(stateSpace,spaceLabels,"0") )
        gateset1.set_evec(   pygsti.construction.build_vector(stateSpace,spaceLabels,"1") )
        gateset1.set_gate('Gi', pygsti.construction.build_gate(stateSpace,spaceLabels,"I(Q0)"))
        gateset1.set_gate('Gx', pygsti.construction.build_gate(stateSpace,spaceLabels,"X(pi/2,Q0)"))
        gateset1.set_gate('Gy', pygsti.construction.build_gate(stateSpace,spaceLabels,"Y(pi/2,Q0)"))
        gateset1.set_identity_vec( pygsti.construction.build_identity_vec(stateSpace) )
        gateset1.add_spam_label(0,0,'plus')
        gateset1.add_spam_label(0,-1,'minus')
        
        with self.assertRaises(ValueError):
            gateset1.add_spam_label(0,0,'plus2') # "plus" already refers to this pair
        with self.assertRaises(ValueError):
            gateset1.add_spam_label(1,0,'badrho') # bad rho index
        with self.assertRaises(ValueError):
            gateset1.add_spam_label(0,1,'bade') # bad E index


        gateset_evec_first = pygsti.objects.GateSet() #set identity vector first
        gateset_evec_first.set_evec( pygsti.construction.build_vector(stateSpace,spaceLabels,"1") )

        gateset_id_first = pygsti.objects.GateSet() #set identity vector first
        gateset_id_first.set_identity_vec( pygsti.construction.build_identity_vec(stateSpace) )
        with self.assertRaises(ValueError):
            gateset_id_first.set_identity_vec( np.array([1,2,3],'d') ) #wrong dimension
        with self.assertRaises(ValueError):
            gateset_id_first.set_rhovec( np.array([1,2,3],'d') ) #wrong dimension
        with self.assertRaises(ValueError):
            gateset_id_first.set_evec( np.array([1,2,3],'d') ) #wrong dimension
        with self.assertRaises(ValueError):
            gateset_id_first.set_rhovec( np.array([1,2,3,4],'d'), 10) #index too large
        with self.assertRaises(ValueError):
            gateset_id_first.set_evec( np.array([1,2,3,4],'d'), 10) #index too large




        gateset2 = pygsti.construction.build_gateset( [2], [('Q0',)],['Gi','Gx','Gy'], 
                                                      [ "I(Q0)","X(pi/2,Q0)", "Y(pi/2,Q0)"],
                                                      rhoExpressions=["0"], EExpressions=["1"], 
                                                      spamLabelDict={'plus': (0,0), 'minus': (0,-1) })

        gateset3 = self.assertWarns(pygsti.construction.build_gateset, 
                [2], [('Q0',)],['Gi','Gx','Gy'], 
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
        gateset4 = pygsti.io.load_gateset("temp_test_files/Test_Gateset.txt")

        std_gateset = pygsti.construction.build_gateset( [2], [('Q0',)],['Gi','Gx','Gy'], 
                                                         [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"],
                                                         rhoExpressions=["0"], EExpressions=["1"], 
                                                         spamLabelDict={'plus': (0,0), 'minus': (0,-1) },
                                                         basis="std")
        pp_gateset = pygsti.construction.build_gateset( [2], [('Q0',)],['Gi','Gx','Gy'], 
                                                        [ "I(Q0)","X(pi/8,Q0)", "Z(pi/8,Q0)"],
                                                        rhoExpressions=["0"], EExpressions=["1"], 
                                                        spamLabelDict={'plus': (0,0), 'minus': (0,-1) },
                                                        basis="pp")

        with self.assertRaises(ValueError):
            pygsti.construction.build_gateset( [2], [('A0',)],['Gi','Gx','Gy'], 
                                               [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"],
                                               rhoExpressions=["FooBar"], EExpressions=["1"], 
                                               spamLabelDict={'plus': (0,0), 'minus': (0,-1) }) # invalid state specifier (A0)

        with self.assertRaises(ValueError):
            pygsti.construction.build_gateset( [4], [('Q0',)],['Gi','Gx','Gy'], 
                                               [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"],
                                               rhoExpressions=["FooBar"], EExpressions=["1"], 
                                               spamLabelDict={'plus': (0,0), 'minus': (0,-1) }) # state space dimension mismatch (4 != 2)
            
        with self.assertRaises(ValueError):
            pygsti.construction.build_gateset( [2], [('Q0',)],['Gi','Gx','Gy'], 
                                               [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"],
                                               rhoExpressions=["FooBar"], EExpressions=["1"], 
                                               spamLabelDict={'plus': (0,0), 'minus': (0,-1) },
                                               basis="FooBar") #Bad basis

        with self.assertRaises(ValueError):
            pygsti.construction.build_gateset( [2], [('Q0',)],['Gi','Gx','Gy'], 
                                               [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"],
                                               rhoExpressions=["FooBar"], EExpressions=["1"], 
                                               spamLabelDict={'plus': (0,0), 'minus': (0,-1) }) #Bad rhoExpression
        with self.assertRaises(ValueError):
            pygsti.construction.build_gateset( [2], [('Q0',)],['Gi','Gx','Gy'], 
                                               [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"],
                                               rhoExpressions=["0"], EExpressions=["FooBar"], 
                                               spamLabelDict={'plus': (0,0), 'minus': (0,-1) }) #Bad EExpression


    def test_two_qubit_gate(self):
        gate = pygsti.two_qubit_gate(xx=0.5, xy=0.5, xz=0.5, yy=0.5, yz=0.5, zz=0.5)

        
    def test_gateset_tools(self):

        gateset = pygsti.construction.build_gateset( [2], [('Q0',)],['Gi','Gx','Gy'], 
                                                     [ "I(Q0)","X(pi/2,Q0)", "Y(pi/2,Q0)"],
                                                     rhoExpressions=["0"], EExpressions=["1"], 
                                                     spamLabelDict={'plus': (0,0), 'minus': (0,-1) })

        gateset_2q = pygsti.construction.build_gateset( 
            [4], [('Q0','Q1')],['GIX','GIY','GXI','GYI','GCNOT'], 
            [ "I(Q0):X(pi/2,Q1)", "I(Q0):Y(pi/2,Q1)", "X(pi/2,Q0):I(Q1)", "Y(pi/2,Q0):I(Q1)", "CX(pi,Q0,Q1)" ],
            rhoExpressions=["0"], EExpressions=["0","1","2"], 
            spamLabelDict={'upup': (0,0), 'updn': (0,1), 'dnup': (0,2), 'dndn': (0,-1) }, basis="pp" )

        gateset_rot = pygsti.objects.gatesettools.rotate_gateset(gateset, (np.pi/2,0,0) ) #rotate all gates by pi/2 about X axis
        gateset_randu = pygsti.objects.gatesettools.randomize_gateset_with_unitary(gateset,0.01)
        gateset_randu = pygsti.objects.gatesettools.randomize_gateset_with_unitary(gateset,0.01,seed=1234)
        #print gateset_rot

        rotXPi   = pygsti.construction.build_gate( [2],[('Q0',)], "X(pi,Q0)").matrix
        rotXPiOv2   = pygsti.construction.build_gate( [2],[('Q0',)], "X(pi/2,Q0)").matrix        
        rotYPiOv2   = pygsti.construction.build_gate( [2],[('Q0',)], "Y(pi/2,Q0)").matrix        

        self.assertArraysAlmostEqual(gateset_rot['Gi'], rotXPiOv2)
        self.assertArraysAlmostEqual(gateset_rot['Gx'], rotXPi)
        self.assertArraysAlmostEqual(gateset_rot['Gx'], np.dot(rotXPiOv2,rotXPiOv2))
        self.assertArraysAlmostEqual(gateset_rot['Gy'], np.dot(rotXPiOv2,rotYPiOv2))

        gateset_2q_rot = pygsti.objects.gatesettools.rotate_2q_gateset(gateset_2q, rotate=list(np.zeros(15,'d')))
        gateset_2q_randu = pygsti.objects.gatesettools.randomize_gateset_with_unitary(gateset_2q,0.01)
        gateset_2q_randu = pygsti.objects.gatesettools.randomize_gateset_with_unitary(gateset_2q,0.01,seed=1234)

        #TODO: test 2q rotated gates??

        gateset_dep = pygsti.objects.gatesettools.depolarize_gateset(gateset,noise=0.1)
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


        gateset_spam = pygsti.objects.gatesettools.depolarize_spam(gateset,noise=0.1)
        #print gateset_spam
        self.assertAlmostEqual(np.dot(gateset.EVecs[0].T,gateset.rhoVecs[0]),0)
        self.assertAlmostEqual(np.dot(gateset_spam.EVecs[0].T,gateset_spam.rhoVecs[0]),0.095)
        self.assertArraysAlmostEqual(gateset_spam.rhoVecs[0], 1/np.sqrt(2)*np.array([1,0,0,0.9]).reshape(-1,1) )
        self.assertArraysAlmostEqual(gateset_spam.EVecs[0], 1/np.sqrt(2)*np.array([1,0,0,-0.9]).reshape(-1,1) )

        gateset_rand_rot = pygsti.objects.gatesettools.rotate_gateset(gateset,max_rotate=0.2)
        gateset_rand_rot = pygsti.objects.gatesettools.rotate_gateset(gateset,max_rotate=0.2,seed=1234)
        with self.assertRaises(ValueError):
            pygsti.objects.gatesettools.rotate_gateset(gateset,rotate=0.2,max_rotate=0.2) #can't specify both
        with self.assertRaises(ValueError):
            pygsti.objects.gatesettools.rotate_gateset(gateset) #must specify rotate or max_rotate
        with self.assertRaises(ValueError):
            pygsti.objects.gatesettools.rotate_gateset(gateset, (1,2,3,4) ) #tuple must be length 3 (or a float)
        with self.assertRaises(ValueError):
            pygsti.objects.gatesettools.rotate_gateset(gateset, "a string!" ) #must be a 3-tuple or float
        with self.assertRaises(ValueError):
            pygsti.objects.gatesettools.rotate_gateset(gateset_2q, rotate=(0,0,0)) #wrong dimension gateset


        gateset_2q_rand_rot = pygsti.objects.gatesettools.rotate_2q_gateset(gateset_2q,max_rotate=0.2)
        gateset_2q_rand_rot = pygsti.objects.gatesettools.rotate_2q_gateset(gateset_2q,max_rotate=0.2,seed=1234)
        with self.assertRaises(ValueError):
            pygsti.objects.gatesettools.rotate_2q_gateset(gateset_2q,rotate=0.2,max_rotate=0.2) #can't specify both
        with self.assertRaises(ValueError):
            pygsti.objects.gatesettools.rotate_2q_gateset(gateset_2q) #must specify rotate or max_rotate
        with self.assertRaises(ValueError):
            pygsti.objects.gatesettools.rotate_2q_gateset(gateset_2q, (1,2,3,4) ) #tuple must be length 15 (or a float)
        with self.assertRaises(ValueError):
            pygsti.objects.gatesettools.rotate_2q_gateset(gateset_2q, "a string!" ) #must be a 3-tuple or float
        with self.assertRaises(ValueError):
            pygsti.objects.gatesettools.rotate_2q_gateset(gateset, rotate=np.zeros(15,'d')) #wrong dimension gateset

        gateset_rand_dep = pygsti.objects.gatesettools.depolarize_gateset(gateset,max_noise=0.1)
        gateset_rand_dep = pygsti.objects.gatesettools.depolarize_gateset(gateset,max_noise=0.1,seed=1234)
        with self.assertRaises(ValueError):
            pygsti.objects.gatesettools.depolarize_gateset(gateset,noise=0.1,max_noise=0.1) #can't specify both
        with self.assertRaises(ValueError):
            pygsti.objects.gatesettools.depolarize_gateset(gateset) #must specify noise or max_noise

        gateset_rand_spam = pygsti.objects.gatesettools.depolarize_spam(gateset,max_noise=0.1)
        gateset_rand_spam = pygsti.objects.gatesettools.depolarize_spam(gateset,max_noise=0.1,seed=1234)
        with self.assertRaises(ValueError):
            pygsti.objects.gatesettools.depolarize_spam(gateset,noise=0.1,max_noise=0.1) #can't specify both
        with self.assertRaises(ValueError):
            pygsti.objects.gatesettools.depolarize_spam(gateset) #must specify noise or max_noise

        
    def test_spamspecs(self):
        strs = pygsti.construction.gatestring_list( [('Gx',),('Gy',),('Gx','Gx')] )
        rhoSpecs, ESpecs = pygsti.construction.build_spam_specs(fiducialGateStrings=strs)

        with self.assertRaises(ValueError):
            pygsti.construction.build_spam_specs(rhoSpecs=rhoSpecs, rhoStrs=strs) #can't specify both...
            
        with self.assertRaises(ValueError):
            pygsti.construction.build_spam_specs(rhoStrs=strs, fiducialGateStrings=strs) #can't specify both...

        with self.assertRaises(ValueError):
            pygsti.construction.build_spam_specs() # must specify something!

        with self.assertRaises(ValueError):
            pygsti.construction.build_spam_specs(rhoStrs=strs, ESpecs=ESpecs, EStrs=strs) #can't specify both...
            
        with self.assertRaises(ValueError):
            pygsti.construction.build_spam_specs(EStrs=strs, fiducialGateStrings=strs) #can't specify both...
            
        with self.assertRaises(ValueError):
            pygsti.construction.build_spam_specs(rhoStrs=strs) # must specify some E-thing!

    def test_gate_object(self):
        gate_full = pygsti.construction.build_gate( [2],[('Q0',)], "I(Q0)","gm", parameterization="full")
        gate_linear = pygsti.construction.build_gate( [2],[('Q0',)], "D(Q0)","gm", parameterization="linear")
        gate_linear_copy = gate_linear.copy()
        gate_full_copy = gate_full.copy()

        value_dim_full = gate_full.value_dimension()
        value_dim_linear = gate_linear.value_dimension()

        with self.assertRaises(ValueError):
            gate_full.set_value( np.zeros((5,5),'d') ) #wrong size - must be 4x4

        gate_linear.set_value( np.zeros(value_dim_linear,'d') )
        with self.assertRaises(ValueError):
            gate_linear.set_value( np.zeros(value_dim_linear+1,'d') ) #wrong size

        full_as_str = str(gate_full)
        linear_as_str = str(gate_linear)

        #Linear from scratch
        baseMx = np.zeros( (2,2) )
        paramArray = np.array( [1.0,1.0] )
        parameterToBaseIndicesMap = { 0: [(0,0)], 1: [(1,1)] } #parameterize only the diagonal els
        gate_linear_B = pygsti.obj.LinearlyParameterizedGate(baseMx, paramArray,
                                                             parameterToBaseIndicesMap, real=True)
        with self.assertRaises(ValueError):
            pygsti.obj.LinearlyParameterizedGate(baseMx, np.array( [1.0+1j, 1.0] ),
                                                 parameterToBaseIndicesMap, real=True) #must be real
            
        numParams = gate_linear_B.get_num_params()
        v = gate_linear_B.to_vector()
        gate_linear_B.from_vector(v)
        deriv = gate_linear_B.deriv_wrt_params()
        with self.assertRaises(ValueError):
            gate_linear_B.get_num_params(bG0=False) #not implemented
        with self.assertRaises(ValueError):
            gate_linear_B.to_vector(bG0=False) #not implemented
        with self.assertRaises(ValueError):
            gate_linear_B.from_vector(v,bG0=False) #not implemented
        with self.assertRaises(ValueError):
            gate_linear_B.deriv_wrt_params(bG0=False) #not implemented

        I = np.identity(2)
        gate_linear_B.transform(I,I)


        #Full from scratch
        mx = np.array( [[1,0],[0,1]], 'd' )
        gate_full_B = pygsti.obj.FullyParameterizedGate(mx)
            
        numParams = gate_full_B.get_num_params()
        v = gate_full_B.to_vector()
        gate_full_B.from_vector(v)
        deriv = gate_full_B.deriv_wrt_params()

        numParams_noG0 = gate_full_B.get_num_params(bG0=False)
        v_noG0 = gate_full_B.to_vector(bG0=False)
        gate_full_B.from_vector(v_noG0,bG0=False)
        deriv_noG0 = gate_full_B.deriv_wrt_params(bG0=False)

        I = np.identity(2)
        gate_full_B.transform(I,I)

        
      
if __name__ == "__main__":
    unittest.main(verbosity=2)
