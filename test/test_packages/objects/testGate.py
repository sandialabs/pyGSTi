import unittest
import pygsti
import numpy as np
import pickle

from  pygsti.objects import GateMatrix
#from  pygsti.objects import GateMap
import pygsti.construction as pc
import scipy.sparse as sps

from ..testutils import BaseTestCase, compare_files, temp_files

class GateTestCase(BaseTestCase):

    def setUp(self):
        super(GateTestCase, self).setUp()

    def test_lpg_deriv(self):
        gs_target_lp = pc.build_gateset(
            [2], [('Q0',)],['Gi','Gx'], [ "D(Q0)","X(pi/2,Q0)" ],
            basis="pp", parameterization="linear" )

    
        gs_target_lp2 = pc.build_gateset(
            [4], [('Q0','Q1')],['Gix','Giy','Gxi','Gyi','Gcnot'], 
            [ "I(Q0):X(pi/2,Q1)", "I(Q0):Y(pi/2,Q1)", "X(pi/2,Q0):I(Q1)", "Y(pi/2,Q0):I(Q1)", "CX(pi,Q0,Q1)" ],
            basis="pp", parameterization="linearTP" )

        gs_target_lp2.preps['rho0'] = pygsti.objects.TPParameterizedSPAMVec(gs_target_lp2.preps['rho0'])
        #because there's no easy way to specify this TP parameterization...
        # (above is leftover from a longer feature test)
        
        check = pygsti.objects.gate.check_deriv_wrt_params
        testDeriv = check(gs_target_lp.gates['Gi'])
        testDeriv = check(gs_target_lp.gates['Gx'])
        testDeriv = check(gs_target_lp2.gates['Gix'])
        testDeriv = check(gs_target_lp2.gates['Giy'])
        testDeriv = check(gs_target_lp2.gates['Gxi'])
        testDeriv = check(gs_target_lp2.gates['Gyi'])
        testDeriv = check(gs_target_lp2.gates['Gcnot'])

    def test_gate_base(self):
        #check that everything is not implemented
        gate = pygsti.objects.Gate(4,"densitymx")

        state = np.zeros( (4,1), 'd' )
        state[0] = state[3] = 1.0

        T = pygsti.objects.FullGaugeGroupElement(
            np.array( [ [0,1],
                        [1,0] ], 'd') )
            
        self.assertEqual( gate.get_dimension(), 4 )
        self.assertEqual( gate.num_params(), 0) #default is no params
        v = gate.to_vector()
        self.assertEqual(len(v),0)
        gate.from_vector(v)
        gate.copy()

        #with self.assertRaises(NotImplementedError):
        #    gate.acton(state) # REMOVED from gate - now use reps
        with self.assertRaises(NotImplementedError):
            gate.transform(T)
        with self.assertRaises(NotImplementedError):
            gate.depolarize(0.05)
        with self.assertRaises(NotImplementedError):
            gate.rotate((0.01,0,0),'gm')
        with self.assertRaises(NotImplementedError):
            gate.frobeniusdist2(gate)
        with self.assertRaises(NotImplementedError):
            gate.frobeniusdist(gate)
        with self.assertRaises(NotImplementedError):
            gate.jtracedist(gate)
        with self.assertRaises(NotImplementedError):
            gate.diamonddist(gate)

        s = pickle.dumps(gate)
        gate2 = pickle.loads(s)

        
    def test_gate_matrix(self):
        gate = GateMatrix(np.zeros(2),"densitymx")

        gate[:] #calls __getslice__ in python 2.7

        # default is no params, so can compute deriv & hessian (all zeros)
        self.assertFalse(gate.has_nonzero_hessian()) 
        gate.deriv_wrt_params()
        gate.hessian_wrt_params()

        with self.assertRaises(ValueError):
            gate.set_value(np.identity(2))

        with self.assertRaises(ValueError):
            GateMatrix.convert_to_matrix( np.zeros( (2,2,2), 'd') ) #must be 2D
        with self.assertRaises(ValueError):
            GateMatrix.convert_to_matrix( np.zeros( (2,4), 'd') ) #must be square

        bad_mxs = ['akdjsfaksdf',
                  [[], [1, 2]],
                  [[[]], [[1, 2]]]]
        for bad_mx in bad_mxs:
            with self.assertRaises(ValueError):
                GateMatrix.convert_to_matrix(bad_mx)


#GateMap removed
#    def test_gate_map(self):
#        gatemap = GateMap(dim=4)
#        self.assertEqual(gatemap.size,4**2)

        
    def test_gate_methods(self):
        dummyGS = pygsti.objects.GateSet()
        mx = np.identity(4,'d')
        mx2 = np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,0,1],
                        [0,0,-1,0]],'d')

        #build a list of gates to test
        gates_to_test = []
        
        gates_to_test.append( pygsti.objects.StaticGate(mx) )
        gates_to_test.append( pygsti.objects.FullyParameterizedGate(mx) )
        gates_to_test.append( pygsti.objects.TPParameterizedGate(mx) )

        parameterArray = np.zeros(2,'d')
        parameterToBaseIndicesMap = {0: [(0,3),(3,0)], 1: [(1,2),(2,1)] }
        gates_to_test.append( pygsti.objects.LinearlyParameterizedGate(
            mx,parameterArray, parameterToBaseIndicesMap,
            leftTransform=None, rightTransform=None, real=True) )
        
        gates_to_test.append( pygsti.objects.EigenvalueParameterizedGate(
            mx,includeOffDiagsInDegen2Blocks=False,
            TPconstrainedAndUnital=False) )

        gates_to_test.append( pygsti.objects.EigenvalueParameterizedGate(
            mx2,includeOffDiagsInDegen2Blocks=False,
            TPconstrainedAndUnital=False) )
        
        gates_to_test.append( pygsti.objects.LindbladParameterizedGate.from_gate_matrix(
            mx,unitaryPostfactor=None,
            ham_basis="pp", nonham_basis="pp", cptp=True,
            nonham_diagonal_only=False, truncate=True, mxBasis="pp") )

        gates_to_test.append( pygsti.objects.LindbladParameterizedGate.from_gate_matrix(
            mx,unitaryPostfactor=None,
            ham_basis="pp", nonham_basis="pp", cptp=True,
            nonham_diagonal_only=True, truncate=True, mxBasis="pp") )

        ppBasis = pygsti.obj.Basis("pp",2)
        gates_to_test.append( pygsti.objects.LindbladParameterizedGate.from_gate_matrix(
            mx,unitaryPostfactor=mx,
            ham_basis=ppBasis, nonham_basis=ppBasis, cptp=False,
            nonham_diagonal_only=False, truncate=True, mxBasis="pp") )

        ppMxs = pygsti.tools.pp_matrices(2)
        gates_to_test.append( pygsti.objects.LindbladParameterizedGate.from_gate_matrix(
            mx,unitaryPostfactor=None,
            ham_basis=ppMxs, nonham_basis=ppMxs, cptp=False,
            nonham_diagonal_only=True, truncate=True, mxBasis="pp") )

        compGate = pygsti.objects.ComposedGate(
            [pygsti.objects.StaticGate(mx),
             pygsti.objects.FullyParameterizedGate(mx),
             pygsti.objects.FullyParameterizedGate(mx2),
             pygsti.objects.StaticGate(mx),
             pygsti.objects.FullyParameterizedGate(mx2)] )
        dummyGS.gates['Gcomp'] = compGate # so to/from vector work in tests below
        gates_to_test.append( dummyGS.gates['Gcomp'] )

        embedGate = pygsti.objects.EmbeddedGate( [('Q0',)], ['Q0'], pygsti.objects.FullyParameterizedGate(mx))
        dummyGS.gates['Gembed'] = embedGate # so to/from vector work in tests below
        gates_to_test.append( dummyGS.gates['Gembed'] )


        #with self.assertRaises(AssertionError): #need to truncate... WHY THOUGH?
        # no need to truncate anymore... (?)
        pygsti.objects.LindbladParameterizedGate.from_gate_matrix(
            mx,unitaryPostfactor=mx,
            ham_basis=ppBasis, nonham_basis=ppBasis, cptp=False,
            nonham_diagonal_only=False, truncate=False, mxBasis="pp")

        for gate in gates_to_test:
            state = np.zeros( (4,1), 'd' )
            state[0] = state[3] = 1.0

            T = pygsti.objects.FullGaugeGroupElement(
                np.array( [ [0,1,0,0],
                            [1,0,0,0],
                            [0,0,1,0],
                            [0,0,0,1] ], 'd') )
            T2 = pygsti.objects.UnitaryGaugeGroupElement(
                np.array( [ [0,1,0,0],
                            [1,0,0,0],
                            [0,0,1,0],
                            [0,0,0,1] ], 'd') )

            #test Gate methods
            self.assertEqual( gate.get_dimension(), 4 )
            gate.torep().acton( pygsti.objects.FullyParameterizedSPAMVec(state).torep("prep"))
            gate.has_nonzero_hessian()

            try:
                gate.transform(T)
            except NotImplementedError: pass #OK, as this is unallowed for some gate types
            except ValueError: pass #OK, as this is unallowed for some gate types
            
            try:
            #if isinstance(gate, pygsti.obj.LindbladParameterizedGate):
                gate.transform(T2)
            except NotImplementedError: pass #OK, as this is unallowed for some gate types
            except ValueError: pass #OK, as this is unallowed for some gate types

            try:
                gate.depolarize(0.05)
                gate.depolarize([0.05,0.10,0.15])
            except ValueError: pass #OK, as this is unallowed for some gate types

            try:
                #gate.rotate(0.01,'gm') #float as arg is no longer allowed
                gate.rotate([0.01,0.02,0.03],'gm')
            except ValueError: pass #OK, as this is unallowed for some gate types
            
            self.assertAlmostEqual( gate.frobeniusdist2(gate), 0.0 )
            self.assertAlmostEqual( gate.frobeniusdist(gate), 0.0 )
            self.assertAlmostEqual( gate.jtracedist(gate), 0.0 )
            self.assertAlmostEqual( gate.diamonddist(gate), 0.0 )

            nP = gate.num_params()
            gate2 = gate.copy()
            self.assertTrue( np.allclose(gate,gate2) )
            
            v = gate.to_vector()
            gate.from_vector(v)
            self.assertTrue( np.allclose(gate,gate2) )

            s = pickle.dumps(gate)
            gate2 = pickle.loads(s)

            #test GateMatrix methods (since all our gates_to_test are GateMatrix objs)
            a = gate[0,0]
            b = gate[:]
            c = gate[0,:]
            s = str(gate)
            #with self.assertRaises(ValueError):
            #    gate.<some method that changes shape in place> #not allowed to reshape

            try:
                gate[1,1] = 2.0
            except ValueError: pass #OK, as this is unallowed for some gate types

            deriv = gate.deriv_wrt_params()
            self.assertEqual( deriv.shape, (gate.dim**2, nP))
            if nP > 0:
                deriv2 = gate.deriv_wrt_params([0])
                self.assertEqual( deriv2.shape, (gate.dim**2, 1))
            try:
                hessian = gate.hessian_wrt_params()
                hessian = gate.hessian_wrt_params([1,2],None)
                hessian = gate.hessian_wrt_params(None,[1,2])
                hessian = gate.hessian_wrt_params([1,2],[1,2])
            except NotImplementedError: pass #OK, as this is unallowed for some gate types

            #other methods
            try:
                cgate = gate.compose(gate)
            except NotImplementedError: pass #Still todo for ComposedGate (and maybe more?)
            except ValueError: pass #Other gates may just not allow compositions

            try:
                gate.set_value( np.identity(4,'d') )
            except ValueError: pass #OK, as this is unallowed for some gate types

            try:
                gate.set_value( np.random.random((4,4)) )
            except ValueError: pass #OK, as this is unallowed for some gate types

            with self.assertRaises(ValueError):
                gate.set_value( np.random.random((4,2)) )
            with self.assertRaises(ValueError):
                gate.set_value( np.identity(5,'d') )



    def test_gatemap_methods(self):
        dummyGS = pygsti.objects.GateSet()
        densemx = np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,0,1],
                        [0,0,-1,0]],'d')
        sparsemx = sps.csr_matrix(densemx, dtype='d')

        #build a list of gates to test
        gates_to_test = []
                
        gates_to_test.append( pygsti.objects.LindbladParameterizedGateMap.from_gate_matrix(
            densemx,unitaryPostfactor=None,
            ham_basis="pp", nonham_basis="pp", cptp=True,
            nonham_diagonal_only=False, truncate=True, mxBasis="pp") )

        gates_to_test.append( pygsti.objects.LindbladParameterizedGateMap.from_gate_matrix(
            sparsemx,unitaryPostfactor=None,
            ham_basis="pp", nonham_basis="pp", cptp=True,
            nonham_diagonal_only=False, truncate=True, mxBasis="pp") )

        gates_to_test.append( pygsti.objects.LindbladParameterizedGateMap.from_gate_matrix(
            None,unitaryPostfactor=densemx,
            ham_basis="pp", nonham_basis="pp", cptp=True,
            nonham_diagonal_only=False, truncate=True, mxBasis="pp") )

        gates_to_test.append( pygsti.objects.LindbladParameterizedGateMap.from_gate_matrix(
            None, unitaryPostfactor=sparsemx,
            ham_basis="pp", nonham_basis="pp", cptp=True,
            nonham_diagonal_only=False, truncate=True, mxBasis="pp") )

        ppBasis = pygsti.obj.Basis("pp",2)
        gates_to_test.append( pygsti.objects.LindbladParameterizedGateMap.from_gate_matrix(
            densemx,unitaryPostfactor=None,
            ham_basis=ppBasis, nonham_basis=ppBasis, cptp=False,
            nonham_diagonal_only=True, truncate=True, mxBasis="pp") )

        ppMxs = pygsti.tools.pp_matrices(2)
        testGate= pygsti.objects.LindbladParameterizedGateMap.from_gate_matrix(
            densemx,unitaryPostfactor=None,
            ham_basis=ppMxs, nonham_basis=ppMxs, cptp=False,
            nonham_diagonal_only=True, truncate=True, mxBasis="pp")
        gates_to_test.append( testGate )

        gates_to_test.append(pygsti.objects.LindbladParameterizedGateMap.from_gate_matrix(
            densemx,unitaryPostfactor=None,
            ham_basis=None, nonham_basis=ppMxs, cptp=False,
            nonham_diagonal_only=True, truncate=True, mxBasis="pp"))

        compGate = pygsti.objects.ComposedGateMap( [testGate, testGate, testGate] )
        dummyGS.gates['Gcomp'] = compGate # so to/from vector work in tests below
        gates_to_test.append( dummyGS.gates['Gcomp'] )

        embedGate = pygsti.objects.EmbeddedGateMap( [('Q0',)], ['Q0'], testGate)
        dummyGS.gates['Gembed'] = embedGate # so to/from vector work in tests below
        gates_to_test.append( dummyGS.gates['Gembed'] )

        dummyGS2 = pygsti.objects.GateSet() # b/c will have different dim from dummyGS
        ppBasis2x2 = pygsti.obj.Basis("pp",(2,2))
        embedGate2 = pygsti.objects.EmbeddedGateMap( [('Q0',),('Q1',)], ['Q0'], testGate) # 2 blocks
        dummyGS2.gates['Gembed2'] = embedGate2 # so to/from vector work in tests below
        gates_to_test.append( dummyGS2.gates['Gembed2'] )

        with self.assertRaises(ValueError):
            pygsti.objects.EmbeddedGateMap( [('L0','foobar')], ['Q0'], testGate)
        with self.assertRaises(ValueError):
            pygsti.objects.EmbeddedGateMap( [('Q0',),('Q1',)], ['Q0','Q1'], testGate) #labels correspond to diff blocks            


        for gate in gates_to_test:
            state = np.zeros( (4,1), 'd' )
            state[0] = state[3] = 1.0

            T = pygsti.objects.FullGaugeGroupElement(
                np.array( [ [0,1,0,0],
                            [1,0,0,0],
                            [0,0,1,0],
                            [0,0,0,1] ], 'd') )
            T2 = pygsti.objects.UnitaryGaugeGroupElement(
                np.array( [ [0,1,0,0],
                            [1,0,0,0],
                            [0,0,1,0],
                            [0,0,0,1] ], 'd') )

            #test Gate methods
            self.assertTrue( gate.get_dimension() in  (4,8) ) # embedded gate2 has dim==8
            gate.torep().acton( pygsti.objects.FullyParameterizedSPAMVec(state).torep("prep"))
            if hasattr(gate, '_slow_acton'):
                gate._slow_acton(state) # for EmbeddedGateMaps

            sparseMx = gate.tosparse()

            try:
                gate.transform(T)
            except NotImplementedError: pass #OK, as this is unallowed for some gate types
            except ValueError: pass #OK, as this is unallowed for some gate types

            try:
            #if isinstance(gate, pygsti.obj.LindbladParameterizedGateMap):
                gate.transform(T2)
            except NotImplementedError: pass #OK, as this is unallowed for some gate types
            except ValueError: pass #OK, as this is unallowed for some gate types

            try:
                gate.depolarize(0.05)
                gate.depolarize([0.05,0.10,0.15])
            except NotImplementedError: pass #OK, as this is unallowed for some gate types
            except ValueError: pass #OK, as this is unallowed for some gate types

            try:
                gate.rotate([0.01,0.02,0.03],'gm')
            except NotImplementedError: pass #OK, as this is unallowed for some gate types
            except ValueError: pass #OK, as this is unallowed for some gate types
            
            nP = gate.num_params()
            gate2 = gate.copy()
            #Dense only: self.assertTrue( np.allclose(gate,gate2) )
            
            v = gate.to_vector()
            gate.from_vector(v)
            #Dense only: self.assertTrue( np.allclose(gate,gate2) )

            s = pickle.dumps(gate)
            gate2 = pickle.loads(s)

            #other methods
            s = str(gate)
            try:
                cgate = gate.compose(gate)
            except ValueError: pass #OK, as this is unallowed for some gate types
            except NotImplementedError: pass # still a TODO item for some types (ComposedGateMap)

    def test_convert(self):
        densemx = np.array([[1,0,0,0],
                            [0,1,0,0],
                            [0,0,0,1],
                            [0,0,-1,0]],'d')

        basis = pygsti.obj.Basis("pp",2)
        lndgate = pygsti.objects.LindbladParameterizedGate.from_gate_matrix(
            densemx,unitaryPostfactor=densemx,
            ham_basis=basis, nonham_basis=basis, cptp=True,
            nonham_diagonal_only=False, truncate=True, mxBasis=basis)
        g = pygsti.objects.gate.convert(lndgate,"CPTP",basis) 
        self.assertTrue(g is lndgate) #should be trivial (no) conversion

    def test_eigenvalue_param_gate(self):
        mx = np.array( [[ 1,   0,     0,       0],
                        [ 0,   1,     0,       0],
                        [ 0,   0,     -1, -1e-10],
                        [ 0,   0,  1e-10,     -1]], 'd')
        # degenerate (to tol) -1 evals will generate *complex* evecs
        g1 = pygsti.objects.EigenvalueParameterizedGate(
            mx,includeOffDiagsInDegen2Blocks=False,
            TPconstrainedAndUnital=False)

        mx = np.array( [[ 1,   0,     0,       0],
                        [ 0,   1,     0,       0],
                        [ 0,   0,     -1,      0],
                        [ 0,   0,     0,      -1]], 'complex')
        # 2 degenerate real pairs of evecs => should add off-diag els
        g2 = pygsti.objects.EigenvalueParameterizedGate(
            mx,includeOffDiagsInDegen2Blocks=True,
            TPconstrainedAndUnital=False)
        self.assertEqual(g2.params, [[(1.0, (0, 0))], [(1.0, (1, 1))],
                                     [(1.0, (0, 1))], [(1.0, (1, 0))], # off diags blk 1
                                     [(1.0, (2, 2))], [(1.0, (3, 3))],
                                     [(1.0, (2, 3))], [(1.0, (3, 2))]]) # off diags blk 2


        mx = np.array( [[ 1,   -0.1,     0,      0],
                        [ 0.1,    1,     0,      0],
                        [ 0,      0,     1+1,   -0.1],
                        [ 0,      0,   0.1,      1+1]], 'complex')
        # complex pairs of evecs => make sure combined parameters work
        g3 = pygsti.objects.EigenvalueParameterizedGate(
            mx,includeOffDiagsInDegen2Blocks=True,
            TPconstrainedAndUnital=False)
        self.assertEqual(g3.params, [
            [(1.0, (0, 0)), (1.0, (1, 1))], # single param that is Re part of 0,0 and 1,1 els
            [(1j, (0, 0)), (-1j, (1, 1))],  # Im part of 0,0 and 1,1 els
            [(1.0, (2, 2)), (1.0, (3, 3))], # Re part of 2,2 and 3,3 els
            [(1j, (2, 2)), (-1j, (3, 3))]   # Im part of 2,2 and 3,3 els
        ])


        mx = np.array( [[ 1,   -0.1,     0,      0],
                        [ 0.1,    1,     0,      0],
                        [ 0,      0,     1,   -0.1],
                        [ 0,      0,   0.1,      1]], 'complex')
        # 2 degenerate complex pairs of evecs => should add off-diag els
        g4 = pygsti.objects.EigenvalueParameterizedGate(
            mx,includeOffDiagsInDegen2Blocks=True,
            TPconstrainedAndUnital=False)
        self.assertArraysAlmostEqual(g4.evals, [1.+0.1j, 1.+0.1j, 1.-0.1j, 1.-0.1j]) # Note: evals are sorted!
        self.assertEqual(g4.params,[  
            [(1.0, (0, 0)), (1.0, (2, 2))], # single param that is Re part of 0,0 and 2,2 els (conj eval pair, since sorted)
            [(1j, (0, 0)), (-1j, (2, 2))],  # Im part of 0,0 and 2,2 els
            [(1.0, (1, 1)), (1.0, (3, 3))], # Re part of 1,1 and 3,3 els
            [(1j, (1, 1)), (-1j, (3, 3))],  # Im part of 1,1 and 3,3 els
            [(1.0, (0, 1)), (1.0, (2, 3))], # Re part of 0,1 and 2,3 els (upper triangle)
            [(1j, (0, 1)), (-1j, (2, 3))],  # Im part of 0,1 and 2,3 els (upper triangle); (0,1) and (2,3) must be conjugates
            [(1.0, (1, 0)), (1.0, (3, 2))], # Re part of 1,0 and 3,2 els (lower triangle)
            [(1j, (1, 0)), (-1j, (3, 2))]   # Im part of 1,0 and 3,2 els (lower triangle); (1,0) and (3,2) must be conjugates
        ])

            


if __name__ == '__main__':
    unittest.main(verbosity=2)
