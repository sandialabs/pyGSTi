import numpy as np
import scipy

import pygsti
import pygsti.construction.modelconstruction as mc
import pygsti.modelmembers.operations as op
import pygsti.tools.basistools as bt
from ..util import BaseCase


class ModelConstructionTester(BaseCase):
    def setUp(self):
        #OK for these tests, since we test user interface?
        #Set Model objects to "strict" mode for testing
        pygsti.models.ExplicitOpModel._strict = False

    def test_build_basis_gateset(self):
        modelA = mc.create_explicit_model(
            [('Q0',)], ['Gi', 'Gx', 'Gy'],
            ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"]
        )
        modelB = mc.basis_create_explicit_model(
            [('Q0',)], pygsti.Basis.cast('gm', 4),
            ['Gi', 'Gx', 'Gy'], ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"]
        )
        self.assertAlmostEqual(modelA.frobeniusdist(modelB), 0)

    def test_build_model(self):
        model1 = pygsti.models.ExplicitOpModel(['Q0'])
        model1['rho0'] = mc._basis_create_spam_vector("0", model1.basis)
        model1['Mdefault'] = pygsti.modelmembers.povms.UnconstrainedPOVM([('0', mc._basis_create_spam_vector("0", model1.basis)),
                                                                          ('1', mc._basis_create_spam_vector("1", model1.basis))],
                                                                         evotype='default')
        model1['Gi'] = mc._basis_create_operation(model1.state_space, "I(Q0)")
        model1['Gx'] = mc._basis_create_operation(model1.state_space, "X(pi/2,Q0)")
        model1['Gy'] = mc._basis_create_operation(model1.state_space, "Y(pi/2,Q0)")
    
        model2 = mc.create_explicit_model(
            [('Q0',)], ['Gi', 'Gx', 'Gy'],
            ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"]
        )

        self.assertAlmostEqual(model1.frobeniusdist(model2), 0)

    def test_build_explicit_model(self):
        model = mc.create_explicit_model([('Q0',)], ['Gi', 'Gx', 'Gy'], ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"])
        self.assertEqual(set(model.operations.keys()), set(['Gi', 'Gx', 'Gy']))
        self.assertAlmostEqual(sum(model.probabilities(('Gx', 'Gi', 'Gy')).values()), 1.0)
        self.assertEqual(model.num_params, 60)

        gateset2b = mc.create_explicit_model([('Q0',)], ['Gi', 'Gx', 'Gy'],
                                            ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"],
                                            effect_labels=['1', '0'])
        self.assertArraysAlmostEqual(model.effects['0'], gateset2b.effects['1'])
        self.assertArraysAlmostEqual(model.effects['1'], gateset2b.effects['0'])

        # This is slightly confusing. Single qubit rotations are always stored in "pp" basis internally
        # UPDATE: now this isn't even allowed, as the 'densitymx' type represents states as *real* vectors.
        #std_gateset = mc.create_explicit_model([('Q0',)], ['Gi', 'Gx', 'Gy'],
        #                                      ["I(Q0)", "X(pi/8,Q0)", "Y(pi/8,Q0)"],
        #                                      basis="std")

        pp_gateset = mc.create_explicit_model([('Q0',)], ['Gi', 'Gx', 'Gy'],
                                             ["I(Q0)", "X(pi/8,Q0)", "Y(pi/8,Q0)"],
                                             basis="pp")

        #for op in ['Gi', 'Gx', 'Gy']:
        #    self.assertArraysAlmostEqual(std_gateset[op], pp_gateset[op])

    def test_build_crosstalk_free_model(self):
        nQubits = 2

        mdl = mc.create_crosstalk_free_model(
            nQubits, ('Gi', 'Gx', 'Gy', 'Gcnot'),
            ensure_composed_gates=True,
            independent_gates=False
        )
        assert(set(mdl.operation_blks['gates'].keys()) == set(["Gi", "Gx", "Gy", "Gcnot"]))
        assert(set(mdl.operation_blks['layers'].keys()) == set(
            [('Gi', 0), ('Gi', 1), ('Gx', 0), ('Gx', 1), ('Gy', 0), ('Gy', 1), ('Gcnot', 0, 1), ('Gcnot', 1, 0)]))
        self.assertEqual(mdl.num_params, 0)

        addlErr = pygsti.modelmembers.operations.FullTPOp(np.identity(4, 'd'))  # adds 12 params
        addlErr2 = pygsti.modelmembers.operations.FullTPOp(np.identity(4, 'd'))  # adds 12 params

        mdl.operation_blks['gates']['Gx'].append(addlErr)
        mdl.operation_blks['gates']['Gy'].append(addlErr2)
        mdl.operation_blks['gates']['Gi'].append(addlErr)

        # TODO: If you call mdl.num_params between the 3 calls above, this second one has an error...

        self.assertEqual(mdl.num_params, 24)

        # TODO: These are maybe not deterministic? Sometimes are swapped for me...
        if mdl.operation_blks['layers'][('Gx', 0)].gpindices == slice(0, 12):
            slice1 = slice(0, 12)
            slice2 = slice(12, 24)
        else:
            slice1 = slice(12, 24)
            slice2 = slice(0, 12)
        self.assertEqual(mdl.operation_blks['layers'][('Gx', 0)].gpindices, slice1)
        self.assertEqual(mdl.operation_blks['layers'][('Gy', 0)].gpindices, slice2)
        self.assertEqual(mdl.operation_blks['layers'][('Gi', 0)].gpindices, slice1)
        self.assertEqual(mdl.operation_blks['gates']['Gx'].gpindices, slice1)
        self.assertEqual(mdl.operation_blks['gates']['Gy'].gpindices, slice2)
        self.assertEqual(mdl.operation_blks['gates']['Gi'].gpindices, slice1)

        # Case: ensure_composed_gates=False, independent_gates=True
        cfmdl = mc.create_crosstalk_free_model(
            nQubits, ('Gx', 'Gy', 'Gcnot'),
            depolarization_strengths={'Gx': 0.1, 'idle': 0.01, 'prep': 0.01, 'povm': 0.01},
            stochastic_error_probs={'Gy': (0.02, 0.02, 0.02)},
            lindblad_error_coeffs={
                'Gcnot': {('H', 'ZZ'): 0.01, ('S', 'IX'): 0.01},
            }, qubit_labels=['qb{}'.format(i) for i in range(nQubits)], 
            ensure_composed_gates=False, independent_gates=True)

        self.assertEqual(cfmdl.num_params, 17)

        # Case: ensure_composed_gates=True, independent_gates=False
        cfmdl2 = mc.create_crosstalk_free_model(
            nQubits, ('Gx', 'Gy', 'Gcnot'),
            depolarization_strengths={'Gx': 0.1, 'idle': 0.01, 'prep': 0.01, 'povm': 0.01},
            stochastic_error_probs={'Gy': (0.02, 0.02, 0.02)},
            lindblad_error_coeffs={
                'Gcnot': {('H', 'ZZ'): 0.01, ('S', 'IX'): 0.01},
             }, qubit_labels=['qb{}'.format(i) for i in range(nQubits)],
            ensure_composed_gates=True, independent_gates=False)
        self.assertEqual(cfmdl2.num_params, 11)

        # Same as above but add ('Gx','qb0') to test giving qubit-specific error rates
        cfmdl3 = mc.create_crosstalk_free_model(
            nQubits, ('Gx', 'Gy', 'Gcnot'),
            depolarization_strengths={'Gx': 0.1, ('Gx', 'qb0'): 0.2, 'idle': 0.01, 'prep': 0.01, 'povm': 0.01},
            stochastic_error_probs={'Gy': (0.02, 0.02, 0.02)},
            lindblad_error_coeffs={
                'Gcnot': {('H', 'ZZ'): 0.01, ('S', 'IX'): 0.01},
             }, qubit_labels=['qb{}'.format(i) for i in range(nQubits)],
            ensure_composed_gates=True, independent_gates=False)
        self.assertEqual(cfmdl3.num_params, 12)

    def test_build_crosstalk_free_model_depolarize_parameterizations(self):
        nQubits = 2

        # Test depolarizing
        mdl_depol1 = mc.create_crosstalk_free_model(
            nQubits, ('Gi',), depolarization_strengths={'Gi': 0.1}
        )
        Gi_op = mdl_depol1.operation_blks['gates']['Gi']
        self.assertTrue(isinstance(Gi_op, op.ComposedOp))
        self.assertTrue(isinstance(Gi_op.factorops[0], op.StaticStandardOp))
        self.assertTrue(isinstance(Gi_op.factorops[1], op.DepolarizeOp))
        self.assertEqual(mdl_depol1.num_params, 1)

        # Expand into StochasticNoiseOp
        mdl_depol2 = mc.create_crosstalk_free_model(
            nQubits, ('Gi',), depolarization_strengths={'Gi': 0.1},
            depolarization_parameterization='stochastic'
        )
        Gi_op = mdl_depol2.operation_blks['gates']['Gi']
        self.assertTrue(isinstance(Gi_op, op.ComposedOp))
        self.assertTrue(isinstance(Gi_op.factorops[0], op.StaticStandardOp))
        self.assertTrue(isinstance(Gi_op.factorops[1], op.StochasticNoiseOp))
        self.assertEqual(mdl_depol2.num_params, 3) 

        # Use LindbladOp with "depol", "diagonal" param
        mdl_depol3 = mc.create_crosstalk_free_model(
            nQubits, ('Gi',), depolarization_strengths={'Gi': 0.1},
            depolarization_parameterization='lindblad'
        )
        Gi_op = mdl_depol3.operation_blks['gates']['Gi']
        self.assertTrue(isinstance(Gi_op, op.ComposedOp))
        self.assertEqual(mdl_depol3.num_params, 1)

        # For prep and povm, only lindblad parameterization is allowed
        # Test that proper warnings are raised
        with self.assertWarns(UserWarning):
            mc.create_crosstalk_free_model(
                nQubits, ('Gi',), depolarization_strengths={'Gi': 0.1, 'prep': 0.1},
                depolarization_parameterization='depolarize'
            )
        with self.assertWarns(UserWarning):
            mc.create_crosstalk_free_model(
                nQubits, ('Gi',), depolarization_strengths={'Gi': 0.1, 'prep': 0.1},
                depolarization_parameterization='stochastic'
            )
        
        with self.assertWarns(UserWarning):
            mc.create_crosstalk_free_model(
                nQubits, ('Gi',), depolarization_strengths={'Gi': 0.1, 'povm': 0.1},
                depolarization_parameterization='depolarize'
            )
        with self.assertWarns(UserWarning):
            mc.create_crosstalk_free_model(
                nQubits, ('Gi',), depolarization_strengths={'Gi': 0.1, 'povm': 0.1},
                depolarization_parameterization='stochastic'
            )

    def test_build_crosstalk_free_model_stochastic_parameterizations(self):
        nQubits = 2

        # Test stochastic
        mdl_sto1 = mc.create_crosstalk_free_model(
            nQubits, ('Gi',), stochastic_error_probs={'Gi': (0.1, 0.1, 0.1)},
        )
        Gi_op = mdl_sto1.operation_blks['gates']['Gi']
        self.assertTrue(isinstance(Gi_op, op.ComposedOp))
        self.assertTrue(isinstance(Gi_op.factorops[0], op.StaticStandardOp))
        self.assertTrue(isinstance(Gi_op.factorops[1], op.StochasticNoiseOp))
        self.assertEqual(mdl_sto1.num_params, 3)

        # Use LindbladOp with "cptp", "diagonal" param
        mdl_sto3 = mc.create_crosstalk_free_model(
            nQubits, ('Gi',), stochastic_error_probs={'Gi': (0.1, 0.1, 0.1)},
            stochastic_parameterization='lindblad'
        )
        Gi_op = mdl_sto3.operation_blks['gates']['Gi']
        self.assertTrue(isinstance(Gi_op, op.ComposedOp))
        self.assertEqual(mdl_sto3.num_params, 3)

        # For prep and povm, only lindblad parameterization is allowed
        # Test that proper warnings are raised
        with self.assertWarns(UserWarning):
            mc.create_crosstalk_free_model(
                nQubits, ('Gi',), stochastic_error_probs={'Gi': (0.1, 0.1, 0.1), 'prep': (0.01,)*3},
                stochastic_parameterization='stochastic'
            )
        with self.assertWarns(UserWarning):
            mc.create_crosstalk_free_model(
                nQubits, ('Gi',), stochastic_error_probs={'Gi': (0.1,)*3, 'povm': (0.01,)*3},
                stochastic_parameterization='stochastic'
            )

    def test_build_crosstalk_free_model_lindblad_parameterizations(self):
        nQubits = 2

        # Test Lindblad
        mdl_lb1 = mc.create_crosstalk_free_model(
            nQubits, ('Gi',), lindblad_error_coeffs={'Gi': {('H', 'X'): 0.1, ('S', 'Y'): 0.1}},
        )
        Gi_op = mdl_lb1.operation_blks['gates']['Gi']
        self.assertTrue(isinstance(Gi_op, op.ComposedOp))
        self.assertEqual(Gi_op.errorgen_coefficients(), {('H', 'X'): 0.1, ('S', 'Y'): 0.1})
        self.assertEqual(mdl_lb1.num_params, 2)
    
        # Test param passthrough
        mdl_lb2 = mc.create_crosstalk_free_model(
            nQubits, ('Gi',), lindblad_error_coeffs={'Gi': {('H', 'X'): 0.1, ('S', 'Y'): 0.1}},
            lindblad_parameterization='H+S'
        )
        Gi_op = mdl_lb2.operation_blks['gates']['Gi']
        self.assertTrue(isinstance(Gi_op, op.ComposedOp))
        self.assertEqual(Gi_op.errorgen_coefficients(), {('H', 'X'): 0.1, ('S', 'Y'): 0.1})
        self.assertEqual(mdl_lb2.num_params, 2)

    def test_build_crosstalk_free_model_with_nonstd_gate_unitary_factory(self):
        nQubits = 2

        def fn(args):
            if args is None: args = (0,)
            a, = args
            sigmaZ = np.array([[1, 0], [0, -1]], 'd')
            return scipy.linalg.expm(1j * float(a) * sigmaZ)

        cfmdl = mc.create_crosstalk_free_model(nQubits, ('Gx', 'Gy', 'Gcnot', 'Ga'),
                                              nonstd_gate_unitaries={'Ga': fn})

        c = pygsti.obj.Circuit("Gx:1Ga;0.3:1Gx:1@(0,1)")
        p = cfmdl.probabilities(c)

        self.assertAlmostEqual(p['00'], 0.08733219254516078)
        self.assertAlmostEqual(p['01'], 0.9126678074548386)
    
    def test_build_crosstalk_free_model_with_custom_gates(self):
        nQubits = 2

        class XRotationOpFactory(pygsti.modelmembers.operations.OpFactory):
            def __init__(self):
                ss = pygsti.models.statespace.QubitSpace(1)
                pygsti.modelmembers.operations.OpFactory.__init__(self, state_space=ss, evotype="densitymx")

            def create_object(self, args=None, sslbls=None):
                theta = float(args[0])/2.0
                b = 2*np.cos(theta)*np.sin(theta)
                c = np.cos(theta)**2 - np.sin(theta)**2
                superop = np.array([[1,   0,   0,   0],
                                    [0,   1,   0,   0],
                                    [0,   0,   c,  -b],
                                    [0,   0,   b,   c]],'d')
                return pygsti.modelmembers.operations.StaticArbitraryOp(superop, self.evotype, self.state_space)

        xrot_fact = XRotationOpFactory()
    
        cfmdl = mc.create_crosstalk_free_model(nQubits, ('Gi', 'Gxr'),
                                               custom_gates={'Gxr': xrot_fact})

        c = pygsti.obj.Circuit("Gxr;3.1415926536:1@(0,1)")
        p = cfmdl.probabilities(c)

        self.assertAlmostEqual(p['01'], 1.0)

        c = pygsti.obj.Circuit("Gxr;1.5707963268:1@(0,1)")
        p = cfmdl.probabilities(c)
        
        self.assertAlmostEqual(p['00'], 0.5)
        self.assertAlmostEqual(p['01'], 0.5)

    def test_build_operation_raises_on_bad_parameterization(self):
        with self.assertRaises(ValueError):
            mc._basis_create_operation([('Q0', 'Q1')], "X(pi,Q0)", "gm", parameterization="FooBar")

    def test_build_explicit_model_raises_on_bad_state(self):
        with self.assertRaises(ValueError):
            mc.create_explicit_model([('A0',)], ['Gi', 'Gx', 'Gy'],
                                    ["I(Q0)", "X(pi/8,Q0)", "Y(pi/8,Q0)"])

    def test_build_explicit_model_raises_on_bad_basis(self):
        with self.assertRaises(AssertionError):
            mc.create_explicit_model([('Q0',)], ['Gi', 'Gx', 'Gy'],
                                    ["I(Q0)", "X(pi/8,Q0)", "Y(pi/8,Q0)"],
                                    basis="FooBar")

    def test_build_explicit_model_raises_on_bad_rho_expression(self):
        with self.assertRaises(ValueError):
            mc.create_explicit_model([('Q0',)], ['Gi', 'Gx', 'Gy'],
                                    ["I(Q0)", "X(pi/8,Q0)", "Y(pi/8,Q0)"],
                                    prep_labels=['rho0'], prep_expressions=["FooBar"],)

    def test_build_explicit_model_raises_on_bad_effect_expression(self):
        with self.assertRaises(ValueError):
            mc.create_explicit_model([('Q0',)], ['Gi', 'Gx', 'Gy'],
                                    ["I(Q0)", "X(pi/8,Q0)", "Y(pi/8,Q0)"],
                                    effect_labels=['0', '1'], effect_expressions=["FooBar", "1"])


class GateConstructionBase(object):
    def setUp(self):
        pygsti.models.ExplicitOpModel._strict = False

    def _construct_gates(self, param):
        # TODO these aren't really unit tests
        #CNOT gate
        Ucnot = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 1],
                          [0, 0, 1, 0]], 'd')
        cnotMx = pygsti.tools.unitary_to_process_mx(Ucnot)
        self.CNOT_chk = pygsti.tools.change_basis(cnotMx, "std", self.basis)

        #CPHASE gate
        Ucphase = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, -1]], 'd')
        cphaseMx = pygsti.tools.unitary_to_process_mx(Ucphase)
        self.CPHASE_chk = pygsti.tools.change_basis(cphaseMx, "std", self.basis)
        self.ident = mc._basis_create_operation([('Q0',)], "I(Q0)", self.basis, param)
        self.rotXa = mc._basis_create_operation([('Q0',)], "X(pi/2,Q0)", self.basis, param)
        self.rotX2 = mc._basis_create_operation([('Q0',)], "X(pi,Q0)", self.basis, param)
        self.rotYa = mc._basis_create_operation([('Q0',)], "Y(pi/2,Q0)", self.basis, param)
        self.rotZa = mc._basis_create_operation([('Q0',)], "Z(pi/2,Q0)", self.basis, param)
        self.rotNa = mc._basis_create_operation([('Q0',)], "N(pi/2,1.0,0.5,0,Q0)", self.basis, param)
        self.iwL = mc._basis_create_operation([('Q0', 'L0')], "I(Q0)", self.basis, param)
        self.CnotA = mc._basis_create_operation([('Q0', 'Q1')], "CX(pi,Q0,Q1)", self.basis, param)
        self.CY = mc._basis_create_operation([('Q0', 'Q1')], "CY(pi,Q0,Q1)", self.basis, param)
        self.CZ = mc._basis_create_operation([('Q0', 'Q1')], "CZ(pi,Q0,Q1)", self.basis, param)
        self.CNOT = mc._basis_create_operation([('Q0', 'Q1')], "CNOT(Q0,Q1)", self.basis, param)
        self.CPHASE = mc._basis_create_operation([('Q0', 'Q1')], "CPHASE(Q0,Q1)", self.basis, param)

    def test_construct_gates_static(self):
        self._construct_gates('static')

    def test_construct_gates_TP(self):
        self._construct_gates('TP')

    def test_construct_gates_full(self):
        self._construct_gates('full')

        self.leakA = mc._basis_create_operation([('L0',), ('L1',), ('L2',)],
                                        "LX(pi,0,1)", self.basis, 'full')
        self.rotLeak = mc._basis_create_operation([('Q0',), ('L0',)],
                                          "X(pi,Q0):LX(pi,0,2)", self.basis, 'full')
        self.leakB = mc._basis_create_operation([('Q0',), ('L0',)], "LX(pi,0,2)", self.basis, 'full')
        self.rotXb = mc._basis_create_operation([('Q0',), ('L0',), ('L1',)],
                                        "X(pi,Q0)", self.basis, 'full')
        self.CnotB = mc._basis_create_operation([('Q0', 'Q1'), ('L0',)], "CX(pi,Q0,Q1)", self.basis, 'full')

    def _test_leakA(self):
        leakA_ans = np.array([[0., 1., 0.],
                              [1., 0., 0.],
                              [0., 0., 1.]], 'd')
        self.assertArraysAlmostEqual(self.leakA, leakA_ans)

    def _test_rotXa(self):
        rotXa_ans = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.],
                              [0., 0., 0, -1.],
                              [0., 0., 1., 0]], 'd')
        self.assertArraysAlmostEqual(self.rotXa, rotXa_ans)

    def _test_rotX2(self):
        rotX2_ans = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.],
                              [0., 0., -1., 0.],
                              [0., 0., 0., -1.]], 'd')
        self.assertArraysAlmostEqual(self.rotX2, rotX2_ans)

    def _test_rotLeak(self):
        rotLeak_ans = np.array([[0.5, 0., 0., -0.5, 0.70710678],
                                [0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0.],
                                [0.5, 0., 0., -0.5, -0.70710678],
                                [0.70710678, 0., 0., 0.70710678, 0.]], 'd')
        self.assertArraysAlmostEqual(self.rotLeak, rotLeak_ans)

    def _test_leakB(self):
        leakB_ans = np.array([[0.5, 0., 0., -0.5, 0.70710678],
                              [0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0.],
                              [-0.5, 0., 0., 0.5, 0.70710678],
                              [0.70710678, 0., 0., 0.70710678, 0.]], 'd')
        self.assertArraysAlmostEqual(self.leakB, leakB_ans)

    def _test_rotXb(self):
        rotXb_ans = np.array([[1., 0., 0., 0., 0., 0.],
                              [0., 1., 0., 0., 0., 0.],
                              [0., 0., -1., 0., 0., 0.],
                              [0., 0., 0., -1., 0., 0.],
                              [0., 0., 0., 0., 1., 0.],
                              [0., 0., 0., 0., 0., 1.]], 'd')
        self.assertArraysAlmostEqual(self.rotXb, rotXb_ans)

    def _test_CnotA(self):
        CnotA_ans = np.array([[1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0],
                              [0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.assertArraysAlmostEqual(self.CnotA, CnotA_ans)

    def _test_CnotB(self):
        CnotB_ans = np.array([[1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0],
                              [0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]])
        self.assertArraysAlmostEqual(self.CnotB, CnotB_ans)

    def test_raises_on_bad_basis(self):
        with self.assertRaises(AssertionError):
            mc._basis_create_operation([('Q0',)], "X(pi/2,Q0)", "FooBar")

    def test_raises_on_bad_gate_name(self):
        with self.assertRaises(ValueError):
            mc._basis_create_operation([('Q0',)], "FooBar(Q0)", self.basis)

    def test_raises_on_bad_state_spec(self):
        with self.assertRaises(ValueError):
            mc._basis_create_operation([('A0',)], "I(Q0)", self.basis)

    def test_raises_on_bad_label(self):
        with self.assertRaises(KeyError):
            mc._basis_create_operation([('Q0', 'L0')], "I(Q0,A0)", self.basis)

    def test_raises_on_qubit_state_space_mismatch(self):
        with self.assertRaises(ValueError):
            mc._basis_create_operation([('Q0',), ('Q1',)], "CZ(pi,Q0,Q1)", self.basis)

    def test_raises_on_LX_with_bad_basis_spec(self):
        with self.assertRaises(AssertionError):
            mc._basis_create_operation([('Q0',), ('L0',)], "LX(pi,0,2)", "foobar")


class PauliGateConstructionTester(GateConstructionBase, BaseCase):
    basis = 'pp'


class StdGateConstructionTester(GateConstructionBase, BaseCase):
    basis = 'std'

    def test_construct_gates_full(self):
        super(StdGateConstructionTester, self).test_construct_gates_full()
        self._test_leakA()


class GellMannGateConstructionTester(GateConstructionBase, BaseCase):
    basis = 'gm'

    def test_construct_gates_TP(self):
        super(GellMannGateConstructionTester, self).test_construct_gates_TP()
        self._test_rotXa()
        self._test_rotX2()

        self._test_CnotA()

    def test_construct_gates_static(self):
        super(GellMannGateConstructionTester, self).test_construct_gates_static()
        self._test_rotXa()
        self._test_rotX2()

        self._test_CnotA()

    def test_construct_gates_full(self):
        super(GellMannGateConstructionTester, self).test_construct_gates_full()
        self._test_leakA()
        self._test_rotXa()
        self._test_rotX2()

        self._test_rotLeak()
        self._test_leakB()
        self._test_rotXb()

        self._test_CnotA()
        self._test_CnotB()
