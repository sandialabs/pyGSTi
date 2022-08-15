import numpy as np
import scipy
import unittest
import itertools
from functools import reduce

import pygsti
from pygsti.modelpacks import smq1Q_XYI
from pygsti.modelpacks import smq2Q_XYICNOT
import pygsti.models.modelconstruction as mc
import pygsti.modelmembers.operations as op
import pygsti.modelmembers.states as st
import pygsti.tools.basistools as bt
from pygsti.processors.processorspec import QubitProcessorSpec as _ProcessorSpec
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel as GEEL
from ..util import BaseCase


def save_and_load_pspec(pspec):
    s = pspec.dumps()
    return pygsti.processors.QubitProcessorSpec.loads(s)


class ModelConstructionTester(BaseCase):
    def setUp(self):
        #OK for these tests, since we test user interface?
        #Set Model objects to "strict" mode for testing
        pygsti.models.ExplicitOpModel._strict = False

    def test_model_states(self):
        mdl1Q = smq1Q_XYI.target_model()
        E0, E1 = mdl1Q.povms['Mdefault']['0'].to_dense(), mdl1Q.povms['Mdefault']['1'].to_dense()

        mdl2Q = smq2Q_XYICNOT.target_model()
        E01 = mdl2Q.povms['Mdefault']['01']

        self.assertArraysAlmostEqual(np.kron(E0, E1), E01.to_dense())

    def test_create_spam_vector(self):
        def multikron(args):
            return reduce(np.kron, args)

        mdl1Q = smq1Q_XYI.target_model()
        mdlE0, mdlE1 = mdl1Q.povms['Mdefault']['0'].to_dense(), mdl1Q.povms['Mdefault']['1'].to_dense()
        ss1Q = pygsti.baseobjs.statespace.QubitSpace(1)
        E0 = mc.create_spam_vector(0, ss1Q, 'pp').flatten()
        E1 = mc.create_spam_vector(1, ss1Q, 'pp').flatten()

        self.assertArraysAlmostEqual(mdlE0, E0)
        self.assertArraysAlmostEqual(mdlE1, E1)

        E0_chk = st.create_from_pure_vector(np.array([1, 0]), 'static', 'pp', 'default', state_space=ss1Q).to_dense()
        E1_chk = st.create_from_pure_vector(np.array([0, 1]), 'static', 'pp', 'default', state_space=ss1Q).to_dense()
        self.assertArraysAlmostEqual(E0_chk, E0)
        self.assertArraysAlmostEqual(E1_chk, E1)

        nQubits = 4
        ssNQ = pygsti.baseobjs.statespace.QubitSpace(nQubits)
        E = {'0': E0, '1': E1}
        for i in range(2**nQubits):
            bin_i = '{{0:0{}b}}'.format(nQubits).format(i)  # first .format creates format str, e.g. '{0:04b}'
            print("Testing state %d (%s)" % (i, bin_i))
            created = mc.create_spam_vector(i, ssNQ, 'pp').flatten()
            krond = multikron([E[digit] for digit in bin_i])
            v = np.zeros(2**nQubits, complex); v[i] = 1.0
            alt_created = st.create_from_pure_vector(v, 'static', 'pp', 'default', state_space=ssNQ).to_dense()
            self.assertArraysAlmostEqual(created, krond)
            self.assertArraysAlmostEqual(created, alt_created)

    def test_build_basis_gateset(self):
        modelA = mc.create_explicit_model_from_expressions(
            [('Q0',)], ['Gi', 'Gx', 'Gy'],
            ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"]
        )
        modelB = mc._create_explicit_model_from_expressions(
            [('Q0',)], pygsti.baseobjs.Basis.cast('gm', 4),
            ['Gi', 'Gx', 'Gy'], ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"]
        )
        self.assertAlmostEqual(modelA.frobeniusdist(modelB), 0)

    def test_build_model(self):
        model1 = pygsti.models.ExplicitOpModel(['Q0'])
        model1['rho0'] = mc.create_spam_vector("0", model1.state_space, model1.basis)
        model1['Mdefault'] = pygsti.modelmembers.povms.UnconstrainedPOVM(
            [('0', mc.create_spam_vector("0", model1.state_space, model1.basis)),
             ('1', mc.create_spam_vector("1", model1.state_space, model1.basis))],
                                                                         evotype='default')
        model1['Gi'] = mc.create_operation("I(Q0)", model1.state_space, model1.basis)
        model1['Gx'] = mc.create_operation("X(pi/2,Q0)", model1.state_space, model1.basis)
        model1['Gy'] = mc.create_operation("Y(pi/2,Q0)", model1.state_space, model1.basis)
    
        model2 = mc.create_explicit_model_from_expressions(
            [('Q0',)], ['Gi', 'Gx', 'Gy'],
            ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"]
        )

        self.assertAlmostEqual(model1.frobeniusdist(model2), 0)

    def test_build_explicit_model(self):
        model = mc.create_explicit_model_from_expressions([('Q0',)], ['Gi', 'Gx', 'Gy'], ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"])
        self.assertEqual(set(model.operations.keys()), set(['Gi', 'Gx', 'Gy']))
        self.assertAlmostEqual(sum(model.probabilities(('Gx', 'Gi', 'Gy')).values()), 1.0)
        self.assertEqual(model.num_params, 60)

        gateset2b = mc.create_explicit_model_from_expressions([('Q0',)], ['Gi', 'Gx', 'Gy'],
                                                              ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"],
                                                              effect_labels=['1', '0'])
        self.assertArraysAlmostEqual(model.effects['0'], gateset2b.effects['1'])
        self.assertArraysAlmostEqual(model.effects['1'], gateset2b.effects['0'])

        # This is slightly confusing. Single qubit rotations are always stored in "pp" basis internally
        # UPDATE: now this isn't even allowed, as the 'densitymx' type represents states as *real* vectors.
        #std_gateset = mc.create_explicit_model([('Q0',)], ['Gi', 'Gx', 'Gy'],
        #                                      ["I(Q0)", "X(pi/8,Q0)", "Y(pi/8,Q0)"],
        #                                      basis="std")

        pp_gateset = mc.create_explicit_model_from_expressions([('Q0',)], ['Gi', 'Gx', 'Gy'],
                                                               ["I(Q0)", "X(pi/8,Q0)", "Y(pi/8,Q0)"],
                                                               basis="pp")

        #for op in ['Gi', 'Gx', 'Gy']:
        #    self.assertArraysAlmostEqual(std_gateset[op], pp_gateset[op])

    def test_build_crosstalk_free_model(self):
        nQubits = 2

        pspec = _ProcessorSpec(nQubits, ('Gi', 'Gx', 'Gy', 'Gcnot'), geometry='line')

        mdl = mc.create_crosstalk_free_model(
            pspec,
            ensure_composed_gates=True,
            independent_gates=False,
            independent_spam=False
        )
        assert(set(mdl.operation_blks['gates'].keys()) == set(["Gi", "Gx", "Gy", "Gcnot"]))
        assert(set(mdl.operation_blks['layers'].keys()) == set(
            [('Gi', 0), ('Gi', 1), ('Gx', 0), ('Gx', 1), ('Gy', 0), ('Gy', 1), ('Gcnot', 0, 1), ('Gcnot', 1, 0), '{auto_global_idle}']))
        self.assertEqual(mdl.num_params, 0)

        addlErr = pygsti.modelmembers.operations.FullTPOp(np.identity(4, 'd'))  # adds 12 params
        addlErr2 = pygsti.modelmembers.operations.FullTPOp(np.identity(4, 'd'))  # adds 12 params

        mdl.operation_blks['gates']['Gi'].append(addlErr)
        mdl.operation_blks['gates']['Gx'].append(addlErr)
        mdl.operation_blks['gates']['Gy'].append(addlErr2)

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
        pspec = _ProcessorSpec(nQubits, ('Gx', 'Gy', 'Gcnot', 'Gidle'), qubit_labels=['qb{}'.format(i) for i in range(nQubits)],
                               geometry='line')
        cfmdl = mc.create_crosstalk_free_model(
            pspec,
            depolarization_strengths={'Gx': 0.1, 'Gidle': 0.01, 'prep': 0.01, 'povm': 0.01},
            stochastic_error_probs={'Gy': (0.02, 0.02, 0.02)},
            lindblad_error_coeffs={
                'Gcnot': {('H', 'ZZ'): 0.01, ('S', 'IX'): 0.01},
            },
            ensure_composed_gates=False, independent_gates=True, independent_spam=True,
            ideal_spam_type="computational")

        self.assertEqual(cfmdl.num_params, 17)

        # Case: ensure_composed_gates=True, independent_gates=False
        cfmdl2 = mc.create_crosstalk_free_model(
            pspec,
            depolarization_strengths={'Gx': 0.1, 'Gidle': 0.01, 'prep': 0.01, 'povm': 0.01},
            stochastic_error_probs={'Gy': (0.02, 0.02, 0.02)},
            lindblad_error_coeffs={
                'Gcnot': {('H', 'ZZ'): 0.01, ('S', 'IX'): 0.01},
             },
            ensure_composed_gates=True, independent_gates=False, independent_spam=False)

        self.assertEqual(cfmdl2.num_params, 9)

        # Same as above but add ('Gx','qb0') to test giving qubit-specific error rates
        cfmdl3 = mc.create_crosstalk_free_model(
            pspec,
            depolarization_strengths={'Gx': 0.1, ('Gx', 'qb0'): 0.2, 'Gidle': 0.01, 'prep': 0.01, 'povm': 0.01},
            stochastic_error_probs={'Gy': (0.02, 0.02, 0.02)},
            lindblad_error_coeffs={
                'Gcnot': {('H', 'ZZ'): 0.01, ('S', 'IX'): 0.01},
             },
            ensure_composed_gates=True, independent_gates=False, independent_spam=False)

        self.assertEqual(cfmdl3.num_params, 10)

    def test_build_crosstalk_free_model_depolarize_parameterizations(self):
        nQubits = 2
        pspec = _ProcessorSpec(nQubits, ('Gi',), geometry='line')

        # Test depolarizing
        mdl_depol1 = mc.create_crosstalk_free_model(
            pspec, depolarization_strengths={'Gi': 0.1},
            ideal_spam_type="tensor product static"
        )
        Gi_op = mdl_depol1.operation_blks['gates']['Gi']
        self.assertTrue(isinstance(Gi_op, op.ComposedOp))
        self.assertTrue(isinstance(Gi_op.factorops[0], op.StaticStandardOp))
        self.assertTrue(isinstance(Gi_op.factorops[1], op.DepolarizeOp))
        self.assertEqual(mdl_depol1.num_params, 1)

        # Expand into StochasticNoiseOp
        mdl_depol2 = mc.create_crosstalk_free_model(
            pspec, depolarization_strengths={'Gi': 0.1},
            depolarization_parameterization='stochastic'
        )
        Gi_op = mdl_depol2.operation_blks['gates']['Gi']
        self.assertTrue(isinstance(Gi_op, op.ComposedOp))
        self.assertTrue(isinstance(Gi_op.factorops[0], op.StaticStandardOp))
        self.assertTrue(isinstance(Gi_op.factorops[1], op.StochasticNoiseOp))
        self.assertEqual(mdl_depol2.num_params, 3) 

        # Use LindbladOp with "depol", "diagonal" param
        mdl_depol3 = mc.create_crosstalk_free_model(
            pspec, depolarization_strengths={'Gi': 0.1},
            depolarization_parameterization='lindblad'
        )
        Gi_op = mdl_depol3.operation_blks['gates']['Gi']
        self.assertTrue(isinstance(Gi_op, op.ComposedOp))
        self.assertEqual(mdl_depol3.num_params, 1)

        mdl_prep1 = mc.create_crosstalk_free_model(
            pspec, depolarization_strengths={'Gi': 0.1, 'prep': 0.1},
            depolarization_parameterization='depolarize',
            independent_spam=False
        )
        rho0 = mdl_prep1.prep_blks['layers']['rho0']
        self.assertTrue(isinstance(rho0, pygsti.modelmembers.states.ComposedState))
        self.assertEqual(mdl_prep1.num_params, 2)
    
        mdl_prep2 = mc.create_crosstalk_free_model(
            pspec, depolarization_strengths={'Gi': 0.1, 'prep': 0.1},
            depolarization_parameterization='stochastic',
            independent_spam=False
        )
        rho0 = mdl_prep2.prep_blks['layers']['rho0']
        self.assertTrue(isinstance(rho0, pygsti.modelmembers.states.ComposedState))
        self.assertEqual(mdl_prep2.num_params, 6)
    
        mdl_povm1 = mc.create_crosstalk_free_model(
            pspec, depolarization_strengths={'Gi': 0.1, 'povm': 0.1},
            depolarization_parameterization='depolarize',
            independent_spam=False
        )
        Mdefault = mdl_povm1.povm_blks['layers']['Mdefault']
        self.assertTrue(isinstance(Mdefault, pygsti.modelmembers.povms.ComposedPOVM))
        self.assertEqual(mdl_povm1.num_params, 2)
    
        mdl_povm2 = mc.create_crosstalk_free_model(
            pspec, depolarization_strengths={'Gi': 0.1, 'povm': 0.1},
            depolarization_parameterization='stochastic',
            independent_spam=False
        )
        Mdefault = mdl_povm2.povm_blks['layers']['Mdefault']
        self.assertTrue(isinstance(Mdefault, pygsti.modelmembers.povms.ComposedPOVM))
        self.assertEqual(mdl_povm2.num_params, 6)

    def test_build_crosstalk_free_model_stochastic_parameterizations(self):
        nQubits = 2
        pspec = _ProcessorSpec(nQubits, ('Gi',), geometry='line')

        # Test stochastic
        mdl_sto1 = mc.create_crosstalk_free_model(
            pspec, stochastic_error_probs={'Gi': (0.1, 0.1, 0.1)},
            ideal_spam_type="tensor product static",
            independent_spam=False
        )
        Gi_op = mdl_sto1.operation_blks['gates']['Gi']
        self.assertTrue(isinstance(Gi_op, op.ComposedOp))
        self.assertTrue(isinstance(Gi_op.factorops[0], op.StaticStandardOp))
        self.assertTrue(isinstance(Gi_op.factorops[1], op.StochasticNoiseOp))
        self.assertEqual(mdl_sto1.num_params, 3)

        # Use LindbladOp with "cptp", "diagonal" param
        mdl_sto3 = mc.create_crosstalk_free_model(
            pspec, stochastic_error_probs={'Gi': (0.1, 0.1, 0.1)},
            stochastic_parameterization='lindblad',
            independent_spam=False
        )
        Gi_op = mdl_sto3.operation_blks['gates']['Gi']
        self.assertTrue(isinstance(Gi_op, op.ComposedOp))
        self.assertEqual(mdl_sto3.num_params, 3)

        mdl_prep1 = mc.create_crosstalk_free_model(
            pspec, stochastic_error_probs={'Gi': (0.1, 0.1, 0.1), 'prep': (0.01,)*3},
            stochastic_parameterization='stochastic',
            independent_spam=False
        )
        rho0 = mdl_prep1.prep_blks['layers']['rho0']
        self.assertTrue(isinstance(rho0, pygsti.modelmembers.states.ComposedState))
        self.assertEqual(mdl_prep1.num_params, 6)

        mdl_povm1 = mc.create_crosstalk_free_model(
            pspec, stochastic_error_probs={'Gi': (0.1,)*3, 'povm': (0.01,)*3},
            stochastic_parameterization='stochastic',
            independent_spam=False
        )
        Mdefault = mdl_povm1.povm_blks['layers']['Mdefault']
        self.assertTrue(isinstance(Mdefault, pygsti.modelmembers.povms.ComposedPOVM))
        self.assertEqual(mdl_povm1.num_params, 6)

    def test_build_crosstalk_free_model_lindblad_parameterizations(self):
        nQubits = 2
        pspec = _ProcessorSpec(nQubits, ('Gi',), geometry='line')

        # Test Lindblad
        mdl_lb1 = mc.create_crosstalk_free_model(
            pspec, lindblad_error_coeffs={'Gi': {('H', 'X'): 0.1, ('S', 'Y'): 0.1}},
            ideal_spam_type="tensor product static",
            independent_spam=False
        )
        Gi_op = mdl_lb1.operation_blks['gates']['Gi']
        self.assertTrue(isinstance(Gi_op, op.ComposedOp))
        self.assertEqual(Gi_op.errorgen_coefficients(), {GEEL('H', ['X'], [0]): 0.1, GEEL('S', ['Y'], [0]): 0.1})
        self.assertEqual(mdl_lb1.num_params, 2)
    
        # Test param passthrough
        mdl_lb2 = mc.create_crosstalk_free_model(
            pspec, lindblad_error_coeffs={'Gi': {('H', 'X'): 0.1, ('S', 'Y'): 0.1}},
            lindblad_parameterization='H+S',
            independent_spam=False
        )
        Gi_op = mdl_lb2.operation_blks['gates']['Gi']
        self.assertTrue(isinstance(Gi_op, op.ComposedOp))
        self.assertEqual(Gi_op.errorgen_coefficients(), {GEEL('H', ['X'], [0]): 0.1, GEEL('S', ['Y'], [0]): 0.1})
        self.assertEqual(mdl_lb2.num_params, 2)

        mdl_prep1 = mc.create_crosstalk_free_model(
            pspec, lindblad_error_coeffs={
                'Gi': {('H', 'X'): 0.1, ('S', 'Y'): 0.1},
                'prep': {('H', 'Y'): 0.01}},
            ideal_spam_type='tensor product static',
        )
        rho0 = mdl_prep1.prep_blks['layers']['rho0']
        self.assertTrue(isinstance(rho0, pygsti.modelmembers.states.TensorProductState))
        self.assertEqual(mdl_prep1.num_params, 4)

        mdl_povm1 = mc.create_crosstalk_free_model(
            pspec, lindblad_error_coeffs={
                'Gi': {('H', 'X'): 0.1, ('S', 'Y'): 0.1},
                'povm': {('H', 'Y'): 0.01}},
            ideal_spam_type='tensor product static',
        )
        Mdefault = mdl_povm1.povm_blks['layers']['Mdefault']
        self.assertTrue(isinstance(Mdefault, pygsti.modelmembers.povms.TensorProductPOVM))
        self.assertEqual(mdl_povm1.num_params, 4)

        # Test Composed variants of prep/povm
        mdl_prep2 = mc.create_crosstalk_free_model(
            pspec, lindblad_error_coeffs={
                'Gi': {('H', 'X'): 0.1, ('S', 'Y'): 0.1},
                'prep': {('H', 'Y'): 0.01}},
            ideal_spam_type="computational",
            independent_spam=False
        )
        rho0 = mdl_prep2.prep_blks['layers']['rho0']
        self.assertTrue(isinstance(rho0, pygsti.modelmembers.states.ComposedState))
        self.assertEqual(mdl_prep2.num_params, 3)

        mdl_povm2 = mc.create_crosstalk_free_model(
            pspec, lindblad_error_coeffs={
                'Gi': {('H', 'X'): 0.1, ('S', 'Y'): 0.1},
                'povm': {('H', 'Y'): 0.01}},
            ideal_spam_type="computational",
            independent_spam=False
        )
        Mdefault = mdl_povm2.povm_blks['layers']['Mdefault']
        self.assertTrue(isinstance(Mdefault, pygsti.modelmembers.povms.ComposedPOVM))
        self.assertEqual(mdl_povm2.num_params, 3)

    def test_build_crosstalk_free_model_with_nonstd_gate_unitary_factory(self):
        nQubits = 2

        def fn(args):
            if args is None: args = (0,)
            a, = args
            sigmaZ = np.array([[1, 0], [0, -1]], 'd')
            return scipy.linalg.expm(1j * float(a) * sigmaZ)
        fn.udim = 2
        fn.shape = (2,2)

        pspec = _ProcessorSpec(nQubits, ('Gx', 'Gy', 'Gcnot', 'Ga'), nonstd_gate_unitaries={'Ga': fn}, geometry='line')
        cfmdl = mc.create_crosstalk_free_model(pspec)

        c = pygsti.circuits.Circuit("Gx:1Ga;0.3:1Gx:1@(0,1)")
        p = cfmdl.probabilities(c)

        self.assertAlmostEqual(p['00'], 0.08733219254516078)
        self.assertAlmostEqual(p['01'], 0.9126678074548386)
    
    def test_build_crosstalk_free_model_with_custom_gates(self):
        nQubits = 2

        def fn(args):
            if args is None: args = (0,)
            theta, = args
            sigmaX = np.array([[0, 1], [1, 0]], 'd')
            return scipy.linalg.expm(1j * float(theta) / 4 * sigmaX)
        fn.udim = 2
        fn.shape = (2,2)

        class XRotationOpFactory(pygsti.modelmembers.operations.OpFactory):
            def __init__(self):
                ss = pygsti.baseobjs.statespace.QubitSpace(1)
                pygsti.modelmembers.operations.OpFactory.__init__(self, state_space=ss, evotype="default")

            def create_object(self, args=None, sslbls=None):
                theta = float(args[0])/2.0
                b = 2*np.cos(theta)*np.sin(theta)
                c = np.cos(theta)**2 - np.sin(theta)**2
                superop = np.array([[1,   0,   0,   0],
                                    [0,   1,   0,   0],
                                    [0,   0,   c,  -b],
                                    [0,   0,   b,   c]],'d')
                return pygsti.modelmembers.operations.StaticArbitraryOp(superop, evotype=self.evotype,
                                                                        state_space=self.state_space)

        xrot_fact = XRotationOpFactory()

        pspec = _ProcessorSpec(nQubits, ('Gi', 'Gxr'), nonstd_gate_unitaries={'Gxr': fn},  geometry='line')
        cfmdl = mc.create_crosstalk_free_model(pspec, custom_gates={'Gxr': xrot_fact})

        c = pygsti.circuits.Circuit("Gxr;3.1415926536:1@(0,1)")
        p = cfmdl.probabilities(c)

        self.assertAlmostEqual(p['01'], 1.0)

        c = pygsti.circuits.Circuit("Gxr;1.5707963268:1@(0,1)")
        p = cfmdl.probabilities(c)
        
        self.assertAlmostEqual(p['00'], 0.5)
        self.assertAlmostEqual(p['01'], 0.5)

    def test_build_operation_raises_on_bad_parameterization(self):
        with self.assertRaises(ValueError):
            mc.create_operation("X(pi,Q0)", [('Q0', 'Q1')], "gm", parameterization="FooBar")

    def test_build_explicit_model_raises_on_bad_state(self):
        with self.assertRaises(ValueError):
            mc.create_explicit_model_from_expressions([('A0',)], ['Gi', 'Gx', 'Gy'],
                                                      ["I(Q0)", "X(pi/8,Q0)", "Y(pi/8,Q0)"])

    def test_build_explicit_model_raises_on_bad_basis(self):
        with self.assertRaises(AssertionError):
            mc.create_explicit_model_from_expressions([('Q0',)], ['Gi', 'Gx', 'Gy'],
                                                      ["I(Q0)", "X(pi/8,Q0)", "Y(pi/8,Q0)"],
                                                      basis="FooBar")

    def test_build_explicit_model_raises_on_bad_rho_expression(self):
        with self.assertRaises(ValueError):
            mc.create_explicit_model_from_expressions([('Q0',)], ['Gi', 'Gx', 'Gy'],
                                                      ["I(Q0)", "X(pi/8,Q0)", "Y(pi/8,Q0)"],
                                                      prep_labels=['rho0'], prep_expressions=["FooBar"], )

    def test_build_explicit_model_raises_on_bad_effect_expression(self):
        with self.assertRaises(ValueError):
            mc.create_explicit_model_from_expressions([('Q0',)], ['Gi', 'Gx', 'Gy'],
                                                      ["I(Q0)", "X(pi/8,Q0)", "Y(pi/8,Q0)"],
                                                      effect_labels=['0', '1'], effect_expressions=["FooBar", "1"])

    def test_default_spam_in_processorspecs(self):
        pspec_defaults = pygsti.processors.QubitProcessorSpec(4, ['Gxpi2', 'Gypi2'], geometry='line')
        pspec_defaults = save_and_load_pspec(pspec_defaults)  # make sure serialization works too

        self.assertEqual(pspec_defaults.prep_names, ('rho0',))
        self.assertEqual(pspec_defaults.povm_names, ('Mdefault',))

        mdl_default = pygsti.models.modelconstruction.create_crosstalk_free_model(pspec_defaults)

        all4Qoutcomes = list(map(lambda x: ''.join(x), itertools.product(['0','1'], repeat=4)))
        self.assertArraysAlmostEqual(mdl_default.prep_blks['layers']['rho0']._zvals, [0, 0, 0, 0])
        self.assertEqual(list(mdl_default.povm_blks['layers']['Mdefault'].keys()), all4Qoutcomes)

        mdl_default_explicit = pygsti.models.modelconstruction.create_explicit_model(pspec_defaults)
        self.assertArraysAlmostEqual(mdl_default_explicit.preps['rho0']._zvals, [0, 0, 0, 0])
        self.assertEqual(list(mdl_default_explicit.povms['Mdefault'].keys()), all4Qoutcomes)

        mdl_default_cloud = pygsti.models.modelconstruction.create_cloud_crosstalk_model(pspec_defaults)
        self.assertArraysAlmostEqual(mdl_default_cloud.prep_blks['layers']['rho0']._zvals, [0, 0, 0, 0])
        self.assertEqual(list(mdl_default_cloud.povm_blks['layers']['Mdefault'].keys()), all4Qoutcomes)

    def test_named_spam_in_processorspecs(self):
        pspec_names = pygsti.processors.QubitProcessorSpec(4, ['Gxpi2', 'Gypi2'], geometry='line',
                                                           prep_names=("rho1", "rho_1100"), povm_names=("Mz",))
        pspec_names = save_and_load_pspec(pspec_names)  # make sure serialization works too

        self.assertEqual(pspec_names.prep_names, ('rho1', 'rho_1100'))
        self.assertEqual(pspec_names.povm_names, ('Mz',))

        mdl_names = pygsti.models.modelconstruction.create_crosstalk_free_model(pspec_names)

        all4Qoutcomes = list(map(lambda x: ''.join(x), itertools.product(['0','1'], repeat=4)))
        self.assertArraysAlmostEqual(mdl_names.prep_blks['layers']['rho1']._zvals, [0, 0, 0, 1])
        self.assertArraysAlmostEqual(mdl_names.prep_blks['layers']['rho_1100']._zvals, [1, 1, 0, 0])
        self.assertEqual(list(mdl_names.povm_blks['layers']['Mz'].keys()), all4Qoutcomes)

    def test_spamvecs_in_processorspecs(self):
        prep_vec = np.zeros(2**4, complex)
        prep_vec[4] = 1.0
        EA = np.zeros(2**4, complex)
        EA[14] = 1.0
        EB = np.zeros(2**4, complex)
        EB[15] = 1.0

        pspec_vecs = pygsti.processors.QubitProcessorSpec(4, ['Gxpi2', 'Gypi2'], geometry='line',
                                                          prep_names=("rhoA", "rhoC"), povm_names=("Ma", "Mc"),
                                                          nonstd_preps={'rhoA': "rho0", 'rhoC': prep_vec},
                                                          nonstd_povms={'Ma': {'0': "0000", '1': EA},
                                                                        'Mc': {'OutA': "0000", 'OutB': [EA, EB]}})
        pspec_vecs = save_and_load_pspec(pspec_vecs)  # make sure serialization works too

        self.assertEqual(pspec_vecs.prep_names, ('rhoA', 'rhoC'))
        self.assertEqual(pspec_vecs.povm_names, ('Ma','Mc'))

        mdl_vecs = pygsti.models.modelconstruction.create_crosstalk_free_model(pspec_vecs, ideal_spam_type='full TP')

        from pygsti.tools import state_to_dmvec, change_basis
        prep_supervec = change_basis(state_to_dmvec(prep_vec), 'std', 'pp')
        prep0_vec = np.zeros(2**4, complex); prep0_vec[0] = 1.0
        prep0_supervec = change_basis(state_to_dmvec(prep0_vec), 'std', 'pp')

        self.assertArraysAlmostEqual(mdl_vecs.prep_blks['layers']['rhoA'].to_dense(), prep0_supervec)
        self.assertArraysAlmostEqual(mdl_vecs.prep_blks['layers']['rhoC'].to_dense(), prep_supervec)

        self.assertEqual(list(mdl_vecs.povm_blks['layers']['Ma'].keys()), ['0', '1'])
        self.assertEqual(list(mdl_vecs.povm_blks['layers']['Mc'].keys()), ['OutA', 'OutB'])

        def Zeffect(index):
            v = np.zeros(2**4, complex); v[index] = 1.0
            return change_basis(state_to_dmvec(v), 'std', 'pp')

        self.assertArraysAlmostEqual(mdl_vecs.povm_blks['layers']['Ma']['0'].to_dense(), Zeffect(0))
        self.assertArraysAlmostEqual(mdl_vecs.povm_blks['layers']['Ma']['1'].to_dense(), Zeffect(14))

        self.assertArraysAlmostEqual(mdl_vecs.povm_blks['layers']['Mc']['OutA'].to_dense(), Zeffect(0))
        self.assertArraysAlmostEqual(mdl_vecs.povm_blks['layers']['Mc']['OutB'].to_dense(), Zeffect(14) + Zeffect(15))

    def test_instruments_in_processorspecs(self):
        #Instruments
        pspec_with_instrument = pygsti.processors.QubitProcessorSpec(4, ['Gxpi2', 'Gypi2'], geometry='line',
                                                                     instrument_names=('Iz',))
        mdl_default_explicit = pygsti.models.modelconstruction.create_explicit_model(pspec_with_instrument)

        all4Qoutcomes = list(map(lambda x: ''.join(x), itertools.product(['0','1'], repeat=4)))
        self.assertEqual(list(mdl_default_explicit.instruments['Iz'].keys()), all4Qoutcomes)

        pspec_with_instrument2 = pygsti.processors.QubitProcessorSpec(
            2, ['Gxpi2', 'Gypi2'], geometry='line', instrument_names=('Iparity',),
            nonstd_instruments={'Iparity': {'plus': [('00', '00'), ('11','11')],
                                            'minus': [('10', '10'), ('01','01')]}})
        mdl_default_explicit2 = pygsti.models.modelconstruction.create_explicit_model(pspec_with_instrument2)
        self.assertEqual(list(mdl_default_explicit2.instruments['Iparity']), ['plus', 'minus'])

        from pygsti.tools import state_to_dmvec, change_basis

        def Ivec(index):
            v = np.zeros(2**2, complex); v[index] = 1.0
            return v

        def Isupervec(index):
            return change_basis(state_to_dmvec(Ivec(index)), 'std', 'pp')

        sv00 = Isupervec(0)
        sv01 = Isupervec(1)
        sv10 = Isupervec(2)
        sv11 = Isupervec(3)
        Iplus = np.outer(sv00, sv00) + np.outer(sv11, sv11)
        Iminus = np.outer(sv01, sv01) + np.outer(sv10, sv10)
        Itot = Iplus + Iminus
        self.assertAlmostEqual(Itot[0,0], 1.0)
        self.assertArraysAlmostEqual(mdl_default_explicit2.instruments['Iparity']['plus'].to_dense(), Iplus)
        self.assertArraysAlmostEqual(mdl_default_explicit2.instruments['Iparity']['minus'].to_dense(), Iminus)
        
        
        # In[14]:
        
        
        v00 = Ivec(0)
        v01 = Ivec(1)
        v10 = Ivec(2)
        v11 = Ivec(3)
        
        pspec_with_instrument3 = pygsti.processors.QubitProcessorSpec(
            2, ['Gxpi2', 'Gypi2'], geometry='line', instrument_names=('Iparity',),
            nonstd_instruments={'Iparity': {'plus': [(v00, v00), (v11,v11)],
                                            'minus': [(v10, v10), (v01,v01)]}})
        mdl_default_explicit3 = pygsti.models.modelconstruction.create_explicit_model(pspec_with_instrument3)
        self.assertEqual(list(mdl_default_explicit3.instruments['Iparity']), ['plus', 'minus'])
        
        self.assertArraysAlmostEqual(mdl_default_explicit3.instruments['Iparity']['plus'].to_dense(), Iplus)
        self.assertArraysAlmostEqual(mdl_default_explicit3.instruments['Iparity']['minus'].to_dense(), Iminus)






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
        self.ident = mc.create_operation("I(Q0)", [('Q0',)], self.basis, param)
        self.rotXa = mc.create_operation("X(pi/2,Q0)", [('Q0',)], self.basis, param)
        self.rotX2 = mc.create_operation("X(pi,Q0)", [('Q0',)], self.basis, param)
        self.rotYa = mc.create_operation("Y(pi/2,Q0)", [('Q0',)], self.basis, param)
        self.rotZa = mc.create_operation("Z(pi/2,Q0)", [('Q0',)], self.basis, param)
        self.rotNa = mc.create_operation("N(pi/2,1.0,0.5,0,Q0)", [('Q0',)], self.basis, param)
        self.iwL = mc.create_operation("I(Q0)", [('Q0', 'L0')], self.basis, param)
        self.CnotA = mc.create_operation("CX(pi,Q0,Q1)", [('Q0', 'Q1')], self.basis, param)
        self.CY = mc.create_operation("CY(pi,Q0,Q1)", [('Q0', 'Q1')], self.basis, param)
        self.CZ = mc.create_operation("CZ(pi,Q0,Q1)", [('Q0', 'Q1')], self.basis, param)
        self.CNOT = mc.create_operation("CNOT(Q0,Q1)", [('Q0', 'Q1')], self.basis, param)
        self.CPHASE = mc.create_operation("CPHASE(Q0,Q1)", [('Q0', 'Q1')], self.basis, param)

    def test_construct_gates_static(self):
        self._construct_gates('static')

    def test_construct_gates_TP(self):
        self._construct_gates('full TP')

    @unittest.skip("Need to fix default state space to work with non-square dims!")
    def test_construct_gates_full(self):
        self._construct_gates('full')

        self.leakA = mc.create_operation("LX(pi,0,1)", [('L0',), ('L1',), ('L2',)], self.basis, 'full')
        self.rotLeak = mc.create_operation("X(pi,Q0):LX(pi,0,2)", [('Q0',), ('L0',)], self.basis, 'full')
        self.leakB = mc.create_operation("LX(pi,0,2)", [('Q0',), ('L0',)], self.basis, 'full')
        self.rotXb = mc.create_operation("X(pi,Q0)", [('Q0',), ('L0',), ('L1',)], self.basis, 'full')
        self.CnotB = mc.create_operation("CX(pi,Q0,Q1)", [('Q0', 'Q1'), ('L0',)], self.basis, 'full')

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
            mc.create_operation("X(pi/2,Q0)", [('Q0',)], "FooBar")

    def test_raises_on_bad_gate_name(self):
        with self.assertRaises(ValueError):
            mc.create_operation("FooBar(Q0)", [('Q0',)], self.basis)

    def test_raises_on_bad_state_spec(self):
        with self.assertRaises(ValueError):
            mc.create_operation("I(Q0)", [('A0',)], self.basis)

    def test_raises_on_bad_label(self):
        with self.assertRaises(KeyError):
            mc.create_operation("I(Q0,A0)", [('Q0', 'L0')], self.basis)

    def test_raises_on_qubit_state_space_mismatch(self):
        with self.assertRaises(ValueError):
            mc.create_operation("CZ(pi,Q0,Q1)", [('Q0',), ('Q1',)], self.basis)

    def test_raises_on_LX_with_bad_basis_spec(self):
        with self.assertRaises(AssertionError):
            mc.create_operation("LX(pi,0,2)", [('Q0',), ('L0',)], "foobar")


class PauliGateConstructionTester(GateConstructionBase, BaseCase):
    basis = 'pp'


class StdGateConstructionTester(GateConstructionBase, BaseCase):
    basis = 'std'

    def test_construct_gates_full(self):
        super(StdGateConstructionTester, self).test_construct_gates_full()
        self._test_leakA()

    @unittest.skip("Cannot parameterize as TP using std basis (TP requires *real* op mxs)")
    def test_construct_gates_TP(self):
        pass


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
