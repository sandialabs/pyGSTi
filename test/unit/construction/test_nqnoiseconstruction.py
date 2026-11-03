import numpy as np
import scipy

import pygsti.models.modelconstruction as mc
import pygsti.circuits.cloudcircuitconstruction as cc
from pygsti.circuits import Circuit
from pygsti.processors import QubitProcessorSpec as _QubitProcessorSpec
from ..util import BaseCase


class KCoverageTester(BaseCase):
    def test_kcoverage(self):
        # TODO optimize
        n = 10  # nqubits
        k = 4  # number of "labels" needing distribution
        rows = cc.create_kcoverage_template(n, k, verbosity=2)
        cc._check_kcoverage_template(rows, n, k, verbosity=1)


class NQNoiseConstructionTester(BaseCase):
    def test_build_cloud_crosstalk_model(self):
        nQubits = 2
        pspec = _QubitProcessorSpec(nQubits, ('Gx', 'Gy', 'Gcnot', '{idle}'), qubit_labels=['qb{}'.format(i) for i in range(nQubits)], geometry='line')
        pspec2 = _QubitProcessorSpec(nQubits, ('Gx', 'Gy', 'Gcnot', '{idle}'), geometry='line')  # just integer qubit labels

        ccmdl1 = mc.create_cloud_crosstalk_model(
            pspec,
            lindblad_error_coeffs={('Gx', 'qb0'): {('H', 'X'): 0.01, ('S', 'XY:qb0,qb1'): 0.01},
             ('Gcnot', 'qb0', 'qb1'): {('H', 'ZZ'): 0.02, ('S', 'XX:qb0,qb1'): 0.02},
             '{idle}': {('S', 'XX:qb0,qb1'): 0.01},
             'prep': {('S', 'XX:qb0,qb1'): 0.01},
             'povm': {('S', 'XX:qb0,qb1'): 0.01}
             })        
        self.assertEqual(ccmdl1.num_params, 7)

        #Using sparse=True and a map-based simulator
        ccmdl1 = mc.create_cloud_crosstalk_model(
            pspec,
            lindblad_error_coeffs={('Gx', 'qb0'): {('H', 'X'): 0.01, ('S', 'XY:qb0,qb1'): 0.01},
             ('Gcnot', 'qb0', 'qb1'): {('H', 'ZZ'): 0.02, ('S', 'XX:qb0,qb1'): 0.02},
             '{idle}': {('S', 'XX:qb0,qb1'): 0.01},
             'prep': {('S', 'XX:qb0,qb1'): 0.01},
             'povm': {('S', 'XX:qb0,qb1'): 0.01}
             },
            simulator="map")
        self.assertEqual(ccmdl1.num_params, 7)


        #Using compact notation:
        ccmdl2 = mc.create_cloud_crosstalk_model(
            pspec2,
            lindblad_error_coeffs={'Gx:0': {('HX'): 0.01, 'SXY:0,1': 0.01},
             'Gcnot:0:1': {'HZZ': 0.02, 'SXX:0,1': 0.02},
             '{idle}': {'SXX:0,1': 0.01},
             'prep': {'SXX:0,1': 0.01},
             'povm': {'SXX:0,1': 0.01}
             })
        self.assertEqual(ccmdl2.num_params, 7)

        #also using qubit_labels
        ccmdl3 = mc.create_cloud_crosstalk_model(
            pspec,
            lindblad_error_coeffs={'Gx:qb0': {('HX'): 0.01, 'SXY:qb0,qb1': 0.01},
             'Gcnot:qb0:qb1': {'HZZ': 0.02, 'SXX:qb0,qb1': 0.02},
             '{idle}': {'SXX:qb0,qb1': 0.01},
             'prep': {'SXX:qb0,qb1': 0.01},
             'povm': {'SXX:qb0,qb1': 0.01}
             })
        self.assertEqual(ccmdl3.num_params, 7)

        with self.assertRaises(ValueError):  # cannot use non-lindblad errors with "errorgens" composition type
            mc.create_cloud_crosstalk_model(
                pspec2, errcomp_type="errorgens",
                depolarization_strengths={'Gx': 0.15}
            )
        with self.assertRaises(ValueError):
            mc.create_cloud_crosstalk_model(
                pspec2, errcomp_type="errorgens",
                stochastic_error_probs={'Gx': (0.01,)*15}
            )

    def test_build_cloud_crosstalk_model_stencils(self):
        nQubits = 2
        pspec = _QubitProcessorSpec(nQubits, ('Gx', 'Gy', 'Gcnot', 'Gidle'), qubit_labels=['qb{}'.format(i) for i in range(nQubits)],
                                    geometry='line')

        ccmdl1 = mc.create_cloud_crosstalk_model(
            pspec,
            lindblad_error_coeffs={'Gx': {('H', 'X'): 0.01, ('S', 'X:@0+left'): 0.01},  # ('S','XX:@1+right,@0+left'): 0.02
             'Gcnot': {('H', 'ZZ'): 0.02, ('S', 'XX:@1+right,@0+left'): 0.02},
             'Gidle': {('S', 'XX:qb0,qb1'): 0.01}
             })
        self.assertEqual(ccmdl1.num_params, 5)

        #Using compact notation:
        ccmdl2 = mc.create_cloud_crosstalk_model(
            pspec,
            lindblad_error_coeffs={'Gx': {'HX': 0.01, 'SX:@0+left': 0.01},  # ('S','XX:@1+right,@0+left'): 0.02
             'Gcnot': {'HZZ': 0.02, 'SXX:@1+right,@0+left': 0.02},
             'Gidle': {'SXX:qb0,qb1': 0.01}
             })
        self.assertEqual(ccmdl2.num_params, 5)

    def test_build_cloud_crosstalk_model_indepgates(self):
        #Same as test_cloud_crosstalk_stencils case but set independent_gates=True
        nQubits = 2
        pspec = _QubitProcessorSpec(nQubits, ('Gx', 'Gy', 'Gcnot', 'Gidle'), qubit_labels=['qb{}'.format(i) for i in range(nQubits)],
                                    geometry='line')

        ccmdl1 = mc.create_cloud_crosstalk_model(
            pspec,
            lindblad_error_coeffs={'Gx': {('H', 'X'): 0.01, ('S', 'X:@0+left'): 0.01},  # ('S','XX:@1+right,@0+left'): 0.02
             'Gcnot': {('H', 'ZZ'): 0.02, ('S', 'XX:@1+right,@0+left'): 0.02},
             'Gidle': {('S', 'XX:qb0,qb1'): 0.01}
             }, independent_gates=True)
        self.assertEqual(ccmdl1.num_params, 7)

    def test_build_cloud_crosstalk_model_with_nonstd_gate_unitary_factory(self):
        nQubits = 2

        def fn(args):
            if args is None: args = (0,)
            a, = args
            sigmaZ = np.array([[1, 0], [0, -1]], 'd')
            return scipy.linalg.expm(1j * float(a) * sigmaZ)
        fn.udim = 2
        fn.shape = (2,2)

        pspec = _QubitProcessorSpec(nQubits, ('Gx', 'Gy', 'Gcnot', 'Ga'), nonstd_gate_unitaries={'Ga': fn}, geometry='line')
        ccmdl = mc.create_cloud_crosstalk_model(pspec)
        c = Circuit("Gx:1Ga;0.3:1Gx:1@(0,1)")
        p1 = ccmdl.probabilities(c)

        self.assertAlmostEqual(p1['00'], 0.08733219254516078)
        self.assertAlmostEqual(p1['01'], 0.9126678074548386)


class CloudNoiseFirstOrderJacobianTester(BaseCase):
    """ Tests of the (non-term-simulator) first-order machinery used by create_cloudnoise_circuits """

    def test_linearized_lindblad_parameterization(self):
        lin = cc._linearized_lindblad_parameterization
        self.assertEqual(lin("H+S"), "H+s")
        self.assertEqual(lin("H+s"), "H+s")
        self.assertEqual(lin("H+D"), "H+d")
        self.assertEqual(lin("S"), "s")
        self.assertEqual(lin("CPTPLND"), "GLND")
        self.assertEqual(lin("GLND"), "GLND")
        self.assertEqual(lin("exp(H+S)"), "exp(H+s)")
        with self.assertRaises(ValueError):
            lin("full TP")

    def test_first_order_jacobian_matches_finite_differences(self):
        from pygsti.forwardsims import MatrixForwardSimulator
        from pygsti.baseobjs import Label
        pspec = _QubitProcessorSpec(2, ('Gx', 'Gy', 'Gcnot', '{idle}'), geometry='line')

        def build(param):
            return mc.create_cloud_crosstalk_model_from_hops_and_weights(
                pspec, None, 1, 0, 1, 0, 0, simulator=MatrixForwardSimulator(), gate_type=param,
                spam_type=param, errcomp_type="gates", implicit_idle_mode="add_global")

        orig_mdl = build("H+S")  # stochastic rates are *quadratic* in the parameters
        jac_calc = cc._FirstOrderProbabilityJacobian(orig_mdl, build("H+s"))
        Np = jac_calc.num_params
        self.assertEqual(Np, orig_mdl.num_params)

        rng = np.random.default_rng(1234)
        theta0 = rng.uniform(0.5, 1.5, Np)
        wrt = np.arange(0, Np, 2)
        transform = jac_calc.wrt_param_transform(theta0, wrt)
        c1 = Circuit([Label('Gx', 0), Label('Gcnot', (0, 1)), Label('Gy', 1), ()], line_labels=(0, 1))
        c0 = Circuit([Label('Gx', 0), Label('Gy', 1)], line_labels=(0, 1))
        J, = jac_calc.amplified_jacobians([(c1, c0)], transform)

        # J = (dp/dtheta_lin at zero error) . (chain-rule transform evaluated at theta0)
        lin_d = jac_calc.dprobs([c1], transform[0])[0] - jac_calc.dprobs([c0], transform[0])[0]
        G = jac_calc.param_transform(theta0)[transform[0], :][:, wrt]
        self.assertArraysAlmostEqual(J.real, lin_d @ G)
        self.assertTrue(np.linalg.norm(J) > 0)

        # finite-difference check of the zero-error derivatives of the linear model
        lin_mdl = jac_calc.linear_model
        eps = 1e-6
        full = jac_calc.dprobs([c1])[0]
        for j in rng.choice(Np, 5, replace=False):
            v = np.zeros(Np); v[j] = eps
            lin_mdl.from_vector(v); pp = lin_mdl.probabilities(c1)
            lin_mdl.from_vector(-v); pm = lin_mdl.probabilities(c1)
            fd = [(pp[o] - pm[o]) / (2 * eps) for o in jac_calc.outcomes]
            self.assertArraysAlmostEqual(full[:, j], fd, places=5)
        lin_mdl.from_vector(np.zeros(Np))
