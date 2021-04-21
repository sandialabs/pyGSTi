import scipy
import numpy as np

from ..util import BaseCase

from pygsti.objects import DataSet, Circuit
from pygsti.modelpacks.legacy import std1Q_XYI, std2Q_XXYYII, std2Q_XYICNOT
import pygsti.construction.nqnoiseconstruction as nc


class KCoverageTester(BaseCase):
    def test_kcoverage(self):
        # TODO optimize
        n = 10  # nqubits
        k = 4  # number of "labels" needing distribution
        rows = nc.get_kcoverage_template(n, k, verbosity=2)
        nc._check_kcoverage_template(rows, n, k, verbosity=1)


class StdModuleBase(object):
    def test_upgrade_to_multiq_module(self):
        newmod = nc.stdmodule_to_smqmodule(self.std)
        opLabels = list(newmod.target_model().operations.keys())
        germStrs = newmod.germs

        for gl in opLabels:
            if gl != "Gi" and gl != ():
                self.assertGreater(len(gl.sslbls), 0)

        for str in germStrs:
            for gl in str:
                if gl != "Gi" and gl != ():
                    self.assertGreater(len(gl.sslbls), 0)


class Std1Q_XYITester(StdModuleBase, BaseCase):
    std = std1Q_XYI


class Std2Q_XXYYIITester(StdModuleBase, BaseCase):
    std = std2Q_XXYYII


class Std2Q_XYICNOTTester(StdModuleBase, BaseCase):
    std = std2Q_XYICNOT

    def test_upgrade_dataset(self):
        #Test upgrade of 2Q dataset
        ds = DataSet(outcome_labels=('00', '01', '10', '11'))
        ds.outcome_labels
        ds.add_count_dict(('Gix',), {'00': 90, '10': 10})
        ds.add_count_dict(('Giy',), {'00': 80, '10': 20})
        ds.add_count_dict(('Gxi',), {'00': 55, '10': 45})
        ds.add_count_dict(('Gyi',), {'00': 40, '10': 60})

        from pygsti.objects import Circuit as C
        ds2 = ds.copy()
        newmod = nc.stdmodule_to_smqmodule(self.std)
        newmod.upgrade_dataset(ds2)
        qlbls = (0, 1)  # qubit labels
        self.assertEqual(ds2[C((('Gx', 0),), qlbls)].counts, {('00',): 55, ('10',): 45})
        self.assertEqual(ds2[C((('Gy', 0),), qlbls)].counts, {('00',): 40, ('10',): 60})
        self.assertEqual(ds2[C((('Gx', 1),), qlbls)].counts, {('00',): 90, ('10',): 10})
        self.assertEqual(ds2[C((('Gy', 1),), qlbls)].counts, {('00',): 80, ('10',): 20})


class NQNoiseConstructionTester(BaseCase):
    def test_build_cloud_crosstalk_model(self):
        nQubits = 2
        ccmdl1 = nc.create_cloud_crosstalk_model(
            nQubits, ('Gx', 'Gy', 'Gcnot'),
            lindblad_error_coeffs={('Gx', 'qb0'): {('H', 'X'): 0.01, ('S', 'XY:qb0,qb1'): 0.01},
             ('Gcnot', 'qb0', 'qb1'): {('H', 'ZZ'): 0.02, ('S', 'XX:qb0,qb1'): 0.02},
             'idle': {('S', 'XX:qb0,qb1'): 0.01},
             'prep': {('S', 'XX:qb0,qb1'): 0.01},
             'povm': {('S', 'XX:qb0,qb1'): 0.01}
             }, qubit_labels=['qb{}'.format(i) for i in range(nQubits)])
        self.assertEqual(ccmdl1.num_params, 7)

        #Using sparse=True and a map-based simulator
        ccmdl1 = nc.create_cloud_crosstalk_model(
            nQubits, ('Gx', 'Gy', 'Gcnot'),
            lindblad_error_coeffs={('Gx', 'qb0'): {('H', 'X'): 0.01, ('S', 'XY:qb0,qb1'): 0.01},
             ('Gcnot', 'qb0', 'qb1'): {('H', 'ZZ'): 0.02, ('S', 'XX:qb0,qb1'): 0.02},
             'idle': {('S', 'XX:qb0,qb1'): 0.01},
             'prep': {('S', 'XX:qb0,qb1'): 0.01},
             'povm': {('S', 'XX:qb0,qb1'): 0.01}
             }, qubit_labels=['qb{}'.format(i) for i in range(nQubits)],
            simulator="map", sparse_lindblad_basis=True, sparse_lindblad_reps=True)
        self.assertEqual(ccmdl1.num_params, 7)


        #Using compact notation:
        ccmdl2 = nc.create_cloud_crosstalk_model(
            nQubits, ('Gx', 'Gy', 'Gcnot'),
            lindblad_error_coeffs={'Gx:0': {('HX'): 0.01, 'SXY:0,1': 0.01},
             'Gcnot:0:1': {'HZZ': 0.02, 'SXX:0,1': 0.02},
             'idle': {'SXX:0,1': 0.01},
             'prep': {'SXX:0,1': 0.01},
             'povm': {'SXX:0,1': 0.01}
             })
        self.assertEqual(ccmdl2.num_params, 7)

        #also using qubit_labels
        ccmdl3 = nc.create_cloud_crosstalk_model(
            nQubits, ('Gx', 'Gy', 'Gcnot'),
            lindblad_error_coeffs={'Gx:qb0': {('HX'): 0.01, 'SXY:qb0,qb1': 0.01},
             'Gcnot:qb0:qb1': {'HZZ': 0.02, 'SXX:qb0,qb1': 0.02},
             'idle': {'SXX:qb0,qb1': 0.01},
             'prep': {'SXX:qb0,qb1': 0.01},
             'povm': {'SXX:qb0,qb1': 0.01}
             }, qubit_labels=['qb{}'.format(i) for i in range(nQubits)])
        self.assertEqual(ccmdl3.num_params, 7)

        # Assert if try to use non-lindblad error specification (will be removed in the future when implemented)
        with self.assertRaises(NotImplementedError):
            nc.create_cloud_crosstalk_model(
                nQubits, ('Gx', 'Gy', 'Gcnot'),
                depolarization_strengths={'Gx': 0.15}
            )
        with self.assertRaises(NotImplementedError):
            nc.create_cloud_crosstalk_model(
                nQubits, ('Gx', 'Gy', 'Gcnot'),
                stochastic_error_probs={'Gx': (0.01,)*15}
            )


    def test_build_cloud_crosstalk_model_stencils(self):
        nQubits = 2
        ccmdl1 = nc.create_cloud_crosstalk_model(
            nQubits, ('Gx', 'Gy', 'Gcnot'),
            lindblad_error_coeffs={'Gx': {('H', 'X'): 0.01, ('S', 'X:@0+left'): 0.01},  # ('S','XX:@1+right,@0+left'): 0.02
             'Gcnot': {('H', 'ZZ'): 0.02, ('S', 'XX:@1+right,@0+left'): 0.02},
             'idle': {('S', 'XX:qb0,qb1'): 0.01}
             }, qubit_labels=['qb{}'.format(i) for i in range(nQubits)])
        self.assertEqual(ccmdl1.num_params, 5)

        #Using compact notation:
        ccmdl2 = nc.create_cloud_crosstalk_model(
            nQubits, ('Gx', 'Gy', 'Gcnot'),
            lindblad_error_coeffs={'Gx': {'HX': 0.01, 'SX:@0+left': 0.01},  # ('S','XX:@1+right,@0+left'): 0.02
             'Gcnot': {'HZZ': 0.02, 'SXX:@1+right,@0+left': 0.02},
             'idle': {'SXX:qb0,qb1': 0.01}
             }, qubit_labels=['qb{}'.format(i) for i in range(nQubits)])
        self.assertEqual(ccmdl2.num_params, 5)

    def test_build_cloud_crosstalk_model_indepgates(self):
        #Same as test_cloud_crosstalk_stencils case but set independent_gates=True
        nQubits = 2
        ccmdl1 = nc.create_cloud_crosstalk_model(
            nQubits, ('Gx', 'Gy', 'Gcnot'),
            lindblad_error_coeffs={'Gx': {('H', 'X'): 0.01, ('S', 'X:@0+left'): 0.01},  # ('S','XX:@1+right,@0+left'): 0.02
             'Gcnot': {('H', 'ZZ'): 0.02, ('S', 'XX:@1+right,@0+left'): 0.02},
             'idle': {('S', 'XX:qb0,qb1'): 0.01}
             }, qubit_labels=['qb{}'.format(i) for i in range(nQubits)], independent_gates=True)
        self.assertEqual(ccmdl1.num_params, 8)

    def test_build_cloud_crosstalk_model_with_nonstd_gate_unitary_factory(self):
        nQubits = 2

        def fn(args):
            if args is None: args = (0,)
            a, = args
            sigmaZ = np.array([[1, 0], [0, -1]], 'd')
            return scipy.linalg.expm(1j * float(a) * sigmaZ)

        ccmdl = nc.create_cloud_crosstalk_model(nQubits, ('Gx', 'Gy', 'Gcnot', 'Ga'),
                                                nonstd_gate_unitaries={'Ga': fn})
        c = Circuit("Gx:1Ga;0.3:1Gx:1@(0,1)")
        p1 = ccmdl.probabilities(c)

        self.assertAlmostEqual(p1['00'], 0.08733219254516078)
        self.assertAlmostEqual(p1['01'], 0.9126678074548386)
