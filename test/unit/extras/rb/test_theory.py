import numpy as np

from ...util import BaseCase

from pygsti.tools import optools as ot
from pygsti.modelpacks.legacy import std1Q_Cliffords, std1Q_XY, std1Q_XYI
#from pygsti.extras import rb


class GaugeTransformBase(object):
    def test_transform_to_rb_gauge(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        # A test that the RB gauge transformation behaves as expected -- a model that does not
        # have r = infidelity in its initial gauge does have this in the RB gauge. This also
        # tests that the r predictions are working for not-all-the-Cliffords models.
        mdl_in_RB_gauge = rb.theory.transform_to_rb_gauge(self.mdl, self.target_model, eigenvector_weighting=0.5)
        r_pred_EI = rb.theory.predicted_rb_number(self.mdl, self.target_model, rtype='EI')
        REI = rb.theory.gateset_infidelity(mdl_in_RB_gauge, self.target_model, itype='EI')
        self.assertAlmostEqual(r_pred_EI, REI, places=10)


class InfidelityBase(object):
    weights = None

    def test_gateset_infidelity_AGI(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        AGsI = rb.theory.gateset_infidelity(self.mdl, self.target_model, weights=self.weights, itype='AGI')
        self.assertAlmostEqual(AGsI, self.expected_AGI, places=10)

    def test_predicted_RB_number_AGI(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        r_pred_AGI = rb.theory.predicted_rb_number(self.mdl, self.target_model, weights=self.weights, rtype='AGI')
        self.assertAlmostEqual(r_pred_AGI, self.expected_AGI, places=10)

    def test_gateset_infidelity_EI(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        REI = rb.theory.gateset_infidelity(self.mdl, self.target_model, weights=self.weights, itype='EI')
        self.assertAlmostEqual(REI, self.expected_EI, places=10)

    def test_predicted_RB_number_EI(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        r_pred_EI = rb.theory.predicted_rb_number(self.mdl, self.target_model, weights=self.weights, rtype='EI')
        self.assertAlmostEqual(r_pred_EI, self.expected_EI, places=10)


class RBTheoryZrotModelTester(GaugeTransformBase, BaseCase):
    @classmethod
    def setUpClass(cls):
        super(RBTheoryZrotModelTester, cls).setUpClass()
        cls.target_model = std1Q_XY.target_model()
        cls.mdl = cls.target_model.copy()

        Zrot_unitary = np.array([[1., 0.], [0., np.exp(-1j * 0.01)]])
        Zrot_channel = ot.unitary_to_pauligate(Zrot_unitary)

        for key in cls.target_model.operations.keys():
            cls.mdl.operations[key] = np.dot(Zrot_channel, cls.target_model.operations[key])


class RBTheoryWeightedInfidelityTester(GaugeTransformBase, InfidelityBase, BaseCase):
    @classmethod
    def setUpClass(cls):
        super(RBTheoryWeightedInfidelityTester, cls).setUpClass()
        cls.target_model = std1Q_XY.target_model()
        cls.mdl = cls.target_model.copy()

        depol_strength_X = 1e-3
        depol_strength_Y = 3e-3

        lx = 1. - depol_strength_X
        depmap_X = np.array([[1., 0., 0., 0.], [0., lx, 0., 0.], [0., 0., lx, 0.], [0, 0., 0., lx]])
        ly = 1. - depol_strength_Y
        depmap_Y = np.array([[1., 0., 0., 0.], [0., ly, 0., 0.], [0., 0., ly, 0.], [0, 0., 0., ly]])
        cls.mdl.operations['Gx'] = np.dot(depmap_X, cls.target_model.operations['Gx'])
        cls.mdl.operations['Gy'] = np.dot(depmap_Y, cls.target_model.operations['Gy'])

        Gx_weight = 1
        Gy_weight = 2
        cls.weights = {'Gx': Gx_weight, 'Gy': Gy_weight}
        GxAGI = ot.average_gate_infidelity(cls.mdl.operations['Gx'], cls.target_model.operations['Gx'])
        GyAGI = ot.average_gate_infidelity(cls.mdl.operations['Gy'], cls.target_model.operations['Gy'])
        cls.expected_AGI = (Gx_weight * GxAGI + Gy_weight * GyAGI) / (Gx_weight + Gy_weight)
        GxAEI = ot.entanglement_infidelity(cls.mdl.operations['Gx'], cls.target_model.operations['Gx'])
        GyAEI = ot.entanglement_infidelity(cls.mdl.operations['Gy'], cls.target_model.operations['Gy'])
        cls.expected_EI = (Gx_weight * GxAEI + Gy_weight * GyAEI) / (Gx_weight + Gy_weight)


class RBTheoryCliffordsModelTester(GaugeTransformBase, InfidelityBase, BaseCase):
    @classmethod
    def setUpClass(cls):
        return  # SKIP TESTS
        super(RBTheoryCliffordsModelTester, cls).setUpClass()
        cls.target_model = std1Q_Cliffords.target_model()
        cls.depol_strength = 1e-3
        cls.mdl = cls.target_model.depolarize(op_noise=cls.depol_strength)
        cls.expected_AGI = rb.analysis.p_to_r(1 - cls.depol_strength, d=2, rtype='AGI')
        cls.expected_EI = rb.analysis.p_to_r(1 - cls.depol_strength, d=2, rtype='EI')
        cls.clifford_group = rb.group.construct_1q_clifford_group()

    def test_R_matrix(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        R = rb.theory.R_matrix(self.target_model, self.clifford_group, group_to_model=None, weights=None)
        # TODO assert correctness

    def test_exact_RB_ASPs(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        m, ASPs = rb.theory.exact_rb_asps(
            self.mdl, self.clifford_group, m_max=1000, m_min=0,
            m_step=100, success_outcomelabel=('0',),
            group_to_model=None, weights=None, compilation=None,
            group_twirled=False
        )
        self.assertLess(abs(ASPs[1] - (0.5 + 0.5 * (1.0 - self.depol_strength)**101)), 10**(-10))

        m, ASPs = rb.theory.exact_rb_asps(
            self.mdl, self.clifford_group, m_max=1000, m_min=0,
            m_step=100, success_outcomelabel=('0',),
            group_to_model=None, weights=None, compilation=None,
            group_twirled=True
        )
        self.assertLess(abs(ASPs[1] - (0.5 + 0.5 * (1.0 - self.depol_strength)**102)), 10**(-10))

    def test_L_matrix_ASPs(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        # TODO optimize
        # Check it works with a Clifford model, and gives plausable output
        m, ASPs = rb.theory.L_matrix_asps(
            self.mdl, self.target_model, m_max=10, m_min=0, m_step=1,
            success_outcomelabel=('0',), compilation=None,
            group_twirled=False, weights=None, gauge_optimize=False,
            return_error_bounds=False, norm='1to1'
        )
        self.assertTrue((ASPs > 0.98).all())


class RBTheoryXYModelTester(BaseCase):
    @classmethod
    def setUpClass(cls):
        return  # SKIP TESTS
        super(RBTheoryXYModelTester, cls).setUpClass()
        cls.target_model = std1Q_XY.target_model()
        cls.mdl = cls.target_model.depolarize(op_noise=1e-3)

        cls.clifford_group = rb.group.construct_1q_clifford_group()
        cls.clifford_compilation = std1Q_XY.clifford_compilation

    def test_exact_AB_ASPs(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        # Tests the exact RB ASPs function on a subset-of-Cliffords model.
        group_to_model = {'Gc16': 'Gx', 'Gc21': 'Gy'}
        weights = {'Gx': 5, 'Gy': 10}
        m, ASPs = rb.theory.exact_rb_asps(
            self.target_model, self.clifford_group, m_max=10, m_min=0,
            m_step=1, success_outcomelabel=('0',),
            group_to_model=group_to_model, weights=None,
            compilation=self.clifford_compilation, group_twirled=False
        )
        self.assertLess(abs(np.sum(ASPs) - len(ASPs)), 10**(-10))

        # Tests the function behaves reasonably with a depolarized model + works with group_twirled + weights.
        m, ASPs = rb.theory.exact_rb_asps(
            self.mdl, self.clifford_group, m_max=10, m_min=0, m_step=1,
            success_outcomelabel=('0',),
            group_to_model=group_to_model, weights=None,
            compilation=self.clifford_compilation, group_twirled=False
        )
        self.assertLess(abs(ASPs[0] - 1), 10**(-10))

        m, ASPs = rb.theory.exact_rb_asps(
            self.mdl, self.clifford_group, m_max=10, m_min=0, m_step=3,
            success_outcomelabel=('0',),
            group_to_model=group_to_model, weights=weights,
            compilation=self.clifford_compilation, group_twirled=True
        )
        self.assertTrue((ASPs > 0.99).all())

    def test_L_matrix_ASPs(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        # TODO optimize
        # Check the L-matrix theory predictions work and are consistent with the exact predictions
        group_to_model = {'Gc16': 'Gx', 'Gc21': 'Gy'}
        weights = {'Gx': 5, 'Gy': 10}

        m, ASPs = rb.theory.exact_rb_asps(
            self.mdl, self.clifford_group, m_max=10, m_min=0, m_step=1,
            success_outcomelabel=('0',),
            group_to_model=group_to_model, weights=weights,
            compilation=self.clifford_compilation, group_twirled=True
        )

        # Todo : change '1to1' to 'diamond' in 2 of 3 of the following, when diamonddist is working.
        L_m, L_ASPs, L_LASPs, L_UASPs = rb.theory.L_matrix_asps(
            self.mdl, self.target_model, m_max=10, m_min=0, m_step=1,
            success_outcomelabel=('0',), compilation=self.clifford_compilation,
            group_twirled=True, weights=weights, gauge_optimize=True,
            return_error_bounds=True, norm='1to1'
        )
        self.assertTrue((abs(ASPs - L_ASPs) < 0.001).all())

        # Check it works without the twirl, and gives plausable output
        L_m, L_ASPs = rb.theory.L_matrix_asps(
            self.mdl, self.target_model, m_max=10, m_min=0, m_step=1,
            success_outcomelabel=('0',),
            compilation=self.clifford_compilation,
            group_twirled=False, weights=None, gauge_optimize=False,
            return_error_bounds=False, norm='1to1'
        )
        self.assertTrue((ASPs > 0.98).all())


class RBTheoryXYIModelTester(BaseCase):
    @classmethod
    def setUpClass(cls):
        return  # SKIP TESTS
        super(RBTheoryXYIModelTester, cls).setUpClass()
        cls.target_model = std1Q_XYI.target_model()
        cls.depol_strength = 1e-3
        cls.mdl = cls.target_model.depolarize(op_noise=cls.depol_strength)

        cls.clifford_group = rb.group.construct_1q_clifford_group()
        cls.clifford_compilation = std1Q_XYI.clifford_compilation

    def test_R_matrix(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        # Test constructing the R matrix for a group-subset model with weights
        group_to_model = {'Gc0': 'Gi', 'Gc16': 'Gx', 'Gc21': 'Gy'}
        weights = {'Gi': 1., 'Gx': 1, 'Gy': 1}
        R = rb.theory.R_matrix(self.target_model, self.clifford_group,
                               group_to_model=group_to_model, weights=weights)
        # TODO assert correctness

    def test_R_matrix_predicted_RB_decay_parameter(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        group_to_model = {'Gc0': 'Gi', 'Gc16': 'Gx', 'Gc21': 'Gy'}
        weights = {'Gi': 1., 'Gx': 1, 'Gy': 1}
        # Tests the p-prediction function works, and that we get the correct predictions from the R-matrix.
        p = rb.theory.R_matrix_predicted_rb_decay_parameter(self.target_model, self.clifford_group,
                                                            group_to_model=group_to_model,
                                                            weights=weights)
        self.assertAlmostEqual(p, 1., places=10)

        p = rb.theory.R_matrix_predicted_rb_decay_parameter(self.mdl, self.clifford_group,
                                                            group_to_model=group_to_model,
                                                            weights=weights)
        self.assertAlmostEqual(p, (1.0 - self.depol_strength), places=10)
