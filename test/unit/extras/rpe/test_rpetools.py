import pygsti.construction as pc
from pygsti.extras.rpe import rpetools as tools, rpeconstruction as rpc
from pygsti.modelpacks.legacy import std1Q_XYI as std, std1Q_XYI as stdXY
from ...util import BaseCase

_SEED = 1969


class RPEToolsFuncBase(object):
    def setUp(self):
        super(RPEToolsFuncBase, self).setUp()
        self.target = stdXY.target_model()
        self.target.operations['Gi'] = std.target_model().operations['Gi']  # need a Gi gate...
        self.stringListD = rpc.create_rpe_angle_circuits_dict(2, self.config)
        self.mdl_depolXZ = self.target.depolarize(op_noise=0.1, spam_noise=0.1, seed=_SEED)
        self.ds = pc.simulate_data(self.mdl_depolXZ, self.stringListD['totalStrList'],
                                   num_samples=1000, sample_error='binomial', seed=_SEED)

    def test_extract_rotation_hat(self):
        xhat = 10  # 1 counts for sin string
        yhat = 90  # 1 counts for cos string
        k = 1  # experiment generation
        Nx = 100  # sin string clicks
        Ny = 100  # cos string clicks
        k1Alpha = tools.extract_rotation_hat(xhat, yhat, k, Nx, Ny, "alpha",
                                             previous_angle=None, rpeconfig_inst=self.config)
        k1Eps = tools.extract_rotation_hat(xhat, yhat, k, Nx, Ny, "epsilon",
                                           previous_angle=None, rpeconfig_inst=self.config)
        # self.assertAlmostEqual(k1Alpha, 0.785398163397)
        self.assertAlmostEqual(k1Alpha, -2.35619449019)
        self.assertAlmostEqual(k1Eps, -2.35619449019)

        k = 2  # experiment generation
        k2Alpha = tools.extract_rotation_hat(xhat, yhat, k, Nx, Ny, "alpha",
                                             previous_angle=k1Alpha, rpeconfig_inst=self.config)
        k2Eps = tools.extract_rotation_hat(xhat, yhat, k, Nx, Ny, "epsilon",
                                           previous_angle=k1Eps, rpeconfig_inst=self.config)
        # self.assertAlmostEqual(k2Alpha, 0.392699081699)
        self.assertAlmostEqual(k2Alpha, -1.1780972451)
        self.assertAlmostEqual(k2Eps, -1.1780972451)

    def test_est_angle_list(self):
        epslist = tools.estimate_angles(
            self.ds, self.stringListD['epsilon', 'sin'],
            self.stringListD['epsilon', 'cos'], angle_name="epsilon",
            rpeconfig_inst=self.config
        )
        # TODO assert correctness

    def test_est_theta_list(self):
        epslist = tools.estimate_angles(
            self.ds, self.stringListD['epsilon', 'sin'],
            self.stringListD['epsilon', 'cos'], angle_name="epsilon",
            rpeconfig_inst=self.config
        )
        tlist, dummy = tools.estimate_thetas(
            self.ds, self.stringListD['theta', 'sin'], self.stringListD['theta', 'cos'],
            epslist, return_phi_fun_list=True, rpeconfig_inst=self.config
        )
        # TODO assert correctness
        tlist = tools.estimate_thetas(
            self.ds, self.stringListD['theta', 'sin'], self.stringListD['theta', 'cos'],
            epslist, return_phi_fun_list=False, rpeconfig_inst=self.config
        )
        # TODO assert correctness

    def test_extract_alpha(self):
        alpha = tools.extract_alpha(stdXY.target_model(), self.config)
        # TODO assert correctness

    def test_extract_epsilon(self):
        epsilon = tools.extract_epsilon(stdXY.target_model(), self.config)
        # TODO assert correctness

    def test_extract_theta(self):
        theta = tools.extract_theta(stdXY.target_model(), self.config)
        # TODO assert correctness

    def test_analyze_rpe_data(self):
        results = tools.analyze_rpe_data(self.ds, self.mdl_depolXZ, self.stringListD, self.config)
        # TODO assert correctness

    def test_extract_rotation_hat_raises_on_missing_previous_angle(self):
        with self.assertRaises(Exception):
            tools.extract_rotation_hat(10, 90, 2, 100, 100, "epsilon",
                                       previous_angle=None, rpeconfig_inst=self.config)

    def test_extract_rotation_hat_raises_on_bad_angle_name(self):
        with self.assertRaises(Exception):
            tools.extract_rotation_hat(10, 90, 1, 100, 100, "foobar",
                                       previous_angle=None, rpeconfig_inst=self.config)


class RPETools00ConfigTester(RPEToolsFuncBase, BaseCase):
    from pygsti.extras.rpe import rpeconfig_GxPi2_GyPi2_00 as config


class RPEToolsUpDnConfigTester(RPEToolsFuncBase, BaseCase):
    from pygsti.extras.rpe import rpeconfig_GxPi2_GyPi2_UpDn as config
