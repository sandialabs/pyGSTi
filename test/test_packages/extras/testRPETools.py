from ..testutils import BaseTestCase, compare_files, temp_files
from pygsti.construction import std1Q_XYI as std
from pygsti.extras.rpe.rpeconfig_GxPi2_GyPi2_UpDn import rpeconfig_GxPi2_GyPi2_UpDn
from pygsti.extras.rpe.rpeconfig_GxPi2_GyPi2_00 import rpeconfig_GxPi2_GyPi2_00
import pygsti
import unittest

class RPETestCase(BaseTestCase):
    def test_rpe_tools(self):
        from pygsti.extras import rpe

        rpeconfig_inst_list = [rpeconfig_GxPi2_GyPi2_UpDn,rpeconfig_GxPi2_GyPi2_00]

        for rpeconfig_inst in rpeconfig_inst_list:

            xhat = 10 #1 counts for sin string
            yhat = 90 #1 counts for cos string
            k = 1 #experiment generation
            Nx = 100 # sin string clicks
            Ny = 100 # cos string clicks
            k1Alpha = rpe.extract_rotation_hat(xhat,yhat,k,Nx,Ny,"alpha",
                                               previousAngle=None, rpeconfig_inst=rpeconfig_inst)
            k1Eps = rpe.extract_rotation_hat(xhat,yhat,k,Nx,Ny,"epsilon",
                                             previousAngle=None, rpeconfig_inst=rpeconfig_inst)
            #self.assertAlmostEqual(k1Alpha, 0.785398163397)
            #self.assertAlmostEqual(k1Eps, -2.35619449019)

            k = 2 #experiment generation
            k2Alpha = rpe.extract_rotation_hat(xhat,yhat,k,Nx,Ny,"alpha",
                                               previousAngle=k1Alpha, rpeconfig_inst=rpeconfig_inst)
            k2Eps = rpe.extract_rotation_hat(xhat,yhat,k,Nx,Ny,"epsilon",
                                             previousAngle=k1Eps, rpeconfig_inst=rpeconfig_inst)
            #self.assertAlmostEqual(k2Alpha, 0.392699081699)
            #self.assertAlmostEqual(k2Eps, -1.1780972451)


            with self.assertRaises(Exception):
                rpe.extract_rotation_hat(xhat,yhat,2,Nx,Ny,"epsilon",
                                         previousAngle=None, rpeconfig_inst=rpeconfig_inst) #need previous angle

            with self.assertRaises(Exception):
                rpe.extract_rotation_hat(xhat,yhat,1,Nx,Ny,"foobar", rpeconfig_inst=rpeconfig_inst) #bad angle name


            from pygsti.construction import std1Q_XY as stdXY
            target = stdXY.gs_target.copy()
            target.gates['Gi'] =  std.gs_target.gates['Gi'] #need a Gi gate...
            stringListD = rpe.make_rpe_angle_string_list_dict(2,rpeconfig_inst)
            gs_depolXZ = target.depolarize(gate_noise=0.1,spam_noise=0.1)
            ds = pygsti.construction.generate_fake_data(gs_depolXZ, stringListD['totalStrList'],
                                                        nSamples=1000, sampleError='binomial')

            epslist = rpe.est_angle_list(ds,stringListD['epsilon','sin'],stringListD['epsilon','cos'],
                                         angleName="epsilon", rpeconfig_inst=rpeconfig_inst)

            tlist,dummy = rpe.est_theta_list(ds,stringListD['theta','sin'],stringListD['theta','cos'],
                                             epslist,returnPhiFunList=True, rpeconfig_inst=rpeconfig_inst)
            tlist = rpe.est_theta_list(ds,stringListD['theta','sin'],stringListD['theta','cos'],
                                       epslist,returnPhiFunList=False, rpeconfig_inst=rpeconfig_inst)

            alpha = rpe.extract_alpha( stdXY.gs_target, rpeconfig_inst)
            eps = rpe.extract_epsilon( stdXY.gs_target, rpeconfig_inst)
            theta = rpe.extract_theta( stdXY.gs_target, rpeconfig_inst)
            rpe.analyze_rpe_data(ds,gs_depolXZ,stringListD,rpeconfig_inst)

if __name__ == '__main__':
    unittest.main(verbosity=2)
