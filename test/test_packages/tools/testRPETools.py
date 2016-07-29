from ..testutils import BaseTestCase
from pygsti.construction import std1Q_XYI as std
import pygsti
import unittest

class RPETestCase(BaseTestCase):
    def test_rpe_tools(self):
        from pygsti.tools import rpe

        xhat = 10 #plus counts for sin string
        yhat = 90 #plus counts for cos string
        k = 1 #experiment generation
        Nx = 100 # sin string clicks
        Ny = 100 # cos string clicks
        k1Alpha = rpe.extract_rotation_hat(xhat,yhat,k,Nx,Ny,"alpha",
                                           previousAngle=None)
        k1Eps = rpe.extract_rotation_hat(xhat,yhat,k,Nx,Ny,"epsilon",
                                         previousAngle=None)
        self.assertAlmostEqual(k1Alpha, 0.785398163397)
        self.assertAlmostEqual(k1Eps, -2.35619449019)

        k = 2 #experiment generation
        k2Alpha = rpe.extract_rotation_hat(xhat,yhat,k,Nx,Ny,"alpha",
                                           previousAngle=k1Alpha)
        k2Eps = rpe.extract_rotation_hat(xhat,yhat,k,Nx,Ny,"epsilon",
                                         previousAngle=k1Eps)
        self.assertAlmostEqual(k2Alpha, 0.392699081699)
        self.assertAlmostEqual(k2Eps, -1.1780972451)


        with self.assertRaises(Exception):
            rpe.extract_rotation_hat(xhat,yhat,2,Nx,Ny,"epsilon",
                                     previousAngle=None) #need previous angle

        with self.assertRaises(Exception):
            rpe.extract_rotation_hat(xhat,yhat,1,Nx,Ny,"foobar") #bad angle name


        from pygsti.construction import std1Q_XZ as stdXZ
        target = stdXZ.gs_target.copy()
        target.gates['Gi'] =  std.gs_target.gates['Gi'] #need a Gi gate...
        stringListD = pygsti.construction.make_rpe_string_list_d(2)
        gs_depolXZ = target.depolarize(gate_noise=0.1,spam_noise=0.1)
        ds = pygsti.construction.generate_fake_data(gs_depolXZ, stringListD['totalStrList'],
                                                    nSamples=1000, sampleError='binomial')

        epslist = rpe.est_angle_list(ds,stringListD['epsilon','sin'],stringListD['epsilon','cos'],
                                     angleName="epsilon")

        tlist,dummy = rpe.est_theta_list(ds,stringListD['theta','sin'],stringListD['theta','cos'],
                                         epslist,returnPhiFunList=True)
        tlist = rpe.est_theta_list(ds,stringListD['theta','sin'],stringListD['theta','cos'],
                                   epslist,returnPhiFunList=False)

        alpha = rpe.extract_alpha( stdXZ.gs_target )
        eps = rpe.extract_epsilon( stdXZ.gs_target )
        theta = rpe.extract_theta( stdXZ.gs_target )
        rpe.analyze_simulated_rpe_experiment(ds,gs_depolXZ,stringListD)

if __name__ == '__main__':
    unittest.main(verbosity=2)
