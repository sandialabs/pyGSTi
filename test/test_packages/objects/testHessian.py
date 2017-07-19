import unittest
import warnings
import pygsti
from pygsti.construction import std1Q_XYI as stdxyi
from pygsti.construction import std1Q_XY as stdxy

import numpy as np
import sys, os

from ..testutils import BaseTestCase, compare_files, temp_files


class TestHessianMethods(BaseTestCase):

    def setUp(self):
        super(TestHessianMethods, self).setUp()

        self.gateset = pygsti.io.load_gateset(compare_files + "/analysis.gateset")
        self.ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/analysis.dataset")


    def test_parameter_counting(self):
        #XY Gateset: SPAM=True
        n = stdxy.gs_target.num_params()
        self.assertEqual(n,40) # 2*16 + 2*4 = 40

        n = stdxy.gs_target.num_nongauge_params()
        self.assertEqual(n,24) # full 16 gauge params: SPAM gate + 2 others

        #XY Gateset: SPAM=False
        tst = stdxy.gs_target.copy()
        del tst.preps['rho0']
        del tst.effects['E0']
        n = tst.num_params()
        self.assertEqual(n,32) # 2*16 = 32

        n = tst.num_nongauge_params()
        self.assertEqual(n,18) # gates are all unital & TP => only 14 gauge params (2 casimirs)


        #XYI Gateset: SPAM=True
        n = stdxyi.gs_target.num_params()
        self.assertEqual(n,56) # 3*16 + 2*4 = 56

        n = stdxyi.gs_target.num_nongauge_params()
        self.assertEqual(n,40) # full 16 gauge params: SPAM gate + 3 others

        #XYI Gateset: SPAM=False
        tst = stdxyi.gs_target.copy()
        del tst.preps['rho0']
        del tst.effects['E0']
        n = tst.num_params()
        self.assertEqual(n,48) # 3*16 = 48

        n = tst.num_nongauge_params()
        self.assertEqual(n,34) # gates are all unital & TP => only 14 gauge params (2 casimirs)

        #XYI Gateset: SP0=False
        tst = stdxyi.gs_target.copy()
        tst.preps['rho0'] = pygsti.obj.TPParameterizedSPAMVec(tst.preps['rho0'])
        n = tst.num_params()
        self.assertEqual(n,55) # 3*16 + 4 + 3 = 55

        n = tst.num_nongauge_params()
        self.assertEqual(n,40) # 15 gauge params (minus one b/c can't change rho?)

        #XYI Gateset: G0=SP0=False
        tst.gates['Gi'] = pygsti.obj.TPParameterizedGate(tst.gates['Gi'])
        tst.gates['Gx'] = pygsti.obj.TPParameterizedGate(tst.gates['Gx'])
        tst.gates['Gy'] = pygsti.obj.TPParameterizedGate(tst.gates['Gy'])
        n = tst.num_params()
        self.assertEqual(n,43) # 3*12 + 4 + 3 = 43

        n = tst.num_nongauge_params()
        self.assertEqual(n,31) # full 12 gauge params of single 4x3 gate


    def test_hessian_projection(self):

        chi2, chi2Grad, chi2Hessian = pygsti.chi2(self.ds, self.gateset,
                                                  returnGradient=True,
                                                  returnHessian=True)

        proj_non_gauge = self.gateset.get_nongauge_projector()
        projectedHessian = np.dot(proj_non_gauge,
                                  np.dot(chi2Hessian, proj_non_gauge))

        self.assertEqual( projectedHessian.shape, (56,56) )
        self.assertEqual( np.linalg.matrix_rank(proj_non_gauge), 40)
        self.assertEqual( np.linalg.matrix_rank(projectedHessian), 40)

        eigvals = np.sort(abs(np.linalg.eigvals(projectedHessian)))

        eigvals_chk = np.array(
            [  2.36932621e-11,   4.57739349e-11,   8.49888013e-11,   8.49888013e-11,
               1.22859895e-10,   1.22859895e-10,   1.38705957e-10,   1.38705957e-10,
               3.75441328e-10,   6.46644807e-10,   6.46644807e-10,   7.06181642e-10,
               7.06181642e-10,   7.65472749e-10,   1.62672899e-09,   1.62672899e-09,
               9.23280752e+04,   9.88622140e+04,   1.24577730e+05,   1.40500652e+05,
               1.88461602e+05,   1.98466608e+05,   2.07084497e+05,   2.27489018e+05,
               2.33403285e+05,   2.52573024e+05,   2.82350782e+05,   3.12087185e+05,
               3.21855420e+05,   3.31659734e+05,   3.52649174e+05,   3.60682071e+05,
               3.90777833e+05,   4.59913853e+05,   5.02652879e+05,   5.59311926e+05,
               5.78891070e+05,   6.82325323e+05,   7.57318263e+05,   8.16739390e+05,
               1.06466062e+06,   1.20075694e+06,   1.37368639e+06,   1.58356629e+06,
               1.68898356e+06,   2.12277359e+06,   3.30650801e+06,   3.75869331e+06,
               4.00195245e+06,   4.42427797e+06,   5.06956256e+06,   7.31166332e+06,
               9.19432790e+06,   9.99944236e+06,   1.31027722e+07,   5.80310818e+07] )

        TOL = 1e-7
        for val,chk in zip(eigvals,eigvals_chk):
            if abs(val) > TOL or abs(chk) > TOL:
                self.assertAlmostEqual(abs(val-chk)/(abs(chk)+TOL), 0.0, places=3)
            # (else both chk and val are <= TOL, so both == 0 for our purposes)
        #print "eigvals = ",eigvals

    def test_confidenceRegion(self):

        chi2, chi2Hessian = pygsti.chi2(self.ds, self.gateset,
                                        returnHessian=True)
        ci_std = pygsti.obj.ConfidenceRegion(self.gateset, chi2Hessian, 95.0,
                                             hessianProjection="std")
        ci_noproj = pygsti.obj.ConfidenceRegion(self.gateset, chi2Hessian, 95.0,
                                             hessianProjection="none")
        ci_opt = pygsti.obj.ConfidenceRegion(self.gateset, chi2Hessian, 95.0,
                                             hessianProjection="optimal gate CIs",
                                             tol=0.1) #very low tol so doesn't take long
        ci_intrinsic = pygsti.obj.ConfidenceRegion(self.gateset, chi2Hessian, 95.0,
                                             hessianProjection="intrinsic error")
        ci_linresponse = pygsti.obj.ConfidenceRegion(self.gateset, None, 95.0,
                                                     hessianProjection="linear response",
                                                     linresponse_mlgst_params={'dataset': self.ds,
                                                                               'gateStringsToUse': list(self.ds.keys())} )

        self.assertTrue( ci_std.has_hessian() )

        with self.assertRaises(ValueError):
            pygsti.obj.ConfidenceRegion(self.gateset, chi2Hessian, 95.0,
                                             hessianProjection="FooBar") #bad hessianProjection

        self.assertWarns(pygsti.obj.ConfidenceRegion, self.gateset,
                         chi2Hessian, 0.95, hessianProjection="none") # percentage < 1.0

        for ci_cur in (ci_std, ci_noproj, ci_opt, ci_intrinsic, ci_linresponse):
            try: 
                ar_of_intervals_Gx = ci_cur.get_profile_likelihood_confidence_intervals("Gx")
                ar_of_intervals_rho0 = ci_cur.get_profile_likelihood_confidence_intervals("rho0")
                ar_of_intervals_E0 = ci_cur.get_profile_likelihood_confidence_intervals("E0")
                ar_of_intervals = ci_cur.get_profile_likelihood_confidence_intervals()
            except NotImplementedError: 
                pass #linear response CI doesn't support profile likelihood intervals
    
            def fnOfGate_float(mx):
                return float(mx[0,0])
            def fnOfGate_0D(mx):
                return np.array( float(mx[0,0]) )
            def fnOfGate_1D(mx):
                return mx[0,:]
            def fnOfGate_2D(mx):
                return mx[:,:]
            def fnOfGate_3D(mx):
                return np.zeros( (2,2,2), 'd') #just to test for error
    
    
            df = ci_cur.get_gate_fn_confidence_interval(fnOfGate_float, 'Gx', verbosity=0)
            df = ci_cur.get_gate_fn_confidence_interval(fnOfGate_0D, 'Gx', verbosity=0)
            df = ci_cur.get_gate_fn_confidence_interval(fnOfGate_1D, 'Gx', verbosity=0)
            df = ci_cur.get_gate_fn_confidence_interval(fnOfGate_2D, 'Gx', verbosity=0)
            df, f0 = self.runSilent(ci_cur.get_gate_fn_confidence_interval,
                                    fnOfGate_float, 'Gx', returnFnVal=True, verbosity=4)
    
            with self.assertRaises(ValueError):
                ci_cur.get_gate_fn_confidence_interval(fnOfGate_3D, 'Gx', verbosity=0)

            #SHORT-CIRCUIT linear reponse here to reduce run time
            if ci_cur is ci_linresponse: continue
    
            def fnOfVec_float(v):
                return float(v[0])
            def fnOfVec_0D(v):
                return np.array( float(v[0]) )
            def fnOfVec_1D(v):
                return np.array(v[:])
            def fnOfVec_2D(v):
                return np.dot(v.T,v)
            def fnOfVec_3D(v):
                return np.zeros( (2,2,2), 'd') #just to test for error
    
    
            df = ci_cur.get_prep_fn_confidence_interval(fnOfVec_float, 'rho0', verbosity=0)
            df = ci_cur.get_prep_fn_confidence_interval(fnOfVec_0D, 'rho0', verbosity=0)
            df = ci_cur.get_prep_fn_confidence_interval(fnOfVec_1D, 'rho0', verbosity=0)
            df = ci_cur.get_prep_fn_confidence_interval(fnOfVec_2D, 'rho0', verbosity=0)
            df, f0 = self.runSilent(ci_cur.get_prep_fn_confidence_interval,
                                    fnOfVec_float, 'rho0', returnFnVal=True, verbosity=4)
    
            with self.assertRaises(ValueError):
                ci_cur.get_prep_fn_confidence_interval(fnOfVec_3D, 'rho0', verbosity=0)
    
            df = ci_cur.get_effect_fn_confidence_interval(fnOfVec_float, 'E0', verbosity=0)
            df = ci_cur.get_effect_fn_confidence_interval(fnOfVec_0D, 'E0', verbosity=0)
            df = ci_cur.get_effect_fn_confidence_interval(fnOfVec_1D, 'E0', verbosity=0)
            df = ci_cur.get_effect_fn_confidence_interval(fnOfVec_2D, 'E0', verbosity=0)
            df, f0 = self.runSilent(ci_cur.get_effect_fn_confidence_interval,
                                    fnOfVec_float, 'E0', returnFnVal=True, verbosity=4)
    
            with self.assertRaises(ValueError):
                ci_cur.get_effect_fn_confidence_interval(fnOfVec_3D, 'E0', verbosity=0)
    
    
            def fnOfSpam_float(rhoVecs, EVecs):
                return float( np.dot( rhoVecs[0].T, EVecs[0] ) )
            def fnOfSpam_0D(rhoVecs, EVecs):
                return np.array( float( np.dot( rhoVecs[0].T, EVecs[0] ) ) )
            def fnOfSpam_1D(rhoVecs, EVecs):
                return np.array( [ np.dot( rhoVecs[0].T, EVecs[0] ), 0] )
            def fnOfSpam_2D(rhoVecs, EVecs):
                return np.array( [[ np.dot( rhoVecs[0].T, EVecs[0] ), 0],[0,0]] )
            def fnOfSpam_3D(rhoVecs, EVecs):
                return np.zeros( (2,2,2), 'd') #just to test for error
    
            df = ci_cur.get_spam_fn_confidence_interval(fnOfSpam_float, verbosity=0)
            df = ci_cur.get_spam_fn_confidence_interval(fnOfSpam_0D, verbosity=0)
            df = ci_cur.get_spam_fn_confidence_interval(fnOfSpam_1D, verbosity=0)
            df = ci_cur.get_spam_fn_confidence_interval(fnOfSpam_2D, verbosity=0)
            df, f0 = self.runSilent(ci_cur.get_spam_fn_confidence_interval,
                                    fnOfSpam_float, returnFnVal=True, verbosity=4)
    
            with self.assertRaises(ValueError):
                ci_cur.get_spam_fn_confidence_interval(fnOfSpam_3D, verbosity=0)
    
    
    
            def fnOfGateSet_float(gs):
                return float( gs.gates['Gx'][0,0] )
            def fnOfGateSet_0D(gs):
                return np.array( gs.gates['Gx'][0,0]  )
            def fnOfGateSet_1D(gs):
                return np.array( gs.gates['Gx'][0,:] )
            def fnOfGateSet_2D(gs):
                return np.array( gs.gates['Gx'] )
            def fnOfGateSet_3D(gs):
                return np.zeros( (2,2,2), 'd') #just to test for error
    
            df = ci_cur.get_gateset_fn_confidence_interval(fnOfGateSet_float, verbosity=0)
            df = ci_cur.get_gateset_fn_confidence_interval(fnOfGateSet_0D, verbosity=0)
            df = ci_cur.get_gateset_fn_confidence_interval(fnOfGateSet_1D, verbosity=0)
            df = ci_cur.get_gateset_fn_confidence_interval(fnOfGateSet_2D, verbosity=0)
            df, f0 = self.runSilent(ci_cur.get_gateset_fn_confidence_interval,
                                    fnOfGateSet_float, returnFnVal=True, verbosity=4)
    
            with self.assertRaises(ValueError):
                ci_cur.get_gateset_fn_confidence_interval(fnOfGateSet_3D, verbosity=0)

        #TODO: assert values of df & f0 ??

    def tets_pickle_ConfidenceRegion(self):
        chi2, chi2Hessian = pygsti.chi2(self.ds, self.gateset,
                                        returnHessian=True)
        ci_std = pygsti.obj.ConfidenceRegion(self.gateset, chi2Hessian, 95.0,
                                             hessianProjection="std")
        import pickle
        s = pickle.dumps(ci_std)
        ci_std2 = pickle.loads(s)
        #TODO: make sure ci_std and ci_std2 are the same


    def test_mapcalc_hessian(self):
        chi2, chi2Hessian = pygsti.chi2(self.ds, self.gateset,
                                        returnHessian=True)
        
        gs_mapcalc = self.gateset.copy()
        gs_mapcalc._calcClass = pygsti.objects.gatemapcalc.GateMapCalc
        chi2, chi2Hessian_mapcalc = pygsti.chi2(self.ds, self.gateset,
                                        returnHessian=True)

        self.assertArraysAlmostEqual(chi2Hessian, chi2Hessian_mapcalc)



if __name__ == "__main__":
    unittest.main(verbosity=2)
