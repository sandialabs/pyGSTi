import unittest
import warnings
import pygsti
from pygsti.construction import std1Q_XYI as stdxyi
from pygsti.construction import std1Q_XY as stdxy
from pygsti.objects import gatesetfunction as gsf
from pygsti.objects.gatemapcalc import GateMapCalc

import numpy as np
import sys, os

from ..testutils import BaseTestCase, compare_files, temp_files


class TestHessianMethods(BaseTestCase):

    def setUp(self):
        super(TestHessianMethods, self).setUp()

        self.gateset = pygsti.io.load_gateset(compare_files + "/analysis.gateset")
        self.ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/analysis.dataset%s" % self.versionsuffix)


        fiducials = stdxyi.fiducials
        germs = stdxyi.germs
        gateLabels = list(self.gateset.gates.keys()) # also == std.gates
        self.maxLengthList = [1,2]
        self.gss = pygsti.construction.make_lsgst_structs(gateLabels, fiducials, fiducials, germs, self.maxLengthList)


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

        print("eigvals = ",eigvals)
        
        eigvals_chk = np.array(
            [  2.53636344e-10,   3.87263955e-10,   4.49523968e-10,
               8.17955744e-10,   8.17955744e-10,   1.22910388e-09,
               1.23803907e-09,   1.67547571e-09,   1.67547571e-09,
               1.75147770e-09,   2.20582127e-09,   2.20582127e-09,
               2.84333714e-09,   4.43169431e-09,   4.43169431e-09,
               1.75164250e-08,   9.38919779e+05,   9.57169426e+05,
               9.69271265e+05,   1.41963844e+06,   1.52443387e+06,
               1.89627852e+06,   1.97543819e+06,   2.05177386e+06,
               2.18333142e+06,   2.30078215e+06,   2.31036461e+06,
               2.40108194e+06,   2.63301339e+06,   2.72062783e+06,
               2.73970548e+06,   2.90332118e+06,   3.15705184e+06,
               3.86079309e+06,   3.87209620e+06,   4.70586582e+06,
               8.76738379e+06,   9.73067464e+06,   1.04062266e+07,
               1.15408214e+07,   1.21868610e+07,   1.33524791e+07,
               1.34516720e+07,   1.50757108e+07,   1.74290255e+07,
               1.83023511e+07,   2.15141700e+07,   2.22614418e+07,
               2.32690752e+07,   2.88149432e+07,   3.04306844e+07,
               3.10300863e+07,   4.25290585e+07,   8.95794195e+07,
               1.29723323e+08,   5.69560469e+08])
        
        #OLD2: eigvals_chk = np.array(
        #    [  9.81005045e-11,   4.43460242e-10,   4.43460242e-10,   4.49453396e-10,
        #       7.85029052e-10,   9.21626711e-10,   1.25309376e-09,   1.25309376e-09,
        #       2.04195500e-09,   2.04195500e-09,   2.55671745e-09,   2.55671745e-09,
        #       2.93033892e-09,   5.37903360e-09,   5.37903360e-09,   6.64245605e-09,
        #       8.03549557e+05,   8.11254999e+05,   9.92299959e+05,   1.27568332e+06,
        #       1.44009186e+06,   1.81498172e+06,   1.86373602e+06,   1.99016763e+06,
        #       2.18874496e+06,   2.25389002e+06,   2.32916014e+06,   2.41498248e+06,
        #       2.55666608e+06,   2.57241718e+06,   2.64636883e+06,   2.76332854e+06,
        #       3.06395675e+06,   3.73022730e+06,   3.80623619e+06,   4.44053284e+06,
        #       8.15965448e+06,   9.17787491e+06,   9.81390517e+06,   1.12014162e+07,
        #       1.18245119e+07,   1.28066806e+07,   1.34654943e+07,   1.45921884e+07,
        #       1.68278502e+07,   1.78792278e+07,   2.13539797e+07,   2.19466159e+07,
        #       2.21679148e+07,   2.77724533e+07,   2.94553174e+07,   3.00224355e+07,
        #       4.05213545e+07,   8.41911981e+07,   1.23090462e+08,   5.34640416e+08  ] )
        
        #OLD: eigvals_chk = np.array(
        #    [  2.36932621e-11,   4.57739349e-11,   8.49888013e-11,   8.49888013e-11,
        #       1.22859895e-10,   1.22859895e-10,   1.38705957e-10,   1.38705957e-10,
        #       3.75441328e-10,   6.46644807e-10,   6.46644807e-10,   7.06181642e-10,
        #       7.06181642e-10,   7.65472749e-10,   1.62672899e-09,   1.62672899e-09,
        #       9.23280752e+04,   9.88622140e+04,   1.24577730e+05,   1.40500652e+05,
        #       1.88461602e+05,   1.98466608e+05,   2.07084497e+05,   2.27489018e+05,
        #       2.33403285e+05,   2.52573024e+05,   2.82350782e+05,   3.12087185e+05,
        #       3.21855420e+05,   3.31659734e+05,   3.52649174e+05,   3.60682071e+05,
        #       3.90777833e+05,   4.59913853e+05,   5.02652879e+05,   5.59311926e+05,
        #       5.78891070e+05,   6.82325323e+05,   7.57318263e+05,   8.16739390e+05,
        #       1.06466062e+06,   1.20075694e+06,   1.37368639e+06,   1.58356629e+06,
        #       1.68898356e+06,   2.12277359e+06,   3.30650801e+06,   3.75869331e+06,
        #       4.00195245e+06,   4.42427797e+06,   5.06956256e+06,   7.31166332e+06,
        #       9.19432790e+06,   9.99944236e+06,   1.31027722e+07,   5.80310818e+07] )

        TOL = 1e-7
        for val,chk in zip(eigvals,eigvals_chk):
            if abs(val) > TOL or abs(chk) > TOL:
                self.assertAlmostEqual(abs(val-chk)/(abs(chk)+TOL), 0.0, places=3)
            # (else both chk and val are <= TOL, so both == 0 for our purposes)
        #print "eigvals = ",eigvals

    def test_confidenceRegion(self):

        res = pygsti.obj.Results()
        res.init_dataset(self.ds)
        res.init_gatestrings(self.gss)
        res.add_estimate(stdxyi.gs_target.copy(), stdxyi.gs_target.copy(),
                         [self.gateset]*len(self.maxLengthList), parameters={'objective': 'logl'},
                         estimate_key="default")
        
        est = res.estimates['default']
        est.add_confidence_region_factory('final iteration estimate', 'final')
        self.assertTrue( est.has_confidence_region_factory('final iteration estimate', 'final'))

        cfctry = est.get_confidence_region_factory('final iteration estimate', 'final')
        cfctry.compute_hessian()
        self.assertTrue( cfctry.has_hessian() )

        cfctry.project_hessian('std')
        cfctry.project_hessian('none')
        cfctry.project_hessian('optimal gate CIs')
        cfctry.project_hessian('intrinsic error')

        cfctry.enable_linear_response_errorbars()
        #{'dataset': self.ds, 'gateStringsToUse': list(self.ds.keys())})

        ci_std = cfctry.view( 95.0, 'normal', 'std')
        ci_noproj = cfctry.view( 95.0, 'normal', 'none')
        ci_intrinsic = cfctry.view( 95.0, 'normal', 'intrinsic error')
        ci_opt = cfctry.view( 95.0, 'normal', 'optimal gate CIs')
        #ci_linresponse = ??
        
        with self.assertRaises(ValueError):
            cfctry.project_hessian(95.0, 'normal', 'FooBar') #bad hessianProjection

        self.assertWarns(cfctry.view, 0.95, 'normal', 'none') # percentage < 1.0

        for ci_cur in (ci_std, ci_noproj, ci_opt, ci_intrinsic): # , ci_linresponse
            try: 
                ar_of_intervals_Gx = ci_cur.get_profile_likelihood_confidence_intervals("Gx")
                ar_of_intervals_rho0 = ci_cur.get_profile_likelihood_confidence_intervals("rho0")
                ar_of_intervals_E0 = ci_cur.get_profile_likelihood_confidence_intervals("E0")
                ar_of_intervals = ci_cur.get_profile_likelihood_confidence_intervals()
            except NotImplementedError: 
                pass #linear response CI doesn't support profile likelihood intervals
    
            def fnOfGate_float(mx,b):
                return float(mx[0,0])
            def fnOfGate_0D(mx,b):
                return np.array( float(mx[0,0]) )
            def fnOfGate_1D(mx,b):
                return mx[0,:]
            def fnOfGate_2D(mx,b):
                return mx[:,:]
            def fnOfGate_3D(mx,b):
                return np.zeros( (2,2,2), 'd') #just to test for error

            for fnOfGate in (fnOfGate_float, fnOfGate_0D, fnOfGate_1D, fnOfGate_2D, fnOfGate_3D):
                FnClass = gsf.gatefn_factory(fnOfGate)
                FnObj = FnClass(self.gateset, 'Gx')
                if fnOfGate is fnOfGate_3D:
                    with self.assertRaises(ValueError):
                        df = ci_cur.get_fn_confidence_interval(FnObj, verbosity=0)
                else:
                    df = ci_cur.get_fn_confidence_interval(FnObj, verbosity=0)
                    df, f0 = self.runSilent(ci_cur.get_fn_confidence_interval,
                                            FnObj, returnFnVal=True, verbosity=4)

            ##SHORT-CIRCUIT linear reponse here to reduce run time
            #if ci_cur is ci_linresponse: continue
    
            def fnOfVec_float(v,b):
                return float(v[0])
            def fnOfVec_0D(v,b):
                return np.array( float(v[0]) )
            def fnOfVec_1D(v,b):
                return np.array(v[:])
            def fnOfVec_2D(v,b):
                return np.dot(v.T,v)
            def fnOfVec_3D(v,b):
                return np.zeros( (2,2,2), 'd') #just to test for error
    
            for fnOfVec in (fnOfVec_float, fnOfVec_0D, fnOfVec_1D, fnOfVec_2D, fnOfVec_3D):
                FnClass = gsf.vecfn_factory(fnOfVec)
                FnObj = FnClass(self.gateset, 'rho0', 'prep')
                if fnOfVec is fnOfVec_3D:
                    with self.assertRaises(ValueError):
                        df = ci_cur.get_fn_confidence_interval(FnObj, verbosity=0)
                else:
                    df = ci_cur.get_fn_confidence_interval(FnObj, verbosity=0)
                    df, f0 = self.runSilent(ci_cur.get_fn_confidence_interval,
                                            FnObj, returnFnVal=True, verbosity=4)

            for fnOfVec in (fnOfVec_float, fnOfVec_0D, fnOfVec_1D, fnOfVec_2D, fnOfVec_3D):
                FnClass = gsf.vecfn_factory(fnOfVec)
                FnObj = FnClass(self.gateset, 'E0', 'effect')
                if fnOfVec is fnOfVec_3D:
                    with self.assertRaises(ValueError):
                        df = ci_cur.get_fn_confidence_interval(FnObj, verbosity=0)
                else:
                    df = ci_cur.get_fn_confidence_interval(FnObj, verbosity=0)
                    df, f0 = self.runSilent(ci_cur.get_fn_confidence_interval,
                                            FnObj, returnFnVal=True, verbosity=4)
    
    
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

            for fnOfSpam in (fnOfSpam_float, fnOfSpam_0D, fnOfSpam_1D, fnOfSpam_2D, fnOfSpam_3D):
                FnClass = gsf.spamfn_factory(fnOfSpam)
                FnObj = FnClass(self.gateset)
                if fnOfSpam is fnOfSpam_3D:
                    with self.assertRaises(ValueError):
                        df = ci_cur.get_fn_confidence_interval(FnObj, verbosity=0)
                else:
                    df = ci_cur.get_fn_confidence_interval(FnObj, verbosity=0)
                    df, f0 = self.runSilent(ci_cur.get_fn_confidence_interval,
                                            FnObj, returnFnVal=True, verbosity=4)

    
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

            for fnOfGateSet in (fnOfGateSet_float, fnOfGateSet_0D, fnOfGateSet_1D, fnOfGateSet_2D, fnOfGateSet_3D):
                FnClass = gsf.gatesetfn_factory(fnOfGateSet)
                FnObj = FnClass(self.gateset)
                if fnOfGateSet is fnOfGateSet_3D:
                    with self.assertRaises(ValueError):
                        df = ci_cur.get_fn_confidence_interval(FnObj, verbosity=0)
                else:
                    df = ci_cur.get_fn_confidence_interval(FnObj, verbosity=0)
                    df, f0 = self.runSilent(ci_cur.get_fn_confidence_interval,
                                            FnObj, returnFnVal=True, verbosity=4)

        #TODO: assert values of df & f0 ??

    def tets_pickle_ConfidenceRegion(self):
        res = pygsti.obj.Results()
        res.init_dataset(self.ds)
        res.init_gatestrings(self.gss)
        res.add_estimate(stdxyi.gs_target.copy(), stdxyi.gs_target.copy(),
                         [self.gateset]*len(self.maxLengthList), parameters={'objective': 'logl'},
                         estimate_key="default")
        
        res.add_confidence_region_factory('final iteration estimate', 'final')
        self.assertTrue( res.has_confidence_region_factory('final iteration estimate', 'final'))

        cfctry = res.get_confidence_region_factory('final iteration estimate', 'final')
        cfctry.compute_hessian()
        self.assertTrue( cfctry.has_hessian() )

        cfctry.project_hessian('std')
        ci_std = cfctry.view( 95.0, 'normal', 'std')

        import pickle
        s = pickle.dumps(cfctry)
        cifctry2 = pickle.loads(s)
        
        s = pickle.dumps(ci_std)
        ci_std2 = pickle.loads(s)
        
        #TODO: make sure ci_std and ci_std2 are the same


    def test_mapcalc_hessian(self):
        chi2, chi2Hessian = pygsti.chi2(self.ds, self.gateset,
                                        returnHessian=True)
        
        gs_mapcalc = self.gateset.copy()
        gs_mapcalc._calcClass = GateMapCalc
        chi2, chi2Hessian_mapcalc = pygsti.chi2(self.ds, self.gateset,
                                        returnHessian=True)

        self.assertArraysAlmostEqual(chi2Hessian, chi2Hessian_mapcalc)



if __name__ == "__main__":
    unittest.main(verbosity=2)
