import unittest
import warnings
import pygsti
from pygsti.modelpacks.legacy import std1Q_XYI as stdxyi
from pygsti.modelpacks.legacy import std1Q_XY as stdxy
from pygsti.objects import modelfunction as gsf
from pygsti.objects.mapforwardsim import MapForwardSimulator
from pygsti.objects import Label as L

import numpy as np
import sys, os
import pickle

from ..testutils import BaseTestCase, compare_files, temp_files


class TestHessianMethods(BaseTestCase):

    def setUp(self):
        super(TestHessianMethods, self).setUp()

        self.model = pygsti.io.load_model(compare_files + "/analysis.model")
        self.ds = pygsti.objects.DataSet(file_to_load_from=compare_files + "/analysis.dataset")


        fiducials = stdxyi.fiducials
        germs = stdxyi.germs
        opLabels = list(self.model.operations.keys()) # also == std.gates
        self.maxLengthList = [1,2]
        self.gss = pygsti.construction.make_lsgst_structs(opLabels, fiducials, fiducials, germs, self.maxLengthList)


    def test_parameter_counting(self):
        #XY Model: SPAM=True
        n = stdxy.target_model().num_params()
        self.assertEqual(n,44) # 2*16 + 3*4 = 44

        n = stdxy.target_model().num_nongauge_params()
        self.assertEqual(n,28) # full 16 gauge params

        #XY Model: SPAM=False
        tst = stdxy.target_model()
        del tst.preps['rho0']
        del tst.povms['Mdefault']
        n = tst.num_params()
        self.assertEqual(n,32) # 2*16 = 32

        n = tst.num_nongauge_params()
        self.assertEqual(n,18) # gates are all unital & TP => only 14 gauge params (2 casimirs)


        #XYI Model: SPAM=True
        n = stdxyi.target_model().num_params()
        self.assertEqual(n,60) # 3*16 + 3*4 = 60

        n = stdxyi.target_model().num_nongauge_params()
        self.assertEqual(n,44) # full 16 gauge params: SPAM gate + 3 others

        #XYI Model: SPAM=False
        tst = stdxyi.target_model()
        del tst.preps['rho0']
        del tst.povms['Mdefault']
        n = tst.num_params()
        self.assertEqual(n,48) # 3*16 = 48

        n = tst.num_nongauge_params()
        self.assertEqual(n,34) # gates are all unital & TP => only 14 gauge params (2 casimirs)

        #XYI Model: SP0=False
        tst = stdxyi.target_model()
        tst.preps['rho0'] = pygsti.obj.TPSPAMVec(tst.preps['rho0'])
        n = tst.num_params()
        self.assertEqual(n,59) # 3*16 + 2*4 + 3 = 59

        n = tst.num_nongauge_params()
        self.assertEqual(n,44) # 15 gauge params (minus one b/c can't change rho?)

        #XYI Model: G0=SP0=False
        tst.operations['Gi'] = pygsti.obj.TPDenseOp(tst.operations['Gi'])
        tst.operations['Gx'] = pygsti.obj.TPDenseOp(tst.operations['Gx'])
        tst.operations['Gy'] = pygsti.obj.TPDenseOp(tst.operations['Gy'])
        n = tst.num_params()
        self.assertEqual(n,47) # 3*12 + 2*4 + 3 = 47

        n = tst.num_nongauge_params()
        self.assertEqual(n,35) # full 12 gauge params of single 4x3 gate


    def test_hessian_projection(self):

        chi2, chi2Grad, chi2Hessian = pygsti.chi2(self.model, self.ds,
                                                  return_gradient=True,
                                                  return_hessian=True)

        proj_non_gauge = self.model.get_nongauge_projector()
        projectedHessian = np.dot(proj_non_gauge,
                                  np.dot(chi2Hessian, proj_non_gauge))

        print(self.model.num_params())
        print(proj_non_gauge.shape)
        self.assertEqual( projectedHessian.shape, (60,60) )
        #print("Evals = ")
        #print("\n".join( [ "%d: %g" % (i,ev) for i,ev in enumerate(np.linalg.eigvals(projectedHessian))] ))
        self.assertEqual( np.linalg.matrix_rank(proj_non_gauge), 44)
        self.assertEqual( np.linalg.matrix_rank(projectedHessian), 44)

        eigvals = np.sort(abs(np.linalg.eigvals(projectedHessian)))

        print("eigvals = ",eigvals)

        eigvals_chk = np.array([2.51663034e-10, 2.51663034e-10, 6.81452335e-10, 7.72039792e-10,
                                8.76915081e-10, 8.76915081e-10, 1.31455011e-09, 3.03808236e-09,
                                3.03808236e-09, 3.13457752e-09, 3.21805358e-09, 3.21805358e-09,
                                4.78549720e-09, 7.83389490e-09, 1.82493106e-08, 1.82493106e-08,
                                9.23087831e+05, 1.05783101e+06, 1.16457705e+06, 1.39492929e+06,
                                1.84015484e+06, 2.10613947e+06, 2.37963392e+06, 2.47192689e+06,
                                2.64566761e+06, 2.68722871e+06, 2.82383377e+06, 2.86584033e+06,
                                2.94590436e+06, 2.96180212e+06, 3.08322015e+06, 3.29389050e+06,
                                3.66581786e+06, 3.76266448e+06, 3.81921738e+06, 3.86624688e+06,
                                3.89045873e+06, 4.72831630e+06, 4.96416855e+06, 6.53286834e+06,
                                1.01424911e+07, 1.11347312e+07, 1.26152967e+07, 1.30081040e+07,
                                1.36647082e+07, 1.49293583e+07, 1.58234599e+07, 1.80999182e+07,
                                2.09155048e+07, 2.17444267e+07, 2.46870311e+07, 2.64427393e+07,
                                2.72410297e+07, 3.34988002e+07, 3.45005948e+07, 3.69040745e+07,
                                5.08647137e+07, 9.43153151e+07, 1.36088308e+08, 6.30304807e+08])

        TOL = 1e-7
        for val,chk in zip(eigvals,eigvals_chk):
            if abs(val) > TOL or abs(chk) > TOL:
                self.assertAlmostEqual(abs(val-chk)/(abs(chk)+TOL), 0.0, places=3)
            # (else both chk and val are <= TOL, so both == 0 for our purposes)
        #print "eigvals = ",eigvals

    def test_confidenceRegion(self):

        res = pygsti.obj.Results()
        res.init_dataset(self.ds)
        res.init_circuits(self.gss)

        #Add estimate for hessian-based CI --------------------------------------------------
        res.add_estimate(stdxyi.target_model(), stdxyi.target_model(),
                         [self.model]*len(self.maxLengthList), parameters={'objective': 'logl'},
                         estimate_key="default")

        est = res.estimates['default']
        est.add_confidence_region_factory('final iteration estimate', 'final')
        self.assertWarns(est.add_confidence_region_factory, 'final iteration estimate','final') #overwrites former
        self.assertTrue( est.has_confidence_region_factory('final iteration estimate', 'final'))

        cfctry = est.get_confidence_region_factory('final iteration estimate', 'final')
        self.assertFalse( cfctry.can_construct_views() ) # b/c no hessian or LR enabled yet...
        cfctry.compute_hessian(approximate=True)
        cfctry.compute_hessian()
        self.assertTrue( cfctry.has_hessian() )
        self.assertFalse( cfctry.can_construct_views() ) # b/c hessian isn't projected yet...

        mdl_dummy = cfctry.get_model() # test method
        s = pickle.dumps(cfctry) # test pickle
        pickle.loads(s)

        cfctry.project_hessian('std')
        cfctry.project_hessian('none')
        cfctry.project_hessian('optimal gate CIs')
        cfctry.project_hessian('intrinsic error')
        with self.assertRaises(ValueError):
            cfctry.project_hessian(95.0, 'normal', 'FooBar') #bad hessianProjection

        self.assertTrue( cfctry.can_construct_views() )

        ci_std = cfctry.view( 95.0, 'normal', 'std')
        ci_noproj = cfctry.view( 95.0, 'normal', 'none')
        ci_intrinsic = cfctry.view( 95.0, 'normal', 'intrinsic error')
        ci_opt = cfctry.view( 95.0, 'normal', 'optimal gate CIs')

        with self.assertRaises(ValueError):
            cfctry.view(95.0, 'foobar') #bad region type

        self.assertWarns(cfctry.view, 0.95, 'normal', 'none') # percentage < 1.0


        #Add estimate for linresponse-based CI --------------------------------------------------
        res.add_estimate(stdxyi.target_model(), stdxyi.target_model(),
                         [self.model]*len(self.maxLengthList), parameters={'objective': 'logl'},
                         estimate_key="linresponse")

        estLR = res.estimates['linresponse']

        #estLR.add_confidence_region_factory('final iteration estimate', 'final') #Could do this, but use alt. method for more coverage
        with self.assertRaises(KeyError):
            estLR.get_confidence_region_factory('final iteration estimate', 'final') #won't create by default
        cfctryLR = estLR.get_confidence_region_factory('final iteration estimate', 'final', create_if_needed=True) #now it will
        self.assertTrue( estLR.has_confidence_region_factory('final iteration estimate', 'final'))

        #cfctryLR = estLR.get_confidence_region_factory('final iteration estimate', 'final') #done by 'get' call above
        self.assertFalse( cfctryLR.can_construct_views() ) # b/c no hessian or LR enabled yet...
        cfctryLR.enable_linear_response_errorbars() #parent results object is used to automatically populate params

        #self.assertTrue( cfctryLR.can_construct_views() )
        ci_linresponse = cfctryLR.view( 95.0, 'normal', None)

        mdl_dummy = cfctryLR.get_model() # test method
        s = pickle.dumps(cfctryLR) # test pickle
        pickle.loads(s)


        #Add estimate for with bad objective ---------------------------------------------------------
        res.add_estimate(stdxyi.target_model(), stdxyi.target_model(),
                         [self.model]*len(self.maxLengthList), parameters={'objective': 'foobar'},
                         estimate_key="foo")
        est = res.estimates['foo']
        est.add_confidence_region_factory('final iteration estimate', 'final')
        with self.assertRaises(ValueError): # bad objective
            est.get_confidence_region_factory('final iteration estimate', 'final').compute_hessian()



        # Now test each of the views we created above ------------------------------------------------
        for ci_cur in (ci_std, ci_noproj, ci_opt, ci_intrinsic, ci_linresponse):

            s = pickle.dumps(ci_cur) # test pickle
            pickle.loads(s)

            #linear response CI doesn't support profile likelihood intervals
            if ci_cur is not ci_linresponse: # (profile likelihoods not implemented in this case)
                ar_of_intervals_Gx = ci_cur.get_profile_likelihood_confidence_intervals(L("Gx"))
                ar_of_intervals_rho0 = ci_cur.get_profile_likelihood_confidence_intervals(L("rho0"))
                ar_of_intervals_M0 = ci_cur.get_profile_likelihood_confidence_intervals(L("Mdefault"))
                ar_of_intervals = ci_cur.get_profile_likelihood_confidence_intervals()

                with self.assertRaises(ValueError):
                    ci_cur.get_profile_likelihood_confidence_intervals("foobar") #invalid label

            def fnOfGate_float(mx,b):
                return float(mx[0,0])
            def fnOfGate_complex(mx,b):
                return complex(mx[0,0] + 1.0j)
            def fnOfGate_0D(mx,b):
                return np.array( float(mx[0,0]) )
            def fnOfGate_1D(mx,b):
                return mx[0,:]
            def fnOfGate_2D(mx,b):
                return mx[:,:]
            def fnOfGate_2D_complex(mx,b):
                return np.array(mx[:,:] + 1j*mx[:,:],'complex')
            def fnOfGate_3D(mx,b):
                return np.zeros( (2,2,2), 'd') #just to test for error

            fns = (fnOfGate_float, fnOfGate_0D, fnOfGate_1D,
                   fnOfGate_2D, fnOfGate_3D)
            if ci_cur is not ci_linresponse: # complex functions not supported by linresponse CIs
                fns += (fnOfGate_complex, fnOfGate_2D_complex)

            for fnOfOp in fns:
                FnClass = gsf.opfn_factory(fnOfOp)
                FnObj = FnClass(self.model, 'Gx')
                if fnOfOp is fnOfGate_3D:
                    with self.assertRaises(ValueError):
                        df = ci_cur.get_fn_confidence_interval(FnObj, verbosity=0)
                else:
                    df = ci_cur.get_fn_confidence_interval(FnObj, verbosity=0)
                    df, f0 = self.runSilent(ci_cur.get_fn_confidence_interval,
                                            FnObj, return_fn_val=True, verbosity=4)

            ##SHORT-CIRCUIT linear reponse here to reduce run time
            if ci_cur is ci_linresponse: continue

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
                FnObj = FnClass(self.model, 'rho0', 'prep')
                if fnOfVec is fnOfVec_3D:
                    with self.assertRaises(ValueError):
                        df = ci_cur.get_fn_confidence_interval(FnObj, verbosity=0)
                else:
                    df = ci_cur.get_fn_confidence_interval(FnObj, verbosity=0)
                    df, f0 = self.runSilent(ci_cur.get_fn_confidence_interval,
                                            FnObj, return_fn_val=True, verbosity=4)

            for fnOfVec in (fnOfVec_float, fnOfVec_0D, fnOfVec_1D, fnOfVec_2D, fnOfVec_3D):
                FnClass = gsf.vecfn_factory(fnOfVec)
                FnObj = FnClass(self.model, 'Mdefault:0', 'effect')
                if fnOfVec is fnOfVec_3D:
                    with self.assertRaises(ValueError):
                        df = ci_cur.get_fn_confidence_interval(FnObj, verbosity=0)
                else:
                    df = ci_cur.get_fn_confidence_interval(FnObj, verbosity=0)
                    df, f0 = self.runSilent(ci_cur.get_fn_confidence_interval,
                                            FnObj, return_fn_val=True, verbosity=4)


            def fnOfSpam_float(rhoVecs, povms):
                lbls = list(povms[0].keys())
                return float( np.dot( rhoVecs[0].T, povms[0][lbls[0]] ) )
            def fnOfSpam_0D(rhoVecs, povms):
                lbls = list(povms[0].keys())
                return np.array( float( np.dot( rhoVecs[0].T, povms[0][lbls[0]] ) ) )
            def fnOfSpam_1D(rhoVecs, povms):
                lbls = list(povms[0].keys())
                return np.array( [ np.dot( rhoVecs[0].T, povms[0][lbls[0]] ), 0] )
            def fnOfSpam_2D(rhoVecs, povms):
                lbls = list(povms[0].keys())
                return np.array( [[ np.dot( rhoVecs[0].T, povms[0][lbls[0]] ), 0],[0,0]] )
            def fnOfSpam_3D(rhoVecs, povms):
                return np.zeros( (2,2,2), 'd') #just to test for error

            for fnOfSpam in (fnOfSpam_float, fnOfSpam_0D, fnOfSpam_1D, fnOfSpam_2D, fnOfSpam_3D):
                FnClass = gsf.spamfn_factory(fnOfSpam)
                FnObj = FnClass(self.model)
                if fnOfSpam is fnOfSpam_3D:
                    with self.assertRaises(ValueError):
                        df = ci_cur.get_fn_confidence_interval(FnObj, verbosity=0)
                else:
                    df = ci_cur.get_fn_confidence_interval(FnObj, verbosity=0)
                    df, f0 = self.runSilent(ci_cur.get_fn_confidence_interval,
                                            FnObj, return_fn_val=True, verbosity=4)


            def fnOfGateSet_float(mdl):
                return float( mdl.operations['Gx'][0,0] )
            def fnOfGateSet_0D(mdl):
                return np.array( mdl.operations['Gx'][0,0]  )
            def fnOfGateSet_1D(mdl):
                return np.array( mdl.operations['Gx'][0,:] )
            def fnOfGateSet_2D(mdl):
                return np.array( mdl.operations['Gx'] )
            def fnOfGateSet_3D(mdl):
                return np.zeros( (2,2,2), 'd') #just to test for error

            for fnOfGateSet in (fnOfGateSet_float, fnOfGateSet_0D, fnOfGateSet_1D, fnOfGateSet_2D, fnOfGateSet_3D):
                FnClass = gsf.modelfn_factory(fnOfGateSet)
                FnObj = FnClass(self.model)
                if fnOfGateSet is fnOfGateSet_3D:
                    with self.assertRaises(ValueError):
                        df = ci_cur.get_fn_confidence_interval(FnObj, verbosity=0)
                else:
                    df = ci_cur.get_fn_confidence_interval(FnObj, verbosity=0)
                    df, f0 = self.runSilent(ci_cur.get_fn_confidence_interval,
                                            FnObj, return_fn_val=True, verbosity=4)

        #TODO: assert values of df & f0 ??

    def tets_pickle_ConfidenceRegion(self):
        res = pygsti.obj.Results()
        res.init_dataset(self.ds)
        res.init_circuits(self.gss)
        res.add_estimate(stdxyi.target_model(), stdxyi.target_model(),
                         [self.model]*len(self.maxLengthList), parameters={'objective': 'logl'},
                         estimate_key="default")

        res.add_confidence_region_factory('final iteration estimate', 'final')
        self.assertTrue( res.has_confidence_region_factory('final iteration estimate', 'final'))

        cfctry = res.get_confidence_region_factory('final iteration estimate', 'final')
        cfctry.compute_hessian()
        self.assertTrue( cfctry.has_hessian() )

        cfctry.project_hessian('std')
        ci_std = cfctry.view( 95.0, 'normal', 'std')

        s = pickle.dumps(cfctry)
        cifctry2 = pickle.loads(s)

        s = pickle.dumps(ci_std)
        ci_std2 = pickle.loads(s)

        #TODO: make sure ci_std and ci_std2 are the same


    def test_mapcalc_hessian(self):
        chi2, chi2Hessian = pygsti.chi2(self.model, self.ds,
                                        return_hessian=True)

        mdl_mapcalc = self.model.copy()
        mdl_mapcalc._calcClass = MapForwardSimulator
        chi2, chi2Hessian_mapcalc = pygsti.chi2(self.model, self.ds,
                                        return_hessian=True)

        self.assertArraysAlmostEqual(chi2Hessian, chi2Hessian_mapcalc)



if __name__ == "__main__":
    unittest.main(verbosity=2)
