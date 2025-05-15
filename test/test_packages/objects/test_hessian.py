import pickle
import unittest

import numpy as np
from pygsti.forwardsims.mapforwardsim import MapForwardSimulator

import pygsti
from pygsti import protocols as proto
from pygsti.modelpacks import smq1Q_XY
from pygsti.modelpacks import smq1Q_XYI
from pygsti.baseobjs import Label as L
from pygsti.report import modelfunction as gsf
from ..testutils import BaseTestCase, compare_files


class TestHessianMethods(BaseTestCase):

    def setUp(self):
        super(TestHessianMethods, self).setUp()

        self.model = smq1Q_XY.target_model()
        self.model = self.model.depolarize(spam_noise = .01, op_noise = .001)
        self.model = self.model.rotate(max_rotate=.005, seed=1234)

        prep_fiducials = smq1Q_XY.prep_fiducials()
        meas_fiducials = smq1Q_XY.meas_fiducials()
        germs = smq1Q_XY.germs()
        op_labels = list(self.model.operations.keys()) # also == std.gates
        self.maxLengthList = [1]
        #circuits for XY model.
        self.gss = pygsti.circuits.make_lsgst_structs(op_labels, prep_fiducials[0:4], 
                                                          meas_fiducials[0:3], smq1Q_XY.germs(), self.maxLengthList)

        self.edesign =  proto.CircuitListsDesign([pygsti.circuits.CircuitList(circuit_struct) for circuit_struct in self.gss])

        self.ds = pygsti.data.simulate_data(self.model, self.edesign.all_circuits_needing_data, 1000, seed = 1234)


    def test_parameter_counting(self):
        #XY Model: SPAM=True
        n = smq1Q_XY.target_model().num_params
        self.assertEqual(n,44) # 2*16 + 3*4 = 44

        n = smq1Q_XY.target_model().num_nongauge_params
        self.assertEqual(n,28) # full 16 gauge params

        #XY Model: SPAM=False
        tst = smq1Q_XY.target_model()
        del tst.preps['rho0']
        del tst.povms['Mdefault']
        n = tst.num_params
        self.assertEqual(n,32) # 2*16 = 32

        n = tst.num_nongauge_params
        self.assertEqual(n,18) # gates are all unital & TP => only 14 gauge params (2 casimirs)


        #XYI Model: SPAM=True
        n = smq1Q_XYI.target_model().num_params
        self.assertEqual(n,60) # 3*16 + 3*4 = 60

        n = smq1Q_XYI.target_model().num_nongauge_params
        self.assertEqual(n,44) # full 16 gauge params: SPAM gate + 3 others

        #XYI Model: SPAM=False
        tst = smq1Q_XYI.target_model()
        del tst.preps['rho0']
        del tst.povms['Mdefault']
        n = tst.num_params
        self.assertEqual(n,48) # 3*16 = 48

        n = tst.num_nongauge_params
        self.assertEqual(n,34) # gates are all unital & TP => only 14 gauge params (2 casimirs)

        #XYI Model: SP0=False
        tst = smq1Q_XYI.target_model()
        tst.preps['rho0'] = pygsti.modelmembers.states.TPState(tst.preps['rho0'])
        n = tst.num_params
        self.assertEqual(n,59) # 3*16 + 2*4 + 3 = 59

        n = tst.num_nongauge_params
        self.assertEqual(n,44) # 15 gauge params (minus one b/c can't change rho?)

        #XYI Model: G0=SP0=False
        tst.operations[L(())] = pygsti.modelmembers.operations.FullTPOp(tst.operations[L(())])
        tst.operations['Gxpi2',0] = pygsti.modelmembers.operations.FullTPOp(tst.operations['Gxpi2',0])
        tst.operations['Gypi2',0] = pygsti.modelmembers.operations.FullTPOp(tst.operations['Gypi2',0])
        n = tst.num_params
        self.assertEqual(n,47) # 3*12 + 2*4 + 3 = 47

        n = tst.num_nongauge_params
        self.assertEqual(n,35) # full 12 gauge params of single 4x3 gate

    def test_hessian_projection(self):
        chi2Hessian = pygsti.chi2_hessian(self.model, self.ds)

        proj_non_gauge = self.model.compute_nongauge_projector()
        projectedHessian = proj_non_gauge@chi2Hessian@proj_non_gauge

        self.assertEqual( projectedHessian.shape, (44,44) )
        self.assertEqual( np.linalg.matrix_rank(proj_non_gauge), 28)
        self.assertEqual( np.linalg.matrix_rank(projectedHessian), 28)

        eigvals = np.sort(abs(np.linalg.eigvals(projectedHessian)))

        print("eigvals = ",eigvals)

        eigvals_chk = np.array([ 5.45537035e-13, 5.45537035e-13, 1.47513013e-12, 1.47513013e-12,
                                 1.57813273e-12, 4.87695508e-12, 1.22061302e-11, 3.75982961e-11,
                                 5.49796401e-11, 5.62019047e-11, 5.62019047e-11, 7.06418308e-11,
                                 1.44881858e-10, 1.48934891e-10, 1.48934891e-10, 2.06194475e-10,
                                 1.91727543e+01, 2.26401298e+02, 5.23331036e+02, 1.16447879e+03,
                                 1.45737904e+03, 1.93375238e+03, 2.02017169e+03, 3.55570313e+03,
                                 3.95986905e+03, 5.52173250e+03, 8.20436174e+03, 9.93573257e+03,
                                 1.36092721e+04, 1.87334336e+04, 2.07723720e+04, 2.17070806e+04,
                                 2.72168569e+04, 3.31886655e+04, 3.72430633e+04, 4.64233389e+04,
                                 6.35672652e+04, 8.61196820e+04, 1.08248150e+05, 1.65647618e+05,
                                 5.72597674e+05, 9.44823397e+05, 1.45785061e+06, 6.85705713e+06])

        TOL = 1e-7
        for val,chk in zip(eigvals,eigvals_chk):
            if abs(val) > TOL or abs(chk) > TOL:
                self.assertAlmostEqual(abs(val-chk)/(abs(chk)+TOL), 0.0, places=2)
            # (else both chk and val are <= TOL, so both == 0 for our purposes)
        #print "eigvals = ",eigvals

    def test_confidenceRegion(self):

        data = proto.ProtocolData(self.edesign, self.ds)
        res = proto.ModelEstimateResults(data, proto.StandardGST(modes="full TP"))

        #Add estimate for hessian-based CI --------------------------------------------------
        builder = pygsti.objectivefns.ObjectiveFunctionBuilder(pygsti.objectivefns.PoissonPicDeltaLogLFunction)
        res.add_estimate(
            proto.estimate.Estimate.create_gst_estimate(
                res, smq1Q_XY.target_model(), smq1Q_XY.target_model(),
                [self.model] * len(self.maxLengthList), parameters={'final_objfn_builder': builder}),
            estimate_key="default"
        )

        est = res.estimates['default']
        est.add_confidence_region_factory('final iteration estimate', 'final')
        self.assertWarns(est.add_confidence_region_factory, 'final iteration estimate','final') #overwrites former
        self.assertTrue( est.has_confidence_region_factory('final iteration estimate', 'final'))

        cfctry = est.create_confidence_region_factory('final iteration estimate', 'final')
        self.assertFalse( cfctry.can_construct_views() ) # b/c no hessian or LR enabled yet...
        cfctry.compute_hessian(approximate=True)
        cfctry.compute_hessian()
        self.assertTrue( cfctry.has_hessian )
        self.assertFalse( cfctry.can_construct_views() ) # b/c hessian isn't projected yet...

        mdl_dummy = cfctry.model # test method
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
        res.add_estimate(
            proto.estimate.Estimate.create_gst_estimate(
                res, smq1Q_XY.target_model(), smq1Q_XY.target_model(),
                [self.model]*len(self.maxLengthList), parameters={'final_objfn_builder': builder}),
            estimate_key="linresponse"
        )

        estLR = res.estimates['linresponse']

        #estLR.add_confidence_region_factory('final iteration estimate', 'final') #Could do this, but use alt. method for more coverage
        with self.assertRaises(KeyError):
            estLR.create_confidence_region_factory('final iteration estimate', 'final') #won't create by default
        cfctryLR = estLR.create_confidence_region_factory('final iteration estimate', 'final', create_if_needed=True) #now it will
        self.assertTrue( estLR.has_confidence_region_factory('final iteration estimate', 'final'))

        #cfctryLR = estLR.create_confidence_region_factory('final iteration estimate', 'final') #done by 'get' call above
        self.assertFalse( cfctryLR.can_construct_views() ) # b/c no hessian or LR enabled yet...
        cfctryLR.enable_linear_response_errorbars() #parent results object is used to automatically populate params

        #self.assertTrue( cfctryLR.can_construct_views() )
        ci_linresponse = cfctryLR.view( 95.0, 'normal', None)

        mdl_dummy = cfctryLR.model # test method
        s = pickle.dumps(cfctryLR) # test pickle
        pickle.loads(s)


        #Add estimate for with bad objective ---------------------------------------------------------
        class FooBar:
            def __init__(self):
                self.cls_to_build = list  # an invalid objective class
                self.regularization = {}
                self.penalties = {}
        
        res.add_estimate(
            proto.estimate.Estimate.create_gst_estimate(
                res, smq1Q_XY.target_model(), smq1Q_XY.target_model(),
                [self.model]*len(self.maxLengthList), parameters={'final_objfn_builder': FooBar()}),
            estimate_key="foo"
        )

        est = res.estimates['foo']
        est.add_confidence_region_factory('final iteration estimate', 'final')
        with self.assertRaises(ValueError): # bad objective
            est.create_confidence_region_factory('final iteration estimate', 'final').compute_hessian()

        # Now test each of the views we created above ------------------------------------------------
        for ci_cur in (ci_std, ci_noproj, ci_opt, ci_intrinsic, ci_linresponse):

            s = pickle.dumps(ci_cur) # test pickle
            pickle.loads(s)

            #linear response CI doesn't support profile likelihood intervals
            if ci_cur is not ci_linresponse: # (profile likelihoods not implemented in this case)
                ar_of_intervals_Gx = ci_cur.retrieve_profile_likelihood_confidence_intervals(L("Gxpi2", 0))
                ar_of_intervals_rho0 = ci_cur.retrieve_profile_likelihood_confidence_intervals(L("rho0"))
                ar_of_intervals_M0 = ci_cur.retrieve_profile_likelihood_confidence_intervals(L("Mdefault"))
                ar_of_intervals = ci_cur.retrieve_profile_likelihood_confidence_intervals()

                with self.assertRaises(ValueError):
                    ci_cur.retrieve_profile_likelihood_confidence_intervals("foobar") #invalid label

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
                FnObj = FnClass(self.model, L('Gxpi2',0))
                if fnOfOp is fnOfGate_3D:
                    with self.assertRaises(ValueError):
                        df = ci_cur.compute_confidence_interval(FnObj, verbosity=0)
                else:
                    df = ci_cur.compute_confidence_interval(FnObj, verbosity=0)
                    df, f0 = self.runSilent(ci_cur.compute_confidence_interval,
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
                        df = ci_cur.compute_confidence_interval(FnObj, verbosity=0)
                else:
                    df = ci_cur.compute_confidence_interval(FnObj, verbosity=0)
                    df, f0 = self.runSilent(ci_cur.compute_confidence_interval,
                                            FnObj, return_fn_val=True, verbosity=4)

            for fnOfVec in (fnOfVec_float, fnOfVec_0D, fnOfVec_1D, fnOfVec_2D, fnOfVec_3D):
                FnClass = gsf.vecfn_factory(fnOfVec)
                FnObj = FnClass(self.model, 'Mdefault:0', 'effect')
                if fnOfVec is fnOfVec_3D:
                    with self.assertRaises(ValueError):
                        df = ci_cur.compute_confidence_interval(FnObj, verbosity=0)
                else:
                    df = ci_cur.compute_confidence_interval(FnObj, verbosity=0)
                    df, f0 = self.runSilent(ci_cur.compute_confidence_interval,
                                            FnObj, return_fn_val=True, verbosity=4)


            def fnOfSpam_float(rhoVecs, povms):
                lbls = list(povms[0].keys())
                return float( np.dot( rhoVecs[0].T, povms[0][lbls[0]] ) )
            def fnOfSpam_0D(rhoVecs, povms):
                lbls = list(povms[0].keys())
                return np.array( float( np.dot( rhoVecs[0].T, povms[0][lbls[0]] ) ) )
            def fnOfSpam_1D(rhoVecs, povms):
                lbls = list(povms[0].keys())
                return np.array( [ float(np.dot( rhoVecs[0].T, povms[0][lbls[0]]) ), 0] )
            def fnOfSpam_2D(rhoVecs, povms):
                lbls = list(povms[0].keys())
                return np.array( [[ float(np.dot( rhoVecs[0].T, povms[0][lbls[0]] )), 0],[0,0]] )
            def fnOfSpam_3D(rhoVecs, povms):
                return np.zeros( (2,2,2), 'd') #just to test for error

            for fnOfSpam in (fnOfSpam_float, fnOfSpam_0D, fnOfSpam_1D, fnOfSpam_2D, fnOfSpam_3D):
                FnClass = gsf.spamfn_factory(fnOfSpam)
                FnObj = FnClass(self.model)
                if fnOfSpam is fnOfSpam_3D:
                    with self.assertRaises(ValueError):
                        df = ci_cur.compute_confidence_interval(FnObj, verbosity=0)
                else:
                    df = ci_cur.compute_confidence_interval(FnObj, verbosity=0)
                    df, f0 = self.runSilent(ci_cur.compute_confidence_interval,
                                            FnObj, return_fn_val=True, verbosity=4)


            def fnOfGateSet_float(mdl):
                return float( mdl.operations['Gxpi2',0][0,0] )
            def fnOfGateSet_0D(mdl):
                return np.array( mdl.operations['Gxpi2',0][0,0]  )
            def fnOfGateSet_1D(mdl):
                return np.array( mdl.operations['Gxpi2',0][0,:] )
            def fnOfGateSet_2D(mdl):
                return np.array( mdl.operations['Gxpi2',0] )
            def fnOfGateSet_3D(mdl):
                return np.zeros( (2,2,2), 'd') #just to test for error

            for fnOfGateSet in (fnOfGateSet_float, fnOfGateSet_0D, fnOfGateSet_1D, fnOfGateSet_2D, fnOfGateSet_3D):
                FnClass = gsf.modelfn_factory(fnOfGateSet)
                FnObj = FnClass(self.model)
                if fnOfGateSet is fnOfGateSet_3D:
                    with self.assertRaises(ValueError):
                        df = ci_cur.compute_confidence_interval(FnObj, verbosity=0)
                else:
                    df = ci_cur.compute_confidence_interval(FnObj, verbosity=0)
                    df, f0 = self.runSilent(ci_cur.compute_confidence_interval,
                                            FnObj, return_fn_val=True, verbosity=4)

        #TODO: assert values of df & f0 ??

    def test_pickle_ConfidenceRegion(self):

        data = proto.ProtocolData(self.edesign, self.ds)
        res = proto.ModelEstimateResults(data, proto.StandardGST(modes="full TP"))

        res.add_estimate(
            proto.estimate.Estimate.create_gst_estimate(
                res, smq1Q_XY.target_model(), smq1Q_XY.target_model(),
                [self.model]*len(self.maxLengthList), parameters={'objective': 'logl'}),
            estimate_key="default"
        )

        est = res.estimates['default']
        est.add_confidence_region_factory('final iteration estimate', 'final')
        self.assertTrue( est.has_confidence_region_factory('final iteration estimate', 'final'))

        cfctry = est.create_confidence_region_factory('final iteration estimate', 'final')
        cfctry.compute_hessian()
        self.assertTrue( cfctry.has_hessian )

        cfctry.project_hessian('std')
        ci_std = cfctry.view( 95.0, 'normal', 'std')

        s = pickle.dumps(cfctry)
        cifctry2 = pickle.loads(s)

        s = pickle.dumps(ci_std)
        ci_std2 = pickle.loads(s)

        #TODO: make sure ci_std and ci_std2 are the same


    def test_mapcalc_hessian(self):
        chi2Hessian = pygsti.chi2_hessian(self.model, self.ds)

        mdl_mapcalc = self.model.copy()
        mdl_mapcalc._calcClass = MapForwardSimulator
        chi2Hessian_mapcalc = pygsti.chi2_hessian(self.model, self.ds)

        self.assertArraysAlmostEqual(chi2Hessian, chi2Hessian_mapcalc)



if __name__ == "__main__":
    unittest.main(verbosity=2)
