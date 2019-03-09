import unittest
import pygsti
import numpy as np
import warnings
import pickle
import os
import copy

from pygsti.construction import std1Q_XYI as std

from ..testutils import BaseTestCase, compare_files, temp_files

# This class is for unifying some models that get used in this file and in testGateSets2.py
class ResultsEstimateTestCase(BaseTestCase):

    def setUp(self):
        super(ResultsEstimateTestCase, self).setUp()

    def test_results_and_estimate_objects(self):

        class DummyComm(object):
            def __init__(self): pass
            def Get_rank(self): return 0
            def Get_size(self): return 1
            def bcast(self, x, root): return x
        
        #prepare a results object
        model = pygsti.io.load_model(compare_files + "/analysis.model")
        ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/analysis.dataset%s" % self.versionsuffix)

        fiducials = std.fiducials
        germs = std.germs
        opLabels = list(model.operations.keys()) # also == std.gates
        maxLengthList = [1,2]
        gss = pygsti.construction.make_lsgst_structs(opLabels, fiducials, fiducials, germs, maxLengthList)

        #init results
        res = pygsti.obj.Results()

        with self.assertRaises(ValueError):
            res.add_estimate(None,None,None,None) # dataset not init yet
        res.init_dataset(ds)

        with self.assertRaises(ValueError):
            res.add_estimate(None,None,None,None) # gss not init yet
        res.init_circuits(gss)

        self.assertWarns( res.init_dataset, ds ) # usually don't want to re-init
        self.assertWarns( res.init_circuits, gss ) # usually don't want to re-init
        with self.assertRaises(ValueError):
            res.init_circuits("foobar")
        res.init_circuits(gss) # make sure "foobar" test above doesn't leave results in an un-init state


        #add estimates
        res.add_estimate(std.target_model(), std.target_model(),
                         [model]*len(maxLengthList), parameters={'objective': 'logl'},
                         estimate_key="default")
        self.assertWarns(res.add_estimate, std.target_model(), std.target_model(),
                         [model]*len(maxLengthList), parameters={'objective': 'logl'},
                         estimate_key="default") #re-init existing estimate


        est = res.estimates['default']

          #Get effective estimate dataset
        effds = est.get_effective_dataset()
        effds, subMxs = est.get_effective_dataset(return_subMxs=True)

          #add gauge-optimized
        goparams = {'itemWeights': {'gates': 1.0, 'spam': 0.1}, 'method': 'BFGS' } #method so we don't need a legit comm
        est.add_gaugeoptimized(goparams, label="test", comm=None, verbosity=None)
        est.add_gaugeoptimized(goparams, label="test2", comm=DummyComm(), verbosity=None)

        #create estimate from scratch
        emptyEst = pygsti.objects.estimate.Estimate(res, targetModel=None, seedModel=None,
                                                    modeslByIter=None, parameters=None)
        res.estimates['empty'] = emptyEst # add in non-standard way so we can get away with not specifying
                                          # a target model adn modeslByIter.  Don't do this!

        goparams = {'itemWeights': {'gates': 1.0, 'spam': 0.1}, 'targetModel': model, 'model': model }
        emptyEst.add_gaugeoptimized(goparams, label="test", comm=None, verbosity=None) #OK

        with self.assertRaises(ValueError):
            goparams = {'itemWeights': {'gates': 1.0, 'spam': 0.1}, 'targetModel': model }
            emptyEst.add_gaugeoptimized(goparams, label="test", comm=None, verbosity=None) # goparams must have 'model'
        with self.assertRaises(ValueError):
            goparams = {'itemWeights': {'gates': 1.0, 'spam': 0.1}, 'model': model }
            emptyEst.add_gaugeoptimized(goparams, label="test", comm=None, verbosity=None) # goparams must have 'targetModel'

        #Estimate views
        est_view = est.view(None)
        est_view = est.view(['test'])

        #Estimate & results render as str
        print(str(est))
        print(str(res))

        #pickle Estimate
        s = pickle.dumps(est)
        est2 = pickle.loads(s)

        #Results views
        rview = res.view(['default'])
        rview2 = res.view('default') # this works too


        # add_estimates from other results
        res2 = pygsti.obj.Results()

        with self.assertRaises(ValueError):
            res2.add_estimates(res, ['default']) # ds not init yet...
        res2.init_dataset(ds)
        
        with self.assertRaises(ValueError):
            res2.add_estimates(res, ['default']) # gss not init yet...
        res2.init_circuits(gss)

        res2.add_estimates(res, ['default']) #now it works!
        self.assertWarns( res2.add_estimates, res, ['default'] ) # b/c re-init exising estimate

        # rename estimates
        res2.rename_estimate('default','renamed_default')
        with self.assertRaises(KeyError):
            res2.rename_estimate('foobar','renamed_foobar')

        # add estimate from model test
        mdl_guess = std.target_model().depolarize(op_noise=0.07,spam_noise=0.03)
        res2.add_model_test(std.target_model(), mdl_guess, estimate_key='Test', gauge_opt_keys="auto")

        chi2_res = pygsti.obj.Results()
        chi2_res.init_dataset(ds)        
        chi2_res.init_circuits(gss)
        chi2_res.add_estimate(std.target_model(), std.target_model(),
                         [model]*len(maxLengthList), parameters={'objective': 'chi2'},
                         estimate_key="default")
        chi2_res.add_model_test(std.target_model(), mdl_guess, estimate_key='Test', gauge_opt_keys="auto")

        chi2_res.estimates['default'].parameters['objective'] = "foobar" #sets up error below
        chi2_res.estimates['Test'].parameters['objective'] = "foobar"
        #print("DB: ",chi2_res.estimates.keys())
        with self.assertRaises(ValueError):
            chi2_res.add_model_test(std.target_model(), mdl_guess,
                                    estimate_key='Test', gauge_opt_keys="auto") # invalid "objective"


    def test_deprecated_report_fns(self):
        #deprecated functions that issue warnings
        res = pygsti.obj.Results()
        self.assertWarns( res.create_full_report_pdf )
        self.assertWarns( res.create_brief_report_pdf )
        self.assertWarns( res.create_presentation_pdf )
        self.assertWarns( res.create_presentation_ppt )
        self.assertWarns( res.create_general_report_pdf )

    def test_load_old_results(self):
        vs = "v2" if self.versionsuffix == "" else "v3"
        #pygsti.obj.results.enable_old_python_results_unpickling()
        with pygsti.io.enable_old_object_unpickling():
            with open(compare_files + "/pygsti0.9.6.results.pkl.%s" % vs,'rb') as f:
                results = pickle.load(f)
        #pygsti.obj.results.disable_old_python_results_unpickling()
        #pygsti.io.disable_old_object_unpickling()
        with open(temp_files + "/repickle_old_results.pkl.%s" % vs,'wb') as f:
            #pickle.dump(results.estimates['TP'].models['single'], f) # Debug
            pickle.dump(results, f)

        with pygsti.io.enable_old_object_unpickling("0.9.7"):
            with open(compare_files + "/pygsti0.9.7.results.pkl.%s" % vs,'rb') as f:
                results = pickle.load(f)
        with open(temp_files + "/repickle_old_results.pkl.%s" % vs,'wb') as f:
            pickle.dump(results, f)


        
if __name__ == '__main__':
    unittest.main(verbosity=2)
