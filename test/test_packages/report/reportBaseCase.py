import unittest
import warnings
import pickle
import collections
import pygsti
import os
from pygsti.modelpacks.legacy import std1Q_XYI as std
from ..testutils import BaseTestCase, compare_files, temp_files, regenerate_references

import numpy as np


class ReportBaseCase(BaseTestCase):

    @classmethod
    def setUpClass(cls):
        """
        Handle all once-per-class (slow) computation and loading,
         to avoid calling it for each test (like setUp).  Store
         results in class variable for use within setUp.
        """
        super(ReportBaseCase, cls).setUpClass()

        orig_cwd = os.getcwd()
        os.chdir(os.path.abspath(os.path.dirname(__file__)))
        os.chdir('..') # The test_packages directory

        targetModel = std.target_model()
        datagen_gateset = targetModel.depolarize(op_noise=0.05, spam_noise=0.1)
        datagen_gateset2 = targetModel.depolarize(op_noise=0.1, spam_noise=0.05).rotate((0.15,-0.03,0.03))

        #cls.specs = pygsti.construction.build_spam_specs(std.fiducials, effect_labels=['E0'])
        #  #only use the first EVec

        opLabels = std.gates
        cls.lgstStrings = pygsti.construction.list_lgst_circuits(std.fiducials, std.fiducials, opLabels)
        cls.maxLengthList = [1,2,4,8]

        cls.lsgstStrings = pygsti.construction.make_lsgst_lists(
            opLabels, std.fiducials, std.fiducials, std.germs, cls.maxLengthList)
        cls.lsgstStructs = pygsti.construction.make_lsgst_structs(
            opLabels, std.fiducials, std.fiducials, std.germs, cls.maxLengthList)


        # RUN BELOW LINES TO GENERATE ANALYSIS DATASET (SAVE)
        if regenerate_references():
            ds = pygsti.construction.generate_fake_data(datagen_gateset, cls.lsgstStrings[-1], n_samples=1000,
                                                        sample_error='binomial', seed=100)
            ds.save(compare_files + "/reportgen.dataset")
            ds2 = pygsti.construction.generate_fake_data(datagen_gateset2, cls.lsgstStrings[-1], n_samples=1000,
                                                         sample_error='binomial', seed=100)
            ds2.save(compare_files + "/reportgen2.dataset")


        cls.ds = pygsti.objects.DataSet(file_to_load_from=compare_files + "/reportgen.dataset")
        cls.ds2 = pygsti.objects.DataSet(file_to_load_from=compare_files + "/reportgen2.dataset")

        mdl_lgst = pygsti.do_lgst(cls.ds, std.fiducials, std.fiducials, targetModel, svdTruncateTo=4, verbosity=0)
        mdl_lgst_go = pygsti.gaugeopt_to_target(mdl_lgst, targetModel, {'gates': 1.0, 'spam': 0.0})
        cls.mdl_clgst = pygsti.contract(mdl_lgst_go, "CPTP")
        cls.mdl_clgst_tp = pygsti.contract(cls.mdl_clgst, "vSPAM")
        cls.mdl_clgst_tp.set_all_parameterizations("TP")

        #Compute results for MC2GST
        lsgst_gatesets_prego = pygsti.do_iterative_mc2gst(
            cls.ds, cls.mdl_clgst, cls.lsgstStrings, verbosity=0,
            minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6),
            returnAll=True)

        cls.results = pygsti.objects.Results()
        cls.results.init_dataset(cls.ds)
        cls.results.init_circuits(cls.lsgstStructs)
        cls.results.add_estimate(targetModel, cls.mdl_clgst,
                                 lsgst_gatesets_prego,
                                 {'objective': "chi2",
                                  'minProbClipForWeighting': 1e-4,
                                  'probClipInterval': (-1e6,1e6), 'radius': 1e-4,
                                  'weights': None, 'defaultDirectory': temp_files + "",
                                  'defaultBasename': "MyDefaultReportName"})

        gaugeOptParams = collections.OrderedDict([
                ('model', lsgst_gatesets_prego[-1]),  #so can gauge-propagate CIs
                ('targetModel', targetModel),       #so can gauge-propagate CIs
                ('cptp_penalty_factor', 0),
                ('gates_metric',"frobenius"),
                ('spam_metric',"frobenius"),
                ('item_weights', {'gates': 1.0, 'spam': 0.001}),
                ('returnAll', True) ])

        _, gaugeEl, go_final_gateset = pygsti.gaugeopt_to_target(**gaugeOptParams)
        gaugeOptParams['_gaugeGroupEl'] = gaugeEl  #so can gauge-propagate CIs
        cls.results.estimates['default'].add_gaugeoptimized(gaugeOptParams, go_final_gateset)
        cls.results.estimates['default'].add_gaugeoptimized(gaugeOptParams, go_final_gateset, "go_dup")

        #Compute results for MLGST with TP constraint
        # Use do_long_sequence_gst with a non-mark dataset to trigger data scaling
        tp_target = targetModel.copy(); tp_target.set_all_parameterizations("TP")


        cls.ds3 = cls.ds.copy_nonstatic()
        cls.ds3.add_counts_from_dataset(cls.ds2)
        cls.ds3.done_adding_data()

        cls.results_logL = pygsti.do_long_sequence_gst(cls.ds3, tp_target, std.fiducials, std.fiducials,
                                                       std.germs, cls.maxLengthList, verbosity=0,
                                                       advanced_options={'tolerance': 1e-6, 'starting point': 'LGST',
                                                                        'onBadFit': ["robust","Robust","robust+","Robust+"],
                                                                        'badFitThreshold': -1.0,
                                                                        'germLengthLimits': {('Gx','Gi','Gi'): 2} })
        #OLD
        #lsgst_gatesets_TP = pygsti.do_iterative_mlgst(cls.ds, cls.mdl_clgst_tp, cls.lsgstStrings, verbosity=0,
        #                                           minProbClip=1e-4, probClipInterval=(-1e6,1e6),
        #                                           returnAll=True) #TP initial model => TP output models
        #cls.results_logL = pygsti.objects.Results()
        #cls.results_logL.init_dataset(cls.ds)
        #cls.results_logL.init_circuits(cls.lsgstStructs)
        #cls.results_logL.add_estimate(targetModel, cls.mdl_clgst_tp,
        #                         lsgst_gatesets_TP,
        #                         {'objective': "logl",
        #                          'minProbClip': 1e-4,
        #                          'probClipInterval': (-1e6,1e6), 'radius': 1e-4,
        #                          'weights': None, 'defaultDirectory': temp_files + "",
        #                          'defaultBasename': "MyDefaultReportName"})
        #
        #tp_target = targetModel.copy(); tp_target.set_all_parameterizations("TP")
        #gaugeOptParams = gaugeOptParams.copy() #just to be safe
        #gaugeOptParams['model'] = lsgst_gatesets_TP[-1]  #so can gauge-propagate CIs
        #gaugeOptParams['targetModel'] = tp_target  #so can gauge-propagate CIs
        #_, gaugeEl, go_final_gateset = pygsti.gaugeopt_to_target(**gaugeOptParams)
        #gaugeOptParams['_gaugeGroupEl'] = gaugeEl #so can gauge-propagate CIs
        #cls.results_logL.estimates['default'].add_gaugeoptimized(gaugeOptParams, go_final_gateset)
        #
        ##self.results_logL.options.precision = 3
        ##self.results_logL.options.polar_precision = 2

        os.chdir(orig_cwd)



    def setUp(self):
        super(ReportBaseCase, self).setUp()

        cls = self.__class__

        self.targetModel = std.target_model()
        self.fiducials = std.fiducials[:]
        self.germs = std.germs[:]
        self.opLabels = std.gates

        #self.specs = cls.specs
        self.maxLengthList = cls.maxLengthList[:]
        self.lgstStrings = cls.lgstStrings
        self.ds = cls.ds

        self.mdl_clgst = cls.mdl_clgst.copy()
        self.mdl_clgst_tp = cls.mdl_clgst_tp.copy()

        self.results = cls.results.copy()
        self.results_logL = cls.results_logL.copy()
