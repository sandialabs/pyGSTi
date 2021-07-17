import collections
import os

import pygsti
from pygsti.modelpacks.legacy import std1Q_XYI as std
from ..testutils import BaseTestCase, compare_files, temp_files, regenerate_references


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

        target_model = std.target_model()
        datagen_gateset = target_model.depolarize(op_noise=0.05, spam_noise=0.1)
        datagen_gateset2 = target_model.depolarize(op_noise=0.1, spam_noise=0.05).rotate((0.15,-0.03,0.03))

        #cls.specs = pygsti.construction.build_spam_specs(std.fiducials, effect_labels=['E0'])
        #  #only use the first EVec

        op_labels = std.gates
        cls.lgstStrings = pygsti.circuits.create_lgst_circuits(std.fiducials, std.fiducials, op_labels)
        cls.maxLengthList = [1,2,4,8]

        cls.lsgstStrings = pygsti.circuits.create_lsgst_circuit_lists(
            op_labels, std.fiducials, std.fiducials, std.germs, cls.maxLengthList)
        cls.lsgstStructs = pygsti.circuits.make_lsgst_structs(
            op_labels, std.fiducials, std.fiducials, std.germs, cls.maxLengthList)


        # RUN BELOW LINES TO GENERATE ANALYSIS DATASET (SAVE)
        if regenerate_references():
            ds = pygsti.data.simulate_data(datagen_gateset, cls.lsgstStrings[-1], num_samples=1000,
                                                   sample_error='binomial', seed=100)
            ds.save(compare_files + "/reportgen.dataset")
            ds2 = pygsti.data.simulate_data(datagen_gateset2, cls.lsgstStrings[-1], num_samples=1000,
                                                    sample_error='binomial', seed=100)
            ds2.save(compare_files + "/reportgen2.dataset")


        cls.ds = pygsti.objects.DataSet(file_to_load_from=compare_files + "/reportgen.dataset")
        cls.ds2 = pygsti.objects.DataSet(file_to_load_from=compare_files + "/reportgen2.dataset")

        mdl_lgst = pygsti.run_lgst(cls.ds, std.fiducials, std.fiducials, target_model, svd_truncate_to=4, verbosity=0)
        mdl_lgst_go = pygsti.gaugeopt_to_target(mdl_lgst, target_model, {'gates': 1.0, 'spam': 0.0})
        cls.mdl_clgst = pygsti.contract(mdl_lgst_go, "CPTP")
        cls.mdl_clgst_tp = pygsti.contract(cls.mdl_clgst, "vSPAM")
        cls.mdl_clgst_tp.set_all_parameterizations("TP")

        #Compute results for MC2GST
        lsgst_gatesets_prego, *_ = pygsti.run_iterative_gst(
            cls.ds, cls.mdl_clgst, cls.lsgstStrings,
            optimizer={'tol': 1e-5},
            iteration_objfn_builders=['chi2'],
            final_objfn_builders=[],
            resource_alloc=None,
            verbosity=0
        )

        experiment_design = pygsti.protocols.StandardGSTDesign(
            target_model, std.fiducials, std.fiducials, std.germs, cls.maxLengthList
        )
        data = pygsti.protocols.ProtocolData(experiment_design, cls.ds)
        protocol = pygsti.protocols.StandardGST()
        cls.results = pygsti.protocols.gst.ModelEstimateResults(data, protocol)
        cls.results.add_estimate(pygsti.protocols.estimate.Estimate.create_gst_estimate(
            cls.results, target_model, cls.mdl_clgst,lsgst_gatesets_prego,
            {'objective': "chi2",
             'min_prob_clip_for_weighting': 1e-4,
             'prob_clip_interval': (-1e6,1e6), 'radius': 1e-4,
             'weights': None, 'defaultDirectory': temp_files + "",
             'defaultBasename': "MyDefaultReportName"}
        ))

        gaugeOptParams = collections.OrderedDict([
                ('model', lsgst_gatesets_prego[-1]),  #so can gauge-propagate CIs
                ('target_model', target_model),       #so can gauge-propagate CIs
                ('cptp_penalty_factor', 0),
                ('gates_metric',"frobenius"),
                ('spam_metric',"frobenius"),
                ('item_weights', {'gates': 1.0, 'spam': 0.001}),
                ('return_all', True) ])

        _, gaugeEl, go_final_gateset = pygsti.gaugeopt_to_target(**gaugeOptParams)
        gaugeOptParams['_gaugeGroupEl'] = gaugeEl  #so can gauge-propagate CIs
        cls.results.estimates['default'].add_gaugeoptimized(gaugeOptParams, go_final_gateset)
        cls.results.estimates['default'].add_gaugeoptimized(gaugeOptParams, go_final_gateset, "go_dup")

        #Compute results for MLGST with TP constraint
        # Use run_long_sequence_gst with a non-mark dataset to trigger data scaling
        tp_target = target_model.copy(); tp_target.set_all_parameterizations("TP")


        cls.ds3 = cls.ds.copy_nonstatic()
        cls.ds3.add_counts_from_dataset(cls.ds2)
        cls.ds3.done_adding_data()

        cls.results_logL = pygsti.run_long_sequence_gst(cls.ds3, tp_target, std.fiducials, std.fiducials,
                                                        std.germs, cls.maxLengthList, verbosity=0,
                                                        advanced_options={'tolerance': 1e-6, 'starting_point': 'LGST',
                                                                        'on_bad_fit': ["robust","Robust","robust+","Robust+"],
                                                                        'bad_fit_threshold': -1.0,
                                                                        'germ_length_limits': {('Gx','Gi','Gi'): 2} })
        #OLD
        #lsgst_gatesets_TP = pygsti.do_iterative_mlgst(cls.ds, cls.mdl_clgst_tp, cls.lsgstStrings, verbosity=0,
        #                                           min_prob_clip=1e-4, prob_clip_interval=(-1e6,1e6),
        #                                           returnAll=True) #TP initial model => TP output models
        #cls.results_logL = pygsti.objects.Results()
        #cls.results_logL.init_dataset(cls.ds)
        #cls.results_logL.init_circuits(cls.lsgstStructs)
        #cls.results_logL.add_estimate(target_model, cls.mdl_clgst_tp,
        #                         lsgst_gatesets_TP,
        #                         {'objective': "logl",
        #                          'min_prob_clip': 1e-4,
        #                          'prob_clip_interval': (-1e6,1e6), 'radius': 1e-4,
        #                          'weights': None, 'defaultDirectory': temp_files + "",
        #                          'defaultBasename': "MyDefaultReportName"})
        #
        #tp_target = target_model.copy(); tp_target.set_all_parameterizations("TP")
        #gaugeOptParams = gaugeOptParams.copy() #just to be safe
        #gaugeOptParams['model'] = lsgst_gatesets_TP[-1]  #so can gauge-propagate CIs
        #gaugeOptParams['target_model'] = tp_target  #so can gauge-propagate CIs
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

        self.target_model = std.target_model()
        self.fiducials = std.fiducials[:]
        self.germs = std.germs[:]
        self.op_labels = std.gates

        #self.specs = cls.specs
        self.maxLengthList = cls.maxLengthList[:]
        self.lgstStrings = cls.lgstStrings
        self.ds = cls.ds

        self.mdl_clgst = cls.mdl_clgst.copy()
        self.mdl_clgst_tp = cls.mdl_clgst_tp.copy()

        self.results = cls.results.copy()
        self.results_logL = cls.results_logL.copy()
