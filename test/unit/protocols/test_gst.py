from pygsti.data import simulate_data
from pygsti.forwardsims.mapforwardsim import MapForwardSimulator
from pygsti.modelpacks import smq1Q_XYI
from pygsti.modelpacks.legacy import std1Q_XYI, std2Q_XYICNOT
from pygsti.objectivefns.objectivefns import PoissonPicDeltaLogLFunction
from pygsti.models.gaugegroup import TrivialGaugeGroup
from pygsti.objectivefns import FreqWeightedChi2Function
from pygsti.optimize.simplerlm import SimplerLMOptimizer
from pygsti.protocols import gst
from pygsti.protocols.estimate import Estimate
from pygsti.protocols.protocol import ProtocolData, Protocol
from pygsti.protocols.gst import GSTGaugeOptSuite
from pygsti.tools import two_delta_logl
from ..util import BaseCase
import pytest


class GSTUtilTester(BaseCase):

    @classmethod
    def setUpClass(cls):

        #Construct a results object
        gst_design = smq1Q_XYI.create_gst_experiment_design(max_max_length=4)
        mdl_target = smq1Q_XYI.target_model()
        mdl_datagen = mdl_target.depolarize(op_noise=0.05, spam_noise=0.025)

        ds = simulate_data(mdl_datagen, gst_design.all_circuits_needing_data, 1000, seed=2020)
        data = ProtocolData(gst_design, ds)
        cls.results = gst.ModelEstimateResults(data, Protocol("test-protocol"))
        cls.results.add_estimate(
            Estimate.create_gst_estimate(
                cls.results, mdl_target, mdl_target,
                [mdl_datagen] * len(gst_design.circuit_lists), parameters={'objective': 'logl'}),
            estimate_key="test-estimate"
        )
        cls.target_model = mdl_target

    def test_gaugeopt_suite_to_dictionary(self):
        model_1Q = std1Q_XYI.target_model()
        model_2Q = std2Q_XYICNOT.target_model()
        model_trivialgg = model_2Q.copy()
        model_trivialgg.default_gauge_group = TrivialGaugeGroup(4)

        d = GSTGaugeOptSuite("stdgaugeopt").to_dictionary(model_1Q, verbosity=1)
        d2 = GSTGaugeOptSuite(gaugeopt_argument_dicts=d).to_dictionary(model_1Q, verbosity=1)  # with dictionary - basically a pass-through

        d = GSTGaugeOptSuite(("varySpam", "varySpamWt", "varyValidSpamWt", "toggleValidSpam", "none")).to_dictionary(
            model_1Q, verbosity=1)
        d = GSTGaugeOptSuite(("varySpam", "varySpamWt", "varyValidSpamWt", "toggleValidSpam", "stdgaugeopt-unreliable2Q")).to_dictionary(
            model_trivialgg, verbosity=1)

        d = GSTGaugeOptSuite(("stdgaugeopt", "stdgaugeopt-unreliable2Q")).to_dictionary(model_1Q, verbosity=1)  # non-2Q gates
        d = GSTGaugeOptSuite(("stdgaugeopt", "stdgaugeopt-unreliable2Q")).to_dictionary(model_2Q, verbosity=1)

        unreliable_ops = ('Gx', 'Gcnot')
        d = GSTGaugeOptSuite(("stdgaugeopt", "stdgaugeopt-unreliable2Q")).to_dictionary(model_2Q, unreliable_ops, verbosity=1)
        d = GSTGaugeOptSuite(("varySpam", "varySpam-unreliable2Q")).to_dictionary(model_2Q, unreliable_ops, verbosity=1)
        # TODO assert correctness

    def test_gaugeopt_suite_raises_on_bad_suite(self):
        model_1Q = std1Q_XYI.target_model()
        with self.assertRaises(ValueError):
            GSTGaugeOptSuite("foobar").to_dictionary(model_1Q, verbosity=1)

    def test_add_badfit_estimates(self):
        builder = PoissonPicDeltaLogLFunction.builder()
        opt = SimplerLMOptimizer()
        badfit_opts = gst.GSTBadFitOptions(threshold=-10, actions=("robust", "Robust", "robust+", "Robust+",
                                                                   "wildcard", "do nothing"))
        res = self.results.copy()
        res.estimates['test-estimate'].parameters['final_objfn_builder'] = builder
        gst._add_badfit_estimates(res, 'test-estimate', badfit_opts, opt)
        estimate_names = set(res.estimates.keys())
        self.assertEqual(estimate_names, set(['test-estimate',
                                              'test-estimate.robust', 'test-estimate.Robust',
                                              'test-estimate.robust+', 'test-estimate.Robust+']))
        self.assertTrue('unmodeled_error' in res.estimates['test-estimate'].parameters)  # wildcard budget

    def test_add_gauge_opt(self):
        res = self.results.copy()
        unreliable = ()
        gaugeopt_suite = GSTGaugeOptSuite('stdgaugeopt', gaugeopt_target=self.target_model)
        gst._add_gauge_opt(res, 'test-estimate', gaugeopt_suite, self.target_model, unreliable)
        self.assertTrue('stdgaugeopt' in res.estimates['test-estimate'].models)
        self.assertTrue('stdgaugeopt' in res.estimates['test-estimate'].goparameters)


class StandardGSTDesignTester(BaseCase):
    """
    Tests for methods in the StandardGSTDesign class.
    """

    def test_creation(self):
        gst.GateSetTomographyDesign(smq1Q_XYI.processor_spec(),
                                    smq1Q_XYI.prep_fiducials(),
                                    smq1Q_XYI.meas_fiducials(),
                                    smq1Q_XYI.germs(),
                                    [1, 2])


class GSTInitialModelTester(BaseCase):
    """
    Tests for methods in the GSTInitialModel class.
    """

    def setUp(self):
        self.edesign = smq1Q_XYI.create_gst_experiment_design(max_max_length=2)
        self.target_model = smq1Q_XYI.target_model()

    def tearDown(self):
        pass  # TODO

    def test_create_from(self):
        im = gst.GSTInitialModel.cast(self.target_model)
        im2 = gst.GSTInitialModel.cast(im)
        self.assertTrue(im2 is im)

        im3 = gst.GSTInitialModel.cast(self.target_model)
        self.assertEqual(im3.starting_point, "User-supplied-Model")

    def test_get_model_target(self):
        #Default
        im = gst.GSTInitialModel(target_model=self.target_model)
        mdl = im.retrieve_model(self.edesign, None, None, None)
        self.assertEqual(im.starting_point, 'target')
        self.assertTrue(self.target_model.frobeniusdist(mdl) < 1e-6)

    def test_get_model_custom(self):
        #Custom model
        custom_model = self.target_model.rotate(max_rotate=0.05, seed=1234)
        im = gst.GSTInitialModel(custom_model)
        mdl = im.retrieve_model(self.edesign, None, None, None)
        self.assertEqual(im.starting_point, "User-supplied-Model")
        self.assertArraysAlmostEqual(mdl.to_vector(), custom_model.to_vector())
        #self.assertTrue(mdl is custom_model)  # No longer the case
        # - see commit 72856fb23d5711e4b0a8e2373e02e4dd08e2ae46 -- now we copy the custom model

    def test_get_model_depolarized(self):
        #Depolarized start
        depol_model = self.target_model.depolarize(op_noise=0.1)
        im = gst.GSTInitialModel(target_model=self.target_model, depolarize_start=0.1)
        mdl = im.retrieve_model(self.edesign, None, None, None)
        self.assertEqual(im.starting_point, 'target')
        self.assertTrue(depol_model.frobeniusdist(mdl) < 1e-6)

    def test_get_model_lgst(self):
        #LGST
        datagen_model = self.target_model.depolarize(op_noise=0.1)
        ds = simulate_data(datagen_model, self.edesign.all_circuits_needing_data, 1000, sample_error='none')  # no error for reproducibility

        im1 = gst.GSTInitialModel(self.target_model, "LGST")
        mdl1 = im1.retrieve_model(self.edesign, None, ds, None)

        im2 = gst.GSTInitialModel(self.target_model, "LGST-if-possible")
        mdl2 = im2.retrieve_model(self.edesign, None, ds, None)

        self.assertTrue(mdl1.frobeniusdist(mdl2) < 1e-6)
        #TODO: would like some gauge-inv metric between mdl? and datagen_model to be ~0 (FUTURE)


class GSTBadFitOptionsTester(BaseCase):
    """
    Tests for methods in the GSTBadFitOptions class.
    """

    def test_create_from(self):
        bfo = gst.GSTBadFitOptions.cast(None)
        bfo2 = gst.GSTBadFitOptions.cast(bfo)
        self.assertTrue(bfo2 is bfo)

        bfo3 = gst.GSTBadFitOptions.cast({'threshold': 3.0, 'actions': ('wildcard',)})
        self.assertEqual(bfo3.threshold, 3.0)
        self.assertEqual(bfo3.actions, ('wildcard',))


class GSTObjFnBuildersTester(BaseCase):
    """
    Tests for methods in the GSTObjFnBuilders class.
    """

    def test_create_from(self):
        builders0 = gst.GSTObjFnBuilders.cast(None)
        builders = gst.GSTObjFnBuilders.cast(builders0)
        self.assertTrue(builders is builders0)

        builders = gst.GSTObjFnBuilders.cast([('A', 'B'), ('C', 'D')])  # pass args as tuple
        self.assertEqual(builders.iteration_builders, ('A', 'B'))
        self.assertEqual(builders.final_builders, ('C', 'D'))

    def test_init_simple(self):
        builders = gst.GSTObjFnBuilders.create_from()
        self.assertEqual(len(builders.iteration_builders), 1)
        self.assertEqual(len(builders.final_builders), 1)

        builders = gst.GSTObjFnBuilders.create_from('logl', always_perform_mle=True)
        self.assertEqual(len(builders.iteration_builders), 2)
        self.assertEqual(len(builders.final_builders), 0)

        builders = gst.GSTObjFnBuilders.create_from('logl', always_perform_mle=True, only_perform_mle=True)
        self.assertEqual(len(builders.iteration_builders), 1)
        self.assertEqual(len(builders.final_builders), 0)

        builders = gst.GSTObjFnBuilders.create_from('logl', freq_weighted_chi2=True)
        self.assertEqual(builders.iteration_builders[0].cls_to_build, FreqWeightedChi2Function)


class BaseProtocolData(object):

    @classmethod
    def setUpClass(cls):
        cls.gst_design = smq1Q_XYI.create_gst_experiment_design(max_max_length=4)
        cls.mdl_target = smq1Q_XYI.target_model()
        cls.mdl_datagen = cls.mdl_target.depolarize(op_noise=0.05, spam_noise=0.025)

        ds = simulate_data(cls.mdl_datagen, cls.gst_design.all_circuits_needing_data, 1000, sample_error='none')
        cls.gst_data = ProtocolData(cls.gst_design, ds)


class MapForwardSimulatorWrapper(MapForwardSimulator):

    Message = """
        Hit the forward simulator wrapper!
    """

    def _bulk_fill_probs(self, array_to_fill, layout):
        print(self.Message)
        super(MapForwardSimulatorWrapper, self)._bulk_fill_probs(array_to_fill, layout)

    def _bulk_fill_probs_atom(self, array_to_fill, layout_atom, resource_alloc):
        print(self.Message)
        super(MapForwardSimulatorWrapper, self)._bulk_fill_probs_atom(array_to_fill, layout_atom, resource_alloc)


class TestGateSetTomography(BaseProtocolData):
    """
    Tests for methods in the GateSetTomography class.

    We can't subclass BaseCase since we use some advanced PyTest features.
    """

    def test_run(self):
        self.setUpClass()
        proto = gst.GateSetTomography(smq1Q_XYI.target_model("CPTPLND"), 'stdgaugeopt', name="testGST")
        results = proto.run(self.gst_data)

        mdl_result = results.estimates["testGST"].models['stdgaugeopt']
        twoDLogL = two_delta_logl(mdl_result, self.gst_data.dataset)
        assert twoDLogL <= 1.0  # should be near 0 for perfect data

    def test_run_custom_sim(self, capfd: pytest.LogCaptureFixture):
        self.setUpClass()
        proto = gst.GateSetTomography(smq1Q_XYI.target_model("CPTPLND"), 'stdgaugeopt', name="testGST")
        results = proto.run(self.gst_data, simulator=MapForwardSimulatorWrapper())
        stdout, _ = capfd.readouterr()
        assert MapForwardSimulatorWrapper.Message in stdout

        mdl_result = results.estimates["testGST"].models['stdgaugeopt']
        twoDLogL = two_delta_logl(mdl_result, self.gst_data.dataset)
        assert twoDLogL <= 1.0  # should be near 0 for perfect data

        for estimate in results.estimates.values():
            for model in estimate.models.values():
                assert isinstance(model, MapForwardSimulatorWrapper)
        pass

    
    def test_write_and_read_to_dir(self):
        #integration test to at least confirm we are writing and reading
        #to and from the directory serializations.
        proto = gst.GateSetTomography(smq1Q_XYI.target_model("CPTPLND"), 'stdgaugeopt', name="testGST")
        proto.write('../../test_packages/temp_test_files/test_GateSetTomography_serialization')
        #then read this back in
        proto_read = gst.GateSetTomography.from_dir('../../test_packages/temp_test_files/test_GateSetTomography_serialization')

        #spot check some of the values of the protocol objects
        assert all([elem1==elem2 for elem1, elem2 in 
                    zip(proto_read.initial_model.model.to_vector(), 
                        proto.initial_model.model.to_vector())])
        assert proto_read.gaugeopt_suite.gaugeopt_suite_names == proto.gaugeopt_suite.gaugeopt_suite_names
        assert proto_read.name == proto.name
        assert proto_read.badfit_options.actions == proto.badfit_options.actions

class LinearGateSetTomographyTester(BaseProtocolData, BaseCase):
    """
    Tests for methods in the LinearGateSetTomography class.
    """

    def test_check_if_runnable(self):
        proto = gst.LinearGateSetTomography(self.mdl_target.copy(), 'stdgaugeopt', name="testGST")
        lgst_data = ProtocolData(self.gst_data.edesign.copy_with_maxlengths([1]), self.gst_data.dataset)
        proto.check_if_runnable(lgst_data)  # throws an error if there's a problem

    def test_run(self):
        proto = gst.LinearGateSetTomography(self.mdl_target.copy(), 'stdgaugeopt', name="testLGST")

        lgst_data = ProtocolData(self.gst_data.edesign.copy_with_maxlengths([1]), self.gst_data.dataset)
        results = proto.run(lgst_data)

        mdl_result = results.estimates["testLGST"].models['stdgaugeopt']
        twoDLogL = two_delta_logl(mdl_result, self.gst_data.dataset, self.gst_design.circuit_lists[0])
        self.assertLessEqual(twoDLogL, 1.0)  # should be near 0 for perfect data

    def test_write_and_read_to_dir(self):
        #integration test to at least confirm we are writing and reading
        #to and from the directory serializations.
        proto = gst.LinearGateSetTomography(self.mdl_target.copy(), 'stdgaugeopt', name="testGST")
        proto.write('../../test_packages/temp_test_files/test_LinearGateSetTomography_serialization')
        #then read this back in
        proto_read = gst.LinearGateSetTomography.from_dir('../../test_packages/temp_test_files/test_LinearGateSetTomography_serialization')

        #spot check some of the values of the protocol objects
        assert all([elem1==elem2 for elem1, elem2 in 
                    zip(proto_read.target_model.to_vector(), 
                        proto.target_model.to_vector())])
        assert proto_read.gaugeopt_suite.gaugeopt_suite_names == proto.gaugeopt_suite.gaugeopt_suite_names
        assert proto_read.name == proto.name
        assert proto_read.badfit_options.actions == proto.badfit_options.actions

class TestStandardGST(BaseProtocolData):
    """
    Tests for methods in the StandardGST class.

    We can't subclass BaseCase since we use some advanced PyTest features.
    """

    def test_run(self):
        self.setUpClass()
        proto = gst.StandardGST(modes=["full TP","CPTPLND","Target"])
        results = proto.run(self.gst_data)

        mdl_result = results.estimates["full TP"].models['stdgaugeopt']
        twoDLogL = two_delta_logl(mdl_result, self.gst_data.dataset)
        assert twoDLogL <= 1.0  # should be near 0 for perfect data

        mdl_result = results.estimates["CPTPLND"].models['stdgaugeopt']
        twoDLogL = two_delta_logl(mdl_result, self.gst_data.dataset)
        assert twoDLogL <= 1.0  # should be near 0 for perfect data

    def test_run_custom_sim(self, capfd: pytest.LogCaptureFixture):
        self.setUpClass()
        # We have to test GST modes separately, since we aren't sure how many times
        # the forward simulator's methods will be called.
        self._test_run_custom_sim('full TP', capfd, True)
        self._test_run_custom_sim('Target', capfd, False)

    def _test_run_custom_sim(self, mode, parent_capfd, check_output):
        proto = gst.StandardGST(modes=[mode])
        results = proto.run(self.gst_data, simulator=MapForwardSimulatorWrapper)
        stdout, _ = parent_capfd.readouterr()
        assert MapForwardSimulatorWrapper.Message in stdout
        if check_output:
            mdl_result = results.estimates[mode].models['stdgaugeopt']
            twoDLogL = two_delta_logl(mdl_result, self.gst_data.dataset)
            assert twoDLogL <= 1.0  # should be near 0 for perfect data
        for estimate in results.estimates.values():
            for model in estimate.models.values():
                assert isinstance(model, MapForwardSimulatorWrapper)
        pass

    def test_write_and_read_to_dir(self):
        #integration test to at least confirm we are writing and reading
        #to and from the directory serializations.
        proto = gst.StandardGST(modes=["full TP","CPTPLND","Target"])
        proto.write('../../test_packages/temp_test_files/test_StandardGateSetTomography_serialization')
        #then read this back in
        proto_read = gst.StandardGST.from_dir('../../test_packages/temp_test_files/test_StandardGateSetTomography_serialization')

        #spot check some of the values of the protocol objects
        assert proto_read.gaugeopt_suite.gaugeopt_suite_names == proto.gaugeopt_suite.gaugeopt_suite_names
        assert proto_read.name == proto.name
        assert proto_read.modes == proto.modes
        assert proto_read.badfit_options.actions == proto.badfit_options.actions
    


#Unit tests are currently performed in objects/test_results.py - TODO: move these tests here
# or move ModelEstimateResults class (?) and update/add tests
#class ModelEstimateResultsTester(BaseCase):
#    """
#    Tests for methods in the ModelEstimateResults class.
#    """
#
#    @classmethod
#    def setUpClass(cls):
#        pass  # TODO
#
#    @classmethod
#    def tearDownClass(cls):
#        pass  # TODO
#
#    def setUp(self):
#        pass  # TODO
#
#    def tearDown(self):
#        pass  # TODO
#
#    def test_from_dir(self):
#        raise NotImplementedError()  # TODO: test from_dir
#
#    def test_dataset(self):
#        raise NotImplementedError()  # TODO: test dataset
#
#    def test_as_nameddict(self):
#        raise NotImplementedError()  # TODO: test to_nameddict
#
#    def test_add_estimates(self):
#        raise NotImplementedError()  # TODO: test add_estimates
#
#    def test_rename_estimate(self):
#        raise NotImplementedError()  # TODO: test rename_estimate
#
#    def test_add_estimate(self):
#        raise NotImplementedError()  # TODO: test add_estimate
#
#    def test_add_model_test(self):
#        raise NotImplementedError()  # TODO: test add_model_test
#
#    def test_view(self):
#        raise NotImplementedError()  # TODO: test view
#
#    def test_copy(self):
#        raise NotImplementedError()  # TODO: test copy
