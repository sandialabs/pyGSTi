from ..util import BaseCase

from pygsti.modelpacks.legacy import std1Q_XYI, std2Q_XYICNOT
from pygsti.objects import TrivialGaugeGroup
from pygsti.protocols import gst


class GSTUtilTester(BaseCase):
    def test_gaugeopt_suite_to_dictionary(self):
        model_1Q = std1Q_XYI.target_model()
        model_2Q = std2Q_XYICNOT.target_model()
        model_trivialgg = model_2Q.copy()
        model_trivialgg.default_gauge_group = TrivialGaugeGroup(4)

        d = gst.gaugeopt_suite_to_dictionary("stdgaugeopt", model_1Q, verbosity=1)
        d2 = gst.gaugeopt_suite_to_dictionary(d, model_1Q, verbosity=1)  # with dictionary - basically a pass-through

        d = gst.gaugeopt_suite_to_dictionary(("varySpam", "varySpamWt", "varyValidSpamWt", "toggleValidSpam", "none"),
                                             model_1Q, verbosity=1)
        d = gst.gaugeopt_suite_to_dictionary(
            ("varySpam", "varySpamWt", "varyValidSpamWt", "toggleValidSpam", "stdgaugeopt-unreliable2Q"),
            model_trivialgg, verbosity=1
        )

        d = gst.gaugeopt_suite_to_dictionary(
            ("stdgaugeopt", "stdgaugeopt-unreliable2Q"), model_1Q, verbosity=1)  # non-2Q gates
        d = gst.gaugeopt_suite_to_dictionary(("stdgaugeopt", "stdgaugeopt-unreliable2Q"), model_2Q, verbosity=1)

        unreliable_ops = ('Gx', 'Gcnot')
        d = gst.gaugeopt_suite_to_dictionary(
            ("stdgaugeopt", "stdgaugeopt-unreliable2Q"), model_2Q, unreliable_ops, verbosity=1)
        d = gst.gaugeopt_suite_to_dictionary(("varySpam", "varySpam-unreliable2Q"),
                                             model_2Q, unreliable_ops, verbosity=1)
        # TODO assert correctness

    def test_gaugeopt_suite_to_dictionary_raises_on_bad_suite(self):
        model_1Q = std1Q_XYI.target_model()
        with self.assertRaises(ValueError):
            gst.gaugeopt_suite_to_dictionary("foobar", model_1Q, verbosity=1)

    def test_add_badfit_estimates(self):
        raise NotImplementedError()  # TODO: test add_badfit_estimates

    def test_add_gauge_opt(self):
        raise NotImplementedError()  # TODO: test add_gauge_opt


class HasTargetModelTester(BaseCase):
    """
    Tests for methods in the HasTargetModel class.
    """

    @classmethod
    def setUpClass(cls):
        pass  # TODO

    @classmethod
    def tearDownClass(cls):
        pass  # TODO

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO


class GateSetTomographyDesignTester(BaseCase):
    """
    Tests for methods in the GateSetTomographyDesign class.
    """

    @classmethod
    def setUpClass(cls):
        pass  # TODO

    @classmethod
    def tearDownClass(cls):
        pass  # TODO

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO


class StructuredGSTDesignTester(BaseCase):
    """
    Tests for methods in the StructuredGSTDesign class.
    """

    @classmethod
    def setUpClass(cls):
        pass  # TODO

    @classmethod
    def tearDownClass(cls):
        pass  # TODO

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO


class StandardGSTDesignTester(BaseCase):
    """
    Tests for methods in the StandardGSTDesign class.
    """

    @classmethod
    def setUpClass(cls):
        pass  # TODO

    @classmethod
    def tearDownClass(cls):
        pass  # TODO

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO


class GSTInitialModelTester(BaseCase):
    """
    Tests for methods in the GSTInitialModel class.
    """

    @classmethod
    def setUpClass(cls):
        pass  # TODO

    @classmethod
    def tearDownClass(cls):
        pass  # TODO

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_build(self):
        raise NotImplementedError()  # TODO: test build

    def test_get_model(self):
        raise NotImplementedError()  # TODO: test get_model


class GSTBadFitOptionsTester(BaseCase):
    """
    Tests for methods in the GSTBadFitOptions class.
    """

    @classmethod
    def setUpClass(cls):
        pass  # TODO

    @classmethod
    def tearDownClass(cls):
        pass  # TODO

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_build(self):
        raise NotImplementedError()  # TODO: test build


class GSTObjFnBuildersTester(BaseCase):
    """
    Tests for methods in the GSTObjFnBuilders class.
    """

    @classmethod
    def setUpClass(cls):
        pass  # TODO

    @classmethod
    def tearDownClass(cls):
        pass  # TODO

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_build(self):
        raise NotImplementedError()  # TODO: test build

    def test_init_simple(self):
        raise NotImplementedError()  # TODO: test init_simple


class GateSetTomographyTester(BaseCase):
    """
    Tests for methods in the GateSetTomography class.
    """

    @classmethod
    def setUpClass(cls):
        pass  # TODO

    @classmethod
    def tearDownClass(cls):
        pass  # TODO

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_run(self):
        raise NotImplementedError()  # TODO: test run


class LinearGateSetTomographyTester(BaseCase):
    """
    Tests for methods in the LinearGateSetTomography class.
    """

    @classmethod
    def setUpClass(cls):
        pass  # TODO

    @classmethod
    def tearDownClass(cls):
        pass  # TODO

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_check_if_runnable(self):
        raise NotImplementedError()  # TODO: test check_if_runnable

    def test_run(self):
        raise NotImplementedError()  # TODO: test run


class StandardGSTTester(BaseCase):
    """
    Tests for methods in the StandardGST class.
    """

    @classmethod
    def setUpClass(cls):
        pass  # TODO

    @classmethod
    def tearDownClass(cls):
        pass  # TODO

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_run(self):
        raise NotImplementedError()  # TODO: test run


class ModelEstimateResultsTester(BaseCase):
    """
    Tests for methods in the ModelEstimateResults class.
    """

    @classmethod
    def setUpClass(cls):
        pass  # TODO

    @classmethod
    def tearDownClass(cls):
        pass  # TODO

    def setUp(self):
        pass  # TODO

    def tearDown(self):
        pass  # TODO

    def test_from_dir(self):
        raise NotImplementedError()  # TODO: test from_dir

    def test_dataset(self):
        raise NotImplementedError()  # TODO: test dataset

    def test_as_nameddict(self):
        raise NotImplementedError()  # TODO: test as_nameddict

    def test_add_estimates(self):
        raise NotImplementedError()  # TODO: test add_estimates

    def test_rename_estimate(self):
        raise NotImplementedError()  # TODO: test rename_estimate

    def test_add_estimate(self):
        raise NotImplementedError()  # TODO: test add_estimate

    def test_add_model_test(self):
        raise NotImplementedError()  # TODO: test add_model_test

    def test_view(self):
        raise NotImplementedError()  # TODO: test view

    def test_copy(self):
        raise NotImplementedError()  # TODO: test copy
