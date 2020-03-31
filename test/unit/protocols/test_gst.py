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
