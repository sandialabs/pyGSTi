from ..util import BaseCase, with_temp_path
from ..fixture_gen import drivers_gen

from pygsti import io
from pygsti.construction import std1Q_XYI as std
from pygsti.objects import DataSet, operation, UnitaryGaugeGroup, mapforwardsim
from pygsti.drivers import longsequence as ls


# TODO optimize everything
class LongSequenceBase(BaseCase):
    def setUp(self):
        self.maxLens = [1, 2, 4]
        self.lsgstStrings = drivers_gen._lsgstStrings
        self.fiducials = drivers_gen._fiducials
        self.germs = drivers_gen._germs
        self.ds = DataSet(fileToLoadFrom=self.fixture_path('drivers.dataset'))
        self.model = io.load_model(self.fixture_path('drivers.model'))
        self.options = {}

    def test_long_sequence_gst(self):
        result = ls.do_long_sequence_gst(
            self.ds, self.model, self.fiducials, self.fiducials,
            self.germs, self.maxLens, advancedOptions={**self.options})
        # TODO assert correctness


class LongSequenceExtendedBase(LongSequenceBase):
    def test_long_sequence_gst_chi2(self):
        result = ls.do_long_sequence_gst(
            self.ds, self.model, self.fiducials, self.fiducials,
            self.germs, self.maxLens,
            advancedOptions={**self.options, 'objective': "chi2"})
        # TODO assert correctness


class LongSequenceTester(LongSequenceExtendedBase):

    def test_long_sequence_gst_advanced_options(self):
        # TODO what exactly is being tested?
        result = ls.do_long_sequence_gst(
            self.ds, self.model, self.fiducials, None,
            self.germs, self.maxLens,
            advancedOptions={**self.options,
                             'starting point': self.model,
                             'depolarizeStart': 0.05,
                             'cptpPenaltyFactor': 1.0}
        )
        # TODO assert correctness

    def test_long_sequence_gst_raises_on_bad_profile_options(self):
        #check invalid profile options
        with self.assertRaises(ValueError):
            ls.do_long_sequence_gst(
                self.ds, self.model, self.fiducials, self.fiducials,
                self.germs, self.maxLens,
                advancedOptions={'profile': 3}
            )

    def test_long_sequence_gst_raises_on_bad_advanced_options(self):
        with self.assertRaises(ValueError):
            ls.do_long_sequence_gst(
                self.ds, self.model, self.fiducials, None,
                self.germs, self.maxLens,
                advancedOptions={'objective': "FooBar"}
            )  # bad objective
        with self.assertRaises(ValueError):
            ls.do_long_sequence_gst(
                self.ds, self.model, self.fiducials, None,
                self.germs, self.maxLens,
                advancedOptions={'starting point': "FooBar"}
            )  # bad starting point


class WholeGermPowersTester(LongSequenceExtendedBase):
    def setUp(self):
        super(WholeGermPowersTester, self).setUp()
        self.options = {'truncScheme': "whole germ powers"}

    @with_temp_path
    @with_temp_path
    @with_temp_path
    def test_long_sequence_gst_with_file_args(self, ds_path, fiducial_path, germ_path):
        model_path = self.fixture_path('drivers.model')
        io.write_dataset(ds_path, self.ds, self.lsgstStrings[-1])
        io.write_circuit_list(fiducial_path, self.fiducials)
        io.write_circuit_list(germ_path, self.germs)

        result = ls.do_long_sequence_gst(
            ds_path, model_path, fiducial_path, fiducial_path, germ_path, self.maxLens,
            advancedOptions={**self.options,
                             'randomizeStart': 1e-6,
                             'profile': 2,
                             'verbosity': 10,
                             'memoryLimitInBytes': 2 * 1000**3}
        )
        # TODO assert correctness


class TruncatedGermPowersTester(LongSequenceExtendedBase):
    def setUp(self):
        super(TruncatedGermPowersTester, self).setUp()
        self.ds = DataSet(fileToLoadFrom=self.fixture_path('drivers_tgp.dataset'))
        self.options = {'truncScheme': "truncated germ powers"}


class LengthAsExponentTester(LongSequenceExtendedBase):
    def setUp(self):
        super(LengthAsExponentTester, self).setUp()
        self.ds = DataSet(fileToLoadFrom=self.fixture_path('drivers_lae.dataset'))
        self.options = {'truncScheme': "length as exponent"}


class CPTPGatesTester(LongSequenceBase):
    # TODO optimize!!
    def setUp(self):
        super(CPTPGatesTester, self).setUp()
        self.model.set_all_parameterizations("CPTP")


class SGatesTester(LongSequenceBase):
    def setUp(self):
        super(SGatesTester, self).setUp()
        self.model.set_all_parameterizations("S")


class HPlusSGatesTester(LongSequenceBase):
    # TODO optimize!!!!
    def setUp(self):
        super(HPlusSGatesTester, self).setUp()
        self.model.set_all_parameterizations("H+S")


class GLNDModelTester(LongSequenceBase):
    def setUp(self):
        super(GLNDModelTester, self).setUp()
        for lbl, gate in self.model.operations.items():
            self.model.operations[lbl] = operation.convert(gate, "GLND", "gm")
        self.model.default_gauge_group = UnitaryGaugeGroup(self.model.dim, "gm")


class MapCalcTester(LongSequenceBase):
    def setUp(self):
        super(MapCalcTester, self).setUp()
        self.model._calcClass = mapforwardsim.MapForwardSimulator
        self.options = {'truncScheme': "whole germ powers"}


class BadFitTester(LongSequenceExtendedBase):
    def setUp(self):
        super(BadFitTester, self).setUp()
        self.options = {
            'truncScheme': "whole germ powers",
            'badFitThreshold': -100
        }


class RobustDataScalingTester(LongSequenceBase):
    def setUp(self):
        super(RobustDataScalingTester, self).setUp()
        self.ds = DataSet(fileToLoadFrom=self.fixture_path('drivers2.dataset'))
        self.options = {
            'badFitThreshold': -100,
            'onBadFit': ["do nothing", "robust", "Robust", "robust+", "Robust+"]
        }

    def test_long_sequence_gst_raises_on_bad_badfit_options(self):
        with self.assertRaises(ValueError):
            ls.do_long_sequence_gst(
                self.ds, self.model, self.fiducials, self.fiducials,
                self.germs, self.maxLens,
                advancedOptions={'badFitThreshold': -100,
                                 'onBadFit': ["foobar"]}
            )
