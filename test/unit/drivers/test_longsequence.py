from ..util import BaseCase, with_temp_path
from . import fixtures as pkg

from io import BytesIO
from pygsti import io
import pygsti.construction as pc
from pygsti.modelpacks.legacy import std1Q_XYI as std
from pygsti.modelpacks.legacy import std2Q_XYICNOT as std2Q
from pygsti.objects import DataSet, operation, UnitaryGaugeGroup, TrivialGaugeGroup, mapforwardsim
from pygsti.drivers import longsequence as ls


# TODO optimize everything
class LongSequenceBase(BaseCase):
    @classmethod
    def setUpClass(cls):
        cls.model = pkg.model
        cls.maxLens = pkg.maxLengthList
        cls.opLabels = pkg.opLabels
        cls.fiducials = pkg.fiducials
        cls.germs = pkg.germs
        cls.lsgstStrings = pkg.lsgstStrings
        cls.ds = pkg.dataset

    def setUp(self):
        self.model = self.model.copy()
        self.ds = self.ds.copy()


class ModelTestTester(LongSequenceBase):
    def setUp(self):
        super(ModelTestTester, self).setUp()
        self.mdl_guess = self.model.depolarize(op_noise=0.01, spam_noise=0.01)

    def test_model_test(self):
        result = ls.run_model_test(
            self.mdl_guess, self.ds, self.model, self.fiducials,
            self.fiducials, self.germs, self.maxLens
        )
        # TODO assert correctness

    def test_model_test_advanced_options(self):
        result = ls.run_model_test(
            self.mdl_guess, self.ds, self.model, self.fiducials,
            self.fiducials, self.germs, self.maxLens,
            advanced_options=dict(objective='chi2', profile=2)
        )
        # TODO assert correctness

    def test_model_test_pickle_output(self):
        with BytesIO() as pickle_stream:
            result = ls.run_model_test(
                self.mdl_guess, self.ds, self.model, self.fiducials,
                self.fiducials, self.germs, self.maxLens, output_pkl=pickle_stream
            )
            self.assertTrue(len(pickle_stream.getvalue()) > 0)
            # TODO assert correctness

    def test_model_test_raises_on_bad_options(self):
        with self.assertRaises(ValueError):
            ls.run_model_test(
                self.mdl_guess, self.ds, self.model, self.fiducials,
                self.fiducials, self.germs, self.maxLens,
                advanced_options=dict(objective='foobar')
            )
        with self.assertRaises(ValueError):
            ls.run_model_test(
                self.mdl_guess, self.ds, self.model, self.fiducials,
                self.fiducials, self.germs, self.maxLens,
                advanced_options=dict(profile='foobar')
            )


class StdPracticeGSTTester(LongSequenceBase):
    def setUp(self):
        super(StdPracticeGSTTester, self).setUp()
        self.mdl_guess = self.model.depolarize(op_noise=0.01, spam_noise=0.01)

    def test_stdpractice_gst_TP(self):
        result = ls.run_stdpractice_gst(
            self.ds, self.model, self.fiducials, self.fiducials,
            self.germs, self.maxLens, modes="TP",
            models_to_test={"Test": self.mdl_guess}, comm=None,
            mem_limit=None, verbosity=5
        )
        # TODO assert correctness

    def test_stdpractice_gst_CPTP(self):
        result = ls.run_stdpractice_gst(
            self.ds, self.model, self.fiducials, self.fiducials,
            self.germs, self.maxLens, modes="CPTP",
            models_to_test={"Test": self.mdl_guess}, comm=None,
            mem_limit=None, verbosity=5
        )
        # TODO assert correctness

    def test_stdpractice_gst_Test(self):
        result = ls.run_stdpractice_gst(
            self.ds, self.model, self.fiducials, self.fiducials,
            self.germs, self.maxLens, modes="Test",
            models_to_test={"Test": self.mdl_guess}, comm=None,
            mem_limit=None, verbosity=5
        )
        # TODO assert correctness

    def test_stdpractice_gst_Target(self):
        result = ls.run_stdpractice_gst(
            self.ds, self.model, self.fiducials, self.fiducials,
            self.germs, self.maxLens, modes="Target",
            models_to_test={"Test": self.mdl_guess}, comm=None,
            mem_limit=None, verbosity=5
        )
        # TODO assert correctness

    @with_temp_path
    @with_temp_path
    @with_temp_path
    @with_temp_path
    def test_stdpractice_gst_file_args(self, ds_path, model_path, fiducial_path, germ_path):
        io.write_dataset(ds_path, self.ds, self.lsgstStrings[-1])
        io.write_model(self.model, model_path)
        io.write_circuit_list(fiducial_path, self.fiducials)
        io.write_circuit_list(germ_path, self.germs)

        result = ls.run_stdpractice_gst(
            ds_path, model_path, fiducial_path, fiducial_path, germ_path,
            self.maxLens, modes="TP", comm=None, mem_limit=None, verbosity=5
        )
        # TODO assert correctness

    def test_stdpractice_gst_gaugeOptTarget(self):
        myGaugeOptSuiteDict = {
            'MyGaugeOpt': {
                'item_weights': {'gates': 1, 'spam': 0.0001}
            }
        }
        result = ls.run_stdpractice_gst(
            self.ds, self.model, self.fiducials, self.fiducials,
            self.germs, self.maxLens, modes="TP", comm=None,
            mem_limit=None, verbosity=5, gauge_opt_target=self.mdl_guess,
            gauge_opt_suite=myGaugeOptSuiteDict
        )
        # TODO assert correctness

    def test_stdpractice_gst_gaugeOptTarget_warns_on_target_override(self):
        myGaugeOptSuiteDict = {
            'MyGaugeOpt': {
                'item_weights': {'gates': 1, 'spam': 0.0001},
                'target_model': self.model  # to test overriding internal target model (prints a warning)
            }
        }
        with self.assertWarns(Warning):
            result = ls.run_stdpractice_gst(
                self.ds, self.model, self.fiducials, self.fiducials,
                self.germs, self.maxLens, modes="TP", comm=None,
                mem_limit=None, verbosity=5, gauge_opt_target=self.mdl_guess,
                gauge_opt_suite=myGaugeOptSuiteDict
            )
            # TODO assert correctness

    def test_stdpractice_gst_advanced_options(self):
        result = ls.run_stdpractice_gst(
            self.ds, self.model, self.fiducials, self.fiducials,
            self.germs, self.maxLens, modes="TP", comm=None,
            mem_limit=None, verbosity=5,
            advanced_options={'all': {
                'objective': 'chi2',
                'bad_fit_threshold': -100,  # so we create a robust estimate and convey guage opt to it.
                'on_bad_fit': ["robust"]
            }}
        )
        # TODO assert correctness

    def test_stdpractice_gst_pickle_output(self):
        with BytesIO() as pickle_stream:
            result = ls.run_stdpractice_gst(
                self.ds, self.model, self.fiducials, self.fiducials,
                self.germs, self.maxLens, modes="Target", output_pkl=pickle_stream
            )
            self.assertTrue(len(pickle_stream.getvalue()) > 0)
            # TODO assert correctness

    def test_stdpractice_gst_raises_on_bad_mode(self):
        with self.assertRaises(ValueError):
            result = ls.run_stdpractice_gst(
                self.ds, self.model, self.fiducials, self.fiducials,
                self.germs, self.maxLens, modes="Foobar"
            )


class LongSequenceGSTBase(LongSequenceBase):
    def setUp(self):
        super(LongSequenceGSTBase, self).setUp()
        self.options = {}

    def test_long_sequence_gst(self):
        result = ls.run_long_sequence_gst(
            self.ds, self.model, self.fiducials, self.fiducials,
            self.germs, self.maxLens, advanced_options=self.options)
        # TODO assert correctness


class LongSequenceGSTWithChi2(LongSequenceGSTBase):
    def test_long_sequence_gst_chi2(self):
        self.options.update(
            objective='chi2'
        )
        result = ls.run_long_sequence_gst(
            self.ds, self.model, self.fiducials, self.fiducials,
            self.germs, self.maxLens,
            advanced_options=self.options)
        # TODO assert correctness


class LongSequenceGSTTester(LongSequenceGSTWithChi2):
    def test_long_sequence_gst_advanced_options(self):
        # TODO what exactly is being tested?
        self.options.update({
            #'starting point': self.model,  #this is deprecated now - need to use protocol objects
            'depolarize_start': 0.05,
            'cptp_penalty_factor': 1.0
        })
        result = ls.run_long_sequence_gst(
            self.ds, self.model, self.fiducials, None,
            self.germs, self.maxLens,
            advanced_options=self.options
        )
        # TODO assert correctness

    def test_long_sequence_gst_raises_on_bad_profile_options(self):
        #check invalid profile options
        with self.assertRaises(ValueError):
            ls.run_long_sequence_gst(
                self.ds, self.model, self.fiducials, self.fiducials,
                self.germs, self.maxLens,
                advanced_options={'profile': 3}
            )

    def test_long_sequence_gst_raises_on_bad_advanced_options(self):
        with self.assertRaises(ValueError):
            ls.run_long_sequence_gst(
                self.ds, self.model, self.fiducials, None,
                self.germs, self.maxLens,
                advanced_options={'objective': "FooBar"}
            )  # bad objective
        with self.assertRaises(ValueError):
            ls.run_long_sequence_gst(
                self.ds, self.model, self.fiducials, None,
                self.germs, self.maxLens,
                advanced_options={'starting_point': "FooBar"}
            )  # bad starting point


class WholeGermPowersTester(LongSequenceGSTWithChi2):
    def setUp(self):
        super(WholeGermPowersTester, self).setUp()
        self.options = {} # 'truncScheme': "whole germ powers"}
        # Trunce scheme has been removed as an option - we only ever use whole germ powers now

    @with_temp_path
    @with_temp_path
    @with_temp_path
    @with_temp_path
    def test_long_sequence_gst_with_file_args(self, ds_path, model_path, fiducial_path, germ_path):
        io.write_dataset(ds_path, self.ds, self.lsgstStrings[-1])
        io.write_model(self.model, model_path)
        io.write_circuit_list(fiducial_path, self.fiducials)
        io.write_circuit_list(germ_path, self.germs)

        self.options.update(
            randomize_start=1e-6,
            profile=2,
        )
        result = ls.run_long_sequence_gst(
            ds_path, model_path, fiducial_path, fiducial_path, germ_path, self.maxLens,
            advanced_options=self.options, verbosity=10
        )
        # TODO assert correctness


class CPTPGatesTester(LongSequenceGSTBase):
    # TODO optimize!!
    def setUp(self):
        super(CPTPGatesTester, self).setUp()
        self.model.set_all_parameterizations("CPTP")


class SGatesTester(LongSequenceGSTBase):
    def setUp(self):
        super(SGatesTester, self).setUp()
        self.model.set_all_parameterizations("S")


class HPlusSGatesTester(LongSequenceGSTBase):
    # TODO optimize!!!!
    def setUp(self):
        super(HPlusSGatesTester, self).setUp()
        self.model.set_all_parameterizations("H+S")


class GLNDModelTester(LongSequenceGSTBase):
    def setUp(self):
        super(GLNDModelTester, self).setUp()
        for lbl, gate in self.model.operations.items():
            self.model.operations[lbl] = operation.convert(gate, "GLND", "gm")
        self.model.default_gauge_group = UnitaryGaugeGroup(self.model.dim, "gm")


class MapCalcTester(LongSequenceGSTBase):
    def setUp(self):
        super(MapCalcTester, self).setUp()
        self.model._calcClass = mapforwardsim.MapForwardSimulator
        self.options = {}


class BadFitTester(LongSequenceGSTWithChi2):
    def setUp(self):
        super(BadFitTester, self).setUp()
        self.options = {
            'bad_fit_threshold': -100
        }


class RobustDataScalingTester(LongSequenceGSTBase):
    @classmethod
    def setUpClass(cls):
        super(RobustDataScalingTester, cls).setUpClass()
        datagen_gateset = cls.model.depolarize(op_noise=0.1, spam_noise=0.03).rotate((0.05, 0.13, 0.02))
        ds2 = pc.simulate_data(
            datagen_gateset, cls.lsgstStrings[-1], num_samples=1000, sample_error='binomial', seed=100
        ).copy_nonstatic()
        ds2.add_counts_from_dataset(cls.ds)
        ds2.done_adding_data()
        cls.ds = ds2

    def setUp(self):
        super(RobustDataScalingTester, self).setUp()
        self.options = {
            'bad_fit_threshold': -100,
            'on_bad_fit': ["do nothing", "robust", "Robust", "robust+", "Robust+"]
        }

    def test_long_sequence_gst_raises_on_bad_badfit_options(self):
        with self.assertRaises(ValueError):
            ls.run_long_sequence_gst(
                self.ds, self.model, self.fiducials, self.fiducials,
                self.germs, self.maxLens,
                advanced_options={'bad_fit_threshold': -100,
                                 'on_bad_fit': ["foobar"]}
            )
