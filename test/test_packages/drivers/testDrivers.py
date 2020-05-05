import unittest
import pygsti
from pygsti.modelpacks.legacy import std1Q_XYI as std
from pygsti.modelpacks.legacy import std2Q_XYICNOT as std2Q
from pygsti.objects.mapforwardsim import MapForwardSimulator

from ..testutils import BaseTestCase, compare_files, temp_files, regenerate_references

class DriversTestCase(BaseTestCase):

    def setUp(self):
        super(DriversTestCase, self).setUp()

        self.model = std.target_model()

        self.germs = std.germs
        self.fiducials = std.fiducials
        self.maxLens = [1,2,4]
        self.op_labels = list(self.model.operations.keys())

        self.lsgstStrings = pygsti.construction.make_lsgst_lists(
            self.op_labels, self.fiducials, self.fiducials, self.germs, self.maxLens )

        ## RUN BELOW LINES TO GENERATE SAVED DATASETS
        if regenerate_references():
            datagen_gateset = self.model.depolarize(op_noise=0.05, spam_noise=0.1)
            ds = pygsti.construction.generate_fake_data(
                datagen_gateset, self.lsgstStrings[-1],
                n_samples=1000,sample_error='binomial', seed=100)
            ds.save(compare_files + "/drivers.dataset")

class TestDriversMethods(DriversTestCase):
    def test_longSequenceGST_fiducialPairReduction(self):
        ds = pygsti.objects.DataSet(file_to_load_from=compare_files + "/drivers.dataset")
        maxLens = self.maxLens

        #Make list-of-lists of GST operation sequences
        fullStructs = pygsti.construction.make_lsgst_structs(
            std.target_model(), std.fiducials, std.fiducials, std.germs, maxLens)

        lens = [ len(strct.allstrs) for strct in fullStructs ]
        self.assertEqual(lens, [92,168,450]) # ,817,1201, 1585]


        #Global FPR
        fidPairs = pygsti.alg.find_sufficient_fiducial_pairs(
            std.target_model(), std.fiducials, std.fiducials, std.germs,
            search_mode="random", n_random=100, seed=1234,
            verbosity=1, mem_limit=int(2*(1024)**3), minimum_pairs=2)

        gfprStructs = pygsti.construction.make_lsgst_structs(
            std.target_model(), std.fiducials, std.fiducials, std.germs, maxLens,
            fid_pairs=fidPairs)

        lens = [ len(strct.allstrs) for strct in gfprStructs ]
        #self.assertEqual(lens, [92,100,130]) #,163,196,229]
          #can't test reliably b/c "random" above
          # means different answers on different systems

        gfprExperiments = pygsti.construction.make_lsgst_experiment_list(
            std.target_model(), std.fiducials, std.fiducials, std.germs, maxLens,
            fid_pairs=fidPairs)

        result = pygsti.do_long_sequence_gst_base(ds, std.target_model(), gfprStructs, verbosity=0)
        pygsti.report.create_standard_report(result, temp_files + "/full_report_GFPR",
                                             "GFPR report", verbosity=2)


        #Per-germ FPR
        fidPairsDict = pygsti.alg.find_sufficient_fiducial_pairs_per_germ(
            std.target_model(), std.fiducials, std.fiducials, std.germs,
            search_mode="random", constrain_to_tp=True,
            n_random=100, seed=1234, verbosity=1,
            mem_limit=int(2*(1024)**3))

        pfprStructs = pygsti.construction.make_lsgst_structs(
            std.target_model(), std.fiducials, std.fiducials, std.germs, maxLens,
            fid_pairs=fidPairsDict) #note: fidPairs arg can be a dict too!

        lens = [ len(strct.allstrs) for strct in pfprStructs ]
        #self.assertEqual(lens, [92,99,138]) # ,185,233,281]
          #can't test reliably b/c "random" above
          # means different answers on different systems


        pfprExperiments = pygsti.construction.make_lsgst_experiment_list(
            std.target_model(), std.fiducials, std.fiducials, std.germs, maxLens,
            fid_pairs=fidPairsDict)

        result = pygsti.do_long_sequence_gst_base(ds, std.target_model(), pfprStructs, verbosity=0)
        pygsti.report.create_standard_report(result, temp_files + "/full_report_PFPR",
                                             "PFPR report", verbosity=2)



    def test_longSequenceGST_randomReduction(self):
        ds = pygsti.objects.DataSet(file_to_load_from=compare_files + "/drivers.dataset")
        ts = "whole germ powers"
        maxLens = self.maxLens

        #Without fixed initial fiducial pairs
        fidPairs = None
        reducedLists = pygsti.construction.make_lsgst_structs(
            std.target_model().operations.keys(), std.fiducials, std.fiducials, std.germs,
            maxLens, fidPairs, ts, keep_fraction=0.5, keep_seed=1234)
        result = self.runSilent(pygsti.do_long_sequence_gst_base,
            ds, std.target_model(), reducedLists,
            advanced_options={'truncScheme': ts})

        #create a report...
        pygsti.report.create_standard_report(result, temp_files + "/full_report_RFPR",
                                             "RFPR report", verbosity=2)

        #With fixed initial fiducial pairs
        fidPairs = pygsti.alg.find_sufficient_fiducial_pairs(
            std.target_model(), std.fiducials, std.fiducials, std.germs, verbosity=0)
        reducedLists = pygsti.construction.make_lsgst_structs(
            std.target_model().operations.keys(), std.fiducials, std.fiducials, std.germs,
            maxLens, fidPairs, ts, keep_fraction=0.5, keep_seed=1234)
        result2 = self.runSilent(pygsti.do_long_sequence_gst_base,
                                 ds, std.target_model(), reducedLists,
                                 advanced_options={'truncScheme': ts})

        #create a report...
        pygsti.report.create_standard_report(result2, temp_files + "/full_report_RFPR2.html",
                                             verbosity=2)

    def test_longSequenceGST_CPTP(self):
        ds = pygsti.objects.DataSet(file_to_load_from=compare_files + "/drivers.dataset")

        target_model = std.target_model()
        target_model.set_all_parameterizations("CPTP")

        maxLens = self.maxLens
        result = self.runSilent(pygsti.do_long_sequence_gst,
                                ds, target_model, std.fiducials, std.fiducials,
                                std.germs, maxLens)

        #create a report...
        pygsti.report.create_standard_report(result, temp_files + "/full_report_CPTPGates",
                                             "CPTP Gates report", verbosity=2)


    def test_longSequenceGST_Sonly(self):
        ds = pygsti.objects.DataSet(file_to_load_from=compare_files + "/drivers.dataset")

        target_model = std.target_model()
        target_model.set_all_parameterizations("S")

        maxLens = self.maxLens
        result = self.runSilent(pygsti.do_long_sequence_gst,
                                ds, target_model, std.fiducials, std.fiducials,
                                std.germs, maxLens)

        #create a report...
        pygsti.report.create_standard_report(result, temp_files + "/full_report_SGates.html",
                                             "SGates report", verbosity=2)


    def test_longSequenceGST_GLND(self):
        #General Lindbladian parameterization (allowed to be non-CPTP)
        ds = pygsti.objects.DataSet(file_to_load_from=compare_files + "/drivers.dataset")

        target_model = std.target_model()

        #No set_all_parameterizations option for this one, since it probably isn't so useful
        for lbl,gate in target_model.operations.items():
            target_model.operations[lbl] = pygsti.objects.operation.convert(gate, "GLND", "gm")
        target_model.default_gauge_group = pygsti.objects.UnitaryGaugeGroup(target_model.dim,"gm")
          #Lindblad gates only know how to do unitary transforms currently, even though
          # in the non-cptp case it they should be able to transform generally.

        maxLens = self.maxLens
        result = self.runSilent(pygsti.do_long_sequence_gst,
                                ds, target_model, std.fiducials, std.fiducials,
                                std.germs, maxLens)

        #create a report...
        pygsti.report.create_standard_report(result, temp_files + "/full_report_SGates",
                                             "SGates report", verbosity=2)


    def test_longSequenceGST_HplusS(self):
        ds = pygsti.objects.DataSet(file_to_load_from=compare_files + "/drivers.dataset")

        target_model = std.target_model()
        target_model.set_all_parameterizations("H+S")

        maxLens = self.maxLens
        result = self.runSilent(pygsti.do_long_sequence_gst,
                                ds, target_model, std.fiducials, std.fiducials,
                                std.germs, maxLens)

        #create a report...
        pygsti.report.create_standard_report(result, temp_files + "/full_report_HplusSGates",
                                             "HpS report", verbosity=2)



    def test_longSequenceGST_badfit(self):
        ds = pygsti.objects.DataSet(file_to_load_from=compare_files + "/drivers.dataset")

        #lower bad-fit threshold to zero to trigger bad-fit additional processing
        maxLens = self.maxLens
        result = self.runSilent(pygsti.do_long_sequence_gst,
                                ds, std.target_model(), std.fiducials, std.fiducials,
                                std.germs, maxLens, advanced_options={'bad_fit_threshold': -100})

        pygsti.report.create_standard_report(result, temp_files + "/full_report_badfit",
                                             "badfit report", verbosity=2)

    def test_stdpracticeGST(self):
        ds = pygsti.objects.DataSet(file_to_load_from=compare_files + "/drivers.dataset")
        mdl_guess = std.target_model().depolarize(op_noise=0.01,spam_noise=0.01)

        #lower bad-fit threshold to zero to trigger bad-fit additional processing
        maxLens = self.maxLens
        result = self.runSilent(pygsti.do_stdpractice_gst,
                                ds, std.target_model(), std.fiducials, std.fiducials,
                                std.germs, maxLens, modes="TP,CPTP,Test,Target",
                                models_to_test = {"Test": mdl_guess},
                                comm=None, mem_limit=None, verbosity=5)
        pygsti.report.create_standard_report(result, temp_files + "/full_report_stdpractice",
                                             "Std Practice Test Report", verbosity=2)

    def test_bootstrap(self):
        """Test bootstrap model generation"""
        ds = pygsti.objects.DataSet(file_to_load_from=compare_files + "/drivers.dataset")
        tp_target = std.target_model()
        tp_target.set_all_parameterizations("TP")
        mdl = pygsti.do_lgst(ds, std.fiducials, std.fiducials, target_model=tp_target, svd_truncate_to=4, verbosity=0)

        default_maxLens = [0]+[2**k for k in range(10)]
        circuits = pygsti.construction.make_lsgst_experiment_list(
            self.op_labels, self.fiducials, self.fiducials, self.germs,
            default_maxLens, fid_pairs=None, trunc_scheme="whole germ powers")
        ds_defaultMaxLens = pygsti.construction.generate_fake_data(
            mdl, circuits, n_samples=10000, sample_error='round')

        bootgs_p_defaultMaxLens = \
            pygsti.drivers.make_bootstrap_models(
                2, ds_defaultMaxLens, 'parametric', std.fiducials, std.fiducials,
                std.germs, None, input_model=mdl,
                return_data=False) #test when max_lengths == None


if __name__ == "__main__":
    unittest.main(verbosity=2)
