import unittest
import pygsti
from pygsti.construction import std1Q_XYI as std

import numpy as np
import sys

class DriversTestCase(unittest.TestCase):

    def setUp(self):
        #Set GateSet objects to "strict" mode for testing
        pygsti.objects.GateSet._strict = True

        self.gateset = std.gs_target

        self.germs = std.germs
        self.fiducials = std.fiducials
        self.maxLens = [0,1,2,4]
        self.gateLabels = self.gateset.gates.keys()
        
        self.elgstStrings = pygsti.construction.make_elgst_lists(
            self.gateLabels, self.germs, self.maxLens )
        
        self.lsgstStrings = pygsti.construction.make_lsgst_lists(
            self.gateLabels, self.fiducials, self.fiducials, self.germs, self.maxLens )

        self.lsgstStrings_tgp = pygsti.construction.make_lsgst_lists(
            self.gateLabels, self.fiducials, self.fiducials, self.germs, self.maxLens, 
            truncScheme="truncated germ powers" )

        self.lsgstStrings_lae = pygsti.construction.make_lsgst_lists(
            self.gateLabels, self.fiducials, self.fiducials, self.germs, self.maxLens, 
            truncScheme='length as exponent' )

        # RUN BELOW LINES TO GENERATE SAVED DATASETS
        #datagen_gateset = self.gateset.depolarize(gate_noise=0.05, spam_noise=0.1)
        #ds = pygsti.construction.generate_fake_data(
        #    datagen_gateset, self.lsgstStrings[-1],
        #    nSamples=1000,sampleError='binomial', seed=100)
        #
        #ds_tgp = pygsti.construction.generate_fake_data(
        #    datagen_gateset, self.lsgstStrings_tgp[-1],
        #    nSamples=1000,sampleError='binomial', seed=100)
        #
        #ds_lae = pygsti.construction.generate_fake_data(
        #    datagen_gateset, self.lsgstStrings_lae[-1],
        #    nSamples=1000,sampleError='binomial', seed=100)
        #
        #ds.save("cmp_chk_files/drivers.dataset")
        #ds_tgp.save("cmp_chk_files/drivers_tgp.dataset")
        #ds_lae.save("cmp_chk_files/drivers_lae.dataset")

    def runSilent(self, callable, *args, **kwds):
        orig_stdout = sys.stdout
        sys.stdout = open("temp_test_files/silent.txt","w")
        result = callable(*args, **kwds)
        sys.stdout.close()
        sys.stdout = orig_stdout
        return result


class TestDriversMethods(DriversTestCase):

    def test_longSequenceGST_WholeGermPowers(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom="cmp_chk_files/drivers.dataset")
        ts = "whole germ powers"

        maxLens = self.maxLens
        result = self.runSilent(pygsti.do_long_sequence_gst,
                                ds, std.gs_target, std.fiducials, std.fiducials,
                                std.germs, maxLens, truncScheme=ts)

        result = self.runSilent(pygsti.do_long_sequence_gst,
                                ds, std.gs_target, std.fiducials, std.fiducials,
                                std.germs, maxLens, truncScheme=ts, objective="chi2")

        result = self.runSilent(pygsti.do_long_sequence_gst,
                                ds, std.gs_target, std.fiducials, std.fiducials,
                                std.germs, maxLens, truncScheme=ts, constrainToTP=False)
        

        #Try using files instead of objects
        pygsti.io.write_gateset(std.gs_target, "temp_test_files/driver.gateset")
        pygsti.io.write_dataset("temp_test_files/driver_test_dataset.txt",
                                ds, self.lsgstStrings[-1])
        pygsti.io.write_gatestring_list("temp_test_files/driver_fiducials.txt", std.fiducials)
        pygsti.io.write_gatestring_list("temp_test_files/driver_germs.txt", std.germs)

        result = self.runSilent(pygsti.do_long_sequence_gst,
                                "temp_test_files/driver_test_dataset.txt", 
                                "temp_test_files/driver.gateset", 
                                "temp_test_files/driver_fiducials.txt",
                                "temp_test_files/driver_fiducials.txt",
                                "temp_test_files/driver_germs.txt",
                                maxLens, truncScheme=ts)

        #Try using effectStrs == None, gaugeOptToCPTP, and advanced options
        result = self.runSilent(pygsti.do_long_sequence_gst,
                                ds, std.gs_target, std.fiducials, None,
                                std.germs, maxLens, truncScheme=ts,
                                gaugeOptToCPTP=True, 
                                advancedOptions={'contractLGSTtoCPTP': True,
                                                 'depolarizeLGST': 0.05 })


        #Check errors
        with self.assertRaises(ValueError):
            self.runSilent(pygsti.do_long_sequence_gst,
                           ds, std.gs_target, std.fiducials, None,
                           std.germs, maxLens, truncScheme=ts, objective="FooBar") #bad objective



    def test_longSequenceGST_TruncGermPowers(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom="cmp_chk_files/drivers_tgp.dataset")
        ts = "truncated germ powers"

        maxLens = self.maxLens
        result = self.runSilent(pygsti.do_long_sequence_gst,
            ds, std.gs_target, std.fiducials, std.fiducials,
            std.germs, maxLens, truncScheme=ts)

        result = self.runSilent(pygsti.do_long_sequence_gst,
            ds, std.gs_target, std.fiducials, std.fiducials,
            std.germs, maxLens, truncScheme=ts, objective="chi2")

        result = self.runSilent(pygsti.do_long_sequence_gst,
            ds, std.gs_target, std.fiducials, std.fiducials,
            std.germs, maxLens, truncScheme=ts, constrainToTP=False)

    def test_longSequenceGST_LengthAsExponent(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom="cmp_chk_files/drivers_lae.dataset")
        ts = "length as exponent"

        maxLens = self.maxLens
        result = self.runSilent(pygsti.do_long_sequence_gst,
            ds, std.gs_target, std.fiducials, std.fiducials,
            std.germs, maxLens, truncScheme=ts)

        result = self.runSilent(pygsti.do_long_sequence_gst,
            ds, std.gs_target, std.fiducials, std.fiducials,
            std.germs, maxLens, truncScheme=ts, objective="chi2")

        result = self.runSilent(pygsti.do_long_sequence_gst,
            ds, std.gs_target, std.fiducials, std.fiducials,
            std.germs, maxLens, truncScheme=ts, constrainToTP=False)



    def test_longSequenceGST_fiducialPairReduction(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom="cmp_chk_files/drivers.dataset")
        ts = "whole germ powers"

        fidPairs = pygsti.alg.find_sufficient_fiducial_pairs(
            std.gs_target, std.fiducials, std.fiducials, std.germs, verbosity=0)

        maxLens = self.maxLens
        result = self.runSilent(pygsti.do_long_sequence_gst,
                                ds, std.gs_target, std.fiducials, std.fiducials,
                                std.germs, maxLens, truncScheme=ts, fidPairs=fidPairs)

        #create a report...
        result.create_full_report_pdf(filename="temp_test_files/full_report_FPR.pdf",
                                      debugAidsAppendix=False, gaugeOptAppendix=False,
                                      pixelPlotAppendix=False, whackamoleAppendix=False,
                                      verbosity=2)


    def test_longSequenceGST_parameterizedGates(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom="cmp_chk_files/drivers.dataset")
        ts = "whole germ powers"

        gs_target = pygsti.construction.build_gateset([2],[('Q0',)], ['Gi','Gx','Gy'], 
                                                      [ "D(Q0)","X(pi/2,Q0)", "Y(pi/2,Q0)"],
                                                      prepLabels=['rho0'], prepExpressions=["0"],
                                                      effectLabels=['E0'], effectExpressions=["1"], 
                                                      spamdefs={'plus': ('rho0','E0'),
                                                                     'minus': ('rho0','remainder') },
                                                      parameterization="linear")

        maxLens = self.maxLens
        result = self.runSilent(pygsti.do_long_sequence_gst,
                                ds, gs_target, std.fiducials, std.fiducials,
                                std.germs, maxLens, truncScheme=ts, constrainToTP=False)

        #create a report...
        result.create_full_report_pdf(filename="temp_test_files/full_report_LPGates.pdf",
                                      debugAidsAppendix=False, gaugeOptAppendix=False,
                                      pixelPlotAppendix=False, whackamoleAppendix=False,
                                      verbosity=2)



        #prepStrsListOrFilename, effectStrsListOrFilename,
        #germsListOrFilename, maxLengths, gateLabels, 
        #weightsDict, fidPairs, constrainToTP, 
        #gaugeOptToCPTP, gaugeOptRatio, objective="logl",
        #advancedOptions={}, lsgstLists=None,
        #truncScheme="whole germ powers"):

    def test_bootstrap(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom="cmp_chk_files/drivers.dataset")        
        specs = pygsti.construction.build_spam_specs(std.fiducials)
        gs = pygsti.do_lgst(ds, specs, targetGateset=std.gs_target, svdTruncateTo=4, verbosity=0)

        bootds_p = pygsti.drivers.make_bootstrap_dataset(
            ds,'parametric', gs, seed=1234 )
        bootds_np = pygsti.drivers.make_bootstrap_dataset(
            ds,'nonparametric', seed=1234 )

        with self.assertRaises(ValueError):
            pygsti.drivers.make_bootstrap_dataset(ds,'foobar', seed=1)
              #bad generationMethod
        with self.assertRaises(ValueError):
            pygsti.drivers.make_bootstrap_dataset(ds,'parametric', seed=1)
              # must specify gateset for parametric mode
        with self.assertRaises(ValueError):
            pygsti.drivers.make_bootstrap_dataset(ds,'nonparametric',gs,seed=1)
              # must *not* specify gateset for nonparametric mode

        
        maxLengths = [0] #just do LGST strings to make this fast...
        bootgs_p = self.runSilent(pygsti.drivers.make_bootstrap_gatesets,
            2, ds, 'parametric', std.fiducials, std.fiducials,
            std.germs, maxLengths, inputGateSet=gs, constrainToTP=True,
            returnData=False)

        default_maxLens = [0]+[2**k for k in range(10)]
        gateStrings = pygsti.construction.make_lsgst_experiment_list(
            self.gateLabels, self.fiducials, self.fiducials, self.germs,
            default_maxLens, fidPairs=None, truncScheme="whole germ powers")
        ds_defaultMaxLens = pygsti.construction.generate_fake_data(
            gs, gateStrings, nSamples=1000, sampleError='round')

        bootgs_p_defaultMaxLens = self.runSilent(
            pygsti.drivers.make_bootstrap_gatesets,
            2, ds_defaultMaxLens, 'parametric', std.fiducials, std.fiducials,
            std.germs, None, inputGateSet=gs, constrainToTP=True,
            returnData=False) #test when maxLengths == None

        bootgs_np, bootds_np2 = self.runSilent(
            pygsti.drivers.make_bootstrap_gatesets,
            2, ds, 'nonparametric', std.fiducials, std.fiducials,
            std.germs, maxLengths, targetGateSet=gs, 
            constrainToTP=True, returnData=True)

        with self.assertRaises(ValueError):
            pygsti.drivers.make_bootstrap_gatesets(
                2, ds, 'parametric', std.fiducials, std.fiducials,
                std.germs, maxLengths, constrainToTP=True,
                returnData=False) 
                #must specify either inputGateSet or targetGateSet

        with self.assertRaises(ValueError):
            pygsti.drivers.make_bootstrap_gatesets(
                2, ds, 'parametric', std.fiducials, std.fiducials,
                std.germs, maxLengths, inputGateSet=gs, targetGateSet=gs,
                constrainToTP=True, returnData=False) 
                #cannot specify both inputGateSet and targetGateSet


        self.runSilent(pygsti.drivers.gauge_optimize_gs_list,
                       bootgs_p, std.gs_target, constrainToTP=True,
                       gateMetric = 'frobenius', spamMetric = 'frobenius',
                       plot=False)

        #Test plotting -- seems to work.
        self.runSilent(pygsti.drivers.gauge_optimize_gs_list,
                       bootgs_p, std.gs_target, constrainToTP=True,
                       gateMetric = 'frobenius', spamMetric = 'frobenius',
                       plot=True)


        #Test utility functions -- just make sure they run for now...
        def gsFn(gs):
            return gs.get_dimension()

        tp_target = std.gs_target.copy()
        tp_target.set_all_parameterizations("TP")
        
        pygsti.drivers.gs_stdev(gsFn, bootgs_p)
        pygsti.drivers.gs_mean(gsFn, bootgs_p)
        pygsti.drivers.gs_stdev1(gsFn, bootgs_p)
        pygsti.drivers.gs_mean1(gsFn, bootgs_p)
        pygsti.drivers.to_vector(bootgs_p[0])

        pygsti.drivers.to_mean_gateset(bootgs_p, tp_target)
        pygsti.drivers.to_std_gateset(bootgs_p, tp_target)
        pygsti.drivers.to_rms_gateset(bootgs_p, tp_target)
        
        pygsti.drivers.gateset_jtracedist(bootgs_p[0], tp_target)
        pygsti.drivers.gateset_process_fidelity(bootgs_p[0], tp_target)
        pygsti.drivers.gateset_diamonddist(bootgs_p[0], tp_target)
        pygsti.drivers.gateset_decomp_angle(bootgs_p[0])
        pygsti.drivers.gateset_decomp_decay_diag(bootgs_p[0])
        pygsti.drivers.gateset_decomp_decay_offdiag(bootgs_p[0])
        pygsti.drivers.spamrameter(bootgs_p[0])

    
      
if __name__ == "__main__":
    unittest.main(verbosity=2)
