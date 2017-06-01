import unittest
import pygsti
from pygsti.construction import std1Q_XYI as std

import sys, os

from ..testutils import BaseTestCase, compare_files, temp_files

class DriversTestCase(BaseTestCase):

    def setUp(self):
        super(DriversTestCase, self).setUp()

        self.gateset = std.gs_target

        self.germs = std.germs
        self.fiducials = std.fiducials
        self.maxLens = [1,2,4]
        self.gateLabels = list(self.gateset.gates.keys())

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
        #ds.save(compare_files + "/drivers.dataset")
        #ds_tgp.save(compare_files + "/drivers_tgp.dataset")
        #ds_lae.save(compare_files + "/drivers_lae.dataset")

class TestDriversMethods(DriversTestCase):

    def test_longSequenceGST_WholeGermPowers(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/drivers.dataset")
        ts = "whole germ powers"

        maxLens = self.maxLens
        result = self.runSilent(pygsti.do_long_sequence_gst,
                                ds, std.gs_target, std.fiducials, std.fiducials,
                                std.germs, maxLens, advancedOptions={'truncScheme': ts})

        result = self.runSilent(pygsti.do_long_sequence_gst,
                                ds, std.gs_target, std.fiducials, std.fiducials,
                                std.germs, maxLens, 
                                advancedOptions={'truncScheme': ts, 'objective': "chi2"})


        #Try using files instead of objects
        pygsti.io.write_gateset(std.gs_target, temp_files + "/driver.gateset")
        pygsti.io.write_dataset(temp_files + "/driver_test_dataset.txt",
                                ds, self.lsgstStrings[-1])
        pygsti.io.write_gatestring_list(temp_files + "/driver_fiducials.txt", std.fiducials)
        pygsti.io.write_gatestring_list(temp_files + "/driver_germs.txt", std.germs)

        result = self.runSilent(pygsti.do_long_sequence_gst,
                                temp_files + "/driver_test_dataset.txt",
                                temp_files + "/driver.gateset",
                                temp_files + "/driver_fiducials.txt",
                                temp_files + "/driver_fiducials.txt",
                                temp_files + "/driver_germs.txt",
                                maxLens, advancedOptions={'truncScheme': ts})

        #Try using effectStrs == None and some advanced options
        result = self.runSilent(pygsti.do_long_sequence_gst,
                                ds, std.gs_target, std.fiducials, None,
                                std.germs, maxLens,
                                advancedOptions={'contractStartToCPTP': True,
                                                 'depolarizeStart': 0.05,
                                                 'truncScheme': ts})


        #Check errors
        with self.assertRaises(ValueError):
            self.runSilent(pygsti.do_long_sequence_gst,
                           ds, std.gs_target, std.fiducials, None,
                           std.germs, maxLens, 
                           advancedOptions={'truncScheme': ts, 'objective': "FooBar"}) #bad objective



    def test_longSequenceGST_TruncGermPowers(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/drivers_tgp.dataset")
        ts = "truncated germ powers"

        maxLens = self.maxLens
        result = self.runSilent(pygsti.do_long_sequence_gst,
            ds, std.gs_target, std.fiducials, std.fiducials,
            std.germs, maxLens, advancedOptions={'truncScheme': ts})

        result = self.runSilent(pygsti.do_long_sequence_gst,
            ds, std.gs_target, std.fiducials, std.fiducials,
            std.germs, maxLens, 
            advancedOptions={'truncScheme': ts, 'objective': "chi2"})

        #result = self.runSilent(pygsti.do_long_sequence_gst,
        #    ds, std.gs_target, std.fiducials, std.fiducials,
        #    std.germs, maxLens, truncScheme=ts, constrainToTP=False)

    def test_longSequenceGST_LengthAsExponent(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/drivers_lae.dataset")
        ts = "length as exponent"

        maxLens = self.maxLens
        result = self.runSilent(pygsti.do_long_sequence_gst,
            ds, std.gs_target, std.fiducials, std.fiducials,
            std.germs, maxLens, advancedOptions={'truncScheme': ts})

        result = self.runSilent(pygsti.do_long_sequence_gst,
            ds, std.gs_target, std.fiducials, std.fiducials,
            std.germs, maxLens,
            advancedOptions={'truncScheme': ts, 'objective': "chi2"})

        #result = self.runSilent(pygsti.do_long_sequence_gst,
        #    ds, std.gs_target, std.fiducials, std.fiducials,
        #    std.germs, maxLens, truncScheme=ts, constrainToTP=False)



    def test_longSequenceGST_fiducialPairReduction(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/drivers.dataset")
        ts = "whole germ powers"
        maxLens = self.maxLens

        fidPairs = pygsti.alg.find_sufficient_fiducial_pairs(
            std.gs_target, std.fiducials, std.fiducials, std.germs, verbosity=0)

        gfprStructs = pygsti.construction.make_lsgst_structs(
            std.gs_target, std.fiducials, std.fiducials, std.germs, maxLens,
            fidPairs=fidPairs)
    
        result = self.runSilent(pygsti.do_long_sequence_gst_base,
                                ds, std.gs_target, gfprStructs,
                                advancedOptions={'truncScheme': ts})

        #create a report...
        pygsti.report.create_single_qubit_report(result, temp_files + "/full_report_FPR.html",
                                                 verbosity=2)
        #import os
        #print("LOG DEBUG")
        #os.system("cat " + temp_files + "/full_report_FPR.log")


    def test_longSequenceGST_randomReduction(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/drivers.dataset")
        ts = "whole germ powers"
        maxLens = self.maxLens

        #Without fixed initial fiducial pairs
        fidPairs = None
        reducedLists = pygsti.construction.make_lsgst_structs(
            std.gs_target.gates.keys(), std.fiducials, std.fiducials, std.germs,
            maxLens, fidPairs, ts, keepFraction=0.5, keepSeed=1234)
        result = pygsti.do_long_sequence_gst_base(
            ds, std.gs_target, reducedLists,
            advancedOptions={'truncScheme': ts}) #self.runSilent(

        #create a report...
        pygsti.report.create_single_qubit_report(result, temp_files + "/full_report_RFPR.html",
                                                 verbosity=2)

        #With fixed initial fiducial pairs
        fidPairs = pygsti.alg.find_sufficient_fiducial_pairs(
            std.gs_target, std.fiducials, std.fiducials, std.germs, verbosity=0)
        reducedLists = pygsti.construction.make_lsgst_structs(
            std.gs_target.gates.keys(), std.fiducials, std.fiducials, std.germs,
            maxLens, fidPairs, ts, keepFraction=0.5, keepSeed=1234)
        result2 = self.runSilent(pygsti.do_long_sequence_gst_base,
                                 ds, std.gs_target, reducedLists,
                                 advancedOptions={'truncScheme': ts})

        #create a report...
        pygsti.report.create_single_qubit_report(result2, temp_files + "/full_report_RFPR2.html",
                                                 verbosity=2)


    def test_longSequenceGST_parameterizedGates(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/drivers.dataset")
        ts = "whole germ powers"

        gs_target = pygsti.construction.build_gateset([2],[('Q0',)], ['Gi','Gx','Gy'],
                                                      [ "D(Q0)","X(pi/2,Q0)", "Y(pi/2,Q0)"],
                                                      prepLabels=['rho0'], prepExpressions=["0"],
                                                      effectLabels=['E0'], effectExpressions=["1"],
                                                      spamdefs={'plus': ('rho0','E0'),
                                                                     'minus': ('rho0','remainder') },
                                                      parameterization="linear")

        maxLens = self.maxLens
        result = pygsti.do_long_sequence_gst( #self.runSilent(
                                ds, gs_target, std.fiducials, std.fiducials,
                                std.germs, maxLens,
                                advancedOptions={'truncScheme': ts, 'tolerance':1e-4} )
                                #decrease tolerance
                                # b/c this problem seems hard to converge at the very end
                                # very small changes (~0.0001) to the total chi^2.

        #create a report...
        pygsti.report.create_single_qubit_report(result, temp_files + "/full_report_LPGates.html",
                                                 verbosity=2)
                


    def test_longSequenceGST_wMapCalc(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/drivers.dataset")
        ts = "whole germ powers"

        gs_target = std.gs_target.copy()
        gs_target._calcClass = pygsti.objects.gatemapcalc.GateMapCalc

        maxLens = self.maxLens
        result = self.runSilent(pygsti.do_long_sequence_gst,
                                ds, gs_target, std.fiducials, std.fiducials,
                                std.germs, maxLens, advancedOptions={'truncScheme': ts})


    def test_bootstrap(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/drivers.dataset")
        specs = self.runSilent(pygsti.construction.build_spam_specs, std.fiducials)
        tp_target = std.gs_target.copy(); tp_target.set_all_parameterizations("TP")
        gs = pygsti.do_lgst(ds, specs, targetGateset=tp_target, svdTruncateTo=4, verbosity=0)

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
            std.germs, maxLengths, inputGateSet=gs,
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
            std.germs, None, inputGateSet=gs,
            returnData=False) #test when maxLengths == None

        bootgs_np, bootds_np2 = self.runSilent(
            pygsti.drivers.make_bootstrap_gatesets,
            2, ds, 'nonparametric', std.fiducials, std.fiducials,
            std.germs, maxLengths, targetGateSet=gs,
            returnData=True)

        with self.assertRaises(ValueError):
            pygsti.drivers.make_bootstrap_gatesets(
                2, ds, 'parametric', std.fiducials, std.fiducials,
                std.germs, maxLengths,returnData=False)
                #must specify either inputGateSet or targetGateSet

        with self.assertRaises(ValueError):
            pygsti.drivers.make_bootstrap_gatesets(
                2, ds, 'parametric', std.fiducials, std.fiducials,
                std.germs, maxLengths, inputGateSet=gs, targetGateSet=gs,
                returnData=False) #cannot specify both inputGateSet and targetGateSet


        self.runSilent(pygsti.drivers.gauge_optimize_gs_list,
                       bootgs_p, std.gs_target, gateMetric = 'frobenius',
                       spamMetric = 'frobenius', plot=False)

        ##Test plotting -- removed b/c plotting was removed w/matplotlib removal
        #self.runSilent(pygsti.drivers.gauge_optimize_gs_list,
        #               bootgs_p, std.gs_target,
        #               gateMetric = 'frobenius', spamMetric = 'frobenius',
        #               plot=True)


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
