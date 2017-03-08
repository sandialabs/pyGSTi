import unittest
import pygsti
from pygsti.construction import std1Q_XYI as std

import numpy as np
from scipy import polyfit
import sys, os

from ..testutils import BaseTestCase, compare_files, temp_files

class TestCoreMethods(BaseTestCase):

    def setUp(self):
        super(TestCoreMethods, self).setUp()

        self.gateset = std.gs_target
        self.datagen_gateset = self.gateset.depolarize(gate_noise=0.05, spam_noise=0.1)

        self.fiducials = std.fiducials
        self.germs = std.germs
        self.specs = pygsti.construction.build_spam_specs(self.fiducials, effect_labels=['E0']) #only use the first EVec

        self.gateLabels = list(self.gateset.gates.keys()) # also == std.gates
        self.lgstStrings = pygsti.construction.list_lgst_gatestrings(self.specs, self.gateLabels)

        self.maxLengthList = [0,1,2,4,8]

        self.elgstStrings = pygsti.construction.make_elgst_lists(
            self.gateLabels, self.germs, self.maxLengthList )

        self.lsgstStrings = pygsti.construction.make_lsgst_lists(
            self.gateLabels, self.fiducials, self.fiducials, self.germs, self.maxLengthList )

        #Created in testAnalysis...
        self.ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/analysis.dataset")

        ##UNCOMMENT to create LGST analysis dataset
        #ds_lgst = pygsti.construction.generate_fake_data(self.datagen_gateset, self.lgstStrings,
        #                                                 nSamples=10000,sampleError='binomial', seed=100)
        #ds_lgst.save(compare_files + "/analysis_lgst.dataset")
        self.ds_lgst = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/analysis_lgst.dataset")


    def test_gram(self):
        ds = self.ds
        #pygsti.construction.generate_fake_data(self.datagen_gateset, self.lgstStrings,
        #                                            nSamples=1000, sampleError='none')

        rank,evals,target_evals = pygsti.gram_rank_and_evals(ds, self.specs, self.gateset)
        print("gram rank = ",rank)
        print("gram evals = ",evals)
        print("target gram evals = ",target_evals)

        with self.assertRaises(ValueError):
            pygsti.gram_rank_and_evals(ds, self.specs, None) #no spam labels

    def test_LGST(self):

        ds = self.ds
        #pygsti.construction.generate_fake_data(self.datagen_gateset, self.lgstStrings, nSamples=1000,
        #                                            sampleError='binomial', seed=None)

        print("GG0 = ",self.gateset.default_gauge_group)
        gs_lgst = pygsti.do_lgst(ds, self.specs, self.gateset, svdTruncateTo=4, verbosity=0)
        gs_lgst_verb = self.runSilent(pygsti.do_lgst, ds, self.specs, self.gateset, svdTruncateTo=4, verbosity=10)
        self.assertAlmostEqual(gs_lgst.frobeniusdist(gs_lgst_verb),0)

        print("GG = ",gs_lgst.default_gauge_group)
        gs_lgst_go = pygsti.optimize_gauge(gs_lgst,"target",targetGateset=self.gateset, spamWeight=1.0, gateWeight=1.0) #DEPRECATED
        gs_lgst_go = pygsti.gaugeopt_to_target(gs_lgst,self.gateset, {'spam':1.0, 'gates': 1.0})
        gs_clgst = pygsti.contract(gs_lgst_go, "CPTP")

        # RUN BELOW LINES TO SEED SAVED GATESET FILES
        #pygsti.io.write_gateset(gs_lgst,compare_files + "/lgst.gateset", "Saved LGST Gateset before gauge optimization")
        #pygsti.io.write_gateset(gs_lgst_go,compare_files + "/lgst_go.gateset", "Saved LGST Gateset after gauge optimization")
        #pygsti.io.write_gateset(gs_clgst,compare_files + "/clgst.gateset", "Saved LGST Gateset after G.O. and CPTP contraction")

        gs_lgst_compare = pygsti.io.load_gateset(compare_files + "/lgst.gateset")
        gs_lgst_go_compare = pygsti.io.load_gateset(compare_files + "/lgst_go.gateset")
        gs_clgst_compare = pygsti.io.load_gateset(compare_files + "/clgst.gateset")

        self.assertAlmostEqual( gs_lgst.frobeniusdist(gs_lgst_compare), 0)
        self.assertAlmostEqual( gs_lgst_go.frobeniusdist(gs_lgst_go_compare), 0)
        self.assertAlmostEqual( gs_clgst.frobeniusdist(gs_clgst_compare), 0)

        #Check for error conditions
        with self.assertRaises(ValueError):
            gs_lgst = pygsti.do_lgst(ds, self.specs, None, svdTruncateTo=4, verbosity=0) #no gate labels

        with self.assertRaises(ValueError):
            gs_lgst = pygsti.do_lgst(ds, self.specs, None, gateLabels=list(self.gateset.gates.keys()),
                                     svdTruncateTo=4, verbosity=0) #no spam dict

        with self.assertRaises(ValueError):
            gs_lgst = pygsti.do_lgst(ds, self.specs, None, gateLabels=list(self.gateset.gates.keys()),
                                     spamDict=self.gateset.get_reverse_spam_defs(),
                                     svdTruncateTo=4, verbosity=0) #no identity vector

        with self.assertRaises(ValueError):
            bad_specs = pygsti.construction.build_spam_specs(
                pygsti.construction.gatestring_list([('Gx',),('Gx',),('Gx',),('Gx',)]), effect_labels=['E0'])
            gs_lgst = pygsti.do_lgst(ds, bad_specs, self.gateset, svdTruncateTo=4, verbosity=0) # bad specs (rank deficient)


        with self.assertRaises(KeyError): # AB-matrix construction error
            incomplete_strings = self.lgstStrings[5:] #drop first 5 strings...
            bad_ds = pygsti.construction.generate_fake_data(
                self.datagen_gateset, incomplete_strings,
                nSamples=10, sampleError='none')
            gs_lgst = pygsti.do_lgst(bad_ds, self.specs, self.gateset,
                                     svdTruncateTo=4, verbosity=0)
                      # incomplete dataset

        with self.assertRaises(KeyError): # X-matrix construction error
            incomplete_strings = self.lgstStrings[:-5] #drop last 5 strings...
            bad_ds = pygsti.construction.generate_fake_data(
                self.datagen_gateset, incomplete_strings,
                nSamples=10, sampleError='none')
            gs_lgst = pygsti.do_lgst(bad_ds, self.specs, self.gateset,
                                     svdTruncateTo=4, verbosity=0)
                      # incomplete dataset





    def test_LGST_no_sample_error(self):
        ds = pygsti.construction.generate_fake_data(self.datagen_gateset, self.lgstStrings,
                                                    nSamples=1000, sampleError='none')
        gs_lgst = pygsti.do_lgst(ds, self.specs, self.gateset, svdTruncateTo=4, verbosity=0)
        #gs_lgst = pygsti.optimize_gauge(gs_lgst, "target", targetGateset=self.datagen_gateset, gateWeight=1.0, spamWeight=1.0) #DEPRECATED
        gs_lgst = pygsti.gaugeopt_to_target(gs_lgst,self.datagen_gateset, {'spam':1.0, 'gates': 1.0})
        self.assertAlmostEqual( gs_lgst.frobeniusdist(self.datagen_gateset), 0)


    def test_eLGST(self):

        ds = self.ds
        #pygsti.construction.generate_fake_data(self.datagen_gateset, self.lsgstStrings[-1],
        #                                            nSamples=1000,sampleError='binomial', seed=100)

        gs_lgst = pygsti.do_lgst(ds, self.specs, self.gateset, svdTruncateTo=4, verbosity=0)
        gs_lgst_go = pygsti.optimize_gauge(gs_lgst,"target",targetGateset=self.gateset, spamWeight=1.0, gateWeight=1.0) #DEPRECATED
        gs_lgst_go = pygsti.gaugeopt_to_target(gs_lgst,self.gateset, {'spam':1.0, 'gates': 1.0})
        gs_clgst = pygsti.contract(gs_lgst_go, "CPTP")

        gs_single_exlgst = pygsti.do_exlgst(ds, gs_clgst, self.elgstStrings[0], self.specs,
                                            self.gateset, regularizeFactor=1e-3, svdTruncateTo=4,
                                            verbosity=0)
        gs_single_exlgst_verb = self.runSilent(pygsti.do_exlgst, ds, gs_clgst, self.elgstStrings[0], self.specs,
                                               self.gateset, regularizeFactor=1e-3, svdTruncateTo=4,
                                               verbosity=10)

        gs_exlgst = pygsti.do_iterative_exlgst(ds, gs_clgst, self.specs, self.elgstStrings,
                                               targetGateset=self.gateset, svdTruncateTo=4, verbosity=0)
        all_minErrs, all_gs_exlgst_tups = pygsti.do_iterative_exlgst(
            ds, gs_clgst, self.specs, [ [gs.tup for gs in gsList] for gsList in self.elgstStrings],
            targetGateset=self.gateset, svdTruncateTo=4, verbosity=0, returnAll=True, returnErrorVec=True)

        gs_exlgst_verb = self.runSilent(pygsti.do_iterative_exlgst, ds, gs_clgst, self.specs, self.elgstStrings,
                                        targetGateset=self.gateset, svdTruncateTo=4, verbosity=10)
        gs_exlgst_reg = pygsti.do_iterative_exlgst(ds, gs_clgst, self.specs, self.elgstStrings,
                                               targetGateset=self.gateset, svdTruncateTo=4, verbosity=0,
                                               regularizeFactor=10)
        self.assertAlmostEqual(gs_exlgst.frobeniusdist(gs_exlgst_verb),0)
        self.assertAlmostEqual(gs_exlgst.frobeniusdist(all_gs_exlgst_tups[-1]),0)


        #Run internal checks on less max-L values (so it doesn't take forever)
        gs_exlgst_chk = pygsti.do_iterative_exlgst(ds, gs_clgst, self.specs, self.elgstStrings[0:2],
                                                   targetGateset=self.gateset, svdTruncateTo=4, verbosity=0,
                                                   check_jacobian=True)
        gs_exlgst_chk_verb = self.runSilent(pygsti.do_iterative_exlgst,ds, gs_clgst, self.specs, self.elgstStrings[0:2],
                                                   targetGateset=self.gateset, svdTruncateTo=4, verbosity=10,
                                                   check_jacobian=True)

        # RUN BELOW LINES TO SEED SAVED GATESET FILES
        #pygsti.io.write_gateset(gs_exlgst,compare_files + "/exlgst.gateset", "Saved Extended-LGST (eLGST) Gateset")
        #pygsti.io.write_gateset(gs_exlgst_reg,compare_files + "/exlgst_reg.gateset", "Saved Extended-LGST (eLGST) Gateset w/regularization")

        gs_exlgst_compare = pygsti.io.load_gateset(compare_files + "/exlgst.gateset")
        gs_exlgst_reg_compare = pygsti.io.load_gateset(compare_files + "/exlgst_reg.gateset")
        gs_exlgst.set_all_parameterizations("full") # b/c ex-LGST sets spam to StaticSPAMVec objects (b/c they're not optimized)
        gs_exlgst_reg.set_all_parameterizations("full") # b/c ex-LGST sets spam to StaticSPAMVec objects (b/c they're not optimized)
        gs_exlgst_go = pygsti.optimize_gauge(gs_exlgst, 'target', targetGateset=gs_exlgst_compare, spamWeight=1.0)  #DEPRECATED
        gs_exlgst_go = pygsti.gaugeopt_to_target(gs_exlgst,gs_exlgst_compare, {'spam':1.0 })
        gs_exlgst_reg_go = pygsti.optimize_gauge(gs_exlgst_reg, 'target', targetGateset=gs_exlgst_reg_compare, spamWeight=1.0) #DEPRECATED
        gs_exlgst_reg_go = pygsti.gaugeopt_to_target(gs_exlgst_reg,gs_exlgst_reg_compare, {'spam':1.0 })

        self.assertAlmostEqual( gs_exlgst_go.frobeniusdist(gs_exlgst_compare), 0, places=5)
        self.assertAlmostEqual( gs_exlgst_reg_go.frobeniusdist(gs_exlgst_reg_compare), 0, places=5)


    def test_MC2GST(self):

        ds = self.ds
        #pygsti.construction.generate_fake_data(self.datagen_gateset, self.lsgstStrings[-1],
        #                                            nSamples=1000, sampleError='binomial', seed=100)

        gs_lgst = pygsti.do_lgst(ds, self.specs, self.gateset, svdTruncateTo=4, verbosity=0)
        gs_lgst_go = pygsti.optimize_gauge(gs_lgst,"target",targetGateset=self.gateset, spamWeight=1.0, gateWeight=1.0) #DEPRECATED
        gs_lgst_go = pygsti.gaugeopt_to_target(gs_lgst,self.gateset, {'spam':1.0, 'gates': 1.0})
        gs_clgst = pygsti.contract(gs_lgst_go, "CPTP")
        CM = pygsti.objects.profiler._get_mem_usage()

        gs_single_lsgst = pygsti.do_mc2gst(ds, gs_clgst, self.lsgstStrings[0], minProbClipForWeighting=1e-6,
                                           probClipInterval=(-1e6,1e6), regularizeFactor=1e-3,
                                           verbosity=0) #uses regularizeFactor

        gs_single_lsgst_cp = pygsti.do_mc2gst(ds, gs_clgst, self.lsgstStrings[0], minProbClipForWeighting=1e-6,
                                           probClipInterval=(-1e6,1e6), cptp_penalty_factor=1.0,
                                           verbosity=0) #uses cptp_penalty_factor

        gs_lsgst = pygsti.do_iterative_mc2gst(ds, gs_clgst, self.lsgstStrings, verbosity=0,
                                             minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6),
                                             memLimit=CM + 1024**3)
        all_minErrs, all_gs_lsgst_tups = pygsti.do_iterative_mc2gst(
            ds, gs_clgst, [ [gs.tup for gs in gsList] for gsList in self.lsgstStrings],
            minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6), returnAll=True, returnErrorVec=True)
        gs_lsgst_verb = self.runSilent(pygsti.do_iterative_mc2gst, ds, gs_clgst, self.lsgstStrings, verbosity=10,
                                             minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6),
                                             memLimit=CM + 1024**3)
        gs_lsgst_reg = self.runSilent(pygsti.do_iterative_mc2gst,ds, gs_clgst,
                                      self.lsgstStrings, verbosity=10,
                                      minProbClipForWeighting=1e-6,
                                      probClipInterval=(-1e6,1e6),
                                      regularizeFactor=10, memLimit=CM + 1024**3)
        self.assertAlmostEqual(gs_lsgst.frobeniusdist(gs_lsgst_verb),0)
        self.assertAlmostEqual(gs_lsgst.frobeniusdist(all_gs_lsgst_tups[-1]),0)


        #Run internal checks on less max-L values (so it doesn't take forever)
        gs_lsgst_chk = pygsti.do_iterative_mc2gst(ds, gs_clgst, self.lsgstStrings[0:2], verbosity=0,
                                                 minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6),
                                                 check=True, check_jacobian=True)
        gs_lsgst_chk_verb = self.runSilent(pygsti.do_iterative_mc2gst, ds, gs_clgst, self.lsgstStrings[0:2], verbosity=10,
                                                      minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6),
                                                      check=True, check_jacobian=True, memLimit=CM + 1024**3)

        #Other option variations - just make sure they run at this point
        gs_lsgst_chk_opts = pygsti.do_iterative_mc2gst(ds, gs_clgst, self.lsgstStrings[0:2], verbosity=0,
                                                      minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6),
                                                      useFreqWeightedChiSq=True, gateStringSetLabels=["Set1","Set2"],
                                                      gatestringWeightsDict={ ('Gx',): 2.0 } )

        #Check with small but ok memlimit -- not anymore since new mem estimation uses current memory, making this non-robust
        #self.runSilent(pygsti.do_mc2gst,ds, gs_clgst, self.lsgstStrings[0], minProbClipForWeighting=1e-6,
        #                 probClipInterval=(-1e6,1e6), regularizeFactor=1e-3,
        #                 verbosity=10, memLimit=CM + 1024**3)


        #Check errors:
        with self.assertRaises(MemoryError):
            pygsti.do_mc2gst(ds, gs_clgst, self.lsgstStrings[0], minProbClipForWeighting=1e-6,
                             probClipInterval=(-1e6,1e6), regularizeFactor=1e-3,
                             verbosity=0, memLimit=1)

        with self.assertRaises(AssertionError):
            pygsti.do_mc2gst(ds, gs_clgst, self.lsgstStrings[0], minProbClipForWeighting=1e-6,
                             probClipInterval=(-1e6,1e6), regularizeFactor=1e-3,
                             verbosity=0, cptp_penalty_factor=1.0) #can't specify both cptp_penalty_factor and regularizeFactor


        # RUN BELOW LINES TO SEED SAVED GATESET FILES
        #pygsti.io.write_gateset(gs_lsgst,compare_files + "/lsgst.gateset", "Saved LSGST Gateset")
        #pygsti.io.write_gateset(gs_lsgst_reg,compare_files + "/lsgst_reg.gateset", "Saved LSGST Gateset w/Regularization")

        gs_lsgst_compare = pygsti.io.load_gateset(compare_files + "/lsgst.gateset")
        gs_lsgst_reg_compare = pygsti.io.load_gateset(compare_files + "/lsgst_reg.gateset")

        gs_lsgst_go = pygsti.optimize_gauge(gs_lsgst, 'target', targetGateset=gs_lsgst_compare, spamWeight=1.0) #DEPRECATED
        gs_lsgst_go = pygsti.gaugeopt_to_target(gs_lsgst, gs_lsgst_compare, {'spam':1.0})

        gs_lsgst_reg_go = pygsti.optimize_gauge(gs_lsgst_reg, 'target', targetGateset=gs_lsgst_reg_compare, spamWeight=1.0) #DEPRECATED
        gs_lsgst_reg_go = pygsti.gaugeopt_to_target(gs_lsgst_reg, gs_lsgst_reg_compare, {'spam':1.0})

        self.assertAlmostEqual( gs_lsgst_go.frobeniusdist(gs_lsgst_compare), 0, places=4)
        self.assertAlmostEqual( gs_lsgst_reg_go.frobeniusdist(gs_lsgst_reg_compare), 0, places=4)


    def test_MLGST(self):

        ds = self.ds
        #pygsti.construction.generate_fake_data(self.datagen_gateset, self.lsgstStrings[-1],
        #                                            nSamples=1000, sampleError='binomial', seed=100)

        gs_lgst = pygsti.do_lgst(ds, self.specs, self.gateset, svdTruncateTo=4, verbosity=0)
        gs_lgst_go = pygsti.optimize_gauge(gs_lgst,"target",targetGateset=self.gateset, spamWeight=1.0, gateWeight=1.0) #DEPRECATED
        gs_lgst_go = pygsti.gaugeopt_to_target(gs_lgst,self.gateset, {'spam':1.0, 'gates': 1.0})
        gs_clgst = pygsti.contract(gs_lgst_go, "CPTP")
        CM = pygsti.objects.profiler._get_mem_usage()

        gs_single_mlgst = pygsti.do_mlgst(ds, gs_clgst, self.lsgstStrings[0], minProbClip=1e-6,
                                          probClipInterval=(-1e2,1e2), verbosity=0)

        gs_mlegst = pygsti.do_iterative_mlgst(ds, gs_clgst, self.lsgstStrings, verbosity=0,
                                               minProbClip=1e-6, probClipInterval=(-1e2,1e2),
                                               memLimit=CM + 1024**3)
        maxLogL, all_gs_mlegst_tups = pygsti.do_iterative_mlgst(
            ds, gs_clgst, [ [gs.tup for gs in gsList] for gsList in self.lsgstStrings],
            minProbClip=1e-6, probClipInterval=(-1e2,1e2), returnAll=True, returnMaxLogL=True)

        gs_mlegst_verb = self.runSilent(pygsti.do_iterative_mlgst, ds, gs_clgst, self.lsgstStrings, verbosity=10,
                                             minProbClip=1e-6, probClipInterval=(-1e2,1e2),
                                             memLimit=CM + 1024**3)
        self.assertAlmostEqual(gs_mlegst.frobeniusdist(gs_mlegst_verb),0)
        self.assertAlmostEqual(gs_mlegst.frobeniusdist(all_gs_mlegst_tups[-1]),0)


        #Run internal checks on less max-L values (so it doesn't take forever)
        gs_mlegst_chk = pygsti.do_iterative_mlgst(ds, gs_clgst, self.lsgstStrings[0:2], verbosity=0,
                                                 minProbClip=1e-6, probClipInterval=(-1e2,1e2),
                                                 check=True)

        #Other option variations - just make sure they run at this point
        gs_mlegst_chk_opts = pygsti.do_iterative_mlgst(ds, gs_clgst, self.lsgstStrings[0:2], verbosity=0,
                                                       minProbClip=1e-6, probClipInterval=(-1e2,1e2),
                                                       gateStringSetLabels=["Set1","Set2"], useFreqWeightedChiSq=True,
                                                       gatestringWeightsDict={ ('Gx',): 2.0 } )

        aliased_list = [ pygsti.obj.GateString( [ (x if x != "Gx" else "GA1") for x in gs]) for gs in self.lsgstStrings[0] ]
        gs_withA1 = gs_clgst.copy(); gs_withA1.gates["GA1"] = gs_clgst.gates["Gx"]
        del gs_withA1.gates["Gx"] # otherwise gs_withA1 will have Gx params that we have no knowledge of!
        gs_mlegst_chk_opts2 = pygsti.do_mlgst(ds, gs_withA1, aliased_list, minProbClip=1e-6,
                                              probClipInterval=(-1e2,1e2), verbosity=10,
                                              gateLabelAliases={ 'GA1': ('Gx',) })

        #Other option variations - just make sure they run at this point
        gs_mlegst_chk_opts = pygsti.do_iterative_mlgst(ds, gs_clgst, self.lsgstStrings[0:2], verbosity=0,
                                                       minProbClip=1e-6, probClipInterval=(-1e2,1e2),
                                                       gateStringSetLabels=["Set1","Set2"], useFreqWeightedChiSq=True,
                                                       gatestringWeightsDict={ ('Gx',): 2.0 } )


        #Check with small but ok memlimit -- not anymore since new mem estimation uses current memory, making this non-robust
        #self.runSilent(pygsti.do_mlgst, ds, gs_clgst, self.lsgstStrings[0], minProbClip=1e-6,
        #                probClipInterval=(-1e2,1e2), verbosity=4, memLimit=curMem+8500000) #invoke memory control

        pygsti.do_mlgst(ds, gs_clgst, self.lsgstStrings[0], minProbClip=1e-6,
                        probClipInterval=(-1e2,1e2), verbosity=0, poissonPicture=False)
                       #non-Poisson picture - should use (-1,-1) gateset for consistency?


        #Check errors:
        with self.assertRaises(MemoryError):
            pygsti.do_mlgst(ds, gs_clgst, self.lsgstStrings[0], minProbClip=1e-6,
                            probClipInterval=(-1e2,1e2),verbosity=0, memLimit=1)


        # RUN BELOW LINES TO SEED SAVED GATESET FILES
        #pygsti.io.write_gateset(gs_mlegst,compare_files + "/mle_gst.gateset", "Saved MLE-GST Gateset")

        gs_mle_compare = pygsti.io.load_gateset(compare_files + "/mle_gst.gateset")
        gs_mlegst_go = pygsti.optimize_gauge(gs_mlegst, 'target', targetGateset=gs_mle_compare, spamWeight=1.0) #DEPRECATED
        gs_mlegst_go = pygsti.gaugeopt_to_target(gs_mlegst, gs_mle_compare, {'spam':1.0})

        self.assertAlmostEqual( gs_mlegst_go.frobeniusdist(gs_mle_compare), 0, places=5)

    def test_LGST_1overSqrtN_dependence(self):
        my_datagen_gateset = self.gateset.depolarize(gate_noise=0.05, spam_noise=0)
        # !!don't depolarize spam or 1/sqrt(N) dependence saturates!!

        nSamplesList = np.array([ 16, 128, 1024, 8192 ])
        diffs = []
        for nSamples in nSamplesList:
            ds = pygsti.construction.generate_fake_data(my_datagen_gateset, self.lgstStrings, nSamples,
                                                        sampleError='binomial', seed=100)
            gs_lgst = pygsti.do_lgst(ds, self.specs, self.gateset, svdTruncateTo=4, verbosity=0)
            gs_lgst_go = pygsti.optimize_gauge(gs_lgst,"target",targetGateset=my_datagen_gateset,
                                               spamWeight=1.0, gateWeight=1.0) #DEPRECATED
            gs_lgst_go = pygsti.gaugeopt_to_target(gs_lgst, my_datagen_gateset, {'spam':1.0, 'gate': 1.0})
            diffs.append( my_datagen_gateset.frobeniusdist(gs_lgst_go) )

        diffs = np.array(diffs, 'd')
        a,b = polyfit(np.log10(nSamplesList), np.log10(diffs), deg=1)
        #print "\n",nSamplesList; print diffs; print a #DEBUG
        self.assertLess( a+0.5, 0.05 )


    def test_model_selection(self):

        ds = self.ds
        #pygsti.construction.generate_fake_data(self.datagen_gateset, self.lsgstStrings[-1],
        #                                            nSamples=1000,sampleError='binomial', seed=100)


        gs_lgst4 = pygsti.do_lgst(ds, self.specs, self.gateset, svdTruncateTo=4, verbosity=0)
        gs_lgst6 = pygsti.do_lgst(ds, self.specs, self.gateset, svdTruncateTo=6, verbosity=0)
        sys.stdout.flush()

        self.runSilent(pygsti.do_lgst, ds, self.specs, self.gateset, svdTruncateTo=6, verbosity=4) # test verbose prints

        chiSq4 = pygsti.chi2(ds, gs_lgst4, self.lgstStrings, minProbClipForWeighting=1e-4)
        chiSq6 = pygsti.chi2(ds, gs_lgst6, self.lgstStrings, minProbClipForWeighting=1e-4)

        print("LGST dim=4 chiSq = ",chiSq4)
        print("LGST dim=6 chiSq = ",chiSq6)
        #self.assertAlmostEqual(chiSq4, 174.061524953) #429.271983052)
        #self.assertAlmostEqual(chiSq6, 267012993.861, places=1) #1337.74222467) #Why is this so large??? -- DEBUG later

        # Least squares GST with model selection
        gs_lsgst = self.runSilent(pygsti.do_iterative_mc2gst_with_model_selection, ds, gs_lgst4, 1, self.lsgstStrings[0:3],
                                  verbosity=10, minProbClipForWeighting=1e-3, probClipInterval=(-1e5,1e5))

        # Run again with other parameters
        tuple_strings = [ list(map(tuple, gsList)) for gsList in self.lsgstStrings[0:3] ] #to test tuple argument
        errorVecs, gs_lsgst_wts = self.runSilent(pygsti.do_iterative_mc2gst_with_model_selection, ds, gs_lgst4,
                                                 1, tuple_strings, verbosity=10, minProbClipForWeighting=1e-3,
                                                 probClipInterval=(-1e5,1e5), gatestringWeightsDict={ ('Gx',): 2.0 },
                                                 returnAll=True, returnErrorVec=True)

        # Do non-iterative to cover GateString->tuple conversion
        gs_non_iterative = self.runSilent( pygsti.do_mc2gst_with_model_selection, ds,
                                           gs_lgst4, 1, self.lsgstStrings[0],
                                           verbosity=10, probClipInterval=(-1e5,1e5) )


        # RUN BELOW LINES TO SEED SAVED GATESET FILES
        #pygsti.io.write_gateset(gs_lsgst,compare_files + "/lsgstMS.gateset", "Saved LSGST Gateset with model selection")

        gs_lsgst_compare = pygsti.io.load_gateset(compare_files + "/lsgstMS.gateset")
        gs_lsgst_go = pygsti.optimize_gauge(gs_lsgst, 'target', targetGateset=gs_lsgst_compare, spamWeight=1.0) #DEPRECATED
        gs_lsgst_go = pygsti.gaugeopt_to_target(gs_lsgst, gs_lsgst_compare, {'spam':1.0})
        self.assertAlmostEqual( gs_lsgst_go.frobeniusdist(gs_lsgst_compare), 0, places=4)

    def test_miscellaneous(self):
        ds = self.ds
        #pygsti.construction.generate_fake_data(self.datagen_gateset, self.lgstStrings,
        #                                            nSamples=1000, sampleError='none')

        strs = pygsti.construction.list_strings_lgst_can_estimate(ds, self.specs)

        self.runSilent(self.gateset.print_info) #just make sure it works


    def test_gaugeopt_and_contract(self):
        ds = self.ds_lgst
        #pygsti.construction.generate_fake_data(self.datagen_gateset, self.lgstStrings,
        #                                            nSamples=10000,sampleError='binomial', seed=100)

        gs_lgst = pygsti.do_lgst(ds, self.specs, self.gateset, svdTruncateTo=4, verbosity=0)

        #Gauge Opt to Target
        gs_lgst_target     = self.runSilent(pygsti.optimize_gauge, gs_lgst,"target",targetGateset=self.gateset,verbosity=10) #DEPRECATED
        gs_lgst_target     = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst, self.gateset, verbosity=10)

        gs_clgst_cp    = self.runSilent(pygsti.contract, gs_lgst_target, "CP",verbosity=10, tol=10.0, useDirectCP=False) #DEBUG

        #Gauge Opt to Target using non-frobenius metrics
        gs_lgst_targetAlt  = self.runSilent(pygsti.optimize_gauge, gs_lgst_target,"target",targetGateset=self.gateset,
                                            targetGatesMetric='fidelity', verbosity=10) #DEPRECATED
        gs_lgst_targetAlt  = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst_target, self.gateset,
                                            gatesMetric='fidelity', verbosity=10)

        gs_lgst_targetAlt  = self.runSilent(pygsti.optimize_gauge, gs_lgst_target,"target",targetGateset=self.gateset,
                                            targetGatesMetric='tracedist', verbosity=10) #DEPRECATED
        gs_lgst_targetAlt  = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst_target, self.gateset,
                                            gatesMetric='tracedist', verbosity=10)

        gs_lgst_targetAlt  = self.runSilent(pygsti.optimize_gauge, gs_lgst_target,"target",targetGateset=self.gateset,
                                            targetSpamMetric='fidelity', verbosity=10) #DEPRECATED
        gs_lgst_targetAlt  = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst_target, self.gateset,
                                            spamMetric='fidelity', verbosity=10)

        gs_lgst_targetAlt  = self.runSilent(pygsti.optimize_gauge, gs_lgst_target,"target",targetGateset=self.gateset,
                                            targetSpamMetric='tracedist', verbosity=10) #DEPRECATED
        gs_lgst_targetAlt  = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst_target, self.gateset,
                                            spamMetric='tracedist', verbosity=10)


        with self.assertRaises(ValueError):
            self.runSilent(pygsti.optimize_gauge, gs_lgst_target,"target",targetGateset=self.gateset,
                           targetGatesMetric='foobar', verbosity=10) #bad targetGatesMetric #DEPRECATED
        with self.assertRaises(ValueError):
            self.runSilent(pygsti.gaugeopt_to_target, gs_lgst_target, self.gateset,
                           gatesMetric='foobar', verbosity=10) #bad gatesMetric

        with self.assertRaises(ValueError):
            self.runSilent(pygsti.optimize_gauge, gs_lgst_target,"target",targetGateset=self.gateset,
                           targetSpamMetric='foobar', verbosity=10) #bad targetSpamMetric #DEPRECATED
        with self.assertRaises(ValueError):
            self.runSilent(pygsti.gaugeopt_to_target, gs_lgst_target, self.gateset,
                           spamMetric='foobar', verbosity=10) #bad spamMetric

        with self.assertRaises(ValueError):
            self.runSilent(pygsti.optimize_gauge, gs_lgst_target,"foobar",targetGateset=self.gateset,
                           targetSpamMetric='target', verbosity=10) #bad toGetTo #DEPRECATED


        #Contractions
        gs_clgst_tp    = self.runSilent(pygsti.contract, gs_lgst_target, "TP",verbosity=10, tol=10.0)
        gs_clgst_cp    = self.runSilent(pygsti.contract, gs_lgst_target, "CP",verbosity=10, tol=10.0)
        gs_clgst_cptp  = self.runSilent(pygsti.contract, gs_lgst_target, "CPTP",verbosity=10, tol=10.0)
        gs_clgst_cptp2 = self.runSilent(pygsti.contract, gs_lgst_target, "CPTP",verbosity=10, useDirectCP=False)
        gs_clgst_cptp3 = self.runSilent(pygsti.contract, gs_lgst_target, "CPTP",verbosity=10, tol=10.0, maxiter=0)
        gs_clgst_xp    = self.runSilent(pygsti.contract, gs_lgst_target, "XP", ds,verbosity=10, tol=10.0)
        gs_clgst_xptp  = self.runSilent(pygsti.contract, gs_lgst_target, "XPTP", ds,verbosity=10, tol=10.0)
        gs_clgst_vsp   = self.runSilent(pygsti.contract, gs_lgst_target, "vSPAM",verbosity=10, tol=10.0)
        gs_clgst_none  = self.runSilent(pygsti.contract, gs_lgst_target, "nothing",verbosity=10, tol=10.0)

          #test bad effect vector cases
        gs_bad_effect = gs_lgst_target.copy()
        gs_bad_effect.effects['E0'] = [100.0,0,0,0] # E eigvals all > 1.0
        self.runSilent(pygsti.contract, gs_bad_effect, "vSPAM",verbosity=10, tol=10.0)
        gs_bad_effect.effects['E0'] = [-100.0,0,0,0] # E eigvals all < 0
        self.runSilent(pygsti.contract, gs_bad_effect, "vSPAM",verbosity=10, tol=10.0)

        with self.assertRaises(ValueError):
            self.runSilent(pygsti.contract, gs_lgst_target, "foobar",verbosity=10, tol=10.0) #bad toWhat



        #More gauge optimizations
        TP_gauge_group = pygsti.obj.TPGaugeGroup(gs_lgst.dim)
        gs_lgst_target_cp  = self.runSilent(pygsti.optimize_gauge, gs_clgst_cptp,"target",targetGateset=self.gateset,
                                            constrainToCP=True,constrainToTP=True,constrainToValidSpam=True,verbosity=10) #DEPRECATED
        gs_lgst_target_cp  = self.runSilent(pygsti.gaugeopt_to_target, gs_clgst_cptp, self.gateset, 
                                            CPpenalty=1.0, gauge_group=TP_gauge_group, verbosity=10)

        gs_lgst.set_basis("gm",4) #so CPTP optimizations can work on gs_lgst
        gs_lgst_cptp       = self.runSilent(pygsti.optimize_gauge, gs_lgst,"CPTP",verbosity=10) #DEPRECATED
        gs_lgst_cptp       = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst, None,
                                            CPpenalty=1.0, TPpenalty=1.0, validSpamPenalty=1.0, verbosity=10)

        gs_lgst_cptp_tp    = self.runSilent(pygsti.optimize_gauge, gs_lgst,"CPTP",verbosity=10, constrainToTP=True) #DEPRECATED
        gs_lgst_cptp_tp    = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst, None,
                                            CPpenalty=1.0, TPpenalty=1.0, validSpamPenalty=1.0, gauge_group=TP_gauge_group, verbosity=10) #no point? (remove?)

        gs_lgst_tp         = self.runSilent(pygsti.optimize_gauge, gs_lgst,"TP",verbosity=10) #DEPRECATED
        gs_lgst_tp         = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst, None,
                                            TPpenalty=1.0, validSpamPenalty=1.0, verbosity=10)

        gs_lgst_tptarget   = self.runSilent(pygsti.optimize_gauge, gs_lgst,"TP and target",targetGateset=self.gateset,verbosity=10) #DEPRECATED
        gs_lgst_tptarget   = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst, self.gateset,
                                            TPpenalty=1.0, validSpamPenalty=1.0, verbosity=10)

        gs_lgst_cptptarget = self.runSilent(pygsti.optimize_gauge, gs_lgst,"CPTP and target",targetGateset=self.gateset,verbosity=10) #DEPRECATED
        gs_lgst_cptptarget = self.runSilent(pygsti.gaugeopt_to_target, gs_lgst, self.gateset,
                                            CPpenalty=1.0, TPpenalty=1.0, validSpamPenalty=1.0, verbosity=10)

        gs_lgst_cptptarget2= self.runSilent(pygsti.optimize_gauge, gs_lgst,"CPTP and target",targetGateset=self.gateset,
                                            verbosity=10, constrainToTP=True) #DEPRECATED
        gs_lgst_cptptarget2= self.runSilent(pygsti.gaugeopt_to_target, gs_lgst, self.gateset,
                                            CPpenalty=1.0, TPpenalty=1.0, validSpamPenalty=1.0, gauge_group=TP_gauge_group, verbosity=10) #no point? (remove?)

        gs_lgst_cd         = self.runSilent(pygsti.optimize_gauge, gs_lgst,"Completely Depolarized",targetGateset=self.gateset,verbosity=10) #DEPRECATED

        #TODO: check output lies in space desired

        # big kick that should land it outside XP, TP, etc, so contraction
        # routines are more tested
        gs_bigkick = gs_lgst_target.kick(absmag=1.0)
        gs_badspam = gs_bigkick.copy()
        gs_badspam.effects['E0'] =  np.array( [[2],[0],[0],[4]], 'd') #set a bad evec so vSPAM has to work...

        gs_clgst_tp    = self.runSilent(pygsti.contract,gs_bigkick, "TP", verbosity=10, tol=10.0)
        gs_clgst_cp    = self.runSilent(pygsti.contract,gs_bigkick, "CP", verbosity=10, tol=10.0)
        gs_clgst_cptp  = self.runSilent(pygsti.contract,gs_bigkick, "CPTP", verbosity=10, tol=10.0)
        gs_clgst_xp    = self.runSilent(pygsti.contract,gs_bigkick, "XP", ds, verbosity=10, tol=10.0)
        gs_clgst_xptp  = self.runSilent(pygsti.contract,gs_bigkick, "XPTP", ds, verbosity=10, tol=10.0)
        gs_clgst_vsp   = self.runSilent(pygsti.contract,gs_badspam, "vSPAM", verbosity=10, tol=10.0)
        gs_clgst_none  = self.runSilent(pygsti.contract,gs_bigkick, "nothing", verbosity=10, tol=10.0)

        #TODO: check output lies in space desired

        #Check Errors
        with self.assertRaises(ValueError):
            pygsti.optimize_gauge(gs_lgst,"FooBar",verbosity=0) # bad toGetTo argument

        with self.assertRaises(ValueError):
            pygsti.contract(gs_lgst_target, "FooBar",verbosity=0) # bad toWhat argument

        # No longer raise value error for failure to contract...
        #with self.assertRaises(ValueError):
        #    self.runSilent(pygsti.contract,gs_bigkick, "CP", verbosity=10,
        #                   maxiter=1) # fail to contract to CP


if __name__ == "__main__":
    unittest.main(verbosity=2)
