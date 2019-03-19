import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import unittest
import pygsti
from pygsti.construction import std1Q_XYI as std
from pygsti.baseobjs.basis import Basis
from pygsti.objects import Label as L

import numpy as np
from scipy import polyfit
import sys, os

from ..testutils import compare_files, temp_files
from .basecase import AlgorithmsBase

class TestCoreMethods(AlgorithmsBase):
    def test_gram(self):
        ds = self.ds
        #pygsti.construction.generate_fake_data(self.datagen_gateset, self.lgstStrings,
        #                                            nSamples=1000, sampleError='none')

        rank,evals,target_evals = pygsti.gram_rank_and_evals(ds, self.fiducials, self.fiducials, self.model)
        print("gram rank = ",rank)
        print("gram evals = ",evals)
        print("target gram evals = ",target_evals)

        with self.assertRaises(ValueError):
            pygsti.gram_rank_and_evals(ds, self.fiducials, self.fiducials, None) #no target

    def test_LGST(self):

        ds = self.ds
        #pygsti.construction.generate_fake_data(self.datagen_gateset, self.lgstStrings, nSamples=1000,
        #                                            sampleError='binomial', seed=None)

        print("GG0 = ",self.model.default_gauge_group)
        mdl_lgst = pygsti.do_lgst(ds, self.fiducials, self.fiducials, self.model, svdTruncateTo=4, verbosity=0)
        mdl_lgst_verb = self.runSilent(pygsti.do_lgst, ds, self.fiducials, self.fiducials, self.model, svdTruncateTo=4, verbosity=10)
        self.assertAlmostEqual(mdl_lgst.frobeniusdist(mdl_lgst_verb),0)

        print("GG = ",mdl_lgst.default_gauge_group)
        mdl_lgst_go = pygsti.gaugeopt_to_target(mdl_lgst,self.model, {'spam':1.0, 'gates': 1.0}, checkJac=True)
        mdl_clgst = pygsti.contract(mdl_lgst_go, "CPTP")

        # RUN BELOW LINES TO SEED SAVED GATESET FILES
        if os.environ.get('PYGSTI_REGEN_REF_FILES','no').lower() in ("yes","1","true"):
            pygsti.io.write_model(mdl_lgst,compare_files + "/lgst.model", "Saved LGST Model before gauge optimization")
            pygsti.io.write_model(mdl_lgst_go,compare_files + "/lgst_go.model", "Saved LGST Model after gauge optimization")
            pygsti.io.write_model(mdl_clgst,compare_files + "/clgst.model", "Saved LGST Model after G.O. and CPTP contraction")

        mdl_lgst_compare = pygsti.io.load_model(compare_files + "/lgst.model")
        mdl_lgst_go_compare = pygsti.io.load_model(compare_files + "/lgst_go.model")
        mdl_clgst_compare = pygsti.io.load_model(compare_files + "/clgst.model")

        self.assertAlmostEqual( mdl_lgst.frobeniusdist(mdl_lgst_compare), 0, places=5)
        self.assertAlmostEqual( mdl_lgst_go.frobeniusdist(mdl_lgst_go_compare), 0, places=5)
        self.assertAlmostEqual( mdl_clgst.frobeniusdist(mdl_clgst_compare), 0, places=5)

        #Check for error conditions
        with self.assertRaises(ValueError):
            mdl_lgst = pygsti.do_lgst(ds, self.fiducials, self.fiducials, None, svdTruncateTo=4, verbosity=0) #no target model

        with self.assertRaises(ValueError):
            mdl_lgst = pygsti.do_lgst(ds, self.fiducials, self.fiducials, None, opLabels=list(self.model.operations.keys()),
                                     svdTruncateTo=4, verbosity=0) #no spam dict

        #No need for identity vector anymore
        #with self.assertRaises(ValueError):
        #    mdl_lgst = pygsti.do_lgst(ds, self.fiducials, self.fiducials, None, opLabels=list(self.model.operations.keys()),
        #                             spamDict=self.model.get_reverse_spam_defs(),
        #                             svdTruncateTo=4, verbosity=0) #no identity vector

        with self.assertRaises(ValueError):
            bad_fids =pygsti.construction.circuit_list([('Gx',),('Gx',),('Gx',),('Gx',)])
            mdl_lgst = pygsti.do_lgst(ds, bad_fids, bad_fids, self.model, svdTruncateTo=4, verbosity=0) # bad fiducials (rank deficient)


        with self.assertRaises(KeyError): # AB-matrix construction error
            incomplete_strings = self.lgstStrings[5:] #drop first 5 strings...
            bad_ds = pygsti.construction.generate_fake_data(
                self.datagen_gateset, incomplete_strings,
                nSamples=10, sampleError='none')
            mdl_lgst = pygsti.do_lgst(bad_ds, self.fiducials, self.fiducials, self.model,
                                     svdTruncateTo=4, verbosity=0)
                      # incomplete dataset

        with self.assertRaises(KeyError): # X-matrix construction error
            incomplete_strings = self.lgstStrings[:-5] #drop last 5 strings...
            bad_ds = pygsti.construction.generate_fake_data(
                self.datagen_gateset, incomplete_strings,
                nSamples=10, sampleError='none')
            mdl_lgst = pygsti.do_lgst(bad_ds, self.fiducials, self.fiducials, self.model,
                                     svdTruncateTo=4, verbosity=0)
                      # incomplete dataset

        #Deprecated / removed:
        #LGST on an "old-style" model
        #old_style_gateset = pygsti.construction.build_explicit_model(
        #    [2], [('Q0',)],['Gi','Gx','Gy'],
        #    [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"],
        #    prepLabels=["rho0"], prepExpressions=["0"],
        #    effectLabels=["E0"], effectExpressions=["0"],
        #    spamdefs={'0': ('rho0','E0'),
        #              '1': ('remainder','remainder') } )
        #mdl_lgst = pygsti.do_lgst(ds, self.fiducials, self.fiducials, old_style_gateset,
        #                         svdTruncateTo=4, verbosity=0)





    def test_LGST_no_sample_error(self):
        #change rep-count type so dataset can hold fractional counts for sampleError = 'none'
        oldType = pygsti.objects.dataset.Repcount_type
        pygsti.objects.dataset.Repcount_type = np.float64
        ds = pygsti.construction.generate_fake_data(self.datagen_gateset, self.lgstStrings,
                                                    nSamples=10000, sampleError='none')
        pygsti.objects.dataset.Repcount_type = oldType
        
        mdl_lgst = pygsti.do_lgst(ds, self.fiducials, self.fiducials, self.model, svdTruncateTo=4, verbosity=0)
        print("DATAGEN:")
        print(self.datagen_gateset)
        print("\nLGST RAW:")
        print(mdl_lgst)
        mdl_lgst = pygsti.gaugeopt_to_target(mdl_lgst,self.datagen_gateset, {'spam':1.0, 'gates': 1.0}, checkJac=False)
        print("\nAfter gauge opt:")
        print(mdl_lgst)
        print(mdl_lgst.strdiff(self.datagen_gateset))
        self.assertAlmostEqual( mdl_lgst.frobeniusdist(self.datagen_gateset), 0, places=4)


    def test_eLGST(self):

        ds = self.ds
        #pygsti.construction.generate_fake_data(self.datagen_gateset, self.lsgstStrings[-1],
        #                                            nSamples=1000,sampleError='binomial', seed=100)

        assert(pygsti.obj.Model._pcheck)
        mdl_lgst = pygsti.do_lgst(ds, self.fiducials, self.fiducials, self.model, svdTruncateTo=4, verbosity=0)
        #mdl_lgst._check_paramvec() #will fail, but OK, since paramvec is computed only when *needed* now
        mdl_lgst_go = pygsti.gaugeopt_to_target(mdl_lgst,self.model, {'spam':1.0, 'gates': 1.0}, checkJac=True)
        mdl_lgst_go._check_paramvec()
        mdl_clgst = pygsti.contract(mdl_lgst_go, "CPTP")
        mdl_clgst.to_vector() # to make sure we're in sync
        mdl_clgst._check_paramvec()
        self.model._check_paramvec()
        
        _,mdl_single_exlgst = pygsti.do_exlgst(ds, mdl_clgst, self.elgstStrings[0], self.fiducials, self.fiducials,
                                            self.model, regularizeFactor=1e-3, svdTruncateTo=4,
                                              verbosity=0)
        mdl_single_exlgst._check_paramvec()

        _,mdl_single_exlgst_verb = self.runSilent(pygsti.do_exlgst, ds, mdl_clgst, self.elgstStrings[0], self.fiducials, self.fiducials,
                                               self.model, regularizeFactor=1e-3, svdTruncateTo=4,
                                               verbosity=10)
        mdl_single_exlgst_verb._check_paramvec()
        
        self.assertAlmostEqual(mdl_single_exlgst.frobeniusdist(mdl_single_exlgst_verb),0)

        mdl_exlgst = pygsti.do_iterative_exlgst(ds, mdl_clgst, self.fiducials, self.fiducials, self.elgstStrings,
                                               targetModel=self.model, svdTruncateTo=4, verbosity=0)

        all_minErrs, all_gs_exlgst_tups = pygsti.do_iterative_exlgst(
            ds, mdl_clgst, self.fiducials, self.fiducials, [ [mdl.tup for mdl in gsList] for gsList in self.elgstStrings],
            targetModel=self.model, svdTruncateTo=4, verbosity=0, returnAll=True, returnErrorVec=True)

        mdl_exlgst_verb = self.runSilent(pygsti.do_iterative_exlgst, ds, mdl_clgst, self.fiducials, self.fiducials, self.elgstStrings,
                                        targetModel=self.model, svdTruncateTo=4, verbosity=10)
        mdl_exlgst_reg = pygsti.do_iterative_exlgst(ds, mdl_clgst, self.fiducials, self.fiducials, self.elgstStrings,
                                               targetModel=self.model, svdTruncateTo=4, verbosity=0,
                                               regularizeFactor=10)
        self.assertAlmostEqual(mdl_exlgst.frobeniusdist(mdl_exlgst_verb),0)
        self.assertAlmostEqual(mdl_exlgst.frobeniusdist(all_gs_exlgst_tups[-1]),0)


        #Run internal checks on less max-L values (so it doesn't take forever)
        mdl_exlgst_chk = pygsti.do_iterative_exlgst(ds, mdl_clgst, self.fiducials, self.fiducials, self.elgstStrings[0:2],
                                                   targetModel=self.model, svdTruncateTo=4, verbosity=0,
                                                   check_jacobian=True)
        mdl_exlgst_chk_verb = self.runSilent(pygsti.do_iterative_exlgst,ds, mdl_clgst, self.fiducials, self.fiducials, self.elgstStrings[0:2],
                                                   targetModel=self.model, svdTruncateTo=4, verbosity=10,
                                                   check_jacobian=True)

        # RUN BELOW LINES TO SEED SAVED GATESET FILES
        if os.environ.get('PYGSTI_REGEN_REF_FILES','no').lower() in ("yes","1","true"):
            pygsti.io.write_model(mdl_exlgst,compare_files + "/exlgst.model", "Saved Extended-LGST (eLGST) Model")
            pygsti.io.write_model(mdl_exlgst_reg,compare_files + "/exlgst_reg.model", "Saved Extended-LGST (eLGST) Model w/regularization")

        mdl_exlgst_compare = pygsti.io.load_model(compare_files + "/exlgst.model")
        mdl_exlgst_reg_compare = pygsti.io.load_model(compare_files + "/exlgst_reg.model")
        mdl_exlgst.set_all_parameterizations("full") # b/c ex-LGST sets spam to StaticSPAMVec objects (b/c they're not optimized)
        mdl_exlgst_reg.set_all_parameterizations("full") # b/c ex-LGST sets spam to StaticSPAMVec objects (b/c they're not optimized)
        mdl_exlgst_go = pygsti.gaugeopt_to_target(mdl_exlgst,mdl_exlgst_compare, {'spam':1.0 }, checkJac=True)
        mdl_exlgst_reg_go = pygsti.gaugeopt_to_target(mdl_exlgst_reg,mdl_exlgst_reg_compare, {'spam':1.0 }, checkJac=True)

        #self.assertAlmostEqual( mdl_exlgst_go.frobeniusdist(mdl_exlgst_compare), 0, places=5)
        #self.assertAlmostEqual( mdl_exlgst_reg_go.frobeniusdist(mdl_exlgst_reg_compare), 0, places=5)


    def test_MC2GST(self):

        ds = self.ds
        #pygsti.construction.generate_fake_data(self.datagen_gateset, self.lsgstStrings[-1],
        #                                            nSamples=1000, sampleError='binomial', seed=100)

        mdl_lgst = pygsti.do_lgst(ds, self.fiducials, self.fiducials, self.model, svdTruncateTo=4, verbosity=0)
        mdl_lgst_go = pygsti.gaugeopt_to_target(mdl_lgst,self.model, {'spam':1.0, 'gates': 1.0}, checkJac=True)
        mdl_clgst = pygsti.contract(mdl_lgst_go, "CPTP")
        CM = pygsti.baseobjs.profiler._get_mem_usage()

        mdl_single_lsgst = pygsti.do_mc2gst(ds, mdl_clgst, self.lsgstStrings[0], minProbClipForWeighting=1e-4,
                                           probClipInterval=(-1e6,1e6), regularizeFactor=1e-3,
                                           verbosity=0) #uses regularizeFactor

        mdl_single_lsgst_cp = pygsti.do_mc2gst(ds, mdl_clgst, self.lsgstStrings[0], minProbClipForWeighting=1e-4,
                                           probClipInterval=(-1e6,1e6), cptp_penalty_factor=1.0,
                                           verbosity=0) #uses cptp_penalty_factor

        mdl_single_lsgst_sp = pygsti.do_mc2gst(ds, mdl_clgst, self.lsgstStrings[0], minProbClipForWeighting=1e-4,
                                              probClipInterval=(-1e6,1e6), spam_penalty_factor=1.0,
                                              verbosity=0) #uses spam_penalty_factor

        mdl_single_lsgst_cpsp = pygsti.do_mc2gst(ds, mdl_clgst, self.lsgstStrings[0], minProbClipForWeighting=1e-4,
                                                probClipInterval=(-1e6,1e6), cptp_penalty_factor=1.0,
                                                spam_penalty_factor=1.0, verbosity=0) #uses both penalty factors

        mdl_single_lsgst_cpsp = self.runSilent(pygsti.do_mc2gst, ds, mdl_clgst, self.lsgstStrings[0], minProbClipForWeighting=1e-4,
                                              probClipInterval=(-1e6,1e6), cptp_penalty_factor=1.0,
                                              spam_penalty_factor=1.0, verbosity=10) #uses both penalty factors w/verbosity high
        mdl_single_lsgst_cp = self.runSilent(pygsti.do_mc2gst, ds, mdl_clgst, self.lsgstStrings[0], minProbClipForWeighting=1e-4,
                                            probClipInterval=(-1e6,1e6), cptp_penalty_factor=1.0,
                                            verbosity=10) #uses cptp_penalty_factor w/verbosity high
        mdl_single_lsgst_sp = self.runSilent(pygsti.do_mc2gst, ds, mdl_clgst, self.lsgstStrings[0], minProbClipForWeighting=1e-4,
                                            probClipInterval=(-1e6,1e6), spam_penalty_factor=1.0,
                                            verbosity=10) #uses spam_penalty_factor w/verbosity high


        
        mdl_lsgst = pygsti.do_iterative_mc2gst(ds, mdl_clgst, self.lsgstStrings, verbosity=0,
                                             minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6),
                                             memLimit=CM + 1024**3)
        all_minErrs, all_gs_lsgst_tups = pygsti.do_iterative_mc2gst(
            ds, mdl_clgst, [ [mdl.tup for mdl in gsList] for gsList in self.lsgstStrings],
            minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6), returnAll=True, returnErrorVec=True)
        mdl_lsgst_verb = self.runSilent(pygsti.do_iterative_mc2gst, ds, mdl_clgst, self.lsgstStrings, verbosity=10,
                                             minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6),
                                             memLimit=CM + 1024**3)
        mdl_lsgst_reg = self.runSilent(pygsti.do_iterative_mc2gst,ds, mdl_clgst,
                                      self.lsgstStrings, verbosity=10,
                                      minProbClipForWeighting=1e-6,
                                      probClipInterval=(-1e6,1e6),
                                      regularizeFactor=10, memLimit=CM + 1024**3)
        self.assertAlmostEqual(mdl_lsgst.frobeniusdist(mdl_lsgst_verb),0)
        self.assertAlmostEqual(mdl_lsgst.frobeniusdist(all_gs_lsgst_tups[-1]),0)


        #Run internal checks on less max-L values (so it doesn't take forever)
        mdl_lsgst_chk = pygsti.do_iterative_mc2gst(ds, mdl_clgst, self.lsgstStrings[0:2], verbosity=0,
                                                 minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6),
                                                 check=True, check_jacobian=True)
        mdl_lsgst_chk_verb = self.runSilent(pygsti.do_iterative_mc2gst, ds, mdl_clgst, self.lsgstStrings[0:2], verbosity=10,
                                                      minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6),
                                                      check=True, check_jacobian=True, memLimit=CM + 1024**3)

        #Other option variations - just make sure they run at this point
        mdl_lsgst_chk_opts = pygsti.do_iterative_mc2gst(ds, mdl_clgst, self.lsgstStrings[0:2], verbosity=0,
                                                      minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6),
                                                      useFreqWeightedChiSq=True, circuitSetLabels=["Set1","Set2"],
                                                      circuitWeightsDict={ ('Gx',): 2.0 } )

        aliased_list = [ pygsti.obj.Circuit( [ (x if x != L("Gx") else L("GA1")) for x in mdl]) for mdl in self.lsgstStrings[0] ]
        mdl_withA1 = mdl_clgst.copy(); mdl_withA1.operations["GA1"] = mdl_clgst.operations["Gx"]
        del mdl_withA1.operations["Gx"] # otherwise mdl_withA1 will have Gx params that we have no knowledge of!
        mdl_lsgst_chk_opts2 = pygsti.do_mc2gst(ds, mdl_withA1, aliased_list, minProbClipForWeighting=1e-6,
                                              probClipInterval=(-1e2,1e2), verbosity=10,
                                              opLabelAliases={ L('GA1'): (L('Gx'),) })

        
        #Check with small but ok memlimit -- not anymore since new mem estimation uses current memory, making this non-robust
        #self.runSilent(pygsti.do_mc2gst,ds, mdl_clgst, self.lsgstStrings[0], minProbClipForWeighting=1e-6,
        #                 probClipInterval=(-1e6,1e6), regularizeFactor=1e-3,
        #                 verbosity=10, memLimit=CM + 1024**3)


        #Check errors:
        with self.assertRaises(MemoryError):
            pygsti.do_mc2gst(ds, mdl_clgst, self.lsgstStrings[0], minProbClipForWeighting=1e-6,
                             probClipInterval=(-1e6,1e6), regularizeFactor=1e-3,
                             verbosity=0, memLimit=1)

        with self.assertRaises(AssertionError):
            pygsti.do_mc2gst(ds, mdl_clgst, self.lsgstStrings[0], minProbClipForWeighting=1e-6,
                             probClipInterval=(-1e6,1e6), regularizeFactor=1e-3,
                             verbosity=0, cptp_penalty_factor=1.0) #can't specify both cptp_penalty_factor and regularizeFactor


        # RUN BELOW LINES TO SEED SAVED GATESET FILES
        if os.environ.get('PYGSTI_REGEN_REF_FILES','no').lower() in ("yes","1","true"):
            pygsti.io.write_model(mdl_lsgst,compare_files + "/lsgst.model", "Saved LSGST Model")
            pygsti.io.write_model(mdl_lsgst_reg,compare_files + "/lsgst_reg.model", "Saved LSGST Model w/Regularization")

        mdl_lsgst_compare = pygsti.io.load_model(compare_files + "/lsgst.model")
        mdl_lsgst_reg_compare = pygsti.io.load_model(compare_files + "/lsgst_reg.model")

        mdl_lsgst_go = pygsti.gaugeopt_to_target(mdl_lsgst, mdl_lsgst_compare, {'spam':1.0}, checkJac=True)

        mdl_lsgst_reg_go = pygsti.gaugeopt_to_target(mdl_lsgst_reg, mdl_lsgst_reg_compare, {'spam':1.0}, checkJac=True)

        self.assertAlmostEqual( mdl_lsgst_go.frobeniusdist(mdl_lsgst_compare), 0, places=4)
        self.assertAlmostEqual( mdl_lsgst_reg_go.frobeniusdist(mdl_lsgst_reg_compare), 0, places=4)

        # RUN BELOW LINES TO SEED SAVED GATESET FILES
        if os.environ.get('PYGSTI_REGEN_REF_FILES','no').lower() in ("yes","1","true"):
            mdl_lsgst_go = pygsti.gaugeopt_to_target(mdl_lsgst, self.model, {'spam':1.0})
            pygsti.io.write_model(mdl_lsgst_go,compare_files + "/analysis.model", "Saved LSGST Analysis Model")
            print("DEBUG: analysis.model = "); print(mdl_lgst_go)


    def test_MLGST(self):

        ds = self.ds
        #pygsti.construction.generate_fake_data(self.datagen_gateset, self.lsgstStrings[-1],
        #                                            nSamples=1000, sampleError='binomial', seed=100)

        mdl_lgst = pygsti.do_lgst(ds, self.fiducials, self.fiducials, self.model, svdTruncateTo=4, verbosity=0)
        mdl_lgst_go = pygsti.gaugeopt_to_target(mdl_lgst,self.model, {'spam':1.0, 'gates': 1.0}, checkJac=True)
        mdl_clgst = pygsti.contract(mdl_lgst_go, "CPTP") 
        mdl_clgst = mdl_clgst.depolarize(op_noise=0.02, spam_noise=0.02) # just to avoid infinity objective funct & jacs below
        CM = pygsti.baseobjs.profiler._get_mem_usage()

        mdl_single_mlgst = pygsti.do_mlgst(ds, mdl_clgst, self.lsgstStrings[0], minProbClip=1e-4,
                                          probClipInterval=(-1e2,1e2), verbosity=0)

        #this test often gives an assetion error "finite Jacobian has inf norm!" on Travis CI Python 3 case
        try:
            mdl_single_mlgst_cpsp = pygsti.do_mlgst(ds, mdl_clgst, self.lsgstStrings[0], minProbClip=1e-4,
                                                  probClipInterval=(-1e2,1e2), cptp_penalty_factor=1.0,
                                                  spam_penalty_factor=1.0, verbosity=10) #uses both penalty factors w/verbosity > 0
        except ValueError: pass # ignore when assertions in customlm.py are disabled
        except AssertionError:
            pass # just ignore for now.  FUTURE: see what we can do in custom LM about scaling large jacobians...
        
        try:
            mdl_single_mlgst_cp = pygsti.do_mlgst(ds, mdl_clgst, self.lsgstStrings[0], minProbClip=1e-4,
                                                  probClipInterval=(-1e2,1e2), cptp_penalty_factor=1.0,
                                                  verbosity=10)
        except ValueError: pass # ignore when assertions in customlm.py are disabled
        except AssertionError:
            pass # just ignore for now.  FUTURE: see what we can do in custom LM about scaling large jacobians...
        
        try:
            mdl_single_mlgst_sp = pygsti.do_mlgst(ds, mdl_clgst, self.lsgstStrings[0], minProbClip=1e-4,
                                                  probClipInterval=(-1e2,1e2), spam_penalty_factor=1.0,
                                                  verbosity=10)
        except ValueError: pass # ignore when assertions in customlm.py are disabled
        except AssertionError:
            pass # just ignore for now.  FUTURE: see what we can do in custom LM about scaling large jacobians...
            

        mdl_mlegst = pygsti.do_iterative_mlgst(ds, mdl_clgst, self.lsgstStrings, verbosity=0,
                                               minProbClip=1e-4, probClipInterval=(-1e2,1e2),
                                               memLimit=CM + 1024**3)
        maxLogL, all_gs_mlegst_tups = pygsti.do_iterative_mlgst(
            ds, mdl_clgst, [ [mdl.tup for mdl in gsList] for gsList in self.lsgstStrings],
            minProbClip=1e-4, probClipInterval=(-1e2,1e2), returnAll=True, returnMaxLogL=True)

        mdl_mlegst_verb = self.runSilent(pygsti.do_iterative_mlgst, ds, mdl_clgst, self.lsgstStrings, verbosity=10,
                                             minProbClip=1e-4, probClipInterval=(-1e2,1e2),
                                             memLimit=CM + 1024**3)
        self.assertAlmostEqual(mdl_mlegst.frobeniusdist(mdl_mlegst_verb),0, places=5)
        self.assertAlmostEqual(mdl_mlegst.frobeniusdist(all_gs_mlegst_tups[-1]),0,places=5)


        #Run internal checks on less max-L values (so it doesn't take forever)
        mdl_mlegst_chk = pygsti.do_iterative_mlgst(ds, mdl_clgst, self.lsgstStrings[0:2], verbosity=0,
                                                 minProbClip=1e-4, probClipInterval=(-1e2,1e2),
                                                 check=True)

        #Other option variations - just make sure they run at this point
        mdl_mlegst_chk_opts = pygsti.do_iterative_mlgst(ds, mdl_clgst, self.lsgstStrings[0:2], verbosity=0,
                                                       minProbClip=1e-4, probClipInterval=(-1e2,1e2),
                                                       circuitSetLabels=["Set1","Set2"], useFreqWeightedChiSq=True,
                                                       circuitWeightsDict={ (L('Gx'),): 2.0 } )

        aliased_list = [ pygsti.obj.Circuit( [ (x if x != L("Gx") else L("GA1")) for x in mdl]) for mdl in self.lsgstStrings[0] ]
        mdl_withA1 = mdl_clgst.copy(); mdl_withA1.operations["GA1"] = mdl_clgst.operations["Gx"]
        del mdl_withA1.operations["Gx"] # otherwise mdl_withA1 will have Gx params that we have no knowledge of!
        mdl_mlegst_chk_opts2 = pygsti.do_mlgst(ds, mdl_withA1, aliased_list, minProbClip=1e-4,
                                              probClipInterval=(-1e2,1e2), verbosity=10,
                                              opLabelAliases={ L('GA1'): (L('Gx'),) })

        #Other option variations - just make sure they run at this point
        mdl_mlegst_chk_opts3 = pygsti.do_iterative_mlgst(ds, mdl_clgst, self.lsgstStrings[0:2], verbosity=0,
                                                       minProbClip=1e-4, probClipInterval=(-1e2,1e2),
                                                       circuitSetLabels=["Set1","Set2"], useFreqWeightedChiSq=True,
                                                        circuitWeightsDict={ (L('Gx'),): 2.0 }, alwaysPerformMLE=True )

        #Forcing function used by linear response error bars
        forcingfn_grad = np.ones((1,mdl_clgst.num_params()), 'd')
        mdl_lsgst_chk_opts3 = pygsti.algorithms.core._do_mlgst_base(
            ds, mdl_clgst, self.lsgstStrings[0], verbosity=0,
            minProbClip=1e-4, probClipInterval=(-1e2,1e2),
            forcefn_grad=forcingfn_grad)
        mdl_lsgst_chk_opts4 = pygsti.algorithms.core._do_mlgst_base(
            ds, mdl_clgst, self.lsgstStrings[0], verbosity=0, poissonPicture=False, 
            minProbClip=1e-4, probClipInterval=(-1e2,1e2),
            forcefn_grad=forcingfn_grad) # non-poisson picture

        #Check with small but ok memlimit -- not anymore since new mem estimation uses current memory, making this non-robust
        #self.runSilent(pygsti.do_mlgst, ds, mdl_clgst, self.lsgstStrings[0], minProbClip=1e-6,
        #                probClipInterval=(-1e2,1e2), verbosity=4, memLimit=curMem+8500000) #invoke memory control

        #non-Poisson picture - should use (-1,-1) model for consistency?
        pygsti.do_mlgst(ds, mdl_clgst, self.lsgstStrings[0], minProbClip=1e-4,
                        probClipInterval=(-1e2,1e2), verbosity=0, poissonPicture=False)
        try:
            pygsti.do_mlgst(ds, mdl_clgst, self.lsgstStrings[0], minProbClip=1e-1, # 1e-1 b/c get inf Jacobians...
                            probClipInterval=(-1e2,1e2), verbosity=0, poissonPicture=False,
                            spam_penalty_factor=1.0, cptp_penalty_factor=1.0)
        except ValueError: pass # ignore when assertions in customlm.py are disabled
        except AssertionError:
            pass # just ignore for now.  FUTURE: see what we can do in custom LM about scaling large jacobians...



        #Check errors:
        with self.assertRaises(MemoryError):
            pygsti.do_mlgst(ds, mdl_clgst, self.lsgstStrings[0], minProbClip=1e-4,
                            probClipInterval=(-1e2,1e2),verbosity=0, memLimit=1)


        # RUN BELOW LINES TO SEED SAVED GATESET FILES
        if os.environ.get('PYGSTI_REGEN_REF_FILES','no').lower() in ("yes","1","true"):
            pygsti.io.write_model(mdl_mlegst,compare_files + "/mle_gst.model", "Saved MLE-GST Model")

        mdl_mle_compare = pygsti.io.load_model(compare_files + "/mle_gst.model")
        mdl_mlegst_go = pygsti.gaugeopt_to_target(mdl_mlegst, mdl_mle_compare, {'spam':1.0}, checkJac=True)

        self.assertAlmostEqual( mdl_mlegst_go.frobeniusdist(mdl_mle_compare), 0, places=4)

    def test_LGST_1overSqrtN_dependence(self):
        my_datagen_gateset = self.model.depolarize(op_noise=0.05, spam_noise=0)
        # !!don't depolarize spam or 1/sqrt(N) dependence saturates!!

        nSamplesList = np.array([ 16, 128, 1024, 8192 ])
        diffs = []
        for nSamples in nSamplesList:
            ds = pygsti.construction.generate_fake_data(my_datagen_gateset, self.lgstStrings, nSamples,
                                                        sampleError='binomial', seed=100)
            mdl_lgst = pygsti.do_lgst(ds, self.fiducials, self.fiducials, self.model, svdTruncateTo=4, verbosity=0)
            mdl_lgst_go = pygsti.gaugeopt_to_target(mdl_lgst, my_datagen_gateset, {'spam':1.0, 'gate': 1.0}, checkJac=True)
            diffs.append( my_datagen_gateset.frobeniusdist(mdl_lgst_go) )

        diffs = np.array(diffs, 'd')
        a, b = polyfit(np.log10(nSamplesList), np.log10(diffs), deg=1)
        #print "\n",nSamplesList; print diffs; print a #DEBUG
        self.assertLess( a+0.5, 0.05 )


    def test_model_selection(self):

        ds = self.ds
        #pygsti.construction.generate_fake_data(self.datagen_gateset, self.lsgstStrings[-1],
        #                                            nSamples=1000,sampleError='binomial', seed=100)


        mdl_lgst4 = pygsti.do_lgst(ds, self.fiducials, self.fiducials, self.model, svdTruncateTo=4, verbosity=0)
        mdl_lgst6 = pygsti.do_lgst(ds, self.fiducials, self.fiducials, self.model, svdTruncateTo=6, verbosity=0)
        sys.stdout.flush()

        self.runSilent(pygsti.do_lgst, ds, self.fiducials, self.fiducials, self.model, svdTruncateTo=6, verbosity=4) # test verbose prints

        chiSq4 = pygsti.chi2(mdl_lgst4, ds, self.lgstStrings, minProbClipForWeighting=1e-4)
        chiSq6 = pygsti.chi2(mdl_lgst6, ds, self.lgstStrings, minProbClipForWeighting=1e-4)

        print("LGST dim=4 chiSq = ",chiSq4)
        print("LGST dim=6 chiSq = ",chiSq6)
        #self.assertAlmostEqual(chiSq4, 174.061524953) #429.271983052)
        #self.assertAlmostEqual(chiSq6, 267012993.861, places=1) #1337.74222467) #Why is this so large??? -- DEBUG later

        # Least squares GST with model selection
        mdl_lsgst = self.runSilent(pygsti.do_iterative_mc2gst_with_model_selection, ds, mdl_lgst4, 1, self.lsgstStrings[0:3],
                                  verbosity=10, minProbClipForWeighting=1e-3, probClipInterval=(-1e5,1e5))

        # Run again with other parameters
        tuple_strings = [ list(map(tuple, gsList)) for gsList in self.lsgstStrings[0:3] ] #to test tuple argument
        errorVecs, mdl_lsgst_wts = self.runSilent(pygsti.do_iterative_mc2gst_with_model_selection, ds, mdl_lgst4,
                                                 1, tuple_strings, verbosity=10, minProbClipForWeighting=1e-3,
                                                 probClipInterval=(-1e5,1e5), circuitWeightsDict={ ('Gx',): 2.0 },
                                                 returnAll=True, returnErrorVec=True)

        # Do non-iterative to cover Circuit->tuple conversion
        mdl_non_iterative = self.runSilent( pygsti.do_mc2gst_with_model_selection, ds,
                                           mdl_lgst4, 1, self.lsgstStrings[0],
                                           verbosity=10, probClipInterval=(-1e5,1e5) )


        # RUN BELOW LINES TO SEED SAVED GATESET FILES
        if os.environ.get('PYGSTI_REGEN_REF_FILES','no').lower() in ("yes","1","true"):
            pygsti.io.write_model(mdl_lsgst,compare_files + "/lsgstMS.model", "Saved LSGST Model with model selection")

        mdl_lsgst_compare = pygsti.io.load_model(compare_files + "/lsgstMS.model")
        mdl_lsgst_go = pygsti.gaugeopt_to_target(mdl_lsgst, mdl_lsgst_compare, {'spam':1.0}, checkJac=True)
        self.assertAlmostEqual( mdl_lsgst_go.frobeniusdist(mdl_lsgst_compare), 0, places=4)

    def test_miscellaneous(self):
        ds = self.ds
        #pygsti.construction.generate_fake_data(self.datagen_gateset, self.lgstStrings,
        #                                            nSamples=1000, sampleError='none')

        strs = pygsti.construction.list_strings_lgst_can_estimate(ds, self.fiducials, self.fiducials)

        self.runSilent(self.model.print_info) #just make sure it works

        #test boundary case:
        gate2Q = np.identity(16,'d')
        with self.assertRaises(ValueError):
            pygsti.alg.find_closest_unitary_opmx(gate2Q) #doesn't work for > 1 qubits

if __name__ == "__main__":
    unittest.main(verbosity=2)
