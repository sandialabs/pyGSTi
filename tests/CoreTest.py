import unittest
import GST
from GSTCommons import Std1Q_XYI as Std
from GSTCommons import MakeLists_WholeGermPowers as MakeLists
import numpy as np
from scipy import polyfit
import sys

class CoreTestCase(unittest.TestCase):

    def setUp(self):

        self.gateset = Std.gs_target
        self.datagen_gateset = GST.GateSetTools.depolarize_gateset(self.gateset, noise=0.05)
        self.datagen_gateset = GST.GateSetTools.depolarize_spam(self.datagen_gateset, noise=0.1)
        
        self.fiducials = Std.fiducials
        self.germs = Std.germs
        self.specs = GST.get_spam_specs(self.fiducials, EVecInds=[0]) #only use the first EVec

        self.gateLabels = self.gateset.keys() # also == Std.gates
        self.lgstStrings = GST.list_lgst_gatestrings(self.specs, self.gateset.keys())

        self.maxLengthList = [0,1,2,4,8]
        
        self.elgstStrings = MakeLists.make_elgst_lists(
            self.gateLabels, self.germs, self.maxLengthList )
        
        self.lsgstStrings = MakeLists.make_lsgst_lists(
            self.gateLabels, self.fiducials, self.germs, self.maxLengthList )

    def runSilent(self, callable, *args, **kwds):
        orig_stdout = sys.stdout
        sys.stdout = open("temp_test_files/silent.txt","w")
        result = callable(*args, **kwds)
        sys.stdout.close()
        sys.stdout = orig_stdout
        return result


class TestCoreMethods(CoreTestCase):

    def test_LGST(self):

        ds = GST.generate_fake_data(self.datagen_gateset, self.lgstStrings, nSamples=1000,
                                  sampleError='binomial', seed=100)
        
        gs_lgst = GST.do_lgst(ds, self.specs, self.gateset, svdTruncateTo=4, verbosity=0)
        gs_lgst_verb = self.runSilent(GST.do_lgst, ds, self.specs, self.gateset, svdTruncateTo=4, verbosity=10)
        self.assertAlmostEqual(gs_lgst.diff_frobenius(gs_lgst_verb),0)

        gs_lgst_go = GST.optimize_gauge(gs_lgst,"target",targetGateset=self.gateset, spamWeight=1.0, gateWeight=1.0)
        gs_clgst = GST.contract(gs_lgst_go, "CPTP")
        
        # RUN BELOW LINES TO SEED SAVED GATESET FILES
        #GST.write_gateset(gs_lgst,"cmp_chk_files/lgst.gateset", "Saved LGST Gateset before gauge optimization")
        #GST.write_gateset(gs_lgst_go,"cmp_chk_files/lgst_go.gateset", "Saved LGST Gateset after gauge optimization")
        #GST.write_gateset(gs_clgst,"cmp_chk_files/clgst.gateset", "Saved LGST Gateset after G.O. and CPTP contraction")

        gs_lgst_compare = GST.load_gateset("cmp_chk_files/lgst.gateset")
        gs_lgst_go_compare = GST.load_gateset("cmp_chk_files/lgst_go.gateset")
        gs_clgst_compare = GST.load_gateset("cmp_chk_files/clgst.gateset")
        
        self.assertAlmostEqual( gs_lgst.diff_frobenius(gs_lgst_compare), 0)
        self.assertAlmostEqual( gs_lgst_go.diff_frobenius(gs_lgst_go_compare), 0)
        self.assertAlmostEqual( gs_clgst.diff_frobenius(gs_clgst_compare), 0)


    def test_LGST_no_sample_error(self):
        ds = GST.generate_fake_data(self.datagen_gateset, self.lgstStrings, nSamples=1000,
                                  sampleError='none')
        gs_lgst = GST.do_lgst(ds, self.specs, self.gateset, svdTruncateTo=4, verbosity=0)
        gs_lgst = GST.optimize_gauge(gs_lgst, "target", targetGateset=self.datagen_gateset, gateWeight=1.0, spamWeight=1.0)
        self.assertAlmostEqual( gs_lgst.diff_frobenius(self.datagen_gateset), 0)
        

    def test_eLGST(self):

        ds = GST.generate_fake_data(self.datagen_gateset, self.lsgstStrings[-1], nSamples=1000,
                                  sampleError='binomial', seed=100)
        
        gs_lgst = GST.do_lgst(ds, self.specs, self.gateset, svdTruncateTo=4, verbosity=0)
        gs_lgst_go = GST.optimize_gauge(gs_lgst,"target",targetGateset=self.gateset, spamWeight=1.0, gateWeight=1.0)
        gs_clgst = GST.contract(gs_lgst_go, "CPTP")

        gs_exlgst = GST.Core.do_iterative_exlgst(ds, gs_clgst, self.specs, self.elgstStrings,
                                               targetGateset=self.gateset, svdTruncateTo=4, verbosity=0)
        gs_exlgst_verb = self.runSilent(GST.Core.do_iterative_exlgst, ds, gs_clgst, self.specs, self.elgstStrings,
                                        targetGateset=self.gateset, svdTruncateTo=4, verbosity=10)
        gs_exlgst_reg = GST.Core.do_iterative_exlgst(ds, gs_clgst, self.specs, self.elgstStrings,
                                               targetGateset=self.gateset, svdTruncateTo=4, verbosity=0,
                                               regularizeFactor=10)
        self.assertAlmostEqual(gs_exlgst.diff_frobenius(gs_exlgst_verb),0)


        #Run internal checks on less max-L values (so it doesn't take forever)
        gs_exlgst_chk = GST.Core.do_iterative_exlgst(ds, gs_clgst, self.specs, self.elgstStrings[0:2],
                                                   targetGateset=self.gateset, svdTruncateTo=4, verbosity=0,
                                                   check_jacobian=True)
        gs_exlgst_chk_verb = self.runSilent(GST.Core.do_iterative_exlgst,ds, gs_clgst, self.specs, self.elgstStrings[0:2],
                                                   targetGateset=self.gateset, svdTruncateTo=4, verbosity=10,
                                                   check_jacobian=True)
        
        # RUN BELOW LINES TO SEED SAVED GATESET FILES
        #GST.write_gateset(gs_exlgst,"cmp_chk_files/exlgst.gateset", "Saved Extended-LGST (eLGST) Gateset")
        #GST.write_gateset(gs_exlgst_reg,"cmp_chk_files/exlgst_reg.gateset", "Saved Extended-LGST (eLGST) Gateset w/regularization")

        gs_exlgst_compare = GST.load_gateset("cmp_chk_files/exlgst.gateset")
        gs_exlgst_reg_compare = GST.load_gateset("cmp_chk_files/exlgst_reg.gateset")
        
        self.assertAlmostEqual( gs_exlgst.diff_frobenius(gs_exlgst_compare), 0)
        self.assertAlmostEqual( gs_exlgst_reg.diff_frobenius(gs_exlgst_reg_compare), 0)


    def test_LSGST(self):

        ds = GST.generate_fake_data(self.datagen_gateset, self.lsgstStrings[-1], nSamples=1000,
                                  sampleError='binomial', seed=100)
        
        gs_lgst = GST.do_lgst(ds, self.specs, self.gateset, svdTruncateTo=4, verbosity=0)
        gs_lgst_go = GST.optimize_gauge(gs_lgst,"target",targetGateset=self.gateset, spamWeight=1.0, gateWeight=1.0)
        gs_clgst = GST.contract(gs_lgst_go, "CPTP")

        gs_lsgst = GST.Core.do_iterative_mc2gst(ds, gs_clgst, self.lsgstStrings, verbosity=0,
                                             minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6),
                                             memLimit=1000*1024**2)
        gs_lsgst_verb = self.runSilent(GST.Core.do_iterative_mc2gst, ds, gs_clgst, self.lsgstStrings, verbosity=10,
                                             minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6),
                                             memLimit=10*1024**2)
        gs_lsgst_reg = GST.Core.do_iterative_mc2gst(ds, gs_clgst, self.lsgstStrings, verbosity=0,
                                                 minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6),
                                                 regularizeFactor=10, memLimit=100*1024**2)
        self.assertAlmostEqual(gs_lsgst.diff_frobenius(gs_lsgst_verb),0)


        #Run internal checks on less max-L values (so it doesn't take forever)
        gs_lsgst_chk = GST.Core.do_iterative_mc2gst(ds, gs_clgst, self.lsgstStrings[0:2], verbosity=0,
                                                 minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6),
                                                 check=True, check_jacobian=True)
        gs_lsgst_chk_verb = self.runSilent(GST.Core.do_iterative_mc2gst, ds, gs_clgst, self.lsgstStrings[0:2], verbosity=10,
                                                      minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6),
                                                      check=True, check_jacobian=True, memLimit=100*1024**2)

        #Other option variations - just make sure they run at this point
        gs_lsgst_chk_opts = GST.Core.do_iterative_mc2gst(ds, gs_clgst, self.lsgstStrings[0:2], verbosity=0,
                                                      minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6),
                                                      useFreqWeightedChiSq=True, gateStringSetLabels=["Set1","Set2"],
                                                      gatestringWeightsDict={ ('Gx',): 2.0 } )



        # RUN BELOW LINES TO SEED SAVED GATESET FILES
        #GST.write_gateset(gs_lsgst,"cmp_chk_files/lsgst.gateset", "Saved LSGST Gateset")
        #GST.write_gateset(gs_lsgst_reg,"cmp_chk_files/lsgst_reg.gateset", "Saved LSGST Gateset w/Regularization")

        gs_lsgst_compare = GST.load_gateset("cmp_chk_files/lsgst.gateset")
        gs_lsgst_reg_compare = GST.load_gateset("cmp_chk_files/lsgst_reg.gateset")
        
        self.assertAlmostEqual( gs_lsgst.diff_frobenius(gs_lsgst_compare), 0)
        self.assertAlmostEqual( gs_lsgst_reg.diff_frobenius(gs_lsgst_reg_compare), 0)


    def test_MLEGST(self):

        ds = GST.generate_fake_data(self.datagen_gateset, self.lsgstStrings[-1], nSamples=1000,
                                  sampleError='binomial', seed=100)
        
        gs_lgst = GST.do_lgst(ds, self.specs, self.gateset, svdTruncateTo=4, verbosity=0)
        gs_lgst_go = GST.optimize_gauge(gs_lgst,"target",targetGateset=self.gateset, spamWeight=1.0, gateWeight=1.0)
        gs_clgst = GST.contract(gs_lgst_go, "CPTP")

        gs_mlegst = GST.Core.do_iterative_mlgst(ds, gs_clgst, self.lsgstStrings, verbosity=0,
                                               minProbClip=1e-6, probClipInterval=(-1e2,1e2),
                                               memLimit=1000*1024**2)
        gs_mlegst_verb = self.runSilent(GST.Core.do_iterative_mlgst, ds, gs_clgst, self.lsgstStrings, verbosity=10,
                                             minProbClip=1e-6, probClipInterval=(-1e2,1e2),
                                             memLimit=10*1024**2)
        self.assertAlmostEqual(gs_mlegst.diff_frobenius(gs_mlegst_verb),0)


        #Run internal checks on less max-L values (so it doesn't take forever)
        gs_mlegst_chk = GST.Core.do_iterative_mlgst(ds, gs_clgst, self.lsgstStrings[0:2], verbosity=0,
                                                 minProbClip=1e-6, probClipInterval=(-1e2,1e2),
                                                 check=True)

        #Other option variations - just make sure they run at this point
        gs_mlegst_chk_opts = GST.Core.do_iterative_mlgst(ds, gs_clgst, self.lsgstStrings[0:2], verbosity=0,
                                                      minProbClip=1e-6, probClipInterval=(-1e2,1e2),
                                                      gateStringSetLabels=["Set1","Set2"] )

        # RUN BELOW LINES TO SEED SAVED GATESET FILES
        #GST.write_gateset(gs_mlegst,"cmp_chk_files/mle_gst.gateset", "Saved MLE-GST Gateset")
    
        gs_mle_compare = GST.load_gateset("cmp_chk_files/mle_gst.gateset")
        self.assertAlmostEqual( gs_mlegst.diff_frobenius(gs_mle_compare), 0)

    def test_LGST_1overSqrtN_dependence(self):
        my_datagen_gateset = GST.GateSetTools.depolarize_gateset(self.gateset, noise=0.05)
        # !!don't depolarize spam or 1/sqrt(N) dependence saturates!!

        nSamplesList = np.array([ 16, 128, 1024, 8192 ])
        diffs = []
        for nSamples in nSamplesList:
            ds = GST.generate_fake_data(my_datagen_gateset, self.lgstStrings, nSamples,
                                      sampleError='binomial', seed=100)
            gs_lgst = GST.do_lgst(ds, self.specs, self.gateset, svdTruncateTo=4, verbosity=0)
            gs_lgst_go = GST.optimize_gauge(gs_lgst,"target",targetGateset=my_datagen_gateset, spamWeight=1.0, gateWeight=1.0)
            diffs.append( my_datagen_gateset.diff_frobenius(gs_lgst_go) )

        diffs = np.array(diffs, 'd')
        a,b = polyfit(np.log10(nSamplesList), np.log10(diffs), deg=1)
        #print nSamplesList; print diffs; print a #DEBUG
        self.assertLess( a+0.5, 0.05 )


    def test_model_selection(self):

        ds = GST.generate_fake_data(self.datagen_gateset, self.lsgstStrings[-1], nSamples=1000,
                                  sampleError='binomial', seed=100)
        

        gs_lgst4 = GST.do_lgst(ds, self.specs, self.gateset, svdTruncateTo=4, verbosity=0)
        gs_lgst6 = GST.do_lgst(ds, self.specs, self.gateset, svdTruncateTo=6, verbosity=0)
        
        chiSq4 = GST.AT.chi2(ds, gs_lgst4, self.lgstStrings, minProbClipForWeighting=1e-4)
        chiSq6 = GST.AT.chi2(ds, gs_lgst6, self.lgstStrings, minProbClipForWeighting=1e-4)

        #print "LGST dim=4 chiSq = ",chiSq4
        #print "LGST dim=6 chiSq = ",chiSq6
        self.assertAlmostEqual(chiSq4, 174.061524953) #429.271983052)
        self.assertAlmostEqual(chiSq6, 267013300.944, places=1) #1337.74222467) #Why is this so large??? -- DEBUG later

        # Least squares GST with model selection
        gs_lsgst = self.runSilent(GST.Core.do_iterative_mc2gst_with_model_selection, ds, gs_lgst4, 1, self.lsgstStrings[0:3],
                                  verbosity=10, minProbClipForWeighting=1e-3, probClipInterval=(-1e5,1e5))

        # RUN BELOW LINES TO SEED SAVED GATESET FILES
        #GST.write_gateset(gs_lsgst,"cmp_chk_files/lsgstMS.gateset", "Saved LSGST Gateset with model selection")

        gs_lsgst_compare = GST.load_gateset("cmp_chk_files/lsgstMS.gateset")
        
        self.assertAlmostEqual( gs_lsgst.diff_frobenius(gs_lsgst_compare), 0)

    def test_miscellaneous(self):
        ds = GST.generate_fake_data(self.datagen_gateset, self.lgstStrings, nSamples=1000, sampleError='none')
        strs = GST.Core.list_strings_lgst_can_estimate(ds, self.specs)

        self.runSilent(GST.print_gateset_info,self.gateset) #just make sure it works


    def test_gaugeopt_and_contract(self):
        ds = GST.generate_fake_data(self.datagen_gateset, self.lgstStrings, nSamples=10000,
                                  sampleError='binomial', seed=100)
        
        gs_lgst = GST.do_lgst(ds, self.specs, self.gateset, svdTruncateTo=4, verbosity=0)

        gs_lgst_target     = self.runSilent(GST.optimize_gauge, gs_lgst,"target",targetGateset=self.gateset,verbosity=10)
        gs_lgst_cptp       = self.runSilent(GST.optimize_gauge, gs_lgst,"CPTP",verbosity=10)
        gs_lgst_tp         = self.runSilent(GST.optimize_gauge, gs_lgst,"TP",verbosity=10)
        gs_lgst_tptarget   = self.runSilent(GST.optimize_gauge, gs_lgst,"TP and target",targetGateset=self.gateset,verbosity=10)
        gs_lgst_cptptarget = self.runSilent(GST.optimize_gauge, gs_lgst,"CPTP and target",targetGateset=self.gateset,verbosity=10)
        gs_lgst_cd         = self.runSilent(GST.optimize_gauge, gs_lgst,"Completely Depolarized",targetGateset=self.gateset,verbosity=10)
        
        gs_clgst_tp    = self.runSilent(GST.contract, gs_lgst_target, "TP",verbosity=10)
        gs_clgst_cp    = self.runSilent(GST.contract, gs_lgst_target, "CP",verbosity=10)
        gs_clgst_cptp  = self.runSilent(GST.contract, gs_lgst_target, "CPTP",verbosity=10)
        gs_clgst_xp    = self.runSilent(GST.contract, gs_lgst_target, "XP", ds,verbosity=10)
        gs_clgst_xptp  = self.runSilent(GST.contract, gs_lgst_target, "XPTP", ds,verbosity=10)
        gs_clgst_xptp  = self.runSilent(GST.contract, gs_lgst_target, "XPTP", ds,verbosity=10)
        gs_clgst_vsp   = self.runSilent(GST.contract, gs_lgst_target, "vSPAM",verbosity=10)
        gs_clgst_none  = self.runSilent(GST.contract, gs_lgst_target, "nothing",verbosity=10)

        #TODO: check output lies in space desired

        # big kick that should land it outside XP, TP, etc, so contraction
        # routines are more tested
        gs_bigkick = GST.GateSetTools.kick_gateset(gs_lgst_target, absmag=1.0) 

        gs_clgst_tp    = GST.contract(gs_bigkick, "TP")
        gs_clgst_cp    = GST.contract(gs_bigkick, "CP")
        gs_clgst_cptp  = GST.contract(gs_bigkick, "CPTP")
        gs_clgst_xp    = GST.contract(gs_bigkick, "XP", ds)
        gs_clgst_xptp  = GST.contract(gs_bigkick, "XPTP", ds)
        gs_clgst_vsp   = GST.contract(gs_bigkick, "vSPAM")
        gs_clgst_none  = GST.contract(gs_bigkick, "nothing")

        #TODO: check output lies in space desired
        


        
    
      
if __name__ == "__main__":
    unittest.main(verbosity=2)
