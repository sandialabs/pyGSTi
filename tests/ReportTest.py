import unittest
import GST
from GSTCommons import Std1Q_XYI as Std
from GSTCommons import MakeLists_WholeGermPowers as MakeLists
import numpy as np
import os

class ReportTestCase(unittest.TestCase):

    def setUp(self):

        targetGateset = Std.gs_target
        datagen_gateset = GST.GateSetTools.depolarize_gateset(targetGateset, noise=0.05)
        datagen_gateset = GST.GateSetTools.depolarize_spam(datagen_gateset, noise=0.1)
        
        fiducials = Std.fiducials
        germs = Std.germs
        specs = GST.get_spam_specs(fiducials, EVecInds=[0]) #only use the first EVec
        strs = GST.get_spam_strs(specs)

        gateLabels = targetGateset.keys() # also == Std.gates
        lgstStrings = GST.list_lgst_gatestrings(specs, targetGateset.keys())

        maxLengthList = [0,1,2,4,8]
        
        lsgstStrings = MakeLists.make_lsgst_lists(gateLabels, fiducials, germs, maxLengthList)

        ds = GST.DataSet(fileToLoadFrom="cmp_chk_files/reportgen.dataset")

        # RUN BELOW LINES TO GENERATE ANALYSIS DATASET
        #ds = GST.generate_fake_data(datagen_gateset, lsgstStrings[-1], nSamples=1000,
        #                          sampleError='binomial', seed=100)
        #ds.save("cmp_chk_files/reportgen.dataset")

        gs_lgst = GST.do_lgst(ds, specs, targetGateset, svdTruncateTo=4, verbosity=0)
        gs_lgst_go = GST.optimize_gauge(gs_lgst,"target",targetGateset=targetGateset)
        gs_clgst = GST.contract(gs_lgst_go, "CPTP")
        lsgst_gatesets = GST.Core.do_iterative_mc2gst(ds, gs_clgst, lsgstStrings, verbosity=0,
                                                   minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6),
                                                   returnAll=True)

        self.results = GST.Results()
        self.results.init_Ls_and_germs("chi2", targetGateset, ds, gs_clgst, maxLengthList, germs,
                                     lsgst_gatesets, lsgstStrings, fiducials, fiducials, 
                                     GST.GateStringTools.repeat_with_max_length, False)



class TestReport(ReportTestCase):
    
    def test_reportsA(self):
        self.results.create_full_report_pdf(filename="temp_test_files/full_reportA.pdf", confidenceLevel=None,
                                         debugAidsAppendix=False, gaugeOptAppendix=False,
                                         pixelPlotAppendix=False, whackamoleAppendix=False)
        self.results.create_brief_report_pdf(filename="temp_test_files/brief_reportA.pdf", confidenceLevel=None)
        self.results.create_presentation_pdf(filename="temp_test_files/slidesA.pdf", confidenceLevel=None,
                                           debugAidsAppendix=False, pixelPlotAppendix=False, whackamoleAppendix=False)
        self.results.create_presentation_ppt(filename="temp_test_files/slidesA.ppt", confidenceLevel=None,
                                           debugAidsAppendix=False, pixelPlotAppendix=False, whackamoleAppendix=False)

        def checkFile(fn):
            linesToTest = open("temp_test_files/%s" % fn).readlines()
            linesOK = open("cmp_chk_files/%s" % fn).readlines()
            self.assertEqual(linesToTest,linesOK)

        #Compare the text files, assume if these match the PDFs are equivalent
        checkFile("full_reportA.tex")
        checkFile("brief_reportA.tex")
        checkFile("slidesA.tex")

    def test_reportsB(self):
        self.results.create_full_report_pdf(filename="temp_test_files/full_reportB.pdf", confidenceLevel=95,
                                         debugAidsAppendix=True, gaugeOptAppendix=True,
                                         pixelPlotAppendix=True, whackamoleAppendix=True,
                                         verbosity=2)
        self.results.create_brief_report_pdf(filename="temp_test_files/brief_reportB.pdf", confidenceLevel=95, verbosity=2)
        self.results.create_presentation_pdf(filename="temp_test_files/slidesB.pdf", confidenceLevel=95,
                                           debugAidsAppendix=True, pixelPlotAppendix=True, whackamoleAppendix=True,
                                           verbosity=2)
        self.results.create_presentation_ppt(filename="temp_test_files/slidesB.ppt", confidenceLevel=95,
                                           debugAidsAppendix=True, pixelPlotAppendix=True, whackamoleAppendix=True,
                                           verbosity=2)

        def checkFile(fn):
            linesToTest = open("temp_test_files/%s" % fn).readlines()
            linesOK = open("cmp_chk_files/%s" % fn).readlines()
            self.assertEqual(linesToTest,linesOK)

        #Compare the text files, assume if these match the PDFs are equivalent
        checkFile("full_reportB.tex")
        checkFile("full_reportB_appendices.tex")
        checkFile("brief_reportB.tex")
        checkFile("slidesB.tex")
        
if __name__ == "__main__":
    unittest.main(verbosity=2)
