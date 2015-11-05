import unittest
import GST
from GSTCommons import Std1Q_XYI as Std
from GSTCommons import MakeLists_WholeGermPowers as MakeLists
import numpy as np
import os

class ReportTestCase(unittest.TestCase):

    def setUp(self):

        targetGateset = Std.gs_target
        datagen_gateset = GST.GateSetTools.depolarizeGateset(targetGateset, noise=0.05)
        datagen_gateset = GST.GateSetTools.depolarizeSPAM(datagen_gateset, noise=0.1)
        
        fiducials = Std.fiducials
        germs = Std.germs
        specs = GST.getRhoAndESpecs(fiducials, EVecInds=[0]) #only use the first EVec
        strs = GST.getRhoAndEStrs(specs)

        gateLabels = targetGateset.keys() # also == Std.gates
        lgstStrings = GST.listLGSTGateStrings(specs, targetGateset.keys())

        maxLengthList = [0,1,2,4,8]
        
        lsgstStrings = MakeLists.make_lsgst_lists(gateLabels, fiducials, germs, maxLengthList)

        ds = GST.DataSet(fileToLoadFrom="cmp_chk_files/reportgen.dataset")

        # RUN BELOW LINES TO GENERATE ANALYSIS DATASET
        #ds = GST.generateFakeData(datagen_gateset, lsgstStrings[-1], nSamples=1000,
        #                          sampleError='binomial', seed=100)
        #ds.save("cmp_chk_files/reportgen.dataset")

        gs_lgst = GST.doLGST(ds, specs, targetGateset, svdTruncateTo=4, verbosity=0)
        gs_lgst_go = GST.optimizeGauge(gs_lgst,"target",targetGateset=targetGateset)
        gs_clgst = GST.contract(gs_lgst_go, "CPTP")
        lsgst_gatesets = GST.Core.doIterativeLSGST(ds, gs_clgst, lsgstStrings, verbosity=0,
                                                   minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6),
                                                   returnAll=True)

        self.results = GST.Results()
        self.results.init_LsAndGerms("chi2", targetGateset, ds, gs_clgst, maxLengthList, germs,
                                     lsgst_gatesets, lsgstStrings, fiducials, fiducials, 
                                     GST.GateStringTools.repeatWithMaxLength, False)



class TestReport(ReportTestCase):
    
    def test_reportsA(self):
        self.results.createFullReportPDF(filename="temp_test_files/full_reportA.pdf", confidenceLevel=None,
                                         debugAidsAppendix=False, gaugeOptAppendix=False,
                                         pixelPlotAppendix=False, whackamoleAppendix=False)
        self.results.createBriefReportPDF(filename="temp_test_files/brief_reportA.pdf", confidenceLevel=None)
        self.results.createPresentationPDF(filename="temp_test_files/slidesA.pdf", confidenceLevel=None,
                                           debugAidsAppendix=False, pixelPlotAppendix=False, whackamoleAppendix=False)
        self.results.createPresentationPPT(filename="temp_test_files/slidesA.ppt", confidenceLevel=None,
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
        self.results.createFullReportPDF(filename="temp_test_files/full_reportB.pdf", confidenceLevel=95,
                                         debugAidsAppendix=True, gaugeOptAppendix=True,
                                         pixelPlotAppendix=True, whackamoleAppendix=True,
                                         verbosity=2)
        self.results.createBriefReportPDF(filename="temp_test_files/brief_reportB.pdf", confidenceLevel=95, verbosity=2)
        self.results.createPresentationPDF(filename="temp_test_files/slidesB.pdf", confidenceLevel=95,
                                           debugAidsAppendix=True, pixelPlotAppendix=True, whackamoleAppendix=True,
                                           verbosity=2)
        self.results.createPresentationPPT(filename="temp_test_files/slidesB.ppt", confidenceLevel=95,
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
