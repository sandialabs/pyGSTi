from pygsti.construction import std1Q_XYI as std

import unittest
import warnings
import pygsti
import os

from ..testutils import BaseTestCase, compare_files, temp_files

try:
    from PIL import Image, ImageChops # stackoverflow.com/questions/19230991/image-open-cannot-identify-image-file-python
    haveImageLibs = True
except ImportError:
    haveImageLibs = False


class AnalysisTestCase(BaseTestCase):

    def setUp(self):
        super(AnalysisTestCase, self).setUp()

        self.gateset = std.gs_target
        self.datagen_gateset = self.gateset.depolarize(gate_noise=0.05, spam_noise=0.1)

        self.fiducials = std.fiducials
        self.germs = std.germs
        self.specs = pygsti.construction.build_spam_specs(self.fiducials, prep_labels=['rho0'], effect_labels=['E0'])
        self.strs = pygsti.construction.get_spam_strs(self.specs)

        self.gateLabels = list(self.gateset.gates.keys()) # also == std.gates
        self.lgstStrings = pygsti.construction.list_lgst_gatestrings(self.specs, self.gateLabels)

        self.maxLengthList = [0,1,2,4,8]

        self.lsgstStrings = pygsti.construction.make_lsgst_lists(
            self.gateLabels, self.fiducials, self.fiducials, self.germs, self.maxLengthList )

        self.ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/analysis.dataset")
        self.lsgst_gateset = pygsti.io.load_gateset(compare_files + "/analysis.gateset")

        # RUN BELOW LINES TO GENERATE SAVED ANALYSIS GATESET AND DATASET
        #self.ds = pygsti.construction.generate_fake_data(self.datagen_gateset, self.lsgstStrings[-1], nSamples=1000,
        #                              sampleError='binomial', seed=100)
        #self.ds.save(compare_files + "/analysis.dataset")
        #gs_lgst = pygsti.alg.do_lgst(self.ds, self.specs, self.gateset, svdTruncateTo=4, verbosity=0)
        #gs_lgst_go = pygsti.alg.optimize_gauge(gs_lgst,"target",targetGateset=self.gateset)
        #gs_clgst = pygsti.alg.contract(gs_lgst_go, "CPTP")
        #self.lsgst_gateset = pygsti.alg.do_iterative_mc2gst(self.ds, gs_clgst, self.lsgstStrings, verbosity=0,
        #                                               minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6) )
        #pygsti.io.write_gateset(self.lsgst_gateset, compare_files + "/analysis.gateset")

        #Collect data we need for making plots
        self.Xs = self.maxLengthList[1:]
        self.xlbl = "L (max length)"
        self.Ys = self.germs
        self.ylbl = "Germ"
        self.gateStrDict = { (x,y):pygsti.construction.repeat_with_max_length(y,x,False) \
                                 for x in self.Xs for y in self.Ys }

        #remove duplicates by replacing duplicate strings with None
        runningList = []
        for x in self.Xs:
            for y in self.Ys:
                if self.gateStrDict[(x,y)] in runningList:
                    self.gateStrDict[(x,y)] = None
                else: runningList.append( self.gateStrDict[(x,y)] )

class TestAnalysis(AnalysisTestCase):

    def test_blank_boxes(self):
        pygsti.report.blank_boxplot( self.Xs, self.Ys, self.gateStrDict, self.strs, self.xlbl, self.ylbl,
                             sumUp=True, ticSize=20, save_to=temp_files + "/blankBoxes.jpg")
        self.assertEqualImages(temp_files + "/blankBoxes.jpg", compare_files + "/blankBoxes_ok.jpg")

    def test_chi2_boxes(self):
        pygsti.report.chi2_boxplot( self.Xs, self.Ys, self.gateStrDict, self.ds, self.lsgst_gateset, self.strs,
                         self.xlbl, self.ylbl, linlg_pcntle=.05, scale=1.0, sumUp=False, histogram=True,
                         save_to=temp_files + "/chi2boxes.jpg")
        self.assertEqualImages(temp_files + "/chi2boxes.jpg", compare_files + "/chi2boxes_ok.jpg")
        self.assertEqualImages(temp_files + "/chi2boxes_hist.jpg", compare_files + "/chi2boxes_hist_ok.jpg")

        pygsti.report.chi2_boxplot( self.Xs, self.Ys, self.gateStrDict, self.ds, self.lsgst_gateset, self.strs,
                         self.xlbl, self.ylbl, linlg_pcntle=.05, scale=1.0, sumUp=False, histogram=True, invert=True,
                         save_to=temp_files + "/chi2boxes_inv.jpg")
        self.assertEqualImages(temp_files + "/chi2boxes_inv.jpg", compare_files + "/chi2boxes_inv_ok.jpg")
        self.assertEqualImages(temp_files + "/chi2boxes_inv_hist.jpg", compare_files + "/chi2boxes_inv_hist_ok.jpg")

        pygsti.report.chi2_boxplot( self.Xs, self.Ys, self.gateStrDict, self.ds, self.lsgst_gateset, self.strs,
                         self.xlbl, self.ylbl, linlg_pcntle=.05, scale=1.0, sumUp=True,
                         save_to=temp_files + "/chi2boxes_summed.jpg")
        self.assertEqualImages(temp_files + "/chi2boxes_summed.jpg", compare_files + "/chi2boxes_summed_ok.jpg")

    def test_direct_boxes(self):
        directLGST = pygsti.report.direct_lgst_gatesets( [gs for gs in list(self.gateStrDict.values()) if gs is not None],
                                                self.ds, self.specs, self.gateset, svdTruncateTo=4, verbosity=0)
        directLSGST = pygsti.report.direct_mc2gst_gatesets( [gs for gs in list(self.gateStrDict.values()) if gs is not None],
                                                  self.ds, self.specs, self.gateset, svdTruncateTo=4,
                                                  minProbClipForWeighting=1e-2,
                                                  probClipInterval=(-1e6,1e6), verbosity=0)

        pygsti.report.direct_chi2_boxplot( self.Xs, self.Ys, self.gateStrDict, self.ds, directLSGST,
                                   self.strs, self.xlbl, self.ylbl,
                                   linlg_pcntle=.05, scale=1.0, boxLabels=True,
                                   save_to=temp_files + "/direct_chi2_boxes.jpg")
        self.assertEqualImages(temp_files + "/direct_chi2_boxes.jpg", compare_files + "/direct_chi2_boxes_ok.jpg")

        pygsti.report.direct_deviation_boxplot(self.Xs, self.Ys, self.gateStrDict, self.ds, self.lsgst_gateset,
                                      directLSGST, self.xlbl, self.ylbl, prec=4,
                                      scale=1.0, boxLabels=True,
                                      save_to=temp_files + "/direct_deviation.jpg")
        self.assertEqualImages(temp_files + "/direct_deviation.jpg", compare_files + "/direct_deviation_ok.jpg")

        pygsti.report.direct2x_comp_boxplot( self.Xs, self.Ys, self.gateStrDict, self.ds, directLSGST,
                                    self.strs, self.xlbl, self.ylbl,
                                    M=10, scale=1.0, boxLabels=True,
                                    save_to=temp_files + "/direct_2x_compare.jpg")
        self.assertEqualImages(temp_files + "/direct_2x_compare.jpg", compare_files + "/direct_2x_compare_ok.jpg")

        pygsti.report.small_eigval_err_rate_boxplot(self.Xs, self.Ys, self.gateStrDict, self.ds, directLSGST,
                                         self.xlbl, self.ylbl, scale=1.0, boxLabels=True,
                                         save_to=temp_files + "/small_eigval_err.jpg")
        self.assertEqualImages(temp_files + "/small_eigval_err.jpg", compare_files + "/small_eigval_err_ok.jpg")


    def test_whack_a_mole(self):
        whack = pygsti.obj.GateString( ('Gi',)*4 )
        fullGatestringList = self.lsgstStrings[-1]
        pygsti.report.whack_a_chi2_mole_boxplot( whack, fullGatestringList, self.Xs, self.Ys, self.gateStrDict,
                                       self.ds, self.lsgst_gateset, self.strs, self.xlbl, self.ylbl,
                                       scale=1.0, sumUp=False, histogram=True,
                                       save_to=temp_files + "/whackamole.jpg")
        self.assertEqualImages(temp_files + "/whackamole.jpg", compare_files + "/whackamole_ok.jpg")
        self.assertEqualImages(temp_files + "/whackamole_hist.jpg", compare_files + "/whackamole_hist_ok.jpg")

    def test_total_chi2(self):
        chi2 = pygsti.chi2( self.ds, self.lsgst_gateset )
        self.assertAlmostEqual(chi2, 729.182015795)


if __name__ == "__main__":
    unittest.main(verbosity=2)
