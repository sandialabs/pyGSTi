import unittest
import pygsti
from pygsti.construction import std1Q_XYI as std

import numpy as np
import os

class ReportTestCase(unittest.TestCase):

    def setUp(self):
        #Set GateSet objects to "strict" mode for testing
        pygsti.objects.GateSet._strict = True

        self.targetGateset = std.gs_target
        datagen_gateset = self.targetGateset.depolarize(gate_noise=0.05, spam_noise=0.1)
        
        self.fiducials = std.fiducials
        self.germs = std.germs
        self.specs = pygsti.construction.build_spam_specs(self.fiducials, EVecLbls=['E0']) #only use the first EVec

        self.gateLabels = self.targetGateset.gates.keys() # also == std.gates
        self.lgstStrings = pygsti.construction.list_lgst_gatestrings(self.specs, self.gateLabels)

        self.maxLengthList = [0,1,2,4,8]
        
        self.lsgstStrings = pygsti.construction.make_lsgst_lists(self.gateLabels, self.fiducials, self.germs, self.maxLengthList)

        self.ds = pygsti.objects.DataSet(fileToLoadFrom="cmp_chk_files/reportgen.dataset")

        # RUN BELOW LINES TO GENERATE ANALYSIS DATASET
        #ds = pygsti.construction.generate_fake_data(datagen_gateset, lsgstStrings[-1], nSamples=1000,
        #                                            sampleError='binomial', seed=100)
        #ds.save("cmp_chk_files/reportgen.dataset")

        gs_lgst = pygsti.do_lgst(self.ds, self.specs, self.targetGateset, svdTruncateTo=4, verbosity=0)
        gs_lgst_go = pygsti.optimize_gauge(gs_lgst,"target",targetGateset=self.targetGateset)
        self.gs_clgst = pygsti.contract(gs_lgst_go, "CPTP")
        self.gs_clgst_tp = pygsti.contract(self.gs_clgst, "vSPAM")
        self.gs_clgst_tp.set_all_parameterizations("TP")
        

    def checkFile(self, fn):
        linesToTest = open("temp_test_files/%s" % fn).readlines()
        linesOK = open("cmp_chk_files/%s" % fn).readlines()
        self.assertEqual(linesToTest,linesOK)



class TestReport(ReportTestCase):
    
    def test_reports_chi2(self):

        lsgst_gatesets = pygsti.do_iterative_mc2gst(self.ds, self.gs_clgst, self.lsgstStrings, verbosity=0,
                                                   minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6),
                                                   returnAll=True)
        lsgst_gatesets = [ pygsti.optimize_gauge(gs, "target", targetGateset=self.targetGateset,
                                                 gateWeight=1,spamWeight=0.001) for gs in lsgst_gatesets]

        self.results = pygsti.report.Results()
        self.results.init_Ls_and_germs("chi2", self.targetGateset, self.ds, self.gs_clgst, self.maxLengthList, self.germs,
                                     lsgst_gatesets, self.lsgstStrings, self.fiducials, self.fiducials, 
                                     pygsti.construction.repeat_with_max_length, False)
        self.results.set_additional_info(minProbClip=1e-6, minProbClipForWeighting=1e-4,
                                         probClipInterval=(-1e6,1e6), radius=1e-4,
                                         weightsDict=None, defaultDirectory="temp_test_files",
                                         defaultBasename="MyDefaultReportName")

        self.results.create_full_report_pdf(filename="temp_test_files/full_reportA.pdf", confidenceLevel=None,
                                         debugAidsAppendix=False, gaugeOptAppendix=False,
                                         pixelPlotAppendix=False, whackamoleAppendix=False)
        self.results.create_brief_report_pdf(filename="temp_test_files/brief_reportA.pdf", confidenceLevel=None)
        self.results.create_presentation_pdf(filename="temp_test_files/slidesA.pdf", confidenceLevel=None,
                                           debugAidsAppendix=False, pixelPlotAppendix=False, whackamoleAppendix=False)
        self.results.create_presentation_ppt(filename="temp_test_files/slidesA.ppt", confidenceLevel=None,
                                           debugAidsAppendix=False, pixelPlotAppendix=False, whackamoleAppendix=False)

        #Run again using default filenames
        self.results.create_full_report_pdf(filename="auto", confidenceLevel=None,
                                         debugAidsAppendix=False, gaugeOptAppendix=False,
                                         pixelPlotAppendix=False, whackamoleAppendix=False)
        self.results.create_brief_report_pdf(filename="auto", confidenceLevel=None)
        self.results.create_presentation_pdf(filename="auto", confidenceLevel=None,
                                           debugAidsAppendix=False, pixelPlotAppendix=False, whackamoleAppendix=False)
        self.results.create_presentation_ppt(filename="auto", confidenceLevel=None,
                                           debugAidsAppendix=False, pixelPlotAppendix=False, whackamoleAppendix=False)

        #Compare the text files, assume if these match the PDFs are equivalent
        self.checkFile("full_reportA.tex")
        self.checkFile("brief_reportA.tex")
        self.checkFile("slidesA.tex")


        self.results.create_full_report_pdf(filename="temp_test_files/full_reportB.pdf", confidenceLevel=95,
                                         debugAidsAppendix=True, gaugeOptAppendix=True,
                                         pixelPlotAppendix=True, whackamoleAppendix=True,
                                         verbosity=2)
        self.results.create_full_report_pdf(filename="temp_test_files/full_reportB-noGOpt.pdf", confidenceLevel=95,
                                         debugAidsAppendix=True, gaugeOptAppendix=False,
                                         pixelPlotAppendix=True, whackamoleAppendix=True) # to test blank GOpt tables
        self.results.create_brief_report_pdf(filename="temp_test_files/brief_reportB.pdf", confidenceLevel=95, verbosity=2)
        self.results.create_presentation_pdf(filename="temp_test_files/slidesB.pdf", confidenceLevel=95,
                                           debugAidsAppendix=True, pixelPlotAppendix=True, whackamoleAppendix=True,
                                           verbosity=2)
        self.results.create_presentation_ppt(filename="temp_test_files/slidesB.ppt", confidenceLevel=95,
                                           debugAidsAppendix=True, pixelPlotAppendix=True, whackamoleAppendix=True,
                                           verbosity=2)

        #Compare the text files, assume if these match the PDFs are equivalent
        self.checkFile("full_reportB.tex")
        self.checkFile("full_reportB_appendices.tex")
        self.checkFile("brief_reportB.tex")
        self.checkFile("slidesB.tex")

    def test_reports_logL_TP(self):
        lsgst_gatesets_TP = pygsti.do_iterative_mlgst(self.ds, self.gs_clgst_tp, self.lsgstStrings, verbosity=0,
                                                   minProbClip=1e-4, probClipInterval=(-1e6,1e6),
                                                   returnAll=True)
        lsgst_gatesets_TP = [ pygsti.optimize_gauge(gs, "target", targetGateset=self.targetGateset, constrainToTP=True,
                                                 gateWeight=1,spamWeight=0.001) for gs in lsgst_gatesets_TP]

        self.results_logL = pygsti.report.Results()
        self.results_logL.init_Ls_and_germs("logl", self.targetGateset, self.ds, self.gs_clgst_tp, self.maxLengthList, self.germs,
                                     lsgst_gatesets_TP, self.lsgstStrings, self.fiducials, self.fiducials, 
                                     pygsti.construction.repeat_with_max_length, True)

        #Run a few tests to generate figures we don't use in reports
        self.results_logL.get_figure("bestEstimateSummedColorBoxPlot")
        self.results_logL.get_figure("blankBoxPlot")
        self.results_logL.get_figure("blankSummedBoxPlot")
        self.results_logL.get_figure("directLGSTColorBoxPlot")
        self.results_logL.get_figure("directLGSTDeviationColorBoxPlot")
        with self.assertRaises(ValueError):
            self.results_logL.get_figure("FooBar")
        with self.assertRaises(ValueError):
            self.results_logL.get_special('FooBar')

        #Run tests to generate tables we don't use in reports
        self.results_logL.get_table("bestGatesetVsTargetAnglesTable")



        self.results_logL.create_full_report_pdf(filename="temp_test_files/full_reportC.pdf", confidenceLevel=None,
                                                 debugAidsAppendix=False, gaugeOptAppendix=False,
                                                 pixelPlotAppendix=False, whackamoleAppendix=False,
                                                 verbosity=2)
        self.results_logL.create_brief_report_pdf(filename="temp_test_files/brief_reportC.pdf", confidenceLevel=None, verbosity=2)
        self.results_logL.create_presentation_pdf(filename="temp_test_files/slidesC.pdf", confidenceLevel=None,
                                                  debugAidsAppendix=False, pixelPlotAppendix=False, whackamoleAppendix=False,
                                                  verbosity=2)
        self.results_logL.create_presentation_ppt(filename="temp_test_files/slidesC.ppt", confidenceLevel=None,
                                                  debugAidsAppendix=False, pixelPlotAppendix=False, whackamoleAppendix=False,
                                                  verbosity=2)

        ##Compare the text files, assume if these match the PDFs are equivalent
        self.checkFile("full_reportC.tex")
        self.checkFile("brief_reportC.tex")
        self.checkFile("slidesC.tex")


        self.results_logL.create_full_report_pdf(filename="temp_test_files/full_reportD.pdf", confidenceLevel=95,
                                                 debugAidsAppendix=True, gaugeOptAppendix=True,
                                                 pixelPlotAppendix=True, whackamoleAppendix=True,
                                                 verbosity=2)
        self.results_logL.create_brief_report_pdf(filename="temp_test_files/brief_reportD.pdf", confidenceLevel=95, verbosity=2)
        self.results_logL.create_presentation_pdf(filename="temp_test_files/slidesD.pdf", confidenceLevel=95,
                                                  debugAidsAppendix=True, pixelPlotAppendix=True, whackamoleAppendix=True,
                                                  verbosity=2)
        self.results_logL.create_presentation_ppt(filename="temp_test_files/slidesD.ppt", confidenceLevel=95,
                                                  debugAidsAppendix=True, pixelPlotAppendix=True, whackamoleAppendix=True,
                                                  verbosity=2)

        ##Compare the text files, assume if these match the PDFs are equivalent
        self.checkFile("full_reportD.tex")
        self.checkFile("full_reportD_appendices.tex")
        self.checkFile("brief_reportD.tex")
        self.checkFile("slidesD.tex")


    def test_table_generation(self):
        import pygsti.report.generation as gen
        formats = ['latex','html','py','ppt'] #all the formats we know
        tableclass = "cssClass"
        longtable = False
        confidenceRegionInfo = None

        gateset = pygsti.io.load_gateset("cmp_chk_files/analysis.gateset")
        ds = pygsti.objects.DataSet(fileToLoadFrom="cmp_chk_files/analysis.dataset")

        chi2, chi2Hessian = pygsti.chi2(ds, gateset, returnHessian=True)
        ci = pygsti.obj.ConfidenceRegion(gateset, chi2Hessian, 95.0,
                                         hessianProjection="std")
        gateset_tp = pygsti.contract(gateset,"TP"); gateset_tp.set_all_parameterizations("TP")
        chi2, chi2Hessian_TP = pygsti.chi2(ds, gateset_tp, returnHessian=True)
        ci_TP = pygsti.obj.ConfidenceRegion(gateset_tp, chi2Hessian_TP, 95.0,
                                            hessianProjection="std")

        chi2, chi2Hessian_tgt = pygsti.chi2(ds, std.gs_target, returnHessian=True)
        ci_tgt = pygsti.obj.ConfidenceRegion(std.gs_target, chi2Hessian_tgt, 95.0,
                                         hessianProjection="std")
        target_tp = std.gs_target.copy(); target_tp.set_all_parameterizations("TP")
        chi2, chi2Hessian_tgt_TP = pygsti.chi2(ds, target_tp, returnHessian=True)
        ci_TP_tgt = pygsti.obj.ConfidenceRegion(target_tp, chi2Hessian_tgt_TP, 95.0,
                                            hessianProjection="std")

        gateset_2q = pygsti.construction.build_gateset( 
            [4], [('Q0','Q1')],['GIX','GIY','GXI','GYI','GCNOT'], 
            [ "I(Q0):X(pi/2,Q1)", "I(Q0):Y(pi/2,Q1)", "X(pi/2,Q0):I(Q1)", "Y(pi/2,Q0):I(Q1)", "CX(pi,Q0,Q1)" ],
            rhoLabelList=['rho0'], rhoExpressions=["0"], ELabelList=['E0','E1','E2'], EExpressions=["0","1","2"], 
            spamLabelDict={'upup': ('rho0','E0'), 'updn': ('rho0','E1'), 'dnup': ('rho0','E2'),
                           'dndn': ('rho0','remainder') }, basis="pp" )
        

        #tests which fill in the cracks of the full-report tests
        with self.assertRaises(ValueError):
            gen.get_gateset_spam_table(gateset, formats, tableclass,
                                       longtable, None, mxBasis="fooBar")
        gen.get_gateset_gates_table(gateset_tp, formats, tableclass, longtable, ci_TP) #test zero-padding
        gen.get_unitary_gateset_gates_table(std.gs_target, formats, tableclass, longtable, ci_tgt) #unitary gates w/CIs
        gen.get_unitary_gateset_gates_table(target_tp, formats, tableclass, longtable, ci_TP_tgt) #unitary gates w/CIs
        gen.get_gateset_closest_unitary_table(gateset_2q, formats, tableclass, longtable, None) #test higher-dim gateset
        gen.get_gateset_closest_unitary_table(gateset, formats, tableclass, longtable, ci) #test with CIs (long...)
        gen.get_gateset_rotn_axis_table(std.gs_target, formats, tableclass, longtable, None) #try to get "--"s and "X"s to display
        gen.get_chi2_progress_table([0], [gateset_tp], [ [('Gx',)],], ds, formats, tableclass, longtable) #TP case
        gen.get_chi2_confidence_region(gateset_tp, ds, 95) #TP case

        gen.get_gatestring_multi_table([ [('Gx',),('Gz',)], [('Gy',)] ], ["list1","list2"], formats,
                                       tableclass, longtable, commonTitle=None) #commonTitle == None case w/diff length lists

        with self.assertRaises(ValueError):
            gen.get_unitary_gateset_gates_table(std.gs_target, formats, tableclass, longtable, ci) #gateset-CI mismatch
        with self.assertRaises(ValueError):
            gen.get_gateset_spam_parameters_table(std.gs_target, formats, tableclass, longtable, ci) #gateset-CI mismatch

        #LogL case tests
        gen.get_logl_progress_table([0], [gateset_tp], [ [('Gx',)],], ds, formats, tableclass, longtable) # logL case
        gen.get_logl_progress_table([0], [gateset], [ [('Gx',)],], ds, formats, tableclass, longtable) # logL case

        gen.get_logl_confidence_region(gateset_tp, ds, 95,
                                       gatestring_list=None, probClipInterval=(-1e6,1e6),
                                       minProbClip=1e-4, radius=1e-4, hessianProjection="std")
        gen.get_logl_confidence_region(gateset, ds, 95,
                                       gatestring_list=None, probClipInterval=(-1e6,1e6),
                                       minProbClip=1e-4, radius=1e-4, hessianProjection="std")

        
    def test_table_formatting(self):
        vec = np.array( [1.0,2.0,3.0] )
        mx = np.identity( 2, 'd' )
        rank3Tensor = np.zeros( (2,2,2), 'd')
        f = 10.0
        l = [10.0, 20.0]
        
        class weirdType:
            def __init__(self):
                pass
            def __str__(self):
                return "weird"
        w = weirdType()

        print "Float formatting"
        pygsti.report.html.html(f)
        pygsti.report.latex.latex(f)
        pygsti.report.ppt.ppt(f)

        print "List formatting"
        pygsti.report.html.html(l)
        pygsti.report.latex.latex(l)
        pygsti.report.ppt.ppt(l)

        print "Arbitrary class formatting"
        pygsti.report.html.html(w)
        pygsti.report.latex.latex(w)
        pygsti.report.ppt.ppt(w)

        print "Vector formatting"
        pygsti.report.html.html(vec)
        pygsti.report.latex.latex(vec)
        pygsti.report.ppt.ppt(vec)

        print "Vector formatting (w/brackets)"
        pygsti.report.html.html(vec, brackets=True)
        pygsti.report.latex.latex(vec, brackets=True)
        pygsti.report.ppt.ppt(vec, brackets=True)

        print "Matrix formatting"
        pygsti.report.html.html_matrix(mx, fontsize=8, brackets=False)
        pygsti.report.latex.latex_matrix(mx, fontsize=8, brackets=False)
        pygsti.report.ppt.ppt_matrix(mx, fontsize=8, brackets=False)

        print "Value formatting"
        ROUND = 2; complxAsPolar=True
        for complxAsPolar in (True,False):
            for x in (0.001,0.01,1.0,10.0,100.0,1000.0,10000.0,1.0+1.0j,10j,1.0+1e-10j,1e-10j,"N/A"):
                pygsti.report.html.html_value(x, ROUND, complxAsPolar)
                pygsti.report.latex.latex_value(x, ROUND, complxAsPolar)
                pygsti.report.ppt.ppt_value(x, ROUND, complxAsPolar)        

        with self.assertRaises(ValueError):
            pygsti.report.html.html(rank3Tensor)
            pygsti.report.latex.latex(rank3Tensor)

    def test_plotting(self):
        test_data = np.array( [[1e-8,1e-7,1e-6,1e-5],
                               [1e-4,1e-3,1e-2,1e-1],
                               [1.0,10.0,100.0,1000.0],
                               [1.0e4,1.0e5,1.0e6,1.0e7]],'d' )
        gstFig = pygsti.report.plotting.color_boxplot( test_data, size=(10,10), prec="compacthp",save_to="temp_test_files/test.pdf")
        gstFig = pygsti.report.plotting.color_boxplot( test_data, size=(10,10), prec="compact",save_to="")
        gstFig = pygsti.report.plotting.color_boxplot( test_data, size=(10,10), prec=3,save_to="")
        gstFig = pygsti.report.plotting.color_boxplot( test_data, size=(10,10), prec=-3,save_to="")
        gstFig = pygsti.report.plotting.color_boxplot( test_data, size=(10,10), prec="foobar",colorbar=True,save_to="")
        gstFig.check()

        pygsti.report.plotting.gateset_with_lgst_gatestring_estimates( [('Gx','Gx')], self.ds, self.specs,
                                                                       self.targetGateset,includeTargetGates=False,
                                                                       gateStringLabels=None, svdTruncateTo=4, verbosity=0)

        gsWithGxgx = pygsti.report.plotting.focused_mc2gst_gatesets( 
            pygsti.construction.gatestring_list([('Gx','Gx')]), self.ds, self.specs, self.gs_clgst)

    def test_reportables(self):
        #Test that None is returned when qty cannot be computed
        qty = pygsti.report.reportables.compute_dataset_qty("FooBar",self.ds)
        self.assertIsNone(qty)
        qty = pygsti.report.reportables.compute_gateset_qty("FooBar",self.gs_clgst)
        self.assertIsNone(qty)
        qty = pygsti.report.reportables.compute_gateset_dataset_qty("FooBar",self.gs_clgst, self.ds)
        self.assertIsNone(qty)
        qty = pygsti.report.reportables.compute_gateset_gateset_qty("FooBar",self.gs_clgst, self.gs_clgst)
        self.assertIsNone(qty)

        #test ignoring gate strings not in dataset        
        qty = pygsti.report.reportables.compute_dataset_qty("gate string length", self.ds,
                                                            pygsti.construction.gatestring_list([('Gx','Gx'),('Gfoobar',)]) )
        qty = pygsti.report.reportables.compute_gateset_dataset_qty("prob(plus) diff", self.gs_clgst, self.ds,
                                                            pygsti.construction.gatestring_list([('Gx','Gx'),('Gfoobar',)]) )
        qty_str = str(qty) #test __str__

        #Test gateset gates mismatch
        from pygsti.construction import std1Q_XY as stdXY          
        with self.assertRaises(ValueError):
            qty = pygsti.report.reportables.compute_gateset_gateset_qty(
                "Gx fidelity",std.gs_target, stdXY.gs_target) #Gi missing from 2nd gateset
        with self.assertRaises(ValueError):
            qty = pygsti.report.reportables.compute_gateset_gateset_qty(
                "Gx fidelity",stdXY.gs_target, std.gs_target) #Gi missing from 1st gateset



    def test_results_object(self):
        results = pygsti.report.Results()
        results.init_single("logl", self.targetGateset, self.ds, self.gs_clgst,
                            self.lgstStrings, False)

        results.create_full_report_pdf(filename="temp_test_files/singleReport.pdf")
        results.create_presentation_pdf(filename="temp_test_files/singleSlides.pdf")
        results.create_presentation_ppt(filename="temp_test_files/singleSlides.ppt", pptTables=True)

        results2 = pygsti.report.Results(restrictToFormats=('py','latex'))
        results2.set_template_path("/some/path/to/templates")
        results2.set_latex_cmd("myCustomLatex")

        #bad objective function name
        results_badObjective = pygsti.report.Results()
        #results_badObjective.init_single("foobar", self.targetGateset, self.ds, self.gs_clgst,
        #                                 self.lgstStrings, False)
        results_badObjective.init_Ls_and_germs("foobar", self.targetGateset, self.ds, self.gs_clgst, [0], self.germs,
                                               [self.gs_clgst], [self.lgstStrings], self.fiducials, self.fiducials, 
                                               pygsti.construction.repeat_with_max_length, True)
        
        with self.assertRaises(ValueError):
            results_badObjective.get_confidence_region(95) 
        with self.assertRaises(ValueError):
            results_badObjective.get_special('DirectLongSeqGatesets')
        with self.assertRaises(ValueError):
            results_badObjective.create_full_report_pdf(filename="temp_test_files/badReport.pdf")
        with self.assertRaises(ValueError):
            results_badObjective.create_presentation_pdf(filename="temp_test_files/badSlides.pdf")
        with self.assertRaises(ValueError):
            results_badObjective.create_presentation_ppt(filename="temp_test_files/badSlides.pptx")
        
        
    

        
if __name__ == "__main__":
    unittest.main(verbosity=2)
