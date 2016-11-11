import unittest
import warnings
import pickle
import pygsti
import os
from pygsti.construction import std1Q_XYI as std
from ..testutils import compare_files, temp_files

import numpy as np

# Inherit setup from here
from .reportBaseCase import ReportBaseCase

class TestReport(ReportBaseCase):

    def checkFile(self, fn):

        if 'PYGSTI_DEEP_TESTING' in os.environ and \
           os.environ['PYGSTI_DEEP_TESTING'].lower() in ("yes","1","true"):
            # Deep testing -- do latex comparison
            cmpFilenm = compare_files + "/%s" % fn
            if os.path.exists(cmpFilenm):
                linesToTest = open(temp_files + "/%s" % fn).readlines()
                linesOK = open(cmpFilenm).readlines()
                self.assertEqual(linesToTest,linesOK)
            else:
                print("WARNING! No reference file to compare against: %s" % cmpFilenm)
        else:
            # Normal testing -- no latex comparison
            pass

    def test_reports_chi2_noCIs(self):

        vs = self.versionsuffix
        self.results.create_full_report_pdf(filename=temp_files + "/full_reportA%s.pdf" % vs, confidenceLevel=None,
                                         debugAidsAppendix=False, gaugeOptAppendix=False,
                                         pixelPlotAppendix=False, whackamoleAppendix=False)
        self.results.create_brief_report_pdf(filename=temp_files + "/brief_reportA%s.pdf" % vs, confidenceLevel=None)
        self.results.create_presentation_pdf(filename=temp_files + "/slidesA%s.pdf" % vs, confidenceLevel=None,
                                           debugAidsAppendix=False, pixelPlotAppendix=False, whackamoleAppendix=False)
        if self.have_python_pptx:
            self.results.create_presentation_ppt(filename=temp_files + "/slidesA.ppt", confidenceLevel=None,
                                                 debugAidsAppendix=False, pixelPlotAppendix=False, whackamoleAppendix=False)

        #Run again using default filenames
        self.results.create_full_report_pdf(filename="auto", confidenceLevel=None,
                                         debugAidsAppendix=False, gaugeOptAppendix=False,
                                         pixelPlotAppendix=False, whackamoleAppendix=False)
        self.results.create_brief_report_pdf(filename="auto", confidenceLevel=None)
        self.results.create_presentation_pdf(filename="auto", confidenceLevel=None,
                                           debugAidsAppendix=False, pixelPlotAppendix=False, whackamoleAppendix=False)
        if self.have_python_pptx:
            self.results.create_presentation_ppt(filename="auto", confidenceLevel=None,
                                                 debugAidsAppendix=False, pixelPlotAppendix=False, whackamoleAppendix=False)

        #Compare the text files, assume if these match the PDFs are equivalent
        self.checkFile("full_reportA%s.tex" % vs)
        self.checkFile("brief_reportA%s.tex" % vs)
        self.checkFile("slidesA%s.tex" % vs)



    def test_reports_chi2_wCIs(self):
        vs = self.versionsuffix

        self.results.create_full_report_pdf(filename=temp_files + "/full_reportB%s.pdf" % vs, confidenceLevel=95,
                                         debugAidsAppendix=True, gaugeOptAppendix=True,
                                         pixelPlotAppendix=True, whackamoleAppendix=True,
                                         verbosity=2)
        self.results.create_full_report_pdf(filename=temp_files + "/full_reportB-noGOpt%s.pdf" % vs, confidenceLevel=95,
                                         debugAidsAppendix=True, gaugeOptAppendix=False,
                                         pixelPlotAppendix=True, whackamoleAppendix=True) # to test blank GOpt tables
        self.results.create_brief_report_pdf(filename=temp_files + "/brief_reportB%s.pdf" % vs, confidenceLevel=95, verbosity=2)
        self.results.create_presentation_pdf(filename=temp_files + "/slidesB%s.pdf" % vs, confidenceLevel=95,
                                           debugAidsAppendix=True, pixelPlotAppendix=True, whackamoleAppendix=True,
                                           verbosity=2)
        if self.have_python_pptx:
            self.results.create_presentation_ppt(filename=temp_files + "/slidesB%s.ppt" % vs, confidenceLevel=95,
                                                 debugAidsAppendix=True, pixelPlotAppendix=True, whackamoleAppendix=True,
                                                 verbosity=2)

        #Compare the text files, assume if these match the PDFs are equivalent
        self.checkFile("full_reportB%s.tex" % vs)
        self.checkFile("full_reportB%s_appendices.tex" % vs)
        self.checkFile("brief_reportB%s.tex" % vs)
        self.checkFile("slidesB%s.tex" % vs)


    def test_reports_chi2_nonMarkCIs(self):
        vs = self.versionsuffix

        #Non-markovian error bars (negative confidenceLevel) & tooltips
        self.results.create_full_report_pdf(filename=temp_files + "/full_reportE%s.pdf" % vs, confidenceLevel=-95,
                                         debugAidsAppendix=True, gaugeOptAppendix=True,
                                         pixelPlotAppendix=True, whackamoleAppendix=True,
                                         verbosity=2, tips=True)
        self.results.create_brief_report_pdf(filename=temp_files + "/brief_reportE%s.pdf" % vs, confidenceLevel=-95,
                                             verbosity=2, tips=True)

        #Compare the text files, assume if these match the PDFs are equivalent
        self.checkFile("full_reportE%s.tex" % vs)
        self.checkFile("full_reportE%s_appendices.tex" % vs)


    def test_reports_logL_TP_noCIs(self):

        vs = self.versionsuffix
        self.results_logL.create_full_report_pdf(filename=temp_files + "/full_reportC%s.pdf" % vs, confidenceLevel=None,
                                                 debugAidsAppendix=False, gaugeOptAppendix=False,
                                                 pixelPlotAppendix=False, whackamoleAppendix=False,
                                                 verbosity=2)
        self.results_logL.create_brief_report_pdf(filename=temp_files + "/brief_reportC%s.pdf" % vs, confidenceLevel=None, verbosity=2)
        self.results_logL.create_presentation_pdf(filename=temp_files + "/slidesC%s.pdf" % vs, confidenceLevel=None,
                                                  debugAidsAppendix=False, pixelPlotAppendix=False, whackamoleAppendix=False,
                                                  verbosity=2)
        self.results_logL.create_general_report_pdf(filename=temp_files + "/general_reportC%s.pdf" % vs, confidenceLevel=None,
                                                    verbosity=2, showAppendix=True)

        if self.have_python_pptx:
            self.results_logL.create_presentation_ppt(filename=temp_files + "/slidesC%s.ppt" % vs, confidenceLevel=None,
                                                      debugAidsAppendix=False, pixelPlotAppendix=False, whackamoleAppendix=False,
                                                      verbosity=2)
        
        ##Compare the text files, assume if these match the PDFs are equivalent
        self.checkFile("full_reportC%s.tex" % vs)
        self.checkFile("brief_reportC%s.tex" % vs)
        self.checkFile("slidesC%s.tex" % vs)
        #self.checkFile("general_reportC%s.tex" % vs)


    def test_reports_logL_TP_wCIs(self):

        vs = self.versionsuffix
        self.results_logL.create_full_report_pdf(filename=temp_files + "/full_reportD%s.pdf" % vs, confidenceLevel=95,
                                                 debugAidsAppendix=True, gaugeOptAppendix=True,
                                                 pixelPlotAppendix=True, whackamoleAppendix=True,
                                                 verbosity=2)
        self.results_logL.create_brief_report_pdf(filename=temp_files + "/brief_reportD%s.pdf" % vs, confidenceLevel=95, verbosity=2)
        self.results_logL.create_presentation_pdf(filename=temp_files + "/slidesD%s.pdf" % vs, confidenceLevel=95,
                                                  debugAidsAppendix=True, pixelPlotAppendix=True, whackamoleAppendix=True,
                                                  verbosity=2)
        self.results_logL.create_general_report_pdf(filename=temp_files + "/general_reportD%s.pdf" % vs, confidenceLevel=95,
                                                    verbosity=2, tips=True) #test tips here too

        if self.have_python_pptx:
            self.results_logL.create_presentation_ppt(filename=temp_files + "/slidesD%s.ppt" % vs, confidenceLevel=95,
                                                      debugAidsAppendix=True, pixelPlotAppendix=True, whackamoleAppendix=True,
                                                      verbosity=2)

        ##Compare the text files, assume if these match the PDFs are equivalent
        self.checkFile("full_reportD%s.tex" % vs)
        self.checkFile("full_reportD%s_appendices.tex" % vs)
        self.checkFile("brief_reportD%s.tex" % vs)
        self.checkFile("slidesD%s.tex" % vs)
        #self.checkFile("general_reportD%s.tex" % vs)



    def test_extra_logL_TP_figs_and_tables(self):

        #Run a few tests to generate tables & figures we don't use in reports
        self.results_logL.tables["chi2ProgressTable"]
        self.results_logL.tables["logLProgressTable"]
        self.results_logL.figures["bestEstimateSummedColorBoxPlot"]
        self.results_logL.figures["blankBoxPlot"]
        self.results_logL.figures["blankSummedBoxPlot"]
        self.results_logL.figures["directLGSTColorBoxPlot"]
        self.results_logL.figures["directLGSTDeviationColorBoxPlot"]
        with self.assertRaises(KeyError):
            self.results_logL.figures["FooBar"]
        with self.assertRaises(KeyError):
            self.results_logL._specials['FooBar']

        #Run tests to generate tables we don't use in reports
        self.results_logL.tables["bestGatesetVsTargetAnglesTable"]



    def test_table_generation(self):
        import pygsti.report.generation as gen
        formats = ['latex','html','test','ppt'] #all the formats we know
        tableclass = "cssClass"
        longtable = False
        confidenceRegionInfo = None

        gateset = pygsti.io.load_gateset(compare_files + "/analysis.gateset")
        ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/analysis.dataset")

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
            prepLabels=['rho0'], prepExpressions=["0"], effectLabels=['E0','E1','E2'], effectExpressions=["0","1","2"],
            spamdefs={'upup': ('rho0','E0'), 'updn': ('rho0','E1'), 'dnup': ('rho0','E2'),
                           'dndn': ('rho0','remainder') }, basis="pp" )


        #tests which fill in the cracks of the full-report tests
        tab = gen.get_gateset_spam_table(gateset, None)
        tab_wCI = gen.get_gateset_spam_table(gateset, ci)
        table_wCI_as_str = str(tab_wCI)

        gen.get_gateset_spam_table(gateset, None)
        gen.get_gateset_gates_table(gateset_tp, ci_TP) #test zero-padding
        gen.get_unitary_gateset_gates_table(std.gs_target, ci_tgt) #unitary gates w/CIs
        gen.get_unitary_gateset_gates_table(target_tp, ci_TP_tgt) #unitary gates w/CIs
        gen.get_gateset_closest_unitary_table(gateset_2q, None) #test higher-dim gateset
        gen.get_gateset_closest_unitary_table(gateset, ci) #test with CIs (long...)
        gen.get_gateset_rotn_axis_table(std.gs_target, None) #try to get "--"s and "X"s to display
        gen.get_chi2_progress_table([0], [gateset_tp], [ [('Gx',)],], ds) #TP case
        gen.get_chi2_confidence_region(gateset_tp, ds, 95) #TP case

        gen.get_gatestring_multi_table([ [('Gx',),('Gz',)], [('Gy',)] ],
                                       ["list1","list2"], commonTitle=None) #commonTitle == None case w/diff length lists

        with self.assertRaises(ValueError):
            gen.get_unitary_gateset_gates_table(std.gs_target, ci) #gateset-CI mismatch
        with self.assertRaises(ValueError):
            gen.get_gateset_spam_parameters_table(std.gs_target, ci) #gateset-CI mismatch




        #LogL case tests
        gen.get_logl_progress_table([0], [gateset_tp], [ [('Gx',)],], ds) # logL case
        gen.get_logl_progress_table([0], [gateset], [ [('Gx',)],], ds) # logL case

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

        print("Float formatting")
        pygsti.report.html.html(f)
        pygsti.report.latex.latex(f)
        pygsti.report.ppt.ppt(f)

        print("List formatting")
        pygsti.report.html.html(l)
        pygsti.report.latex.latex(l)
        pygsti.report.ppt.ppt(l)

        print("Arbitrary class formatting")
        pygsti.report.html.html(w)
        pygsti.report.latex.latex(w)
        pygsti.report.ppt.ppt(w)

        print("Vector formatting")
        pygsti.report.html.html(vec)
        pygsti.report.latex.latex(vec)
        pygsti.report.ppt.ppt(vec)

        print("Vector formatting (w/brackets)")
        pygsti.report.html.html(vec, brackets=True)
        pygsti.report.latex.latex(vec, brackets=True)
        pygsti.report.ppt.ppt(vec, brackets=True)

        print("Matrix formatting")
        pygsti.report.html.html_matrix(mx, fontsize=8, brackets=False)
        pygsti.report.latex.latex_matrix(mx, fontsize=8, brackets=False)
        pygsti.report.ppt.ppt_matrix(mx, fontsize=8, brackets=False)

        print("Value formatting")
        ROUND = 2; complxAsPolar=True
        for complxAsPolar in (True,False):
            for x in (0.001,0.01,1.0,10.0,100.0,1000.0,10000.0,1.0+1.0j,10j,1.0+1e-10j,1e-10j,"N/A"):
                pygsti.report.html.html_value(x, ROUND, complxAsPolar)
                pygsti.report.latex.latex_value(x, ROUND, complxAsPolar)
                pygsti.report.ppt.ppt_value(x, ROUND, complxAsPolar)

        with self.assertRaises(ValueError):
            pygsti.report.html.html(rank3Tensor)

        with self.assertRaises(ValueError):
            pygsti.report.latex.latex(rank3Tensor)

        with self.assertRaises(ValueError):
            pygsti.report.ppt.ppt(rank3Tensor)


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
                            self.lgstStrings, False, self.targetGateset)

        results.parameters.update(
            {'minProbClip': 1e-6, 'minProbClipForWeighting': 1e-4,
             'probClipInterval': (-1e6,1e6), 'radius': 1e-4,
             'weights': None, 'defaultDirectory': temp_files + "",
             'defaultBasename': "MyDefaultReportName",
             'hessianProjection': 'std'} )

        results.create_full_report_pdf(
            filename=temp_files + "/singleReport.pdf")
        results.create_brief_report_pdf(
            filename=temp_files + "/singleBrief.pdf")
        results.create_presentation_pdf(
            filename=temp_files + "/singleSlides.pdf")
        if self.have_python_pptx:
            results.create_presentation_ppt(
                filename=temp_files + "/singleSlides.ppt", pptTables=True)

        #test tree splitting of hessian
        results.parameters['memLimit'] = 10*(1024)**2 #10MB
        results.create_brief_report_pdf(confidenceLevel=95,
            filename=temp_files + "/singleBriefMemLimit.pdf")
        results.parameters['memLimit'] = 10 #10 bytes => too small
        with self.assertRaises(MemoryError):
            results.create_brief_report_pdf(confidenceLevel=90,
               filename=temp_files + "/singleBriefMemLimit.pdf")


        #similar test for chi2 hessian
        results2 = pygsti.report.Results()
        results2.init_single("chi2", self.targetGateset, self.ds, self.gs_clgst,
                            self.lgstStrings, False, self.targetGateset)
        results2.parameters.update(
            {'minProbClip': 1e-6, 'minProbClipForWeighting': 1e-4,
             'probClipInterval': (-1e6,1e6), 'radius': 1e-4,
             'weights': None, 'defaultDirectory': temp_files + "",
             'defaultBasename': "MyDefaultReportName",
             'hessianProjection': "std"} )
        results2.parameters['memLimit'] = 10*(1024)**2 #10MB
        results2.create_brief_report_pdf(confidenceLevel=95,
            filename=temp_files + "/singleBriefMemLimit2.pdf")
        results2.parameters['memLimit'] = 10 #10 bytes => too small
        with self.assertRaises(MemoryError):
            results2.create_brief_report_pdf(confidenceLevel=90,
               filename=temp_files + "/singleBriefMemLimit2.pdf")




        results_str = str(results)
        tableNames = list(results.tables.keys())
        figNames = list(results.figures.keys())
        for g in results.gatesets:
            s = str(g)
        for g in results.gatestring_lists:
            s = str(g)
        s = str(results.dataset)
        s = str(results.options)

        self.assertTrue(tableNames[0] in results.tables)

        with self.assertRaises(KeyError):
            x = results.tables.get('foobar')
        with self.assertRaises(ValueError):
            results.tables['newKey'] = "notAllowed"
        with self.assertRaises(NotImplementedError):
            for x in results.tables: # cannot __iter__
                print(x)
        with self.assertRaises(NotImplementedError):
            for x in results.tables.iteritems(): # cannot iter
                print(x)
        with self.assertRaises(NotImplementedError):
            for x in list(results.tables.values()): # cannot iter
                print(x)

        pkl = pickle.dumps(results)
        results_copy = pickle.loads(pkl)
        self.assertEqual(tableNames, list(results_copy.tables.keys()))
        self.assertEqual(figNames, list(results_copy.figures.keys()))
        #self.assertEqual(results.options, results_copy.options) #need to add equal test to ResultsOptions
        self.assertEqual(results.parameters, results_copy.parameters)

        results2 = pygsti.report.Results()
        results2.options.template_path = "/some/path/to/templates"
        results2.options.latex_cmd = "myCustomLatex"

        #bad objective function name
        results_badObjective = pygsti.report.Results()
        #results_badObjective.init_single("foobar", self.targetGateset, self.ds, self.gs_clgst,
        #                                 self.lgstStrings, False)
        results_badObjective.init_Ls_and_germs("foobar", self.targetGateset, self.ds, self.gs_clgst, [0], self.germs,
                                               [self.gs_clgst], [self.lgstStrings], self.fiducials, self.fiducials,
                                               pygsti.construction.repeat_with_max_length, True)

        with self.assertRaises(ValueError):
            results_badObjective._get_confidence_region(95)
        with self.assertRaises(ValueError):
            results_badObjective._specials['DirectLongSeqGatesets']
        with self.assertRaises(ValueError):
            results_badObjective.create_full_report_pdf(filename=temp_files + "/badReport.pdf")
        with self.assertRaises(ValueError):
            results_badObjective.create_presentation_pdf(filename=temp_files + "/badSlides.pdf")
        if self.have_python_pptx:
            with self.assertRaises(ValueError):
                results_badObjective.create_presentation_ppt(filename=temp_files + "/badSlides.pptx")

    # Commented out by LSaldyt - dummy axes can't be given to figures, since they save their contents immediately to file
    #def test_report_figure_object(self):
    #    axes = {'dummy': "matplotlib axes"}
    #    fig = pygsti.report.figure.ReportFigure(axes, {})
    #    fig.set_extra_info("extra!")
    #    self.assertEqual(fig.get_extra_info(), "extra!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
