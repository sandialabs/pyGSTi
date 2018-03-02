import unittest
import warnings
import collections
import pickle
import pygsti
import os
import shutil
import subprocess
from pygsti.construction import std1Q_XYI as std
from ..testutils import compare_files, temp_files

import numpy as np

# Inherit setup from here
from .reportBaseCase import ReportBaseCase

bLatex = bool('PYGSTI_LATEX_TESTING' in os.environ and 
              os.environ['PYGSTI_LATEX_TESTING'].lower() in ("yes","1","true"))
try:
    import pandas
    bPandas = True
except ImportError:
    bPandas = False

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

    def test_offline_zip(self):
        pygsti.report.create_offline_zip(temp_files + "/.")

    def test_failures(self):
        self.assertWarns(pygsti.report.create_general_report, self.results, temp_files + "/XXX")
        with self.assertRaises(ValueError): # backward compat catch - when forget to specify title
            pygsti.report.create_standard_report(self.results,temp_files+"/XXX", 95)

        with self.assertRaises(ValueError): #PDF report with multiple gauge opts
            pygsti.report.create_standard_report(self.results,temp_files + "/XXX.pdf")

    def test_std_clifford_comp(self):
        self.assertTrue(pygsti.report.factory.find_std_clifford_compilation(std.gs_target,3) is not None)
        nonStdGS = std.gs_target.rotate((0.15,-0.03,0.03))
        self.assertTrue(pygsti.report.factory.find_std_clifford_compilation(nonStdGS) is None)
        

    def test_reports_chi2_noCIs(self):
        vs = self.versionsuffix
        pygsti.report.create_standard_report(self.results,temp_files + "/general_reportA",
                                            confidenceLevel=None, verbosity=3,  auto_open=False) # omit title as test

        #Test advanced options
        linkto = ()
        if bLatex: linkto = ('tex','pdf') + linkto #Note: can't render as 'tex' without matplotlib b/c of figs
        if bPandas: linkto = ('pkl',) + linkto
        results_odict = collections.OrderedDict([("One", self.results), ("Two",self.results)])
        pygsti.report.create_standard_report(results_odict,temp_files + "/general_reportA_adv1",
                                             confidenceLevel=None, verbosity=3,  auto_open=False,
                                             advancedOptions={'errgen_type': "logG-logT",
                                                              'precision': {'normal': 2, 'polar': 1, 'sci': 1}},
                                             link_to=linkto)
        
        pygsti.report.create_standard_report({"One": self.results, "Two": self.results_logL},temp_files + "/general_reportA_adv2",
                                             confidenceLevel=None, verbosity=3,  auto_open=False,
                                             advancedOptions={'errgen_type': "logTiG",
                                                              'precision': 2, #just a single int
                                                              'resizable': False,
                                                              'autosize': 'none'})

        #test latex reporting
        if bLatex:
            pygsti.report.create_standard_report(self.results.view("default","go0"),temp_files + "/general_reportA.pdf",
                                                 confidenceLevel=None, verbosity=3,  auto_open=False)

        

        #Compare the html files?
        #self.checkFile("general_reportA%s.html" % vs)


    def test_reports_chi2_wCIs(self):
        vs = self.versionsuffix
        crfact = self.results.estimates['default'].add_confidence_region_factory('go0', 'final')
        crfact.compute_hessian(comm=None)
        crfact.project_hessian('intrinsic error')

        pygsti.report.create_standard_report(self.results,temp_files + "/general_reportB",
                                            "Report B", confidenceLevel=95, verbosity=3,  auto_open=False)
        #Compare the html files?
        #self.checkFile("general_reportB%s.html" % vs)


    def test_reports_chi2_nonMarkCIs(self):
        vs = self.versionsuffix
        crfact = self.results.estimates['default'].add_confidence_region_factory('go0', 'final')
        crfact.compute_hessian(comm=None)
        crfact.project_hessian('std')

        #Note: Negative confidence levels no longer trigger non-mark error bars; this is done via "nm threshold"
        pygsti.report.create_standard_report(self.results,temp_files + "/general_reportE",
                                             "Report E", confidenceLevel=95, verbosity=3,  auto_open=False,
                                             advancedOptions={'nm threshold': -10})
        #Compare the html files?
        #self.checkFile("general_reportC%s.html" % vs)


    def test_reports_logL_TP_noCIs(self):
        vs = self.versionsuffix

        #Also test adding a model-test estimate to this report
        gs_guess = std.gs_target.depolarize(gate_noise=0.07,spam_noise=0.03)
        results = self.results_logL.copy()
        results.add_model_test(std.gs_target, gs_guess, estimate_key='Test', gauge_opt_keys="auto")

        
        #Note: this report will have (un-combined) Robust estimates too
        pygsti.report.create_standard_report(results,temp_files + "/general_reportC",
                                             "Report C", confidenceLevel=None, verbosity=3,  auto_open=False,
                                             advancedOptions={'combine_robust': False} )
        #Compare the html files?
        #self.checkFile("general_reportC%s.html" % vs)


    def test_reports_logL_TP_wCIs(self):
        vs = self.versionsuffix

        #Use propagation method instead of directly computing a factory for the go0 gauge-opt
        crfact = self.results.estimates['default'].add_confidence_region_factory('final iteration estimate', 'final')
        crfact.compute_hessian(comm=None)

        self.results.estimates['default'].gauge_propagate_confidence_region_factory('go0') #instead of computing one        
        crfact = self.results.estimates['default'].get_confidence_region_factory('go0') #was created by propagation
        crfact.project_hessian('optimal gate CIs')

        #Note: this report will have Robust estimates too
        pygsti.report.create_standard_report(self.results_logL,temp_files + "/general_reportD",
                                             "Report D", confidenceLevel=95, verbosity=3,  auto_open=False)
        #Compare the html files?
        #self.checkFile("general_reportD%s.html" % vs)

    def test_reports_multiple_ds(self):
        vs = self.versionsuffix
        
        #Note: this report will have (un-combined) Robust estimates too
        pygsti.report.create_standard_report({"chi2": self.results, "logl": self.results_logL},
                                             temp_files + "/general_reportF",
                                             "Report F", confidenceLevel=None, verbosity=3,  auto_open=False)
        #Compare the html files?
        #self.checkFile("general_reportC%s.html" % vs)


    def test_report_notebook(self):
        pygsti.report.create_report_notebook(self.results_logL, temp_files + "/report_notebook.ipynb", None,
                                             verbosity=3)
        pygsti.report.create_report_notebook({'one': self.results_logL, 'two': self.results_logL},
                                             temp_files + "/report_notebook.ipynb", None,
                                             verbosity=3) # multiple comparable datasets
        

    def test_inline_template(self):
        #Generate some results (quickly)
        gs_tgt = std.gs_target.copy()
        gs_datagen = gs_tgt.depolarize(gate_noise=0.01,spam_noise=0.01)
        gateStrings = pygsti.construction.make_lsgst_experiment_list(
            gs_tgt, std.fiducials, std.fiducials, std.germs,[1])
        ds = pygsti.construction.generate_fake_data(
            gs_datagen, gateStrings, nSamples=10000, sampleError='round')
        gs_test = gs_tgt.depolarize(gate_noise=0.01,spam_noise=0.01)
        results = pygsti.do_model_test(gs_test, ds, gs_tgt, std.fiducials, std.fiducials, std.germs, [1])
        
        #Mimic factory report creation to test "inline" rendering of switchboards, tables, and figures:
        qtys = {}
        qtys['title'] = "Test Inline Report"
        qtys['date'] = "THE DATE"
        qtys['confidenceLevel'] = "NOT-SET"
        qtys['linlg_pcntle'] = "95"
        qtys['linlg_pcntle_inv'] = "5"
        #qtys['errorgenformula'], qtys['errorgendescription'] = _errgen_formula(errgen_type, fmt)

        qtys['pdfinfo'] = "PDFINFO"

        # Generate Switchboard
        ws = pygsti.report.Workspace()
        printer = pygsti.obj.VerbosityPrinter(1)
        switchBd, dataset_labels, est_labels, gauge_opt_labels, Ls, swLs = \
            pygsti.report.factory._create_master_switchboard(ws, {'MyTest': results}, None, 10, 
                                                             printer, 'html', False)

        gsTgt = switchBd.gsTarget
        ds = switchBd.ds
        eff_ds = switchBd.eff_ds
        modvi_ds = switchBd.modvi_ds
        prepStrs = switchBd.prepStrs
        effectStrs = switchBd.effectStrs
        germs = switchBd.germs
        strs = switchBd.strs
        cliffcomp = switchBd.clifford_compilation
 
        def addqty(b, name, fn, *args, **kwargs):
            qtys[name] = fn(*args, **kwargs)
                
        addqty(2,'targetSpamBriefTable', ws.SpamTable, gsTgt, None, display_as='boxes', includeHSVec=False)
        addqty(2,'targetGatesBoxTable', ws.GatesTable, gsTgt, display_as="boxes")
        addqty(2,'datasetOverviewTable', ws.DataSetOverviewTable, ds)

        gsFinal = switchBd.gsFinal
        gsGIRep = switchBd.gsGIRep
        gsEP = switchBd.gsGIRepEP
        cri = None

        addqty(4,'bestGatesetSpamParametersTable', ws.SpamParametersTable, switchBd.gsTargetAndFinal,
               ['Target','Estimated'], cri )
        addqty(4,'bestGatesetSpamBriefTable', ws.SpamTable, switchBd.gsTargetAndFinal,
               ['Target','Estimated'], 'boxes', cri, includeHSVec=False)
        addqty(4,'bestGatesetSpamVsTargetTable', ws.SpamVsTargetTable, gsFinal, gsTgt, cri)
        addqty(4,'bestGatesetGaugeOptParamsTable', ws.GaugeOptParamsTable, switchBd.goparams)
        addqty(4,'bestGatesetGatesBoxTable', ws.GatesTable, switchBd.gsTargetAndFinal,
               ['Target','Estimated'], "boxes", cri)

        # Generate plots                                                                                                                     
        addqty(4,'gramBarPlot', ws.GramMatrixBarPlot, ds,gsTgt,10,strs)
        
        # 3) populate template file => report file                                                                                        
        templateFile = "../../../../test/test_packages/cmp_chk_files/report_dashboard_template.html"
          # trickery to use a template in nonstadard location
        linkto = ()
        if bLatex: linkto = ('tex','pdf') + linkto #Note: can't render as 'tex' without matplotlib b/c of figs
        if bPandas: linkto = ('pkl',) + linkto
        toggles = {'CompareDatasets': False, 'ShowScaling': False, 'CombineRobust': True }
        if os.path.exists(temp_files + "/inline_report.html.files"):
            shutil.rmtree(temp_files + "/inline_report.html.files") #clear figures directory
        pygsti.report.merge_helpers.merge_html_template(qtys, templateFile, temp_files + "/inline_report.html",
                                                        auto_open=False, precision=None, link_to=linkto,
                                                        connected=False, toggles=toggles, renderMath=True,
                                                        resizable=True, autosize='none', verbosity=printer)


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

        from pygsti.report.convert import converter
        specs = dict(longtables=False, tableID=None, tableclass=None,
               scratchDir=None, precision=6, polarprecision=3, sciprecision=0,
               resizable=False, autosize=False, fontsize=None, complexAsPolar=True,
               brackets=False)
        html  = converter('html')  # Retrieve low-level formatters
        latex = converter('latex')

        print("Float formatting")
        html(f, specs)
        latex(f, specs)

        print("List formatting")
        html(l, specs)
        latex(l, specs)

        print("Arbitrary class formatting")
        html(w, specs)
        latex(w, specs)

        print("Vector formatting")
        html(vec, specs)
        latex(vec, specs)

        print("Vector formatting (w/brackets)")
        specs['brackets'] = True
        html(vec, specs)
        latex(vec, specs)
        specs['brackets'] = False

        print("Matrix formatting")
        specs['fontsize'] = 8
        html(mx, specs)
        latex(mx, specs)
        specs['fontsize'] = None

        print("Value formatting")
        specs['precision'] = 2
        specs['complexAsPolar'] = True
        for complxAsPolar in (True,False):
            for x in (0.001,0.01,1.0,10.0,100.0,1000.0,10000.0,1.0+1.0j,10j,1.0+1e-10j,1e-10j,"N/A"):
                html(x, specs)
                latex(x, specs)

        with self.assertRaises(ValueError):
            html(rank3Tensor, specs)

        with self.assertRaises(ValueError):
            latex(rank3Tensor, specs)

    def test_factory_helpers(self):
        pygsti.report.factory._errgen_formula("logTiG", "latex")
        pygsti.report.factory._errgen_formula("logGTi", "latex")
        pygsti.report.factory._errgen_formula("logG-logT", "latex")
        pygsti.report.factory._errgen_formula("foobar", "latex")

    def test_merge_helpers(self):
        """ Tests boundary cases for merge_helpers.py functinos """
        import pygsti.report.merge_helpers as mh

        # ---- insert_resource ----
        mh.insert_resource(connected=True, online_url="http://myurl.com/myfile.js",
                           offline_filename="myOfflineFile.js")
        mh.insert_resource(connected=True, online_url="http://myurl.com/myfile.js",
                           offline_filename=None, integrity="TEST", crossorigin="TEST")

        with self.assertRaises(ValueError):
            mh.insert_resource(connected=True, online_url=None, offline_filename="myOfflineFile.foobar") 
            #unknown resource type (extension)
        with self.assertRaises(ValueError):
            mh.insert_resource(connected=False, online_url=None, offline_filename="myOfflineFile.foobar") 
            #unknown resource type (extension)

        # ---- rsync_offline_dir ----
        outputDir = temp_files + "/rsync_offline_testdir"
        if os.path.exists(outputDir):
            shutil.rmtree(outputDir) #make sure no directory exists
        mh.rsync_offline_dir(outputDir) #creates outputDir
        os.remove(os.path.join(outputDir, "offline/README.txt")) # remove a single file
        mh.rsync_offline_dir(outputDir) #creates *only* the single file removed

        # ---- read_and_preprocess_template ----
        tmpl = "#iftoggle(tname)\nSomething\n#elsetoggle\nNO END TOGGLE!"
        with open(temp_files + "/test_toggles.txt","w") as f:
            f.write(tmpl)
        with self.assertRaises(AssertionError):
            mh.read_and_preprocess_template(temp_files + "/test_toggles.txt", {'tname': True}) # no #endtoggle

        tmpl = "#iftoggle(tname)\nSomething\nNO ELSE OR END TOGGLE!"
        with open(temp_files + "/test_toggles.txt","w") as f:
            f.write(tmpl)
        with self.assertRaises(AssertionError):
            mh.read_and_preprocess_template(temp_files + "/test_toggles.txt", {'tname': True}) # no #elsetoggle or #endtoggle

        # ---- makeEmptyDir ----
        dirname = temp_files + "/empty_testdir"
        if os.path.exists(dirname):
            shutil.rmtree(dirname) #make sure no directory exists
        mh.makeEmptyDir(dirname)


        # ---- fill_std_qtys ----
        qtys = {}
        mh.fill_std_qtys(qtys, connected=True, renderMath=True, CSSnames=[]) #test connected=True case

        # ---- evaluate_call ----
        printer = pygsti.obj.VerbosityPrinter(1)
        stdout, stderr, returncode = mh.process_call(['ls','foobar.foobar'])
        with self.assertRaises(subprocess.CalledProcessError):
            mh.evaluate_call(['ls','foobar.foobar'], stdout, stderr, returncode, printer)

        # ---- to_pdfinfo ----
        pdfinfo =  mh.to_pdfinfo([ ('key','value'),
                                   ('key2', ("TUP","LE")),
                                   ('key3', collections.OrderedDict([("One", 1), ("Two",1)])) ])
        self.assertEqual(pdfinfo, "key={value},\nkey2={[TUP, LE]},\nkey3={Dict[One: 1, Two: 1]}")

        
        
        

#Test functions within reportables separately? This version of the test is outdated:
#    def test_reportables(self):
#        #Test that None is returned when qty cannot be computed
#        qty = pygsti.report.reportables.compute_dataset_qty("FooBar",self.ds)
#        self.assertIsNone(qty)
#        qty = pygsti.report.reportables.compute_gateset_qty("FooBar",self.gs_clgst)
#        self.assertIsNone(qty)
#        qty = pygsti.report.reportables.compute_gateset_dataset_qty("FooBar",self.gs_clgst, self.ds)
#        self.assertIsNone(qty)
#        qty = pygsti.report.reportables.compute_gateset_gateset_qty("FooBar",self.gs_clgst, self.gs_clgst)
#        self.assertIsNone(qty)
#
#        #test ignoring gate strings not in dataset
#        qty = pygsti.report.reportables.compute_dataset_qty("gate string length", self.ds,
#                                                            pygsti.construction.gatestring_list([('Gx','Gx'),('Gfoobar',)]) )
#        qty = pygsti.report.reportables.compute_gateset_dataset_qty("prob(0) diff", self.gs_clgst, self.ds,
#                                                            pygsti.construction.gatestring_list([('Gx','Gx'),('Gfoobar',)]) )
#        qty_str = str(qty) #test __str__
#
#        #Test gateset gates mismatch
#        from pygsti.construction import std1Q_XY as stdXY
#        with self.assertRaises(ValueError):
#            qty = pygsti.report.reportables.compute_gateset_gateset_qty(
#                "Gx fidelity",std.gs_target, stdXY.gs_target) #Gi missing from 2nd gateset
#        with self.assertRaises(ValueError):
#            qty = pygsti.report.reportables.compute_gateset_gateset_qty(
#                "Gx fidelity",stdXY.gs_target, std.gs_target) #Gi missing from 1st gateset



#def test_results_object(self):
#    results = pygsti.report.Results()
#    results.init_single("logl", self.targetGateset, self.ds, self.gs_clgst,
#                        self.lgstStrings, self.targetGateset)
#
#    results.parameters.update(
#        {'minProbClip': 1e-6, 'minProbClipForWeighting': 1e-4,
#         'probClipInterval': (-1e6,1e6), 'radius': 1e-4,
#         'weights': None, 'defaultDirectory': temp_files + "",
#         'defaultBasename': "MyDefaultReportName",
#         'hessianProjection': 'std'} )
#
#    results.create_full_report_pdf(
#        filename=temp_files + "/singleReport.pdf")
#    results.create_brief_report_pdf(
#        filename=temp_files + "/singleBrief.pdf")
#    results.create_presentation_pdf(
#        filename=temp_files + "/singleSlides.pdf")
#    if self.have_python_pptx:
#        results.create_presentation_ppt(
#            filename=temp_files + "/singleSlides.ppt", pptTables=True)
#
#    #test tree splitting of hessian
#    results.parameters['memLimit'] = 10*(1024)**2 #10MB
#    results.create_brief_report_pdf(confidenceLevel=95,
#        filename=temp_files + "/singleBriefMemLimit.pdf")
#    results.parameters['memLimit'] = 10 #10 bytes => too small
#    with self.assertRaises(MemoryError):
#        results.create_brief_report_pdf(confidenceLevel=90,
#           filename=temp_files + "/singleBriefMemLimit.pdf")
#
#
#    #similar test for chi2 hessian
#    results2 = pygsti.report.Results()
#    results2.init_single("chi2", self.targetGateset, self.ds, self.gs_clgst,
#                        self.lgstStrings, self.targetGateset)
#    results2.parameters.update(
#        {'minProbClip': 1e-6, 'minProbClipForWeighting': 1e-4,
#         'probClipInterval': (-1e6,1e6), 'radius': 1e-4,
#         'weights': None, 'defaultDirectory': temp_files + "",
#         'defaultBasename': "MyDefaultReportName",
#         'hessianProjection': "std"} )
#    results2.parameters['memLimit'] = 10*(1024)**2 #10MB
#    results2.create_brief_report_pdf(confidenceLevel=95,
#        filename=temp_files + "/singleBriefMemLimit2.pdf")
#    results2.parameters['memLimit'] = 10 #10 bytes => too small
#    with self.assertRaises(MemoryError):
#        results2.create_brief_report_pdf(confidenceLevel=90,
#           filename=temp_files + "/singleBriefMemLimit2.pdf")
#
#
#
#
#    results_str = str(results)
#    tableNames = list(results.tables.keys())
#    figNames = list(results.figures.keys())
#    for g in results.gatesets:
#        s = str(g)
#    for g in results.gatestring_lists:
#        s = str(g)
#    s = str(results.dataset)
#    s = str(results.options)
#
#    self.assertTrue(tableNames[0] in results.tables)
#
#    with self.assertRaises(KeyError):
#        x = results.tables.get('foobar')
#    with self.assertRaises(ValueError):
#        results.tables['newKey'] = "notAllowed"
#    with self.assertRaises(NotImplementedError):
#        for x in results.tables: # cannot __iter__
#            print(x)
#    with self.assertRaises(NotImplementedError):
#        for x in results.tables.iteritems(): # cannot iter
#            print(x)
#    with self.assertRaises(NotImplementedError):
#        for x in list(results.tables.values()): # cannot iter
#            print(x)
#
#    pkl = pickle.dumps(results)
#    results_copy = pickle.loads(pkl)
#    self.assertEqual(tableNames, list(results_copy.tables.keys()))
#    self.assertEqual(figNames, list(results_copy.figures.keys()))
#    #self.assertEqual(results.options, results_copy.options) #need to add equal test to ResultsOptions
#    self.assertEqual(results.parameters, results_copy.parameters)
#
#    results2 = pygsti.report.Results()
#    results2.options.template_path = "/some/path/to/templates"
#    results2.options.latex_cmd = "myCustomLatex"
#
#    #bad objective function name
#    results_badObjective = pygsti.report.Results()
#    #results_badObjective.init_single("foobar", self.targetGateset, self.ds, self.gs_clgst,
#    #                                 self.lgstStrings)
#    results_badObjective.init_Ls_and_germs("foobar", self.targetGateset, self.ds, self.gs_clgst, [0], self.germs,
#                                           [self.gs_clgst], [self.lgstStrings], self.fiducials, self.fiducials,
#                                           pygsti.construction.repeat_with_max_length, True)
#
#    with self.assertRaises(ValueError):
#        results_badObjective._get_confidence_region(95)
#    with self.assertRaises(ValueError):
#        results_badObjective._specials['DirectLongSeqGatesets']
#    with self.assertRaises(ValueError):
#        results_badObjective.create_full_report_pdf(filename=temp_files + "/badReport.pdf")
#    with self.assertRaises(ValueError):
#        results_badObjective.create_presentation_pdf(filename=temp_files + "/badSlides.pdf")
#    if self.have_python_pptx:
#        with self.assertRaises(ValueError):
#            results_badObjective.create_presentation_ppt(filename=temp_files + "/badSlides.pptx")


if __name__ == "__main__":
    unittest.main(verbosity=2)
