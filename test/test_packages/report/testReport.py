import unittest
import warnings
import collections
import pickle
import pygsti
import os
import shutil
import subprocess
from pygsti.modelpacks.legacy import std1Q_XYI as std
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
        self.assertTrue(pygsti.report.factory.find_std_clifford_compilation(std.target_model(),3) is not None)
        nonStdGS = std.target_model().rotate((0.15,-0.03,0.03))
        self.assertTrue(pygsti.report.factory.find_std_clifford_compilation(nonStdGS) is None)


    def test_reports_chi2_noCIs(self):
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
        crfact = self.results.estimates['default'].add_confidence_region_factory('go0', 'final')
        crfact.compute_hessian(comm=None)
        crfact.project_hessian('intrinsic error')

        pygsti.report.create_standard_report(self.results,temp_files + "/general_reportB",
                                            "Report B", confidenceLevel=95, verbosity=3,  auto_open=False)
        #Compare the html files?
        #self.checkFile("general_reportB%s.html" % vs)


    def test_reports_chi2_nonMarkCIs(self):
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
        #Also test adding a model-test estimate to this report
        mdl_guess = std.target_model().depolarize(op_noise=0.07,spam_noise=0.03)
        results = self.results_logL.copy()
        results.add_model_test(std.target_model(), mdl_guess, estimate_key='Test', gauge_opt_keys="auto")


        #Note: this report will have (un-combined) Robust estimates too
        pygsti.report.create_standard_report(results,temp_files + "/general_reportC",
                                             "Report C", confidenceLevel=None, verbosity=3,  auto_open=False,
                                             advancedOptions={'combine_robust': False} )
        #Compare the html files?
        #self.checkFile("general_reportC%s.html" % vs)


    def test_reports_logL_TP_wCIs(self):
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
        mdl_tgt = std.target_model()

        #Mimic factory report creation to test "inline" rendering of switchboards, tables, and figures:
        qtys = {}
        qtys['title'] = "Test Inline Report"
        qtys['date'] = "THE DATE"
        qtys['pdfinfo'] = "PDFINFO"

        ws = pygsti.report.Workspace()
        printer = pygsti.obj.VerbosityPrinter(1)
        qtys['targetGatesBoxTable'] = ws.GatesTable(mdl_tgt, display_as="boxes")

        # 3) populate template file => report file
        linkto = ()
        if bLatex: linkto = ('tex', 'pdf') + linkto #Note: can't render as 'tex' without matplotlib b/c of figs
        if bPandas: linkto = ('pkl',) + linkto
        toggles = {'CompareDatasets': False, 'ShowScaling': False, 'CombineRobust': True }
        if os.path.exists(temp_files + "/inline_report.html.files"):
            shutil.rmtree(temp_files + "/inline_report.html.files") #clear figures directory
        pygsti.report.merge_helpers.merge_jinja_template(qtys, temp_files + "/inline_report.html",
                                                         templateDir=compare_files, templateName="report_dashboard_template.html",
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


        # ---- fill_std_qtys ---- Not a function anymore
        #qtys = {}
        #mh.fill_std_qtys(qtys, connected=True, renderMath=True, CSSnames=[]) #test connected=True case HERE

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
