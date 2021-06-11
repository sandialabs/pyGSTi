import os
import pickle
import unittest

import numpy as np

import pygsti.protocols.estimate
from pygsti.extras import drift
from pygsti.modelpacks.legacy import stdQT_XYIMS
from ..report.reportBaseCase import ReportBaseCase
from ..testutils import compare_files, temp_files

bLatex = bool('PYGSTI_LATEX_TESTING' in os.environ and
              os.environ['PYGSTI_LATEX_TESTING'].lower() in ("yes","1","true"))
try:
    import pandas
    bPandas = True
except ImportError:
    bPandas = False

class TestWorkspace(ReportBaseCase):

    def setUp(self):
        super(TestWorkspace, self).setUp()

        self.tgt = self.results.estimates['default'].models['target']
        self.ds = self.results.dataset.copy()
        self.ds.comment = "Hello\nWorld!" # for testing DS overview table
        self.mdl = self.results.estimates['default'].models['go0']
        self.gss = self.results.circuit_lists['final']

    def test_notebook_mode(self):
        wnb = pygsti.report.Workspace()
        with self.assertRaises(ValueError):
            wnb.init_notebook_mode()

    def test_caching(self):
        ws = pygsti.report.Workspace()
        def output_fn():
            return ws.NotApplicable() # a WorkspaceOutput obj
        key, result = ws.smartCache.cached_compute(output_fn, [])
        #print(key,result)

        ws.save_cache(temp_files + "/saved_workspace_testcache.pkl", show_unpickled=True)
        ws.load_cache(temp_files + "/saved_workspace_testcache.pkl")

    def test_plotly_pickling(self):
        import plotly.graph_objs as go

        # Create random data with numpy
        N = 1000
        random_x = np.random.randn(N)
        random_y = np.random.randn(N)

        # Create a trace
        trace = go.Scatter(
            x = random_x,
            y = random_y,
            mode = 'markers'
        )
        data = [trace]
        fig = go.Figure(data=[trace], layout={})
        pygsti.report.workspace.enable_plotly_pickling()

        #DEBUG
        #Import dill
        #dill.detect.trace(True)
        #dill.detect.badobjects(fig['frames'], depth=1)
        #print(dir(fig))
        #print("frames: ",type(fig['frames']))
        #s = pickle.dumps(fig['frames'])
        #print("layout",type(fig['layout']))
        #s = pickle.dumps(fig['layout'])
        #print("data",type(fig['data']))
        #s = pickle.dumps(fig['data'])

        s = pickle.dumps(fig)
        fig2 = pickle.loads(s)
        pygsti.report.workspace.disable_plotly_pickling()

        #Again for good measure
        pygsti.report.workspace.enable_plotly_pickling()
        pygsti.report.workspace.disable_plotly_pickling()

    def test_custom_digest(self):
        import hashlib
        M = hashlib.md5()
        ws = pygsti.report.Workspace()

        NA = ws.NotApplicable()
        pygsti.report.workspace.ws_custom_digest(M, NA)

        switchbd = ws.Switchboard(["My Switch"],[["On","Off"]],["buttons"])
        switchbd.add("mdl", [0])
        switchval = switchbd.mdl
        pygsti.report.workspace.ws_custom_digest(M, switchval)


    def test_table_creation(self):
        w = pygsti.report.Workspace()
        tbls = []; cr = None

        gsMultiSpam = self.mdl.copy()
        gsMultiSpam.povms['Msecondpovm'] = self.mdl.povms['Mdefault'].copy()
        gsTP = self.tgt.depolarize(0.01,0.01); gsTP.set_all_parameterizations("TP")
        gsCPTP = self.tgt.depolarize(0.01,0.01); gsCPTP.set_all_parameterizations("CPTP")
        gsGM = self.mdl.depolarize(0.01,0.01); gsGM.basis = pygsti.obj.Basis.cast("gm",4)
        gsSTD = self.mdl.depolarize(0.01,0.01); gsSTD.basis = pygsti.obj.Basis.cast("std",4)
        gsQT = stdQT_XYIMS.target_model().depolarize(0.01,0.01)

        #Construct confidence regions
        def make_cr(mdl):
            hessian = pygsti.tools.logl_hessian(mdl, self.ds, self.gss,
                                                min_prob_clip=1e-4, prob_clip_interval=(-1e6,1e6),
                                                radius=1e-4)
            est = pygsti.protocols.estimate.Estimate.create_gst_estimate(None, mdl, None, []) #dummy w/out parent
            crfactory = pygsti.obj.ConfidenceRegionFactory(
                parent=est, model_lbl="target", circuit_list_lbl=None,
                hessian=hessian, non_mark_radius_sq=0.0)
            crfactory.project_hessian('std')
            return crfactory.view(95)

        cr2 = make_cr(self.mdl)
        cr2TP = make_cr(gsTP)
        cr2CPTP = make_cr(gsCPTP)

        #----------------- Test table creation ---------------------------
        tbls.append( w.BlankTable() )
        tbls.append( w.SpamTable(self.mdl, ["mytitle"], "boxes", cr, True ) )
        tbls.append( w.SpamTable(gsTP, ["mytitle"], "boxes", cr2TP, True ) )
        tbls.append( w.SpamTable(self.mdl, ["mytitle"], "numbers", cr, False ) )
        with self.assertRaises(ValueError):
            w.SpamTable(self.mdl, ["mytitle"], "foobar", cr, False ) #invalid display_as

        tbls.append( w.SpamParametersTable(self.mdl, cr ) )
        tbls.append( w.SpamParametersTable(gsMultiSpam, cr ) )

        tbls.append( w.GatesTable(self.mdl, ["mytitle"], display_as="boxes", confidence_region_info=cr ) )
        tbls.append( w.GatesTable(self.mdl, ["mytitle"], display_as="numbers", confidence_region_info=cr ) )
        tbls.append( w.GatesTable(gsTP, ["mytitle"], display_as="numbers", confidence_region_info=cr2TP ) )
        tbls.append( w.GatesTable(gsCPTP, ["mytitle"], display_as="numbers", confidence_region_info=cr2CPTP ) )
        with self.assertRaises(ValueError):
            w.GatesTable(self.mdl, ["mytitle"], display_as="foobar", confidence_region_info=cr )

        tbls.append( w.ChoiTable(self.mdl, ["mytitle"], cr ) )
        tbls.append( w.ChoiTable(gsQT, ["mytitle"], None ) )
        with self.assertRaises(ValueError):
            w.ChoiTable(self.mdl, ["mytitle"], cr, display=("foobar",) )

        tbls.append( w.GatesVsTargetTable(self.mdl, self.tgt, cr) )
        with self.assertRaises(ValueError):
            w.GatesVsTargetTable(self.mdl, self.tgt, cr, display=("foobar",) )

        tbls.append( w.SpamVsTargetTable(self.mdl, self.tgt, cr ) )
        tbls.append( w.ErrgenTable(self.mdl, self.tgt, cr, display_as="boxes", gen_type="logTiG") )
        tbls.append( w.ErrgenTable(self.mdl, self.tgt, cr, display_as="numbers", gen_type="logTiG") )
        tbls.append( w.ErrgenTable(self.mdl, self.tgt, cr, display_as="numbers", gen_type="logG-logT") )
        tbls.append( w.ErrgenTable(gsGM, self.tgt, cr, display_as="numbers", gen_type="logGTi") )
        tbls.append( w.ErrgenTable(gsSTD, self.tgt, cr, display_as="numbers", gen_type="logGTi") )
        tbls.append( w.ErrgenTable(gsQT, stdQT_XYIMS.target_model(), cr, display_as="numbers", gen_type="logGTi") )
        with self.assertRaises(ValueError):
            w.ErrgenTable(self.mdl, self.tgt, cr, display=('foobar',))
        with self.assertRaises(AssertionError):
            w.ErrgenTable(self.mdl, self.tgt, cr, display_as='foobar')
        with self.assertRaises(ValueError):
            w.ErrgenTable(self.mdl, self.tgt, cr, gen_type='foobar')

        tbls.append( w.GateDecompTable(self.mdl, self.tgt, cr) )

        tbls.append( w.GateEigenvalueTable(self.mdl, self.tgt, cr) )
        tbls.append( w.GateEigenvalueTable(self.mdl, None, cr, display=("polar",) ) ) # polar with no target model
        tbls.append( w.GateEigenvalueTable(self.mdl, self.tgt, cr, display=("evdm","evinf","rel"),
                                           virtual_ops=[pygsti.obj.Circuit(('Gx','Gx'))] ) )
        with self.assertRaises(ValueError):
            tbls.append( w.GateEigenvalueTable(self.mdl, self.tgt, cr, display=("foobar",)) )

        tbls.append( w.DataSetOverviewTable(self.ds,max_length_list=[1,2,4,8]) )
        tbls.append( w.FitComparisonTable(self.gss.xs, self.results.circuit_lists['iteration'],
                                          self.results.estimates['default'].models['iteration estimates'], self.ds) )
        with self.assertRaises(ValueError):
            w.FitComparisonTable(self.gss.xs, self.results.circuit_lists['iteration'],
                                 self.results.estimates['default'].models['iteration estimates'], self.ds, objfn_builder="foobar")

        tbls.append( w.GaugeRobustErrgenTable(self.mdl, self.tgt) )

        prepStrs = self.results.circuit_lists['prep fiducials']
        effectStrs = self.results.circuit_lists['meas fiducials']
        tbls.append( w.CircuitTable((prepStrs,effectStrs),
                                    ["Prep.","Measure"], common_title="Fiducials"))

        metric_abbrevs = ["evinf", "evagi","evnuinf","evnuagi","evdiamond",
                          "evnudiamond", "inf","agi","trace","diamond","nuinf","nuagi",
                          "frob"]
        for metric in metric_abbrevs:
            tbls.append( w.GatesSingleMetricTable(
                metric, [self.mdl,self.mdl],[self.tgt,self.tgt], ['one','two'])) #1D
            tbls.append( w.GatesSingleMetricTable(
                metric, [[self.mdl],[self.mdl]],[[self.tgt],[self.tgt]],
                ['column one'], ['row one','row two'], op_label="Gx")) #2D
            tbls.append( w.GatesSingleMetricTable(
                metric, [self.mdl,None],[self.tgt,self.tgt], ['one','two'])) #1D w/None model

        tbls.append( w.StandardErrgenTable(4, "hamiltonian", "pp") )  # 1Q
        tbls.append( w.StandardErrgenTable(9, "stochastic", "gm") )   # qutrit
        tbls.append( w.StandardErrgenTable(16, "hamiltonian", "gm") ) # 2Q
        tbls.append( w.StandardErrgenTable(64, "stochastic", "pp") )  # >2Q (3Q)

        goparams = self.results.estimates['default'].goparameters['go0'].copy()
        goparams.update({'method': "LM", 'cptp_penalty_factor': 1.0}) # for coverage
        tbls.append( w.GaugeOptParamsTable(goparams) )
        tbls.append( w.GaugeOptParamsTable(False) )

        params = self.results.estimates['default'].parameters.copy()
        params['gaugeOptParams'] = goparams # add for coverate
        tbls.append( w.MetadataTable(self.mdl, params) )
        params['gaugeOptParams'] = [goparams] # can also be a list (for GOpt stages)
        tbls.append( w.MetadataTable(gsTP, params) )

        weirdGS = pygsti.construction.create_explicit_model(
            [('Q0','Q1')],['Gi'], ["I(Q0)"])
        #weirdGS.preps['rho1'] = pygsti.obj.ComplementSPAMVec(weirdGS.preps['rho0'],[]) #num_params not implemented!
        weirdGS.povms['Mtensor'] = pygsti.obj.TensorProdPOVM([self.mdl.povms['Mdefault'],self.mdl.povms['Mdefault']])
        tbls.append( w.MetadataTable(weirdGS, params) )

        tbls.append( w.SoftwareEnvTable() )

        profiler = pygsti.obj.Profiler()
        tbls.append( w.ProfilerTable(profiler,"time") )
        tbls.append( w.ProfilerTable(profiler,"name") )
        if profiler is not None:
            with self.assertRaises(ValueError):
                w.ProfilerTable(profiler,"foobar")

        #OLD tables
        tbls.append( w.old_RotationAxisVsTargetTable(self.mdl, self.tgt) )
        tbls.append( w.old_GateDecompTable(self.mdl) )
        tbls.append( w.old_RotationAxisTable(self.mdl) )


        #Now test table rendering in html
        for i,tbl in enumerate(tbls):
            print("Table: %s" % str(type(tbl)))
            out_html = tbl.render("html")
            #out_latex = tbl.render("latex") #not supported yet (figure formatting wants scratchdir)

            tbl.saveas(temp_files + "/saved_table_%d.html" % i)
            if bPandas:
                tbl.saveas(temp_files + "/saved_table_%d.pkl" % i)
            if bLatex:
                tbl.saveas(temp_files + "/saved_table_%d.tex" % i)
                tbl.saveas(temp_files + "/saved_table_%d.pdf" % i)
                tbl.set_render_options(render_includes=False, leave_includes_src=False) #dumb case where nothing is output
                tbl.render('latex')
            with self.assertRaises(ValueError):
                tbl.saveas(temp_files + "/saved_table_%d.foobar" % i) # Unknown file type

        w.save_cache(temp_files + "/saved_workspace_cache1.pkl", show_unpickled=True)
        w.load_cache(temp_files + "/saved_workspace_cache1.pkl")

        w2 = pygsti.report.Workspace(temp_files + "/saved_workspace_cache1.pkl") # init from cache

        #printOpenFiles()
        #print("PSUTIL open files (%d) = " % len(psutil.Process().open_files()), psutil.Process().open_files())
        #assert(False),"STOP"


    def test_plot_creation(self):
        w = pygsti.report.Workspace()
        prepStrs = self.results.circuit_lists['prep fiducials']
        effectStrs = self.results.circuit_lists['meas fiducials']
        non_gatestring_strs = [ 'GxString', 'GyString' ]

        plts = []
        plts.append( w.BoxKeyPlot(prepStrs, effectStrs) )
        plts.append( w.BoxKeyPlot(non_gatestring_strs, non_gatestring_strs) )
        plts.append( w.ColorBoxPlot(("chi2","logl"), self.gss, self.ds, self.mdl, box_labels=True,
                                    hover_info=True, sum_up=False, invert=False) )
        plts.append( w.ColorBoxPlot(("chi2","logl"), self.gss, self.ds, self.mdl, box_labels=False,
                                    hover_info=True, sum_up=True, invert=False) )
        #plts.append( w.ColorBoxPlot(("chi2","logl"), self.gss, self.ds, self.mdl, box_labels=False,
        #                            hover_info=True, sum_up=False, invert=True) )  # invert no longer supported
        plts.append( w.ColorBoxPlot(("chi2","logl"), self.gss, self.ds, self.mdl, box_labels=False,
                                    hover_info=True, sum_up=False, invert=False, typ="scatter") )

        mds = pygsti.objects.MultiDataSet()
        mds.add_dataset("DS0",self.ds)
        mds.add_dataset("DS1",self.ds)
        dsc = pygsti.objects.DataComparator([self.ds,self.ds], op_exclusions=['Gfoo'], op_inclusions=['Gx','Gy','Gi'])
        dsc2 = pygsti.objects.DataComparator(mds)
        dsc.run()
        dsc2.run()
        plts.append( w.ColorBoxPlot(("dscmp",), self.gss, None, self.mdl, dscomparator=dsc) ) # dscmp with 'None' dataset specified
        plts.append( w.ColorBoxPlot(("dscmp",), self.gss, None, self.mdl, dscomparator=dsc2) )

        tds = pygsti.io.load_time_dependent_dataset(compare_files + "/timeseries_data_trunc.txt")
        #OLD: driftresults = drift.do_basic_drift_characterization(tds)
        results_gst = drift.StabilityAnalyzer(tds, ids=True)
        results_gst.compute_spectra()
        results_gst.run_instability_detection(0.05)
        results_gst.run_instability_characterization(estimator='filter', modelselector=('default',()),verbosity=0)
        plts.append( w.ColorBoxPlot('driftsize', self.gss, self.ds, self.mdl, box_labels=False,
                                    hover_info=False, sum_up=True, invert=False, stabilityanalyzer=results_gst) )

        with self.assertRaises(ValueError):
            w.ColorBoxPlot(("foobar",), self.gss, self.ds, self.mdl)
        with self.assertRaises(ValueError):
            w.ColorBoxPlot(("chi2",), self.gss, self.ds, self.mdl, typ="foobar")

        from pygsti.algorithms import directx as dx
        #specs = pygsti.construction.build_spam_specs(
        #        prepStrs=prepStrs,
        #        effectStrs=effectStrs,
        #        prep_labels=list(self.mdl.preps.keys()),
        #        effect_labels=self.mdl.get_effect_labels() )

        baseStrs = [plaq.base for _, plaq in self.gss.iter_plaquettes()]
        directModels = dx.direct_mlgst_models(
            baseStrs, self.ds, prepStrs, effectStrs, self.tgt, svd_truncate_to=4)
        plts.append( w.ColorBoxPlot(["chi2","logl","blank"], self.gss,
                                    self.ds, self.mdl, box_labels=False, direct_gst_models=directModels) )
        plts.append( w.ColorBoxPlot(["errorrate"], self.gss,
                                    self.ds, self.mdl, box_labels=False, sum_up=True,
                                    direct_gst_models=directModels) )

        gmx = np.identity(4,'d'); gmx[3,0] = 0.5
        plts.append( w.MatrixPlot(gmx, -1,1, ['a','b','c','d'], ['e','f','g','h'], "X", "Y",
                                  colormap = pygsti.report.colormaps.DivergingColormap(vmin=-2, vmax=2)) )
        plts.append( w.MatrixPlot(gmx, -1,1, ['a','b','c','d'], ['e','f','g','h'], "X", "Y",colormap=None))
        plts.append( w.GateMatrixPlot(gmx, -1,1, "pp", "in", "out", box_labels=True) )
        plts.append( w.PolarEigenvaluePlot([np.linalg.eigvals(self.mdl.operations['Gx'])],["purple"],scale=1.5) )
        plts.append( w.PolarEigenvaluePlot([np.linalg.eigvals(self.mdl.operations['Gx'])],["purple"],amp=2.0) )

        projections = np.zeros(16,'d')
        plts.append( w.ProjectionsBoxPlot(projections, "pp", box_labels=False) )
        plts.append( w.ProjectionsBoxPlot(projections, "gm", box_labels=True) )
        plts.append( w.ProjectionsBoxPlot(np.zeros(9,'d'), "gm", box_labels=True) )  # non-qubit case
        plts.append( w.ProjectionsBoxPlot(np.zeros(64,'d'), "gm", box_labels=True) ) # >2Q case

        choievals = np.array([-0.03, 0.02, 0.04, 0.98])
        choieb = np.array([0.05, 0.01, 0.02, 0.01])
        plts.append( w.ChoiEigenvalueBarPlot(choievals, None) )
        plts.append( w.ChoiEigenvalueBarPlot(choievals, choieb) )

        plts.append( w.FitComparisonBarPlot(self.gss.xs, self.results.circuit_lists['iteration'],
                                          self.results.estimates['default'].models['iteration estimates'], self.ds,) )
        plts.append( w.GramMatrixBarPlot(self.ds,self.tgt) )

        plts.append( w.DatasetComparisonHistogramPlot(dsc, nbins=50, frequency=True,
                                                      log=True, display='pvalue'))
        plts.append( w.DatasetComparisonHistogramPlot(dsc, nbins=None, frequency=True,
                                                      log=False, display='llr'))
        with self.assertRaises(ValueError):
            w.DatasetComparisonHistogramPlot(dsc, nbins=50, frequency=True,
                                             log=False, display='foobar')

        # Note: RandomizedBenchmarkingPlot is tested in extras/test_rb.py

        #Now test table rendering in html
        for i,plt in enumerate(plts):
            print("Plot: %s" % str(type(plt)))
            out_html = plt.render("html")
            plt.saveas(temp_files + "/saved_plot_%d.html" % i)
            plt.saveas(temp_files + "/saved_plot_%d.pkl" % i) #Note: plots don't need Pandas for pickling
            if bLatex:
                plt.saveas(temp_files + "/saved_plot_%d.pdf" % i)
            with self.assertRaises(ValueError):
                plt.saveas(temp_files + "/saved_plot_%d.foobar" % i) # Unknown file type

        w.save_cache(temp_files + "/saved_workspace_cache1.pkl")
        w.load_cache(temp_files + "/saved_workspace_cache1.pkl")

        # printOpenFiles()
        # print("PSUTIL open files (%d) = " % len(psutil.Process().open_files()), psutil.Process().open_files())
        # assert(False),"STOP"


    def test_switchboard(self):
        w = pygsti.report.Workspace()
        ds = self.ds
        mdl = self.mdl
        gs2 = self.mdl.depolarize(op_noise=0.01, spam_noise=0.15)
        gs3 = self.mdl.depolarize(op_noise=0.011, spam_noise=0.1)

        switchbd = w.Switchboard(["dataset","model"],
                                 [["one","two"],["One","Two"]],
                                 ["dropdown","slider"])
        switchbd.add("ds",(0,))
        switchbd.add("mdl",(1,))
        switchbd["ds"][:] = [ds, ds]
        switchbd["mdl"][:] = [mdl, gs2]

        switchbd2 = w.Switchboard(["spamWeight"], [["0.0","0.1","0.2","0.5","0.9","0.95"]], ["slider"])

        tbl = w.SpamTable(switchbd["mdl"])
        plt = w.ColorBoxPlot(("chi2","logl"), self.gss, switchbd["ds"], switchbd["mdl"], box_labels=False)

        switchbd.render("html")
        switchbd2.render("html")
        tbl.render("html")
        plt.render("html")

        switchbd3 = w.Switchboard(["My Switch"],[["On","Off"]],["buttons"])
        switchbd3.add("mdl", [0])
        switchbd3.mdl[:] = [gs2,gs3]
        tbl2 = w.GatesVsTargetTable(switchbd3.mdl, self.tgt)

        switchbd3.render("html")
        tbl2.render("html")

        switchbd3.render("html")
        try:
            switchbd3.display()
        except:
            pass #might work, might not, depending on how much ipython support exists...

        switchbd4 = w.Switchboard(["My Numeric Slider"],[["0.0","1.0","1.5"]],["numslider"])
        switchbd5 = w.Switchboard(["My Switch"],[["On","Off"]],["buttons"],show="none") #test show=="none"
        with self.assertRaises(KeyError):
            switchbd5['key'] = 10 # can't add switched values like this.
        switchbd5.switchTypes[0] = "foobar" #mess with Switchboard internals to trigger errors below
        with self.assertRaises(ValueError):
            switchbd5.get_switch_change_handlerjs(0)
        with self.assertRaises(ValueError):
            switchbd5.get_switch_valuejs(0)
        with self.assertRaises(ValueError):
            switchbd5.render("html")

        switchbd4.render("html")
        #switchbd5.render("html")

        view1 = switchbd.view()
        view2 = switchbd.view(['dataset'])
        view3 = switchbd.view([True,False])
        view4 = pygsti.report.workspace.SwitchboardView(switchbd, show="all")
        view5 = pygsti.report.workspace.SwitchboardView(switchbd, show="none")

        try:
            view1.display()
        except:
            pass #might work, might not, depending on how much ipython support exists...

        #Test a switched WorkspaceTable & Plot & Text
        switchbd6 = w.Switchboard(["Plot type"],[["BoxeS","NumberS"]],["buttons"])
        switchbd6.add("typ", [0])
        switchbd6.add("strs", [0])
        switchbd6.typ[:] = ["boxes","numbers"]
        switchbd6.strs[:] = [ [('Gx',)], [('Gy',)] ]

        tbl = w.SpamTable(self.mdl, ["mytitle"], switchbd6.typ, None)
        tbl.saveas(temp_files + "/saved_switched_table.html", index=0)
        if bPandas: tbl.saveas(temp_files + "/saved_switched_table.pkl") # OK to not specify index in pkl case
        with self.assertRaises(ValueError):
            tbl.saveas(temp_files + "/saved_switched_table.html") # must supply index
        with self.assertRaises(ValueError):
            tbl.saveas(temp_files + "/saved_switched_table.tex") # must supply index

        plt = w.BoxKeyPlot(switchbd6.strs, switchbd6.strs)
        plt.saveas(temp_files + "/saved_switched_plot.html", index=0)
        plt.saveas(temp_files + "/saved_switched_plot.pkl") # OK to not specify index in pkl case
        with self.assertRaises(ValueError):
            plt.saveas(temp_files + "/saved_switched_plot.html") # must supply index
        with self.assertRaises(ValueError):
            plt.saveas(temp_files + "/saved_switched_plot.tex") # cannot save plots as LaTeX
        with self.assertRaises(ValueError):
            plt.saveas(temp_files + "/saved_switched_plot.pdf") # must supply index

        def textfn(s): return pygsti.report.textblock.ReportText(s, None)
        txt = pygsti.report.workspace.WorkspaceText(w, textfn, switchbd6.typ) # just "boxes" / "numbers" text
        txt.saveas(temp_files + "/saved_switched_text.html", index=0)
        txt.saveas(temp_files + "/saved_switched_text.pkl") # OK to not specify index in pkl case
        with self.assertRaises(ValueError):
            txt.saveas(temp_files + "/saved_switched_text.html") # must supply index
        with self.assertRaises(ValueError):
            txt.saveas(temp_files + "/saved_switched_text.tex") # must supply index


        # Test "Opaque figure" python rendering (fig w/out "pythonvalue")
        plt.figs[0].pythonvalue = None # mimic fig w/out pythonvalue
        py = plt.render("python")
        self.assertEqual( py['python']['index0']['value'], "Opaque Figure")


    def test_text_creation(self):
        printer = pygsti.obj.VerbosityPrinter(1)
        printer.start_recording()
        printer.log("Hello World")
        lineinfo = printer.stop_recording()

        ws = pygsti.report.Workspace()
        texts = []
        texts.append( ws.StdoutText( lineinfo ) )

        #Now test rendering
        for i,text in enumerate(texts):
            print("Text: %s" % str(type(text)))
            out_html = text.render("html")
            text.saveas(temp_files + "/saved_text_%d.html" % i)
            text.saveas(temp_files + "/saved_text_%d.pkl" % i)
            if bLatex:
                text.saveas(temp_files + "/saved_text_%d.tex" % i)
                text.saveas(temp_files + "/saved_text_%d.pdf" % i)
                text.set_render_options(render_includes=False, leave_includes_src=False) #dumb case where nothing is output
                text.render("latex")
            with self.assertRaises(ValueError):
                text.saveas(temp_files + "/saved_text_%d.foobar" % i) # Unknown file type


    def test_workspaceoutput(self):
        ws = pygsti.report.Workspace()
        raw_wo = pygsti.report.workspace.WorkspaceOutput(ws)

        s = pickle.dumps(raw_wo)
        raw_wo2 = pickle.loads(s)

        #test some base class paths
        with self.assertRaises(ValueError):
            raw_wo.set_render_options(foobar="10")
        with self.assertRaises(NotImplementedError):
            raw_wo.render("html")
        with self.assertRaises(NotImplementedError):
            raw_wo.saveas("doesntmatter.html")

        raw_wo.set_render_options(global_requirejs=True)
        raw_wo._create_onready_handler("myJavascriptCode;", "MyID")

        try:
            raw_wo.display()
        except:
            pass #might work or not, depending on how much ipython is installed

    def test_notapplicable(self):
        ws = pygsti.report.Workspace()
        NA = ws.NotApplicable()
        NA.render("html")
        NA.render("latex")
        NA.render("python")
        with self.assertRaises(ValueError):
            NA.render("foobar")

    def test_plot_and_table_objects(self):
        ws = pygsti.report.Workspace()

        printer = pygsti.obj.VerbosityPrinter(1)
        printer.start_recording()
        printer.log("Hello World (with $\\alpha$ math latex)")
        lineinfo = printer.stop_recording()
        strs = pygsti.construction.to_circuits([ (), ('Gx',), ('Gx','Gy')])

        table = ws.BlankTable()
        plot = ws.BoxKeyPlot(strs, strs)
        text = ws.StdoutText( lineinfo )
        objs = [table, plot, text] # list of "representative" objects for testing

        for obj in objs:
            print("Testing ",type(obj))
            obj.set_render_options(precision=4,
                                   output_dir=temp_files) # needed for html rendering
            obj.render("html")

            obj.set_render_options(switched_item_mode='foobar')
            with self.assertRaises(ValueError):
                obj.render("html") #invalid switched-item-mode
            if bLatex:
                with self.assertRaises(ValueError):
                    obj.render("latex") #invalid switched-item-mode
            if bPandas:
                with self.assertRaises(ValueError):
                    obj.render("python") #invalid switched-item-mode

            for mode in ('separate files','inline'):
                obj.set_render_options(switched_item_mode=mode,
                                       autosize="continual") #pick options not used elsewhere
                obj.render("html")
                if bLatex:  obj.render("latex")
                if bPandas: obj.render("python")

            with self.assertRaises(NotImplementedError):
                obj.render("foobar")

        #Individual item tests
        with self.assertRaises(ValueError):
            plot.set_render_options(valign="foobar")
            plot.render("html")


    def test_plot_helpers(self):
        from pygsti.report import plothelpers as ph

        self.assertEqual(ph._eformat(0.1, "compacthp"),".10")
        self.assertEqual(ph._eformat(1.0, "compacthp"),"1.0")
        self.assertEqual(ph._eformat(5.2, "compacthp"),"5.2")
        self.assertEqual(ph._eformat(63.2, "compacthp"),"63")
        self.assertEqual(ph._eformat(2.1e-4, "compacthp"),"2m4")
        self.assertEqual(ph._eformat(2.1e+4, "compacthp"),"2e4")
        self.assertEqual(ph._eformat(-3.2e-4, "compacthp"),"-3m4")
        self.assertEqual(ph._eformat(-3.2e+4, "compacthp"),"-3e4")
        self.assertEqual(ph._eformat(4e+40, "compacthp"),"*40")
        self.assertEqual(ph._eformat(6e+102, "compacthp"),"B")
        self.assertEqual(ph._eformat(10, "compacthp"),"10")
        self.assertEqual(ph._eformat(1.234, 2),"1.23")
        self.assertEqual(ph._eformat(-1.234, 2),"-1.23")
        self.assertEqual(ph._eformat(-1.234, "foobar"), "-1.234") #just prints in general format

        subMxs = np.nan * np.ones((3,3,2,2),'d') # a 3x3 grid of 2x2 matrices
        nBoxes, dof_per_box = ph._compute_num_boxes_dof(subMxs, sum_up=True, element_dof=1)
        self.assertEqual(nBoxes, 0)
        self.assertEqual(dof_per_box, None)

        subMxs[0,0,1,1] = 1.0 # matrix [0,0] has a single non-Nan element
        subMxs[0,2,0,1] = 1.0
        subMxs[0,2,1,1] = 1.0 # matrix [0,2] has a two non-Nan elements

        # now the mxs that aren't all-NaNs don't all have the same # of Nans => warning
        #self.assertWarns(ph._compute_num_boxes_dof, subMxs, sum_up=True, element_dof=1)
        ph._compute_num_boxes_dof( subMxs, sum_up=True, element_dof=1) # Python2.7 doesn't always warn...


    def test_plot_basefns(self):
        #Tests non-covered boundary cases for plot-supporting functions
        import pygsti.report.colormaps
        import pygsti.report.workspaceplots

        colormap = pygsti.report.colormaps.DivergingColormap(0,10)
        mx = np.identity(2,'d')
        mxs = [ [mx, mx],
                [mx, mx] ]
        gstrs = pygsti.construction.to_circuits([ (), ('Gx',) ])

        # ---- _nested_color_boxplot ----
        pygsti.report.workspaceplots._nested_color_boxplot(mxs, colormap)

        # ---- _summable_color_boxplot ----
        pygsti.report.workspaceplots._summable_color_boxplot(
            mxs, gstrs, gstrs, 'Xlbl','Ylbl', colormap) #circuit labels

        pygsti.report.workspaceplots._summable_color_boxplot(
            mxs, ['xlbl1','xlbl2'], ['ylbl1','ylbl2'],
            'Xlbl','Ylbl', colormap,
            sum_up=True, hover_info=True) #sumup test

        pygsti.report.workspaceplots._summable_color_boxplot(
            mxs, ['xlbl1','xlbl2'], ['ylbl1','ylbl2'],
            'Xlbl','Ylbl', colormap,
            sum_up=True, hover_info=False) #no hover info

        pygsti.report.workspaceplots._summable_color_boxplot(
            mxs, ['xlbl1','xlbl2'], ['ylbl1','ylbl2'],
            'Xlbl','Ylbl', colormap,
            sum_up=False, hover_info=False) # no hover info or sum_up

        # invert is no longer an option
        #pygsti.report.workspaceplots._summable_color_boxplot(
        #    mxs, ['xlbl1','xlbl2'], ['ylbl1','ylbl2'],
        #    'Xlbl','Ylbl', colormap,
        #    sum_up=True, invert=True) #ignores invert

        # ----- _circuit_color_boxplot -----
        germs = preps = effects = gstrs
        plaquettes = {}
        fidpairs = {(i,j): (prep, meas) for j,prep in enumerate(preps) for i,meas in enumerate(effects)}
        for L in [1,2]:
            for germ in germs:
                plaquettes[(L, germ)] = pygsti.obj.GermFiducialPairPlaquette(germ, L, fidpairs)
        gss = pygsti.obj.PlaquetteGridCircuitStructure(plaquettes, [1,2], germs, 'L', 'germ')
        gss2 = gss.copy()
        
        #cls = type('DummyClass', pygsti.obj.LsGermsStructure.__bases__, dict(pygsti.obj.LsGermsStructure.__dict__))
        #gss3.__class__ = cls  # mimic a non-LsGermsStructure object when we don't actually have any currently (HACK)
        #assert(not isinstance(gss3, pygsti.obj.LsGermsStructure))

        pygsti.report.workspaceplots._circuit_color_boxplot(
            gss, mxs, colormap, sum_up=True)
        pygsti.report.workspaceplots._circuit_color_boxplot(
            gss, mxs, colormap, sum_up=False)
        pygsti.report.workspaceplots._circuit_color_boxplot(
            gss2, mxs, colormap, sum_up=True)
        pygsti.report.workspaceplots._circuit_color_boxplot(
            gss2, mxs, colormap, sum_up=False)


        # ----- _circuit_color_scatterplot -----
        pygsti.report.workspaceplots._circuit_color_scatterplot(
            gss, mxs, colormap, sum_up=True)
        pygsti.report.workspaceplots._circuit_color_scatterplot(
            gss, mxs, colormap, hover_info=False) # no hoverinfo
        pygsti.report.workspaceplots._circuit_color_scatterplot(
            gss2, mxs, colormap, hover_info=True) # gss2 case
        pygsti.report.workspaceplots._circuit_color_scatterplot(
            gss2, mxs, colormap, hover_info=True) # gss2 case
        pygsti.report.workspaceplots._circuit_color_scatterplot(
            gss, mxs, colormap, sum_up=True, addl_hover_submxs={'qty': mxs} ) # just reuse mxs
        pygsti.report.workspaceplots._circuit_color_scatterplot(
            gss, mxs, colormap, sum_up=False, addl_hover_submxs={'qty': mxs} ) # just reuse mxs
        #pygsti.report.workspaceplots._circuit_color_scatterplot(
        #    gss3, mxs, colormap, sum_up=True, hover_info=True) # gss3 case
        #pygsti.report.workspaceplots._circuit_color_scatterplot(
        #    gss3, mxs, colormap, sum_up=False, hover_info=True) # gss3 case


        # ---- _circuit_color_histogram ----
        negmx = -1*np.ones((2,2),'d')
        mxs_allneg = [ [negmx, negmx],
                         [negmx, negmx] ]
        pygsti.report.workspaceplots._circuit_color_histogram(
            gss, mxs_allneg, colormap) # when there's no counts to plot

        # ---- _opmatrix_color_boxplot ----
        pygsti.report.workspaceplots._opmatrix_color_boxplot(
            np.identity(4,'d'), -1.0, 1.0, mx_basis="pp", mx_basis_y="gm")
          # test weird case when there's different bases on diff axes

        # ---- _matrix_color_boxplot ----
        pygsti.report.workspaceplots._matrix_color_boxplot(
            np.identity(16,'d'), colormap=colormap, grid="black:2") # test "gridcolor:linewidth" format of grid argument.


if __name__ == "__main__":
    unittest.main(verbosity=2)
