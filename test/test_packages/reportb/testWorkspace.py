import os
import unittest
import pygsti
#import psutil
from pygsti.extras import drift

from ..testutils import compare_files, temp_files

import numpy as np

from ..report.reportBaseCase import ReportBaseCase

bLatex = bool('PYGSTI_LATEX_TESTING' in os.environ and 
              os.environ['PYGSTI_LATEX_TESTING'].lower() in ("yes","1","true"))


#HACK for tracking open files
# try:
#     import __builtin__ as builtins # Python2.7
#     ver = 2
# except ImportError:
#     import builtins
#     ver = 3
#     
# openfiles = set()
# if ver == 2:
#     oldfile = builtins.file
#     class newfile(oldfile):
#         def __init__(self, *args):
#             self.x = args[0]
#             #print("### OPENING %s ###" % str(self.x))
#             oldfile.__init__(self, *args)
#             openfiles.add(self)
#     
#         def close(self):
#             #print("### CLOSING %s ###" % str(self.x))
#             oldfile.close(self)
#             openfiles.remove(self)
#     oldopen = builtins.open
#     def newopen(*args):
#         return newfile(*args)
#     builtins.file = newfile # file() only exists in python2
#     builtins.open = newopen
#     
# def printOpenFiles():
#     print("### %d OPEN FILES: [%s]" % (len(openfiles), ", ".join(f.x for f in openfiles)))

class TestWorkspace(ReportBaseCase):

    def setUp(self):
        super(TestWorkspace, self).setUp()

        self.tgt = self.results.estimates['default'].gatesets['target']
        self.ds = self.results.dataset
        self.gs = self.results.estimates['default'].gatesets['go0']
        self.gss = self.results.gatestring_structs['final']

    def test_notebook_mode(self):
        wnb = pygsti.report.Workspace()
        wnb.init_notebook_mode(connected=True, autodisplay=True)
        wnb = pygsti.report.Workspace()
        wnb.init_notebook_mode(connected=True, autodisplay=False)
        wnb = pygsti.report.Workspace()
        wnb.init_notebook_mode(connected=False, autodisplay=True)
        wnb = pygsti.report.Workspace()
        wnb.init_notebook_mode(connected=False, autodisplay=False)

    def test_table_creation(self):
        w = pygsti.report.Workspace()
        tbls = []; cr = None
        
        tbls.append( w.BlankTable() )
        tbls.append( w.SpamTable(self.gs, ["mytitle"], "boxes", cr, True ) )
        tbls.append( w.SpamTable(self.gs, ["mytitle"], "numbers", cr, False ) )
        tbls.append( w.SpamParametersTable(self.gs, cr ) )
        tbls.append( w.GatesTable(self.gs, ["mytitle"], display_as="boxes", confidenceRegionInfo=cr ) )
        tbls.append( w.GatesTable(self.gs, ["mytitle"], display_as="numbers", confidenceRegionInfo=cr ) )
        tbls.append( w.ChoiTable(self.gs, ["mytitle"], cr ) )
        tbls.append( w.GatesVsTargetTable(self.gs, self.tgt, cr) )
        tbls.append( w.SpamVsTargetTable(self.gs, self.tgt, cr ) )
        tbls.append( w.ErrgenTable(self.gs, self.tgt, cr, display_as="boxes", genType="logTiG") )
        tbls.append( w.ErrgenTable(self.gs, self.tgt, cr, display_as="numbers", genType="logTiG") )
        tbls.append( w.ErrgenTable(self.gs, self.tgt, cr, display_as="numbers", genType="logG-logT") )
        tbls.append( w.GateDecompTable(self.gs, self.tgt, cr) )
        #tbls.append( w.RotationAxisVsTargetTable(self.gs, self.tgt, cr ) )
        #tbls.append( w.RotationAxisTable(self.gs, cr) )
        tbls.append( w.GateEigenvalueTable(self.gs, self.tgt, cr) )
        tbls.append( w.DataSetOverviewTable(self.ds) )
        tbls.append( w.FitComparisonTable(self.gss.Ls, self.results.gatestring_structs['iteration'],
                                          self.results.estimates['default'].gatesets['iteration estimates'], self.ds,) )
        tbls.append( w.GaugeRobustErrgenTable(self.gs, self.tgt) )

        prepStrs = self.results.gatestring_lists['prep fiducials']
        effectStrs = self.results.gatestring_lists['effect fiducials']
        tbls.append( w.GatestringTable((prepStrs,effectStrs),
                                       ["Prep.","Measure"], commonTitle="Fiducials"))

        metric_abbrevs = ["evinf", "evagi","evnuinf","evnuagi","evdiamond",
                          "evnudiamond", "inf","agi","trace","diamond","nuinf","nuagi",
                          "frob"]
        for metric in metric_abbrevs:
            tbls.append( w.GatesSingleMetricTable(
                metric, [self.gs,self.gs],[self.tgt,self.tgt], ['one','two'])) #1D
            tbls.append( w.GatesSingleMetricTable(
                metric, [[self.gs],[self.gs]],[[self.tgt],[self.tgt]],
                ['column one'], ['row one','row two'], gateLabel="Gx")) #2D

        tbls.append( w.StandardErrgenTable(4, "hamiltonian", "pp") )
        tbls.append( w.StandardErrgenTable(4, "stochastic", "pp") )
        tbls.append( w.StandardErrgenTable(4, "hamiltonian", "gm") )
        tbls.append( w.StandardErrgenTable(4, "stochastic", "gm") )
        
        tbls.append( w.GaugeOptParamsTable(self.results.estimates['default'].goparameters['go0']) )
        tbls.append( w.MetadataTable(self.gs, self.results.estimates['default'].parameters ) )
        tbls.append( w.SoftwareEnvTable() )

        #Now test table rendering in html
        for tbl in tbls:
            print("Table: %s" % str(type(tbl)))
            out_html = tbl.render("html")
            #out_latex = tbl.render("latex") #not supported yet (figure formatting wants scratchdir)
            if bLatex:
                tbl.saveas(temp_files + "/saved_table_%s.pdf" % str(id(tbl)))

        #printOpenFiles()
        #print("PSUTIL open files (%d) = " % len(psutil.Process().open_files()), psutil.Process().open_files())
        #assert(False),"STOP"


    def test_plot_creation(self):
        w = pygsti.report.Workspace()
        prepStrs = self.results.gatestring_lists['prep fiducials']
        effectStrs = self.results.gatestring_lists['effect fiducials']
        
        plts = []
        plts.append( w.BoxKeyPlot(prepStrs, effectStrs) )
        plts.append( w.ColorBoxPlot(("chi2","logl"), self.gss, self.ds, self.gs, boxLabels=True,
                                    hoverInfo=True, sumUp=False, invert=False) )
        plts.append( w.ColorBoxPlot(("chi2","logl"), self.gss, self.ds, self.gs, boxLabels=False,
                                    hoverInfo=True, sumUp=True, invert=False) )
        plts.append( w.ColorBoxPlot(("chi2","logl"), self.gss, self.ds, self.gs, boxLabels=False,
                                    hoverInfo=True, sumUp=False, invert=True) )
        plts.append( w.ColorBoxPlot(("chi2","logl"), self.gss, self.ds, self.gs, boxLabels=False,
                                    hoverInfo=True, sumUp=False, invert=False, typ="scatter") )

        tds = pygsti.io.load_tddataset(compare_files + "/timeseries_data_trunc.txt")
        driftresults = drift.do_basic_drift_characterization(tds)
        plts.append( w.ColorBoxPlot(("driftpv","driftpwr"), self.gss, self.ds, self.gs, boxLabels=False,
                                    hoverInfo=True, sumUp=True, invert=False, driftresults=driftresults) )

        from pygsti.algorithms import directx as dx
        #specs = pygsti.construction.build_spam_specs(
        #        prepStrs=prepStrs,
        #        effectStrs=effectStrs,
        #        prep_labels=list(self.gs.preps.keys()),
        #        effect_labels=self.gs.get_effect_labels() )
        baseStrs = self.gss.get_basestrings()
        directGatesets = dx.direct_mlgst_gatesets(
            baseStrs, self.ds, prepStrs, effectStrs, self.tgt, svdTruncateTo=4)
        plts.append( w.ColorBoxPlot(["chi2","logl","blank",'directchi2','directlogl'], self.gss,
                                    self.ds, self.gs, boxLabels=False, directGSTgatesets=directGatesets) )
        plts.append( w.ColorBoxPlot(["errorrate"], self.gss,
                                    self.ds, self.gs, boxLabels=False, sumUp=True,
                                    directGSTgatesets=directGatesets) )
        
        gmx = np.identity(4,'d'); gmx[3,0] = 0.5
        plts.append( w.MatrixPlot(gmx, -1,1, ['a','b','c','d'], ['e','f','g','h'], "X", "Y",
                                  colormap = pygsti.report.colormaps.DivergingColormap(vmin=-2, vmax=2)) )
        plts.append( w.GateMatrixPlot(gmx, -1,1, "pp", "in", "out", boxLabels=True) )
        plts.append( w.PolarEigenvaluePlot([np.linalg.eigvals(self.gs.gates['Gx'])],["purple"],scale=1.5) )

        projections = np.zeros(16,'d')
        plts.append( w.ProjectionsBoxPlot(projections, "pp", boxLabels=False) )
        plts.append( w.ProjectionsBoxPlot(projections, "gm", boxLabels=True) )
        
        choievals = np.array([-0.03, 0.02, 0.04, 0.98])
        choieb = np.array([0.05, 0.01, 0.02, 0.01])
        plts.append( w.ChoiEigenvalueBarPlot(choievals, None) )
        plts.append( w.ChoiEigenvalueBarPlot(choievals, choieb) )

        plts.append( w.FitComparisonBarPlot(self.gss.Ls, self.results.gatestring_structs['iteration'],
                                          self.results.estimates['default'].gatesets['iteration estimates'], self.ds,) )
        plts.append( w.GramMatrixBarPlot(self.ds,self.tgt) )
                     
        #Now test table rendering in html
        for plt in plts:
            print("Plot: %s" % str(type(plt)))
            out_html = plt.render("html")
            if bLatex:
                plt.saveas(temp_files + "/saved_plot_%s.pdf" % str(id(plt)))

        # printOpenFiles()
        # print("PSUTIL open files (%d) = " % len(psutil.Process().open_files()), psutil.Process().open_files())
        # assert(False),"STOP"


    def test_switchboard(self):
        w = pygsti.report.Workspace()
        ds = self.ds
        gs = self.gs
        gs2 = self.gs.depolarize(gate_noise=0.01, spam_noise=0.15)
        gs3 = self.gs.depolarize(gate_noise=0.011, spam_noise=0.1)

        switchbd = w.Switchboard(["dataset","gateset"],
                                 [["one","two"],["One","Two"]],
                                 ["dropdown","slider"])
        switchbd.add("ds",(0,))
        switchbd.add("gs",(1,))
        switchbd["ds"][:] = [ds, ds]
        switchbd["gs"][:] = [gs, gs2]

        switchbd2 = w.Switchboard(["spamWeight"], [["0.0","0.1","0.2","0.5","0.9","0.95"]], ["slider"])

        tbl = w.SpamTable(switchbd["gs"])
        plt = w.ColorBoxPlot(("chi2","logl"), self.gss, switchbd["ds"], switchbd["gs"], boxLabels=False)

        switchbd.render("html")
        switchbd2.render("html")
        tbl.render("html")
        plt.render("html")

        switchbd3 = w.Switchboard(["My Switch"],[["On","Off"]],["buttons"])
        switchbd3.add("gs", [0])
        switchbd3.gs[:] = [gs2,gs3]
        tbl2 = w.GatesVsTargetTable(switchbd3.gs, self.tgt)
        
        switchbd3.render("html")
        tbl2.render("html")


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
        nBoxes, dof_per_box = ph._compute_num_boxes_dof(subMxs, sumUp=True, element_dof=1)
        self.assertEqual(nBoxes, 0)
        self.assertEqual(dof_per_box, None)

        subMxs[0,0,1,1] = 1.0 # matrix [0,0] has a single non-Nan element
        subMxs[0,2,0,1] = 1.0 
        subMxs[0,2,1,1] = 1.0 # matrix [0,2] has a two non-Nan elements

        # now the mxs that aren't all-NaNs don't all have the same # of Nans => warning
        #self.assertWarns(ph._compute_num_boxes_dof, subMxs, sumUp=True, element_dof=1)
        ph._compute_num_boxes_dof( subMxs, sumUp=True, element_dof=1) # Python2.7 doesn't always warn...
        
