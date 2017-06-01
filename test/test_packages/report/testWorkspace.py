import unittest
import pygsti
from ..testutils import compare_files, temp_files

import numpy as np

from .reportBaseCase import ReportBaseCase

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
        tbls.append( w.SpamTable(self.gs, ["mytitle"], cr, True ) )
        tbls.append( w.SpamTable(self.gs, ["mytitle"], cr, False ) )
        tbls.append( w.SpamParametersTable(self.gs, cr ) )
        tbls.append( w.GatesTable(self.gs, ["mytitle"], display_as="boxes", confidenceRegionInfo=cr ) )
        tbls.append( w.GatesTable(self.gs, ["mytitle"], display_as="numbers", confidenceRegionInfo=cr ) )
        tbls.append( w.ChoiTable(self.gs, ["mytitle"], cr ) )
        tbls.append( w.GatesVsTargetTable(self.gs, self.tgt, cr) )
        tbls.append( w.SpamVsTargetTable(self.gs, self.tgt, cr ) )
        tbls.append( w.ErrgenTable(self.gs, self.tgt, cr, display_as="boxes", genType="logTiG") )
        tbls.append( w.ErrgenTable(self.gs, self.tgt, cr, display_as="numbers", genType="logTiG") )
        tbls.append( w.ErrgenTable(self.gs, self.tgt, cr, display_as="numbers", genType="logG-logT") )
        tbls.append( w.RotationAxisVsTargetTable(self.gs, self.tgt, cr ) )
        tbls.append( w.GateDecompTable(self.gs, cr) )
        tbls.append( w.RotationAxisTable(self.gs, cr) )
        tbls.append( w.GateEigenvalueTable(self.gs, self.tgt, cr) )
        tbls.append( w.DataSetOverviewTable(self.ds, self.tgt, maxLengthList=self.gss.Ls) )
        tbls.append( w.FitComparisonTable(self.gss.Ls, self.results.gatestring_structs['iteration'],
                                          self.results.estimates['default'].gatesets['iteration estimates'], self.ds,) )

        prepStrs = self.results.gatestring_lists['prep fiducials']
        effectStrs = self.results.gatestring_lists['effect fiducials']
        tbls.append( w.GatestringTable((prepStrs,effectStrs),
                                       ["Prep.","Measure"], commonTitle="Fiducials"))
        
        tbls.append( w.GatesSingleMetricTable( [self.gs,self.gs], ['one','two'],
                                               self.tgt, metric="infidelity") )
        tbls.append( w.GatesSingleMetricTable( [self.gs,self.gs], ['one','two'],
                                               self.tgt, metric="diamond") )
        tbls.append( w.GatesSingleMetricTable( [self.gs,self.gs], ['one','two'],
                                               self.tgt, metric="jtrace") )


        tbls.append( w.StandardErrgenTable(4, "hamiltonian", "pp") )
        tbls.append( w.StandardErrgenTable(4, "stochastic", "pp") )
        tbls.append( w.StandardErrgenTable(4, "hamiltonian", "gm") )
        tbls.append( w.StandardErrgenTable(4, "stochastic", "gm") )
        
        tbls.append( w.GaugeOptParamsTable(self.results.estimates['default'].goparameters['go0']) )
        tbls.append( w.MetadataTable(self.gs, self.results.estimates['default'].parameters ) )
        tbls.append( w.SoftwareEnvTable() )

        #Now test table rendering in html
        for tbl in tbls:
            out_html = tbl.render("html")
            #out_latex = tbl.render("latex") #not supported yet (figure formatting wants scratchdir)


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

        from pygsti.algorithms import directx as dx
        specs = pygsti.construction.build_spam_specs(
                prepStrs=prepStrs,
                effectStrs=effectStrs,
                prep_labels=self.gs.get_prep_labels(),
                effect_labels=self.gs.get_effect_labels() )
        baseStrs = self.gss.get_basestrings()
        directGatesets = dx.direct_mlgst_gatesets(
            baseStrs, self.ds, specs, self.tgt, svdTruncateTo=4)
        plts.append( w.ColorBoxPlot(["chi2","logl","blank",'directchi2','directlogl'], self.gss,
                                    self.ds, self.gs, boxLabels=False, directGSTgatesets=directGatesets) )
        
        gmx = np.identity(4,'d'); gmx[3,0] = 0.5
        plts.append( w.GateMatrixPlot(gmx, -1,1, "pp", 2, "in", "out", boxLabels=True) )
        plts.append( w.PolarEigenvaluePlot([np.linalg.eigvals(self.gs.gates['Gx'])],["purple"],scale=1.5) )

        projections = np.zeros(16,'d')
        plts.append( w.ProjectionsBoxPlot(projections, "pp", boxLabels=False) )
        plts.append( w.ProjectionsBoxPlot(projections, "gm", boxLabels=True) )
        
        choievals = np.array([-0.03, 0.02, 0.04, 0.98])
        choieb = np.array([0.05, 0.01, 0.02, 0.01])
        plts.append( w.ChoiEigenvalueBarPlot(choievals, None) )
        plts.append( w.ChoiEigenvalueBarPlot(choievals, choieb) )
                     
        #Now test table rendering in html
        for plt in plts:
            out_html = plt.render("html")


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
