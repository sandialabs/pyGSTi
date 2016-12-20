import unittest
import warnings
import pickle
import collections
import pygsti
import os
from pygsti.construction import std1Q_XYI as std
from ..testutils import BaseTestCase, compare_files, temp_files

import numpy as np


class ReportBaseCase(BaseTestCase):

    def setUp(self):
        super(ReportBaseCase, self).setUp()

        self.targetGateset = std.gs_target
        datagen_gateset = self.targetGateset.depolarize(gate_noise=0.05, spam_noise=0.1)

        self.fiducials = std.fiducials
        self.germs = std.germs

        self.specs = pygsti.construction.build_spam_specs(self.fiducials, effect_labels=['E0']) #only use the first EVec

        self.gateLabels = list(self.targetGateset.gates.keys()) # also == std.gates
        self.lgstStrings = pygsti.construction.list_lgst_gatestrings(self.specs, self.gateLabels)

        self.maxLengthList = [0,1,2,4,8]

        self.lsgstStrings = pygsti.construction.make_lsgst_lists(
            self.gateLabels, self.fiducials, self.fiducials, self.germs, self.maxLengthList)

        self.ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/reportgen.dataset")

        # RUN BELOW LINES TO GENERATE ANALYSIS DATASET
        #ds = pygsti.construction.generate_fake_data(datagen_gateset, lsgstStrings[-1], nSamples=1000,
        #                                            sampleError='binomial', seed=100)
        #ds.save(compare_files + "/reportgen.dataset")

        gs_lgst = pygsti.do_lgst(self.ds, self.specs, self.targetGateset, svdTruncateTo=4, verbosity=0)
        #gs_lgst_go = pygsti.optimize_gauge(gs_lgst,"target",targetGateset=self.targetGateset,gateWeight=1.0,spamWeight=0.0) #DEPRECATED
        gs_lgst_go = pygsti.gaugeopt_to_target(gs_lgst, self.targetGateset, {'gates': 1.0, 'spam': 0.0})
        self.gs_clgst = pygsti.contract(gs_lgst_go, "CPTP")
        self.gs_clgst_tp = pygsti.contract(self.gs_clgst, "vSPAM")
        self.gs_clgst_tp.set_all_parameterizations("TP")


        try:
            import pptx
            self.have_python_pptx = True
        except ImportError:
            warnings.warn("**** IMPORT: Cannot import pptx (python-pptx), and so" +
                         " Powerpoint slide generation tests have been disabled.")
            self.have_python_pptx = False


        #Compute results for MC2GST
        lsgst_gatesets_prego = pygsti.do_iterative_mc2gst(self.ds, self.gs_clgst, self.lsgstStrings, verbosity=0,
                                                          minProbClipForWeighting=1e-6, probClipInterval=(-1e6,1e6),
                                                          returnAll=True)
        gaugeOptParams = collections.OrderedDict([
                ('TPpenalty', 0),
                ('CPpenalty', 0),
                ('gatesMetric',"frobenius"),
                ('spamMetric',"frobenius"),
                ('itemWeights', {'gates': 1.0, 'spam': 0.001}) ])

        lsgst_gatesets = []
        for gs in lsgst_gatesets_prego:
            lsgst_gatesets.append( pygsti.gaugeopt_to_target(gs,self.targetGateset,
                                                             **gaugeOptParams) )

        self.results = pygsti.report.Results()
        self.results.init_Ls_and_germs("chi2", self.targetGateset, self.ds, self.gs_clgst,
                                       self.maxLengthList, self.germs,
                                       lsgst_gatesets, self.lsgstStrings, self.fiducials, self.fiducials,
                                       pygsti.construction.repeat_with_max_length, None, lsgst_gatesets_prego)
        self.results.parameters.update({'minProbClip': 1e-6, 'minProbClipForWeighting': 1e-4,
                                        'probClipInterval': (-1e6,1e6), 'radius': 1e-4,
                                        'weights': None, 'defaultDirectory': temp_files + "",
                                        'defaultBasename': "MyDefaultReportName",
                                        'gaugeOptParams': gaugeOptParams} )
        self.results.options.precision = 3
        self.results.options.polar_precision = 2



        #Compute results for MLGST with TP constraint
        lsgst_gatesets_TP = pygsti.do_iterative_mlgst(self.ds, self.gs_clgst_tp, self.lsgstStrings, verbosity=0,
                                                   minProbClip=1e-4, probClipInterval=(-1e6,1e6),
                                                   returnAll=True) #TP initial gateset => TP output gatesets
        tp_target = self.targetGateset.copy(); tp_target.set_all_parameterizations("TP")
        lsgst_gatesets_TP = [ pygsti.gaugeopt_to_target(gs, tp_target, {'gates': 1.0, 'spam': 0.001})
                              for gs in lsgst_gatesets_TP ]

        self.results_logL = pygsti.report.Results()
        self.results_logL.init_Ls_and_germs("logl", self.targetGateset, self.ds, self.gs_clgst_tp, self.maxLengthList, self.germs,
                                     lsgst_gatesets_TP, self.lsgstStrings, self.fiducials, self.fiducials,
                                     pygsti.construction.repeat_with_max_length)
        self.results_logL.options.precision = 3
        self.results_logL.options.polar_precision = 2

        try:
            basestring #Only defined in Python 2
            self.versionsuffix = "" #Python 2
        except NameError:
            self.versionsuffix = "v3" #Python 3
