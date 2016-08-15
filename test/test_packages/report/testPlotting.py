import unittest
import pygsti
from ..testutils import compare_files, temp_files

import numpy as np

from .reportBaseCase import ReportBaseCase

class TestPlotting(ReportBaseCase):

    def setUp(self):
        super(TestPlotting, self).setUp()

    def test_plotting(self):
        test_data = np.array( [[1e-8,1e-7,1e-6,1e-5],
                               [1e-4,1e-3,1e-2,1e-1],
                               [1.0,10.0,100.0,1000.0],
                               [1.0e4,1.0e5,1.0e6,1.0e7]],'d' )
        cmap = pygsti.report.plotting.StdColormapFactory('seq', n_boxes=10, vmin=0, vmax=1, dof=1)
        gstFig = pygsti.report.plotting.color_boxplot( test_data, cmap, size=(10,10), prec="compacthp",save_to=temp_files + "/test.pdf")
        gstFig = pygsti.report.plotting.color_boxplot( test_data, cmap, size=(10,10), prec="compact",save_to="")
        gstFig = pygsti.report.plotting.color_boxplot( test_data, cmap, size=(10,10), prec=3,save_to="")
        gstFig = pygsti.report.plotting.color_boxplot( test_data, cmap, size=(10,10), prec=-3,save_to="")
        gstFig = pygsti.report.plotting.color_boxplot( test_data, cmap, size=(10,10), prec="foobar",colorbar=True,save_to="")
        gstFig.check()

        pygsti.report.plotting.gateset_with_lgst_gatestring_estimates( [('Gx','Gx')], self.ds, self.specs,
                                                                       self.targetGateset,includeTargetGates=False,
                                                                       gateStringLabels=None, svdTruncateTo=4, verbosity=0)

        gsWithGxgx = pygsti.report.plotting.focused_mc2gst_gatesets(
            pygsti.construction.gatestring_list([('Gx','Gx')]), self.ds, self.specs, self.gs_clgst)
