import unittest
import pygsti
import numpy as np
import sys
import warnings
import os

from ..testutils import BaseTestCase, temp_files, compare_files
from pygsti.report import colormaps as cmap

try:
    import matplotlib
    bMPL = True
except ImportError:
    bMPL = False


class ColormapTests(BaseTestCase):
    def setUp(self):
        super(ColormapTests, self).setUp()

    def test_colormap_helpers(self):
        self.assertTrue(np.all(cmap.as_rgb_array("rgb(0,255,128)") ==  np.array([0,255,128])))
        self.assertTrue(np.all(cmap.as_rgb_array("rgba(0,255,128,0.5)") ==  np.array([0,255,128])))
        self.assertTrue(np.all(cmap.as_rgb_array("#00FF80") ==  np.array([0,255,128])))
        with self.assertRaises(ValueError):
            cmap.as_rgb_array("foobar")

        plotlyColorscale = [ (0.0,"#FFFFFF"), (1.0,"rgb(0,0,0)") ]
        print(cmap.interpolate_plotly_colorscale(plotlyColorscale, 0.5))
        self.assertEqual(cmap.interpolate_plotly_colorscale(plotlyColorscale, 0.5), 'rgb(128,128,128)')
        self.assertEqual(cmap.interpolate_plotly_colorscale(plotlyColorscale, 0.0), 'rgb(255,255,255)')
        self.assertEqual(cmap.interpolate_plotly_colorscale(plotlyColorscale, 1.0), 'rgb(0,0,0)')

    def test_colormap_construction(self):

        colors = [ (0.0, (1.0,1.0,1.0)), (1.0,(0.,0.,0.)) ]
        base_cmap = cmap.Colormap(colors, 1.0, 10.0)
        if bMPL:
            mpl_norm, mpl_cmap = base_cmap.get_matplotlib_norm_and_cmap()

        for llcolor in ("red","blue","green","cyan","yellow","purple"):
            llcmap = cmap.LinlogColormap(1.0, 10.0, 16, 0.95, 1, llcolor)
            if bMPL:
                mpl_norm, mpl_cmap = llcmap.get_matplotlib_norm_and_cmap()
            
        with self.assertRaises(ValueError):
            cmap.LinlogColormap(1.0, 10.0, 16, 0.95, 1, color="foobar")

        divcmap = cmap.DivergingColormap(1.0, 10.0, color="RdBu")
        with self.assertRaises(ValueError):
            cmap.DivergingColormap(1.0, 10.0, color="foobar")

        for seqcolor in ("whiteToBlack","blackToWhite","whiteToBlue","whiteToRed"):
            seqcmap = cmap.SequentialColormap(1.0, 10.0, seqcolor)
        with self.assertRaises(ValueError):
            cmap.SequentialColormap(1.0, 10.0, color="foobar")

        cmap.PiecewiseLinearColormap( colors )
            

        
if __name__ == '__main__':
    import unittest
    unittest.main(verbosity=1)
