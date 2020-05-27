import numpy as np

from ..util import BaseCase, needs_matplotlib

from pygsti.report import colormaps as cmap


class ColormapHelperTester(BaseCase):
    def test_as_rgb_array(self):
        self.assertArraysEqual(cmap.to_rgb_array("rgb(0,255,128)"), np.array([0, 255, 128]))
        self.assertArraysEqual(cmap.to_rgb_array("rgba(0,255,128,0.5)"), np.array([0, 255, 128]))
        self.assertArraysEqual(cmap.to_rgb_array("#00FF80"), np.array([0, 255, 128]))

    def test_as_rgb_array_raises_on_arg_parse_error(self):
        with self.assertRaises(ValueError):
            cmap.to_rgb_array("foobar")

    def test_interpolate_plotly_colorscale(self):
        plotlyColorscale = [(0.0, "#FFFFFF"), (1.0, "rgb(0,0,0)")]
        self.assertEqual(cmap.interpolate_plotly_colorscale(plotlyColorscale, 0.5), 'rgb(128,128,128)')
        self.assertEqual(cmap.interpolate_plotly_colorscale(plotlyColorscale, 0.0), 'rgb(255,255,255)')
        self.assertEqual(cmap.interpolate_plotly_colorscale(plotlyColorscale, 1.0), 'rgb(0,0,0)')


class ColormapInstanceBase(object):
    @needs_matplotlib
    def test_get_matplotlib_norm_and_cmap(self):
        mpl_norm, mpl_cmap = self.cmap.create_matplotlib_norm_and_cmap()
        # TODO assert correctness


class MulticolorInstance(object):
    """Derived test case base class for Colormaps supporting a finite set of color parameters"""

    def setUp(self):
        super(MulticolorInstance, self).setUp()
        self.cmap = self._construct()  # default color

    def test_color_support(self):
        # assert that the expected range of colors is supported
        for color in self.colors:
            self._construct(color=color)

    def test_constructor_raises_on_unknown_color(self):
        with self.assertRaises(ValueError):
            self._construct(color='foobar')


class BasicColormapTester(ColormapInstanceBase, BaseCase):
    def setUp(self):
        super(BasicColormapTester, self).setUp()
        colors = [(0.0, (1.0, 1.0, 1.0)), (1.0, (0., 0., 0.))]
        self.cmap = cmap.Colormap(colors, 1.0, 10.0)

    def test_construction(self):
        self.skipTest("TODO assert correctness")


class LinlogColormapTester(MulticolorInstance, BaseCase):
    colors = ['red', 'blue', 'green', 'cyan', 'yellow', 'purple']

    def _construct(self, **kwargs):
        return cmap.LinlogColormap(1.0, 10.0, 16, 0.95, 1, **kwargs)

    def test_construction(self):
        self.skipTest("TODO assert correctness")


class DivergingColormapTester(MulticolorInstance, BaseCase):
    colors = ['RdBu']

    def _construct(self, **kwargs):
        return cmap.DivergingColormap(1.0, 10.0, **kwargs)

    def test_construction(self):
        self.skipTest("TODO assert correctness")


class SequentialColormapTester(MulticolorInstance, BaseCase):
    colors = ['whiteToBlack', 'blackToWhite', 'whiteToBlue', 'whiteToRed']

    def _construct(self, **kwargs):
        return cmap.SequentialColormap(1.0, 10.0, **kwargs)

    def test_construction(self):
        self.skipTest("TODO assert correctness")


class PiecewiseLinearColormapTester(ColormapInstanceBase, BaseCase):
    def setUp(self):
        super(PiecewiseLinearColormapTester, self).setUp()
        colors = [(0.0, (1.0, 1.0, 1.0)), (1.0, (0., 0., 0.))]
        self.cmap = cmap.PiecewiseLinearColormap(colors)

    def test_construction(self):
        self.skipTest("TODO assert correctness")
