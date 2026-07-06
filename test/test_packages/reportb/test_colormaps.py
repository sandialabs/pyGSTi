import numpy as np
import plotly.graph_objs as go

import pygsti
from pygsti.report import colormaps as cmap
from pygsti.report.figure import ReportFigure
from ..testutils import BaseTestCase, temp_files


class ColormapTests(BaseTestCase):

    def test_mpl_conversion(self):
        try:
            import matplotlib
            from pygsti.report import mpl_colormaps
            from pygsti.report.mpl_colormaps import plotly_to_matplotlib
        except ImportError:
            return # no matplotlib => stop here

        #Quick test of linlog inverse function (never used)
        llcmap = cmap.LinlogColormap(1.0, 10.0, 16, 0.95, 1, "red")
        mpl = pygsti.report.mpl_colormaps.MplLinLogNorm(llcmap)
        x = mpl.inverse(1.0) # test out inverse() function
        xar = mpl.inverse(np.array([1.0,2.0],'d'))


        # Test plotly -> matplotlib conversion
        data = []
        data.append( go.Scatter(
                x = [1,2,3],
                y = [3,4,5],
                mode = 'lines+markers',
                line = dict(width=2, color="blue",dash='dash'),
                name = 'Test1',
                showlegend=True
            ))
        data.append( go.Scattergl(x=[2,4,6], y=[1,3,9],
                                  mode="markers",
                                  showlegend=True) )

        layout = go.Layout(
	    width=800,
            height=400,
            title=dict(text="my title", font=dict(size=16)),
	    xaxis=dict(
                title=dict(text="my xlabel", font=dict(size=14)),
                side="top",
                type="log",
            ),
            yaxis=dict(
                title=dict(text='Mean survival probability', font=dict(size=14)),
                side="right",
                type="log",
            ),
            legend=dict(
                font=dict(
                    size=13,
                ),
            )
        )
        plotly_fig = go.Figure(data=list(data), layout=layout)
        pygsti_fig = ReportFigure(plotly_fig, colormap=None, python_value=None)

        mpl_fig = plotly_to_matplotlib(pygsti_fig)

        plotly_to_matplotlib(pygsti_fig,temp_files + "/testMPL.pdf")

        with self.assertRaises(ValueError):
            fig = ReportFigure(plotly_fig, colormap=None, python_value=None, special="foobar")
            plotly_to_matplotlib(fig)

        #Heatmap
        nX = 5
        nY = 3
        heatmap_data = [ go.Heatmap(z=np.ones((nY,nX),'d'),
                                    colorscale=[ [0, 'white'], [1, 'black'] ],
                                    showscale=False, zmin=0,zmax=1,hoverinfo='none') ]
        seqcmap = cmap.SequentialColormap(0, 10.0, "whiteToBlack")
        layout['xaxis']['type'] = "linear"  # heatmaps don't play well with
        layout['yaxis']['type'] = "linear"  # log scales
        heatmap_fig = ReportFigure(go.Figure(data=heatmap_data, layout=layout), seqcmap, plt_data=np.ones((nY,nX),'d'))
        plotly_to_matplotlib(heatmap_fig, temp_files + "/testMPLHeatmap.pdf")

        #bad mode
        with self.assertRaises(ValueError):
            data = [ go.Scatter(x = [0,1,2], y = [3,4,5],
                                mode = 'foobar') ] # invalid mode

            #Can't always test plotly_to_matplotlib w/bad mode now b/c plotly v3 validates it above
            fig = ReportFigure(go.Figure(data=data, layout=layout))
            plotly_to_matplotlib(fig) #invalid mode

    def test_mpl_conversion_heatmap_without_plt_data(self):
        """Minimal reproducer for the ``KeyError: 'plt_data'`` seen in report generation.

        The multi-subplot matrix plot builders ``_matrices_color_boxplot`` and
        ``_opmatrices_color_boxplot`` (pygsti/report/workspaceplots.py) return a
        ``ReportFigure`` that contains a ``heatmap`` trace but does *not* carry
        ``plt_data`` (or a ``colormap``) in its metadata.  When such a figure is
        rendered to matplotlib (PDF reports, or HTML reports with ``link_to``
        including 'pdf'/'pkl'), ``plotly_to_matplotlib`` fails at
        ``pygsti/report/mpl_colormaps.py`` with ``KeyError: 'plt_data'``.

        This test constructs the minimal figure that triggers the bug and asserts
        the *desired* behavior: the figure should render to matplotlib without
        raising.  While the bug is present this test FAILS (the KeyError is the
        undesired behavior we want surfaced).
        """
        try:
            import matplotlib
            from pygsti.report.mpl_colormaps import plotly_to_matplotlib
        except ImportError:
            return  # no matplotlib => stop here

        nX, nY = 5, 3
        heatmap_data = [go.Heatmap(z=np.ones((nY, nX), 'd'),
                                   colorscale=[[0, 'white'], [1, 'black']],
                                   showscale=False, zmin=0, zmax=1, hoverinfo='none')]
        layout = go.Layout(width=800, height=400,
                           xaxis=dict(type="linear"),
                           yaxis=dict(type="linear"))

        # Mimic _matrices_color_boxplot / _opmatrices_color_boxplot: the figure is
        # built with ReportFigure(fig) -- no colormap and no plt_data metadata.
        fig = ReportFigure(go.Figure(data=heatmap_data, layout=layout))

        # Desired behavior: this should render without raising.  While the bug is
        # present, plotly_to_matplotlib raises ``KeyError: 'plt_data'`` and this
        # call fails the test.
        plotly_to_matplotlib(fig, temp_files + "/testMPLHeatmapNoPltData.pdf")


if __name__ == '__main__':
    import unittest
    unittest.main(verbosity=1)
