'''
Graph
======

The :class:`Graph` widget is a widget for displaying plots. It supports
drawing multiple plot with different colors on the Graph. It also supports
axes titles, ticks, labeled ticks, grids and a log or linear representation on
both the x and y axis, independently.

To display a plot. First create a graph which will function as a "canvas" for
the plots. Then create plot objects e.g. MeshLinePlot and add them to the
graph.

To create a graph with x-axis between 0-100, y-axis between -1 to 1, x and y
labels of and X and Y, respectively, x major and minor ticks every 25, 5 units,
respectively, y major ticks every 1 units, full x and y grids and with
a red line plot containing a sin wave on this range::

    from kivy_garden.graph import Graph, MeshLinePlot
    graph = Graph(xlabel='X', ylabel='Y', x_ticks_minor=5,
                  x_ticks_major=25, y_ticks_major=1,
                  y_grid_label=True, x_grid_label=True, padding=5,
                  x_grid=True, y_grid=True, xmin=-0, xmax=100, ymin=-1, ymax=1)
    plot = MeshLinePlot(color=[1, 0, 0, 1])
    plot.points = [(x, sin(x / 10.)) for x in range(0, 101)]
    graph.add_plot(plot)

The MeshLinePlot plot is a particular plot which draws a set of points using
a mesh object. The points are given as a list of tuples, with each tuple
being a (x, y) coordinate in the graph's units.

You can create different types of plots other than MeshLinePlot by inheriting
from the Plot class and implementing the required functions. The Graph object
provides a "canvas" to which a Plot's instructions are added. The plot object
is responsible for updating these instructions to show within the bounding
box of the graph the proper plot. The Graph notifies the Plot when it needs
to be redrawn due to changes. See the MeshLinePlot class for how it is done.

The current availables plots are:

    * `MeshStemPlot`
    * `MeshLinePlot`
    * `SmoothLinePlot` - require Kivy 1.8.1

.. note::

    The graph uses a stencil view to clip the plots to the graph display area.
    As with the stencil graphics instructions, you cannot stack more than 8
    stencil-aware widgets.

'''

__all__ = ('Graph', 'Plot', 'MeshLinePlot', 'MeshStemPlot', 'LinePlot',
           'SmoothLinePlot', 'ContourPlot', 'ScatterPlot', 'PointPlot')

import itertools
import numpy as _np

from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.stencilview import StencilView
from kivy.properties import NumericProperty, BooleanProperty,\
    BoundedNumericProperty, StringProperty, ListProperty, ObjectProperty,\
    DictProperty, AliasProperty
from kivy.clock import Clock
from kivy.graphics import Mesh, Color, Rectangle, Point
from kivy.graphics import Fbo
from kivy.graphics.texture import Texture
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.logger import Logger
from kivy import metrics
from kivy.core.text import Label as CoreLabel
from kivy.core.window import Window

from pygsti.report.kivywidget import WrappedLabel

from math import log10, floor, ceil
from decimal import Decimal
from itertools import chain
try:
    import numpy as np
except ImportError as e:
    np = None

from pygsti.report import plothelpers as _ph


def identity(x):
    return x


def exp10(x):
    return 10 ** x


Builder.load_string("""
<GraphRotatedLabel>:
    canvas.before:
        PushMatrix
        Rotate:
            angle: root.angle
            axis: 0, 0, 1
            origin: root.center
    canvas.after:
        PopMatrix
""")


class GraphRotatedLabel(Label):
    angle = NumericProperty(0)


class Axis(EventDispatcher):
    pass


class XAxis(Axis):
    pass


class YAxis(Axis):
    pass


class Graph(Widget):
    '''Graph class, see module documentation for more information.
    '''

    # triggers a full reload of graphics
    _trigger = ObjectProperty(None)
    # triggers only a repositioning of objects due to size/pos updates
    _trigger_size = ObjectProperty(None)
    # triggers only a update of colors, e.g. tick_color
    _trigger_color = ObjectProperty(None)
    # holds widget with the x-axis label
    _xlabel = ObjectProperty(None)
    # holds widget with the y-axis label
    _ylabel = ObjectProperty(None)
    # holds all the x-axis tick mark labels
    _x_grid_label = ListProperty([])
    # holds all the y-axis tick mark labels
    _y_grid_label = ListProperty([])
    # the mesh drawing all the ticks/grids
    _mesh_ticks = ObjectProperty(None)
    # the mesh which draws the surrounding rectangle
    _mesh_rect = ObjectProperty(None)
    # a list of locations of major and minor ticks. The values are not
    # but is in the axis min - max range
    _ticks_majorx = ListProperty([])
    _ticks_minorx = ListProperty([])
    _ticks_majory = ListProperty([])
    _ticks_minory = ListProperty([])

    tick_color = ListProperty([.5, .5, .5, 1])
    '''Color of the grid/ticks, default to 1/2. grey.
    '''

    background_color = ListProperty([0, 0, 0, 0])
    '''Color of the background, defaults to transparent
    '''

    border_color = ListProperty([1, 1, 1, 1])
    '''Color of the border, defaults to white
    '''

    label_options = DictProperty()
    '''Label options that will be passed to `:class:`kivy.uix.Label`.
    '''

    _with_stencilbuffer = BooleanProperty(True)
    '''Whether :class:`Graph`'s FBO should use FrameBuffer (True) or not
    (False).

    .. warning:: This property is internal and so should be used with care.
    It can break some other graphic instructions used by the :class:`Graph`,
    for example you can have problems when drawing :class:`SmoothLinePlot`
    plots, so use it only when you know what exactly you are doing.

    :data:`_with_stencilbuffer` is a :class:`~kivy.properties.BooleanProperty`,
    defaults to True.
    '''

    def __init__(self, **kwargs):
        super(Graph, self).__init__(**kwargs)

        with self.canvas:
            self._fbo = Fbo(
                size=self.size, with_stencilbuffer=self._with_stencilbuffer)

        with self._fbo:
            self._background_color = Color(*self.background_color)
            self._background_rect = Rectangle(size=self.size)
            self._mesh_ticks_color = Color(*self.tick_color)
            self._mesh_ticks = Mesh(mode='lines')
            self._mesh_rect_color = Color(*self.border_color)
            self._mesh_rect = Mesh(mode='line_strip')

        with self.canvas:
            Color(1, 1, 1)
            self._fbo_rect = Rectangle(
                size=self.size, texture=self._fbo.texture)

        mesh = self._mesh_rect
        mesh.vertices = [0] * (5 * 4)
        mesh.indices = range(5)

        self._plot_area = StencilView()
        self.add_widget(self._plot_area)

        t = self._trigger = Clock.create_trigger(self._redraw_all)
        ts = self._trigger_size = Clock.create_trigger(self._redraw_size)
        tc = self._trigger_color = Clock.create_trigger(self._update_colors)

        self.bind(center=ts, padding=ts, precision=ts, plots=ts, x_grid=ts,
                  y_grid=ts, draw_border=ts)
        self.bind(xmin=t, xmax=t, xlog=t, x_ticks_major=t, x_ticks_minor=t,
                  xlabel=t, x_grid_label=t, ymin=t, ymax=t, ylog=t,
                  y_ticks_major=t, y_ticks_minor=t, ylabel=t, y_grid_label=t,
                  font_size=t, label_options=t, x_ticks_angle=t,
                  x_grid_labels=t, y_grid_labels=t, x_tick_offset=t, y_tick_offset=t)
        self.bind(tick_color=tc, background_color=tc, border_color=tc)
        self._trigger()

    def add_widget(self, widget):
        if widget is self._plot_area:
            canvas = self.canvas
            self.canvas = self._fbo
        super(Graph, self).add_widget(widget)
        if widget is self._plot_area:
            self.canvas = canvas

    def remove_widget(self, widget):
        if widget is self._plot_area:
            canvas = self.canvas
            self.canvas = self._fbo
        super(Graph, self).remove_widget(widget)
        if widget is self._plot_area:
            self.canvas = canvas

    def _get_ticks(self, major, minor, log, s_min, s_max):
        if major and s_max > s_min:
            if log:
                s_min = log10(s_min)
                s_max = log10(s_max)
                # count the decades in min - max. This is in actual decades,
                # not logs.
                n_decades = floor(s_max - s_min)
                # for the fractional part of the last decade, we need to
                # convert the log value, x, to 10**x but need to handle
                # differently if the last incomplete decade has a decade
                # boundary in it
                if floor(s_min + n_decades) != floor(s_max):
                    n_decades += 1 - (10 ** (s_min + n_decades + 1) - 10 **
                                      s_max) / 10 ** floor(s_max + 1)
                else:
                    n_decades += ((10 ** s_max - 10 ** (s_min + n_decades)) /
                                  10 ** floor(s_max + 1))
                # this might be larger than what is needed, but we delete
                # excess later
                n_ticks_major = n_decades / float(major)
                n_ticks = int(floor(n_ticks_major * (minor if minor >=
                                                     1. else 1.0))) + 2
                # in decade multiples, e.g. 0.1 of the decade, the distance
                # between ticks
                decade_dist = major / float(minor if minor else 1.0)

                points_minor = [0] * n_ticks
                points_major = [0] * n_ticks
                k = 0  # position in points major
                k2 = 0  # position in points minor
                # because each decade is missing 0.1 of the decade, if a tick
                # falls in < min_pos skip it
                min_pos = 0.1 - 0.00001 * decade_dist
                s_min_low = floor(s_min)
                # first real tick location. value is in fractions of decades
                # from the start we have to use decimals here, otherwise
                # floating point inaccuracies results in bad values
                start_dec = ceil((10 ** Decimal(s_min - s_min_low - 1)) /
                                 Decimal(decade_dist)) * decade_dist
                count_min = (0 if not minor else
                             floor(start_dec / decade_dist) % minor)
                start_dec += s_min_low
                count = 0  # number of ticks we currently have passed start
                while True:
                    # this is the current position in decade that we are.
                    # e.g. -0.9 means that we're at 0.1 of the 10**ceil(-0.9)
                    # decade
                    pos_dec = start_dec + decade_dist * count
                    pos_dec_low = floor(pos_dec)
                    diff = pos_dec - pos_dec_low
                    zero = abs(diff) < 0.001 * decade_dist
                    if zero:
                        # the same value as pos_dec but in log scale
                        pos_log = pos_dec_low
                    else:
                        pos_log = log10((pos_dec - pos_dec_low
                                         ) * 10 ** ceil(pos_dec))
                    if pos_log > s_max:
                        break
                    count += 1
                    if zero or diff >= min_pos:
                        if minor and count_min % minor:
                            points_minor[k2] = pos_log
                            k2 += 1
                        else:
                            points_major[k] = pos_log
                            k += 1
                        # OLD (broken?) if minor and not count_min % minor:
                        # OLD (broken?)     points_major[k] = pos_log
                        # OLD (broken?)     k += 1
                        # OLD (broken?) else:
                        # OLD (broken?)     points_minor[k2] = pos_log
                        # OLD (broken?)     k2 += 1
                    count_min += 1
            else:
                # distance between each tick
                tick_dist = major / float(minor if minor else 1.0)
                n_ticks = int(floor((s_max - s_min) / tick_dist) + 1)
                points_major = [0] * int(floor((s_max - s_min) / float(major))
                                         + 1)
                points_minor = [0] * (n_ticks - len(points_major) + 1)
                k = 0  # position in points major
                k2 = 0  # position in points minor
                for m in range(0, n_ticks):
                    if minor and m % minor:
                        points_minor[k2] = m * tick_dist + s_min
                        k2 += 1
                    else:
                        points_major[k] = m * tick_dist + s_min
                        k += 1
            del points_major[k:]
            del points_minor[k2:]
        else:
            points_major = []
            points_minor = []
        return points_major, points_minor

    def _update_labels(self):
        xlabel = self._xlabel
        ylabel = self._ylabel
        x = self.x
        y = self.y
        width = self.width
        height = self.height
        padding = self.padding
        x_next = padding + x
        y_next = padding + y
        xextent = width + x
        yextent = height + y
        ymin = self.ymin
        ymax = self.ymax
        xmin = self.xmin
        precision = self.precision
        x_overlap = False
        y_overlap = False
        # set up x and y axis labels
        if xlabel:
            xlabel.text = self.xlabel
            xlabel.texture_update()
            xlabel.size = xlabel.texture_size
            xlabel.pos = int(
                x + width / 2. - xlabel.width / 2.), int(padding + y)
            y_next += padding + xlabel.height
        if ylabel:
            ylabel.text = self.ylabel
            ylabel.texture_update()
            ylabel.size = ylabel.texture_size
            ylabel.x = padding + x - (ylabel.width / 2. - ylabel.height / 2.)
            x_next += padding + ylabel.height
        xpoints = self._ticks_majorx
        xlabels = self._x_grid_label
        xlabel_grid = self.x_grid_label
        ylabel_grid = self.y_grid_label
        x_grid_labels = self.x_grid_labels
        y_grid_labels = self.y_grid_labels
        ypoints = self._ticks_majory
        ylabels = self._y_grid_label
        # now x and y tick mark labels
        if len(ylabels) and ylabel_grid:
            # horizontal size of the largest tick label, to have enough room
            funcexp = exp10 if self.ylog else identity
            funclog = log10 if self.ylog else identity
            if y_grid_labels == 'auto':
                ylabels[0].text = precision % funcexp(ypoints[0])
            else:
                ylabels[0].text = y_grid_labels[0] if len(y_grid_labels) > 0 else ''
            ylabels[0].texture_update()

            y1 = ylabels[0].texture_size
            y_start = y_next + (padding + y1[1] if len(xlabels) and xlabel_grid
                                else 0) + \
                               (padding + y1[1] if not y_next else 0)
            yextent = y + height - padding - y1[1] / 2.

            ymin = funclog(ymin)
            ratio = (yextent - y_start) / float(funclog(ymax) - ymin)
            y_start -= y1[1] / 2.
            y1 = y1[0]
            for k in range(len(ylabels)):
                if y_grid_labels == 'auto':
                    ylabels[k].text = precision % funcexp(ypoints[k])
                else:
                    ylabels[k].text = y_grid_labels[k] if len(y_grid_labels) > k else ''
                ylabels[k].texture_update()

                ylabels[k].size = ylabels[k].texture_size
                y1 = max(y1, ylabels[k].texture_size[0])
                ylabels[k].pos = (
                    int(x_next),
                    int(y_start + (ypoints[k] - ymin) * ratio))
            if len(ylabels) > 1 and ylabels[0].top > ylabels[1].y:
                y_overlap = True
            else:
                x_next += y1 + padding
        if len(xlabels) and xlabel_grid:
            funcexp = exp10 if self.xlog else identity
            funclog = log10 if self.xlog else identity
            if x_grid_labels == 'auto':
                # find the distance from the end that'll fit the last tick label
                xlabels[0].text = precision % funcexp(xpoints[-1])
            else:
                xlabels[0].text = x_grid_labels[0] if len(x_grid_labels) > 0 else ''
            xlabels[0].texture_update()

            xextent = x + width - xlabels[0].texture_size[0] / 2. - padding
            # find the distance from the start that'll fit the first tick label
            if not x_next:
                xlabels[0].text = precision % funcexp(xpoints[0])
                xlabels[0].texture_update()
                x_next = padding + xlabels[0].texture_size[0] / 2.
            xmin = funclog(xmin)
            ratio = (xextent - x_next) / float(funclog(self.xmax) - xmin)
            right = -1
            for k in range(len(xlabels)):
                if x_grid_labels == 'auto':
                    xlabels[k].text = precision % funcexp(xpoints[k])
                else:
                    xlabels[k].text = x_grid_labels[k] if len(x_grid_labels) > k else ''
                # update the size so we can center the labels on ticks
                xlabels[k].texture_update()
                xlabels[k].size = xlabels[k].texture_size
                half_ts = xlabels[k].texture_size[0] / 2.
                xlabels[k].pos = (
                    int(x_next + (xpoints[k] - xmin) * ratio - half_ts),
                    int(y_next))
                if xlabels[k].x < right:
                    x_overlap = True
                    break
                right = xlabels[k].right
            if not x_overlap:
                y_next += padding + xlabels[0].texture_size[1]
        # now re-center the x and y axis labels
        if xlabel:
            xlabel.x = int(
                x_next + (xextent - x_next) / 2. - xlabel.width / 2.)
        if ylabel:
            ylabel.y = int(
                y_next + (yextent - y_next) / 2. - ylabel.height / 2.)
            ylabel.angle = 90
        if x_overlap: print("Graph X overlap -- omitting x labels!")
        if y_overlap: print("Graph Y overlap -- omitting y labels!")
        if x_overlap:
            for k in range(len(xlabels)):
                xlabels[k].text = ''
        if y_overlap:
            for k in range(len(ylabels)):
                ylabels[k].text = ''
        return x_next - x, y_next - y, xextent - x, yextent - y

    def _update_ticks(self, size):
        # re-compute the positions of the bounding rectangle
        mesh = self._mesh_rect
        vert = mesh.vertices
        if self.draw_border:
            s0, s1, s2, s3 = size
            vert[0] = s0
            vert[1] = s1
            vert[4] = s2
            vert[5] = s1
            vert[8] = s2
            vert[9] = s3
            vert[12] = s0
            vert[13] = s3
            vert[16] = s0
            vert[17] = s1
        else:
            vert[0:18] = [0 for k in range(18)]
        mesh.vertices = vert
        # re-compute the positions of the x/y axis ticks
        mesh = self._mesh_ticks
        vert = mesh.vertices
        start = 0
        xpoints = self._ticks_majorx
        ypoints = self._ticks_majory
        xpoints2 = self._ticks_minorx
        ypoints2 = self._ticks_minory
        ylog = self.ylog
        xlog = self.xlog
        xmin = self.xmin
        xmax = self.xmax
        if xlog:
            xmin = log10(xmin)
            xmax = log10(xmax)
        ymin = self.ymin
        ymax = self.ymax
        if ylog:
            ymin = log10(ymin)
            ymax = log10(ymax)
        if len(xpoints):
            top = size[3] if self.x_grid else metrics.dp(12) + size[1]
            ratio = (size[2] - size[0]) / float(xmax - xmin)
            for k in range(start, len(xpoints) + start):
                vert[k * 8] = size[0] + (xpoints[k - start] - xmin) * ratio
                vert[k * 8 + 1] = size[1]
                vert[k * 8 + 4] = vert[k * 8]
                vert[k * 8 + 5] = top
            start += len(xpoints)
        if len(xpoints2):
            top = metrics.dp(8) + size[1]
            ratio = (size[2] - size[0]) / float(xmax - xmin)
            for k in range(start, len(xpoints2) + start):
                vert[k * 8] = size[0] + (xpoints2[k - start] - xmin) * ratio
                vert[k * 8 + 1] = size[1]
                vert[k * 8 + 4] = vert[k * 8]
                vert[k * 8 + 5] = top
            start += len(xpoints2)
        if len(ypoints):
            top = size[2] if self.y_grid else metrics.dp(12) + size[0]
            ratio = (size[3] - size[1]) / float(ymax - ymin)
            for k in range(start, len(ypoints) + start):
                vert[k * 8 + 1] = size[1] + (ypoints[k - start] - ymin) * ratio
                vert[k * 8 + 5] = vert[k * 8 + 1]
                vert[k * 8] = size[0]
                vert[k * 8 + 4] = top
            start += len(ypoints)
        if len(ypoints2):
            top = metrics.dp(8) + size[0]
            ratio = (size[3] - size[1]) / float(ymax - ymin)
            for k in range(start, len(ypoints2) + start):
                vert[k * 8 + 1] = size[1] + (
                    ypoints2[k - start] - ymin) * ratio
                vert[k * 8 + 5] = vert[k * 8 + 1]
                vert[k * 8] = size[0]
                vert[k * 8 + 4] = top
        mesh.vertices = vert

    x_axis = ListProperty([None])
    y_axis = ListProperty([None])

    def get_x_axis(self, axis=0):
        if axis == 0:
            return self.xlog, self.xmin, self.xmax
        info = self.x_axis[axis]
        return info["log"], info["min"], info["max"]

    def get_y_axis(self, axis=0):
        if axis == 0:
            return self.ylog, self.ymin, self.ymax
        info = self.y_axis[axis]
        return info["log"], info["min"], info["max"]

    def add_x_axis(self, xmin, xmax, xlog=False):
        data = {
            "log": xlog,
            "min": xmin,
            "max": xmax
        }
        self.x_axis.append(data)
        return data

    def add_y_axis(self, ymin, ymax, ylog=False):
        data = {
            "log": ylog,
            "min": ymin,
            "max": ymax
        }
        self.y_axis.append(data)
        return data

    def _update_plots(self, size):
        for plot in self.plots:
            xlog, xmin, xmax = self.get_x_axis(plot.x_axis)
            ylog, ymin, ymax = self.get_y_axis(plot.y_axis)
            plot._update(xlog, xmin, xmax, ylog, ymin, ymax, size)

    def _update_colors(self, *args):
        self._mesh_ticks_color.rgba = tuple(self.tick_color)
        self._background_color.rgba = tuple(self.background_color)
        self._mesh_rect_color.rgba = tuple(self.border_color)

    def _redraw_all(self, *args):
        # add/remove all the required labels
        xpoints_major, xpoints_minor = self._redraw_x(*args)
        ypoints_major, ypoints_minor = self._redraw_y(*args)

        mesh = self._mesh_ticks
        n_points = (len(xpoints_major) + len(xpoints_minor) +
                    len(ypoints_major) + len(ypoints_minor))
        mesh.vertices = [0] * (n_points * 8)
        mesh.indices = [k for k in range(n_points * 2)]
        self._redraw_size()

    def _redraw_x(self, *args):
        font_size = self.font_size
        if self.xlabel:
            xlabel = self._xlabel
            if not xlabel:
                xlabel = Label()
                self.add_widget(xlabel)
                self._xlabel = xlabel

            xlabel.font_size = font_size
            for k, v in self.label_options.items():
                setattr(xlabel, k, v)

        else:
            xlabel = self._xlabel
            if xlabel:
                self.remove_widget(xlabel)
                self._xlabel = None
        grids = self._x_grid_label
        xpoints_major, xpoints_minor = self._get_ticks(self.x_ticks_major,
                                                       self.x_ticks_minor,
                                                       self.xlog, self.xmin + self.x_tick_offset,
                                                       self.xmax)
        self._ticks_majorx = xpoints_major
        self._ticks_minorx = xpoints_minor

        if not self.x_grid_label:
            n_labels = 0
        else:
            n_labels = len(xpoints_major)

        for k in range(n_labels, len(grids)):
            self.remove_widget(grids[k])
        del grids[n_labels:]

        grid_len = len(grids)
        grids.extend([None] * (n_labels - len(grids)))
        for k in range(grid_len, n_labels):
            grids[k] = GraphRotatedLabel(
                font_size=font_size, angle=self.x_ticks_angle,
                **self.label_options)
            self.add_widget(grids[k])
        return xpoints_major, xpoints_minor

    def _redraw_y(self, *args):
        font_size = self.font_size
        if self.ylabel:
            ylabel = self._ylabel
            if not ylabel:
                ylabel = GraphRotatedLabel()
                self.add_widget(ylabel)
                self._ylabel = ylabel

            ylabel.font_size = font_size
            for k, v in self.label_options.items():
                setattr(ylabel, k, v)
        else:
            ylabel = self._ylabel
            if ylabel:
                self.remove_widget(ylabel)
                self._ylabel = None
        grids = self._y_grid_label
        ypoints_major, ypoints_minor = self._get_ticks(self.y_ticks_major,
                                                       self.y_ticks_minor,
                                                       self.ylog, self.ymin + self.y_tick_offset,
                                                       self.ymax)
        self._ticks_majory = ypoints_major
        self._ticks_minory = ypoints_minor

        if not self.y_grid_label:
            n_labels = 0
        else:
            n_labels = len(ypoints_major)

        for k in range(n_labels, len(grids)):
            self.remove_widget(grids[k])
        del grids[n_labels:]

        grid_len = len(grids)
        grids.extend([None] * (n_labels - len(grids)))
        for k in range(grid_len, n_labels):
            grids[k] = Label(font_size=font_size, **self.label_options)
            self.add_widget(grids[k])
        return ypoints_major, ypoints_minor

    def _redraw_size(self, *args):
        # size a 4-tuple describing the bounding box in which we can draw
        # graphs, it's (x0, y0, x1, y1), which correspond with the bottom left
        # and top right corner locations, respectively
        self._clear_buffer()
        size = self._update_labels()
        self.view_pos = self._plot_area.pos = (size[0], size[1])
        self.view_size = self._plot_area.size = (
            size[2] - size[0], size[3] - size[1])

        if self.size[0] and self.size[1]:
            self._fbo.size = self.size
        else:
            self._fbo.size = 1, 1  # gl errors otherwise
        self._fbo_rect.texture = self._fbo.texture
        self._fbo_rect.size = self.size
        self._fbo_rect.pos = self.pos
        self._background_rect.size = self.size
        self._update_ticks(size)
        self._update_plots(size)

    def _clear_buffer(self, *largs):
        fbo = self._fbo
        fbo.bind()
        fbo.clear_buffer()
        fbo.release()

    def add_plot(self, plot):
        '''Add a new plot to this graph.

        :Parameters:
            `plot`:
                Plot to add to this graph.

        >>> graph = Graph()
        >>> plot = MeshLinePlot(mode='line_strip', color=[1, 0, 0, 1])
        >>> plot.points = [(x / 10., sin(x / 50.)) for x in range(-0, 101)]
        >>> graph.add_plot(plot)
        '''
        if plot in self.plots:
            return
        add = self._plot_area.canvas.add
        for instr in plot.get_drawings():
            add(instr)
        plot.bind(on_clear_plot=self._clear_buffer)
        self.plots.append(plot)

    def remove_plot(self, plot):
        '''Remove a plot from this graph.

        :Parameters:
            `plot`:
                Plot to remove from this graph.

        >>> graph = Graph()
        >>> plot = MeshLinePlot(mode='line_strip', color=[1, 0, 0, 1])
        >>> plot.points = [(x / 10., sin(x / 50.)) for x in range(-0, 101)]
        >>> graph.add_plot(plot)
        >>> graph.remove_plot(plot)
        '''
        if plot not in self.plots:
            return
        remove = self._plot_area.canvas.remove
        for instr in plot.get_drawings():
            remove(instr)
        plot.unbind(on_clear_plot=self._clear_buffer)
        self.plots.remove(plot)
        self._clear_buffer()

    def collide_plot(self, x, y):
        '''Determine if the given coordinates fall inside the plot area. Use
        `x, y = self.to_widget(x, y, relative=True)` to first convert into
        widget coordinates if it's in window coordinates because it's assumed
        to be given in local widget coordinates, relative to the graph's pos.

        :Parameters:
            `x, y`:
                The coordinates to test.
        '''
        adj_x, adj_y = x - self._plot_area.pos[0], y - self._plot_area.pos[1]
        return 0 <= adj_x <= self._plot_area.size[0] \
            and 0 <= adj_y <= self._plot_area.size[1]

    def to_data(self, x, y):
        '''Convert widget coords to data coords. Use
        `x, y = self.to_widget(x, y, relative=True)` to first convert into
        widget coordinates if it's in window coordinates because it's assumed
        to be given in local widget coordinates, relative to the graph's pos.

        :Parameters:
            `x, y`:
                The coordinates to convert.

        If the graph has multiple axes, use :class:`Plot.unproject` instead.
        '''
        adj_x = float(x - self._plot_area.pos[0])
        adj_y = float(y - self._plot_area.pos[1])
        norm_x = adj_x / self._plot_area.size[0]
        norm_y = adj_y / self._plot_area.size[1]
        if self.xlog:
            xmin, xmax = log10(self.xmin), log10(self.xmax)
            conv_x = 10.**(norm_x * (xmax - xmin) + xmin)
        else:
            conv_x = norm_x * (self.xmax - self.xmin) + self.xmin
        if self.ylog:
            ymin, ymax = log10(self.ymin), log10(self.ymax)
            conv_y = 10.**(norm_y * (ymax - ymin) + ymin)
        else:
            conv_y = norm_y * (self.ymax - self.ymin) + self.ymin
        return [conv_x, conv_y]

    xmin = NumericProperty(0.)
    '''The x-axis minimum value.

    If :data:`xlog` is True, xmin must be larger than zero.

    :data:`xmin` is a :class:`~kivy.properties.NumericProperty`, defaults to 0.
    '''

    xmax = NumericProperty(100.)
    '''The x-axis maximum value, larger than xmin.

    :data:`xmax` is a :class:`~kivy.properties.NumericProperty`, defaults to 0.
    '''

    xlog = BooleanProperty(False)
    '''Determines whether the x-axis should be displayed logarithmically (True)
    or linearly (False).

    :data:`xlog` is a :class:`~kivy.properties.BooleanProperty`, defaults
    to False.
    '''

    x_tick_offset = NumericProperty(0.)
    ''' Coordinate offset of the first major x tick '''
    
    x_ticks_major = BoundedNumericProperty(0, min=0)
    '''Distance between major tick marks on the x-axis.

    Determines the distance between the major tick marks. Major tick marks
    start from min and re-occur at every ticks_major until :data:`xmax`.
    If :data:`xmax` doesn't overlap with a integer multiple of ticks_major,
    no tick will occur at :data:`xmax`. Zero indicates no tick marks.

    If :data:`xlog` is true, then this indicates the distance between ticks
    in multiples of current decade. E.g. if :data:`xmin` is 0.1 and
    ticks_major is 0.1, it means there will be a tick at every 10th of the
    decade, i.e. 0.1 ... 0.9, 1, 2... If it is 0.3, the ticks will occur at
    0.1, 0.3, 0.6, 0.9, 2, 5, 8, 10. You'll notice that it went from 8 to 10
    instead of to 20, that's so that we can say 0.5 and have ticks at every
    half decade, e.g. 0.1, 0.5, 1, 5, 10, 50... Similarly, if ticks_major is
    1.5, there will be ticks at 0.1, 5, 100, 5,000... Also notice, that there's
    always a major tick at the start. Finally, if e.g. :data:`xmin` is 0.6
    and this 0.5 there will be ticks at 0.6, 1, 5...

    :data:`x_ticks_major` is a
    :class:`~kivy.properties.BoundedNumericProperty`, defaults to 0.
    '''

    x_ticks_minor = BoundedNumericProperty(0, min=0)
    '''The number of sub-intervals that divide x_ticks_major.

    Determines the number of sub-intervals into which ticks_major is divided,
    if non-zero. The actual number of minor ticks between the major ticks is
    ticks_minor - 1. Only used if ticks_major is non-zero. If there's no major
    tick at xmax then the number of minor ticks after the last major
    tick will be however many ticks fit until xmax.

    If self.xlog is true, then this indicates the number of intervals the
    distance between major ticks is divided. The result is the number of
    multiples of decades between ticks. I.e. if ticks_minor is 10, then if
    ticks_major is 1, there will be ticks at 0.1, 0.2...0.9, 1, 2, 3... If
    ticks_major is 0.3, ticks will occur at 0.1, 0.12, 0.15, 0.18... Finally,
    as is common, if ticks major is 1, and ticks minor is 5, there will be
    ticks at 0.1, 0.2, 0.4... 0.8, 1, 2...

    :data:`x_ticks_minor` is a
    :class:`~kivy.properties.BoundedNumericProperty`, defaults to 0.
    '''

    x_grid = BooleanProperty(False)
    '''Determines whether the x-axis has tick marks or a full grid.

    If :data:`x_ticks_major` is non-zero, then if x_grid is False tick marks
    will be displayed at every major tick. If x_grid is True, instead of ticks,
    a vertical line will be displayed at every major tick.

    :data:`x_grid` is a :class:`~kivy.properties.BooleanProperty`, defaults
    to False.
    '''

    x_grid_label = BooleanProperty(False)
    '''Whether labels should be displayed beneath each major tick. If true,
    each major tick will have a label containing the axis value.

    :data:`x_grid_label` is a :class:`~kivy.properties.BooleanProperty`,
    defaults to False.
    '''

    x_grid_labels = ObjectProperty('auto')
    '''List of custom label strings for the x ticks or 'auto' to create numerical values
    '''

    xlabel = StringProperty('')
    '''The label for the x-axis. If not empty it is displayed in the center of
    the axis.

    :data:`xlabel` is a :class:`~kivy.properties.StringProperty`,
    defaults to ''.
    '''

    ymin = NumericProperty(0.)
    '''The y-axis minimum value.

    If :data:`ylog` is True, ymin must be larger than zero.

    :data:`ymin` is a :class:`~kivy.properties.NumericProperty`, defaults to 0.
    '''

    ymax = NumericProperty(100.)
    '''The y-axis maximum value, larger than ymin.

    :data:`ymax` is a :class:`~kivy.properties.NumericProperty`, defaults to 0.
    '''

    ylog = BooleanProperty(False)
    '''Determines whether the y-axis should be displayed logarithmically (True)
    or linearly (False).

    :data:`ylog` is a :class:`~kivy.properties.BooleanProperty`, defaults
    to False.
    '''

    y_tick_offset = NumericProperty(0.)
    ''' Coordinate offset of the first major y tick '''

    y_ticks_major = BoundedNumericProperty(0, min=0)
    '''Distance between major tick marks. See :data:`x_ticks_major`.

    :data:`y_ticks_major` is a
    :class:`~kivy.properties.BoundedNumericProperty`, defaults to 0.
    '''

    y_ticks_minor = BoundedNumericProperty(0, min=0)
    '''The number of sub-intervals that divide ticks_major.
    See :data:`x_ticks_minor`.

    :data:`y_ticks_minor` is a
    :class:`~kivy.properties.BoundedNumericProperty`, defaults to 0.
    '''

    y_grid = BooleanProperty(False)
    '''Determines whether the y-axis has tick marks or a full grid. See
    :data:`x_grid`.

    :data:`y_grid` is a :class:`~kivy.properties.BooleanProperty`, defaults
    to False.
    '''

    y_grid_label = BooleanProperty(False)
    '''Whether labels should be displayed beneath each major tick. If true,
    each major tick will have a label containing the axis value.

    :data:`y_grid_label` is a :class:`~kivy.properties.BooleanProperty`,
    defaults to False.
    '''

    y_grid_labels = ObjectProperty('auto')
    '''List of custom label strings for the y ticks or 'auto' to create numerical values
    '''

    ylabel = StringProperty('')
    '''The label for the y-axis. If not empty it is displayed in the center of
    the axis.

    :data:`ylabel` is a :class:`~kivy.properties.StringProperty`,
    defaults to ''.
    '''

    padding = NumericProperty('5dp')
    '''Padding distances between the labels, axes titles and graph, as
    well between the widget and the objects near the boundaries.

    :data:`padding` is a :class:`~kivy.properties.NumericProperty`, defaults
    to 5dp.
    '''

    font_size = NumericProperty('15sp')
    '''Font size of the labels.

    :data:`font_size` is a :class:`~kivy.properties.NumericProperty`, defaults
    to 15sp.
    '''

    x_ticks_angle = NumericProperty(0)
    '''Rotate angle of the x-axis tick marks.

    :data:`x_ticks_angle` is a :class:`~kivy.properties.NumericProperty`,
    defaults to 0.
    '''

    precision = StringProperty('%g')
    '''Determines the numerical precision of the tick mark labels. This value
    governs how the numbers are converted into string representation. Accepted
    values are those listed in Python's manual in the
    "String Formatting Operations" section.

    :data:`precision` is a :class:`~kivy.properties.StringProperty`, defaults
    to '%g'.
    '''

    draw_border = BooleanProperty(True)
    '''Whether a border is drawn around the canvas of the graph where the
    plots are displayed.

    :data:`draw_border` is a :class:`~kivy.properties.BooleanProperty`,
    defaults to True.
    '''

    plots = ListProperty([])
    '''Holds a list of all the plots in the graph. To add and remove plots
    from the graph use :data:`add_plot` and :data:`add_plot`. Do not add
    directly edit this list.

    :data:`plots` is a :class:`~kivy.properties.ListProperty`,
    defaults to [].
    '''

    view_size = ObjectProperty((0, 0))
    '''The size of the graph viewing area - the area where the plots are
    displayed, excluding labels etc.
    '''

    view_pos = ObjectProperty((0, 0))
    '''The pos of the graph viewing area - the area where the plots are
    displayed, excluding labels etc. It is relative to the graph's pos.
    '''


class Plot(EventDispatcher):
    '''Plot class, see module documentation for more information.

    :Events:
        `on_clear_plot`
            Fired before a plot updates the display and lets the fbo know that
            it should clear the old drawings.

    ..versionadded:: 0.4
    '''

    __events__ = ('on_clear_plot', )

    # most recent values of the params used to draw the plot
    params = DictProperty({'xlog': False, 'xmin': 0, 'xmax': 100,
                           'ylog': False, 'ymin': 0, 'ymax': 100,
                           'size': (0, 0, 0, 0)})

    color = ListProperty([1, 0, 0, 1])
    '''Color of the plot.
    '''

    colors = ListProperty([])
    '''Colors for individual points/bars/etc.
    '''
    
    points = ListProperty([])
    '''List of (x, y) points to be displayed in the plot.

    The elements of points are 2-tuples, (x, y). The points are displayed
    based on the mode setting.

    :data:`points` is a :class:`~kivy.properties.ListProperty`, defaults to
    [].
    '''

    x_axis = NumericProperty(0)
    '''Index of the X axis to use, defaults to 0
    '''

    y_axis = NumericProperty(0)
    '''Index of the Y axis to use, defaults to 0
    '''

    def __init__(self, **kwargs):
        super(Plot, self).__init__(**kwargs)
        self.ask_draw = Clock.create_trigger(self.draw)
        self.bind(params=self.ask_draw, points=self.ask_draw, colors=self.ask_draw)
        self._drawings = self.create_drawings()

    def funcx(self):
        """Return a function that convert or not the X value according to plot
        prameters"""
        return log10 if self.params["xlog"] else lambda x: x

    def funcy(self):
        """Return a function that convert or not the Y value according to plot
        prameters"""
        return log10 if self.params["ylog"] else lambda y: y

    def x_px(self):
        """Return a function that convert the X value of the graph to the
        pixel coordinate on the plot, according to the plot settings and axis
        settings. It's relative to the graph pos.
        """
        funcx = self.funcx()
        params = self.params
        size = params["size"]
        xmin = funcx(params["xmin"])
        xmax = funcx(params["xmax"])
        xrange = float(xmax - xmin)
        ratiox = 0
        if xrange:
            ratiox = (size[2] - size[0]) / xrange

        return lambda x: (funcx(x) - xmin) * ratiox + size[0]

    def y_px(self):
        """Return a function that convert the Y value of the graph to the
        pixel coordinate on the plot, according to the plot settings and axis
        settings. The returned value is relative to the graph pos.
        """
        funcy = self.funcy()
        params = self.params
        size = params["size"]
        ymin = funcy(params["ymin"])
        ymax = funcy(params["ymax"])
        yrange = float(ymax - ymin)
        ratioy = 0
        if yrange:
            ratioy = (size[3] - size[1]) / yrange

        return lambda y: (funcy(y) - ymin) * ratioy + size[1]

    def unproject(self, x, y):
        """Return a function that unproject a pixel to a X/Y value on the plot
        (works only for linear, not log yet). `x`, `y`, is relative to the
        graph pos, so the graph's pos needs to be subtracted from x, y before
        passing it in.
        """
        params = self.params
        size = params["size"]

        xmin = params["xmin"]
        xmax = params["xmax"]
        xrange = float(xmax - xmin)
        ratiox = 0
        if xrange:
            ratiox = (size[2] - size[0]) / xrange

        ymin = params["ymin"]
        ymax = params["ymax"]
        yrange = float(ymax - ymin)
        ratioy = 0
        if yrange:
            ratioy = (size[3] - size[1]) / yrange

        x0 = (x - size[0]) / ratiox + xmin
        y0 = (y - size[1]) / ratioy + ymin
        return x0, y0

    def get_px_bounds(self):
        """Returns a dict containing the pixels bounds from the plot parameters.
         The returned values are relative to the graph pos.
        """
        params = self.params
        x_px = self.x_px()
        y_px = self.y_px()
        return {
            "xmin": x_px(params["xmin"]),
            "xmax": x_px(params["xmax"]),
            "ymin": y_px(params["ymin"]),
            "ymax": y_px(params["ymax"]),
        }

    def update(self, xlog, xmin, xmax, ylog, ymin, ymax, size):
        '''Called by graph whenever any of the parameters
        change. The plot should be recalculated then.
        log, min, max indicate the axis settings.
        size a 4-tuple describing the bounding box in which we can draw
        graphs, it's (x0, y0, x1, y1), which correspond with the bottom left
        and top right corner locations, respectively.
        '''
        self.params.update({
            'xlog': xlog, 'xmin': xmin, 'xmax': xmax, 'ylog': ylog,
            'ymin': ymin, 'ymax': ymax, 'size': size})

    def get_group(self):
        '''returns a string which is unique and is the group name given to all
        the instructions returned by _get_drawings. Graph uses this to remove
        these instructions when needed.
        '''
        return ''

    def get_drawings(self):
        '''returns a list of canvas instructions that will be added to the
        graph's canvas.
        '''
        if isinstance(self._drawings, (tuple, list)):
            return self._drawings
        return []

    def create_drawings(self):
        '''called once to create all the canvas instructions needed for the
        plot
        '''
        pass

    def draw(self, *largs):
        '''draw the plot according to the params. It dispatches on_clear_plot
        so derived classes should call super before updating.
        '''
        self.dispatch('on_clear_plot')

    def iterate_points(self):
        '''Iterate on all the points adjusted to the graph settings
        '''
        x_px = self.x_px()
        y_px = self.y_px()
        for x, y in self.points:
            yield x_px(x), y_px(y)

    def on_clear_plot(self, *largs):
        pass

    # compatibility layer
    _update = update
    _get_drawings = get_drawings
    _params = params


class MeshLinePlot(Plot):
    '''MeshLinePlot class which displays a set of points similar to a mesh.
    '''

    def _set_mode(self, value):
        if hasattr(self, '_mesh'):
            self._mesh.mode = value

    mode = AliasProperty(lambda self: self._mesh.mode, _set_mode)
    '''VBO Mode used for drawing the points. Can be one of: 'points',
    'line_strip', 'line_loop', 'lines', 'triangle_strip', 'triangle_fan'.
    See :class:`~kivy.graphics.Mesh` for more details.

    Defaults to 'line_strip'.
    '''

    def create_drawings(self):
        #Note (by Erik): cannot use self.colors here until we write a custom GL shader
        self._color = Color(*self.color)
        self._mesh = Mesh(mode='line_strip')
        self.bind(
            color=lambda instr, value: setattr(self._color, "rgba", value))
        return [self._color, self._mesh]

    def draw(self, *args):
        super(MeshLinePlot, self).draw(*args)
        self.plot_mesh()

    def plot_mesh(self):
        points = [p for p in self.iterate_points()]
        mesh, vert, _ = self.set_mesh_size(len(points))
        for k, (x, y) in enumerate(points):
            vert[k * 4] = x
            vert[k * 4 + 1] = y
        mesh.vertices = vert

    def set_mesh_size(self, size):
        mesh = self._mesh
        vert = mesh.vertices
        ind = mesh.indices
        diff = size - len(vert) // 4
        if diff < 0:
            del vert[4 * size:]
            del ind[size:]
        elif diff > 0:
            ind.extend(range(len(ind), len(ind) + diff))
            vert.extend([0] * (diff * 4))
        mesh.vertices = vert
        return mesh, vert, ind


class MeshStemPlot(MeshLinePlot):
    '''MeshStemPlot uses the MeshLinePlot class to draw a stem plot. The data
    provided is graphed from origin to the data point.
    '''

    def plot_mesh(self):
        points = [p for p in self.iterate_points()]
        mesh, vert, _ = self.set_mesh_size(len(points) * 2)
        y0 = self.y_px()(0)
        for k, (x, y) in enumerate(self.iterate_points()):
            vert[k * 8] = x
            vert[k * 8 + 1] = y0
            vert[k * 8 + 4] = x
            vert[k * 8 + 5] = y
        mesh.vertices = vert


class LinePlot(Plot):
    """LinePlot draws using a standard Line object.
    """

    line_width = NumericProperty(1)

    def create_drawings(self):
        from kivy.graphics import Line, RenderContext

        self._grc = RenderContext(
                use_parent_modelview=True,
                use_parent_projection=True)
        with self._grc:
            self._gcolor = Color(*self.color)
            self._gline = Line(
                points=[], cap='none',
                width=self.line_width, joint='round')

        return [self._grc]

    def draw(self, *args):
        super(LinePlot, self).draw(*args)
        # flatten the list
        points = []
        for x, y in self.iterate_points():
            points += [x, y]
        self._gline.points = points

    def on_line_width(self, *largs):
        if hasattr(self, "_gline"):
            self._gline.width = self.line_width


class SmoothLinePlot(Plot):
    '''Smooth Plot class, see module documentation for more information.
    This plot use a specific Fragment shader for a custom anti aliasing.
    '''

    SMOOTH_FS = '''
    $HEADER$

    void main(void) {
        float edgewidth = 0.015625 * 64.;
        float t = texture2D(texture0, tex_coord0).r;
        float e = smoothstep(0., edgewidth, t);
        gl_FragColor = frag_color * vec4(1, 1, 1, e);
    }
    '''

    # XXX This gradient data is a 64x1 RGB image, and
    # values goes from 0 -> 255 -> 0.
    GRADIENT_DATA = (
        b"\x00\x00\x00\x07\x07\x07\x0f\x0f\x0f\x17\x17\x17\x1f\x1f\x1f"
        b"'''///777???GGGOOOWWW___gggooowww\x7f\x7f\x7f\x87\x87\x87"
        b"\x8f\x8f\x8f\x97\x97\x97\x9f\x9f\x9f\xa7\xa7\xa7\xaf\xaf\xaf"
        b"\xb7\xb7\xb7\xbf\xbf\xbf\xc7\xc7\xc7\xcf\xcf\xcf\xd7\xd7\xd7"
        b"\xdf\xdf\xdf\xe7\xe7\xe7\xef\xef\xef\xf7\xf7\xf7\xff\xff\xff"
        b"\xf6\xf6\xf6\xee\xee\xee\xe6\xe6\xe6\xde\xde\xde\xd5\xd5\xd5"
        b"\xcd\xcd\xcd\xc5\xc5\xc5\xbd\xbd\xbd\xb4\xb4\xb4\xac\xac\xac"
        b"\xa4\xa4\xa4\x9c\x9c\x9c\x94\x94\x94\x8b\x8b\x8b\x83\x83\x83"
        b"{{{sssjjjbbbZZZRRRJJJAAA999111)))   \x18\x18\x18\x10\x10\x10"
        b"\x08\x08\x08\x00\x00\x00")

    def create_drawings(self):
        from kivy.graphics import Line, RenderContext

        # very first time, create a texture for the shader
        if not hasattr(SmoothLinePlot, '_texture'):
            tex = Texture.create(size=(1, 64), colorfmt='rgb')
            tex.add_reload_observer(SmoothLinePlot._smooth_reload_observer)
            SmoothLinePlot._texture = tex
            SmoothLinePlot._smooth_reload_observer(tex)

        self._grc = RenderContext(
            fs=SmoothLinePlot.SMOOTH_FS,
            use_parent_modelview=True,
            use_parent_projection=True)
        with self._grc:
            self._gcolor = Color(*self.color)
            self._gline = Line(
                points=[], cap='none', width=2.,
                texture=SmoothLinePlot._texture)

        return [self._grc]

    @staticmethod
    def _smooth_reload_observer(texture):
        texture.blit_buffer(SmoothLinePlot.GRADIENT_DATA, colorfmt="rgb")

    def draw(self, *args):
        super(SmoothLinePlot, self).draw(*args)
        # flatten the list
        points = []
        for x, y in self.iterate_points():
            points += [x, y]
        self._gline.points = points


class ContourPlot(Plot):
    """
    ContourPlot visualizes 3 dimensional data as an intensity map image.
    The user must first specify 'xrange' and 'yrange' (tuples of min,max) and
    then 'data', the intensity values.
    `data`, is a MxN matrix, where the first dimension of size M specifies the
    `y` values, and the second dimension of size N specifies the `x` values.
    Axis Y and X values are assumed to be linearly spaced values from
    xrange/yrange and the dimensions of 'data', `MxN`, respectively.
    The color values are automatically scaled to the min and max z range of the
    data set.
    """
    _image = ObjectProperty(None)
    data = ObjectProperty(None, force_dispatch=True)
    xrange = ListProperty([0, 100])
    yrange = ListProperty([0, 100])

    def __init__(self, **kwargs):
        super(ContourPlot, self).__init__(**kwargs)
        self.bind(data=self.ask_draw, xrange=self.ask_draw,
                  yrange=self.ask_draw)

    def create_drawings(self):
        self._image = Rectangle()
        self._color = Color([1, 1, 1, 1])
        self.bind(
            color=lambda instr, value: setattr(self._color, 'rgba', value))
        return [self._color, self._image]

    def draw(self, *args):
        super(ContourPlot, self).draw(*args)
        data = self.data
        xdim, ydim = data.shape

        # Find the minimum and maximum z values
        zmax = data.max()
        zmin = data.min()
        rgb_scale_factor = 1.0 / (zmax - zmin) * 255
        # Scale the z values into RGB data
        buf = np.array(data, dtype=float, copy=True)
        np.subtract(buf, zmin, out=buf)
        np.multiply(buf, rgb_scale_factor, out=buf)
        # Duplicate into 3 dimensions (RGB) and convert to byte array
        buf = np.asarray(buf, dtype=np.uint8)
        buf = np.expand_dims(buf, axis=2)
        buf = np.concatenate((buf, buf, buf), axis=2)
        buf = np.reshape(buf, (xdim, ydim, 3))

        charbuf = bytearray(np.reshape(buf, (buf.size)))
        self._texture = Texture.create(size=(xdim, ydim), colorfmt='rgb')
        self._texture.blit_buffer(charbuf, colorfmt='rgb', bufferfmt='ubyte')
        image = self._image
        image.texture = self._texture

        x_px = self.x_px()
        y_px = self.y_px()
        bl = x_px(self.xrange[0]), y_px(self.yrange[0])
        tr = x_px(self.xrange[1]), y_px(self.yrange[1])
        image.pos = bl
        w = tr[0] - bl[0]
        h = tr[1] - bl[1]
        image.size = (w, h)


class BarPlot(Plot):
    '''BarPlot class which displays a bar graph.
    '''

    bar_width = NumericProperty(1)
    bar_spacing = NumericProperty(1.)
    graph = ObjectProperty(allownone=True)

    def __init__(self, *ar, **kw):
        super(BarPlot, self).__init__(*ar, **kw)
        self.bind(bar_width=self.ask_draw)
        self.bind(points=self.update_bar_width)
        self.bind(graph=self.update_bar_width)

    def update_bar_width(self, *ar):
        if not self.graph:
            return
        if len(self.points) < 2:
            return
        if self.graph.xmax == self.graph.xmin:
            return

        point_width = (
            len(self.points) *
            float(abs(self.graph.xmax) - abs(self.graph.xmin)) /
            float(abs(max(self.points)[0]) - abs(min(self.points)[0])))

        if not self.points:
            self.bar_width = 1
        else:
            #TODO: could update this logic to make bar_spacing work better
            self.bar_width = (
                (self.graph.width - self.graph.padding) /
                point_width * self.bar_spacing)

        #print("DB: point width = ", point_width, " bar width = ", self.bar_width)

    def create_drawings(self):
        from kivy.graphics import RenderContext

        #test = RenderContext(
        #    use_parent_modelview=True,
        #    use_parent_projection=True)
        #print("TEST VERTEX SOURCE:")
        #print(test.shader.vs)
        #print("TEST FRAG SOURCE:")
        #print(test.shader.fs)

        if not self.colors:  # original all-one-color bars
            self._color = Color(*self.color)
            self._mesh = Mesh(mode='triangles')
            self.bind(
                color=lambda instr, value: setattr(self._color, 'rgba', value))
            return [self._color, self._mesh]

        else:  # per-bar colors -> need custom shader

            import os
            self._grc = RenderContext(
                use_parent_modelview=True,
                use_parent_projection=True)
            self._grc.shader.source = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   'perptcolorshaders.glsl')
            vertex_format = [
                (b'vPosition', 2, 'float'),  # <--- GLSL shader variable names.
                (b'vColor', 4, 'float')
            ]
            with self._grc:
                self._color = Color(*self.color)
                self._mesh = Mesh(fmt=vertex_format,
                                  mode='triangles')
                self.bind(
                    color=lambda instr, value: setattr(self._color, 'rgba', value))
            return [self._grc]

    def draw(self, *args):
        super(BarPlot, self).draw(*args)
        points = self.points

        # The mesh only supports (2^16) - 1 indices, so...
        if len(points) * 6 > 65535:
            Logger.error(
                "BarPlot: cannot support more than 10922 points. "
                "Ignoring extra points.")
            points = points[:10922]

        point_len = len(points)
        mesh = self._mesh
        vert = mesh.vertices
        ind = mesh.indices

        vertex_size = 4 if len(self.colors) == 0 else 6
        diff = len(points) * 6 - len(vert) // vertex_size
        if diff < 0:
            del vert[6 * vertex_size * point_len:]
            del ind[6 * point_len:]
        elif diff > 0:
            ind.extend(range(len(ind), len(ind) + diff))
            vert.extend([0] * (diff * vertex_size))

        bounds = self.get_px_bounds()
        x_px = self.x_px()
        y_px = self.y_px()
        ymin = y_px(self.graph.ymin)

        bar_width = self.bar_width
        if bar_width < 0:
            bar_width = x_px(bar_width) - bounds["xmin"]

        nColors = len(self.colors)
            
        for k in range(point_len):
            p = points[k]
            x1 = x_px(p[0])
            x2 = x1 + bar_width
            y1 = ymin
            y2 = y_px(p[1])
            idx = k * 6 * vertex_size

            if not self.colors:  # all bars one color - original method
                # first triangle
                vert[idx] = x1
                vert[idx + 1] = y2
                vert[idx + 4] = x1
                vert[idx + 5] = y1
                vert[idx + 8] = x2
                vert[idx + 9] = y1
                # second triangle
                vert[idx + 12] = x1
                vert[idx + 13] = y2
                vert[idx + 16] = x2
                vert[idx + 17] = y2
                vert[idx + 20] = x2
                vert[idx + 21] = y1
            else:
                # per-vertex coloring: each vertex is (x, y, r, g, b, a)
                color = self.colors[k % nColors]
                r, g, b, a = color

                # first triangle
                vert[idx] = x1
                vert[idx + 1] = y2
                vert[idx + 2] = r
                vert[idx + 3] = g
                vert[idx + 4] = b
                vert[idx + 5] = a

                vert[idx + 6] = x1
                vert[idx + 7] = y1
                vert[idx + 8] = r
                vert[idx + 9] = g
                vert[idx + 10] = b
                vert[idx + 11] = a

                vert[idx + 12] = x2
                vert[idx + 13] = y1
                vert[idx + 14] = r
                vert[idx + 15] = g
                vert[idx + 16] = b
                vert[idx + 17] = a

                # second triangle
                vert[idx + 18] = x1
                vert[idx + 19] = y2
                vert[idx + 20] = r
                vert[idx + 21] = g
                vert[idx + 22] = b
                vert[idx + 23] = a

                vert[idx + 24] = x2
                vert[idx + 25] = y2
                vert[idx + 26] = r
                vert[idx + 27] = g
                vert[idx + 28] = b
                vert[idx + 29] = a

                vert[idx + 30] = x2
                vert[idx + 31] = y1
                vert[idx + 32] = r
                vert[idx + 33] = g
                vert[idx + 34] = b
                vert[idx + 35] = a

        mesh.vertices = vert

    def _unbind_graph(self, graph):
        graph.unbind(width=self.update_bar_width,
                     xmin=self.update_bar_width,
                     ymin=self.update_bar_width)

    def bind_to_graph(self, graph):
        old_graph = self.graph

        if old_graph:
            # unbind from the old one
            self._unbind_graph(old_graph)

        # bind to the new one
        self.graph = graph
        graph.bind(width=self.update_bar_width,
                   xmin=self.update_bar_width,
                   ymin=self.update_bar_width)

    def unbind_from_graph(self):
        if self.graph:
            self._unbind_graph(self.graph)


class HBar(MeshLinePlot):
    '''HBar draw horizontal bar on all the Y points provided
    '''

    def plot_mesh(self, *args):
        points = self.points
        mesh, vert, ind = self.set_mesh_size(len(points) * 2)
        mesh.mode = "lines"

        bounds = self.get_px_bounds()
        px_xmin = bounds["xmin"]
        px_xmax = bounds["xmax"]
        y_px = self.y_px()
        for k, y in enumerate(points):
            y = y_px(y)
            vert[k * 8] = px_xmin
            vert[k * 8 + 1] = y
            vert[k * 8 + 4] = px_xmax
            vert[k * 8 + 5] = y
        mesh.vertices = vert


class VBar(MeshLinePlot):
    '''VBar draw vertical bar on all the X points provided
    '''

    def plot_mesh(self, *args):
        points = self.points
        mesh, vert, ind = self.set_mesh_size(len(points) * 2)
        mesh.mode = "lines"

        bounds = self.get_px_bounds()
        px_ymin = bounds["ymin"]
        px_ymax = bounds["ymax"]
        x_px = self.x_px()
        for k, x in enumerate(points):
            x = x_px(x)
            vert[k * 8] = x
            vert[k * 8 + 1] = px_ymin
            vert[k * 8 + 4] = x
            vert[k * 8 + 5] = px_ymax
        mesh.vertices = vert


class ScatterPlot(Plot):
    """
    ScatterPlot draws using a standard Point object.
    The pointsize can be controlled with :attr:`point_size`.

    >>> plot = ScatterPlot(color=[1, 0, 0, 1], point_size=5)
    """

    point_size = NumericProperty(1)
    """The point size of the scatter points. Defaults to 1.
    """

    def create_drawings(self):
        from kivy.graphics import Point, RenderContext

        self._points_context = RenderContext(
                use_parent_modelview=True,
                use_parent_projection=True)
        with self._points_context:
            self._gcolor = Color(*self.color)
            self._gpts = Point(points=[], pointsize=self.point_size)

        return [self._points_context]

    def draw(self, *args):
        super(ScatterPlot, self).draw(*args)
        # flatten the list
        self._gpts.points = list(chain(*self.iterate_points()))

    def on_point_size(self, *largs):
        if hasattr(self, "_gpts"):
            self._gpts.pointsize = self.point_size


class PointPlot(Plot):
    '''Displays a set of points.
    '''

    point_size = NumericProperty(1)
    '''
    Defaults to 1.
    '''

    _color = None

    _point = None

    def __init__(self, **kwargs):
        super(PointPlot, self).__init__(**kwargs)

        def update_size(*largs):
            if self._point:
                self._point.pointsize = self.point_size
        self.fbind('point_size', update_size)

        def update_color(*largs):
            if self._color:
                self._color.rgba = self.color
        self.fbind('color', update_color)

    def create_drawings(self):
        from kivy.graphics import RenderContext

        if not self.colors:  # original all-one-color points
            self._color = Color(*self.color)
            self._point = Point(pointsize=self.point_size)
            return [self._color, self._point]

        else:  # per-point colors -> need custom shader
            import os
            self._grc = RenderContext(
                use_parent_modelview=True,
                use_parent_projection=True)
            self._grc.shader.source = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   'perptcolorshaders.glsl')
            vertex_format = [
                (b'vPosition', 2, 'float'),  # <--- GLSL shader variable names.
                (b'vColor', 4, 'float')
            ]
            with self._grc:
                self._color = Color(*self.color)
                self._mesh = Mesh(fmt=vertex_format, mode='triangles')
                self.bind(color=lambda instr, value: setattr(self._color, 'rgba', value))
            return [self._grc]

    def draw(self, *args):
        #super(PointPlot, self).draw(*args)  # for some reason, leaving this uncommented causes
        # problems where the scatterplot just gets cleared and doesn't redraw properly.
        # I have no idea why this happens for this PointPlot but not others...

        if not self.colors:
            self._point.points = [v for p in self.iterate_points() for v in p]

        else:
            points = self.points
            sz = self.point_size / 2  # offset size of rectangle edges from point center

            point_len = len(points)
            mesh = self._mesh
            vert = mesh.vertices
            ind = mesh.indices

            vertex_size = 6  # 2 pos + 4 color
            diff = point_len * 6 - len(vert) // vertex_size  # 6 points from 2 triangls
            if diff < 0:
                del vert[6 * vertex_size * point_len:]
                del ind[6 * point_len:]
            elif diff > 0:
                ind.extend(range(len(ind), len(ind) + diff))
                vert.extend([0] * (diff * vertex_size))

            #bounds = self.get_px_bounds()
            #x_px = self.x_px()
            #y_px = self.y_px()
            #ymin = y_px(self.graph.ymin)
            #
            #bar_width = self.bar_width
            #if bar_width < 0:
            #    bar_width = x_px(bar_width) - bounds["xmin"]

            nColors = len(self.colors)

            for k, p in enumerate(self.iterate_points()):
                x1 = p[0] - sz
                x2 = p[0] + sz
                y1 = p[1] - sz
                y2 = p[1] + sz
                idx = k * 6 * vertex_size

                # per-vertex coloring: each vertex is (x, y, r, g, b, a)
                color = self.colors[k % nColors]
                r, g, b, a = color

                # first triangle
                vert[idx] = x1
                vert[idx + 1] = y2
                vert[idx + 2] = r
                vert[idx + 3] = g
                vert[idx + 4] = b
                vert[idx + 5] = a

                vert[idx + 6] = x1
                vert[idx + 7] = y1
                vert[idx + 8] = r
                vert[idx + 9] = g
                vert[idx + 10] = b
                vert[idx + 11] = a

                vert[idx + 12] = x2
                vert[idx + 13] = y1
                vert[idx + 14] = r
                vert[idx + 15] = g
                vert[idx + 16] = b
                vert[idx + 17] = a

                # second triangle
                vert[idx + 18] = x1
                vert[idx + 19] = y2
                vert[idx + 20] = r
                vert[idx + 21] = g
                vert[idx + 22] = b
                vert[idx + 23] = a

                vert[idx + 24] = x2
                vert[idx + 25] = y2
                vert[idx + 26] = r
                vert[idx + 27] = g
                vert[idx + 28] = b
                vert[idx + 29] = a

                vert[idx + 30] = x2
                vert[idx + 31] = y1
                vert[idx + 32] = r
                vert[idx + 33] = g
                vert[idx + 34] = b
                vert[idx + 35] = a

            mesh.vertices = vert



#class MeshBoxPlot(Plot):
#    '''MeshBoxPlot class which displays a set of boxes similar to a mesh.
#    '''
#
#    #REMOVE?
#    #def _set_mode(self, value):
#    #    if hasattr(self, '_mesh'):
#    #        self._mesh.mode = value
#    #
#    #mode = AliasProperty(lambda self: self._mesh.mode, _set_mode)
#    #'''VBO Mode used for drawing the points. Can be one of: 'points',
#    #'line_strip', 'line_loop', 'lines', 'triangle_strip', 'triangle_fan'.
#    #See :class:`~kivy.graphics.Mesh` for more details.
#    #
#    #Defaults to 'line_strip'.
#    #'''
#
#    def __init__(self, *ar, **kw):
#        super(BarPlot, self).__init__(*ar, **kw)
#        self.bind(bar_width=self.ask_draw)
#        self.bind(points=self.update_bar_width)
#        self.bind(graph=self.update_bar_width)
#
#
#    def create_drawings(self):
#        self._mesh = Mesh(mode='triangles')
#
#        # Leave this for now, even though we may not need/want it
#        self._color = Color(*self.color)
#        self.bind(
#            color=lambda instr, value: setattr(self._color, "rgba", value))
#
#        return [self._color, self._mesh]
#
#    def draw(self, *args):
#        super(MeshBoxPlot, self).draw(*args)
#        self.plot_mesh()
#
#    def plot_mesh(self):
#        points = [p for p in self.iterate_points()]
#        mesh, vert, _ = self.set_mesh_size(len(points))
#        for k, (x, y) in enumerate(points):
#            vert[k * 4] = x
#            vert[k * 4 + 1] = y
#        mesh.vertices = vert
#
#    def set_mesh_size(self, size):
#        mesh = self._mesh
#        vert = mesh.vertices
#        ind = mesh.indices
#        diff = size - len(vert) // 4
#        if diff < 0:
#            del vert[4 * size:]
#            del ind[size:]
#        elif diff > 0:
#            ind.extend(range(len(ind), len(ind) + diff))
#            vert.extend([0] * (diff * 4))
#        mesh.vertices = vert
#        return mesh, vert, ind


class MatrixBoxPlotGraph(Graph):
    '''TODO: docstring
    '''

    def __init__(self, data_matrix, box_labels=True, colormap=None, prec=0, **kwargs):
        self.data_matrix = data_matrix  # allow to be sparse also?
        # maybe allow to send a data layout so data values can be updated quickly?

        Window.bind(mouse_pos=self.on_mouse_pos)
        kwargs.update(dict(xmin=0, xmax=self.data_matrix.shape[1],
                           ymin=0, ymax=self.data_matrix.shape[0],
                           x_grid=True, y_grid=True,
                           #x_grid_labels=["One", "Two", "Three", "4", "5", "6", "7", "-8-", "nine", "ten"],
                           ))
        super(MatrixBoxPlotGraph, self).__init__(**kwargs)
        self._sidebar_layout = self._status_label = None

        # create a 64x1 texture, defaults to rgba / ubyte
        self.colormap = colormap
        self.tsize = 64  # texture / colorbar size should be a power of 2 for GL efficiency
        texture = Texture.create(size=(self.tsize, 2), colorfmt='rgb', bufferfmt='ubyte')
        texture.add_reload_observer(self.populate_texture)
        self.populate_texture(texture)

        with self._fbo:
            Color(1.0, 1.0, 1.0, 1.0)  # use white base so doesn't alter box colors
            self._mesh_boxes = Mesh(mode='triangles', texture=texture)

        if box_labels:
            self.labels = []
            self.label_colors = []
            self.label_rects = []
            for i in range(self.data_matrix.shape[0]):
                row_labels = []
                row_rects = []
                row_colors = []
                for j in range(self.data_matrix.shape[1]):
                    label = CoreLabel(text=_ph._eformat(self.data_matrix[i, j], prec), font_size=16)
                    label.refresh()
                    if self.colormap is not None:
                        color = Color(0, 0, 0) if (self.colormap.besttxtcolor(self.data_matrix[i, j]) == 'black') \
                            else Color(1, 1, 1)
                    else:  # assume black-to-white default colormap
                        color = Color(0, 0, 0) if (self.data_matrix[i, j] > 0.5) else Color(1, 1, 1)
                    rect = Rectangle(size=label.texture.size, pos=(0, 0), texture=label.texture)
                    row_labels.append(label)
                    row_rects.append(rect)
                    row_colors.append(color)
                    self._fbo.add(color)
                    self._fbo.add(rect)
                self.labels.append(row_labels)
                self.label_rects.append(row_rects)
                self.label_colors.append(row_colors)
                #self.canvas.add(Color(self.colour, 1-self.colour,0, 1))

    def populate_texture(self, texture):
        # create self.tsizex2 rgb colorbar table, and fill with values from 0 to 255
        if self.colormap is not None:
            bufline =  list(itertools.chain(
                *[self.colormap.interpolate_color_tuple(v, value_is_normalized=True)
                  for v in _np.linspace(0, 1, self.tsize)]))
        else:
            # simple black to white color scale
            bufline =  list(itertools.chain(*[(int(x * 255 / self.tsize),) * 3 for x in range(self.tsize)]))
        buf = bytes(bufline + bufline)
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

    #def _redraw_all(self, *args):
    #    super()._redraw_all(*args)
    #
    #    self._redraw_boxes(*args)
    #    #mesh = self._mesh_ticks
    #    #n_points = (len(xpoints_major) + len(xpoints_minor) +
    #    #            len(ypoints_major) + len(ypoints_minor))
    #    #mesh.vertices = [0] * (n_points * 8)
    #    #mesh.indices = [k for k in range(n_points * 2)]

    def _update_plots(self, size):

        nboxes = self.data_matrix.size

        # The mesh only supports (2^16) - 1 indices, so...
        if nboxes * 4 > 65535:
            #Logger.error(  "Ignoring extra boxes.")
            raise ValueError("MatrixBoxPlotGraph: cannot support more than 16383 boxes.")

        mesh = self._mesh_boxes
        vert = mesh.vertices
        ind = mesh.indices

        #DEBUG
        #if len(vert) < 16 * nboxes:
        #    vert.extend([0] * (16 * nboxes - len(vert)))
        #    ind.extend(range(len(ind), 4 * nboxes))

        scalex = (size[2] - size[0]) / (self.xmax - self.xmin)
        scaley = (size[3] - size[1]) / (self.ymax - self.ymin)
        pad = 2

        current_boxes = len(vert) // 16
        box_diff = nboxes - current_boxes  # number of vertices
        if box_diff < 0:
            del vert[16 * nboxes:]  # 4 entries per vertex * 4 vertices per box
            del ind[6 * nboxes:]  # 6 indices per box (2 triangles)
        elif box_diff > 0:
            for i in range(current_boxes, nboxes):  # assumes vertex order is tr, tl, bl, br
                ind.extend([4 * i, 4 * i + 1, 4 * i + 2,  # triangle 1
                            4 * i + 2, 4 * i + 3, 4 * i])  # triangle 2
            vert.extend([0] * (box_diff * 16))

        M, N = self.data_matrix.shape
        if self.colormap is not None:
            data_max = self.colormap.hmax
            data_min = self.colormap.hmin
        else:
            data_max = max(self.data_matrix.flat)
            data_min = min(self.data_matrix.flat)

        for i in range(M):
            for j in range(N):
                idx = 16 * (i * N + j)  # first vertex index
                x1, x2 = j * scalex + size[0] + pad, (j + 1) * scalex + size[0] - pad,
                y1, y2 = (M - 1 - i) * scaley + size[1] + pad, (M - i) * scaley + size[1] - pad
                norm_v = self.colormap.normalize(self.data_matrix[i, j]) \
                    if (self.colormap is not None) else self.data_matrix[i, j]  # in [hmin, hmax]
                u = ((norm_v - data_min) / (data_max - data_min))  # in [0, 1]
                delta = 1.0 / self.tsize

                vert[idx] = x2
                vert[idx + 1] = y2
                vert[idx + 2] = u + delta
                vert[idx + 3] = 1.0

                vert[idx + 4] = x1
                vert[idx + 5] = y2
                vert[idx + 6] = u
                vert[idx + 7] = 1.0

                vert[idx + 8] = x1
                vert[idx + 9] = y1
                vert[idx + 10] = u
                vert[idx + 11] = 0

                vert[idx + 12] = x2
                vert[idx + 13] = y1
                vert[idx + 14] = u + delta
                vert[idx + 15] = 0

                lbl = self.label_rects[i][j]
                lbl.pos = ((x1 + x2 - lbl.size[0]) / 2.0, (y1 + y2 - lbl.size[1]) / 2.0)
        mesh.vertices = vert  # needed?

    def set_info_containers(self, sidebar_layout, status_label):
        self._sidebar_layout = sidebar_layout
        self._status_label = status_label

    def on_touch_down(self, touch):

        #wpos = self.to_widget(*touch.pos, relative=True)  # DON'T do this - we're in a RelativeLayout
        # and so touch.pos is *not* in window coordinates - it's in RelativeLayout coordinates which area
        # the same as this widget's local (but not relative) coordinates.

        # Graph.collide_point needs coordinates in the same system as self.pos, which is the
        # relative coordinate system of a RelativeLayout when this Graph is within such a
        # layout (and it will be since our data area is a RelativeLayout).  In RelativeLayout.on_touch_down
        # the touch coordinates are mapped to the RelativeLayout's system, and so we're fine to
        # use touch.pos as-is
        if self.collide_point(*touch.pos):

            # self.view_pos, on the other hand, is in *relative* local coordinates,
            # so we need to transform "local" -> "relative local" coords:
            tpos = self.to_local(*touch.pos, relative=True)
            if (self.view_pos[0] <= tpos[0] <= self.view_pos[0] + self.view_size[0]
               and self.view_pos[1] <= tpos[1] <= self.view_pos[1] + self.view_size[1]):
                x, y = self.to_data(*tpos)
                if self._status_label and self._sidebar_layout:
                    self._sidebar_layout.clear_widgets()
                    j, i = int(_np.floor(x)), self.data_matrix.shape[0] - 1 - int(_np.floor(y))
                    if 0 <= i < self.data_matrix.shape[0] and 0 <= j < self.data_matrix.shape[1]:
                        anchor = AnchorLayout(padding=50, anchor_x='center', anchor_y='top')
                        #anchor.add_widget(Label(text="Clicked at %f, %f" % (x,y)))
                        anchor.add_widget(Label(text="Value %g" % (self.data_matrix[i, j])))
                        self._sidebar_layout.add_widget(anchor)
                    #self._status_label.text = "Value: %g" % self.data[k]
                return True
        return super().on_touch_down(touch)

    def on_mouse_pos(self, *args):
        if not self.get_root_window():
            return  # don't proceed if I'm not displayed <=> If have no parent

        pos = args[1]
        tpos = self.to_widget(*pos, relative=False)  # because pos is in window coords
        if self.collide_point(*tpos):
            tpos = self.to_local(*tpos, relative=True)
            if (self.view_pos[0] <= tpos[0] <= self.view_pos[0] + self.view_size[0]
               and self.view_pos[1] <= tpos[1] <= self.view_pos[1] + self.view_size[1]):
                x, y = self.to_data(*tpos)
                if self._status_label:
                    j, i = int(_np.floor(x)), self.data_matrix.shape[0] - 1 - int(_np.floor(y))
                    if 0 <= i < self.data_matrix.shape[0] and 0 <= j < self.data_matrix.shape[1]:
                        self._status_label.text = "%g" % self.data_matrix[i, j]
                    elif len(self._status_label.text) > 0:
                        self._status_label.text = ''
            elif self._status_label and len(self._status_label.text) > 0:
                self._status_label.text = ''

        #DB REMOVE
        #print("Update vertices: %d vertices, %d indices" % (len(mesh.vertices), len(mesh.indices)))
        #print(mesh.vertices)
        #print(mesh.indices)
        #print()

    #def _redraw_x(self, *args):
    #    font_size = self.font_size
    #    if self.xlabel:
    #        xlabel = self._xlabel
    #        if not xlabel:
    #            xlabel = Label()
    #            self.add_widget(xlabel)
    #            self._xlabel = xlabel
    #
    #        xlabel.font_size = font_size
    #        for k, v in self.label_options.items():
    #            setattr(xlabel, k, v)
    #
    #    else:
    #        xlabel = self._xlabel
    #        if xlabel:
    #            self.remove_widget(xlabel)
    #            self._xlabel = None
    #    grids = self._x_grid_label
    #    xpoints_major, xpoints_minor = self._get_ticks(self.x_ticks_major,
    #                                                   self.x_ticks_minor,
    #                                                   self.xlog, self.xmin,
    #                                                   self.xmax)
    #    self._ticks_majorx = xpoints_major
    #    self._ticks_minorx = xpoints_minor
    #
    #    if not self.x_grid_label:
    #        n_labels = 0
    #    else:
    #        n_labels = len(xpoints_major)
    #
    #    for k in range(n_labels, len(grids)):
    #        self.remove_widget(grids[k])
    #    del grids[n_labels:]
    #
    #    grid_len = len(grids)
    #    grids.extend([None] * (n_labels - len(grids)))
    #    for k in range(grid_len, n_labels):
    #        grids[k] = GraphRotatedLabel(
    #            font_size=font_size, angle=self.x_ticks_angle,
    #            **self.label_options)
    #        self.add_widget(grids[k])
    #    return xpoints_major, xpoints_minor

    #def _redraw_size(self, *args):
    #    super()._redraw_size(*args)
    #    size = self._update_labels() #??
    #    self._update_boxes(size)

    #REMOVE
    #    # size a 4-tuple describing the bounding box in which we can draw
    #    # graphs, it's (x0, y0, x1, y1), which correspond with the bottom left
    #    # and top right corner locations, respectively
    #    self._clear_buffer()
    #    size = self._update_labels()
    #    self.view_pos = self._plot_area.pos = (size[0], size[1])
    #    self.view_size = self._plot_area.size = (
    #        size[2] - size[0], size[3] - size[1])
    #
    #    if self.size[0] and self.size[1]:
    #        self._fbo.size = self.size
    #    else:
    #        self._fbo.size = 1, 1  # gl errors otherwise
    #    self._fbo_rect.texture = self._fbo.texture
    #    self._fbo_rect.size = self.size
    #    self._fbo_rect.pos = self.pos
    #    self._background_rect.size = self.size
    #    self._update_ticks(size)
    #    self._update_plots(size)

class NestedMatrixBoxPlotGraph(Graph):
    '''TODO: docstring
    '''

    def __init__(self, plt_data_list_of_lists, box_labels=False, colormap=None,
                 hover_label_fn=None, **kwargs):
        self.plt_data_list_of_lists = plt_data_list_of_lists  # better sparse format?
        Window.bind(mouse_pos=self.on_mouse_pos)

        data_size = 0
        for plt_data_list in plt_data_list_of_lists:
            data_size += sum([plt_data.size for plt_data in plt_data_list])
        self.data = _np.zeros(data_size, 'd')
        self.xs = _np.zeros(data_size, int)
        self.ys = _np.zeros(data_size, int)
        self.hover_labels = _np.empty(data_size, dtype=object) if hover_label_fn else None

        # Same as plotly code; maybe upgrade to allow sub-matrices of different shapes?
        elRows, elCols = plt_data_list_of_lists[0][0].shape  # nE,nr
        self.num_rows = len(plt_data_list_of_lists)  # number of rows of plaquettes
        self.num_cols = len(plt_data_list_of_lists[0])  # number of columns of plaquettes

        k = 0
        gap = 1.0
        nRows, nCols = self.num_rows, self.num_cols
        xmax = elCols * nCols + gap * (nCols - 1)
        ymax = elRows * nRows + gap * (nRows - 1)
        for i, plt_data_list in enumerate(plt_data_list_of_lists):  # for each row of data matrices
            for j, plt_data in enumerate(plt_data_list):  # for each column
                for ii in range(plt_data.shape[0]):
                    for jj in range(plt_data.shape[1]):
                        if _np.isnan(plt_data[ii, jj]):
                            continue
                        self.xs[k] = j * (elCols + gap) + jj
                        self.ys[k] = i * (elRows + gap) + ii
                        #self.ys[k] = ymax - (i * (elRows + gap) + ii)  # FLIPY (WRONG)
                        self.data[k] = plt_data[ii, jj]
                        if hover_label_fn:
                            lbl = hover_label_fn(self.data[k], i, j, ii, jj)
                            lbl = lbl.replace('<br>', '\n').replace('<sup>', '^').replace('</sup>', '')
                            self.hover_labels[k] = lbl
                        k += 1

        self.xs = self.xs[:k]
        self.ys = self.ys[:k]
        self.data = self.data[:k]
        if self.hover_labels is not None:
            self.hover_labels = self.hover_labels[:k]

        xspacing = elCols + gap
        yspacing = elRows + gap
        kwargs.update(dict(xmin=0, xmax=xmax,
                           ymin=0, ymax=ymax,
                           #x_grid=True, y_grid=True,
                           x_ticks_major=xspacing, y_ticks_major=yspacing,
                           x_tick_offset=0.5 * xspacing, y_tick_offset=0.5 * yspacing,
                           ))
        super(NestedMatrixBoxPlotGraph, self).__init__(**kwargs)
        self._sidebar_layout = self._status_label = None

        # create a 64x2 texture, defaults to rgba / ubyte
        self.colormap = colormap
        self.tsize = 64  # texture / colorbar size should be a power of 2 for GL efficiency
        texture = Texture.create(size=(self.tsize, 2), colorfmt='rgb', bufferfmt='ubyte')
        texture.add_reload_observer(self.populate_texture)
        self.populate_texture(texture)

        with self._fbo:
            Color(0, 0, 0, 1)  # box border color
            self._mesh_box_bg = Mesh(mode='triangles')
            Color(1, 1, 1, 1)  # base color = white
            self._mesh_boxes = Mesh(mode='triangles', texture=texture)

        #if box_labels:
        #    self.labels = []
        #    self.label_colors = []
        #    self.label_rects = []
        #    for i in range(self.data_matrix.shape[0]):
        #        row_labels = []
        #        row_rects = []
        #        row_colors = []
        #        for j in range(self.data_matrix.shape[1]):
        #            label = CoreLabel(text="%.2g" % self.data_matrix[i, j], font_size=20)
        #            label.refresh()
        #            color = Color(0, 0, 0) if (self.data_matrix[i, j] > 0.5) else Color(1, 1, 1)
        #            rect = Rectangle(size=label.texture.size, pos=(0, 0), texture=label.texture)
        #            row_labels.append(label)
        #            row_rects.append(rect)
        #            row_colors.append(color)
        #            self._fbo.add(color)
        #            self._fbo.add(rect)
        #        self.labels.append(row_labels)
        #        self.label_rects.append(row_rects)
        #        self.label_colors.append(row_colors)
        #        #self.canvas.add(Color(self.colour, 1-self.colour,0, 1))

    def populate_texture(self, texture):
        # create self.tsizex2 rgb colorbar table, and fill with color values
        if self.colormap is not None:
            bufline =  list(itertools.chain(
                *[self.colormap.interpolate_color_tuple(v, value_is_normalized=True)
                  for v in _np.linspace(0, 1, self.tsize)]))
        else:
            # simple black to white color scale
            bufline =  list(itertools.chain(*[(int(x * 255 / self.tsize),) * 3 for x in range(self.tsize)]))
        buf = bytes(bufline + bufline)
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

    #def _redraw_all(self, *args):
    #    super()._redraw_all(*args)
    #
    #    self._redraw_boxes(*args)
    #    #mesh = self._mesh_ticks
    #    #n_points = (len(xpoints_major) + len(xpoints_minor) +
    #    #            len(ypoints_major) + len(ypoints_minor))
    #    #mesh.vertices = [0] * (n_points * 8)
    #    #mesh.indices = [k for k in range(n_points * 2)]

    def _update_plots(self, size):

        nboxes = self.data.size

        # The mesh only supports (2^16) - 1 indices, so...
        if nboxes * 4 > 65535:
            #Logger.error(  "Ignoring extra boxes.")
            raise ValueError("NestedMatrixBoxPlotGraph: cannot support more than 16383 boxes.")

        mesh = self._mesh_boxes
        vert = mesh.vertices
        ind = mesh.indices

        vert_bg = self._mesh_box_bg.vertices

        #DEBUG
        #if len(vert) < 16 * nboxes:
        #    vert.extend([0] * (16 * nboxes - len(vert)))
        #    ind.extend(range(len(ind), 4 * nboxes))

        scalex = (size[2] - size[0]) / (self.xmax - self.xmin)
        scaley = (size[3] - size[1]) / (self.ymax - self.ymin)
        pad = 2

        current_boxes = len(vert) // 16
        box_diff = nboxes - current_boxes  # number of vertices
        if box_diff < 0:
            del vert[16 * nboxes:]  # 4 entries per vertex * 4 vertices per box
            del ind[6 * nboxes:]  # 6 indices per box (2 triangles)
            del vert_bg[16 * nboxes:]
        elif box_diff > 0:
            for i in range(current_boxes, nboxes):  # assumes vertex order is tr, tl, bl, br
                ind.extend([4 * i, 4 * i + 1, 4 * i + 2,  # triangle 1
                            4 * i + 2, 4 * i + 3, 4 * i])  # triangle 2

            vert.extend([0] * (box_diff * 16))
            vert_bg.extend([0] * (box_diff * 16))

        if self.colormap is not None:
            data_max = self.colormap.hmax
            data_min = self.colormap.hmin
        else:
            data_max = max(self.data)
            data_min = min(self.data)

        for i, (x, y, val) in enumerate(zip(self.xs, self.ys, self.data)):
            idx = 16 * i  # first vertex index
            x1, x2 = x * scalex + size[0] + pad, (x + 1) * scalex + size[0] - pad
            y1, y2 = y * scaley + size[1] + pad, (y + 1) * scaley + size[1] - pad
            #y1, y2 = y * scaley + size[1] + pad, (y - 1) * scaley + size[1] - pad  # FLIPY (WRONG)

            norm_v = self.colormap.normalize(val) if (self.colormap is not None) else val  # in [hmin, hmax]
            u = ((norm_v - data_min) / (data_max - data_min))  # in [0, 1]
            #delta = 1.0 / self.tsize

            vert[idx] = x2
            vert[idx + 1] = y2
            vert[idx + 2] = u #+ delta   -- if needed, need to double texture x-size
            vert[idx + 3] = 1.0

            vert[idx + 4] = x1
            vert[idx + 5] = y2
            vert[idx + 6] = u
            vert[idx + 7] = 1.0

            vert[idx + 8] = x1
            vert[idx + 9] = y1
            vert[idx + 10] = u
            vert[idx + 11] = 0

            vert[idx + 12] = x2
            vert[idx + 13] = y1
            vert[idx + 14] = u #+ delta  -- if needed, need to double texture x-size
            vert[idx + 15] = 0

            x1, x2 = x * scalex + size[0], (x + 1) * scalex + size[0]  # no padding
            y1, y2 = y * scaley + size[1], (y + 1) * scaley + size[1]  # no padding
            vert_bg[idx] = x2
            vert_bg[idx + 1] = y2
            vert_bg[idx + 4] = x1
            vert_bg[idx + 5] = y2
            vert_bg[idx + 8] = x1
            vert_bg[idx + 9] = y1
            vert_bg[idx + 12] = x2
            vert_bg[idx + 13] = y1

            #lbl = self.label_rects[i][j]
            #lbl.pos = ((x1 + x2 - lbl.size[0]) / 2.0, (y1 + y2 - lbl.size[1]) / 2.0)
        mesh.vertices = vert  # needed?
        self._mesh_box_bg.vertices = vert_bg
        self._mesh_box_bg.indices = ind  # (same indices as color-box mesh)

    def set_info_containers(self, sidebar_layout, status_label):
        self._sidebar_layout = sidebar_layout
        self._status_label = status_label

    def _find_index(self, x, y):
        xx = _np.floor(x)
        yy = _np.floor(y)
        kset = set(_np.where(self.xs == xx)[0]).intersection(_np.where(self.ys == yy)[0])
        assert(len(kset) <= 1)
        return next(iter(kset)) if kset else None
        
    def on_touch_down(self, touch):
        # Graph.collide_point needs coordinates in the same system as self.pos, which is the
        # relative coordinate system of a RelativeLayout when this Graph is within such a
        # layout (and it will be since our data area is a RelativeLayout).  In RelativeLayout.on_touch_down
        # the touch coordinates are mapped to the RelativeLayout's system, and so we're fine to
        # use touch.pos as-is:
        if self.collide_point(*touch.pos):

            # self.view_pos, on the other hand, is in *relative* local coordinates,
            # so we need to transform "local" -> "relative local" coords:
            tpos = self.to_local(*touch.pos, relative=True)
            if (self.view_pos[0] <= tpos[0] <= self.view_pos[0] + self.view_size[0]
               and self.view_pos[1] <= tpos[1] <= self.view_pos[1] + self.view_size[1]):
                x, y = self.to_data(*tpos)
                if self._status_label and self._sidebar_layout:
                    k = self._find_index(x, y)
                    self._sidebar_layout.clear_widgets()
                    if k is not None:
                        anchor = AnchorLayout(padding=50, anchor_x='center', anchor_y='top')
                        anchor.add_widget(WrappedLabel(text=self.hover_labels[k]))
                        self._sidebar_layout.add_widget(anchor)
                        self._status_label.text = "Value: %g" % self.data[k]
                        #self._status_label.text = "Touch event at %g, %g => index %d" % (x, y, k)
                    elif len(self._status_label.text) > 0:
                        self._status_label.text = ''
                #print("Touch event at ", touch.pos, ' -> ', (x, y))
                return True
        return super().on_touch_down(touch)

    def on_mouse_pos(self, *args):
        if not self.get_root_window():
            return  # don't proceed if I'm not displayed <=> If have no parent
        pos = args[1]
        tpos = self.to_widget(*pos, relative=False)  # because pos is in window coords
        if self.collide_point(*tpos):
            tpos = self.to_local(*tpos, relative=True)
            if (self.view_pos[0] <= tpos[0] <= self.view_pos[0] + self.view_size[0]
               and self.view_pos[1] <= tpos[1] <= self.view_pos[1] + self.view_size[1]):
                x, y = self.to_data(*tpos)
                if self._status_label:
                    k = self._find_index(x, y)
                    if k is not None:
                        #self._status_label.text = "Move event at %g, %g" % (x, y)
                        if self.hover_labels is not None:
                            self._status_label.text = self.hover_labels[k].replace('\n', ', ')
                        else:
                            self._status_label.text = "value: %g, " % self.data[k]
                    elif len(self._status_label.text) > 0:
                        self._status_label.text = ''
                #print("Move event at ", pos, ' -> ', (x, y))


if __name__ == '__main__':
    import itertools
    from math import sin, cos, pi
    from random import randrange
    from kivy.utils import get_color_from_hex as rgb
    from kivy.uix.boxlayout import BoxLayout
    from kivy.app import App

    class TestApp(App):

        def build(self):
            b = BoxLayout(orientation='vertical')
            # example of a custom theme
            colors = itertools.cycle([
                rgb('7dac9f'), rgb('dc7062'), rgb('66a8d4'), rgb('e5b060')])
            graph_theme = {
                'label_options': {
                    'color': rgb('444444'),  # color of tick labels and titles
                    'bold': True},
                'background_color': rgb('f8f8f2'),  # canvas background color
                'tick_color': rgb('808080'),  # ticks and grid
                'border_color': rgb('808080')}  # border drawn around each graph

            graph = Graph(
                xlabel='Cheese',
                ylabel='Apples',
                x_ticks_minor=5,
                x_ticks_major=25,
                y_ticks_major=1,
                y_grid_label=True,
                x_grid_label=True,
                padding=5,
                xlog=False,
                ylog=False,
                x_grid=True,
                y_grid=True,
                xmin=-50,
                xmax=50,
                ymin=-1,
                ymax=1,
                **graph_theme)

            plot = SmoothLinePlot(color=next(colors))
            plot.points = [(x / 10., sin(x / 50.)) for x in range(-500, 501)]
            # for efficiency, the x range matches xmin, xmax
            graph.add_plot(plot)

            plot = MeshLinePlot(color=next(colors))
            plot.points = [(x / 10., cos(x / 50.)) for x in range(-500, 501)]
            graph.add_plot(plot)
            self.plot = plot  # this is the moving graph, so keep a reference

            plot = MeshStemPlot(color=next(colors))
            graph.add_plot(plot)
            plot.points = [(x, x / 50.) for x in range(-50, 51)]

            plot = BarPlot(color=next(colors), bar_spacing=.72)
            graph.add_plot(plot)
            plot.bind_to_graph(graph)
            plot.points = [(x, .1 + randrange(10) / 10.) for x in range(-50, 1)]

            Clock.schedule_interval(self.update_points, 1 / 60.)

            graph2 = Graph(
                xlabel='Position (m)',
                ylabel='Time (s)',
                x_ticks_minor=0,
                x_ticks_major=1,
                y_ticks_major=10,
                y_grid_label=True,
                x_grid_label=True,
                padding=5,
                xlog=False,
                ylog=False,
                xmin=0,
                ymin=0,
                **graph_theme)
            b.add_widget(graph)

            if np is not None:
                (xbounds, ybounds, data) = self.make_contour_data()
                # This is required to fit the graph to the data extents
                graph2.xmin, graph2.xmax = xbounds
                graph2.ymin, graph2.ymax = ybounds

                plot = ContourPlot()
                plot.data = data
                plot.xrange = xbounds
                plot.yrange = ybounds
                plot.color = [1, 0.7, 0.2, 1]
                graph2.add_plot(plot)

                b.add_widget(graph2)
                self.contourplot = plot

                Clock.schedule_interval(self.update_contour, 1 / 60.)

            # Test the scatter plot
            plot = ScatterPlot(color=next(colors), point_size=5)
            graph.add_plot(plot)
            plot.points = [(x, .1 + randrange(10) / 10.) for x in range(-50, 1)]
            return b

        def make_contour_data(self, ts=0):
            omega = 2 * pi / 30
            k = (2 * pi) / 2.0

            ts = sin(ts * 2) + 1.5  # emperically determined 'pretty' values
            npoints = 100
            data = np.ones((npoints, npoints))

            position = [ii * 0.1 for ii in range(npoints)]
            time = [(ii % 100) * 0.6 for ii in range(npoints)]

            for ii, t in enumerate(time):
                for jj, x in enumerate(position):
                    data[ii, jj] = sin(
                        k * x + omega * t) + sin(-k * x + omega * t) / ts
            return (0, max(position)), (0, max(time)), data

        def update_points(self, *args):
            self.plot.points = [
                (x / 10., cos(Clock.get_time() + x / 50.))
                for x in range(-500, 501)]

        def update_contour(self, *args):
            _, _, self.contourplot.data[:] = self.make_contour_data(
                Clock.get_time())
            # this does not trigger an update, because we replace the
            # values of the arry and do not change the object.
            # However, we cannot do "...data = make_contour_data()" as
            # kivy will try to check for the identity of the new and
            # old values.  In numpy, 'nd1 == nd2' leads to an error
            # (you have to use np.all).  Ideally, property should be patched
            # for this.
            self.contourplot.ask_draw()

    TestApp().run()
