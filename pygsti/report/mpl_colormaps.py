"""
Plotly-to-Matplotlib conversion functions.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations

import gc as _gc
from typing import Any, Optional, TYPE_CHECKING

import numpy as _np

from pygsti.report.plothelpers import _eformat
from pygsti.circuits.circuit import Circuit as _Circuit

try:
    import matplotlib as _matplotlib
    import matplotlib.pyplot as _plt
except ImportError:
    raise ValueError(("While not a core requirement of pyGSTi, Matplotlib is "
                      "required to generate PDF plots.  It looks like you "
                      "don't have it installed on your system (it failed to "
                      "import)."))

if TYPE_CHECKING:
    # These imports are only used for type annotations, and are gated behind
    # TYPE_CHECKING (together with `from __future__ import annotations`, which
    # makes all annotations lazily-evaluated strings) so that they impose no
    # runtime import requirements beyond what this module already needs.
    from matplotlib.axes import Axes as _MplAxes
    from matplotlib.collections import QuadMesh as _MplQuadMesh
    from matplotlib.colors import Colormap as _MplColormap
    from matplotlib.colors import Normalize as _MplNormalize
    from matplotlib.figure import Figure as _MplFigure

    from pygsti.report.colormaps import Colormap as _PygstiColormap
    from pygsti.report.figure import ReportFigure


class MplLinLogNorm(_matplotlib.colors.Normalize):
    """
    Matplotlib version of lin-log colormap normalization

    Parameters
    ----------
    linlog_colormap : LinlogColormap
        pyGSTi linear-logarithmic color map to base this colormap off of.

    clip : bool, optional
        Whether clipping should be performed. See :class:`matplotlib.colors.Normalize`.
    """

    def __init__(self, linlog_colormap, clip=False):
        cm = linlog_colormap
        super(MplLinLogNorm, self).__init__(vmin=cm.vmin, vmax=cm.vmax, clip=clip)
        self.trans = cm.trans
        self.cm = cm

    def inverse(self, value):
        """
        Inverse of __call__ as per matplotlib spec.

        Parameters
        ----------
        value : float or numpy.ndarray
            Color-value to invert back.

        Returns
        -------
        float or numpy.ndarray
        """
        norm_trans = super(MplLinLogNorm, self).__call__(self.trans)
        deltav = self.vmax - self.vmin
        return_value = _np.where(_np.greater(0.5, value),
                                 2 * value * (self.trans - self.vmin) + self.vmin,
                                 deltav * _np.power(norm_trans, 2 * (1 - value)))
        if return_value.shape == ():
            return return_value.item()
        else:
            return return_value.view(_np.ma.MaskedArray)

    def __call__(self, value, clip=None):
        return self.cm.normalize(value)


def mpl_make_linear_norm(vmin, vmax, clip=False):
    """
    Create a linear matplotlib normalization

    Parameters
    ----------
    vmin : float
        Minimum mapped color value.

    vmax : float
        Maximum mapped color value.

    clip : bool, optional
        Whether clipping should be performed. See :class:`matplotlib.colors.Normalize`.

    Returns
    -------
    matplotlib.colors.Normalize
    """
    return _matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=clip)


def mpl_make_linear_cmap(rgb_colors, name=None):
    """
    Make a color map that simply linearly interpolates between a set of colors in RGB space.

    Parameters
    ----------
    rgb_colors : list
        Each element is a `(value, (r, g, b))` tuple specifying a value and an
        RGB color.  Both `value` and `r`, `g`, and `b` should be floating point
        numbers between 0 and 1.

    name : string, optional
        A name for the colormap. If not provided, a name will be constructed
        from an random integer.

    Returns
    -------
    cmap
    """
    if name is None:
        name = "pygsti-cmap-" + str(_np.random.randint(0, 100000000))

    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}
    for val, rgb_tup in rgb_colors:
        for k, v in zip(('red', 'green', 'blue'), rgb_tup):
            cdict[k].append((val, v, v))
        cdict['alpha'].append((val, 1.0, 1.0))  # alpha channel always 1.0

    return _matplotlib.colors.LinearSegmentedColormap(name, cdict)


def _get_plotly_attr(obj: Any, x: str, default: Any) -> Any:
    """ Needed b/c in plotly v3 layout no longer is a dict """
    try:
        ret = obj[x]
        return ret if (ret is not None) else default
    except KeyError:
        return default
    raise ValueError("Non-KeyError raised when trying to access a plotly hierarchy object.")


def _fallback_norm_and_cmap(traceDict: Any) -> tuple[_MplNormalize, _MplColormap]:
    """
    Build a matplotlib Normalize + Colormap directly from a plotly heatmap
    trace's own `colorscale`/`zmin`/`zmax` attributes.

    This is used as a fallback when a `ReportFigure` doesn't carry a pyGSTi
    `Colormap` object in its `.colormap` attribute (e.g. a "raw" plotly
    figure that wasn't built via pyGSTi's report-figure machinery).

    Parameters
    ----------
    traceDict : dict-like
        A plotly heatmap trace.

    Returns
    -------
    norm : matplotlib.colors.Normalize
    cmap : matplotlib.colors.Colormap
    """
    colorscale = _get_plotly_attr(traceDict, 'colorscale', None) \
        or [[0, 'white'], [1, 'black']]
    zmin = _get_plotly_attr(traceDict, 'zmin', 0.0)
    zmax = _get_plotly_attr(traceDict, 'zmax', 1.0)
    norm = _matplotlib.colors.Normalize(vmin=zmin, vmax=zmax)
    stops = [(val, _matplotlib.colors.to_rgb(mpl_color(color))) for val, color in colorscale]
    cmap = _matplotlib.colors.LinearSegmentedColormap.from_list("pygsti-fallback-cmap", stops)
    return norm, cmap


def _draw_heatmap_panel(axes: _MplAxes, traceDict: Any, plt_data: Optional[_np.ndarray],
                        colormap: Optional[_PygstiColormap], xtickvals: Optional[Any],
                        ytickvals: Optional[Any], xlabels: Optional[Any], ylabels: Optional[Any],
                        shapes: list, box_labels_font_size: int, prec: Any, fontsize: int,
                        title: Optional[str] = None, mpl_fig: Optional[_MplFigure] = None,
                        resize_fig: bool = False) -> _MplQuadMesh:
    """
    Draw a single pyGSTi/plotly heatmap trace onto a matplotlib `Axes`.

    This factors out the heatmap-drawing logic shared between the
    single-panel (`plotly_to_matplotlib`) and multi-panel
    (`_composite_heatmap_to_matplotlib`) conversion paths.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The axes to draw onto.

    traceDict : dict-like
        The plotly heatmap trace.

    plt_data : numpy.ndarray or None
        The (un-normalized) data underlying the heatmap.  If None, falls
        back to the trace's own (normalized) `z` data.

    colormap : Colormap or None
        A pyGSTi `Colormap` object.  If None, a matplotlib norm/cmap pair
        is derived directly from the trace's `colorscale`/`zmin`/`zmax`.

    xtickvals, ytickvals : list or None
        Tick locations (in original, un-shifted data coordinates).

    xlabels, ylabels : list or None
        Tick labels.

    shapes : list
        Plotly shape dicts (gridlines) belonging to this panel.

    box_labels_font_size : int
        Font size for box value labels; if <= 0 no labels are drawn.

    prec : int or str
        Precision for box value labels.

    fontsize : int
        Base font size for the colorbar ticks.

    title : str, optional
        A title to set on this panel's axes.

    mpl_fig : matplotlib.figure.Figure, optional
        The parent figure; only needed when `resize_fig` is True.

    resize_fig : bool, optional
        Whether to resize `mpl_fig` based on this panel's data shape (only
        appropriate for the single-panel case).

    Returns
    -------
    matplotlib.collections.QuadMesh
        The created heatmap object (useful for building a colorbar).
    """
    if plt_data is None:
        plt_data = _np.asarray(_get_plotly_attr(traceDict, 'z', None))

    show_colorscale = _get_plotly_attr(traceDict, 'showscale', True)

    if resize_fig and (mpl_fig is not None):
        mpl_size = (plt_data.shape[1] * 0.5, plt_data.shape[0] * 0.5)
        mpl_fig.set_size_inches(*mpl_size)

    if colormap is not None:
        norm, cmap = colormap.create_matplotlib_norm_and_cmap()
    else:
        norm, cmap = _fallback_norm_and_cmap(traceDict)

    masked_data = _np.ma.array(plt_data, mask=_np.isnan(plt_data))
    heatmap = axes.pcolormesh(masked_data, cmap=cmap, norm=norm)

    axes.set_xlim(0, plt_data.shape[1])
    axes.set_ylim(0, plt_data.shape[0])

    xtics = ytics = None
    if xtickvals is not None:
        xtics = _np.array(xtickvals) + 0.5
        axes.set_xticks(xtics, minor=False)

    if ytickvals is not None:
        ytics = _np.array(ytickvals) + 0.5
        axes.set_yticks(ytics, minor=False)

    grid = bool(len(shapes) > 1)
    if grid and (xtics is not None) and (ytics is not None):
        def _get_minor_tics(t):
            return [(t[i] + t[i + 1]) / 2.0 for i in range(len(t) - 1)]
        axes.set_xticks(_get_minor_tics(xtics), minor=True)
        axes.set_yticks(_get_minor_tics(ytics), minor=True)
        axes.grid(which='minor', axis='both', linestyle='-', linewidth=2)

    off = False  # Matplotlib used to allow 'off', but now False should be used
    if xlabels is None and ylabels is None:
        axes.tick_params(labelcolor='w', top=off, bottom=off, left=off, right=off)  # white tics
    else:
        axes.tick_params(top=off, bottom=off, left=off, right=off)

    if xlabels is not None:
        axes.set_xticklabels(mpl_process_lbls(xlabels), rotation=0, fontsize=(fontsize - 2))

    if ylabels is not None:
        axes.set_yticklabels(mpl_process_lbls(ylabels), fontsize=(fontsize - 2))

    if box_labels_font_size > 0:
        # Write values on colored squares
        for y in range(plt_data.shape[0]):
            for x in range(plt_data.shape[1]):
                if _np.isnan(plt_data[y, x]): continue
                assert(_np.isfinite(plt_data[y, x])), "%s is not finite!" % str(plt_data[y, x])
                axes.text(x + 0.5, y + 0.5, mpl_process_lbl(_eformat(plt_data[y, x], prec), math=True),
                          horizontalalignment='center',
                          verticalalignment='center',
                          color=mpl_besttxtcolor(plt_data[y, x], cmap, norm),
                          fontsize=box_labels_font_size)

    if show_colorscale:
        cbar = _plt.colorbar(heatmap, ax=axes)
        cbar.ax.tick_params(labelsize=(fontsize - 2))

    if title is not None:
        axes.set_title(mpl_process_lbl(title), fontsize=fontsize)

    return heatmap


def _composite_heatmap_to_matplotlib(pygsti_fig: ReportFigure, save_to: Optional[str] = None,
                                     fontsize: int = 12, prec: Any = 'compacthp',
                                     box_labels_font_size: int = 6) -> Optional[_MplFigure]:
    """
    Convert a composite (multi-panel) pyGSTi heatmap `ReportFigure` -- as
    produced by `_matrices_color_boxplot`/`_opmatrices_color_boxplot` -- into
    a matplotlib figure with one `Axes` per panel.

    Parameters
    ----------
    pygsti_fig : ReportFigure
        A pyGSTi figure whose metadata contains `num_matrix_rows` and
        `num_matrix_cols`.

    save_to : str
        Output filename.  Extension determines type.  If None, then the
        matplotlib figure is returned instead of saved.

    fontsize : int, optional
        Base fontsize to use for converted figure.

    prec : int or {"compact","compacth"}
        Digits of precision to include in labels.

    box_labels_font_size : int, optional
        The size for labels on the boxes. If 0 then no labels are
        put on the boxes

    Returns
    -------
    matplotlib.Figure
        Matplotlib figure, unless save_to is not None, in which case
        the figure is closed and None is returned.
    """
    numMPLFigs = len(_plt.get_fignums())
    fig = pygsti_fig.plotlyfig
    layout_dict = fig.to_dict()['layout']

    num_rows = pygsti_fig.metadata['num_matrix_rows']
    num_cols = pygsti_fig.metadata['num_matrix_cols']
    plt_data_list = pygsti_fig.metadata.get('plt_data', None)
    subtitles = pygsti_fig.metadata.get('subtitles', None)
    colormap = pygsti_fig.colormap

    mpl_fig, axes_grid = _plt.subplots(num_rows, num_cols, squeeze=False)

    h, w = layout_dict.get('height'), layout_dict.get('width')
    if (w is not None) and (h is not None):
        mpl_size = w / 100.0, h / 100.0  # same heuristic as the single-panel path
        mpl_fig.set_size_inches(*mpl_size)
        pygsti_fig.metadata['mpl_fig_size'] = mpl_size

    shapes_all = layout_dict.get('shapes', []) or []

    heatmap_traces = [tr for tr in fig['data']
                      if _get_plotly_attr(tr, 'type', 'unknown') == 'heatmap']

    for idx, traceDict in enumerate(heatmap_traces):
        row_i, col_j = divmod(idx, num_cols)
        axes = axes_grid[row_i][col_j]

        xaxis_key = _get_plotly_attr(traceDict, 'xaxis', 'x')
        yaxis_key = _get_plotly_attr(traceDict, 'yaxis', 'y')
        xaxis_layout = layout_dict.get('xaxis' + xaxis_key[1:], {}) or {}
        yaxis_layout = layout_dict.get('yaxis' + yaxis_key[1:], {}) or {}

        xtickvals = xaxis_layout.get('tickvals')
        ytickvals = yaxis_layout.get('tickvals')
        xlabels = xaxis_layout.get('ticktext')
        ylabels = yaxis_layout.get('ticktext')

        #Shapes (gridlines) belonging to this panel are tagged with this
        #panel's xref/yref by the composite-figure builders.
        panel_shapes = [s for s in shapes_all
                        if (s.get('xref') == xaxis_key) and (s.get('yref') == yaxis_key)]

        plt_data = plt_data_list[idx] if (plt_data_list is not None) else None
        title = subtitles[idx] if (subtitles is not None) and (idx < len(subtitles)) else None

        _draw_heatmap_panel(axes, traceDict, plt_data, colormap, xtickvals, ytickvals,
                            xlabels, ylabels, panel_shapes, box_labels_font_size, prec,
                            fontsize, title=title)

    if save_to:
        _gc.collect()
        _plt.savefig(save_to, bbox_inches='tight')
        _plt.close(mpl_fig)
        del mpl_fig
        _gc.collect()
        if len(_plt.get_fignums()) != numMPLFigs:
            raise ValueError("WARNING: MORE FIGURES OPEN NOW (%d) THAN WHEN WE STARTED %d)!!" %
                             (len(_plt.get_fignums()), numMPLFigs))
        return None  # figure is closed!
    else:
        return mpl_fig


def mpl_besttxtcolor(x, cmap, norm):
    """
    Determining function for whether text should be white or black

    Parameters
    ----------
    x : float
        Value of the cell in question

    cmap : matplotlib colormap
        Colormap assigning colors to the cells

    norm : matplotlib normalizer
        Function to map cell values to the interval [0, 1] for use by a
        colormap

    Returns
    -------
    {"white","black"}
    """
    cell_color = cmap(norm(x))
    R, G, B = cell_color[:3]
    # Perceived brightness calculation from http://alienryderflex.com/hsp.html
    P = _np.sqrt(0.299 * R**2 + 0.587 * G**2 + 0.114 * B**2)
    return "black" if 0.5 <= P else "white"


def mpl_process_lbl(lbl, math=False):
    """
    Process a (plotly-compatible) text label `lbl` to matplotlb text.

    Parameters
    ----------
    lbl : str
        A text label to process.

    math : bool, optional
        Whether math-formatting (latex) should be used.

    Returns
    -------
    str
    """
    if not isinstance(lbl, str):
        lbl = str(lbl)  # just force as a string
    math = math or ('<sup>' in lbl) or ('<sub>' in lbl) or \
        ('_' in lbl) or ('|' in lbl) or (len(lbl) == 1)
    try:
        float(lbl)
        math = True
    except: pass

    l = lbl
    l = l.replace("<i>", "").replace("</i>", "")
    l = l.replace("<sup>", "^{").replace("</sup>", "}")
    l = l.replace("<sub>", "_{").replace("</sub>", "}")
    l = l.replace("<br>", "\n")

    if math:
        l = l.replace("alpha", "\\alpha")
        l = l.replace("beta", "\\beta")
        l = l.replace("sigma", "\\sigma")

    if math or (len(l) == 1): l = "$" + l + "$"
    return l


def mpl_process_lbls(lbl_list):
    """
    Process a list of plotly labels into matplotlib ones

    Parameters
    ----------
    lbl_list : list
        A list of string-valued labels to process.

    Returns
    -------
    list
        the processed labels (strings).
    """
    return [mpl_process_lbl(lbl) for lbl in lbl_list]


def mpl_color(plotly_color):
    """
    Convert a plotly color name to a matplotlib compatible one.

    Parameters
    ----------
    plotly_color : str
        A plotly color value, e.g. `"#FF0023"` or `"rgb(0,255,128)"`.

    Returns
    -------
    str
    """
    plotly_color = plotly_color.strip()  # remove any whitespace
    if plotly_color.startswith('#'):
        return plotly_color  # matplotlib understands "#FF0013"
    elif plotly_color.startswith('rgb(') and plotly_color.endswith(')'):
        tupstr = plotly_color[len('rgb('):-1]
        tup = [float(x) / 256.0 for x in tupstr.split(',')]
        return tuple(tup)
    elif plotly_color.startswith('rgba(') and plotly_color.endswith(')'):
        tupstr = plotly_color[len('rgba('):-1]
        rgba = tupstr.split(',')
        tup = [float(x) / 256.0 for x in rgba[0:3]] + [float(rgba[3])]
        return tuple(tup)
    else:
        return plotly_color  # hope this is a color name matplotlib understands


def plotly_to_matplotlib(pygsti_fig: ReportFigure, save_to: Optional[str] = None, fontsize: int = 12,
                         prec: Any = 'compacthp', box_labels_font_size: int = 6) -> Optional[_MplFigure]:
    """
    Convert a pygsti (plotly) figure to a matplotlib figure.

    Parameters
    ----------
    pygsti_fig : ReportFigure
        A pyGSTi figure.

    save_to : str
        Output filename.  Extension determines type.  If None, then the
        matplotlib figure is returned instead of saved.

    fontsize : int, optional
        Base fontsize to use for converted figure.

    prec : int or {"compact","compacth"}
        Digits of precision to include in labels.

    box_labels_font_size : int, optional
        The size for labels on the boxes. If 0 then no labels are
        put on the boxes

    Returns
    -------
    matplotlib.Figure
        Matplotlib figure, unless save_to is not None, in which case
        the figure is closed and None is returned.
    """
    numMPLFigs = len(_plt.get_fignums())
    fig = pygsti_fig.plotlyfig
    data_trace_list = fig['data']

    if 'special' in pygsti_fig.metadata:
        if pygsti_fig.metadata['special'] == "keyplot":
            return special_keyplot(pygsti_fig, save_to, fontsize)
        else: raise ValueError("Invalid `special` label: %s" % pygsti_fig.metadata['special'])

    if 'num_matrix_rows' in pygsti_fig.metadata:
        # A composite/multi-panel heatmap figure (e.g. from
        # _matrices_color_boxplot / _opmatrices_color_boxplot) needs its own
        # multi-Axes rendering path -- a single Axes can't represent more
        # than one heatmap panel correctly.
        return _composite_heatmap_to_matplotlib(pygsti_fig, save_to, fontsize,
                                                prec, box_labels_font_size)

    #if axes is None:
    mpl_fig, axes = _plt.subplots()  # create a new figure if no axes are given

    layout = fig['layout']
    h, w = layout['height'], layout['width']
    # todo: get margins and subtract from h,w

    if mpl_fig is not None and w is not None and h is not None:
        mpl_size = w / 100.0, h / 100.0  # heusistic
        mpl_fig.set_size_inches(*mpl_size)  # was 12,8 for "super" color plot
        pygsti_fig.metadata['mpl_fig_size'] = mpl_size  # record for later use by rendering commands

    def get(obj, x, default):
        """ Needed b/c in plotly v3 layout no longer is a dict """
        try:
            ret = obj[x]
            return ret if (ret is not None) else default
        except KeyError:
            return default
        raise ValueError("Non-KeyError raised when trying to access a plotly hierarchy object.")

    xaxis, yaxis = layout['xaxis'], layout['yaxis']
    #annotations = get(layout,'annotations',[])
    title = get(layout, 'title', None)
    shapes = get(layout, 'shapes', [])  # assume only shapes are grid lines
    bargap = get(layout, 'bargap', 0)

    xlabel = get(xaxis, 'title', None)
    ylabel = get(yaxis, 'title', None)
    xlabels = get(xaxis, 'ticktext', None)
    ylabels = get(yaxis, 'ticktext', None)
    xtickvals = get(xaxis, 'tickvals', None)
    ytickvals = get(yaxis, 'tickvals', None)
    xaxistype = get(xaxis, 'type', None)
    yaxistype = get(yaxis, 'type', None)
    xaxisside = get(xaxis, 'side', 'bottom')
    yaxisside = get(yaxis, 'side', 'left')
    xtickangle = get(xaxis, 'tickangle', 0)
    xlim = get(xaxis, 'range', None)
    ylim = get(yaxis, 'range', None)

    if xaxisside == "top":
        axes.xaxis.set_label_position('top')
        axes.xaxis.tick_top()
        #axes.yaxis.set_ticks_position('both')

    if yaxisside == "right":
        axes.yaxis.set_label_position('right')
        axes.yaxis.tick_right()
        #axes.yaxis.set_ticks_position('both')

    if title is not None:
        # Sometimes Title object still is nested
        title_text = title if isinstance(title, str) else get(title, 'text', '')
        if xaxisside == "top":
            axes.set_title(mpl_process_lbl(title_text), fontsize=fontsize, y=4)  # push title up higher
        axes.set_title(mpl_process_lbl(title_text), fontsize=fontsize)

    if xlabel is not None:
        xlabel_text = xlabel if isinstance(xlabel, str) else get(xlabel, 'text', '')
        axes.set_xlabel(mpl_process_lbl(xlabel_text), fontsize=fontsize)

    if ylabel is not None:
        ylabel_text = ylabel if isinstance(ylabel, str) else get(ylabel, 'text', '')
        axes.set_ylabel(mpl_process_lbl(ylabel_text), fontsize=fontsize)

    if xtickvals is not None:
        axes.set_xticks(xtickvals, minor=False)

    if ytickvals is not None:
        axes.set_yticks(ytickvals, minor=False)

    if xlabels is not None:
        axes.set_xticklabels(mpl_process_lbls(xlabels), rotation=0, fontsize=(fontsize - 2))

    if ylabels is not None:
        axes.set_yticklabels(mpl_process_lbls(ylabels), fontsize=(fontsize - 2))

    if xtickangle != 0:
        _plt.xticks(rotation=-xtickangle)  # minus b/c ploty & matplotlib have different sign conventions

    if xaxistype == 'log':
        axes.set_xscale("log")
    if yaxistype == 'log':
        axes.set_yscale("log")

    if xlim is not None:
        if xaxistype == 'log':  # plotly's limits are already log10'd in this case
            xlim = 10.0**xlim[0], 10.0**xlim[1]  # but matplotlib's aren't
        axes.set_xlim(xlim)

    if ylim is not None:
        if yaxistype == 'log':  # plotly's limits are already log10'd in this case
            ylim = 10.0**ylim[0], 10.0**ylim[1]  # but matplotlib's aren't
        axes.set_ylim(ylim)

    #figure out barwidth and offsets for bar plots
    num_bars = sum([get(d, 'type', '') == 'bar' for d in data_trace_list])
    currentBarOffset = 0
    barWidth = (1.0 - bargap) / num_bars if num_bars > 0 else 1.0

    #process traces
    handles = []; labels = []  # for the legend
    boxes = []  # for violins
    for traceDict in data_trace_list:
        typ = get(traceDict, 'type', 'unknown')

        name = get(traceDict, 'name', None)
        showlegend = get(traceDict, 'showlegend', True)

        if typ == "heatmap":
            # traceDict['z'] is *normalized* already - maybe would work here but not for box value labels
            plt_data = pygsti_fig.metadata.get('plt_data', None)
            colormap = pygsti_fig.colormap
            _draw_heatmap_panel(axes, traceDict, plt_data, colormap, xtickvals, ytickvals,
                                xlabels, ylabels, shapes, box_labels_font_size, prec, fontsize,
                                mpl_fig=mpl_fig, resize_fig=True)

        elif typ == "scatter":
            mode = get(traceDict, 'mode', 'lines')
            marker = get(traceDict, 'marker', None)
            line = get(traceDict, 'line', None)
            if marker and (line is None):
                line = marker['line']  # 2nd attempt to get line props

            if marker:
                color = get(marker, 'color', None)
            if line and (color is None):
                color = get(line, 'color', None)
            if color is None:
                color = 'rgb(0,0,0)'

            if isinstance(color, tuple):
                color = [mpl_color(c) for c in color]
            else:
                color = mpl_color(color)

            linewidth = float(line['width']) if (line and get(line, 'width', None) is not None) else 1.0

            x = y = None
            if 'x' in traceDict and 'y' in traceDict:
                x = traceDict['x']
                y = traceDict['y']
            elif 'r' in traceDict and 't' in traceDict:
                x = traceDict['r']
                y = traceDict['t']

            assert(x is not None and y is not None), "x and y both None in trace: %s" % traceDict
            if mode == 'lines':
                if isinstance(color, list):
                    raise ValueError('List of colors incompatible with lines mode')
                lines = _plt.plot(x, y, linestyle='-', marker=None, color=color, linewidth=linewidth)
            elif mode == 'markers':
                lines = _plt.scatter(x, y, marker=".", color=color)
            elif mode == 'lines+markers':
                if isinstance(color, list):
                    # List of colors only works for markers with scatter, have default black line
                    lines = _plt.plot(x, y, linestyle='-', color=(0, 0, 0), linewidth=linewidth)
                    _plt.scatter(x, y, marker='.', color=color)
                else:
                    lines = _plt.plot(x, y, linestyle='-', marker='.', color=color, linewidth=linewidth)
            else: raise ValueError("Unknown mode: %s" % mode)

            if showlegend and name:
                handles.append(lines[0])
                labels.append(name)

        elif typ == "scattergl":  # currently used only for colored points...
            x = traceDict['x']
            y = traceDict['y']
            assert(x is not None and y is not None), "x and y both None in trace: %s" % traceDict

            colormap = pygsti_fig.colormap
            if colormap:
                norm, cmap = colormap.create_matplotlib_norm_and_cmap()
                s = _plt.scatter(x, y, c=y, s=50, cmap=cmap, norm=norm)
            else:
                s = _plt.scatter(x, y, c=y, s=50, cmap='gray')

            if showlegend and name:
                handles.append(s)
                labels.append(name)

        elif typ == "bar":
            xlabels = [str(xl) for xl in traceDict['x']]  # x "values" are actually bar labels in plotly

            #always grey=pos, red=neg type of bar plot for now (since that's all pygsti uses)
            y = _np.asarray(traceDict['y'])
            if 'plt_yerr' in pygsti_fig.metadata:
                yerr = pygsti_fig.metadata['plt_yerr']
            else:
                yerr = None

            # actual x values are just the integers + offset
            x = _np.arange(y.size) + currentBarOffset
            currentBarOffset += barWidth  # so next bar trace will be offset correctly

            marker = get(traceDict, 'marker', None)
            if marker and ('color' in marker):
                if isinstance(marker['color'], str):
                    color = mpl_color(marker['color'])
                elif isinstance(marker['color'], list):
                    color = [mpl_color(c) for c in marker['color']]  # b/c axes.bar can take a list of colors
                else: color = "gray"

            if yerr is None:
                axes.bar(x, y, barWidth, color=color)
            else:
                axes.bar(x, y, barWidth, color=color,
                         yerr=yerr.ravel().real)

            if xtickvals is not None:
                xtics = _np.array(xtickvals) + 0.5  # _np.arange(plt_data.shape[1])+0.5
            else: xtics = x
            axes.set_xticks(xtics, minor=False)
            axes.set_xticklabels(mpl_process_lbls(xlabels), rotation=0, fontsize=(fontsize - 4))

        elif typ == "histogram":
            #histnorm = get(traceDict,'histnorm',None)
            marker = get(traceDict, 'marker', None)
            color = mpl_color(marker['color'] if marker and isinstance(marker['color'], str) else "gray")
            xbins = traceDict['xbins']
            histdata = traceDict['x']

            if abs(xbins['size']) < 1e-6:
                histBins = 1
            else:
                histBins = int(round((xbins['end'] - xbins['start']) / xbins['size']))

            histdata_finite = _np.take(histdata, _np.where(_np.isfinite(histdata)))[
                0]  # take gives back (1,N) shaped array (why?)
            if yaxistype == 'log':
                if len(histdata_finite) == 0:
                    axes.set_yscale("linear")  # no data, and will get an error with log-scale, so switch to linear

            #histMin = min( histdata_finite ) if cmapFactory.vmin is None else cmapFactory.vmin
            #histMax = max( histdata_finite ) if cmapFactory.vmax is None else cmapFactory.vmax
            #_plt.hist(_np.clip(histdata_finite,histMin,histMax), histBins,
            #          range=[histMin, histMax], facecolor='gray', align='mid')
            _, _, patches = _plt.hist(histdata_finite, histBins,
                                      facecolor=color, align='mid')

            #If we've been given an array of colors
            if marker and ('color' in marker) and isinstance(marker['color'], list):
                for p, c in zip(patches, marker['color']):
                    _plt.setp(p, 'facecolor', mpl_color(c))

        elif typ == "box":
            boxes.append(traceDict)

    if len(boxes) > 0:
        _plt.violinplot([box['y'] for box in boxes], [box['x0'] for box in boxes],
                        points=10, widths=1., showmeans=False,
                        showextrema=False, showmedians=False)
        # above kwargs taken from Tim's original RB plot - we could set some of
        # these from boxes[0]'s properties like 'boxmean' (a boolean) FUTURE?

    extraartists = [axes]
    if len(handles) > 0:
        lgd = _plt.legend(handles, labels, bbox_to_anchor=(1.01, 1.0),
                          borderaxespad=0., loc="upper left")
        extraartists.append(lgd)

    if save_to:
        _gc.collect()  # too many open files (b/c matplotlib doesn't close everything) can cause the below to fail
        _plt.savefig(save_to, bbox_extra_artists=extraartists,
                     bbox_inches='tight')  # need extra artists otherwise
        #axis labels get clipped
        _plt.cla()
        _plt.close(mpl_fig)
        del mpl_fig
        _gc.collect()  # again, to be safe...
        if len(_plt.get_fignums()) != numMPLFigs:
            raise ValueError("WARNING: MORE FIGURES OPEN NOW (%d) THAN WHEN WE STARTED %d)!!" %
                             (len(_plt.get_fignums()), numMPLFigs))
        return None  # figure is closed!
    else:
        return mpl_fig


#Special processing for the key-plot: since it uses so much weird
# plotly and matplotlib construction it makes no sense to try to
# automatically convert.
def special_keyplot(pygsti_fig: ReportFigure, save_to: Optional[str], fontsize: int) -> Optional[_MplFigure]:
    """
    Create a plot showing the layout of a single sub-block of a goodness-of-fit box plot.

    Parameters
    ----------
    pygsti_fig : ReportFigure
        The pyGSTi figure to process.

    save_to : str
        Filename to save to.

    fontsize : int
        Fone size to use

    Returns
    -------
    matplotlib.Figure
    """

    #Hardcoded
    title = ""
    prepStrs, effectStrs, xlabel, ylabel = pygsti_fig.metadata['args']

    fig, axes = _plt.subplots()
    mpl_size = (len(prepStrs) * 0.5, len(effectStrs) * 0.5)
    fig.set_size_inches(*mpl_size)
    pygsti_fig.metadata['mpl_fig_size'] = mpl_size  # record for later use by rendering commands

    if title is not None:
        axes.set_title(title, fontsize=(fontsize + 4))

    if xlabel is not None:
        axes.set_xlabel(xlabel, fontsize=(fontsize + 4))

    if ylabel is not None:
        axes.set_ylabel(ylabel, fontsize=(fontsize + 4))

    #Copied from _summable_color_boxplot
    def _val_filter(vals):  # filter to latex-ify circuits.  Later add filter as a possible parameter
        formatted_vals = []
        for val in vals:
            if type(val) in (tuple, _Circuit) and all([type(el) == str for el in val]):
                if len(val) == 0:
                    formatted_vals.append(r"$\{\}$")
                else:
                    formatted_vals.append("$" + "\\cdot".join([("\\mathrm{%s}" % el) for el in val]) + "$")
            else:
                formatted_vals.append(val)
        return formatted_vals

    # for suppression of "UserWarning: FixedFormatter should only be used together with FixedLocator" warning
    import matplotlib.ticker as mticker

    axes.yaxis.tick_right()
    axes.xaxis.set_label_position("top")
    axes.xaxis.set_major_locator(mticker.FixedLocator(axes.get_xticks().tolist()))  # avoids matplotlib warning (above)
    axes.yaxis.set_major_locator(mticker.FixedLocator(axes.get_yticks().tolist()))  # when calling set_[x|y]ticklabels
    axes.set_xticklabels(_val_filter(prepStrs), rotation=90, ha='center', fontsize=fontsize)
    axes.set_yticklabels(list(reversed(_val_filter(effectStrs))), fontsize=fontsize)  # FLIP
    axes.set_xticks(_np.arange(len(prepStrs)) + 0.5)
    axes.set_xticks(_np.arange(len(prepStrs) + 1), minor=True)
    axes.set_yticks(_np.arange(len(effectStrs)) + 0.5)
    axes.set_yticks(_np.arange(len(effectStrs) + 1), minor=True)
    axes.tick_params(which='major', bottom='off', top='off', left='off', right='off', pad=5)
    axes.yaxis.grid(True, linestyle='-', linewidth=1.0, which='minor')
    axes.xaxis.grid(True, linestyle='-', linewidth=1.0, which='minor')

    if save_to is not None:
        if len(save_to) > 0:  # So you can pass save_to="" and figure will be closed but not saved to a file
            _plt.savefig(save_to, bbox_extra_artists=(axes,), bbox_inches='tight')

        _plt.cla()
        _plt.close(fig)  # close the figure if we're saving it to a file
    else:
        return fig
