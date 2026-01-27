"""
Matplotlib volumetric benchmarking plotting routines.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np


def _build_default_colormap():
    try:
        import matplotlib.pyplot as _plt
        from matplotlib.colors import ListedColormap as _ListedColormap
        from matplotlib import cm as _cm
        import seaborn as _sns

        _sns.set_style('white')
        _sns.set_style('ticks')

        # Utility color maps.
        blues = _sns.color_palette(_sns.color_palette("Blues", 200)).as_hex()
        blues[0] = '#ffffff'
        blues = _ListedColormap(blues)

        #reds = _sns.color_palette(_sns.color_palette("Reds", 200)).as_hex()
        #reds[0] = '#ffffff'
        #reds = _ListedColormap(reds)
        #
        #greens = _sns.color_palette(_sns.color_palette("Greens", 200)).as_hex()
        #greens[0] = '#ffffff'
        #greens = _ListedColormap(greens)
        #
        #binary_blue = _sns.color_palette(_sns.color_palette("Blues", 200)).as_hex()
        #binary_blue[0] = '#ffffff'
        #binary_blue = _ListedColormap([binary_blue[0], binary_blue[50]])

        #spectral = _cm.get_cmap('Spectral')

        # The default color map.
        my_cmap = blues

    except ImportError:
        my_cmap = None
    return my_cmap


def empty_volumetric_plot(figsize=None, y_values=None, x_values=None, title=None, xlabel='Depth', ylabel='Width'):
    """
    Creates an empty volumetric plot with just the axes set.

    Parameters
    ----------
    figsize : tuple or None, optional
        The figure size.

    y_values : list or None, optional
        The y-axis values, typically corresponding to circuit widths.

    x_values : list or None, optional
        The x-axis values, typically corresponding to circuit depths.

    title : string or None, optional
        Plot title

    xlabel : string, optional
        x-axis label

    ylabel : string, optional
        y-axis label.

    Return
    ------
    fig, ax : matplolib fig and ax.
    """
    try:
        import matplotlib.pyplot as _plt
        import seaborn as _sns
    except ImportError:
        raise ValueError(("While not a core requirement of pyGSTi, Matplotlib and Seaborn are "
                          "required to generate VB plots.  It looks like you "
                          "don't have them installed on your system (it failed to import)."))

    fig, ax = _plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    _plt.xlabel(xlabel, fontsize=20)
    _plt.ylabel(ylabel, fontsize=20)
    _plt.title(title, fontsize=24, y=1.02)
    _plt.xlim(-1, len(x_values))
    _plt.ylim(-1, len(y_values))
    depth_labels = [str(d)[0:len(str(d)) - ((len(str(d)) - 1) // 3) * 3]
                    + ['', 'k', 'M', 'G'][(len(str(d)) - 1) // 3] for d in x_values]
    _plt.xticks(range(len(x_values)), depth_labels, rotation=-60, fontsize=14)
    _plt.yticks(range(len(y_values)), y_values, fontsize=14)

    _sns.despine()

    return fig, ax


def _get_xy(data, y_values=None, x_values=None):
    # Helper function for setting the x and y axes of VB plots.
    if x_values is None:
        x_values = list(set([shape[0] for shape in data.keys()]))
        x_values.sort()
    if y_values is None:
        y_values = list(set([shape[1] for shape in data.keys()]))
        y_values.sort()

    return y_values, x_values


def volumetric_plot(data, y_values=None, x_values=None, title=None, fig=None, ax=None,
                    cmap=None, color=None, flagQV=False, qv_threshold=None,
                    figsize=(10, 10), scale=1., centerscale=1., linescale=1.,
                    pass_threshold=0, show_threshold=0):
    """
    Creates a volumetric benchmarking plot.
    """
    y_values, x_values = _get_xy(data, y_values, x_values)

    if fig is None:
        fig, ax = empty_volumetric_plot(figsize=figsize, y_values=y_values, x_values=x_values, title=title)

    if qv_threshold is None:
        qv_threshold = pass_threshold

    if color is not None:
        cmap = None
        point_color = color
    elif cmap is None:
        cmap = _build_default_colormap()

    for indw, w in enumerate(y_values):
        for indd, d in enumerate(x_values):

            edgecolor = 'k'
            linewidth = 1 * linescale
            datapoint = data.get((d, w), None)

            if (datapoint is not None) and (not _np.isnan(datapoint)):

                if w == d and flagQV:
                    if datapoint > qv_threshold:
                        edgecolor = 'r'
                        linewidth = 5 * scale * linescale

                if datapoint >= show_threshold:
                    if datapoint < pass_threshold:
                        datapoint = 0

                    if color is None:
                        point_color = [datapoint]
                    ax.scatter([indd], [indw], marker="s", s=280 * scale - 30 * linewidth, c=point_color,
                               cmap=cmap, vmin=0, vmax=1, edgecolor=edgecolor, linewidth=linewidth)

    return fig, ax


def volumetric_boundary_plot(data, y_values=None, x_values=None, boundary=None, threshold=.5,
                             missing_data_action='continue', monotonic=True, color='k', linewidth=4,
                             linestyle='-', dashing=None, fig=None, ax=None, figsize=None, title=None,
                             label=None):
    """
    Creates a volumetric benchmarking boundary plot, that displays boundary at which the given data
    drops below the specified threshold
    """
    y_values, x_values = _get_xy(data, y_values, x_values)

    if fig is None:
        fig, ax = empty_volumetric_plot(figsize=figsize, y_values=y_values, x_values=x_values, title=title)

    if boundary is not None:
        boundaries = _np.array([-1 if boundary[d] == 0 else y_values.index(boundary[d]) for d in x_values])
        # x-values for a jagged line that outlines the boxes (one pair for each box)
        xvals = [y for x in range(len(x_values)) for y in [x - .5, x + .5]]
        # y-values for a jagged line that outlines the boxes (one pair for each box)
        yvals = [y + .5 for boundary in boundaries for y in [boundary, boundary]]

    else:
        # For each depth, find the widest circuit that achieves the threshold performance (return -1 if none)
        if missing_data_action == 'none':
            boundaries = _np.array([_np.max([-1] + [y_values.index(w) for w in y_values if (d, w) in data.keys()
                                                    and data[d, w] >= threshold]) for d in x_values])
            # x-values for a jagged line that outlines the boxes (one pair for each box)
            xvals = [y for x in range(len(x_values)) for y in [x - .5, x + .5]]
            # y-values for a jagged line that outlines the boxes (one pair for each box)
            yvals = [y + .5 for boundary in boundaries for y in [boundary, boundary]]

        elif missing_data_action == 'continue' or missing_data_action == 'hedge':
            boundaries = []
            d = x_values[0]
            boundary_at_d = _np.max([-1] + [y_values.index(w) for w in y_values if (d, w) in data.keys()
                                            and data[d, w] >= threshold])
            boundaries.append(boundary_at_d)
            previous_boundary = boundary_at_d
            hedged_x_values = []
            for i, d in enumerate(x_values[1:]):
                max_width_at_depth = _np.max([-1] + [w for w in y_values if (d, w) in data.keys()])
                if max_width_at_depth < previous_boundary:
                    boundary_at_d = previous_boundary
                    hedged_x_values.append(d)
                else:
                    boundary_at_d = _np.max([-1] + [y_values.index(w) for w in y_values if (d, w) in data.keys()
                                                    and data[d, w] >= threshold])
                boundaries.append(boundary_at_d)
                previous_boundary = boundary_at_d

            if missing_data_action == 'continue':
                # x-values for a jagged line that outlines the boxes (one pair for each box)
                xvals = [y for x in range(len(x_values)) for y in [x - .5, x + .5]]
                # y-values for a jagged line that outlines the boxes (one pair for each box)
                yvals = [y + .5 for boundary in boundaries for y in [boundary, boundary]]

            elif missing_data_action == 'hedge':
                # x-values for a jagged line that outlines the boxes (one pair for each box)
                xvals = []
                yvals = []
                last_xval = -0.5
                for x, boundary in zip(range(len(x_values)), boundaries):
                    d = x_values[x]
                    if d in hedged_x_values:
                        # Only hedge when there's actually some data at larger x_values.
                        if not all([d in hedged_x_values for d in x_values[x:]]):
                            xvals += [last_xval, x]
                            yvals += [boundary + .5, boundary + .5]
                    else:
                        xvals += [last_xval, x + .5]
                        yvals += [boundary + .5, boundary + .5]
                    last_xval = xvals[-1]

    if monotonic:
        monotonic_yvals = [yvals[0]]
        for y in yvals[1:]:
            if y > monotonic_yvals[-1]:
                monotonic_yvals.append(monotonic_yvals[-1])
            else:
                monotonic_yvals.append(y)
        yvals = monotonic_yvals

    line, = ax.plot(xvals, yvals, color, linewidth=linewidth, label=label, linestyle=linestyle)
    if dashing is not None:
        line.set_dashes(dashing)
    return fig, ax


def capability_region_plot(vbdataframe, metric='polarization', threshold=1 / _np.e, significance=0.05, figsize=(10, 10),
                           scale=1., title=None, colors=None):
    """
    Creates a capability regions plot from a VBDataFrame. Default options creates plots like those shown
    in Fig. 3 of "Measuring the Capabilities of Quantum Computers" arXiv:2008.11294.
    """
    x_values = vbdataframe.x_values
    y_values = vbdataframe.y_values

    fig, ax = empty_volumetric_plot(figsize=figsize, y_values=y_values, x_values=x_values, title=title)

    creg = vbdataframe.capability_regions(metric=metric, threshold=threshold, significance=significance, monotonic=True)

    # Split the data up into dicts for the three different regions: 'success', 'indeterminate' and 'fail'.
    creg_split = {}
    creg_split['success'] = {(w, d): 1 for (w, d), val in creg.items() if val == 2}
    creg_split['indeterminate'] = {(w, d): 1 for (w, d), val in creg.items() if val == 1}
    creg_split['fail'] = {(w, d): 1 for (w, d), val in creg.items() if val == 0}

    if colors is None:
        colors = {'success': [(0.2, 0.6274509803921569, 0.17254901960784313)],
                  'indeterminate': [(0.9921568627450981, 0.7490196078431373, 0.43529411764705883)],
                  'fail': 'w'}

    for region in ('success', 'indeterminate', 'fail'):
        fig, ax = volumetric_plot(creg_split[region], y_values=y_values, x_values=x_values, scale=scale, fig=fig, ax=ax,
                                  color=colors[region])

    return fig, ax


def volumetric_distribution_plot(vbdataframe, metric='polarization', threshold=1 / _np.e, hypothesis_test='standard',
                                 significance=0.05, figsize=(10, 10), scale=None, title=None, cmap=None):
    """
    Creates volumetric benchmarking plots that display the maximum, mean and minimum of a given figure-of-merit (by
    default, circuit polarization) as a function of circuit shape. This function can be used to create figures like
    those shown in Fig. 1 of "Measuring the Capabilities of Quantum Computers" arXiv:2008.11294.

    Parameters
    ----------
    vbdataframe : VBDataFrame
        A VBDataFrame object containing the data to be plotted in a VB plot.

    metric : string, optional
        The quantity to plot. Default is 'polarization' as used and defined in arXiv:2008.11294. The plot
        will show the maximum, mean, and minimum of this metric at each circuit shape.

    threshold : float, optional
        The threshold for "success" for the figure-of-merit defined by `metric`. This threshold is used
        to compute the three "success" boundaries that are shown in the plot.

    hypothesis_test : string, optional
        The type of statistical significance adjustment to apply to the boundaries. The options are
        
        * 'standard': this reproduces the method used and described in arXiv:2008.11294 (see the appendices for details). With this option, there will be a difference between the boundary for the minimum and maximum polarization only if there is statistically significant evidence in the data for this.
        * 'none': no statistical significance adjustment: all three boundaries show the point at which relevant statistic (maximum, mean, minimum) drops below the threshold.

    significance : float, optional
        The statistical significance in the hypothesis tests. Only used in `hypothesis_test` is not 'none'.

    figsize : tuple, optional
        The figure size

    scale : dict, optional
        The scale for the three concentric squares, showing the maximum, mean and minimum.
        Defaults to {'min': 1.95, 'mean': 1, 'max': 0.13}.

    title : sting, optional
        The figure title.

    cmap : ColorMap, optional
        A matplotlib colormap.

    Return
    ------
    fig, ax : matplolib fig and ax.
    """
    if scale is None:
        scale = {'min': 1.95, 'mean': 1, 'max': 0.13}
    linescale = {'min': 1, 'mean': 0, 'max': 0}
    boundary_color = {'min': '#ff0000', 'mean': '#000000', 'max': '#2ecc71'}
    boundary_dashing = {'min': [1, 1], 'mean': None, 'max': [0.5, 0.5]}
    boundary_linewidth = {'min': 3, 'mean': 6, 'max': 5}
    x_values = vbdataframe.x_values
    y_values = vbdataframe.y_values

    fig, ax = empty_volumetric_plot(figsize=figsize, y_values=y_values, x_values=x_values, title=title)

    # Dictionary containing the three types of VB data that are used in this plot.
    vb_data = {stat: vbdataframe.vb_data(metric=metric, statistic=stat, no_data_action='discard')
               for stat in ('min', 'mean', 'max')}
    # Used to find the min and max boundaries if they are adjusted for statistical significance.
    capability_regions = vbdataframe.capability_regions(metric=metric, threshold=threshold, significance=significance,
                                                        monotonic=True)

    if hypothesis_test == 'standard':
        adjusted_boundaries = ('max', 'min')
        unadjusted_boundaries = ('mean',)

    elif hypothesis_test == 'none':
        adjusted_boundaries = ()
        unadjusted_boundaries = ('max', 'mean', 'min',)

    else:
        raise ValueError("`hypothesis_test` must be 'standard' or 'none'!")

    # Plots the data.
    for statistic in ('min', 'mean', 'max'):
        fig, ax = volumetric_plot(vb_data[statistic], y_values=y_values, x_values=x_values, fig=fig, ax=ax,
                                  scale=scale[statistic], linescale=linescale[statistic], cmap=cmap)

    # Plots the boundaries that have been adjusted for statistical significance.
    for statistic in adjusted_boundaries:
        if statistic == 'max': effective_threshold = 0.99
        elif statistic == 'min': effective_threshold = 1.99
        volumetric_boundary_plot(capability_regions, y_values=y_values, x_values=x_values,
                                 threshold=effective_threshold,
                                 missing_data_action='hedge', fig=fig, ax=ax, linestyle='-',
                                 color=boundary_color[statistic], linewidth=boundary_linewidth[statistic],
                                 dashing=boundary_dashing[statistic])

    # Plots the boundaries that are not adjusted for statistical significance.
    for statistic in unadjusted_boundaries:
        volumetric_boundary_plot(vb_data[statistic], y_values=y_values, x_values=x_values, threshold=threshold,
                                 monotonic=False, missing_data_action='hedge', fig=fig, ax=ax, linestyle='-',
                                 color=boundary_color[statistic], linewidth=boundary_linewidth[statistic],
                                 dashing=boundary_dashing[statistic])

    return fig, ax
