"""
Matplotlib volumetric benchmarking plotting routines.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np

try:
    import matplotlib as _matplotlib
    import matplotlib.pyplot as _plt
except ImportError:
    raise ValueError(("While not a core requirement of pyGSTi, Matplotlib is "
                      "required to generate VB plots.  It looks like you "
                      "don't have it installed on your system (it failed to "
                      "import)."))

try:
    import seaborn as _sns
except ImportError:
    raise ValueError(("While not a core requirement of pyGSTi, Seaborn is "
                      "required to generate VB plots.  It looks like you "
                      "don't have it installed on your system (it failed to "
                      "import)."))

_sns.set_style('white')
_sns.set_style('ticks')

from matplotlib.colors import ListedColormap as _ListedColormap
# The default color map
my_cmap = _sns.color_palette(_sns.color_palette("Blues", 200)).as_hex()
my_cmap[0] = '#ffffff'
my_cmap = _ListedColormap(my_cmap)

# Utility color maps.
reds = _sns.color_palette(_sns.color_palette("Reds", 200)).as_hex()
reds[0] = '#ffffff'
reds = _ListedColormap(reds)

greens = _sns.color_palette(_sns.color_palette("Greens", 200)).as_hex()
greens[0] = '#ffffff'
greens = _ListedColormap(greens)

binary_blue = _sns.color_palette(_sns.color_palette("Blues", 200)).as_hex()
binary_blue[0] = '#ffffff'
binary_blue = _ListedColormap([binary_blue[0], binary_blue[50]])


def create_empty_volumetric_plot(figsize=None, widths=None, depths=None, title=None,
                                 xlabel='Depth', ylabel='Width'):

    fig, ax = _plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    _plt.xlabel(xlabel, fontsize=20)
    _plt.ylabel(ylabel, fontsize=20)
    _plt.title(title, fontsize=24, y=1.02)
    _plt.xlim(-1, len(depths))
    _plt.ylim(-1, len(widths))
    depth_labels = [str(d)[0:len(str(d)) - ((len(str(d)) - 1) // 3) * 3]
                    + ['', 'k', 'M', 'G'][(len(str(d)) - 1) // 3] for d in depths]
    _plt.xticks(range(len(depths)), depth_labels, rotation=-60, fontsize=14)
    _plt.yticks(range(len(widths)), widths, fontsize=14)

    _sns.despine()

    return fig, ax


def _get_xy(data, widths=None, depths=None):

    if depths is None:
        depths = list(set([shape[0] for shape in data.keys()]))
        depths.sort()
    if widths is None:
        widths = list(set([shape[1] for shape in data.keys()]))
        widths.sort()

    return widths, depths


def volumetric_plot(data, widths=None, depths=None, title=None, fig=None, ax=None,
                    cmap=my_cmap, color=None, flagQV=False, qv_threshold=None,
                    figsize=(10, 10), scale=1., centerscale=1., linescale=1.,
                    pass_threshold=0, show_threshold=0):

    widths, depths = _get_xy(data, widths, depths)

    if fig is None:
        fig, ax = create_empty_volumetric_plot(figsize=figsize, widths=widths, depths=depths, title=title)
    
    if qv_threshold is None:
        qv_threshold = pass_threshold
        
    if color is not None:
        cmap = None
        point_color = color
        
    for indw, w in enumerate(widths):
        for indd, d in enumerate(depths):
            
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


def boundaryPlot(data, widths=None, depths=None, boundary=None, threshold=.5, missing_data_action='continue',
                 monotonic=True, color='k', linewidth=4, linestyle='-', dashing=None, 
                 fig=None, ax=None, figsize=None, title=None, label=None):

    widths, depths = _get_xy(data, widths, depths)
    
    if fig is None:
        fig, ax = create_empty_volumetric_plot(figsize=figsize, widths=widths, depths=depths, title=title)
    
    if boundary is not None:
        #print(boundary)
        #print(widths)
        #print(depths)
        boundaries = _np.array([-1 if boundary[d] == 0 else widths.index(boundary[d]) for d in depths])
        # x-values for a jagged line that outlines the boxes (one pair for each box)        
        xvals = [y for x in range(len(depths)) for y in [x - .5, x + .5]]
        # y-values for a jagged line that outlines the boxes (one pair for each box)
        yvals = [y + .5 for boundary in boundaries for y in [boundary, boundary]]

    else:
        # For each depth, find the widest circuit that achieves the threshold performance (return -1 if none)
        if missing_data_action == 'none':
            boundaries = _np.array([_np.max([-1] + [widths.index(w) for w in widths if (d, w) in data.keys() and data[d, w] >= threshold]) for d in depths])
            # x-values for a jagged line that outlines the boxes (one pair for each box)
            xvals = [y for x in range(len(depths)) for y in [x - .5, x + .5]]
            # y-values for a jagged line that outlines the boxes (one pair for each box)
            yvals = [y + .5 for boundary in boundaries for y in [boundary, boundary]]

        elif missing_data_action == 'continue' or missing_data_action == 'hedge':
            boundaries = []
            d = depths[0]
            boundary_at_d = _np.max([-1] + [widths.index(w) for w in widths if (d, w) in data.keys() and data[d, w] >= threshold])
            boundaries.append(boundary_at_d)
            previous_boundary = boundary_at_d
            hedged_depths = []
            for i, d in enumerate(depths[1:]):
                max_width_at_depth = _np.max([-1] + [w for w in widths if (d, w) in data.keys()])
                if max_width_at_depth < previous_boundary:
                    boundary_at_d = previous_boundary
                    hedged_depths.append(d)
                else:
                    boundary_at_d = _np.max([-1] + [widths.index(w) for w in widths if (d, w) in data.keys() and data[d, w] >= threshold])
                boundaries.append(boundary_at_d)
                previous_boundary = boundary_at_d

            if missing_data_action == 'continue':
                # x-values for a jagged line that outlines the boxes (one pair for each box)        
                xvals = [y for x in range(len(depths)) for y in [x - .5, x + .5]]
                # y-values for a jagged line that outlines the boxes (one pair for each box)
                yvals = [y + .5 for boundary in boundaries for y in [boundary, boundary]]
           
            elif missing_data_action == 'hedge':
                # x-values for a jagged line that outlines the boxes (one pair for each box)
                xvals = []
                yvals = []
                last_xval = -0.5
                for x, boundary in zip(range(len(depths)), boundaries):
                    d = depths[x]
                    if d in hedged_depths:
                        # Only hedge when there's actually some data at larger depths.
                        if not all([d in hedged_depths for d in depths[x:]]):
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


def capability_region_plot(vbdataframe, threshold=1 / _np.e, significance=0.05, figsize=(10, 10), scale=1., title=None):

    x_values = vbdataframe.x_values
    y_values = vbdataframe.y_values

    fig, ax = create_empty_volumetric_plot(figsize=figsize, widths=y_values, depths=x_values, title=title)

    creg = vbdataframe.get_capability_regions(metric='polarization', threshold=threshold, significance=significance, monotonic=True)

    creg_split = {}
    creg_split['success'] = {(w,d):1 for (w,d), val in creg.items() if val == 2}
    creg_split['indeterminate'] = {(d, w):1 for (d, w), val in creg.items() if val == 1}
    creg_split['fail'] = {(d, w):1 for (d, w), val in creg.items() if val == 0}

    color = {'success':[(0.2, 0.6274509803921569, 0.17254901960784313)],
             'indeterminate':[(0.9921568627450981, 0.7490196078431373, 0.43529411764705883)],
             'fail':'w'}

    for region in ('success', 'indeterminate', 'fail'):
        fig, ax = volumetric_plot(creg_split[region], widths=y_values, depths=x_values, scale=scale, fig=fig, ax=ax, color=color[region])

    return fig, ax
