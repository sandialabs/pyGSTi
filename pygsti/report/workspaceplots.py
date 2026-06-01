"""
Classes corresponding to plots within a Workspace context.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import collections as _collections
import warnings as _warnings
from pathlib import Path

import numpy as _np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from itertools import product as _product

import scipy as _scipy
from scipy.stats import chi2 as _chi2
from PIL import ImageFont as _ImageFont

from pygsti.objectivefns.objectivefns import ModelDatasetCircuitsStore as _ModelDatasetCircuitStore
from pygsti.report import colormaps as _colormaps
from pygsti.report import plothelpers as _ph
from pygsti.report.figure import ReportFigure
from pygsti.report.workspace import WorkspacePlot, NotApplicable
from pygsti import algorithms as _alg
from pygsti import baseobjs as _baseobjs
from pygsti.objectivefns import objectivefns as _objfns
from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti.circuits.circuitstructure import PlaquetteGridCircuitStructure as _PlaquetteGridCircuitStructure, \
    GermFiducialPairPlaquette as _GermFiducialPairPlaquette
from pygsti.circuits.circuitlist import CircuitList as _CircuitList
from pygsti.data import DataSet as _DataSet


go_x_axis = go.layout.XAxis
go_y_axis = go.layout.YAxis
go_margin = go.layout.Margin
go_annotation = go.layout.Annotation


def _color_boxplot(plt_data, colormap, colorbar=False, box_label_size=0,
                   prec=0, hover_label_fn=None, hover_labels=None, return_hover_labels=False):
    """
    Create a color box plot.

    Creates a plot.ly heatmap figure composed of colored boxes and
    possibly labels.

    Parameters
    ----------
    plt_data : numpy array
        A 2D array containing the values to be plotted.  None values will
        show up as white.

    colormap : Colormap
        The colormap used to determine box color.

    colorbar : bool, optional
        Whether or not to show the color scale bar.

    box_label_size : int, optional
        If greater than 0, display static labels on each box with font
        size equal to `box_label_size`.

    prec : int or {'compact','compacthp'}, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    hover_label_fn : function, optional
        A function with signature `f(z,i,j)` where `z ==plt_data[i,j]` which
        computes the hover label for the each element of `plt_data`.  Cannot
        be used with `hover_labels`.

    hover_labels : list of lists, optional
        Strings specifying the hover labels for each element of `plt_data`.
        E.g. `hover_labels[i,j]` is the string for the i-th row (y-value)
        and j-th column (x-value) of the plot.

    return_hover_labels : bool, optional (default False)
        If True, additionally return the parsed (nested) lists of
        hover labels for each point.

    Returns
    -------
    plotly.Figure
    """

    masked_data = _np.ma.array(plt_data, mask=_np.isnan(plt_data))
    heatmapArgs = {'z': colormap.normalize(masked_data),
                   'colorscale': colormap.create_plotly_colorscale(),
                   'showscale': colorbar, 'hoverinfo': 'none',
                   'zmin': colormap.hmin, 'zmax': colormap.hmax,
                   'xgap':1, 'ygap':1}
    
    heatmapArgsborderbg = {'z': masked_data,
                   'colorscale': ['#000', '#000'],
                   'showscale': False,
                   'xgap':0, 'ygap':0,
                   'hoverinfo': 'none'}

    #if xlabels is not None: heatmapArgs['x'] = xlabels
    #if ylabels is not None: heatmapArgs['y'] = ylabels

    annotations = []
    if box_label_size:
        # Write values on colored squares
        for y in range(plt_data.shape[0]):
            for x in range(plt_data.shape[1]):
                if _np.isnan(plt_data[y, x]): continue
                annotations.append(
                    dict(
                        text=_ph._eformat(plt_data[y, x], prec),
                        x=x, y=y,
                        xref='x1', yref='y1',
                        font=dict(size=box_label_size,
                                  color=colormap.besttxtcolor(plt_data[y, x])),
                        showarrow=False)
                )

    if hover_label_fn:
        assert(not hover_labels), "Cannot specify hover_label_fn and hover_labels!"
        hover_labels = []
        for y in range(plt_data.shape[0]):
            hover_labels.append([hover_label_fn(plt_data[y, x], y, x)
                                 for x in range(plt_data.shape[1])])
    if hover_labels:
        heatmapArgs['hoverinfo'] = 'text'
        heatmapArgs['text'] = hover_labels

    trace = go.Heatmap(**heatmapArgs)
    trace_border = go.Heatmap(**heatmapArgsborderbg)

    data = [trace_border, trace]

    xaxis = go_x_axis(
        showgrid=False,
        zeroline=False,
        showline=True,
        ticks="",
        showticklabels=True,
        mirror=True,
        linewidth=2,
        range=[-1, plt_data.shape[1]]
    )

    yaxis = go_y_axis(
        showgrid=False,
        zeroline=False,
        showline=True,
        ticks="",
        showticklabels=True,
        mirror=True,
        linewidth=2,
        range=[-1, plt_data.shape[0]]
    )

    layout = go.Layout(
        xaxis=xaxis,
        yaxis=yaxis,
        annotations=annotations,
        hoverlabel= dict(font_family = 'monospace') #add hoverlabel formatting
    )

    fig = go.Figure(data=data, layout=layout)
    rfig = ReportFigure(fig, colormap, plt_data, plt_data=plt_data)
    if return_hover_labels: 
        return rfig, hover_labels
    else:
        return rfig


def _nested_color_boxplot(plt_data_list_of_lists, colormap,
                          colorbar=False, box_label_size=0, prec=0,
                          hover_label_fn=None, return_hover_labels= False):
    """
    Creates a "nested" color box plot.

    Tiles the plaquettes given by `plt_data_list_of_lists`
    onto a single heatmap.

    Parameters
    ----------
    plt_data_list_of_lists : list of lists of numpy arrays
        A complete square 2D list of lists, such that each element is a
        2D numpy array of the same size.

    colormap : Colormap
        The colormap used to determine box color.

    colorbar : bool, optional
        Whether or not to show the color scale bar.

    box_label_size : int, optional
        If greater than 0, display static labels on each box with font
        size equal to `box_label_size`.

    prec : int or {'compact','compacthp'}, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    hover_label_fn : function, optional
        A function with signature `f(z,i,j)` where `z ==plt_data[i,j]` which
        computes the hover label for the each element of `plt_data`.  Cannot
        be used with `hoverLabels`.

    return_hover_labels : bool, optional (default False)
        If True, additionally return the parsed (nested) lists of
        hover labels for each point.

    Returns
    -------
    plotly.Figure
    """

    #Assemble the single 2D grid to pass to _color_boxplot
    # (assume a complete 2D rectangular list of lists, and that
    #  each element is a numpy array of the same size)
    if len(plt_data_list_of_lists) == 0 or len(plt_data_list_of_lists[0]) == 0: return

    elRows, elCols = plt_data_list_of_lists[0][0].shape  # nE,nr
    nRows = len(plt_data_list_of_lists)
    nCols = len(plt_data_list_of_lists[0])

    data = _np.zeros((elRows * nRows + (nRows - 1), elCols * nCols + (nCols - 1)))
    for i in range(1, nRows):
        data[(elRows + 1) * i - 1:(elRows + 1) * i, :] = _np.nan
    for j in range(1, nCols):
        data[:, (elCols + 1) * j - 1:(elCols + 1) * j] = _np.nan

    for i in range(nRows):
        for j in range(nCols):
            data[(elRows + 1) * i:(elRows + 1) * (i + 1) - 1, (elCols + 1)
                 * j:(elCols + 1) * (j + 1) - 1] = plt_data_list_of_lists[i][j]

    xtics = []; ytics = []
    for i in range(nRows): ytics.append(float((elRows + 1) * i) - 0.5 + 0.5 * float(elRows))
    for j in range(nCols): xtics.append(float((elCols + 1) * j) - 0.5 + 0.5 * float(elCols))

    if hover_label_fn:
        hoverLabels = []
        for _ in range(elRows * nRows + (nRows - 1)):
            hoverLabels.append([""] * (elCols * nCols + (nCols - 1)))

        for i in range(nRows):
            for j in range(nCols):
                for ii in range(elRows):
                    for jj in range(elCols):
                        hoverLabels[(elRows + 1) * i + ii][(elCols + 1) * j + jj] = \
                            hover_label_fn(plt_data_list_of_lists[i][j][ii][jj], i, j, ii, jj)
    else:
        hoverLabels = None

    fig = _color_boxplot(data, colormap, colorbar, box_label_size,
                         prec, None, hoverLabels)

    #Layout updates: add tic marks (but not labels - leave that to user)
    fig.plotlyfig['layout']['xaxis'].update(tickvals=xtics)
    fig.plotlyfig['layout']['yaxis'].update(tickvals=ytics)
    
    #I *think* it is the case that the only context in which this function is currrently
    #used is in the per-sequence detail plots, so adding the boundaries between
    #plaquettes here should (hopefully) not break anything.

    #add boundaries between plaquettes.
    #if either of these is length 1 then the diff will be empty
    #this should stick the boundaries halfway between the ticks.
    x_boundaries = _np.array(xtics[:-1]) + 0.5*(xtics[1]-xtics[0]) if len(xtics)>1 else []
    y_boundaries = _np.array(ytics[:-1]) + 0.5*(ytics[1]-ytics[0]) if len(ytics)>1 else []
    
    plaquette_boundary_lines = []

    for x_bnd in x_boundaries:
        #fig.plotlyfig.add_vline(x_bnd, line_width=1, line_color = "#616263")
        #a ref value of 'x'/'y' means this value is relative to the axis data.
        #a ref value of 'paper' means this is relative/proportional to the plot area.
        plaquette_boundary_lines.append(dict(type="line", xref= 'x', yref='paper', x0=x_bnd, y0=0, x1=x_bnd, y1=1, 
             line={'color':"#616263", 'width':1, 'dash':"solid"}))
    for y_bnd in y_boundaries:
        #fig.plotlyfig.add_hline(y_bnd, line_width=1, line_color = "#616263")
        plaquette_boundary_lines.append(dict(type="line", xref= 'paper', yref='y', x0=0, y0=y_bnd, x1=1, y1=y_bnd, 
             line={'color':"#616263", 'width':1, 'dash':"solid"}))

    fig.plotlyfig.update_layout(shapes=plaquette_boundary_lines)
    #Add grid lines between the squares within a plaquette
    #Can use a construction similar to the x/y boundary one to
    #get some reference points, but we need to include the endpoints here.
    if len(xtics)>1:
        x_ref = _np.zeros(len(xtics)+1)
        x_ref[1:] = _np.array(xtics) + 0.5*(xtics[1]-xtics[0])
        #The left edge of the figure isn't zero, since we've shifted that
        #in _color_boxplot, so shift the first reference point back a bit.
        x_ref[0] = -1
    else: #just pick out the end points
        x_ref = _np.array([-1, 2*xtics[0]+1])
    if len(ytics)>1:
        y_ref = _np.zeros(len(ytics)+1)
        y_ref[1:] = _np.array(ytics) + 0.5*(ytics[1]-ytics[0])
        #The bottom edge of the figure isn't zero, since we've shifted that
        #in _color_boxplot, so shift the first reference point back a bit.
        y_ref[0] = -1
    else: #just pick out the end points
        y_ref = _np.array([-1, 2*ytics[0]+1]) 

    #Now we want to construct pairs of end points. We can do this by iterating
    #through the x_ref and y_ref lists pairwise and then adjusting their values
    #to match the size of the plaquette.
    x_endpoints = [(x_ref[i-1]+0.5 , x_ref[i]-0.5) for i in range(1, len(x_ref))]
    y_endpoints = [(y_ref[i-1]+0.5 , y_ref[i]-0.5) for i in range(1, len(y_ref))]

    #also create a flattened list of these endpoints for use in the next filtering
    #step.
    x_endpoint_list = [pt for pair in x_endpoints for pt in pair]
    y_endpoint_list = [pt for pair in y_endpoints for pt in pair]
    
    #need to couple these end points with either constant x or y coordinates
    #which are halfway betweent the boxes.
    y_pos = _np.arange(0.5, data.shape[0]-1, 1)
    x_pos = _np.arange(0.5, data.shape[1]-1, 1)

    #To get just the ones between the boxes, remove the points in x_endpoints and
    #y_endpoints, which correspond to the edges of the plaquettes.
    y_pos_filtered = [y for y in y_pos if not any([abs(y-elem)<1e-6 for elem in y_endpoint_list])]
    x_pos_filtered = [x for x in x_pos if not any([abs(x-elem)<1e-6 for elem in x_endpoint_list])]
    
    plaquette_grid_lines = []
    for y in y_pos_filtered:
        for endpoints in x_endpoints:
            plaquette_grid_lines.append(dict(type="line", x0=endpoints[0], y0=y, x1=endpoints[1], y1=y, 
                                             line={'color':"MediumPurple", 'width':.35, 'dash':"1px"}))
    for x in x_pos_filtered:
        for endpoints in y_endpoints:
            plaquette_grid_lines.append(dict(type="line", x0=x, y0=endpoints[0], x1=x, y1=endpoints[1], 
                                             line={'color':"MediumPurple", 'width':.35, 'dash':"1px"}))
    
    #Add an alternative annotation option for click activated versions of the hover label information
    #with the information plotted off to one of the sides of the figure.
    #I believe hoverLabels should be organized into y, x formatting for the indexing.
    on_click_annotations = []
    for j in range(data.shape[0]):
        for i in range(data.shape[1]):
            #the clicktoshow functionality appears to be (after some extensive testing) bugged and not
            #working properly when using paper and domain x and y references. The coordinate versions
            #work, but they refuse to place annotations at positions fully outside of the plotting area.
            #but, it looks like adding in a manual xshift value hacks around this limitation, since we *can*
            #place an annotation at the very edge of the plotable area, and then semi-manually shift it over.
            
            if hoverLabels is not None and hoverLabels[j][i]: #unpopulated squares should have the empty string as their hover labels, skip those.
                on_click_annotations.append(dict(x= data.shape[1], y= .5*data.shape[0],
                                            yanchor= 'middle', xanchor= 'left',
                                            text = hoverLabels[j][i], align= 'left',
                                            bordercolor= 'black', borderwidth= 1,
                                            clicktoshow= 'onout', xclick=i, yclick=j,
                                            xshift= 20,
                                            visible= False, font = dict(size=12, family='monospace'),
                                            showarrow=True))
                on_click_annotations.append(dict(x= i, y= j,
                                            yanchor= 'middle', xanchor= 'center',
                                            bordercolor= 'purple', borderwidth= 0,
                                            clicktoshow= 'onout',
                                            text='', #border pad value is a complete guess...
                                            visible= False, font = dict(size=12, family='monospace'),
                                            showarrow=True, arrowhead=3))
    #need to add these annotation to the layout here to have them properly work by default with the
    #button menu. (otherwise you need to toggle the button off and on again before they appear).
    fig.plotlyfig.update_layout(annotations = on_click_annotations)
    
    #create a pair of buttons for toggling on and off the inner grids:
    grid_button = dict(type="buttons",
                        active=1,
                        x= 0,
                        y= -0.075,
                        direction= 'right',
                        xanchor= 'right',
                        buttons=[dict(label="Grid On/Off", method="relayout", 
                                      args=  ["shapes", plaquette_boundary_lines+plaquette_grid_lines], 
                                      args2= ["shapes", plaquette_boundary_lines])])

    #create a pair of buttons for toggling on-click and hover labels on and off.
    #there is an interaction (maybe a bug? maybe just undocumented?) between hovermode and the clicktoshow behavior of
    #the annotations such that when hovermode if off the clicktoshow no longer does anything. To circumvent this
    #try toggling on the hoverlabels using the hoverinfo attribute instead.
    #The last of the entries in the list for args and args2 is a list of indices into the figures data attribute selecting
    #which traces to apply the update to (in our case the heatmap we care about is the second trace).
    hover_button = dict(type="buttons",
                        active=0,
                        x= 1,
                        y= -0.075,
                        direction= 'right',
                        xanchor= 'left',
                        buttons=[dict(label="Hover On/Off", method="restyle", args=['hoverinfo', 'text', [1]], args2= ['hoverinfo', 'none', [1]] )])

    click_button = dict(type="buttons",
                        active=0,
                        x= 1,
                        y= -0.19,
                        direction= 'right',
                        xanchor= 'left',
                        buttons=[dict(label="Click On/Off", method="relayout", args=['annotations', on_click_annotations], args2=['annotations', []] )])

    fig.plotlyfig.update_layout(updatemenus=[grid_button, hover_button, click_button])
    
    if return_hover_labels:
        return fig, hoverLabels
    else:
        return fig


def _summable_color_boxplot(sub_mxs, xlabels, ylabels, xlabel, ylabel,
                            colormap, colorbar=False, box_labels=True, prec=0, hover_info=True,
                            sum_up=False, scale=1.0, bgcolor='white'):
    """
    A helper function for generating typical nested color box plots used in pyGSTi.

    Given the list-of-lists, `sub_mxs`, along with x and y labels for both the "outer"
    (i.e. the list-indices), this function will produce a nested color box plot with
    the option of summing over the sub-matrix elements (the "inner" axes).

    Parameters
    ----------
    sub_mxs : list
        A list of lists of 2D numpy.ndarrays.  sub_mxs[iy][ix] specifies the matrix of values
        or sum (if sum_up == True) displayed in iy-th row and ix-th column of the plot.  NaNs
        indicate elements should not be displayed.

    xlabels : list
        Labels for the outer x-axis values.

    ylabels : list
        Labels for the outer y-axis values.

    xlabel : str
        Outer x-axis label.

    ylabel : str
        Outer y-axis label.

    colormap : Colormap
        The colormap used to determine box color.

    colorbar : bool, optional
        Whether or not to show the color scale bar.

    box_labels : bool, optional
        Whether to display static value-labels over each box.

    prec : int or {'compact','compacthp'}, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    hover_info : bool or function, optional
        If a boolean, indicates whether to include interactive hover labels. If
        a function, then must take arguments `(val, i, j, ii, jj)` if
        `sum_up == False` or `(val, i, j)` if `sum_up == True` and return a
        label string, where `val` is the box value, `j` and `i` index
        `xlabels` and `ylabels`, and `ii` and `jj` index the row and column index
        of the sub-matrix element the label is for.

    sum_up : bool, optional
        False displays each matrix element as it's own color box
        True sums the elements of each (x,y) matrix and displays
        a single color box for the sum.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    bgcolor : str, optional
        Background color for this plot.  Can be common color names, e.g.
        `"black"`, or string RGB values, e.g. `"rgb(255,128,0)"`.

    Returns
    -------
    plotly.Figure
    """
    nYs = len(sub_mxs)
    nXs = len(sub_mxs[0]) if nYs > 0 else 0

    nIYs = nIXs = 0
    for ix in range(nXs):
        for iy in range(nYs):
            if sub_mxs[iy][ix] is not None:
                nIYs, nIXs = sub_mxs[iy][ix].shape; break

    # flip so [0,0] el of original sub_mxs is at *top*-left (FLIP)
    sub_mxs = [[_np.flipud(subMx) for subMx in row] for row in sub_mxs]
    #inner_ylabels = list(reversed(inner_ylabels))

    #FUTURE: to restore "invert" functionality, make PlaquetteGridCircuitStructure invertible
    #if invert:
    #    if sum_up:
    #        _warnings.warn("Cannot invert a summed-up plot.  Ignoring invert=True.")
    #    else:
    #        invertedSubMxs = []  # will be indexed as invertedSubMxs[inner-y][inner-x]
    #        for iny in range(nIYs):
    #            invertedSubMxs.append([])
    #            for inx in range(nIXs):
    #                mx = _np.array([[sub_mxs[iy][ix][iny, inx] for ix in range(nXs)]
    #                                for iy in range(nYs)], 'd')
    #                invertedSubMxs[-1].append(mx)
    #
    #        # flip the now-inverted mxs to counteract the flip that will occur upon
    #        # entering _summable_color_boxplot again (with invert=False this time), since we
    #        # *don't* want the now-inner dimension (the germs) actually flipped (FLIP)
    #        invertedSubMxs = [[_np.flipud(subMx) for subMx in row] for row in invertedSubMxs]
    #        ylabels = list(reversed(ylabels))
    #
    #        return _summable_color_boxplot(invertedSubMxs,
    #                                       inner_xlabels, inner_ylabels,
    #                                xlabels, ylabels, inner_xlabel, inner_ylabel, xlabel, ylabel,
    #                                colormap, colorbar, box_labels, prec, hover_info,
    #                                sum_up, False, scale, bgcolor)

    def val_filter(vals):
        """filter to latex-ify circuits.  Later add filter as a possible parameter"""
        formatted_vals = []
        for val in vals:
            if isinstance(val, _Circuit):
                if len(val) == 0:
                    #formatted_vals.append(r"$\{\}$")
                    formatted_vals.append(r"{}")
                else:
                    #formatted_vals.append( "$" + "\\cdot".join([("\\mathrm{%s}" % el) for el in val]) + "$" )
                    formatted_vals.append(val.str)
            else:
                formatted_vals.append(str(val))
        return formatted_vals

    def sum_up_mx(mx):
        """ Sum up `mx` in a NAN-ignoring way """
        flat_mx = mx.ravel()
        if _np.any(_np.isnan(flat_mx)):
            if _np.all(_np.isnan(flat_mx)):
                return _np.nan
            # replace NaNs with zeros for purpose of summing (when there's at least one non-NaN)
            return sum(_np.nan_to_num(flat_mx))
        else:
            return sum(flat_mx)

    #Setup and create plotting functions
    if sum_up:
        subMxSums = _np.array([[sum_up_mx(sub_mxs[iy][ix]) for ix in range(nXs)] for iy in range(nYs)], 'd')

        if hover_info is True:
            def hover_label_fn(val, i, j):
                """ Standard hover labels """
                if _np.isnan(val): return ""
                return "%s: %s<br>%s: %s<br>%g" % \
                    (xlabel, str(xlabels[j]), ylabel, str(ylabels[i]), val)
        elif callable(hover_info):
            hover_label_fn = hover_info
        else: hover_label_fn = None

        boxLabelSize = 8 * scale if box_labels else 0
        fig, hover_labels = _color_boxplot(subMxSums, colormap, colorbar, boxLabelSize,
                             prec, hover_label_fn, return_hover_labels=True)
        #update tickvals b/c _color_boxplot doesn't do this (unlike _nested_color_boxplot)
        if fig is not None:
            fig.plotlyfig['layout']['xaxis'].update(tickvals=list(range(nXs)))
            fig.plotlyfig['layout']['yaxis'].update(tickvals=list(range(nYs)))

        xBoxes = nXs
        yBoxes = nYs
        
    else:  # not summing up

        if hover_info is True:
            def hover_label_fn(val, i, j, ii, jj):
                """ Standard hover labels """
                if _np.isnan(val): return ""
                return "%s: %s<br>%s: %s<br>%s: %s<br>%s: %s<br>%g" % \
                    (xlabel, str(xlabels[j]), ylabel, str(ylabels[i]),
                     "row", str(ii), "column", str(jj), val)
        elif callable(hover_info):
            def hover_label_fn(val, i, j, ii, jj):
                N = len(sub_mxs[i][j])  # number of rows in submatrix
                return hover_info(val, i, j, N - 1 - ii, jj)  # FLIP row index
        else: hover_label_fn = None

        boxLabelSize = 8 if box_labels else 0  # do not scale (OLD: 8*scale)
        fig, hover_labels = _nested_color_boxplot(sub_mxs, colormap, colorbar, boxLabelSize,
                                    prec, hover_label_fn, return_hover_labels=True)

        xBoxes = nXs * (nIXs + 1) - 1
        yBoxes = nYs * (nIYs + 1) - 1
    
    if fig is not None:  # i.e., if there was data to plot
        pfig = fig.plotlyfig
        if xlabel: pfig['layout']['xaxis'].update(title=dict(text=xlabel,
                                                  font={'size': 12 * scale, 'color': "black"}))
        if ylabel: pfig['layout']['yaxis'].update(title=dict(text=ylabel,
                                                  font={'size': 12 * scale, 'color': "black"}))
        if xlabels:
            pfig['layout']['xaxis'].update(tickmode="array",
                                           ticktext=val_filter(xlabels),
                                           tickfont={'size': 10 * scale, 'color': "black"})
        if ylabels:
            pfig['layout']['yaxis'].update(tickmode="array",
                                           ticktext=val_filter(ylabels),
                                           tickfont={'size': 10 * scale, 'color': "black"})

        #Set plot size and margins
        lmargin = rmargin = tmargin = bmargin = 20
        if xlabel: bmargin += 30
        if ylabel: lmargin += 30
        if xlabels:
            max_xl = max([len(xl) for xl in pfig['layout']['xaxis']['ticktext']])
            if max_xl > 0: bmargin += max_xl * 5
        if ylabels:
            max_yl = max([len(yl) for yl in pfig['layout']['yaxis']['ticktext']])
            if max_yl > 0: lmargin += max_yl * 5
        if colorbar: rmargin = 100

        #make sure there's enough margin for hover tooltips
        if 10 * xBoxes < 200: rmargin = max(200 - 10 * xBoxes, rmargin)
        if 10 * yBoxes < 200: bmargin = max(200 - 10 * xBoxes, bmargin)

        #We also need to add additional margin on the right to account for
        #on-click text annotations. Loop through hover_labels and use PIL to
        #estimate the rendered width of the text. It is likely that the longest strings
        #are in the final few columns of the color box plot, so only check those
        #to same some computation (the rendering for size checking takes non-trivial time).
        hover_label_widths = []
        #create a PIL ImageFont object which will be used as a helper for estimating the
        #rendered width of the hover labels/annotations below.
        font_path = str(Path(__file__).parent / 'fonts'/ 'NotoSansMono-Regular.ttf')
        font = _ImageFont.truetype(font_path, 12)
        if hover_labels: #if not None or empty list:
            for j, label_row in enumerate(hover_labels): 
                for i in range(len(label_row)-nIXs, len(label_row)):
                    #nIXs is the width of the plaquette
                    #split the label using the linebreaks
                    split_label= hover_labels[j][i].split('<br>')
                    #loop through elements of split label and get the
                    #widths of each substring using PIL. Add these to a running
                    #list.
                    hover_label_widths.extend([font.getlength(substring) for substring in split_label])
            #Now get the maximum width.
            max_annotation_width = max(hover_label_widths)
        else:
            max_annotation_width = 0

        width = lmargin + 10 * xBoxes + rmargin + max_annotation_width
        rmargin +=max_annotation_width
        #manually add in some additional bottom margin for the new toggle buttons for controlling
        #display
        button_wiggle_factor = 50
        bmargin += button_wiggle_factor
        height = tmargin + 10 * yBoxes + bmargin
        
        width *= scale
        height *= scale
        lmargin *= scale
        rmargin *= scale
        tmargin *= scale
        bmargin *= scale

        pfig['layout'].update(width=width,
                              height=height,
                              margin=go_margin(l=lmargin, r=rmargin, b=bmargin, t=tmargin),
                              plot_bgcolor=bgcolor)
        
        #it is only at this point that we have a height attribute officially set. We need to use this to tweak the placement
        #of the toggleable control buttons, as those only have the option to set their positions in normalized coordinates,
        #so the spacing gets all messed up as the height of the figure gets taller.

        #decide on some absolute distances (in pixels?) between the buttons and the bottom of the printable area.
        y_abs_0= 25
        y_abs_1= 75
        #vertical plotting area should be (approximately at least) height - tmargin - bmargin
        plottable_height = height - tmargin - bmargin
        new_y_0 = -y_abs_0/plottable_height
        new_y_1 = -y_abs_1/plottable_height
        #Now let's update the updatemenus
        #updatemenus should be a tuple, but for some reason this looks like
        #it works...   
        if pfig['layout']['updatemenus']:     
            pfig['layout']['updatemenus'][0]['y'] = new_y_0
            pfig['layout']['updatemenus'][1]['y'] = new_y_0
            pfig['layout']['updatemenus'][2]['y'] = new_y_1
        
    else:  # fig is None => use a "No data to display" placeholder figure
        trace = go.Heatmap(z=_np.zeros((10, 10), 'd'),
                           colorscale=[[0, 'white'], [1, 'black']],
                           showscale=False, zmin=0, zmax=1, hoverinfo='none')
        layout = go.Layout(
            width=100, height=100,
            annotations=[go_annotation(x=5, y=5, text="NO DATA", showarrow=False,
                                       font={'size': 20, 'color': "black"},
                                       xref='x', yref='y')],
            xaxis=dict(showline=False, zeroline=False,
                       showticklabels=False, showgrid=False,
                       ticks=""),
            yaxis=dict(showline=False, zeroline=False,
                       showticklabels=False, showgrid=False,
                       ticks="")
        )
        fig = ReportFigure(go.Figure(data=[trace], layout=layout),
                           None, "No data!")

    return fig


def _create_hover_info_fn(circuit_structure, xvals, yvals, sum_up, addl_hover_submxs):
    if sum_up:
        def hover_label_fn(val, iy, ix):
            """ Standard hover labels """
            if _np.isnan(val): return ""
            plaq = circuit_structure.plaquette(xvals[ix], yvals[iy], empty_if_missing=True)
            txt = plaq.summary_label()
            txt += "<br>value: %g" % val
            for lbl, addl_subMxs in addl_hover_submxs.items():
                txt += "<br>%s: %s" % (lbl, str(addl_subMxs[iy][ix]))
            return txt

    else:
        def hover_label_fn(val, iy, ix, iiy, iix):
            """ Standard hover labels """
            #Note: in this case, we need to "flip" the iiy index because
            # the matrices being plotted are flipped within _summable_color_boxplot(...)
            if _np.isnan(val): return ""
            plaq = circuit_structure.plaquette(xvals[ix], yvals[iy], empty_if_missing=True)
            txt = plaq.element_label(iiy, iix)  # note: *row* index = iiy
            txt += (f"<br>val: {val:.3g}")
            for lbl, addl_subMxs in addl_hover_submxs.items():
                txt += f"<br>{lbl:<9}: {addl_subMxs[iy][ix][iiy][iix]}"
            return txt
    return hover_label_fn
    
def _create_hover_info_fn_circuit_list(circuit_structure, sum_up, addl_hover_submxs):
    
    if sum_up:
        pass
    else:
        if isinstance(circuit_structure, _CircuitList):
            def hover_label_fn(val, i):
                """ Standard hover labels """
                #Note: in this case, we need to "flip" the iiy index because
                # the matrices being plotted are flipped within _summable_color_boxplot(...)
                if _np.isnan(val): return ""
                ckt = circuit_structure[i].copy(editable=True)
                ckt.factorize_repetitions_inplace()
                txt = ckt.layerstr # note: *row* index = iiy
                txt += ("<br>value: %g" % val)
                for lbl, addl_subMxs in addl_hover_submxs.items():
                    txt += "<br>%s: %s" % (lbl, str(addl_subMxs[i]))
                return txt

        elif isinstance(circuit_structure, list) and all([isinstance(el, _CircuitList) for el in circuit_structure]):
            def hover_label_fn(val, i, j):
                """ Standard hover labels """
                #Note: in this case, we need to "flip" the iiy index because
                # the matrices being plotted are flipped within _summable_color_boxplot(...)
                if _np.isnan(val): return ""
                ckt = circuit_structure[i][j].copy(editable=True)
                ckt.factorize_repetitions_inplace()
                txt = ckt.layerstr # note: *row* index = iiy
                txt += ("<br>value: %g" % val)
                for lbl, addl_subMxs in addl_hover_submxs.items():
                    txt += "<br>%s: %s" % (lbl, str(addl_subMxs[i][j]))
                return txt
    return hover_label_fn


def _circuit_color_boxplot(circuit_structure, sub_mxs, colormap,
                           colorbar=False, box_labels=True, prec='compact', hover_info=True,
                           sum_up=False, invert=False, scale=1.0, bgcolor="white", addl_hover_submxs=None):
    """
    A wrapper around :func:`_summable_color_boxplot` for creating color box plots displaying circuits.

    Generates a plot from the structure of the circuits as contained in a
    `CircuitStructure` object.

    Parameters
    ----------
    circuit_structure : CircuitStructure
        Specifies a set of circuits along with their structure, e.g. fiducials, germs,
        and maximum lengths.

    sub_mxs : list
        A list of lists of 2D numpy.ndarrays.  sub_mxs[iy][ix] specifies the matrix of values
        or sum (if sum_up == True) displayed in iy-th row and ix-th column of the plot.  NaNs
        indicate elements should not be displayed.

    colormap : Colormap
        The colormap used to determine box color.

    colorbar : bool, optional
        Whether or not to show the color scale bar.

    box_labels : bool, optional
        Whether to display static value-labels over each box.

    prec : int or {'compact','compacthp'}, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    hover_info : bool, optional
        Whether to incude interactive hover labels.

    sum_up : bool, optional
        False displays each matrix element as it's own color box
        True sums the elements of each (x,y) matrix and displays
        a single color box for the sum.

    invert : bool, optional
        If True, invert the nesting order of the nested color box plot
        (applicable only when sum_up == False).  E.g. use inner_x_labels and
        inner_y_labels to label the x and y axes.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    bgcolor : str, optional
        Background color for this plot.  Can be common color names, e.g.
        `"black"`, or string RGB values, e.g. `"rgb(255,128,0)"`.

    addl_hover_submxs : dict, optional
        If not None, a dictionary whose values are lists-of-lists in the same
        format as `sub_mxs` which specify additional values to add to the
        hover-info of the corresponding boxes.  The keys of this dictionary
        are used as labels within the hover-info text.

    Returns
    -------
    plotly.Figure
    """
    xvals = circuit_structure.used_xs
    yvals = circuit_structure.used_ys

    if addl_hover_submxs is None:
        addl_hover_submxs = {}

    # Note: invert == True case not handled yet
    assert(invert is False), "`invert=True` is no longer supported."

    if hover_info:
        hover_info = _create_hover_info_fn(circuit_structure, xvals, yvals, sum_up, addl_hover_submxs)

    return _summable_color_boxplot(sub_mxs, circuit_structure.used_xs, circuit_structure.used_ys,
                                   circuit_structure.xlabel, circuit_structure.ylabel, colormap, colorbar,
                                   box_labels, prec, hover_info, sum_up, scale, bgcolor)


def _circuit_color_scatterplot(circuit_structure, sub_mxs, colormap,
                               colorbar=False, hover_info=True, sum_up=False,
                               ylabel="", scale=1.0, addl_hover_submxs=None):
    """
    Similar to :func:`_circuit_color_boxplot` except a scatter plot is created.

    Parameters
    ----------
    circuit_structure : PlaquetteGridCircuitStructure
        Specifies a set of circuits along with their outer and inner x,y
        structure, e.g. fiducials, germs, and maximum lengths.

    sub_mxs : list
        A list of lists of 2D numpy.ndarrays.  sub_mxs[iy][ix] specifies the matrix of values
        or sum (if sum_up == True) displayed in iy-th row and ix-th column of the plot.  NaNs
        indicate elements should not be displayed.

    colormap : Colormap
        The colormap used to determine box color.

    colorbar : bool, optional
        Whether or not to show the color scale bar.

    hover_info : bool, optional
        Whether to incude interactive hover labels.

    sum_up : bool, optional
        False displays each matrix element as it's own color box
        True sums the elements of each (x,y) matrix and displays
        a single color box for the sum.

    ylabel : str, optional
        The y-axis label to use.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    addl_hover_submxs : dict, optional
        If not None, a dictionary whose values are lists-of-lists in the same
        format as `sub_mxs` which specify additional values to add to the
        hover-info of the corresponding boxes.  The keys of this dictionary
        are used as labels within the hover-info text.

    Returns
    -------
    plotly.Figure
    """
    g = circuit_structure

    if addl_hover_submxs is None:
        addl_hover_submxs = {}

    if hover_info:
        if isinstance(g, _PlaquetteGridCircuitStructure):
            hover_info = _create_hover_info_fn(circuit_structure, g.used_xs, g.used_ys, sum_up, addl_hover_submxs)
        elif isinstance(g, _CircuitList) or (isinstance(g, list) and all([isinstance(el, _CircuitList) for el in g])):
            hover_info = _create_hover_info_fn_circuit_list(circuit_structure, sum_up, addl_hover_submxs)
        
    xs = []; ys = []; texts = []
    gstrs = set()  # to eliminate duplicate strings
    
    if isinstance(g, _PlaquetteGridCircuitStructure):
        for ix, x in enumerate(g.used_xs):
            for iy, y in enumerate(g.used_ys):
                plaq = g.plaquette(x, y, empty_if_missing=True)
                if sum_up:
                    if plaq.base not in gstrs:
                        tot = sum([sub_mxs[iy][ix][iiy][iix] for iiy, iix, _ in plaq])
                        xs.append(len(plaq.base))  # x-coord is len of *base* string
                        ys.append(tot)
                        gstrs.add(plaq.base)
                        if hover_info:
                            if callable(hover_info):
                                texts.append(hover_info(tot, iy, ix))
                            else:
                                texts.append(str(tot))
                else:
                    for iiy, iix, opstr in plaq:
                        if opstr in gstrs: continue  # skip duplicates
                        xs.append(len(opstr))
                        ys.append(sub_mxs[iy][ix][iiy][iix])
                        gstrs.add(opstr)
                        if hover_info:
                            if callable(hover_info):
                                texts.append(hover_info(sub_mxs[iy][ix][iiy][iix], iy, ix, iiy, iix))
                            else:
                                texts.append(str(sub_mxs[iy][ix][iiy][iix]))
    elif isinstance(g, _CircuitList):
        for i, ckt in enumerate(g):
            if ckt in gstrs:
                continue
            else:
                if sum_up:
                    pass
                    #TODO: Implement sum_up behavior mirroring that above.
                gstrs.add(ckt)
                ys.append(sub_mxs[i])
                xs.append(len(ckt))
                if hover_info:
                    if callable(hover_info):
                        texts.append(hover_info(sub_mxs[i], i))
                    else:
                        texts.append(str(sub_mxs[i]))
    elif isinstance(g, list) and all([isinstance(el, _CircuitList) for el in g]):
        for i, circuit_list in enumerate(g):
            for j, ckt in enumerate(circuit_list):
                if ckt in gstrs:
                    continue
                else:
                    if sum_up:
                        pass
                        #TODO: Implement sum_up behavior mirroring that above.
                    gstrs.add(ckt)
                    ys.append(sub_mxs[i][j])
                    xs.append(len(ckt))
                    if hover_info:
                        if callable(hover_info):
                            texts.append(hover_info(sub_mxs[i][j], i, j))
                        else:
                            texts.append(str(sub_mxs[i][j]))

    trace = go.Scatter(x=xs, y=ys, mode="markers",
                       marker=dict(size=8,
                                   color=[colormap.interpolate_color(y) for y in ys],
                                   colorscale=colormap.create_plotly_colorscale(),
                                   line=dict(width=1)))

    if hover_info:
        trace['hoverinfo'] = 'text'
        trace['text'] = texts
    else:
        trace['hoverinfo'] = 'none'

    xaxis = go_x_axis(
        title={'text': 'sequence length'},
        showline=False,
        zeroline=True,
    )
    yaxis = go_y_axis(
        title={'text': ylabel}
    )

    layout = go.Layout(
        #title="Sum = %.2f" % sum(ys), #DEBUG
        width=400 * scale,
        height=400 * scale,
        hovermode='closest',
        xaxis=xaxis,
        yaxis=yaxis,
        hoverlabel= dict(font_family = 'monospace', bgcolor='#dbd9d9') #add hoverlabel formatting
    )
    return ReportFigure(go.Figure(data=[trace], layout=layout), colormap,
                        {'x': xs, 'y': ys})


def _circuit_color_histogram(circuit_structure, sub_mxs, colormap,
                             ylabel="", scale=1.0, include_chi2=True):
    """
    Similar to :func:`_circuit_color_boxplot` except a histogram is created.

    Parameters
    ----------
    circuit_structure : PlaquetteGridCircuitStructure
        Specifies a set of circuits along with their outer and inner x,y
        structure, e.g. fiducials, germs, and maximum lengths.

    sub_mxs : list
        A list of lists of 2D numpy.ndarrays.  sub_mxs[iy][ix] specifies the matrix of values
        or sum (if sum_up == True) displayed in iy-th row and ix-th column of the plot.  NaNs
        indicate elements should not be displayed.

    colormap : Colormap
        The colormap used to determine box color.

    ylabel : str, optional
        The y-axis label to use.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    include_chi2 : bool, optional (default True)
        If True then include a trace corresponding to a chi^2
        distributions with a number of degrees of freedom inferred
        from the input data.

    Returns
    -------
    plotly.Figure
    """
    g = circuit_structure
        
    #For all of the fanciness below, this all essentially looks like it just produces
    #a flattened list of all of the contents of sub_mxs, so we can still do that with the
    #submx structures we get from using CircuitList objects.
    ys = []  # artificially add minval so
    gstrs = set()  # to eliminate duplicate strings
    
    if isinstance(g, _PlaquetteGridCircuitStructure):
        for ix, x in enumerate(g.used_xs):
            for iy, y in enumerate(g.used_ys):
                plaq = g.plaquette(x, y, empty_if_missing=True)
                #TODO: if sum_up then need to sum before appending...
                for iiy, iix, opstr in plaq:
                    if opstr in gstrs: continue  # skip duplicates
                    ys.append(sub_mxs[iy][ix][iiy][iix])
                    gstrs.add(opstr)
    
    elif isinstance(g, _CircuitList):
        for i, ckt in enumerate(g):
            if ckt in gstrs:
                continue
            else:
                gstrs.add(ckt)
                ys.append(sub_mxs[i])
    
    elif isinstance(g, list) and all([isinstance(el, _CircuitList) for el in g]):
        for i, circuit_list in enumerate(g):
            for j, ckt in enumerate(circuit_list):
                if ckt in gstrs:
                    continue
                else:
                    gstrs.add(ckt)
                    ys.append(sub_mxs[i][j])
    else:
        raise ValueError('Can only handle PlaquetteGridCircuitStructure, CircuitList or lists of CircuitList objects at present.')
    
    if len(ys) == 0: ys = [0]  # case of no data - dummy so max works below

    minval = 0
    maxval = max(minval + 1e-3, _np.max(ys))  # don't let minval==maxval
    nvals = len(ys)
    cumulative = dict(enabled=False)
    nbins = 50
    binsize = (maxval - minval) / (nbins - 1)
    bincenters = _np.linspace(minval, maxval, nbins)
    bindelta = (maxval - minval) / (nbins - 1)  # spacing between bin centers


    trace = go.Histogram(
        x=[bincenters[0]] + ys,  # artificially add 1 count in lowest bin so plotly anchors histogram properly
        autobinx=False,
        xbins=dict(
            start=minval - binsize / 2.0,
            end=maxval + binsize / 2.0,
            size=binsize
        ),
        name="count",
        marker=dict(
            color=[colormap.interpolate_color(t) for t in bincenters],
            line=dict(
                color='black',
                width=1.0,
            )
        ),
        cumulative=cumulative,
        opacity=1.00,
        showlegend=False,
    )

    if include_chi2:
        dof = colormap.dof if hasattr(colormap, "dof") else 1
        line_trace = go.Scatter(
            x=bincenters,
            y=[nvals * bindelta * _chi2.pdf(xval, dof) for xval in bincenters],
            name="expected",
            showlegend=False,
            line=dict(
                color=('rgb(0, 0, 0)'),
                width=1)  # dash = 'dash') # dash options include 'dash', 'dot', and 'dashdot'
        )

    hist_values, np_bins = _np.histogram(ys, nbins, range=(minval - binsize / 2.0,
                                                           maxval + binsize / 2.0))
    if len(hist_values) > 0 and len(hist_values[hist_values > 0]) > 0:
        minlog = _np.log10(max(_np.min(hist_values[hist_values > 0]) / 10.0, 1e-3))
        maxlog = _np.log10(1.5 * _np.max(hist_values))
    else:
        minlog, maxlog = -3, 0  # defaults to (1e-3,1) when there's no data

    layout = go.Layout(
        #title="Sum = %.2f" % sum(ys), #DEBUG
        width=500 * scale,
        height=350 * scale,
        font=dict(size=10),
        xaxis=dict(
            title={'text': ylabel},  # b/c "y-values" are along x-axis in histogram
            showline=True
        ),
        yaxis=dict(
            type='log',
            #tickformat='g',
            exponentformat='power',
            showline=True,
            range=[minlog, maxlog]
        ),
        bargap=0,
        bargroupgap=0,
        legend=dict(orientation="h")
    )

    pythonVal = {'histogram values': ys}
    return ReportFigure(go.Figure(data=[trace, line_trace] if include_chi2 else [trace], layout=layout),
                        colormap, pythonVal)


def _opmatrix_color_boxplot(op_matrix, color_min, color_max, mx_basis_x=None, mx_basis_y=None,
                            xlabel=None, ylabel=None,
                            box_labels=False, colorbar=None, prec=0, scale=1.0,
                            eb_matrix=None, title=None):
    """
    Creates a color box plot for visualizing a single matrix.

    Parameters
    ----------
    op_matrix : numpy array
        The matrix to visualize.

    color_min : float
        Color scale minimum.

    color_max : float
        Color scale maximum.

    mx_basis_x : str or Basis, optional
        The name abbreviation for the basis or a Basis object. Used to label the
        columns (x-ticklabels).  Typically in {"pp","gm","std","qt"}.  
        If you don't want labels, leave as None.

    mx_basis_y : str or Basis, optional
        Same as `mx_basis_x` but for just the y-ticklabels.

    xlabel : str, optional
        X-axis label of the plot.

    ylabel : str, optional
        Y-axis label of the plot.

    box_labels : bool, optional
        Whether box labels are displayed.

    colorbar : bool optional
        Whether to display a color bar to the right of the box plot.  If None,
        then a colorbar is displayed when `box_labels == False`.

    prec : int or {'compact','compacthp'}, optional
        Precision for box labels.  Only relevant when box_labels == True. Allowed
        values are:

        - 'compact' = round to nearest whole number using at most 3 characters
        - 'compacthp' = show as much precision as possible using at most 3 characters
        - int >= 0 = fixed precision given by int
        - int <  0 = number of significant figures given by -int

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    eb_matrix : numpy array, optional
        An array, of the same size as `op_matrix`, which gives error bars to be
        be displayed in the hover info.

    title : str, optional
        A title for the plot

    Returns
    -------
    plotly.Figure
    """

    if isinstance(mx_basis_x, str):
        mx_basis_x = _baseobjs.BuiltinBasis(mx_basis_x, op_matrix.shape[1])
    if isinstance(mx_basis_y, str):
        mx_basis_y = _baseobjs.BuiltinBasis(mx_basis_y, op_matrix.shape[0])
    
    if mx_basis_x is not None:
        if len(mx_basis_x.labels) - 1 == op_matrix.shape[1]:
            xlabels = [("<i>%s</i>" % x) if len(x) else "" for x in mx_basis_x.labels[1:]]
        else:
            xlabels = [("<i>%s</i>" % x) if len(x) else "" for x in mx_basis_x.labels]
    else:
        xlabels = None

    if mx_basis_y is not None:
        if len(mx_basis_y.labels) - 1 == op_matrix.shape[0]:
            ylabels = [("<i>%s</i>" % x) if len(x) else "" for x in mx_basis_y.labels[1:]]
        else:
            ylabels = [("<i>%s</i>" % x) if len(x) else "" for x in mx_basis_y.labels]
    else:
        ylabels=None

    colormap = _colormaps.DivergingColormap(vmin=color_min, vmax=color_max)
    thickLineInterval = 4 if (mx_basis_x is not None and mx_basis_x.name == "pp") \
        else None  # TODO: separate X and Y thick lines?
    return _matrix_color_boxplot(op_matrix, xlabels, ylabels,
                                 xlabel, ylabel, box_labels, thickLineInterval,
                                 colorbar, colormap, prec, scale,
                                 eb_matrix, title)


def _opmatrices_color_boxplot(op_matrices, color_min, color_max, mx_basis_x=None, mx_basis_y=None,
                              xlabel=None, ylabel=None,
                              box_labels=False, colorbar=None, prec=0, scale=1.0,
                              eb_matrices=None, title=None, arrangement='row', subtitles= None):
    """
    Creates a color box plot for visualizing a single matrix.

    Parameters
    ----------
    op_matrices : list of numpy arrays
        The matrices to visualize. Note that there is presently an implicit
        assumption that these are all the same shape.

    color_min : float
        Color scale minimum.

    color_max : float
        Color scale maximum.

    mx_basis : str or Basis, optional
        The name abbreviation for the basis or a Basis object. Used to label the
        columns & rows (x- and y-ticklabels).  Typically in
        {"pp","gm","std","qt"}.  If you don't want labels, leave as None.

    mx_basis_y : str or Basis, optional
        Same as `mx_basis` but for just the y-ticklabels, overriding `mx_basis` and
        allowing the y-ticklabels to be different.

    xlabel : str, optional
        X-axis label of the plot.

    ylabel : str, optional
        Y-axis label of the plot.

    box_labels : bool, optional
        Whether box labels are displayed.

    colorbar : bool optional
        Whether to display a color bar to the right of the box plot.  If None,
        then a colorbar is displayed when `box_labels == False`.

    prec : int or {'compact','compacthp'}, optional
        Precision for box labels.  Only relevant when box_labels == True. Allowed
        values are:

        - 'compact' = round to nearest whole number using at most 3 characters
        - 'compacthp' = show as much precision as possible using at most 3 characters
        - int >= 0 = fixed precision given by int
        - int <  0 = number of significant figures given by -int

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    eb_matrices : list of numpy arrays, optional
        List of arrays, of the same size as elements of `op_matrices`, which gives 
        error bars to be be displayed in the hover info.

    title : str, optional
        A title for the plot
    
    arrangement : string or tuple, optional (default 'row')
        Arrangement of the subplots to use in constructing composite
        figure. Supported string options are 'row' and 'col', which will
        arrange the subplots into a single row or column. Also supported is
        a tuple corresponding to the shape of the grid to arrange the subplots
        into. When using a tuple to specify a grid that grid is filled rowwise
        from left to right.

    subtitles : list of str, optional (default None)
        A list of strings for the sub-titles on each matrix's subplot.

    Returns
    -------
    plotly.Figure
    """

    #Initialize the subplot
    if arrangement == 'row':
        num_rows = 1
        num_cols = len(op_matrices)
    elif arrangement == 'col':
        num_rows = len(op_matrices)
        num_cols = 1
    elif isinstance(arrangement, tuple):
        num_rows = arrangement[0]
        num_cols= arrangement[1]
    else:
        raise ValueError(f'Unrecognized argument for arrangement: {arrangement}')
    #plotly is 1-indexed for subplots.
    row_col_indices = list(_product(range(1,num_rows+1), range(1,num_cols+1)))
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=subtitles)

    #Need to bump up the height of the subtitles
    #if the annotation is one of the subtitles we need to bump up
    #the y value a bit.
    if subtitles is not None:
        for annotation in fig.layout.annotations:
            if annotation['text'] in subtitles:
                annotation['y']+=0.06

    if isinstance(mx_basis_x, str):
        mx_basis_x = _baseobjs.BuiltinBasis(mx_basis_x, op_matrices[0].shape[1])
    if isinstance(mx_basis_y, str):
        mx_basis_y = _baseobjs.BuiltinBasis(mx_basis_y, op_matrices[0].shape[0])
    
    if mx_basis_x is not None:
        if len(mx_basis_x.labels) - 1 == op_matrices[0].shape[1]:
            xlabels = [("<i>%s</i>" % x) if len(x) else "" for x in mx_basis_x.labels[1:]]
        else:
            xlabels = [("<i>%s</i>" % x) if len(x) else "" for x in mx_basis_x.labels]
    else:
        xlabels = None

    if mx_basis_y is not None:
        if len(mx_basis_y.labels) - 1 == op_matrices[0].shape[0]:
            ylabels = [("<i>%s</i>" % x) if len(x) else "" for x in mx_basis_y.labels[1:]]
        else:
            ylabels = [("<i>%s</i>" % x) if len(x) else "" for x in mx_basis_y.labels]
    else:
        ylabels = None

    colormap = _colormaps.DivergingColormap(vmin=color_min, vmax=color_max)
    thickLineInterval = 4 if (mx_basis_x is not None and mx_basis_x.name == "pp") \
        else None  # TODO: separate X and Y thick lines?
    
    if eb_matrices is None:
        eb_matrices = [None]*len(op_matrices)

    #loop through generation of each subplot.
    op_matrix_plots = [[] for _ in range(num_rows)]
    for matrix, (row_idx, col_idx), eb_matrix in zip(op_matrices, row_col_indices, eb_matrices):
        op_matrix_plots[row_idx-1].append(_matrix_color_boxplot(matrix, xlabels, ylabels,
                                                                xlabel, ylabel, box_labels, thickLineInterval,
                                                                colorbar, colormap, prec, scale,
                                                                eb_matrix, None))
    remapped_annotations = []
    remapped_shapes = []
    #add in the existing annotations in fig to start (should at minimum include "Real" and "Imag" headings).
    remapped_annotations.extend(fig.layout['annotations'])
    remapped_shapes.extend(fig.layout['shapes'])
    for i, row in enumerate(op_matrix_plots):
        for j, matrix in enumerate(row):
            #add the first trace (data[0]) to figure
            fig.add_trace(matrix.plotlyfig.data[0], row=i+1, col=j+1)
            #set the subfigures axes layout to match the parents
            matrix_dict_layout= matrix.plotlyfig.to_dict()['layout']
            fig.update_xaxes(row=i+1, col=j+1, **matrix_dict_layout['xaxis'])
            fig.update_yaxes(row=i+1, col=j+1, **matrix_dict_layout['yaxis'])
            #add the shapes from the subplot to the main figure but update
            #the axis references to point to the correct subplot.
            flattened_idx = num_rows*i+ (j+1)
            matrix_shapes = matrix_dict_layout['shapes']
            for shape in matrix_shapes:
                shape['xref'] = f'x{flattened_idx}'
                shape['yref'] = f'y{flattened_idx}'
                remapped_shapes.append(shape)
            #do this same remapping with the annotations
            matrix_annotations = matrix_dict_layout['annotations']
            for annotation in matrix_annotations:
                annotation['xref'] = f'x{flattened_idx}'
                annotation['yref'] = f'y{flattened_idx}'
                remapped_annotations.append(annotation)
    fig.update_layout(annotations=remapped_annotations, shapes=remapped_shapes)

    #set the width of the composite figure to be equal to the width of the widest row.
    row_widths = [[] for _ in range(num_rows)]
    for i, row in enumerate(op_matrix_plots):
        for j, matrix in enumerate(row):
            row_widths[i].append(matrix.plotlyfig.layout.width)
    total_row_widths = [sum(row) for row in row_widths]
    max_total_row_width = max(total_row_widths)
    width = max_total_row_width*scale

    #Next we need to set the height in an analogous fashion.
    col_heights = [[] for _ in range(num_cols)]
    for i, row in enumerate(op_matrix_plots):
        for j, matrix in enumerate(row):
            col_heights[j].append(matrix.plotlyfig.layout.height)
    total_col_heights = [sum(col) for col in col_heights]
    max_total_col_height = max(total_col_heights)
    height = max_total_col_height*scale
    
    #set margins
    b=10*scale
    t=30*scale
    r=40*scale
    l=40*scale
          
    layout = dict(
        title=dict(text=title, font=dict(size=12 * scale)),
        width=width,
        height=height,
        margin_b=b, 
        margin_t=t, 
        margin_r=r, 
        margin_l=l
    )

    fig.update_layout(layout)
    
    return ReportFigure(fig)



def _matrix_color_boxplot(matrix, xlabels=None, ylabels=None,
                          xlabel=None, ylabel=None, box_labels=False,
                          thick_line_interval=None, colorbar=None, colormap=None,
                          prec=0, scale=1.0, eb_matrix=None, title=None, grid="black"):
    """
    Creates a color box plot for visualizing a single matrix.

    Parameters
    ----------
    matrix : numpy array
        The matrix to visualize.

    xlabels : list, optional
        List of (str) box labels along the x-axis.

    ylabels : list, optional
        List of (str) box labels along the y-axis.

    xlabel : str, optional
        X-axis label of the plot.

    ylabel : str, optional
        Y-axis label of the plot.

    box_labels : bool, optional
        Whether box labels are displayed.

    thick_line_interval : int, optional
        If not None, the interval at thicker (darker) lines should be placed.
        For example, if 2 then every other grid line will be thick.

    colorbar : bool optional
        Whether to display a color bar to the right of the box plot.  If None,
        then a colorbar is displayed when `box_labels == False`.

    colormap : Colormap, optional
        An a color map object used to convert the numerical matrix values into
        colors.

    prec : int or {'compact','compacthp'}, optional
        Precision for box labels.  Only relevant when box_labels == True. Allowed
        values are:

        - 'compact' = round to nearest whole number using at most 3 characters
        - 'compacthp' = show as much precision as possible using at most 3 characters
        - int >= 0 = fixed precision given by int
        - int <  0 = number of significant figures given by -int

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    eb_matrix : numpy array, optional
        An array, of the same size as `matrix`, which gives error bars to be
        be displayed in the hover info.

    title : str, optional
        A title for the plot

    grid : {"white","black",None}
        What color grid lines, if any, to add to the plot.  Advanced usage
        allows the addition of `:N` where `N` is an integer giving the line
        width.

    Returns
    -------
    plotly.Figure
    """
    if xlabels is None: xlabels = [""] * matrix.shape[1]
    if ylabels is None: ylabels = [""] * matrix.shape[0]
    HOVER_PREC = 7  # precision for hover labels

    colorbar = colorbar if (colorbar is not None) else (not box_labels)

    flipped_mx = _np.flipud(matrix)  # FLIP so [0,0] matrix el is at *top* left
    ylabels = list(reversed(ylabels))  # FLIP y-labels to match

    #Create hoverlabels manually, since hoverinfo='z' arg to Heatmap
    # doesn't work for certain (e.g. linear-log) color maps
    if eb_matrix is None:
        def hover_label_fn(i, j):
            """ Standard hover labels """
            val = flipped_mx[i, j]
            if _np.isnan(val): return ""
            return "%s" % round(val, HOVER_PREC)  # TODO: something better - or user-specifiable
    else:
        flipped_EBmx = _np.flipud(eb_matrix)  # FLIP so [0,0] matrix el is at *top* left

        def hover_label_fn(i, j):
            """ Standard hover labels w/error bars"""
            val, eb = flipped_mx[i, j], flipped_EBmx[i, j]
            if _np.isnan(val): return ""
            # TODO: something better - or user-specifiable
            return "%s <br>+/- %s" % (round(val, HOVER_PREC), round(eb, HOVER_PREC))

    hoverLabels = []
    for i in range(matrix.shape[0]):
        hoverLabels.append([hover_label_fn(i, j)
                            for j in range(matrix.shape[1])])
        
    #allow for possibilities of nans.
    masked_matrix = _np.ma.array(flipped_mx, mask=_np.isnan(flipped_mx))
    trace = go.Heatmap(z=colormap.normalize(masked_matrix),
                       colorscale=colormap.create_plotly_colorscale(),
                       showscale=colorbar, zmin=colormap.hmin,
                       zmax=colormap.hmax, hoverinfo='text',
                       text=hoverLabels)
    #trace = dict(type='heatmapgl', z=colormap.normalize(flipped_mx),
    #             colorscale=colormap.create_plotly_colorscale(),
    #             showscale=colorbar, zmin=colormap.hmin,
    #             zmax=colormap.hmax, hoverinfo='text', text=hoverLabels) #hoverinfo='z')

    data = [trace]

    nX = matrix.shape[1]
    nY = matrix.shape[0]

    gridlines = []
    showframe = True  # show black axes except with white grid lines (aesthetic)

    if grid:
        if ':' in grid:
            gridlinecolor, w = grid.split(':')
            gridlinewidth = int(w)
        else:
            gridlinecolor = grid  # then 'grid' is the line color
            gridlinewidth = None

        if gridlinecolor == "white":
            showframe = False

        # Vertical lines
        for i in range(nX - 1):
            if gridlinewidth:
                w = gridlinewidth
            else:
                #add darker lines at multiples of thick_line_interval boxes
                w = 3 if (thick_line_interval
                          and (i + 1) % thick_line_interval == 0) else 1

            gridlines.append(
                {
                    'type': 'line',
                    'x0': i + 0.5, 'y0': -0.5,
                    'x1': i + 0.5, 'y1': nY - 0.5,
                    'line': {'color': gridlinecolor, 'width': w},
                })

        #Horizontal lines
        for i in range(nY - 1):
            if gridlinewidth:
                w = gridlinewidth
            else:
                #add darker lines at multiples of thick_line_interval boxes
                w = 3 if (thick_line_interval and (i + 1) % thick_line_interval == 0) else 1

            gridlines.append(
                {
                    'type': 'line',
                    'x0': -0.5, 'y0': i + 0.5,
                    'x1': nX - 0.5, 'y1': i + 0.5,
                    'line': {'color': gridlinecolor, 'width': w},
                })

    annotations = []
    if box_labels:
        for ix in range(nX):
            for iy in range(nY):
                annotations.append(
                    dict(
                        text=_ph._eformat(matrix[iy, ix], prec),
                        x=ix, y=nY - 1 - iy, xref='x1', yref='y1',
                        font=dict(size=8,  # don't scale box labels (OLD: 8*scale)
                                  color=colormap.besttxtcolor(matrix[iy, ix])),
                        showarrow=False)
                )

    #Set plot size and margins
    lmargin = rmargin = tmargin = bmargin = 20
    if title: tmargin += 30
    if xlabel: tmargin += 30
    if ylabel: lmargin += 30
    max_xl = max([len(str(xl)) for xl in xlabels])
    if max_xl > 0: tmargin += max_xl * 7
    max_yl = max([len(str(yl)) for yl in ylabels])
    if max_yl > 0: lmargin += max_yl * 7
    if colorbar: rmargin = 100

    boxSizeX = boxSizeY = 15

    maxTextLen = -1  # DB
    if box_labels:
        if prec in ('compact', 'compacthp'):
            precnum = 3 + 1  # +1 for - sign, e.g. "-1e4"
        else: precnum = abs(prec) + 1
        boxSizeX = boxSizeY = 8 * precnum

        if len(annotations) > 0:
            maxTextLen = max([len(ann['text']) for ann in annotations])
            boxSizeX = boxSizeY = max(8 * maxTextLen, 8 * precnum)

    if (matrix.shape[0] < 3):
        # there is issue with hover info not working for
        # small row of horizontal boxes
        boxSizeY = max(35, boxSizeY)

    width = lmargin + boxSizeX * matrix.shape[1] + rmargin
    height = tmargin + boxSizeY * matrix.shape[0] + bmargin

    #create a PIL ImageFont object which will be used as a helper for estimating the
    #rendered width of the title. If this is larger than the manually specified width above
    #then change the width to this value.
    if title is not None:
        font_path = str(Path(__file__).parent / 'fonts'/ 'OpenSans-Regular.ttf')
        font = _ImageFont.truetype(font_path, 10)
        split_title= title.split('<br>')
        max_title_width = max([font.getlength(subtitle) for subtitle in split_title])
        width = max(width, max_title_width)
        
    width *= scale
    height *= scale
    lmargin *= scale
    rmargin *= scale
    tmargin *= scale
    bmargin *= scale

    layout = go.Layout(
        title={'text': title, 'font': dict(size=10 * scale)},
        width=width,
        height=height,
        margin=go_margin(l=lmargin, r=rmargin, b=bmargin, t=tmargin),  # pad=0
        xaxis=dict(
            side="top",
            title=dict(text=xlabel, font=dict(size=10 * scale)),
            showgrid=False,
            zeroline=False,
            showline=showframe,
            showticklabels=True,
            mirror=True,
            ticks="",
            linewidth=2,
            ticktext=[str(xl) for xl in xlabels],
            tickvals=[i for i in range(len(xlabels))],
            range=[-0.5, len(xlabels) - 0.5]
        ),
        yaxis=dict(
            side="left",
            title=dict(text=ylabel, font=dict(size=10 * scale)),
            showgrid=False,
            zeroline=False,
            showline=showframe,
            showticklabels=True,
            mirror=True,
            ticks="",
            linewidth=2,
            ticktext=[str(yl) for yl in ylabels],
            tickvals=[i for i in range(len(ylabels))],
            range=[-0.5, len(ylabels) - 0.5],
        ),
        shapes=gridlines,
        annotations=annotations
    )

    return ReportFigure(go.Figure(data=data, layout=layout),
                        colormap, flipped_mx, plt_data=flipped_mx)

def _matrices_color_boxplot(matrices, xlabels=None, ylabels=None,
                          xlabel=None, ylabel=None, box_labels=False,
                          thick_line_interval=None, colorbar=None, colormap=None,
                          prec=0, scale=1.0, eb_matrices=None, title=None, grid="black",
                          arrangement='row', subtitles= None):
    """
    Creates a color box plot for visualizing a single matrix.

    Parameters
    ----------
    matrices : list of numpy arrays
        The matrices to visualize.

    xlabels : list, optional
        List of (str) box labels along the x-axis.

    ylabels : list, optional
        List of (str) box labels along the y-axis.

    xlabel : str, optional
        X-axis label of the plot.

    ylabel : str, optional
        Y-axis label of the plot.

    box_labels : bool, optional
        Whether box labels are displayed.

    thick_line_interval : int, optional
        If not None, the interval at thicker (darker) lines should be placed.
        For example, if 2 then every other grid line will be thick.

    colorbar : bool optional
        Whether to display a color bar to the right of the box plot.  If None,
        then a colorbar is displayed when `box_labels == False`.

    colormap : Colormap, optional
        An a color map object used to convert the numerical matrix values into
        colors.

    prec : int or {'compact','compacthp'}, optional
        Precision for box labels.  Only relevant when box_labels == True. Allowed
        values are:

        - 'compact' = round to nearest whole number using at most 3 characters
        - 'compacthp' = show as much precision as possible using at most 3 characters
        - int >= 0 = fixed precision given by int
        - int <  0 = number of significant figures given by -int

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    eb_matrix : numpy array, optional
        An array, of the same size as `matrix`, which gives error bars to be
        be displayed in the hover info.

    title : str, optional
        A title for the plot

    grid : {"white","black",None}
        What color grid lines, if any, to add to the plot.  Advanced usage
        allows the addition of `:N` where `N` is an integer giving the line
        width.

    eb_matrices : list of numpy arrays, optional
        List of arrays, of the same size as elements of `op_matrices`, which gives 
        error bars to be be displayed in the hover info.

    arrangement : string or tuple, optional (default 'row')
        Arrangement of the subplots to use in constructing composite
        figure. Supported string options are 'row' and 'col', which will
        arrange the subplots into a single row or column. Also supported is
        a tuple corresponding to the shape of the grid to arrange the subplots
        into. When using a tuple to specify a grid that grid is filled rowwise
        from left to right.

    subtitles : list of str, optional (default None)
        A list of strings for the sub-titles on each matrix's subplot.

    Returns
    -------
    plotly.Figure
    """

    #Initialize the subplot
    if arrangement == 'row':
        num_rows = 1
        num_cols = len(matrices)
    elif arrangement == 'col':
        num_rows = len(matrices)
        num_cols = 1
    elif isinstance(arrangement, tuple):
        num_rows = arrangement[0]
        num_cols= arrangement[1]
    else:
        raise ValueError(f'Unrecognized argument for arrangement: {arrangement}')
    #plotly is 1-indexed for subplots.
    row_col_indices = list(_product(range(1,num_rows+1), range(1,num_cols+1)))
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=subtitles)

    if eb_matrices is None:
        eb_matrices = [None]*len(matrices)

    #loop through the matrices and draw the submatrices for each. To minimize
    #code duplication use the function _matrix_color_boxplot to create each of these,
    #but ignore the layout it generates and extract just the trace.
    matrix_plots = [[] for _ in range(num_rows)]
    for matrix, (row_idx, col_idx), eb_matrix in zip(matrices, row_col_indices, eb_matrices):
        matrix_plots[row_idx-1].append(_matrix_color_boxplot(matrix, xlabels, ylabels, xlabel, ylabel,
                                            box_labels, thick_line_interval, colorbar, colormap, prec, 
                                            1, eb_matrix, None, grid))
    for i, row in enumerate(matrix_plots):
        for j, matrix in enumerate(row):
            #add the first trace (data[0]) to figure
            fig.add_trace(matrix.plotlyfig.data[0], row=i+1, col=j+1)
            #set the subfigures axes layout to match the parents
            matrix_dict_layout= matrix.plotlyfig.to_dict()['layout']
            fig.update_xaxes(row=i+1, col=j+1, **matrix_dict_layout['xaxis'])
            fig.update_yaxes(row=i+1, col=j+1, **matrix_dict_layout['yaxis'])
            #add the shapes from the subplot to the main figure but update
            #the axis references to point to the correct subplot.
            flattened_idx = num_rows*i+ (j+1)
            matrix_shapes = matrix_dict_layout['shapes']
            for shape in matrix_shapes:
                shape['xref'] = f'x{flattened_idx}'
                shape['yref'] = f'y{flattened_idx}'
                fig.add_shape(shape)
            #do this same remapping with the annotations
            matrix_annotations = matrix_dict_layout['annotations']
            for annotation in matrix_annotations:
                annotation['xref'] = f'x{flattened_idx}'
                annotation['yref'] = f'y{flattened_idx}'
                fig.add_annotation(annotation)
            
    #set the width of the composite figure to be equal to the width of the widest row.
    row_widths = [[] for _ in range(num_rows)]
    for i, row in enumerate(matrix_plots):
        for j, matrix in enumerate(row):
            row_widths[i].append(matrix.plotlyfig.layout.width)
    total_row_widths = [sum(row) for row in row_widths]
    max_total_row_width = max(total_row_widths)
    width = max_total_row_width*scale

    #Next we need to set the height in an analogous fashion.
    col_heights = [[] for _ in range(num_cols)]
    for i, row in enumerate(matrix_plots):
        for j, matrix in enumerate(row):
            col_heights[j].append(matrix.plotlyfig.layout.height)
    total_col_heights = [sum(col) for col in col_heights]
    max_total_col_height = max(total_col_heights)
    height = max_total_col_height*scale
    
    #set margins
    b=10*scale
    t=30*scale
    r = 10*scale
    l=10*scale
          
    layout = dict(
        title=dict(text=title, font=dict(size=12*scale)),
        width=width,
        height=height,
        margin_b=b, 
        margin_t=t, 
        margin_r=r, 
        margin_l=l
    )

    fig.update_layout(layout)

    return ReportFigure(fig)
    #return ReportFigure(go.Figure(data=data, layout=layout),
    #                    colormap, flipped_mx, plt_data=flipped_mx)



class BoxKeyPlot(WorkspacePlot):
    """
    Plot serving as a key for fiducial rows/columns of each plaquette of a circuit color box plot.

    This plot shows the layout of a single sub-block of a goodness-of-fit
    box plot (such as those produced by ColorBoxPlot)

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    prep_fiducials : list of Circuits
        Preparation fiducials.

    meas_fiducials : list of Circuits
        Measurement fiducials.

    xlabel: str, optional
        X-axis label

    ylabel: str, optional
        Y-axis label

    scale : float, optional
        Scaling factor to adjust the size of the final figure.
    """

    def __init__(self, ws, prep_fiducials, meas_fiducials,
                 xlabel="Preparation fiducial", ylabel="Measurement fiducial", scale=1.0):
        """
        Create a plot showing the layout of a single sub-block of a goodness-of-fit
        box plot (such as those produced by ColorBoxPlot)

        Parameters
        ----------
        prep_fiducials, meas_fiducials : list of Circuits
            Preparation and measurement fiducials.

        xlabel, ylabel : str, optional
            X and Y axis labels

        scale : float, optional
            Scaling factor to adjust the size of the final figure.
        """
        # separate in rendering/saving: save_to=None, ticSize=20, scale=1.0 (?)
        super(BoxKeyPlot, self).__init__(ws, self._create, prep_fiducials, meas_fiducials,
                                         xlabel, ylabel, scale)

        #size, save_to,

    def _create(self, prep_fiducials, meas_fiducials, xlabel, ylabel, scale):

        #Copied from _summable_color_boxplot
        def val_filter(vals):
            """filter to latex-ify circuits.  Later add filter as a possible parameter"""
            formatted_vals = []
            for val in vals:
                if isinstance(val, _Circuit):
                    if len(val) == 0:
                        #formatted_vals.append(r"$\{\}$")
                        formatted_vals.append(r"{}")
                    else:
                        #formatted_vals.append( "$" + "\\cdot".join([("\\mathrm{%s}" % el) for el in val]) + "$" )
                        formatted_vals.append(val.str)
                else:
                    formatted_vals.append(str(val))
            return formatted_vals

        nX = len(prep_fiducials)
        nY = len(meas_fiducials)
        trace = go.Heatmap(z=_np.zeros((nY, nX), 'd'),
                           colorscale=[[0, 'white'], [1, 'black']],
                           showscale=False, zmin=0, zmax=1, hoverinfo='none')
        #trace = dict(type='heatmapgl', z=_np.zeros((nY,nX),'d'),
        #                   colorscale=[ [0, 'white'], [1, 'black'] ],
        #                   showscale=False, zmin=0,zmax=1,hoverinfo='none')
        data = [trace]

        gridlines = []

        # Vertical lines
        for i in range(nX - 1):
            gridlines.append(
                {
                    'type': 'line',
                    'x0': i + 0.5, 'y0': -0.5,
                    'x1': i + 0.5, 'y1': nY - 0.5,
                    'line': {'color': 'black', 'width': 1},
                })

        #Horizontal lines
        for i in range(nY - 1):
            gridlines.append(
                {
                    'type': 'line',
                    'x0': -0.5, 'y0': i + 0.5,
                    'x1': nX - 0.5, 'y1': i + 0.5,
                    'line': {'color': 'black', 'width': 1},
                })

        layout = go.Layout(
            width=40 * (nX + 1.5) * scale,
            height=40 * (nY + 1.5) * scale,
            xaxis=dict(
                side="bottom",
                showgrid=False,
                zeroline=False,
                showline=True,
                showticklabels=True,
                mirror=True,
                ticks="",
                linewidth=2,
                ticktext=val_filter(prep_fiducials),
                tickvals=[i for i in range(len(prep_fiducials))],
                tickangle=90
            ),
            yaxis=dict(
                side="right",
                showgrid=False,
                zeroline=False,
                showline=True,
                showticklabels=True,
                mirror=True,
                ticks="",
                linewidth=2,
                ticktext=list(reversed(val_filter(meas_fiducials))),
                tickvals=[i for i in range(len(meas_fiducials))],
            ),
            shapes=gridlines,
            annotations=[
                go_annotation(
                    x=0.5,
                    y=1.2,
                    showarrow=False,
                    text=xlabel,
                    font={'size': 12 * scale, 'color': "black"},
                    xref='paper',
                    yref='paper'),
                go_annotation(
                    x=-0.2,
                    y=0.5,
                    showarrow=False,
                    textangle=-90,
                    text=ylabel,
                    font={'size': 12 * scale, 'color': "black"},
                    xref='paper',
                    yref='paper'
                )
            ]
        )
        # margin = go_margin(l=50,r=50,b=50,t=50) #pad=0
        return ReportFigure(go.Figure(data=data, layout=layout),
                            None, "No data in box key plot!",
                            special='keyplot', args=(prep_fiducials, meas_fiducials, xlabel, ylabel))


class ColorBoxPlot(WorkspacePlot):
    """
    Plot of colored boxes arranged into plaquettes showing various quanties for each circuit in an analysis.

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    plottype : {"chi2","logl","tvd","blank","errorrate","dscmp", "driftdetector", "driftsize"}
        Specifies the type of plot. "errorate" requires that `direct_gst_models` be set.

    circuits : CircuitList or list of Circuits
        Specifies the set of circuits, usually along with their structure, e.g.
        fiducials, germs, and maximum lengths.

    dataset : DataSet
        The data used to specify frequencies and counts.

    model : Model
        The model used to specify the probabilities and SPAM labels.

    sum_up : bool, optional
        False displays each matrix element as it's own color box
        True sums the elements of each (x,y) matrix and displays
        a single color box for the sum.

    box_labels : bool, optional
        Whether box labels are displayed.  It takes much longer to
        generate the figure when this is set to True.

    hover_info : bool, optional
        Whether to include interactive hover labels.

    invert : bool, optional
        If True, invert the nesting order of the color box plot (applicable
        only when sum_up == False).

    prec : int, optional
        Precision for box labels.  Allowed values are:
        * 'compact' = round to nearest whole number using at most 3 characters
        * 'compacthp' = show as much precision as possible using at most 3 characters
        *  int >= 0 = fixed precision given by int
        *  int <  0 = number of significant figures given by -int

    linlg_pcntle : float, optional
        Specifies the (1 - linlg_pcntle) percentile to compute for the boxplots

    direct_gst_models : dict, optional
        A dictionary of "direct" Models used when displaying certain plot
        types.  Keys are circuits and values are corresponding gate
        sets (see `plottype` above).

    dscomparator : DataComparator, optional
        The data set comparator used to produce the "dscmp" plot type.

    stabilityanalyzer : StabilityAnalyzer or 3-tuple, optional
        Only used to produce the "driftdetector" and "driftsize" boxplot. If a StabilityAnalyzer, then
        this contains the results of the drift / stability analysis to be displayed.
        For non-expert users, this is the best option. If a tuple, then the first
        element of the tuple is this StabilityAnalyzer object,
        and the second and third elements of the tuple label which instability detection
        results to display (a StabilityAnalyzer can contain multiple distinct tests for
        instability). The second element is the "detectorkey", which can be None (the
        default), or a string specifying which of the drift detection results to use for
        the plot. If it is None, then the default set of results are used. The third element
        of the tuple is either None, or a tuple that specifies which "level" of tests to
        use from the drift detection run (specified by the detectorkey), e.g., per-circuit
        with outcomes averaged or per-circuit per-outcome.

    mdc_store : ModelDatasetCircuitStore, optional (default None)
        A existing ModelDatasetCircuitStore from which to extract the circuits,
        dataset and model used for the construction of objective functions in certain
        family of color box plots. Specifying this when a pre-existing store exists
        can speed up run time significantly as MDC store creation is expensive.
        While the MDC store contains copies of the circuits, dataset and model,
        the format of these is not guaranteed to be in a format amenable for efficient
        calculation, and as such the values of `circuits`, `dataset` and `model` 
        passed in as arguments is only overridden with those from mdc_store 
        if they are None.

    submatrices : dict, optional
        A dictionary whose keys correspond to other potential plot
        types and whose values are each a list-of-lists of the sub
        matrices to plot, corresponding to the used x and y values
        of the structure of `circuits`.

    typ : {"boxes","scatter","histogram"}
        Which type of plot to make: the standard grid of "boxes", a
        "scatter" plot of the values vs. sequence length, or a "histogram"
        of all the values.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    wildcard : WildcardBudget
        A wildcard budget to apply to the objective function that increases
        the goodness of fit by adjusting (by an amount measured in TVD) the
        probabilities produced by `model` before comparing with the
        frequencies in `dataset`.  Currently, this functionality is only
        supported for `plottype == "logl"`.

    colorbar : bool, optional
        Whether to include a colorbar.

    bgcolor : str, optional
        Background color for this plot.  Can be common color names, e.g.
        `"black"`, or string RGB values, e.g. `"rgb(255,128,0)"`.
    """

    def __init__(self, ws, plottype, circuits, dataset, model,
                 sum_up=False, box_labels=False, hover_info=True, invert=False,
                 prec='compact', linlg_pcntle=0.05, direct_gst_models=None,
                 dscomparator=None, stabilityanalyzer=None, mdc_store=None,
                 submatrices=None, typ="boxes", scale=1.0, comm=None,
                 wildcard=None, colorbar=False, bgcolor="white", genericdict=None,
                 genericdict_threshold=None):
        """
        Create a plot displaying the value of per-circuit quantities.

        Values are shown on a grid of colored boxes, organized according to
        the structure of the circuits (e.g. by germ and "L").

        Parameters
        ----------
        plottype : {"chi2","logl","tvd","blank","errorrate","dscmp", "driftdetector", "driftsize"}
            Specifies the type of plot. "errorate" requires that `direct_gst_models` be set.

        circuits : CircuitList or list of Circuits
            Specifies the set of circuits, usually along with their structure, e.g.
            fiducials, germs, and maximum lengths.

        dataset : DataSet
            The data used to specify frequencies and counts.

        model : Model
            The model used to specify the probabilities and SPAM labels.

        sum_up : bool, optional
            False displays each matrix element as it's own color box
            True sums the elements of each (x,y) matrix and displays
            a single color box for the sum.

        box_labels : bool, optional
            Whether box labels are displayed.  It takes much longer to
            generate the figure when this is set to True.

        hover_info : bool, optional
            Whether to include interactive hover labels.

        invert : bool, optional
            If True, invert the nesting order of the color box plot (applicable
            only when sum_up == False).

        prec : int, optional
            Precision for box labels.  Allowed values are:
              'compact' = round to nearest whole number using at most 3 characters
              'compacthp' = show as much precision as possible using at most 3 characters
              int >= 0 = fixed precision given by int
              int <  0 = number of significant figures given by -int

        linlg_pcntle : float, optional
            Specifies the (1 - linlg_pcntle) percentile to compute for the boxplots

        direct_gst_models : dict, optional
            A dictionary of "direct" Models used when displaying certain plot
            types.  Keys are circuits and values are corresponding gate
            sets (see `plottype` above).

        dscomparator : DataComparator, optional
            The data set comparator used to produce the "dscmp" plot type.

        stabilityanalyzer : StabilityAnalyzer or 3-tuple, optional
            Only used to produce the "driftdetector" and "driftsize" boxplot. If a StabilityAnalyzer, then
            this contains the results of the drift / stability analysis to be displayed.
            For non-expert users, this is the best option. If a tuple, then the first
            element of the tuple is this StabilityAnalyzer object,
            and the second and third elements of the tuple label which instability detection
            results to display (a StabilityAnalyzer can contain multiple distinct tests for
            instability). The second element is the "detectorkey", which can be None (the
            default), or a string specifying which of the drift detection results to use for
            the plot. If it is None, then the default set of results are used. The third element
            of the tuple is either None, or a tuple that specifies which "level" of tests to
            use from the drift detection run (specified by the detectorkey), e.g., per-circuit
            with outcomes averaged or per-circuit per-outcome.

        mdc_store : ModelDatasetCircuitStore, optional (default None)
            A existing ModelDatasetCircuitStore from which to extract the circuits,
            dataset and model used for the construction of objective functions in certain
            family of color box plots. Specifying this when a pre-existing store exists
            can speed up run time significantly as MDC store creation is expensive.
            While the MDC store contains copies of the circuits, dataset and model,
            the format of these is not guaranteed to be in a format amenable for efficient
            calculation, and as such the values of `circuits`, `dataset` and `model` 
            passed in as arguments is only overridden with those from mdc_store 
            if they are None.

        submatrices : dict, optional
            A dictionary whose keys correspond to other potential plot
            types and whose values are each a list-of-lists of the sub
            matrices to plot, corresponding to the used x and y values
            of the structure of `circuits`.

        typ : {"boxes","scatter","histogram"}
            Which type of plot to make: the standard grid of "boxes", a
            "scatter" plot of the values vs. sequence length, or a "histogram"
            of all the values.

        scale : float, optional
            Scaling factor to adjust the size of the final figure.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        wildcard : WildcardBudget
            A wildcard budget to apply to the objective function that increases
            the goodness of fit by adjusting (by an amount measured in TVD) the
            probabilities produced by `model` before comparing with the
            frequencies in `dataset`.  Currently, this functionality is only
            supported for `plottype == "logl"`.

        colorbar : bool, optional
            Whether to include a colorbar.

        bgcolor : str, optional
            Background color for this plot.  Can be common color names, e.g.
            `"black"`, or string RGB values, e.g. `"rgb(255,128,0)"`.
        """
        # separate in rendering/saving: save_to=None, ticSize=20, scale=1.0 (?)
        super(ColorBoxPlot, self).__init__(ws, self._create, plottype, circuits, dataset, model,
                                           prec, sum_up, box_labels, hover_info, invert, linlg_pcntle,
                                           direct_gst_models, dscomparator, stabilityanalyzer, mdc_store,
                                           submatrices, typ, scale, comm, wildcard, colorbar, bgcolor,
                                           genericdict, genericdict_threshold)

    def _create(self, plottypes, circuits, dataset, model, prec, sum_up, box_labels, hover_info,
                invert, linlg_pcntle, direct_gst_models, dscomparator, stabilityanalyzer, mdc_store,
                submatrices, typ, scale, comm, wildcard, colorbar, bgcolor, genericdict, genericdict_threshold):

        fig = None
        addl_hover_info_fns = dict()
        if mdc_store is not None:  # then it overrides
            circuits = mdc_store.circuits if circuits is None else circuits
            dataset = mdc_store.dataset if dataset is None else dataset
            model = mdc_store.model if model is None else model

        if not isinstance(plottypes, (list, tuple)):
            plottypes = [plottypes]

        for ptyp in plottypes:
            if ptyp in ("logl", "chi2", "tvd"):
                ptyp = _objfns.ObjectiveFunctionBuilder.create_from(ptyp)

            if isinstance(ptyp, _objfns.ObjectiveFunctionBuilder):
                if mdc_store is None:
                    mdc_store = _ModelDatasetCircuitStore(model, dataset, circuits, array_types=('E',))

                objfn_builder = ptyp
                objfn = objfn_builder.build_from_store(mdc_store)

                if wildcard:
                    objfn.terms()  # objfn used within wildcard objective fn must be pre-evaluated
                    objfn = _objfns.LogLWildcardFunction(objfn, model.to_vector(), wildcard)
                terms = objfn.terms()  # also assumed to set objfn.probs, objfn.freqs, and objfn.counts

                if isinstance(objfn, (_objfns.PoissonPicDeltaLogLFunction, _objfns.DeltaLogLFunction)):
                    terms *= 2.0  # show 2 * deltaLogL values, not just deltaLogL
                if isinstance(objfn, _objfns.TVDFunction):
                    colormapType = "blueseq"
                else:
                    colormapType = "linlog"
                    linlog_color = "red"

                ytitle = objfn.description  # "chi<sup>2</sup>" OR "2 log(L ratio)"
                
                if isinstance(circuits, _PlaquetteGridCircuitStructure):
                    mx_fn = _mx_fn_from_elements  # use a *global* function so cache can tell it's the same
                elif isinstance(circuits, _CircuitList):
                    mx_fn = _mx_fn_from_elements_circuit_list
                elif isinstance(circuits, list) and all([isinstance(el, _CircuitList) for el in circuits]):
                    mx_fn = _mx_fn_from_elements_circuit_list
                
                extra_arg = (terms, objfn.layout, "sum")
                
                if isinstance(circuits, _PlaquetteGridCircuitStructure):
                    # (function, extra_arg) tuples
                    addl_hover_info_fns['outcomes'] = (_addl_mx_fn_outcomes, (objfn.layout, '>11s'))
                    addl_hover_info_fns['probs'] = (_mx_fn_from_elements, (objfn.probs, objfn.layout, "%11.4g"))
                    addl_hover_info_fns['freqs'] = (_mx_fn_from_elements, (objfn.freqs, objfn.layout, "%11.4g"))
                    addl_hover_info_fns['counts'] = (_mx_fn_from_elements, (objfn.counts, objfn.layout, "%11d"))
                elif isinstance(circuits, _CircuitList) or \
                    (isinstance(circuits, list) and all([isinstance(el, _CircuitList) for el in circuits])):
                     # (function, extra_arg) tuples
                    addl_hover_info_fns['outcomes'] = (_addl_mx_fn_outcomes_circuit_list, (objfn.layout, '>11s'))
                    addl_hover_info_fns['probs'] = (_mx_fn_from_elements_circuit_list, (objfn.probs, objfn.layout, "%11.4g"))
                    addl_hover_info_fns['freqs'] = (_mx_fn_from_elements_circuit_list, (objfn.freqs, objfn.layout, "%11.4g"))
                    addl_hover_info_fns['counts'] = (_mx_fn_from_elements_circuit_list, (objfn.counts, objfn.layout, "%11d"))
                
            elif ptyp == "blank":
                colormapType = "trivial"
                ytitle = ""
                mx_fn = _mx_fn_blank  # use a *global* function so cache can tell it's the same
                extra_arg = None

            elif ptyp == "errorrate":
                colormapType = "seq"
                ytitle = "error rate"
                mx_fn = _mx_fn_errorrate  # use a *global* function so cache can tell it's the same
                extra_arg = direct_gst_models
                assert(sum_up is True), "Can only use 'errorrate' plot with sum_up == True"

            elif ptyp == "dscmp":
                assert(dscomparator is not None), \
                    "Must specify `dscomparator` argument to create `dscmp` plot!"
                colormapType = "manuallinlog"
                linlog_color = "green"
                linlog_trans = dscomparator.llr_pseudothreshold
                ytitle = "2 log(L ratio)"
                mx_fn = _mx_fn_dscmp  # use a *global* function so cache can tell it's the same
                extra_arg = dscomparator

            elif ptyp == "dict":
                assert(genericdict is not None), \
                    "Must specify `dscomparator` argument to create `dscmp` plot!"
                if genericdict_threshold is None:
                    colormapType = "blueseq"
                else:
                    colormapType = "manuallinlog"
                    linlog_color = "green"
                    linlog_trans = genericdict_threshold
                ytitle = "."
                mx_fn = _mx_fn_dict  # use a *global* function so cache can tell it's the same
                extra_arg = genericdict

            elif ptyp == "driftdetector":

                assert(stabilityanalyzer is not None), \
                    "Must specify `stabilityanalyzer` argument to create `drift` plot!"
                # If stabilityanalyzer  is a tuple, we expand into its components
                if isinstance(stabilityanalyzer, tuple):

                    detectorkey = stabilityanalyzer[1]
                    test = stabilityanalyzer[2]
                    stabilityanalyzer = stabilityanalyzer[0]
                # If stabilityanalyzer is a StabilityAnalyzer, we initialize the other components not given
                else:
                    detectorkey = None
                    test = None

                # If these componentes aren't given, we use defaults
                if detectorkey is None: detectorkey = stabilityanalyzer._def_detection
                if test is None: test = ('circuit', )
                assert('circuit' in test), \
                    "Cannot create this plot unless considering a per-circuit instability test!"
                test = stabilityanalyzer._equivalent_implemented_test(test, detectorkey)
                assert(test is not None), \
                    "The automatic test that we default to was not implemented! Must specify the test to use!"
                assert(len(stabilityanalyzer.data.keys()) == 1), \
                    "Currently cannot create a box-plot for multi-DataSet StabilityAnalyzers!"

                colormapType = "manuallinlog"
                linlog_color = "green"
                pvaluethreshold, junk = stabilityanalyzer.pvalue_threshold(test, detectorkey=detectorkey)
                linlog_trans = -1 * _np.log10(pvaluethreshold)
                ytitle = "Evidence for instability as quantified by -log10(pvalue)"
                mx_fn = _mx_fn_driftpv  # use a *global* function so cache can tell it's the same
                extra_arg = (stabilityanalyzer, test)

            elif ptyp == "driftsize":

                assert(stabilityanalyzer is not None), \
                    "Must specify `stabilityanalyzer` argument to create `drift` plot!"
                # If stabilityanalyzer  is a tuple, we expand into its components
                if isinstance(stabilityanalyzer, tuple):

                    estimatekey = stabilityanalyzer[1]
                    estimator = stabilityanalyzer[2]
                    stabilityanalyzer = stabilityanalyzer[0]
                # If stabilityanalyzer is a StabilityAnalyzer, we initialize the other components not given
                else:
                    estimatekey = None
                    estimator = None

                colormapType = "blueseq"
                ytitle = "Total Variational Distance (TVD) Bound"
                mx_fn = _mx_fn_drifttvd  # use a *global* function so cache can tell it's the same
                extra_arg = (stabilityanalyzer, estimatekey, estimator)

            # future: delete this, or update it and added it back in.
            # elif ptyp == "driftpwr":
            #     assert(driftresults is not None), \
            #         "Must specify `driftresults` argument to create `driftpv` plot!"
            #     detectorkey = driftresults[1]
            #     driftresults = driftresults[0]
            #     assert(driftresults.number_of_entities == 1), \
            #         "Currently cannot create a box-plot for multi-entity DriftResults!"
            #     colormapType = "manuallinlog"
            #     linlog_color = "green"
            #     linlog_trans = driftresults.get_power_significance_threshold(sequence='per', detectorkey=detectorkey)
            #     ytitle = "Maximum power in spectrum"
            #     mx_fn = _mx_fn_driftpwr  # use a *global* function so cache can tell it's the same
            #     extra_arg = driftresults

            elif (submatrices is not None) and ptyp in submatrices:
                ytitle = ptyp
                colormapType = submatrices.get(ptyp + ".colormap", "seq")
            else:
                raise ValueError("Invalid plot type: %s" % ptyp)
            
            #TODO: propagate mdc_store down into compute_sub_mxs?
            if (submatrices is not None) and ptyp in submatrices:
                subMxs = submatrices[ptyp]  # "custom" type -- all mxs precomputed by user

                #some of the branches below rely on circuit_struct being defined, which is previously
                #wasn't when hitting this condition on the if statement, so add those definitions here.
                #also need to built the addl_hover_info as well, based on circuit_struct.
                if isinstance(circuits, _PlaquetteGridCircuitStructure):
                    circuit_struct = circuits

                    addl_hover_info = dict()
                    for lbl, (addl_mx_fn, addl_extra_arg) in addl_hover_info_fns.items():
                        if (submatrices is not None) and lbl in submatrices:
                            addl_subMxs = submatrices[lbl]  # ever useful?
                        else:
                            addl_subMxs = self._ccompute(_ph._compute_sub_mxs, circuit_struct, model,
                                                        addl_mx_fn, dataset, addl_extra_arg)
                        addl_hover_info[lbl] = addl_subMxs

                elif isinstance(circuits, _CircuitList):
                    circuit_struct = [circuits]

                    addl_hover_info = dict()
                    for lbl, (addl_mx_fn, addl_extra_arg) in addl_hover_info_fns.items():
                        if (submatrices is not None) and lbl in submatrices:
                            addl_subMxs = submatrices[lbl]  # ever useful?
                        else:
                            addl_subMxs = self._ccompute(_ph._compute_sub_mxs_circuit_list, circuit_struct, model,
                                                        addl_mx_fn, dataset, addl_extra_arg)
                        addl_hover_info[lbl] = addl_subMxs

                elif isinstance(circuits, list) and all([isinstance(el, _CircuitList) for el in circuits]):
                    circuit_struct = circuits

                    addl_hover_info = dict()
                    for lbl, (addl_mx_fn, addl_extra_arg) in addl_hover_info_fns.items():
                        if (submatrices is not None) and lbl in submatrices:
                            addl_subMxs = submatrices[lbl]  # ever useful?
                        else:
                            addl_subMxs = self._ccompute(_ph._compute_sub_mxs_circuit_list, circuit_struct, model,
                                                        addl_mx_fn, dataset, addl_extra_arg)
                        addl_hover_info[lbl] = addl_subMxs

                #Otherwise fall-back to the old casting behavior and proceed
                else:
                    circuit_struct = _PlaquetteGridCircuitStructure.cast(circuits)
                    addl_hover_info = dict()
                    for lbl, (addl_mx_fn, addl_extra_arg) in addl_hover_info_fns.items():
                        if (submatrices is not None) and lbl in submatrices:
                            addl_subMxs = submatrices[lbl]  # ever useful?
                        else:
                            addl_subMxs = self._ccompute(_ph._compute_sub_mxs, circuit_struct, model,
                                                        addl_mx_fn, dataset, addl_extra_arg)
                        addl_hover_info[lbl] = addl_subMxs
                
            elif isinstance(circuits, _PlaquetteGridCircuitStructure):
                circuit_struct= circuits
                subMxs = self._ccompute(_ph._compute_sub_mxs, circuit_struct, model, mx_fn, dataset, extra_arg)
                
                addl_hover_info = dict()
                for lbl, (addl_mx_fn, addl_extra_arg) in addl_hover_info_fns.items():
                    if (submatrices is not None) and lbl in submatrices:
                        addl_subMxs = submatrices[lbl]  # ever useful?
                    else:
                        addl_subMxs = self._ccompute(_ph._compute_sub_mxs, circuit_struct, model,
                                                     addl_mx_fn, dataset, addl_extra_arg)
                    addl_hover_info[lbl] = addl_subMxs
                
            #Add in alternative logic for constructing sub-matrices when we have either a CircuitList or a
            #list of circuit lists:
            elif isinstance(circuits, _CircuitList):
                circuit_struct= [circuits]
                subMxs = self._ccompute(_ph._compute_sub_mxs_circuit_list, circuit_struct, model, mx_fn, dataset, extra_arg)
                
                addl_hover_info = _collections.OrderedDict()
                for lbl, (addl_mx_fn, addl_extra_arg) in addl_hover_info_fns.items():
                    if (submatrices is not None) and lbl in submatrices:
                        addl_subMxs = submatrices[lbl]  # ever useful?
                    else:
                        addl_subMxs = self._ccompute(_ph._compute_sub_mxs_circuit_list, circuit_struct, model,
                                                     addl_mx_fn, dataset, addl_extra_arg)
                    addl_hover_info[lbl] = addl_subMxs

            elif isinstance(circuits, list) and all([isinstance(el, _CircuitList) for el in circuits]):
                circuit_struct= circuits
                subMxs = self._ccompute(_ph._compute_sub_mxs_circuit_list, circuit_struct, model, mx_fn, dataset, extra_arg)
                
                addl_hover_info = _collections.OrderedDict()
                for lbl, (addl_mx_fn, addl_extra_arg) in addl_hover_info_fns.items():
                    if (submatrices is not None) and lbl in submatrices:
                        addl_subMxs = submatrices[lbl]  # ever useful?
                    else:
                        addl_subMxs = self._ccompute(_ph._compute_sub_mxs_circuit_list, circuit_struct, model,
                                                     addl_mx_fn, dataset, addl_extra_arg)
                    addl_hover_info[lbl] = addl_subMxs

            #Otherwise fall-back to the old casting behavior and proceed
            else:
                circuit_struct = _PlaquetteGridCircuitStructure.cast(circuits) # , dataset?
                subMxs = self._ccompute(_ph._compute_sub_mxs, circuit_struct, model, mx_fn, dataset, extra_arg)
                
                addl_hover_info = _collections.OrderedDict()
                for lbl, (addl_mx_fn, addl_extra_arg) in addl_hover_info_fns.items():
                    if (submatrices is not None) and lbl in submatrices:
                        addl_subMxs = submatrices[lbl]  # ever useful?
                    else:
                        addl_subMxs = self._ccompute(_ph._compute_sub_mxs, circuit_struct, model,
                                                     addl_mx_fn, dataset, addl_extra_arg)
                    addl_hover_info[lbl] = addl_subMxs

            if colormapType == "linlog":
                if dataset is None:
                    _warnings.warn("No dataset specified: using DOF-per-element == 1")
                    element_dof = 1
                else:
                    #element_dof = len(dataset.outcome_labels) - 1
                    #Instead of the above, which doesn't work well when there are circuits with different
                    # outcomes, the line below just takes the average degrees of freedom per circuit
                    element_dof = dataset.degrees_of_freedom(circuits) / len(circuits)

                n_boxes, dof_per_box = _ph._compute_num_boxes_dof(subMxs, sum_up, element_dof)
                # NOTE: currently dof_per_box is constant, and takes the total
                # number of outcome labels in the DataSet, which can be incorrect
                # when different sequences have different outcome labels.

            if len(subMxs) > 0:
                dataMax = max([(0 if (mx is None or _np.all(_np.isnan(mx))) else _np.nanmax(mx))
                               for subMxRow in subMxs for mx in subMxRow])
            else: dataMax = 0

            if colormapType == "linlog":
                colormap = _colormaps.LinlogColormap(0, dataMax, len(circuits),
                                                     linlg_pcntle, dof_per_box, linlog_color)
            elif colormapType == "manuallinlog":
                colormap = _colormaps.LinlogColormap.set_manual_transition_point(
                    0, dataMax, linlog_trans, linlog_color)

            elif colormapType == "trivial":
                colormap = _colormaps.SequentialColormap(vmin=0, vmax=1)

            elif colormapType in ("seq", "revseq", "blueseq", "redseq"):
                if len(subMxs) > 0:
                    if isinstance(circuit_struct, _PlaquetteGridCircuitStructure):
                        max_abs = max([_np.max(_np.abs(_np.nan_to_num(subMxs[iy][ix])))
                                       for ix in range(len(circuit_struct.used_xs))
                                       for iy in range(len(circuit_struct.used_ys))])
                    #circuit_struct logic above should mean that we always have at least a length 1 list of
                    #CircuitList objects if not a plaquette circuit structure by this point.
                    elif isinstance(circuit_struct, list) and all([isinstance(el, _CircuitList) for el in circuit_struct]):
                        max_abs = max([_np.max(_np.abs(_np.nan_to_num(subMxs[i][j])))
                                       for i, ckt_list in enumerate(circuit_struct) 
                                       for j in range(len(ckt_list))])
                    
                else: max_abs = 0
                if max_abs == 0: max_abs = 1e-6  # pick a nonzero value if all entries are zero or nan
                if colormapType == "seq": color = "whiteToBlack"
                elif colormapType == "revseq": color = "blackToWhite"
                elif colormapType == "blueseq": color = "whiteToBlue"
                elif colormapType == "redseq": color = "whiteToRed"
                colormap = _colormaps.SequentialColormap(vmin=0, vmax=max_abs, color=color)

            else: assert(False), "Internal logic error"  # pragma: no cover

            if typ == "boxes":
                if not isinstance(circuit_struct, _PlaquetteGridCircuitStructure):
                    #if not a plaquette structure then maybe try returning a NotApplicable object
                    #for the figure?
                    return NotApplicable(self.ws)
                else:
                    #I am expecting this cast won't do anything at the moment, but
                    #maybe down the line it will.
                    circuit_struct= _PlaquetteGridCircuitStructure.cast(circuits)
                    newfig = _circuit_color_boxplot(circuit_struct, subMxs, colormap,
                                                colorbar, box_labels, prec,
                                                hover_info, sum_up, invert,
                                                scale, bgcolor, addl_hover_info)

            elif typ == "scatter":
                newfig = _circuit_color_scatterplot(circuit_struct, subMxs, colormap,
                                                    colorbar, hover_info, sum_up, ytitle,
                                                    scale, addl_hover_info)
            elif typ == "histogram":
                if isinstance(ptyp, _objfns.ObjectiveFunctionBuilder) and ptyp.name=='tvd':
                    include_chi2 = False
                else:
                    include_chi2 = True
                newfig = _circuit_color_histogram(circuit_struct, subMxs, colormap,
                                                  ytitle, scale, include_chi2=include_chi2)
            else:
                raise ValueError("Invalid `typ` argument: %s" % typ)

            if fig is None:
                fig = newfig
            else:
                newfig.plotlyfig['data'][0].update(visible=False)
                combined_fig_data = list(fig.plotlyfig['data']) + [newfig.plotlyfig['data'][0]]
                fig = ReportFigure(go.Figure(data=combined_fig_data, layout=fig.plotlyfig['layout']),
                                   fig.colormap, fig.pythonvalue)  # just add newfig's data
                #Note: can't do fig.plotlyfig['data'].append(newfig.plotlyfig['data'][0]) as of plotly v3

        nTraces = len(fig.plotlyfig['data'])
        assert(nTraces >= len(plottypes))  # not == b/c histogram adds line trace

        if len(plottypes) > 1:
            buttons = []
            for i, nm in enumerate(plottypes):
                visible = [False] * nTraces
                visible[i] = True
                buttons.append(
                    dict(args=['visible', visible],
                         label=nm,
                         method='restyle'))
            fig.plotlyfig['layout'].update(
                updatemenus=list([
                    dict(buttons=buttons,
                         direction='left',
                         pad={'r': 10, 't': 10},
                         showactive=True, type='buttons',
                         x=0.1, xanchor='left',
                         y=1.1, yanchor='top')
                ]))

        #colormap2 = _colormaps.LinlogColormap(0, dataMax, n_boxes, linlg_pcntle, dof_per_box, "blue")
        #fig2 = _circuit_color_boxplot(gss, subMxs, colormap2,
        #                                False, box_labels, prec, hover_info, sum_up, invert)
        #fig['data'].append(fig2['data'][0])
        #fig['layout'].update( )
        return fig


#Helper function for ColorBoxPlot matrix computation
def _mx_fn_from_elements(plaq, x, y, extra):
    return plaq.elementvec_to_array(extra[0], extra[1], mergeop=extra[2])

#modified version of the above meant for working with circuit lists
def _mx_fn_from_elements_circuit_list(circuit_list, extra):
    #Based on the convention above in the ColorBoxPlot code it looks likelihood
    #extra[0] is the thing we want to index into, extra[1] is the layout and extra[2]
    #is something called the merge op, which indicated how to combine the elements of extra[0]
    #for each circuit in the circuit_list
    if isinstance(circuit_list, _CircuitList):
        pass
    elif isinstance(circuit_list, list) and all([isinstance(el, _CircuitList) for el in circuit_list]):
        circuit_list = _CircuitList.cast(circuit_list)
    else:
        msg = 'Invalid type. _mx_fn_from_elements_circuit_list is only presently implemented for CircuitList'\
             +'objects and lists of Circuit objects.'
        raise ValueError(msg)

    return circuit_list.elementvec_to_array(extra[0], extra[1], mergeop=extra[2])
    
def _mx_fn_blank(plaq, x, y, unused):
    return _np.nan * _np.zeros((plaq.num_rows, plaq.num_cols), 'd')


def _mx_fn_errorrate(plaq, x, y, direct_gst_models):  # error rate as 1x1 matrix which we have plotting function sum up
    base_circuit = plaq.base if isinstance(plaq, _GermFiducialPairPlaquette) \
        else _Circuit((), line_labels=list(direct_gst_models.keys())[0].line_labels) #Taking the line labels from the first circuit in direct_gst_models will probably work
        #most of the time. TODO: Cook up a better scheme.
    return _np.array([[_ph.small_eigenvalue_err_rate(base_circuit, direct_gst_models)]])


def _mx_fn_directchi2(plaq, x, y, extra):
    dataset, directGSTmodels, minProbClipForWeighting, gss = extra
    return _ph.direct_chi2_matrix(
        plaq, gss, dataset, directGSTmodels.get(plaq.base, None),
        minProbClipForWeighting)


def _mx_fn_directlogl(plaq, x, y, extra):
    dataset, directGSTmodels, minProbClipForWeighting, gss = extra
    return _ph.direct_logl_matrix(
        plaq, gss, dataset, directGSTmodels.get(plaq.base, None),
        minProbClipForWeighting)


def _mx_fn_dscmp(plaq, x, y, dscomparator):
    return _ph.dscompare_llr_matrices(plaq, dscomparator)


def _mx_fn_dict(plaq, x, y, genericdict):
    return _ph.genericdict_matrices(plaq, genericdict)


def _mx_fn_driftpv(plaq, x, y, instabilityanalyzertuple):
    return _ph.drift_neglog10pvalue_matrices(plaq, instabilityanalyzertuple)


def _mx_fn_drifttvd(plaq, x, y, instabilityanalyzertuple):
    return _ph.drift_maxtvd_matrices(plaq, instabilityanalyzertuple)


def _outcome_to_str(x):  # same function as in writers.py
    if isinstance(x, str): return x
    else: return ":".join([str(i) for i in x])


def _addl_mx_fn_outcomes(plaq, x, y, extra):
    layout = extra[0]
    fmt_spec = extra[1]
    slmx = _np.empty((plaq.num_rows, plaq.num_cols), dtype=_np.object_)
    for i, j, opstr in plaq:
        if fmt_spec is not None:
            slmx[i, j] = "".join([f'{_outcome_to_str(ol):{fmt_spec}}' for ol in layout.outcomes(opstr)])
        else:
            slmx[i, j] = "".join([_outcome_to_str(ol) for ol in layout.outcomes(opstr)])
    return slmx

#modified version of the above function meant to work for CircuitList objects
def _addl_mx_fn_outcomes_circuit_list(circuit_list, extra):
    layout = extra[0]
    fmt_spec = extra[1]
    slmx = _np.empty(len(circuit_list), dtype=_np.object_)
    for i,ckt in enumerate(circuit_list):
        if fmt_spec is not None:
            slmx[i] = ", ".join([f'{_outcome_to_str(ol):{fmt_spec}}' for ol in layout.outcomes(ckt)])
        else:
            slmx[i] = ", ".join([_outcome_to_str(ol) for ol in layout.outcomes(ckt)])
    return slmx
    

class GateMatrixPlot(WorkspacePlot):
    """
    Plot of a operation matrix using colored boxes.

    More specific than :class:`MatrixPlot` because of basis formatting
    for x and y labels.

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    op_matrix : ndarray
      The operation matrix data to display.

    color_min : float, optional
      Minimum value of the color scale.

    color_max : float, optional
      Maximum value of the color scale.

    mx_basis : str or Basis object, optional
        The basis, often of `op_matrix`, used to create the x-labels (and
        y-labels when `mx_basis_y` is None). Typically in {"pp","gm","std","qt"}.
        If you don't want labels, leave as None.

    xlabel : str, optional
        A x-axis label for the plot.

    ylabel : str, optional
        A y-axis label for the plot.

    box_labels : bool, optional
        Whether box labels are displayed.

    colorbar : bool optional
        Whether to display a color bar to the right of the box plot.  If None,
        then a colorbar is displayed when `box_labels == False`.

    prec : int or {'compact','compacthp'}, optional
        Precision for box labels.  Only relevant when box_labels == True. Allowed
        values are:

        - 'compact' = round to nearest whole number using at most 3 characters
        - 'compacthp' = show as much precision as possible using at most 3 characters
        - int >= 0 = fixed precision given by int
        - int <  0 = number of significant figures given by -int

    mx_basis_y : str or Basis object, optional
        The basis, used to create the y-labels (for rows) when these are
        *different* from the x-labels.  Typically in
        {"pp","gm","std","qt"}.  If you don't want labels, leave as None.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    eb_matrix : numpy array, optional
        An array, of the same size as `op_matrix`, which gives error bars to be
        be displayed in the hover info.
    
    title : str, optional
            A title for the plot
    """
    # separate in rendering/saving: size=None,fontSize=20, save_to=None, title=None, scale

    def __init__(self, ws, op_matrix, color_min=-1.0, color_max=1.0,
                 mx_basis_x=None, mx_basis_y=None, xlabel=None, ylabel=None,
                 box_labels=False, colorbar=None, prec=0,
                 scale=1.0, eb_matrix=None, title=None):
        """
        Creates a color box plot of a operation matrix using a diverging color map.

        This can be a useful way to display large matrices which have so many
        entries that their entries cannot easily fit within the width of a page.

        Parameters
        ----------
        op_matrix : ndarray
            The operation matrix data to display.

        color_min, color_max : float, optional
            Min and max values of the color scale.

        mx_basis : str or Basis object, optional
            The basis, often of `op_matrix`, used to create the x-labels (and
            y-labels when `mx_basis_y` is None). Typically in {"pp","gm","std","qt"}.
            If you don't want labels, leave as None.

        xlabel : str, optional
            A x-axis label for the plot.

        ylabel : str, optional
            A y-axis label for the plot.

        box_labels : bool, optional
            Whether box labels are displayed.

        colorbar : bool optional
            Whether to display a color bar to the right of the box plot.  If None,
            then a colorbar is displayed when `box_labels == False`.

        prec : int or {'compact','compacthp'}, optional
            Precision for box labels.  Only relevant when box_labels == True. Allowed
            values are:

            - 'compact' = round to nearest whole number using at most 3 characters
            - 'compacthp' = show as much precision as possible using at most 3 characters
            - int >= 0 = fixed precision given by int
            - int <  0 = number of significant figures given by -int

        mx_basis_y : str or Basis object, optional
            The basis, used to create the y-labels (for rows) when these are
            *different* from the x-labels.  Typically in
            {"pp","gm","std","qt"}.  If you don't want labels, leave as None.

        scale : float, optional
            Scaling factor to adjust the size of the final figure.

        eb_matrix : numpy array, optional
            An array, of the same size as `op_matrix`, which gives error bars to be
            be displayed in the hover info.
        
        title : str, optional
            A title for the plot
        """
        super(GateMatrixPlot, self).__init__(ws, self._create, op_matrix, color_min, color_max,
                                             mx_basis_x, mx_basis_y, xlabel, ylabel,
                                             box_labels, colorbar, prec,  scale, eb_matrix, title)

    def _create(self, op_matrix, color_min, color_max,
                mx_basis_x, mx_basis_y, xlabel, ylabel,
                box_labels, colorbar, prec, scale, eb_matrix, title):

        return _opmatrix_color_boxplot(
            op_matrix, color_min, color_max, mx_basis_x, mx_basis_y,
            xlabel, ylabel, box_labels, colorbar, prec, scale, eb_matrix, title)
    
class GateMatricesPlot(WorkspacePlot):
    """
    Plot of operation matrices using colored boxes.

    More specific than :class:`MatricesPlot` because of basis formatting
    for x and y labels.

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    op_matrices : list of ndarrays
      The operation matrices to display.

    color_min : float, optional
      Minimum value of the color scale.

    color_max : float, optional
      Maximum value of the color scale.

    mx_basis : str or Basis object, optional
        The basis, often of `op_matrix`, used to create the x-labels (and
        y-labels when `mx_basis_y` is None). Typically in {"pp","gm","std","qt"}.
        If you don't want labels, leave as None.

    xlabel : str, optional
        A x-axis label for the plot.

    ylabel : str, optional
        A y-axis label for the plot.

    box_labels : bool, optional
        Whether box labels are displayed.

    colorbar : bool optional
        Whether to display a color bar to the right of the box plot.  If None,
        then a colorbar is displayed when `box_labels == False`.

    prec : int or {'compact','compacthp'}, optional
        Precision for box labels.  Only relevant when box_labels == True. Allowed
        values are:

        - 'compact' = round to nearest whole number using at most 3 characters
        - 'compacthp' = show as much precision as possible using at most 3 characters
        - int >= 0 = fixed precision given by int
        - int <  0 = number of significant figures given by -int

    mx_basis_y : str or Basis object, optional
        The basis, used to create the y-labels (for rows) when these are
        *different* from the x-labels.  Typically in
        {"pp","gm","std","qt"}.  If you don't want labels, leave as None.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    eb_matrices : numpy array, optional
        A list of arrays, of the same size as each element of `op_matrices`, 
        which gives error bars to be displayed in the hover info.
    
    title : str, optional
        A title for the plot
        
    arrangement : string or tuple, optional (default 'row')
        Arrangement of the subplots to use in constructing composite
        figure. Supported string options are 'row' and 'col', which will
        arrange the subplots into a single row or column. Also supported is
        a tuple corresponding to the shape of the grid to arrange the subplots
        into. When using a tuple to specify a grid that grid is filled rowwise
        from left to right.

    subtitles : list of str, optional (default None)
        A list of strings for the sub-titles on each matrix's subplot.
    """

    def __init__(self, ws, op_matrices, color_min=-1.0, color_max=1.0,
                 mx_basis_x=None, mx_basis_y=None, xlabel=None, ylabel=None,
                 box_labels=False, colorbar=None, prec=0,
                 scale=1.0, eb_matrices=None, title=None, arrangement='row', subtitles=None):
        """
        Creates a color box plot of a operation matrix using a diverging color map.

        This can be a useful way to display large matrices which have so many
        entries that their entries cannot easily fit within the width of a page.

        Parameters
        ----------
        op_matrices : list of ndarrays
            The operation matrices to display.

        color_min, color_max : float, optional
            Min and max values of the color scale.

        mx_basis : str or Basis object, optional
            The basis, often of `op_matrix`, used to create the x-labels (and
            y-labels when `mx_basis_y` is None). Typically in {"pp","gm","std","qt"}.
            If you don't want labels, leave as None.

        xlabel : str, optional
            A x-axis label for the plot.

        ylabel : str, optional
            A y-axis label for the plot.

        box_labels : bool, optional
            Whether box labels are displayed.

        colorbar : bool optional
            Whether to display a color bar to the right of the box plot.  If None,
            then a colorbar is displayed when `box_labels == False`.

        prec : int or {'compact','compacthp'}, optional
            Precision for box labels.  Only relevant when box_labels == True. Allowed
            values are:

            - 'compact' = round to nearest whole number using at most 3 characters
            - 'compacthp' = show as much precision as possible using at most 3 characters
            - int >= 0 = fixed precision given by int
            - int <  0 = number of significant figures given by -int

        mx_basis_y : str or Basis object, optional
            The basis, used to create the y-labels (for rows) when these are
            *different* from the x-labels.  Typically in
            {"pp","gm","std","qt"}.  If you don't want labels, leave as None.

        scale : float, optional
            Scaling factor to adjust the size of the final figure.

        eb_matrices : numpy array, optional
            A list of arrays, of the same size as each element of `op_matrices`, 
            which gives error bars to be displayed in the hover info.
        
        title : str, optional
            A title for the plot

        arrangement : string or tuple, optional (default 'row')
            Arrangement of the subplots to use in constructing composite
            figure. Supported string options are 'row' and 'col', which will
            arrange the subplots into a single row or column. Also supported is
            a tuple corresponding to the shape of the grid to arrange the subplots
            into. When using a tuple to specify a grid that grid is filled rowwise
            from left to right.

        subtitles : list of str, optional (default None)
            A list of strings for the sub-titles on each matrix's subplot.
        """
        super(GateMatricesPlot, self).__init__(ws, self._create, op_matrices, color_min, color_max,
                                               mx_basis_x, mx_basis_y, xlabel, ylabel,
                                               box_labels, colorbar, prec, scale, eb_matrices, title,
                                               arrangement, subtitles)

    def _create(self, op_matrices, color_min, color_max,
                mx_basis_x, mx_basis_y, xlabel, ylabel,
                box_labels, colorbar, prec, scale, eb_matrices,
                title, arrangement, subtitles):

        return _opmatrices_color_boxplot(
            op_matrices, color_min, color_max, mx_basis_x, mx_basis_y,
            xlabel, ylabel, box_labels, colorbar, prec, scale, eb_matrices,
            title, arrangement, subtitles)

class MatrixPlot(WorkspacePlot):
    """
    Plot of a general matrix using colored boxes

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    matrix : ndarray
      The operation matrix data to display.

    color_min : float
        Color scale minimum.

    color_max : float
        Color scale maximum.

    xlabels : list, optional
        List of (str) box labels along the x-axis.

    ylabels : list, optional
        List of (str) box labels along the y-axis.

    xlabel : str, optional
        A x-axis label for the plot.

    ylabel : str, optional
        A y-axis label for the plot.

    box_labels : bool, optional
        Whether box labels are displayed.

    colorbar : bool optional
        Whether to display a color bar to the right of the box plot.  If None,
        then a colorbar is displayed when `box_labels == False`.

    colormap : Colormap, optional
        A color map object used to convert the numerical matrix values into
        colors.

    prec : int or {'compact','compacthp'}, optional
        Precision for box labels.  Only relevant when box_labels == True. Allowed
        values are:

        - 'compact' = round to nearest whole number using at most 3 characters
        - 'compacthp' = show as much precision as possible using at most 3 characters
        - int >= 0 = fixed precision given by int
        - int <  0 = number of significant figures given by -int

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    grid : {"white","black",None}
        What color grid lines, if any, to add to the plot.  Advanced usage
        allows the addition of `:N` where `N` is an integer giving the line
        width.

    title : str, optional
        A title for the plot

    eb_matrix : numpy array, optional
        An array, of the same size as `op_matrix`, which gives error bars to be
        be displayed in the hover info.
    """

    def __init__(self, ws, matrix, color_min=-1.0, color_max=1.0,
                 xlabels=None, ylabels=None, xlabel=None, ylabel=None,
                 box_labels=False, colorbar=None, colormap=None, prec=0,
                 scale=1.0, grid="black", title= None, eb_matrix= None):
        """
        Creates a color box plot of a matrix using the given color map.

        This can be a useful way to display large matrices which have so many
        entries that their entries cannot easily fit within the width of a page.

        Parameters
        ----------
        matrix : ndarray
            The operation matrix data to display.

        color_min, color_max : float, optional
            Min and max values of the color scale.

        xlabels, ylabels: list, optional
            List of (str) box labels for each axis.

        xlabel : str, optional
            A x-axis label for the plot.

        ylabel : str, optional
            A y-axis label for the plot.

        box_labels : bool, optional
            Whether box labels are displayed.

        colorbar : bool optional
            Whether to display a color bar to the right of the box plot.  If None,
            then a colorbar is displayed when `box_labels == False`.

        colormap : Colormap, optional
            A color map object used to convert the numerical matrix values into
            colors.

        prec : int or {'compact','compacthp'}, optional
            Precision for box labels.  Only relevant when box_labels == True. Allowed
            values are:

            - 'compact' = round to nearest whole number using at most 3 characters
            - 'compacthp' = show as much precision as possible using at most 3 characters
            - int >= 0 = fixed precision given by int
            - int <  0 = number of significant figures given by -int

        scale : float, optional
            Scaling factor to adjust the size of the final figure.

        grid : {"white","black",None}
            What color grid lines, if any, to add to the plot.  Advanced usage
            allows the addition of `:N` where `N` is an integer giving the line
            width.
        
        title : str, optional
            A title for the plot

        eb_matrix : numpy array, optional
            An array, of the same size as `op_matrix`, which gives error bars to be
            be displayed in the hover info.
        """
        super(MatrixPlot, self).__init__(ws, self._create, matrix, color_min, color_max,
                                         xlabels, ylabels, xlabel, ylabel,
                                         box_labels, colorbar, colormap, prec, scale, grid, title,
                                         eb_matrix)

    def _create(self, matrix, color_min, color_max,
                xlabels, ylabels, xlabel, ylabel,
                box_labels, colorbar, colormap, prec, scale, grid,title,
                eb_matrix):

        if colormap is None:
            colormap = _colormaps.DivergingColormap(vmin=color_min, vmax=color_max)

        ret = _matrix_color_boxplot(
            matrix, xlabels, ylabels, xlabel, ylabel,
            box_labels, None, colorbar, colormap, prec, scale, grid=grid, title=title,
            eb_matrix=eb_matrix)
        return ret

class MatricesPlot(WorkspacePlot):
    """
    Plot of a set of general matrices using colored boxes.
    More general than :class:`MatrixPlot` as this allows
    plotting multiple matrices in a single figure using
    subplots.

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    matrices : list of ndarrays
      The data for operation matrices to display.

    color_min : float
        Color scale minimum.

    color_max : float
        Color scale maximum.

    xlabels : list, optional
        List of (str) box labels along the x-axis.

    ylabels : list, optional
        List of (str) box labels along the y-axis.

    xlabel : str, optional
        A x-axis label for the plot.

    ylabel : str, optional
        A y-axis label for the plot.

    box_labels : bool, optional
        Whether box labels are displayed.

    colorbar : bool optional
        Whether to display a color bar to the right of the box plot.  If None,
        then a colorbar is displayed when `box_labels == False`.

    colormap : Colormap, optional
        A color map object used to convert the numerical matrix values into
        colors.

    prec : int or {'compact','compacthp'}, optional
        Precision for box labels.  Only relevant when box_labels == True. Allowed
        values are:

        - 'compact' = round to nearest whole number using at most 3 characters
        - 'compacthp' = show as much precision as possible using at most 3 characters
        - int >= 0 = fixed precision given by int
        - int <  0 = number of significant figures given by -int

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    grid : {"white","black",None}
        What color grid lines, if any, to add to the plot.  Advanced usage
        allows the addition of `:N` where `N` is an integer giving the line
        width.
    
    arrangement : string or tuple, optional (default 'row')
        Arrangement of the subplots to use in constructing composite
        figure. Supported string options are 'row' and 'col', which will
        arrange the subplots into a single row or column. Also supported is
        a tuple corresponding to the shape of the grid to arrange the subplots
        into. When using a tuple to specify a grid that grid is filled rowwise
        from left to right.

    eb_matrices : list of numpy arrays, optional
        List of arrays, of the same size as elements of `op_matrices`, which gives 
        error bars to be be displayed in the hover info.
        
    title : str, optional
        A title for the plot    

    subtitles : list of str, optional (default None)
        A list of strings for the sub-titles on each matrix's subplot.
    """

    def __init__(self, ws, matrices, color_min=-1.0, color_max=1.0,
                 xlabels=None, ylabels=None, xlabel=None, ylabel=None,
                 box_labels=False, colorbar=None, colormap=None, prec=0,
                 scale=1.0, grid="black", arrangement='row', eb_matrices = None, 
                 title=None, subtitles=None):
        """
        Creates a color box plot of a matrix using the given color map.

        This can be a useful way to display large matrices which have so many
        entries that their entries cannot easily fit within the width of a page.

        Parameters
        ----------
        matrix : ndarray
            The operation matrix data to display.

        color_min, color_max : float, optional
            Min and max values of the color scale.

        xlabels, ylabels: list, optional
            List of (str) box labels for each axis.

        xlabel : str, optional
            A x-axis label for the plot.

        ylabel : str, optional
            A y-axis label for the plot.

        box_labels : bool, optional
            Whether box labels are displayed.

        colorbar : bool optional
            Whether to display a color bar to the right of the box plot.  If None,
            then a colorbar is displayed when `box_labels == False`.

        colormap : Colormap, optional
            A color map object used to convert the numerical matrix values into
            colors.

        prec : int or {'compact','compacthp'}, optional
            Precision for box labels.  Only relevant when box_labels == True. Allowed
            values are:

            - 'compact' = round to nearest whole number using at most 3 characters
            - 'compacthp' = show as much precision as possible using at most 3 characters
            - int >= 0 = fixed precision given by int
            - int <  0 = number of significant figures given by -int

        scale : float, optional
            Scaling factor to adjust the size of the final figure.

        grid : {"white","black",None}
            What color grid lines, if any, to add to the plot.  Advanced usage
            allows the addition of `:N` where `N` is an integer giving the line
            width.
        
        arrangement : string or tuple, optional (default 'row')
            Arrangement of the subplots to use in constructing composite
            figure. Supported string options are 'row' and 'col', which will
            arrange the subplots into a single row or column. Also supported is
            a tuple corresponding to the shape of the grid to arrange the subplots
            into. When using a tuple to specify a grid that grid is filled rowwise
            from left to right.

        eb_matrices : list of numpy arrays, optional
            List of arrays, of the same size as elements of `op_matrices`, which gives 
            error bars to be be displayed in the hover info.

        title : str, optional
            A title for the plot

        subtitles : list of str, optional (default None)
            A list of strings for the sub-titles on each matrix's subplot.
        """
        super(MatricesPlot, self).__init__(ws, self._create, matrices, color_min, color_max,
                                         xlabels, ylabels, xlabel, ylabel,box_labels, 
                                         colorbar, colormap, prec, scale, grid, arrangement,
                                         eb_matrices, title, subtitles)

    def _create(self, matrices, color_min, color_max,
                xlabels, ylabels, xlabel, ylabel,
                box_labels, colorbar, colormap, prec, scale, grid, arrangement, eb_matrices,
                title, subtitles):

        if colormap is None:
            colormap = _colormaps.DivergingColormap(vmin=color_min, vmax=color_max)

        ret = _matrices_color_boxplot(
            matrices, xlabels, ylabels, xlabel, ylabel,
            box_labels, None, colorbar, colormap, prec, scale, grid=grid, 
            arrangement=arrangement, eb_matrices=eb_matrices, title= title, subtitles=subtitles)
        return ret

class PolarEigenvaluePlot(WorkspacePlot):
    """
    Polar plot of complex eigenvalues

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    evals_list : list
        A list of eigenvalue arrays to display.

    colors : list
        A corresponding list of color names to use for arrays given
        by `evals_list` (must have `len(colors) == len(evals_list)`).
        Colors can be standard names, e.g. `"blue"`, or rgb strings
        such as `"rgb(23,92,64)"`.

    labels : list, optional
        A list of labels, one for each element of `evals_list` to be
        placed in the plot legend.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    amp : float, optional
        An amount to amplify (raise to the exponent `amp`) each set of
        eigenvalues.  (Amplified eigenvalues are shown in the same color
        but with smaller markers.) If `amp` is None, no amplification is
        performed.

    center_text : str, optional
        Text to be placed at the very center of the polar plot (sometimes
        useful to use as a title).
    """

    def __init__(self, ws, evals_list, colors, labels=None, scale=1.0, amp=None,
                 center_text=None):
        """
        Creates a polar plot of one or more sets of eigenvalues (or any complex #s).

        Parameters
        ----------
        evals_list : list
            A list of eigenvalue arrays to display.

        colors : list
            A corresponding list of color names to use for arrays given
            by `evals_list` (must have `len(colors) == len(evals_list)`).
            Colors can be standard names, e.g. `"blue"`, or rgb strings
            such as `"rgb(23,92,64)"`.

        labels : list, optional
            A list of labels, one for each element of `evals_list` to be
            placed in the plot legend.

        scale : float, optional
            Scaling factor to adjust the size of the final figure.

        amp : float, optional
            An amount to amplify (raise to the exponent `amp`) each set of
            eigenvalues.  (Amplified eigenvalues are shown in the same color
            but with smaller markers.) If `amp` is None, no amplification is
            performed.

        center_text : str, optional
            Text to be placed at the very center of the polar plot (sometimes
            useful to use as a title).
        """
        super(PolarEigenvaluePlot, self).__init__(ws, self._create, evals_list,
                                                  colors, labels, scale, amp,
                                                  center_text)

    def _create(self, evals_list, colors, labels, scale, amp, center_text):

        annotations = []
        if center_text is not None:
            annotations.append(
                dict(text=center_text,
                     r=0, t=0,
                     font=dict(size=10 * scale,
                               color="black",
                               showarrow=False)
                     ))

        #Note: plotly needs a plain lists for r and t, otherwise it
        # produces javascript [ [a], [b], ... ] instead of [a, b, ...]
        data = []
        for i, evals in enumerate(evals_list):
            color = colors[i] if (colors is not None) else "black"
            trace = go.Scatterpolar(
                r=list(_np.absolute(evals).ravel()),
                theta=list(_np.angle(evals).ravel() * (180.0 / _np.pi)),
                mode='markers',
                marker=dict(
                    color=color,
                    size=45,
                    line=dict(
                        color='white'
                    ),
                    opacity=0.7
                ))
            if labels is None or len(labels[i]) == 0:
                trace.update(showlegend=False)
            else:
                trace.update(name=labels[i])
            data.append(trace)

            #Add amplified eigenvalues
            if amp is not None:
                amp_evals = evals**amp
                trace = go.Scatterpolar(
                    r=list(_np.absolute(amp_evals).ravel()),
                    theta=list(_np.angle(amp_evals).ravel() * (180.0 / _np.pi)),
                    showlegend=False,
                    mode='markers',
                    marker=dict(
                        color=color,
                        size=20,
                        line=dict(
                            color='white'
                        ),
                        opacity=0.5
                    ))
                #if labels is not None:
                #    trace.update(name="%s^%g" % (labels[i],amp))
                data.append(trace)

        layout = go.Layout(
            width=300 * scale,
            height=300 * scale,
            #title='Test Polar',
            #font=dict(size=10),
            plot_bgcolor='rgb(240, 240, 240)',
            polar=dict(
                radialaxis=dict(
                    range=[0, 1.25]),
                angularaxis=dict(
                    tickcolor='rgb(180,180,180)',
                    #range=[0,2]
                    #ticktext=['A','B','C','D']
                    direction="counterclockwise",
                    rotation=-90,
                ),
            ),
        )

        #HACK around plotly bug: Plotly somehow holds residual polar plot data
        # which gets plotted unless new data overwrites it.  This residual data
        # takes up 4 points of data for the first 3 data traces - so we make
        # sure this data is filled in with undisplayed data here (because we
        # don't want to see the residual data!).
        for trace in data:
            if len(trace['r']) < 4:  # hopefully never needed
                extra = 4 - len(trace['r'])  # pragma: no cover
                trace['r'] += [1e3] * extra  # pragma: no cover
                trace['t'] += [0.0] * extra  # pragma: no cover
        while len(data) < 3:
            data.append(go.Scatterpolar(
                r=[1e3] * 4,
                theta=[0.0] * 4,
                name="Dummy",
                mode='markers',
                showlegend=False,
            ))
        assert(len(data) >= 3)

        pythonVal = {}
        for i, tr in enumerate(data):
            key = tr['name'] if ("name" in tr) else "trace%d" % i
            pythonVal[key] = {'r': tr['r'], 'theta': tr['theta']}

        return ReportFigure(go.Figure(data=data, layout=layout),
                            None, pythonVal)


class ProjectionsBoxPlot(WorkspacePlot):
    """
    Plot of matrix of (usually error-generator) projections

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    projections : ndarray
        A 1-dimensional array of length equal to the numer of elements in
        the given basis (usually equal to the gate dimension).  Ordering of
        the values is assumed to correspond to the ordering given by the
        routines in `pygsti.tools`, (e.g. :func:`pp_matrices` when
        `projection_basis` equals "pp").

    projection_basis : {'std', 'gm', 'pp', 'qt'}
        The basis is used to construct the error generators onto which
        the gate  error generator is projected.  Allowed values are
        Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp) and Qutrit (qt).

    color_min : float, optional
        Minimum value of the color scale. If None, then computed
        automatically from the data range.

    color_max : float, optional
        Maximum value of the color scale. If None, then computed
        automatically from the data range.

    box_labels : bool, optional
        Whether box labels are displayed.

    colorbar : bool optional
        Whether to display a color bar to the right of the box plot.  If None,
        then a colorbar is displayed when `box_labels == False`.

    prec : int or {'compact','compacthp'}, optional
        Precision for box labels.  Only relevant when box_labels == True. Allowed
        values are:

        - 'compact' = round to nearest whole number using at most 3 characters
        - 'compacthp' = show as much precision as possible using at most 3 characters
        - int >= 0 = fixed precision given by int
        - int <  0 = number of significant figures given by -int

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    eb_matrix : numpy array, optional
        An array, of the same size as `projections`, which gives error bars to be
        be displayed in the hover info.

    title : str, optional
        A title for the plot
    """

    def __init__(self, ws, projections, projection_basis, color_min=None, color_max=None,
                 box_labels=False, colorbar=None, prec="compacthp", scale=1.0,
                 eb_matrix=None, title=None, proj_type=None):
        """
        Creates a color box plot displaying projections.

        Typically `projections` is obtained by calling
        :func:`std_errorgen_projections`, and so holds the projections of a gate
        error generator onto the generators corresponding to a set of standard
        errors constructed from the given basis.

        Parameters
        ----------
        projections : ndarray
          A 1-dimensional array of length equal to the numer of elements in
          the given basis (usually equal to the gate dimension).  Ordering of
          the values is assumed to correspond to the ordering given by the
          routines in `pygsti.tools`, (e.g. :func:`pp_matrices` when
          `projection_basis` equals "pp").

        projection_basis : {'std', 'gm', 'pp', 'qt'}
          The basis is used to construct the error generators onto which
          the gate  error generator is projected.  Allowed values are
          Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp) and Qutrit (qt).

        color_min,color_max : float, optional
          Color scale min and max values, respectivey.  If None, then computed
          automatically from the data range.

        box_labels : bool, optional
          Whether box labels are displayed.

        colorbar : bool optional
          Whether to display a color bar to the right of the box plot.  If None,
          then a colorbar is displayed when `box_labels == False`.

        prec : int or {'compact','compacthp'}, optional
            Precision for box labels.  Only relevant when box_labels == True. Allowed
            values are:

            - 'compact' = round to nearest whole number using at most 3 characters
            - 'compacthp' = show as much precision as possible using at most 3 characters
            - int >= 0 = fixed precision given by int
            - int <  0 = number of significant figures given by -int

        scale : float, optional
            Scaling factor to adjust the size of the final figure.

        eb_matrix : numpy array, optional
            An array, of the same size as `projections`, which gives error bars to be
            be displayed in the hover info.

        title : str, optional
            A title for the plot

        proj_type : str, optional
            Optional flag for controlling special formatting behavior on certain
            project types. E.g. enabling additional annotations when plotting CA
            error generator projections.
        """
        super(ProjectionsBoxPlot, self).__init__(ws, self._create, projections,
                                                 projection_basis, color_min, color_max,
                                                 box_labels, colorbar, prec, scale,
                                                 eb_matrix, title, proj_type)

    def _create(self, projections,
                projection_basis, color_min, color_max,
                box_labels, colorbar, prec, scale,
                eb_matrix, title, proj_type):
        
        masked_projections = _np.ma.array(projections, mask=_np.isnan(projections))
        absMax = _np.max(_np.abs(masked_projections))
        if color_min is None: color_min = -absMax
        if color_max is None: color_max = absMax

        d2 = len(projections) + 1  # number of projections == dim of gate  (+1 b/c identity is not included)
        d = _np.sqrt(d2)  # dim of density matrix
        nQubits = _np.log2(d)  # note: 4^nQ = d2

        if not _np.isclose(round(nQubits), nQubits) and projections.size == d2 - 1:
            #Non-integral # of qubits, and a single projection axis (H or S, not CA), so just show as a single row
            projections = projections.reshape((1, projections.size))
            xlabel = ""; ylabel = ""
            yd, xd = projections.shape
        elif nQubits == 1:
            if projections.size == 3:
                projections = projections.reshape((1, 3))
                xlabel = "Q1"; ylabel = ""
                xd, yd = 4, 1  # include identity in basis dimensions
            else:  # projections.size == 3*3 = 9
                projections = projections.reshape((3, 3))
                xlabel = ""; ylabel = ""
                xd = yd = 4
        elif nQubits == 2:
            if projections.size == 15:
                projections = _np.concatenate(([0.0], projections)).reshape((4, 4))
                eb_matrix = _np.concatenate(([0.0], eb_matrix)) if (eb_matrix is not None) else None
                xlabel = "Q2"; ylabel = "Q1"
                yd, xd = projections.shape
            else:  # projections.size == 15*15
                projections = projections.reshape((15, 15))
                xlabel = ""; ylabel = ""
                xd = yd = 16  # include identity in basis dimensions
        else:
            if projections.size == d2 - 1:  # == 4**nQubits - 1
                projections = _np.concatenate(([0.0], projections)).reshape((4, (projections.size+1) // 4))
                eb_matrix = _np.concatenate(([0.0], eb_matrix)) if (eb_matrix is not None) else None
                xlabel = "Q*"; ylabel = "Q1"
                yd, xd = projections.shape
            else:  # projections.size == (d2-1)**2 == (4**nQ-1)**2  (CA-size square, works for non-integral nQubits too)
                projections = projections.reshape((d2 - 1, d2 - 1))
                xlabel = ""; ylabel = ""
                xd = yd = d2  # 4**nQubits

        if eb_matrix is not None:
            eb_matrix = eb_matrix.reshape(projections.shape)

        if isinstance(projection_basis, _baseobjs.Basis):
            if isinstance(projection_basis, _baseobjs.TensorProdBasis) and len(projection_basis.component_bases) == 2 \
               and xd == projection_basis.component_bases[0].dim and yd == projection_basis.component_bases[1].dim:
                basis_for_xlabels = projection_basis.component_bases[0]
                basis_for_ylabels = projection_basis.component_bases[1]
            elif xd == projection_basis.dim and yd == 1:
                basis_for_xlabels = projection_basis
                basis_for_ylabels = None
            elif xd == yd == projection_basis.dim:
                basis_for_xlabels = projection_basis
                basis_for_ylabels = projection_basis
            else:
                try:
                    basis_for_xlabels = _baseobjs.BuiltinBasis(projection_basis.name, xd)
                    basis_for_ylabels = _baseobjs.BuiltinBasis(projection_basis.name, yd)
                except:
                    basis_for_xlabels = basis_for_ylabels = None
        else:
            try:
                basis_for_xlabels = _baseobjs.BuiltinBasis(projection_basis, xd)
                basis_for_ylabels = _baseobjs.BuiltinBasis(projection_basis, yd)
            except:
                basis_for_xlabels = basis_for_ylabels = None

        fig = _opmatrix_color_boxplot(projections, color_min, color_max,
                                      basis_for_xlabels, basis_for_ylabels,
                                      xlabel, ylabel, box_labels, colorbar, 
                                      prec, scale, eb_matrix, title)

        #if plotting CA error generator projections add some additional
        #shapes to delineate between the C and A parts of the array.
        if proj_type == 'CA':
            #--------bottom triangle--------#
            #bottom vertical
            fig.plotlyfig.add_shape(type='line', xref='paper', yref='paper', 
                                    x0=-0.06, y0=-0.06, x1=-0.06, y1=1.03,
                                    line={'color':"#2cb802", 'width':1, 'dash':"dot"})
            #bottom horizontal
            fig.plotlyfig.add_shape(type='line', xref='paper', yref='paper', 
                                    x0=-0.06, y0=-0.06, x1=0.80, y1=-0.06,
                                    line={'color':"#2cb802", 'width':1, 'dash':"dot"})
            #bottom diagonal
            fig.plotlyfig.add_shape(type='line', xref='paper', yref='paper', 
                                    x0=-0.06, y0=1.03, x1=1.03, y1=-0.06,
                                    line={'color':"#2cb802", 'width':1, 'dash':"dot"})
            #Add a tab-like area for the annotation
            fig.plotlyfig.add_shape(type='line', xref='paper', yref='paper', 
                                    x0=0.80, y0=-0.06, x1=0.80, y1=-0.26,
                                    line={'color':"#2cb802", 'width':1, 'dash':"dot"})
            fig.plotlyfig.add_shape(type='line', xref='paper', yref='paper', 
                                    x0=0.80, y0=-0.26, x1=1.03, y1=-0.26,
                                    line={'color':"#2cb802", 'width':1, 'dash':"dot"})
            fig.plotlyfig.add_shape(type='line', xref='paper', yref='paper', 
                                    x0=1.03, y0=-0.26, x1=1.03, y1=-0.06,
                                    line={'color':"#2cb802", 'width':1, 'dash':"dot"})


            #----------top triangle---------#
            #top horizontal
            fig.plotlyfig.add_shape(type='line', xref='paper', yref='paper', 
                                    x0=-0.04, y0=1.06, x1=1.06, y1=1.06,
                                    line={'color':"#6002b8", 'width':1, 'dash':"dot"})
            #top vertical
            fig.plotlyfig.add_shape(type='line', xref='paper', yref='paper', 
                                    x0=1.06, y0=1.06, x1=1.06, y1=0.2,
                                    line={'color':"#6002b8", 'width':1, 'dash':"dot"})
            #top diagonal
            fig.plotlyfig.add_shape(type='line', xref='paper', yref='paper', 
                                    x0=-0.04, y0=1.06, x1=1.06, y1=-0.03,
                                    line={'color':"#6002b8", 'width':1, 'dash':"dot"})
            #Add a tab-like area for the annotation
            fig.plotlyfig.add_shape(type='line', xref='paper', yref='paper', 
                                    x0=1.06, y0=0.2, x1=1.26, y1=0.2,
                                    line={'color':"#6002b8", 'width':1, 'dash':"dot"})
            fig.plotlyfig.add_shape(type='line', xref='paper', yref='paper', 
                                    x0=1.26, y0=0.2, x1=1.26, y1=-0.03,
                                    line={'color':"#6002b8", 'width':1, 'dash':"dot"})
            fig.plotlyfig.add_shape(type='line', xref='paper', yref='paper', 
                                    x0=1.26, y0=-0.03, x1=1.06, y1=-0.03,
                                    line={'color':"#6002b8", 'width':1, 'dash':"dot"})
            
            #adjust margins to allow space to annotations
            new_bmargin = fig.plotlyfig.layout.margin.b + 20
            new_height = fig.plotlyfig.layout.height + 20
            new_rmargin = fig.plotlyfig.layout.margin.r + 20
            #the CA plots currently have excessive area on the left, so steal
            #some of this area from there and leave the width alone.
            new_lmargin = fig.plotlyfig.layout.margin.l - 20

            #add annotations for the upper and lower triangles.
            fig.plotlyfig.add_annotation(x= 0.92, y= -0.16, xref= 'paper', yref= 'paper',
                                         text= 'A', showarrow=False, xanchor='center',
                                         yanchor='middle', font=dict(color='#2cb802'))
            fig.plotlyfig.add_annotation(x= 1.16, y= 0.1, xref= 'paper', yref= 'paper',
                                         text= 'C', showarrow=False, xanchor='center',
                                         yanchor='middle', font=dict(color='#6002b8'))

            
            #hack in extra spacing between tick labels and axes (plotly does not
            #appear to have a handle on this presently). A natural way to do this
            #would be with ticksuffix, but this does not work in tandem with ticktext
            #being set.
            new_yaxis_ticktext = [text + (' '*5) for text in fig.plotlyfig.layout.yaxis.ticktext]
            new_xaxis_ticktext = [text+ ('<br>'*1) for text in fig.plotlyfig.layout.xaxis.ticktext]
            
            
            fig.plotlyfig.update_layout(yaxis_ticktext=new_yaxis_ticktext,
                                        xaxis_ticktext=new_xaxis_ticktext,
                                        margin_b= new_bmargin,
                                        margin_r= new_rmargin,
                                        margin_l= new_lmargin,
                                        height = new_height)
            
        return fig


class ChoiEigenvalueBarPlot(WorkspacePlot):
    """
    Bar plot of eigenvalues showing red bars for negative values

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    evals : ndarray
       An array containing the eigenvalues to plot.

    errbars : ndarray, optional
       An array containing the lengths of the error bars
       to place on each bar of the plot.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.
    """

    def __init__(self, ws, evals, errbars=None, scale=1.0):
        """
        Creates a bar plot showing the real parts of each of the eigenvalues
        given.  This is useful for plotting the eigenvalues of Choi matrices,
        since all elements are positive for a CPTP map.

        Parameters
        ----------
        evals : ndarray
           An array containing the eigenvalues to plot.

        errbars : ndarray, optional
           An array containing the lengths of the error bars
           to place on each bar of the plot.

        scale : float, optional
            Scaling factor to adjust the size of the final figure.
        """
        super(ChoiEigenvalueBarPlot, self).__init__(ws, self._create, evals,
                                                    errbars, scale)

    def _create(self, evals, errbars, scale):
        if errbars is not None:
            flat_errbars = errbars.ravel()
        HOVER_PREC = 7
        xs = list(range(evals.size))
        ys, colors, texts = [], [], []
        for i, ev in enumerate(evals.ravel()):
            ys.append(abs(ev.real))
            colors.append('rgb(200,200,200)' if ev.real > 0 else 'red')
            if errbars is not None:
                texts.append("%g +/- %g" % (round(ev.real, HOVER_PREC),
                                            round(flat_errbars[i].real, HOVER_PREC)))
            else:
                texts.append("%g" % round(ev.real, HOVER_PREC))

        trace = go.Bar(
            x=xs, y=ys, text=texts,
            marker=dict(color=colors),
            hoverinfo='text'
        )

        LOWER_LOG_THRESHOLD = -6  # so don't plot all the way down to, e.g., 1e-13
        ys = _np.clip(ys, 1e-30, 1e100)  # to avoid log(0) errors
        log_ys = _np.log10(_np.array(ys, 'd'))
        minlog = max(_np.floor(min(log_ys)) - 0.1, LOWER_LOG_THRESHOLD)
        maxlog = max(_np.ceil(max(log_ys)), minlog + 1)

        #Set plot size and margins
        lmargin = rmargin = tmargin = bmargin = 10
        lmargin += 30  # for y-tics
        bmargin += 40  # for x-tics & xlabel

        width = lmargin + max(20 * len(xs), 120) + rmargin
        height = tmargin + 120 + bmargin

        width *= scale
        height *= scale
        lmargin *= scale
        rmargin *= scale
        tmargin *= scale
        bmargin *= scale

        data = [trace]
        layout = go.Layout(
            width=width,
            height=height,
            margin=go_margin(l=lmargin, r=rmargin, b=bmargin, t=tmargin),
            xaxis=dict(
                title={'text':"index"},
                tickvals=xs
            ),
            yaxis=dict(
                type='log',
                range=[minlog, maxlog]
            ),
            bargap=0.02
        )

        return ReportFigure(go.Figure(data=data, layout=layout),
                            None, evals, plt_y=evals, plt_yerr=errbars,
                            pythonErrorBar=errbars)


class GramMatrixBarPlot(WorkspacePlot):
    """
    Bar plot of Gram matrix eigenvalues stacked against those of target

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    dataset : DataSet
        The DataSet

    target : Model
        A target model which is used for it's mapping of SPAM labels to
        SPAM specifiers and for Gram matrix comparision.

    maxlen : integer, optional
        The maximum length string used when searching for the
        maximal (best) Gram matrix.  It's useful to make this
        at least twice the maximum length fiducial sequence.

    fixed_lists : (prep_fiducials, meas_fiducials), optional
        2-tuple of circuit lists, specifying the preparation and
        measurement fiducials to use when constructing the Gram matrix,
        and thereby bypassing the search for such lists.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.
    """

    def __init__(self, ws, dataset, target, maxlen=10,
                 fixed_lists=None, scale=1.0):
        """
        Creates a bar plot showing eigenvalues of the Gram matrix compared to
        those of the a target model's Gram matrix.

        Parameters
        ----------
        dataset : DataSet
            The DataSet

        target : Model
            A target model which is used for it's mapping of SPAM labels to
            SPAM specifiers and for Gram matrix comparision.

        maxlen : integer, optional
            The maximum length string used when searching for the
            maximal (best) Gram matrix.  It's useful to make this
            at least twice the maximum length fiducial sequence.

        fixed_lists : (prep_fiducials, meas_fiducials), optional
            2-tuple of circuit lists, specifying the preparation and
            measurement fiducials to use when constructing the Gram matrix,
            and thereby bypassing the search for such lists.

        scale : float, optional
            Scaling factor to adjust the size of the final figure.
        """
        super(GramMatrixBarPlot, self).__init__(ws, self._create,
                                                dataset, target, maxlen, fixed_lists, scale)

    def _create(self, dataset, target, maxlen, fixed_lists, scale):

        if fixed_lists is not None and \
                (len(fixed_lists[0]) == 0 or len(fixed_lists[1]) == 0):
            #Empty fixed lists => create empty gram plot
            svals = target_svals = _np.array([], 'd')
        else:
            _, svals, target_svals = _alg.max_gram_rank_and_eigenvalues(dataset, target, maxlen, fixed_lists)
            svals = _np.sort(_np.abs(svals)).reshape(-1, 1)
            target_svals = _np.sort(_np.abs(target_svals)).reshape(-1, 1)

        xs = list(range(svals.size))
        trace1 = go.Bar(
            x=xs, y=list(svals.ravel()),
            marker=dict(color="blue"),
            hoverinfo='y',
            name="from Data"
        )
        trace2 = go.Bar(
            x=xs, y=list(target_svals.ravel()),
            marker=dict(color="black"),
            hoverinfo='y',
            name="from Target"
        )

        if svals.size > 0:
            ymin = min(_np.min(svals), _np.min(target_svals))
            ymax = max(_np.max(svals), _np.max(target_svals))
            ymin = max(ymin, 1e-8)  # prevent lower y-limit from being riduculously small
        else:
            ymin = 0.1
            ymax = 1.0  # just pick some values for empty plot

        data = [trace1, trace2]
        layout = go.Layout(
            width=400 * scale,
            height=300 * scale,
            xaxis=dict(
                title={'text':"index"},
                tickvals=xs
            ),
            yaxis=dict(
                title={'text': "eigenvalue"},
                type='log',
                exponentformat='power',
                range=[_np.log10(ymin), _np.log10(ymax)],
            ),
            bargap=0.1
        )

        pythonVal = {}
        for tr in data:
            pythonVal[tr['name']] = tr['y']
        return ReportFigure(go.Figure(data=data, layout=layout),
                            None, pythonVal)


class FitComparisonBarPlot(WorkspacePlot):
    """
    Bar plot showing the overall (aggregate) goodness of fit (along one dimension).

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    x_names : list
        List of x-values. Typically these are the integer maximum lengths or
        exponents used to index the different iterations of GST, but they
        can also be strings.

    circuits_by_x : list of (CircuitLists or lists of Circuits)
        Specifies the set of circuits used at each x-value.

    model_by_x : list of Models
        `Model` corresponding to each x-value.

    dataset_by_x : DataSet or list of DataSets
        The data sets to compare each model against.  If a single
        :class:`DataSet` is given, then it is used for all comparisons.

    objfn_builder : ObjectiveFunctionBuilder or {"logl", "chi2"}, optional
        The objective function to use, or one of the given strings
        to use a defaut log-likelihood or chi^2 function.

    x_label : str, optional
        A label for the 'x' variable which indexes the different models.
        This string will be the x-label of the resulting bar plot.

    np_by_x : list of ints, optional
        A list of parameter counts to use for each x.  If None, then
        the number of non-gauge parameters for each model is used.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    wildcard : WildcardBudget
        A wildcard budget to apply to the objective function (`objective`),
        which increases the goodness of fit by adjusting (by an amount
        measured in TVD) the probabilities produced by a model before
        comparing with the frequencies in `dataset`.  Currently, this
        functionality is only supported for `objective == "logl"`.

    mdc_stores : list of ModelDatasetCircuitStore, optional (default None)
        A optional list of precomputed ModelDatasetCircuitStore objects
        of length equal to the length of `model_by_x` and `circuits_by_x` 
        (and with internal values for the model, circuits and dataset assumed
        to be commensurate with the corresponding input values) used to
        accelerate objective function construction.
    """

    def __init__(self, ws, x_names, circuits_by_x, model_by_x, dataset_by_x,
                 objfn_builder="logl", x_label='L', np_by_x=None, scale=1.0,
                 comm=None, wildcard=None, mdc_stores=None):
        """
        Creates a bar plot showing the overall (aggregate) goodness of fit
        for one or more model estimates to corresponding data sets.

        Parameters
        ----------
        x_names : list
            List of x-values. Typically these are the integer maximum lengths or
            exponents used to index the different iterations of GST, but they
            can also be strings.

        circuits_by_x : list of (CircuitLists or lists of Circuits)
            Specifies the set of circuits used at each x-value.

        model_by_x : list of Models
            `Model` corresponding to each x-value.

        dataset_by_x : DataSet or list of DataSets
            The data sets to compare each model against.  If a single
            :class:`DataSet` is given, then it is used for all comparisons.

        objfn_builder : ObjectiveFunctionBuilder or {"logl", "chi2"}, optional
            The objective function to use, or one of the given strings
            to use a defaut log-likelihood or chi^2 function.

        x_label : str, optional
            A label for the 'x' variable which indexes the different models.
            This string will be the x-label of the resulting bar plot.

        np_by_x : list of ints, optional
            A list of parameter counts to use for each x.  If None, then
            the number of non-gauge parameters for each model is used.

        scale : float, optional
            Scaling factor to adjust the size of the final figure.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        wildcard : WildcardBudget
            A wildcard budget to apply to the objective function (`objective`),
            which increases the goodness of fit by adjusting (by an amount
            measured in TVD) the probabilities produced by a model before
            comparing with the frequencies in `dataset`.  Currently, this
            functionality is only supported for `objective == "logl"`.

        mdc_stores : list of ModelDatasetCircuitStore, optional (default None)
            An optional list of precomputed ModelDatasetCircuitStore objects
            of length equal to the length of `model_by_x` and `circuits_by_x` 
            (and with internal values for the model, circuits and dataset assumed
            to be commensurate with the corresponding input values) used to
            accelerate objective function construction.
        """
        super(FitComparisonBarPlot, self).__init__(ws, self._create,
                                                   x_names, circuits_by_x, model_by_x, dataset_by_x,
                                                   objfn_builder, x_label, np_by_x, scale,
                                                   comm, wildcard, mdc_stores)

    def _create(self, x_names, circuits_by_x, model_by_x, dataset_by_x, objfn_builder, x_label,
                np_by_x, scale, comm, wildcard, mdc_stores):

        xs = list(range(len(x_names)))
        xtics = []; ys = []; colors = []; texts = []

        if np_by_x is None:
            np_by_x = [mdl.num_modeltest_params if (mdl is not None) else 0
                       for mdl in model_by_x]  # Note: models can be None => N/A

        if isinstance(dataset_by_x, _DataSet):
            dataset_by_x = [dataset_by_x] * len(model_by_x)
        
        if mdc_stores is None:
            mdc_stores = [None]*len(model_by_x)

        for X, mdl, circuits, dataset, Np, mdc_store in zip(
            x_names, model_by_x, circuits_by_x, dataset_by_x, np_by_x, mdc_stores
        ):
            if circuits is None or mdl is None:
                Nsig, rating = _np.nan, 5
            else:
                Nsig, rating, _, _, _, _ = self._ccompute(_ph.rated_n_sigma, dataset, mdl,
                                                          circuits, objfn_builder, Np, wildcard,
                                                          return_all=True, comm=comm, mdc_store=mdc_store)
                # Note: don't really need return_all=True, but helps w/caching b/c other fns use it.

            if rating == 5: color = "darkgreen"
            elif rating == 4: color = "lightgreen"
            elif rating == 3: color = "yellow"
            elif rating == 2: color = "orange"
            else: color = "red"

            xtics.append(str(X))
            ys.append(Nsig)
            texts.append("%g<br>rating: %d" % (Nsig, rating))
            colors.append(color)

        MIN_BAR = 1e-4  # so all bars have positive height (since log scale)
        plotted_ys = [max(y, MIN_BAR) for y in ys]
        trace = go.Bar(
            x=xs, y=plotted_ys, text=texts,
            marker=dict(color=colors),
            hoverinfo='text'
        )

        #Set plot size and margins
        lmargin = rmargin = tmargin = bmargin = 10
        if x_label: bmargin += 20
        lmargin += 20  # y-label is always present
        if xtics:
            max_xl = max([len(xl) for xl in xtics])
            if max_xl > 0: bmargin += max_xl * 5
        lmargin += 20  # for y-labels (guess)

        width = lmargin + max(30 * len(xs), 150) + rmargin
        height = tmargin + 200 + bmargin

        width *= scale
        height *= scale
        lmargin *= scale
        rmargin *= scale
        tmargin *= scale
        bmargin *= scale

        data = [trace]
        layout = go.Layout(
            width=width,
            height=height,
            margin=go_margin(l=lmargin, r=rmargin, b=bmargin, t=tmargin),
            xaxis=dict(
                title={'text':x_label},
                tickvals=xs,
                ticktext=xtics
            ),
            yaxis=dict(
                title={'text': "N<sub>sigma</sub>"},
                type='log'
            ),
            bargap=0.1
        )
        if len(plotted_ys) == 0:
            layout['yaxis']['range'] = [_np.log10(0.1),
                                        _np.log10(1.0)]  # empty plot: range doesn't matter
        elif max(plotted_ys) < 1.0:
            layout['yaxis']['range'] = [_np.log10(min(plotted_ys) / 2.0),
                                        _np.log10(1.0)]
        else:
            layout['yaxis']['range'] = [_np.log10(min(plotted_ys) / 2.0),
                                        _np.log10(max(plotted_ys) * 2.0)]

        return ReportFigure(go.Figure(data=data, layout=layout),
                            None, {'x': xs, 'y': ys})


class FitComparisonBoxPlot(WorkspacePlot):
    """
    Box plot showing the overall (aggregate) goodness of fit (along 2 dimensions).

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    xs : list
        List of X-values (converted to strings).

    ys : list
        List of Y-values (converted to strings).

    circuits_by_y_then_x : list of lists of PlaquetteGridCircuitStructure objects
        Specifies the circuits used at each Y and X value, indexed as
        `circuits_by_y_then_x[iY][iX]`, where `iX` and `iY`
        are X and Y indices, respectively.

    model_by_y_then_x : list of lists of Models
        `Model` corresponding to each X and Y value.

    dataset_by_y_then_x : list of lists of DataSets
        `DataSet` corresponding to each X and Y value.

    objfn_builder : ObjectiveFunctionBuilder or {"logl", "chi2"}, optional
        The objective function to use, or one of the given strings
        to use a defaut log-likelihood or chi^2 function.

    x_label : str, optional
        Label for the 'X' variable which indexes different models.
        This string will be the x-label of the resulting box plot.

    y_label : str, optional
        Label for the 'Y' variable which indexes different models.
        This string will be the y-label of the resulting box plot.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    wildcard : WildcardBudget
        A wildcard budget to apply to the objective function (`objective`),
        which increases the goodness of fit by adjusting (by an amount
        measured in TVD) the probabilities produced by a model before
        comparing with the frequencies in `dataset`.  Currently, this
        functionality is only supported for `objective == "logl"`.
    
    mdc_stores : list of lists of ModelDatasetCircuitStore, optional (default None)
        An optional list of lists of precomputed ModelDatasetCircuitStore objects
        with a structure matching `model_by_y_then_x` and `circuits_by_y_then_x` 
        (and with internal values for the model, circuits and dataset assumed
        to be commensurate with the corresponding input values) used to
        accelerate objective function construction.
    """

    def __init__(self, ws, xs, ys, circuits_by_y_then_x, model_by_y_then_x, dataset_by_y_then_x,
                 objfn_builder="logl", x_label=None, y_label=None, scale=1.0, comm=None,
                 wildcard=None, mdc_stores=None):
        """
        Creates a box plot showing the overall (aggregate) goodness of fit
        for one or more model estimates to their respective  data sets.

        Parameters
        ----------
        xs, ys : list
            List of X-values and Y-values (converted to strings).

        circuits_by_y_then_x : list of lists of PlaquetteGridCircuitStructure objects
            Specifies the circuits used at each Y and X value, indexed as
            `circuits_by_y_then_x[iY][iX]`, where `iX` and `iY`
            are X and Y indices, respectively.

        model_by_y_then_x : list of lists of Models
            `Model` corresponding to each X and Y value.

        dataset_by_y_then_x : list of lists of DataSets
            `DataSet` corresponding to each X and Y value.

        objfn_builder : ObjectiveFunctionBuilder or {"logl", "chi2"}, optional
            The objective function to use, or one of the given strings
            to use a defaut log-likelihood or chi^2 function.

        x_label, y_label : str, optional
            Labels for the 'X' and 'Y' variables which index the different gate
            sets. These strings will be the x- and y-label of the resulting box
            plot.

        scale : float, optional
            Scaling factor to adjust the size of the final figure.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        wildcard : WildcardBudget
            A wildcard budget to apply to the objective function (`objective`),
            which increases the goodness of fit by adjusting (by an amount
            measured in TVD) the probabilities produced by a model before
            comparing with the frequencies in `dataset`.  Currently, this
            functionality is only supported for `objective == "logl"`.
        
        mdc_stores : list of lists of ModelDatasetCircuitStore, optional (default None)
            An optional list of lists of precomputed ModelDatasetCircuitStore objects
            with a structure matching `model_by_y_then_x` and `circuits_by_y_then_x` 
            (and with internal values for the model, circuits and dataset assumed
            to be commensurate with the corresponding input values) used to
            accelerate objective function construction.
        """
        super(FitComparisonBoxPlot, self).__init__(
            ws, self._create, xs, ys, circuits_by_y_then_x, model_by_y_then_x,
            dataset_by_y_then_x, objfn_builder, x_label, y_label, scale, comm,
            wildcard, mdc_stores)

    def _create(self, xs, ys, circuits_by_yx, model_by_yx, dataset_by_yx, objfn_builder,
                x_label, y_label, scale, comm, wildcard, mdc_stores):

        xlabels = list(map(str, xs))
        ylabels = list(map(str, ys))

        NsigMx = _np.empty((len(ys), len(xs)), 'd')
        cmap = _colormaps.PiecewiseLinearColormap(
            [[0, (0, 0.5, 0)], [2, (0, 0.5, 0)],  # rating=5 darkgreen
             [20, (0, 1.0, 0)],  # rating=4 lightgreen
             [100, (1.0, 1.0, 0)],  # rating=3 yellow
             [500, (1.0, 0.5, 0)],  # rating=2 orange
             [1000, (1.0, 0, 0)]])  # rating=1 red
        
        if mdc_stores is None:
            mdc_stores = [[None for y in ys] for x in xs]

        for iY, Y in enumerate(ys):
            for iX, X in enumerate(xs):
                dataset = dataset_by_yx[iY][iX]
                mdl = model_by_yx[iY][iX]
                circuits = circuits_by_yx[iY][iX]
                mdc_store = mdc_stores[iY][iX]

                if dataset is None or circuits is None or mdl is None:
                    NsigMx[iY][iX] = _np.nan
                    continue

                Nsig, rating, _, _, _, _ = self._ccompute(
                    _ph.rated_n_sigma, dataset, mdl, circuits, objfn_builder,
                    None, wildcard, return_all=True, comm=comm, mdc_store=mdc_store)  # self.ws.smartCache,
                NsigMx[iY][iX] = Nsig

        return _matrix_color_boxplot(
            NsigMx, xlabels, ylabels, x_label, y_label,
            box_labels=True, colorbar=False, colormap=cmap,
            prec='compact', scale=scale, grid="white")


class DatasetComparisonSummaryPlot(WorkspacePlot):
    """
    A grid of grayscale boxes comparing data sets pair-wise.

    This class creates a plot showing the total 2*deltaLogL values for each
    pair of :class:`DataSet` out of some number of total `DataSets`.

    Background: For every pair of data sets, the likelihood is computed for
    two different models: 1) the model in which a single set of
    probabilities (one per gate sequence, obtained by the combined outcome
    frequencies) generates both data sets, and 2) the model in which each
    data is generated from different sets of probabilities.  Twice the ratio
    of these log-likelihoods can be compared to the value that is expected
    when model 1) is valid.  This plot shows the difference between the
    expected and actual twice-log-likelihood ratio in units of standard
    deviations.  Zero or negative values indicate the data sets appear to be
    generated by the same underlying probabilities.  Large positive values
    indicate the data sets appear to be generated by different underlying
    probabilities.


    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    dslabels : list
        A list of data set labels, specifying the ordering and the number
        of data sets.

    dsc_dict : dict
        A dictionary of `DataComparator` objects whose keys are 2-tuples of
        integers such that the value associated with `(i,j)` is a
        `DataComparator` object that compares the `i`-th and `j`-th data
        sets (as indexed by `dslabels`.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.
    """

    def __init__(self, ws, dslabels, dsc_dict, scale=1.0):
        """
        Creates a plot showing the total 2*deltaLogL values for each pair of
        DataSets out of some number of total DataSets.

        Background: For every pair of data sets, the likelihood is computed for
        two different models: 1) the model in which a single set of
        probabilities (one per gate sequence, obtained by the combined outcome
        frequencies) generates both data sets, and 2) the model in which each
        data is generated from different sets of probabilities.  Twice the ratio
        of these log-likelihoods can be compared to the value that is expected
        when model 1) is valid.  This plot shows the difference between the
        expected and actual twice-log-likelihood ratio in units of standard
        deviations.  Zero or negative values indicate the data sets appear to be
        generated by the same underlying probabilities.  Large positive values
        indicate the data sets appear to be generated by different underlying
        probabilities.

        Parameters
        ----------
        dslabels : list
            A list of data set labels, specifying the ordering and the number
            of data sets.

        dsc_dict : dict
            A dictionary of `DataComparator` objects whose keys are 2-tuples of
            integers such that the value associated with `(i,j)` is a
            `DataComparator` object that compares the `i`-th and `j`-th data
            sets (as indexed by `dslabels`.
        """
        super(DatasetComparisonSummaryPlot, self).__init__(ws, self._create, dslabels, dsc_dict, scale)

    def _create(self, dslabels, dsc_dict, scale):
        nSigmaMx = _np.zeros((len(dslabels), len(dslabels)), 'd') * _np.nan
        logLMx = _np.zeros((len(dslabels), len(dslabels)), 'd') * _np.nan
        max_nSigma = max_2DeltaLogL = 0.0
        for i, _ in enumerate(dslabels):
            for j, _ in enumerate(dslabels[i + 1:], start=i + 1):
                dsc = dsc_dict.get((i, j), dsc_dict.get((j, i), None))
                val = dsc.aggregate_nsigma if (dsc is not None) else None
                nSigmaMx[i, j] = nSigmaMx[j, i] = val
                if val and val > max_nSigma: max_nSigma = val

                val = dsc.aggregate_llr if (dsc is not None) else None
                logLMx[i, j] = logLMx[j, i] = val
                if val and val > max_2DeltaLogL: max_2DeltaLogL = val

        colormap = _colormaps.SequentialColormap(vmin=0, vmax=max_nSigma)
        nSigma_fig = _matrix_color_boxplot(
            nSigmaMx, dslabels, dslabels, "Dataset 1", "Dataset 2",
            box_labels=True, prec=1, colormap=colormap, scale=scale)

        colormap = _colormaps.SequentialColormap(vmin=0, vmax=max_2DeltaLogL)
        logL_fig = _matrix_color_boxplot(
            logLMx, dslabels, dslabels, "Dataset 1", "Dataset 2",
            box_labels=True, prec=1, colormap=colormap, scale=scale)

        #Combine plotly figures into one
        nSigma_figdict = nSigma_fig.plotlyfig
        if hasattr(nSigma_figdict, 'to_dict'):
            nSigma_figdict = nSigma_figdict.to_dict()  # so we can work with normal dicts
            # and not weird plotly objects.  Older versions of plotly do not support this syntax, so upgrade if needed.
        logL_figdict = logL_fig.plotlyfig
        if hasattr(logL_figdict, 'to_dict'):
            logL_figdict = logL_figdict.to_dict()
        combined_fig_data = list(nSigma_figdict['data']) + [logL_figdict['data'][0]]
        combined_fig_data[-1].update(visible=False)
        combined_fig = ReportFigure(go.Figure(data=combined_fig_data, layout=nSigma_figdict['layout']),
                                    nSigma_fig.colormap, nSigma_fig.pythonvalue)

        annotations = [nSigma_figdict['layout']['annotations'],
                       logL_figdict['layout']['annotations']]

        buttons = []; nTraces = 2
        for i, nm in enumerate(['Nsigma', '2DeltaLogL']):
            visible = [False] * nTraces
            visible[i] = True
            buttons.append(
                dict(args=[{'visible': visible},
                           {'annotations': annotations[i]}],
                     label=nm,
                     method='update'))  # 'restyle'
        #.update( updatemenus=
        combined_fig.plotlyfig['layout']['updatemenus'] = list([
            dict(buttons=buttons,
                 direction='left',
                 #pad = {'r': 1, 'b': 1},
                 showactive=True, type='buttons',
                 x=0.0, xanchor='left',
                 y=-0.1, yanchor='top')
        ])  # )
        m = combined_fig.plotlyfig['layout']['margin']
        w = combined_fig.plotlyfig['layout']['width']
        h = combined_fig.plotlyfig['layout']['height']
        exr = 0 if w > 240 else 240 - w  # extend to right
        combined_fig.plotlyfig['layout'].update(
            margin=go_margin(l=m['l'], r=m['r'] + exr, b=m['b'] + 40, t=m['t']),
            width=w + exr,
            height=h + 40
        )

        return combined_fig


class DatasetComparisonHistogramPlot(WorkspacePlot):
    """
    Histogram of p-values comparing two data sets

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    dsc : DataComparator
        The data set comparator, which holds and compares the data.

    nbins : int, optional
        Bins in the histogram.

    frequency : bool, optional
        Whether the frequencies (instead of the counts) are used.
        TODO: more detail.

    log : bool, optional
        Whether to set a log-scale on the x-axis or not.

    display : {'pvalue', 'llr'}, optional
        What quantity to display (in histogram).

    scale : float, optional
        Scaling factor to adjust the size of the final figure.
    """

    def __init__(self, ws, dsc, nbins=50, frequency=True,
                 log=True, display='pvalue', scale=1.0):
        super(DatasetComparisonHistogramPlot, self).__init__(ws, self._create, dsc, nbins, frequency,
                                                             log, display, scale)

    def _create(self, dsc, nbins, frequency, log, display, scale):
        if display == 'llr' and nbins is None:
            nbins = len(dsc.llrs)

        TOL = 1e-10
        pVals = _np.array(list(dsc.pVals.values()), 'd')
        pVals_nz = _np.array([x for x in pVals if abs(x) > TOL])
        pVals0 = (len(pVals) - len(pVals_nz)) if log else dsc.pVals0
        llrVals = _np.array(list(dsc.llrs.values()), 'd')

        if log:
            if len(pVals_nz) == 0:
                minval = maxval = thres = 0.0
                lastBinCount = 0
            else:
                minval = _np.floor(_np.log10(min(pVals_nz)))
                maxval = _np.ceil(_np.log10(max(pVals_nz)))
                thres = (maxval - minval) / (nbins - 1) * (nbins - 2)
                lastBinCount = (_np.log10(pVals_nz) > thres).sum()
                #Kenny: why use this as a normalization?  Is this correct?
        else:
            minval = min(pVals)
            maxval = max(pVals)
            thres = (maxval - minval) / (nbins - 1) * (nbins - 2)
            lastBinCount = (pVals > thres).sum()

        if display == 'pvalue':
            norm = 'probability' if frequency else 'count'
            vals = _np.log10(pVals_nz) if log else pVals
            cumulative = dict(enabled=False)
            barcolor = '#43C6DB'  # turquoise
        elif display == 'llr':
            norm = 'probability'
            vals = _np.log10(llrVals) if log else llrVals
            cumulative = dict(enabled=True)
            barcolor = '#FFD801'  # rubber ducky yellow
        else:
            raise ValueError("Invalid display value: %s" % display)

        histTrace = go.Histogram(
            x=vals, histnorm=norm,
            autobinx=False,
            xbins=dict(
                start=minval,
                end=maxval,
                size=(maxval - minval) / (nbins - 1)
            ),
            cumulative=cumulative,
            marker=dict(color=barcolor),
            opacity=0.75,
            showlegend=False,
        )

        if display == 'pvalue':
            bin_edges = _np.linspace(minval, maxval, nbins)
            linear_bin_edges = 10**(bin_edges) if log else bin_edges
            M = 1.0 if frequency else lastBinCount

            noChangeTrace = go.Scatter(
                x=bin_edges,
                y=M * _scipy.stats.chi2.pdf(
                    _scipy.stats.chi2.isf(linear_bin_edges, dsc.dof), dsc.dof),
                mode="lines",
                marker=dict(
                    color='rgba(0,0,255,0.8)',
                    line=dict(
                        width=2,
                    )),
                name='No-change prediction'
            )

            data = [histTrace, noChangeTrace]
            xlabel = 'p-value'
            ylabel = "Relative frequency" if frequency else "Number of occurrences"
            title = 'p-value histogram for experimental coins;'

        elif display == 'llr':
            data = [histTrace]
            xlabel = 'log-likelihood'
            ylabel = 'Cumulative frequency'
            title = 'Cumulative log-likelihood ratio histogram for experimental coins;'

        if log:
            minInt = int(_np.floor(minval))
            maxInt = int(_np.ceil(maxval))
            xaxis_dict = dict(
                title={'text': xlabel},
                tickvals=list(range(minInt, maxInt + 1)),
                ticktext=["10<sup>%d</sup>" % i for i in range(minInt, maxInt + 1)]
            )
        else:
            xaxis_dict = dict(title={'text':xlabel})  # auto-tick labels

        datasetnames = dsc.DS_names
        if dsc.op_exclusions:
            title += ' ' + str(dsc.op_exclusions) + ' excluded'
            if dsc.op_inclusions:
                title += ';'
        if dsc.op_inclusions:
            title += ' ' + str(dsc.op_inclusions) + ' included'
        title += '<br>Comparing data ' + str(datasetnames)
        title += ' p=0 ' + str(pVals0) + ' times; ' + str(len(dsc.pVals)) + ' total sequences'

        layout = go.Layout(
            width=700 * scale,
            height=400 * scale,
            title={'text':title},
            font=dict(size=10),
            xaxis=xaxis_dict,
            yaxis=dict(
                title={'text':ylabel},
                type='log' if log else 'linear',
                #tickformat='g',
                exponentformat='power',
            ),
            bargap=0,
            bargroupgap=0,
            legend=dict(orientation="h")
        )

        pythonVal = {'histogram values': vals}
        if display == 'pvalue':
            pythonVal['noChangeTrace'] = {'x': noChangeTrace['x'], 'y': noChangeTrace['y']}
        return ReportFigure(go.Figure(data=data, layout=layout),
                            None, pythonVal)


class WildcardSingleScaleBarPlot(WorkspacePlot):
    """
    Stacked bar plot showing per-gate reference values and wildcard budgets.

    Typically these reference values are a gate metric comparable to wildcard
    budget such as diamond distance, and the bars show the relative modeled vs.
    unmodeled error.

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    budget : PrimitiveOpsSingleScaleWildcardBudget
        Wildcard budget to be plotted.
    """

    def __init__(self, ws, budget, scale=1.0, reference_name='Reference Value'):
        super(WildcardSingleScaleBarPlot, self).__init__(ws, self._create, budget, scale, reference_name)

    def _create(self, budget, scale, reference_name):

        per_op_wildcard_values = budget.per_op_wildcard_vector
        ref_values = budget.reference_values
        gate_labels = budget.primitive_op_labels

        x_axis = go.layout.XAxis(dtick=1, tickmode='array', tickvals=list(range(len(gate_labels))),
                                 ticktext=[str(label) for label in gate_labels])
        y_axis = go.layout.YAxis(tickformat='.1e')

        layout = go.Layout(barmode='stack', xaxis=x_axis, yaxis=y_axis,
                           width=650 * scale, height=350 * scale)

        ref_bar = go.Bar(y=ref_values, name=reference_name, width=0.5)
        wildcard_bar = go.Bar(y=per_op_wildcard_values, name='Wildcard', width=0.5)

        return ReportFigure(go.Figure(data=[ref_bar, wildcard_bar], layout=layout))


class RandomizedBenchmarkingPlot(WorkspacePlot):
    """
    Plot of RB Decay curve

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

    rb_r : RandomizedBenchmarkingResults
        The RB results object containing all the relevant RB data.

    fitkey : dict key, optional
        The key of the self.fits dictionary to plot the fit for. If None, will
        look for a 'full' key (the key for a full fit to A + Bp^m if the standard
        analysis functions are used) and plot this if possible. It otherwise checks
        that there is only one key in the dict and defaults to this. If there are
        multiple keys and none of them are 'full', `fitkey` must be specified when
        `decay` is True.

    decay : bool, optional
        Whether to plot a fit, or just the data.

    success_probabilities : bool, optional
        Whether to plot the success probabilities distribution, as a
        "box & whisker" plot.

    ylim : tuple, optional
        The y limits for the figure.

    xlim : tuple, optional
        The x limits for the figure.

    showpts : bool, optional
        When `success_probabilities == True`, whether individual points
        should be shown along with a "box & whisker".

    legend : bool, optional
        Whether to show a legend.

    title : str, optional
        A title to put on the figure.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.
    """

    def __init__(self, ws, rb_r, fitkey=None, decay=True,
                 success_probabilities=True, ylim=None, xlim=None,
                 showpts=True, legend=True, title=None, scale=1.0):
        """
        Plot RB decay curve, as a function of sequence length.  Optionally
        includes a fitted exponential decay.

        Parameters
        ----------
        rb_r : RandomizedBenchmarkingResults
            The RB results object containing all the relevant RB data.

        fitkey : dict key, optional
            The key of the self.fits dictionary to plot the fit for. If None, will
            look for a 'full' key (the key for a full fit to A + Bp^m if the standard
            analysis functions are used) and plot this if possible. It otherwise checks
            that there is only one key in the dict and defaults to this. If there are
            multiple keys and none of them are 'full', `fitkey` must be specified when
            `decay` is True.

        decay : bool, optional
            Whether to plot a fit, or just the data.

        success_probabilities : bool, optional
            Whether to plot the success probabilities distribution, as a
            "box & whisker" plot.

        ylim, xlim : tuple, optional
            The x and y limits for the figure.

        showpts : bool, optional
            When `success_probabilities == True`, whether individual points
            should be shown along with a "box & whisker".

        legend : bool, optional
            Whether to show a legend.

        title : str, optional
            A title to put on the figure.

        Returns
        -------
        None
        """
        super(RandomizedBenchmarkingPlot, self).__init__(
            ws, self._create, rb_r, fitkey, decay, success_probabilities,
            ylim, xlim, showpts, legend, title, scale)

    def _create(self, rb_r, fitkey, decay, success_probabilities, ylim, xlim,
                showpts, legend, title, scale):

        if decay and fitkey is None:
            allfitkeys = list(rb_r.fits.keys())
            if 'full' in allfitkeys:
                fitkey = 'full'
            elif len(allfitkeys) == 1:
                fitkey = allfitkeys[0]
            else:
                raise ValueError(("There are multiple fits and none have the "
                                  "key 'full'. Please specify the fit to plot!"))

        ASPs = []  # (avg success probs)
        data_per_depth = rb_r.data.cache[rb_r.protocol.datatype]
        for depth in rb_r.depths:
            percircuitdata = data_per_depth[depth]
            ASPs.append(_np.mean(percircuitdata))  # average [adjusted] success probabilities

        xdata = _np.asarray(rb_r.depths)
        ydata = _np.asarray(ASPs)

        data = []  # list of traces
        data.append(go.Scatter(
            x=xdata, y=ydata,
            mode='markers',
            marker=dict(
                color="rgb(0,0,0)",
                size=5
            ),
            name='Average success probabilities',
            showlegend=legend,
        ))

        if decay:
            lengths = _np.linspace(0, max(rb_r.depths), 200)
            try:
                A = rb_r.fits[fitkey].estimates['a']
                B = rb_r.fits[fitkey].estimates['b']
                p = rb_r.fits[fitkey].estimates['p']

                data.append(go.Scatter(
                    x=lengths,
                    y=A + B * p**lengths,
                    mode='lines',
                    line=dict(width=1, color="rgb(120,120,120)"),
                    name='Fit, r = {:.2} +/- {:.1}'.format(rb_r.fits[fitkey].estimates['r'],
                                                        rb_r.fits[fitkey].stds['r']),
                    showlegend=legend,
                ))
            except KeyError:
                _warnings.warn(f'RB fit for {fitkey} likely failed, skipping plot...')

        if success_probabilities:
            all_success_probs_by_depth = [data_per_depth[depth] for depth in rb_r.depths]
            for depth, prob_dist in zip(rb_r.depths, all_success_probs_by_depth):
                data.append(go.Box(
                    x0=depth, y=prob_dist,
                    whiskerwidth=0.2, opacity=0.7, showlegend=False,
                    boxpoints='all' if showpts else False,
                    pointpos=0, jitter=0.5,
                    boxmean=False,  # or True or 'sd'
                    hoveron="boxes", hoverinfo="all",
                    name='m=%d' % depth))

        #pad by 10%
        ymin = min(ydata)
        ymin -= 0.1 * abs(1.0 - ymin)
        xmin = -0.1 * max(xdata)
        xmax = max(xdata) * 1.1

        layout = go.Layout(
            width=800 * scale,
            height=400 * scale,
            title=dict(text=title, font=dict(size=16)),
            xaxis=dict(
                title=dict(text="RB sequence length (m)",
                font=dict(size=14)),
                range=xlim if xlim else [xmin, xmax],
            ),
            yaxis=dict(
                title=dict(text="Success probability",
                font=dict(size=14)),
                range=ylim if ylim else [ymin, 1.0],
            ),
            legend=dict(
                x=0.5, y=1.2,
                font=dict(
                    size=13
                ),
            )
        )

        pythonVal = {}
        for i, tr in enumerate(data):
            if 'x0' in tr: continue  # don't put boxes in python val for now
            key = tr['name'] if ("name" in tr) else "trace%d" % i
            pythonVal[key] = {'x': tr['x'], 'y': tr['y']}

        #reverse order of data so z-ordering is nicer
        return ReportFigure(go.Figure(data=list(data), layout=layout),
                            None, pythonVal)
