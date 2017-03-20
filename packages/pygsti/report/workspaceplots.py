from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Functions for generating plots """

import numpy             as _np
import matplotlib.pyplot as _plt
import matplotlib        as _matplotlib
import os                as _os
import warnings          as _warnings

from .. import algorithms   as _alg
from .. import tools        as _tools
from .. import construction as _construction
from .. import objects      as _objs

from .figure import ReportFigure as _ReportFigure

from .workspace import WorkspacePlot
from . import colormaps as _colormaps
from . import plothelpers as _ph

import plotly.graph_objs as go


import time as _time  #DEBUG TIMER
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot #DEBUG


def color_boxplot(plt_data, colormap, colorbar=False, boxLabelSize=0,
                  prec=0, hoverLabelFn=None, hoverLabels=None):
    """ TODO: docstring
    Create a color box plot.

    Creates a figure composed of colored boxes and possibly labels.

    Parameters
    ----------
    plt_data : numpy array
        A 2D array containing the values to be plotted.

    cmapFactory: ColormapFactory class
        An instance of a ColormapFactory class

    title : string, optional
        Plot title (latex can be used)

    xlabels, ylabels : list of strings, optional
        Tic labels for x and y axes.  If both are None, then tics are not drawn.

    xtics, ytics : list or array of floats, optional
        Values of x and y axis tics.  If None, then half-integers from 0.5 to
        0.5 + (nCols-1) or 0.5 + (nRows-1) are used, respectively.

    colorbar : bool, optional
        Whether to display a colorbar or not.

    fig, axes : matplotlib figure and axes, optional
        If non-None, use these figure and axes objects instead of creating new ones
        via fig,axes = pyplot.supblots()

    size : 2-tuple, optional
        The width and heigh of the final figure in inches.

    prec : int or {'compact','compacthp'}, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    boxLabels : bool, optional
        Whether box labels are displayed.  It takes much longer to
        generate the figure when this is set to True.

    xlabel, ylabel : str, optional
        X and Y axis labels

    save_to : str, optional
        save figure as this filename (usually ending in .pdf)

    ticSize : int, optional
        size of tic marks

    grid : bool, optional
        Whether or not grid lines should be displayed.

    Returns
    -------
    ReportFigure
        The encapsulated matplotlib figure that was generated
    """

    heatmapArgs = { 'z': colormap.normalize(plt_data),
                    'colorscale': colormap.get_colorscale(),
                    'showscale': colorbar, 'hoverinfo': 'none',
                    'zmin':0, 'zmax': 1.0 } #so colormap normalization works as expected
    
    #if xlabels is not None: heatmapArgs['x'] = xlabels
    #if ylabels is not None: heatmapArgs['y'] = ylabels

    annotations = []
    if boxLabelSize:
        # Write values on colored squares
        for y in range(plt_data.shape[0]):
            for x in range(plt_data.shape[1]):
                if _np.isnan(plt_data[y, x]): continue
                annotations.append(
                    dict(
                        text=_ph._eformat(plt_data[y, x], prec),
                        x= x, y= y,
                        xref='x1', yref='y1',
                        font=dict(size=boxLabelSize,
                                  color=colormap.besttxtcolor(plt_data[y,x])),
                        showarrow=False)
                )


    if hoverLabelFn:
        assert(not hoverLabels), "Cannot specify hoverLabelFn and hoverLabels!"
        hoverLabels = []
        for y in range(plt_data.shape[0]):
            hoverLabels.append([ hoverLabelFn(plt_data[y, x],y,x) 
                                 for x in range(plt_data.shape[1]) ] )
    if hoverLabels:
        heatmapArgs['hoverinfo'] = 'text'
        heatmapArgs['text'] = hoverLabels                            
        
    trace = go.Heatmap(**heatmapArgs)
    data = [trace]

    xaxis = go.XAxis(
        showgrid=False,
        zeroline=False,
        showline=True,
        ticks="", 
        showticklabels=True,
        mirror=True,
        linewidth=2,
        range=[-0.5, plt_data.shape[1]-0.5]
        )
    yaxis = go.YAxis(
        showgrid=False,
        zeroline=False,
        showline=True,
        ticks="", 
        showticklabels=True,
        mirror=True,
        linewidth=2,
        range=[-0.5, plt_data.shape[0]-0.5]
        )

    layout = go.Layout(
        xaxis=xaxis,
        yaxis=yaxis,
        annotations=annotations
    )
    
    fig = go.Figure(data=data, layout=layout)
    return fig



def nested_color_boxplot(plt_data_list_of_lists, colormap, 
                         colorbar=False, boxLabelSize=0, prec=0,
                         hoverLabelFn=None):
    """ TODO: docstring
    Create a color box plot.

    Creates a figure composed of colored boxes and possibly labels.

    Parameters
    ----------
    plt_data_list_of_lists : list of lists of numpy arrays
        A complete square 2D list of lists, such that each element is a
        2D numpy array of the same size.

    cmapFactory: instance of the ColormapFactory class

    title : string, optional
        Plot title (latex can be used)

    xlabels, ylabels : list of strings, optional
        Tic labels for x and y axes.  If both are None, then tics are not drawn.

    xtics, ytics : list or array of floats, optional
        Values of x and y axis tics.  If None, then half-integers from 0.5 to
        0.5 + (nCols-1) or 0.5 + (nRows-1) are used, respectively.

    colorbar : bool, optional
        Whether to display a colorbar or not.

    fig, axes : matplotlib figure and axes, optional
        If non-None, use these figure and axes objects instead of creating new ones
        via fig,axes = pyplot.supblots()

    size : 2-tuple, optional
        The width and heigh of the final figure in inches.

    prec : int, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    boxLabels : bool, optional
        Whether box labels are displayed.  It takes much longer to
        generate the figure when this is set to True.

    xlabel, ylabel : str, optional
        X and Y axis labels

    save_to : str, optional
        save figure to this filename (usually ending in .pdf)

    ticSize : int, optional
        size of tic marks

    grid : bool, optional
        Whether or not grid lines should be displayed.

    Returns
    -------
    ReportFigure
        The encapsulated matplotlib figure that was generated
    """

    #Assemble the single 2D grid to pass to color_boxplot
    # (assume a complete 2D rectangular list of lists, and that
    #  each element is a numpy array of the same size)
    if len(plt_data_list_of_lists) == 0 or len(plt_data_list_of_lists[0]) == 0: return
    elRows,elCols = plt_data_list_of_lists[0][0].shape #nE,nr
    nRows = len(plt_data_list_of_lists)
    nCols = len(plt_data_list_of_lists[0])

    data = _np.zeros( ( elRows*nRows + (nRows-1), elCols*nCols + (nCols-1)) )
    for i in range(1,nRows):
        data[(elRows+1)*i-1:(elRows+1)*i,:] = _np.nan
    for j in range(1,nCols):
        data[:, (elCols+1)*j-1:(elCols+1)*j] = _np.nan

    for i in range(nRows):
        for j in range(nCols):
            data[(elRows+1)*i:(elRows+1)*(i+1)-1, (elCols+1)*j:(elCols+1)*(j+1)-1] = plt_data_list_of_lists[i][j]

    xtics = []; ytics = []
    for i in range(nRows):   ytics.append( float((elRows+1)*i)-0.5+0.5*float(elRows) )
    for j in range(nCols):   xtics.append( float((elCols+1)*j)-0.5+0.5*float(elCols) )

    if hoverLabelFn:
        hoverLabels = []
        for k in range(elRows*nRows + (nRows-1)): hoverLabels.append( [""]*(elCols*nCols + (nCols-1)) )

        for i in range(nRows):
            for j in range(nCols):
                for ii in range(elRows):
                    for jj in range(elCols):
                        hoverLabels[(elRows+1)*i + ii][(elCols+1)*j + jj] = \
                            hoverLabelFn(plt_data_list_of_lists[i][j][ii][jj],i,j,ii,jj)
    else:
        hoverLabels = None

    fig = color_boxplot(data, colormap, colorbar, boxLabelSize,
                        prec, None, hoverLabels)

    #Layout updates: add tic marks (but not labels - leave that to user)
    fig['layout']['xaxis'].update(tickvals=xtics)
    fig['layout']['yaxis'].update(tickvals=ytics)
    return fig


def generate_boxplot(subMxs,
                     xlabels, ylabels, inner_xlabels, inner_ylabels,
                     xlabel,  ylabel,  inner_xlabel,  inner_ylabel,
                     colormap, colorbar=False, boxLabels=True, prec=0, hoverInfo=True,
                     sumUp=False, invert=False, save_to=None, scale=1.0):
    """ TODO: docstring
    Creates a view of nested box plot data (i.e. a matrix for each (x,y) pair).

    Given lists of x and y values, a dictionary to convert (x,y) pairs into gate strings,
    and a function to convert a "base" gate string into a matrix of floating point values,
    this function computes (x,y) => matrix data and displays it in one of two ways:

    1. As a full nested color box plot, showing all the matrix values individually
    2. As a color box plot containing the sum of the elements in the (x,y) matrix as
       the (x,y) box.

    A histogram of the values can also be computed and displayed.

    Parameters
    ----------
    xvals, yvals : list
        List of x and y values. Elements can be any hashable quantity, and will be converted
        into x and y tic labels.  Tuples of strings are converted specially for nice latex
        rendering of gate strings.

    xyGateStringDict : dict
        Dictionary with keys == (x_value,y_value) tuples and values == gate strings, where
        a gate string can either be a GateString object or a tuple of gate labels.  Provides
        the mapping between x,y pairs and gate strings.  None values are allowed, and
        indicate that there is not data for that x,y pair and nothing should be plotted.

    subMxs : list
        A list of lists of 2D numpy.ndarrays.  subMxs[iy][ix] specifies the matrix of values
        or sum (if sumUp == True) displayed in iy-th row and ix-th column of the plot.  NaNs
        indicate elements should not be displayed.

    cmapFactory: instance of the ColormapFactory class

    xlabel, ylabel : str, optional
        X and Y axis labels

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    prec : int, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    title : string, optional
        Plot title (latex can be used)

    sumUp : bool, optional
        False displays each matrix element as it's own color box
        True sums the elements of each (x,y) matrix and displays
        a single color box for the sum.

    boxLabels : bool, optional
        Whether box labels are displayed.  It takes much longer to
        generate the figure when this is set to True.

    histogram : bool, optional
        Whether a histogram of the matrix element values or summed matrix
        values (depending on sumUp) should also be computed and displayed.

    histBins : int, optional
        The number of bins to use in the histogram.

    save_to : str, optional
        save figure to this filename (usually ending in .pdf)

    ticSize : int, optional
        size of tic marks

    invert : bool, optional
        If True, invert the nesting order of the nested color box plot (applicable
        only when sumUp == False).  Use inner_x_labels and inner_y_labels to label
        the x and y axes.

    inner_x_labels, inner_y_labels : list, optional
        Similar to xvals, yvals but labels for the columns and rows of the (x,y) matrices
        computed by subMxCreationFn.  Used when invert == True.

    grid : bool, optional
        Whether or not grid lines should be displayed.

    Returns
    -------
    rptFig : ReportFigure
        The encapsulated matplotlib figure that was generated.  Note that
        figure extra info is a dict with keys:

        nUsedXs : int
            The number of used X-values, proportional to the overall final figure width
        nUsedYs : int
            The number of used Y-values, proportional to the overall final figure height
    """
    nYs = len(subMxs)
    nXs = len(subMxs[0]) if nYs>0 else 0
    
    nIYs = nIXs = 0
    for ix in range(nXs):
        for iy in range(nYs):
            if subMxs[iy][ix] is not None:
                nIYs,nIXs = subMxs[iy][ix].shape; break

    # flip so [0,0] el of original subMxs is at *top*-left (FLIP)
    subMxs = [ [ _np.flipud(subMx) for subMx in row ] for row in subMxs]
    inner_ylabels = list(reversed(inner_ylabels))

    if invert:
        if sumUp:
            _warnings.warn("Cannot invert a summed-up plot.  Ignoring invert=True.")
        else:
            invertedSubMxs = []  #will be indexed as invertedSubMxs[inner-y][inner-x]
            for iny in range(nIYs):
                invertedSubMxs.append( [] )
                for inx in range(nIXs):
                    mx = _np.array( [[ subMxs[iy][ix][iny,inx] for ix in range(nXs) ] for iy in range(nYs)],  'd' )
                    invertedSubMxs[-1].append( mx )

            # flip the now-inverted mxs to counteract the flip that will occur upon
            # entering generate_boxplot again (with invert=False this time), since we
            # *don't* want the now-inner dimension (the germs) actually flipped (FLIP)
            invertedSubMxs = [ [ _np.flipud(subMx) for subMx in row ] for row in invertedSubMxs]
            ylabels = list(reversed(ylabels))
                        
            return generate_boxplot(invertedSubMxs,
                         inner_xlabels, inner_ylabels,
                         xlabels, ylabels, inner_xlabel,  inner_ylabel,  xlabel,  ylabel,
                         colormap, colorbar, boxLabels, prec, hoverInfo,
                         sumUp, False, save_to, scale)
                                

    def val_filter(vals):  #filter to latex-ify gate strings.  Later add filter as a possible parameter
        formatted_vals = []
        for val in vals:
            if (isinstance(val,tuple) or isinstance(val,_objs.GateString)) \
               and all([isinstance(el,str) for el in val]):
                if len(val) == 0:
                    formatted_vals.append(r"$\{\}$")
                else:
                    #formatted_vals.append( "$" + "\\cdot".join([("\\mathrm{%s}" % el) for el in val]) + "$" )
                    formatted_vals.append(str(val))
            else:
                formatted_vals.append(val)
        return formatted_vals

    def sum_up_mx(mx):
        flat_mx = mx.flatten()
        if any([_np.isnan(x) for x in flat_mx]):
            if all([_np.isnan(x) for x in flat_mx]):
                return _np.nan
            return sum(_np.nan_to_num(flat_mx)) #replace NaNs with zeros for purpose of summing (when there's at least one non-NaN)
        else:
            return sum(flat_mx)

        
    #Setup and create plotting functions
    if sumUp:
        subMxSums = _np.array( [ [ sum_up_mx(subMxs[iy][ix]) for ix in range(nXs) ] for iy in range(nYs) ], 'd' )

        if hoverInfo:
            def hoverLabelFn(val,i,j):
                if _np.isnan(val): return ""
                return "%s: %s<br>%s: %s<br>%g" % \
                    (xlabel,str(xlabels[j]),ylabel,str(ylabels[i]), val)
        else: hoverLabelFn = None

        boxLabelSize = 8*scale if boxLabels else 0
        fig = color_boxplot( subMxSums, colormap, colorbar, boxLabelSize,
                             prec, hoverLabelFn)
        #update tickvals b/c color_boxplot doesn't do this (unlike nested_color_boxplot)
        fig['layout']['xaxis'].update(tickvals=list(range(nXs)))
        fig['layout']['yaxis'].update(tickvals=list(range(nYs)))
        fig['layout'].update(width=100*(nXs+1)*scale,
                             height=100*(nYs+1)*scale)

    else: #not summing up
                
        if hoverInfo:
            def hoverLabelFn(val,i,j,ii,jj):
                if _np.isnan(val): return ""
                return "%s: %s<br>%s: %s<br>%s: %s<br>%s: %s<br>%g" % \
                    (xlabel,str(xlabels[j]),ylabel,str(ylabels[i]),
                     inner_xlabel,str(inner_xlabels[jj]),
                     inner_ylabel,str(inner_ylabels[ii]), val)
        else: hoverLabelFn = None

        boxLabelSize = 8*scale if boxLabels else 0
        fig = nested_color_boxplot(subMxs, colormap, colorbar, boxLabelSize,
                                   prec, hoverLabelFn)

        fig['layout'].update(width=30*(nXs*nIXs+1)*scale,
                             height=30*(nYs*nIYs+1)*scale)
        
    if xlabel: fig['layout']['xaxis'].update(title=xlabel,
                                             titlefont={'size': 12*scale, 'color': "black"})
    if ylabel: fig['layout']['yaxis'].update(title=ylabel,
                                             titlefont={'size': 12*scale, 'color': "black"})
    if xlabels:
        fig['layout']['xaxis'].update(tickmode="array",
                                      ticktext=val_filter(xlabels),
                                      tickfont={'size': 10*scale, 'color': "black"})
    if ylabels:
        fig['layout']['yaxis'].update(tickmode="array",
                                      ticktext=val_filter(ylabels),
                                      tickfont={'size': 10*scale, 'color': "black"})
        #lblLen = max(map(len,val_filter(ylabels)))
        #fig['layout'].update(margin=go.Margin(l=5*lblLen*scale ) ) # r=50, b=100, t=100, pad=4

    # ticSize, grid, title?

    #TODO: figure saving
    #if rptFig is not None: #can be None if there's nothing to plot
    #    rptFig.save_to(save_to)
    #if rptFig is not None:
    #    rptFig.set_extra_info( { 'nUsedXs': len(xvals),
    #                             'nUsedYs': len(yvals) } )
    return fig

def gatestring_color_boxplot(gatestring_structure, subMxs, colormap,
                             colorbar=False, boxLabels=True, prec='compact', hoverInfo=True,
                             invert=False,sumUp=False,save_to=None,scale=1.0):
    """
    Creates a color boxplot for a structured set of gate strings.
    TODO: docstring
    """
    g = gatestring_structure
    return generate_boxplot(subMxs, g.used_maxLs, g.used_germs, g.prepfids, g.effectfids,
                            "L","germ","rho_i","E_i", colormap,
                            colorbar, boxLabels, prec, hoverInfo,
                            sumUp, invert, save_to, scale)  #"$\\rho_i$","$\\E_i$"      


def gatematrix_color_boxplot(gateMatrix, m, M, mxBasis, mxBasisDims=None,
                             mxBasisDimsY=None, xlabel=None, ylabel=None,
                             boxLabels=False, prec=0, scale=1.0):
        
    def one_sigfig(x):
        if abs(x) < 1e-9: return 0
        if x < 0: return -one_sigfig(-x)
        e = -int(_np.floor(_np.log10(abs(x)))) #exponent
        trunc_x = _np.floor(x * 10**e)/ 10**e #truncate decimal to make sure it gets *smaller*
        return round(trunc_x, e) #round to truncation point just to be sure

    if mxBasis is not None and mxBasisDims is not None:
        if mxBasisDimsY is None: mxBasisDimsY = mxBasisDims
        xlabels=[("$%s$" % x) if len(x) else "" \
                 for x in _tools.basis_element_labels(mxBasis,mxBasisDims)]
        ylabels=[("$%s$" % x) if len(x) else "" \
                 for x in _tools.basis_element_labels(mxBasis,mxBasisDimsY)]
    else:
        xlabels = [""] * gateMatrix.shape[1]
        ylabels = [""] * gateMatrix.shape[0]

    colorscale = [[0, '#3D9970'], [0.75,'#ffffff'], [1, '#001f3f']]  # custom colorscale
    colormap = _colormaps.DivergingColormap(vmin=m, vmax=M)

    flipped_mx = _np.flipud(gateMatrix)  # FLIP so [0,0] matrix el is at *top* left
    ylabels    = list(reversed(ylabels)) # FLIP y-labels to match
    trace = go.Heatmap(z=flipped_mx, colorscale=colormap.get_colorscale(),
                       showscale=(not boxLabels))
    data = [trace]
    
    scale = 1.0
    nX = gateMatrix.shape[1]
    nY = gateMatrix.shape[0]
    
    gridlines = []
    
    # Vertical lines
    for i in range(nX-1):
        #add darker lines at multiples of 4 boxes
        w = 3 if (mxBasis == "pp" and i-1 % 4 == 0) else 1
        
        gridlines.append(     
            {
                'type': 'line',
                'x0': i+0.5, 'y0': -0.5,
                'x1': i+0.5, 'y1': nY-0.5,
                'line': {'color': 'black', 'width': w},
            } )
        
    #Horizontal lines
    for i in range(nY-1):
        #add darker lines at multiples of 4 boxes
        w = 3 if (mxBasis == "pp" and i-1 % 4 == 0) else 1

        gridlines.append(     
            {
                'type': 'line',
                'x0': -0.5, 'y0': i+0.5,
                'x1': nX-0.5, 'y1': i+0.5,
                'line': {'color': 'black', 'width': w},
            } )


    annotations = []
    if boxLabels:
        for ix in range(nX):
            for iy in range(nY):
                annotations.append(
                    dict(
                    text=_ph._eformat(gateMatrix[iy,ix],prec),
                    x=ix, y=nY-1-iy, xref='x1', yref='y1',
                    font=dict(size=scale*10,
                              color=colormap.besttxtcolor(gateMatrix[iy,ix])),
                        showarrow=False)
                )

    layout = go.Layout(
        width = 80*(gateMatrix.shape[1]+1)*scale,
        height = 80*(gateMatrix.shape[0]+1.5)*scale,
        xaxis=dict(
            side="top",
            title=xlabel,
            showgrid=False,
            zeroline=False,
            showline=True,
            showticklabels=True,
            mirror=True,
            ticks="",
            linewidth=2,
            ticktext=xlabels,
            tickvals=[i for i in range(len(xlabels))],
            tickangle=-90,
            range=[-0.5,len(xlabels)-0.5]
            ),
        yaxis=dict(
            side="left",
            title=ylabel,
            showgrid=False,
            zeroline=False,
            showline=True,
            showticklabels=True,
            mirror=True,
            ticks="",
            linewidth=2,
            ticktext=ylabels,
            tickvals=[i for i in range(len(ylabels))],
            range=[-0.5,len(ylabels)-0.5],
            ),
        shapes = gridlines,
        annotations = annotations
    )
            
    return go.Figure(data=data, layout=layout)











    """
    Create a plot showing the layout of a single sub-block of a goodness-of-fit
    box plot (such as those produced by chi2_boxplot or logl_boxplot).

    Parameters
    ----------
    strs : 2-tuple
        A (prepStrs,effectStrs) tuple usually generated by calling get_spam_strs(...)

    xlabel, ylabel : str, optional
        X and Y axis labels

    title : string, optional
        Plot title (latex can be used)

    size : tuple, optional
      The (width,height) figure size in inches.  None
      enables automatic calculation based on gateMatrix
      size.

    save_to : str, optional
        save figure to this filename (usually ending in .pdf)

    ticSize : int, optional
        size of tic marks

    Returns
    -------
    rptFig : ReportFigure
        The encapsulated matplotlib figure that was generated.
    """
class BoxKeyPlot(WorkspacePlot):
    def __init__(self, ws, prepStrs, effectStrs,
                 xlabel="$\\rho_i$", ylabel="$E_i$", scale=1.0):
        # separate in rendering/saving: save_to=None, ticSize=20, scale=1.0 (?)
        super(BoxKeyPlot,self).__init__(ws, self._create, prepStrs, effectStrs,
                                         xlabel, ylabel, scale)

          #size, save_to,
          
    def _create(self, prepStrs, effectStrs, xlabel, ylabel, scale):
        
        #Copied from generate_boxplot
        def val_filter(vals):  #filter to latex-ify gate strings.  Later add filter as a possible parameter
            formatted_vals = []
            for val in vals:
                if type(val) in (tuple,_objs.GateString) and all([type(el) == str for el in val]):
                    if len(val) == 0:
                        formatted_vals.append(r"$\{\}$")
                    else:
                        #formatted_vals.append( "$" + "\\cdot".join([("\\mathrm{%s}" % el) for el in val]) + "$" )
                        formatted_vals.append(str(val))
                else:
                    formatted_vals.append(val)
            return formatted_vals
        
        layout = go.Layout(
            width=70*scale*(len(prepStrs)+1),
            height=70*scale*(len(effectStrs)+1),
            xaxis=dict(
                side="bottom",
                showgrid=False,
                zeroline=False,
                showline=True,
                showticklabels=True,
                mirror=True,
                ticks="",
                linewidth=2,
                ticktext=val_filter(prepStrs),
                tickvals=[i+0.5 for i in range(len(prepStrs))],
                tickangle=90,
               range=[0,len(prepStrs)]
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
                ticktext=list(reversed(val_filter(effectStrs))),
                tickvals=[i+0.5 for i in range(len(effectStrs))],
                range=[0,len(effectStrs)]
            ),
            annotations = [
                go.Annotation(
                    x=0.5,
                    y=1.1,
                    showarrow=False,
                    text=xlabel,
                    font={'size': 12*scale, 'color': "black"},
                    xref='paper',
                    yref='paper'),
                go.Annotation(
                    x=-0.1,
                    y=0.5,
                    showarrow=False,
                    text=ylabel,
                    font={'size': 12*scale, 'color': "black"},
                    xref='paper',
                    yref='paper'
                )
            ]
        )

        xs = [i+0.5 for i in range(len(prepStrs))]
        ys = [i+0.5 for i in range(len(effectStrs))]
        allys = []
        for y in ys: allys.extend( [y]*len(xs) )
        allxs = xs*len(ys)
        trace = go.Scatter(x=allxs, y=allys, mode="markers",
                           marker=dict(
                               size = 12,
                               color = 'rgba(0,0,255,0.8)',
                               line = dict(
                                   width = 2,
                               )),
                           xaxis='x1', yaxis='y1', hoverinfo='none')
                          
        fig = go.Figure(data=[trace], layout=layout)
        return fig


        
    """
    Create a color box plot of chi^2 values.

    Parameters
    ----------
    xvals, yvals : list
        List of x and y values. Elements can be any hashable quantity, and will be converted
        into x and y tic labels.  Tuples of strings are converted specially for nice latex
        rendering of gate strings.

    xy_gatestring_dict : dict
        Dictionary with keys == (x_value,y_value) tuples and values == gate strings, where
        a gate string can either be a GateString object or a tuple of gate labels.  Provides
        the mapping between x,y pairs and gate strings.  None values are allowed, and
        indicate that there is not data for that x,y pair and nothing should be plotted.

    dataset : DataSet
        The data used to specify frequencies and counts

    gateset : GateSet
        The gate set used to specify the probabilities and SPAM labels

    strs : 2-tuple
        A (prepStrs,effectStrs) tuple usually generated by calling get_spam_strs(...)

    xlabel, ylabel : str, optional
        X and Y axis labels

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    prec : int, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    title : string, optional
        Plot title (latex can be used)

    linlg_pcntle: float, optional
        Specifies the (1 - linlg_pcntle) percentile to compute for the boxplots

    sumUp : bool, optional
        False displays each matrix element as it's own color box
        True sums the elements of each (x,y) matrix and displays
        a single color box for the sum.

    boxLabels : bool, optional
        Whether box labels are displayed.  It takes much longer to
        generate the figure when this is set to True.

    histogram : bool, optional
        Whether a histogram of the matrix element values or summed matrix
        values (depending on sumUp) should also be computed and displayed.

    histBins : int, optional
        The number of bins to use in the histogram.

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see chi2fn).

    save_to : str, optional
        save figure to this filename (usually ending in .pdf)

    ticSize : int, optional
        size of tic marks

    invert : bool, optional
        If True, invert the nesting order of the nested color box plot (applicable
        only when sumUp == False).  Use inner_x_labels and inner_y_labels to label
        the x and y axes.

    fidpair_filters : dict, optional
        If not None, a dictionary whose keys are (x,y) tuples and whose values
        are lists of (iRhoStr,iEStr) tuples specifying a subset of all the
        prepStr,effectStr pairs to include in the plot for each particular
        (x,y) sub-block.

    gatestring_filters : dict, optional
        If not None, a dictionary whose keys are (x,y) tuples and whose
        values are lists of GateString objects specifying which elements should
        be computed and displayed in the (x,y) sub-block of the plot.

    gateLabelAliases : dictionary, optional
        Dictionary whose keys are gate label "aliases" and whose values are tuples
        corresponding to what that gate label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. gateLabelAliases['Gx^3'] = ('Gx','Gx','Gx')


    Returns
    -------
    rptFig : ReportFigure
        The encapsulated matplotlib figure that was generated.  Extra figure
        info is a dict with keys:

        nUsedXs : int
            The number of used X-values, proportional to the overall final figure width
        nUsedYs : int
            The number of used Y-values, proportional to the overall final figure height
    """

class ColorBoxPlot(WorkspacePlot):
    def __init__(self, ws, plottype, gss, dataset, gateset,
                 prec='compact', title='', sumUp=False,
                 boxLabels=True, hoverInfo=True, invert=False,
                 linlg_pcntle=.05, minProbClipForWeighting=1e-4,
                 directGSTgatesets=None): 
        # separate in rendering/saving: save_to=None, ticSize=20, scale=1.0 (?)
        super(ColorBoxPlot,self).__init__(ws, self._create, plottype, gss, dataset, gateset,
                                          prec, title, sumUp, boxLabels, hoverInfo,
                                          invert, linlg_pcntle, minProbClipForWeighting,
                                          directGSTgatesets)

    def _create(self, plottypes, gss, dataset, gateset,
                prec, title, sumUp, boxLabels, hoverInfo,
                invert, linlg_pcntle, minProbClipForWeighting,
                directGSTgatesets):

        maps = _ph._computeGateStringMaps(gss, dataset)
        probs_precomp_dict = None
        fig = None
        
        if isinstance(plottypes,str):
            plottypes = [plottypes]
            
        for typ in plottypes:
            if typ == "chi2":
                precomp=True
                colormapType = "linlog"
                linlog_color = "red"
                
                def mx_fn(gateStr,x,y):
                    return _ph.chi2_matrix( maps[gateStr], dataset, gateset, minProbClipForWeighting,
                                            probs_precomp_dict, gss.aliases)
            elif typ == "logl":
                precomp=True
                colormapType = "linlog"
                linlog_color = "green"
                
                def mx_fn(gateStr,x,y):
                    return _ph.logl_matrix( maps[gateStr], dataset, gateset, minProbClipForWeighting,
                                            probs_precomp_dict, gss.aliases)

            elif typ == "blank":
                precomp=False
                colormapType = "trivial"

                def mx_fn(gateStr,x,y):
                    return _np.nan * _np.zeros( (len(gss.effectfids),len(gss.prepfids)), 'd')

            elif typ == "errorrate":
                precomp=False
                colormapType = "seq"

                assert(sumUp == True),"Can only use 'errorrate' plot with sumUp == True"
                def mx_fn(gateStr,x,y): #error rate as 1x1 matrix which we have plotting function sum up
                    return _np.array( [[ _ph.small_eigval_err_rate(gateStr, dataset, directGSTgatesets) ]] )

            elif typ == "directchi2":
                precomp=False
                colormapType = "linlog"
                linlog_color = "blue"
                
                def mx_fn(gateStr,x,y):
                    return _ph.direct_chi2_matrix(
                        gateStr, dataset, directGSTgatesets.get(gateStr,None),
                        (gss.prepfids, gss.effectfids), minProbClipForWeighting,
                        gss.fidpair_filters[(x,y)] if (gss.fidpair_filters is not None) else None,
                        gss.gatestring_filters[(x,y)] if (gss.gatestring_filters is not None) else None,
                        gss.aliases)

            elif typ == "directlogl":
                precomp=False
                colormapType = "linlog"
                linlog_color = "yellow"
                
                def mx_fn(gateStr,x,y):
                    return _ph.direct_logl_matrix(
                        gateStr, dataset, directGSTgatesets.get(gateStr,None),
                        (gss.prepfids, gss.effectfids), minProbClipForWeighting,
                        gss.fidpair_filters[(x,y)] if (gss.fidpair_filters is not None) else None,
                        gss.gatestring_filters[(x,y)] if (gss.gatestring_filters is not None) else None,
                        gss.aliases)

            else:
                raise ValueError("Invalid plot type: %s" % typ)

            if precomp and probs_precomp_dict is None: #bulk-compute probabilities for performance
                probs_precomp_dict = _ph._computeProbabilities(maps, gateset, dataset)

            subMxs = _ph._computeSubMxs(gss,mx_fn,sumUp)
            n_boxes, dof_per_box = _ph._compute_num_boxes_dof(subMxs, gss.used_maxLs, gss.used_germs, sumUp)
            dataMax = max( [ (_np.max(mx) if (mx is not None) else 0) for subMxRow in subMxs for mx in subMxRow] )

            if colormapType == "linlog":
                colormap = _colormaps.LinlogColormap(0, dataMax, n_boxes,
                                    linlg_pcntle, dof_per_box, linlog_color)

            elif colormapType == "trivial":
                colormap = _colormaps.SequentialColormap(vmin=0, vmax=1)

            elif colormapType == "seq":
                max_abs = max([ _np.max(_np.abs(_np.nan_to_num(subMxs[iy][ix])))
                                for ix in range(len(gss.used_maxLs))
                                for iy in range(len(gss.used_germs)) ])
                if max_abs == 0: max_abs = 1e-6 # pick a nonzero value if all entries are zero or nan
                colormap = _colormaps.SequentialColormap(vmin=0, vmax=max_abs)

            else: assert(False) #invalid colormapType was set above
            
            newfig = gatestring_color_boxplot(gss, subMxs, colormap,
                                              False, boxLabels, prec,
                                              hoverInfo, invert,sumUp)
            if fig is None:
                fig = newfig
            else:
                newfig['data'][0].update(visible=False)
                fig['data'].append(newfig['data'][0])

        nTraces = len(fig['data'])
        assert(nTraces == len(plottypes))

        if nTraces > 1:
            buttons = []
            for i,nm in enumerate(plottypes):
                visible = [False]*nTraces
                visible[i] = True
                buttons.append(
                    dict(args=['visible', visible],
                         label=nm,
                         method='restyle') )
            fig['layout'].update(
                updatemenus=list([
                    dict(buttons=buttons,
                         direction = 'left',
                         pad = {'r': 10, 't': 10},
                         showactive = True, type = 'buttons',
                         x = 0.1, xanchor = 'left',
                         y = 1.1, yanchor = 'top')
                    ]) )

        #colormap2 = _colormaps.LinlogColormap(0, dataMax, n_boxes, linlg_pcntle, dof_per_box, "blue")
        #fig2 = gatestring_color_boxplot(gss, subMxs, colormap2,
        #                                False, boxLabels, prec, hoverInfo, invert,sumUp)
        #fig['data'].append(fig2['data'][0])
        #fig['layout'].update(
        #    )
        return fig

    
#def gate_matrix_boxplot(gateMatrix, size=None, m=-1.0, M=1.0,
#                        save_to=None, fontSize=20, mxBasis=None,
#                        mxBasisDims=None, xlabel=None, ylabel=None,
#                        title=None, boxLabels=False, prec=0, mxBasisDimsY=None):
    """
    Creates a color box plot of a gate matrix using a diverging color map.

    This can be a useful way to display large matrices which have so many
    entries that their entries cannot easily fit within the width of a page.

    Parameters
    ----------
    gateMatrix : ndarray
      The gate matrix data to display.

    size : tuple, optional
      The (width,height) figure size in inches.  None
      enables automatic calculation based on gateMatrix
      size.

    m, M : float, optional
      Min and max values of the color scale.

    save_to : str, optional
      save figure as this filename (usually ending in .pdf)

    fontSize : int, optional
      size of font for title

    mxBasis : str, optional
      The name abbreviation for the basis. Typically in {"pp","gm","std","qt"}.
      Used to label the rows & columns.  If you don't want labels, leave as
      None.

    mxBasisDims : int or list, optional
      The dimension of the density matrix space this basis spans, or a
      list specifying the dimensions of terms in a direct-sum
      decomposition of the density matrix space.  Used to label the
      rows & columns.  If you don't want labels, leave as None.

    xlabel : str, optional
      An x-axis label for the plot.

    ylabel : str, optional
      A y-axis label for the plot.

    title : str, optional
      A title for the plot.

    boxLabels : bool, optional
        Whether box labels are displayed.  If False, then a colorbar is
        displayed to the right of the box plot.

    prec : int or {'compact','compacthp'}, optional
        Precision for box labels.  Only relevant when boxLabels == True. Allowed
        values are:

        - 'compact' = round to nearest whole number using at most 3 characters
        - 'compacthp' = show as much precision as possible using at most 3 characters
        - int >= 0 = fixed precision given by int
        - int <  0 = number of significant figures given by -int

    mxBasisDimsY : int or list, optional
        Specifies the dimension of the basis along the Y-axis direction
        if and when this is *different* from the X-axis direction.  If
        the two are the same, this parameter can be set to None.


    Returns
    -------
    ReportFigure
    """
class GateMatrixPlot(WorkspacePlot):
    # separate in rendering/saving: size=None,fontSize=20, save_to=None, title=None, scale
    def __init__(self, ws, gateMatrix, m=-1.0, M=1.0,
                 mxBasis=None, mxBasisDims=None, xlabel=None, ylabel=None,
                 boxLabels=False, prec=0, mxBasisDimsY=None, scale=1.0):
        super(GateMatrixPlot,self).__init__(ws, self._create, gateMatrix, m, M,
                                            mxBasis, mxBasisDims, xlabel, ylabel,
                                            boxLabels, prec, mxBasisDimsY, scale)
          
    def _create(self, gateMatrix, m, M, 
                mxBasis, mxBasisDims, xlabel, ylabel,
                boxLabels, prec, mxBasisDimsY, scale):
        return gatematrix_color_boxplot(
            gateMatrix, m, M, mxBasis, mxBasisDims, mxBasisDimsY,
            xlabel, ylabel, boxLabels, prec, scale)


    """
    Creates a color box plot of a the error generator of a gate matrix.

    The error generator is given by log( inv(targetMatrix) * gateMatrix ).
    This can be a useful way to display large matrices which have so many
    entries that their entries cannot easily fit within the width of a page.

    Parameters
    ----------
    gate : ndarray
      The gate matrix data used when constructing the generator.

    targetGate : ndarray
      The target gate matrix data to use when constructing the the
      generator.

    size : tuple, optional
      The (width,height) figure size in inches.

    title : str, optional
      A title for the plot.

    save_to : str, optional
      save figure as this filename (usually ending in .pdf)

    fontSize : int, optional
      size of font for title

    showNormal : bool, optional
      whether to display the actual eigenvalues of the gate
      and the target gate on the plot.

    showRelative : bool, optional
      whether to display the relative eigenvalues of the gate
      relative to the target gate on the plot.

    Returns
    -------
    ReportFigure
    """

#    evals = _np.linalg.eigvals(gate)
#    target_evals = _np.linalg.eigvals(targetGate)
#    rel_gate = _np.dot(_np.linalg.inv(targetGate), gate)
#    rel_evals = _np.linalg.eigvals(rel_gate)
#    rel_evals10 = rel_evals**10
   
class PolarEigenvaluePlot(WorkspacePlot):
    def __init__(self, ws, evals_list, colors, labels=None, scale=1.0, amp=10,
                 centerText=None):
        super(PolarEigenvaluePlot,self).__init__(ws, self._create, evals_list,
                                                 colors, labels, scale, amp,
                                                 centerText)
        
    def _create(self, evals_list, colors, labels, scale, amp, centerText):

        annotations = []
        if centerText is not None:
            annotations.append(
                dict(text=centerText,
                     r=0, t=0,
                     font=dict(size=10*scale,
                               color="black",
                               showarrow=False)
                ))

        data = []
        for i,evals in enumerate(evals_list):
            color = colors[i] if (colors is not None) else "black"
            trace = go.Scatter(
                r = _np.absolute(evals),
                t = _np.angle(evals) * (180.0/_np.pi),
                mode='markers',
                marker=dict(
                    color=color,
                    size=110,
                    line=dict(
                        color='white'
                    ),
                    opacity=0.7
                ))
            if labels is not None:
                trace.update(name=labels[i])
            data.append(trace)

            #Add amplified eigenvalues
            if amp is not None:
                amp_evals = evals**amp
                trace = go.Scatter(
                    r = _np.absolute(amp_evals),
                    t = _np.angle(amp_evals) * (180.0/_np.pi),
                    mode='markers',
                    marker=dict(
                        color=color,
                        size=50,
                        line=dict(
                            color='white'
                        ),
                        opacity=0.5
                    ))
                if labels is not None:
                    trace.update(name="%s^%g" % (labels[i],amp))
                data.append(trace)
            
        layout = go.Layout(
            #title='Test Polar',
            #font=dict(size=15),
            plot_bgcolor='rgb(240, 240, 240)',
            radialaxis=dict(
                range=[0,1.25]),
            angularaxis=dict(
                tickcolor='rgb(180,180,180)',
                #range=[0,2]
                #ticktext=['A','B','C','D']
            ),
            direction="counterclockwise",
            orientation=-90
        )
        
        return go.Figure(data=data, layout=layout)



    """
    Creates a color box plot displaying the projections of a gate error
    generator onto generators corresponding to a set of standard error
    generators constructed from the given basis.  Typically `projections`
    is obtained by calling :func:`std_errgen_projections`.

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

    m,M : float, optional
      Color scale min and max values, respectivey.  If None, then computed
      automatically from the data range.

    size : tuple, optional
      The (width,height) figure size in inches.  None
      enables automatic calculation based on gateMatrix
      size.

    title : str, optional
      A title for the plot.

    save_to : str, optional
      save figure as this filename (usually ending in .pdf)

    fontSize : int, optional
        size of font for title

    boxLabels : bool, optional
        Whether box labels are displayed.  If False, then a colorbar is
        displayed to the right of the box plot.

    prec : int or {'compact','compacthp'}, optional
        Precision for box labels.  Only relevant when boxLabels == True. Allowed
        values are:

        - 'compact' = round to nearest whole number using at most 3 characters
        - 'compacthp' = show as much precision as possible using at most 3 characters
        - int >= 0 = fixed precision given by int
        - int <  0 = number of significant figures given by -int


    Returns
    -------
    ReportFigure
    """
class ProjectionsBoxPlot(WorkspacePlot):
    def __init__(self, ws, projections, projection_basis, m=None, M=None,
                 boxLabels=False, prec="compacthp", scale=1.0):
        super(ProjectionsBoxPlot,self).__init__(ws, self._create, projections,
                                                 projection_basis, m, M,
                                                 boxLabels, prec, scale)
        
    def _create(self, projections,
                projection_basis, m, M,
                boxLabels, prec, scale):

        absMax = _np.max(_np.abs(projections))
        if m is None: m = -absMax
        if M is None: M =  absMax
    
        d2 = len(projections) # number of projections == dim of gate
        d = int(_np.sqrt(d2)) # dim of density matrix
        nQubits = _np.log2(d)
    
        if not _np.isclose(round(nQubits),nQubits):
            #Non-integral # of qubits, so just show as a single row
            projections = projections.reshape( (1,projections.size) )
            xlabel = ""; ylabel = ""        
        elif nQubits == 1:
            projections = projections.reshape( (1,4) )
            xlabel = "Q1"; ylabel = ""
        elif nQubits == 2:
            projections = projections.reshape( (4,4) )
            xlabel = "Q2"; ylabel="Q1"
        else:
            projections = projections.reshape( (4,projections.size/4) )
            xlabel = "Q*"; ylabel="Q1"
    
        xd = int(round(_np.sqrt(projections.shape[1]))) #x-basis-dim
        yd = int(round(_np.sqrt(projections.shape[0]))) #y-basis-dim
    
        return gatematrix_color_boxplot(
            projections, m, M, projection_basis, xd, yd,
            xlabel, ylabel, boxLabels, prec,  scale)



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

    size : tuple, optional
      The (width,height) figure size in inches.

    barWidth : float, optional
      The width of the bars in the plot.

    save_to : str, optional
      save figure as this filename (usually ending in .pdf)

    fontSize : int, optional
      size of font for title

    xlabel : str, optional
      An x-axis label for the plot.

    ylabel : str, optional
      A y-axis label for the plot.

    title : str, optional
      A title for the plot.

    Returns
    -------
    ReportFigure
    """

#    def choi_eigenvalue_barplot(evals, errbars=None, size=(8,5), barWidth=1,
#                            save_to=None, fontSize=15, xlabel="index",
#                            ylabel="Re[eigenvalue]", title=None):

# xlabel="index", ylabel="Re[eigenvalue]", title=None
# TODO: maybe a "postFormat" or "addToFigure" fn to add title & axis labels to any figure?
class ChoiEigenvalueBarPlot(WorkspacePlot):
    def __init__(self, ws, evals, errbars=None):
        super(ChoiEigenvalueBarPlot,self).__init__(ws, self._create, evals,
                                                   errbars)
        
    def _create(self, evals, errbars):

        xs = list(range(evals.size))
        ys = []; colors = []; texts=[]
        for i,ev in enumerate(evals.flatten()):
            ys.append( abs(ev.real) )
            colors.append('rgb(200,200,200)' if ev.real > 0 else 'red')
            if errbars is not None:
                texts.append("%g +/- %g" % (ev.real,errbars.flatten()[i].real))
            else:
                texts.append("%g" % ev.real)
                
        trace = go.Bar(
            x=xs, y=ys, text=texts,
            marker=dict(color=colors)
        )

        log_ys = _np.log10(_np.array(ys,'d'))
        minlog = _np.floor(min(log_ys))
        maxlog = _np.ceil(max(log_ys))
        
        data = [trace]
        layout = go.Layout(
            xaxis = dict(
                title="index",
                tickvals=xs
                ),
            yaxis = dict(
                type='log',
                range=[minlog,maxlog]
                ),
            bargap=0.02
        )
        
        return go.Figure(data=data, layout=layout)


#Histograms??
#TODO: histogram
#        if histogram:
#            fig = _plt.figure()
#            histdata = subMxSums.flatten()
#            histdata_finite = _np.take(histdata, _np.where(_np.isfinite(histdata)))[0] #take gives back (1,N) shaped array (why?)
#            histMin = min( histdata_finite ) if cmapFactory.vmin is None else cmapFactory.vmin
#            histMax = max( histdata_finite ) if cmapFactory.vmax is None else cmapFactory.vmax
#            _plt.hist(_np.clip(histdata_finite,histMin,histMax), histBins,
#                      range=[histMin, histMax], facecolor='gray', align='mid')
#            if save_to is not None:
#                if len(save_to) > 0:
#                    _plt.savefig( _makeHistFilename(save_to) )
#                _plt.close(fig)

#           if histogram:
#                fig = _plt.figure()
#                histdata = _np.concatenate( [ subMxs[iy][ix].flatten() for ix in range(nXs) for iy in range(nYs)] )
#                histdata_finite = _np.take(histdata, _np.where(_np.isfinite(histdata)))[0] #take gives back (1,N) shaped array (why?)
#                histMin = min( histdata_finite ) if cmapFactory.vmin is None else cmapFactory.vmin
#                histMax = max( histdata_finite ) if cmapFactory.vmax is None else cmapFactory.vmax
#                _plt.hist(_np.clip(histdata_finite,histMin,histMax), histBins,
#                          range=[histMin, histMax], facecolor='gray', align='mid')
#                if save_to is not None:
#                    if len(save_to) > 0:
#                        _plt.savefig( _makeHistFilename(save_to) )
#                    _plt.close(fig)
