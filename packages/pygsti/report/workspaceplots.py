from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Classes corresponding to plots within a Workspace context."""

import numpy             as _np
import scipy             as _scipy
import os                as _os
import warnings          as _warnings

from .. import algorithms   as _alg
from .. import tools        as _tools
from .. import construction as _construction
from .. import objects      as _objs

from .workspace import WorkspacePlot
from . import colormaps as _colormaps
from . import plothelpers as _ph

import plotly.graph_objs as go


import time as _time  #DEBUG TIMER
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot #DEBUG


def color_boxplot(plt_data, colormap, colorbar=False, boxLabelSize=0,
                  prec=0, hoverLabelFn=None, hoverLabels=None):
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

    boxLabelSize : int, optional
        If greater than 0, display static labels on each box with font
        size equal to `boxLabelSize`.

    prec : int or {'compact','compacthp'}, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    hoverLabelFn : function, optional
        A function with signature `f(z,i,j)` where `z ==plt_data[i,j]` which
        computes the hover label for the each element of `plt_data`.  Cannot
        be used with `hoverLabels`.

    hoverLabels : list of lists, optional
        Strings specifying the hover labels for each element of `plt_data`.
        E.g. `hoverLabels[i,j]` is the string for the i-th row (y-value) 
        and j-th column (x-value) of the plot.

    Returns
    -------
    plotly.Figure
    """

    masked_data = _np.ma.array(plt_data, mask=_np.isnan(plt_data))
    heatmapArgs = { 'z': colormap.normalize(masked_data),
                    'colorscale': colormap.get_colorscale(),
                    'showscale': colorbar, 'hoverinfo': 'none',
                    'zmin': colormap.hmin, 'zmax': colormap.hmax }
    
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
    """
    Creates a "nested" color box plot by tiling the plaquettes given
    by `plt_data_list_of_lists` onto a single heatmap.

    Parameters
    ----------
    plt_data_list_of_lists : list of lists of numpy arrays
        A complete square 2D list of lists, such that each element is a
        2D numpy array of the same size.

    colormap : Colormap
        The colormap used to determine box color.

    colorbar : bool, optional
        Whether or not to show the color scale bar.

    boxLabelSize : int, optional
        If greater than 0, display static labels on each box with font
        size equal to `boxLabelSize`.

    prec : int or {'compact','compacthp'}, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    hoverLabelFn : function, optional
        A function with signature `f(z,i,j)` where `z ==plt_data[i,j]` which
        computes the hover label for the each element of `plt_data`.  Cannot
        be used with `hoverLabels`.

    Returns
    -------
    plotly.Figure
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
                     sumUp=False, invert=False, scale=1.0):
    """
    A helper function for generating typical nested color box plots used in pyGSTi.

    Given the list-of-lists, `subMxs`, along with x and y labels for both the "outer"
    (i.e. the list-indices) and "inner" (i.e. the sub-matrix-indices) axes, this function
    will produce a nested color box plot with the option of summing over the inner axes
    or inverting (swapping) the inner and outer axes.

    Parameters
    ----------
    subMxs : list
        A list of lists of 2D numpy.ndarrays.  subMxs[iy][ix] specifies the matrix of values
        or sum (if sumUp == True) displayed in iy-th row and ix-th column of the plot.  NaNs
        indicate elements should not be displayed.

    x_labels, y_labels : list
        Labels for the outer x- and y-axis values.

    inner_x_labels, inner_y_labels : list
        Labels for the inner x- and y-axis values.

    xlabel, ylabel : str
        Outer X and Y axis labels.

    inner_xlabel, inner_ylabel : str
        Inner X and Y axis labels.

    colormap : Colormap
        The colormap used to determine box color.

    colorbar : bool, optional
        Whether or not to show the color scale bar.

    boxLabels : bool, optional
        Whether to display static value-labels over each box.

    prec : int or {'compact','compacthp'}, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    hoverInfo : bool, optional
        Whether to incude interactive hover labels.

    sumUp : bool, optional
        False displays each matrix element as it's own color box
        True sums the elements of each (x,y) matrix and displays
        a single color box for the sum.

    invert : bool, optional
        If True, invert the nesting order of the nested color box plot
        (applicable only when sumUp == False).  E.g. use inner_x_labels and
        inner_y_labels to label the x and y axes.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    Returns
    -------
    plotly.Figure
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
                    mx = _np.array( [[ subMxs[iy][ix][iny,inx] for ix in range(nXs) ]
                                     for iy in range(nYs)],  'd' )
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
                         sumUp, False, scale)
                                

    def val_filter(vals):  #filter to latex-ify gate strings.  Later add filter as a possible parameter
        formatted_vals = []
        for val in vals:
            if (isinstance(val,tuple) or isinstance(val,_objs.GateString)) \
               and all([_tools.isstr(el) for el in val]):
                if len(val) == 0:
                    #formatted_vals.append(r"$\{\}$")
                    formatted_vals.append(r"{}")
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
        fig['layout'].update(width=80*(nXs+3)*scale,
                             height=80*(nYs+3)*scale)

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
        assert(fig is not None), "No data to display!"
        
        fig['layout'].update(width=30*(nXs*nIXs+5)*scale,
                             height=30*(nYs*nIYs+5)*scale)
        
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
                             sumUp=False,invert=False,scale=1.0):
    """
    A wrapper around :func:`generate_boxplot` for creating color box plots
    when the structure of the gate strings is contained in  a
    `GatestringStructure` object.

    Parameters
    ----------
    gatestring_structure : GatestringStructure
        Specifies a set of gate sequences along with their outer and inner x,y
        structure, e.g. fiducials, germs, and maximum lengths.

    subMxs : list
        A list of lists of 2D numpy.ndarrays.  subMxs[iy][ix] specifies the matrix of values
        or sum (if sumUp == True) displayed in iy-th row and ix-th column of the plot.  NaNs
        indicate elements should not be displayed.

    colormap : Colormap
        The colormap used to determine box color.

    colorbar : bool, optional
        Whether or not to show the color scale bar.

    boxLabels : bool, optional
        Whether to display static value-labels over each box.

    prec : int or {'compact','compacthp'}, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    hoverInfo : bool, optional
        Whether to incude interactive hover labels.

    sumUp : bool, optional
        False displays each matrix element as it's own color box
        True sums the elements of each (x,y) matrix and displays
        a single color box for the sum.

    invert : bool, optional
        If True, invert the nesting order of the nested color box plot
        (applicable only when sumUp == False).  E.g. use inner_x_labels and
        inner_y_labels to label the x and y axes.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    Returns
    -------
    plotly.Figure
    """
    g = gatestring_structure
    return generate_boxplot(subMxs,
                            list(map(str,g.used_xvals())), list(map(str,g.used_yvals())),
                            list(map(str,g.minor_xvals())), list(map(str,g.minor_yvals())),
                            "L","germ","rho","E<sub>i</sub>", colormap,
                            colorbar, boxLabels, prec, hoverInfo,
                            sumUp, invert, scale)  #"$\\rho_i$","$\\E_i$"      


def gatematrix_color_boxplot(gateMatrix, m, M, mxBasis=None, mxBasisDims=None,
                             mxBasisDimsY=None, xlabel=None, ylabel=None,
                             boxLabels=False, prec=0, scale=1.0):
    """
    Creates a color box plot for visualizing a single matrix.

    Parameters
    ----------
    gateMatrix : numpy array
        The matrix to visualize.

    m, M : float
        Minimum and maximum of the color scale.

    mxBasis : str
      The name abbreviation for the basis. Typically in {"pp","gm","std","qt"}.
      Used to label the rows & columns.  If you don't want labels, set to None.

    mxBasisDims, mxBasisDimsY : int or list, optional
      The dimension of the density matrix space, or a list specifying the
      dimensions of terms in a direct-sum decomposition of the density matrix
      space, for the X and Y axes (columns and rows).  Used to label the 
      columns and rows, respectively.  If `myBasisDimsY` is None if will 
      default to `mxBasisDims`.

    xlabel, ylabel : str, optional
      Axis labels for the plot.

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

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    Returns
    -------
    plotly.Figure
    """

    def one_sigfig(x):
        if abs(x) < 1e-9: return 0
        if x < 0: return -one_sigfig(-x)
        e = -int(_np.floor(_np.log10(abs(x)))) #exponent
        trunc_x = _np.floor(x * 10**e)/ 10**e #truncate decimal to make sure it gets *smaller*
        return round(trunc_x, e) #round to truncation point just to be sure

    xextra = 2.25 if ylabel is None else 2.5
    yextra = 2.25 if xlabel is None else 2.5

    if mxBasis is not None and mxBasisDims is not None:
        if mxBasisDimsY is None: mxBasisDimsY = mxBasisDims
        xlabels=[("<i>%s</i>" % x) if len(x) else "" \
                 for x in _tools.basis_element_labels(mxBasis,mxBasisDims)]
        ylabels=[("<i>%s</i>" % x) if len(x) else "" \
                 for x in _tools.basis_element_labels(mxBasis,mxBasisDimsY)]
        yextra += 0.5 if (mxBasisDims > 1) else 0
        xextra += 0.5 if (mxBasisDimsY > 1) else 0
    else:
        xlabels = [""] * gateMatrix.shape[1]
        ylabels = [""] * gateMatrix.shape[0]

    colormap = _colormaps.DivergingColormap(vmin=m, vmax=M)
    
    flipped_mx = _np.flipud(gateMatrix)  # FLIP so [0,0] matrix el is at *top* left
    ylabels    = list(reversed(ylabels)) # FLIP y-labels to match
    trace = go.Heatmap(z=colormap.normalize(flipped_mx),
                       colorscale=colormap.get_colorscale(),
                       showscale=(not boxLabels), zmin=colormap.hmin,
                       zmax=colormap.hmax, hoverinfo='z')
    data = [trace]
    
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
        width = 40*(gateMatrix.shape[1]+xextra)*scale,
        height = 40*(gateMatrix.shape[0]+yextra)*scale,
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
        annotations = annotations,
        margin = go.Margin(l=50,r=50,b=50,t=50) #pad=0
    )
    #print("DEBUG width = ",40*(gateMatrix.shape[1]+xextra)*scale)
    #print("DEBUG height = ",40*(gateMatrix.shape[0]+yextra)*scale)
    return go.Figure(data=data, layout=layout)



class BoxKeyPlot(WorkspacePlot):
    def __init__(self, ws, prepStrs, effectStrs,
                 xlabel="Preparation fiducial", ylabel="Measurement fiducial", scale=1.0):
        """
        Create a plot showing the layout of a single sub-block of a goodness-of-fit
        box plot (such as those produced by ColorBoxPlot)
    
        Parameters
        ----------
        prepStrs, effectStrs : list of GateStrings
            Preparation and measurement fiducials.
    
        xlabel, ylabel : str, optional
            X and Y axis labels
    
        scale : float, optional
            Scaling factor to adjust the size of the final figure.
        """
        # separate in rendering/saving: save_to=None, ticSize=20, scale=1.0 (?)
        super(BoxKeyPlot,self).__init__(ws, self._create, prepStrs, effectStrs,
                                         xlabel, ylabel, scale)

          #size, save_to,
          
    def _create(self, prepStrs, effectStrs, xlabel, ylabel, scale):
        
        #Copied from generate_boxplot
        def val_filter(vals):  #filter to latex-ify gate strings.  Later add filter as a possible parameter
            formatted_vals = []
            for val in vals:
                if isinstance(val, (tuple,_objs.GateString)) and all([_tools.isstr(el) for el in val]):
                    if len(val) == 0:
                        #formatted_vals.append(r"$\{\}$")
                        formatted_vals.append(r"{}")
                    else:
                        #formatted_vals.append( "$" + "\\cdot".join([("\\mathrm{%s}" % el) for el in val]) + "$" )
                        formatted_vals.append(str(val))
                else:
                    formatted_vals.append(val)
            return formatted_vals

        nX = len(prepStrs)
        nY = len(effectStrs)
        trace = go.Heatmap(z=_np.zeros((nY,nX),'d'),
                           colorscale=[ [0, 'white'], [1, 'black'] ],
                           showscale=False, zmin=0,zmax=1,hoverinfo='none')
        data = [trace]

        gridlines = []
    
        # Vertical lines
        for i in range(nX-1):
            gridlines.append(     
                {
                    'type': 'line',
                    'x0': i+0.5, 'y0': -0.5,
                    'x1': i+0.5, 'y1': nY-0.5,
                    'line': {'color': 'black', 'width': 1},
                } )
            
        #Horizontal lines
        for i in range(nY-1):
            gridlines.append(     
                {
                    'type': 'line',
                    'x0': -0.5, 'y0': i+0.5,
                    'x1': nX-0.5, 'y1': i+0.5,
                    'line': {'color': 'black', 'width': 1},
                } )

        layout = go.Layout(
            width = 40*(nX+1.5)*scale,
            height = 40*(nY+1.5)*scale,
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
                tickvals=[i for i in range(len(prepStrs))],
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
                ticktext=list(reversed(val_filter(effectStrs))),
                tickvals=[i for i in range(len(effectStrs))],
            ),
            shapes = gridlines,
            annotations = [
                go.Annotation(
                    x=0.5,
                    y=1.2,
                    showarrow=False,
                    text=xlabel,
                    font={'size': 12*scale, 'color': "black"},
                    xref='paper',
                    yref='paper'),
                go.Annotation(
                    x=-0.2,
                    y=0.5,
                    showarrow=False,
                    textangle=-90,
                    text=ylabel,
                    font={'size': 12*scale, 'color': "black"},
                    xref='paper',
                    yref='paper'
                )
            ]
        )
        # margin = go.Margin(l=50,r=50,b=50,t=50) #pad=0
        return go.Figure(data=data, layout=layout)






        
#        layout = go.Layout(
#            width=70*scale*(len(prepStrs)+1),
#            height=70*scale*(len(effectStrs)+1),
#
#
#        xs = [i+0.5 for i in range(len(prepStrs))]
#        ys = [i+0.5 for i in range(len(effectStrs))]
#        allys = []
#        for y in ys: allys.extend( [y]*len(xs) )
#        allxs = xs*len(ys)
#        trace = go.Scatter(x=allxs, y=allys, mode="markers",
#                           marker=dict(
#                               size = 12,
#                               color = 'rgba(0,0,255,0.8)',
#                               line = dict(
#                                   width = 2,
#                               )),
#                           xaxis='x1', yaxis='y1', hoverinfo='none')
#                          
#        fig = go.Figure(data=[trace], layout=layout)
#        return fig


    
class ColorBoxPlot(WorkspacePlot):
    def __init__(self, ws, plottype, gss, dataset, gateset,
                 sumUp=False, boxLabels=False, hoverInfo=True, invert=False,
                 prec='compact', linlg_pcntle=.05, minProbClipForWeighting=1e-4,
                 directGSTgatesets=None, dscomparator=None, submatrices=None, scale=1.0):
        """
        Create a plot displaying the value of per-gatestring quantities.

        Values are shown on a grid of colored boxes, organized according to
        the structure of the gate strings (e.g. by germ and "L").
    
        Parameters
        ----------
        plottype : {"chi2","logl","blank","errorrate","directchi2","directlogl","dscmp"}
            Specifies the type of plot. "errorate", "directchi2" and
            "directlogl" require that `directGSTgatesets` be set.

        gss : GatestringStructure
            Specifies the set of gate strings along with their structure, e.g.
            fiducials, germs, and maximum lengths.
    
        dataset : DataSet
            The data used to specify frequencies and counts.
    
        gateset : GateSet
            The gate set used to specify the probabilities and SPAM labels.
                
        sumUp : bool, optional
            False displays each matrix element as it's own color box
            True sums the elements of each (x,y) matrix and displays
            a single color box for the sum.
    
        boxLabels : bool, optional
            Whether box labels are displayed.  It takes much longer to
            generate the figure when this is set to True.

        hoverInfo : bool, optional
            Whether to incude interactive hover labels.

        invert : bool, optional
            If True, invert the nesting order of the color box plot (applicable
            only when sumUp == False).

        prec : int, optional
            Precision for box labels.  Allowed values are:
              'compact' = round to nearest whole number using at most 3 characters
              'compacthp' = show as much precision as possible using at most 3 characters
              int >= 0 = fixed precision given by int
              int <  0 = number of significant figures given by -int

        linlg_pcntle : float, optional
            Specifies the (1 - linlg_pcntle) percentile to compute for the boxplots    
    
        minProbClipForWeighting : float, optional
            Defines a clipping point for the statistical weight used
            within the chi^2 or logl functions.

        directGSTgatesets : dict, optional
            A dictionary of "direct" Gatesets used when displaying certain plot
            types.  Keys are gate strings and values are corresponding gate
            sets (see `plottype` above).        

        dscomparator : DataComparator, optional
            The data set comparator used to produce the "dscmp" plot type.

        submatrices : dict, optional
            A dictionary whose keys correspond to other potential plot
            types and whose values are each a list-of-lists of the sub
            matrices to plot, corresponding to the used x and y values
            of `gss`.

        scale : float, optional
            Scaling factor to adjust the size of the final figure.
        """
        # separate in rendering/saving: save_to=None, ticSize=20, scale=1.0 (?)
        super(ColorBoxPlot,self).__init__(ws, self._create, plottype, gss, dataset, gateset,
                                          prec, sumUp, boxLabels, hoverInfo,
                                          invert, linlg_pcntle, minProbClipForWeighting,
                                          directGSTgatesets, dscomparator, submatrices, scale)

    def _create(self, plottypes, gss, dataset, gateset,
                prec, sumUp, boxLabels, hoverInfo,
                invert, linlg_pcntle, minProbClipForWeighting,
                directGSTgatesets, dscomparator, submatrices, scale):

        #OLD: maps = _ph._computeGateStringMaps(gss, dataset)
        probs_precomp_dict = None
        fig = None

        if _tools.isstr(plottypes):
            plottypes = [plottypes]

        for typ in plottypes:
            if typ == "chi2":
                precomp=True
                colormapType = "linlog"
                linlog_color = "red"
                
                def mx_fn(plaq,x,y):
                    return _ph.chi2_matrix( plaq, dataset, gateset, minProbClipForWeighting,
                                            probs_precomp_dict)
            elif typ == "logl":
                precomp=True
                colormapType = "linlog"
                linlog_color = "red"
                
                def mx_fn(plaq,x,y):
                    return _ph.logl_matrix( plaq, dataset, gateset, minProbClipForWeighting,
                                            probs_precomp_dict)

            elif typ == "blank":
                precomp=False
                colormapType = "trivial"

                def mx_fn(plaq,x,y):
                    return _np.nan * _np.zeros( (len(gss.minor_yvals()),
                                                 len(gss.minor_xvals())), 'd')

            elif typ == "errorrate":
                precomp=False
                colormapType = "seq"

                assert(sumUp == True),"Can only use 'errorrate' plot with sumUp == True"
                def mx_fn(plaq,x,y): #error rate as 1x1 matrix which we have plotting function sum up
                    return _np.array( [[ _ph.small_eigval_err_rate(plaq.base, dataset, directGSTgatesets) ]] )

            elif typ == "directchi2":
                precomp=False
                colormapType = "linlog"
                linlog_color = "yellow"
                
                def mx_fn(plaq,x,y):
                    return _ph.direct_chi2_matrix(
                        plaq, gss, dataset,
                        directGSTgatesets.get(plaq.base,None),
                        minProbClipForWeighting)

            elif typ == "directlogl":
                precomp=False
                colormapType = "linlog"
                linlog_color = "yellow"
                
                def mx_fn(plaq,x,y):
                    return _ph.direct_logl_matrix(
                        plaq, gss, dataset,
                        directGSTgatesets.get(plaq.base,None),
                        minProbClipForWeighting)

            elif typ == "dscmp":
                precomp=False
                colormapType = "linlog"
                linlog_color = "green"
                assert(dscomparator is not None), \
                    "Must specify `dscomparator` argument to create `dscmp` plot!"

                def mx_fn(plaq,x,y):
                    return _ph.dscompare_llr_matrices(plaq, dscomparator)                

            elif (submatrices is not None) and typ in submatrices:
                precomp = False
                if typ + ".colormap" in submatrices:
                    colormapType = submatrices[typ + ".colormap"]
                else:
                    colormapType = "seq"
            
            else:
                raise ValueError("Invalid plot type: %s" % typ)

            if precomp and probs_precomp_dict is None: #bulk-compute probabilities for performance
                probs_precomp_dict = _ph._computeProbabilities(gss, gateset, dataset)

            if (submatrices is not None) and typ in submatrices:
                subMxs = submatrices[typ] # "custom" type -- all mxs precomputed by user
            else:
                subMxs = _ph._computeSubMxs(gss,mx_fn,sumUp)
            
            n_boxes, dof_per_box = _ph._compute_num_boxes_dof(subMxs, gss.used_xvals(), gss.used_yvals(), sumUp)
            if len(subMxs) > 0:                
                dataMax = max( [ (0 if (mx is None or _np.all(_np.isnan(mx))) else _np.nanmax(mx))
                                 for subMxRow in subMxs for mx in subMxRow] )
            else: dataMax = 0

            if colormapType == "linlog":
                colormap = _colormaps.LinlogColormap(0, dataMax, n_boxes,
                                    linlg_pcntle, dof_per_box, linlog_color)

            elif colormapType == "trivial":
                colormap = _colormaps.SequentialColormap(vmin=0, vmax=1)

            elif colormapType in ("seq","revseq"):
                max_abs = max([ _np.max(_np.abs(_np.nan_to_num(subMxs[iy][ix])))
                                for ix in range(len(gss.used_xvals()))
                                for iy in range(len(gss.used_yvals())) ])
                if max_abs == 0: max_abs = 1e-6 # pick a nonzero value if all entries are zero or nan
                color = "whiteToBlack" if colormapType == "seq" else "blackToWhite"
                colormap = _colormaps.SequentialColormap(vmin=0, vmax=max_abs, color=color)
                
            else: assert(False) #invalid colormapType was set above
            
            newfig = gatestring_color_boxplot(gss, subMxs, colormap,
                                              False, boxLabels, prec,
                                              hoverInfo, sumUp, invert,
                                              scale)
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
        #                                False, boxLabels, prec, hoverInfo, sumUp, invert)
        #fig['data'].append(fig2['data'][0])
        #fig['layout'].update(
        #    )
        return fig

    
#def gate_matrix_boxplot(gateMatrix, size=None, m=-1.0, M=1.0,
#                        save_to=None, fontSize=20, mxBasis=None,
#                        mxBasisDims=None, xlabel=None, ylabel=None,
#                        title=None, boxLabels=False, prec=0, mxBasisDimsY=None):
class GateMatrixPlot(WorkspacePlot):
    # separate in rendering/saving: size=None,fontSize=20, save_to=None, title=None, scale
    def __init__(self, ws, gateMatrix, m=-1.0, M=1.0,
                 mxBasis=None, mxBasisDims=None, xlabel=None, ylabel=None,
                 boxLabels=False, prec=0, mxBasisDimsY=None, scale=1.0):
        """
        Creates a color box plot of a gate matrix using a diverging color map.
    
        This can be a useful way to display large matrices which have so many
        entries that their entries cannot easily fit within the width of a page.
    
        Parameters
        ----------
        gateMatrix : ndarray
          The gate matrix data to display.
        
        m, M : float, optional
          Min and max values of the color scale.
    
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

        scale : float, optional
            Scaling factor to adjust the size of the final figure.
        """
        super(GateMatrixPlot,self).__init__(ws, self._create, gateMatrix, m, M,
                                            mxBasis, mxBasisDims, xlabel, ylabel,
                                            boxLabels, prec, mxBasisDimsY, scale)
          
    def _create(self, gateMatrix, m, M, 
                mxBasis, mxBasisDims, xlabel, ylabel,
                boxLabels, prec, mxBasisDimsY, scale):
        
        return gatematrix_color_boxplot(
            gateMatrix, m, M, mxBasis, mxBasisDims, mxBasisDimsY,
            xlabel, ylabel, boxLabels, prec, scale)



#    evals = _np.linalg.eigvals(gate)
#    target_evals = _np.linalg.eigvals(targetGate)
#    rel_gate = _np.dot(_np.linalg.inv(targetGate), gate)
#    rel_evals = _np.linalg.eigvals(rel_gate)
#    rel_evals10 = rel_evals**10
   
class PolarEigenvaluePlot(WorkspacePlot):
    def __init__(self, ws, evals_list, colors, labels=None, scale=1.0, amp=None,
                 centerText=None):
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

        centerText : str, optional
            Text to be placed at the very center of the polar plot (sometimes 
            useful to use as a title).
        """
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

        #Note: plotly needs a plain lists for r and t, otherwise it
        # produces javascript [ [a], [b], ... ] instead of [a, b, ...]  
        data = []
        for i,evals in enumerate(evals_list):
            color = colors[i] if (colors is not None) else "black"
            trace = go.Scatter(
                r = list(_np.absolute(evals).flat), 
                t = list(_np.angle(evals).flatten() * (180.0/_np.pi)),
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
                trace = go.Scatter(
                    r = list(_np.absolute(amp_evals).flat),
                    t = list(_np.angle(amp_evals).flatten() * (180.0/_np.pi)),
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
            width = 300*scale,
            height = 300*scale,
            #title='Test Polar',
            #font=dict(size=10),
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

        #HACK around plotly bug: Plotly somehow holds residual polar plot data
        # which gets plotted unless new data overwrites it.  This residual data
        # takes up 4 points of data for the first 3 data traces - so we make
        # sure this data is filled in with undisplayed data here (because we
        # don't want to see the residual data!).
        for trace in data:
            if len(trace['r']) < 4:
                extra = 4-len(trace['r'])
                trace['r'] += [1e3]*extra
                trace['t'] += [0.0]*extra
        while len(data) < 3:
            data.append( go.Scatter(
                r = [1e3]*4,
                t = [0.0]*4,
                name="Dummy",
                mode='markers',
                showlegend=False,
                ))
        assert(len(data) >= 3)
        
        return go.Figure(data=data, layout=layout)



class ProjectionsBoxPlot(WorkspacePlot):
    def __init__(self, ws, projections, projection_basis, m=None, M=None,
                 boxLabels=False, prec="compacthp", scale=1.0):
        """
        Creates a color box plot displaying projections.

        Typically `projections` is obtained by calling
        :func:`std_errgen_projections`, and so holds the projections of a gate
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
    
        m,M : float, optional
          Color scale min and max values, respectivey.  If None, then computed
          automatically from the data range.
    
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

        scale : float, optional
            Scaling factor to adjust the size of the final figure.
        """
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



#    def choi_eigenvalue_barplot(evals, errbars=None, size=(8,5), barWidth=1,
#                            save_to=None, fontSize=15, xlabel="index",
#                            ylabel="Re[eigenvalue]", title=None):

# xlabel="index", ylabel="Re[eigenvalue]", title=None
# TODO: maybe a "postFormat" or "addToFigure" fn to add title & axis labels to any figure?
class ChoiEigenvalueBarPlot(WorkspacePlot):
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
        super(ChoiEigenvalueBarPlot,self).__init__(ws, self._create, evals,
                                                   errbars, scale)
        
    def _create(self, evals, errbars, scale):

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
            marker=dict(color=colors),
            hoverinfo='text'
        )

        log_ys = _np.log10(_np.array(ys,'d'))
        minlog = _np.floor(min(log_ys))
        maxlog = _np.ceil(max(log_ys))
        
        data = [trace]
        layout = go.Layout(
            width = 400*scale,
            height = 300*scale,
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



class DatasetComparisonPlot(WorkspacePlot):
    def __init__(self, ws, dsc, nbins=50, frequency=True,
                  log=True, display='pvalue', scale=1.0):
        super(DatasetComparisonPlot,self).__init__(ws, self._create, dsc, nbins, frequency,
                                                   log, display, scale)
        
    def _create(self, dsc, nbins, frequency, log, display, scale):
        if display == 'llr' and nbins is None:
            nbins = len(dsc.llrVals)

        TOL=1e-10
        pVals = dsc.pVals.astype('d')
        pVals_nz = _np.array([x for x in pVals if abs(x)>TOL])
        pVals0 = (len(pVals)-len(pVals_nz)) if log else dsc.pVals0
        llrVals = dsc.llrVals.astype('d')
        
        if log:
            minval = _np.floor(_np.log10(min(pVals_nz)))
            maxval = _np.ceil(_np.log10(max(pVals_nz)))
            thres = (maxval-minval)/(nbins-1) * (nbins-2)
            lastBinCount = (_np.log10(pVals_nz)>thres).sum()
             #Kenny: why use this as a normalization?  Is this correct?
        else:
            minval = min(pVals)
            maxval = max(pVals)
            thres = (maxval-minval)/(nbins-1) * (nbins-2)
            lastBinCount = (pVals>thres).sum()

        if display == 'pvalue':
            norm = 'probability' if frequency else 'count'
            vals = _np.log10(pVals_nz) if log else pVals
            cumulative = dict(enabled=False)
            barcolor = '#43C6DB' #turquoise
        elif display == 'llr':
            norm = 'probability'
            vals = _np.log10(llrVals) if log else llrVals
            cumulative = dict(enabled=True)
            barcolor = '#FFD801' #rubber ducky yellow
        else:
            raise ValueError("Invalid display value: %s" % display)

        histTrace = go.Histogram(
            x=vals, histnorm=norm,
            autobinx=False,
            xbins=dict(
                start=minval,
                end=maxval,
                size=(maxval-minval)/(nbins-1)
            ),
            cumulative=cumulative,
            marker=dict(color=barcolor),
            opacity=0.75,
            showlegend=False,
        )

        if display == 'pvalue':
            bin_edges = _np.linspace(minval,maxval,nbins)
            linear_bin_edges = 10**(bin_edges) if log else bin_edges
            M = 1.0 if frequency else lastBinCount
            
            noChangeTrace = go.Scatter(
                x=bin_edges,
                y=M*_scipy.stats.chi2.pdf(
                    _scipy.stats.chi2.isf(linear_bin_edges,dsc.dof)
                                          ,dsc.dof),
                mode="lines",
                marker=dict(
                    color = 'rgba(0,0,255,0.8)',
                    line = dict(
                        width = 2,
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
                title=xlabel,
                tickvals=list(range(minInt,maxInt+1)),
                ticktext=["10<sup>%d</sup>" % i for i in range(minInt,maxInt+1)]
            )
        else:
            xaxis_dict = dict(title=xlabel) #auto-tick labels


        datasetnames = dsc.DS_names
        if dsc.gate_exclusions:
            title += ' '+str(dsc.gate_exclusions)+' excluded'
            if dsc.gate_inclusions:
                title += ';'
        if dsc.gate_inclusions:
            title += ' '+str(dsc.gate_inclusions)+' included'
        title += '<br>Comparing datasets '+str(datasetnames)
        title += ' p=0 '+str(pVals0)+' times; '+str(len(dsc.pVals))+' total sequences'

        layout = go.Layout(
            width=700*scale,
            height=400*scale,
            title=title,
            font=dict(size=10),
            xaxis=xaxis_dict,
            yaxis=dict(
                title=ylabel,
                type='log' if log else 'linear',
                #tickformat='g',
                exponentformat='power',
            ),
            bargap=0,
            bargroupgap=0,
            legend=dict(orientation="h")
        )

        return go.Figure(data=data, layout=layout)
    

class RandomizedBenchmarkingPlot(WorkspacePlot):
    def __init__(self, ws, rbR, gstyp='clifford',xlim=None, ylim=None,
                 fitting='standard', Magesan_zeroth=False, Magesan_first=False,
                 exact_decay=False,L_matrix_decay=False, Magesan_zeroth_SEB=False,
                 Magesan_first_SEB=False, L_matrix_decay_SEB=False,gs=False,
                 gs_target=False,group=False, norm='1to1', legend=True,
                 title=True, scale=1.0):
        """
        Plot RB decay curve, as a function of some the sequence length
        computed using the `gstyp` gate-label-set.

        Parameters
        ----------
        rbR : RBResults
            The RB results object containing all the relevant RB data.

        gstyp : str, optional
            The gate-label-set specifying which translation (i.e. strings with
            which gate labels) to use when computing sequence lengths.

        xlim : tuple, optional
            The x-range as (xmin,xmax).

        ylim : tuple, optional
            The y-range as (ymin,ymax).

        save_fig_path : str, optional
            If not None, the filename where the resulting plot should be saved.
            
        fitting : str, optional
            Allowed values are 'standard', 'first order' or 'all'. Specifies 
            whether the zeroth or first order fitting model results are plotted,
            or both.
            
        Magesan_zeroth : bool, optional
            If True, plots the decay predicted by the 'zeroth order' theory of Magesan
            et al. PRA 85 042311 2012. Requires gs and gs_target to be specified.
            
        Magesan_first : bool, optional
            If True, plots the decay predicted by the 'first order' theory of Magesan
            et al. PRA 85 042311 2012. Requires gs and gs_target to be specified.
            
        Magesan_zeroth_SEB : bool, optional
            If True, plots the systematic error bound for the 'zeroth order' theory 
            predicted decay. This is the region around the zeroth order decay in which
            the exact RB average survival probabilities are guaranteed to fall.
        
        Magesan_first_SEB : bool, optional
            As above, but for 'first order' theory.
            
        exact_decay : bool, optional
            If True, plots the exact RB decay, as predicted by the 'R matrix' theory
            of arXiv:1702.01853. Requires gs and group to be specified
            
        L_matrix_decay : bool, optional
            If True, plots the RB decay, as predicted by the approximate 'L matrix'
            theory of arXiv:1702.01853. Requires gs and gs_target to be specified.
            
        L_matrix_decay_SEB : bool, optional
            If True, plots the systematic error bound for approximate 'L matrix'
            theory of arXiv:1702.01853. This is the region around predicted decay
            in which the exact RB average survival probabilities are guaranteed 
            to fall.
            
        gs : gateset, optional
            Required, if plotting any of the theory decays. The gateset for which 
            these decays should be plotted for.
            
        gs_target : Gateset, optional
            Required, if plotting certain theory decays. The target gateset for which 
            these decays should be plotted for. 
            
        group : MatrixGroup, optional
            Required, if plotting R matrix theory decay. The matrix group that gs
            is an implementation of.
            
        norm : str, optional
            The norm used for calculating the Magesan theory bounds.
            
        legend : bool, optional
            Specifies whether a legend is added to the graph
            
        title : bool, optional
            Specifies whether a title is added to the graph
            
        Returns
        -------
        None
        """
#         loc : str, optional
#            Specifies the location of the legend.
        super(RandomizedBenchmarkingPlot,self).__init__(
            ws, self._create, rbR, gstyp, xlim, ylim, fitting, Magesan_zeroth,
            Magesan_first, exact_decay, L_matrix_decay, Magesan_zeroth_SEB,
            Magesan_first_SEB, L_matrix_decay_SEB, gs, gs_target, group, norm,
            legend, title, scale)
        
    def _create(self, rbR, gstyp, xlim, ylim, fitting, Magesan_zeroth,
                Magesan_first, exact_decay, L_matrix_decay, Magesan_zeroth_SEB,
                Magesan_first_SEB, L_matrix_decay_SEB, gs, gs_target, group,
                norm, legend, title, scale):

        from ..extras.rb import rbutils as _rbutils
        #TODO: maybe move the computational/fitting part of this function
        #  back to the RBResults object to reduce the logic (and dependence
        #  on rbutils) here.
        
        if gstyp not in rbR.dicts:
            raise ValueError("%s data not found!" % gstyp)

        #newplot = _plt.figure(figsize=(8, 4))
        #newplotgca = newplot.gca()

        # Note: minus one to get xdata that discounts final Clifford-inverse
        xdata = _np.asarray(rbR.dicts[gstyp]['lengths']) - 1
        ydata = _np.asarray(rbR.dicts[gstyp]['successes'])
        A = rbR.dicts[gstyp]['A']
        B = rbR.dicts[gstyp]['B']
        f = rbR.dicts[gstyp]['f']
        A1 = rbR.dicts[gstyp]['A1']
        B1 = rbR.dicts[gstyp]['B1']
        C1 = rbR.dicts[gstyp]['C1']
        f1 = rbR.dicts[gstyp]['f1']
        pre_avg = rbR.pre_avg
        
        if (Magesan_zeroth_SEB is True) and (Magesan_zeroth is False):
            print("As Magesan_zeroth_SEB is True, Setting Magesan_zeroth to True\n")
            Magesan_zeroth = True
        if (Magesan_first_SEB is True) and (Magesan_first is False):
            print("As Magesan_first_SEB is True, Setting Magesan_first to True\n")
            Magesan_first = True
            
        if gstyp != 'clifford':
            if (Magesan_zeroth is True) or (Magesan_zeroth is True):
                print("Magesan Analytical deacays curves for Cliffords only." +
                "Setting all analytic parameters to False.")
                Magesan_zeroth=False
                Magesan_first=False
            if (exact_decay is True) or (L_matrix_decay is True):  
                print("Exact and L matrix decay curves for Cliffords only." +
                "Setting all analytic parameters to False.")
                exact_decay=False
                L_matrix_decay=False
                
        if (Magesan_zeroth is True) or (Magesan_first is True):
            if (gs is False) or (gs_target is False):
                raise ValueError("To plot Magesan et al theory decay curves a gateset" +
                           " and a target gateset is required.")
            else:
                MTP = _rbutils.Magesan_theory_parameters(gs, gs_target, 
                                                success_spamlabel=rbR.success_spamlabel,
                                                         norm=norm,d=rbR.d)
                f_an = MTP['p']
                A_an = MTP['A']
                B_an = MTP['B']
                A1_an = MTP['A1']
                B1_an = MTP['B1']
                C1_an = MTP['C1']
                delta = MTP['delta']
                
        if exact_decay is True:
            if (gs is False) or (group is False):
                raise ValueError("To plot the exact decay curve a gateset" +
                           "and the target group are required.")
            else:
                mvalues,ASPs = _rbutils.exact_RB_ASPs(gs,group,max(xdata),m_min=1,m_step=1,
                                                      d=rbR.d,
                                                      success_spamlabel=rbR.success_spamlabel)
                
        if L_matrix_decay is True:
            if (gs is False) or (gs_target is False):
                raise ValueError("To plot the L matrix theory decay curve a gateset" +
                           " and a target gateset is required.")
            else:
                mvalues, LM_ASPs, LM_ASPs_SEB_lower, LM_ASPs_SEB_upper = \
                _rbutils.L_matrix_ASPs(gs,gs_target,max(xdata),m_min=1,m_step=1,d=rbR.d,
                                             success_spamlabel=rbR.success_spamlabel)
                
        if gstyp!='clifford':
            xlabel = '{0} sequence length'.format(gstyp.capitalize())
        
        if gstyp=='clifford':
            xlabel = 'Sequence length'

        #OLD cmap = _plt.cm.get_cmap('Set1')
        
        data = [] # list of traces
        data.append( go.Scatter(
            x = xdata, y = ydata,
            mode = 'markers',
            marker = dict(
                color = "rgb(0,0,0)",
                size = 6 if pre_avg else 3
            ),
            name = 'Averaged RB data' if pre_avg else 'RB data',
        ))
        
        if fitting=='standard' or fitting=='first order':
            fit_label_1='Fit'
            fit_label_2='Fit'
            color2 = "black"
        if fitting=='all':
            fit_label_1='Fit (Std)'
            fit_label_2='Fit (1st order)'
            color2 = "red"

        theory_color2 = "green"
        theory_fill2 = "rgba(0,128,0,0.1)"
        if Magesan_zeroth is True and Magesan_first is True:
            theory_color2 = "magenta"
            theory_fill2 = "rgba(255,0,255,0.1)"
                    
        if fitting=='standard' or fitting=='all':
            data.append( go.Scatter(
                x = _np.arange(max(xdata)),
                y = _rbutils.standard_fit_function(_np.arange(max(xdata)),A,B,f),
                mode = 'lines',
                line = dict(width=1, color="black"),
                name = fit_label_1,
                showlegend=legend,
            ))
          
        if fitting=='first order' or fitting=='all':
            data.append( go.Scatter(
                x = _np.arange(max(xdata)),
                y = _rbutils.first_order_fit_function(_np.arange(max(xdata)),A1,B1,C1,f1),
                mode = 'lines',
                line = dict(width=1, color=color2),
                name = fit_label_2,
                showlegend=legend,
            ))

        if Magesan_zeroth is True:
            data.append( go.Scatter(
                x = _np.arange(max(xdata)),
                y = _rbutils.standard_fit_function(_np.arange(max(xdata)),A_an,B_an,f_an),
                mode = 'lines',
                line = dict(width=2, color="green", dash='dash'),
                name = '0th order theory',
                showlegend=legend,
            ))

            if Magesan_zeroth_SEB is True:
                data.append( go.Scatter(
                    x = _np.arange(max(xdata)),
                    y = _rbutils.seb_upper(
                        _rbutils.standard_fit_function(_np.arange(max(xdata)),A_an,B_an,f_an), 
                        _np.arange(max(xdata)), delta, order='zeroth'),
                    mode = 'lines',
                    line = dict(width=0.5, color="green"),
                    name = '0th order bound',
                    fill='tonexty',
                    fillcolor='rgba(0,128,0,0.1)',
                    showlegend=False,
                ))
                data.append( go.Scatter(
                    x = _np.arange(max(xdata)),
                    y = _rbutils.seb_lower( 
                        _rbutils.standard_fit_function(_np.arange(max(xdata)),A_an,B_an,f_an),
                        _np.arange(max(xdata)), delta, order='zeroth'),
                    mode = 'lines',
                    line = dict(width=0.5, color="green"),
                    name = '0th order bound',
                    showlegend=False,
                ))


        if Magesan_first is True:
            data.append( go.Scatter(
                x = _np.arange(max(xdata)),
                y = _rbutils.first_order_fit_function(_np.arange(max(xdata)),A1_an,B1_an,C1_an,f_an),
                mode = 'lines',
                line = dict(width=2, color=theory_color2, dash='dash'),
                name = '1st order theory',
                showlegend=legend,
            ))

            if Magesan_first_SEB is True:
                data.append( go.Scatter(
                    x = _np.arange(max(xdata)),
                    y = _rbutils.seb_upper(
                        _rbutils.first_order_fit_function(_np.arange(max(xdata)),A1_an,B1_an,C1_an,f_an),
                        _np.arange(max(xdata)), delta, order='first'),
                    mode = 'lines',
                    line = dict(width=0.5, color=theory_color2), #linewidth=4?
                    name = '1st order bound',
                    fill='tonexty',
                    fillcolor=theory_fill2,
                    showlegend=False,
                ))
                data.append( go.Scatter(
                    x = _np.arange(max(xdata)),
                    y = _rbutils.seb_lower( 
                        _rbutils.first_order_fit_function(_np.arange(max(xdata)),A1_an,B1_an,C1_an,f_an),
                        _np.arange(max(xdata)), delta, order='first'),
                    mode = 'lines',
                    line = dict(width=0.5, color=theory_color2),
                    name = '1st order bound',
                    showlegend=False,
                ))


        if exact_decay is True:
            data.append( go.Scatter(
                x = mvalues,
                y = ASPs,
                mode = 'lines',
                line = dict(width=2, color="blue",dash='dash'),
                name = 'Exact decay',
                showlegend=legend,
            ))
        
        if L_matrix_decay is True:
            data.append( go.Scatter(
                x = mvalues,
                y = LM_ASPs,
                mode = 'lines',
                line = dict(width=2, color="cyan",dash='dash'),
                name = 'L matrix decay',
                showlegend=legend,
            ))
            if L_matrix_decay_SEB is True:
                data.append( go.Scatter(
                    x = mvalues,
                    y = LM_ASPs_SEB_upper,
                    mode = 'lines',
                    line = dict(width=0.5, color="cyan"),
                    name = 'LM bound',
                    fill='tonexty',
                    fillcolor='rgba(0,255,255,0.1)',
                    showlegend=False,
                ))
                data.append( go.Scatter(
                    x = mvalues,
                    y = LM_ASPs_SEB_lower,
                    mode = 'lines',
                    line = dict(width=0.5, color="cyan"),
                    name = 'LM bound',
                    showlegend=False,
                ))

        ymin = min([min(trace['y']) for trace in data])
        ymin -= 0.1*abs(1.0-ymin) #pad by 10%
        
        layout = go.Layout(
            width=800*scale,
            height=400*scale,
            title='Randomized Benchmarking Decay',
            titlefont=dict(size=16),
            xaxis=dict(
                title=xlabel,
                titlefont=dict(size=14),
                range=xlim if xlim else [0,max(xdata)],
            ),
            yaxis=dict(
                title='Mean survival probability',
                titlefont=dict(size=14),
                range=ylim if ylim else [ymin,1.0],
            ),
            legend=dict(
                font=dict(
                    size=13,
                ),
            )
        )

        #reverse order of data so z-ordering is nicer
        return go.Figure(data=list(reversed(data)), layout=layout)

        #newplotgca.set_xlabel(xlabel, fontsize=15)
        #newplotgca.set_ylabel('Mean survival probability',fontsize=15)
        #if title==True:
        #    newplotgca.set_title('Randomized Benchmarking Decay', fontsize=18)  
        #newplotgca.set_frame_on(True)
        #newplotgca.yaxis.grid(False)
        #newplotgca.tick_params(axis='x', top='off', labelsize=12)
        #newplotgca.tick_params(axis='y', left='off', right='off', labelsize=12)
        
        #if legend==True:
        #    leg = _plt.legend(fancybox=True, loc=loc)
        #    leg.get_frame().set_alpha(0.9)
        
        #newplotgca.spines["top"].set_visible(False)
        #newplotgca.spines["right"].set_visible(False)    
        #newplotgca.spines["bottom"].set_alpha(.7)
        #newplotgca.spines["left"].set_alpha(.7)  

    

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
