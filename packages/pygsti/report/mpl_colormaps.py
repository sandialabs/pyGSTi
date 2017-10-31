from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Plotly-to-Matplotlib conversion functions. """

import numpy as _np
from .plothelpers import _eformat
from ..tools import compattools as _compat

try:
    import matplotlib as _matplotlib
    import matplotlib.pyplot as _plt
except ImportError:
    raise ValueError(("While not a core requirement of pyGSTi, Matplotlib is "
                      "required to generate PDF plots.  It looks like you "
                      "don't have it installed on your system (it failed to "
                      "import)."))


class mpl_LinLogNorm(_matplotlib.colors.Normalize):
    def __init__(self, linLogColormap, clip=False):
        cm = linLogColormap
        super(mpl_LinLogNorm, self).__init__(vmin=cm.vmin, vmax=cm.vmax, clip=clip)
        self.trans = cm.trans
        self.cm = cm

    def inverse(self, value):
        norm_trans = super(mpl_LinLogNorm, self).__call__(self.trans)
        deltav = self.vmax - self.vmin
        return_value = _np.where(_np.greater(0.5, value),
                                 2*value*(self.trans - self.vmin) + self.vmin,
                                 deltav*_np.power(norm_trans, 2*(1 - value)))
        if return_value.shape==():
            return return_value.item()
        else:
            return return_value.view(_np.ma.MaskedArray)

    def __call__(self, value, clip=None):
        return self.cm.normalize(value)

def mpl_make_linear_norm(vmin, vmax, clip=False):
    return _matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=clip)

def mpl_make_linear_cmap(rgb_colors, name=None):
    """
    Make a color map that simply linearly interpolates between a set of
    colors in RGB space.                                                                                                             

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
        name = "pygsti-cmap-" + str(_np.random.randint(0,100000000))
        
    cdict = { 'red':[], 'green':[], 'blue':[], 'alpha':[] }
    for val,rgb_tup in rgb_colors:
        for k,v in zip(('red','green','blue'),rgb_tup):
            cdict[k].append( (val, v, v) )
        cdict['alpha'].append( (val, 1.0, 1.0) ) #alpha channel always 1.0
        
    return _matplotlib.colors.LinearSegmentedColormap(name, cdict)
        
    cdict = {label: []
             for label, idx in zip(labels, list(range(len(start_color))))}

    if name is None:
        name = 'linear_' + str(start_color) + '-' + str(final_color)

    return _matplotlib.colors.LinearSegmentedColormap(name, cdict)


def mpl_besttxtcolor( x, cmap, norm ):
    """
    Determinining function for whether text should be white or black
 
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
    P = _np.sqrt(0.299*R**2 + 0.587*G**2 + 0.114*B**2)
    return "black" if 0.5 <= P else "white"

def mpl_process_lbl(lbl, math=False):
    math = math or ('<sup>' in lbl) or ('<sub>' in lbl) or \
           ('_' in lbl) or ('|' in lbl) or (len(lbl) == 1)
    try:
        float(lbl)
        math=True
    except: pass

    
    l = lbl
    l = l.replace("<i>","").replace("</i>","")
    l = l.replace("<sup>","^{").replace("</sup>","}")
    l = l.replace("<sub>","_{").replace("</sub>","}")
    l = l.replace("<br>","\n")

    if math:
        l = l.replace("alpha","\\alpha")
        l = l.replace("beta","\\beta")
        l = l.replace("sigma","\\sigma")
    
    if math or (len(l) == 1): l = "$" + l + "$"
    return l

def mpl_process_lbls(lblList):
    return [ mpl_process_lbl(lbl) for lbl in lblList ]
    
def mpl_color(plotly_color):
    #_compat.isstr
    plotly_color = plotly_color.strip() #remove any whitespace
    if plotly_color.startswith('#'):
        return plotly_color # matplotlib understands "#FF0013"
    elif plotly_color.startswith('rgb(') and plotly_color.endswith(')'):
        tupstr = plotly_color[len('rgb('):-1]
        tup = [float(x)/256.0 for x in tupstr.split(',')]
        return tuple(tup)
    elif plotly_color.startswith('rgba(') and plotly_color.endswith(')'):
        tupstr = plotly_color[len('rgba('):-1]
        rgba = tupstr.split(',')
        tup = [float(x)/256.0 for x in rgba[0:3]] + [float(rgba[3])]
        return tuple(tup)
    else:
        return plotly_color #hope this is a color name matplotlib understands

def plotly_to_matplotlib(pygsti_fig, save_to=None, fontsize=12, prec='compacthp'):
    fig = pygsti_fig['plotlyfig']
    data_trace_list = fig['data']
    
    #if axes is None: 
    mpl_fig,axes = _plt.subplots()  # create a new figure if no axes are given    
    
    layout = fig['layout']
    h,w = layout['height'], layout['width']
    # todo: get margins and subtract from h,w
    
    if mpl_fig is not None and w is not None and h is not None:
        mpl_size = w/100.0, h/100.0 #heusistic
        mpl_fig.set_size_inches(*mpl_size) # was 12,8 for "super" color plot
        pygsti_fig['mpl_fig_size'] = mpl_size #record for later use by rendering commands
    
    xaxis, yaxis = layout['xaxis'], layout['yaxis']
    annotations = layout.get('annotations',[])
    title = layout.get('title',None)
    shapes = layout.get('shapes',[]) # assume only shapes are grid lines
    bargap = layout.get('bargap',0)
    
    xlabel = xaxis.get('title',None)
    ylabel = yaxis.get('title',None)
    xlabels = xaxis.get('ticktext',None)
    ylabels = yaxis.get('ticktext',None)
    xtickvals = xaxis.get('tickvals',None)
    ytickvals = yaxis.get('tickvals',None)
    xaxistype = xaxis.get('type',None)
    yaxistype = yaxis.get('type',None)
    xaxisside = xaxis.get('side','bottom')
    yaxisside = yaxis.get('side','left')
    xtickangle = xaxis.get('tickangle',0)

    if xaxisside == "top":
        axes.xaxis.set_label_position('top') 
        axes.xaxis.tick_top()
        #axes.yaxis.set_ticks_position('both')

    if yaxisside == "right":
        axes.yaxis.set_label_position('right') 
        axes.yaxis.tick_right()
        #axes.yaxis.set_ticks_position('both')
    
    if title is not None:
        if xaxisside == "top":
            axes.set_title( mpl_process_lbl(title), fontsize=fontsize, y=2.5 ) #push title up higher
        axes.set_title( mpl_process_lbl(title), fontsize=fontsize )

    if xlabel is not None:
        axes.set_xlabel( mpl_process_lbl(xlabel), fontsize=fontsize )

    if ylabel is not None:
        axes.set_ylabel( mpl_process_lbl(ylabel), fontsize=fontsize )
        
    if xtickvals is not None:
        axes.set_xticks(xtickvals, minor=False)
        
    if ytickvals is not None:
        axes.set_yticks(ytickvals, minor=False)
            
    if xlabels is not None:    
        axes.set_xticklabels( mpl_process_lbls(xlabels) ,rotation=0, fontsize=(fontsize-2) )
    
    if ylabels is not None:
        axes.set_yticklabels( mpl_process_lbls(ylabels), fontsize=(fontsize-2) )

    if xtickangle != 0:
        _plt.xticks(rotation=-xtickangle) #minus b/c ploty & matplotlib have different sign conventions
        
    if xaxistype == 'log':
        axes.set_xscale("log")
    if yaxistype == 'log':
        axes.set_yscale("log")

    #figure out barwidth and offsets for bar plots
    num_bars = sum([d.get('type','')=='bar' for d in data_trace_list])
    currentBarOffset = 0
    barWidth = (1.0-bargap)/num_bars if num_bars > 0 else 1.0

    #process traces
    handles = []; labels = [] #for the legend
    for i,traceDict in enumerate(data_trace_list):
        typ = traceDict.get('type','unknown')
        
        name = traceDict.get('name',None)
        showlegend = traceDict.get('showlegend',True)
        
        if typ == "heatmap":
            colorscale = traceDict.get('colorscale','unknown')
            plt_data = pygsti_fig['plt_data'] #traceDict['z'] is *normalized* already - maybe would work here but not for box value labels
            zmin = traceDict.get('zmin','default')
            zmax = traceDict.get('zmin','default')
            show_colorscale = traceDict.get('showscale',True)
            
            mpl_fig.set_size_inches( plt_data.shape[1]*0.4, plt_data.shape[0]*0.4)
            
            colormap = pygsti_fig['colormap']
            assert(colormap is not None), 'Must separately specify a colormap...'
            norm, cmap = colormap.get_matplotlib_norm_and_cmap()

            masked_data = _np.ma.array(plt_data, mask=_np.isnan(plt_data))
            heatmap = axes.pcolormesh( masked_data, cmap=cmap, norm=norm)

            axes.set_xlim(0,plt_data.shape[1])
            axes.set_ylim(0,plt_data.shape[0])

            xtics = _np.array(xtickvals)+0.5 #_np.arange(plt_data.shape[1])+0.5
            axes.set_xticks(xtics, minor=False)
                
            ytics = _np.array(ytickvals)+0.5 # _np.arange(plt_data.shape[0])+0.5
            axes.set_yticks(ytics, minor=False)
 
            grid = bool(len(shapes) > 1)
            if grid:
                def get_minor_tics(t):
                    return [ (t[i]+t[i+1])/2.0 for i in range(len(t)-1) ]
                axes.set_xticks(get_minor_tics(xtics), minor=True)
                axes.set_yticks(get_minor_tics(ytics), minor=True)
                axes.grid(which='minor', axis='both', linestyle='-', linewidth=2)

            if xlabels is None and ylabels is None:
                axes.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off') #white tics                                           
            else:
                axes.tick_params(top='off', bottom='off', left='off', right='off')

            #print("DB ann = ", len(annotations))
            #boxLabels = bool( len(annotations) >= 1 ) #TODO: why not plt_data.size instead of 1?
            boxLabels = True # maybe should always be true?
            if boxLabels:
                # Write values on colored squares                                                                                                        
                for y in range(plt_data.shape[0]):
                    for x in range(plt_data.shape[1]):
                        if _np.isnan(plt_data[y, x]): continue
                        axes.text(x + 0.5, y + 0.5, mpl_process_lbl(_eformat(plt_data[y, x], prec),math=True),
                                horizontalalignment='center',
                                verticalalignment='center',
                                color=mpl_besttxtcolor( plt_data[y,x], cmap, norm),
                                fontsize=(fontsize-6))

            if show_colorscale:
                _plt.colorbar(heatmap)

        elif typ == "scatter":
            mode = traceDict.get('mode','lines')
            marker = traceDict.get('marker',None)
            line = marker['line'] if marker else None
            color = mpl_color(marker['color'] if marker else "rgba(0,0,0,1.0)")
            linewidth = float(line['width']) if line else 1.0
    
            x = traceDict['x'] 
            y = traceDict['y']
            lines = _plt.plot(x,y)
            if mode == 'lines': ls = '-'
            elif mode == 'markers': ls = '.'
            elif mode == 'lines+markers': ls = '-.'
            else: raise ValueError("Unknown mode: %s" % mode)
            _plt.setp(lines, linestyle=ls, color=color, linewidth=linewidth)
            
            if showlegend and name:
                handles.append(lines[0])
                labels.append(name)
                
        elif typ == "scattergl": #currently used only for colored points...
            x = traceDict['x'] 
            y = traceDict['y'] 
            colormap = pygsti_fig.get('colormap',None)
            if colormap:
                norm, cmap = colormap.get_matplotlib_norm_and_cmap()
                s = _plt.scatter(x, y, c=y, s=50, cmap=cmap, norm=norm)
            else:
                s = _plt.scatter(x, y, c=y, s=50, cmap='gray')
            
            if showlegend and name:
                handles.append(s)
                labels.append(name)   
            
        elif typ == "bar": 
            xlabels = [str(xl) for xl in traceDict['x']] # x "values" are actually bar labels in plotly

            #always grey=pos, red=neg type of bar plot for now (since that's all pygsti uses)
            y = _np.asarray(traceDict['y'])
            if 'plt_yerr' in pygsti_fig:
                yerr = pygsti_fig['plt_yerr']
            else:
                yerr = None

            # actual x values are just the integers + offset
            x = _np.arange(y.size) + currentBarOffset
            currentBarOffset += barWidth #so next bar trace will be offset correctly

            marker = traceDict.get('marker',None)
            if marker and ('color' in marker):
                if _compat.isstr(marker['color']):
                    color = mpl_color(marker['color'])
                elif isinstance(marker['color'],list):
                    color = [mpl_color(c) for c in marker['color']] # b/c axes.bar can take a list of colors
                else: color = "gray"

            if yerr is None:
                rects = axes.bar(x, y, barWidth, color=color)
            else:
                rects = axes.bar(x, y, barWidth, color=color,
                                 yerr=yerr.flatten().real)
                                
            if xtickvals is not None:
                xtics = _np.array(xtickvals)+0.5 #_np.arange(plt_data.shape[1])+0.5
            else: xtics = x
            axes.set_xticks(xtics, minor=False)
            axes.set_xticklabels( mpl_process_lbls(xlabels) ,rotation=0, fontsize=(fontsize-4) )    
            
        elif typ == "histogram":
            histnorm = traceDict.get('histnorm',None)
            marker = traceDict.get('marker',None)
            color = mpl_color(marker['color'] if marker and _compat.isstr(marker['color']) else "gray")
            xbins = traceDict['xbins']
            histdata = traceDict['x'] 

            if abs(xbins['size']) < 1e-6:
                histBins = 1
            else:
                histBins = int(round( (xbins['end'] - xbins['start'])/xbins['size']))
             
            histdata_finite = _np.take(histdata, _np.where(_np.isfinite(histdata)))[0] #take gives back (1,N) shaped array (why?)
            if yaxistype == 'log':
                TOL = 1e-6
                if len(histdata_finite) == 0:
                    axes.set_yscale("linear") #no data, and will get an error with log-scale, so switch to linear
            
            #histMin = min( histdata_finite ) if cmapFactory.vmin is None else cmapFactory.vmin
            #histMax = max( histdata_finite ) if cmapFactory.vmax is None else cmapFactory.vmax
            #_plt.hist(_np.clip(histdata_finite,histMin,histMax), histBins,
            #          range=[histMin, histMax], facecolor='gray', align='mid')
            n, bins, patches = _plt.hist(histdata_finite, histBins,
                                         facecolor=color, align='mid')

            #If we've been given an array of colors
            if marker and ('color' in marker) and isinstance(marker['color'],list):
                for p,c in zip(patches,marker['color']):
                    _plt.setp(p, 'facecolor', mpl_color(c))
        
    if len(handles) > 0:
        _plt.legend(handles, labels, bbox_to_anchor=(1.01, 1.0),
                    borderaxespad=0., loc="upper left")
        
    if save_to:
        _plt.savefig(save_to, bbox_extra_artists=(axes,),
                     bbox_inches='tight') #need extra artists otherwise
                                          #axis labels get clipped 
        _plt.cla()
        _plt.close(mpl_fig)
        del mpl_fig
        return None #figure is closed!
    else:
        return mpl_fig
