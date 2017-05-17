from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Extends Plolty python library for additional needed functionality."""

import os as _os
from plotly import tools as _plotlytools
from plotly.offline.offline import _plot_html
from plotly.offline.offline import get_plotlyjs
from plotly.offline.offline import __PLOTLY_OFFLINE_INITIALIZED
from pkg_resources import resource_string

def plot_ex(figure_or_data, show_link=True, link_text='Export to plot.ly',
            validate=True, resizable=False, autosize=False,
            lock_aspect_ratio=False, master=True):
    """ 
    Create a pyGSTi plotly graph locally, returning HTML & JS separately.

    figure_or_data -- a plotly.graph_objs.Figure or plotly.graph_objs.Data or
                      dict or list that describes a Plotly graph.
                      See https://plot.ly/python/ for examples of
                      graph descriptions.

    Keyword arguments:
    show_link (default=True) -- display a link in the bottom-right corner of
        of the chart that will export the chart to Plotly Cloud or
        Plotly Enterprise
    link_text (default='Export to plot.ly') -- the text of export link
    validate (default=True) -- validate that all of the keys in the figure
        are valid? omit if your version of plotly.js has become outdated
        with your version of graph_reference.json or if you need to include
        extra, unnecessary keys in your figure.
    """

    #Processing to enable automatic-resizing & aspect ratio locking
    fig = _plotlytools.return_figure_from_figure_or_data(
        figure_or_data, False)
    orig_width = fig.get('layout', {}).get('width', None)
    orig_height = fig.get('layout', {}).get('height', None)

    if lock_aspect_ratio and orig_width and orig_height:
        aspect_ratio = orig_width / orig_height
    else: aspect_ratio = None

    if autosize or resizable:
        #Remove original dimensions of plot default of 100% is used below
        # (and triggers resize-script creation)
        if orig_width: del fig['layout']['width']
        if orig_height: del fig['layout']['height']
        
        #Special polar plot case - see below - add dummy width & height so
        # we can find/replace them with variables in generated javascript.
        if 'angularaxis' in fig['layout']:
            fig['layout']['width'] = 123
            fig['layout']['height'] = 123

    config = {}
    config['showLink'] = show_link
    config['linkText'] = link_text

    #Note: removing width and height from layout above causes default values to
    # be used (the '100%'s hardcoded below) which subsequently trigger adding a resize script.
    plot_html, plotdivid, width, height = _plot_html(
        figure_or_data, config, validate,
        '100%', '100%', global_requirejs=False)
       #Note: no need for global_requirejs here since we now extract js and remake full script.

    if autosize or resizable:
        if orig_width: fig['layout']['width'] = orig_width
        if orig_height: fig['layout']['height'] = orig_height

    # Separate the HTML and JS in plot_html so we can insert
    # initial-sizing JS between them.  NOTE: this is FRAGILE and depends
    # on Plotly output (_plot_html) being HTML followed by JS
    tag = '<script type="text/javascript">'; end_tag = '</script>'
    iTag = plot_html.index(tag)
    plot_js = plot_html[iTag+len(tag):-len(end_tag)]
    plot_html = plot_html[0:iTag]

    full_script = ''
    if resizable or autosize:
        #Note: in this case, upper logic (usually in an on-ready hander of the table/plot
        # group creation) is responsible for triggering a "create" event on the plot div
        # when appropriate (see workspace.py).

        #Below if/else block added by EGN
        if 'angularaxis' in fig['layout']:
            #Special case of polar plots: Plotly does *not* allow resizing of polar plots.
            # (I don't know why, and it's not documented, but in plotly.js there are explict conditions
            #  in Plotly.relayout that short-circuit when "gd.frameworks.isPolar" is true).  So,
            #  we just re-create the plot with a different size to mimic resizing.
            plot_js = plot_js.replace('"width": 123','"width": pw').replace('"height": 123','"height": ph')
            plotly_resize_js = (
                'var plotlydiv = $("#{id}");\n'
                'plotlydiv.children(".plotly").remove();\n'
                'var pw = plotlydiv.width();\n'
                'var ph = plotlydiv.height();\n'
                '{resized}\n').format(id=plotdivid, resized=plot_js)
            plotly_create_js = plotly_resize_js
        else:
            #the ususal plotly creation & resize javascript
            plotly_create_js = plot_js
            plotly_resize_js = '  Plotly.Plots.resize(document.getElementById("{id}"));'.format(id=plotdivid)

        aspect_val = aspect_ratio if aspect_ratio else "null"
            
        groupclass = "pygsti-plotgroup-master" \
                     if master else "pygsti-plotgroup-slave"

        full_script = (  #(assume this will all be run within an on-ready handler)
            '  $("#{id}").addClass("{groupclass}");\n' #perform this right away
            '  $("#{id}").on("init", function(event) {{\n' #always add init-size handler
            '    pex_init_plotdiv($("#{id}"), {ow}, {oh});\n'
            '    pex_init_slaves($("#{id}"));\n'
            '    console.log("Initialized {id}");\n'
            '  }});\n'
            '  $("#{id}").on("create", function(event, fracw, frach) {{\n' #always add create handler
            '     pex_update_plotdiv_size($("#{id}"), {ratio}, fracw, frach, {ow}, {oh});\n'
            '     {plotlyCreateJS}\n'
            '     pex_create_slaves($("#{id}"), {ow}, {oh});\n'
            '     console.log("Created {id}");\n'
            '  }});\n'            
            '  $("#{id}").on("resize", function(event,fracw,frach) {{\n'
            '    pex_update_plotdiv_size($("#{id}"), {ratio}, fracw, frach, {ow}, {oh});\n'
            '    {plotlyResizeJS}\n'
            '    pex_resize_slaves($("#{id}"), {ow}, {oh});\n'
            '    console.log("Resized {id}");\n'
            '  }});\n'
        ).format(id=plotdivid, ratio=aspect_val,
                 groupclass=groupclass,
                 ow=orig_width if orig_width else "null",
                 oh=orig_height if orig_height else "null",
                 plotlyCreateJS=plotly_create_js,
                 plotlyResizeJS=plotly_resize_js)

    else:
        #NOTE: In this case, JS is placed which creates
        # the plot immediately.  When resize or autosize
        # are True, creation is deferred to *handlers* which
        # must be called by "parent" object.  Maybe this should
        # be the case here too?
        full_script = (
            '  {plotlyCreateJS}\n'
            ).format(plotlyCreateJS=plot_js)

    return {'html': plot_html, 'js': full_script }
        

def init_notebook_mode_ex(connected=False):
    """ 
    A copy of init_notebook_mode in plotly.offline except this version
    loads the pyGSTi-customized plotly library when connected=False
    (which contains fixes relevant to pyGSTi plots).
    """
    global __PLOTLY_OFFLINE_INITIALIZED

    if connected:
        # Inject plotly.js into the output cell                                 
        script_inject = (
            ''
            '<script>'
            'requirejs.config({'
            'paths: { '
            # Note we omit the extension .js because require will include it.   
            '\'plotly\': [\'https://cdn.plot.ly/plotly-latest.min\']},'
            '});'
            'if(!window.Plotly) {{'
            'require([\'plotly\'],'
            'function(plotly) {window.Plotly=plotly;});'
            '}}'
            '</script>'
        )
    else:
        # Inject plotly.js into the output cell                                 
        script_inject = (
            ''
            '<script type=\'text/javascript\'>'
            'if(!window.Plotly){{'
            'define(\'plotly\', function(require, exports, module) {{'
            '{script}'
            '}});'
            'require([\'plotly\'], function(Plotly) {{'
            'window.Plotly = Plotly;'
            '}});'
            '}}'
            '</script>'
            '').format(script=get_plotlyjs_ex())  #EGN changed to _ex

    #ORIG: ipython_display.display(ipython_display.HTML(script_inject))
    __PLOTLY_OFFLINE_INITIALIZED = True
    return script_inject #EGN: just return so we can combine with other HTML

def get_plotlyjs_ex():
    """ Gets the custom pyGSTi version of plotly """
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                         "templates","offline", "plotly-polarfixed.min.js") # "plotly-polarfixed.js"

    #EGN this block mocks-up resource_string to also work when using a
    # local package... could look into whether this is unecessary if we
    # just do a "pip -e pygsti" install instead of install_locally.py...
    with open(path) as f:
        plotlyjs = f.read()
        try: # to convert to unicode since we use unicode literals
            plotlyjs = plotlyjs.decode('utf-8')
        except AttributeError:
            pass #Python3 case when unicode is read in natively (no need to decode)
    
    #ORIG plotlyjs = resource_string('plotly', path).decode('utf-8')
    return plotlyjs
