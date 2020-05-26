"""
Extends Plolty python library for additional needed functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import os as _os
#from plotly.offline.offline import get_plotlyjs
#from plotly.offline.offline import __PLOTLY_OFFLINE_INITIALIZED
#from pkg_resources import resource_string
DEFAULT_PLOTLY_TEMPLATE = 'none'

# Try to set the default plotly template.  This isn't essential, but makes the
# figures look nicer.  It must be done at import time, before any plotly.Figure
# objects are created, so this import must be place *here*, not within a function.
try:
    from plotly import io as _pio
    _pio.templates.default = DEFAULT_PLOTLY_TEMPLATE
except ImportError: pass


def plot_ex(figure_or_data, show_link=True, link_text='Export to plot.ly',
            validate=True, resizable=False, lock_aspect_ratio=False,
            master=True, click_to_display=False, link_to=None, link_to_id=False,
            rel_figure_dir="figures"):
    """
    Create a pyGSTi plotly graph locally, returning HTML & JS separately.

    Parameters
    ----------
    figure_or_data : plotly.graph_objs.Figure or .Data or dict or list
        object that describes a Plotly graph. See https://plot.ly/python/
        for examples of graph descriptions.

    show_link : bool, optional
        display a link in the bottom-right corner of

    link_text : str, optional
        the text of export link

    validate : bool, optional
        validate that all of the keys in the figure are valid? omit if you
        need to include extra, unnecessary keys in your figure.

    resizable : bool, optional
        Make the plot resizable by including a "resize" event handler and
        any additional initialization.

    lock_aspect_ratio : bool, optional
        Whether the aspect ratio of the plot should be allowed to change
        when it is sized based on it's container.

    master : bool, optional
        Whether this plot represents the "master" of a group of plots,
        all of the others which are "slaves".  The relative sizing of the
        master of a group will determine the relative sizing of the slaves,
        rather than the slave's containing element.  Useful for preserving the
        size of the features in a group of plots that may be different overall
        sizes.

    click_to_display : bool, optional
        Whether the plot should be rendered immediately or whether a "click"
        icon should be shown instead, which must be clicked on to render the
        plot.

    link_to : None or tuple of {"pdf", "pkl"}
        If not-None, the types of pre-rendered/computed versions of this plot
        that can be assumed to be present, and therefore linked to by additional
        items in the hover-over menu of the plotly plot.

    link_to_id : str, optional
        The base name (without extension) of the ".pdf" or ".pkl" files that are
        to be linked to by menu items.  For example, if `link_to` equals
        `("pdf",)` and `link_to_id` equals "plot1234", then a menu item linking
        to the file "plot1234.pdf" will be added to the renderd plot.

    rel_figure_dir : str, optional
        A relative path from the "current" path (the path of the generated
        html documents) to figure files.  Usually something like `"figures"`.

    Returns
    -------
    dict
        With 'html' and 'js' keys separately specifying the HTML and javascript
        needed to embed the plot.
    """
    from plotly import __version__ as _plotly_version
    from plotly import tools as _plotlytools

    #Processing to enable automatic-resizing & aspect ratio locking
    fig = _plotlytools.return_figure_from_figure_or_data(
        figure_or_data, False)
    orig_width = fig.get('layout', {}).get('width', None)
    orig_height = fig.get('layout', {}).get('height', None)

    if lock_aspect_ratio and orig_width and orig_height:
        aspect_ratio = orig_width / orig_height
    else: aspect_ratio = None

    #Remove original dimensions of plot so default of 100% is used below
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

    #Add version-dependent kwargs to _plot_html call below
    plotly_version = tuple(map(int, _plotly_version.split('.')))
    if plotly_version < (3, 8, 0):  # "old" plotly with _plot_html function
        from plotly.offline.offline import _plot_html

        kwargs = {}
        if plotly_version >= (3, 7, 0):  # then auto_play arg exists
            kwargs['auto_play'] = False

        #Note: removing width and height from layout above causes default values to
        # be used (the '100%'s hardcoded below) which subsequently trigger adding a resize script.
        plot_html, plotdivid, _, _ = _plot_html(
            fig, config, validate, '100%', '100%',
            global_requirejs=False,  # no need for global_requirejs here
            **kwargs)  # since we now extract js and remake full script.
    else:
        from plotly.io import to_html as _to_html
        import uuid as _uuid
        plot_html = _to_html(fig, config, auto_play=False, include_plotlyjs=False,
                             include_mathjax=False, post_script=None, full_html=False,
                             animation_opts=None, validate=validate)
        assert(plot_html.startswith("<div>") and plot_html.endswith("</div>"))
        plot_html = plot_html[len("<div>"):-len("</div>")].strip()
        assert(plot_html.endswith("</script>"))
        id_index = plot_html.find('id="')
        id_index_end = plot_html.find('"', id_index + len('id="'))
        plotdivid = _uuid.UUID(plot_html[id_index + len('id="'):id_index_end])

    if orig_width: fig['layout']['width'] = orig_width
    if orig_height: fig['layout']['height'] = orig_height

    # Separate the HTML and JS in plot_html so we can insert
    # initial-sizing JS between them.  NOTE: this is FRAGILE and depends
    # on Plotly output (_plot_html) being HTML followed by JS
    tag = '<script type="text/javascript">'; end_tag = '</script>'
    iTag = plot_html.index(tag)
    plot_js = plot_html[iTag + len(tag):-len(end_tag)].strip()
    plot_html = plot_html[0:iTag].strip()

    full_script = ''

    #Note: in this case, upper logic (usually in an on-ready hander of the table/plot
    # group creation) is responsible for triggering a "create" event on the plot div
    # when appropriate (see workspace.py).

    #Get javascript for create and (possibly) resize handlers
    plotly_create_js = plot_js  # the ususal plotly creation javascript
    plotly_resize_js = None

    if resizable:
        #the ususal plotly resize javascript
        plotly_resize_js = '  Plotly.Plots.resize(document.getElementById("{id}"));'.format(id=plotdivid)

        if 'angularaxis' in fig['layout']:
            #Special case of polar plots: Plotly does *not* allow resizing of polar plots.
            # (I don't know why, and it's not documented, but in plotly.js there are explict conditions
            #  in Plotly.relayout that short-circuit when "gd.frameworks.isPolar" is true).  So,
            #  we just re-create the plot with a different size to mimic resizing.
            plot_js = plot_js.replace('"width": 123', '"width": pw').replace('"height": 123', '"height": ph')
            plotly_resize_js = (
                'var plotlydiv = $("#{id}");\n'
                'plotlydiv.children(".plotly").remove();\n'
                'var pw = plotlydiv.width();\n'
                'var ph = plotlydiv.height();\n'
                '{resized}\n').format(id=plotdivid, resized=plot_js)
            plotly_create_js = plotly_resize_js

    aspect_val = aspect_ratio if aspect_ratio else "null"

    groupclass = "pygsti-plotgroup-master" \
                 if master else "pygsti-plotgroup-slave"

    if link_to and ('pdf' in link_to) and link_to_id:
        link_to_pdf_js = (
            "\n"
            "  btn = $('#{id}').find('.modebar-btn[data-title=\"Show closest data on hover\"]');\n"
            "  btn = cloneAndReplace( btn ); //Strips all event handlers\n"
            "  btn.attr('data-title','Download PDF');\n"
            "  btn.click( function() {{\n"
            "     window.open('{relfigdir}/{pdfid}.pdf');\n"
            "  }});\n").format(id=plotdivid, pdfid=link_to_id, relfigdir=rel_figure_dir)
        plotly_create_js += link_to_pdf_js
    if link_to and ('pkl' in link_to) and link_to_id:
        link_to_pkl_js = (
            "\n"
            "  btn = $('#{id}').find('.modebar-btn[data-title=\"Zoom\"]');\n"
            "  btn = cloneAndReplace( btn ); //Strips all event handlers\n"
            "  btn.attr('data-title','Download python pickle');\n"
            "  btn.click( function() {{\n"
            "     window.open('{relfigdir}/{pklid}.pkl');\n"
            "  }});\n").format(id=plotdivid, pklid=link_to_id, relfigdir=rel_figure_dir)
        plotly_create_js += link_to_pkl_js

    plotly_click_js = ""
    if click_to_display and master:
        # move plotly plot creation from "create" to "click" handler
        plotly_click_js = plotly_create_js
        plotly_create_js = ""

    full_script = (  # (assume this will all be run within an on-ready handler)
        '  $("#{id}").addClass("{groupclass}");\n'  # perform this right away
        '  $("#{id}").on("init", function(event) {{\n'  # always add init-size handler
        '    pex_init_plotdiv($("#{id}"), {ow}, {oh});\n'
        '    pex_init_slaves($("#{id}"));\n'
        '    console.log("Initialized {id}");\n'
        '  }});\n'
        '  $("#{id}").on("click.pygsti", function(event) {{\n'
        '     plotman.enqueue(function() {{ \n'
        '       {plotlyClickJS} \n'
        '     }}, "Click-creating Plot {id}" );\n'
        '     $("#{id}").off("click.pygsti");\n'  # remove this event handler
        '     console.log("Click-Created {id}");\n'
        '  }});\n'
        '  $("#{id}").on("create", function(event, fracw, frach) {{\n'  # always add create handler
        '     pex_update_plotdiv_size($("#{id}"), {ratio}, fracw, frach, {ow}, {oh});\n'
        '     plotman.enqueue(function() {{ \n'
        '       $("#{id}").addClass("pygBackground");\n'
        '       {plotlyCreateJS} \n'
        '       pex_create_slaves($("#{id}"), {ow}, {oh});\n'
        '     }}, "Creating Plot {id}" );\n'
        '     console.log("Created {id}");\n'
        '  }});\n'
    ).format(id=plotdivid, ratio=aspect_val,
             groupclass=groupclass,
             ow=orig_width if orig_width else "null",
             oh=orig_height if orig_height else "null",
             plotlyClickJS=plotly_click_js,
             plotlyCreateJS=plotly_create_js)

    #Add resize handler if needed
    if resizable:
        full_script += (
            '  $("#{id}").on("resize", function(event,fracw,frach) {{\n'
            '    pex_update_plotdiv_size($("#{id}"), {ratio}, fracw, frach, {ow}, {oh});\n'
            '    plotman.enqueue(function() {{ \n'
            '      {plotlyResizeJS} \n'
            '      pex_resize_slaves($("#{id}"), {ow}, {oh});\n'
            '     }}, "Resizing Plot {id}" );\n'
            '    //console.log("Resized {id}");\n'
            '  }});\n'
        ).format(id=plotdivid, ratio=aspect_val,
                 ow=orig_width if orig_width else "null",
                 oh=orig_height if orig_height else "null",
                 plotlyResizeJS=plotly_resize_js)

    return {'html': plot_html, 'js': full_script}


def init_notebook_mode_ex(connected=False):
    """
    Similar to `init_notebook_mode` in `plotly.offline`.

    The main difference is that this function loads the pyGSTi-customized plotly library
    when `connected=False` (which contains fixes relevant to pyGSTi plots).

    Parameters
    ----------
    connected : bool, optional
        Whether an active internet connection should be assumed.

    Returns
    -------
    str
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
            '').format(script=get_plotlyjs_ex())  # EGN changed to _ex

    #ORIG: ipython_display.display(ipython_display.HTML(script_inject))
    __PLOTLY_OFFLINE_INITIALIZED = True
    return script_inject  # EGN: just return so we can combine with other HTML


def get_plotlyjs_ex():
    """
    Gets the custom pyGSTi version of plotly

    Returns
    -------
    str
    """
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                         "templates", "offline", "plotly-latest.min.js")  # "plotly-polarfixed.js"

    #EGN this block mocks-up resource_string to also work when using a
    # local package... could look into whether this is unecessary if we
    # just do a "pip -e pygsti" install instead of install_locally.py...
    with open(path) as f:
        plotlyjs = f.read()
        try:  # to convert to unicode since we use unicode literals
            plotlyjs = plotlyjs.decode('utf-8')
        except AttributeError:
            pass  # Python3 case when unicode is read in natively (no need to decode)

    #ORIG plotlyjs = resource_string('plotly', path).decode('utf-8')
    return plotlyjs
