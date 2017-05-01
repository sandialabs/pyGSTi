from plotly import tools as _plotlytools
#from plotly.offline.offline import *
from plotly.offline.offline import _plot_html

def plot_ex(figure_or_data, show_link=True, link_text='Export to plot.ly',
            validate=True, output_type='file', include_plotlyjs=True,
            filename='temp-plot.html', auto_open=True, image=None,
            image_filename='plot_image', image_width=800, image_height=600,
            global_requirejs=False, resizable=False, autosize=False, lock_aspect_ratio=False):
    """ Create a plotly graph locally as an HTML document or string.

    Example:
    ```
    from plotly.offline import plot
    import plotly.graph_objs as go

    plot([go.Scatter(x=[1, 2, 3], y=[3, 2, 6])], filename='my-graph.html')
    # We can also download an image of the plot by setting the image parameter
    # to the image format we want
    plot([go.Scatter(x=[1, 2, 3], y=[3, 2, 6])], filename='my-graph.html'
         image='jpeg')
    ```
    More examples below.

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
    output_type ('file' | 'div' - default 'file') -- if 'file', then
        the graph is saved as a standalone HTML file and `plot`
        returns None.
        If 'div', then `plot` returns a string that just contains the
        HTML <div> that contains the graph and the script to generate the
        graph.
        Use 'file' if you want to save and view a single graph at a time
        in a standalone HTML file.
        Use 'div' if you are embedding these graphs in an HTML file with
        other graphs or HTML markup, like a HTML report or an website.
    include_plotlyjs (default=True) -- If True, include the plotly.js
        source code in the output file or string.
        Set as False if your HTML file already contains a copy of the plotly.js
        library.
    filename (default='temp-plot.html') -- The local filename to save the
        outputted chart to. If the filename already exists, it will be
        overwritten. This argument only applies if `output_type` is 'file'.
    auto_open (default=True) -- If True, open the saved file in a
        web browser after saving.
        This argument only applies if `output_type` is 'file'.
    image (default=None |'png' |'jpeg' |'svg' |'webp') -- This parameter sets
        the format of the image to be downloaded, if we choose to download an
        image. This parameter has a default value of None indicating that no
        image should be downloaded. Please note: for higher resolution images
        and more export options, consider making requests to our image servers.
        Type: `help(py.image)` for more details.
    image_filename (default='plot_image') -- Sets the name of the file your
        image will be saved to. The extension should not be included.
    image_height (default=600) -- Specifies the height of the image in `px`.
    image_width (default=800) -- Specifies the width of the image in `px`.
    """
    if output_type not in ['div', 'file']:
        raise ValueError(
            "`output_type` argument must be 'div' or 'file'. "
            "You supplied `" + output_type + "``")
    if not filename.endswith('.html') and output_type == 'file':
        warnings.warn(
            "Your filename `" + filename + "` didn't end with .html. "
            "Adding .html to the end of your file.")
        filename += '.html'

    #BEGIN block added by EGN: processing to enable automatic-resizing & aspect ratio locking
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
    # END block added by EGN

    config = {}
    config['showLink'] = show_link
    config['linkText'] = link_text

    #Note (by EGN): removing width and height from layout above causes default values to
    # be used (the '100%'s hardcoded below) which subsequently trigger adding a resize script.
    plot_html, plotdivid, width, height = _plot_html(
        figure_or_data, config, validate,
        '100%', '100%', global_requirejs=global_requirejs) #EGN added global_requirejs argument & plumbing

    resize_script = ''
    if resizable: # EGN added this entire block
        if aspect_ratio:
            #'    var plotdiv = $("#{id}").closest(".pygsti-wsoutput-group");\n' Maybe use this element later,
            #  instead of real plotdiv so that all plots resize at the same time -- but we'd need some way of
            #  creating the JQueryUI resizable from the workspace group and not here, since doing it here would
            #  create multiple resizable widgets on the same wsoutput-group element (one per plotly figure).
            resize_script = (  #very similar to script below except for plotdiv -> el and Math.min call (TODO: COMBINE?)
                '<script type="text/javascript">\n'
                '    var plotdiv = $("#{id}");'
                '    var box = null;'  #take as reference first parent with max-height or max-width set
                '    plotdiv.parents().each(function(i,el) {{'
    	        '      if(box == null && ($(el).css("max-height") != "none"'
                '         || $(el).css("max-width") != "none")) {{ box=$(el) }} }});'
                '    var w, h;'
                '    if(box == null) {{' #just use original dimensions
                '      plotdiv.css("width", {ow});'
                '      plotdiv.css("height",{oh});'
                '    }} else {{'
                '      w = Math.min(box.width(),{ow})\n;' # w < max-width (just add *min*)
                '      h = w / {ratio};\n' # but h may be > max-height
                '      plotdiv.css("width", w);\n' # adjust plotdiv based (*same* as original height)
                '      plotdiv.css("height",h);\n' # on box width
                '      if(box.height() < h) {{\n' # (due to max-height restriction)
                '        h = box.height();\n'
                '        w = h * {ratio};\n'
                '        plotdiv.css("width", w);\n' # adjust plotdiv based
                '        plotdiv.css("height",h);\n' # on (max) box height
                '      }}\n'
                '     box.css("max-width","none");\n'  # so plot can be resized
                '     box.css("max-height","none");\n' #  larger if desired
                '    }}\n'
                '    plotdiv.resizable({{\n'
                '      aspectRatio: {ratio},\n'
                '      resize: function( event, ui ) {{\n'
                '        var plotdiv = document.getElementById("{id}");\n'
                '        Plotly.Plots.resize(plotdiv); }}\n'
                '    }});\n'
                '$( document ).ready(function() {{\n'
                '   Plotly.Plots.resize(document.getElementById("{id}"));\n'
                '   console.log("Initial Resizing {id}");\n'
                '}});</script>\n'
            ).format(id=plotdivid, ratio=aspect_ratio, ow=orig_width, oh=orig_height)
        else:
            resize_script = (  #very similar to script below except for plotdiv -> el and Math.min call (TODO: COMBINE?)
                '<script type="text/javascript">\n'
                '    var el = $("#{id}").closest(".pygsti-wsoutput-group");\n'
                '   el.resizable({{\n'
                '     aspectRatio: {ratio},\n'
                '     resize: function( event, ui ) {{\n'
                '       var plotdiv = document.getElementById("{id}");\n'
                '       Plotly.Plots.resize(plotdiv); }}\n'
                '   }});\n'
            ).format(id=plotdivid)
        
    elif autosize:
    # OLD (original plotly) if width == '100%' or height == '100%':
    # EGN note: don't allow %'s layout yet - assume they're all pixels
        
        #EGN adds aspect_ratio (width/height) argument & effect
        if aspect_ratio is None: 
            aspect_ratio_js = ""
        else:
            #Alter width & height of plotdiv based on size of "box" (reference div)
            # Note that "box" div's width & height change dynamically and are unaltered.
            aspect_ratio_js = (
                '    var plotdiv = $("#{id}");'
                '    var box = null;'  #take as reference first parent with max-height or max-width set
                '    plotdiv.parents().each(function(i,el) {{'
	        '      if(box == null && ($(el).css("max-height") != "none"'
                '         || $(el).css("max-width") != "none")) {{ box=$(el) }} }});'
                '    if(box == null) box = plotdiv.parent();'
                '    var w = box.width();' # w < max-width
                '    var h = w / {ratio};' # but h may be > max-height
                '    plotdiv.css("width", w);' # adjust plotdiv based
                '    plotdiv.css("height",h);' # on box width
                '    if(box.height() < h) {{ ' # (due to max-height restriction)
                '      h = box.height();'
                '      w = h * {ratio};'
                '      plotdiv.css("width", w);' # adjust plotdiv based
                '      plotdiv.css("height",h);' # on (max) box height
                '    }}'
                '    console.log("Resize to " + w + ", " + h + " (ratio {ratio})");'
                ).format(id=plotdivid, ratio=aspect_ratio)

        resize_script = (
            '<script type="text/javascript">'
            'function {resizeFn}() {{\n'
            '  {aspectjs}\n'
            '  Plotly.Plots.resize(document.getElementById("{id}"));\n'
            '}}\n'
            'window.addEventListener("resize", function(){{'
            '  {resizeFn}(); }});'
            '$( document ).ready(function() {{'
            '  {resizeFn}();' #resize initially
            '  console.log("Initial Resizing {id}");}});'
            '</script>'
        ).format(id=plotdivid,
                 resizeFn="resize_%s" % str(plotdivid).replace('-',''),
                 aspectjs=aspect_ratio_js)
    
    if output_type == 'file':
        with open(filename, 'w') as f:
            if include_plotlyjs:
                plotly_js_script = ''.join([
                    '<script type="text/javascript">',
                    get_plotlyjs(),
                    '</script>',
                ])
            else:
                plotly_js_script = ''

            if image:
                if image not in __IMAGE_FORMATS:
                    raise ValueError('The image parameter must be one of the '
                                     'following: {}'.format(__IMAGE_FORMATS)
                                     )
                # if the check passes then download script is injected.
                # write the download script:
                script = get_image_download_script('plot')
                script = script.format(format=image,
                                       width=image_width,
                                       height=image_height,
                                       filename=image_filename,
                                       plot_id=plotdivid)
            else:
                script = ''

            f.write(''.join([
                '<html>',
                '<head><meta charset="utf-8" /></head>',
                '<body>',
                plotly_js_script,
                plot_html,
                resize_script,
                script,
                '</body>',
                '</html>']))

        url = 'file://' + os.path.abspath(filename)
        if auto_open:
            webbrowser.open(url)

        return url

    elif output_type == 'div':
        #EGN adds resize_script to 'div' case
        if include_plotlyjs:
            return ''.join([
                '<div>'
                '<script type="text/javascript">',
                get_plotlyjs(),
                '</script>',
                plot_html,
                resize_script,  #EGN added
                '</div>'
            ])
        #'<div id="box%s" class="pygsti-plotly-box">' % plotdivid, #EGN added attributes
        #  id="box%s" 
        else:
            #ORIGINAL Plotly: return plot_html
            return  '<div>' + plot_html + resize_script + "</div>" # EGN added

