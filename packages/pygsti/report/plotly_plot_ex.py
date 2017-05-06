from plotly import tools as _plotlytools
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
        
        #Special polar plot case - see below - add dummy width & height so
        # we can find/replace them with variables in generated javascript.
        if 'angularaxis' in fig['layout']:
            fig['layout']['width'] = 123
            fig['layout']['height'] = 123

    # END block added by EGN

    config = {}
    config['showLink'] = show_link
    config['linkText'] = link_text

    #Note (by EGN): removing width and height from layout above causes default values to
    # be used (the '100%'s hardcoded below) which subsequently trigger adding a resize script.
    plot_html, plotdivid, width, height = _plot_html(
        figure_or_data, config, validate,
        '100%', '100%', global_requirejs=False) #EGN added global_requirejs argument & plumbing
       #Note: no need for global_requirejs here since we now extract js and remake full script.

    #EGN added: reset removed elements to avoid corrupting dictionary downstream
    if autosize or resizable:
        if orig_width: fig['layout']['width'] = orig_width
        if orig_height: fig['layout']['height'] = orig_height

    #EGN added: separate the HTML and JS in plot_html so we can insert
    # initial-sizing JS between them.  NOTE: this is FRAGILE and depends
    # on Plotly output (plot_html) being HTML followed by JS
    tag = '<script type="text/javascript">'; end_tag = '</script>'
    iTag = plot_html.index(tag)
    plot_js = plot_html[iTag+len(tag):-len(end_tag)]
    plot_html = plot_html[0:iTag]

    #EGN added
    if global_requirejs:
        requireJSbegin = "require(['jquery','jquery-UI','plotly'],function($,ui,Plotly) {"
        requireJSend = "});"
    else:
        requireJSbegin = requireJSend = ""

    full_script = ''
    if resizable or autosize: # EGN added this entire block
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
    
        createFn="create_%s" % str(plotdivid).replace('-','') #EGN added
        resizeFn="resize_%s" % str(plotdivid).replace('-','') #EGN added    
        aspect_val = aspect_ratio if aspect_ratio else "null"

        if resizable: #bind a "resize" event to plotdiv for the container to call
            resizable_js = (
                '     var plotdiv = $("#{id}");\n'
                '     plotdiv.addClass("resizable-plot");\n'
                '     plotdiv.on("resize", function(event, ui) {{\n'
                '       pex_update_size($("#{id}"), {ratio});\n'
                '       {resizeFn}();\n'
                '       //console.log("Resized {id}");\n'
                '     }});\n'
            ).format(id=plotdivid, resizeFn=resizeFn, ratio=aspect_val)
        else: resizable_js = ''
                    
        if autosize: # add a resize handler for the window resizing
            autosize_js = (
                'var timeout_{resizeFn};\n'
                'window.addEventListener("resize",function() {{\n'
                '    clearTimeout(timeout_{resizeFn});\n'
                '    timeout_{resizeFn} = setTimeout(window_{resizeFn}, 300);\n'
                '}});\n'
                'function window_{resizeFn}(){{\n'
                '    var plotdiv = $("#{id}");\n'
                '    pex_update_size(plotdiv, {ratio});\n'
                '    {resizeFn}();\n'
                '    //console.log("Window resize on {id}");\n'
                '}}'
            ).format(id=plotdivid, ratio=aspect_val, resizeFn=resizeFn)
        else: autosize_js = ''

        full_script = (
            '<script type="text/javascript">\n'
            '  {requireJSbegin}\n'
            '  console.log("Setting initial size of {id}");\n'
            '  pex_init_container($("#{id}"), {ow}, {oh});\n'
            '  function {createFn}() {{\n'
            '     pex_update_size($("#{id}"), {ratio});\n' # set *initial* size of plotdiv
            '     {plotlyCreateJS} \n}}\n'
            '  function {resizeFn}() {{ {plotlyResizeJS} }}\n'
            '  $("#{id}").on("create", function(event, ui) {{\n' #always add create handler
            '    {createFn}();\n'
            '    console.log("Created {id}");\n'
            '  }});\n'
            '  {resizableJS}\n'
            '  {autosizeJS}\n'
            '  {requireJSend}\n'
            '</script>\n'
        ).format(id=plotdivid, ratio=aspect_val,
                 ow=orig_width if orig_width else "null",
                 oh=orig_height if orig_height else "null",
                 createFn=createFn,
                 resizeFn=resizeFn,
                 plotlyCreateJS=plotly_create_js,
                 plotlyResizeJS=plotly_resize_js,
                 resizableJS=resizable_js,
                 autosizeJS=autosize_js,
                 requireJSbegin=requireJSbegin,
                 requireJSend=requireJSend)
    else:
        #Note: this is why we set global_requirejs to False
        # in _plot_html -- we add it here.
        full_script = (
            '<script type="text/javascript">\n'
            '  {requireJSbegin}\n'
            '  {plotlyCreateJS}\n'
            '  {requireJSend}\n'
            '</script>\n').format(
                requireJSbegin=requireJSbegin,
                plotlyCreateJS=plot_js,
                requireJSend=requireJSend)

    
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
                full_script,
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
                full_script,
                '</div>'
            ])
        else:
            #ORIGINAL Plotly: return plot_html (would be + full_script now)
            return  '<div>' + plot_html + full_script + "</div>" # EGN added

