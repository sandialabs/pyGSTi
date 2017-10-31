from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Helper functions for creating HTML documents by "merging" with a template"""

import collections as _collections
import os          as _os
import sys         as _sys
import shutil      as _shutil
import webbrowser as _webbrowser

import re  as _re
import subprocess  as _subprocess

from ..tools import compattools as _compat
from ..objects import VerbosityPrinter

def read_contents(filename):
    contents = None
    try: #on Windows using python3 open can fail when trying to read text files. encoding fixes this
        f = open(filename)
        contents = f.read()
    except UnicodeDecodeError:
        f = open(filename, encoding='utf-8') #try this, but not available in python 2.7!
        contents = f.read()

    f.close()
    
    try: # to convert to unicode since we use unicode literals
        contents = contents.decode('utf-8')
    except AttributeError: pass #Python3 case when unicode is read in natively (no need to decode)
    
    return contents


def insert_resource(connected, online_url, offline_filename,
                    integrity=None, crossorigin=None):
    if connected:
        if online_url:
            url = online_url
        else:
            #insert resource inline, since we don't want
            # to depend on offline/ directory when connected=True
            assert(offline_filename), \
                "connected=True without `online_url` requires offline filename!"
            absname = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                    "templates","offline",offline_filename)
            
            if offline_filename.endswith("js"):
                return '<script type="text/javascript">\n' + \
                    read_contents(absname) + "</script>\n"
            
            elif offline_filename.endswith("css"):
                return '<style>\n' + read_contents(absname) + "</style>\n"
            
            else:
                raise ValueError("Unknown resource type for %s" % offline_filename)
            
    else:
        assert(offline_filename), "connected=False requires offline filename"
        url = "offline/" + offline_filename
        
    if url.endswith("js"):
        
        tag = '<script src="%s"' % url
        if connected:
            if integrity: tag += ' integrity="%s"' % integrity
            if crossorigin: tag += ' crossorigin="%s"' % crossorigin
        tag += '></script>'
        return tag

    elif url.endswith("css"):
        return '<link rel="stylesheet" href="%s">' % url
    
    else:
        raise ValueError("Unknown resource type for %s" % url)


def rsync_offline_dir(outputDir):
    #Copy offline directory into outputDir updating any outdated files
    destDir = _os.path.join(outputDir, "offline")
    offlineDir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                               "templates","offline")
    if not _os.path.exists(destDir):
        _shutil.copytree(offlineDir, destDir)
        
    else:
        for dirpath, dirnames, filenames in _os.walk(offlineDir):
            for nm in filenames:
                srcnm = _os.path.join(dirpath, nm)
                relnm = _os.path.relpath(srcnm, offlineDir)
                destnm = _os.path.join(destDir, relnm)

                if not _os.path.isfile(destnm) or \
                    (_os.path.getmtime(destnm) < _os.path.getmtime(srcnm)):
                    _shutil.copyfile(srcnm, destnm)
                    #print("COPYING to %s" % destnm)


def read_and_preprocess_template(templateFilename, toggles):
    template = ''
    with open(templateFilename, 'r') as templatefile:
        template = templatefile.read()
        
    try: # convert to unicode if Python2 
        template = template.decode('utf-8')
    except AttributeError: pass #Python3 case

    if toggles is None:
        toggles = {}
        
    def preprocess(txt):
        try: i = txt.index("#iftoggle(")
        except ValueError: i = None

        try: k = txt.index("#elsetoggle")
        except ValueError: k = None

        try: j = txt.index("#endtoggle")
        except ValueError: j = None

        if i is None:
            return txt #no iftoggle, so no further processing to do
            
        if (k is not None and k < i) or (j is not None and j < i):
            return txt # else/end appears *before* if - so don't process the if

        #Process the #iftoggle
        off = len("#iftoggle(")
        end = txt[i+off:].index(')')
        toggleName = txt[i+off:i+off+end]
        pre_text = txt[0:i]  #text before our #iftoggle
        post_text = preprocess(txt[i+off+end+1:]) #text after
        
        if_text = ""
        else_text = ""

        #Process #elsetoggle or #endtoggle - whichever is first
        try: k = post_text.index("#elsetoggle") # index in (new) *post_text*
        except ValueError: k = None
        try: j = post_text.index("#endtoggle") # index in (new) *post_text*
        except ValueError: j = None

        if k is not None and (j is None or k < j): # if-block ends at #else
            #process #elsetoggle
            if_text = post_text[0:k]
            post_text = preprocess(post_text[k+len("#elsetoggle"):])
            else_processed = True
        else: else_processed = False

        #Process #endtoggle
        try: j = post_text.index("#endtoggle") # index in (new) *post_text*
        except ValueError: j = None
        assert(j is not None), "#iftoggle(%s) without corresponding #endtoggle" % toggleName
        
        if not else_processed: # if-block ends at #endtoggle
            if_text = post_text[0:j]
        else: # if-block already captured; else-block ends at #endtoggle
            else_text = post_text[0:j]
        post_text = preprocess(post_text[j+len("#endtoggle"):])
                
        if toggles[toggleName]:
            return pre_text + if_text + post_text
        else:
            return pre_text + else_text + post_text
    
    return preprocess(template)

def clearDir(path):
    if not _os.path.isdir(path): return
    for fn in _os.listdir(path):
        full_fn = _os.path.join(path,fn)
        if _os.path.isdir(full_fn):
            clearDir(full_fn)
            _os.rmdir(full_fn)
        else:
            _os.remove( full_fn )

def makeEmptyDir(dirname):
    if not _os.path.exists(dirname):
        _os.makedirs(dirname)
    else:
        assert(_os.path.isdir(dirname)), "%s exists but isn't a directory!" % dirname
        clearDir(dirname)
    return dirname


def fill_std_qtys(qtys, connected, renderMath, CSSnames):
    #Add favicon
    if 'favicon' not in qtys:
        if connected:
            favpath = "https://raw.githubusercontent.com/pyGSTio/pyGSTi/gh-pages"
        else:
            favpath = "offline/images"
            
        qtys['favicon'] = (
            '<link rel="icon" type="image/png" sizes="16x16" href="{fp}/favicon-16x16.png">\n'
            '<link rel="icon" type="image/png" sizes="32x32" href="{fp}/favicon-32x32.png">\n'
            '<link rel="icon" type="image/png" sizes="96x96" href="{fp}/favicon-96x96.png">\n'
            ).format(fp=favpath)
            
    #Add inline or CDN javascript    
    if 'jqueryLIB' not in qtys:
        qtys['jqueryLIB'] = insert_resource(
            connected, "https://code.jquery.com/jquery-3.2.1.min.js", "jquery-3.2.1.min.js",
            "sha256-hwg4gsxgFZhOsEEamdOYGBf13FyQuiTwlAQgxVSNgt4=",
            "anonymous")
        
    if 'jqueryUILIB' not in qtys:
        qtys['jqueryUILIB'] = insert_resource(
            connected, "https://code.jquery.com/ui/1.12.1/jquery-ui.min.js", "jquery-ui.min.js",
            "sha256-VazP97ZCwtekAsvgPBSUwPFKdrwD3unUfSGVYrahUqU=",
            "anonymous")
        
        qtys['jqueryUILIB'] += insert_resource(
            connected, "https://code.jquery.com/ui/1.12.1/themes/smoothness/jquery-ui.css",
            "smoothness-jquery-ui.css")
        
    if 'plotlyLIB' not in qtys:
        qtys['plotlyLIB'] = insert_resource(
            connected, "https://cdn.plot.ly/plotly-latest.min.js", "plotly-polarfixed.min.js")

    #if 'mathjaxLIB' not in qtys:
    #    assert(connected),"MathJax cannot be used unless connected=True."
    #    src = insert_resource(
    #        connected, "http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML",
    #        None)
    #
    #    #this *might* need to go at the very top of the HTML page to work
    #    qtys['mathjaxLIB'] = ('<script>'
    #         'var waitForPlotly = setInterval( function() {'
    #         '    if( typeof(window.Plotly) !== "undefined" && typeof(window.MathJax) !== "undefined" ){'
    #         '            MathJax.Hub.Config({ SVG: { font: "STIX-Web" }, displayAlign: "center" });'
    #         '            MathJax.Hub.Queue(["setRenderer", MathJax.Hub, "SVG"]);'
    #         '            clearInterval(waitForPlotly);'
    #         '    }}, 250 );'
    #         '</script>')
    #
    #    qtys['mathjaxLIB'] += '<script type="text/x-mathjax-config"> MathJax.Hub.Config({ ' + \
    #                         'tex2jax: {inlineMath: [["$","$"] ]} ' + \
    #                         '}); </script>' + src + \
    #                         '<style type="text/css"> ' + \
    #                         '.MathJax_MathML {text-indent: 0;} ' + \
    #                         '</style>'
    #    # removed ,["\\(","\\)"] from inlineMath so parentheses work in html

    if 'masonryLIB' not in qtys:
        qtys['masonryLIB'] = insert_resource(
            connected, "https://unpkg.com/masonry-layout@4/dist/masonry.pkgd.min.js",
            "masonry.pkgd.min.js")

    if 'katexLIB' not in qtys:
        qtys['katexLIB'] = insert_resource(
            connected, "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.7.1/katex.min.css",
            "katex.css")

        qtys['katexLIB'] += insert_resource(
            connected, "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.7.1/katex.min.js",
            "katex.min.js")
        
        qtys['katexLIB'] += insert_resource(
            connected, "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.7.1/contrib/auto-render.min.js",
            "auto-render.min.js")

        if renderMath:
            qtys['katexLIB'] += (
                '\n<script>'
                'document.addEventListener("DOMContentLoaded", function() {'
                '  $("#status").show();\n'
                '  $("#status").text("Rendering body math");\n'
                '  $(".math").each(function() {\n'
                '    console.log("Rendering KateX");\n'
                '    var texTxt = $(this).text();\n'
                '    el = $(this).get(0);\n'
                '    if(el.tagName == "DIV"){\n'
                '       addDisp = "\\displaystyle";\n'
                '    } else {\n'
                '    addDisp = "";\n'
                '    }\n'
                '    try {\n'
                '      katex.render(addDisp+texTxt, el);\n'
                '    }\n'
                '    catch(err) {\n'
                '      $(this).html("<span class=\'err\'>"+err);\n'
                '    }\n'
                '  });\n'
                '});\n'
                '</script>' )

#OLD: auto-render entire document
#        qtys['katexLIB'] += ('\n<script>'
#                'document.addEventListener("DOMContentLoaded", function() {'
#                'renderMathInElement(document.body, { delimiters: ['
#                '{left: "$$", right: "$$", display: true},'
#                '{left: "$", right: "$", display: false},'
#                '] } ); });'
#                '</script>')
        # removed so parens work:
        # '{left: "\\[", right: "\\]", display: true},'
        # '{left: "\\(", right: "\\)", display: false}'

    if 'plotlyexLIB' not in qtys:
        qtys['plotlyexLIB'] = insert_resource(
            connected, None, "pygsti_plotly_ex.js")

    if 'dashboardLIB' not in qtys:
        qtys['dashboardLIB'] = insert_resource(
            connected, None, "pygsti_dashboard.js")
    
    #Add inline CSS
    if 'CSS' not in qtys:
        qtys['CSS'] = "\n".join( [insert_resource(
            connected, None, cssFile)
                for cssFile in CSSnames] )

def render_as_html(qtys, render_options, link_to, verbosity):
    printer = VerbosityPrinter.build_printer(verbosity)
    
    #render quantities as HTML
    qtys_html = _collections.defaultdict(lambda x=0: "BLANK")
    for key,val in qtys.items():
        if _compat.isstr(val):
            qtys_html[key] = val
        else:
            printer.log("Rendering %s" % key, 3)
            if hasattr(val,'set_render_options'):
                val.set_render_options(**render_options)
                
                out = val.render("html")
                if link_to:
                    val.set_render_options(leave_includes_src=('tex' in link_to),
                                           render_includes=('pdf' in link_to) )
                    if 'tex' in link_to or 'pdf' in link_to: val.render("latex") 
                    if 'pkl' in link_to: val.render("python")

            else: #switchboards usually
                out = val.render("html")
                
            # Note: out is a dictionary of rendered portions
            qtys_html[key] = "<script>\n%(js)s\n</script>\n\n%(html)s" % out
            
    return qtys_html

        
            
def merge_html_template(qtys, templateFilename, outputFilename, auto_open=False,
                        precision=None, link_to=None, connected=False, toggles=None,
                        renderMath=True, resizable=True, autosize='none', verbosity=0,
                        CSSnames=("pygsti_dataviz.css", "pygsti_dashboard.css",
                                  "pygsti_fonts.css")):

    printer = VerbosityPrinter.build_printer(verbosity)

    assert(outputFilename.endswith(".html")), "outputFilename should have ended with .html!"
    outputDir = _os.path.dirname(outputFilename)
            
    #Copy offline directory into position
    if not connected:
        rsync_offline_dir(outputDir)

    fill_std_qtys(qtys, connected, renderMath, CSSnames)

    #render quantities as HTML
    qtys_html = render_as_html(qtys, dict(switched_item_mode="inline",
                                          global_requirejs=False,
                                          resizable=resizable, autosize=autosize,
                                          output_dir=None, link_to=link_to,
                                          precision=precision), link_to, printer)

    fullTemplateFilename = _os.path.join( _os.path.dirname(_os.path.abspath(__file__)),
                                          "templates", templateFilename )
    template = read_and_preprocess_template(fullTemplateFilename, toggles)
    
    #Do actual fill -- everything needs to be unicode at this point.
    filled_template = template % qtys_html
      #.format_map(qtys_html) #need python 3.2+
      
    if _sys.version_info <= (3, 0): # Python2: need to re-encode for write(...)
        filled_template = filled_template.encode('utf-8')

    with open(outputFilename, 'w') as outputfile:
        outputfile.write(filled_template)

    printer.log("Output written to %s" % outputFilename)
        
    if auto_open:
        url = 'file://' + _os.path.abspath(outputFilename)
        printer.log("Opening %s..." % outputFilename)
        _webbrowser.open(url)


def merge_html_template_dir(qtys, templateDir, outputDir, auto_open=False,
                            precision=None, link_to=None, connected=False, toggles=None,
                            renderMath=True, resizable=True, autosize='none', verbosity=0,
                            CSSnames=("pygsti_dataviz.css", "pygsti_dashboard.css",
                                      "pygsti_fonts.css")):
    
    printer = VerbosityPrinter.build_printer(verbosity)
        
    #Create directories if needed; otherwise clear it
    figDir = makeEmptyDir(_os.path.join(outputDir, 'figures'))
    tabDir = makeEmptyDir(_os.path.join(outputDir, 'tabs'))

    #FIX
    ##clear offline dir if it exists
    #offlineDir = _os.path.join(outputDir, 'offline')
    #if _os.path.isdir(offlineDir):
    #    _clearDir(offlineDir)
    #    _os.rmdir(offlineDir) #otherwise rsync doesn't work (?)
            
    #Copy offline directory into position
    if not connected:
        rsync_offline_dir(outputDir)

    fill_std_qtys(qtys, connected, renderMath, CSSnames)

    #render quantities as HTML
    qtys_html = render_as_html(qtys, dict(switched_item_mode="separate files",
                                          global_requirejs=False,
                                          resizable=resizable, autosize=autosize,
                                          output_dir=figDir, link_to=link_to,
                                          precision=precision), link_to, printer)
        
    #Insert qtys into template file(s)
    baseTemplateDir = _os.path.join( _os.path.dirname(_os.path.abspath(__file__)), "templates", templateDir)
    templateFilenames = [fn for fn in _os.listdir(baseTemplateDir) if fn.endswith(".html")]
    outputFilenames = []
    for fn in templateFilenames:
        outfn = _os.path.join(outputDir, fn) if (fn == 'main.html') else \
                _os.path.join(outputDir, 'tabs', fn)
        outputFilenames.append( outfn )
        
    for templateFilename,outputName in zip(templateFilenames,outputFilenames):
        templateFilename = _os.path.join( baseTemplateDir, templateFilename )
        template = read_and_preprocess_template(templateFilename, toggles)
    
        #Do actual fill -- everything needs to be unicode at this point.
        filled_template = template % qtys_html
          #.format_map(qtys_html) #need python 3.2+
      
        if _sys.version_info <= (3, 0): # Python2: need to re-encode for write(...)
            filled_template = filled_template.encode('utf-8')

        with open(outputName, 'w') as outputfile:
            outputfile.write(filled_template)

    printer.log("Output written to %s directory" % outputDir)

    if auto_open:
        outputFilename = _os.path.join(outputDir, 'main.html')
        url = 'file://' + _os.path.abspath(outputFilename)
        printer.log("Opening %s..." % outputFilename)
        _webbrowser.open(url)


def process_call(call):
    process = _subprocess.Popen(call, stdout=_subprocess.PIPE,
                                stderr=_subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout, stderr, process.returncode

def evaluate_call(call, stdout, stderr, returncode, printer):
    if len(stderr) > 0:
        printer.error(stderr)
    if returncode > 0:
        raise _subprocess.CalledProcessError(returncode, call)

def merge_latex_template(qtys, templateFilename, outputFilename):
    templateFilename = _os.path.join( _os.path.dirname(_os.path.abspath(__file__)),
                                          "templates", templateFilename )
    template = ''
    with open(templateFilename, 'r') as templatefile:
        template = templatefile.read()
    template = template.replace("{", "{{").replace("}", "}}") #double curly braces (for format processing)                                   
    # Replace template field markers with `str.format` fields.
    template = _re.sub( r"\\putfield\{\{([^}]+)\}\}\{\{[^}]*\}\}", "{\\1}", template)

    # Replace str.format fields with values and write to output file
    template = template.format(**qtys)
    
    with open(outputFilename, 'w') as outputfile:
        outputfile.write(template)
