from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Report generation functions. """

import pickle as _pickle
import os  as _os
import sys as _sys
import time as _time
import numpy as _np
import warnings as _warnings
import collections as _collections
import webbrowser as _webbrowser
import zipfile as _zipfile
from scipy.stats import chi2 as _chi2

from ..objects import VerbosityPrinter, Basis, SmartCache
from ..objects import DataComparator as _DataComparator
from ..tools   import compattools as _compat
from ..tools   import timed_block as _timed_block

from ..tools.mpitools import distribute_indices as _distribute_indices

from .. import tools as _tools

from . import workspace as _ws
from . import autotitle as _autotitle
from .notebook import Notebook as _Notebook

import functools as _functools

from pprint import pprint as _pprint

def _read_and_preprocess_template(templateFilename, toggles):
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

def _merge_template(qtys, templateFilenameOrDir, outputFilename, auto_open,
                    precision, pdf_links, python_links,
                    CSSnames=("pygsti_dataviz.css","pygsti_report.css","pygsti_fonts.css"),
                    connected=False, toggles=None, renderMath=True, verbosity=0):

    printer = VerbosityPrinter.build_printer(verbosity)

    #Figure out which rendering mode we'll use
    full = _os.path.join( _os.path.dirname(_os.path.abspath(__file__)),
                          "templates", templateFilenameOrDir )
    if _os.path.isdir(full) or not outputFilename.endswith(".html"):
        #template is a directory, so we must outputFilename must also be a directory
        render_typ = "htmldir" # output is a dir of html files
        if outputFilename.endswith(".html") or outputFilename.endswith(".pdf"):
            outputDir = _os.path.splitext(outputFilename)[0] #remove extension
        else:
            outputDir = outputFilename #assume any .ext is desired in folder name
        outputFilename = _os.path.join(outputDir, 'main.html')

        def clearDir(path):
            if not _os.path.isdir(path): return
            for fn in _os.listdir(path):
                full_fn = _os.path.join(path,fn)
                if _os.path.isdir(full_fn):
                    clearDir(full_fn)
                    _os.rmdir(full_fn)
                else:
                    _os.remove( full_fn )
        
        #Create figures directory if it doesn't already exist,
        # otherwise clear it
        figDir = _os.path.join(outputDir, 'figures')
        if not _os.path.exists(figDir):
            _os.makedirs(figDir)
        else:
            assert(_os.path.isdir(figDir)), "%s exists but isn't a directory!" % figDir
            clearDir(figDir)

        #Create tabs directory if it doesn't already exist,
        # otherwise clear it
        tabDir = _os.path.join(outputDir, 'tabs')
        if not _os.path.exists(tabDir):
            _os.makedirs(tabDir)
        else:
            assert(_os.path.isdir(tabDir)), "%s exists but isn't a directory!" % tabDir
            clearDir(tabDir)

        #clear offline dir if it exists
        offlineDir = _os.path.join(outputDir, 'offline')
        if _os.path.isdir(offlineDir):
            clearDir(offlineDir)
            _os.rmdir(offlineDir) #otherwise rsync doesn't work (?)
            
    else:
        assert(outputFilename.endswith(".html")), "outputFilename should have ended with .html!"
        render_typ = "html"
        outputDir = _os.path.dirname(outputFilename)

    #Copy offline directory into position
    if not connected:
        _ws.rsync_offline_dir(outputDir)

    figureDir = _os.path.join(outputDir, 'figures')

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
        qtys['jqueryLIB'] = _ws.insert_resource(
            connected, "https://code.jquery.com/jquery-3.2.1.min.js", "jquery-3.2.1.min.js",
            "sha256-hwg4gsxgFZhOsEEamdOYGBf13FyQuiTwlAQgxVSNgt4=",
            "anonymous")
        
    if 'jqueryUILIB' not in qtys:
        qtys['jqueryUILIB'] = _ws.insert_resource(
            connected, "https://code.jquery.com/ui/1.12.1/jquery-ui.min.js", "jquery-ui.min.js",
            "sha256-VazP97ZCwtekAsvgPBSUwPFKdrwD3unUfSGVYrahUqU=",
            "anonymous")
        
        qtys['jqueryUILIB'] += _ws.insert_resource(
            connected, "https://code.jquery.com/ui/1.12.1/themes/smoothness/jquery-ui.css",
            "smoothness-jquery-ui.css")
        
    if 'plotlyLIB' not in qtys:
        qtys['plotlyLIB'] = _ws.insert_resource(
            connected, "https://cdn.plot.ly/plotly-latest.min.js", "plotly-polarfixed.min.js")

    #if 'mathjaxLIB' not in qtys:
    #    assert(connected),"MathJax cannot be used unless connected=True."
    #    src = _ws.insert_resource(
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

    if 'katexLIB' not in qtys:
        qtys['katexLIB'] = _ws.insert_resource(
            connected, "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.7.1/katex.min.css",
            "katex.css")

        qtys['katexLIB'] += _ws.insert_resource(
            connected, "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.7.1/katex.min.js",
            "katex.min.js")
        
        qtys['katexLIB'] += _ws.insert_resource(
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
        qtys['plotlyexLIB'] = _ws.insert_resource(
            connected, None, "pygsti_plotly_ex.js")

    if 'dashboardLIB' not in qtys:
        qtys['dashboardLIB'] = _ws.insert_resource(
            connected, None, "pygsti_dashboard.js")
    
    #Add inline CSS
    if 'CSS' not in qtys:
        qtys['CSS'] = "\n".join( [_ws.insert_resource(
            connected, None, cssFile)
                for cssFile in CSSnames] )

    #render quantities as HTML
    qtys_html = _collections.defaultdict(lambda x=0: "BLANK")
    for key,val in qtys.items():
        if _compat.isstr(val):
            qtys_html[key] = val

        else:

            printer.log("Rendering %s" % key, 3)
            if isinstance(val,_ws.WorkspaceOutput): #switchboards don't have render options yet...
                val.set_render_options(output_dir=figureDir,
                                       link_to_pdf=pdf_links, link_to_pkl=python_links)
            
            if isinstance(val,_ws.WorkspaceTable):
                #supply precision argument
                out = val.render(render_typ, precision=precision, resizable=True, autosize=False)
                if pdf_links:    val.render("latexdir") 
                if python_links: val.render("pythondir")
            elif isinstance(val,_ws.WorkspacePlot):
                out = val.render(render_typ, resizable=True, autosize=False)
                if pdf_links:    val.render("latexdir")
                if python_links: val.render("pythondir")
            else: #switchboards usually
                out = val.render(render_typ)
                
            # Note: out is a dictionary of rendered portions
            qtys_html[key] = "<script>\n%(js)s\n</script>\n\n%(html)s" % out

        
    #Insert qtys into template file(s)
    if _os.path.isdir(full):
        baseTemplateDir = full
        templateFilenames = [fn for fn in _os.listdir(baseTemplateDir) if fn.endswith(".html")]
        outputFilenames = []
        for fn in templateFilenames:
            outfn = _os.path.join(outputDir, fn) if (fn == 'main.html') else \
                    _os.path.join(outputDir, 'tabs', fn)
            outputFilenames.append( outfn )
    else:
        baseTemplateDir = _os.path.join( _os.path.dirname(_os.path.abspath(__file__)), "templates" )
        templateFilenames = [templateFilenameOrDir]
        outputFilenames = [outputFilename]
        
    for templateFilename,outputName in zip(templateFilenames,outputFilenames):
        templateFilename = _os.path.join( baseTemplateDir, templateFilename )
        template = _read_and_preprocess_template(templateFilename, toggles)
    
        #Do actual fill -- everything needs to be unicode at this point.
        filled_template = template % qtys_html
          #.format_map(qtys_html) #need python 3.2+
      
        if _sys.version_info <= (3, 0): # Python2: need to re-encode for write(...)
            filled_template = filled_template.encode('utf-8')

        with open(outputName, 'w') as outputfile:
            outputfile.write(filled_template)

    if render_typ == "html":
        printer.log("Output written to %s" % outputFilename)
    else: # render_typ == "htmldir"
        printer.log("Output written to %s directory" % outputDir)

    if auto_open:
        url = 'file://' + _os.path.abspath(outputFilename)
        printer.log("Opening %s..." % outputFilename)
        _webbrowser.open(url)
        
def _errgen_formula(errgen_type, typ):
    assert(typ in ('html','latex'))
    if errgen_type == "logTiG":
        ret = '<span class="math">\hat{G} = G_{\mathrm{target}}e^{\mathbb{L}}</span>'
    elif errgen_type == "logG-logT":
        ret = '<span class="math">\hat{G} = e^{\mathbb{L} + \log G_{\mathrm{target}}}</span>'
    else:
        ret = "???"
    
    if typ == "latex": #minor modifications for latex versino
        ret = ret.replace('<span class="math">','$')
        ret = ret.replace('</span>','$')

    return ret

def _add_new_labels(running_lbls, current_lbls):
    """ 
    Simple routine to add current-labels to a list of
    running-labels without introducing duplicates and
    preserving order as best we can.
    """
    if running_lbls is None:
        return current_lbls[:] #copy!
    elif running_lbls != current_lbls:
        for lbl in current_lbls:
            if lbl not in running_lbls:
                running_lbls.append(lbl)
    return running_lbls


def create_offline_zip(outputDir="."):
    """ 
    Creates a zip file containing the a directory ("offline") of files
    need to display "offline" reports (generated with `connected=False`).  

    For offline reports to display, the "offline" folder must be placed
    in the same directory as the report's HTML file.  This function can
    be used to easily obtain a copy of the offline folder for the purpose
    of sharing offline reports with other people.  If you're just creating
    your own offline reports using pyGSTi, the offline folder is
    automatically copied into it's proper position - so you don't need 
    to call this function.

    Parameters
    ----------
    outputDir : str, optional
        The directory in which "offline.zip" should be place.

    Returns
    -------
    None
    """
    templatePath = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                "templates")

    zipFName = _os.path.join(outputDir, "offline.zip")
    zipHandle = _zipfile.ZipFile(zipFName, 'w', _zipfile.ZIP_DEFLATED)    
    for root, dirs, files in _os.walk(_os.path.join(templatePath,"offline")):
        for f in files:
            fullPath = _os.path.join(root, f)
            zipHandle.write(fullPath, _os.path.relpath(fullPath,templatePath))
    zipHandle.close()

def _set_toggles(results_dict):
    #Determine when to get gatestring weight (scaling) values and show via
    # ColorBoxPlots below by checking whether any estimate has "weights"
    # parameter (a dict) with > 0 entries.
    toggles = {}
    
    toggles["ShowScaling"] = False
    for res in results_dict.values():
        for i,(lbl,est) in enumerate(res.estimates.items()):
            weights = est.parameters.get("weights",None)
            if weights is not None and len(weights) > 0:
                toggles["ShowScaling"] = True

    return toggles
    
def _create_master_switchboard(ws, results_dict, confidenceLevel,
                               nmthreshold, computecrs, comm):
    """
    Creates the "master switchboard" used by several of the reports
    """
    
    dataset_labels = list(results_dict.keys())
    est_labels = None
    gauge_opt_labels = None
    Ls = None        

    for results in results_dict.values():
        est_labels = _add_new_labels(est_labels, list(results.estimates.keys()))
        Ls = _add_new_labels(Ls, results.gatestring_structs['final'].Ls)    
        for est in results.estimates.values():
            gauge_opt_labels = _add_new_labels(gauge_opt_labels,
                                               list(est.goparameters.keys()))            

    Ls = list(sorted(Ls)) #make sure Ls are sorted in increasing order
    
    multidataset = bool(len(dataset_labels) > 1)
    multiest = bool(len(est_labels) > 1)
    multiGO = bool(len(gauge_opt_labels) > 1)
    multiL = bool(len(Ls) > 1)
            
    switchBd = ws.Switchboard(
        ["Dataset","Estimate","G-Opt","max(L)"],
        [dataset_labels, est_labels, gauge_opt_labels, list(map(str,Ls))],
        ["dropdown","dropdown", "buttons", "slider"], [0,0,0,len(Ls)-1],
        show=[multidataset,multiest,False,False] # "global" switches only
    )

    switchBd.add("ds",(0,))
    switchBd.add("prepStrs",(0,))
    switchBd.add("effectStrs",(0,))
    switchBd.add("strs",(0,))
    switchBd.add("germs",(0,))

    switchBd.add("eff_ds",(0,1))    
    switchBd.add("scaledSubMxsDict",(0,1))
    switchBd.add("gsTarget",(0,1))
    switchBd.add("params",(0,1))
    switchBd.add("objective",(0,1))
    switchBd.add("mpc",(0,1))
    switchBd.add("clifford_compilation",(0,1))

    switchBd.add("gsFinalIter",(0,1))
    switchBd.add("gsFinal",(0,1,2))
    switchBd.add("gsTargetAndFinal",(0,1,2)) #general only!
    switchBd.add("goparams",(0,1,2))
    switchBd.add("gsL",(0,1,3))
    switchBd.add("gss",(0,3))
    switchBd.add("gssFinal",(0,))
    switchBd.add("gsAllL",(0,1))
    switchBd.add("gssAllL",(0,))

    if confidenceLevel is not None:
        switchBd.add("cri",(0,1,2))

    for d,dslbl in enumerate(dataset_labels):
        results = results_dict[dslbl]
        
        switchBd.ds[d] = results.dataset
        switchBd.prepStrs[d] = results.gatestring_lists['prep fiducials']
        switchBd.effectStrs[d] = results.gatestring_lists['effect fiducials']
        switchBd.strs[d] = (results.gatestring_lists['prep fiducials'],
                            results.gatestring_lists['effect fiducials'])
        switchBd.germs[d] = results.gatestring_lists['germs']

        switchBd.gssFinal[d] = results.gatestring_structs['final']
        for iL,L in enumerate(Ls): #allow different results to have different Ls
            if L in results.gatestring_structs['final'].Ls:
                k = results.gatestring_structs['final'].Ls.index(L)
                switchBd.gss[d,iL] = results.gatestring_structs['iteration'][k]
        #OLD switchBd.gss[d,:] = results.gatestring_structs['iteration']
        switchBd.gssAllL[d] = results.gatestring_structs['iteration']

        for i,lbl in enumerate(est_labels):
            est = results.estimates.get(lbl,None)
            if est is None: continue

            switchBd.params[d,i] = est.parameters
            switchBd.objective[d,i] = est.parameters['objective']
            if est.parameters['objective'] == "logl":
                switchBd.mpc[d,i] = est.parameters['minProbClip']
            else:
                switchBd.mpc[d,i] = est.parameters['minProbClipForWeighting']
            switchBd.clifford_compilation[d,i] = est.parameters.get("clifford compilation",None)

            NA = ws.NotApplicable()
            effds, scale_subMxs = est.get_effective_dataset(True)
            switchBd.eff_ds[d,i] = effds
            switchBd.scaledSubMxsDict[d,i] = {'scaling': scale_subMxs, 'scaling.colormap': "revseq"}
            switchBd.gsTarget[d,i] = est.gatesets['target']
            switchBd.gsFinalIter[d,i] = est.gatesets['final iteration estimate']
            switchBd.gsFinal[d,i,:] = [ est.gatesets.get(l,NA) for l in gauge_opt_labels ]
            switchBd.gsTargetAndFinal[d,i,:] = \
                        [ [est.gatesets['target'], est.gatesets[l]] if (l in est.gatesets) else NA
                          for l in gauge_opt_labels ]
            switchBd.goparams[d,i,:] = [ est.goparameters.get(l,NA) for l in gauge_opt_labels]

            for iL,L in enumerate(Ls): #allow different results to have different Ls
                if L in results.gatestring_structs['final'].Ls:
                    k = results.gatestring_structs['final'].Ls.index(L)
                    switchBd.gsL[d,i,iL] = est.gatesets['iteration estimates'][k]
            #OLD switchBd.gsL[d,i,:] = est.gatesets['iteration estimates']
            switchBd.gsAllL[d,i] = est.gatesets['iteration estimates']
        
            if confidenceLevel is not None:
                #FUTURE: reuse Hessian for multiple gauge optimizations of the same gate set (or leave this to user?)
        
                #Check whether we should use non-Markovian error bars:
                # If fit is bad, check if any reduced fits were computed
                # that we can use with in-model error bars.  If not, use
                # experimental non-markovian error bars.
                if est.misfit_sigma() > nmthreshold:
                    est_confidenceLevel = -abs(confidenceLevel)
                else: est_confidenceLevel = confidenceLevel

                for il,l in enumerate(gauge_opt_labels):
                    if l in est.gatesets:
                        switchBd.cri[d,i,il] = est.get_confidence_region(
                            est_confidenceLevel, l, "final",
                            allowcreate=computecrs, comm=comm)
                    else: switchBd.cri[d,i,il] = NA
                    
    return switchBd, dataset_labels, est_labels, gauge_opt_labels, Ls



def create_general_report(results, filename, title="auto",
                          confidenceLevel=None,                          
                          linlogPercentile=5, errgen_type="logTiG",
                          nmthreshold=50, precision=None,
                          comm=None, ws=None, auto_open=False,
                          cachefile=None, brief=False,
                          connected=False, computecrs=True,
                          pdf_links=False, python_links=False, verbosity=1):
    """
    Create a "general" GST report.  This report is "general" in that it is
    suited to display results for any number of qubits/qutrits.  Along with
    the results, it includes background and explanation text.

    Parameters
    ----------
    results : Results
        An object which represents the set of results from one *or more* GST
        estimation runs, typically obtained from running 
        :func:`do_long_sequence_gst` or :func:`do_stdpractice_gst`, OR a
        dictionary of such objects, representing multiple GST runs to be
        compared (typically all with *different* data sets). The keys of this
        dictionary are used to label different data sets that are selectable
        in the report.

    filename : string, optional
       The output filename where the report file(s) will be saved.  If
       None, then no output file is produced (but returned Workspace
       still caches all intermediate results).

    title : string, optional
       The title of the report.  "auto" causes a random title to be
       generated (which you may or may not like).

    confidenceLevel : int, optional
       If not None, then the confidence level (between 0 and 100) used in
       the computation of confidence regions/intervals. If None, no
       confidence regions or intervals are computed.

    linlogPercentile : float, optional
        Specifies the colorscale transition point for any logL or chi2 color
        box plots.  The lower `(100 - linlogPercentile)` percentile of the
        expected chi2 distribution is shown in a linear grayscale, and the 
        top `linlogPercentile` is shown on a logarithmic colored scale.

    errgen_type: {"logG-logT", "logTiG"}
        The type of error generator to compute.  Allowed values are:
        
        - "logG-logT" : errgen = log(gate) - log(target_gate)
        - "logTiG" : errgen = log( dot(inv(target_gate), gate) )

    nmthreshold : float, optional
        The threshold, in units of standard deviations, that triggers the
        usage of non-Markovian error bars.  If None, then non-Markovian
        error bars are never computed.

    precision : int or dict, optional
        The amount of precision to display.  A dictionary with keys
        "polar", "sci", and "normal" can separately specify the 
        precision for complex angles, numbers in scientific notation, and 
        everything else, respectively.  If an integer is given, it this
        same value is taken for all precision types.  If None, then
        `{'normal': 6, 'polar': 3, 'sci': 0}` is used.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    ws : Workspace, optional
        The workspace used as a scratch space for performing the calculations
        and visualizations required for this report.  If you're creating
        multiple reports with similar tables, plots, etc., it may boost
        performance to use a single Workspace for all the report generation.

    auto_open : bool, optional
        If True, automatically open the report in a web browser after it
        has been generated.

    cachefile : str, optional
        filename with cached workspace results

    brief : bool, optional
        Whether large, disk-space-consuming plots are omitted.

    connected : bool, optional
        Whether output HTML should assume an active internet connection.  If
        True, then the resulting HTML file size will be reduced because it
        will link to web resources (e.g. CDN libraries) instead of embedding
        them.

    computecrs : bool, optional
        Whether to compute confidence regions as needed, according to
        `confidenceLevel`.  If True, any absent confidence regions that would
        be used are computed.  If False, only existing confidence regions are
        used, and error bars are simply omitted when a confidence region is 
        absent.

    pdf_links : bool, optional
        If True, render PDF versions of plots and tables, and create links 
        to them in the report.

    python_links : bool, optional
        If True, render Python versions of plots (pickled python data) and tables
        (pickled pandas DataFrames), and create links to them in the report.

    verbosity : int, optional
       How much detail to send to stdout.
    

    Returns
    -------
    Workspace
        The workspace object used to create the report
    """
    printer = VerbosityPrinter.build_printer(verbosity, comm=comm)
    printer.log('*** Creating workspace ***')
    if ws is None: ws = _ws.Workspace(cachefile)

    if isinstance(title,int): #to catch backward compatibility issues
        raise ValueError(("'title' argument must be a string.  You may be accidentally"
                          " specifying an int here because in older versions of pyGSTi"
                          " the third argument to create_general_report was the"
                          " confidence interval - please note the updated function signature"))
    
    if title is None or title == "auto":
        autoname = _autotitle.generate_name()
        title = "GST Report for " + autoname
        _warnings.warn( ("You should really specify `title=` when generating reports,"
                         "as this makes it much easier to identify them later on.  "
                         "Since you didn't, pyGSTi will has generated a random one"
                         " for you: '{}'.").format(autoname))

    results_dict = results if isinstance(results, dict) else {"unique": results}
    toggles = _set_toggles(results_dict)

    #DEBUG
    renderMath = True
    #_ws.WorkspaceOutput.default_render_options['click_to_display'] = True #don't render any plots until they're clicked
    #_ws.WorkspaceOutput.default_render_options['render_math'] = renderMath #don't render any math

    qtys = {} # stores strings to be inserted into report template
    def addqty(name, fn, *args, **kwargs):
        with _timed_block(name, formatStr='{:45}', printer=printer, verbosity=2):
            qtys[name] = fn(*args, **kwargs)

    qtys['title'] = title
    qtys['date'] = _time.strftime("%B %d, %Y")
    qtys['confidenceLevel'] = "%d" % \
        confidenceLevel if confidenceLevel is not None else "NOT-SET"
    qtys['linlg_pcntle'] = "%d" % round(linlogPercentile) #to nearest %
    qtys['errorgenformula'] = _errgen_formula(errgen_type, 'html')

    # Generate Tables
    printer.log("*** Generating switchboard tables ***")

    #Create master switchboard
    switchBd, dataset_labels, est_labels, gauge_opt_labels, Ls = \
            _create_master_switchboard(ws, results_dict,
                                       confidenceLevel, nmthreshold,
                                       computecrs, comm)

    if confidenceLevel is not None:
        #TODO: make plain text fields which update based on switchboards?
        for some_cri in switchBd.cri.flat: #can have only some confidence regions
            if some_cri is not None: # OLD: switchBd.cri[0,0,0]
                qtys['confidenceIntervalScaleFctr'] = "%.3g" % some_cri.intervalScaling
                qtys['confidenceIntervalNumNonGaugeParams'] = "%d" % some_cri.nNonGaugeParams

    multidataset = bool(len(dataset_labels) > 1)
    multiest = bool(len(est_labels) > 1)
    multiGO = bool(len(gauge_opt_labels) > 1)
    multiL = bool(len(Ls) > 1)

    #goView = [multidataset,multiest,multiGO,False]
    #maxLView = [multidataset,multiest,False,multiL]
    goView = [False,False,multiGO,False]
    maxLView = [False,False,False,multiL]

    qtys['topSwitchboard'] = switchBd
    qtys['goSwitchboard1'] = switchBd.view(goView,"v1")
    qtys['goSwitchboard2'] = switchBd.view(goView,"v2")
    qtys['maxLSwitchboard1'] = switchBd.view(maxLView,"v6")

    gsTgt = switchBd.gsTarget
    ds = switchBd.ds
    eff_ds = switchBd.eff_ds
    prepStrs = switchBd.prepStrs
    effectStrs = switchBd.effectStrs
    germs = switchBd.germs
    strs = switchBd.strs
    cliffcomp = switchBd.clifford_compilation

    addqty('targetSpamBriefTable', ws.SpamTable, gsTgt, None, includeHSVec=False)
    addqty('targetGatesBoxTable', ws.GatesTable, gsTgt, display_as="boxes")
    addqty('datasetOverviewTable', ws.DataSetOverviewTable, ds)

    gsFinal = switchBd.gsFinal
    cri = switchBd.cri if (confidenceLevel is not None) else None
    addqty('bestGatesetSpamParametersTable', ws.SpamParametersTable, gsFinal, cri)
    addqty('bestGatesetSpamBriefTable', ws.SpamTable, switchBd.gsTargetAndFinal,
                                                         ['Target','Estimated'],
                                                         cri, includeHSVec=False)
    addqty('bestGatesetSpamVsTargetTable', ws.SpamVsTargetTable, gsFinal, gsTgt, cri)
    addqty('bestGatesetGaugeOptParamsTable', ws.GaugeOptParamsTable, switchBd.goparams)
    addqty('bestGatesetGatesBoxTable', ws.GatesTable, switchBd.gsTargetAndFinal,
                                                     ['Target','Estimated'], "boxes", cri)
    addqty('bestGatesetChoiEvalTable', ws.ChoiTable, gsFinal, None, cri, display=("barplot",))
    addqty('bestGatesetDecompTable', ws.GateDecompTable, gsFinal, gsTgt, cri) #TEST
    addqty('bestGatesetEvalTable', ws.GateEigenvalueTable, gsFinal, gsTgt, cri,
           display=('evals','target','absdiff-evals','infdiff-evals','log-evals','absdiff-log-evals'),
           virtual_gates=germs)
    addqty('bestGatesetRelEvalTable', ws.GateEigenvalueTable, gsFinal, gsTgt, cri, display=('rel','log-rel'))
    addqty('bestGatesetVsTargetTable', ws.GatesetVsTargetTable, gsFinal, gsTgt, cliffcomp, cri)
    addqty('bestGatesVsTargetTable_gv', ws.GatesVsTargetTable, gsFinal, gsTgt, cri, #TEST
                                        display=('inf','trace','diamond','uinf','agi'), virtual_gates=germs)
    addqty('bestGatesVsTargetTable_gi', ws.GatesVsTargetTable, gsFinal, gsTgt, cri, #TEST
                                        display=('giinf','gidm'), virtual_gates=germs)
    addqty('bestGatesVsTargetTable_sum', ws.GatesVsTargetTable, gsFinal, gsTgt, cri,
                                         display=('inf','trace','diamond','uinf','agi','giinf','gidm'))
    addqty('bestGatesetErrGenBoxTable', ws.ErrgenTable, gsFinal, gsTgt, cri, ("errgen","H","S"),
                                                           "boxes", errgen_type)
    addqty('metadataTable', ws.MetadataTable, gsFinal, switchBd.params)
    addqty('softwareEnvTable', ws.SoftwareEnvTable)

    #Ls and Germs specific
    gss = switchBd.gss
    gsL = switchBd.gsL
    gssAllL = switchBd.gssAllL
    addqty('fiducialListTable', ws.GatestringTable, strs,["Prep.","Measure"], commonTitle="Fiducials")
    addqty('prepStrListTable', ws.GatestringTable, prepStrs,"Preparation Fiducials")
    addqty('effectStrListTable', ws.GatestringTable, effectStrs,"Measurement Fiducials")
    addqty('germList2ColTable', ws.GatestringTable, germs, "Germ", nCols=2)
    addqty('progressTable', ws.FitComparisonTable, 
           Ls, gssAllL, switchBd.gsAllL, eff_ds, switchBd.objective, 'L')
    
    # Generate plots
    printer.log("*** Generating plots ***")

    addqty('gramBarPlot', ws.GramMatrixBarPlot, ds,gsTgt,10,strs)
    addqty('progressBarPlot', ws.FitComparisonBarPlot, 
           Ls, gssAllL, switchBd.gsAllL, eff_ds, switchBd.objective, 'L')

    if not brief: 
        addqty('dataScalingColorBoxPlot', ws.ColorBoxPlot, 
            "scaling", switchBd.gssFinal, eff_ds, switchBd.gsFinalIter,
                submatrices=switchBd.scaledSubMxsDict)
    
        #Not pagniated currently... just set to same full plot
        addqty('bestEstimateColorBoxPlotPages', ws.ColorBoxPlot,
            switchBd.objective, gss, eff_ds, gsL,
            linlg_pcntle=float(linlogPercentile) / 100,
            minProbClipForWeighting=switchBd.mpc)
        qtys['bestEstimateColorBoxPlotPages'].set_render_options(click_to_display=True)
        
        addqty('bestEstimateColorScatterPlot', ws.ColorBoxPlot,
            switchBd.objective, gss, eff_ds, gsL,
            linlg_pcntle=float(linlogPercentile) / 100,
            minProbClipForWeighting=switchBd.mpc, scatter=True) #TODO: L-switchboard on summary page?
        qtys['bestEstimateColorScatterPlot'].set_render_options(click_to_display=True)
        #  Fast enough now thanks to scattergl, but webgl render issues so need to delay creation 

    if multidataset:
        #initialize a new "dataset comparison switchboard"
        dscmp_switchBd = ws.Switchboard(
            ["Dataset1","Dataset2"],
            [dataset_labels, dataset_labels],
            ["buttons","buttons"], [0,1]
        )
        dscmp_switchBd.add("dscmp",(0,1))
        dscmp_switchBd.add("dscmp_gss",(0,))

        for d1, dslbl1 in enumerate(dataset_labels):
            dscmp_switchBd.dscmp_gss[d1] = results_dict[dslbl1].gatestring_structs['final']

        dsComp = dict()
        all_dsComps = dict()        
        indices = []
        for i in range(len(dataset_labels)):
            for j in range(len(dataset_labels)):
                indices.append((i, j))
        if comm is not None:
            _, indexDict, _ = _distribute_indices(indices, comm)
            rank = comm.Get_rank()
            for k, v in indexDict.items():
                if v == rank:
                    d1, d2 = k
                    dslbl1 = dataset_labels[d1]
                    dslbl2 = dataset_labels[d2]

                    ds1 = results_dict[dslbl1].dataset
                    ds2 = results_dict[dslbl2].dataset
                    dsComp[(d1, d2)] = _DataComparator(
                        [ds1, ds2], DS_names=[dslbl1, dslbl2])
            dicts = comm.gather(dsComp, root=0)
            if rank == 0:
                for d in dicts:
                    for k, v in d.items():
                        d1, d2 = k
                        dscmp_switchBd.dscmp[d1, d2] = v
                        all_dsComps[(d1,d2)] = v
        else:
            for d1, d2 in indices:
                dslbl1 = dataset_labels[d1]
                dslbl2 = dataset_labels[d2]
                ds1 = results_dict[dslbl1].dataset
                ds2 = results_dict[dslbl2].dataset
                all_dsComps[(d1,d2)] =  _DataComparator([ds1, ds2], DS_names=[dslbl1,dslbl2])                
                dscmp_switchBd.dscmp[d1, d2] = all_dsComps[(d1,d2)]
        
        qtys['dscmpSwitchboard'] = dscmp_switchBd
        addqty('dsComparisonSummary', ws.DatasetComparisonSummaryPlot, dataset_labels, all_dsComps)
        addqty('dsComparisonHistogram', ws.DatasetComparisonHistogramPlot, dscmp_switchBd.dscmp)
        if not brief: 
            addqty('dsComparisonBoxPlot', ws.ColorBoxPlot, 'dscmp', dscmp_switchBd.dscmp_gss,
                   None, None, dscomparator=dscmp_switchBd.dscmp)
        toggles['CompareDatasets'] = True
    else:
        toggles['CompareDatasets'] = False

    if filename is not None:
        if comm is None or comm.Get_rank() == 0:
            # 3) populate template html file => report html file
            printer.log("*** Merging into template file ***")
            #template = "report_dashboard.html"
            template = "general_report"
            _merge_template(qtys, template, filename, auto_open, precision, pdf_links, python_links,
                            connected=connected, toggles=toggles, renderMath=renderMath, verbosity=printer,
                            CSSnames=("pygsti_dataviz.css","pygsti_dashboard.css","pygsti_fonts.css"))
            #SmartCache.global_status(printer)
    else:
        printer.log("*** NOT Merging into template file (filename is None) ***")
        
    return ws


def create_report_notebook(results, filename, title="auto",
                           confidenceLevel=None,    
                           auto_open=False, connected=False, verbosity=0):
    """ TODO: docstring - but just a subset of args for create_general_report"""
    printer = VerbosityPrinter.build_printer(verbosity)
    templatePath = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                 "templates")
    assert(_os.path.splitext(filename)[1] == '.ipynb'), 'Output file extension must be .ipynb'
    outputDir = _os.path.dirname(filename)
    
    #Copy offline directory into position
    if not connected:
        _ws.rsync_offline_dir(outputDir)

    #Save results to file
    basename = _os.path.splitext(_os.path.basename(filename))[0]
    results_file_base = basename + '_results.pkl'
    results_file = _os.path.join(outputDir, results_file_base)
    with open(results_file,'wb') as f:
        _pickle.dump(results, f)

    if title is None or title == "auto":
        autoname = _autotitle.generate_name()
        title = "GST Report for " + autoname
        _warnings.warn( ("You should really specify `title=` when generating reports,"
                         "as this makes it much easier to identify them later on.  "
                         "Since you didn't, pyGSTi will has generated a random one"
                         " for you: '{}'.").format(autoname))

    nb = _Notebook()
    nb.add_markdown('# {title}\n(Created on {date})'.format(
        title=title, date=_time.strftime("%B %d, %Y")))

    nb.add_code("""\
        from __future__ import print_function
        import pickle
        import pygsti""")

    if isinstance(results, dict):
        dsKeys = list(results.keys())
        results = results[dsKeys[0]]
          #Note: `results` is always a single Results obj from here down
          
        nb.add_code("""\
        #Load results dictionary
        with open('{infile}', 'rb') as infile:
            results_dict = pickle.load(infile)
        print("Available dataset keys: ", ', '.join(results_dict.keys()))\
        """.format(infile = results_file_base))

        nb.add_code("""\
        #Set which dataset should be used below
        results = results_dict['{dsKey}']
        print("Available estimates: ", ', '.join(results.estimates.keys()))\
        """.format(dsKey=dsKeys[0]))

    else:
        dsKeys = []
        nb.add_code("""\
        #Load results
        with open('{infile}', 'rb') as infile:
            results = pickle.load(infile)
        print("Available estimates: ", ', '.join(results.estimates.keys()))\
        """.format(infile = results_file_base))

    estLabels = list(results.estimates.keys())
    estimate = results.estimates[estLabels[0]]
    nb.add_code("""\
    #Set which estimate is to be used below
    estimate = results.estimates['{estLabel}']
    print("Available gauge opts: ", ', '.join(estimate.goparameters.keys()))\
    """.format(estLabel=estLabels[0]))

    goLabels = list(estimate.goparameters.keys())
    nb.add_code("""\
        gopt      = '{goLabel}'
        ds        = results.dataset

        gssFinal  = results.gatestring_structs['final']
        Ls        = results.gatestring_structs['final'].Ls
        gssPerIter = results.gatestring_structs['iteration'] #ALL_L

        prepStrs = results.gatestring_lists['prep fiducials']
        effectStrs = results.gatestring_lists['effect fiducials']
        germs = results.gatestring_lists['germs']
        strs = (prepStrs, effectStrs)

        params = estimate.parameters
        objective = estimate.parameters['objective']
        if objective == "logl":
            mpc = estimate.parameters['minProbClip']
        else:
            mpc = estimate.parameters['minProbClipForWeighting']
        clifford_compilation = estimate.parameters.get('clifford_compilation',None)

        effective_ds, scale_subMxs = estimate.get_effective_dataset(True)
        scaledSubMxsDict = {{'scaling': scale_subMxs, 'scaling.colormap': "revseq"}}

        gatesets   = estimate.gatesets
        gs         = gatesets[gopt] #FINAL
        gs_final   = gatesets['final iteration estimate'] #ITER
        gs_target  = gatesets['target']
        gsPerIter  = gatesets['iteration estimates']

        goparams = estimate.goparameters[gopt]

        confidenceLevel = {CL}
        if confidenceLevel is None:
            cri = None
        else:
            cri = estimate.get_confidence_region(confidenceLevel, gopt)\
    """.format(goLabel=goLabels[0], CL=confidenceLevel))
            
    nb.add_code("""\
        from pygsti.report import Workspace
        ws = Workspace()
        ws.init_notebook_mode(connected={conn}, autodisplay=True)\
        """.format(conn=str(connected)))
    
    nb.add_notebook_text_files([
        _os.path.join(templatePath,'summary.txt'),
        _os.path.join(templatePath,'goodness.txt'),
        _os.path.join(templatePath,'gauge_invariant.txt'),
        _os.path.join(templatePath,'gauge_variant.txt')])

    #Insert multi-dataset specific analysis
    if len(dsKeys) > 1:
        nb.add_markdown( ('# Dataset comparisons\n',
                          'This report contains information for more than one data set.',
                          'This page shows comparisons between different data sets.') )
        
        nb.add_code("""\
        dslbl1 = '{dsLbl1}'
        dslbl2 = '{dsLbl2}'
        dscmp_gss = results_dict[dslbl1].gatestring_structs['final']
        ds1 = results_dict[dslbl1].dataset
        ds2 = results_dict[dslbl2].dataset
        dscmp = pygsti.obj.DataComparator([ds1, ds2], DS_names=[dslbl1, dslbl2])
        """.format(dsLbl1=dsKeys[0], dsLbl2=dsKeys[1]))
        nb.add_notebook_text_files([
            _os.path.join(templatePath,'data_comparison.txt')])

    #Add reference material
    nb.add_notebook_text_files([
        _os.path.join(templatePath,'input.txt'),
        _os.path.join(templatePath,'meta.txt')])

    if auto_open:
        port = "auto" if auto_open == True else int(auto_open)
        nb.launch(filename, port=port)
    else:
        nb.save_to(filename)



##Scratch: SAVE!!! this code generates "projected" gatesets which can be sent to
## FitComparisonTable (with the same gss for each) to make a nice comparison plot.
#        gateLabels = list(gateset.gates.keys())  # gate labels
#        basis = gateset.basis
#    
#        if basis.name != targetGateset.basis.name:
#            raise ValueError("Basis mismatch between gateset (%s) and target (%s)!"\
#                                 % (basis.name, targetGateset.basis.name))
#    
#        #Do computation first
#        # Note: set to "full" parameterization so we can set the gates below
#        #  regardless of what to fo parameterization the original gateset had.
#        gsH = gateset.copy(); gsH.set_all_parameterizations("full"); Np_H = 0
#        gsS = gateset.copy(); gsS.set_all_parameterizations("full"); Np_S = 0
#        gsHS = gateset.copy(); gsHS.set_all_parameterizations("full"); Np_HS = 0
#        gsLND = gateset.copy(); gsLND.set_all_parameterizations("full"); Np_LND = 0
#        #gsHSCP = gateset.copy()
#        gsLNDCP = gateset.copy(); gsLNDCP.set_all_parameterizations("full")
#        for gl in gateLabels:
#            gate = gateset.gates[gl]
#            targetGate = targetGateset.gates[gl]
#    
#            errgen = _tools.error_generator(gate, targetGate, genType)
#            hamProj, hamGens = _tools.std_errgen_projections(
#                errgen, "hamiltonian", basis.name, basis, True)
#            stoProj, stoGens = _tools.std_errgen_projections(
#                errgen, "stochastic", basis.name, basis, True)
#            HProj, OProj, HGens, OGens = \
#                _tools.lindblad_errgen_projections(
#                    errgen, basis, basis, basis, normalize=False,
#                    return_generators=True)
#                #Note: return values *can* be None if an empty/None basis is given
#    
#            ham_error_gen = _np.einsum('i,ijk', hamProj, hamGens)
#            sto_error_gen = _np.einsum('i,ijk', stoProj, stoGens)
#            lnd_error_gen = _np.einsum('i,ijk', HProj, HGens) + \
#                _np.einsum('ij,ijkl', OProj, OGens)
#    
#            ham_error_gen = _tools.change_basis(ham_error_gen,"std",basis)
#            sto_error_gen = _tools.change_basis(sto_error_gen,"std",basis)
#            lnd_error_gen = _tools.change_basis(lnd_error_gen,"std",basis)
#    
#            gsH.gates[gl]  = _tools.gate_from_error_generator(
#                ham_error_gen, targetGate, genType)
#            gsS.gates[gl]  = _tools.gate_from_error_generator(
#                sto_error_gen, targetGate, genType)
#            gsHS.gates[gl] = _tools.gate_from_error_generator(
#                ham_error_gen+sto_error_gen, targetGate, genType)
#            gsLND.gates[gl] = _tools.gate_from_error_generator(
#                lnd_error_gen, targetGate, genType)
#
#            #CPTP projection
#    
#            #Removed attempt to contract H+S to CPTP by removing positive stochastic projections,
#            # but this doesn't always return the gate to being CPTP (maybe b/c of normalization)...
#            #sto_error_gen_cp = _np.einsum('i,ijk', stoProj.clip(None,0), stoGens) #only negative stochastic projections OK
#            #sto_error_gen_cp = _tools.std_to_pp(sto_error_gen_cp)
#            #gsHSCP.gates[gl] = _tools.gate_from_error_generator(
#            #    ham_error_gen, targetGate, genType) #+sto_error_gen_cp
#    
#            evals,U = _np.linalg.eig(OProj)
#            pos_evals = evals.clip(0,1e100) #clip negative eigenvalues to 0
#            OProj_cp = _np.dot(U,_np.dot(_np.diag(pos_evals),_np.linalg.inv(U))) #OProj_cp is now a pos-def matrix
#            lnd_error_gen_cp = _np.einsum('i,ijk', HProj, HGens) + \
#                _np.einsum('ij,ijkl', OProj_cp, OGens)
#            lnd_error_gen_cp = _tools.change_basis(lnd_error_gen_cp,"std",basis)
#    
#            gsLNDCP.gates[gl] = _tools.gate_from_error_generator(
#                lnd_error_gen_cp, targetGate, genType)
#    
#            Np_H += len(hamProj)
#            Np_S += len(stoProj)
#            Np_HS += len(hamProj) + len(stoProj)
#            Np_LND += HProj.size + OProj.size
#    
#        #DEBUG!!!
#        #print("DEBUG: BEST sum neg evals = ",_tools.sum_of_negative_choi_evals(gateset))
#        #print("DEBUG: LNDCP sum neg evals = ",_tools.sum_of_negative_choi_evals(gsLNDCP))
#    
#        #Check for CPTP where expected
#        #assert(_tools.sum_of_negative_choi_evals(gsHSCP) < 1e-6)
#        assert(_tools.sum_of_negative_choi_evals(gsLNDCP) < 1e-6)
#
#        # ...
#        gatesets = (gateset, gsHS, gsH, gsS, gsLND, cptpGateset, gsLNDCP, gsHSCPTP)
#        gatesetTyps = ("Full","H + S","H","S","LND","CPTP","LND CPTP","H + S CPTP")
#        Nps = (Nng, Np_HS, Np_H, Np_S, Np_LND, Nng, Np_LND, Np_HS)

    
