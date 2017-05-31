from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Report generation functions. """

import os  as _os
import sys as _sys
import time as _time
import numpy as _np
import collections as _collections
import webbrowser as _webbrowser
import zipfile as _zipfile
from scipy.stats import chi2 as _chi2

from ..objects      import VerbosityPrinter
from ..objects      import DataComparator as _DataComparator
from ..tools        import compattools as _compat
from ..             import tools as _tools
from .              import workspace as _ws


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

        #Process #elsetoggle
        try: k = post_text.index("#elsetoggle") # index in (new) *post_text*
        except ValueError: k = None
        if k is not None: # if-block ends at #else
            if_text = post_text[0:k]
            post_text = preprocess(post_text[k+len("#elsetoggle"):])

        #Process #endtoggle
        try: j = post_text.index("#endtoggle") # index in (new) *post_text*
        except ValueError: j = None
        assert(j is not None), "#iftoggle(%s) without corresponding #endtoggle" % toggleName
        
        if k is None: # if-block ends at #end
            if_text = post_text[0:j]
        else: # if-block already captured; else-block ends at #end
            else_text = post_text[0:j]
        post_text = preprocess(post_text[j+len("#endtoggle"):])
                
        if toggles[toggleName]:
            return pre_text + if_text + post_text
        else:
            return pre_text + else_text + post_text
    
    return preprocess(template)

def _merge_template(qtys, templateFilename, outputFilename, auto_open, precision,
                    CSSnames=("pygsti_dataviz.css","pygsti_report.css","pygsti_fonts.css"),
                    connected=False, toggles=None, verbosity=0):

    printer = VerbosityPrinter.build_printer(verbosity)

    #Copy offline directory into position
    if not connected:
        outputDir = _os.path.dirname(outputFilename)
        _ws.rsync_offline_dir(outputDir)

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

        qtys['katexLIB'] += ('\n<script>'
                'document.addEventListener("DOMContentLoaded", function() {'
                'renderMathInElement(document.body, { delimiters: ['
                '{left: "$$", right: "$$", display: true},'
                '{left: "$", right: "$", display: false},'
                '] } ); });'
                '</script>')
        # removed so parens work:
        # '{left: "\\[", right: "\\]", display: true},'
        # '{left: "\\(", right: "\\)", display: false}'

    if 'plotlyexLIB' not in qtys:
        qtys['plotlyexLIB'] = _ws.insert_resource(
            connected, None, "pygsti_plotly_ex.js")
    
    #Add inline CSS
    if 'CSS' not in qtys:
        qtys['CSS'] = "\n".join( [_ws.insert_resource(
            connected, None, cssFile)
                for cssFile in CSSnames] )

    #Insert qtys into template file
    templateFilename = _os.path.join( _os.path.dirname(_os.path.abspath(__file__)),
                                      "templates", templateFilename )

    template = _read_and_preprocess_template(templateFilename, toggles)

    qtys_html = _collections.defaultdict(lambda x=0: "BLANK")
    for key,val in qtys.items():
        if _compat.isstr(val):
            qtys_html[key] = val

        else:
            #print("DB: rendering ",key)
            if isinstance(val,_ws.WorkspaceTable):
                #supply precision argument
                out = val.render("html", precision=precision, resizable=True, autosize=True)
            elif isinstance(val,_ws.WorkspacePlot):
                out = val.render("html", resizable=True, autosize=True)
            else: #switchboards usually
                out = val.render("html") 

            # Note: out is a dictionary of rendered portions
            qtys_html[key] = "<script>\n%(js)s\n</script>\n\n%(html)s" % out

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
        
def _errgen_formula(errgen_type):
    if errgen_type == "logTiG":
        return "$\hat{G} = G_{\mathrm{target}}e^{\mathbb{L}}$"
    elif errgen_type == "logG-logT":
        return "$\hat{G} = e^{\mathbb{L} + \log G_{\mathrm{target}}}$"
    else:
        return "???"

def _add_new_labels(running_lbls, current_lbls):
    """ 
    Simple routine to add current-labels to a list of
    running-labels without introducing duplicates and
    preserving order as best we can.
    """
    if running_lbls is None:
        return current_lbls
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
                               nmthreshold, comm):
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
        show=[multidataset,multiest,False,False]
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

            NA = ws.NotApplicable()
            effds, scale_subMxs = est.get_effective_dataset(True)
            switchBd.eff_ds[d,i] = effds
            switchBd.scaledSubMxsDict[d,i] = {'scaling': scale_subMxs, 'scaling.colormap': "revseq"}
            switchBd.gsTarget[d,i] = est.gatesets['target']
            switchBd.gsFinalIter[d,i] = est.gatesets['final iteration estimate']
            switchBd.gsFinal[d,i,:] = [ est.gatesets.get(l,NA) for l in gauge_opt_labels ]
            switchBd.gsTargetAndFinal[d,i,:] = \
                        [ [est.gatesets['target'], est.gatesets.get(l,NA)]
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
                            est_confidenceLevel, l, "final", comm=comm)
                    else: switchBd.cri[d,i,il] = NA
                    
    return switchBd, dataset_labels, est_labels, gauge_opt_labels, Ls



def create_single_qubit_report(results, filename, confidenceLevel=None,
                               title="auto", datasetLabel="$\\mathcal{D}$",
                               linlogPercentile=5, errgen_type="logTiG",
                               nmthreshold=50, precision=None, brief=False,
                               comm=None, ws=None, auto_open=False,
                               connected=False, verbosity=0):

    """
    Create a "full" single-qubit GST report.  This report gives a detailed and
    analysis that is intended to be applied to `results` of single-qubit GST.
    The report includes background and explanation text to help the user
    interpret the contained results.

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
       The output filename where the report file(s) will be saved.

    confidenceLevel : int, optional
       If not None, then the confidence level (between 0 and 100) used in
       the computation of confidence regions/intervals. If None, no
       confidence regions or intervals are computed.

    title : string, optional
       The title of the report.  "auto" uses a default title which
       specifyies the label of the dataset as well.

    datasetLabel : string, optional
       A label given to the dataset.

    linlogPercentile : float, optional
        Specifies the colorscale transition point for any logL or chi2 color
        box plots.  The lower `(100 - linlogPercentile)` percentile of the
        expected chi2 distribution is shown in a linear grayscale, and the 
        top `linlogPercentile` is shown on a logarithmic colored scale.

    errgen_type : {"logG-logT", "logTiG"}
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

    brief : boolean, optional
        If True, then an alternate version of the report is created which 
        removes much of the expanatory text and re-orders the content to
        be more useful for to users who are familiar with the report's
        tables and figures.

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

    connected : bool, optional
        Whether output HTML should assume an active internet connection.  If
        True, then the resulting HTML file size will be reduced because it
        will link to web resources (e.g. CDN libraries) instead of embedding
        them.

    verbosity : int, optional
       How much detail to send to stdout.
    

    Returns
    -------
    None
    """
    printer = VerbosityPrinter.build_printer(verbosity, comm=comm)
    if ws is None: ws = _ws.Workspace()

    if title == "auto":
        title = "GST report for %s" % datasetLabel
    
    results_dict = results if isinstance(results, dict) else {"unique": results}
    toggles = _set_toggles(results_dict)
        
    qtys = {} # stores strings to be inserted into report template
    qtys['title'] = title
    qtys['date'] = _time.strftime("%B %d, %Y")
    qtys['confidenceLevel'] = "%d" % \
        confidenceLevel if confidenceLevel is not None else "NOT-SET"
    qtys['linlg_pcntle'] = "%d" % round(linlogPercentile) #to nearest %
    qtys['datasetLabel'] = datasetLabel
    qtys['errorgenformula'] = _errgen_formula(errgen_type)

    # Generate Tables
    printer.log("*** Generating tables ***")

    #Create master switchboard
    switchBd, dataset_labels, est_labels, gauge_opt_labels, Ls = \
            _create_master_switchboard(ws, results_dict,
                                       confidenceLevel, nmthreshold, comm)

    if confidenceLevel is not None:
        #TODO: make plain text fields which update based on switchboards?
        qtys['confidenceIntervalScaleFctr'] = "%.3g" % switchBd.cri[0,0,0].intervalScaling
        qtys['confidenceIntervalNumNonGaugeParams'] = "%d" % switchBd.cri[0,0,0].nNonGaugeParams

    multidataset = bool(len(dataset_labels) > 1)
    multiest = bool(len(est_labels) > 1)
    multiGO = bool(len(gauge_opt_labels) > 1)
    multiL = bool(len(Ls) > 1)

    goView = [multidataset,multiest,multiGO,False]
    maxLView = [multidataset,multiest,False,multiL]
    qtys['topSwitchboard'] = switchBd
    qtys['goSwitchboard1'] = switchBd.view(goView,"v1")
    qtys['goSwitchboard2'] = switchBd.view(goView,"v2")
    qtys['goSwitchboard3'] = switchBd.view(goView,"v3")
    qtys['goSwitchboard4'] = switchBd.view(goView,"v4")
    qtys['goSwitchboard5'] = switchBd.view(goView,"v5")    
    qtys['maxLSwitchboard1'] = switchBd.view(maxLView,"v6")
    qtys['maxLSwitchboard2'] = switchBd.view(maxLView,"v7")

    gsTgt = switchBd.gsTarget    
    ds = switchBd.ds
    eff_ds = switchBd.eff_ds
    prepStrs = switchBd.prepStrs
    effectStrs = switchBd.effectStrs
    germs = switchBd.germs
    strs = switchBd.strs

    #Target/fixed outputs
    qtys['targetSpamTable'] = ws.SpamTable(gsTgt)
    qtys['targetGatesTable'] = ws.GatesTable(gsTgt)
    qtys['datasetOverviewTable'] = ws.DataSetOverviewTable(ds, gsTgt, 10, strs)
    
    gsFinal = switchBd.gsFinal
    cri = switchBd.cri if (confidenceLevel is not None) else None
    qtys['bestGatesetSpamTable'] = ws.SpamTable(gsFinal, None, cri)
    qtys['bestGatesetSpamParametersTable'] = ws.SpamParametersTable(gsFinal, cri)
    qtys['bestGatesetGaugeOptParamsTable'] = ws.GaugeOptParamsTable(switchBd.goparams)
    qtys['bestGatesetGatesTable'] = ws.GatesTable(gsFinal, display_as="numbers", confidenceRegionInfo=cri)
    qtys['bestGatesetChoiTable'] = ws.ChoiTable(gsFinal, None, cri, display=('matrix','eigenvalues'))
    qtys['bestGatesetDecompTable'] = ws.GateDecompTable(gsFinal, cri)
    qtys['bestGatesetRotnAxisTable'] = ws.RotationAxisTable(gsFinal, cri, True)
    qtys['bestGatesetVsTargetTable'] = ws.GatesVsTargetTable(gsFinal, gsTgt, cri)
    qtys['bestGatesetErrorGenTable'] = ws.ErrgenTable(gsFinal, gsTgt, cri, ("errgen",),
                                                      "numbers", errgen_type)
    qtys['metadataTable'] = ws.MetadataTable(gsFinal, switchBd.params)
    qtys['softwareEnvTable'] = ws.SoftwareEnvTable()

    # Ls and germs specific
    gss = switchBd.gss
    gsL = switchBd.gsL
    gssAllL = switchBd.gssAllL
    qtys['fiducialListTable'] = ws.GatestringTable(strs,["Prep.","Measure"], commonTitle="Fiducials")
    qtys['prepStrListTable'] = ws.GatestringTable(prepStrs,"Preparation Fiducials")
    qtys['effectStrListTable'] = ws.GatestringTable(effectStrs,"Measurement Fiducials")
    qtys['germListTable'] = ws.GatestringTable(germs, "Germ")
    qtys['progressTable'] = ws.FitComparisonTable(
        Ls, gssAllL, switchBd.gsAllL, eff_ds, switchBd.objective, 'L')

    # Generate plots
    printer.log("*** Generating plots ***")

    qtys['colorBoxPlotKeyPlot'] = ws.BoxKeyPlot(prepStrs, effectStrs)
    
    qtys['bestEstimateColorBoxPlot'] = ws.ColorBoxPlot(
        switchBd.objective, gss, eff_ds, gsL,
        linlg_pcntle=float(linlogPercentile) / 100,
        minProbClipForWeighting=switchBd.mpc)

    qtys['dataScalingColorBoxPlot'] = ws.ColorBoxPlot(
        "scaling", switchBd.gssFinal, eff_ds, switchBd.gsFinalIter,
        submatrices=switchBd.scaledSubMxsDict)
    
    qtys['invertedBestEstimateColorBoxPlot'] = ws.ColorBoxPlot(
        switchBd.objective, gss, eff_ds, gsL, 
        linlg_pcntle=float(linlogPercentile) / 100,
        minProbClipForWeighting=switchBd.mpc, invert=True)

    if multidataset:
        #initialize a new "dataset comparison switchboard"
        dscmp_switchBd = ws.Switchboard(
            ["Dataset1","Dataset2"],
            [dataset_labels, dataset_labels],
            ["buttons","buttons"], [0,1]
        )
        dscmp_switchBd.add("dscmp",(0,1))
        dscmp_switchBd.add("dscmp_gss",(0,))

        for d1,dslbl1 in enumerate(dataset_labels):
            dscmp_switchBd.dscmp_gss[d1] = results_dict[dslbl1].gatestring_structs['final']
            # Note: just use gate string structure from the "first" (Dataset1) data set,
            # which means that if the 2nd results object could have *different* gate strings
            # and the 'dscmp' color box plot must gracefully handle this case.
            
            for d2,dslbl2 in enumerate(dataset_labels):
                ds1 = results_dict[dslbl1].dataset
                ds2 = results_dict[dslbl2].dataset
                dscmp_switchBd.dscmp[d1,d2] = _DataComparator(
                    [ds1,ds2], DS_names=[dslbl1,dslbl2])

        qtys['dscmpSwitchboard'] = dscmp_switchBd
        qtys['dsComparisonHistogram'] = ws.DatasetComparisonPlot(dscmp_switchBd.dscmp)
        qtys['dsComparisonBoxPlot'] = ws.ColorBoxPlot('dscmp', dscmp_switchBd.dscmp_gss,
                                                      None, None, dscomparator=dscmp_switchBd.dscmp)
        toggles['CompareDatasets'] = True
    else:
        toggles['CompareDatasets'] = False
    
    # Populate template latex file => report latex file
    printer.log("*** Merging into template file ***")
    #print("DB inserting choi:\n",qtys['bestGatesetChoiTable'].render("html"))
    #print("DB inserting decomp:\n",qtys['bestGatesetDecompTable'].render("html"))
    template = "report_singlequbit_brief.html" if brief else "report_singlequbit.html"
    _merge_template(qtys, template, filename, auto_open, precision,
                    connected=connected, toggles=toggles, verbosity=printer)

    

def create_general_report(results, filename, confidenceLevel=None,
                          title="auto", datasetLabel="$\\mathcal{D}$",
                          linlogPercentile=5, errgen_type="logTiG",
                          nmthreshold=50, precision=None, brief=False,
                          comm=None, ws=None, auto_open=False,
                          connected=False, verbosity=0):
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
       The output filename where the report file(s) will be saved.

    confidenceLevel : int, optional
       If not None, then the confidence level (between 0 and 100) used in
       the computation of confidence regions/intervals. If None, no
       confidence regions or intervals are computed.

    title : string, optional
       The title of the report.  "auto" uses a default title which
       specifyies the label of the dataset as well.

    datasetLabel : string, optional
       A label given to the dataset.

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

    brief : boolean, optional
        If True, then an alternate version of the report is created which 
        removes much of the expanatory text and re-orders the content to
        be more useful for to users who are familiar with the report's
        tables and figures.

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

    connected : bool, optional
        Whether output HTML should assume an active internet connection.  If
        True, then the resulting HTML file size will be reduced because it
        will link to web resources (e.g. CDN libraries) instead of embedding
        them.

    verbosity : int, optional
       How much detail to send to stdout.
    

    Returns
    -------
    None
    """
    printer = VerbosityPrinter.build_printer(verbosity, comm=comm)
    if ws is None: ws = _ws.Workspace()
        
    if title == "auto":
        title = "GST report for %s" % datasetLabel

    results_dict = results if isinstance(results, dict) else {"unique": results}
    toggles = _set_toggles(results_dict)

    qtys = {} # stores strings to be inserted into report template
    qtys['title'] = title
    qtys['date'] = _time.strftime("%B %d, %Y")
    qtys['confidenceLevel'] = "%d" % \
        confidenceLevel if confidenceLevel is not None else "NOT-SET"
    qtys['linlg_pcntle'] = "%d" % round(linlogPercentile) #to nearest %
    qtys['datasetLabel'] = datasetLabel
    qtys['errorgenformula'] = _errgen_formula(errgen_type)

    # Generate Tables
    printer.log("*** Generating tables ***")

    #Create master switchboard
    switchBd, dataset_labels, est_labels, gauge_opt_labels, Ls = \
            _create_master_switchboard(ws, results_dict,
                                       confidenceLevel, nmthreshold, comm)

    if confidenceLevel is not None:
        #TODO: make plain text fields which update based on switchboards?
        qtys['confidenceIntervalScaleFctr'] = "%.3g" % switchBd.cri[0,0,0].intervalScaling
        qtys['confidenceIntervalNumNonGaugeParams'] = "%d" % switchBd.cri[0,0,0].nNonGaugeParams

    multidataset = bool(len(dataset_labels) > 1)
    multiest = bool(len(est_labels) > 1)
    multiGO = bool(len(gauge_opt_labels) > 1)
    multiL = bool(len(Ls) > 1)

    goView = [multidataset,multiest,multiGO,False]
    maxLView = [multidataset,multiest,False,multiL]
    qtys['topSwitchboard'] = switchBd
    qtys['goSwitchboard1'] = switchBd.view(goView,"v1")
    qtys['goSwitchboard2'] = switchBd.view(goView,"v2")
    qtys['goSwitchboard3'] = switchBd.view(goView,"v3")
    qtys['goSwitchboard4'] = switchBd.view(goView,"v4")
    qtys['goSwitchboard5'] = switchBd.view(goView,"v5")    
    qtys['maxLSwitchboard1']  = switchBd.view(maxLView,"v6")
    #qtys['maxLSwitchboard2'] = switchBd.view(maxLView,"v7") #unused

    gsTgt = switchBd.gsTarget
    ds = switchBd.ds
    eff_ds = switchBd.eff_ds
    prepStrs = switchBd.prepStrs
    effectStrs = switchBd.effectStrs
    germs = switchBd.germs
    strs = switchBd.strs

    qtys['targetSpamBriefTable'] = ws.SpamTable(gsTgt, None, includeHSVec=False)
    qtys['targetGatesBoxTable'] = ws.GatesTable(gsTgt, display_as="boxes")
    qtys['datasetOverviewTable'] = ws.DataSetOverviewTable(ds, gsTgt, 10, strs)

    gsFinal = switchBd.gsFinal
    cri = switchBd.cri if (confidenceLevel is not None) else None
    qtys['bestGatesetSpamParametersTable'] = ws.SpamParametersTable(gsFinal, cri)
    qtys['bestGatesetSpamBriefTable'] = ws.SpamTable(switchBd.gsTargetAndFinal,
                                                     ['Target','Estimated'],
                                                     cri, includeHSVec=False)

    qtys['bestGatesetSpamVsTargetTable'] = ws.SpamVsTargetTable(gsFinal, gsTgt, cri)
    qtys['bestGatesetGaugeOptParamsTable'] = ws.GaugeOptParamsTable(switchBd.goparams)
    qtys['bestGatesetGatesBoxTable'] = ws.GatesTable(switchBd.gsTargetAndFinal,
                                                     ['Target','Estimated'], "boxes", cri)
    qtys['bestGatesetChoiEvalTable'] = ws.ChoiTable(gsFinal, None, cri, display=("barplot",))
    qtys['bestGatesetEvalTable'] = ws.GateEigenvalueTable(gsFinal, gsTgt, cri, display=('polar','relpolar'))
    qtys['bestGatesetVsTargetTable'] = ws.GatesVsTargetTable(gsFinal, gsTgt, cri)
    qtys['bestGatesetErrGenBoxTable'] = ws.ErrgenTable(gsFinal, gsTgt, cri, ("errgen","H","S"),
                                                       "boxes", errgen_type)
    qtys['metadataTable'] = ws.MetadataTable(gsFinal, switchBd.params)
    qtys['softwareEnvTable'] = ws.SoftwareEnvTable()

    #Ls and Germs specific
    gss = switchBd.gss
    gsL = switchBd.gsL
    gssAllL = switchBd.gssAllL
    qtys['fiducialListTable'] = ws.GatestringTable(strs,["Prep.","Measure"], commonTitle="Fiducials")
    qtys['prepStrListTable'] = ws.GatestringTable(prepStrs,"Preparation Fiducials")
    qtys['effectStrListTable'] = ws.GatestringTable(effectStrs,"Measurement Fiducials")
    qtys['germList2ColTable'] = ws.GatestringTable(germs, "Germ", nCols=2)
    qtys['progressTable'] = ws.FitComparisonTable(
        Ls, gssAllL, switchBd.gsAllL, eff_ds, switchBd.objective, 'L')
    
    # Generate plots
    printer.log("*** Generating plots ***")
                
    qtys['colorBoxPlotKeyPlot'] = ws.BoxKeyPlot(prepStrs, effectStrs)        
    qtys['bestEstimateSummedColorBoxPlot'] = ws.ColorBoxPlot(
        switchBd.objective, gss, eff_ds, gsL,
        linlg_pcntle=float(linlogPercentile) / 100,
        minProbClipForWeighting=switchBd.mpc, sumUp=True)

    qtys['dataScalingColorBoxPlot'] = ws.ColorBoxPlot(
        "scaling", switchBd.gssFinal, eff_ds, switchBd.gsFinalIter,
        submatrices=switchBd.scaledSubMxsDict)
    
    #Not pagniated currently... just set to same full plot
    qtys['bestEstimateColorBoxPlotPages'] = ws.ColorBoxPlot(
        switchBd.objective, gss, eff_ds, gsL,
        linlg_pcntle=float(linlogPercentile) / 100,
        minProbClipForWeighting=switchBd.mpc)

    if multidataset:
        #initialize a new "dataset comparison switchboard"
        dscmp_switchBd = ws.Switchboard(
            ["Dataset1","Dataset2"],
            [dataset_labels, dataset_labels],
            ["buttons","buttons"], [0,1]
        )
        dscmp_switchBd.add("dscmp",(0,1))
        dscmp_switchBd.add("dscmp_gss",(0,))

        for d1,dslbl1 in enumerate(dataset_labels):
            dscmp_switchBd.dscmp_gss[d1] = results_dict[dslbl1].gatestring_structs['final']
            # Note: just use gate string structure from the "first" (Dataset1) data set,
            # which means that if the 2nd results object could have *different* gate strings
            # and the 'dscmp' color box plot must gracefully handle this case.
            
            for d2,dslbl2 in enumerate(dataset_labels):
                ds1 = results_dict[dslbl1].dataset
                ds2 = results_dict[dslbl2].dataset
                dscmp_switchBd.dscmp[d1,d2] = _DataComparator(
                    [ds1,ds2], DS_names=[dslbl1,dslbl2])

        qtys['dscmpSwitchboard'] = dscmp_switchBd
        qtys['dsComparisonHistogram'] = ws.DatasetComparisonPlot(dscmp_switchBd.dscmp)
        qtys['dsComparisonBoxPlot'] = ws.ColorBoxPlot('dscmp', dscmp_switchBd.dscmp_gss,
                                                      None, None, dscomparator=dscmp_switchBd.dscmp)
        toggles['CompareDatasets'] = True
    else:
        toggles['CompareDatasets'] = False

    # 3) populate template html file => report html file
    printer.log("*** Merging into template file ***")
    template = "report_general_brief.html" if brief else "report_general.html"
    _merge_template(qtys, template, filename, auto_open, precision,
                    connected=connected, toggles=toggles, verbosity=printer)



##Scratch: SAVE!!! this code generates "projected" gatesets which can be sent to
## FitComparisonTable (with the same gss for each) to make a nice comparison plot.
#        gateLabels = list(gateset.gates.keys())  # gate labels
#        basisNm = gateset.get_basis_name()
#        basisDims = gateset.get_basis_dimension()
#    
#        if basisNm != targetGateset.get_basis_name():
#            raise ValueError("Basis mismatch between gateset (%s) and target (%s)!"\
#                                 % (basisNm, targetGateset.get_basis_name()))
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
#                errgen, "hamiltonian", basisNm, basisNm, True)
#            stoProj, stoGens = _tools.std_errgen_projections(
#                errgen, "stochastic", basisNm, basisNm, True)
#            HProj, OProj, HGens, OGens = \
#                _tools.lindblad_errgen_projections(
#                    errgen, basisNm, basisNm, basisNm, normalize=False,
#                    return_generators=True)
#                #Note: return values *can* be None if an empty/None basis is given
#    
#            ham_error_gen = _np.einsum('i,ijk', hamProj, hamGens)
#            sto_error_gen = _np.einsum('i,ijk', stoProj, stoGens)
#            lnd_error_gen = _np.einsum('i,ijk', HProj, HGens) + \
#                _np.einsum('ij,ijkl', OProj, OGens)
#    
#            ham_error_gen = _tools.change_basis(ham_error_gen,"std",basisNm)
#            sto_error_gen = _tools.change_basis(sto_error_gen,"std",basisNm)
#            lnd_error_gen = _tools.change_basis(lnd_error_gen,"std",basisNm)
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
#            lnd_error_gen_cp = _tools.change_basis(lnd_error_gen_cp,"std",basisNm)
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

    
