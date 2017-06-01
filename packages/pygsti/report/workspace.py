from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines the Workspace class and supporting functionality."""

import itertools as _itertools
import collections as _collections
import os as _os
import shutil as _shutil
import numpy as _np
import uuid as _uuid
import random as _random
import inspect as _inspect
import sys as _sys
import hashlib as _hashlib

from ..tools import compattools as _compat

from . import plotly_plot_ex as _plotly_ex
#from IPython.display import clear_output as _clear_output

_PYGSTI_WORKSPACE_INITIALIZED = False


def digest(obj):
    """Returns an MD5 digest of an arbitary Python object, `obj`."""
    if _sys.version_info > (3, 0): # Python3?
        longT = int      # define long and unicode
        unicodeT = str   #  types to mimic Python2
    else:
        longT = long
        unicodeT = unicode

    # a function to recursively serialize 'v' into an md5 object
    def add(md5, v):
        """Add `v` to the hash, recursively if needed."""
        md5.update(str(type(v)).encode('utf-8'))
        if isinstance(v, bytes):
            md5.update(v)  #can add bytes directly
        elif isinstance(v, float) or _compat.isstr(v) or _compat.isint(v):
            md5.update(str(v).encode('utf-8')) #need to encode strings
        elif isinstance(v, _np.ndarray):
            md5.update(v.tostring()) # numpy gives us bytes
        elif isinstance(v, (tuple, list)):
            for el in v:  add(md5,el)
        elif isinstance(v, dict):
            keys = list(v.keys())
            for k in sorted(keys):
                add(md5,k)
                add(md5,v[k])
        elif isinstance(v, SwitchValue):
            md5.update(v.base.tostring()) #don't recurse to parent switchboard
        else:
            #print("Encoding type: ",str(type(v)))
            attribs = list(sorted(dir(v)))
            for k in attribs:
                if k.startswith('__'): continue
                a = getattr(v, k)
                if _inspect.isroutine(a): continue
                add(md5,k)
                add(md5,a)
        return

    M = _hashlib.md5()
    add(M, obj)
    return M.digest() #return the MD5 digest

def _is_hashable(x):
    try:
        dct = { x: 0 }
    except TypeError:
        return False
    return True


def call_key(fn, args):
    """ 
    Returns a hashable key for caching the result of a function call.

    Parameters
    ----------
    fn : function
       The function itself

    args : list or tuple
       The function's arguments.

    Returns
    -------
    tuple
    """
    if hasattr(fn,"__self__"):
        # hash on ClassName.methodName to avoid collisions, e.g. w/ "_create"
        fnName = fn.__self__.__class__.__name__ + "." + fn.__name__
    else:
        fnName = fn.__name__
    return (fnName,) + tuple(map(digest,args))


def randomID():
    """ Returns a random DOM ID """
    return str(int(10000*_random.random()))
    #return str(_uuid.uuid4().hex) #alternative

def read_contents(filename):
    contents = None
    with open(filename) as f:
        contents = f.read()
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


class Workspace(object):
    """
    Central to data analysis, Workspace objects facilitate the building
    of reports and dashboards.  In particular, they serve as a:

    - factory for tables, plots, and other types of output
    - cache manager to optimize the construction of such output
    - serialization manager for saving and loading analysis variables

    Workspace objects are typically used either 1) within an ipython 
    notebook to interactively build a report/dashboard, or 2) within
    a script to build a hardcoded ("fixed") report/dashboard.
    """

    def __init__(self):
        """
        Initialize a Workspace object.
        """

        self.outputObjs = {} #cache of WorkspaceOutput objects (hashable by call_keys)
        self.compCache = {}  # cache of computation function results (hashable by call_keys)
        self._register_components(False)


    def _makefactory(self,cls,autodisplay):

        PY3 = bool(_sys.version_info > (3, 0))

        #Manipulate argument list of cls.__init__
        argspec = _inspect.getargspec(cls.__init__)
        argnames = argspec[0]
        assert(argnames[0] == 'self' and argnames[1] == 'ws'), \
            "__init__ must begin with (self, ws, ...)"
        factoryfn_argnames = argnames[2:] #strip off self & ws args
        newargspec = (factoryfn_argnames,) + argspec[1:]

        #Define a new factory function with appropriate signature
        signature = _inspect.formatargspec(
            formatvalue=lambda val: "", *newargspec)
        signature = signature[1:-1] #strip off parenthesis from ends of "(signature)"
        
        if autodisplay:
            factory_func_def = (
                    'def factoryfn(%(signature)s):\n' 
                    '    ret = cls(self, %(signature)s); ret.display(); return ret' % 
                    {'signature':signature } )
        else:
            factory_func_def = (
                    'def factoryfn(%(signature)s):\n' 
                    '    return cls(self, %(signature)s)' % 
                    {'signature':signature } )

        #print("FACTORY FN DEF = \n",new_func)
        exec_globals = {'cls' : cls, 'self': self}
        if _sys.version_info > (3, 0):
            exec(factory_func_def, exec_globals) #Python 3
        else:
            exec("""exec factory_func_def in exec_globals""") #Python 2
        factoryfn = exec_globals['factoryfn']

        #Copy cls.__init__ info over to factory function
        factoryfn.__name__   = cls.__init__.__name__
        factoryfn.__doc__    = cls.__init__.__doc__
        factoryfn.__module__ = cls.__init__.__module__
        factoryfn.__dict__   = cls.__init__.__dict__            
        if PY3:
            factoryfn.__defaults__ = cls.__init__.__defaults__
        else:
            factoryfn.func_defaults = cls.__init__.func_defaults
            
        return factoryfn


    def _register_components(self, autodisplay):        
        
        # "register" components
        from . import workspacetables as _wt
        from . import workspaceplots as _wp
        makefactory = lambda cls: self._makefactory(cls,autodisplay)

        self.Switchboard = makefactory(Switchboard)
        self.NotApplicable = makefactory(NotApplicable)

        #Tables
          # Gate sequences
        self.GatestringTable = makefactory(_wt.GatestringTable)
        
          # Spam & Gates
        self.SpamTable = makefactory(_wt.SpamTable)
        self.SpamParametersTable = makefactory(_wt.SpamParametersTable)
        self.GatesTable= makefactory(_wt.GatesTable)
        self.ChoiTable = makefactory(_wt.ChoiTable)

          # Spam & Gates vs. a target
        self.SpamVsTargetTable = makefactory(_wt.SpamVsTargetTable)
        self.GatesVsTargetTable = makefactory(_wt.GatesVsTargetTable)
        self.GatesSingleMetricTable = makefactory(_wt.GatesSingleMetricTable)
        self.GateEigenvalueTable = makefactory(_wt.GateEigenvalueTable)
        self.ErrgenTable = makefactory(_wt.ErrgenTable)
        self.StandardErrgenTable = makefactory(_wt.StandardErrgenTable)

          # Specific to 1Q gates
        self.GateDecompTable = makefactory(_wt.GateDecompTable)
        self.RotationAxisTable = makefactory(_wt.RotationAxisTable)
        self.RotationAxisVsTargetTable = makefactory(_wt.RotationAxisVsTargetTable)

          # goodness of fit
        self.FitComparisonTable = makefactory(_wt.FitComparisonTable)

          #Specifically designed for reports
        self.BlankTable = makefactory(_wt.BlankTable)
        self.DataSetOverviewTable = makefactory(_wt.DataSetOverviewTable)
        self.GaugeOptParamsTable = makefactory(_wt.GaugeOptParamsTable)
        self.MetadataTable = makefactory(_wt.MetadataTable)
        self.SoftwareEnvTable = makefactory(_wt.SoftwareEnvTable)

        #Plots
        self.ColorBoxPlot = makefactory(_wp.ColorBoxPlot)
        self.BoxKeyPlot = makefactory(_wp.BoxKeyPlot)
        self.GateMatrixPlot = makefactory(_wp.GateMatrixPlot)
        self.PolarEigenvaluePlot = makefactory(_wp.PolarEigenvaluePlot)
        self.ProjectionsBoxPlot = makefactory(_wp.ProjectionsBoxPlot)
        self.ChoiEigenvalueBarPlot = makefactory(_wp.ChoiEigenvalueBarPlot)
        self.DatasetComparisonPlot = makefactory(_wp.DatasetComparisonPlot)
        self.RandomizedBenchmarkingPlot = makefactory(_wp.RandomizedBenchmarkingPlot)

        
    def init_notebook_mode(self, connected=False, autodisplay=False):
        """
        Initialize this Workspace for use in an iPython notebook environment.

        This function should be called prior to using the Workspace when
        working within an iPython notebook.

        Parameters
        ----------
        connected : bool (optional)
            Whether to assume you are connected to the internet.  If you are,
            then setting this to `True` allows initialization to rely on web-
            hosted resources which will reduce the overall size of your
            notebook.

        autodisplay : bool (optional)
            Whether to automatically display workspace objects after they are
            created.

        Returns
        -------
        None
        """
        try:
            from IPython.core.display import display as _display
            from IPython.core.display import HTML as _HTML
        except ImportError:
            raise ImportError('Only run `init_notebook_mode` from inside an IPython Notebook.')

        global _PYGSTI_WORKSPACE_INITIALIZED

        script = ""
        
        if not connected:
            rsync_offline_dir(_os.getcwd())

        #If offline, add JS to head that will load local requireJS and/or
        # jQuery if needed (jupyter-exported html files always use CDN
        # for these).
        if not connected:
            script += "<script src='offline/jupyterlibload.js'></script>\n"
        
        #Load our custom plotly extension functions            
        script += insert_resource(connected,None,"pygsti_plotly_ex.js")

        # Load style sheets for displaying tables
        script += insert_resource(connected,None,"pygsti_dataviz.css")

        #jQueryUI_CSS = "https://code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css"
        jQueryUI_CSS = "https://code.jquery.com/ui/1.12.1/themes/smoothness/jquery-ui.css"
        script += insert_resource(connected,jQueryUI_CSS,"smoothness-jquery-ui.css")

        #To fix the UI tooltips within Jupyter (b/c they use an old/custom JQueryUI css file)
        if connected:
            imgURL = "https://code.jquery.com/ui/1.12.1/themes/smoothness/images/ui-icons_222222_256x240.png"
        else:
            imgURL = "offline/images/ui-icons_222222_256x240.png"
        script += "<style>\n" + \
                  ".tooltipbuttons .ui-button { padding: 0; border: 0; background: transparent; }\n" + \
                  ".tooltipbuttons .ui-icon { background-image: url(\"%s\"); margin-top: 0; }\n" % imgURL + \
                  "</style>"
        
        # Note: within a jupyter notebook, the requireJS base path appears
        # to be "/static", so just setting the path to "offline/myfile"
        # would attempt to load "/static/offline/myfile.js" which points
        # somewhere like .../site-packages/notebook/static/offline/myfile".
        # So:
        # - when in a notebook, the path needs to be "../files" followed
        # by the notebook's path, which we can obtain via the notebook JS
        # object.
        # - when *not* in a notebook, the requireJS base defaults to the
        # current page, so just using "offline/myfile" works fine then.

        #Tell require.js where jQueryUI and Katex are
        if connected:
            reqscript = (
                "<script>"
                "console.log('ONLINE - using CDN paths');"
                "requirejs.config({{ "
                "   paths: {{ 'jquery-UI': ['{jqueryui}'],"
                "             'katex': ['{katex}'],"
                "             'autorender': ['{auto}'] }},"
                "}});"
                "require(['jquery', 'jquery-UI'],function($,ui) {{"
                "  window.jQueryUI=ui; console.log('jquery-UI loaded'); }});"
                "require(['katex', 'autorender'],function(katex,auto) {{"
                "  window.katex=katex; console.log('Katex loaded'); }});"
                "</script>"
            ).format(jqueryui="https://code.jquery.com/ui/1.12.1/jquery-ui.min",
                     katex="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.7.1/katex.min.js",
                     auto="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.7.1/contrib/auto-render.min.js")

        else:
            reqscript = (
                "<script>"
                "var pth;"
                "if(typeof IPython !== 'undefined') {{"
                "  var nb = IPython.notebook;"
                "  var relpath = nb.notebook_path.substr(0, nb.notebook_path.lastIndexOf('/') + 1 );"
                "  jqueryui_pth = '../files' + nb.base_url + relpath + '{jqueryui}';"
                "  katex_pth = '../files' + nb.base_url + relpath + '{katex}';"
                "  auto_pth = '../files' + nb.base_url + relpath + '{auto}';"
                "  console.log('IPYTHON DETECTED - using path ' + jqueryui_pth);"
                "}}"
                "else {{"
                "  jqueryui_pth = '{jqueryui}';"
                "  katex_pth = '{katex}';"
                "  auto_pth = '{auto}';"
                "  console.log('NO IPYTHON DETECTED - using path ' + jqueryui_pth);"
                "}}"
                "requirejs.config({{ "
                "   paths: {{ 'jquery-UI': [jqueryui_pth], 'katex': [katex_pth], 'autorender': [auto_pth] }},"
                "}});"
                "require(['jquery', 'jquery-UI'],function($,ui) {{"
                "  window.jQueryUI=ui; console.log('jquery & jquery-UI loaded'); }});"
                "require(['katex', 'autorender'],function(katex,auto) {{"
                "  window.katex=katex; console.log('Katex loaded'); }});"
                "</script>"
            ).format(jqueryui="offline/jquery-ui.min",
                     katex="offline/katex.min",
                     auto="offline/auto-render.min")
        script += reqscript

        
        #Initialize Katex as a fallback if MathJax is unavailable (offline), OR,
        # if MathJax is loaded, wait for plotly to load before rendering SVG text
        # so math shows up properly in plots (maybe we could just use a require
        # statement for this instead of polling?)
        script += insert_resource(
            connected, "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.7.1/katex.min.css",
            "katex.css")

        script += (
            "\n<script>"
            "require(['jquery','katex','autorender'],function($,katex,renderMathInElement) {\n"
            "  var mathjaxTimer = setInterval( function() {\n"
            "    if(document.readyState === 'complete' || document.readyState === 'loaded') {\n"
            "        clearInterval(mathjaxTimer);\n"
            "        if(typeof MathJax === 'undefined') {\n"
            "          console.log('MATHJAX not found - attempting to typeset with Katex');\n"
            "          renderMathInElement(document.body, { delimiters: [\n"
            "             {left: '$$', right: '$$', display: true},\n"
            "             {left: '$', right: '$', display: false},\n"
            "             ] } );\n"
            "        }\n"
            "        else { //Mathjax is alive - wait for plotly\n"
            "            var waitForPlotly = setInterval( function() {\n"
            "              if( typeof(window.Plotly) !== 'undefined' ){\n"
            "                MathJax.Hub.Config({ SVG: { font: 'STIX-Web' }, displayAlign: 'center' });\n"
            "                MathJax.Hub.Queue(['setRenderer', MathJax.Hub, 'SVG']);\n"
            "                clearInterval(waitForPlotly);\n"
            "              }\n"
            "            }, 500 );\n"
            "        }\n"
            "    } //end readyState check \n"
            "  }, 500); //end setInterval \n"
            "});\n"
            '</script>\n')
        
        # Initialize Plotly libraries
        script += _plotly_ex.init_notebook_mode_ex(connected)

        # Perform check to see what has been loaded
        script += (
            "<div id='notebook_load_status' style='font-style:italic;color:blue'>Loading...</div>\n"
            "<script type='text/javascript'>\n"
            "  require(['jquery','jquery-UI','plotly','katex', 'autorender'],\n"
            "     function($,ui,Plotly,katex,auto) {\n"
            "     $(document).ready( function() {\n"
            "       var txt = '';\n"
            "       if( typeof($('#notebook_load_status').resizable) === 'undefined') {\n"
            "         txt += '<span class=\"failmsg\">JQueryUI not loaded correctly</span><br>';\n"
            "       }\n"
            "       if( typeof(Plotly.newPlot) === 'undefined') {\n"
            "         txt += '<span class=\"failmsg\">Plotly not loaded correctly</span><br>';\n"
            "       }\n"
            "       if(txt.length == 0) {\n"
            "         txt += '<span class=\"successmsg\">Notebook Initialization Complete</span>';\n"
            "         if( typeof MathJax !== 'undefined') {\n"
            "           txt += '<span class=\"successmsg2\"> (+MathJax)</span>';\n"
            "         } else {\n"
            "           txt += '<span class=\"successmsg2\"> (+KaTeX)</span>';\n"
            "         }\n"
            "       }\n"
            "       $('#notebook_load_status').html(txt);\n"
            "      }); });\n"
            "</script>\n")

        _display(_HTML(script)) #single call to display keeps things simple
        
        _PYGSTI_WORKSPACE_INITIALIZED = True

        self._register_components(autodisplay)
        return
        

    def switchedCompute(self, fn, *args):
        """
        Computes a function, given its name and arguments, when some or all of
        those arguments are SwitchedValue objects.

        Caching is employed to avoid duplicating function evaluations which have
        the same arguments.  Note that the function itself doesn't need to deal 
        with SwitchValue objects, as this routine resolves such objects into a
        series of function evaluations using the underlying value(s) within the
        SwitchValue.  This routine is primarily used internally for the
        computation of tables and plots.

        if any of the arguments is an instance of `NotApplicable` then `fn`
        is *not* evaluated and the instance is returned as the evaluation
        result.  If multiple arguments are `NotApplicable` instances, the
        first is used as the result.

        Parameters
        ----------
        fn : function
            The function to evaluate

        args : list
            The function's arguments

        Returns
        -------
        fn_values : list
            The function return values for all relevant sets of arguments.
            Denote the length of this list by N.
        switchboards : list
            A list of all the relevant Switchboards used during the function
            evaluation.  Denote the length of this list by M.
        switchboard_switch_indices : list
            A list of length M whose elements are tuples containing the 0-based
            indices of the relevant switches (i.e. those used by any of the
            arguments) for each switchboard (element of `switchboards`).
        switchpos_map : dict
            A dictionary whose keys are switch positions, and whose values are
            integers between 0 and N which index the element of `fn_values`
            corresponding to the given switch positions.  Each 
            "switch positions" key is a tuple of length M whose elements (one
            per switchboard) are tuples of 0-based switch-position indices
            indicating the position of the relevant switches of that
            switchboard.  Thus, 
            `len(key[i]) = len(switchboard_switch_indices[i])`, where `key`
            is a dictionary key.
        """
        # Computation functions get stripped-down *value* args
        # (strip SwitchedValue stuff away)

        switchboards = []
        switchBdInfo = []        
        nonSwitchedArgs = []

        switchpos_map = {}
        storedKeys = {}
        resultValues = []
        
        for i,arg in enumerate(args):
            if isinstance(arg,SwitchValue):
                isb = None
                for j,sb in enumerate(switchboards):
                    if arg.parent is sb:
                        isb = j; break
                else:
                    isb = len(switchboards)
                    switchboards.append(arg.parent)
                    switchBdInfo.append({
                        'argument indices': [], # indices of arguments that are children of this switchboard
                        'value names': [], # names of switchboard value correspond to each argument index
                        'switch indices': set() # indices of the switches that are actually used by the args
                        })
                assert(isb is not None)
                info = switchBdInfo[isb]
                                        
                info['argument indices'].append(i)
                info['value names'].append(arg.name)
                info['switch indices'].update(arg.dependencies)
            else:
                nonSwitchedArgs.append( (i,arg) )

        #print("DB: %d arguments" % len(args))
        #print("DB: found %d switchboards" % len(switchboards))
        #print("DB: switchBdInfo = ", switchBdInfo)
        #print("DB: nonSwitchedArgs = ", nonSwitchedArgs)

        #Gate a list of lists, each list holding all of the relevant switch positions for each board
        switch_positions = []
        for i,sb in enumerate(switchboards):
            info = switchBdInfo[isb]
            info['switch indices'] = list(info['switch indices']) # set -> list so definite order
            
            switch_ranges = [ list(range(len(sb.positionLabels[i])))
                              for i in info['switch indices'] ]
            sb_switch_positions = list(_itertools.product( *switch_ranges ))
              # a list of all possible positions for the switches being
              # used for the *single* board sb
            switch_positions.append( sb_switch_positions )

            
        #loop over all relevant switch configurations (across multiple switchboards)
        for pos in _itertools.product( *switch_positions ):
            # pos[i] gives the switch configuration for the i-th switchboard

            #fill in the arguments for our function call
            argVals = [None]*len(args)

            #first, iterate over all the switchboards
            for sw_pos,sb,info in zip(pos, switchboards, switchBdInfo):
                # sw_pos is a tuple of the info['switch indices'] switch positions for sb
                sis = info['switch indices']
                for nm,j in zip(info["value names"],info["argument indices"]):
                    value_swpos = [ sw_pos[sis.index(k)] for k in sb[nm].dependencies ]
                      # potentially a subset of sw_pos, contains only the switch positions
                      # relevant to the particular SwitchedValue named nm (also the j-th argument)
                    argVals[j] = sb[nm][tuple(value_swpos)] # tuple needed for proper indexing

            #next, fill in the non-switched arguments
            for j,arg in nonSwitchedArgs:
                argVals[j] = arg

            #if any argument is a NotApplicable object, just use it as the
            # result. Otherwise, compute the result or pull it from cache.
            for v in argVals:
                if isinstance(v,NotApplicable):
                    key="NA"; result = v; break
            else:
                # argVals now contains all the arguments, so call the function if
                #  we need to and add result.
                key = call_key(fn, argVals) # cache by call key
                if key not in self.compCache:
                    #print("DB: computing with args = ", argVals)
                    #print("DB: computing with arg types = ", [type(x) for x in argVals])
                    self.compCache[key] = fn(*argVals)
                result = self.compCache[key]

            if key not in storedKeys:
                switchpos_map[pos] = len(resultValues)
                storedKeys[key] = len(resultValues)
                resultValues.append( result )
            else:
                switchpos_map[pos] = storedKeys[key]

        switchboard_switch_indices = [ info['switch indices'] for info in switchBdInfo ]
        return resultValues, switchboards, switchboard_switch_indices, switchpos_map


    def cachedCompute(self, fn, *args):
        """
        Call a function with the given arguments (if needed).

        If the function has already been called with the given arguments then
        the cached return value is returned.  Otherwise, the function is
        evaluated and the result is stored in this Workspace's cache.

        Parameters
        ----------
        fn : function
            The function to evaluate

        args : list
            The function's arguments

        Returns
        -------
        object
            Whether `fn` returns.
        """
        curkey = call_key(fn, args) # cache by call key
        
        if curkey not in self.compCache:
            self.compCache[curkey] = fn(*valArgs)
            
        return self.compCache[curkey]
    
    
class Switchboard(_collections.OrderedDict):
    """
    Encapsulates a render-able set of user-interactive switches
    for controlling visualized output.

    Outwardly a Switchboard looks like a dictionary of SwitchValue
    objects, which in turn look like appropriately sized numpy arrays
    of values for some quantity.  Different switch positions select
    different values and thereby what data is visualized in various
    outputs (e.g. tables and plots).
    """
    
    def __init__(self, ws, switches, positions, types, initial_pos=None,
                 descriptions=None, show="all", ID=None):
        """
        Create a new Switchboard.

        Parameters
        ----------
        switches : list
            A list of switch names.  The length of this list is 
            the number of switches.

        positions : list
            Elements are lists of position labels, one per switch.
            Length must be equal to `len(switches)`.

        types : list of {'buttons','dropdown','slider','numslider'}
            A list of switch-type strings specifying what type of switch
            each switch is.  

            - 'buttons': a set of toggle buttons
            - 'dropdown': a drop-down (or combo-box)
            - 'slider': a horizontal slider (equally spaced items)
            - 'numslider': a horizontal slider (spaced by numeric value)

        initial_pos : list or None (optional)
            A list of 0-based integer indices giving the initial
            position of each of the `len(switches)` switches.  None
            defaults to the first (0-th) position for each switch.

        descriptions : list (optional)
            A string description for each of the `len(switches)` switches.

        show : list (optional)
            A list of boolean (one for each of the `len(switches)` switches)
            indicating whether or not that switch should be rendered.  The
            special values "all" and "none" show all or none of the switches,
            respectively.
            
        ID : str (optional) 
            A DOM identifier to use when rendering this Switchboard to HTML.
            Usually leaving this value as `None` is best, in which case a
            random identifier is created.
        """
        # Note: intentionally leave off ws argument desc. in docstring
        assert(len(switches) == len(positions))
        
        self.ID = randomID() if (ID is None) else ID
        self.ws = ws #Workspace
        self.switchNames = switches
        self.switchTypes = types
        self.switchIDs = ["switchbd%s_%d" % (self.ID,i)
                          for i in range(len(switches))]
        self.positionLabels = positions
        if initial_pos is None:
            self.initialPositions = _np.array([0]*len(switches),'i')
        else:
            assert(len(initial_pos) == len(switches))
            self.initialPositions = _np.array(initial_pos,'i')

        self.descriptions = descriptions
        
        if show == "all":
            self.show = [True]*len(switches)
        elif show == "none":
            self.show = [False]*len(switches)
        else:
            assert(len(show) == len(switches))
            self.show = show

        self.widget = None
        super(Switchboard,self).__init__([])


    def add(self, varname, dependencies):
        """
        Adds a new switched-value to this Switchboard.

        Parameters
        ----------
        varname : str
            A name for the variable being added.  This name will be used to
            access the new variable (as either a dictionary key or as an 
            object member).

        dependencies : list or tuple
            The (0-based) switch-indices specifying which switch positions
            the new variable is dependent on.  For example, if the Switchboard
            has two switches, one for "amplitude" and one for "frequencey", and
            this value is only dependent on frequency, then `dependencies`
            should be set to `(1,)` or `[1]`.
        
        Returns
        -------
        None
        """
        super(Switchboard,self).__setitem__(varname, SwitchValue(self, varname, dependencies))

    def __setitem__(self, key, val):
        raise KeyError("Use add(...) to add an item to this swichboard")

    def render(self, typ="html"):
        """
        Render this Switchboard into the requested format.

        The returned string(s) are intended to be used to embedded a 
        visualization of this object within a larger document.

        Parameters
        ----------
        typ : {"html"}
            The format to render as.  Currently only HTML is supported.

        Returns
        -------
        dict
            A dictionary of strings whose keys indicate which portion of
            the embeddable output the value is.  Keys will vary for different
            `typ`.  For `"html"`, keys are `"html"` and `"js"` for HTML and
            and Javascript code, respectively.
        """
        return self._render_base(typ, None, self.show)


    def _render_base(self, typ, view_suffix, show):
        """ 
        Break off this implementation so SwitchboardViews can use.
        """
        assert(typ == "html"), "Can't render Switchboards as anything but HTML"

        switch_html = []; switch_js = []
        for i,(name,baseID,typ,posLbls,ipos,bShow) in enumerate(
                zip(self.switchNames, self.switchIDs, self.switchTypes,
                    self.positionLabels, self.initialPositions, show)):
            
            ID = (baseID + view_suffix) if view_suffix else baseID
            style = "" if bShow else " style='display: none'"
            
            if typ == "buttons":
                html  = "<div class='switch_container'%s>\n" % style
                html += "<fieldset id='%s'>\n" % ID
                if name:
                    html += "<legend>%s: </legend>\n" % name
                for k,lbl in enumerate(posLbls):
                    checked = " checked='checked'" if k==ipos else ""
                    html += "<label for='%s-%d'>%s</label>\n" % (ID, k,lbl)
                    html += "<input type='radio' name='%s' id='%s-%d' value=%d%s>\n" \
                                          % (ID,ID,k,k,checked)
                html += "</fieldset></div>\n"
                js = "  $('#%s > input').checkboxradio({ icon: false });" % ID

                if view_suffix:
                    js += "\n".join( (
                        "function connect_%s_to_base(){" % ID,
                        "  if( $('#%s').hasClass('initializedSwitch') ) {" % baseID,  # "if base switch is ready"
                        "    $('#%s').on('change', function(event, ui) {" % baseID,
                        "      var v = $(\"#%s > input[name='%s']:checked\").val();" % (baseID,baseID),
                        "      var el = $(\"#%s > input[name='%s'][value=\" + v + \"]\");" % (ID,ID),
                        "      if( el.is(':checked') == false ) { ",
                        "        el.click();",
                        "      }",
                        "    });"
                        "    $('#%s').on('change', function(event, ui) {" % ID,
                        "      var v = $(\"#%s > input[name='%s']:checked\").val();" % (ID,ID),
                        "      var el = $(\"#%s > input[name='%s'][value=\" + v + \"]\");" % (baseID,baseID),
                        "      if( el.is(':checked') == false ) { ",
                        "        el.click();",
                        "      }",
                        "    });",
                        "    $('#%s').trigger('change');" % baseID,
                        "  }",
                        "  else {", #need to wait for base switch
                        "    setTimeout(connect_%s_to_base, 500);" % ID,
                        "    console.log('%s base NOT initialized: Waiting...');" % ID,
                        "  }",
                        "};",
                        "connect_%s_to_base();" % ID #start trying to connect
                    ))


            elif typ == "dropdown":
                html = "<div class='switch_container'%s><fieldset>\n" % style
                if name:
                    html += "<label for='%s'>%s</label>\n" % (ID,name)
                html += "<select name='%s' id='%s'>\n" % (ID,ID)
                for k,lbl in enumerate(posLbls):
                    selected = " selected='selected'" if k==ipos else ""
                    html += "<option value=%d%s>%s</option>\n" % (k,selected,lbl)
                html += "</select>\n</fieldset></div>\n"
                js = "  $('#%s').selectmenu();" % ID

                if view_suffix:
                    js += "\n".join( (
                        "function connect_%s_to_base(){" % ID,
                        "  if( $('#%s').hasClass('initializedSwitch') ) {" % baseID,  # "if base switch is ready"
                        "    $('#%s').on('selectmenuchange', function(event, ui) {" % baseID,
                        "      var v = $('#%s').val();" % baseID,
                        "      var el = $('#%s');" % ID,
                        "      if( el.val() != v ) { ",
                        "        el.val(v).selectmenu('refresh');",
                        "      }",
                        "    });"
                        "    $('#%s').on('selectmenuchange', function(event, ui) {" % ID,
                        "      var v = $('#%s').val();" % ID,
                        "      var el = $('#%s');" % baseID,
                        "      if( el.val() != v ) { ",
                        "        el.val(v).selectmenu('refresh').trigger('selectmenuchange');",
                        "      }",
                        "    });",
                        "    $('#%s').trigger('selectmenuchange');" % baseID,
                        "    console.log('%s connected to base');\n" % ID,
                        "  }",
                        "  else {", #need to wait for base switch
                        "    setTimeout(connect_%s_to_base, 500);" % ID,
                        "    console.log('%s base NOT initialized: Waiting...');" % ID,
                        "  }",
                        "};",
                        "connect_%s_to_base();" % ID #start trying to connect
                    ))
            
            elif typ == "slider" or typ == "numslider":
                
                if typ == "numslider":
                    float_vals = list(map(float,posLbls))
                    m,M = min(float_vals),max(float_vals)
                else:
                    float_vals = list(range(len(posLbls)))
                    m,M = 0, len(posLbls)-1

                ml = max(list(map(len,posLbls)))
                w = 3.0 #1.0*ml

                html  = "<div id='%s-container' class='switch_container'%s>\n" \
                                  % (ID,style)
                html += "<fieldset>\n"
                if name:
                    html += "<label for='%s'>%s</label>\n" % (ID,name)
                html += "<div name='%s' id='%s'>\n" % (ID,ID)
                html += "<div id='%s-handle' class='ui-slider-handle'></div>" % ID
                html += "</div>\n</fieldset></div>\n"
                #                    "       $('#%s-container').css({'margin-top':'%fem'});" % (ID,1.7/2),

                js = ""
                if view_suffix is None:
                    js  = "var %s_float_values = [" % ID + \
                                ",".join(map(str,float_vals)) + "];\n"
                    js += "var %s_str_values = [" % ID + \
                                ",".join(["'%s'" % s for s in posLbls]) + "];\n"
                    js += "window.%s_float_values = %s_float_values;\n" % (ID,ID) #ensure declared globally
                    js += "window.%s_str_values = %s_str_values;\n" % (ID,ID) #ensure declared globally

                    js += "\n".join( (
                        "function findNearest_%s(includeLeft, includeRight, value) {" % ID,
                        "  var nearest = null;",
                        "  var diff = null;",
                        "  for (var i = 0; i < %s_float_values.length; i++) {" % ID,
                        "    if ((includeLeft && %s_float_values[i] <= value) ||" % ID,
                        "        (includeRight && %s_float_values[i] >= value)) {" % ID,
                        "      var newDiff = Math.abs(value - %s_float_values[i]);" % ID,
                        "      if (diff == null || newDiff < diff) {",
                        "        nearest = i;",
                        "        diff = newDiff;",
                        "      }",
                        "    }",
                        "  }",
                        "  return nearest;",
                        "}",
                        "window.findNearest_%s = findNearest_%s;\n" % (ID,ID)))
                
                js += "\n".join( (
                    "  $('#%s').slider({" % ID,
                    "     orientation: 'horizontal', range: false,",
                    "     min: %f, max: %f, step: %f," % (m,M,(M-m)/100.0),
                    "     value: %f," % float_vals[ipos],
                    "     create: function() {",
                    "       $('#%s-handle').text('%s');" % (ID,posLbls[ipos]),
                    "       $('#%s-handle').css({'width':'%fem','height':'%fem'});" % (ID,w,1.7),
                    "       $('#%s-handle').css({'margin-left':'%fem','top':'%fem'});" % (ID,-w/2,-1.7/2+0.4),
                    "       $('#%s-handle').css({'text-align':'center','line-height':'1.5em'});" % ID,
                    "       $('#%s').css({'margin-left':'%fem', 'margin-top':'0.4em'});" % (ID,w/2),
                    "     },",
                    "     slide: function(event, ui) {",
                    "        var includeLeft = event.keyCode != $.ui.keyCode.RIGHT;",
                    "        var includeRight = event.keyCode != $.ui.keyCode.LEFT;",
                    "        var iValue = findNearest_%s(includeLeft, includeRight, ui.value);" % baseID,
                    "        if($('#%s').slider('value') != %s_float_values[iValue]) {" % (ID,baseID),
                    "          $('#%s-handle').text(%s_str_values[iValue]);" % (baseID,baseID),
                    "          $('#%s').slider('value', %s_float_values[iValue]);" % (baseID,baseID),
                    "        }"
                    "        return false;"
                    "    },",
                    "  });",
                ))

                if view_suffix:
                    # slide events always change *base* (non-view) slider (see above),
                    # which causes a change event to fire.  Views handle this event
                    # to update their own slider values.
                    js += "\n".join( (
                        "function connect_%s_to_base(){" % ID,
                        "  if( $('#%s').hasClass('initializedSwitch') ) {" % baseID,  # "if base switch is ready"
                        "    $('#%s').on('slidechange', function(event, ui) {" % baseID,
                        "      $('#%s').slider('value', ui.value);" % ID,
                        "      $('#%s-handle').text( $('#%s-handle').text() );" % (ID,baseID),
                        "      });",
                        "    var mock_ui = { value: $('#%s').slider('value') };" % baseID, #b/c handler uses ui.value
                        "    $('#%s').trigger('slidechange', mock_ui);" % baseID,
                        "  }",
                        "  else {", #need to wait for base switch
                        "    setTimeout(connect_%s_to_base, 500);" % ID,
                        "    console.log('%s base NOT initialized: Waiting...');" % ID,
                        "  }",
                        "};",
                        "connect_%s_to_base();" % ID #start trying to connect
                    ))
                
            else:
                raise ValueError("Unknown switch type: %s" % typ)

            js += "$('#%s').addClass('initializedSwitch');\n" % ID

            switch_html.append(html)
            switch_js.append(js)

        html = "\n".join(switch_html)
        js = "$(document).ready(function() {\n" +\
             "\n".join(switch_js) + "\n});"
        return {'html':html, 'js':js}
                

    def get_switch_change_handlerjs(self, switchIndex):
        """
        Returns the Javascript needed to begin an on-change handler
        for a particular switch.

        Parameters
        ----------
        switchIndex : int
            The 0-based index of which switch to get handler JS for.

        Returns
        -------
        str
        """
        ID = self.switchIDs[switchIndex]
        typ = self.switchTypes[switchIndex]
        if typ == "buttons":
            return "$('#%s').on('change', function() {" % ID
        elif typ == "dropdown":
            return "$('#%s').on('selectmenuchange', function() {" % ID
        elif typ == "slider" or typ == "numslider":
            return "$('#%s').on('slidechange', function() {" % ID #only when slider stops
            #return "$('#%s').on('slide', function() {" % ID # continuous on
             # mouse move - but doesn't respond correctly to arrows, so seems
             # better to use 'slidechange'
        else:
            raise ValueError("Unknown switch type: %s" % typ)

        
    def get_switch_valuejs(self, switchIndex):
        """
        Returns the Javascript needed to get the value of a particular switch.

        Parameters
        ----------
        switchIndex : int
            The 0-based index of which switch to get value-extracting JS for.

        Returns
        -------
        str
        """
        ID = self.switchIDs[switchIndex]
        typ = self.switchTypes[switchIndex]
        if typ == "buttons":
            return "$(\"#%s > input[name='%s']:checked\").val()" % (ID,ID)
        elif typ == "dropdown":
            return "$('#%s').val()" % ID
        elif typ == "slider" or typ == "numslider":
            #return "%s_float_values.indexOf($('#%s').slider('option', 'value'))" % (ID,ID)
            return "findNearest_%s(true,true,$('#%s').slider('option', 'value'))" % (ID,ID)
        else:
            raise ValueError("Unknown switch type: %s" % typ)
        
    def display(self):
        """
        Display this switchboard within an iPython notebook.

        Calling this function requires that you are in an
        iPython environment, and really only makes sense 
        within a notebook.

        Returns
        -------
        None
        """
        from IPython.display import display as _display
        from IPython.display import HTML as _HTML

        #if self.widget is None:
        #    self.widget = _widgets.HTMLMath(value="?",
        #                                placeholder='Switch HTML',
        #                                description='Switch HTML',
        #                                disabled=False)
        out = self.render("html")
        content = "<script>\n" + \
                  "require(['jquery','jquery-UI'],function($,ui) {" + \
                  out['js'] + " });</script>" + out['html']
        #self.widget.value = content
        _display(_HTML(content)) #self.widget)

    def view(self, switches="all", idsuffix="auto"):
        """
        Return a view of this Switchboard.
        
        Parameters
        ----------
        switches : list, optional
            The names of the switches to include in this view. The special
            value "all" includes all of the switches in the view.
            Alternatively, this can be an array of boolean values, one
            for each switch.

        idsuffix : str, optional
            A suffix to append to the DOM ID of this switchboard when
            rendering the view.  If "auto", a random suffix is used.

        Returns
        -------
        SwitchboardView
        """
        if switches == "all":
            show = [True]*len(self.switchNames)
        elif all([isinstance(b,bool) for b in switches]):
            assert(len(switches) == len(self.switchNames))
            show = switches
        else:
            show = [False]*len(self.switchNames)
            for nm in switches:
                show[self.switchNames.index(nm)] = True
                
        return SwitchboardView(self, idsuffix, show)

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        return getattr(self.__dict__,attr)


class SwitchboardView(object):
    """
    A duplicate or "view" of an existing switchboard which logically
    represents the *same* set of switches.  Thus, when switches are
    moved on the duplicate board, switches will move on the original
    (and vice versa).
    """

    def __init__(self, switchboard, idsuffix="auto", show="all"):
        """
        Create a new SwitchboardView

        Parameters
        ----------
        switchboard : Switchboard
            The base switch board.

        idsuffix : str, optional
            A suffix to append to the DOM ID of this switchboard
            when rendering the view.  If "auto", a random suffix
            is used.
                
        show : list (optional)
            A list of booleans indicating which switches should be rendered.
            The special values "all" and "none" show all or none of the
            switches, respectively.
        """
        if idsuffix == "auto":
            self.idsuffix = "v" + randomID()
        else:
            self.idsuffix = idsuffix

        if show == "all":
            self.show = [True]*len(switchboard.switchNames)
        elif show == "none":
            self.show = [False]*len(switchboard.switchNames)
        else:
            assert(len(show) == len(switchboard.switchNames))
            self.show = show

        self.switchboard = switchboard

    def render(self, typ="html"):
        """
        Render this Switchboard into the requested format.

        The returned string(s) are intended to be used to embedded a 
        visualization of this object within a larger document.

        Parameters
        ----------
        typ : {"html"}
            The format to render as.  Currently only HTML is supported.

        Returns
        -------
        dict
            A dictionary of strings whose keys indicate which portion of
            the embeddable output the value is.  Keys will vary for different
            `typ`.  For `"html"`, keys are `"html"` and `"js"` for HTML and
            and Javascript code, respectively.
        """
        return self.switchboard._render_base(typ, self.idsuffix, self.show)

    def display(self):
        """
        Display this switchboard within an iPython notebook.

        Calling this function requires that you are in an
        iPython environment, and really only makes sense 
        within a notebook.

        Returns
        -------
        None
        """
        from IPython.display import display as _display
        from IPython.display import HTML as _HTML

        out = self.render("html")
        content = "<script>\n" + \
                  "require(['jquery','jquery-UI'],function($,ui) {" + \
                  out['js'] + " });</script>" + out['html']
        _display(_HTML(content))
        


class SwitchValue(object):
    """
    Encapsulates a "switched value", which is essentially a value (i.e. some
    quantity, usually one used as an argument to visualization functions) that
    is controlled by the switches of a single Switchboard.

    The paradigm is one of a Switchboard being a collection of switches along
    with a dictionary of SwitchValues, whereby each SwitchValue is a mapping
    of switch positions to values.  For efficiency, a SwitchValue need only map
    a "subspace" of the switch positions, that is, the position-space spanned
    by only a subset of the switches.  Which switch-positions are mapped is
    given by the "dependencies" of a SwitchValue.

    SwitchValue behaves much like a numpy array of values in terms of
    element access.
    """
    
    def __init__(self, parent_switchboard, name, dependencies):
        """
        Creates a new SwitchValue.

        Parameters
        ----------
        parent_switchboard : Switchboard
            The switch board this value is associated with.

        name : str
            The name of this value, which is also the key or member
            name used to access this value from its parent `Switchboard`.

        dependencies : iterable
            The 0-based indices identifying which switches this value
            depends upon, and correspondingly, which switch positions
            the different axes of the new `SwitchValue` correspond to.
        """
        self.ws = parent_switchboard.ws #workspace
        self.parent = parent_switchboard
        self.name = name
        self.dependencies = dependencies
        
        shape = [len(self.parent.positionLabels[i]) for i in dependencies]
        self.base = _np.empty(shape, dtype=_np.object)
        index_all = (slice(None,None),)*len(shape)
        self.base[index_all] = NotApplicable(self.ws)

    #Access to underlying ndarray
    def __getitem__( self, key ):
        return self.base.__getitem__(key)

    def __getslice__(self, i,j):
        return self.__getitem__(slice(i,j)) #Called for A[:]

    def __setitem__(self, key, val):
        return self.base.__setitem__(key,val)

    def __getattr__(self, attr):
        #use __dict__ so no chance for recursive __getattr__
        return getattr(self.__dict__['base'],attr)

    def __len__(self):         return len(self.base)
    #Future - arithmetic ops should return a new SwitchValue
    #def __add__(self,x):       return self.base + x
    #def __sub__(self,x):       return self.base - x
    #def __mul__(self,x):       return self.base * x
    #def __truediv__(self, x):  return self.base / x
            
        

class WorkspaceOutput(object):
    """ 
    Base class for all forms of data-visualization within a `Workspace` context.

    WorkspaceOutput sets a common interface for performing data visualization
    using a Workspace.  In particular, `render` is used to create embeddable 
    output in various formats, and `display` is used to show the object within
    an iPython notebook.
    """
    
    def __init__(self, ws):
        """
        Create a new WorkspaceOutput object.  Usually not called directly.
        
        Parameters
        ----------
        ws : Workspace
            The workspace containing the new object.
        """
        self.ws = ws
        #self.widget = None #don't build until 1st display()

        
    # Note: hashing not needed because these objects are not *inputs* to
    # other WorspaceOutput objects or computation functions - these objects
    # are cached using call_key.

    def render(self, typ="html"):
        """
        Renders this object into the specifed format, specifically for
        embedding it within a larger document.

        Parameters
        ----------
        typ : str
            The format to render as.  Currently `"html"` is widely supported
            and `"latex"` is supported for tables.

        Returns
        -------
        dict
            A dictionary of strings whose keys indicate which portion of
            the embeddable output the value is.  Keys will vary for different
            `typ`.  For `"html"`, keys are `"html"` and `"js"` for HTML and
            and Javascript code, respectively.
        """
        raise NotImplementedError("Derived classes must implement their own render()")


    def display(self):
        """
        Display this object within an iPython notebook.
        """
        from IPython.display import display as _display
        from IPython.display import HTML as _HTML

        #import ipywidgets as _widgets
        #if self.widget is None:
        #    self.widget = _widgets.HTMLMath(value="?",
        #                                placeholder='Plot HTML',
        #                                description='Plot HTML',
        #                                disabled=False)
        out = self.render("html", global_requirejs=True) # b/c jupyter uses require.js
        #OLD: content = "<script>\n%s\n</script>\n\n%s" % (js,html)
        content = "<script>\n" + \
                  "require(['jquery','jquery-UI','plotly'],function($,ui,Plotly) {" + \
                  out['js'] + " });</script>" + out['html']

        #self.widget.value = content
        #with open("debug.html","w") as f:
        #    jsincludes = '<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.1.1/jquery.min.js">'
        #    filecontent = "<html><head><script>\n%s\n</script>\n%s\n</head>\n<body> %s </body></html>" % (_get_plotlyjs(),jsincludes,content)
        #    f.write(filecontent)
        #print("DB content:\n",content)
        #_display(self.widget)
        _display(_HTML(content))

    def _render_html(self, ID, div_htmls, div_ids, switchpos_map,
                     switchboards, switchIndices):
        """
        Helper rendering function, which takes care of the (complex)
        common logic which take a series of HTML div blocks corresponding
        to the results of a Workspace.switchedCompute(...) call and 
        builds the HTML and JS necessary for toggling the visibility of
        these divs in response to changes in switch position(s).

        Parameters
        ----------
        ID: str
            The identifier to use when constructing DOM ids.

        div_htmls : list
            A list of html "<div>...</div>" blocks.  This is the content that
            is switched between.

        div_ids : list
            A list giving the DOM ids for the div blocks given by `div_html`.

        switchpos_map : dict
            A dictionary mapping switch positions to div-index.  Keys are switch
            tuples of per-switchboard positions (i.e. a tuple of tuples), giving
            the positions of each switch specified in `switchIndices`.  Values
            are integer indices into `html_divs`.

        switchboards: list
            A list of relevant SwitchBoard objects.

        switchIndices: list
            A list of tuples, one per Switchboard object, giving the relevant
            switch indices (integers) within that Switchboard.

        Returns
        -------
        dict
            A dictionary of strings whose keys indicate which portion of
            the embeddable output the value is.  Keys are `"html"` and `"js"`.
        """

        #build HTML as container div containing one or more plot divs
        # Note: 'display: none' doesn't always work in firefox... (polar plots in ptic)
        html = "<div id='%s' class='pygsti-wsoutput-group'>\n" % ID  # style='display: none' or 'visibility: hidden'
        html += "\n".join(div_htmls) + "\n</div>\n"

        #build javascript to map switch positions to div_ids
        js = "var switchmap_%s = new Array();\n" % ID
        for switchPositions, iDiv in switchpos_map.items():
            #switchPositions is a tuple of tuples of position indices, one tuple per switchboard
            div_id = div_ids[iDiv]
            flatPositions = []
            for singleBoardSwitchPositions in switchPositions:
                flatPositions.extend( singleBoardSwitchPositions )                
            js += "switchmap_%s[ [%s] ] = '%s';\n" % \
                    (ID, ",".join(map(str,flatPositions)), div_id)

        js += "window.switchmap_%s = switchmap_%s;\n" % (ID,ID) #ensure a *global* variable
        js += "\n"


        cnd = " && ".join([ "$('#switchbd%s_%d').hasClass('initializedSwitch')"
                            % (sb.ID,switchIndex)
                            for sb, switchInds in zip(switchboards, switchIndices)
                            for switchIndex in switchInds ])
        if len(cnd) == 0: cnd = "true"

        #define fn to "connect" output object to switchboard, i.e.
        #  register event handlers for relevant switches so output object updates
        js += "function connect_%s_to_switches(){" % ID
        js += "  if(%s) {" % cnd  # "if switches are ready"
        # loop below adds event bindings to the body of this if-block
                
        #build change event listener javascript
        handler_fns_js = []
        for sb, switchInds in zip(switchboards, switchIndices):
            # switchInds is a tuple containing the "used" switch indices of sb
            
            for switchIndex in switchInds:
                #build a handler function to get all of the relevant switch positions,
                # build a (flattened) position array, and perform the lookup.
                fname = "%s_onchange_%s_%d" % (ID,sb.ID,switchIndex)
                handler_js = "function %s() {\n" % fname 
                handler_js += "  var curSwitchPos = new Array();\n"
                for sb2, switchInds2 in zip(switchboards, switchIndices):
                    for switchIndex2 in switchInds2:
                        handler_js += "  curSwitchPos.push(%s);\n" % sb2.get_switch_valuejs(switchIndex2)
                handler_js += "  var idToShow = switchmap_%s[ curSwitchPos ];\n" % ID
                handler_js += "  $( '#%s' ).children().hide();\n" % ID
                handler_js += "  $( '#' + idToShow ).show();\n"
                handler_js += "  $( '#' + idToShow ).parentsUntil('#%s').show();\n" % ID
                handler_js += "}\n"
                handler_fns_js.append(handler_js)

                # part of if-block ensuring switches are ready (i.e. created)
                js += "    " + sb.get_switch_change_handlerjs(switchIndex) + \
                              "%s(); });\n" % fname
                js += "    %s();\n" % fname # call function to update visibility

        # end if-block
        js += "    console.log('Switches initialized: %s handlers set');\n" % ID
        js += "    $( '#%s' ).show()\n" % ID  #visibility updates are done: show parent container
        js += "  }\n" #ends if-block
        js += "  else {\n"  # switches aren't ready - so wait
        js += "    setTimeout(connect_%s_to_switches, 500);\n" % ID
        js += "    console.log('%s switches NOT initialized: Waiting...');\n" % ID
        js += "  }\n"
        js += "};\n" #end of connect function
        
        #on-ready handler starts trying to connect to switches
        js += "$(document).ready(function() {\n" #
        js += "  connect_%s_to_switches();\n" % ID
        js += "});\n\n"
        js += "\n".join(handler_fns_js)

        return {'html':html, 'js':js}


class NotApplicable(WorkspaceOutput):
    """
    Class signifying that an given set of arguments is not applicable
    to a function being evaluated.
    """
    def __init__(self, ws):
        """
        Create a new NotApplicable object.
        """
        super(NotApplicable, self).__init__(ws)

    def render(self, typ="html", ID=None):
        """
        Renders this object into the specifed format, specifically for
        embedding it within a larger document.

        Parameters
        ----------
        typ : str
            The format to render as.  Currently `"html"` is widely supported
            and `"latex"` is supported for tables.

        Returns
        -------
        dict
            A dictionary of strings whose keys indicate which portion of
            the embeddable output the value is.  Keys will vary for different
            `typ`.  For `"html"`, keys are `"html"` and `"js"` for HTML and
            and Javascript code, respectively.
        """
        if ID is None: ID=randomID()
        
        if typ == "html":
            return {'html': "<div id='%s' class='notapplicable'>[NO DATA or N/A]</div>" % ID, 'js':"" }

        elif typ == "latex":
            return {'latex': "Not applicable" }
        else:
            raise ValueError("NotApplicable render type not supported: %s" % typ)

    
class WorkspaceTable(WorkspaceOutput):
    """
    Encapsulates a table within a `Workspace` context.

    A base class which provides the logic required to take a
    single table-generating function and make it into a legitimate
    `WorkspaceOutput` object for using within workspaces.
    """
    
    def __init__(self, ws, fn, *args):
        """
        Create a new WorkspaceTable.  Usually not called directly.

        Parameters
        ----------
        ws : Workspace
            The workspace containing the new object.

        fn : function
            A table-creating function.

        args : various
            The arguments to `fn`.
        """
        super(WorkspaceTable, self).__init__(ws)
        self.tablefn = fn
        self.initargs = args
        self.tables,self.switchboards,self.sbSwitchIndices,self.switchpos_map = \
            self.ws.switchedCompute(self.tablefn, *self.initargs)

        
    def render(self, typ, global_requirejs=False, precision=None,
               resizable=True, autosize=False):
        """
        Renders this table into the specifed format, specifically for
        embedding it within a larger document.

        Parameters
        ----------
        typ : str
            The format to render as.  Currently `"html"` is supported in 
            all cases, and `"latex"` is supported for non-switched tables
            (those which don't depend on any switched values).

        global_requirejs : bool, optional
            Whether the table is going to be embedded in an environment
            with a globally defined RequireJS library.  If True, then
            rendered output will make use of RequireJS.

        precision : int or dict, optional
            The amount of precision to display.  A dictionary with keys
            "polar", "sci", and "normal" can separately specify the 
            precision for complex angles, numbers in scientific notation, and 
            everything else, respectively.  If an integer is given, it this
            same value is taken for all precision types.  If None, then
            `{'normal': 6, 'polar': 3, 'sci': 0}` is used.

        resizable : bool, optional
            Whether or not to place table inside a JQueryUI 
            resizable widget (only applies when `typ == "html"`).

        autosize : bool, optional
            Whether elements within table should be resized when
            the browser window is resized (only applies when
            `typ == "html"`).

        Returns
        -------
        dict
            A dictionary of strings giving the different portions of the
            embeddable output.  For `"html"`, keys are `"html"` and `"js"`.
            For `"latex"`, there is a single key `"latex"`.
        """
        if precision is None:
            precDict = {'normal': 6, 'polar': 3, 'sci': 0}
        elif _compat.isint(precDict):
            precDict = {'normal':precision, 'polar':precision, 'sci':precision}
        else:
            assert('normal' in precDict), "Must at least specify 'normal' precision"
            p = precision['normal']
            precDict = { 'normal': p,
                         'polar': precision.get(['polar'],p),
                         'sci': precision.get(['sci'],p) }
        
        if typ == "html":
            tableID = "table_" + randomID()
            
            divHTML = []
            divIDs = []
            divJS = []
            for i,table in enumerate(self.tables):
                tableDivID = tableID + "_%d" % i
                if isinstance(table,NotApplicable):
                    table_dict = table.render("html",tableDivID)
                else:
                    table_dict = table.render("html", tableID=tableDivID + "_tbl",
                                              tableclass="dataTable",
                                              precision=precDict['normal'],
                                              polarprecision=precDict['polar'],
                                              sciprecision=precDict['sci'],
                                              resizable=resizable, autosize=autosize)
                

                divHTML.append("<div id='%s'>\n%s\n</div>\n" %
                               (tableDivID,table_dict['html']))
                divJS.append(table_dict['js'])
                divIDs.append(tableDivID)
                
            base = self._render_html(tableID, divHTML, divIDs, self.switchpos_map,
                                     self.switchboards, self.sbSwitchIndices)
            ret = { 'html': base['html'], 'js': '' }

            if global_requirejs:
                ret['js'] += "require(['jquery','jquery-UI','plotly'],function($,ui,Plotly) {\n"
            ret['js'] += '  $(document).ready(function() {\n'
            ret['js'] += '\n'.join(divJS) + base['js'] #insert plot handlers above switchboard init JS
            
            if resizable:
                ret['js'] += 'make_wstable_resizable("{tableID}");\n'.format(tableID=tableID)
            if autosize:
                ret['js'] += 'make_wsobj_autosize("{tableID}");\n'.format(tableID=tableID)
            if resizable or autosize:
                ret['js'] += 'trigger_wstable_plot_creation("{tableID}");\n'.format(tableID=tableID)

            ret['js'] += '}); //end on-ready handler\n'
            if global_requirejs:
                ret['js'] += '}); //end require block\n'

            return ret
        else:
            assert(len(self.tables) == 1), \
                "Can only render %s format for a non-switched table" % typ
            return {typ: self.tables[0].render(typ)}


        
class WorkspacePlot(WorkspaceOutput):
    """
    Encapsulates a plot within a `Workspace` context.

    A base class which provides the logic required to take a
    single plot.ly figure-generating function and make it into a
    legitimate `WorkspaceOutput` object for using within workspaces.
    """

    def __init__(self, ws, fn, *args):
        """
        Create a new WorkspaceTable.  Usually not called directly.

        Parameters
        ----------
        ws : Workspace
            The workspace containing the new object.

        fn : function
            A table-creating function.

        args : various
            The arguments to `fn`.
        """
        super(WorkspacePlot, self).__init__(ws)
        self.plotfn = fn
        self.initargs = args
        self.figs, self.switchboards, self.sbSwitchIndices, self.switchpos_map = \
            self.ws.switchedCompute(self.plotfn, *self.initargs)

        
    def render(self, typ="html", global_requirejs=False, resizable=True, autosize=False):
        """
        Renders this plot into the specifed format, specifically for
        embedding it within a larger document.

        Parameters
        ----------
        typ : str
            The format to render as.  Currently only `"html"` is supported.

        global_requirejs : bool, optional
            Whether the table is going to be embedded in an environment
            with a globally defined RequireJS library.  If True, then
            rendered output will make use of RequireJS.

        resizable : bool, optional
            Whether or not to place plot inside a JQueryUI 
            resizable widget (only applies when `typ == "html"`).

        autosize : bool, optional
            Whether this plot should be resized resized when
            the browser window is resized (only applies when
            `typ == "html"`).

        Returns
        -------
        dict
            A dictionary of strings giving the HTML and Javascript portions
            of the embeddable output.  Keys are `"html"` and `"js"`.
        """
        assert(typ == "html"), "Only HTML rendering supported currently"

        plotID = "plot_" + randomID()

        def getPlotlyDivID(html):
            #could make this more robust using lxml or something later...
            iStart = html.index('div id="')
            iEnd = html.index('"', iStart+8)
            return html[iStart+8:iEnd]

        #pick "master" plot, whose resizing dictates the resizing of other plots,
        # as the largest-height plot.
        iMaster = None; maxH = 0;
        for i,fig in enumerate(self.figs):
            if isinstance(fig,NotApplicable): continue
            if 'height' in fig['layout']:
                if fig['layout']['height'] > maxH:
                    iMaster, maxH = i, fig['layout']['height'];
        assert(iMaster is not None)
          #maybe we'll deal with this case in the future by setting 
          # master=None below, but it's unclear whether this will is needed.
        
        divHTML = []
        divIDs = []
        divJS = []
        for i,fig in enumerate(self.figs):
            if isinstance(fig,NotApplicable):
                NAid = randomID()
                fig_dict = fig.render(typ, NAid)
                divIDs.append(NAid)
            else:
                #use auto-sizing (fluid layout)
                fig_dict = _plotly_ex.plot_ex(
                    fig, show_link=False,
                    autosize=autosize, resizable=resizable,
                    lock_aspect_ratio=True, master=True ) # bool(i==iMaster)
                divIDs.append(getPlotlyDivID(fig_dict['html']))
                
            divHTML.append("<div class='relwrap'><div class='abswrap'>%s</div></div>" % fig_dict['html'])
            divJS.append( fig_dict['js'] )
            
        base = self._render_html(plotID, divHTML, divIDs, self.switchpos_map,
                                self.switchboards, self.sbSwitchIndices)        
        ret = { 'html': base['html'], 'js': '' }

        # "handlers only" mode is when plot is embedded in something
        #  larger (e.g. a table) that takes responsibility for putting
        #  the JS in an on-ready handler and initializing and creating
        #  the plots.
        handlersOnly = bool(resizable == "handlers only")

        
        if not handlersOnly:
            if global_requirejs:
                ret['js'] += "require(['jquery','jquery-UI','plotly'],function($,ui,Plotly) {\n"
            ret['js'] += '  $(document).ready(function() {\n'
        ret['js'] += '\n'.join(divJS) + base['js'] #insert plot handlers above switchboard init JS

        if not handlersOnly:
            if resizable: # make a resizable widget
                ret['js'] += 'make_wsplot_resizable("{plotID}");\n'.format(plotID=plotID)
            if autosize: # add window resize handler
                ret['js'] += 'make_wsobj_autosize("{plotID}");\n'.format(plotID=plotID)
            if (autosize or resizable): #trigger init & create of plots
                ret['js'] += 'trigger_wsplot_plot_creation("{plotID}");\n'.format(plotID=plotID)

        if not handlersOnly:
            ret['js'] += '}); //end on-ready handler\n'
            if global_requirejs:
                ret['js'] += '}); //end require block\n'
                        
        return ret
                
                    
