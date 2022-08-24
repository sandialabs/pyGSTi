"""
Defines the Workspace class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import collections as _collections
import inspect as _inspect
import itertools as _itertools
import os as _os
import pickle as _pickle
# import uuid        as _uuid
import random as _random
import shutil as _shutil
import subprocess as _subprocess
from pprint import pprint as _pprint

import numpy as _np

from pygsti.report import merge_helpers as _merge
from pygsti.report import plotly_plot_ex as _plotly_ex
from pygsti import baseobjs as _baseobjs
from pygsti.baseobjs.smartcache import CustomDigestError as _CustomDigestError
from pygsti.baseobjs import _compatibility as _compat

_PYGSTI_WORKSPACE_INITIALIZED = False
VALIDATE_PLOTLY = False  # False increases performance of report rendering; set to True to debug


def in_ipython_notebook():
    """
    Returns true if called from within an IPython/jupyter notebook

    Returns
    -------
    bool
    """
    try:
        # 'ZMQInteractiveShell' in a notebook, 'TerminalInteractiveShell' in IPython REPL, and fails elsewhere.
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except NameError:
        return False


def display_ipynb(content):
    """
    Render HTML content to an IPython notebook cell display

    Parameters
    ----------
    content : str
        HTML content to insert.

    Returns
    -------
    None
    """
    from IPython.core.display import display, HTML
    display(HTML(content))


def enable_plotly_pickling():
    """
    Hacks the plotly python library so that figures may be pickled and un-pickled.

    This hack should be used only temporarily - so all pickling
    and un-pickling should be done between calls to
    :func:`enable_plotly_pickling` and :func:`disable_plotly_pickling`.

    Returns
    -------
    None
    """
    import plotly

    if int(plotly.__version__.split(".")[0]) >= 3:  # plotly version 3 or higher
        BLT = plotly.basedatatypes.BaseLayoutType

        def fix_getattr(self, prop):
            if '_subplotid_props' not in self.__dict__:
                self._subplotid_props = set()
            return self.__saved_getattr__(prop)
        if hasattr(BLT, '__getattr__'):
            BLT.__saved_getattr__ = BLT.__getattr__
            del BLT.__getattr__
        BLT.__getattr__ = fix_getattr

    else:
        def setitem(self, key, value, _raise=True):
            """Sets an item of a dict using the standard dict's  __setitem__
               to restore normal dict behavior"""
            return dict.__setitem__(self, key, value)

        plotlyDictClass = plotly.graph_objs.Figure.__bases__[0]
        if hasattr(plotlyDictClass, '__setitem__'):
            plotlyDictClass.__saved_setitem__ = plotlyDictClass.__setitem__
        if hasattr(plotlyDictClass, '__getattr__'):
            plotlyDictClass.__saved_getattr__ = plotlyDictClass.__getattr__
            del plotlyDictClass.__getattr__
        if hasattr(plotlyDictClass, '__setattr__'):
            plotlyDictClass.__saved_setattr__ = plotlyDictClass.__setattr__
            del plotlyDictClass.__setattr__
        plotlyDictClass.__setitem__ = setitem


def disable_plotly_pickling():
    """
    Reverses the effect of :func:`enable_plotly_pickling`

    Returns
    -------
    None
    """
    import plotly

    if int(plotly.__version__.split(".")[0]) >= 3:  # plotly version 3 or higher
        BLT = plotly.basedatatypes.BaseLayoutType
        if hasattr(BLT, '__saved_getattr__'):
            BLT.__getattr__ = BLT.__saved_getattr__
            del BLT.__saved_getattr__

    else:
        plotlyDictClass = plotly.graph_objs.Figure.__bases__[0]
        if hasattr(plotlyDictClass, '__saved_setitem__'):
            plotlyDictClass.__setitem__ = plotlyDictClass.__saved_setitem__
            del plotlyDictClass.__saved_setitem__
        if hasattr(plotlyDictClass, '__saved_getattr__'):
            plotlyDictClass.__getattr__ = plotlyDictClass.__saved_getattr__
            del plotlyDictClass.__saved_getattr__
        if hasattr(plotlyDictClass, '__saved_setattr__'):
            plotlyDictClass.__setattr__ = plotlyDictClass.__saved_setattr__
            del plotlyDictClass.__saved_setattr__


def ws_custom_digest(md5, v):
    """
    A "digest" function for hashing several special types

    Parameters
    ----------
    md5 : hashlib.HASH
        The MD5 hash object.

    v : object
        the value to add to `md5` (using `md5.update(...)`).

    Returns
    -------
    None
    """
    if isinstance(v, NotApplicable):
        md5.update("NOTAPPLICABLE".encode('utf-8'))
    elif isinstance(v, SwitchValue):
        md5.update(v.base.tostring())  # don't recurse to parent switchboard
    else:
        raise _CustomDigestError()


def random_id():
    """
    Returns a random document-objet-model (DOM) ID

    Returns
    -------
    str
    """
    return str(int(1000000 * _random.random()))
    #return str(_uuid.uuid4().hex) #alternative


class Workspace(object):
    """
    Central to data analysis, Workspace objects facilitate the building of reports and dashboards.

    In particular, they serve as a:

    - factory for tables, plots, and other types of output
    - cache manager to optimize the construction of such output
    - serialization manager for saving and loading analysis variables

    Workspace objects are typically used either 1) within an ipython
    notebook to interactively build a report/dashboard, or 2) within
    a script to build a hardcoded ("fixed") report/dashboard.

    Parameters
    ----------
    cachefile : str, optional
        filename with cached workspace results
    """

    def __init__(self, cachefile=None):
        """
        Initialize a Workspace object.

        Parameters
        ----------
        cachefile : str, optional
            filename with cached workspace results
        """
        self._register_components(False)
        self.smartCache = _baseobjs.SmartCache()
        if cachefile is not None:
            self.load_cache(cachefile)
        self.smartCache.add_digest(ws_custom_digest)

    def save_cache(self, cachefile, show_unpickled=False):
        """
        Save this Workspace's cache to a file.

        Parameters
        ----------
        cachefile : str
            The filename to save the cache to.

        show_unpickled : bool, optional
            Whether to print quantities (keys) of cache that could not be
            saved because they were not pickle-able.

        Returns
        -------
        None
        """
        with open(cachefile, 'wb') as outfile:
            enable_plotly_pickling()
            _pickle.dump(self.smartCache, outfile)
            disable_plotly_pickling()
        if show_unpickled:
            print('Unpickled keys:')
            _pprint(self.smartCache.unpickleable)

    def load_cache(self, cachefile):
        """
        Load this Workspace's cache from `cachefile`.

        Parameters
        ----------
        cachefile : str
            The filename to load the cache from.

        Returns
        -------
        None
        """
        with open(cachefile, 'rb') as infile:
            enable_plotly_pickling()
            oldCache = _pickle.load(infile).cache
            disable_plotly_pickling()
            for v in oldCache.values():
                if isinstance(v, WorkspaceOutput):  # hasattr(v,'ws') == True for plotly dicts (why?)
                    print('Updated {} object to set ws to self'.format(type(v)))
                    v.ws = self
            self.smartCache.cache.update(oldCache)

    def __getstate__(self):
        return {'smartCache': self.smartCache}

    def __setstate__(self, state_dict):
        self._register_components(False)
        self.smartCache = state_dict['smartCache']

    def _makefactory(self, cls, autodisplay):  # , printer=_objs.VerbosityPrinter(1)):
        # XXX this indirection is so wild -- can we please rewrite directly?
        #Manipulate argument list of cls.__init__
        argspec = _inspect.getargspec(cls.__init__)
        argnames = argspec[0]
        assert(argnames[0] == 'self' and argnames[1] == 'ws'), \
            "__init__ must begin with (self, ws, ...)"

        factoryfn_argnames = argnames[2:]  # strip off self & ws args
        newargspec = (factoryfn_argnames,) + argspec[1:]

        #Define a new factory function with appropriate signature
        signature = _inspect.formatargspec(
            formatvalue=lambda val: "", *newargspec)
        signature = signature[1:-1]  # strip off parenthesis from ends of "(signature)"

        if autodisplay:
            factory_func_def = (
                'def factoryfn(%(signature)s):\n'
                '    ret = cls(self, %(signature)s); ret.display(); return ret' %
                {'signature': signature})
        else:
            factory_func_def = (
                'def factoryfn(%(signature)s):\n'
                '    return cls(self, %(signature)s)' %
                {'signature': signature})

        #print("FACTORY FN DEF = \n",new_func)
        exec_globals = {'cls': cls, 'self': self}
        exec(factory_func_def, exec_globals)
        factoryfn = exec_globals['factoryfn']

        #Copy cls.__init__ info over to factory function
        factoryfn.__name__ = cls.__init__.__name__
        factoryfn.__doc__ = cls.__init__.__doc__
        factoryfn.__module__ = cls.__init__.__module__
        factoryfn.__dict__ = cls.__init__.__dict__
        factoryfn.__defaults__ = cls.__init__.__defaults__

        return factoryfn

    def _register_components(self, autodisplay):
        # "register" components
        from . import workspacetables as _wt
        from . import workspaceplots as _wp
        from . import workspacetexts as _wtxt

        def makefactory(cls): return self._makefactory(cls, autodisplay)

        self.Switchboard = makefactory(Switchboard)
        self.NotApplicable = makefactory(NotApplicable)

        #Tables
        # Circuits
        self.CircuitTable = makefactory(_wt.CircuitTable)

        # Spam & Gates
        self.SpamTable = makefactory(_wt.SpamTable)
        self.SpamParametersTable = makefactory(_wt.SpamParametersTable)
        self.GatesTable = makefactory(_wt.GatesTable)
        self.ChoiTable = makefactory(_wt.ChoiTable)

        # Spam & Gates vs. a target
        self.SpamVsTargetTable = makefactory(_wt.SpamVsTargetTable)
        self.ModelVsTargetTable = makefactory(_wt.ModelVsTargetTable)
        self.GatesVsTargetTable = makefactory(_wt.GatesVsTargetTable)
        self.GatesSingleMetricTable = makefactory(_wt.GatesSingleMetricTable)
        self.GateEigenvalueTable = makefactory(_wt.GateEigenvalueTable)
        self.ErrgenTable = makefactory(_wt.ErrgenTable)
        self.GaugeRobustErrgenTable = makefactory(_wt.GaugeRobustErrgenTable)
        self.GaugeRobustModelTable = makefactory(_wt.GaugeRobustModelTable)
        self.GaugeRobustMetricTable = makefactory(_wt.GaugeRobustMetricTable)
        self.NQubitErrgenTable = makefactory(_wt.NQubitErrgenTable)
        self.StandardErrgenTable = makefactory(_wt.StandardErrgenTable)

        # Specific to 1Q gates
        self.GateDecompTable = makefactory(_wt.GateDecompTable)
        self.old_GateDecompTable = makefactory(_wt.OldGateDecompTable)
        self.old_RotationAxisVsTargetTable = makefactory(_wt.OldRotationAxisVsTargetTable)
        self.old_RotationAxisTable = makefactory(_wt.OldRotationAxisTable)

        # goodness of fit
        self.FitComparisonTable = makefactory(_wt.FitComparisonTable)
        self.WildcardBudgetTable = makefactory(_wt.WildcardBudgetTable)
        self.WildcardDiamondDistanceBarPlot = makefactory(_wp.WildcardDiamondDistanceBarPlot)

        #Specifically designed for reports
        self.BlankTable = makefactory(_wt.BlankTable)
        self.DataSetOverviewTable = makefactory(_wt.DataSetOverviewTable)
        self.GaugeOptParamsTable = makefactory(_wt.GaugeOptParamsTable)
        self.MetadataTable = makefactory(_wt.MetadataTable)
        self.SoftwareEnvTable = makefactory(_wt.SoftwareEnvTable)
        self.ProfilerTable = makefactory(_wt.ProfilerTable)
        self.ExampleTable = makefactory(_wt.ExampleTable)

        #Plots
        self.ColorBoxPlot = makefactory(_wp.ColorBoxPlot)
        self.BoxKeyPlot = makefactory(_wp.BoxKeyPlot)
        self.MatrixPlot = makefactory(_wp.MatrixPlot)
        self.GateMatrixPlot = makefactory(_wp.GateMatrixPlot)
        self.PolarEigenvaluePlot = makefactory(_wp.PolarEigenvaluePlot)
        self.ProjectionsBoxPlot = makefactory(_wp.ProjectionsBoxPlot)
        self.ChoiEigenvalueBarPlot = makefactory(_wp.ChoiEigenvalueBarPlot)
        self.GramMatrixBarPlot = makefactory(_wp.GramMatrixBarPlot)
        self.FitComparisonBarPlot = makefactory(_wp.FitComparisonBarPlot)
        self.FitComparisonBoxPlot = makefactory(_wp.FitComparisonBoxPlot)
        self.DatasetComparisonHistogramPlot = makefactory(_wp.DatasetComparisonHistogramPlot)
        self.DatasetComparisonSummaryPlot = makefactory(_wp.DatasetComparisonSummaryPlot)
        self.RandomizedBenchmarkingPlot = makefactory(_wp.RandomizedBenchmarkingPlot)

        #Text blocks
        self.StdoutText = makefactory(_wtxt.StdoutText)

        #Extras
        from ..extras import idletomography as _idt
        self.IdleTomographyIntrinsicErrorsTable = makefactory(_idt.IdleTomographyIntrinsicErrorsTable)
        self.IdleTomographyObservedRatePlot = makefactory(_idt.IdleTomographyObservedRatePlot)
        self.IdleTomographyObservedRatesTable = makefactory(_idt.IdleTomographyObservedRatesTable)
        self.IdleTomographyObservedRatesForIntrinsicRateTable = makefactory(
            _idt.IdleTomographyObservedRatesForIntrinsicRateTable)

        from ..extras.drift import driftreport as _driftrpt
        self.DriftSummaryTable = makefactory(_driftrpt.DriftSummaryTable)
        self.DriftDetailsTable = makefactory(_driftrpt.DriftDetailsTable)
        self.PowerSpectraPlot = makefactory(_driftrpt.PowerSpectraPlot)
        self.ProbTrajectoriesPlot = makefactory(_driftrpt.ProbTrajectoriesPlot)
        self.GermFiducialProbTrajectoriesPlot = makefactory(_driftrpt.GermFiducialProbTrajectoriesPlot)
        self.GermFiducialPowerSpectraPlot = makefactory(_driftrpt.GermFiducialPowerSpectraPlot)

    def init_notebook_mode(self, connected=False, autodisplay=False):
        """
        Initialize this Workspace for use in an iPython notebook environment.

        This function should be called prior to using the Workspace when
        working within an iPython notebook.

        Parameters
        ----------
        connected : bool , optional
            Whether to assume you are connected to the internet.  If you are,
            then setting this to `True` allows initialization to rely on web-
            hosted resources which will reduce the overall size of your
            notebook.

        autodisplay : bool , optional
            Whether to automatically display workspace objects after they are
            created.

        Returns
        -------
        None
        """
        if not in_ipython_notebook():
            raise ValueError('Only run `init_notebook_mode` from inside an IPython Notebook.')

        global _PYGSTI_WORKSPACE_INITIALIZED

        script = ""

        if not connected:
            _merge.rsync_offline_dir(_os.getcwd())

        #If offline, add JS to head that will load local requireJS and/or
        # jQuery if needed (jupyter-exported html files always use CDN
        # for these).
        if not connected:
            script += "<script src='offline/jupyterlibload.js'></script>\n"

        #Load our custom plotly extension functions
        script += _merge.insert_resource(connected, None, "pygsti_plotly_ex.js")
        script += "<script type='text/javascript'> window.plotman = new PlotManager(); </script>"

        #jQueryUI_CSS = "https://code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css"
        jQueryUI_CSS = "https://code.jquery.com/ui/1.12.1/themes/smoothness/jquery-ui.css"
        script += _merge.insert_resource(connected, jQueryUI_CSS, "smoothness-jquery-ui.css")

        # Load style sheets for displaying tables
        script += _merge.insert_resource(connected, None, "pygsti_dataviz.css")

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
        script += _merge.insert_resource(
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

        display_ipynb(script)  # single call to display keeps things simple

        _PYGSTI_WORKSPACE_INITIALIZED = True

        self._register_components(autodisplay)
        return

    def switched_compute(self, fn, *args):
        """
        Calls a compute function with special handling of :class:`SwitchedValue` arguments.

        This is similar to calling `fn`, given its name and arguments, when
        some or all of those arguments are :class:`SwitchedValue` objects.

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

        for i, arg in enumerate(args):
            if isinstance(arg, SwitchValue):
                isb = None
                for j, sb in enumerate(switchboards):
                    if arg.parent is sb:
                        isb = j; break
                else:
                    isb = len(switchboards)
                    switchboards.append(arg.parent)
                    switchBdInfo.append({
                        'argument indices': [],  # indices of arguments that are children of this switchboard
                        'value names': [],  # names of switchboard value correspond to each argument index
                        'switch indices': set()  # indices of the switches that are actually used by the args
                    })
                assert(isb is not None)
                info = switchBdInfo[isb]

                info['argument indices'].append(i)
                info['value names'].append(arg.name)
                info['switch indices'].update(arg.dependencies)
            else:
                nonSwitchedArgs.append((i, arg))

        #print("DB: %d arguments" % len(args))
        #print("DB: found %d switchboards" % len(switchboards))
        #print("DB: switchBdInfo = ", switchBdInfo)
        #print("DB: nonSwitchedArgs = ", nonSwitchedArgs)

        #Create a list of lists, each list holding all of the relevant switch positions for each board
        switch_positions = []
        for isb, sb in enumerate(switchboards):
            info = switchBdInfo[isb]
            info['switch indices'] = list(info['switch indices'])  # set -> list so definite order

            switch_ranges = [list(range(len(sb.positionLabels[i])))
                             for i in info['switch indices']]
            sb_switch_positions = list(_itertools.product(*switch_ranges))
            # a list of all possible positions for the switches being
            # used for the *single* board sb
            switch_positions.append(sb_switch_positions)

        #loop over all relevant switch configurations (across multiple switchboards)
        for pos in _itertools.product(*switch_positions):
            # pos[i] gives the switch configuration for the i-th switchboard

            #fill in the arguments for our function call
            argVals = [None] * len(args)

            #first, iterate over all the switchboards
            for sw_pos, sb, info in zip(pos, switchboards, switchBdInfo):
                # sw_pos is a tuple of the info['switch indices'] switch positions for sb
                sis = info['switch indices']
                for nm, j in zip(info["value names"], info["argument indices"]):
                    value_swpos = [sw_pos[sis.index(k)] for k in sb[nm].dependencies]
                    # potentially a subset of sw_pos, contains only the switch positions
                    # relevant to the particular SwitchedValue named nm (also the j-th argument)
                    argVals[j] = sb[nm][tuple(value_swpos)]  # tuple needed for proper indexing

            #next, fill in the non-switched arguments
            for j, arg in nonSwitchedArgs:
                argVals[j] = arg

            for v in argVals:
                if isinstance(v, NotApplicable):
                    key = "NA"; result = v; break
            else:
                key, result = self.smartCache.cached_compute(fn, argVals)

            if key not in storedKeys or key == 'INEFFECTIVE':
                switchpos_map[pos] = len(resultValues)
                storedKeys[key] = len(resultValues)
                resultValues.append(result)
            else:
                switchpos_map[pos] = storedKeys[key]

        switchboard_switch_indices = [info['switch indices'] for info in switchBdInfo]
        return resultValues, switchboards, switchboard_switch_indices, switchpos_map


class Switchboard(_collections.OrderedDict):
    """
    Encapsulates a render-able set of user-interactive switches for controlling visualized output.

    Outwardly a Switchboard looks like a dictionary of SwitchValue
    objects, which in turn look like appropriately sized numpy arrays
    of values for some quantity.  Different switch positions select
    different values and thereby what data is visualized in various
    outputs (e.g. tables and plots).

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.

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

    id : str (optional)
        A DOM identifier to use when rendering this Switchboard to HTML.
        Usually leaving this value as `None` is best, in which case a
        random identifier is created.

    use_loadable_items : bool, optional
        Whether "loadable" items are being used.  When using loadable items,
        elements of a web page are explicitly told when to initialize themselves
        by a "load_loadable_item" signal instead of loading as soon as the DOM
        is ready (a traditional on-ready handler).  Using this option increases
        the performance of large/complex web pages.
    """

    def __init__(self, ws, switches, positions, types, initial_pos=None,
                 descriptions=None, show="all", id=None, use_loadable_items=False):
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

        id : str (optional)
            A DOM identifier to use when rendering this Switchboard to HTML.
            Usually leaving this value as `None` is best, in which case a
            random identifier is created.

        use_loadable_items : bool, optional
            Whether "loadable" items are being used.  When using loadable items,
            elements of a web page are explicitly told when to initialize themselves
            by a "load_loadable_item" signal instead of loading as soon as the DOM
            is ready (a traditional on-ready handler).  Using this option increases
            the performance of large/complex web pages.
        """
        # Note: intentionally leave off ws argument desc. in docstring
        assert(len(switches) == len(positions))

        self.ID = random_id() if (id is None) else id
        self.ws = ws  # Workspace
        self.switchNames = switches
        self.switchTypes = types
        self.switchIDs = ["switchbd%s_%d" % (self.ID, i)
                          for i in range(len(switches))]
        self.positionLabels = positions
        self.use_loadable_items = use_loadable_items
        if initial_pos is None:
            self.initialPositions = _np.array([0] * len(switches), _np.int64)
        else:
            assert(len(initial_pos) == len(switches))
            self.initialPositions = _np.array(initial_pos, _np.int64)

        self.descriptions = descriptions

        if show == "all":
            self.show = [True] * len(switches)
        elif show == "none":
            self.show = [False] * len(switches)
        else:
            assert(len(show) == len(switches))
            self.show = show

        self.widget = None
        super(Switchboard, self).__init__([])

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
            has two switches, one for "amplitude" and one for "frequency", and
            this value is only dependent on frequency, then `dependencies`
            should be set to `(1,)` or `[1]`.

        Returns
        -------
        None
        """
        super(Switchboard, self).__setitem__(varname, SwitchValue(self, varname, dependencies))

    def add_unswitched(self, varname, value):
        """
        Adds a new non-switched-value to this Switchboard.

        This can be convenient for attaching related non-switched data to
        a :class:`Switchboard`.

        Parameters
        ----------
        varname : str
            A name for the variable being added.  This name will be used to
            access the new variable (as either a dictionary key or as an
            object member).

        value : object
            The un-switched value to associate with `varname`.

        Returns
        -------
        None
        """
        super(Switchboard, self).__setitem__(varname, value)

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
        for name, baseID, styp, posLbls, ipos, bShow in zip(
                self.switchNames, self.switchIDs, self.switchTypes,
                self.positionLabels, self.initialPositions, show):

            ID = (baseID + view_suffix) if view_suffix else baseID
            style = "" if bShow else " style='display: none'"

            if styp == "buttons":
                html = "<div class='switch_container'%s>\n" % style
                html += "<fieldset id='%s'>\n" % ID
                if name:
                    html += "<legend>%s: </legend>\n" % name
                for k, lbl in enumerate(posLbls):
                    checked = " checked='checked'" if k == ipos else ""
                    html += "<label for='%s-%d'>%s</label>\n" % (ID, k, lbl)
                    html += "<input type='radio' name='%s' id='%s-%d' value=%d%s>\n" \
                        % (ID, ID, k, k, checked)
                html += "</fieldset></div>\n"
                js = "  $('#%s > input').checkboxradio({ icon: false });" % ID

                if view_suffix:
                    js += "\n".join((
                        "function connect_%s_to_base(){" % ID,
                        "  if( $('#%s').hasClass('initializedSwitch') ) {" % baseID,  # "if base switch is ready"
                        "    $('#%s').on('change', function(event, ui) {" % baseID,
                        "      var v = $(\"#%s > input[name='%s']:checked\").val();" % (baseID, baseID),
                        "      var el = $(\"#%s > input[name='%s'][value=\" + v + \"]\");" % (ID, ID),
                        "      if( el.is(':checked') == false ) { ",
                        "        el.click();",
                        "      }",
                        "    });"
                        "    $('#%s').on('change', function(event, ui) {" % ID,
                        "      var v = $(\"#%s > input[name='%s']:checked\").val();" % (ID, ID),
                        "      var el = $(\"#%s > input[name='%s'][value=\" + v + \"]\");" % (baseID, baseID),
                        "      if( el.is(':checked') == false ) { ",
                        "        el.click();",
                        "      }",
                        "    });",
                        "    $('#%s').trigger('change');" % baseID,
                        "  }",
                        "  else {",  # need to wait for base switch
                        "    setTimeout(connect_%s_to_base, 500);" % ID,
                        "    console.log('%s base NOT initialized: Waiting...');" % ID,
                        "  }",
                        "};",
                        "connect_%s_to_base();" % ID  # start trying to connect
                    ))

            elif styp == "dropdown":
                html = "<div class='switch_container'%s><fieldset>\n" % style
                if name:
                    html += "<label for='%s'>%s</label>\n" % (ID, name)
                html += "<select name='%s' id='%s'>\n" % (ID, ID)
                for k, lbl in enumerate(posLbls):
                    selected = " selected='selected'" if k == ipos else ""
                    html += "<option value=%d%s>%s</option>\n" % (k, selected, lbl)
                html += "</select>\n</fieldset></div>\n"
                js = "  $('#%s').selectmenu();" % ID

                if view_suffix:
                    js += "\n".join((
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
                        "  else {",  # need to wait for base switch
                        "    setTimeout(connect_%s_to_base, 500);" % ID,
                        "    console.log('%s base NOT initialized: Waiting...');" % ID,
                        "  }",
                        "};",
                        "connect_%s_to_base();" % ID  # start trying to connect
                    ))

            elif styp == "slider" or styp == "numslider":

                if styp == "numslider":
                    float_vals = list(map(float, posLbls))
                    m, M = min(float_vals), max(float_vals)
                else:
                    float_vals = list(range(len(posLbls)))
                    m, M = 0, len(posLbls) - 1

                #ml = max(list(map(len,posLbls)))
                w = 3.0  # 1.0*ml

                html = "<div id='%s-container' class='switch_container'%s>\n" \
                    % (ID, style)
                html += "<fieldset>\n"
                if name:
                    html += "<label for='%s' class='pygsti-slider-label'>%s</label>\n" % (ID, name)
                html += "<div name='%s' id='%s'>\n" % (ID, ID)
                html += "<div id='%s-handle' class='ui-slider-handle'></div>" % ID
                html += "</div>\n</fieldset></div>\n"
                #                    "       $('#%s-container').css({'margin-top':'%fem'});" % (ID,1.7/2),

                js = ""
                if view_suffix is None:
                    js = "var %s_float_values = [" % ID + \
                        ",".join(map(str, float_vals)) + "];\n"
                    js += "var %s_str_values = [" % ID + \
                        ",".join(["'%s'" % s for s in posLbls]) + "];\n"
                    js += "window.%s_float_values = %s_float_values;\n" % (ID, ID)  # ensure declared globally
                    js += "window.%s_str_values = %s_str_values;\n" % (ID, ID)  # ensure declared globally

                    js += "\n".join((
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
                        "window.findNearest_%s = findNearest_%s;\n" % (ID, ID)))

                #allow ipos = something (e.g. -1) when there aren't any position labels
                if len(posLbls) == 0:
                    float_val = 0.0; posLabel = "--"
                else:
                    float_val = float_vals[ipos]
                    posLabel = posLbls[ipos]

                js += "\n".join((
                    "  $('#%s').slider({" % ID,
                    "     orientation: 'horizontal', range: false,",
                    "     min: %f, max: %f, step: %f," % (m, M, (M - m) / 100.0),
                    "     value: %f," % float_val,
                    "     create: function() {",
                    "       $('#%s-handle').text('%s');" % (ID, posLabel),
                    "       $('#%s-handle').css({'width':'%fem','height':'%fem'});" % (ID, w, 1.7),
                    "       $('#%s-handle').css({'margin-left':'%fem','top':'%fem'});" % (ID, -w / 2, -1.7 / 2 + 0.4),
                    "       $('#%s-handle').css({'text-align':'center','line-height':'1.5em'});" % ID,
                    "       $('#%s').css({'margin-left':'%fem', 'margin-top':'0.4em'});" % (ID, w / 2),
                    "     },",
                    "     slide: function(event, ui) {",
                    "        var includeLeft = event.keyCode != $.ui.keyCode.RIGHT;",
                    "        var includeRight = event.keyCode != $.ui.keyCode.LEFT;",
                    "        var iValue = findNearest_%s(includeLeft, includeRight, ui.value);" % baseID,
                    "        if($('#%s').slider('value') != %s_float_values[iValue]) {" % (ID, baseID),
                    "          $('#%s-handle').text(%s_str_values[iValue]);" % (baseID, baseID),
                    "          $('#%s').slider('value', %s_float_values[iValue]);" % (baseID, baseID),
                    "        }"
                    "        return false;"
                    "    },",
                    "  });",
                ))

                if view_suffix:
                    # slide events always change *base* (non-view) slider (see above),
                    # which causes a change event to fire.  Views handle this event
                    # to update their own slider values.
                    js += "\n".join((
                        "function connect_%s_to_base(){" % ID,
                        "  if( $('#%s').hasClass('initializedSwitch') ) {" % baseID,  # "if base switch is ready"
                        "    $('#%s').on('slidechange', function(event, ui) {" % baseID,
                        "      $('#%s').slider('value', ui.value);" % ID,
                        "      $('#%s-handle').text( $('#%s-handle').text() );" % (ID, baseID),
                        "      });",
                        "    var mock_ui = { value: $('#%s').slider('value') };" % baseID,  # b/c handler uses ui.value
                        "    $('#%s').trigger('slidechange', mock_ui);" % baseID,
                        "  }",
                        "  else {",  # need to wait for base switch
                        "    setTimeout(connect_%s_to_base, 500);" % ID,
                        "    console.log('%s base NOT initialized: Waiting...');" % ID,
                        "  }",
                        "};",
                        "connect_%s_to_base();" % ID  # start trying to connect
                    ))

            else:
                raise ValueError("Unknown switch type: %s" % styp)

            js += "$('#%s').addClass('initializedSwitch');\n" % ID

            switch_html.append(html)
            switch_js.append(js)

        html = "\n".join(switch_html)
        if not self.use_loadable_items:  # run JS as soon as the document is ready
            js = "$(document).ready(function() {\n" + \
                "\n".join(switch_js) + "\n});"
        else:  # in a report, where we have a 'loadable' parent and might not want to load right away
            js = "$(document).ready(function() {\n" + \
                 "$('#%s').closest('.loadable').on('load_loadable_item', function(){\n" % ID + \
                "\n".join(switch_js) + "\n}); });"

        return {'html': html, 'js': js}

    def create_switch_change_handlerjs(self, switch_index):
        """
        Returns the Javascript needed to begin an on-change handler for a particular switch.

        Parameters
        ----------
        switch_index : int
            The 0-based index of which switch to get handler JS for.

        Returns
        -------
        str
        """
        ID = self.switchIDs[switch_index]
        typ = self.switchTypes[switch_index]
        if typ == "buttons":
            return "$('#%s').on('change', function() {" % ID
        elif typ == "dropdown":
            return "$('#%s').on('selectmenuchange', function() {" % ID
        elif typ == "slider" or typ == "numslider":
            return "$('#%s').on('slidechange', function() {" % ID  # only when slider stops
            #return "$('#%s').on('slide', function() {" % ID # continuous on
            # mouse move - but doesn't respond correctly to arrows, so seems
            # better to use 'slidechange'
        else:
            raise ValueError("Unknown switch type: %s" % typ)

    def create_switch_valuejs(self, switch_index):
        """
        Returns the Javascript needed to get the value of a particular switch.

        Parameters
        ----------
        switch_index : int
            The 0-based index of which switch to get value-extracting JS for.

        Returns
        -------
        str
        """
        ID = self.switchIDs[switch_index]
        typ = self.switchTypes[switch_index]
        if typ == "buttons":
            return "$(\"#%s > input[name='%s']:checked\").val()" % (ID, ID)
        elif typ == "dropdown":
            return "$('#%s').val()" % ID
        elif typ == "slider" or typ == "numslider":
            #return "%s_float_values.indexOf($('#%s').slider('option', 'value'))" % (ID,ID)
            return "findNearest_%s(true,true,$('#%s').slider('option', 'value'))" % (ID, ID)
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
        if not in_ipython_notebook():
            raise ValueError('Only run `display` from inside an IPython Notebook.')

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
        display_ipynb(content)  # self.widget)

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
            show = [True] * len(self.switchNames)
        elif all([isinstance(b, bool) for b in switches]):
            assert(len(switches) == len(self.switchNames))
            show = switches
        else:
            show = [False] * len(self.switchNames)
            for nm in switches:
                show[self.switchNames.index(nm)] = True

        return SwitchboardView(self, idsuffix, show)

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        return getattr(self.__dict__, attr)


class SwitchboardView(object):
    """
    A duplicate or "view" of an existing switchboard.

    This view is logically represents the *same* set of switches.
    Thus, when switches are moved on the duplicate board, switches
    will move on the original (and vice versa).

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
            self.idsuffix = "v" + random_id()
        else:
            self.idsuffix = idsuffix

        if show == "all":
            self.show = [True] * len(switchboard.switchNames)
        elif show == "none":
            self.show = [False] * len(switchboard.switchNames)
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
        if not in_ipython_notebook():
            raise ValueError('Only run `display` from inside an IPython Notebook.')

        out = self.render("html")
        content = "<script>\n" + \
                  "require(['jquery','jquery-UI'],function($,ui) {" + \
                  out['js'] + " });</script>" + out['html']
        display_ipynb(content)


class SwitchValue(object):
    """
    A value that is controlled by the switches of a single Switchboard.

    "Value" here means an arbitrary quantity, and is usually an argument
    to visualization functions.

    The paradigm is one of a Switchboard being a collection of switches along
    with a dictionary of SwitchValues, whereby each SwitchValue is a mapping
    of switch positions to values.  For efficiency, a SwitchValue need only map
    a "subspace" of the switch positions, that is, the position-space spanned
    by only a subset of the switches.  Which switch-positions are mapped is
    given by the "dependencies" of a SwitchValue.

    SwitchValue behaves much like a numpy array of values in terms of
    element access.

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
        self.ws = parent_switchboard.ws  # workspace
        self.parent = parent_switchboard
        self.name = name
        self.dependencies = dependencies

        shape = [len(self.parent.positionLabels[i]) for i in dependencies]
        self.base = _np.empty(shape, dtype=_np.object)
        index_all = (slice(None, None),) * len(shape)
        self.base[index_all] = NotApplicable(self.ws)

    #Access to underlying ndarray
    def __getitem__(self, key):
        return self.base.__getitem__(key)

    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))  # Called for A[:]

    def __setitem__(self, key, val):
        return self.base.__setitem__(key, val)

    def __getattr__(self, attr):
        #use __dict__ so no chance for recursive __getattr__
        return getattr(self.__dict__['base'], attr)

    def __len__(self): return len(self.base)
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

    Parameters
    ----------
    ws : Workspace
        The workspace containing the new object.

    Attributes
    ----------
    default_render_options : dict
        Class attribute giving default rendering options.
    """
    default_render_options = {
        #General
        'output_dir': False,
        'precision': None,

        'output_name': False,
        'switched_item_mode': 'inline',  # or 'separate files'
        'switched_item_id_overrides': {},

        #HTML specific
        'global_requirejs': False,
        'use_loadable_items': False,
        'click_to_display': False,
        'render_math': True,
        'resizable': True,
        'autosize': 'none',
        'link_to': None,
        'valign': 'top',

        #Latex specific
        'latex_cmd': "pdflatex",
        'latex_flags': ["-interaction=nonstopmode", "-halt-on-error", "-shell-escape"],
        'page_size': (6.5, 8.0),
        'render_includes': True,
        'leave_includes_src': False,
    }

    def __init__(self, ws):
        """
        Create a new WorkspaceOutput object.  Usually not called directly.

        Parameters
        ----------
        ws : Workspace
            The workspace containing the new object.
        """
        self.ws = ws
        self.ID = random_id()  # maybe allow overriding this in the FUTURE
        self.options = WorkspaceOutput.default_render_options.copy()

    def set_render_options(self, **kwargs):
        """
        Sets rendering options, which affect how render() behaves.

        The reason render options are set via this function rather
        than passed directly as arguments to the render(...) call
        is twofold.  First, it allows for global 'default' options
        to be set before creating `WorkspaceOutput`-derived objects;
        Secondly, it allows the user to set render options right after
        an object is constructed, separately from the rendering process
        (which is sometimes desirable).

        Parameters
        ----------
        output_dir : str or False
            The name of the output directory under which all output files
            should be created.  The names of these files just the IDs of the
            items being rendered.

        precision : int or dict, optional
            The amount of precision to display.  A dictionary with keys
            "polar", "sci", and "normal" can separately specify the
            precision for complex angles, numbers in scientific notation, and
            everything else, respectively.  If an integer is given, it this
            same value is taken for all precision types.  If None, then
            `{'normal': 6, 'polar': 3, 'sci': 0}` is used.

        switched_item_mode : {'inline','separate files'}, optional
            Whether switched items should be rendered inline within the 'html'
            and 'js' blocks of the return value of :func:`render`, or whether
            each switched item (corresponding to a single "switch position")
            should be rendered in a separate file and loaded on-demand only
            when it is needed.

        switched_item_id_overrides : dict, optional
            A dictionary of *index*:*id* pairs, where *index* is a 0-based index
            into the list of switched items (plots or tables), and *id* is a
            string ID.  Since the ID is used as the filename when saving files,
            overriding the ID is useful when writing a single plot or table to
            a specific filename.

        global_requirejs : bool, optional
            Whether the table is going to be embedded in an environment
            with a globally defined RequireJS library.  If True, then
            rendered output will make use of RequireJS.

        click_to_display : bool, optional
            If True, table plots are not initially created but must
            be clicked to prompt creation.  This is False by default,
            and can be useful to set to True for tables with
            especially complex plots whose creation would slow down
            page loading significantly.

        resizable : bool, optional
            Whether or not to place table inside a JQueryUI
            resizable widget (only applies when `typ == "html"`).

        autosize : {'none', 'initial', 'continual'}, optional
            Whether tables and plots should be resized either
            initially, i.e. just upon first rendering (`"initial"`) or whenever
            the browser window is resized (`"continual"`).  This option only
            applies for html rendering.

        link_to : tuple of {"tex", "pdf", "pkl"} or None, optional
            If not None, a list of one or more items from the given set
            indicating whether or not to include links to Latex, PDF, and
            Python pickle files, respectively.  Note that setting this
            render option does not automatically *create/render* additional
            formats of this output object (you need to make multiple `render`
            calls for that) - it just creates the *links* to these files when
            rendering as "html".

        valign : {"top","bottom"}
            Whether the switched items should be vertically aligned by their
            tops or bottoms (when they're different heights).

        latex_cmd : str, optional
            The system command or executable used to compile LaTeX documents.
            Usually `"pdflatex"`.

        latex_flags : list, optional
            A list of (string-valued) flags to pass to `latex_cmd` when
            compiling LaTeX documents.  Defaults to
            `["-interaction=nonstopmode", "-halt-on-error", "-shell-escape"]`

        page_size : tuple
            The usable page size for LaTeX documents, as (*width*,*height*)
            where *width* and *height* are in inches.  Note that this does not
            include margins.  Defaults to `(6.5,8.0)`.

        render_includes : bool, optional
            When rendering as "latex", whether included files should also be
            rendered (either by compiling latex to PDF or saving plots as PDFs).

        leave_includes_src : bool, optional
            When LaTeX compilation is done, should the source "*.tex" files be
            removed? If `False`, then they *are* removed.

        Returns
        -------
        None
        """
        for key, val in kwargs.items():
            if key in self.options:
                self.options[key] = val
            else:
                raise ValueError("Invalid render option: %s\nValid options are:\n" % key
                                 + '\n'.join(self.options.keys()))

    def __getstate__(self):
        state_dict = self.__dict__.copy()
        del state_dict['ws']
        return state_dict

    def __setstate__(self, d):
        self.__dict__.update(d)
        if 'ws' not in self.__dict__:
            self.__dict__['ws'] = None

    # Note: hashing not needed because these objects are not *inputs* to
    # other WorspaceOutput objects or computation functions - these objects
    # are cached using call_key.

    def render(self, typ="html"):
        """
        Renders this object into the specified format, specifically for embedding it within a larger document.

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

        Returns
        -------
        None
        """
        if not in_ipython_notebook():
            raise ValueError('Only run `display` from inside an IPython Notebook.')

        self.set_render_options(global_requirejs=True,
                                output_dir=None)  # b/c jupyter uses require.js
        out = self.render("html")
        content = "<script>\n" + \
                  "require(['jquery','jquery-UI','plotly'],function($,ui,Plotly) {" + \
                  out['js'] + " });</script>" + out['html']

        display_ipynb(content)

    def saveas(self, filename, index=None, verbosity=0):
        """
        Saves this workspace output object to a file.

        The type of file that is saved is determined automatically by the
        extension of `filename`.  Recognized extensions are `pdf` (PDF),
        `tex` (LaTeX), `pkl` (Python pickle) and `html` (HTML).  Since this
        object may contain different instances of its data based on switch
        positions, when their are multiple instances the user must specify
        the `index` argument to disambiguate.

        Parameters
        ----------
        filename : str
            The destination filename.  Its extension determines what type
            of file is saved.

        index : int, optional
            An absolute index into the list of different switched "versions"
            of this object's data.  In most cases, the object being saved
            doesn't depend on any switch boards and has only a single "version",
            in which caes this can be left as the default.

        verbosity : int, optional
            Controls the level of detail printed to stdout.

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def _ccompute(self, fn, *args, **kwargs):
        """ Cached-computation using self.ws's smart cache """
        return self.ws.smartCache.cached_compute(fn, args, kwargs)[1]

    def _create_onready_handler(self, content, id):
        global_requirejs = self.options.get('global_requirejs', False)
        use_loadable_items = self.options.get('use_loadable_items', False)
        ret = ""

        if global_requirejs:
            ret += "require(['jquery','jquery-UI','plotly','autorender'],function($,ui,Plotly,renderMathInElement) {\n"

        ret += '  $(document).ready(function() {\n'
        if use_loadable_items:
            ret += "  $('#%s').closest('.loadable').on('load_loadable_item', function(){\n" % id

        ret += content

        if use_loadable_items:
            ret += "  });"  # end load_loadable_item handler

        ret += '}); //end on-ready or on-load handler\n'

        if global_requirejs:
            ret += '}); //end require block\n'

        return ret

    def _render_html(self, id, div_htmls, div_jss, div_ids, switchpos_map,
                     switchboards, switch_indices, div_css_classes=None,
                     link_to=None, link_to_files_dir=None, embed_figures=True):
        """
        Helper rendering function, which takes care of the (complex)
        common logic which take a series of HTML div blocks corresponding
        to the results of a Workspace.switched_compute(...) call and
        builds the HTML and JS necessary for toggling the visibility of
        these divs in response to changes in switch position(s).

        Parameters
        ----------
        id: str
            The identifier to use when constructing DOM ids.

        div_htmls : list
            The html content for each switched block (typically a elements are
            "<div>...</div>" blocks themselves).  This is the content that
            is switched between.

        div_jss : list
            Javascript content to accompany each switched block.

        div_ids : list
            A list giving the DOM ids for the div blocks given by `div_html`.

        switchpos_map : dict
            A dictionary mapping switch positions to div-index.  Keys are switch
            tuples of per-switchboard positions (i.e. a tuple of tuples), giving
            the positions of each switch specified in `switch_indices`.  Values
            are integer indices into `html_divs`.

        switchboards : list
            A list of relevant SwitchBoard objects.

        switch_indices : list
            A list of tuples, one per Switchboard object, giving the relevant
            switch indices (integers) within that Switchboard.

        div_css_classes : list, optional
            A list of (string) CSS classes to add to the div elements created
            by this function.

        link_to : list, optional
            If not None, a list of one or more items from the set
            {"tex", "pdf", "pkl"} indicating whether or not to
            include links to Latex, PDF, and Python pickle files,
            respectively.

        link_to_files_dir : str, optional
            The directory to place linked-to files in. Only used when
            `link_to` is not None.

        embed_figures: bool, optional
           If True (default), figures will be embedded directly into
           the report HTML. Otherwise, figures will be written to
           `link_to_files_dir` and dynamically loaded into the report
           with AJAX requests.

        Returns
        -------
        dict
            A dictionary of strings whose keys indicate which portion of
            the embeddable output the value is.  Keys are `"html"` and `"js"`.
        """

        # within_report = self.options.get('within_report', False)

        #Build list of CSS classes for the created divs
        classes = ['single_switched_value']
        if div_css_classes is not None:
            classes.extend(div_css_classes)
        cls = ' '.join(classes)

        #build HTML as container div containing one or more plot divs
        # Note: 'display: none' doesn't always work in firefox... (polar plots in ptic)
        #   style='display: none' or 'visibility: hidden'
        html = "<div id='%s' class='pygsti-wsoutput-group'>\n" % id

        div_contents = []
        if div_jss is None: div_jss = [""] * len(div_htmls)
        for divHTML, divJS in zip(div_htmls, div_jss):
            scriptJS = "<script>\n%s\n</script>\n" % divJS if divJS else ""
            div_contents.append(("{script}{html}".format(
                script=scriptJS, html=divHTML)))

        if embed_figures:
            #Inline div contents
            html += "\n".join(["<div class='%s' id='%s'>\n%s\n</div>\n" %
                               (cls, divID, divContent) for divID, divContent
                               in zip(div_ids, div_contents)])
        else:
            html += "\n".join(["<div class='%s' id='%s'></div>\n" %
                               (cls, divID) for divID in div_ids])

            #build a list of filenames based on the divIDs
            div_filenames = [(divID + ".html") for divID in div_ids]

            #Create separate files with div contents
            for divContent, divFilenm in zip(div_contents, div_filenames):
                with open(_os.path.join(str(link_to_files_dir), divFilenm), 'w') as f:
                    f.write(divContent)
        html += "\n</div>\n"  # ends pygsti-wsoutput-group div

        #build javascript to map switch positions to div_ids
        js = "var switchmap_%s = new Array();\n" % id
        for switchPositions, iDiv in switchpos_map.items():
            #switchPositions is a tuple of tuples of position indices, one tuple per switchboard
            div_id = div_ids[iDiv]
            flatPositions = []
            for singleBoardSwitchPositions in switchPositions:
                flatPositions.extend(singleBoardSwitchPositions)
            js += "switchmap_%s[ [%s] ] = '%s';\n" % \
                (id, ",".join(map(str, flatPositions)), div_id)

        js += "window.switchmap_%s = switchmap_%s;\n" % (id, id)  # ensure a *global* variable
        js += "\n"

        cnd = " && ".join(["$('#switchbd%s_%d').hasClass('initializedSwitch')"
                           % (sb.ID, switchIndex)
                           for sb, switchInds in zip(switchboards, switch_indices)
                           for switchIndex in switchInds])
        if len(cnd) == 0: cnd = "true"

        #define fn to "connect" output object to switchboard, i.e.
        #  register event handlers for relevant switches so output object updates
        js += "function connect_%s_to_switches(){\n" % id
        js += "  if(%s) {\n" % cnd  # "if switches are ready"
        # loop below adds event bindings to the body of this if-block

        #build a handler function to get all of the relevant switch positions,
        # build a (flattened) position array, and perform the lookup.  Note that
        # this function does the same thing regardless of *which* switch was
        # changed, and so is called by all relevant switch change handlers.
        onchange_name = "%s_onchange" % id
        handler_js = "function %s() {\n" % onchange_name
        handler_js += "  var tabdiv = $( '#%s' ).closest('.tabcontent');\n" % id
        handler_js += "  if( tabdiv.length > 0 && !tabdiv.hasClass('active') ) return;\n"  # short-circuit
        handler_js += "  var curSwitchPos = new Array();\n"
        for sb, switchInds in zip(switchboards, switch_indices):
            for switchIndex in switchInds:
                handler_js += "  curSwitchPos.push(%s);\n" % sb.create_switch_valuejs(switchIndex)
        handler_js += "  var idToShow = switchmap_%s[ curSwitchPos ];\n" % id
        handler_js += "  $( '#%s' ).children().hide();\n" % id
        handler_js += "  divToShow = $( '#' + idToShow );\n"

        #Javascript to switch to a new div
        if embed_figures:
            handler_js += "  divToShow.show();\n"
            handler_js += "  divToShow.parentsUntil('#%s').show();\n" % id
            handler_js += "  caption = divToShow.closest('figure').children('figcaption:first');\n"
            handler_js += "  caption.css('width', Math.round(divToShow.width()*0.9) + 'px');\n"
        else:
            handler_js += "  if( divToShow.children().length == 0 ) {\n"
            handler_js += "    $(`#${idToShow}`).load(`figures/${idToShow}.html`, function() {\n"
            handler_js += "        divToShow = $( '#' + idToShow );\n"
            handler_js += "        divToShow.show();\n"
            handler_js += "        divToShow.parentsUntil('#%s').show();\n" % id
            if link_to and ('tex' in link_to):
                handler_js += "    divToShow.append('<a class=\"dlLink\" href=\"figures/'"
                handler_js += " + idToShow + '.tex\" target=\"_blank\">&#9660;TEX</a>');\n"
            if link_to and ('pdf' in link_to):
                handler_js += "    divToShow.append('<a class=\"dlLink\" href=\"figures/'"
                handler_js += " + idToShow + '.pdf\" target=\"_blank\">&#9660;PDF</a>');\n"
            if link_to and ('pkl' in link_to):
                handler_js += "    divToShow.append('<a class=\"dlLink\" href=\"figures/'"
                handler_js += " + idToShow + '.pkl\" target=\"_blank\">&#9660;PKL</a>');\n"
            handler_js += "        caption = divToShow.closest('figure').children('figcaption:first');\n"
            handler_js += "        caption.css('width', Math.round(divToShow.width()*0.9) + 'px');\n"
            handler_js += "    });\n"  # end load-complete handler
            handler_js += "  }\n"
            handler_js += "  else {\n"
            handler_js += "    divToShow.show();\n"
            handler_js += "    divToShow.parentsUntil('#%s').show();\n" % id
            handler_js += "    caption = divToShow.closest('figure').children('figcaption:first');\n"
            handler_js += "    caption.css('width', Math.round(divToShow.width()*0.9) + 'px');\n"
            handler_js += "  }\n"
        handler_js += "}\n"  # end <id>_onchange function

        #build change event listener javascript
        for sb, switchInds in zip(switchboards, switch_indices):
            # switchInds is a tuple containing the "used" switch indices of sb
            for switchIndex in switchInds:
                # part of if-block ensuring switches are ready (i.e. created)
                js += "    " + sb.create_switch_change_handlerjs(switchIndex) + \
                    "%s(); });\n" % onchange_name

        #bind onchange call to custom 'tabchange' event that we trigger when tab changes
        js += "    $( '#%s' ).closest('.tabcontent').on('tabchange', function(){\n" % id
        js += "%s(); });\n" % onchange_name
        js += "    %s();\n" % onchange_name  # call onchange function *once* at end to update visibility

        # end if-block
        js += "    console.log('Switches initialized: %s handlers set');\n" % id
        js += "    $( '#%s' ).show()\n" % id  # visibility updates are done: show parent container
        js += "  }\n"  # ends if-block
        js += "  else {\n"  # switches aren't ready - so wait
        js += "    setTimeout(connect_%s_to_switches, 500);\n" % id
        js += "    console.log('%s switches NOT initialized: Waiting...');\n" % id
        js += "  }\n"
        js += "};\n"  # end of connect function

        #on-ready handler starts trying to connect to switches
        # - note this is already in a 'load_loadable_item' handler, so no need for that here
        js += "$(document).ready(function() {\n"
        js += "  connect_%s_to_switches();\n" % id

        if link_to:
            # Add download links for all divs at once since they're all ready
            rel_figure_dir = _os.path.basename(str(link_to_files_dir))
            if 'tex' in link_to:
                for div_id in div_ids:
                    js += "  $('#%s').append('<a class=\"dlLink\" href=\"%s/" % (div_id, rel_figure_dir)
                    js += "%s.tex\" target=\"_blank\">&#9660;TEX</a>');\n" % div_id
            if 'pdf' in link_to:
                for div_id in div_ids:
                    js += "  $('#%s').append('<a class=\"dlLink\" href=\"%s/" % (div_id, rel_figure_dir)
                    js += "%s.pdf\" target=\"_blank\">&#9660;PDF</a>');\n" % div_id
            if 'pkl' in link_to:
                for div_id in div_ids:
                    js += "  $('#%s').append('<a class=\"dlLink\" href=\"%s/" % (div_id, rel_figure_dir)
                    js += "%s.pkl\" target=\"_blank\">&#9660;PKL</a>');\n" % div_id

        js += "});\n\n"  # end on-ready handler
        js += handler_js

        return {'html': html, 'js': js}


class NotApplicable(WorkspaceOutput):
    """
    Class signifying that an given set of arguments is not applicable to a function being evaluated.

    Parameters
    ----------
    ws : Workspace
        The containing (parent) workspace.
    """

    def __init__(self, ws):
        """
        Create a new NotApplicable object.
        """
        super(NotApplicable, self).__init__(ws)

    def render(self, typ="html", id=None):
        """
        Renders this object into the specified format, specifically for embedding it within a larger document.

        Parameters
        ----------
        typ : str
            The format to render as.  Allowed options are `"html"`,
            `"latex"`, and `"python"`.

        id : str, optional
            An DOM id used in place of the objects internal id.

        Returns
        -------
        dict
            A dictionary of strings whose keys indicate which portion of
            the embeddable output the value is.  Keys will vary for different
            `typ`.  For `"html"`, keys are `"html"` and `"js"` for HTML and
            and Javascript code, respectively.
        """
        if id is None: id = self.ID

        if typ == "html":
            return {'html': "<div id='%s' class='notapplicable'>[NO DATA or N/A]</div>" % id, 'js': ""}

        elif typ == "latex":
            return {'latex': "Not applicable"}

        elif typ == "python":
            return "Not Applicable"
        else:
            raise ValueError("NotApplicable render type not supported: %s" % typ)


class WorkspaceTable(WorkspaceOutput):
    """
    Encapsulates a table within a `Workspace` context.

    A base class which provides the logic required to take a
    single table-generating function and make it into a legitimate
    `WorkspaceOutput` object for using within workspaces.

    Parameters
    ----------
    ws : Workspace
        The workspace containing the new object.

    fn : function
        A table-creating function.

    args : various
        The arguments to `fn`.
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
        self.tables, self.switchboards, self.sbSwitchIndices, self.switchpos_map = \
            self.ws.switched_compute(self.tablefn, *self.initargs)

    def render(self, typ):
        """
        Renders this table into the specified format, specifically for embedding it within a larger document.

        Parameters
        ----------
        typ : str
            The format to render as.  Currently `"html"`, `"latex"`
            and `"python"` are supported.

        Returns
        -------
        dict
            A dictionary of strings giving the different portions of the
            embeddable output.  For `"html"`, keys are `"html"` and `"js"`.
            For `"latex"`, there is a single key `"latex"`.
        """
        resizable = self.options.get('resizable', True)
        autosize = self.options.get('autosize', 'none')
        precision = self.options.get('precision', None)
        switched_item_mode = self.options.get('switched_item_mode', 'inline')
        overrideIDs = self.options.get('switched_item_id_overrides', {})
        output_dir = self.options.get('output_dir', None)

        if precision is None:
            precDict = {'normal': 6, 'polar': 3, 'sci': 0}
        elif _compat.isint(precision):
            precDict = {'normal': precision, 'polar': precision, 'sci': precision}
        else:
            assert('normal' in precision), "Must at least specify 'normal' precision"
            p = precision['normal']
            precDict = {'normal': p,
                        'polar': precision.get('polar', p),
                        'sci': precision.get('sci', p)}

        ID = self.ID
        tableID = "table_" + ID

        if typ == "html":

            divHTML = []
            divIDs = []
            divJS = []

            for i, table in enumerate(self.tables):
                tableDivID = tableID + "_%d" % i
                if i in overrideIDs: tableDivID = overrideIDs[i]

                if isinstance(table, NotApplicable):
                    table_dict = table.render("html", tableDivID)
                else:
                    table_dict = table.render("html", table_id=tableDivID + "_tbl",
                                              tableclass="dataTable",
                                              precision=precDict['normal'],
                                              polarprecision=precDict['polar'],
                                              sciprecision=precDict['sci'],
                                              resizable=resizable, autosize=(autosize == "continual"),
                                              click_to_display=self.options['click_to_display'],
                                              link_to=self.options['link_to'],
                                              output_dir=output_dir)

                if switched_item_mode == 'separate files':
                    divJS.append(self._form_table_js(tableDivID, table_dict['html'], table_dict['js'], None))
                else:
                    #otherwise just add plot handers (table_dict['js']) to divJS for later
                    divJS.append(table_dict['js'])

                divHTML.append(table_dict['html'])
                divIDs.append(tableDivID)

            if switched_item_mode == 'inline':
                base = self._render_html(tableID, divHTML, None, divIDs, self.switchpos_map,
                                         self.switchboards, self.sbSwitchIndices, None,
                                         self.options.get('link_to', None), output_dir)  # no JS yet...
                js = self._form_table_js(tableID, base['html'], '\n'.join(divJS), base['js'])
                # creates JS for everything: plot creation, switchboard init, autosize
            elif switched_item_mode == 'separate files':
                base = self._render_html(tableID, divHTML, divJS, divIDs, self.switchpos_map,
                                         self.switchboards, self.sbSwitchIndices, None,
                                         self.options.get('link_to', None), output_dir, embed_figures=False)
                js = self._form_table_js(tableID, None, None, base['js'])
            else:
                raise ValueError("Invalid `switched_item_mode` render option: %s" %
                                 switched_item_mode)

            return {'html': base['html'], 'js': js}

        elif typ == "latex":

            render_includes = self.options.get('render_includes', True)
            leave_src = self.options.get('leave_includes_src', False)
            W, H = self.options.get('page_size', (6.5, 8.0))
            printer = _baseobjs.VerbosityPrinter(1)  # TEMP - add verbosity arg?

            #Note: in both cases output_dir needs to be the *relative* path
            # between the current directory and the output directory if
            # \includegraphics statements are to work.  If this isn't needed
            # (e.g. if just the standalone files are needed) then output_dir
            # can be an absolute path as well.

            # table rendering returned in ret dict
            if switched_item_mode == 'inline':
                # Assume current directory is where generated latex
                # code will reside and output_dir is where figs go.
                tablefig_output_dir = output_dir  # (can be None, in
                #which case an error will be raised if table has figs)
                render_dir = None  # no need to chdir for table render

            #render each switched "item" as a separate standalone file
            elif switched_item_mode == 'separate files':
                # Assume current directory is where \includegraphics{...}
                # latex will go, and that separate table TEX files *and*
                # figures go in `output_dir`.  The table latex is given an
                # output_dir of '.' because figure files will be in the same
                # directory.
                assert(output_dir), "Cannot render a table as 'latex' with " + \
                    "switched items as separate files without a valid " + \
                    "'output_dir' render option"
                tablefig_output_dir = '.'
                render_dir = output_dir

            else:
                raise ValueError("Invalid `switched_item_mode` render option: %s" %
                                 switched_item_mode)

            if render_dir is not None and not _os.path.exists(render_dir):
                _os.mkdir(render_dir)

            cwd = _os.getcwd()
            latex_list = []
            for i, table in enumerate(self.tables):
                tableDivID = tableID + "_%d" % i
                if i in overrideIDs: tableDivID = overrideIDs[i]
                if isinstance(table, NotApplicable): continue

                if render_dir: _os.chdir(render_dir)
                table_dict = table.render("latex",
                                          precision=precDict['normal'],
                                          polarprecision=precDict['polar'],
                                          sciprecision=precDict['sci'],
                                          output_dir=tablefig_output_dir,
                                          render_includes=render_includes)
                if render_dir: _os.chdir(cwd)

                if switched_item_mode == 'inline':
                    latex_list.append(table_dict['latex'])

                elif switched_item_mode == 'separate files':
                    if render_includes or leave_src:
                        d = {'toLatex': table_dict['latex']}
                        _merge.merge_latex_template(d, "standalone.tex",
                                                    _os.path.join(str(output_dir), "%s.tex" % tableDivID))

                    if render_includes:
                        assert('latex_cmd' in self.options and self.options['latex_cmd']), \
                            "Cannot render latex include files without a valid 'latex_cmd' render option"

                        try:
                            _os.chdir(render_dir)
                            latex_cmd = self.options['latex_cmd']
                            latex_call = [latex_cmd] + self.options.get('latex_flags', []) \
                                + ["%s.tex" % tableDivID]
                            stdout, stderr, returncode = _merge.process_call(latex_call)
                            _merge.evaluate_call(latex_call, stdout, stderr, returncode, printer)
                            if not _os.path.isfile("%s.pdf" % tableDivID):
                                raise Exception("File %s.pdf was not created by %s"
                                                % (tableDivID, latex_cmd))
                            if not leave_src: _os.remove("%s.tex" % tableDivID)
                            _os.remove("%s.log" % tableDivID)
                            _os.remove("%s.aux" % tableDivID)
                        except _subprocess.CalledProcessError as e:
                            printer.error("%s returned code %d " % (latex_cmd, e.returncode)
                                          + "trying to render standalone %s.tex. " % tableDivID
                                          + "Check %s.log to see details." % tableDivID)
                        finally:
                            _os.chdir(cwd)

                        latex_list.append("\\includegraphics[width=%.2fin,height=%.2fin,keepaspectratio]{%s}" %
                                          (W, H, _os.path.join(str(output_dir), "%s.pdf" % tableDivID)))
                    elif leave_src:
                        latex_list.append("\\input{%s}" % _os.path.join(str(output_dir), "%s.tex" % tableDivID))
                    else:
                        latex_list.append("%% Didn't generated anything for tableID=%s" % tableDivID)
                else:
                    raise ValueError("Invalid `switched_item_mode` render option: %s" %
                                     switched_item_mode)  # pragma: no cover

            return {'latex': "\n".join(latex_list)}

        elif typ == "python":

            if switched_item_mode == 'separate files':
                assert(output_dir), "Cannot render tables as 'python' in separate" \
                    + " files without a valid 'output_dir' render option"

            tables_python = _collections.OrderedDict()
            for i, table in enumerate(self.tables):
                if isinstance(table, NotApplicable): continue
                tableDivID = tableID + "_%d" % i
                if i in overrideIDs: tableDivID = overrideIDs[i]

                if switched_item_mode == "inline":
                    table_dict = table.render("python", output_dir=None)
                    tables_python[tableDivID] = table_dict['python']
                elif switched_item_mode == "separate files":
                    outputFilename = _os.path.join(str(output_dir), "%s.pkl" % tableDivID)
                    table_dict = table.render("python", output_dir=output_dir)
                    #( setting output_dir generates separate files for plots in table )
                    table_dict['python'].to_pickle(outputFilename)  # a DataFrame
                    tables_python[tableDivID] = "df_%s = pd.read_pickle('%s')" \
                                                % (tableDivID, outputFilename)
                else:
                    raise ValueError("Invalid `switched_item_mode` render option: %s" %
                                     switched_item_mode)

            return {'python': tables_python}

        else:
            assert(len(self.tables) == 1), \
                "Can only render %s format for a non-switched table" % typ
            return {typ: self.tables[0].render(typ)}

    def saveas(self, filename, index=None, verbosity=0):
        """
        Saves this workspace table object to a file.

        The type of file that is saved is determined automatically by the
        extension of `filename`.  Recognized extensions are `pdf` (PDF),
        `tex` (LaTeX), `pkl` (Python pickle) and `html` (HTML).  Since this
        object may contain different instances of its data based on switch
        positions, when their are multiple instances the user must specify
        the `index` argument to disambiguate.

        Parameters
        ----------
        filename : str
            The destination filename.  Its extension determines what type
            of file is saved.

        index : int, optional
            An absolute index into the list of different switched "versions"
            of this object's data.  In most cases, the object being saved
            doesn't depend on any switch boards and has only a single "version",
            in which caes this can be left as the default.

        verbosity : int, optional
            Controls the level of detail printed to stdout.

        Returns
        -------
        None
        """
        N = len(self.tables)

        if filename.endswith(".html"):
            if index is None:
                if N == 1:
                    index = 0
                else:
                    raise ValueError("Must supply `index` argument for a non-trivially-switched WorkspaceTable")

            saved_switchposmap = self.switchpos_map
            saved_switchboards = self.switchboards
            saved_switchinds = self.sbSwitchIndices

            #Temporarily pretend we don't depend on any switchboards and
            # by default display the user-specified index
            self.switchboards = []
            self.sbSwitchIndices = []
            self.switchpos_map = {(): index}

            qtys = {'title': _os.path.splitext(_os.path.basename(str(filename)))[0],
                    'singleItem': self}
            _merge.merge_jinja_template(qtys, filename, template_name="standalone.html",
                                        verbosity=verbosity)

            self.switchpos_map = saved_switchposmap
            self.switchboards = saved_switchboards
            self.sbSwitchIndices = saved_switchinds

        elif filename.endswith(".pkl"):
            if index is None and N == 1: index = 0
            overrides = {i: "index%d" % i for i in range(N)}
            self.set_render_options(switched_item_mode="inline",
                                    switched_item_id_overrides=overrides)
            render_out = self.render("python")

            if index is not None:  # just pickle a single element
                to_pickle = render_out['python']['index%d' % index]
            else:  # pickle dictionary of all indices
                to_pickle = render_out['python']

            with open(str(filename), 'wb') as f:
                _pickle.dump(to_pickle, f)

        else:
            if index is None:
                if N == 1:
                    index = 0
                else:
                    raise ValueError("Must supply `index` argument for a non-trivially-switched WorkspaceTable")

            output_dir = _os.path.dirname(str(filename))
            filebase, ext = _os.path.splitext(_os.path.basename(str(filename)))

            tempDir = _os.path.join(str(output_dir), "%s_temp" % filebase)
            if not _os.path.exists(tempDir): _os.mkdir(tempDir)

            self.set_render_options(switched_item_mode="separate files",
                                    switched_item_id_overrides={index: filebase},
                                    output_dir=tempDir)

            if ext == ".tex":
                self.set_render_options(render_includes=False,
                                        leave_includes_src=True)
            elif ext == ".pdf":
                self.set_render_options(render_includes=True,
                                        leave_includes_src=False)
            else:
                raise ValueError("Unknown file type for %s" % filename)

            self.render("latex")  # renders everything in temp dir
            _os.rename(_os.path.join(str(tempDir), "%s%s" % (filebase, ext)),
                       _os.path.join(str(output_dir), "%s%s" % (filebase, ext)))

            #remove all other files
            _shutil.rmtree(tempDir)

    def _form_table_js(self, table_id, table_html, table_plot_handlers,
                       switchboard_init_js):

        resizable = self.options.get('resizable', True)
        autosize = self.options.get('autosize', 'none')
        create_table_plots = bool(table_plot_handlers is not None)
        queue_math_render = bool(table_html and '$' in table_html
                                 and self.options.get('render_math', True))
        add_autosize_handler = bool(switchboard_init_js is not None)
        #only add ws-table-wide autosize handler when initializing the table's switchboard (once
        # per workspace table)

        content = ""

        # put plot handlers *above* switchboard init JS
        if table_plot_handlers: content += table_plot_handlers
        if switchboard_init_js: content += switchboard_init_js

        #Table initialization javascript: this will either be within the math-rendering (queued) function
        # (if '$' in ret['html']) or else at the *end* of the ready handler (if no math needed rendering).
        init_table_js = ''
        if create_table_plots and resizable:  # make a resizable widget on *entire* plot
            # (will only act on first call, but wait until first plots are created)
            init_table_js += '    make_wstable_resizable("{table_id}");\n'.format(table_id=table_id)
        if add_autosize_handler and autosize == "continual":
            init_table_js += '    make_wsobj_autosize("{table_id}");\n'.format(table_id=table_id)
        if create_table_plots:
            init_table_js += '    trigger_wstable_plot_creation("{table_id}",{initautosize});\n'.format(
                table_id=table_id, initautosize=str(autosize in ("initial", "continual")).lower())

        if queue_math_render:
            # then there is math text that needs rendering,
            # so queue this, *then* trigger plot creation
            content += ('  plotman.enqueue(function() {{ \n'
                        '    renderMathInElement(document.getElementById("{table_id}"), {{ delimiters: [\n'
                        '             {{left: "$$", right: "$$", display: true}},\n'
                        '             {{left: "$", right: "$", display: false}},\n'
                        '             ] }} );\n').format(table_id=table_id)
            content += init_table_js
            content += '  }}, "Rendering math in {table_id}" );\n'.format(table_id=table_id)  # end enqueue
        else:
            #Note: this MUST be below plot handler init, when it triggers plot creation
            content += init_table_js

        return self._create_onready_handler(content, table_id)


class WorkspacePlot(WorkspaceOutput):
    """
    Encapsulates a plot within a `Workspace` context.

    A base class which provides the logic required to take a
    single plot.ly figure-generating function and make it into a
    legitimate `WorkspaceOutput` object for using within workspaces.

    Parameters
    ----------
    ws : Workspace
        The workspace containing the new object.

    fn : function
        A table-creating function.

    args : various
        The arguments to `fn`.
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
        '''
        # LSaldyt: removed plotfn for easier pickling? It doesn't seem to be used anywhere
        self.plotfn = fn
        self.initargs = args
        self.figs, self.switchboards, self.sbSwitchIndices, self.switchpos_map = \
            self.ws.switched_compute(self.plotfn, *self.initargs)
        '''
        self.initargs = args
        self.figs, self.switchboards, self.sbSwitchIndices, self.switchpos_map = \
            self.ws.switched_compute(fn, *self.initargs)

    def render(self, typ="html", id=None):
        """
        Renders this plot into the specified format, specifically for embedding it within a larger document.

        Parameters
        ----------
        typ : str
            The format to render as.  Currently `"html"`, `"latex"`
            and `"python"` are supported.

        id : str, optional
            A base id to use when rendering.  If None, the object's
            persistent id is used, which usually what you want.

        Returns
        -------
        dict
            A dictionary of strings giving the HTML and Javascript portions
            of the embeddable output.  Keys are `"html"` and `"js"`.
        """
        resizable = self.options.get('resizable', True)
        valign = self.options.get('valign', 'top')
        overrideIDs = self.options.get('switched_item_id_overrides', {})
        switched_item_mode = self.options.get('switched_item_mode', 'inline')
        output_dir = self.options.get('output_dir', None)

        if valign == 'top':
            abswrap_cls = 'abswrap'
            relwrap_cls = 'relwrap'
        elif valign == 'bottom':
            abswrap_cls = 'bot_abswrap'
            relwrap_cls = 'bot_relwrap'
        else:
            raise ValueError("Invalid 'valign' value: %s" % valign)

        if id is None: id = self.ID
        plotID = "plot_" + id

        if typ == "html":

            #def getPlotlyDivID(html):
            #    #could make this more robust using lxml or something later...
            #    iStart = html.index('div id="')
            #    iEnd = html.index('"', iStart+8)
            #    return html[iStart+8:iEnd]

            ##pick "master" plot, whose resizing dictates the resizing of other plots,
            ## as the largest-height plot.
            #iMaster = None; maxH = 0;
            #for i, fig in enumerate(self.figs):
            #    if isinstance(fig, NotApplicable):
            #        continue
            # NOTE: master=None below, but it's unclear whether this will later be needed.

            # "handlers only" mode is when plot is embedded in something
            #  larger (e.g. a table) that takes responsibility for putting
            #  the JS returned into an on-ready handler and triggering the
            #  initialization and creation of the plots.
            handlersOnly = bool(resizable == "handlers only")

            divHTML = []
            divIDs = []
            divJS = []

            for i, fig in enumerate(self.figs):
                plotDivID = plotID + "_%d" % i
                if i in overrideIDs: plotDivID = overrideIDs[i]

                if isinstance(fig, NotApplicable):
                    fig_dict = fig.render(typ, plotDivID)
                else:
                    #use auto-sizing (fluid layout)
                    #fig.plotlyfig.update_layout(template=DEFAULT_PLOTLY_TEMPLATE)  #slow: set default theme in plot_ex
                    fig_dict = _plotly_ex.plot_ex(
                        fig.plotlyfig, show_link=False, resizable=resizable,
                        lock_aspect_ratio=True, master=True, validate=VALIDATE_PLOTLY,  # bool(i==iMaster)
                        click_to_display=self.options['click_to_display'],
                        link_to=self.options['link_to'], link_to_id=plotDivID,
                        rel_figure_dir=_os.path.basename(
                            str(output_dir)) if not (output_dir in (None, False)) else None)

                if switched_item_mode == 'separate files':
                    assert(handlersOnly is False)  # doesn't make sense to put only handlers in a separate file
                    divJS.append(self._form_plot_js(plotDivID, fig_dict['js'], None))
                else:
                    divJS.append(fig_dict['js'])

                divIDs.append(plotDivID)
                divHTML.append("<div class='%s'>%s</div>" % (abswrap_cls, fig_dict['html']))

            if switched_item_mode == 'inline':
                base = self._render_html(plotID, divHTML, None, divIDs, self.switchpos_map,
                                         self.switchboards, self.sbSwitchIndices, [relwrap_cls])
                # Don't link_to b/c plots will all have download buttons
                if handlersOnly:
                    js = '\n'.join(divJS) + base['js']  # insert plot handlers above switchboard init JS
                else:
                    js = self._form_plot_js(plotID, '\n'.join(divJS), base['js'])

            elif switched_item_mode == 'separate files':
                base = self._render_html(plotID, divHTML, divJS, divIDs, self.switchpos_map,
                                         self.switchboards, self.sbSwitchIndices, [relwrap_cls],
                                         None, self.options['output_dir'], embed_figures=False)
                js = self._form_plot_js(plotID, None, base['js'])
            else:
                raise ValueError("Invalid `switched_item_mode` render option: %s" %
                                 switched_item_mode)

            return {'html': base['html'], 'js': js}

        elif typ == "latex":
            assert('output_dir' in self.options and self.options['output_dir']), \
                "Cannot render a plot as 'latex' without a valid " +\
                "'output_dir' render option (regardless of switched_item_mode)"

            if switched_item_mode not in ('inline', 'separate files'):
                raise ValueError("Invalid `switched_item_mode` render option: %s" %
                                 switched_item_mode)  # for uniformity with other cases,
                # even though it's not used.

            from .mpl_colormaps import plotly_to_matplotlib as _plotly_to_matplotlib

            output_dir = self.options['output_dir']
            maxW, maxH = self.options.get('page_size', (6.5, 8.0))
            includes = []
            for i, fig in enumerate(self.figs):
                if isinstance(fig, NotApplicable): continue
                plotDivID = plotID + "_%d" % i
                if i in overrideIDs: plotDivID = overrideIDs[i]

                if self.options.get('render_includes', True):
                    filename = _os.path.join(str(output_dir), plotDivID + ".pdf")
                    _plotly_to_matplotlib(fig, filename)

                    W, H = maxW, maxH
                    if 'mpl_fig_size' in fig.metadata:  # added by plotly_to_matplotlib call above
                        figW, figH = fig.metadata['mpl_fig_size']  # gives the "normal size" of the figure
                        W = min(W, figW)
                        W = min(H, figH)
                        del fig.metadata['mpl_fig_size']

                    includes.append("\\includegraphics[width=%.2fin,height=%.2fin,keepaspectratio]{%s}" %
                                    (W, H, filename))
                else:
                    includes.append("%% Didn't render plotID=%s" % plotDivID)
            return {'latex': '\n'.join(includes)}

        elif typ == "python":

            if switched_item_mode == 'separate files':
                assert(output_dir), "Cannot render plots as 'python' in separate" \
                    + " files without a valid 'output_dir' render option"

            plots_python = _collections.OrderedDict()
            for i, fig in enumerate(self.figs):
                plotDivID = plotID + "_%d" % i
                if i in overrideIDs: plotDivID = overrideIDs[i]
                if isinstance(fig, NotApplicable): continue

                if fig.pythonvalue is not None:
                    data = {'value': fig.pythonvalue}
                    if "pythonErrorBar" in fig.metadata:
                        data['errorbar'] = fig.metadata['pythonErrorBar']
                else:
                    data = {'value': "Opaque Figure"}

                if switched_item_mode == "inline":
                    plots_python[plotDivID] = data
                elif switched_item_mode == "separate files":
                    outputFilename = _os.path.join(str(output_dir), "%s.pkl" % plotDivID)
                    with open(outputFilename, "wb") as fPkl:
                        _pickle.dump(data, fPkl)
                    plots_python[plotDivID] = "data_%s = pickle.load(open('%s','rb'))" \
                        % (plotDivID, outputFilename)
                else:
                    raise ValueError("Invalid `switched_item_mode` render option: %s" %
                                     switched_item_mode)

            return {'python': plots_python}

        else:
            raise NotImplementedError("Invalid rendering format: %s" % typ)

    def saveas(self, filename, index=None, verbosity=0):
        """
        Saves this workspace plot object to a file.

        The type of file that is saved is determined automatically by the
        extension of `filename`.  Recognized extensions are `pdf` (PDF),
        `pkl` (Python pickle) and `html` (HTML).  Since this object may
        contain different instances of its data based on switch positions,
        when their are multiple instances the user must specify the `index`
        argument to disambiguate.

        Parameters
        ----------
        filename : str
            The destination filename.  Its extension determines what type
            of file is saved.

        index : int, optional
            An absolute index into the list of different switched "versions"
            of this object's data.  In most cases, the object being saved
            doesn't depend on any switch boards and has only a single "version",
            in which caes this can be left as the default.

        verbosity : int, optional
            Controls the level of detail printed to stdout.

        Returns
        -------
        None
        """
        N = len(self.figs)

        if filename.endswith(".html"):
            #Note: Same as WorkspaceTable except for N
            if index is None:
                if N == 1:
                    index = 0
                else:
                    raise ValueError("Must supply `index` argument for a non-trivially-switched WorkspacePlot")

            saved_switchposmap = self.switchpos_map
            saved_switchboards = self.switchboards
            saved_switchinds = self.sbSwitchIndices

            #Temporarily pretend we don't depend on any switchboards and
            # by default display the user-specified index
            self.switchboards = []
            self.sbSwitchIndices = []
            self.switchpos_map = {(): index}

            qtys = {'title': _os.path.splitext(_os.path.basename(str(filename)))[0],
                    'singleItem': self}
            _merge.merge_jinja_template(qtys, filename, template_name="standalone.html",
                                        verbosity=verbosity)

            self.switchpos_map = saved_switchposmap
            self.switchboards = saved_switchboards
            self.sbSwitchIndices = saved_switchinds

        elif filename.endswith(".pkl"):
            #Note: Same as WorkspaceTable except for N
            if index is None and N == 1: index = 0
            overrides = {i: "index%d" % i for i in range(N)}
            self.set_render_options(switched_item_mode="inline",
                                    switched_item_id_overrides=overrides)
            render_out = self.render("python")

            if index is not None:  # just pickle a single element
                to_pickle = render_out['python']['index%d' % index]
            else:  # pickle dictionary of all indices
                to_pickle = render_out['python']

            with open(filename, 'wb') as f:
                _pickle.dump(to_pickle, f)

        elif filename.endswith(".tex"):
            raise ValueError("Cannot save a WorkspacePlot as LaTeX - try PDF.")

        elif filename.endswith(".pdf"):
            from .mpl_colormaps import plotly_to_matplotlib as _plotly_to_matplotlib

            if index is None:
                if N == 1:
                    index = 0
                else:
                    raise ValueError("Must supply `index` argument for a non-trivially-switched WorkspacePlot")
            _plotly_to_matplotlib(self.figs[index], filename)

        else:
            raise ValueError("Unknown file type for %s" % filename)

    def _form_plot_js(self, plot_id, plot_handlers, switchboard_init_js):

        resizable = self.options.get('resizable', True)
        autosize = self.options.get('autosize', 'none')
        create_plots = bool(plot_handlers is not None)
        add_autosize_handler = bool(switchboard_init_js is not None)
        #only add ws-plot-wide autosize handler when initializing the plot's switchboard (once
        # per workspace table)

        content = ""

        #put plot handlers above switchboard init JS
        if plot_handlers: content += plot_handlers
        if switchboard_init_js: content += switchboard_init_js

        if resizable:  # make a resizable widget
            content += 'make_wsplot_resizable("{plot_id}");\n'.format(plot_id=plot_id)
        if add_autosize_handler and autosize == "continual":  # add window resize handler
            content += 'make_wsobj_autosize("{plot_id}");\n'.format(plot_id=plot_id)
        if create_plots:
            #trigger init & create of plots
            content += 'trigger_wsplot_plot_creation("{plot_id}",{initautosize});\n'.format(
                plot_id=plot_id, initautosize=str(autosize in ("initial", "continual")).lower())

        return self._create_onready_handler(content, plot_id)


class WorkspaceText(WorkspaceOutput):
    """
    Encapsulates a block of text within a `Workspace` context.

    A base class which provides the logic required to take a
    single text-generating function and make it into a legitimate
    `WorkspaceOutput` object for using within workspaces.

    Parameters
    ----------
    ws : Workspace
        The workspace containing the new object.

    fn : function
        A text-creating function.

    args : various
        The arguments to `fn`.
    """

    def __init__(self, ws, fn, *args):
        """
        Create a new WorkspaceText object.  Usually not called directly.

        Parameters
        ----------
        ws : Workspace
            The workspace containing the new object.

        fn : function
            A text-creating function.

        args : various
            The arguments to `fn`.
        """
        super(WorkspaceText, self).__init__(ws)
        self.textfn = fn
        self.initargs = args
        self.texts, self.switchboards, self.sbSwitchIndices, self.switchpos_map = \
            self.ws.switched_compute(self.textfn, *self.initargs)

    def render(self, typ):
        """
        Renders this text block into the specified format, specifically for embedding it within a larger document.

        Parameters
        ----------
        typ : str
            The format to render as.  Currently `"html"`, `"latex"`
            and `"python"` are supported.

        Returns
        -------
        dict
            A dictionary of strings giving the different portions of the
            embeddable output.  For `"html"`, keys are `"html"` and `"js"`.
            For `"latex"`, there is a single key `"latex"`.
        """

        switched_item_mode = self.options.get('switched_item_mode', 'inline')
        overrideIDs = self.options.get('switched_item_id_overrides', {})
        output_dir = self.options.get('output_dir', None)

        ID = self.ID
        textID = "text_" + ID

        if typ == "html":

            divHTML = []
            divIDs = []
            divJS = []

            for i, text in enumerate(self.texts):
                textDivID = textID + "_%d" % i
                if i in overrideIDs: textDivID = overrideIDs[i]

                if isinstance(text, NotApplicable):
                    text_dict = text.render("html", textDivID)
                else:
                    text_dict = text.render("html", textDivID)

                if switched_item_mode == 'separate files':
                    divJS.append(self._form_text_js(textDivID, text_dict['html'], None))
                #else: divJS is unused

                divHTML.append(text_dict['html'])
                divIDs.append(textDivID)

            if switched_item_mode == 'inline':
                base = self._render_html(textID, divHTML, None, divIDs, self.switchpos_map,
                                         self.switchboards, self.sbSwitchIndices, None,
                                         self.options.get('link_to', None), output_dir)  # no JS yet...
                js = self._form_text_js(textID, base['html'], base['js'])
                # creates JS for everything: plot creation, switchboard init, autosize

            elif switched_item_mode == 'separate files':
                base = self._render_html(textID, divHTML, divJS, divIDs, self.switchpos_map,
                                         self.switchboards, self.sbSwitchIndices, None,
                                         self.options.get('link_to', None), output_dir, embed_figures=False)
                js = self._form_text_js(textID, None, base['js'])
            else:
                raise ValueError("Invalid `switched_item_mode` render option: %s" %
                                 switched_item_mode)

            return {'html': base['html'], 'js': js}

        elif typ == "latex":

            leave_src = self.options.get('leave_includes_src', False)
            render_includes = self.options.get('render_includes', True)
            W, H = self.options.get('page_size', (6.5, 8.0))
            printer = _baseobjs.VerbosityPrinter(1)  # TEMP - add verbosity arg?

            #Note: in both cases output_dir needs to be the *relative* path
            # between the current directory and the output directory if
            # \includegraphics statements are to work.  If this isn't needed
            # (e.g. if just the standalone files are needed) then output_dir
            # can be an absolute path as well.

            cwd = _os.getcwd()
            latex_list = []
            for i, text in enumerate(self.texts):
                textDivID = textID + "_%d" % i
                if i in overrideIDs: textDivID = overrideIDs[i]
                if isinstance(text, NotApplicable): continue

                text_dict = text.render("latex")

                if switched_item_mode == 'inline':
                    latex_list.append(text_dict['latex'])

                elif switched_item_mode == 'separate files':
                    if render_includes or leave_src:
                        d = {'toLatex': text_dict['latex']}
                        _merge.merge_latex_template(d, "standalone.tex",
                                                    _os.path.join(str(output_dir), "%s.tex" % textDivID))

                    if render_includes:
                        render_dir = output_dir
                        assert('latex_cmd' in self.options and self.options['latex_cmd']), \
                            "Cannot render latex include files without a valid 'latex_cmd' render option"

                        try:
                            _os.chdir(render_dir)
                            latex_cmd = self.options['latex_cmd']
                            latex_call = [latex_cmd] + self.options.get('latex_flags', []) \
                                + ["%s.tex" % textDivID]
                            stdout, stderr, returncode = _merge.process_call(latex_call)
                            _merge.evaluate_call(latex_call, stdout, stderr, returncode, printer)
                            if not _os.path.isfile("%s.pdf" % textDivID):
                                raise Exception("File %s.pdf was not created by %s"
                                                % (textDivID, latex_cmd))
                            if not leave_src: _os.remove("%s.tex" % textDivID)
                            _os.remove("%s.log" % textDivID)
                            _os.remove("%s.aux" % textDivID)
                        except _subprocess.CalledProcessError as e:
                            printer.error("%s returned code %d " % (latex_cmd, e.returncode)
                                          + "trying to render standalone %s.tex. " % textDivID
                                          + "Check %s.log to see details." % textDivID)
                        finally:
                            _os.chdir(cwd)

                        latex_list.append("\\includegraphics[width=%.2fin,height=%.2fin,keepaspectratio]{%s}" %
                                          (W, H, _os.path.join(str(output_dir), "%s.pdf" % textDivID)))
                    elif leave_src:
                        latex_list.append("\\input{%s}" % _os.path.join(str(output_dir), "%s.tex" % textDivID))
                    else:
                        latex_list.append("%% Didn't generated anything for textID=%s" % textDivID)
                else:
                    raise ValueError("Invalid `switched_item_mode` render option: %s" %
                                     switched_item_mode)

            return {'latex': "\n".join(latex_list)}

        elif typ == "python":

            if switched_item_mode == 'separate files':
                assert(output_dir), "Cannot render texts as 'python' in separate" \
                    + " files without a valid 'output_dir' render option"

            texts_python = _collections.OrderedDict()
            for i, text in enumerate(self.texts):
                if isinstance(text, NotApplicable): continue
                textDivID = textID + "_%d" % i
                if i in overrideIDs: textDivID = overrideIDs[i]

                text_dict = text.render("python")

                if switched_item_mode == "inline":
                    texts_python[textDivID] = text_dict['python']
                elif switched_item_mode == "separate files":
                    outputFilename = _os.path.join(str(output_dir), "%s.pkl" % textDivID)
                    with open(outputFilename, 'wb') as f:
                        _pickle.dump(text_dict['python'], f)
                    texts_python[textDivID] = "text_%s = pickle.load(open('%s','rb'))" \
                        % (textDivID, outputFilename)
                else:
                    raise ValueError("Invalid `switched_item_mode` render option: %s" %
                                     switched_item_mode)

            return {'python': texts_python}

        else:
            assert(len(self.texts) == 1), \
                "Can only render %s format for a non-switched text block" % typ
            return {typ: self.texts[0].render(typ)}

    def saveas(self, filename, index=None, verbosity=0):
        """
        Saves this workspace text block object to a file.

        The type of file that is saved is determined automatically by the
        extension of `filename`.  Recognized extensions are `pdf` (PDF),
        `tex` (LaTeX), `pkl` (Python pickle) and `html` (HTML).  Since this
        object may contain different instances of its data based on switch
        positions, when their are multiple instances the user must specify
        the `index` argument to disambiguate.

        Parameters
        ----------
        filename : str
            The destination filename.  Its extension determines what type
            of file is saved.

        index : int, optional
            An absolute index into the list of different switched "versions"
            of this object's data.  In most cases, the object being saved
            doesn't depend on any switch boards and has only a single "version",
            in which caes this can be left as the default.

        verbosity : int, optional
            Controls the level of detail printed to stdout.

        Returns
        -------
        None
        """
        N = len(self.texts)

        if filename.endswith(".html"):
            if index is None:
                if N == 1:
                    index = 0
                else:
                    raise ValueError("Must supply `index` argument for a non-trivially-switched WorkspaceText")

            saved_switchposmap = self.switchpos_map
            saved_switchboards = self.switchboards
            saved_switchinds = self.sbSwitchIndices

            #Temporarily pretend we don't depend on any switchboards and
            # by default display the user-specified index
            self.switchboards = []
            self.sbSwitchIndices = []
            self.switchpos_map = {(): index}

            qtys = {'title': _os.path.splitext(_os.path.basename(str(filename)))[0],
                    'singleItem': self}
            _merge.merge_jinja_template(qtys, filename, template_name="standalone.html",
                                        verbosity=verbosity)

            self.switchpos_map = saved_switchposmap
            self.switchboards = saved_switchboards
            self.sbSwitchIndices = saved_switchinds

        elif filename.endswith(".pkl"):
            if index is None and N == 1: index = 0
            overrides = {i: "index%d" % i for i in range(N)}
            self.set_render_options(switched_item_mode="inline",
                                    switched_item_id_overrides=overrides)
            render_out = self.render("python")

            if index is not None:  # just pickle a single element
                to_pickle = render_out['python']['index%d' % index]
            else:  # pickle dictionary of all indices
                to_pickle = render_out['python']

            with open(filename, 'wb') as f:
                _pickle.dump(to_pickle, f)

        else:
            if index is None:
                if N == 1:
                    index = 0
                else:
                    raise ValueError("Must supply `index` argument for a non-trivially-switched WorkspaceText")

            output_dir = _os.path.dirname(filename)
            filebase, ext = _os.path.splitext(_os.path.basename(filename))

            tempDir = _os.path.join(str(output_dir), "%s_temp" % filebase)
            if not _os.path.exists(tempDir): _os.mkdir(tempDir)

            self.set_render_options(switched_item_mode="separate files",
                                    switched_item_id_overrides={index: filebase},
                                    output_dir=tempDir)

            if ext == ".tex":
                self.set_render_options(render_includes=False,
                                        leave_includes_src=True)
            elif ext == ".pdf":
                self.set_render_options(render_includes=True,
                                        leave_includes_src=False)
            else:
                raise ValueError("Unknown file type for %s" % filename)

            self.render("latex")  # renders everything in temp dir
            _os.rename(_os.path.join(str(tempDir), "%s%s" % (filebase, ext)),
                       _os.path.join(str(output_dir), "%s%s" % (filebase, ext)))

            #remove all other files
            _shutil.rmtree(tempDir)

    def _form_text_js(self, text_id, text_html, switchboard_init_js):

        content = ""
        if switchboard_init_js: content += switchboard_init_js

        queue_math_render = bool(text_html and '$' in text_html
                                 and self.options.get('render_math', True))

        if text_html is not None:
            init_text_js = (
                'el = $("#{textid}");\n'
                'if(el.hasClass("pygsti-wsoutput-group")) {{\n'
                '  el.children("div.single_switched_value").each( function(i,el){{\n'
                '    CollapsibleLists.applyTo( $(el).find("ul").first()[0] );\n'
                '  }});\n'
                '}} else if(el.hasClass("single_switched_value")){{\n'
                '  CollapsibleLists.applyTo(el[0]);\n'
                '}}\n'
                'caption = el.closest("figure").children("figcaption:first");\n'
                'caption.css("width", Math.round(el.width()*0.9) + "px");\n'
            ).format(textid=text_id)
        else:
            init_text_js = ""  # no per-div init needed

        if queue_math_render:
            # then there is math text that needs rendering,
            # so queue this, *then* trigger plot creation
            content += ('  plotman.enqueue(function() {{ \n'
                        '    renderMathInElement(document.getElementById("{text_id}"), {{ delimiters: [\n'
                        '             {{left: "$$", right: "$$", display: true}},\n'
                        '             {{left: "$", right: "$", display: false}},\n'
                        '             ] }} );\n').format(text_id=text_id)
            content += init_text_js
            content += '  }}, "Rendering math in {text_id}" );\n'.format(text_id=text_id)  # end enqueue
        else:
            content += init_text_js

        return self._create_onready_handler(content, text_id)
