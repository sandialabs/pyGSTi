from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines the Workspace class and supporting functionality."""

#import os  as _os
#import re  as _re
#import time as _time
#import subprocess  as _subprocess
#import collections as _collections
#import matplotlib  as _matplotlib
#import itertools   as _itertools
#import copy as _copy
#
#from ..             import objects              as _objs
#from ..objects      import gatestring           as _gs
#from ..objects      import VerbosityPrinter
#from ..construction import spamspecconstruction as _ssc
#from ..algorithms   import gaugeopt_to_target   as _optimizeGauge
#from ..algorithms   import contract             as _contract
#from ..tools        import listtools            as _lt
#from ..             import _version
#
#from . import latex      as _latex
#from . import generation as _generation
#from . import plotting   as _plotting
#
#from .resultcache import ResultCache as _ResultCache

#TODO:
# convert plots & tables -> plot.ly
# separate plot calculation from rendering
# move table and plot *calculation* fns to Workspace (?) and use a calc-cache of some sort
# remove caching & frills from Results object
# add Report classes which use Workspace to construct std reports
# move any new classes to "objects/" ?

#Example:
# w = Workspace( options, autodisplay="widgets" )
# ds = w.load_dataset("file")
# gs = w.Selector([gs1,gs2,gs3], "slider")
# tab = w.MyTable(gs)
# plot = w.MyPlot(gs, ds)
# w.display([gs,plot], "widgets") #if autodisplay is off? layout?
# OR
# gs.selected = 1
# w.display([plot], "static") #if autodisplay is off? layout?
import itertools as _itertools
import collections as _collections
import os as _os
import numpy as _np
import uuid as _uuid
import random as _random

import plotly.offline as _plotly_offline
from .plotly_offline_fixed import plot_ex as _plot
from plotly.offline.offline import get_plotlyjs as _get_plotlyjs
#from IPython.display import clear_output as _clear_output

def _is_hashable(x):
    try:
        dct = { x: 0 }
    except TypeError:
        return False
    return True


import json, collections
class DigestEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, _np.ndarray):
            return hash(o.tostring())
        #if isinstance(o, dict) or isinstance(o,collections.OrderedDict):
        #    pairs = tuple([(k,v) for k,v in dict.items()])
        #    return json.JSONEncoder.default(self, pairs)
        if hasattr(o, 'digest_hash'):
            return o.digest_hash()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)

def call_key(obj, args):
    """ obj can be either a function or a class """
    key = [obj.__name__]
    for arg in args:
        if _is_hashable(arg):
            key.append( arg )
        elif hasattr(arg, 'digest_hash'):
            #Use a digest hash if one is available
            key.append( arg.digest_hash() ) #backup for un-hashable objects e.g. GateSet, DataSet, etc.
            #key.append( id(arg) ) #backup for un-hashable objects e.g. GateSet, DataSet, etc.
        else:
            try:
                json_hash = hash(json.dumps(arg, cls=DigestEncoder, sort_keys=True))
                key.append(json_hash)
            except:
            #    print("WARNING: Object of type %s and val %s is not " % (str(type(arg)),str(arg))
            #          + "hashable, so cannot be used as an argument"
            #          + "in cached computation calls")
                raise TypeError

    return tuple(key) #so hashable

def randomID():
    return str(int(10000*_random.random()))
    #return str(_uuid.uuid4().hex) #alternative


class Workspace(object):
    """
    Encapsulates XXX
    """

    def __init__(self):
        """
        Initialize a Workspace object.
        """

        self.outputObjs = {} #cache of WorkspaceOutput objects (hashable by call_keys)
        self.compCache = {}  # cache of computation function results (hashable by call_keys)

        def makefactory(cls):
            #TODO: set signature of factor_fn to cls.__init__
            def factory_fn(*args,**kwargs):
                factory_fn.__doc__ = cls.__init__.__doc__
                return cls(self, *args, **kwargs)
            
            #else:
            #    def factory_fn(*args,**kwargs):
            #        key = call_key(cls, args) # cache by call key
            #        if key not in self.outputObjs:
            #            #print("DB: new call key = ",key)
            #            self.outputObjs[key] = cls(self, *args, **kwargs) #construct using *full* args
            #        return self.outputObjs[key]
                
            factory_fn.__doc__ = cls.__init__.__doc__
            return factory_fn

        # "register" components
        from . import workspacetables as _wt
        from . import workspaceplots as _wp

        self.Switchboard = makefactory(Switchboard)

        #Tables (consolidate?)
        self.BlankTable = makefactory(_wt.BlankTable)
        self.SpamTable = makefactory(_wt.SpamTable)
        self.SpamParametersTable = makefactory(_wt.SpamParametersTable)
        self.SpamVsTargetTable = makefactory(_wt.SpamVsTargetTable)

        self.GatesTable= makefactory(_wt.GatesTable)
        self.UnitaryGatesTable = makefactory(_wt.UnitaryGatesTable)
        self.ChoiTable = makefactory(_wt.ChoiTable)
        self.GatesVsTargetTable = makefactory(_wt.GatesVsTargetTable)
        self.GateAnglesTable = makefactory(_wt.GateAnglesTable)
        self.GateDecompTable = makefactory(_wt.GateDecompTable)
        self.RotationAxisTable = makefactory(_wt.RotationAxisTable)        
        self.CherryPickedGatesVsTargetTable = makefactory(_wt.CherryPickedGatesVsTargetTable)
        #self.ClosestUnitaryTable = makefactory(_wt.ClosestUnitaryTable)

        self.EigenvalueTable = makefactory(_wt.EigenvalueTable)
        self.ChoiEigenvalueTable = makefactory(_wt.ChoiEigenvalueTable)
        
        self.ErrgenTable = makefactory(_wt.ErrgenTable)
        
        self.DataSetOverviewTable = makefactory(_wt.DataSetOverviewTable)
        
        self.Chi2ProgressTable = makefactory(_wt.Chi2ProgressTable)
        self.LogLProgressTable = makefactory(_wt.LogLProgressTable)
        self.LogLByGermTable = makefactory(_wt.LogLByGermTable)
        self.LogLProjectedErrGenTable = makefactory(_wt.LogLProjectedErrGenTable)

        self.GatestringTable = makefactory(_wt.GatestringTable)
        self.GatestringMultiTable= makefactory(_wt.GatestringMultiTable)

        self.GateBoxesTable = makefactory(_wt.GateBoxesTable)
        self.ErrgenBoxesTable = makefactory(_wt.ErrgenBoxesTable)        
        self.ErrGenComparisonTable = makefactory(_wt.ErrGenComparisonTable)
        self.ErrgenProjectorBoxesTable = makefactory(_wt.ErrgenProjectorBoxesTable)

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

    def init_notebook_mode(self, connected=False):
        try:
            from IPython.core.display import display as _display
            from IPython.core.display import HTML as _HTML
        except ImportError:
            raise ImportError('Only run `init_notebook_mode` from inside an IPython Notebook.')

        global __PYGSTI_WORKSPACE_INITIALIZED
            
        # The polling here is to ensure that plotly.js has already been loaded before
        # setting display alignment in order to avoid a race condition.
        script = """
            <script>
            var waitForPlotly = setInterval( function() {
            if( typeof(window.Plotly) !== "undefined" ){
                MathJax.Hub.Config({ SVG: { font: "STIX-Web" }, displayAlign: "center" });
                MathJax.Hub.Queue(["setRenderer", MathJax.Hub, "SVG"]);
                clearInterval(waitForPlotly);
            }}, 250 );
            </script>"""        
        _display(_HTML(script))

        # Load style sheets for displaying tables
        cssPath = _os.path.join( _os.path.dirname(_os.path.abspath(__file__)),
                                 "templates","css")
        with open(_os.path.join(cssPath,"dataTable.css")) as f:
            script = '<style>\n' + str(f.read()) + '</style>'
            _display(_HTML(script1 + script))

        # Update Mathjax config -- jupyter notebooks already have this so not needed
        #script = '<script type="text/x-mathjax-config">\n' \
        #        + '  MathJax.Hub.Config({ tex2jax: {inlineMath: [["$","$"], ["\\(","\\)"]]}\n' \
        #        + '}); </script>' \
        #        + '<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" ></script>'
        #_display(_HTML(script))

        #Tell require.js where jQueryUI is
        script = """
        <script>requirejs.config({paths: { 'jquery-ui': ['https://code.jquery.com/ui/1.12.1/jquery-ui.min']},});
          require(['jquery-ui'],function(ui) {
          window.jQueryUI=ui;});
        </script>"""
        _display(_HTML(script))


        #MathJax (& jQuery) are already loaded in ipython notebooks
        # '<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" ></script>'
        
        # Initialize Plotly libraries
        _plotly_offline.init_notebook_mode(connected)
        
        __PYGSTI_WORKSPACE_INITIALIZED = True

        

    def switchedCompute(self, fn, *args):

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
                
            # argVals now contains all the arguments, so call the function if
            #  we need to and add result.
            try:
                key = call_key(fn, argVals) # cache by call key
                if key not in self.compCache:
                    #print("DB: computing with args = ", argsVals)
                    self.compCache[key] = fn(*argVals)
                result = self.compCache[key]
            except TypeError:
                #print("DB: computing with args = ", argsVals)
                print("WARNING: unable to cache call to %s" % fn.__name__)
                result = fn(*argVals)
                key = randomID() #something unique so it get's it's own cache slot

            if key not in storedKeys:
                switchpos_map[pos] = len(resultValues)
                storedKeys[key] = len(resultValues)
                resultValues.append( result )
            else:
                switchpos_map[pos] = storedKeys[key]

        switchboard_switch_indices = [ info['switch indices'] for info in switchBdInfo ]
        return resultValues, switchpos_map, switchboards, switchboard_switch_indices


    def cachedCompute(self, fn, *args):
        curkey = call_key(fn, args) # cache by call key
        
        if curkey not in self.compCache:
            self.compCache[curkey] = fn(*valArgs)
            
        return self.compCache[curkey]

    
    
class Switchboard(_collections.OrderedDict):
    def __init__(self, ws, switches, positions, types, initial_pos=None,
                 descriptions=None, ID=None):

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
        self.widget = None
        super(Switchboard,self).__init__([])


    def add(self, varname, dependencies):
        super(Switchboard,self).__setitem__(varname, SwitchValue(self.ws, self, varname, dependencies))

    def __setitem__(self, key, val):
        raise KeyError("Use add(...) to add an item to this swichboard")

    def render(self, typ="html"):
        assert(typ == "html"), "Can't render Switchboards as anything but HTML"

        switch_html = []; switch_js = []
        for i,(name,ID,typ,posLbls,ipos) in enumerate(zip(self.switchNames,
                                                          self.switchIDs,
                                                          self.switchTypes,
                                                          self.positionLabels,
                                                          self.initialPositions)):
            if typ == "buttons":
                html = "<fieldset id='%s'>\n" % ID
                for k,lbl in enumerate(posLbls):
                    checked = " checked='checked'" if k==ipos else ""
                    html += "<label for='%s-%d'>%s</label>\n" % (ID, k,lbl)
                    html += "<input type='radio' name='%s' id='%s-%d' value=%d%s>\n" \
                                          % (ID,ID,k,k,checked)
                html += "</fieldset>\n"
                js = "  $('#%s > input').checkboxradio({ icon: false });" % ID

            elif typ == "dropdown":
                html = ""; js = ""
                raise NotImplementedError()
            
            elif typ == "slider":
                html = ""; js = ""
                raise NotImplementedError()
                
            else:
                raise ValueError("Unknown switch type: %s" % typ)

            switch_html.append(html)
            switch_js.append(js)

        html = "\n".join(switch_html)
        js = "$(document).ready(function() {\n" +\
             "\n".join(switch_js) + "\n});"
        return html, js
                

    def get_switch_change_handlerjs(self, switchIndex):
        ID = self.switchIDs[switchIndex]
        typ = self.switchTypes[switchIndex]
        if typ == "buttons":
            return "$('#%s').on('change', function() {" % ID
        else:
            raise ValueError("Unknown switch type: %s" % typ)

    def get_switch_valuejs(self, switchIndex):
        ID = self.switchIDs[switchIndex]
        typ = self.switchTypes[switchIndex]
        if typ == "buttons":
            return "$(\"#%s > input[name='%s']:checked\").val()" % (ID,ID)
        else:
            raise ValueError("Unknown switch type: %s" % typ)
        
    def display(self):
        import ipywidgets as _widgets
        from IPython.display import display as _display
        from IPython.display import HTML as _HTML

        #if self.widget is None:
        #    self.widget = _widgets.HTMLMath(value="?",
        #                                placeholder='Switch HTML',
        #                                description='Switch HTML',
        #                                disabled=False)
        html, js = self.render("html")
        content = "<script>\n" + \
                  "require(['jquery','jquery-ui'],function($,ui) {" + \
                  js + " });</script>" + html
        #self.widget.value = content
        _display(_HTML(content)) #self.widget)
        


class SwitchValue(object):
    def __init__(self, ws, parent_switchboard, name, dependencies):
        self.ws = ws #workset
        self.parent = parent_switchboard
        self.name = name
        self.dependencies = dependencies
        
        shape = [len(self.parent.positionLabels[i]) for i in dependencies]
        self.base = _np.empty(shape, dtype=_np.object)

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
    """ An output that can be rendered """
    
    def __init__(self, ws):
        self.ws = ws
        self.widget = None #don't build until 1st display()

        
    # Note: hashing not needed because these objects are not *inputs* to
    # other WorspaceOutput objects or computation functions - these objects
    # are cached using call_key.

    def render(self, typ="html"):
        """Renders this object to the specifed format"""
        raise NotImplementedError("Derived classes should implement render()")


    def display(self):
        """Create a new widget associated with this object and display it"""
        import ipywidgets as _widgets
        from IPython.display import display as _display
        from IPython.display import HTML as _HTML

        #if self.widget is None:
        #    self.widget = _widgets.HTMLMath(value="?",
        #                                placeholder='Plot HTML',
        #                                description='Plot HTML',
        #                                disabled=False)
        html, js = self.render("html", global_requirejs=True) # b/c jupyter uses require.js
        #OLD: content = "<script>\n%s\n</script>\n\n%s" % (js,html)
        content = "<script>\n" + \
                  "require(['jquery','jquery-ui'],function($,ui) {" + \
                  js + " });</script>" + html

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
        Helper rendering function

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
        """

        #build HTML as container div containing one or more plot divs
        html = "<div id='%s' style='display: none'>\n" % ID
        html += "\n".join(div_htmls) + "\n</div>\n"

        #build javascript to map switch positions to div_ids
        #js  = "$(document).ready(function() {\n"
        js = "var switchmap_%s = new Array();\n" % ID
          #global varaiable -- do not put in on-ready handler
        js += "$( function() {\n"
        for switchPositions, iDiv in self.switchpos_map.items():
            #switchPositions is a tuple of tuples of position indices, one tuple per switchboard
            div_id = div_ids[iDiv]
            flatPositions = []
            for singleBoardSwitchPositions in switchPositions:
                flatPositions.extend( singleBoardSwitchPositions )                
            js += "  switchmap_%s[ [%s] ] = '%s';\n" % \
                    (ID, ",".join(map(str,flatPositions)), div_id)
        js += "\n"

        
        #build change event listener javascript
        handler_fns_js = []
        for sb, switchInds in zip(self.switchboards, self.sbSwitchIndices):
            # switchInds is a tuple containing the "used" switch indices of sb
            
            for switchIndex in switchInds:
                #build a handler function to get all of the relevant switch positions,
                # build a (flattened) position array, and perform the lookup.
                fname = "%s_onchange_%s_%d" % (ID,sb.ID,switchIndex)
                handler_js = "function %s() {\n" % fname 
                handler_js += "  var curSwitchPos = new Array();\n"
                for sb2, switchInds2 in zip(self.switchboards, self.sbSwitchIndices):
                    for switchIndex2 in switchInds2:
                        handler_js += "  curSwitchPos.push(%s);\n" % sb2.get_switch_valuejs(switchIndex2)
                handler_js += "  var idToShow = switchmap_%s[ curSwitchPos ];\n" % ID
                handler_js += "  $( '#%s' ).children().hide();\n" % ID
                handler_js += "  $( '#' + idToShow ).show();\n"
                handler_js += "}\n"
                handler_fns_js.append(handler_js)

                # on document ready
                js += "  " + sb.get_switch_change_handlerjs(switchIndex) + \
                              "%s(); });\n" % fname
                js += "  %s();\n" % fname # call function to update visibility

        #once all visibility update are done, show parent container
        js += "$( '#%s' ).show()\n" % ID
        js += "});\n\n" # end on-ready handler
        js += "\n".join(handler_fns_js)

        return html, js



    
class WorkspaceTable(WorkspaceOutput):
    def __init__(self, ws, fn, *args):
        super(WorkspaceTable, self).__init__(ws)
        self.tablefn = fn
        self.initargs = args
        self.tables,self.switchpos_map,self.switchboards,self.sbSwitchIndices = \
            self.ws.switchedCompute(self.tablefn, *self.initargs)

        
    def render(self, typ, global_requirejs=False):
        """Renders this table to the specifed format"""
        if typ == "html":
            tableID = "table_" + randomID()

            divHTML = []
            divIDs = []
            for i,table in enumerate(self.tables):
                tableDivID = tableID + "_%d" % i
                table_html = "<div id='%s'>\n%s\n</div>\n" % (tableDivID,
                                table.render("html", tableclass="dataTable"))
                divHTML.append(table_html)
                divIDs.append(tableDivID)
                
            return self._render_html(tableID, divHTML, divIDs, self.switchpos_map,
                                     self.switchboards, self.sbSwitchIndices)
        else:
            assert(len(self.tables) == 1), \
                "Can only render %s format for a non-switched table" % typ
            return self.tables[0].render(typ)


        
class WorkspacePlot(WorkspaceOutput):
    def __init__(self, ws, fn, *args):
        super(WorkspacePlot, self).__init__(ws)
        self.plotfn = fn
        self.initargs = args
        self.figs, self.switchpos_map, self.switchboards, self.sbSwitchIndices = \
            self.ws.switchedCompute(self.plotfn, *self.initargs)

    def render(self, typ="html", global_requirejs=False):
        assert(typ == "html"), "Only HTML rendering supported currently"

        plotID = "plot_" + randomID()

        def getPlotlyDivID(html):
            #could make this more robust using lxml or something later...
            iStart = html.index('div id="')
            iEnd = html.index('"', iStart+8)
            return html[iStart+8:iEnd]

        divHTML = []
        divIDs = []
        for fig in self.figs:
            fig_html = _plot(fig, include_plotlyjs=False, output_type='div',
                             show_link=False, global_requirejs=global_requirejs)
            divHTML.append(fig_html)
            divIDs.append(getPlotlyDivID(fig_html))
            
        return self._render_html(plotID, divHTML, divIDs, self.switchpos_map,
                                 self.switchboards, self.sbSwitchIndices)
                
                    
                


### TEST ------------------------------------------------------------------------
#def add(a,b):
#    print("Adding(%d,%d)!" % (a,b))
#    return a + b
#
#class TestLabel(WorkspaceOutput):
#    def __init__(self, ws, val1, val2):
#        print("Constructing TestLabel(%s,%s)" % (str(val1),str(val2)) )
#        super(TestLabel, self).__init__(ws)        
#        self.initargs = (val1,val2)
#        self.widget = None #don't build until necessary
#        self.update()
#
#    def update(self):
#        val1,val2 = self.initargs
#        print("Updating TestLabel(%s,%s)" % (str(val1),str(val2)) )
#        self.html = self.ws.cachedCompute(add, val1, val2)
#
#        if self.widget:
#            self.widget.value = "Value: " + str(self.html)
#
#    def render(self):
#        if self.widget is None:
#            self.widget = _widgets.Label(value="Value: " + str(self.html),
#                                         placeholder='Some LaTeX',
#                                         description='Some LaTeX',
#                                         disabled=False)
#            self.update()
#        _display(self.widget)
#
#
#class TestPlot(WorkspaceOutput):
#    def __init__(self, ws, val1, val2):
#        print("Constructing TestPlot(%s,%s)" % (str(val1),str(val2)) )
#        super(TestPlot, self).__init__(ws)        
#        self.initargs = (val1,val2)
#        self.widget = None #don't build until necessary
#        self.update()
#
#    def update(self):
#        val1,val2 = self.initargs
#        print("Updating TestPlot(%s,%s)" % (str(val1),str(val2)) )
#        self.val = self.ws.cachedCompute(add, val1, val2)
#
#        if self.widget:
#            fig = _go.Scatter(x=[1,2,3],y=[3, 1, self.val])
#            _clear_output()
#            iplot([fig])
#
#    def render(self):
#        if self.widget is None:
#            self.widget = _widgets.Output()
#            self.update()
#        _display(self.widget)
#        
#class MyPlot(WorkspaceOutput):
#    def __init__(self, ws, gateset, dataset, filename, render=True):
#
#        key = ("MyPlot",gateset.key,dataset.key,filename.key)
#
#        # use ws as cache for (myID, myArgs) value
#        if ws is not None and key in ws:
#            self.__dict__ = ws[key].__dict__.copy() #copy?
#        else:
#            #compute plot data; deal with gateset & dataset being WorkspaceValues that contain mulitple values
#            # - use ws as cache for calculations
#            gs = gateset.value
#            ds = dataset.value
#            #self.logLMx = ws.BoxPlotData("logl", gs, ds, sumUp=False) #will check cache
#            
#            if ws is not None:
#                ws[key] = self
#
#        if render: self.render()
#        super(MyPlot, self).__init__(ws)
#
#    def render(self):
#        #plot self.logLMx within a widget?
#        pass
### TEST ------------------------------------------------------------------------

