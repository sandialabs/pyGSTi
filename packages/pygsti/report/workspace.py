from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines the Dashboard class and supporting functionality."""

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

import ipywidgets as _widgets
from IPython.display import display as _display
from IPython.display import clear_output as _clear_output

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


def _is_hashable(x):
    try:
        dct = { x: 0 }
    except TypeError:
        return False
    return True


def call_key(obj, args):
    """ obj can be either a function or a class """
    key = [obj.__name__]
    for arg in args:
        if _is_hashable(arg):
            key.append( arg )
        else: key.append( id(arg) ) #backup for un-hashable objects e.g. GateSet, DataSet, etc.
    return tuple(key) #so hashable


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


        def makefactory(cls, cache=False):
            
            #TODO: set signature of factor_fn to cls.__init__
            if not cache:
                def factory_fn(*args,**kwargs):
                    factory_fn.__doc__ = cls.__init__.__doc__
                    return cls(self, *args, **kwargs)

            else:
                def factory_fn(*args,**kwargs):
                    key = call_key(cls, args) # cache by call key
                    if key not in self.outputObjs:
                        #print("DB: new call key = ",key)
                        self.outputObjs[key] = cls(self, *args, **kwargs) #construct using *full* args
                    return self.outputObjs[key]
                
            factory_fn.__doc__ = cls.__init__.__doc__
            return factory_fn

        # "register" components
        from . import workspacetables as _wt
        from . import workspaceplots as _wp
        self.NamedValue = makefactory(NamedValue)
        self.Selector = makefactory(Selector)
        self.TestLabel = makefactory(TestLabel, cache=True)
        self.TestPlot = makefactory(TestPlot, cache=True)
        self.BlankTable = makefactory(_wt.BlankTable, cache=True)
        self.SpamTable = makefactory(_wt.SpamTable, cache=True)
        self.ColorBoxPlot = makefactory(_wp.ColorBoxPlot, cache=True)
        self.BoxKeyPlot = makefactory(_wp.BoxKeyPlot, cache=True)
        self.GateMatrixPlot = makefactory(_wp.GateMatrixPlot, cache=True)
        self.PolarEigenvaluePlot = makefactory(_wp.PolarEigenvaluePlot, cache=True)
        self.ProjectionsBoxPlot = makefactory(_wp.ProjectionsBoxPlot, cache=True)
        self.ChoiEigenvalueBarPlot = makefactory(_wp.ChoiEigenvalueBarPlot, cache=True)


    def cachedCompute(self, fn, *args):

        # Computation functions get stripped-down *value* args
        # (strip WorkspaceValue stuff away)

        precomp = True #do all precomputation up front
        valArgs = [ arg.value if isinstance(arg,WorkspaceValue) else arg
                    for arg in args ]
        curkey = call_key(fn, valArgs) # cache by call key
        
        if precomp:
            valLists = [ arg.values if isinstance(arg,WorkspaceValue) else [arg]
                         for arg in args ]
            for valArgs in _itertools.product( *valLists ):
                key = call_key(fn, valArgs) # cache by call key
                if key not in self.compCache:
                    self.compCache[key] = fn(*valArgs)
        else:
            if curkey not in self.compCache:
                self.compCache[curkey] = fn(*valArgs)

        return self.compCache[curkey]

    
    def trigger_update(self, hashval):
        #print("DB: update %s" % str(hashval))
        
        #clear anything dependent on hashval in computational cache
        to_remove = []
        for call_key_tuple in self.compCache:
            if hashval in call_key_tuple:
                to_remove.append(call_key_tuple)
        #print("DB: to_remove = ", to_remove)
        for k in to_remove: del self.compCache[k]

        #update (recompute if needed) anything dependent on hashval in output cache
        for call_key_tuple in self.outputObjs:
            #print("DB: checking ", call_key_tuple)
            if hashval in call_key_tuple:
                #print("DB: recomputing ", call_key_tuple)
                self.outputObjs[call_key_tuple].update()
        
                
        

class WorkspaceValue(object):
    """ 
    An input that can be hashed to make a argument-list key for
    WorspaceOutput object construction.  Wraps a one or more values
    which are individually used by computational routines.  The
    'value' member holds the "current" value of the WorkspaceValue.
    When a computational routine is called, `Workspace.cachedCompute`
    takes care of caching results for *all* of a WorkspaceValue's
    values (including the "current" one) and returns the current one.
    """
    def __init__(self, ws, values, curindex=0):
        self.values = values
        self.value = values[curindex] if len(values) > 0 else None
        self.curindex = curindex if len(values) > 0 else None
        self.ws = ws
        
    def __hash__(self):
        raise TypeError #so this type is "unhashable" and will use id
        # as desired (ideally GateSets and DataSets will do the same thing?)

    def update(self, new_curindex):
        self.curindex = new_curindex
        self.value = self.values[new_curindex]


class NamedWorkspaceValue(WorkspaceValue):

    def __init__(self, ws, values, names=None, groupname=None, curindex=0):
        if names is not None:
            assert(len(names) == len(values))
        else:
            names = [ str(v) for v in values ]
            
        if groupname is None:
            groupname = names[0] if (len(names) == 1) else ""

        self.groupname = groupname
        self.names = names
        self.name = names[curindex] if len(names) > 0 else None
        super(NamedWorkspaceValue, self).__init__(ws, values, curindex)

    def update(self, new_curindex):
        super(NamedWorkspaceValue, self).update(new_curindex)
        self.name = self.names[new_curindex]

        
class NamedValue(NamedWorkspaceValue):
    """ just provides a single-value constructor to NamedWorkspaceValue """
    def __init__(self, ws, value, name=None):
        super(NamedValue, self).__init__(ws, [value],
                                         [name] if (name is not None) else None)


class Selector(NamedWorkspaceValue):
    def __init__(self, ws, values, names=None, groupname="Select:", curindex=0, 
                 description='', typ="buttons"):
        
        self.typ = typ
        self.description = description
        self.widget = None #don't build widget unless needed

        super(Selector, self).__init__(ws, values, names, groupname, curindex)

    def build_widget(self):
        if self.typ == "buttons":
            wig = _widgets.ToggleButtons(
                options=self.names,
                description=self.groupname,
                disabled=False,
                button_style='', # 'success', 'info', 'warning', 'danger' or ''
                tooltip=self.description,
                #     icon='check'
            )
        else:
            raise ValueError("Unknown 'typ' argument: %s" % self.typ)
        self.widget = wig


    #renders a select widget
    def render(self):
        if self.widget is None:
            self.build_widget()            
        _display(self.widget)

        # - observes change event -> calls update on listeners to this variable?
        self.widget.observe(self.on_update, 'value')

    def on_update(self, change_dict):
        #print("DB: CALLBACK called : %s!" % change_dict)
        if change_dict['type'] == "change":
            iCur = self.names.index(change_dict['new'])
            self.update(iCur)
            self.ws.trigger_update(id(self))
            #print("DB: Selected option %d: %s (val=%s)" %
            #      (self.curindex, self.name, str(self.value)))


class WorkspaceOutput(object):
    """ An output that can be rendered """
    
    def __init__(self, ws):
        self.ws = ws
        
    # Note: hashing not needed because these objects are not *inputs* to
    # other WorspaceOutput objects or computation functions - these objects
    # are cached using call_key.

    #renders nothing by default
    def render(self, typ):
        pass

    
class WorkspaceTable(WorkspaceOutput):
    def __init__(self, ws, fn, *args):
        super(WorkspaceTable, self).__init__(ws)
        self.tablefn = fn
        self.initargs = args
        self.widget = None #don't build until necessary
        self.update()

    def update(self):
        self.table = self.ws.cachedCompute(self.tablefn, *self.initargs)
        if self.widget:
            #_clear_output()
            #self.table.render("iplotly") # make iplot([fig]) call
            from IPython.display import HTML
            self.widget.value = self.table.render("html", tableclass="dataTable")
            #self.widget.value = "$$" + self.table.render("latex") + "$$"
            
    def render(self):
        if self.widget is None:
            #self.widget = _widgets.Output()
            self.widget = _widgets.HTML(value="?",
                                        placeholder='Some LaTeX',
                                        description='Some LaTeX',
                                        disabled=False)
            self.update()
        _display(self.widget)


class WorkspacePlot(WorkspaceOutput):
    def __init__(self, ws, fn, *args):
        super(WorkspacePlot, self).__init__(ws)
        self.plotfn = fn
        self.initargs = args
        self.widget = None #don't build until necessary
        self.update()

    def update(self):
        self.fig = self.ws.cachedCompute(self.plotfn, *self.initargs)
        if self.widget:
            _clear_output()
            iplot(self.fig) #basehtml = plot(fig, output_type='div', image_filename="test", image='png', auto_open=False)
            
    def render(self):
        if self.widget is None:
            self.widget = _widgets.Output()
            self.update()
        _display(self.widget)



## TEST ------------------------------------------------------------------------
def add(a,b):
    print("Adding(%d,%d)!" % (a,b))
    return a + b

class TestLabel(WorkspaceOutput):
    def __init__(self, ws, val1, val2):
        print("Constructing TestLabel(%s,%s)" % (str(val1),str(val2)) )
        super(TestLabel, self).__init__(ws)        
        self.initargs = (val1,val2)
        self.widget = None #don't build until necessary
        self.update()

    def update(self):
        val1,val2 = self.initargs
        print("Updating TestLabel(%s,%s)" % (str(val1),str(val2)) )
        self.html = self.ws.cachedCompute(add, val1, val2)

        if self.widget:
            self.widget.value = "Value: " + str(self.html)

    def render(self):
        if self.widget is None:
            self.widget = _widgets.Label(value="Value: " + str(self.html),
                                         placeholder='Some LaTeX',
                                         description='Some LaTeX',
                                         disabled=False)
            self.update()
        _display(self.widget)


class TestPlot(WorkspaceOutput):
    def __init__(self, ws, val1, val2):
        print("Constructing TestPlot(%s,%s)" % (str(val1),str(val2)) )
        super(TestPlot, self).__init__(ws)        
        self.initargs = (val1,val2)
        self.widget = None #don't build until necessary
        self.update()

    def update(self):
        val1,val2 = self.initargs
        print("Updating TestPlot(%s,%s)" % (str(val1),str(val2)) )
        self.val = self.ws.cachedCompute(add, val1, val2)

        if self.widget:
            fig = go.Scatter(x=[1,2,3],y=[3, 1, self.val])
            _clear_output()
            iplot([fig])

    def render(self):
        if self.widget is None:
            self.widget = _widgets.Output()
            self.update()
        _display(self.widget)
## TEST ------------------------------------------------------------------------
    
    
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

