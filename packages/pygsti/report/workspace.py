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
import collections as _collections
import numpy as _np
import uuid as _uuid
import random as _random

import ipywidgets as _widgets
from IPython.display import display as _display
from IPython.display import clear_output as _clear_output

import plotly.graph_objs as go
from plotly.offline import plot, iplot
from plotly.offline.offline import get_plotlyjs


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
        elif hasattr(arg, 'digest_hash'):
            #Use a digest hash if one is available
            key.append( arg.digest_hash() ) #backup for un-hashable objects e.g. GateSet, DataSet, etc.
            #key.append( id(arg) ) #backup for un-hashable objects e.g. GateSet, DataSet, etc.
        else: raise ValueError("Object of type %s is not " % str(type(arg))
                               + "hashable, so cannot be used as an argument"
                               + "in cached computation calls")

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
        self.WorkspaceValue = makefactory(WorkspaceValue)
        self.NamedValue = makefactory(NamedValue)
        self.NamedWorkspaceValue = makefactory(NamedWorkspaceValue)
        self.Selector = makefactory(Selector)
        self.Switchboard = makefactory(Switchboard)
        self.TestLabel = makefactory(TestLabel, cache=False)
        self.TestPlot = makefactory(TestPlot, cache=False)
        self.BlankTable = makefactory(_wt.BlankTable, cache=False)
        self.SpamTable = makefactory(_wt.SpamTable, cache=False)
        self.ColorBoxPlot = makefactory(_wp.ColorBoxPlot, cache=False)
        self.BoxKeyPlot = makefactory(_wp.BoxKeyPlot, cache=False)
        self.GateMatrixPlot = makefactory(_wp.GateMatrixPlot, cache=False)
        self.PolarEigenvaluePlot = makefactory(_wp.PolarEigenvaluePlot, cache=False)
        self.ProjectionsBoxPlot = makefactory(_wp.ProjectionsBoxPlot, cache=False)
        self.ChoiEigenvalueBarPlot = makefactory(_wp.ChoiEigenvalueBarPlot, cache=False)


    def switchedCompute(self, fn, *args):

        # Computation functions get stripped-down *value* args
        # (strip WorkspaceValue stuff away)

        valArgs = [ arg.value if isinstance(arg,WorkspaceValue) else arg
                    for arg in args ]

        switchboards = []
        switchBdInfo = []
        
        #sbVariableNames = []
        #sbArgIndices = []
        
        nonSwitchedArgs = []

        switchpos_map = {}
        storedKeys = {}
        resultValues = []
        
        for i,arg in enumerate(args):
            if isinstance(arg,SwitchVariable):
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
                info['value names'].append(arg.groupname)
                info['switch indices'].update(arg.dependencies)
            else:
                nonSwitchedArgs.append( (i,arg) )

        #print("DB: %d arguments" % len(args))
        #print("DB: found %d switchboards" % len(switchboards))
        #print("DB: sbArgIndices = ", sbArgIndices)
        #print("DB: sbVariableNames = ", sbVariableNames)
        #print("DB: nonSwitchedArgs = ", nonSwitchedArgs)
        #print("DB: nsArgIndices = ",nonSwitchedArgIndices)

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

        #for k,nswarg in zip(nonSwitchedArgIndices,nonSwitchedArgs): #treat each non-switched are like a separate switchboard
        #    sbValueTups.append([(val,) for val in args[k].values] if isinstance(args[k],WorkspaceValue) else [(args[k],)])
        #    sbArgIndices.append( (k,) )

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
                    argVals[j] = sb[nm][value_swpos]

            #next, fill in the non-switched arguments
            for j,arg in nonSwitchedArgs:
                argVals[j] = arg.value if isinstance(arg,WorkspaceValue) else arg
                

            # argVals now contains all the arguments, so call the function if
            #  we need to and add result.

            #TODO: maybe make this a call to worspace?
            key = call_key(fn, argVals) # cache by call key            
            if key not in self.compCache:
                #print("DB: computing with valArgs = ", valArgs)
                self.compCache[key] = fn(*argVals)
            result = self.compCache[key]

            if key not in storedKeys:
                switchpos_map[pos] = len(resultValues)
                storedKeys[key] = len(resultValues)
                resultValues.append( self.compCache[key] )
            else:
                switchpos_map[pos] = storedKeys[key]

        switchboard_switch_indices = [ info['switch indices'] for info in switchBdInfo ]
        return resultValues, switchpos_map, switchboards, switchboard_switch_indices


    def cachedCompute(self, fn, *args):

        # Computation functions get stripped-down *value* args
        # (strip WorkspaceValue stuff away)

        precomp = True #do all precomputation up front
        valArgs = [ arg.value if isinstance(arg,WorkspaceValue) else arg
                    for arg in args ]
        curkey = call_key(fn, valArgs) # cache by call key
        
        if precomp:
            switchboards = []
            sbVariableNames = []
            sbArgIndices = []
            
            nonSwitchedArgs = []
            nonSwitchedArgIndices = []
            
            for i,arg in enumerate(args):
                if isinstance(arg,SwitchVariable):
                    isb = None
                    for j,sb in enumerate(switchboards):
                        if arg.parent is sb:
                            isb = j; break
                    else:
                        isb = len(switchboards)
                        switchboards.append(arg.parent)
                        sbArgIndices.append([])
                        sbVariableNames.append([])
                    assert(isb is not None)

                    sbArgIndices[isb].append(i)
                    sbVariableNames[isb].append(arg.groupname)
                else:
                    nonSwitchedArgs.append(arg)
                    nonSwitchedArgIndices.append(i)

            #print("DB: %d arguments" % len(args))
            #print("DB: found %d switchboards" % len(switchboards))
            #print("DB: sbArgIndices = ", sbArgIndices)
            #print("DB: sbVariableNames = ", sbVariableNames)
            #print("DB: nonSwitchedArgs = ", nonSwitchedArgs)
            #print("DB: nsArgIndices = ",nonSwitchedArgIndices)
                    
            sbValueTups = []
            for sb,variableNames in zip(switchboards,sbVariableNames):
                sbValueTups.append( sb.get_value_tuples(variableNames) )
                
            for k,nswarg in zip(nonSwitchedArgIndices,nonSwitchedArgs): #treat each non-switched are like a separate switchboard
                sbValueTups.append([(val,) for val in args[k].values] if isinstance(args[k],WorkspaceValue) else [(args[k],)])
                sbArgIndices.append( (k,) )

            #print("DB: sbValueTups:")
            #for i,vals in enumerate(sbValueTups):
            #    print(  "%d: len=%d, indices=%s, vals=%s" % (i,len(vals),str(sbArgIndices[i]),str(vals)))

            #Notes:
            # each el of sbValueTups is a list of (val1,val2,...) tuples associated with a single Switchboard (or non-switched arg)
            # each el of sbArgIndices is a tuple  (i1, i2, ...) of integers giving the argument indices of the (va1,val2,...) values
            #   associated with a single Switchboard (or non-switched arg)

            for allValueTups in _itertools.product( *sbValueTups ):
                valArgs = [None]*len(args)
                for argIndices, valueTup in zip(sbArgIndices, allValueTups):
                    for argIndex,val in zip(argIndices, valueTup):
                        valArgs[argIndex] = val
                
                key = call_key(fn, valArgs) # cache by call key
                if key not in self.compCache:
                    #print("DB: computing with valArgs = ", valArgs)
                    self.compCache[key] = fn(*valArgs)

            #OLD
            #valLists = [ arg.values if isinstance(arg,WorkspaceValue) else [arg]
            #             for arg in args ]
            #for valArgs in _itertools.product( *valLists ):
            #    key = call_key(fn, valArgs) # cache by call key
            #    if key not in self.compCache:
            #        self.compCache[key] = fn(*valArgs)
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
    An input that for WorspaceOutput object construction.  Wraps one
    or more values which are individually used by computational routines.
    The 'value' member holds the "current" value of the WorkspaceValue.
    When a computational routine is called, `Workspace.cachedCompute`
    takes care of caching results for *all* of a WorkspaceValue's
    values (including the "current" one) and returns the current one.
    """
    def __init__(self, ws, values, curindex=0):
        self.values = values
        self.value = values[curindex] if len(values) > 0 else None
        self.curindex = curindex if len(values) > 0 else None
        self.ws = ws

    #We never need to hash these objects, since only underlying 'values' are hashed (see cachedCompute)
    #def __hash__(self):
    #    raise TypeError #so this type is "unhashable" and will use id
    #    # as desired (ideally GateSets and DataSets will do the same thing?)

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



#switchbd = SwitchBoard(switches=["Dataset","GaugeOptParam", "SomeOtherParam"],
#                       labels=[["1","2"],["0","0.5","1.0"],['a','b']])
#switchbd.addItem("gs",(0,1)) #OR
#switchbd.addItem("gs",("Dataset","GaugeOptParam"))
#switchbd.addItem("ds",(0,)) #OR
#switchbd.addItem("ds",("Dataset",))
#switchbd.gs["1",:] = gOptSets1 # gs marked as a "level 2" item, so if switch 1 or 2 moves it gets updated
#switchbd.gs["2",:] = gOptSets2
#switchbd.ds[:] = [ds1,ds2]  # a "level 1" item: only updates when switch 1 is moved
#switchbd.addItem("other",(2,)) #OR
#switchbd.addItem("other",("SomeOtherParam",))
#switchbd.otherParams[:] = [Aval, Bval]

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
        super(Switchboard,self).__setitem__(varname, SwitchVariable(self.ws, self, varname, dependencies))

    def get_value_tuples(self, variableNames):
        """ Returns (val1,val2,...) tuples of *distinct* values for the variables corresponding to variableNames """
        switches_to_iter_over = set()
        for varname in variableNames:
            switches_to_iter_over.update(self[varname].dependencies)
        switches_to_iter_over = list(switches_to_iter_over) #so definite ordering
            
        #iterate over the switches that these values depend upon
        value_tuples = []
        ranges = [ list(range(len(self.positionLabels[i]))) for i in switches_to_iter_over ]
        for switchVals in _itertools.product( *ranges ):
            variableVals = []
            for varname in variableNames:
                varSwitchVals = [ switchVals[switches_to_iter_over.index(k)] for k in self[varname].dependencies ]
                variableVals.append( self[varname][varSwitchVals] )
            #OLD: variableVals = [ self[varname][switchVals] for varname in variableNames ]
            tup = tuple(variableVals)
            if tup not in value_tuples:
                value_tuples.append(tup)
                
        return value_tuples

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
        if self.widget is None:
            self.widget = _widgets.HTML(value="?",
                                        placeholder='Switch HTML',
                                        description='Switch HTML',
                                        disabled=False)
        html, js = self.render("html")
        content = "<script>\n" + js + "</script>" + html
        self.widget.value = content
        _display(self.widget)

            
#    def __getattr__(self, attr):
#        ret = getattr(self.__dict__['base'],attr)
#        if isinstance(ret, _np.ndarray) and ret.base is self.base:
#            ret.flags.writeable = False
#        return ret

class SwitchVariable(NamedWorkspaceValue):
    def __init__(self, ws, parent_switchboard, name, dependencies):
        self.parent = parent_switchboard
        self.dependencies = dependencies

        Ns = [len(self.parent.positionLabels[i]) for i in dependencies]
        totalAndFactors = list(reversed(_np.cumprod([1]+list(reversed(Ns)))))
        total = totalAndFactors[0]
        self.factors = totalAndFactors[1:]
        super(SwitchVariable, self).__init__(ws, values=[None]*total, names=[""]*total,
                                             groupname=name, curindex=0)

    def __getitem__(self, currentSwitchPositionInds):
        i = _np.dot(self.factors, _np.array(currentSwitchPositionInds))
        return self.values[i]
        
    def __setitem__(self, inds, vals):
        if isinstance(inds, tuple):
            assert(len(inds) == len(self.dependencies))
        else:
            assert(1 == len(self.dependencies))
            inds = (inds,)

        valIndRanges = []
        myIndRanges = []
        for i,(sliceOrIndex,swIndex) in enumerate(zip(inds,self.dependencies)):
            if isinstance(sliceOrIndex, slice):
                assert(sliceOrIndex.start is None and sliceOrIndex.stop is None), \
                    "Currently, only full support slices are supported"
                myIndRanges.append(list(range(len(self.parent.positionLabels[swIndex]))))
                valIndRanges.append(list(range(len(self.parent.positionLabels[swIndex]))))
            else:
                valIndRanges.append(
                    [self.parent.positionLabels[swIndex].index(sliceOrIndex)])

        def getval(vs, vis):
            if len(vis) == 1: return vs[vis[0]]
            else: return getval(vs[vis[0]], vis[1:])
                
        for myInds,valInds in zip(_itertools.product(*myIndRanges),_itertools.product(*valIndRanges)):
            i = _np.dot(self.factors, _np.array(myInds))
            self.values[i] = getval(vals,valInds)

    #def switch_update(self):
    #    curpos = [ self.parent.currentPositions[k] for k in self.dependencies ]
    #    i = _np.dot(self.factors, _np.array(curpos,'i') )
    #    self.update(i)  # WorkspaceValue.update

        
        
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

    def display(self):
        """Create a new widget associated with this object and display it"""
        #displays nothing by default
        pass

    
class WorkspaceTable(WorkspaceOutput):
    def __init__(self, ws, fn, *args):
        super(WorkspaceTable, self).__init__(ws)
        self.tablefn = fn
        self.initargs = args
        self.widget = None #don't build until necessary
        self.update()

    def update(self):
        """
        Update any widgets associated with this object.  Called when underlying
        data is changed.
        """
        self.table = self.ws.cachedCompute(self.tablefn, *self.initargs)
        if self.widget:
            self.widget.value = self.table.render("html", tableclass="dataTable")
            #self.widget.value = "$$" + self.table.render("latex") + "$$"

    def display(self):
        """Create a new widget associated with this object and display it"""
        if self.widget is None:
            self.widget = _widgets.HTML(value="?",
                                        placeholder='Some LaTeX',
                                        description='Some LaTeX',
                                        disabled=False)
            self.update()
        _display(self.widget)


    def render(self, typ):
        """Renders a static version of this table"""
        return self.table.render(typ, tableclass="dataTable")


class WorkspacePlot(WorkspaceOutput):
    def __init__(self, ws, fn, *args):
        super(WorkspacePlot, self).__init__(ws)
        self.plotfn = fn
        self.initargs = args
        self.widget = None #don't build until necessary

        self.figs,self.switchpos_map,self.switchboards,self.sbSwitchIndices = \
                    self.ws.switchedCompute(self.plotfn, *self.initargs)

        #self.update()

    #def update(self):
    #    """
    #    Update any widgets associated with this object.  Called when underlying
    #    data is changed.
    #    """
    #    self.fig = self.ws.cachedCompute(self.plotfn, *self.initargs)
    #    if self.widget:
    #        _clear_output()
    #        iplot(self.fig) #basehtml = plot(fig, output_type='div', image_filename="test", image='png', auto_open=False)

    def display(self):
        """Create a new widget associated with this object and display it"""
        if self.widget is None:
            self.widget = _widgets.HTML(value="?",
                                        placeholder='Plot HTML',
                                        description='Plot HTML',
                                        disabled=False)
        html, js = self.render("html")
        content = "<script>\n%s\n</script>\n\n%s" % (js,html)
        self.widget.value = content
        #with open("debug.html","w") as f:
        #    jsincludes = '<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.1.1/jquery.min.js">'
        #    filecontent = "<html><head><script>\n%s\n</script>\n%s\n</head>\n<body> %s </body></html>" % (get_plotlyjs(),jsincludes,content)
        #    f.write(filecontent)
        #print("DB content:\n",content)
        _display(self.widget)

        
    def render(self, typ="html"):
        assert(typ == "html"), "Only HTML rendering supported currently"

        containerID = "oobj_" + randomID()

        def getPlotlyDivID(html):
            #could make this more robust using lxml or something later...
            iStart = html.index('div id="')
            iEnd = html.index('"', iStart+8)
            return html[iStart+8:iEnd]

        self.figs,self.switchpos_map,self.switchboards,self.sbSwitchIndices = \
                    self.ws.switchedCompute(self.plotfn, *self.initargs)

        
        #build HTML as container div containing one or more plot divs
        html = "<div id='%s' style='display: none'>\n" % containerID

        figDivIds = []
        for fig in self.figs:
            fig_html = plot(fig, include_plotlyjs=False, output_type='div')
            html += fig_html + "\n"
            figDivIds.append(getPlotlyDivID(fig_html))
        html += "</div>\n"


        #build javascript to map switch positions to divIDs
        #js  = "$(document).ready(function() {\n"
        js = "var switchmap_%s = new Array();\n" % containerID
          #global varaiable -- do not put in on-ready handler
        js += "$( function() {\n"
        for switchPositions, iFig in self.switchpos_map.items():
            #switchPositions is a tuple of tuples of position indices, one tuple per switchboard
            fig_divid = figDivIds[iFig]
            flatPositions = []
            for singleBoardSwitchPositions in switchPositions:
                flatPositions.extend( singleBoardSwitchPositions )                
            js += "  switchmap_%s[ [%s] ] = '%s';\n" % \
                    (containerID, ",".join(map(str,flatPositions)), fig_divid)
        js += "\n"

        
        #build change event listener javascript
        handler_fns_js = []
        for sb, switchInds in zip(self.switchboards, self.sbSwitchIndices):
            # switchInds is a tuple containing the "used" switch indices of sb
            
            for switchIndex in switchInds:
                #build a handler function to get all of the relevant switch positions,
                # build a (flattened) position array, and perform the lookup.
                fname = "%s_onchange_%s_%d" % (containerID,sb.ID,switchIndex)
                handler_js = "function %s() {\n" % fname 
                handler_js += "  var curSwitchPos = new Array();\n"
                for sb2, switchInds2 in zip(self.switchboards, self.sbSwitchIndices):
                    for switchIndex2 in switchInds2:
                        handler_js += "  curSwitchPos.push(%s);\n" % sb2.get_switch_valuejs(switchIndex2)
                handler_js += "  var idToShow = switchmap_%s[ curSwitchPos ];\n" % containerID
                handler_js += "  $( '#%s' ).children().hide();\n" % containerID
                handler_js += "  $( '#' + idToShow ).show();\n"
                handler_js += "}\n"
                handler_fns_js.append(handler_js)

                # on document ready
                js += "  " + sb.get_switch_change_handlerjs(switchIndex) + \
                              "%s(); });\n" % fname
                js += "  %s();\n" % fname # call function to update visibility

        #once all visibility update are done, show parent container
        js += "$( '#%s' ).show()\n" % containerID
        js += "});\n\n" # end on-ready handler
        js += "\n".join(handler_fns_js)

        return html, js
                
                    
                


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

