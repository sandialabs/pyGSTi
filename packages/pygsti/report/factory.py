from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines report factories. """

import os  as _os
#import re  as _re
#import time as _time
#import subprocess  as _subprocess
import collections as _collections
#import matplotlib  as _matplotlib
#import itertools   as _itertools
#import copy as _copy
#
#from ..             import objects              as _objs
#from ..objects      import gatestring           as _gs
from ..objects      import VerbosityPrinter
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

from .gatestringstructure import GatestringStructure
from .workspace import Workspace as _Workspace

#ideas:
# - need to separate computation from plotting, as plotting will always be needed in regen
#   but computation may be "precomputed"
# - check plot.ly:  ability to set colors of heatmap (or custom colormap?)
#                   
#
# Workspace object:
#   - caches, imports Results objects,
#   - holds data sources, controls, and figures (plots/tables) each labeled by a string
#   - keeps dependency graph for interactive updates
#   - data sources = gatesets, datasets, arrays of datasets, etc -- maybe Result objects too? (for params, etc)
#   - examples: ws.makeLogLBoxPlot("LogLPlot1", "myData1", "targetGS")
#           OR  ws.figures["LogLPlot1"] = ws.makeLogLBoxPlot("myData1", "targetGS") #returns a WorkSpaceFigure object?
#               ws.gatesets["varyingGS"] = createGatesetDepolSliderControl(ws.gatesets["target"], 0, 0.1, precomp=False) #returns a WorkspaceControl object?
#               ws.gatesets["v2GS"] = createGatesetGaugeOptSliderControl(ws.gatesets["best"], "spamWeight", 0, 1, nPrecompSamples=10)
#               ws.show() #shows all workspace figures & controls
#     WorkspaceObject as base class: inputs and outputs = arrays holding refs to other WorkspaceObjects
#     WorkspaceData objects only have outputs
#     WorkspaceFigure objects only have inputs
#   - when a control changes, "onupdate" needs to regen everything that depends on the control.
#   - maybe WorkSpaceObject needs a "regen" method?
#   - functions for each plot vs. strings -- fns better b/c clearer args, but what about generation.py replication?
#   - functions take gateset, dataset etc. params but could be arrays of such params -- object type for this?
#   - generation.py /plotting.py fns *compute* & return table or figure that can later be *rendered*; Workspace fns compute
#     only if necessary & render (if auto-render is on) and deal with arrays of params.
#   - calculator object to cache calculations
#
# Results object:
#   - only holds dataset(s), gatesets, CRs, gatestring_structure(s), parameters, etc.
#
# Report object (?):
#   - used to generate a report (PDF file) -- uses latex template and externally or
#     internally (optional param to creation fn) uses a Workspace to create all the
#     static figures needed to produce the report.  Caches tables & figures, but
#     doesn't need to "smart cache" them, as Workspace handles this, e.g. if another
#     report object is given the same Workspace and asks for the same plot the
#     Workspace shouldn't regenerate it.
#  - inherits ResultOptions members like latex cmd, etc.
#  - maybe can generate HTML & Latex (?)


def _merge_template(qtys, templateFilename, outputFilename):
    templateFilename = _os.path.join( _os.path.dirname(_os.path.abspath(__file__)),
                                      "templates", templateFilename )

    template = ''
    with open(templateFilename, 'r') as templatefile:
        template = templatefile.read()
        
    qtys_html = _collections.defaultdict(lambda x=0: "BLANK")
    for key,val in qtys.items():
        if isinstance(val,str):
            qtys_html[key] = val
        else:
            print("DB: rendering ",key)
            html, js = val.render("html")
            qtys_html[key] = "<script>\n%s\n</script>\n\n%s" % (js,html)

    #DEBUG
    #testtmp = "%(targetSpamTable)s" % qtys_html
    #print("TEST = \n",qtys_html['targetSpamTable'])
    #print("TEST2 = \n",testtmp)
            
    filled_template = template % qtys_html
      #.format_map(qtys_html) #need python 3.2+
    with open(outputFilename, 'w') as outputfile:
        outputfile.write(filled_template)


def create_single_qubit_report(results, filename, confidenceLevel=None,
                               title="GST report", datasetLabel="$\\mathcal{D}$",
                               verbosity=0, comm=None, ws=None):
    """TODO: docstring """
    printer = VerbosityPrinter.build_printer(verbosity, comm=comm)
    if ws is None:
        ws = _Workspace()

    # dictionary to store all strings to be inserted into report template
    qtys = {}
      
    qtys['confidenceLevel'] = "%g" % \
        confidenceLevel if confidenceLevel is not None else "NOT-SET"
    qtys['linlg_pcntle'] = str(results.parameters['linlogPercentile'])
    qtys['datasetLabel'] = datasetLabel

    if results.options.errgen_type == "logTiG":
        qtys['errorgenformula'] = "$\hat{G} = G_{\mathrm{target}}e^{\mathbb{L}}$"
    elif results.options.errgen_type == "logG-logT":
        qtys['errorgenformula'] = "$\hat{G} = e^{\mathbb{L} + \log G_{\mathrm{target}}}$"
    else:
        qtys['errorgenformula'] = "???"
        
    if confidenceLevel is not None:
        cri = results._get_confidence_region(confidenceLevel) #TODO
        qtys['confidenceIntervalScaleFctr'] = \
                    "%.3g" % cri.intervalScaling
        qtys['confidenceIntervalNumNonGaugeParams'] = \
                    "%d" % cri.nNonGaugeParams
    else:
        cri = None
        qtys['confidenceIntervalScaleFctr'] = "NOT-SET"
        qtys['confidenceIntervalNumNonGaugeParams'] = "NOT-SET"


    # 1) get latex tables
    printer.log("*** Generating tables ***")

    gsTgt = results.gatesets['target']
    gsFinal = results.gatesets['final estimate']
    ds = results.dataset
    strs = ( tuple(results.gatestring_lists['prep fiducials']),
             tuple(results.gatestring_lists['effect fiducials']) )


    qtys['targetSpamTable'] = ws.SpamTable(gsTgt)
    qtys['targetGatesTable'] = ws.GatesTable(gsTgt)
    qtys['datasetOverviewTable'] = ws.DataSetOverviewTable(ds, gsTgt, 10, strs)
    qtys['bestGatesetSpamTable'] = ws.SpamTable(gsFinal, cri)
    qtys['bestGatesetSpamParametersTable'] = ws.SpamParametersTable(gsFinal, cri)
    qtys['bestGatesetGaugeOptParamsTable'] = ws.GaugeOptParamsTable(tuple(results.parameters['gaugeOptParams']))
    qtys['bestGatesetGatesTable'] = ws.GatesTable(gsFinal, cri)
    qtys['bestGatesetChoiTable'] = ws.ChoiTable(gsFinal, cri)
    qtys['bestGatesetDecompTable'] = ws.GateDecompTable(gsFinal, cri)
    qtys['bestGatesetRotnAxisTable'] = ws.RotationAxisTable(gsFinal, cri, True)
    qtys['bestGatesetVsTargetTable'] = ws.GatesVsTargetTable(gsFinal, gsTgt, cri)
    qtys['bestGatesetErrorGenTable'] = ws.ErrgenTable(gsFinal, gsTgt, cri, results.options.errgen_type)
    qtys['metadataTable'] = ws.MetadataTable(gsFinal, results.options, results.parameters)
    qtys['softwareEnvTable'] = ws.SoftwareEnvTable()

    # if Ls and germs available
    if results._LsAndGermInfoSet: #TODO: better way?
        qtys['fiducialListTable'] = ws.GatestringMultiTable(strs, ["Prep.","Measure"], "Fiducials")
        qtys['prepStrListTable'] = ws.GatestringTable(tuple(results.gatestring_lists['prep fiducials']), "Preparation Fiducials")
        qtys['effectStrListTable'] = ws.GatestringTable(tuple(results.gatestring_lists['effect fiducials']), "Measurement Fiducials")
        qtys['germListTable'] = ws.GatestringTable(tuple(results.gatestring_lists['germs']), "Germ")
        
        if results.parameters['objective'] == "logl":
            qtys['progressTable'] = ws.LogLProgressTable(
                results.parameters['max length list'],
                results.gatesets['iteration estimates'],
                results.gatestring_lists['iteration'], ds,
                results.parameters.get('gateLabelAliases',None))
            plotType = "logl"
            mpc = results.parameters['minProbClip']
        else:
            qtys['progressTable'] = ws.Chi2ProgressTable(
                results.parameters['max length list'],
                results.gatesets['iteration estimates'],
                results.gatestring_lists['iteration'], ds,
                results.parameters.get('gateLabelAliases',None))
            plotType = "chi2"
            mpc = results.parameters['minProbClipForWeighting']


        # 2) generate plots
        printer.log("*** Generating plots ***")

        #Create a GatestringStructure (TODO: maybe move this within class?)
        fidPairs = results.parameters['fiducial pairs']
        if fidPairs is None: fidpair_filters = None
        elif isinstance(fidPairs,dict) or hasattr(fidPairs,"keys"):
            #Assume fidPairs is a dict indexed by germ
            fidpair_filters = { (x,y): fidPairs[y] 
                                for x in Ls[st:] for y in germs }
        else:
            #Assume fidPairs is a list
            fidpair_filters = { (x,y): fidPairs
                                for x in Ls[st:] for y in germs }

        Ls = results.parameters['max length list']
        germs = results.gatestring_lists['germs']
        gstr_filters = { (x,y) : results.gatestring_lists['iteration'][i]
                         for i,x in enumerate(Ls)
                         for y in germs }

        gss = GatestringStructure(results.parameters['max length list'],
                                  results.gatestring_lists['germs'],
                                  results.gatestring_lists['prep fiducials'],
                                  results.gatestring_lists['effect fiducials'],
                                  results._getBaseStrDict(),
                                  fidpair_filters, gstr_filters,
                                  results.parameters.get('gateLabelAliases',None))

        qtys['colorBoxPlotKeyPlot'] = ws.BoxKeyPlot(tuple(results.gatestring_lists['prep fiducials']),
                                                    tuple(results.gatestring_lists['effect fiducials']))
        qtys['bestEstimateColorBoxPlot'] = ws.ColorBoxPlot(plotType, gss, ds, gsFinal,
                    linlg_pcntle=float(results.parameters['linlogPercentile']) / 100,
                    minProbClipForWeighting=mpc)
        qtys['invertedBestEstimateColorBoxPlot'] = ws.ColorBoxPlot(plotType, gss, ds, gsFinal,
                    linlg_pcntle=float(results.parameters['linlogPercentile']) / 100,
                    minProbClipForWeighting=mpc, invert=True)
    

    # 3) populate template latex file => report latex file
    printer.log("*** Merging into template file ***")
    
    templateFile = "report_singlequbit.html"
    _merge_template(qtys, templateFile, filename)




def create_general_report(results, filename, confidenceLevel=None,
                          title="GST report", datasetLabel="$\\mathcal{D}$",
                          verbosity=0, comm=None, ws=None):
    """TODO: docstring """
    printer = VerbosityPrinter.build_printer(verbosity, comm=comm)
    if ws is None:
        ws = _Workspace()

    # dictionary to store all strings to be inserted into report template
    qtys = {}
      
    qtys['confidenceLevel'] = "%g" % \
        confidenceLevel if confidenceLevel is not None else "NOT-SET"
    qtys['linlg_pcntle'] = str(results.parameters['linlogPercentile'])
    qtys['datasetLabel'] = datasetLabel

    if results.options.errgen_type == "logTiG":
        qtys['errorgenformula'] = "$\hat{G} = G_{\mathrm{target}}e^{\mathbb{L}}$"
    elif results.options.errgen_type == "logG-logT":
        qtys['errorgenformula'] = "$\hat{G} = e^{\mathbb{L} + \log G_{\mathrm{target}}}$"
    else:
        qtys['errorgenformula'] = "???"
        
    if confidenceLevel is not None:
        cri = results._get_confidence_region(confidenceLevel) #TODO
        qtys['confidenceIntervalScaleFctr'] = \
                    "%.3g" % cri.intervalScaling
        qtys['confidenceIntervalNumNonGaugeParams'] = \
                    "%d" % cri.nNonGaugeParams
    else:
        cri = None
        qtys['confidenceIntervalScaleFctr'] = "NOT-SET"
        qtys['confidenceIntervalNumNonGaugeParams'] = "NOT-SET"


    # 1) get latex tables
    printer.log("*** Generating tables ***")

    gsTgt = results.gatesets['target']
    gsFinal = results.gatesets['final estimate']
    ds = results.dataset
    strs = ( tuple(results.gatestring_lists['prep fiducials']),
             tuple(results.gatestring_lists['effect fiducials']) )


    qtys['targetSpamBriefTable'] = ws.SpamTable(gsTgt, includeHSVec=False)
    qtys['targetGatesBoxTable'] = ws.GateBoxesTable([gsTgt])
    qtys['datasetOverviewTable'] = ws.DataSetOverviewTable(ds, gsTgt, 10, strs)
    qtys['bestGatesetSpamBriefTable'] = ws.SpamTable(gsFinal, gsTgt, cri, includeHSVec=False)
    qtys['bestGatesetSpamParametersTable'] = ws.SpamParametersTable(gsFinal, cri)
    qtys['bestGatesetSpamVsTargetTable'] = ws.SpamVsTargetTable(gsFinal, gsTgt, cri)
    qtys['bestGatesetGaugeOptParamsTable'] = ws.GaugeOptParamsTable(tuple(results.parameters['gaugeOptParams']))
    qtys['bestGatesetGatesBoxTable'] = ws.GateBoxesTable([gsTgt,gsFinal], ['Target','Estimated'], cri)
    qtys['bestGatesetChoiEvalTable'] = ws.ChoiEigenvalueTable(gsFinal, cri)
    qtys['bestGatesetEvalTable'] = ws.EigenvalueTable(gsFinal, gsTgt, cri)
#    qtys['bestGatesetRelEvalTable'] = OUT!
    qtys['bestGatesetVsTargetTable'] = ws.GatesVsTargetTable(gsFinal, gsTgt, cri)
    qtys['bestGatesetErrGenBoxTable'] = ws.ErrgenBoxesTable(gsFinal, gsTgt, cri, results.options.errgen_type)
    qtys['metadataTable'] = ws.MetadataTable(gsFinal, results.options, results.parameters)
    qtys['softwareEnvTable'] = ws.SoftwareEnvTable()

    # if Ls and germs available
    if results._LsAndGermInfoSet: #TODO: better way?
        qtys['fiducialListTable'] = ws.GatestringMultiTable(strs, ["Prep.","Measure"], "Fiducials")
        qtys['prepStrListTable'] = ws.GatestringTable(tuple(results.gatestring_lists['prep fiducials']), "Preparation Fiducials")
        qtys['effectStrListTable'] = ws.GatestringTable(tuple(results.gatestring_lists['effect fiducials']), "Measurement Fiducials")
        qtys['germList2ColTable'] = ws.GatestringTable(tuple(results.gatestring_lists['germs']), "Germ", nCols=2)
        
        if results.parameters['objective'] == "logl":
            qtys['progressTable'] = ws.LogLProgressTable(
                results.parameters['max length list'],
                results.gatesets['iteration estimates'],
                results.gatestring_lists['iteration'], ds,
                results.parameters.get('gateLabelAliases',None))
            plotType = "logl"
            mpc = results.parameters['minProbClip']
        else:
            qtys['progressTable'] = ws.Chi2ProgressTable(
                results.parameters['max length list'],
                results.gatesets['iteration estimates'],
                results.gatestring_lists['iteration'], ds,
                results.parameters.get('gateLabelAliases',None))
            plotType = "chi2"
            mpc = results.parameters['minProbClipForWeighting']


        # 2) generate plots
        printer.log("*** Generating plots ***")

        #Create a GatestringStructure (TODO: maybe move this within class?)
        fidPairs = results.parameters['fiducial pairs']
        if fidPairs is None: fidpair_filters = None
        elif isinstance(fidPairs,dict) or hasattr(fidPairs,"keys"):
            #Assume fidPairs is a dict indexed by germ
            fidpair_filters = { (x,y): fidPairs[y] 
                                for x in Ls[st:] for y in germs }
        else:
            #Assume fidPairs is a list
            fidpair_filters = { (x,y): fidPairs
                                for x in Ls[st:] for y in germs }

        Ls = results.parameters['max length list']
        germs = results.gatestring_lists['germs']
        gstr_filters = { (x,y) : results.gatestring_lists['iteration'][i]
                         for i,x in enumerate(Ls)
                         for y in germs }

        gss = GatestringStructure(results.parameters['max length list'],
                                  results.gatestring_lists['germs'],
                                  results.gatestring_lists['prep fiducials'],
                                  results.gatestring_lists['effect fiducials'],
                                  results._getBaseStrDict(),
                                  fidpair_filters, gstr_filters,
                                  results.parameters.get('gateLabelAliases',None))

        qtys['colorBoxPlotKeyPlot'] = ws.BoxKeyPlot(tuple(results.gatestring_lists['prep fiducials']),
                                                    tuple(results.gatestring_lists['effect fiducials']))
        
        qtys['bestEstimateSummedColorBoxPlot'] = ws.ColorBoxPlot(plotType, gss, ds, gsFinal,
                    linlg_pcntle=float(results.parameters['linlogPercentile']) / 100,
                    minProbClipForWeighting=mpc, sumUp=True)

        #Not pagniated currently... just set to same full plot
        qtys['bestEstimateColorBoxPlotPages'] = ws.ColorBoxPlot(plotType, gss, ds, gsFinal,
                    linlg_pcntle=float(results.parameters['linlogPercentile']) / 100,
                    minProbClipForWeighting=mpc)   

    # 3) populate template latex file => report latex file
    printer.log("*** Merging into template file ***")
    
    templateFile = "report_general.html"
    _merge_template(qtys, templateFile, filename)
