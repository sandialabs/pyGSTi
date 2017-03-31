from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines report factories. """

import os  as _os
import time as _time
import collections as _collections

from ..objects      import VerbosityPrinter
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
            #print("DB: rendering ",key)
            out = val.render("html") # a dictionary of rendered portions
            qtys_html[key] = "<script>\n%(js)s\n</script>\n\n%(html)s" % out

    #DEBUG
    #testtmp = "%(targetSpamTable)s" % qtys_html
    #print("TEST = \n",qtys_html['targetSpamTable'])
    #print("TEST2 = \n",testtmp)
            
    filled_template = template % qtys_html
      #.format_map(qtys_html) #need python 3.2+
    with open(outputFilename, 'w') as outputfile:
        outputfile.write(filled_template)


def create_single_qubit_report(results, filename, confidenceLevel=None,
                               title="auto", datasetLabel="$\\mathcal{D}$",
                               verbosity=0, comm=None, ws=None):

    """
    Create a "full" single-qubit GST report.  This report gives a detailed and
    analysis that is intended to be applied to `results` of single-qubit GST.
    The report includes background and explanation text to help the user
    interpret the contained results.

    Parameters
    ----------
    results : Results
        A set of GST results, typically obtained from running
        :func:`do_long_sequence_gst`.

    filename : string, optional
       The output filename where the report file(s) will be saved.

    confidenceLevel : float, optional
       If not None, then the confidence level (between 0 and 100) used in
       the computation of confidence regions/intervals. If None, no
       confidence regions or intervals are computed.

    title : string, optional
       The title of the report.  "auto" uses a default title which
       specifyies the label of the dataset as well.

    datasetLabel : string, optional
       A label given to the dataset.

    verbosity : int, optional
       How much detail to send to stdout.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    ws : Workspace, optional
        The workspace used as a scratch space for performing the calculations
        and visualizations required for this report.  If you're creating
        multiple reports with similar tables, plots, etc., it may boost
        performance to use a single Workspace for all the report generation.
    

    Returns
    -------
    None
    """
    printer = VerbosityPrinter.build_printer(verbosity, comm=comm)
    if ws is None:
        ws = _Workspace()

    if title == "auto":
        title = "GST report for %s" % datasetLabel

    # dictionary to store all strings to be inserted into report template
    qtys = {}

    qtys['title'] = title
    qtys['date'] = _time.strftime("%B %d, %Y")
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
    qtys['bestGatesetGatesTable'] = ws.GatesTable(gsFinal, display_as="numbers", confidenceRegionInfo=cri)
    qtys['bestGatesetChoiTable'] = ws.ChoiTable(gsFinal, None, cri, display=('matrix','eigenvalues'))
    qtys['bestGatesetDecompTable'] = ws.GateDecompTable(gsFinal, cri)
    qtys['bestGatesetRotnAxisTable'] = ws.RotationAxisTable(gsFinal, cri, True)
    qtys['bestGatesetVsTargetTable'] = ws.GatesVsTargetTable(gsFinal, gsTgt, cri)
    qtys['bestGatesetErrorGenTable'] = ws.ErrgenTable(gsFinal, gsTgt, cri, ("errgen",),
                                                      "numbers", results.options.errgen_type)
    qtys['metadataTable'] = ws.MetadataTable(gsFinal, results.options, results.parameters)
    qtys['softwareEnvTable'] = ws.SoftwareEnvTable()

    # if Ls and germs available
    if results._LsAndGermInfoSet: #TODO: better way?
        qtys['fiducialListTable'] = ws.GatestringTable(strs, ["Prep.","Measure"], commonTitle="Fiducials")
        qtys['prepStrListTable'] = ws.GatestringTable(results.gatestring_lists['prep fiducials'],
                                                      "Preparation Fiducials")
        qtys['effectStrListTable'] = ws.GatestringTable(results.gatestring_lists['effect fiducials'],
                                                        "Measurement Fiducials")
        qtys['germListTable'] = ws.GatestringTable(results.gatestring_lists['germs'], "Germ")        
        qtys['progressTable'] = ws.FitComparisonTable(
            results.parameters['max length list'],
            results.gatestring_structs['iteration'],
            results.gatesets['iteration estimates'],
            ds, results.parameters['objective'], 'L')

        # 2) generate plots
        printer.log("*** Generating plots ***")

        gss = results.gatestring_structs['final']

        if results.parameters['objective'] == "logl":
            mpc = results.parameters['minProbClip']
        else:
            mpc = results.parameters['minProbClipForWeighting']
        
        qtys['colorBoxPlotKeyPlot'] = ws.BoxKeyPlot(tuple(results.gatestring_lists['prep fiducials']),
                                                    tuple(results.gatestring_lists['effect fiducials']))
        
        qtys['bestEstimateColorBoxPlot'] = ws.ColorBoxPlot(
            results.parameters['objective'], gss, ds, gsFinal,
            linlg_pcntle=float(results.parameters['linlogPercentile']) / 100,
            minProbClipForWeighting=mpc)
        qtys['invertedBestEstimateColorBoxPlot'] = ws.ColorBoxPlot(
            results.parameters['objective'], gss, ds, gsFinal,
            linlg_pcntle=float(results.parameters['linlogPercentile']) / 100,
            minProbClipForWeighting=mpc, invert=True)
    

    # 3) populate template latex file => report latex file
    printer.log("*** Merging into template file ***")

    #Add inline CSS
    qtys['inlineCSS'] = ""
    cssPath = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            "templates","css")
    for cssFile in ("dataTable.css","pygsti_pub.css","pygsti_screen.css"):
        with open(_os.path.join(cssPath,cssFile)) as f:
            qtys['inlineCSS'] += '<style>\n' + str(f.read()) + '\n</style>\n'

    templateFile = "report_singlequbit.html"
    #print("DB inserting choi:\n",qtys['bestGatesetChoiTable'].render("html"))
    #print("DB inserting decomp:\n",qtys['bestGatesetDecompTable'].render("html"))
    _merge_template(qtys, templateFile, filename)
    printer.log("Output written to %s" % filename)




def create_general_report(results, filename, confidenceLevel=None,
                          title="auto", datasetLabel="$\\mathcal{D}$",
                          verbosity=0, comm=None, ws=None):
    """
    Create a "general" GST report.  This report is "general" in that it is
    suited to display results for any number of qubits/qutrits.  Along with
    the results, it includes background and explanation text.

    Parameters
    ----------
    results : Results
        A set of GST results, typically obtained from running
        :func:`do_long_sequence_gst`.

    filename : string, optional
       The output filename where the report file(s) will be saved.

    confidenceLevel : float, optional
       If not None, then the confidence level (between 0 and 100) used in
       the computation of confidence regions/intervals. If None, no
       confidence regions or intervals are computed.

    title : string, optional
       The title of the report.  "auto" uses a default title which
       specifyies the label of the dataset as well.

    datasetLabel : string, optional
       A label given to the dataset.

    verbosity : int, optional
       How much detail to send to stdout.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    ws : Workspace, optional
        The workspace used as a scratch space for performing the calculations
        and visualizations required for this report.  If you're creating
        multiple reports with similar tables, plots, etc., it may boost
        performance to use a single Workspace for all the report generation.
    

    Returns
    -------
    None
    """
    printer = VerbosityPrinter.build_printer(verbosity, comm=comm)
    if ws is None:
        ws = _Workspace()
        
    if title == "auto":
        title = "GST report for %s" % datasetLabel

    # dictionary to store all strings to be inserted into report template
    qtys = {}

    qtys['title'] = title
    qtys['date'] = _time.strftime("%B %d, %Y")
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
    qtys['targetGatesBoxTable'] = ws.GatesTable(gsTgt, display_as="boxes")
    qtys['datasetOverviewTable'] = ws.DataSetOverviewTable(ds, gsTgt, 10, strs)
    qtys['bestGatesetSpamBriefTable'] = ws.SpamTable([gsTgt, gsFinal], ['Target','Estimated'],
                                                     cri, includeHSVec=False)
    qtys['bestGatesetSpamParametersTable'] = ws.SpamParametersTable(gsFinal, cri)
    qtys['bestGatesetSpamVsTargetTable'] = ws.SpamVsTargetTable(gsFinal, gsTgt, cri)
    qtys['bestGatesetGaugeOptParamsTable'] = ws.GaugeOptParamsTable(tuple(results.parameters['gaugeOptParams']))
    qtys['bestGatesetGatesBoxTable'] = ws.GatesTable([gsTgt,gsFinal], ['Target','Estimated'], "boxes", cri)
    qtys['bestGatesetChoiEvalTable'] = ws.ChoiTable(gsFinal, None, cri, display=("eigenvalues","barplot"))
    qtys['bestGatesetEvalTable'] = ws.GateEigenvalueTable(gsFinal, gsTgt, cri)
#    qtys['bestGatesetRelEvalTable'] = OUT!
    qtys['bestGatesetVsTargetTable'] = ws.GatesVsTargetTable(gsFinal, gsTgt, cri)
    qtys['bestGatesetErrGenBoxTable'] = ws.ErrgenTable(gsFinal, gsTgt, cri, ("errgen","H","S"),
                                                       "boxes", results.options.errgen_type)
    qtys['metadataTable'] = ws.MetadataTable(gsFinal, results.options, results.parameters)
    qtys['softwareEnvTable'] = ws.SoftwareEnvTable()

    # if Ls and germs available
    if results._LsAndGermInfoSet: #TODO: better way?
        qtys['fiducialListTable'] = ws.GatestringTable(strs, ["Prep.","Measure"], commonTitle="Fiducials")
        qtys['prepStrListTable'] = ws.GatestringTable(results.gatestring_lists['prep fiducials'],
                                                      "Preparation Fiducials")
        qtys['effectStrListTable'] = ws.GatestringTable(results.gatestring_lists['effect fiducials'],
                                                        "Measurement Fiducials")
        qtys['germList2ColTable'] = ws.GatestringTable(results.gatestring_lists['germs'], "Germ", nCols=2)
        qtys['progressTable'] = ws.FitComparisonTable(
            results.parameters['max length list'],
            results.gatestring_structs['iteration'],
            results.gatesets['iteration estimates'],
            ds, results.parameters['objective'], 'L')

        # 2) generate plots
        printer.log("*** Generating plots ***")

        gss = results.gatestring_structs['final']
        
        if results.parameters['objective'] == "logl":
            mpc = results.parameters['minProbClip']
        else:
            mpc = results.parameters['minProbClipForWeighting']
                
        
        qtys['colorBoxPlotKeyPlot'] = ws.BoxKeyPlot(tuple(results.gatestring_lists['prep fiducials']),
                                                    tuple(results.gatestring_lists['effect fiducials']))
        
        qtys['bestEstimateSummedColorBoxPlot'] = ws.ColorBoxPlot(
            results.parameters['objective'], gss, ds, gsFinal,
            linlg_pcntle=float(results.parameters['linlogPercentile']) / 100,
            minProbClipForWeighting=mpc, sumUp=True)

        #Not pagniated currently... just set to same full plot
        qtys['bestEstimateColorBoxPlotPages'] = ws.ColorBoxPlot(
            results.parameters['objective'], gss, ds, gsFinal,
            linlg_pcntle=float(results.parameters['linlogPercentile']) / 100,
            minProbClipForWeighting=mpc)   

    # 3) populate template latex file => report latex file
    printer.log("*** Merging into template file ***")

    #Add inline CSS
    qtys['inlineCSS'] = ""
    cssPath = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            "templates","css")
    for cssFile in ("dataTable.css","pygsti_pub.css","pygsti_screen.css"):
        with open(_os.path.join(cssPath,cssFile)) as f:
            qtys['inlineCSS'] += '<style>\n' + str(f.read()) + '\n</style>\n'
    
    templateFile = "report_general.html"
    _merge_template(qtys, templateFile, filename)
    printer.log("Output written to %s" % filename)


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
