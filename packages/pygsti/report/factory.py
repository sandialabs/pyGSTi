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
import collections as _collections
import webbrowser as _webbrowser

from ..objects      import VerbosityPrinter
from ..tools        import compattools as _compat
from .workspace import Workspace as _Workspace
from .workspace import WorkspaceTable as _WorkspaceTable

def _merge_template(qtys, templateFilename, outputFilename, auto_open, precision,
                    inlineCSSnames=("dataTable.css","pygsti_pub.css","pygsti_screen.css"),
                    verbosity=0):

    printer = VerbosityPrinter.build_printer(verbosity)

    #Add inline CSS
    if 'inlineCSS' not in qtys:
        qtys['inlineCSS'] = ""
        cssPath = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                "templates","css")
        for cssFile in inlineCSSnames:
            with open(_os.path.join(cssPath,cssFile)) as f:
                contents = f.read()
                try: # to convert to unicode since we're using unicode literals below
                    contents = contents.decode('utf-8')
                except AttributeError: pass #Python3 case when unicode is read in natively (no need to decode)
                qtys['inlineCSS'] += '<style>\n%s</style>\n' % contents

    #Insert qtys into template file
    templateFilename = _os.path.join( _os.path.dirname(_os.path.abspath(__file__)),
                                      "templates", templateFilename )

    template = ''
    with open(templateFilename, 'r') as templatefile:
        template = templatefile.read()
        
    try: # convert to unicode if Python2 
        template = template.decode('utf-8')
    except AttributeError: pass #Python3 case
        
    qtys_html = _collections.defaultdict(lambda x=0: "BLANK")
    for key,val in qtys.items():
        if _compat.isstr(val):
            qtys_html[key] = val

        else:
            #print("DB: rendering ",key)
            if isinstance(val,_WorkspaceTable):
                #supply precision argument
                out = val.render("html",precision=precision)
            else:
                out = val.render("html") # a dictionary of rendered portions
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

    

def create_single_qubit_report(results, filename, confidenceLevel=None,
                               title="auto", datasetLabel="$\\mathcal{D}$",
                               linlogPercentile=5, errgen_type="logTiG",
                               precision=None, brief=False,
                               comm=None, ws=None, auto_open=False, verbosity=0):

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

    verbosity : int, optional
       How much detail to send to stdout.
    

    Returns
    -------
    None
    """
    printer = VerbosityPrinter.build_printer(verbosity, comm=comm)
    if ws is None: ws = _Workspace()

    if title == "auto":
        title = "GST report for %s" % datasetLabel
    
    qtys = {} # stores strings to be inserted into report template
    qtys['title'] = title
    qtys['date'] = _time.strftime("%B %d, %Y")
    qtys['confidenceLevel'] = "%d" % \
        confidenceLevel if confidenceLevel is not None else "NOT-SET"
    qtys['linlg_pcntle'] = "%d" % round(linlogPercentile) #to nearest %
    qtys['datasetLabel'] = datasetLabel
    qtys['errorgenformula'] = _errgen_formula(errgen_type)
        
    if confidenceLevel is not None:
        cri = results.get_confidence_region("go0","final",confidenceLevel, comm=comm)
        qtys['confidenceIntervalScaleFctr'] = "%.3g" % cri.intervalScaling
        qtys['confidenceIntervalNumNonGaugeParams'] = "%d" % cri.nNonGaugeParams
    else: cri = None


    # Generate Tables
    printer.log("*** Generating tables ***")

    gsTgt = results.gatesets['target']
    gsFinal = results.gatesets['go0']
    ds = results.dataset
    prepStrs = results.gatestring_lists['prep fiducials']
    effectStrs = results.gatestring_lists['effect fiducials']
    germs = results.gatestring_lists['germs']

    qtys['targetSpamTable'] = ws.SpamTable(gsTgt)
    qtys['targetGatesTable'] = ws.GatesTable(gsTgt)
    qtys['datasetOverviewTable'] = ws.DataSetOverviewTable(ds, gsTgt, 10, (prepStrs,effectStrs))
    qtys['bestGatesetSpamTable'] = ws.SpamTable(gsFinal, None, cri)
    qtys['bestGatesetSpamParametersTable'] = ws.SpamParametersTable(gsFinal, cri)
    qtys['bestGatesetGaugeOptParamsTable'] = ws.GaugeOptParamsTable(results.goparameters['go0'])
    qtys['bestGatesetGatesTable'] = ws.GatesTable(gsFinal, display_as="numbers", confidenceRegionInfo=cri)
    qtys['bestGatesetChoiTable'] = ws.ChoiTable(gsFinal, None, cri, display=('matrix','eigenvalues'))
    qtys['bestGatesetDecompTable'] = ws.GateDecompTable(gsFinal, cri)
    qtys['bestGatesetRotnAxisTable'] = ws.RotationAxisTable(gsFinal, cri, True)
    qtys['bestGatesetVsTargetTable'] = ws.GatesVsTargetTable(gsFinal, gsTgt, cri)
    qtys['bestGatesetErrorGenTable'] = ws.ErrgenTable(gsFinal, gsTgt, cri, ("errgen",),
                                                      "numbers", errgen_type)
    qtys['metadataTable'] = ws.MetadataTable(gsFinal, results.parameters)
    qtys['softwareEnvTable'] = ws.SoftwareEnvTable()

    # Ls and germs specific
    qtys['fiducialListTable'] = ws.GatestringTable((prepStrs,effectStrs),
                                                   ["Prep.","Measure"], commonTitle="Fiducials")
    qtys['prepStrListTable'] = ws.GatestringTable(prepStrs,"Preparation Fiducials")
    qtys['effectStrListTable'] = ws.GatestringTable(effectStrs,"Measurement Fiducials")
    qtys['germListTable'] = ws.GatestringTable(germs, "Germ")
    qtys['progressTable'] = ws.FitComparisonTable(
        results.parameters['max length list'],
        results.gatestring_structs['iteration'],
        results.gatesets['iteration estimates'],
        ds, results.parameters['objective'], 'L')

    # Generate plots
    printer.log("*** Generating plots ***")

    gss = results.gatestring_structs['final']
    if results.parameters['objective'] == "logl":
        mpc = results.parameters['minProbClip']
    else:
        mpc = results.parameters['minProbClipForWeighting']
    
    qtys['colorBoxPlotKeyPlot'] = ws.BoxKeyPlot(prepStrs, effectStrs)
    qtys['bestEstimateColorBoxPlot'] = ws.ColorBoxPlot(
        results.parameters['objective'], gss, ds, gsFinal,
        linlg_pcntle=float(linlogPercentile) / 100,
        minProbClipForWeighting=mpc)
    qtys['invertedBestEstimateColorBoxPlot'] = ws.ColorBoxPlot(
        results.parameters['objective'], gss, ds, gsFinal,
        linlg_pcntle=float(linlogPercentile) / 100,
        minProbClipForWeighting=mpc, invert=True)
    
    # Populate template latex file => report latex file
    printer.log("*** Merging into template file ***")
    #print("DB inserting choi:\n",qtys['bestGatesetChoiTable'].render("html"))
    #print("DB inserting decomp:\n",qtys['bestGatesetDecompTable'].render("html"))
    template = "report_singlequbit_brief.html" if brief else "report_singlequbit.html"
    _merge_template(qtys, template, filename, auto_open, precision,
                    verbosity=printer)

    

def create_general_report(results, filename, confidenceLevel=None,
                          title="auto", datasetLabel="$\\mathcal{D}$",
                          linlogPercentile=5, errgen_type="logTiG",
                          precision=None, brief=False,
                          comm=None, ws=None, auto_open=False, verbosity=0):
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

    verbosity : int, optional
       How much detail to send to stdout.
    

    Returns
    -------
    None
    """
    printer = VerbosityPrinter.build_printer(verbosity, comm=comm)
    if ws is None: ws = _Workspace()
        
    if title == "auto":
        title = "GST report for %s" % datasetLabel

    qtys = {} # stores strings to be inserted into report template
    qtys['title'] = title
    qtys['date'] = _time.strftime("%B %d, %Y")
    qtys['confidenceLevel'] = "%d" % \
        confidenceLevel if confidenceLevel is not None else "NOT-SET"
    qtys['linlg_pcntle'] = "%d" % round(linlogPercentile) #to nearest %
    qtys['datasetLabel'] = datasetLabel
    qtys['errorgenformula'] = _errgen_formula(errgen_type)
        
    if confidenceLevel is not None:
        cri = results.get_confidence_region("go0","final",confidenceLevel, comm=comm)
        qtys['confidenceIntervalScaleFctr'] = "%.3g" % cri.intervalScaling
        qtys['confidenceIntervalNumNonGaugeParams'] = "%d" % cri.nNonGaugeParams
    else: cri = None


    # Generate Tables
    printer.log("*** Generating tables ***")

    gsTgt = results.gatesets['target']
    gsFinal = results.gatesets['go0']
    ds = results.dataset
    prepStrs = results.gatestring_lists['prep fiducials']
    effectStrs = results.gatestring_lists['effect fiducials']
    germs = results.gatestring_lists['germs']

    qtys['targetSpamBriefTable'] = ws.SpamTable(gsTgt, None, includeHSVec=False)
    qtys['targetGatesBoxTable'] = ws.GatesTable(gsTgt, display_as="boxes")
    qtys['datasetOverviewTable'] = ws.DataSetOverviewTable(ds, gsTgt, 10, (prepStrs,effectStrs))
    qtys['bestGatesetSpamBriefTable'] = ws.SpamTable([gsTgt, gsFinal], ['Target','Estimated'],
                                                     cri, includeHSVec=False)
    qtys['bestGatesetSpamParametersTable'] = ws.SpamParametersTable(gsFinal, cri)
    qtys['bestGatesetSpamVsTargetTable'] = ws.SpamVsTargetTable(gsFinal, gsTgt, cri)
    qtys['bestGatesetGaugeOptParamsTable'] = ws.GaugeOptParamsTable(results.goparameters['go0'])
    qtys['bestGatesetGatesBoxTable'] = ws.GatesTable([gsTgt,gsFinal], ['Target','Estimated'], "boxes", cri)
    qtys['bestGatesetChoiEvalTable'] = ws.ChoiTable(gsFinal, None, cri, display=("eigenvalues","barplot"))
    qtys['bestGatesetEvalTable'] = ws.GateEigenvalueTable(gsFinal, gsTgt, cri)
    qtys['bestGatesetVsTargetTable'] = ws.GatesVsTargetTable(gsFinal, gsTgt, cri)
    qtys['bestGatesetErrGenBoxTable'] = ws.ErrgenTable(gsFinal, gsTgt, cri, ("errgen","H","S"),
                                                       "boxes", errgen_type)
    qtys['metadataTable'] = ws.MetadataTable(gsFinal, results.parameters)
    qtys['softwareEnvTable'] = ws.SoftwareEnvTable()

    #Ls and Germs specific
    qtys['fiducialListTable'] = ws.GatestringTable((prepStrs,effectStrs),
                                                   ["Prep.","Measure"], commonTitle="Fiducials")
    qtys['prepStrListTable'] = ws.GatestringTable(prepStrs,"Preparation Fiducials")
    qtys['effectStrListTable'] = ws.GatestringTable(effectStrs,"Measurement Fiducials")
    qtys['germList2ColTable'] = ws.GatestringTable(germs, "Germ", nCols=2)
    qtys['progressTable'] = ws.FitComparisonTable(
        results.parameters['max length list'],
            results.gatestring_structs['iteration'],
        results.gatesets['iteration estimates'],
        ds, results.parameters['objective'], 'L')
    
    # Generate plots
    printer.log("*** Generating plots ***")
        
    gss = results.gatestring_structs['final']
    if results.parameters['objective'] == "logl":
        mpc = results.parameters['minProbClip']
    else:
        mpc = results.parameters['minProbClipForWeighting']
        
    qtys['colorBoxPlotKeyPlot'] = ws.BoxKeyPlot(prepStrs, effectStrs)        
    qtys['bestEstimateSummedColorBoxPlot'] = ws.ColorBoxPlot(
        results.parameters['objective'], gss, ds, gsFinal,
        linlg_pcntle=float(linlogPercentile) / 100,
        minProbClipForWeighting=mpc, sumUp=True)
    
    #Not pagniated currently... just set to same full plot
    qtys['bestEstimateColorBoxPlotPages'] = ws.ColorBoxPlot(
        results.parameters['objective'], gss, ds, gsFinal,
        linlg_pcntle=float(linlogPercentile) / 100,
        minProbClipForWeighting=mpc)   

    # 3) populate template latex file => report latex file
    printer.log("*** Merging into template file ***")
    template = "report_general_brief.html" if brief else "report_general.html"
    _merge_template(qtys, template, filename, auto_open,
                    precision, verbosity=printer)



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
