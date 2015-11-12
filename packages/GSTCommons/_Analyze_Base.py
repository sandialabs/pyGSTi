import GST as _GST
import os as _os
import warnings as _warnings
import numpy as _np
import sys as _sys

def doLongSequenceGST(longSequenceObjective, 
                      dataFilenameOrSet, targetGateFilenameOrSet,
                      rhoStrsListOrFilename, EStrsListOrFilename,
                      germsListOrFilename, maxLengths, gateLabels, weightsDict,
                      makeListsFn, truncFn, rhoEPairs, constrainToTP, 
                      gaugeOptToCPTP, gaugeOptRatio, advancedOptions,lsgstLists = None):
                    
    cwd = _os.getcwd()

    #Get target gateset
    if isinstance(targetGateFilenameOrSet, str):
        gs_target = _GST.loadGateset(targetGateFilenameOrSet)
    else:
        gs_target = targetGateFilenameOrSet #assume a GateSet object

    #Get dataset
    if isinstance(dataFilenameOrSet, str):
        ds = _GST.loadDataset(dataFilenameOrSet)
        default_dir = _os.path.dirname(dataFilenameOrSet) #default directory for reports, etc
        default_base = _os.path.splitext( _os.path.basename(dataFilenameOrSet) )[0]
    else:
        ds = dataFilenameOrSet #assume a Dataset object
        default_dir = default_base = None

    #Get gate strings and labels
    if gateLabels is None:
        gateLabels = gs_target.keys()

    if isinstance(rhoStrsListOrFilename, str):
        rhoStrs = _GST.loadGatestringList(rhoStrsListOrFilename)
    else: rhoStrs = rhoStrsListOrFilename

    if EStrsListOrFilename is None:
        EStrs = rhoStrs #use same strings for EStrs if EStrsListOrFilename is None
    else:
        if isinstance(EStrsListOrFilename, str):
            EStrs = _GST.loadGatestringList(EStrsListOrFilename)
        else: EStrs = EStrsListOrFilename

    if isinstance(germsListOrFilename, str):
        germs = _GST.loadGatestringList(germsListOrFilename)
    else: germs = germsListOrFilename
    if lsgstLists is None:
        lsgstLists = makeListsFn(gateLabels, rhoStrs, EStrs, germs, maxLengths, rhoEPairs)

    #Starting Point = LGST
    gate_dim = gs_target.get_dimension()
    specs = _GST.getRhoAndESpecs(rhoStrs=rhoStrs, EStrs=EStrs, EVecInds=gs_target.getEVecIndices())
    gs_lgst = _GST.doLGST(ds, specs, gs_target, svdTruncateTo=gate_dim, verbosity=3)

    if constrainToTP: #gauge optimize (and contract if needed) to TP, then lock down first basis element as the identity
        firstElIdentityVec = _np.zeros( (gate_dim,1) )
        firstElIdentityVec[0] = gate_dim**0.25 # first basis el is assumed = sqrt(gate_dim)-dimensional identity density matrix 
        minPenalty, gaugeMx, gs_in_TP = _GST.optimizeGauge(gs_lgst, "TP",  returnAll=True, spamWeight=1.0, gateWeight=1.0, verbosity=3)
        if minPenalty > 0:
            gs_in_TP = _GST.contract(gs_in_TP, "TP")
            if minPenalty > 1e-5: 
                _warnings.warn("Could not gauge optimize to TP (penalty=%g), so contracted LGST gateset to TP" % minPenalty)

        gs_after_gauge_opt = _GST.optimizeGauge(gs_in_TP, "target", targetGateset=gs_target, constrainToTP=True, spamWeight=1.0, gateWeight=1.0)
        gs_after_gauge_opt.set_identityVec( firstElIdentityVec ) # declare that this basis has the identity as its first element

    else: # no TP constraint
        gs_after_gauge_opt = _GST.optimizeGauge(gs_lgst, "target", targetGateset=gs_target, spamWeight=1.0, gateWeight=1.0)
        #OLD: gs_clgst = _GST.contract(gs_after_gauge_opt, "CPTP")
        #TODO: set identity vector, or leave as is, which assumes LGST had the right one and contraction doesn't change it ??

    #Run LSGST on data
    if longSequenceObjective == "chi2":
        gs_lsgst_list = _GST.doIterativeLSGST(ds, gs_after_gauge_opt, lsgstLists, 
                                              minProbClipForWeighting=advancedOptions.get('minProbClipForWeighting',1e-4),
                                              probClipInterval = advancedOptions.get('probClipInterval',(-1e6,1e6)),
                                              returnAll=True, opt_G0=(not constrainToTP), opt_SP0=(not constrainToTP),
                                              gatestringWeightsDict=weightsDict, verbosity=advancedOptions.get('verbosity',2) )
    elif longSequenceObjective == "logL":
        gs_lsgst_list = _GST.doIterativeMLEGST(ds, gs_after_gauge_opt, lsgstLists,
                                               minProbClip = advancedOptions.get('minProbClip',1e-4),
                                               probClipInterval = advancedOptions.get('probClipInterval',(-1e6,1e6)),
                                               radius=advancedOptions.get('radius',1e-4), 
                                               returnAll=True, opt_G0=(not constrainToTP), opt_SP0=(not constrainToTP),
                                               verbosity=advancedOptions.get('verbosity',2) )
    else:
        raise ValueError("Invalid longSequenceObjective: %s" % longSequenceObjective)

    #Run the gatesets through gauge optimization, first to CPTP then to target
    #   so fidelity and frobenius distance w/targets is more meaningful
    if gaugeOptToCPTP:
        print "\nGauge Optimizing to CPTP..."; _sys.stdout.flush()
        go_gs_lsgst_list = [_GST.Core.optimizeGauge(gs,'CPTP', constrainToTP=constrainToTP) for gs in gs_lsgst_list]
    else:
        go_gs_lsgst_list = gs_lsgst_list
        
    for i,gs in enumerate(go_gs_lsgst_list):
        if gaugeOptToCPTP and _GST.JOps.sumOfNegativeJEvals(gs) < 1e-8:  #if a gateset is in CP, then don't let it out (constrain = True)
            go_gs_lsgst_list[i] = _GST.Core.optimizeGauge(gs,'target',targetGateset=gs_target,
                                                          constrainToTP=constrainToTP, constrainToCP=True,
                                                          gateWeight=1, spamWeight=gaugeOptRatio)
        
        else: #otherwise just optimize to the target and forget about CPTP...
            go_gs_lsgst_list[i] = _GST.Core.optimizeGauge(gs,'target',targetGateset=gs_target,
                                                          constrainToTP=constrainToTP, 
                                                          gateWeight=1, spamWeight=gaugeOptRatio)

    ret = _GST.Results()
    ret.init_LsAndGerms(longSequenceObjective, gs_target, ds, 
                        gs_after_gauge_opt, maxLengths, germs,
                        go_gs_lsgst_list, lsgstLists, rhoStrs, EStrs,
                        truncFn,  constrainToTP, rhoEPairs, gs_lsgst_list)
    ret.setAdditionalInfo(advancedOptions.get('minProbClip',1e-4),
                          advancedOptions.get('minProbClipForWeighting',1e-4),
                          advancedOptions.get('probClipInterval',(-1e6,1e6)),
                          advancedOptions.get('radius',1e-4), 
                          weightsDict, default_dir, default_base)

    assert( len(maxLengths) == len(lsgstLists) == len(go_gs_lsgst_list) )
    return ret


#def doLongSequenceAnalysis(longSequenceGSTResults, analysisTypes, analysisOptions, confidenceLevel, templatePath):
#
##HERE
#
##Allowed "types": allPythonTables, allHtmlTables, pdfReport, pdfBrief, pptBrief
## or justTables, 
#
##Basically, this will be a call to:
## 1) GetReportQuantities  <-- formats, whichTables
## 2) writeReport_XXX (optional) <-- type, format of report (full, brief, pdf, ppt, html?); these require certain tables in certain formats
#
##baseFilename, suffix, title, datasetName, appendices, 
##reportFilename, suffix, reportTitle, datasetName, appendices,
#    results = longSequenceGSTResults
#
#    #Get dataset
#    if results.get('dataset filename',None) is not None:
#        ds = _GST.loadDataset(dataFilenameOrSet)
#        default_report_dir = _os.path.dirname(dataFilenameOrSet)
#        default_report_base = _os.path.splitext( _os.path.basename(dataFilenameOrSet) )[0] + suffix
#        if datasetName == "auto":
#            datasetName = _GST.LatexUtil.latex_escaped( _os.path.splitext( _os.path.basename(dataFilenameOrSet) )[0] )
#    else:
#        ds = dataFilenameOrSet #assume a Dataset object
#        default_report_dir = cwd
#        default_report_base = "GSTReport" + suffix 
#        if datasetName == "auto":
#            datasetName = "$\\mathcal{D}$"
#    ret['dataset'] = ds
#
#    
#    #Generate a report
#    Ls = maxLengths
#    L_germ_tuple_to_baseStr_dict = { (L,germ):truncFn(germ,L,False) for L in Ls for germ in germs}
#    ret['color box plot base string dictionary'] = L_germ_tuple_to_baseStr_dict
#
#    st = 1 if maxLengths[0] == 0 else 0 #start index: skips LGST column in report color box plots
#
#
#    formats = []
#    if getPy:      formats.append('py')
#    if getHTML:    formats.append('html')
#    if makeReport: formats.append('latex')
#
#    if len(formats) > 0:
#        print "\nGenerating Tables:"; _sys.stdout.flush()
#        reportQtys = _GST.ReportGeneration.getReportQuantities(formats, longSequenceObjective,
#                                            maxLengths[st:], go_gs_lsgst_list[st:], lsgstLists[st:],
#                                            ds, datasetName, germs, rhoStrs, EStrs, gs_target, "dataTable",
#                                            False, appendices, rhoEPairs, constrainToTP, confidenceLevel,
#                                            minProbClip=1e-6,minProbClipForWeighting=1e-4,
#                                            probClipInterval=(-1e6,1e6),radius=1e-4)
#        ret['confidence region'] = reportQtys['confidenceRegion']
#        ret['gauge appendix gatesets'] = reportQtys['gaugeOptAppendixGatesets']
#
#        if getPy:
#            ret['python tables'] = { tableKey: tableDict['py'] for tableKey,tableDict in reportQtys['tables'].iteritems() }
#
#        if getHTML:
#            ret['html tables'] = { tableKey: tableDict['html'] for tableKey,tableDict in reportQtys['tables'].iteritems() }
#
#        #if makeReport: #commented b/c I don't think this would really be useful (?)
#        #    ret['latex tables'] = { tableKey: tableDict['latex'] for tableKey,tableDict in reportQtys['tables'].iteritems() }
#
#    
#    if makeReport:
#        print "\nGenerating Report:"; _sys.stdout.flush()
#
#        if reportTitle == "auto": reportTitle = "GST report for %s" % datasetName
#
#        if reportFilename != "auto":
#            report_dir = _os.path.dirname(reportFilename)
#            report_base = _os.path.splitext( _os.path.basename(reportFilename) )[0] + suffix
#        else:
#            report_dir = default_report_dir
#            report_base = default_report_base         
#
#        #Note: report_base has no .tex or .pdf extension
#        try:            
#            if report_dir: _os.chdir(report_dir)
#            _GST.ReportGeneration.writeReport_LsAndGerms("%s.pdf" % report_base, longSequenceObjective, 
#                                                         maxLengths[st:], go_gs_lsgst_list[st:], lsgstLists[st:],
#                                                         ds, datasetName, germs, L_germ_tuple_to_baseStr_dict, rhoStrs, EStrs,
#                                                         gs_target, reportTitle,m=0,M=10,appendices=appendices, 
#                                                         templatePath=templatePath, rhoEPairs=rhoEPairs, TPconstrained=constrainToTP,
#                                                         confidenceLevel=confidenceLevel, minProbClip=1e-6,
#                                                         minProbClipForWeighting=1e-4,probClipInterval=(-1e6,1e6),radius=1e-4,
#                                                         precomputed_report_qtys=reportQtys)
#        finally:
#            _os.chdir(cwd)
#
#    return ret
