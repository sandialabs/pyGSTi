def create_general_report(results, filename, confidenceLevel=None,
                          title="auto",
                          datasetLabel="<span class='math'>\\mathcal{D}</span>",
                          linlogPercentile=5, errgen_type="logTiG",
                          nmthreshold=50, precision=None,
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
    printer.log('*** Creating workspace ***')
    if ws is None: ws = _ws.Workspace()
        
    if title == "auto":
        title = "GST report for %s" % datasetLabel

    results_dict = results if isinstance(results, dict) else {"unique": results}
    toggles = _set_toggles(results_dict)

    qtys = {} # stores strings to be inserted into report template

    # Timing is done with these blocks instead:
    printer.log('Adding metadata', 2)

    qtys['title'] = title
    qtys['date'] = _time.strftime("%B %d, %Y")
    qtys['confidenceLevel'] = "%d" % \
        confidenceLevel if confidenceLevel is not None else "NOT-SET"
    qtys['linlg_pcntle'] = "%d" % round(linlogPercentile) #to nearest %
    qtys['datasetLabel'] = datasetLabel
    qtys['errorgenformula'] = _errgen_formula(errgen_type)

    # Generate Tables
    printer.log("*** Generating switchboard tables ***")

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

    #goView = [multidataset,multiest,multiGO,False]
    #maxLView = [multidataset,multiest,False,multiL]
    goView = [False,False,multiGO,False]
    maxLView = [False,False,False,multiL]

    qtys['topSwitchboard'] = switchBd
    qtys['goSwitchboard1'] = switchBd.view(goView,"v1")
    qtys['goSwitchboard2'] = switchBd.view(goView,"v2")
    qtys['maxLSwitchboard1'] = switchBd.view(maxLView,"v6")

    gsTgt = switchBd.gsTarget
    ds = switchBd.ds
    eff_ds = switchBd.eff_ds
    prepStrs = switchBd.prepStrs
    effectStrs = switchBd.effectStrs
    germs = switchBd.germs
    strs = switchBd.strs

    qtys['targetSpamBriefTable'] = ws.SpamTable(gsTgt, None, includeHSVec=False)
    qtys['targetGatesBoxTable'] = ws.GatesTable(gsTgt, display_as="boxes")
    qtys['datasetOverviewTable'] = ws.DataSetOverviewTable(ds)

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
    qtys['bestGatesetDecompTable'] = ws.GateDecompTable(gsFinal, gsTgt, cri)
    qtys['bestGatesetEvalTable'] = ws.GateEigenvalueTable(gsFinal, gsTgt, cri, display=('evals','log-evals'))
    qtys['bestGatesetRelEvalTable'] = ws.GateEigenvalueTable(gsFinal, gsTgt, cri, display=('rel','log-rel'))
    qtys['bestGatesetVsTargetTable'] = ws.GatesVsTargetTable(gsFinal, gsTgt, cri)
    qtys['bestGatesetVsTargetTable_sum'] = ws.GatesVsTargetTable(gsFinal, gsTgt, cri)
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

    qtys['gramBarPlot'] = ws.GramMatrixBarPlot(ds,gsTgt,10,strs)
    qtys['progressBarPlot'] = ws.FitComparisonBarPlot(
        Ls, gssAllL, switchBd.gsAllL, eff_ds, switchBd.objective, 'L')
                
    qtys['dataScalingColorBoxPlot'] = ws.ColorBoxPlot(
        "scaling", switchBd.gssFinal, eff_ds, switchBd.gsFinalIter,
            submatrices=switchBd.scaledSubMxsDict)
    
    #Not pagniated currently... just set to same full plot
    qtys['bestEstimateColorScatterPlot'] = ws.ColorBoxPlot(
        switchBd.objective, gss, eff_ds, gsL,
        linlg_pcntle=float(linlogPercentile) / 100,
        minProbClipForWeighting=switchBd.mpc, scatter=True) #TODO: L-switchboard on summary page?
    qtys['bestEstimateColorScatterPlot'].set_render_options(click_to_display=True)
    #  Fast enough now thanks to scattergl, but webgl render issues so need to delay creation 

    if multidataset:
        #initialize a new "dataset comparison switchboard"
        dscmp_switchBd = ws.Switchboard(
            ["Dataset1","Dataset2"],
            [dataset_labels, dataset_labels],
            ["buttons","buttons"], [0,1]
        )
        dscmp_switchBd.add("dscmp",(0,1))
        dscmp_switchBd.add("dscmp_gss",(0,))

        for d1, dslbl1 in enumerate(dataset_labels):
            dscmp_switchBd.dscmp_gss[d1] = results_dict[dslbl1].gatestring_structs['final']

        dsComp = dict()
        indices = []
        for i in range(len(dataset_labels)):
            for j in range(len(dataset_labels)):
                indices.append((i, j))
        if comm is not None:
            _, indexDict, _ = _distribute_indices(indices, comm)
            rank = comm.Get_rank()
            for k, v in indexDict.items():
                if v == rank:
                    d1, d2 = k
                    dslbl1 = dataset_labels[d1]
                    dslbl2 = dataset_labels[d2]

                    ds1 = results_dict[dslbl1].dataset
                    ds2 = results_dict[dslbl2].dataset
                    dsComp[(d1, d2)] = _DataComparator(
                        [ds1, ds2], DS_names=[dslbl1, dslbl2])
            dicts = comm.gather(dsComp, root=0)
            if rank == 0:
                for d in dicts:
                    for k, v in d.items():
                        d1, d2 = k
                        dscmp_switchBd.dscmp[d1, d2] = v
        else:
            for d1, d2 in indices:
                dslbl1 = dataset_labels[d1]
                dslbl2 = dataset_labels[d2]
                ds1 = results_dict[dslbl1].dataset
                ds2 = results_dict[dslbl2].dataset
                dscmp_switchBd.dscmp[d1, d2] = _DataComparator([ds1, ds2], DS_names=[dslbl1,dslbl2])
        
        qtys['dscmpSwitchboard'] = dscmp_switchBd
        qtys['dsComparisonHistogram'] = ws.DatasetComparisonPlot(dscmp_switchBd.dscmp)
        qtys['dsComparisonBoxPlot'] = ws.ColorBoxPlot('dscmp', dscmp_switchBd.dscmp_gss,
                                                      None, None, dscomparator=dscmp_switchBd.dscmp)
        toggles['CompareDatasets'] = True
    else:
        toggles['CompareDatasets'] = False
