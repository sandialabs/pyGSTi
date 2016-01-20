""" Defines the Results class and supporting functionality."""

import sys as _sys
import os as _os
import re as _re
import collections as _collections
import matplotlib as _matplotlib

from ..objects import gatestring as _gs
from ..construction import spamspecconstruction as _ssc
from ..algorithms import optimize_gauge as _optimizeGauge

import latex as _latex
import generation as _generation
import plotting as _plotting

class Results(object):
    """
    Encapsulates a set of GST results.

    A Results object is constructed from the input parameters and raw output
    gatesets from a GST calculation, typically performed by one of the
    "do<something>" methods of GST.Core, and acts as a end-output factory
    (creating reports, presentations, etc), and a derived-results cache
    (so derived quantities don't need to be recomputed many times for 
    different output formats).
    """

    def __init__(self, restrictToFormats=None, templatePath=None, latexCmd="pdflatex"):
        """ 
        Initialize a Results object.

        Parameters
        ----------
        restrictToFormats : tuple or None
            A tuple of format names to restrict internal computation
            to.  This parameter should be left as None unless you 
            know what you're doing.

        templatePath : string or None
            A local path to the stored GST report template files.  The
            default value of None means to use the default path, which
            is almost always what you want.
        """
        self.confidenceRegions = {} # key == confidence level, val = ConfidenceRegion
        self.tables = {} #dict of dicts.  Outer dict key is confidence level
        self.figures = {} #plain dict.  Key is figure name (a figure applies to all confidence levels)
        self.specials = {} #plain dict.  Key is name of "special" object

        if restrictToFormats is not None:
            self.formatsToCompute = restrictToFormats
        else:
            self.formatsToCompute = ('py','html','latex','ppt') #all formats
        self.longTables = False
        self.tableClass = "dataTable"

        self.templatePath = None
        self.latexCmd = latexCmd
        self.bEssentialResultsSet = False
        self.LsAndGermInfoSet = False
        self.set_additional_info() #set all default values of additional info

    def init_single(self, objective, targetGateset, dataset, gatesetEstimate,
                    gatestring_list, constrainToTP, gatesetEstimate_noGaugeOpt=None):
        """ 
        Initialize this Results object from the inputs and outputs of a
        single (non-iterative) GST method.
        

        Parameters
        ----------
        objective : {'chi2', 'logl'}
            Whether gateset was obtained by minimizing chi^2 or
            maximizing the log-likelihood.
            
        targetGateset : GateSet
            The target gateset used when optimizing the objective.

        dataset : DataSet
            The dataset used when optimizing the objective.

        gatesetEstimate : GateSet
            The (single) gate set obtained which optimizes the objective.

        gatestring_list : list of GateStrings
            The list of gate strings used to optimize the objective.

        constrainToTP : boolean
            Whether or not the gatesetEstimate was constrained to lie
            within TP during the objective optimization.

        gatesetEstimate_noGaugeOpt : GateSet, optional
            The value of the estimated gate set *before* any gauge
            optimization was performed on it.
        
        Returns
        -------
        None
        """
        
        # Set essential info: gateset estimates(s) but no particular
        # structure known about gateStringLists.
        self.gsTarget = targetGateset
        self.gatesets = [ gatesetEstimate ]
        self.gsBestEstimate = gatesetEstimate
        self.gateStringLists = [ gatestring_list ]
        self.bestGateStringList = gatestring_list
        self.dataset = dataset
        self.objective = objective
        self.constrainToTP = constrainToTP
        if gatesetEstimate_noGaugeOpt is not None:
            self.gatesetEstimates_noGO = [ gatesetEstimate_noGaugeOpt ]
        else: 
            self.gatesetEstimates_noGO = None
        self.gsSeed = None
        self.bEssentialResultsSet = True

    def init_Ls_and_germs(self, objective, targetGateset, dataset,
                              seedGateset, Ls, germs, gatesetsByL, gateStringListByL, 
                              rhoStrs, EStrs, truncFn, constrainToTP, rhoEPairs=None,
                              gatesetsByL_noGaugeOpt=None):

        """ 
        Initialize this Results object from the inputs and outputs of
        an iterative GST method based on gate string lists containing
        germs repeated up to a maximum-L value that increases with 
        iteration.

        Parameters
        ----------
        objective : {'chi2', 'logl'}
            Whether gateset was obtained by minimizing chi^2 or
            maximizing the log-likelihood.
            
        targetGateset : GateSet
            The target gateset used when optimizing the objective.

        dataset : DataSet
            The dataset used when optimizing the objective.

        seedGateset : GateSet
            The initial gateset used to seed the iterative part
            of the objective optimization.  Typically this is
            obtained via LGST.

        Ls : list of ints
            List of maximum-L values used in the iterations.

        germs : list of GateStrings
            List of germ gate strings used in the objective optimization.
            
        gatesetsByL : list of GateSets
            The estimated gateset at each L value.

        gateStringListByL : list of lists of GateStrings
            The gate string list used at each L value.

        rhoStrs : list of GateStrings
            The list of state preparation fiducial strings
            in the objective optimization.

        EStrs : list of GateStrings
            The list of measurement fiducial strings
            in the objective optimization.

        truncFn : function
            The truncation function used, indicating how a
            germ should be repeated "L times".  Function should
            take parameters (germ, L) and return the repeated
            gate string.  For example, see
            pygsti.construction.repeat_with_max_length.

        constrainToTP : boolean
            Whether or not the gatesetEstimate was constrained to lie
            within TP during the objective optimization.
            
        rhoEPairs : list of 2-tuples, optional
            Specifies a subset of all rhoStr,EStr string pairs to be used in this
            analysis.  Each element of rhoEPairs is a (iRhoStr, iEStr) 2-tuple of integers,
            which index a string within the state preparation and measurement fiducial
            strings respectively.

        gatesetsByL_noGaugeOpt : list of GateSets, optional
            The value of the estimated gate sets *before* any gauge
            optimization was performed on it.
        
        Returns
        -------
        None
        """

        assert(len(gateStringListByL) == len(gatesetsByL) == len(Ls))

        # Set essential info: gateset estimates(s) but no particular
        # structure known about gateStringLists.
        self.gsTarget = targetGateset
        self.gsSeed = seedGateset
        self.gatesets = gatesetsByL
        self.gsBestEstimate = gatesetsByL[-1]
        self.gateStringLists = gateStringListByL
        self.bestGateStringList = gateStringListByL[-1]
        self.dataset = dataset
        self.objective = objective
        self.constrainToTP = constrainToTP
        if gatesetsByL_noGaugeOpt is not None:
            self.gatesetEstimates_noGO = gatesetsByL_noGaugeOpt
        else: 
            self.gatesetEstimates_noGO = None
        self.bEssentialResultsSet = True

        #Set "Ls and germs" info: gives particular structure
        # to the gateStringLists used to obtain estimates
        self.rhoStrs = rhoStrs
        self.EStrs = EStrs
        self.germs = germs
        self.Ls = Ls
        self.rhoEPairs = rhoEPairs
        self.L_germ_tuple_to_baseStr_dict = { (L,germ):truncFn(germ,L) for L in Ls for germ in germs}
        self.LsAndGermInfoSet = True


    def set_additional_info(self,minProbClip=1e-6, minProbClipForWeighting=1e-4,
                          probClipInterval=(-1e6,1e6), radius=1e-4,
                          weightsDict=None, defaultDirectory=None, defaultBasename=None):
        """
        Set advanced parameters for producing derived outputs.  Usually the default
        values are fine (which is why setting these inputs is separated into a
        separate function).

        Parameters
        ----------
        minProbClip : float, optional
            The minimum probability treated normally in the evaluation of the log-likelihood.
            A penalty function replaces the true log-likelihood for probabilities that lie
            below this threshold so that the log-likelihood never becomes undefined (which improves
            optimizer performance).
    
        minProbClipForWeighting : float, optional
            Sets the minimum and maximum probability p allowed in the chi^2 weights: N/(p*(1-p))
            by clipping probability p values to lie within the interval
            [ minProbClipForWeighting, 1-minProbClipForWeighting ].
    
        probClipInterval : 2-tuple or None, optional
           (min,max) values used to clip the probabilities predicted by gatesets during the
           least squares search for an optimal gateset (if not None).

        radius : float, optional
           Specifies the severity of rounding used to "patch" the zero-frequency
           terms of the log-likelihood.

        weightsDict : dict, optional
           A dictionary with keys == gate strings and values == multiplicative scaling 
           factor for the corresponding gate string. The default is no weight scaling at all.
           
        defaultDirectory : string, optional
           Path to the default directory for generated reports and presentations.

        defaultBasename : string, optional
           Default basename for generated reports and presentations.

        Returns
        -------
        None
        """


        self.additionalInfo = { 'weights': weightsDict, 
                                'minProbClip': minProbClip, 
                                'minProbClipForWeighting': minProbClipForWeighting,
                                'probClipInterval': probClipInterval,
                                'radius': radius,
                                'hessianProjection': 'std',
                                'defaultDirectory': defaultDirectory,
                                'defaultBasename': defaultBasename }

    def set_template_path(self, pathToTemplates):
        """
        Sets the location of GST report and presentation templates.

        Parameters
        ----------
        pathToTemplates : string
           The path to a folder containing GST's template files.  
           Usually this can be determined automatically (the default).
        """
        self.templatePath = pathToTemplates


    def set_latex_cmd(self, latexCmd):
        """
        Sets the shell command used for compiling latex reports and
        presentations.

        Parameters
        ----------
        latexCmd : string
           The command to run to invoke the latex compiler,
           typically just 'pdflatex' when it is on the system
           path. 
        """
        self.latexCmd = latexCmd


    def get_confidence_region(self, confidenceLevel):
        """
        Get the ConfidenceRegion object associated with a given confidence level.
        One will be created and cached the first time given level is requested,
        and future requests will return the cached object.

        Parameters
        ----------
        confidenceLevel : float
           The confidence level (between 0 and 100).

        Returns
        -------
        ConfidenceRegion
        """
        
        assert(self.bEssentialResultsSet)

        if confidenceLevel is None:
            return None

        if confidenceLevel not in self.confidenceRegions:
            if self.objective == "logl":
                cr = _generation.get_logl_confidence_region(
                    self.gsBestEstimate, self.dataset, confidenceLevel,
                    self.constrainToTP, self.bestGateStringList,
                    self.additionalInfo['probClipInterval'],
                    self.additionalInfo['minProbClip'],
                    self.additionalInfo['radius'],
                    self.additionalInfo['hessianProjection'])
            elif self.objective == "chi2":
                cr = _generation.get_chi2_confidence_region(
                    self.gsBestEstimate, self.dataset, confidenceLevel,
                    self.constrainToTP, self.bestGateStringList,
                    self.additionalInfo['probClipInterval'],
                    self.additionalInfo['minProbClipForWeighting'],
                    self.additionalInfo['hessianProjection'])
            else:
                raise ValueError("Invalid objective given in essential info: %s" % self.objective)
            self.confidenceRegions[confidenceLevel] = cr

        return self.confidenceRegions[confidenceLevel]


    def get_table(self, tableName, confidenceLevel=None, fmt="py", verbosity=0):
        """
        Get a report table in a specified format.  Tables are created on 
        the first request then cached for later requests for the same table.
        This method is typically used internally by other Results methods.

        Parameters
        ----------
        tableName : string
           The name of the table.

        confidenceLevel : float, optional
           If not None, then the confidence level (between 0 and 100) used to
           put error bars on the table's values (if possible). If None, no 
           confidence regions or intervals are included.

        fmt : { 'py', 'html', 'latex', 'ppt' }, optional
           The format of the table to be returned.

        verbosity : int, optional
           How much detail to send to stdout.

        Returns
        -------
        string or object
           The requested table in the requested format.  'py' and 'ppt'
           tables are objects, 'html' and 'latex' tables are strings.
        """
        assert(self.bEssentialResultsSet)
        if self.tables.has_key(confidenceLevel) == False:
            self.tables[confidenceLevel] = {}
        if tableName not in self.tables[confidenceLevel]:
            self.tables[confidenceLevel][tableName] = self._generateTable(tableName, confidenceLevel, verbosity)
        return self.tables[confidenceLevel][tableName][fmt]

    def _generateTable(self, tableName, confidenceLevel, verbosity):
        """ 
        Switchboard method for actually creating a table (including computation
        of its values.
        """
        assert(self.bEssentialResultsSet)
        gaugeOptAppendixTablenames = [ 'best%sGateset%sTable' % (a,b) for a in ('Target','TargetSpam','TargetGates','CPTP','TP') \
                                           for b in ('Spam','SpamParameters','Gates','Choi','Decomp','ClosestUnitary','VsTarget') ]

        if verbosity > 0:
            print "Generating %s table..." % tableName; _sys.stdout.flush()

        # target gateset tables
        if tableName == 'targetSpamTable': 
            return _generation.get_gateset_spam_table(
                self.gsTarget, self.formatsToCompute, self.tableClass,
                self.longTables)
        elif tableName == 'targetGatesTable':
            return _generation.get_unitary_gateset_gates_table(
                self.gsTarget, self.formatsToCompute, self.tableClass,
                self.longTables)

        # dataset and gatestring list tables
        elif tableName == 'datasetOverviewTable':
            return _generation.get_dataset_overview_table(
                self.dataset, self.formatsToCompute, self.tableClass,
                self.longTables)

        elif tableName == 'fiducialListTable':
            return _generation.get_gatestring_multi_table(
                [self.rhoStrs, self.EStrs], ["Prep.","Measure"],
                self.formatsToCompute, self.tableClass, self.longTables,
                "Fiducials")

        elif tableName == 'rhoStrListTable':
            return _generation.get_gatestring_table(
                self.rhoStrs, "Preparation Fiducial", self.formatsToCompute, 
                self.tableClass, self.longTables)

        elif tableName == 'EStrListTable':
            return _generation.get_gatestring_table(
                self.EStrs, "Measurement Fiducial", self.formatsToCompute,
                self.tableClass, self.longTables)

        elif tableName == 'germListTable':
            return _generation.get_gatestring_table(
                self.germs, "Germ", self.formatsToCompute, self.tableClass,
                self.longTables)


        # Estimated gateset tables
        elif tableName == 'bestGatesetSpamTable':
            cri = self.get_confidence_region(confidenceLevel)
            return _generation.get_gateset_spam_table(
                self.gsBestEstimate, self.formatsToCompute, self.tableClass,
                self.longTables, cri)

        elif tableName == 'bestGatesetSpamParametersTable':
            cri = self.get_confidence_region(confidenceLevel)
            return _generation.get_gateset_spam_parameters_table(
                self.gsBestEstimate, self.formatsToCompute, self.tableClass,
                self.longTables, cri)

        elif tableName == 'bestGatesetGatesTable':
            cri = self.get_confidence_region(confidenceLevel)
            return _generation.get_gateset_gates_table(
                self.gsBestEstimate, self.formatsToCompute, self.tableClass,
                self.longTables, cri)

        elif tableName == 'bestGatesetChoiTable':
            cri = self.get_confidence_region(confidenceLevel)
            return _generation.get_gateset_choi_table(
                self.gsBestEstimate, self.formatsToCompute, self.tableClass,
                self.longTables, cri)

        elif tableName == 'bestGatesetDecompTable':
            cri = self.get_confidence_region(confidenceLevel)
            return _generation.get_gateset_decomp_table(
                self.gsBestEstimate, self.formatsToCompute, self.tableClass,
                self.longTables, cri)

        elif tableName == 'bestGatesetRotnAxisTable':
            cri = self.get_confidence_region(confidenceLevel)
            return _generation.get_gateset_rotn_axis_table(
                self.gsBestEstimate, self.formatsToCompute, self.tableClass,
                self.longTables, cri)

        elif tableName == 'bestGatesetClosestUnitaryTable':
            #cri = self.get_confidence_region(confidenceLevel)
            return _generation.get_gateset_closest_unitary_table(
                self.gsBestEstimate, self.formatsToCompute, self.tableClass,
                self.longTables) #, cri)

        elif tableName == 'bestGatesetVsTargetTable':
            cri = self.get_confidence_region(confidenceLevel)
            return _generation.get_gateset_vs_target_table(
                self.gsBestEstimate, self.gsTarget, self.formatsToCompute,
                self.tableClass, self.longTables, cri)

        elif tableName == 'bestGatesetErrorGenTable':
            cri = self.get_confidence_region(confidenceLevel)
            return _generation.get_gateset_vs_target_err_gen_table(
                self.gsBestEstimate, self.gsTarget, self.formatsToCompute,
                self.tableClass, self.longTables, cri)


        # progress tables
        elif tableName == 'chi2ProgressTable':
            assert(self.LsAndGermInfoSet)
            return _generation.get_chi2_progress_table(
                self.Ls, self.gatesets, self.gateStringLists, self.dataset,
                self.constrainToTP, self.formatsToCompute, self.tableClass,
                self.longTables)

        elif tableName == 'logLProgressTable':
            assert(self.LsAndGermInfoSet)
            return _generation.get_logl_progress_table(
                self.Ls, self.gatesets, self.gateStringLists, self.dataset,
                self.constrainToTP, self.formatsToCompute, self.tableClass,
                self.longTables)        
        else:
            raise ValueError("Invalid table name: %s" % tableName)

    def get_figure(self, figureName, verbosity=0):
        """
        Get a report figure.  Figures are created on the first
        request then cached for later requests for the same figure.
        This method is typically used internally by other Results methods.

        Parameters
        ----------
        figureName : string
           The name of the figure.

        verbosity : int, optional
           How much detail to send to stdout.

        Returns
        -------
        GSTFigure
            The requested figure object.
        """
        assert(self.bEssentialResultsSet)
        if figureName not in self.figures:
            self.figures[figureName] = self._generateFigure(figureName, verbosity)
        return self.figures[figureName]

    def _generateFigure(self, figureName, verbosity):
        """ 
        Switchboard method for actually creating a figure (including computation
        of its values.
        """
        assert(self.bEssentialResultsSet)
        assert(self.LsAndGermInfoSet)

        if verbosity > 0:
            print "Generating %s figure..." % figureName; _sys.stdout.flush()

        if self.objective == "chi2":
            plotFn = _plotting.chi2_boxplot
            directPlotFn = _plotting.direct_chi2_boxplot
            whackAMolePlotFn = _plotting.whack_a_chi2_mole_boxplot
            #plotFnName,plotFnLatex = "Chi2", "$\chi^2$"
            mpc = self.additionalInfo['minProbClipForWeighting']
        elif self.objective == "logl":
            plotFn = _plotting.logl_boxplot
            directPlotFn = _plotting.direct_logl_boxplot
            whackAMolePlotFn = _plotting.whack_a_logl_mole_boxplot
            #plotFnName,plotFnLatex = "LogL", "$\\log(\\mathcal{L})$"
            mpc = self.additionalInfo['minProbClip']
        else: 
            raise ValueError("Invalid objective value: %s" % self.objective)


        m = 0
        M = 10
        baseStr_dict = self._getBaseStrDict()
        strs  = self.rhoStrs, self.EStrs
        st = 1 if self.Ls[0] == 0 else 0 #start index: skips LGST column in report color box plots        

        if figureName == "bestEstimateColorBoxPlot":
            fig = plotFn( self.Ls[st:], self.germs, baseStr_dict, self.dataset, self.gsBestEstimate, strs,
                          "L", "germ", M=M, m=m, scale=1.0, sumUp=False, histogram=True, title="", rhoEPairs=self.rhoEPairs,
                          minProbClipForWeighting=mpc, save_to="", ticSize=20)

        elif figureName == "invertedBestEstimateColorBoxPlot":
            fig = plotFn( self.Ls[st:], self.germs, baseStr_dict, self.dataset, self.gsBestEstimate, strs,
                          "L", "germ", M=M, m=m, scale=1.0, sumUp=False, histogram=True, title="", rhoEPairs=self.rhoEPairs,
                          save_to="", ticSize=20, minProbClipForWeighting=mpc, invert=True)

        elif figureName == "bestEstimateSummedColorBoxPlot":
            sumScale = len(self.rhoStrs)*len(self.EStrs) if self.rhoEPairs is None else len(self.rhoEPairs)
            fig = plotFn( self.Ls[st:], self.germs, baseStr_dict, self.dataset, self.gsBestEstimate, strs,
                          "L", "germ", M=M*sumScale, m=m*sumScale, scale=1.0, sumUp=True, histogram=False,
                          title="", rhoEPairs=self.rhoEPairs, minProbClipForWeighting=mpc, save_to="", ticSize=14)    
            
        elif figureName.startswith("estimateForLIndex") and figureName.endswith("ColorBoxPlot"):
            i = int(figureName[len("estimateForLIndex"):-len("ColorBoxPlot")])
            fig = plotFn( self.Ls[st:i+1], self.germs, baseStr_dict, self.dataset, self.gatesets[i],
                          strs, "L", "germ", M=M, m=m, scale=1.0, sumUp=False, histogram=False, title="",
                          rhoEPairs=self.rhoEPairs, save_to="", minProbClipForWeighting=mpc, ticSize=20 )

        elif figureName == "blankBoxPlot":
            fig = _plotting.blank_boxplot( 
                self.Ls[st:], self.germs, baseStr_dict, strs, "L", "germ",
                scale=1.0, title="", sumUp=False, save_to="", ticSize=20)

        elif figureName == "blankSummedBoxPlot":
            fig = _plotting.blank_boxplot( 
                self.Ls[st:], self.germs, baseStr_dict, strs, "L", "germ",
                scale=1.0, title="", sumUp=True, save_to="", ticSize=20)

        elif figureName == "directLGSTColorBoxPlot":
            directLGST  = self.get_special('direct_lgst_gatesets')
            fig = directPlotFn( self.Ls[st:], self.germs, baseStr_dict, self.dataset, directLGST, strs,
                                "L", "germ", M=M, m=m, scale=1.0, sumUp=False, title="", minProbClipForWeighting=mpc,
                                rhoEPairs=self.rhoEPairs, save_to="", ticSize=20)

        elif figureName == "directLongSeqGSTColorBoxPlot":
            directLongSeqGST = self.get_special('DirectLongSeqGatesets')
            fig = directPlotFn( self.Ls[st:], self.germs, baseStr_dict, self.dataset, directLongSeqGST, strs,
                          "L", "germ", M=M, m=m, scale=1.0, sumUp=False, title="",minProbClipForWeighting=mpc,
                          rhoEPairs=self.rhoEPairs, save_to="", ticSize=20)

        elif figureName == "directLGSTDeviationColorBoxPlot":
            directLGST  = self.get_special('direct_lgst_gatesets')
            fig = _plotting.direct_deviation_boxplot( 
                self.Ls[st:], self.germs, baseStr_dict, self.dataset,
                self.gsBestEstimate, directLGST, "L", "germ", m=0, scale=1.0,
                prec=-1, title="", save_to="", ticSize=20)

        elif figureName == "directLongSeqGSTDeviationColorBoxPlot":
            directLongSeqGST = self.get_special('DirectLongSeqGatesets')
            fig = _plotting.direct_deviation_boxplot(
                self.Ls[st:], self.germs, baseStr_dict, self.dataset,
                self.gsBestEstimate, directLongSeqGST, "L", "germ", m=0,
                scale=1.0, prec=-1, title="", save_to="", ticSize=20)

        elif figureName == "smallEigvalErrRateColorBoxPlot":
            directLongSeqGST = self.get_special('DirectLongSeqGatesets')
            fig = _plotting.small_eigval_err_rate_boxplot(
                self.Ls[st:], self.germs, baseStr_dict, self.dataset,
                directLongSeqGST, "L", "germ", m=0, scale=1.0, title="",
                save_to="", ticSize=20)

        elif figureName.startswith("whack") and figureName.endswith("MoleBoxes"):
            gateLabel = figureName[len("whack"):-len("MoleBoxes")]

            highestL = self.Ls[-1]; allGateStrings = self.gateStringLists[-1]; hammerWeight = 10.0
            len1GermFirstEls = [ g[0] for g in self.germs if len(g) == 1 ]
            assert(gateLabel in len1GermFirstEls) #only these whack-a-mole plots are available
            strToWhack = _gs.GateString( (gateLabel,)*highestL )

            fig = whackAMolePlotFn( strToWhack, allGateStrings, self.Ls[st:], self.germs, baseStr_dict, self.dataset,
                                    self.gsBestEstimate, strs, "L", "germ", scale=1.0, sumUp=False, title="", whackWith=hammerWeight,
                                    save_to="", minProbClipForWeighting=mpc, ticSize=20, rhoEPairs=self.rhoEPairs, m=0)

        elif figureName.startswith("whack") and figureName.endswith("MoleBoxesSummed"):
            gateLabel = figureName[len("whack"):-len("MoleBoxesSummed")]

            highestL = self.Ls[-1]; allGateStrings = self.gateStringLists[-1]; hammerWeight = 10.0
            len1GermFirstEls = [ g[0] for g in self.germs if len(g) == 1 ]
            assert(gateLabel in len1GermFirstEls) #only these whack-a-mole plots are available
            strToWhack = _gs.GateString( (gateLabel,)*highestL )

            fig = whackAMolePlotFn( strToWhack, allGateStrings, self.Ls[st:], self.germs, baseStr_dict, self.dataset,
                                    self.gsBestEstimate, strs, "L", "germ", scale=1.0, sumUp=True, title="", whackWith=hammerWeight,
                                    save_to="", minProbClipForWeighting=mpc, ticSize=20, rhoEPairs=self.rhoEPairs, m=0)

        else:
            raise ValueError("Invalid figure name: %s" % figureName)

        return fig


    def get_special(self, specialName, verbosity=0):
        """
        Get a "special item", which can be almost anything used in report
        or presentation construction.  This method is almost solely used 
        internally by other Results methods.

        Parameters
        ----------
        tableName : string
           The name of the special item.

        verbosity : int, optional
           How much detail to send to stdout.

        Returns
        -------
        special item (type varies)
        """
        if specialName not in self.specials:
            self.specials[specialName] = self._generateSpecial(specialName, verbosity)
        return self.specials[specialName]

    def _generateSpecial(self, specialName, verbosity):
        """ Switchboard function for creating "special" items """
        if specialName == 'gaugeOptAppendixGatesets':
            assert(self.bEssentialResultsSet)

            if verbosity > 0: 
                print "Performing gauge transforms for appendix..."; _sys.stdout.flush()
            best_gs_gauges = _collections.OrderedDict()

            best_gs_gauges['Target'] = _optimizeGauge(
                self.gsBestEstimate, "target", targetGateset=self.gsTarget,
                constrainToTP=self.constrainToTP, gateWeight=1.0,
                spamWeight=1.0, verbosity=2)

            best_gs_gauges['TargetSpam'] = _optimizeGauge(
                self.gsBestEstimate, "target", targetGateset=self.gsTarget,
                verbosity=2, gateWeight=0.01, spamWeight=0.99,
                constrainToTP=self.constrainToTP)

            best_gs_gauges['TargetGates'] = _optimizeGauge(
                self.gsBestEstimate, "target", targetGateset=self.gsTarget,
                verbosity=2, gateWeight=0.99, spamWeight=0.01,
                constrainToTP=self.constrainToTP)

            best_gs_gauges['CPTP'] = _optimizeGauge(
                self.gsBestEstimate, "CPTP and target", 
                targetGateset=self.gsTarget, verbosity=2,
                targetFactor=1.0e-7, constrainToTP=self.constrainToTP)

            if self.constrainToTP:
                best_gs_gauges['TP'] = best_gs_gauges['Target'].copy() #assume best_gs is already in TP, so just optimize to target (done above)
            else:
                best_gs_gauges['TP'] = _optimizeGauge(
                    self.gsBestEstimate, "TP and target",
                    targetGateset=self.gsTarget, targetFactor=1.0e-7)
            return best_gs_gauges

        elif specialName == 'gaugeOptAppendixTables':
            assert(self.bEssentialResultsSet)

            best_gs_gauges = self.get_special('gaugeOptAppendixGatesets')
            ret = {}

            for gaugeKey,gopt_gs in best_gs_gauges.iteritems():
                #FUTURE: add confidence region support to these appendices? -- would need to compute confidenceRegionInfo (cri)
                #  for each gauge-optimized gateset, gopt_gs and pass to appropriate functions below
                ret['best%sGatesetSpamTable' % gaugeKey] = \
                    _generation.get_gateset_spam_table(
                    gopt_gs, self.formatsToCompute, self.tableClass,
                    self.longTables)
                ret['best%sGatesetSpamParametersTable' % gaugeKey] = \
                    _generation.get_gateset_spam_parameters_table(
                    gopt_gs, self.formatsToCompute, self.tableClass,
                    self.longTables)
                ret['best%sGatesetGatesTable' % gaugeKey] = \
                    _generation.get_gateset_gates_table(
                    gopt_gs, self.formatsToCompute, self.tableClass,
                    self.longTables)
                ret['best%sGatesetChoiTable' % gaugeKey] = \
                    _generation.get_gateset_choi_table(
                    gopt_gs, self.formatsToCompute, self.tableClass,
                    self.longTables)
                ret['best%sGatesetDecompTable' % gaugeKey] = \
                    _generation.get_gateset_decomp_table(
                    gopt_gs, self.formatsToCompute, self.tableClass,
                    self.longTables)                
                ret['best%sGatesetRotnAxisTable' % gaugeKey] = \
                    _generation.get_gateset_rotn_axis_table(
                    gopt_gs, self.formatsToCompute, self.tableClass,
                    self.longTables)
                ret['best%sGatesetClosestUnitaryTable' % gaugeKey] = \
                    _generation.get_gateset_closest_unitary_table(
                    gopt_gs, self.formatsToCompute, self.tableClass,
                    self.longTables)
                ret['best%sGatesetVsTargetTable' % gaugeKey] = \
                    _generation.get_gateset_vs_target_table(
                    gopt_gs, self.gsTarget, self.formatsToCompute, 
                    self.tableClass, self.longTables)
                ret['best%sGatesetErrorGenTable' % gaugeKey] = \
                    _generation.get_gateset_vs_target_err_gen_table(
                    gopt_gs, self.gsTarget, self.formatsToCompute,
                    self.tableClass, self.longTables)

            return ret

        elif specialName == 'blankGaugeOptAppendixTables':
            assert(self.bEssentialResultsSet)

            ret = {}
            for gaugeKey in ('Target','TargetSpam', 'TargetGates', 'CPTP', 'TP'):
                ret['best%sGatesetSpamTable' % gaugeKey] = \
                    _generation.get_blank_table(self.formatsToCompute)
                ret['best%sGatesetSpamParametersTable' % gaugeKey] = \
                    _generation.get_blank_table(self.formatsToCompute)
                ret['best%sGatesetGatesTable' % gaugeKey] = \
                    _generation.get_blank_table(self.formatsToCompute)
                ret['best%sGatesetChoiTable' % gaugeKey] = \
                    _generation.get_blank_table(self.formatsToCompute)
                ret['best%sGatesetDecompTable' % gaugeKey] = \
                    _generation.get_blank_table(self.formatsToCompute)
                ret['best%sGatesetRotnAxisTable' % gaugeKey] = \
                    _generation.get_blank_table(self.formatsToCompute)
                ret['best%sGatesetClosestUnitaryTable' % gaugeKey] = \
                    _generation.get_blank_table(self.formatsToCompute)
                ret['best%sGatesetVsTargetTable' % gaugeKey] = \
                    _generation.get_blank_table(self.formatsToCompute)
                ret['best%sGatesetErrorGenTable' % gaugeKey] = \
                    _generation.get_blank_table(self.formatsToCompute)

            return ret

        elif specialName == "direct_lgst_gatesets":
            assert(self.bEssentialResultsSet)
            assert(self.LsAndGermInfoSet)

            direct_specs = _ssc.build_spam_specs(
                rhoStrs=self.rhoStrs, EStrs=self.EStrs,
                EVecInds=self.gsTarget.get_evec_indices() )
            baseStrs = [] # (L,germ) base strings without duplicates
            for L in self.Ls:
                for germ in self.germs:
                    if self.L_germ_tuple_to_baseStr_dict[(L,germ)] not in baseStrs:
                        baseStrs.append( self.L_germ_tuple_to_baseStr_dict[(L,germ)] )

            return _plotting.direct_lgst_gatesets( 
                baseStrs, self.dataset, direct_specs, self.gsTarget,
                svdTruncateTo=4, verbosity=0) #TODO: svdTruncateTo set elegantly?
        
        elif specialName == "DirectLongSeqGatesets":
            assert(self.bEssentialResultsSet)
            assert(self.LsAndGermInfoSet)

            direct_specs = _ssc.build_spam_specs(
                rhoStrs=self.rhoStrs, EStrs=self.EStrs,
                EVecInds=self.gsTarget.get_evec_indices() )
            baseStrs = [] # (L,germ) base strings without duplicates
            for L in self.Ls:
                for germ in self.germs:
                    if self.L_germ_tuple_to_baseStr_dict[(L,germ)] not in baseStrs:
                        baseStrs.append( self.L_germ_tuple_to_baseStr_dict[(L,germ)] )

            if self.objective == "chi2":
                return _plotting.direct_mc2gst_gatesets(
                    baseStrs, self.dataset, direct_specs, self.gsTarget,
                    svdTruncateTo=self.gsTarget.get_dimension(), 
                    minProbClipForWeighting=self.additionalInfo['minProbClipForWeighting'],
                    probClipInterval=self.additionalInfo['probClipInterval'],
                    verbosity=0)

            elif self.objective == "logl":
                return _plotting.direct_mlgst_gatesets(
                    baseStrs, self.dataset, direct_specs, self.gsTarget,
                    svdTruncateTo=self.gsTarget.get_dimension(),
                    minProbClip=self.additionalInfo['minProbClip'],
                    probClipInterval=self.additionalInfo['probClipInterval'],
                    verbosity=0)
            else:
                raise ValueError("Invalid Objective: %s" % self.objective)

        else:
            raise ValueError("Invalid special name: %s" % specialName)

    def _merge_template(self, qtys, templateFilename, outputFilename):
        if self.templatePath is None:
            templateFilename = _os.path.join( _os.path.dirname(_os.path.abspath(__file__)), 
                                              "templates", templateFilename )
        else:
            templateFilename = _os.path.join( self.templatePath, templateFilename )
            
        template = open(templateFilename,"r").read()
        template = template.replace("{", "{{").replace("}", "}}") #double curly braces (for format processing)

        # Replace template field markers with `str.format` fields.
        template = _re.sub( r"\\putfield\{\{([^}]+)\}\}\{\{[^}]*\}\}", "{\\1}", template)

        # Replace str.format fields with values and write to output file
        template = template.format(**qtys)
        open(outputFilename,'w').write(template)
    

    def _getBaseStrDict(self, remove_dups = True):
        #if remove_dups == True, remove duplicates in 
        #  L_germ_tuple_to_baseStr_dict by replacing with None

        assert(self.bEssentialResultsSet)
        assert(self.LsAndGermInfoSet)

        baseStr_dict = {}
        st = 1 if self.Ls[0] == 0 else 0 #start index: skips LGST column in report color box plots        

        tmpRunningList = []
        for L in self.Ls[st:]:
            for germ in self.germs:
                if remove_dups and self.L_germ_tuple_to_baseStr_dict[(L,germ)] in tmpRunningList:
                    baseStr_dict[(L,germ)] = None
                else: 
                    tmpRunningList.append( self.L_germ_tuple_to_baseStr_dict[(L,germ)] )
                    baseStr_dict[(L,germ)] = self.L_germ_tuple_to_baseStr_dict[(L,germ)]
        return baseStr_dict
    

    def create_full_report_pdf(self, confidenceLevel=None, filename="auto", 
                            title="auto", datasetLabel="auto", suffix="",
                            debugAidsAppendix=False, gaugeOptAppendix=False,
                            pixelPlotAppendix=False, whackamoleAppendix=False,
                            m=0, M=10, verbosity=0):
        """
        Create a "full" GST report.  This report is the most detailed of any of
        the GST reports, and includes background and explanation text to help
        the user interpret the contained results.

        Parameters
        ----------
        confidenceLevel : float, optional
           If not None, then the confidence level (between 0 and 100) used in
           the computation of confidence regions/intervals. If None, no 
           confidence regions or intervals are computed.

        filename : string, optional
           The output filename where the report file(s) will be saved.  Specifying
           "auto" will use the default directory and base name (specified in 
           set_additional_info) if given, otherwise the file "GSTReport.pdf" will
           be output to the current directoy.

        title : string, optional
           The title of the report.  "auto" uses a default title which
           specifyies the label of the dataset as well.

        datasetLabel : string, optional
           A label given to the dataset.  If set to "auto", then the label
           will be the base name of the dataset filename without extension
           (if given) or "$\\mathcal{D}$" (if not).

        suffix : string, optional
           A suffix to add to the end of the report filename.  Useful when
           filename is "auto" and you generate different reports using
           the same dataset.

        debugAidsAppendix : bool, optional
           Whether to include the "debugging aids" appendix.  This 
           appendix contains comparisons of GST and Direct-GST and small-
           eigenvalue error rates among other quantities potentially
           useful for figuring out why the GST estimate did not fit
           the data as well as expected.

        gaugeOptAppendix : bool, optional
           Whether to include the "gauge optimization" appendix.  This
           appendix shows the results of gauge optimizing GST's best
           estimate gate set in several different ways, and thus shows
           how various report quantities can vary by a different gauge
           choice.

        pixelPlotAppendix : bool, optional
           Whether to include the "pixel plots" appendix, which shows
           the goodness of fit, in the form of color box plots, for the
           intermediate iterations of the GST algortihm.

        whackamoleAppendix : bool, optional
           Whether to include the "whack-a-mole" appendix, which contains 
           colr box plots showing the effect of reducing ("whacking") one 
           particular part of the overall goodness of fit box plot.

        m, M : float, optional
           Minimum and Maximum values of the color scale used in the report's
           color box plots.

        verbosity : int, optional
           How much detail to send to stdout.

        Returns
        -------
        None
        """
        assert(self.bEssentialResultsSet)

        #Get report output filename
        default_dir = self.additionalInfo['defaultDirectory']
        default_base = self.additionalInfo['defaultBasename']

        if filename != "auto":
            report_dir = _os.path.dirname(filename)
            report_base = _os.path.splitext( _os.path.basename(filename) )[0] + suffix
        else:
            cwd = _os.getcwd()
            report_dir  = default_dir  if (default_dir  is not None) else cwd
            report_base = default_base if (default_base is not None) else "GSTReport"
            report_base += suffix

        if datasetLabel == "auto":
            if default_base is not None:
                datasetLabel = _latex.latex_escaped( default_base )
            else:
                datasetLabel = "$\\mathcal{D}$"

        if title == "auto": title = "GST report for %s" % datasetLabel

        ######  Generate Report ######
        #Steps:
        # 1) generate latex tables
        # 2) generate plots
        # 3) populate template latex file => report latex file
        # 4) compile report latex file into PDF
        # 5) remove auxiliary files generated during compilation
        #  FUTURE?? determine what we need to compute & plot by reading through the template file?
        
        #Note: for now, we assume the best gateset corresponds to the last L-value
        best_gs = self.gsBestEstimate

        if not self.LsAndGermInfoSet: #cannot create appendices which depend on this structure
            debugAidsAppendix = False
            pixelPlotAppendix = False
            whackamoleAppendix = False
        
        qtys = {} # dictionary to store all latex strings to be inserted into report template
        qtys['title'] = title
        qtys['datasetLabel'] = datasetLabel
        qtys['settoggles'] =  "\\togglefalse{confidences}\n" if confidenceLevel is None else "\\toggletrue{confidences}\n"
        qtys['settoggles'] += "\\toggletrue{LsAndGermsSet}\n" if self.LsAndGermInfoSet else "\\togglefalse{LsAndGermsSet}\n"
        qtys['settoggles'] += "\\toggletrue{debuggingaidsappendix}\n" if debugAidsAppendix else "\\togglefalse{debuggingaidsappendix}\n"
        qtys['settoggles'] += "\\toggletrue{gaugeoptappendix}\n" if gaugeOptAppendix else "\\togglefalse{gaugeoptappendix}\n"
        qtys['settoggles'] += "\\toggletrue{pixelplotsappendix}\n" if pixelPlotAppendix else "\\togglefalse{pixelplotsappendix}\n"
        qtys['settoggles'] += "\\toggletrue{whackamoleappendix}\n" if whackamoleAppendix else "\\togglefalse{whackamoleappendix}\n"
        qtys['confidenceLevel'] = "%g" % confidenceLevel if confidenceLevel is not None else "NOT-SET"
    
        if confidenceLevel is not None:
            cri = self.get_confidence_region(confidenceLevel)
            qtys['confidenceIntervalScaleFctr'] = "%.3g" % cri.intervalScaling
            qtys['confidenceIntervalNumNonGaugeParams'] = "%d" % cri.nNonGaugeParams
        else:
            cri = None
            qtys['confidenceIntervalScaleFctr'] = "NOT-SET"
            qtys['confidenceIntervalNumNonGaugeParams'] = "NOT-SET"

            
        # 1) get latex tables
        if verbosity > 0: 
            print "*** Generating tables ***"; _sys.stdout.flush()

        required_tables = ('targetSpamTable','targetGatesTable','datasetOverviewTable',
                           'bestGatesetSpamTable','bestGatesetSpamParametersTable','bestGatesetGatesTable','bestGatesetChoiTable',
                           'bestGatesetDecompTable','bestGatesetRotnAxisTable','bestGatesetClosestUnitaryTable',
                           'bestGatesetVsTargetTable','bestGatesetErrorGenTable')

        progress_tbl = 'logLProgressTable' if self.objective == "logl" else 'chi2ProgressTable'
        ls_and_germs_tables = ('fiducialListTable','rhoStrListTable','EStrListTable','germListTable', progress_tbl)            
            
        if self.LsAndGermInfoSet:
            required_tables += ls_and_germs_tables
        else:
            #Fill required keys with blank tables so merge still works below
            for key in ls_and_germs_tables:
                qtys[key] = _generation.get_blank_table(self.formatsToCompute)['latex']
            
        for key in required_tables:
            qtys[key] = self.get_table(key, confidenceLevel, 'latex', verbosity)

        if gaugeOptAppendix: #get appendix tables if needed
            goaTables = self.get_special('gaugeOptAppendixTables', verbosity)
            qtys.update( { key : goaTables[key]['latex'] for key in goaTables }  )
        elif any((debugAidsAppendix, pixelPlotAppendix, whackamoleAppendix)):       # if other appendices used, 
            goaTables = self.get_special('blankGaugeOptAppendixTables', verbosity)   # fill keys with blank tables
            qtys.update( { key : goaTables[key]['latex'] for key in goaTables }  )  # for format substitution

    
        # 2) generate plots
        if verbosity > 0: 
            print "*** Generating plots ***"; _sys.stdout.flush()

        if _matplotlib.is_interactive():
            _matplotlib.pyplot.ioff()
            bWasInteractive = True
        else: bWasInteractive = False
    
        D = report_base + "_files" #figure directory relative to reportDir
        if not _os.path.isdir( _os.path.join(report_dir,D)):
            _os.mkdir( _os.path.join(report_dir,D))

        maxW,maxH = 6.5,9.0 #max width and height of graphic in latex document (in inches)

        #Chi2 or logl plots
        if self.LsAndGermInfoSet:
            st = 1 if self.Ls[0] == 0 else 0 #start index: skips LGST column in report color box plots        
            nPlots = (len(self.Ls[st:])-1)+2 if pixelPlotAppendix else 2

            if self.objective == "chi2":
                plotFnName,plotFnLatex = "Chi2", "$\chi^2$"
            elif self.objective == "logl":
                plotFnName,plotFnLatex = "LogL", "$\\log(\\mathcal{L})$"
            else: 
                raise ValueError("Invalid objective value: %s" % self.objective)
            
            if verbosity > 0: 
                print " -- %s plots (%d): " % (plotFnName, nPlots),; _sys.stdout.flush()

            if verbosity > 0: 
                print "1 ",; _sys.stdout.flush()
            fig = self.get_figure("bestEstimateColorBoxPlot",verbosity)
            fig.save_to(_os.path.join(report_dir, D,"best%sBoxes.pdf" % plotFnName))
            maxX = fig.get_extra_info()['nUsedXs']; maxY = fig.get_extra_info()['nUsedYs']

            if verbosity > 0: 
                print "2 ",; _sys.stdout.flush()
            fig = self.get_figure("invertedBestEstimateColorBoxPlot",verbosity)
            fig.save_to(_os.path.join(report_dir, D,"best%sBoxes_inverted.pdf" % plotFnName))
    
        pixplots = ""
        if pixelPlotAppendix:
            for i in range(st,len(self.Ls)-1):

                if verbosity > 0: 
                    print "%d " % (i-st+3),; _sys.stdout.flush()

                fig = self.get_figure("estimateForLIndex%dColorBoxPlot" % i, verbosity)
                fig.save_to( _os.path.join(report_dir, D,"L%d_%sBoxes.pdf" % (i,plotFnName)) )
                lx = fig.get_extra_info()['nUsedXs']; ly = fig.get_extra_info()['nUsedYs']

                W = float(lx+1)/float(maxX+1) * maxW #scale figure size according to number of rows 
                H = float(ly)  /float(maxY)   * maxH # and columns+1 (+1 for labels ~ another col) relative to initial plot
            
                pixplots += "\n"
                pixplots += "\\begin{figure}\n"
                pixplots += "\\begin{center}\n"
                pixplots += "\\includegraphics[width=%fin,height=%fin,keepaspectratio]{%s/L%d_%sBoxes.pdf}\n" % (W,H,D,i,plotFnName)
                pixplots += "\\caption{Box plot of iteration %d (L=%d) gateset %s values.\label{L%dGateset%sBoxPlot}}\n" % (i,self.Ls[i],plotFnLatex,i,plotFnName)
                pixplots += "\\end{center}\n"
                pixplots += "\\end{figure}\n"
    
        if verbosity > 0: 
            print ""; _sys.stdout.flush()
        
        if debugAidsAppendix:
            #DirectLGST and deviation
            if verbosity > 0: 
                print " -- Direct-X plots ",; _sys.stdout.flush()
                print "(2):"; _sys.stdout.flush()    

            #if verbosity > 0: 
            #    print " ?",; _sys.stdout.flush()        
            #fig = self.get_figure("directLGSTColorBoxPlot",verbosity)
            #fig.save_to(_os.path.join(report_dir, D,"directLGST%sBoxes.pdf" % plotFnName))

            if verbosity > 0: 
                print " 1",; _sys.stdout.flush()        
            fig = self.get_figure("directLongSeqGSTColorBoxPlot",verbosity)
            fig.save_to(_os.path.join(report_dir, D,"directLongSeqGST%sBoxes.pdf" % plotFnName))

            #if verbosity > 0: 
            #    print " ?",; _sys.stdout.flush()        
            #fig = self.get_figure("directLGSTDeviationColorBoxPlot",verbosity)
            #fig.save_to(_os.path.join(report_dir, D,"directLGSTDeviationBoxes.pdf"))

            if verbosity > 0: 
                print " 2",; _sys.stdout.flush()
            fig = self.get_figure("directLongSeqGSTDeviationColorBoxPlot",verbosity)
            fig.save_to(_os.path.join(report_dir, D,"directLongSeqGSTDeviationBoxes.pdf"))

            if verbosity > 0: 
                print ""; _sys.stdout.flush()


            #Small eigenvalue error rate
            if verbosity > 0: 
                print " -- Error rate plots..."; _sys.stdout.flush()
            fig = self.get_figure("smallEigvalErrRateColorBoxPlot",verbosity)
            fig.save_to(_os.path.join(report_dir, D,"smallEigvalErrRateBoxes.pdf"))
    

        whackamoleplots = ""
        if whackamoleAppendix:    
            #Whack-a-mole plots for highest L of each length-1 germ
            highestL = self.Ls[-1]; allGateStrings = self.gateStringLists[-1]; hammerWeight = 10.0
            len1Germs = [ g for g in self.germs if len(g) == 1 ]

            if verbosity > 0: 
                print " -- Whack-a-mole plots (%d): " % (2*len(len1Germs)),; _sys.stdout.flush()

            for i,germ in enumerate(len1Germs):
                if verbosity > 0: 
                    print "%d " % (i+1),; _sys.stdout.flush()

                fig = self.get_figure("whack%sMoleBoxes" % germ[0],verbosity)
                fig.save_to(_os.path.join(report_dir, D,"whack%sMoleBoxes.pdf" % germ[0]))
        
                whackamoleplots += "\n"
                whackamoleplots += "\\begin{figure}\n"
                whackamoleplots += "\\begin{center}\n"
                whackamoleplots += "\\includegraphics[width=%fin,height=%fin,keepaspectratio]{%s/whack%sMoleBoxes.pdf}\n" % (maxW,maxH,D,germ[0])
                whackamoleplots += "\\caption{Whack-a-%s-mole box plot for $\mathrm{%s}^{%d}$." % (plotFnLatex,germ[0],highestL)
                whackamoleplots += "  Hitting with hammer of weight %.1f.\label{Whack%sMoleBoxPlot}}\n" % (hammerWeight,germ[0])
                whackamoleplots += "\\end{center}\n"
                whackamoleplots += "\\end{figure}\n"
        
            for i,germ in enumerate(len1Germs):
                if verbosity > 0: 
                    print "%d " % (len(len1Germs)+i+1),; _sys.stdout.flush()

                fig = self.get_figure("whack%sMoleBoxesSummed" % germ[0], verbosity)
                fig.save_to(_os.path.join(report_dir, D,"whack%sMoleBoxesSummed.pdf" % germ[0]))
    
                whackamoleplots += "\n"
                whackamoleplots += "\\begin{figure}\n"
                whackamoleplots += "\\begin{center}\n"
                whackamoleplots += "\\includegraphics[width=4in,height=5in,keepaspectratio]{%s/whack%sMoleBoxesSummed.pdf}\n" % (D,germ[0])
                whackamoleplots += "\\caption{Whack-a-%s-mole box plot for $\mathrm{%s}^{%d}$, summed over fiducial matrix." % (plotFnLatex,germ[0],highestL)
                whackamoleplots += "  Hitting with hammer of weight %.1f.\label{Whack%sMoleBoxPlotSummed}}\n" % (hammerWeight,germ[0])
                whackamoleplots += "\\end{center}\n"
                whackamoleplots += "\\end{figure}\n"
    
            if verbosity > 0: 
                print ""; _sys.stdout.flush()
        
    
        #Note: set qtys keys even if the plots were not created, since the template subsitution occurs before conditional latex inclusion.
        #   Thus, these keys *must* exist otherwise substitution will process will generate an error.
        if self.objective == "chi2":
            qtys['bestGatesetChi2BoxPlot']     = "\\includegraphics[width=%fin,height=%fin,keepaspectratio]{%s/bestChi2Boxes.pdf}" % (maxW,maxH,D)
            qtys['bestGatesetChi2InvBoxPlot']  = "\\includegraphics[width=%fin,height=%fin,keepaspectratio]{%s/bestChi2Boxes_inverted.pdf}" % (maxW,maxH,D)
            qtys['bestGatesetChi2HistPlot']    = "\\includegraphics[width=4in,angle=0]{%s/bestChi2Boxes_hist.pdf}" % D
            qtys['directLGSTChi2BoxPlot']      = "\\includegraphics[width=%fin,height=%fin,keepaspectratio]{%s/directLGSTChi2Boxes.pdf}" % (maxW,maxH,D)
            qtys['directLongSeqGSTChi2BoxPlot']      = "\\includegraphics[width=%fin,height=%fin,keepaspectratio]{%s/directLongSeqGSTChi2Boxes.pdf}" % (maxW,maxH,D)
    
        elif self.objective == "logl":
            qtys['bestGatesetLogLBoxPlot']     = "\\includegraphics[width=%fin,height=%fin,keepaspectratio]{%s/bestLogLBoxes.pdf}" % (maxW,maxH,D)
            qtys['bestGatesetLogLInvBoxPlot']  = "\\includegraphics[width=%fin,height=%fin,keepaspectratio]{%s/bestLogLBoxes_inverted.pdf}" % (maxW,maxH,D)
            qtys['bestGatesetLogLHistPlot']    = "\\includegraphics[width=4in,angle=0]{%s/bestLogLBoxes_hist.pdf}" % D
            qtys['directLGSTLogLBoxPlot']      = "\\includegraphics[width=%fin,height=%fin,keepaspectratio]{%s/directLGSTLogLBoxes.pdf}" % (maxW,maxH,D)
            qtys['directLongSeqGSTLogLBoxPlot']      = "\\includegraphics[width=%fin,height=%fin,keepaspectratio]{%s/directLongSeqGSTLogLBoxes.pdf}" % (maxW,maxH,D)
        else: 
            raise ValueError("Invalid objective value: %s" % self.objective)
    
        qtys['directLGSTDeviationBoxPlot'] = "\\includegraphics[width=4in,height=5in,keepaspectratio]{%s/directLGSTDeviationBoxes.pdf}" % D
        qtys['directLongSeqGSTDeviationBoxPlot'] = "\\includegraphics[width=4in,height=5in,keepaspectratio]{%s/directLongSeqGSTDeviationBoxes.pdf}" % D
        qtys['smallEvalErrRateBoxPlot'] = "\\includegraphics[width=4in,height=5in,keepaspectratio]{%s/smallEigvalErrRateBoxes.pdf}" % D
        qtys['intermediate_pixel_plot_figures'] = pixplots
        qtys['whackamole_plot_figures'] = whackamoleplots
    
        if bWasInteractive:
            _matplotlib.pyplot.ion()
    
        # 3) populate template latex file => report latex file
        if verbosity > 0: 
            print "*** Merging into template file ***"; _sys.stdout.flush()
        
        mainTexFilename = _os.path.join(report_dir, report_base + ".tex")
        appendicesTexFilename = _os.path.join(report_dir, report_base + "_appendices.tex")
        pdfFilename = _os.path.join(report_dir, report_base + ".pdf")
    
        if self.objective == "chi2":    
            mainTemplate = "report_chi2_main.tex"
            appendicesTemplate = "report_chi2_appendices.tex"
        elif self.objective == "logl":
            mainTemplate = "report_logL_main.tex"
            appendicesTemplate = "report_logL_appendices.tex"
        else: 
            raise ValueError("Invalid objective value: %s" % self.objective)
    
        if any( (debugAidsAppendix, gaugeOptAppendix, pixelPlotAppendix, whackamoleAppendix) ):
            qtys['appendices'] = "\\input{%s}" % _os.path.basename(appendicesTexFilename)
            self._merge_template(qtys, appendicesTemplate, appendicesTexFilename)
        else: qtys['appendices'] = ""
        self._merge_template(qtys, mainTemplate, mainTexFilename)
    
    
        # 4) compile report latex file into PDF
        if verbosity > 0: 
            print "Latex file(s) successfully generated.  Attempting to compile with pdflatex..."; _sys.stdout.flush()
        cwd = _os.getcwd()
        if len(report_dir) > 0:  
            _os.chdir(report_dir)
    
        try:
            ret = _os.system( "%s %s > /dev/null" % (self.latexCmd, _os.path.basename(mainTexFilename)) )
            if ret == 0:
                #We could check if the log file contains "Rerun" in it, but we'll just re-run all the time now
                if verbosity > 0: 
                    print "Initial output PDF %s successfully generated." % pdfFilename #mainTexFilename
                ret = _os.system( "%s %s > /dev/null" % (self.latexCmd,_os.path.basename(mainTexFilename)) )
                if ret == 0:
                    if verbosity > 0: 
                        print "Final output PDF %s successfully generated. Cleaning up .aux and .log files." % pdfFilename #mainTexFilename
                    _os.remove( report_base + ".log" )
                    _os.remove( report_base + ".aux" )
                else:
                    print "Error: pdflatex returned code %d. Check %s.log to see details" % (ret, report_base)
            else:
                print "Error: pdflatex returned code %d. Check %s.log to see details" % (ret, report_base)
        except:
            print "Error trying to run pdflatex to generate output PDF %s. Is '%s' on your path?" % (pdfFilename,self.latexCmd)
        finally: 
            _os.chdir(cwd)
    
        return


    def create_brief_report_pdf(self, confidenceLevel=None, 
                             filename="auto", title="auto", datasetLabel="auto",
                             suffix="", m=0, M=10, verbosity=0):
        """
        Create a "brief" GST report.  This report is collects what are typically
        the most relevant tables and plots from the "full" report and presents
        them in an order or efficient analysis by a user familiar with the GST
        full report.  Descriptive text has been all but removed.

        Parameters
        ----------
        confidenceLevel : float, optional
           If not None, then the confidence level (between 0 and 100) used in
           the computation of confidence regions/intervals. If None, no 
           confidence regions or intervals are computed.

        filename : string, optional
           The output filename where the report file(s) will be saved.  Specifying
           "auto" will use the default directory and base name (specified in 
           set_additional_info) if given, otherwise the file "GSTBrief.pdf" will
           be output to the current directoy.

        title : string, optional
           The title of the report.  "auto" uses a default title which
           specifyies the label of the dataset as well.

        datasetLabel : string, optional
           A label given to the dataset.  If set to "auto", then the label
           will be the base name of the dataset filename without extension
           (if given) or "$\\mathcal{D}$" (if not).

        suffix : string, optional
           A suffix to add to the end of the report filename.  Useful when
           filename is "auto" and you generate different reports using
           the same dataset.

        m, M : float, optional
           Minimum and Maximum values of the color scale used in the report's
           color box plots.

        verbosity : int, optional
           How much detail to send to stdout.

        Returns
        -------
        None
        """
        assert(self.bEssentialResultsSet)

        #Get report output filename
        default_dir = self.additionalInfo['defaultDirectory']
        default_base = self.additionalInfo['defaultBasename']

        if filename != "auto":
            report_dir = _os.path.dirname(filename)
            report_base = _os.path.splitext( _os.path.basename(filename) )[0] + suffix
        else:
            cwd = _os.getcwd()
            report_dir  = default_dir  if (default_dir  is not None) else cwd
            report_base = default_base if (default_base is not None) else "GSTBrief"
            report_base += suffix

        if datasetLabel == "auto":
            if default_base is not None:
                datasetLabel = _latex.latex_escaped( default_base )
            else:
                datasetLabel = "$\\mathcal{D}$"

        if title == "auto": title = "Brief GST report for %s" % datasetLabel

        ######  Generate Report ######
        #Steps:
        # 1) generate latex tables
        # 2) generate plots
        # 3) populate template latex file => report latex file
        # 4) compile report latex file into PDF
        # 5) remove auxiliary files generated during compilation
        
                    
        if self.LsAndGermInfoSet:
            baseStr_dict = self._getBaseStrDict()
            st = 1 if self.Ls[0] == 0 else 0 #start index: skips LGST column in report color box plots    
            goodnessOfFitSection = True
        else:
            goodnessOfFitSection = False
    
        #Note: for now, we assume the best gateset corresponds to the last L-value
        best_gs = self.gsBestEstimate
        
        qtys = {} # dictionary to store all latex strings to be inserted into report template
        qtys['title'] = title
        qtys['datasetLabel'] = datasetLabel
        qtys['settoggles'] =  "\\togglefalse{confidences}\n" if confidenceLevel is None else "\\toggletrue{confidences}\n"
        qtys['settoggles'] += "\\toggletrue{goodnessSection}\n" if goodnessOfFitSection else "\\togglefalse{goodnessSection}\n"
        qtys['confidenceLevel'] = "%g" % confidenceLevel if confidenceLevel is not None else "NOT-SET"
        qtys['objective'] = "$\\log{\\mathcal{L}}$" if self.objective == "logl" else "$\\chi^2$"
        qtys['gofObjective'] = "$2\\Delta\\log{\\mathcal{L}}$" if self.objective == "logl" else "$\\chi^2$"

        if confidenceLevel is not None:
            cri = self.get_confidence_region(confidenceLevel)
            qtys['confidenceIntervalScaleFctr'] = "%.3g" % cri.intervalScaling
            qtys['confidenceIntervalNumNonGaugeParams'] = "%d" % cri.nNonGaugeParams
        else:
            cri = None
            qtys['confidenceIntervalScaleFctr'] = "NOT-SET"
            qtys['confidenceIntervalNumNonGaugeParams'] = "NOT-SET"

            
        # 1) get latex tables
        if verbosity > 0: 
            print "*** Generating tables ***"; _sys.stdout.flush()

        required_tables = ('bestGatesetSpamTable','bestGatesetSpamParametersTable','bestGatesetGatesTable',
                           'bestGatesetDecompTable','bestGatesetRotnAxisTable','bestGatesetVsTargetTable',
                           'bestGatesetErrorGenTable')
        for key in required_tables:
            qtys[key] = self.get_table(key, confidenceLevel, 'latex', verbosity)

        if goodnessOfFitSection:
            progress_tbl = 'logLProgressTable' if self.objective == "logl" else 'chi2ProgressTable'
            qtys['progressTable'] = self.get_table(progress_tbl, confidenceLevel, 'latex', verbosity)

    
        # 2) generate plots
        if verbosity > 0: 
            print "*** Generating plots ***"; _sys.stdout.flush()

        if _matplotlib.is_interactive():
            _matplotlib.pyplot.ioff()
            bWasInteractive = True
        else: bWasInteractive = False

        #if goodnessOfFitSection:    
        #    strs  = self.rhoStrs, self.EStrs
        #    D = report_base + "_files" #figure directory relative to reportDir
        #    if not _os.path.isdir( _os.path.join(report_dir,D)):
        #        _os.mkdir( _os.path.join(report_dir,D))
        #
        #    #Chi2 or logl plot
        #    nPlots = 1
        #    if self.objective == "chi2":
        #        plotFnName,plotFnLatex = "Chi2", "$\chi^2$"
        #    elif self.objective == "logl":
        #        plotFnName,plotFnLatex = "LogL", "$\\log(\\mathcal{L})$"
        #    else: 
        #        raise ValueError("Invalid objective value: %s" % self.objective)
        #
        #    if verbosity > 0: 
        #        print " -- %s plots (%d): " % (plotFnName, nPlots),; _sys.stdout.flush()
        #
        #    if verbosity > 0: 
        #        print "1 ",; _sys.stdout.flush()
        #    fig = self.get_figure("bestEstimateColorBoxPlot",verbosity)
        #    fig.save_to(_os.path.join(report_dir, D,"best%sBoxes.pdf" % plotFnName))
        #    maxX = fig.get_extra_info()['nUsedXs']; maxY = fig.get_extra_info()['nUsedYs']
        #    maxW,maxH = 6.5,9.0 #max width and height of graphic in latex document (in inches)
        #
        #    if verbosity > 0: 
        #        print ""; _sys.stdout.flush()    
        #
        #    qtys['bestGatesetBoxPlot']  = "\\includegraphics[width=%fin,height=%fin,keepaspectratio]{%s/best%sBoxes.pdf}" % (maxW,maxH,D,plotFnName)
        
        if bWasInteractive:
            _matplotlib.pyplot.ion()
    
        # 3) populate template latex file => report latex file
        if verbosity > 0: 
            print "*** Merging into template file ***"; _sys.stdout.flush()
        
        mainTexFilename = _os.path.join(report_dir, report_base + ".tex")
        appendicesTexFilename = _os.path.join(report_dir, report_base + "_appendices.tex")
        pdfFilename = _os.path.join(report_dir, report_base + ".pdf")
    
        mainTemplate = "brief_report_main.tex"
        self._merge_template(qtys, mainTemplate, mainTexFilename)
    
        # 4) compile report latex file into PDF
        if verbosity > 0: 
            print "Latex file(s) successfully generated.  Attempting to compile with pdflatex..."; _sys.stdout.flush()
        cwd = _os.getcwd()
        if len(report_dir) > 0:  
            _os.chdir(report_dir)
    
        try:
            ret = _os.system( "%s %s > /dev/null" % (self.latexCmd, _os.path.basename(mainTexFilename)) )
            if ret == 0:
                #We could check if the log file contains "Rerun" in it, but we'll just re-run all the time now
                if verbosity > 0: 
                    print "Initial output PDF %s successfully generated." % pdfFilename #mainTexFilename
                ret = _os.system( "%s %s > /dev/null" % (self.latexCmd, _os.path.basename(mainTexFilename)) )
                if ret == 0:
                    if verbosity > 0: 
                        print "Final output PDF %s successfully generated. Cleaning up .aux and .log files." % pdfFilename #mainTexFilename
                    _os.remove( report_base + ".log" )
                    _os.remove( report_base + ".aux" )
                else:
                    print "Error: pdflatex returned code %d. Check %s.log to see details" % (ret, report_base)
            else:
                print "Error: pdflatex returned code %d. Check %s.log to see details" % (ret, report_base)
        except:
            print "Error trying to run pdflatex to generate output PDF %s. Is '%s' on your path?" % (pdfFilename,self.latexCmd)
        finally: 
            _os.chdir(cwd)

        return


    def create_presentation_pdf(self, confidenceLevel=None, filename="auto", 
                              title="auto", datasetLabel="auto", suffix="",
                              debugAidsAppendix=False, 
                              pixelPlotAppendix=False, whackamoleAppendix=False,
                              m=0, M=10, verbosity=0):
        """
        Create a GST presentation (i.e. slides) using the beamer latex package.

        The slides can contain most (but not all) of the tables and figures from
        the "full" report but contain only minimal descriptive text.  This output
        if useful for those familiar with the GST full report who need to present
        the results in a projector-friendly format.

        Parameters
        ----------
        confidenceLevel : float, optional
           If not None, then the confidence level (between 0 and 100) used in
           the computation of confidence regions/intervals. If None, no 
           confidence regions or intervals are computed.

        filename : string, optional
           The output filename where the presentation file(s) will be saved.  
           Specifying "auto" will use the default directory and base name 
           (specified in set_additional_info) if given, otherwise the file
           "GSTSlides.pdf" will be output to the current directoy.

        title : string, optional
           The title of the presentation.  "auto" uses a default title which
           specifyies the label of the dataset as well.

        datasetLabel : string, optional
           A label given to the dataset.  If set to "auto", then the label
           will be the base name of the dataset filename without extension
           (if given) or "$\\mathcal{D}$" (if not).

        suffix : string, optional
           A suffix to add to the end of the presentation filename.  Useful when
           filename is "auto" and you generate different presentations using
           the same dataset.

        debugAidsAppendix : bool, optional
           Whether to include the "debugging aids" appendix.  This 
           appendix contains comparisons of GST and Direct-GST and small-
           eigenvalue error rates among other quantities potentially
           useful for figuring out why the GST estimate did not fit
           the data as well as expected.

        pixelPlotAppendix : bool, optional
           Whether to include the "pixel plots" appendix, which shows
           the goodness of fit, in the form of color box plots, for the
           intermediate iterations of the GST algortihm.

        whackamoleAppendix : bool, optional
           Whether to include the "whack-a-mole" appendix, which contains 
           colr box plots showing the effect of reducing ("whacking") one 
           particular part of the overall goodness of fit box plot.

        m, M : float, optional
           Minimum and Maximum values of the color scale used in the presentation's
           color box plots.

        verbosity : int, optional
           How much detail to send to stdout.

        Returns
        -------
        None
        """
        assert(self.bEssentialResultsSet)

        #Get report output filename
        default_dir = self.additionalInfo['defaultDirectory']
        default_base = self.additionalInfo['defaultBasename']

        if filename != "auto":
            report_dir = _os.path.dirname(filename)
            report_base = _os.path.splitext( _os.path.basename(filename) )[0] + suffix
        else:
            cwd = _os.getcwd()
            report_dir  = default_dir  if (default_dir  is not None) else cwd
            report_base = default_base if (default_base is not None) else "GSTSlides"
            report_base += suffix

        if datasetLabel == "auto":
            if default_base is not None:
                datasetLabel = _latex.latex_escaped( default_base )
            else:
                datasetLabel = "$\\mathcal{D}$"

        if title == "auto": title = "GST on %s" % datasetLabel

        ######  Generate Presentation ######
        #Steps:
        # 1) generate latex tables
        # 2) generate plots
        # 3) populate template latex file => report latex file
        # 4) compile report latex file into PDF
        # 5) remove auxiliary files generated during compilation
                            
        #Note: for now, we assume the best gateset corresponds to the last L-value
        best_gs = self.gsBestEstimate

        if not self.LsAndGermInfoSet: #cannot create appendices which depend on this structure
            debugAidsAppendix = False
            pixelPlotAppendix = False
            whackamoleAppendix = False
        
        qtys = {} # dictionary to store all latex strings to be inserted into report template
        qtys['title'] = title
        qtys['datasetLabel'] = datasetLabel
        qtys['settoggles'] =  "\\togglefalse{confidences}\n" if confidenceLevel is None else "\\toggletrue{confidences}\n"
        qtys['settoggles'] += "\\toggletrue{LsAndGermsSet}\n" if self.LsAndGermInfoSet else "\\togglefalse{LsAndGermsSet}\n"
        qtys['settoggles'] += "\\toggletrue{debuggingaidsappendix}\n" if debugAidsAppendix else "\\togglefalse{debuggingaidsappendix}\n"
        qtys['settoggles'] += "\\toggletrue{pixelplotsappendix}\n" if pixelPlotAppendix else "\\togglefalse{pixelplotsappendix}\n"
        qtys['settoggles'] += "\\toggletrue{whackamoleappendix}\n" if whackamoleAppendix else "\\togglefalse{whackamoleappendix}\n"
        qtys['confidenceLevel'] = "%g" % confidenceLevel if confidenceLevel is not None else "NOT-SET"
        qtys['objective'] = "$\\log{\\mathcal{L}}$" if self.objective == "logl" else "$\\chi^2$"
        qtys['gofObjective'] = "$2\\Delta\\log{\\mathcal{L}}$" if self.objective == "logl" else "$\\chi^2$"
    
        if confidenceLevel is not None:
            cri = self.get_confidence_region(confidenceLevel)
            qtys['confidenceIntervalScaleFctr'] = "%.3g" % cri.intervalScaling
            qtys['confidenceIntervalNumNonGaugeParams'] = "%d" % cri.nNonGaugeParams
        else:
            cri = None
            qtys['confidenceIntervalScaleFctr'] = "NOT-SET"
            qtys['confidenceIntervalNumNonGaugeParams'] = "NOT-SET"

            
        # 1) get latex tables
        if verbosity > 0: 
            print "*** Generating tables ***"; _sys.stdout.flush()

        required_tables = ('targetSpamTable','targetGatesTable','datasetOverviewTable',
                           'bestGatesetSpamTable','bestGatesetSpamParametersTable','bestGatesetGatesTable','bestGatesetChoiTable',
                           'bestGatesetDecompTable','bestGatesetRotnAxisTable','bestGatesetVsTargetTable','bestGatesetErrorGenTable')
            
        if self.LsAndGermInfoSet:
            progress_tbl = 'logLProgressTable' if self.objective == "logl" else 'chi2ProgressTable'
            qtys['progressTable'] = self.get_table(progress_tbl, confidenceLevel, 'latex', verbosity)
            required_tables += ('fiducialListTable','rhoStrListTable','EStrListTable','germListTable')
        else:
            qtys['progressTable'] = _generation.get_blank_table(self.formatsToCompute)['latex']
            for key in ('fiducialListTable','rhoStrListTable','EStrListTable','germListTable'):
                qtys[key] = _generation.get_blank_table(self.formatsToCompute)['latex']

        for key in required_tables:
            qtys[key] = self.get_table(key, confidenceLevel, 'latex', verbosity)

    
        # 2) generate plots
        if verbosity > 0: 
            print "*** Generating plots ***"; _sys.stdout.flush()

        if _matplotlib.is_interactive():
            _matplotlib.pyplot.ioff()
            bWasInteractive = True
        else: bWasInteractive = False
    
        D = report_base + "_files" #figure directory relative to reportDir
        if not _os.path.isdir( _os.path.join(report_dir,D)):
            _os.mkdir( _os.path.join(report_dir,D))

        maxW,maxH = 4.0,3.0 #max width and height of graphic in latex presentation (in inches)
        maxHc = 2.5 #max height allowed for a figure with a caption (in inches)

        #Chi2 or logl plots
        if self.LsAndGermInfoSet:
            baseStr_dict = self._getBaseStrDict()

            st = 1 if self.Ls[0] == 0 else 0 #start index: skips LGST column in report color box plots        
            nPlots = (len(self.Ls[st:])-1)+1 if pixelPlotAppendix else 1

            if self.objective == "chi2":
                plotFnName,plotFnLatex = "Chi2", "$\chi^2$"
            elif self.objective == "logl":
                plotFnName,plotFnLatex = "LogL", "$\\log(\\mathcal{L})$"
            else: 
                raise ValueError("Invalid objective value: %s" % self.objective)
            
            if verbosity > 0: 
                print " -- %s plots (%d): " % (plotFnName, nPlots),; _sys.stdout.flush()

            if verbosity > 0: 
                print "1 ",; _sys.stdout.flush()
            fig = self.get_figure("bestEstimateColorBoxPlot",verbosity)
            fig.save_to(_os.path.join(report_dir, D,"best%sBoxes.pdf" % plotFnName))
            maxX = fig.get_extra_info()['nUsedXs']; maxY = fig.get_extra_info()['nUsedYs']
        
        pixplots = ""
        if pixelPlotAppendix:
            for i in range(st,len(self.Ls)-1):

                if verbosity > 0: 
                    print "%d " % (i-st+2),; _sys.stdout.flush()

                fig = self.get_figure("estimateForLIndex%dColorBoxPlot" % i, verbosity)
                fig.save_to( _os.path.join(report_dir, D,"L%d_%sBoxes.pdf" % (i,plotFnName)) )
                lx = fig.get_extra_info()['nUsedXs']; ly = fig.get_extra_info()['nUsedYs']

                W = float(lx+1)/float(maxX+1) * maxW #scale figure size according to number of rows 
                H = float(ly)  /float(maxY)   * maxH # and columns+1 (+1 for labels ~ another col) relative to initial plot
            
                pixplots += "\n"
                pixplots += "\\begin{frame}\n"
                pixplots += "\\frametitle{Iteration %d ($L=%d$): %s values}\n" % (i, self.Ls[i], plotFnLatex)
                pixplots += "\\begin{figure}\n"
                pixplots += "\\begin{center}\n"
                #pixplots += "\\adjustbox{max height=\\dimexpr\\textheight-5.5cm\\relax, max width=\\textwidth}{"
                pixplots += "\\includegraphics[width=%fin,height=%fin,keepaspectratio]{%s/L%d_%sBoxes.pdf}\n" % (W,H,D,i,plotFnName)
                pixplots += "\\end{center}\n"
                pixplots += "\\end{figure}\n"
                pixplots += "\\end{frame}\n"
    
        if verbosity > 0: 
            print ""; _sys.stdout.flush()
        
        if debugAidsAppendix:
            #Direct-GST and deviation
            if verbosity > 0: 
                print " -- Direct-X plots (2)",; _sys.stdout.flush()

            if verbosity > 0: 
                print " 1",; _sys.stdout.flush()        
            fig = self.get_figure("directLongSeqGSTColorBoxPlot",verbosity)
            fig.save_to(_os.path.join(report_dir, D,"directLongSeqGST%sBoxes.pdf" % plotFnName))

            if verbosity > 0: 
                print " 2",; _sys.stdout.flush()
            fig = self.get_figure("directLongSeqGSTDeviationColorBoxPlot",verbosity)
            fig.save_to(_os.path.join(report_dir, D,"directLongSeqGSTDeviationBoxes.pdf"))

            if verbosity > 0: 
                print ""; _sys.stdout.flush()

    
            #Small eigenvalue error rate
            if verbosity > 0: 
                print " -- Error rate plots..."; _sys.stdout.flush()
            fig = self.get_figure("smallEigvalErrRateColorBoxPlot",verbosity)
            fig.save_to(_os.path.join(report_dir, D,"smallEigvalErrRateBoxes.pdf"))


        whackamoleplots = ""
        if whackamoleAppendix:    
            #Whack-a-mole plots for highest L of each length-1 germ
            highestL = self.Ls[-1]; allGateStrings = self.gateStringLists[-1]; hammerWeight = 10.0
            len1Germs = [ g for g in self.germs if len(g) == 1 ]

            if verbosity > 0: 
                print " -- Whack-a-mole plots (%d): " % (2*len(len1Germs)),; _sys.stdout.flush()

            for i,germ in enumerate(len1Germs):
                if verbosity > 0: 
                    print "%d " % (i+1),; _sys.stdout.flush()

                fig = self.get_figure("whack%sMoleBoxes" % germ[0],verbosity)
                fig.save_to(_os.path.join(report_dir, D,"whack%sMoleBoxes.pdf" % germ[0]))
        
                whackamoleplots += "\n"
                whackamoleplots += "\\begin{frame}\n"
                whackamoleplots += "\\frametitle{Whack-a-%s-mole plot for $\mathrm{%s}^{%d}$}" % (plotFnLatex,germ[0],highestL)
                whackamoleplots += "\\begin{figure}\n"
                whackamoleplots += "\\begin{center}\n"
                #whackamoleplots += "\\adjustbox{max height=\\dimexpr\\textheight-5.5cm\\relax, max width=\\textwidth}{"
                whackamoleplots += "\\includegraphics[width=%fin,height=%fin,keepaspectratio]{%s/whack%sMoleBoxes.pdf}\n" % (maxW,maxH,D,germ[0])
                whackamoleplots += "\\end{center}\n"
                whackamoleplots += "\\end{figure}\n"
                whackamoleplots += "\\end{frame}\n"
        
            for i,germ in enumerate(len1Germs):
                if verbosity > 0: 
                    print "%d " % (len(len1Germs)+i+1),; _sys.stdout.flush()
    
                fig = self.get_figure("whack%sMoleBoxesSummed" % germ[0], verbosity)
                fig.save_to(_os.path.join(report_dir, D,"whack%sMoleBoxesSummed.pdf" % germ[0]))

                whackamoleplots += "\n"
                whackamoleplots += "\\begin{frame}\n"
                whackamoleplots += "\\frametitle{Summed whack-a-%s-mole plot for $\mathrm{%s}^{%d}$}" % (plotFnLatex,germ[0],highestL)
                whackamoleplots += "\\begin{figure}\n"
                whackamoleplots += "\\begin{center}\n"
                #whackamoleplots += "\\adjustbox{max height=\\dimexpr\\textheight-5.5cm\\relax, max width=\\textwidth}{"
                whackamoleplots += "\\includegraphics[width=%fin,height=%fin,keepaspectratio]{%s/whack%sMoleBoxesSummed.pdf}\n" % (maxW,maxH,D,germ[0])
                whackamoleplots += "\\end{center}\n"
                whackamoleplots += "\\end{figure}\n"
                whackamoleplots += "\\end{frame}\n"
    
            if verbosity > 0: 
                print ""; _sys.stdout.flush()
        
    
        #Note: set qtys keys even if the plots were not created, since the template subsitution occurs before conditional latex inclusion.
        #   Thus, these keys *must* exist otherwise substitution will process will generate an error.
        if self.objective == "chi2":
            qtys['bestGatesetBoxPlot']      = "\\includegraphics[width=%fin,height=%fin,keepaspectratio]{%s/bestChi2Boxes.pdf}" % (maxW,maxH,D)
            qtys['directLongSeqGSTBoxPlot'] = "\\includegraphics[width=%fin,height=%fin,keepaspectratio]{%s/directLongSeqGSTChi2Boxes.pdf}" % (maxW,maxHc,D)
    
        elif self.objective == "logl":
            qtys['bestGatesetBoxPlot']      = "\\includegraphics[width=%fin,height=%fin,keepaspectratio]{%s/bestLogLBoxes.pdf}" % (maxW,maxH,D)
            qtys['directLongSeqGSTBoxPlot'] = "\\includegraphics[width=%fin,height=%fin,keepaspectratio]{%s/directLongSeqGSTLogLBoxes.pdf}" % (maxW,maxHc,D)
        else: 
            raise ValueError("Invalid objective value: %s" % self.objective)
    
        qtys['directLongSeqGSTDeviationBoxPlot'] = "\\includegraphics[width=%fin,height=%fin,keepaspectratio]{%s/directLongSeqGSTDeviationBoxes.pdf}" % (maxW,maxHc,D)
        qtys['smallEvalErrRateBoxPlot'] = "\\includegraphics[width=%fin,height=%fin,keepaspectratio]{%s/smallEigvalErrRateBoxes.pdf}" % (maxW,maxHc,D)
        qtys['intermediate_pixel_plot_slides'] = pixplots
        qtys['whackamole_plot_slides'] = whackamoleplots
    
        if bWasInteractive:
            _matplotlib.pyplot.ion()
    
        # 3) populate template latex file => report latex file
        if verbosity > 0: 
            print "*** Merging into template file ***"; _sys.stdout.flush()
        
        mainTexFilename = _os.path.join(report_dir, report_base + ".tex")
        pdfFilename = _os.path.join(report_dir, report_base + ".pdf")
    
        mainTemplate = "slides_main.tex"    
        self._merge_template(qtys, mainTemplate, mainTexFilename)
    
    
        # 4) compile report latex file into PDF
        if verbosity > 0: 
            print "Latex file(s) successfully generated.  Attempting to compile with pdflatex..."; _sys.stdout.flush()
        cwd = _os.getcwd()
        if len(report_dir) > 0:  
            _os.chdir(report_dir)
    
        try:
            ret = _os.system( "%s %s > /dev/null" % (self.latexCmd, _os.path.basename(mainTexFilename)) )
            if ret == 0:
                #We could check if the log file contains "Rerun" in it, but we'll just re-run all the time now
                if verbosity > 0: 
                    print "Initial output PDF %s successfully generated." % pdfFilename #mainTexFilename
                ret = _os.system( "%s %s > /dev/null" % (self.latexCmd,_os.path.basename(mainTexFilename)) )
                if ret == 0:
                    if verbosity > 0: 
                        print "Final output PDF %s successfully generated. Cleaning up .aux and .log files." % pdfFilename #mainTexFilename
                    _os.remove( report_base + ".log" )
                    _os.remove( report_base + ".aux" )
                else:
                    print "Error: pdflatex returned code %d. Check %s.log to see details" % (ret, report_base)
            else:
                print "Error: pdflatex returned code %d. Check %s.log to see details" % (ret, report_base)
        except:
            print "Error trying to run pdflatex to generate output PDF %s. Is '%s' on your path?" % (pdfFilename,self.latexCmd)
        finally: 
            _os.chdir(cwd)
    
        return
        



    def create_presentation_ppt(self, confidenceLevel=None, filename="auto", 
                            title="auto", datasetLabel="auto", suffix="",
                            debugAidsAppendix=False,
                            pixelPlotAppendix=False, whackamoleAppendix=False,
                            m=0, M=10, verbosity=0, pptTables=False):
        """
        Create a GST Microsoft Powerpoint presentation.

        These slides can contain most (but not all) of the tables and figures from
        the "full" report but contain only minimal descriptive text.  This method 
        uses the python-pptx package to write Powerpoint files.  The resulting 
        powerpoint slides are meant to parallel those of the PDF presentation
        but are not as nice and clean.  This method exists because the Powerpoint
        format is an industry standard and makes it very easy to shamelessly 
        co-opt GST figures or entire slides for incorporation into other 
        presentations.

        Parameters
        ----------
        confidenceLevel : float, optional
           If not None, then the confidence level (between 0 and 100) used in
           the computation of confidence regions/intervals. If None, no 
           confidence regions or intervals are computed.

        filename : string, optional
           The output filename where the presentation file(s) will be saved.  
           Specifying "auto" will use the default directory and base name 
           (specified in set_additional_info) if given, otherwise the file
           "GSTSlides.pptx" will be output to the current directoy.

        title : string, optional
           The title of the presentation.  "auto" uses a default title which
           specifyies the label of the dataset as well.

        datasetLabel : string, optional
           A label given to the dataset.  If set to "auto", then the label
           will be the base name of the dataset filename without extension
           (if given) or "D" (if not).

        suffix : string, optional
           A suffix to add to the end of the presentation filename.  Useful when
           filename is "auto" and you generate different presentations using
           the same dataset.

        debugAidsAppendix : bool, optional
           Whether to include the "debugging aids" appendix.  This 
           appendix contains comparisons of GST and Direct-GST and small-
           eigenvalue error rates among other quantities potentially
           useful for figuring out why the GST estimate did not fit
           the data as well as expected.

        pixelPlotAppendix : bool, optional
           Whether to include the "pixel plots" appendix, which shows
           the goodness of fit, in the form of color box plots, for the
           intermediate iterations of the GST algortihm.

        whackamoleAppendix : bool, optional
           Whether to include the "whack-a-mole" appendix, which contains 
           colr box plots showing the effect of reducing ("whacking") one 
           particular part of the overall goodness of fit box plot.

        m, M : float, optional
           Minimum and Maximum values of the color scale used in the presentation's
           color box plots.

        verbosity : int, optional
           How much detail to send to stdout.

        pptTables : bool, optional
           If True, native powerpoint-format tables are placed in slides instead
           of creating the prettier-looking PNG images of latex-ed tables (which
           are used when False).  This option can be useful when you want to
           modify or extract a part of a table.

        Returns
        -------
        None
        """

        assert(self.bEssentialResultsSet)

        try:
            from pptx import Presentation
            from pptx.util import Inches, Pt
            from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
        except:
            raise ValueError("Cannot import pptx: it seems like python-pptx is not installed on your system!")
        
        try:
            from PIL import Image
        except:
            raise ValueError("Cannot import PIL: it seems like the python imaging library is not installed on your system!")

        #Get report output filename
        default_dir = self.additionalInfo['defaultDirectory']
        default_base = self.additionalInfo['defaultBasename']

        if filename != "auto":
            report_dir = _os.path.dirname(filename)
            report_base = _os.path.splitext( _os.path.basename(filename) )[0] + suffix
        else:
            cwd = _os.getcwd()
            report_dir  = default_dir  if (default_dir  is not None) else cwd
            report_base = default_base if (default_base is not None) else "GSTSlides"
            report_base += suffix

        if datasetLabel == "auto":
            if default_base is not None:
                datasetLabel =  default_base
            else:
                datasetLabel = "D"

        if title == "auto": title = "GST on %s" % datasetLabel

        ######  Generate PPT Presentation ######
                            
        #Note: for now, we assume the best gateset corresponds to the last L-value
        best_gs = self.gsBestEstimate

        if not self.LsAndGermInfoSet: #cannot create appendices which depend on this structure
            debugAidsAppendix = False
            pixelPlotAppendix = False
            whackamoleAppendix = False
        
        qtys = {}
        qtys['title'] = title
        qtys['datasetLabel'] = datasetLabel
        qtys['confidenceLevel'] = "%g" % confidenceLevel if confidenceLevel is not None else "NOT-SET"
        qtys['objective'] = "$\\log{\\mathcal{L}}$" if self.objective == "logl" else "$\\chi^2$"
        qtys['gofObjective'] = "$2\\Delta\\log{\\mathcal{L}}$" if self.objective == "logl" else "$\\chi^2$"
    
        if confidenceLevel is not None:
            cri = self.get_confidence_region(confidenceLevel)
            qtys['confidenceIntervalScaleFctr'] = "%.3g" % cri.intervalScaling
            qtys['confidenceIntervalNumNonGaugeParams'] = "%d" % cri.nNonGaugeParams
        else:
            cri = None
            qtys['confidenceIntervalScaleFctr'] = "NOT-SET"
            qtys['confidenceIntervalNumNonGaugeParams'] = "NOT-SET"

            
        # 1) get ppt tables
        if verbosity > 0: 
            print "*** Generating tables ***"; _sys.stdout.flush()

        required_tables = ('targetSpamTable','targetGatesTable','datasetOverviewTable',
                           'bestGatesetSpamTable','bestGatesetSpamParametersTable','bestGatesetGatesTable','bestGatesetChoiTable',
                           'bestGatesetDecompTable','bestGatesetRotnAxisTable','bestGatesetVsTargetTable','bestGatesetErrorGenTable')
            
        tableFormat = 'ppt' if pptTables else 'latex'
        if self.LsAndGermInfoSet:
            progress_tbl = 'logLProgressTable' if self.objective == "logl" else 'chi2ProgressTable'
            qtys['progressTable'] = self.get_table(progress_tbl, confidenceLevel, tableFormat, verbosity)
            required_tables += ('fiducialListTable','rhoStrListTable','EStrListTable','germListTable')

        for key in required_tables:
            qtys[key] = self.get_table(key, confidenceLevel, tableFormat, verbosity)

    
        # 2) generate plots
        if verbosity > 0: 
            print "*** Generating plots ***"; _sys.stdout.flush()

        if _matplotlib.is_interactive():
            _matplotlib.pyplot.ioff()
            bWasInteractive = True
        else: bWasInteractive = False
    
        D = report_base + "_files" #figure directory relative to reportDir
        if not _os.path.isdir( _os.path.join(report_dir,D)):
            _os.mkdir( _os.path.join(report_dir,D))

        maxW,maxH = 4.0,3.0 #max width and height of graphic in latex presentation (in inches)

        #Chi2 or logl plots
        if self.LsAndGermInfoSet:
            baseStr_dict = self._getBaseStrDict()

            st = 1 if self.Ls[0] == 0 else 0 #start index: skips LGST column in report color box plots        
            nPlots = (len(self.Ls[st:])-1)+1 if pixelPlotAppendix else 1

            if self.objective == "chi2":
                plotFnName,plotFnLatex = "Chi2", "$\chi^2$"
            elif self.objective == "logl":
                plotFnName,plotFnLatex = "LogL", "$\\log(\\mathcal{L})$"
            else: 
                raise ValueError("Invalid objective value: %s" % self.objective)
            
            if verbosity > 0: 
                print " -- %s plots (%d): " % (plotFnName, nPlots),; _sys.stdout.flush()

            if verbosity > 0: 
                print "1 ",; _sys.stdout.flush()
            fig = self.get_figure("bestEstimateColorBoxPlot",verbosity)
            fig.save_to(_os.path.join(report_dir, D,"best%sBoxes.png" % plotFnName))
            maxX = fig.get_extra_info()['nUsedXs']; maxY = fig.get_extra_info()['nUsedYs']

        pixplots = []
        if pixelPlotAppendix:
            for i in range(st,len(self.Ls)-1):

                if verbosity > 0: 
                    print "%d " % (i-st+2),; _sys.stdout.flush()

                fig = self.get_figure("estimateForLIndex%dColorBoxPlot" % i, verbosity)
                fig.save_to( _os.path.join(report_dir, D,"L%d_%sBoxes.png" % (i,plotFnName)) )
                lx = fig.get_extra_info()['nUsedXs']; ly = fig.get_extra_info()['nUsedYs']

                W = float(lx+1)/float(maxX+1) * maxW #scale figure size according to number of rows 
                H = float(ly)  /float(maxY)   * maxH # and columns+1 (+1 for labels ~ another col) relative to initial plot
            
                pixplots.append( _os.path.join(report_dir, D,"L%d_%sBoxes.png" % (i,plotFnName)) )
    
        if verbosity > 0: 
            print ""; _sys.stdout.flush()
        
        if debugAidsAppendix:
            #Direct-GST and deviation
            if verbosity > 0: 
                print " -- Direct-X plots (2)",; _sys.stdout.flush()

            if verbosity > 0: 
                print " 1",; _sys.stdout.flush()        
            fig = self.get_figure("directLongSeqGSTColorBoxPlot",verbosity)
            fig.save_to(_os.path.join(report_dir, D,"directLongSeqGST%sBoxes.png" % plotFnName))

            if verbosity > 0: 
                print " 2",; _sys.stdout.flush()
            fig = self.get_figure("directLongSeqGSTDeviationColorBoxPlot",verbosity)
            fig.save_to(_os.path.join(report_dir, D,"directLongSeqGSTDeviationBoxes.png"))

            if verbosity > 0: 
                print ""; _sys.stdout.flush()
    
            #Small eigenvalue error rate
            if verbosity > 0: 
                print " -- Error rate plots..."; _sys.stdout.flush()
            fig = self.get_figure("smallEigvalErrRateColorBoxPlot",verbosity)
            fig.save_to(_os.path.join(report_dir, D,"smallEigvalErrRateBoxes.png"))
                
    
        whackamoleplots = []
        if whackamoleAppendix:    
            #Whack-a-mole plots for highest L of each length-1 germ
            highestL = self.Ls[-1]; allGateStrings = self.gateStringLists[-1]; hammerWeight = 10.0
            len1Germs = [ g for g in self.germs if len(g) == 1 ]

            if verbosity > 0: 
                print " -- Whack-a-mole plots (%d): " % (2*len(len1Germs)),; _sys.stdout.flush()

            for i,germ in enumerate(len1Germs):
                if verbosity > 0: 
                    print "%d " % (i+1),; _sys.stdout.flush()

                fig = self.get_figure("whack%sMoleBoxes" % germ[0],verbosity)
                fig.save_to(_os.path.join(report_dir, D,"whack%sMoleBoxes.png" % germ[0]))
                whackamoleplots.append( _os.path.join(report_dir, D,"whack%sMoleBoxes.png" % germ[0]) )
        
            for i,germ in enumerate(len1Germs):
                if verbosity > 0: 
                    print "%d " % (len(len1Germs)+i+1),; _sys.stdout.flush()

                fig = self.get_figure("whack%sMoleBoxesSummed" % germ[0],verbosity)
                fig.save_to(_os.path.join(report_dir, D,"whack%sMoleBoxesSummed.png" % germ[0]))
                whackamoleplots.append( _os.path.join(report_dir, D,"whack%sMoleBoxesSummed.png" % germ[0]) )
    
            if verbosity > 0: 
                print ""; _sys.stdout.flush()
        
    
        #Note: set qtys keys even if the plots were not created, since the template subsitution occurs before conditional latex inclusion.
        #   Thus, these keys *must* exist otherwise substitution will process will generate an error.
        fileDir = _os.path.join(report_dir, D)
        if self.objective == "chi2":
            qtys['bestGatesetBoxPlot']      = "%s/bestChi2Boxes.png" % fileDir
            qtys['directLongSeqGSTBoxPlot'] = "%s/directLongSeqGSTChi2Boxes.png" % fileDir
    
        elif self.objective == "logl":
            qtys['bestGatesetBoxPlot']      = "%s/bestLogLBoxes.png" % fileDir
            qtys['directLongSeqGSTBoxPlot'] = "%s/directLongSeqGSTLogLBoxes.png" % fileDir
        else: 
            raise ValueError("Invalid objective value: %s" % self.objective)
    
        qtys['directLongSeqGSTDeviationBoxPlot'] = "%s/directLongSeqGSTDeviationBoxes.png" % fileDir
        qtys['smallEvalErrRateBoxPlot'] = "%s/smallEigvalErrRateBoxes.png" % fileDir
        qtys['intermediate_pixel_plot_slides'] = pixplots
        qtys['whackamole_plot_slides'] = whackamoleplots
    
        if bWasInteractive:
            _matplotlib.pyplot.ion()
    
        # 3) create PPT file via python-pptx
        if verbosity > 0: 
            print "*** Assembling PPT file ***"; _sys.stdout.flush()

        mainPPTFilename = _os.path.join(report_dir, report_base + ".pptx")        
        templatePath = self.templatePath if (self.templatePath is not None) else \
            _os.path.join( _os.path.dirname(_os.path.abspath(__file__)), "templates")
        templateFilename = _os.path.join( templatePath, "slides_main.pptx" )

        # slide layout indices

          #For normal powerpoint templates
        #SLD_LAYOUT_TITLE = 0
        #SLD_LAYOUT_TITLE_AND_CONTENT = 1
        #SLD_LAYOUT_TITLE_NO_CONTENT = 5

          #For sandia's template
        SLD_LAYOUT_TITLE = 0
        SLD_LAYOUT_TITLE_AND_CONTENT = 6
        SLD_LAYOUT_TITLE_NO_CONTENT = 10

        def draw_table_ppt(shapes, key, left, top, width, height, ptSize=10):
            tabDicts = qtys[key]
            tabDict = tabDicts[0] #for now, just draw the first table
            nRows = len(tabDict['row data']) + 1
            nCols = len(tabDict['column names'])

            def maxllen(s): #maximum line length
                return max( [len(ln) for ln in s.split('\n')] )

            max_col_widths = []
            for i in range(nCols):
                max_col_width = max( [ maxllen(rd[i]) for rd in tabDict['row data'] ] )
                max_col_width = max(max_col_width, maxllen(tabDict['column names'][i]) )
                max_col_widths.append(max_col_width)
            
            max_row_heights = []
            max_row_heights.append( max([ len(nm.split('\n')) for nm in tabDict['column names'] ] ) )
            for row_data in tabDict['row data']:
                max_row_heights.append( max([ len(el.split('\n')) for el in row_data ] ) )

            l = Inches(left); t = Inches(top)
            maxW = Inches(width); maxH = Inches(height)

            #table width and height per "pt" - scaling factors determined by trial and error
            Wfctr = 1.2; Hfctr = 1.2
            WperPt = Wfctr * sum(max_col_widths)
            HperPt = Hfctr * sum(max_row_heights)

            #set *desired* font size
            fontSize = Pt(ptSize)

            #compute width and height of table based on desired values
            w = int(WperPt * fontSize)
            h = int(HperPt * fontSize)

            #if table is too big, reduce point size until it fits
            if w > maxW:
                fontSize = maxW / WperPt
                w = maxW
                h = int(HperPt * fontSize)

            if h > maxH:
                fontSize = maxH / HperPt
                h = maxH
                w = int(WperPt * fontSize)            
            
            table = shapes.add_table(nRows, nCols, l, t, w, h).table
            table.first_row = True
            table.horz_banding = True

            # set column widths
            for i in range(nCols):
                table.columns[i].width = int(max_col_widths[i] * Wfctr * fontSize)

            # set row heights
            for i in range(nRows):
                table.rows[i].height = int(max_row_heights[i] * Hfctr * fontSize)

            # write column headings
            for i in range(nCols):
                table.cell(0, i).text = tabDict['column names'][i]
                tf = table.cell(0, i).text_frame
                tf.vertical_anchor = MSO_ANCHOR.MIDDLE
                tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT
                tf.word_wrap = False
                tf.paragraphs[0].alignment = PP_ALIGN.CENTER
                for run in table.cell(0, i).text_frame.paragraphs[0].runs:
                    run.font.bold = True
                    run.font.size = fontSize

            # write body cells
            for i in range(1,nRows):
                for j in range(nCols):
                    table.cell(i, j).text = tabDict['row data'][i-1][j]
                    tf = table.cell(i, j).text_frame
                    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
                    tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT
                    tf.word_wrap = False
                    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
                    for run in table.cell(i, j).text_frame.paragraphs[0].runs:
                        run.font.size = fontSize

            return table

        def draw_table_latex(shapes, key, left, top, width, height, ptSize=10):
            latexTabStr = qtys[key]
            d = {'toLatex': latexTabStr }
            print "Latexing %s table..." % key; _sys.stdout.flush()
            outputFilename = _os.path.join(fileDir, "%s.tex" % key)
            self._merge_template(d, "standalone.tex", outputFilename)

            cwd = _os.getcwd()
            _os.chdir(fileDir)
            try:
                ret = _os.system("%s -shell-escape %s.tex > /dev/null" % (self.latexCmd,key) )
                if ret == 0:
                    _os.remove( "%s.tex" % key )
                    _os.remove( "%s.log" % key )
                    _os.remove( "%s.aux" % key )
                else: raise ValueError("pdflatex returned code %d trying to render standalone %s" % (ret,key))
            except:
                raise ValueError("pdflatex failed to render standalone %s" % key)
            finally:
                _os.chdir(cwd)
            
            pathToImg = _os.path.join(fileDir, "%s.png" % key)
            return draw_pic(shapes, pathToImg, left, top, width, height)


        def draw_pic(shapes, path, left, top, width, height):
            pxWidth, pxHeight = Image.open(open(path)).size
            pxAspect = pxWidth / float(pxHeight) #aspect ratio of image
            maxAspect = width / float(height) #aspect ratio of "max" box
            if pxAspect > maxAspect: 
                w = Inches(width); h = None # image is wider & flatter than max box => constrain width
                #print "%s -> constrain width to %f so height is %f" % (path,width,width / pxAspect)
            else:
                h = Inches(height); w = None # image is taller & thinner than max box => constrain height
                #print "%s -> constrain height to %f so width is %f" % (path,height,height * pxAspect)
            return shapes.add_picture(path, Inches(left), Inches(top), w, h)

        def add_slide(typ, title):
            slide_layout = prs.slide_layouts[typ]
            slide = prs.slides.add_slide(slide_layout)
            slide.shapes.title.text = title
            return slide

        def add_text(shapes, left, top, width, height, text, ptSize=20):
            return add_text_list(shapes, left, top, width, height, [text])

        def add_text_list(shapes, left, top, width, height, textList, ptSize=20):
            l = Inches(left); t = Inches(top); w = Inches(width); h = Inches(height)
            txtBox = slide.shapes.add_textbox(l, t, w, h)
            tf = txtBox.text_frame
            tf.text = textList[0]
            for run in tf.paragraphs[0].runs:
                run.font.size = Pt(ptSize)

            for line in textList[1:]:
                p = tf.add_paragraph()
                p.text = line
                for run in p.runs:
                    run.font.size = Pt(ptSize)

            return txtBox

        drawTable = draw_table_ppt if pptTables else draw_table_latex
        
        # begin presentation creation
        #prs = Presentation() #templateFilename)
        templateDir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"templates") \
            if self.templatePath is None else self.templatePath
        prs = Presentation( _os.path.join( templateDir, "GSTTemplate.pptx" ) )


        # title slide
        slide = add_slide(SLD_LAYOUT_TITLE, title)
        subtitle_shape = slide.placeholders[1]
        subtitle_shape.text = "Your GST results in Powerpoint!"

        # goodness of fit
        if self.LsAndGermInfoSet:
            slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "%s vs GST iteration" % plotFnName)
            #body_shape = slide.shapes.placeholders[1]; tf = body_shape.text_frame
            add_text_list(slide.shapes, 1, 2, 8, 2, ['Ns is the number of gate strings', 'Np is the number of parameters'], 15)
            drawTable(slide.shapes, 'progressTable', 1, 3, 8.5, 4, ptSize=10)
        
            slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "Detailed %s Analysis" % plotFnName)
            draw_pic(slide.shapes, qtys['bestGatesetBoxPlot'], 1, 1.5, 8, 5.5)

        # gate esimtates
        slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "GST Estimate vs. target")
        drawTable(slide.shapes, 'bestGatesetVsTargetTable', 1.5, 1.5, 7, 2, ptSize=10)
        drawTable(slide.shapes, 'bestGatesetErrorGenTable', 1.5, 3.7, 7, 3.5, ptSize=10)

        slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "GST Estimate decomposition")
        drawTable(slide.shapes, 'bestGatesetDecompTable', 1, 1.5, 7.5, 3.5 , ptSize=10)
        drawTable(slide.shapes, 'bestGatesetRotnAxisTable', 1, 5.1, 5, 1.5, ptSize=10)        

        slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "Raw GST Estimate: Gates")
        drawTable(slide.shapes, 'bestGatesetGatesTable', 1, 2, 8, 5, ptSize=10)

        slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "Raw GST Estimate: SPAM")
        drawTable(slide.shapes, 'bestGatesetSpamTable', 1, 1.5, 8, 3, ptSize=10)
        drawTable(slide.shapes, 'bestGatesetSpamParametersTable', 1, 5, 3, 1, ptSize=10)

        slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "Raw GST Choi Matrices")
        drawTable(slide.shapes, 'bestGatesetChoiTable', 0.5, 1.5, 8.5, 5.5, ptSize=10)

        #Inputs to GST
        slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "Target SPAM")
        drawTable(slide.shapes, 'targetSpamTable', 1.5, 1.5, 7, 5, ptSize=10)

        slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "Target Gates")
        drawTable(slide.shapes, 'targetGatesTable', 1, 1.5, 7, 5, ptSize=10)

        if self.LsAndGermInfoSet:
            slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "Fiducial and Germ Gate Strings")
            drawTable(slide.shapes, 'fiducialListTable', 1, 1.5, 4, 3, ptSize=10)
            drawTable(slide.shapes, 'germListTable', 5.5, 1.5, 4, 5, ptSize=10)

        slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "Dataset Overview")
        drawTable(slide.shapes, 'datasetOverviewTable', 1, 2, 5, 4, ptSize=10)

        if debugAidsAppendix:
            #Debugging aids slides
            slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "Direct-GST")
            draw_pic(slide.shapes, qtys['directLongSeqGSTBoxPlot'], 1, 1.5, 8, 5.5)

            slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "Direct-GST Deviation")
            draw_pic(slide.shapes, qtys['directLongSeqGSTDeviationBoxPlot'], 1, 1.5, 8, 5.5)

            slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "Per-gate error rates")
            draw_pic(slide.shapes, qtys['smallEvalErrRateBoxPlot'], 1, 1.5, 8, 5.5)

        if pixelPlotAppendix:
            for i,pixPlotPath in zip( range(st,len(self.Ls)-1), pixplots ):
                slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "Iteration %d (L=%d): %s values" % (i,self.Ls[i],plotFnName))
                draw_pic(slide.shapes, pixPlotPath, 1, 1.5, 8, 5.5)

        if whackamoleAppendix:
            curPlot = 0
            for i,germ in enumerate(len1Germs):
                slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "Whack-a-%s-mole plot for %s^%d" % (plotFnName,germ[0],highestL))
                draw_pic(slide.shapes, whackamoleplots[curPlot], 1, 1.5, 8, 5.5)
                curPlot += 1

            for i,germ in enumerate(len1Germs):
                slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "Summed whack-a-%s-mole plot for %s^%d" % (plotFnName,germ[0],highestL))
                draw_pic(slide.shapes, whackamoleplots[curPlot], 1, 1.5, 8, 5.5)
                curPlot += 1

        # 4) save presenation as PPTX file
        prs.save(mainPPTFilename)
        print "Final output PPT %s successfully generated." % mainPPTFilename
        return


    #FUTURE?
    #def create_brief_html_page(self, tableClass):
    #    pass
    #
    #def create_full_html_page(self, tableClass):
    #    pass


