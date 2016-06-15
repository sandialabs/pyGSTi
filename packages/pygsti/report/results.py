#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation              
#    This Software is released under the GPL license detailed    
#    in the file "license.txt" in the top-level pyGSTi directory 
#*****************************************************************
""" Defines the Results class and supporting functionality."""

import sys as _sys
import os as _os
import re as _re
import collections as _collections
import matplotlib as _matplotlib
import itertools as _itertools

from ..objects import gatestring as _gs
from ..construction import spamspecconstruction as _ssc
from ..algorithms import optimize_gauge as _optimizeGauge
from ..tools import listtools as _lt
from ..tools import basistools as _bt
from .. import _version

import latex as _latex
import generation as _generation
import plotting as _plotting
from resultcache import ResultCache as _ResultCache

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

    def __init__(self, templatePath=None, latexCmd="pdflatex"):
        """ 
        Initialize a Results object.

        Parameters
        ----------
        templatePath : string or None, optional
            A local path to the stored GST report template files.  The
            default value of None means to use the default path, which
            is almost always what you want.
            
        latexCmd : string or None, optional
            The system command used to compile latex documents.
        """

        # Internal Flags
        self._bEssentialResultsSet = False
        self._LsAndGermInfoSet = False

        # Confidence regions: key == confidence level, val = ConfidenceRegion
        self._confidence_regions = {} # plain dict. Key == confidence level
        self._specials = _ResultCache(self._get_special_fns(), self, "special")

        self.tables = _ResultCache(self._get_table_fns(), self, "table")
        self.figures = _ResultCache(self._get_figure_fns(), self, "figure")
        #self.qtys = _ResultCache(self._get_qty_fns(), self, "computable qty")

        #Public API parameters
        self.gatesets = {}
        self.gatestring_lists = {}
        self.dataset = None
        self.parameters = {}
        self.options = ResultOptions()
        self.confidence_level = None #holds "current" (i.e. "last")
        
        # Set default display options (affect how results are displayed)
        self.options.long_tables = False
        self.options.table_class = "pygstiTbl"
        self.options.template_path = templatePath
        self.options.latex_cmd = latexCmd

        # Set default parameter values
        self.parameters = { 'objective': None,
                            'constrainToTP': None,
                            'weights':None, 
                            'minProbClip': 1e-6,
                            'minProbClipForWeighting': 1e-4,
                            'probClipInterval': (-1e6,1e6),
                            'radius': 1e-4,
                            'hessianProjection': 'std',
                            'defaultDirectory': None,
                            'defaultBasename': None,
                            'linlogPercentile':  5,
                            'memLimit': None,
                            'gaugeOptParams': {} }


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
        self.gatesets['target'] = targetGateset
        self.gatesets['iteration estimates'] = [ gatesetEstimate ]
        self.gatesets['final estimate'] = gatesetEstimate
        self.gatestring_lists['iteration'] = [ gatestring_list ]
        self.gatestring_lists['final'] = gatestring_list
        self.dataset = dataset
        self.parameters['objective'] = objective
        self.parameters['constrainToTP'] = constrainToTP

        if gatesetEstimate_noGaugeOpt is not None:
            self.gatesets['iteration estimates pre gauge opt'] = \
                [ gatesetEstimate_noGaugeOpt ]

        self._bEssentialResultsSet = True


    def init_Ls_and_germs(self, objective, targetGateset, dataset,
                              seedGateset, Ls, germs, gatesetsByL, gateStringListByL, 
                              prepStrs, effectStrs, truncFn, constrainToTP, fidPairs=None,
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

        prepStrs : list of GateStrings
            The list of state preparation fiducial strings
            in the objective optimization.

        effectStrs : list of GateStrings
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
            
        fidPairs : list of 2-tuples, optional
            Specifies a subset of all prepStr,effectStr string pairs to be used in this
            analysis.  Each element of fidPairs is a (iRhoStr, iEStr) 2-tuple of integers,
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
        self.gatesets['target'] = targetGateset
        self.gatesets['seed'] = seedGateset
        self.gatesets['iteration estimates'] = gatesetsByL
        self.gatesets['final estimate'] = gatesetsByL[-1]
        self.gatestring_lists['iteration'] = gateStringListByL
        self.gatestring_lists['final'] = gateStringListByL[-1]
        self.gatestring_lists['all'] = _lt.remove_duplicates( 
            list(_itertools.chain(*gateStringListByL)) )
        self.dataset = dataset
        self.parameters['objective'] = objective
        self.parameters['constrainToTP'] = constrainToTP
        if gatesetsByL_noGaugeOpt is not None:
            self.gatesets['iteration estimates pre gauge opt'] = \
                gatesetsByL_noGaugeOpt

        self._bEssentialResultsSet = True

        #Set "Ls and germs" info: gives particular structure
        # to the gateStringLists used to obtain estimates
        self.gatestring_lists['prep fiducials'] = prepStrs
        self.gatestring_lists['effect fiducials'] = effectStrs
        self.gatestring_lists['germs'] = germs
        self.parameters['max length list'] = Ls
        self.parameters['fiducial pairs'] = fidPairs
        self.parameters['L,germ tuple base string dict'] = \
            _collections.OrderedDict( [ ( (L,germ), truncFn(germ,L) ) 
                                        for L in Ls for germ in germs] )
        self._LsAndGermInfoSet = True


    def __setstate__(self, stateDict):
        #Must set ResultCache parent & functions, since these are
        # not pickled (to avoid circular pickle references)
        self.__dict__.update(stateDict)
        self._specials._setparent(self._get_special_fns(), self)
        self.tables._setparent(self._get_table_fns(), self)
        self.figures._setparent(self._get_figure_fns(), self)

    def __str__(self):
        s  = "----------------------------------------------------------\n"
        s += "---------------- pyGSTi Results Object -------------------\n"
        s += "----------------------------------------------------------\n"
        s += "\n"
        s += "I can create reports for you directly, via my create_XXX\n"
        s += "functions, or you can query me for result data via members:\n\n"
        s += " .dataset    -- the DataSet used to generate these results\n\n"
        s += " .gatesets   -- a dictionary of GateSet objects w/keys:\n"
        s += " ---------------------------------------------------------\n"
        s += "  " + "\n  ".join(self.gatesets.keys()) + "\n"
        s += "\n"
        s += " .gatestring_lists   -- a dict of GateString lists w/keys:\n"
        s += " ---------------------------------------------------------\n"
        s += "  " + "\n  ".join(self.gatestring_lists.keys()) + "\n"
        s += "\n"
        s += " .tables   -- a dict of ReportTable objects w/keys:\n"
        s += " ---------------------------------------------------------\n"
        s += "  " + "\n  ".join(self.tables.keys()) + "\n"
        s += "\n"
        s += " .figures   -- a dict of ReportFigure objects w/keys:\n"
        s += " ---------------------------------------------------------\n"
        s += "  " + "\n  ".join(self.figures.keys()) + "\n"
        s += "\n"
        s += " .parameters   -- a dict of simulation parameters:\n"
        s += " ---------------------------------------------------------\n"
        s += "  " + "\n  ".join(self.parameters.keys()) + "\n"
        s += "\n"
        s += " .options   -- a container of display options:\n"
        s += " ---------------------------------------------------------\n"
        s += self.options.describe("   ") + "\n"
        s += "\n"
        s += "NOTE: passing 'tips=True' to create_full_report_pdf or\n"
        s += " create_brief_report_pdf will add markup to the resulting\n"
        s += " PDF indicating how tables and figures in the PDF correspond\n"
        s += " to the values of .tables[ ] and .figures[ ] listed above.\n"
        return s


    def _get_table_fns(self):
        """ 
        Return a dictionary of functions which create a table identified by
        the dictionary key.  These functions are used for the lazy creation
        of tables within the "tables" member of a Results instance.
        """

        #Validation functions: return a list of the computable key(s)
        # which match their single "key" argument.  It can be assumed
        # that "key" is either equal to or matches the corresponding
        # compute-function key.  Since the latter may be a regular expression,
        # "key" may also be this same regular-expression, in which case 
        # a list of currently computable keys (based on current Results
        # parameters, etc.) should be returned.  In the more mundane
        # case where key is just a string, the function simply returns
        # that same string when that key can be computed, and None 
        # otherwise.
        def validate_none(key):
            return [key]
        def validate_essential(key):
            return [key] if self._bEssentialResultsSet else []
        def validate_LsAndGerms(key):
            return [key] if (self._bEssentialResultsSet and
                             self._LsAndGermInfoSet) else []

        def setup():
            return (self.gatesets['target'], self.gatesets['final estimate'])

        fns = _collections.OrderedDict()

        def fn(key, confidenceLevel, vb):
            return _generation.get_blank_table()
        fns['blankTable'] = (fn, validate_none)

        # target gateset tables
        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            return _generation.get_gateset_spam_table(gsTgt, None)
        fns['targetSpamTable'] = (fn, validate_essential)


        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            return _generation.get_gateset_spam_table(gsTgt, None, False)
        fns['targetSpamBriefTable'] = (fn, validate_essential)
             

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            return _generation.get_unitary_gateset_gates_table(
                gsTgt, None)
        fns['targetGatesTable'] = (fn, validate_essential)


        # dataset and gatestring list tables
        def fn(key, confidenceLevel, vb):
            #maxLen = max( 2*max( map(len,self.prepStrs + self.effectStrs) ),
            #             10 ) #heuristic (unused)
            gsTgt, gsBest = setup()
            if self._LsAndGermInfoSet:
                strs = ( self.gatestring_lists['prep fiducials'], 
                         self.gatestring_lists['effect fiducials'] )
            else: strs = None
            return _generation.get_dataset_overview_table(
                self.dataset, gsTgt, 10, strs)
        fns['datasetOverviewTable'] = (fn, validate_essential)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            strs = ( self.gatestring_lists['prep fiducials'], 
                     self.gatestring_lists['effect fiducials'] )

            return _generation.get_gatestring_multi_table(
                strs, ["Prep.","Measure"], "Fiducials")
        fns['fiducialListTable'] = (fn, validate_LsAndGerms)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            return _generation.get_gatestring_table(
                self.gatestring_lists['prep fiducials'],
                "Preparation Fiducial")
        fns['prepStrListTable'] = (fn, validate_LsAndGerms)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            return _generation.get_gatestring_table(
                self.gatestring_lists['effect fiducials'],
                "Measurement Fiducial")
        fns['effectStrListTable'] = (fn, validate_LsAndGerms)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            return _generation.get_gatestring_table(
                self.gatestring_lists['germs'], "Germ")
        fns['germListTable'] = (fn, validate_LsAndGerms)


        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            return _generation.get_gatestring_table(
                self.gatestring_lists['germs'], "Germ", nCols=2)
        fns['germList2ColTable'] = (fn, validate_LsAndGerms)


        # Estimated gateset tables
        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            cri = self._get_confidence_region(confidenceLevel)
            return _generation.get_gateset_spam_table(gsBest, cri)
        fns['bestGatesetSpamTable'] = (fn, validate_essential)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            cri = self._get_confidence_region(confidenceLevel)
            return _generation.get_gateset_spam_table(gsBest, cri, False)
        fns['bestGatesetSpamBriefTable'] = (fn, validate_essential)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            cri = self._get_confidence_region(confidenceLevel)
            return _generation.get_gateset_spam_parameters_table(gsBest, cri)
        fns['bestGatesetSpamParametersTable'] = (fn, validate_essential)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            cri = self._get_confidence_region(confidenceLevel)
            return _generation.get_gateset_gates_table(gsBest, cri)
        fns['bestGatesetGatesTable'] = (fn, validate_essential)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            cri = self._get_confidence_region(confidenceLevel)
            return _generation.get_gateset_choi_table(gsBest, cri)
        fns['bestGatesetChoiTable'] = (fn, validate_essential)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            cri = self._get_confidence_region(confidenceLevel)
            return _generation.get_gateset_decomp_table(gsBest, cri)
        fns['bestGatesetDecompTable'] = (fn, validate_essential)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            cri = self._get_confidence_region(confidenceLevel)
            return _generation.get_gateset_rotn_axis_table(gsBest, cri, True)
        fns['bestGatesetRotnAxisTable'] = (fn, validate_essential)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            cri = self._get_confidence_region(confidenceLevel)
            return _generation.get_gateset_eigenval_table(gsBest, gsTgt, cri)
        fns['bestGatesetEvalTable'] = (fn, validate_essential)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            #cri = self._get_confidence_region(confidenceLevel)
            return _generation.get_gateset_closest_unitary_table(gsBest) #, cri)
        fns['bestGatesetClosestUnitaryTable'] = (fn, validate_essential)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            cri = self._get_confidence_region(confidenceLevel)
            return _generation.get_gates_vs_target_table(gsBest, gsTgt, cri)
        fns['bestGatesetVsTargetTable'] = (fn, validate_essential)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            cri = self._get_confidence_region(confidenceLevel)
            return _generation.get_spam_vs_target_table(gsBest, gsTgt, cri)
        fns['bestGatesetSpamVsTargetTable'] = (fn, validate_essential)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            cri = self._get_confidence_region(confidenceLevel)
            return _generation.get_gates_vs_target_err_gen_table(
                gsBest, gsTgt, cri)
        fns['bestGatesetErrorGenTable'] = (fn, validate_essential)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            cri = self._get_confidence_region(confidenceLevel)
            return _generation.get_gates_vs_target_angles_table(
                gsBest, gsTgt, cri)
        fns['bestGatesetVsTargetAnglesTable'] = (fn, validate_essential)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            cri = self._get_confidence_region(confidenceLevel)
            return _generation.get_gaugeopt_params_table(
                self.parameters['gaugeOptParams'])
        fns['bestGatesetGaugeOptParamsTable'] = (fn, validate_essential)



        # progress tables
        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            return _generation.get_chi2_progress_table(
                self.parameters['max length list'],
                self.gatesets['iteration estimates'],
                self.gatestring_lists['iteration'], self.dataset)
        fns['chi2ProgressTable'] = (fn, validate_LsAndGerms)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            return _generation.get_logl_progress_table(
                self.parameters['max length list'],
                self.gatesets['iteration estimates'],
                self.gatestring_lists['iteration'], self.dataset)
        fns['logLProgressTable'] = (fn, validate_LsAndGerms)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            if self.parameters['objective'] == "logl":
                return _generation.get_logl_progress_table(
                    self.parameters['max length list'],
                    self.gatesets['iteration estimates'],
                    self.gatestring_lists['iteration'], self.dataset)
            elif self.parameters['objective'] == "chi2":
                return _generation.get_chi2_progress_table(
                    self.parameters['max length list'],
                    self.gatesets['iteration estimates'],
                    self.gatestring_lists['iteration'], self.dataset)
            else: raise ValueError("Invalid Objective: %s" % 
                                   self.parameters['objective'])
        fns['progressTable'] = (fn, validate_LsAndGerms)


        # figure-containing tables
        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            return _generation.get_gateset_gate_boxes_table(
                gsTgt, "targetGatesBoxes")
        fns['targetGatesBoxTable'] = (fn, validate_essential)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            return _generation.get_gates_vs_target_err_gen_boxes_table(
                gsBest, gsTgt, "bestErrgenBoxes")
        fns['bestGatesetErrGenBoxTable'] = (fn, validate_essential)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            return _generation.get_gateset_eigenval_table(
                gsBest, gsTgt, "bestEvalPolarPlt")
        fns['bestGatesetEvalTable'] = (fn, validate_essential)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            return _generation.get_gateset_relative_eigenval_table(
                gsBest, gsTgt, "bestRelEvalPolarPlt")
        fns['bestGatesetRelEvalTable'] = (fn, validate_essential)

        def fn(key, confidenceLevel, vb):
            gsTgt, gsBest = setup()
            cri = self._get_confidence_region(confidenceLevel)
            return _generation.get_gateset_choi_eigenval_table(
                gsBest, "bestChoiEvalBars", confidenceRegionInfo=cri)
        fns['bestGatesetChoiEvalTable'] = (fn, validate_essential)



        return fns


    def _get_figure_fns(self):
        """ 
        Return a dictionary of functions which create a figure identified by
        the dictionary key.  These functions are used for the lazy creation
        of figures within the "figures" member of a Results instance.
        """

        def getPlotFn():
            obj = self.parameters['objective']
            assert(obj in ("chi2","logl"))
            if   obj == "chi2": return _plotting.chi2_boxplot
            elif obj == "logl": return _plotting.logl_boxplot

        def getDirectPlotFn():
            obj = self.parameters['objective']
            assert(obj in ("chi2","logl"))
            if   obj == "chi2": return _plotting.direct_chi2_boxplot
            elif obj == "logl": return _plotting.direct_logl_boxplot

        def getWhackAMolePlotFn():
            obj = self.parameters['objective']
            assert(obj in ("chi2","logl"))
            if   obj == "chi2": return _plotting.whack_a_chi2_mole_boxplot
            elif obj == "logl": return _plotting.whack_a_logl_mole_boxplot

        def getMPC():
            obj = self.parameters['objective']
            assert(obj in ("chi2","logl"))
            if obj == "chi2":
                return self.parameters['minProbClipForWeighting']
            elif obj == "logl": 
                return self.parameters['minProbClip']

        def plot_setup():
            m = 0
            M = 10
            baseStr_dict = self._getBaseStrDict()
            strs  = (self.gatestring_lists['prep fiducials'], 
                     self.gatestring_lists['effect fiducials'])
            germs = self.gatestring_lists['germs']
            gsBest = self.gatesets['final estimate']
            fidPairs = self.parameters['fiducial pairs']
            Ls = self.parameters['max length list']
            st = 1 if Ls[0] == 0 else 0 #start index: skip LGST column in plots

            return Ls,germs,gsBest,fidPairs,m,M,baseStr_dict,strs,st

        def noConfidenceLevelDependence(level):
            """ Designates a figure as independent of the confidence level"""
            if level is not None: raise _ResultCache.NoCRDependenceError

        def validate_LsAndGerms(key):
            return [key] if (self._bEssentialResultsSet and
                             self._LsAndGermInfoSet) else []


        fns = _collections.OrderedDict()


        def fn(key, confidenceLevel, vb):
            noConfidenceLevelDependence(confidenceLevel)
            strs  = (self.gatestring_lists['prep fiducials'], 
                     self.gatestring_lists['effect fiducials'])
            return _plotting.gof_boxplot_keyplot(strs)
        fns["colorBoxPlotKeyPlot"] = (fn,validate_LsAndGerms)

        def fn(key, confidenceLevel, vb):
            noConfidenceLevelDependence(confidenceLevel)
            plotFn = getPlotFn();  mpc = getMPC()
            Ls,germs,gsBest,fidPairs,m,M,baseStr_dict,strs,st = plot_setup()
            return plotFn(Ls[st:], germs, baseStr_dict,
                          self.dataset, gsBest, strs,
                          r"$L$", "germ", scale=1.0, sumUp=False,
                          histogram=True, title="", fidPairs=fidPairs, linlg_pcntle=float(self.parameters['linlogPercentile']) / 100,
                          minProbClipForWeighting=mpc, save_to="", ticSize=20)
        fns["bestEstimateColorBoxPlot"] = (fn,validate_LsAndGerms)


        def fn(key, confidenceLevel, vb):
            noConfidenceLevelDependence(confidenceLevel)
            plotFn = getPlotFn(); mpc = getMPC()
            Ls,germs,gsBest,fidPairs,m,M,baseStr_dict,strs,st = plot_setup()
            return plotFn( Ls[st:], germs, baseStr_dict,
                           self.dataset, gsBest, strs,
                           r"$L$", "germ", scale=1.0, sumUp=False,
                           histogram=True, title="", fidPairs=fidPairs, linlg_pcntle=float(self.parameters['linlogPercentile']) / 100,
                           save_to="", ticSize=20, minProbClipForWeighting=mpc,
                           invert=True)
        fns["invertedBestEstimateColorBoxPlot"] = (fn,validate_LsAndGerms)

        def fn(key, confidenceLevel, vb):
            noConfidenceLevelDependence(confidenceLevel)
            plotFn = getPlotFn();  mpc = getMPC()
            Ls,germs,gsBest,fidPairs,m,M,baseStr_dict,strs,st = plot_setup()
            sumScale = len(strs[0])*len(strs[1]) \
                if fidPairs is None else len(fidPairs)
            return plotFn( Ls[st:], germs, baseStr_dict,
                           self.dataset, gsBest, strs,
                          r"$L$", "germ", scale=1.0,
                           sumUp=True, histogram=False, title="",
                           fidPairs=fidPairs, minProbClipForWeighting=mpc,
                           save_to="", ticSize=14, linlg_pcntle=float(self.parameters['linlogPercentile']) / 100)
        fns["bestEstimateSummedColorBoxPlot"] = (fn,validate_LsAndGerms)
            

        expr1 = "estimateForLIndex(\d+?)ColorBoxPlot"
        def fn(key, confidenceLevel, vb):
            noConfidenceLevelDependence(confidenceLevel)
            plotFn = getPlotFn();  mpc = getMPC()
            Ls,germs,gsBest,fidPairs,m,M,baseStr_dict,strs,st = plot_setup()
            i = int(_re.match(expr1,key).group(1))
            return plotFn( Ls[st:i+1], germs, baseStr_dict,
                        self.dataset, self.gatesets['iteration estimates'][i],
                        strs, r"$L$", "germ", scale=1.0, sumUp=False,
                        histogram=False, title="", fidPairs=fidPairs, linlg_pcntle=float(self.parameters['linlogPercentile']) / 100,
                        save_to="", minProbClipForWeighting=mpc, ticSize=20)
        def fn_validate(key):
            if not self._LsAndGermInfoSet: return []
            
            keys = ["estimateForLIndex%dColorBoxPlot" % i 
                    for i in range(len(self.parameters['max length list']))]
            if key == expr1: return keys # all computable keys
            elif key in keys: return [key]
            else: return []
        fns[expr1] = (fn, fn_validate)

        def fn(key, confidenceLevel, vb):
            noConfidenceLevelDependence(confidenceLevel)
            Ls,germs,gsBest,fidPairs,m,M,baseStr_dict,strs,st = plot_setup()
            return _plotting.blank_boxplot( 
                Ls[st:], germs, baseStr_dict, strs, r"$L$", "germ",
                scale=1.0, title="", sumUp=False, save_to="", ticSize=20)
        fns["blankBoxPlot"] = (fn,validate_LsAndGerms)

        def fn(key, confidenceLevel, vb):
            noConfidenceLevelDependence(confidenceLevel)
            Ls,germs,gsBest,fidPairs,m,M,baseStr_dict,strs,st = plot_setup()
            return _plotting.blank_boxplot( 
                Ls[st:], germs, baseStr_dict, strs, r"$L$", "germ",
                scale=1.0, title="", sumUp=True, save_to="", ticSize=20)
        fns["blankSummedBoxPlot"] = (fn,validate_LsAndGerms)


        def fn(key, confidenceLevel, vb):
            noConfidenceLevelDependence(confidenceLevel)
            directPlotFn = getDirectPlotFn(); mpc = getMPC()
            Ls,germs,gsBest,fidPairs,m,M,baseStr_dict,strs,st = plot_setup()
            directLGST = self._specials.get('direct_lgst_gatesets',verbosity=vb)
            return directPlotFn( Ls[st:], germs, baseStr_dict, self.dataset,
                                 directLGST, strs, r"$L$", "germ",
                                 scale=1.0, sumUp=False, title="",
                                 minProbClipForWeighting=mpc, fidPairs=fidPairs,
                                 save_to="", ticSize=20, linlg_pcntle=float(self.parameters['linlogPercentile']) / 100)
        fns["directLGSTColorBoxPlot"] = (fn,validate_LsAndGerms)

        def fn(key, confidenceLevel, vb):
            noConfidenceLevelDependence(confidenceLevel)
            directPlotFn = getDirectPlotFn(); mpc = getMPC()
            Ls,germs,gsBest,fidPairs,m,M,baseStr_dict,strs,st = plot_setup()
            directLongSeqGST = self._specials.get('DirectLongSeqGatesets',
                                                  verbosity=vb)
            return directPlotFn( Ls[st:], germs, baseStr_dict, self.dataset,
                                 directLongSeqGST, strs, r"$L$", "germ",
                                 scale=1.0, sumUp=False, title="",
                                 minProbClipForWeighting=mpc, fidPairs=fidPairs,
                                 save_to="", ticSize=20, linlg_pcntle=float(self.parameters['linlogPercentile']) / 100)
        fns["directLongSeqGSTColorBoxPlot"] = (fn,validate_LsAndGerms)

        def fn(key, confidenceLevel, vb):
            noConfidenceLevelDependence(confidenceLevel)
            Ls,germs,gsBest,fidPairs,m,M,baseStr_dict,strs,st = plot_setup()
            directLGST = self._specials.get('direct_lgst_gatesets',verbosity=vb)
            return _plotting.direct_deviation_boxplot( 
                Ls[st:], germs, baseStr_dict, self.dataset,
                gsBest, directLGST, r"$L$", "germ", scale=1.0,
                prec=-1, title="", save_to="", ticSize=20)
        fns["directLGSTDeviationColorBoxPlot"] = (fn,validate_LsAndGerms)

        def fn(key, confidenceLevel, vb):
            noConfidenceLevelDependence(confidenceLevel)
            Ls,germs,gsBest,fidPairs,m,M,baseStr_dict,strs,st = plot_setup()
            directLongSeqGST = self._specials.get('DirectLongSeqGatesets',
                                                  verbosity=vb)
            return _plotting.direct_deviation_boxplot(
                Ls[st:], germs, baseStr_dict, self.dataset,
                gsBest, directLongSeqGST, r"$L$", "germ",
                scale=1.0, prec=-1, title="", save_to="", ticSize=20)
        fns["directLongSeqGSTDeviationColorBoxPlot"] = (fn,validate_LsAndGerms)

        def fn(key, confidenceLevel, vb):
            noConfidenceLevelDependence(confidenceLevel)
            Ls,germs,gsBest,fidPairs,m,M,baseStr_dict,strs,st = plot_setup()
            directLongSeqGST = self._specials.get('DirectLongSeqGatesets',
                                                  verbosity=vb)
            return _plotting.small_eigval_err_rate_boxplot(
                Ls[st:], germs, baseStr_dict, self.dataset,
                directLongSeqGST, r"$L$", "germ", scale=1.0, title="",
                save_to="", ticSize=20)
        fns["smallEigvalErrRateColorBoxPlot"] = (fn,validate_LsAndGerms)


        expr2 = "whack(.+?)MoleBoxes"
        def fn(key, confidenceLevel, vb):
            noConfidenceLevelDependence(confidenceLevel)
            Ls,germs,gsBest,fidPairs,m,M,baseStr_dict,strs,st = plot_setup()
            highestL = Ls[-1]; hammerWeight = 10.0; mpc = getMPC()
            gateLabel = _re.match(expr2,key).group(1)
            strToWhack = _gs.GateString( (gateLabel,)*highestL )
            whackAMolePlotFn = getWhackAMolePlotFn()
            return whackAMolePlotFn(strToWhack, self.gatestring_lists['all'],
                                    Ls[st:], germs, baseStr_dict, self.dataset,
                                    gsBest, strs, r"$L$", "germ", scale=1.0,
                                    sumUp=False,title="",whackWith=hammerWeight,
                                    save_to="", minProbClipForWeighting=mpc,
                                    ticSize=20, fidPairs=fidPairs)
        def fn_validate(key):
            if not self._LsAndGermInfoSet: return []

            #only whack-a-mole plots for the length-1 germs are available
            len1GermFirstEls = [ g[0] for g in self.gatestring_lists['germs'] 
                                 if len(g) == 1 ]
            
            keys = ["whack%sMoleBoxes" % gl for gl in len1GermFirstEls]
            if key == expr2: return keys # all computable keys
            elif key in keys: return [key]
            else: return []
        fns[expr2] = (fn, fn_validate)


        expr3 = "whack(.+?)MoleBoxesSummed"
        def fn(key, confidenceLevel, vb):
            noConfidenceLevelDependence(confidenceLevel)
            Ls,germs,gsBest,fidPairs,m,M,baseStr_dict,strs,st = plot_setup()
            highestL = Ls[-1]; hammerWeight = 10.0; mpc = getMPC()
            gateLabel = _re.match(expr3,key).group(1)
            strToWhack = _gs.GateString( (gateLabel,)*highestL )
            whackAMolePlotFn = getWhackAMolePlotFn()
            return whackAMolePlotFn(strToWhack, self.gatestring_lists['all'],
                                    Ls[st:], germs, baseStr_dict, self.dataset,
                                    gsBest, strs, r"$L$", "germ", scale=1.0,
                                    sumUp=True, title="",whackWith=hammerWeight,
                                    save_to="", minProbClipForWeighting=mpc,
                                    ticSize=20, fidPairs=fidPairs)
        def fn_validate(key):
            if not self._LsAndGermInfoSet: return []

            #only whack-a-mole plots for the length-1 germs are available
            len1GermFirstEls = [ g[0] for g in self.gatestring_lists['germs'] 
                                 if len(g) == 1 ]
            
            keys = ["whack%sMoleBoxesSummed" % gl for gl in len1GermFirstEls]
            if key == expr3: return keys # all computable keys
            elif key in keys: return [key]
            else: return []
        fns[expr3] = (fn, fn_validate)


        expr4 = "bestGateErrGenBoxes(.+)"
        def fn(key, confidenceLevel, vb):
            #cri = self._get_confidence_region(confidenceLevel)
            noConfidenceLevelDependence(confidenceLevel)
            gateLabel = _re.match(expr4,key).group(1)
            gate = self.gatesets['final estimate'].gates[gateLabel]
            targetGate = self.gatesets['target'].gates[gateLabel]
            basisNm   = self.gatesets['final estimate'].get_basis_name()
            basisDims = self.gatesets['final estimate'].get_basis_dimension()
            assert(basisNm == self.gatesets['target'].get_basis_name())
            return _plotting.gate_matrix_errgen_boxplot(
                gate, targetGate, save_to="", mxBasis=basisNm,
                mxBasisDims=basisDims)

        def fn_validate(key):
            if not self._bEssentialResultsSet: return []            
            keys = ["bestGateErrGenBoxes%s" % gl 
                    for gl in self.gatesets['final estimate'].gates ]
            if key == expr4: return keys # all computable keys
            elif key in keys: return [key]
            else: return []
        fns[expr4] = (fn, fn_validate)


        expr5 = "targetGateBoxes(.+)"
        def fn(key, confidenceLevel, vb):
            #cri = self._get_confidence_region(confidenceLevel)
            noConfidenceLevelDependence(confidenceLevel)
            gateLabel = _re.match(expr5,key).group(1)
            gate = self.gatesets['target'].gates[gateLabel]            
            return _plotting.gate_matrix_boxplot(gate, save_to="", 
                mxBasis=self.gatesets['target'].get_basis_name(),
                mxBasisDims=self.gatesets['target'].get_basis_dimension())

        def fn_validate(key):
            if not self._bEssentialResultsSet: return []            
            keys = ["targetGateBoxes%s" % gl 
                    for gl in self.gatesets['final estimate'].gates ]
            if key == expr5: return keys # all computable keys
            elif key in keys: return [key]
            else: return []
        fns[expr5] = (fn, fn_validate)

        expr6 = "bestEstimatePolar(.+?)EvalPlot"
        def fn(key, confidenceLevel, vb):
            #cri = self._get_confidence_region(confidenceLevel)
            noConfidenceLevelDependence(confidenceLevel)
            gateLabel = _re.match(expr6,key).group(1)
            gate = self.gatesets['final estimate'].gates[gateLabel]
            target_gate = self.gatesets['target'].gates[gateLabel]
            return _plotting.polar_eigenval_plot(gate, target_gate, 
                                                 title=gateLabel, save_to="")

        def fn_validate(key):
            if not self._bEssentialResultsSet: return []            
            keys = ["bestEstimatePolar%sEvalPlot" % gl 
                    for gl in self.gatesets['final estimate'].gates ]
            if key == expr6: return keys # all computable keys
            elif key in keys: return [key]
            else: return []
        fns[expr6] = (fn, fn_validate)


        expr7 = "pauliProdHamiltonianDecompBoxes(.+)"
        def fn(key, confidenceLevel, vb):
            #cri = self._get_confidence_region(confidenceLevel)
            noConfidenceLevelDependence(confidenceLevel)
            gateLabel = _re.match(expr7,key).group(1)
            gate = self.gatesets['final estimate'].gates[gateLabel]
            target_gate = self.gatesets['target'].gates[gateLabel]
            basisNm   = self.gatesets['final estimate'].get_basis_name()
            assert(basisNm == self.gatesets['target'].get_basis_name())
            return _plotting.pauliprod_hamiltonian_boxplot(
                gate, target_gate, save_to="", mxBasis=basisNm, boxLabels=True)

        def fn_validate(key):
            if not self._bEssentialResultsSet: return []            
            keys = ["pauliProdHamiltonianDecompBoxes%s" % gl 
                    for gl in self.gatesets['final estimate'].gates ]
            if key == expr7: return keys # all computable keys
            elif key in keys: return [key]
            else: return []
        fns[expr7] = (fn, fn_validate)

        return fns


    def _get_special_fns(self):
        """ 
        Return a dictionary of functions which create "special objects"
        identified by the dictionary key.  These functions are used for
        the lazy creation of these objects within the "_specials" member
        of a Results instance.
        """

        def validate_essential(key):
            return [key] if self._bEssentialResultsSet else []
        def validate_LsAndGerms(key):
            return [key] if (self._bEssentialResultsSet and
                             self._LsAndGermInfoSet) else []

        def noConfidenceLevelDependence(level):
            """ Designates a "special" as independent of the confidence level"""
            if level is not None: raise _ResultCache.NoCRDependenceError

        fns = _collections.OrderedDict()


        def fn(key, confidenceLevel, vb):
            noConfidenceLevelDependence(confidenceLevel)

            gsTarget = self.gatesets['target']
            gsBestEstimate = self.gatesets['final estimate']
            constrainToTP = self.parameters['constrainToTP']

            if vb > 0: 
                print "Performing gauge transforms for appendix..."
                _sys.stdout.flush()

            best_gs_gauges = _collections.OrderedDict()

            best_gs_gauges['Target'] = _optimizeGauge(
                gsBestEstimate, "target", targetGateset=gsTarget,
                constrainToTP=constrainToTP, gateWeight=1.0,
                spamWeight=1.0, verbosity=vb)

            best_gs_gauges['TargetSpam'] = _optimizeGauge(
                gsBestEstimate, "target", targetGateset=gsTarget,
                verbosity=vb, gateWeight=0.01, spamWeight=0.99,
                constrainToTP=constrainToTP)

            best_gs_gauges['TargetGates'] = _optimizeGauge(
                gsBestEstimate, "target", targetGateset=gsTarget,
                verbosity=vb, gateWeight=0.99, spamWeight=0.01,
                constrainToTP=constrainToTP)

            best_gs_gauges['CPTP'] = _optimizeGauge(
                gsBestEstimate, "CPTP and target", 
                targetGateset=gsTarget, verbosity=vb,
                targetFactor=1.0e-7, constrainToTP=constrainToTP)

            if constrainToTP:
                best_gs_gauges['TP'] = best_gs_gauges['Target'].copy()
                  #assume best_gs is already in TP, so just optimize to
                  # target (done above)
            else:
                best_gs_gauges['TP'] = _optimizeGauge(
                    gsBestEstimate, "TP and target",
                    targetGateset=gsTarget, targetFactor=1.0e-7)
            return best_gs_gauges
        fns['gaugeOptAppendixGatesets'] = (fn, validate_essential)


        def fn(key, confidenceLevel, vb):            
            noConfidenceLevelDependence(confidenceLevel)

            best_gs_gauges = self._specials.get('gaugeOptAppendixGatesets',
                                                verbosity=vb)
            gsTarget = self.gatesets['target']

            ret = {}

            for gaugeKey,gopt_gs in best_gs_gauges.iteritems():
                #FUTURE: add confidence region support to these appendices? 
                # -- would need to compute confidenceRegionInfo (cri)
                #    for each gauge-optimized gateset, gopt_gs and pass
                #    to appropriate functions below
                ret['best%sGatesetSpamTable' % gaugeKey] = \
                    _generation.get_gateset_spam_table(gopt_gs)
                ret['best%sGatesetSpamParametersTable' % gaugeKey] = \
                    _generation.get_gateset_spam_parameters_table(gopt_gs)
                ret['best%sGatesetGatesTable' % gaugeKey] = \
                    _generation.get_gateset_gates_table(gopt_gs)
                ret['best%sGatesetChoiTable' % gaugeKey] = \
                    _generation.get_gateset_choi_table(gopt_gs)
                ret['best%sGatesetDecompTable' % gaugeKey] = \
                    _generation.get_gateset_decomp_table(gopt_gs)
                ret['best%sGatesetRotnAxisTable' % gaugeKey] = \
                    _generation.get_gateset_rotn_axis_table(gopt_gs)
                ret['best%sGatesetClosestUnitaryTable' % gaugeKey] = \
                    _generation.get_gateset_closest_unitary_table(gopt_gs)
                ret['best%sGatesetVsTargetTable' % gaugeKey] = \
                    _generation.get_gates_vs_target_table(
                    gopt_gs, gsTarget, None)
                ret['best%sGatesetErrorGenTable' % gaugeKey] = \
                    _generation.get_gates_vs_target_err_gen_table(
                    gopt_gs, gsTarget)

            return ret
        fns['gaugeOptAppendixTables'] = (fn, validate_essential)


        def fn(key, confidenceLevel, vb):
            noConfidenceLevelDependence(confidenceLevel)
            ret = {}

            for gaugeKey in ('Target','TargetSpam','TargetGates','CPTP','TP'):
                ret['best%sGatesetSpamTable' % gaugeKey] = \
                    _generation.get_blank_table()
                ret['best%sGatesetSpamParametersTable' % gaugeKey] = \
                    _generation.get_blank_table()
                ret['best%sGatesetGatesTable' % gaugeKey] = \
                    _generation.get_blank_table()
                ret['best%sGatesetChoiTable' % gaugeKey] = \
                    _generation.get_blank_table()
                ret['best%sGatesetDecompTable' % gaugeKey] = \
                    _generation.get_blank_table()
                ret['best%sGatesetRotnAxisTable' % gaugeKey] = \
                    _generation.get_blank_table()
                ret['best%sGatesetClosestUnitaryTable' % gaugeKey] = \
                    _generation.get_blank_table()
                ret['best%sGatesetVsTargetTable' % gaugeKey] = \
                    _generation.get_blank_table()
                ret['best%sGatesetErrorGenTable' % gaugeKey] = \
                    _generation.get_blank_table()

            return ret
        fns['blankGaugeOptAppendixTables'] = (fn, validate_essential)


        def fn(key, confidenceLevel, vb):
            noConfidenceLevelDependence(confidenceLevel)

            direct_specs = _ssc.build_spam_specs(
                prepStrs=self.gatestring_lists['prep fiducials'],
                effectStrs=self.gatestring_lists['effect fiducials'],
                prep_labels=self.gatesets['target'].get_prep_labels(),
                effect_labels=self.gatesets['target'].get_effect_labels() )

            baseStrs = [] # (L,germ) base strings without duplicates
            fullDict = self.parameters['L,germ tuple base string dict']
            for L in self.parameters['max length list']:
                for germ in self.gatestring_lists['germs']:
                    if fullDict[(L,germ)] not in baseStrs:
                        baseStrs.append( fullDict[(L,germ)] )

            return _plotting.direct_lgst_gatesets( 
                baseStrs, self.dataset, direct_specs, self.gatesets['target'],
                svdTruncateTo=4, verbosity=0) 
                #TODO: svdTruncateTo set elegantly?
        fns["direct_lgst_gatesets"] = (fn, validate_LsAndGerms)

        
        def fn(key, confidenceLevel, vb):
            noConfidenceLevelDependence(confidenceLevel)

            gsTarget = self.gatesets['target']
            direct_specs = _ssc.build_spam_specs(
                prepStrs=self.gatestring_lists['prep fiducials'],
                effectStrs=self.gatestring_lists['effect fiducials'],
                prep_labels=gsTarget.get_prep_labels(),
                effect_labels=gsTarget.get_effect_labels() )

            baseStrs = [] # (L,germ) base strings without duplicates
            fullDict = self.parameters['L,germ tuple base string dict']
            for L in self.parameters['max length list']:
                for germ in self.gatestring_lists['germs']:
                    if fullDict[(L,germ)] not in baseStrs:
                        baseStrs.append( fullDict[(L,germ)] )

            if self.parameters['objective'] == "chi2":
                mpc = self.parameters['minProbClipForWeighting']
                return _plotting.direct_mc2gst_gatesets(
                    baseStrs, self.dataset, direct_specs, gsTarget,
                    svdTruncateTo=gsTarget.get_dimension(), 
                    minProbClipForWeighting=mpc,
                    probClipInterval=self.parameters['probClipInterval'],
                    verbosity=0)

            elif self.parameters['objective'] == "logl":
                mpc = self.parameters['minProbClip']
                return _plotting.direct_mlgst_gatesets(
                    baseStrs, self.dataset, direct_specs, gsTarget,
                    svdTruncateTo=gsTarget.get_dimension(),
                    minProbClip=mpc,
                    probClipInterval=self.parameters['probClipInterval'],
                    verbosity=0)
            else:
                raise ValueError("Invalid Objective: %s" % 
                                 self.parameters['objective'])
        fns["DirectLongSeqGatesets"] = (fn, validate_LsAndGerms)


        def fn(key, confidenceLevel, vb):
            noConfidenceLevelDependence(confidenceLevel)

            baseStr_dict = self._getBaseStrDict()
            strs  = (self.gatestring_lists['prep fiducials'], 
                     self.gatestring_lists['effect fiducials'])
            germs = self.gatestring_lists['germs']
            gsBest = self.gatesets['final estimate']
            fidPairs = self.parameters['fiducial pairs']
            Ls = self.parameters['max length list']
            st = 1 if Ls[0] == 0 else 0 #start index: skip LGST column in plots

            obj = self.parameters['objective']
            assert(obj in ("chi2","logl"))
            if obj == "chi2":
                plotFn = _plotting.chi2_boxplot
                mpc = self.parameters['minProbClipForWeighting']
                
            elif obj == "logl":
                plotFn = _plotting.logl_boxplot
                mpc = self.parameters['minProbClip']

            maxH = 9.0 # max inches for graphic
            minboxH = 0.075 #min height per box
            germH = (len(self.gatestring_lists['effect fiducials'])+1)*minboxH # + 1 for space
            maxGermsPerFig = max(int(maxH / germH - 2), 1)
            figs = []; n = 0
            while( n < len(germs) ):
                fig_germs = list(reversed(germs[n:n+maxGermsPerFig]))
                fig = plotFn(Ls[st:], fig_germs, baseStr_dict,
                             self.dataset, gsBest, strs,
                             r"$L$", "germ", scale=1.0, sumUp=False,
                             histogram=True, title="", fidPairs=fidPairs,
                             linlg_pcntle=float(self.parameters['linlogPercentile']) / 100,
                             minProbClipForWeighting=mpc, save_to="", ticSize=20)
                figs.append(fig); n += maxGermsPerFig

            return figs

        fns["bestEstimateColorBoxPlotPages"] = (fn,validate_LsAndGerms)


        return fns


#    def set_additional_info(self,minProbClip=1e-6, minProbClipForWeighting=1e-4,
#                          probClipInterval=(-1e6,1e6), radius=1e-4,
#                          weightsDict=None, defaultDirectory=None, defaultBasename=None,
#                          mxBasis="gm"):
#        """
#        Set advanced parameters for producing derived outputs.  Usually the default
#        values are fine (which is why setting these inputs is separated into a
#        separate function).
#
#        Parameters
#        ----------
#        minProbClip : float, optional
#            The minimum probability treated normally in the evaluation of the log-likelihood.
#            A penalty function replaces the true log-likelihood for probabilities that lie
#            below this threshold so that the log-likelihood never becomes undefined (which improves
#            optimizer performance).
#    
#        minProbClipForWeighting : float, optional
#            Sets the minimum and maximum probability p allowed in the chi^2 weights: N/(p*(1-p))
#            by clipping probability p values to lie within the interval
#            [ minProbClipForWeighting, 1-minProbClipForWeighting ].
#    
#        probClipInterval : 2-tuple or None, optional
#           (min,max) values used to clip the probabilities predicted by gatesets during the
#           least squares search for an optimal gateset (if not None).
#
#        radius : float, optional
#           Specifies the severity of rounding used to "patch" the zero-frequency
#           terms of the log-likelihood.
#
#        weightsDict : dict, optional
#           A dictionary with keys == gate strings and values == multiplicative scaling 
#           factor for the corresponding gate string. The default is no weight scaling at all.
#           
#        defaultDirectory : string, optional
#           Path to the default directory for generated reports and presentations.
#
#        defaultBasename : string, optional
#           Default basename for generated reports and presentations.
#
#        mxBasis : {"std", "gm", "pp"}, optional
#           The basis used to interpret the gate matrices.
#
#
#        Returns
#        -------
#        None
#        """
#
#
#        self.additionalInfo = { 'weights': weightsDict, 
#                                'minProbClip': minProbClip, 
#                                'minProbClipForWeighting': minProbClipForWeighting,
#                                'probClipInterval': probClipInterval,
#                                'radius': radius,
#                                'hessianProjection': 'std',
#                                'defaultDirectory': defaultDirectory,
#                                'defaultBasename': defaultBasename,
#                                'mxBasis': "gm"}
#
#    def set_template_path(self, pathToTemplates):
#        """
#        Sets the location of GST report and presentation templates.
#
#        Parameters
#        ----------
#        pathToTemplates : string
#           The path to a folder containing GST's template files.  
#           Usually this can be determined automatically (the default).
#        """
#        self.options.template_path = pathToTemplates
#
#
#    def set_latex_cmd(self, latexCmd):
#        """
#        Sets the shell command used for compiling latex reports and
#        presentations.
#
#        Parameters
#        ----------
#        latexCmd : string
#           The command to run to invoke the latex compiler,
#           typically just 'pdflatex' when it is on the system
#           path. 
#        """
#        self.latexCmd = latexCmd


    def _get_confidence_region(self, confidenceLevel):
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
        
        assert(self._bEssentialResultsSet)

        if confidenceLevel is None:
            return None

        if confidenceLevel not in self._confidence_regions:

            #Negative confidence levels ==> non-Markovian error bars
            if confidenceLevel < 0:
                confidenceLevel = -confidenceLevel
                regionType = "non-markovian"
            else:
                regionType = "std"

            if self.parameters['objective'] == "logl":
                cr = _generation.get_logl_confidence_region(
                    self.gatesets['final estimate'], self.dataset,
                    confidenceLevel,
                    self.gatestring_lists['final'],
                    self.parameters['probClipInterval'],
                    self.parameters['minProbClip'],
                    self.parameters['radius'],
                    self.parameters['hessianProjection'],
                    regionType, self._comm,
                    self.parameters['memLimit'])
            elif self.parameters['objective'] == "chi2":
                cr = _generation.get_chi2_confidence_region(
                    self.gatesets['final estimate'], self.dataset,
                    confidenceLevel,
                    self.gatestring_lists['final'],
                    self.parameters['probClipInterval'],
                    self.parameters['minProbClipForWeighting'],
                    self.parameters['hessianProjection'],
                    regionType, self._comm,
                    self.parameters['memLimit'])
            else:
                raise ValueError("Invalid objective given in essential" +
                                 " info: %s" % self.parameters['objective'])
            self._confidence_regions[confidenceLevel] = cr

        return self._confidence_regions[confidenceLevel]


    def _merge_template(self, qtys, templateFilename, outputFilename):
        if self.options.template_path is None:
            templateFilename = _os.path.join( _os.path.dirname(_os.path.abspath(__file__)), 
                                              "templates", templateFilename )
        else:
            templateFilename = _os.path.join( self.options.template_path,
                                              templateFilename )
            
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

        assert(self._bEssentialResultsSet)
        assert(self._LsAndGermInfoSet)

        baseStr_dict = _collections.OrderedDict()
        st = 1 if self.parameters['max length list'][0] == 0 else 0
          #start index: skips LGST column in report color box plots

        tmpRunningList = []
        fullDict = self.parameters['L,germ tuple base string dict']
        for L in self.parameters['max length list'][st:]:
            for germ in self.gatestring_lists['germs']:
                if remove_dups and fullDict[(L,germ)] in tmpRunningList:
                    baseStr_dict[(L,germ)] = None
                else: 
                    tmpRunningList.append( fullDict[(L,germ)] )
                    baseStr_dict[(L,germ)] = fullDict[(L,germ)]
        return baseStr_dict


    

    def create_full_report_pdf(self, confidenceLevel=None, filename="auto", 
                            title="auto", datasetLabel="auto", suffix="",
                            debugAidsAppendix=False, gaugeOptAppendix=False,
                            pixelPlotAppendix=False, whackamoleAppendix=False,
                            m=0, M=10, tips=False, verbosity=0, comm=None):
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

        tips : boolean, optional
           If True, additional markup and tooltips are included in the produced
           PDF which indicate how tables and figures in the report correspond
           to members of this Results object.  These "tips" can be useful if
           you want to further manipulate the data contained in a table or
           figure.

        verbosity : int, optional
           How much detail to send to stdout.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        Returns
        -------
        None
        """
        assert(self._bEssentialResultsSet)
        self.confidence_level = confidenceLevel 
        self._comm = comm
          #set "current" level, used by ResultCache member dictionaries

        if tips:
            def tooltiptex(directive):
                return '\\pdftooltip{{\\color{blue}\\texttt{%s}}\\quad}' \
                    % directive + '{Access this information in pyGSTi via' \
                    + '<ResultsObject>%s}' % directive


        else:
            def tooltiptex(directive):
                return "" #tooltips disabled


        #Get report output filename
        default_dir = self.parameters['defaultDirectory']
        default_base = self.parameters['defaultBasename']

        if filename != "auto":
            report_dir = _os.path.dirname(filename)
            report_base = _os.path.splitext( _os.path.basename(filename) )[0] \
                           + suffix
        else:
            cwd = _os.getcwd()
            report_dir  = default_dir  if (default_dir  is not None) else cwd
            report_base = default_base if (default_base is not None) \
                                       else "GSTReport"
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
        #  FUTURE?? determine what we need to compute & plot by reading
        #           through the template file?
        
        #Note: for now, we assume the best gateset corresponds to the last
        #      L-value
        best_gs = self.gatesets['final estimate']
        v = verbosity # shorthand

        if not self._LsAndGermInfoSet: #cannot create appendices 
            debugAidsAppendix = False  # which depend on this structure
            pixelPlotAppendix = False
            whackamoleAppendix = False
        
        qtys = {} # dictionary to store all latex strings
                  # to be inserted into report template
        qtys['title'] = title   
        qtys['datasetLabel'] = datasetLabel
        qtys['settoggles'] =  "\\toggle%s{confidences}\n" % \
            ("false" if confidenceLevel is None else "true")
        qtys['settoggles'] += "\\toggle%s{LsAndGermsSet}\n" % \
            ("true" if self._LsAndGermInfoSet else "false")
        qtys['settoggles'] += "\\toggle%s{debuggingaidsappendix}\n" % \
            ("true" if debugAidsAppendix else "false")
        qtys['settoggles'] += "\\toggle%s{gaugeoptappendix}\n" % \
            ("true" if gaugeOptAppendix else "false")
        qtys['settoggles'] += "\\toggle%s{pixelplotsappendix}\n" % \
            ("true" if pixelPlotAppendix else "false")
        qtys['settoggles'] += "\\toggle%s{whackamoleappendix}\n" % \
            ("true" if whackamoleAppendix else "false")
        qtys['confidenceLevel'] = "%g" % \
            confidenceLevel if confidenceLevel is not None else "NOT-SET"
        qtys['linlg_pcntle'] = self.parameters['linlogPercentile']

        if confidenceLevel is not None:
            cri = self._get_confidence_region(confidenceLevel)
            qtys['confidenceIntervalScaleFctr'] = \
                "%.3g" % cri.intervalScaling
            qtys['confidenceIntervalNumNonGaugeParams'] = \
                "%d" % cri.nNonGaugeParams
        else:
            cri = None
            qtys['confidenceIntervalScaleFctr'] = "NOT-SET"
            qtys['confidenceIntervalNumNonGaugeParams'] = "NOT-SET"

        pdfInfo = [('Author','pyGSTi'), ('Title', title),
                   ('Keywords', 'GST'), ('pyGSTi Version',_version.__version__),
                   ('opt_long_tables', self.options.long_tables),
                   ('opt_table_class', self.options.table_class),
                   ('opt_template_path', self.options.template_path),
                   ('opt_latex_cmd', self.options.latex_cmd) ]
                   #('opt_latex_postcmd', self.options.latex_postcmd) #TODO: add this
        for key,val in self.parameters.iteritems():
            pdfInfo.append( (key, val) )
        qtys['pdfinfo'] = _to_pdfinfo( pdfInfo )


        #Get figure directory for figure generation *and* as a 
        # scratch space for tables.
        D = report_base + "_files" #figure directory relative to reportDir
        if not _os.path.isdir( _os.path.join(report_dir,D)):
            _os.mkdir( _os.path.join(report_dir,D))

            
        # 1) get latex tables
        if verbosity > 0: 
            print "*** Generating tables ***"; _sys.stdout.flush()

        
        std_tables = \
            ('targetSpamTable','targetGatesTable','datasetOverviewTable',
             'bestGatesetSpamTable','bestGatesetSpamParametersTable',
             'bestGatesetGatesTable','bestGatesetChoiTable',
             'bestGatesetDecompTable','bestGatesetRotnAxisTable',
             'bestGatesetClosestUnitaryTable',
             'bestGatesetVsTargetTable','bestGatesetErrorGenTable')
        
        ls_and_germs_tables = ('fiducialListTable','prepStrListTable',
                               'effectStrListTable','germListTable',
                               'progressTable')

        tables_to_compute = std_tables
        tables_to_blank = []

        if self._LsAndGermInfoSet:
            tables_to_compute += ls_and_germs_tables
        else:
            tables_to_blank += ls_and_germs_tables

        for key in tables_to_compute:
            qtys[key] = self.tables.get(key, verbosity=v).render(
                'latex',longtables=self.options.long_tables, scratchDir=D)
            qtys["tt_"+key] = tooltiptex(".tables['%s']" % key)

        for key in tables_to_blank:
            qtys[key] = _generation.get_blank_table().render(
                'latex',longtables=self.options.long_tables)
            qtys["tt_"+key] = ""

        #get appendix tables if needed
        if gaugeOptAppendix: 
            goaTables = self._specials.get('gaugeOptAppendixTables',verbosity=v)
            qtys.update( { key : goaTables[key].render(
                        'latex', longtables=self.options.long_tables, scratchDir=D)
                           for key in goaTables }  )
            #TODO: tables[ref] and then tooltips?

        elif any((debugAidsAppendix, pixelPlotAppendix, whackamoleAppendix)):
            goaTables = self._specials.get('blankGaugeOptAppendixTables',
                              verbosity=v)   # fill keys with blank tables
            qtys.update( { key : goaTables[key].render(
                        'latex',longtables=self.options.long_tables)
                           for key in goaTables }  )  # for format substitution
            #TODO: tables[ref] and then tooltips?

    
        # 2) generate plots
        if verbosity > 0: 
            print "*** Generating plots ***"; _sys.stdout.flush()

        if _matplotlib.is_interactive():
            _matplotlib.pyplot.ioff()
            bWasInteractive = True
        else: bWasInteractive = False
    
        maxW,maxH = 6.5,9.0 #max width and height of graphic in latex document (in inches)

        def incgr(figFilenm,W=None,H=None): #includegraphics "macro"
            if W is None: W = maxW
            if H is None: H = maxH
            return "\\includegraphics[width=%.2fin,height=%.2fin" % (W,H) + \
                ",keepaspectratio]{%s/%s}" % (D,figFilenm)

        def set_fig_qtys(figkey, figFilenm, W=None,H=None):
            fig = self.figures.get(figkey, verbosity=v)
            fig.save_to(_os.path.join(report_dir, D, figFilenm))
            qtys[figkey] = incgr(figFilenm,W,H)
            qtys['tt_' + figkey] = tooltiptex(".figures['%s']" % figkey)
            return fig

        #Chi2 or logl plots
        if self._LsAndGermInfoSet:
            Ls = self.parameters['max length list']
            st = 1 if Ls[0] == 0 else 0 #start index: skip LGST column in plots
            nPlots = (len(Ls[st:])-1)+2 if pixelPlotAppendix else 2

            if self.parameters['objective'] == "chi2":
                plotFnName,plotFnLatex = "Chi2", "$\chi^2$"
            elif self.parameters['objective'] == "logl":
                plotFnName,plotFnLatex = "LogL", "$\\log(\\mathcal{L})$"
            else: 
                raise ValueError("Invalid objective value: %s" 
                                 % self.parameters['objective'])
            
            if verbosity > 0: 
                print " -- %s plots (%d): " % (plotFnName, nPlots),
                _sys.stdout.flush()

            if verbosity > 0:
                print "1 ",; _sys.stdout.flush()
            fig = set_fig_qtys("bestEstimateColorBoxPlot",
                               "best%sBoxes.pdf" % plotFnName)
            maxX = fig.get_extra_info()['nUsedXs']
            maxY = fig.get_extra_info()['nUsedYs']

            #qtys["bestEstimateColorBoxPlot_hist"] = \
            #    incgr("best%sBoxes_hist.pdf" % plotFnName figFilenm)
            #    #no tooltip for histogram... - probably should make it 
            #    # it's own element of .figures dict

            if verbosity > 0: 
                print "2 ",; _sys.stdout.flush()
            fig = set_fig_qtys("invertedBestEstimateColorBoxPlot",
                               "best%sBoxes_inverted.pdf" % plotFnName)
        else:
            for figkey in ["bestEstimateColorBoxPlot",
                           "invertedBestEstimateColorBoxPlot"]:
                qtys[figkey] = qtys["tt_"+figkey] = ""

    
        pixplots = ""
        if pixelPlotAppendix:
            Ls = self.parameters['max length list']
            for i in range(st,len(Ls)-1):

                if verbosity > 0: 
                    print "%d " % (i-st+3),; _sys.stdout.flush()
                fig = self.figures.get("estimateForLIndex%dColorBoxPlot" % i,
                                       verbosity=v)
                fig.save_to( _os.path.join(report_dir, D,
                                           "L%d_%sBoxes.pdf" % (i,plotFnName)))
                lx = fig.get_extra_info()['nUsedXs']
                ly = fig.get_extra_info()['nUsedYs']

                #scale figure size according to number of rows and columns+1
                # (+1 for labels ~ another col) relative to initial plot
                W = float(lx+1)/float(maxX+1) * maxW 
                H = float(ly)  /float(maxY)   * maxH 
            
                pixplots += "\n"
                pixplots += "\\begin{figure}\n"
                pixplots += "\\begin{center}\n"
                pixplots += "\\includegraphics[width=%.2fin,height=%.2fin," \
                    % (W,H) + "keepaspectratio]{%s/L%d_%sBoxes.pdf}\n" \
                    %(D,i,plotFnName)
                pixplots += \
                    "\\caption{Box plot of iteration %d (L=%d) " % (i,Ls[i]) \
                    + "gateset %s values.\label{L%dGateset%sBoxPlot}}\n" \
                    % (plotFnLatex,i,plotFnName)
                #TODO: add conditional tooltip string to start of caption
                pixplots += "\\end{center}\n"
                pixplots += "\\end{figure}\n"

        #Set template quantity (empty string if appendix disabled)
        qtys['intermediate_pixel_plot_figures'] = pixplots

        if verbosity > 0: 
            print ""; _sys.stdout.flush()
        
        if debugAidsAppendix:
            #DirectLGST and deviation
            if verbosity > 0: 
                print " -- Direct-X plots ",; _sys.stdout.flush()
                print "(2):"; _sys.stdout.flush()    

            #if verbosity > 0: 
            #    print " ?",; _sys.stdout.flush()
            #fig = set_fig_qtys("directLGSTColorBoxPlot",
            #                   "directLGST%sBoxes.pdf" % plotFnName)

            if verbosity > 0: 
                print " 1",; _sys.stdout.flush()        
            fig = set_fig_qtys("directLongSeqGSTColorBoxPlot",
                           "directLongSeqGST%sBoxes.pdf" % plotFnName)

            #if verbosity > 0: 
            #    print " ?",; _sys.stdout.flush()        
            #fig = set_fig_qtys("directLGSTDeviationColorBoxPlot",
            #                   "directLGSTDeviationBoxes.pdf",W=4,H=5)

            if verbosity > 0: 
                print " 2",; _sys.stdout.flush()
            fig = set_fig_qtys("directLongSeqGSTDeviationColorBoxPlot",
                               "directLongSeqGSTDeviationBoxes.pdf",W=4,H=5)

            if verbosity > 0: 
                print ""; _sys.stdout.flush()


            #Small eigenvalue error rate
            if verbosity > 0: 
                print " -- Error rate plots..."; _sys.stdout.flush()
            fig = set_fig_qtys("smallEigvalErrRateColorBoxPlot",
                               "smallEigvalErrRateBoxes.pdf",W=4,H=5)
        else:
            #UNUSED: "directLGSTColorBoxPlot", "directLGSTDeviationColorBoxPlot"
            for figkey in ["directLongSeqGSTColorBoxPlot",
                           "directLongSeqGSTDeviationColorBoxPlot",
                           "smallEigvalErrRateColorBoxPlot"]:
                qtys[figkey] = qtys["tt_"+figkey] = ""


        whackamoleplots = ""
        if whackamoleAppendix:    
            #Whack-a-mole plots for highest L of each length-1 germ
            Ls = self.parameters['max length list']
            highestL = Ls[-1]; allGateStrings = self.gatestring_lists['all']
            hammerWeight = 10.0
            len1Germs = [ g for g in self.gatestring_lists['germs'] 
                          if len(g) == 1 ]

            if verbosity > 0: 
                print " -- Whack-a-mole plots (%d): " % (2*len(len1Germs)),
                _sys.stdout.flush()

            for i,germ in enumerate(len1Germs):
                if verbosity > 0: 
                    print "%d " % (i+1),; _sys.stdout.flush()

                fig = self.figures.get("whack%sMoleBoxes" % germ[0],verbosity=v)
                fig.save_to(_os.path.join(report_dir, D,"whack%sMoleBoxes.pdf"
                                          % germ[0]))
        
                whackamoleplots += "\n"
                whackamoleplots += "\\begin{figure}\n"
                whackamoleplots += "\\begin{center}\n"
                whackamoleplots += "\\includegraphics[width=%.2fin,height=%.2fin,keepaspectratio]{%s/whack%sMoleBoxes.pdf}\n" % (maxW,maxH,D,germ[0])
                whackamoleplots += "\\caption{Whack-a-%s-mole box plot for $\mathrm{%s}^{%d}$." % (plotFnLatex,germ[0],highestL)
                #TODO: add conditional tooltip string to start of caption
                whackamoleplots += "  Hitting with hammer of weight %.1f.\label{Whack%sMoleBoxPlot}}\n" % (hammerWeight,germ[0])
                whackamoleplots += "\\end{center}\n"
                whackamoleplots += "\\end{figure}\n"
        
            for i,germ in enumerate(len1Germs):
                if verbosity > 0: 
                    print "%d " % (len(len1Germs)+i+1),; _sys.stdout.flush()

                fig = self.figures.get("whack%sMoleBoxesSummed" % germ[0],
                                       verbosity=v)
                fig.save_to(_os.path.join(report_dir, D,"whack%sMoleBoxesSummed.pdf" % germ[0]))
    
                whackamoleplots += "\n"
                whackamoleplots += "\\begin{figure}\n"
                whackamoleplots += "\\begin{center}\n"
                whackamoleplots += "\\includegraphics[width=4in,height=5in,keepaspectratio]{%s/whack%sMoleBoxesSummed.pdf}\n" % (D,germ[0])
                whackamoleplots += "\\caption{Whack-a-%s-mole box plot for $\mathrm{%s}^{%d}$, summed over fiducial matrix." % (plotFnLatex,germ[0],highestL)
                #TODO: add conditional tooltip string to start of caption
                whackamoleplots += "  Hitting with hammer of weight %.1f.\label{Whack%sMoleBoxPlotSummed}}\n" % (hammerWeight,germ[0])
                whackamoleplots += "\\end{center}\n"
                whackamoleplots += "\\end{figure}\n"
    
            if verbosity > 0: 
                print ""; _sys.stdout.flush()

        #Set template quantity (empty string if appendix disabled)
        qtys['whackamole_plot_figures'] = whackamoleplots
            
        if bWasInteractive:
            _matplotlib.pyplot.ion()
    

        # 3) populate template latex file => report latex file
        if verbosity > 0: 
            print "*** Merging into template file ***"; _sys.stdout.flush()
        
        mainTexFilename = _os.path.join(report_dir, report_base + ".tex")
        appendicesTexFilename = _os.path.join(report_dir, report_base + "_appendices.tex")
        pdfFilename = _os.path.join(report_dir, report_base + ".pdf")
    
        if self.parameters['objective'] == "chi2":    
            mainTemplate = "report_chi2_main.tex"
            appendicesTemplate = "report_chi2_appendices.tex"
        elif self.parameters['objective'] == "logl":
            mainTemplate = "report_logL_main.tex"
            appendicesTemplate = "report_logL_appendices.tex"
        else: 
            raise ValueError("Invalid objective value: %s" 
                             % self.parameters['objective'])
    
        if any( (debugAidsAppendix, gaugeOptAppendix,
                 pixelPlotAppendix, whackamoleAppendix) ):
            qtys['appendices'] = "\\input{%s}" % \
                _os.path.basename(appendicesTexFilename)
            self._merge_template(qtys, appendicesTemplate,
                                 appendicesTexFilename)
        else: qtys['appendices'] = ""
        self._merge_template(qtys, mainTemplate, mainTexFilename)
    
    
        # 4) compile report latex file into PDF
        if verbosity > 0: 
            print "Latex file(s) successfully generated.  Attempting to compile with pdflatex..."; _sys.stdout.flush()
        cwd = _os.getcwd()
        if len(report_dir) > 0:  
            _os.chdir(report_dir)
    
        try:
            ret = _os.system( "%s %s %s" % 
                              (self.options.latex_cmd,
                               _os.path.basename(mainTexFilename),
                               self.options.latex_postcmd) )
            if ret == 0:
                #We could check if the log file contains "Rerun" in it, 
                # but we'll just re-run all the time now
                if verbosity > 0: 
                    print "Initial output PDF %s successfully generated." \
                        % pdfFilename

                ret = _os.system( "%s %s %s" % 
                                  (self.options.latex_cmd,
                                   _os.path.basename(mainTexFilename),
                                   self.options.latex_postcmd) )
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
            print "Error trying to run pdflatex to generate output PDF %s. Is '%s' on your path?" % (pdfFilename,self.options.latex_cmd)
        finally: 
            _os.chdir(cwd)
    
        return


    def create_brief_report_pdf(self, confidenceLevel=None, 
                           filename="auto", title="auto", datasetLabel="auto",
                           suffix="", m=0, M=10, tips=False, verbosity=0,
                           comm=None):
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

        tips : boolean, optional
           If True, additional markup and tooltips are included in the produced
           PDF which indicate how tables and figures in the report correspond
           to members of this Results object.  These "tips" can be useful if
           you want to further manipulate the data contained in a table or
           figure.

        verbosity : int, optional
           How much detail to send to stdout.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        Returns
        -------
        None
        """
        assert(self._bEssentialResultsSet)
        self.confidence_level = confidenceLevel 
        self._comm = comm
        v = verbosity # shorthand

        if tips:
            def tooltiptex(directive):
                return '\\pdftooltip{{\\color{blue}\\texttt{%s}}\\quad}' \
                    % directive + '{Access this information in pyGSTi via' \
                    + '<ResultsObject>%s}' % directive
        else:
            def tooltiptex(directive):
                return "" #tooltips disabled


        #Get report output filename
        default_dir  = self.parameters['defaultDirectory']
        default_base = self.parameters['defaultBasename']

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
        
                    
        if self._LsAndGermInfoSet:
            baseStr_dict = self._getBaseStrDict()
            Ls = self.parameters['max length list']
            st = 1 if Ls[0] == 0 else 0 #start index: skip LGST column in plots
            goodnessOfFitSection = True
        else:
            goodnessOfFitSection = False
    
        #Note: for now, we assume the best gateset corresponds to the last L-value
        best_gs = self.gatesets['final estimate']
        obj = self.parameters['objective']
        
        qtys = {} # dictionary to store all latex strings to be inserted into report template
        qtys['title'] = title
        qtys['datasetLabel'] = datasetLabel
        qtys['settoggles'] =  "\\toggle%s{confidences}\n" % \
            ("false" if confidenceLevel is None else "true")
        qtys['settoggles'] += "\\toggle%s{goodnessSection}\n" % \
            ("true" if goodnessOfFitSection else "false")
        qtys['confidenceLevel'] = "%g" % confidenceLevel \
            if confidenceLevel is not None else "NOT-SET"
        qtys['objective'] = "$\\log{\\mathcal{L}}$" \
            if obj == "logl" else "$\\chi^2$"
        qtys['gofObjective'] = "$2\\Delta\\log{\\mathcal{L}}$" \
            if obj == "logl" else "$\\chi^2$"

        if confidenceLevel is not None:
            cri = self._get_confidence_region(confidenceLevel)
            qtys['confidenceIntervalScaleFctr'] = "%.3g" % cri.intervalScaling
            qtys['confidenceIntervalNumNonGaugeParams'] = "%d" % cri.nNonGaugeParams
        else:
            cri = None
            qtys['confidenceIntervalScaleFctr'] = "NOT-SET"
            qtys['confidenceIntervalNumNonGaugeParams'] = "NOT-SET"

        pdfInfo = [('Author','pyGSTi'), ('Title', title),
                   ('Keywords', 'GST'), ('pyGSTi Version',_version.__version__),
                   ('opt_long_tables', self.options.long_tables),
                   ('opt_table_class', self.options.table_class),
                   ('opt_template_path', self.options.template_path),
                   ('opt_latex_cmd', self.options.latex_cmd) ]
        for key,val in self.parameters.iteritems():
            pdfInfo.append( (key, val) )
        qtys['pdfinfo'] = _to_pdfinfo( pdfInfo )

        #Get figure directory for figure generation *and* as a 
        # scratch space for tables.
        D = report_base + "_files" #figure directory relative to reportDir
        if not _os.path.isdir( _os.path.join(report_dir,D)):
            _os.mkdir( _os.path.join(report_dir,D))
            
        # 1) get latex tables
        if verbosity > 0: 
            print "*** Generating tables ***"; _sys.stdout.flush()

        std_tables = ('bestGatesetSpamTable',
                      'bestGatesetSpamParametersTable',
                      'bestGatesetGatesTable',
                      'bestGatesetDecompTable','bestGatesetRotnAxisTable',
                      'bestGatesetVsTargetTable',
                      'bestGatesetErrorGenTable')
        gof_tables = ('progressTable',)

        tables_to_compute = std_tables
        tables_to_blank = []

        if goodnessOfFitSection:
            tables_to_compute += gof_tables
        else:
            tables_to_blank += gof_tables

        for key in tables_to_compute:
            qtys[key] = self.tables.get(key, verbosity=v).render(
                'latex',longtables=self.options.long_tables, scratchDir=D)
            qtys["tt_"+key] = tooltiptex(".tables['%s']" % key)

        for key in tables_to_blank:
            qtys[key] = _generation.get_blank_table().render(
                'latex',longtables=self.options.long_tables)
            qtys["tt_"+key] = ""

    
        # 2) generate plots
        if verbosity > 0: 
            print "*** Generating plots ***"; _sys.stdout.flush()

        if _matplotlib.is_interactive():
            _matplotlib.pyplot.ioff()
            bWasInteractive = True
        else: bWasInteractive = False

        #if goodnessOfFitSection:    
        #    strs = ( self.gatestring_lists['prep fiducials'], 
        #             self.gatestring_lists['effect fiducials'] )
        #    D = report_base + "_files" #figure directory relative to reportDir
        #    if not _os.path.isdir( _os.path.join(report_dir,D)):
        #        _os.mkdir( _os.path.join(report_dir,D))
        #
        #    #Chi2 or logl plot
        #    nPlots = 1
        #    if self.parameters['objective'] == "chi2":
        #        plotFnName,plotFnLatex = "Chi2", "$\chi^2$"
        #    elif self.parameters['objective'] == "logl":
        #        plotFnName,plotFnLatex = "LogL", "$\\log(\\mathcal{L})$"
        #    else: 
        #        raise ValueError("Invalid objective value: %s" 
        #                         % self.parameters['objective'])
        #
        #    if verbosity > 0: 
        #        print " -- %s plots (%d): " % (plotFnName, nPlots),; _sys.stdout.flush()
        #
        #    if verbosity > 0: 
        #        print "1 ",; _sys.stdout.flush()
        #    figkey = 'bestEstimateColorBoxPlot'
        #    figFilenm = "best%sBoxes.pdf" % plotFnName
        #    fig = self.figures.get(figkey, verbosity=v)
        #    fig.save_to(_os.path.join(report_dir, D, figFilenm))
        #    maxX = fig.get_extra_info()['nUsedXs']; maxY = fig.get_extra_info()['nUsedYs']
        #    maxW,maxH = 6.5,9.0 #max width and height of graphic in latex document (in inches)
        #
        #    if verbosity > 0: 
        #        print ""; _sys.stdout.flush()    
        #
        #    qtys[figkey]  = "\\includegraphics[width=%.2fin,height=%.2fin,keepaspectratio]{%s/%s}" % (maxW,maxH,D,figFilenm)
        #    qtys['tt_'+ figkey]  = tooltiptex(".figures['%s']" % figkey)
        
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
            ret = _os.system( "%s %s %s" %
                              (self.options.latex_cmd,
                               _os.path.basename(mainTexFilename),
                               self.options.latex_postcmd) )
            if ret == 0:
                #We could check if the log file contains "Rerun" in it, 
                # but we'll just re-run all the time now
                if verbosity > 0: 
                    print "Initial output PDF %s successfully generated." % \
                        pdfFilename
                ret = _os.system( "%s %s %s" % 
                                  (self.options.latex_cmd,
                                   _os.path.basename(mainTexFilename),
                                   self.options.latex_postcmd) )
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
            print "Error trying to run pdflatex to generate output PDF %s. Is '%s' on your path?" % (pdfFilename,self.options.latex_cmd)
        finally: 
            _os.chdir(cwd)

        return


    def create_presentation_pdf(self, confidenceLevel=None, filename="auto", 
                              title="auto", datasetLabel="auto", suffix="",
                              debugAidsAppendix=False, 
                              pixelPlotAppendix=False, whackamoleAppendix=False,
                              m=0, M=10, verbosity=0, comm=None):
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

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.


        Returns
        -------
        None
        """
        assert(self._bEssentialResultsSet)
        self.confidence_level = confidenceLevel 
        self._comm = comm
        v = verbosity # shorthand

        #Currently, no tooltip option for presentations
        def tooltiptex(directive):
            return "" #tooltips disabled

        #Get report output filename
        default_dir  = self.parameters['defaultDirectory']
        default_base = self.parameters['defaultBasename']

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
        best_gs = self.gatesets['final estimate']

        if not self._LsAndGermInfoSet: #cannot create appendices which depend on this structure
            debugAidsAppendix = False
            pixelPlotAppendix = False
            whackamoleAppendix = False
        
        qtys = {} # dictionary to store all latex strings to be inserted into report template
        qtys['title'] = title
        qtys['datasetLabel'] = datasetLabel
        qtys['settoggles'] =  "\\toggle%s{confidences}\n" % \
            ("false" if confidenceLevel is None else "true" )
        qtys['settoggles'] += "\\toggle%s{LsAndGermsSet}\n" % \
            ("true" if self._LsAndGermInfoSet else "false" )
        qtys['settoggles'] += "\\toggle%s{debuggingaidsappendix}\n" % \
            ("true" if debugAidsAppendix else "false" )
        qtys['settoggles'] += "\\toggle%s{pixelplotsappendix}\n" % \
            ("true" if pixelPlotAppendix else "false" )
        qtys['settoggles'] += "\\toggle%s{whackamoleappendix}\n" % \
            ("true" if whackamoleAppendix else "false" )
        qtys['confidenceLevel'] = "%g" % confidenceLevel \
            if confidenceLevel is not None else "NOT-SET"
        qtys['objective'] = "$\\log{\\mathcal{L}}$" \
            if self.parameters['objective'] == "logl" else "$\\chi^2$"
        qtys['gofObjective'] = "$2\\Delta\\log{\\mathcal{L}}$" \
            if self.parameters['objective'] == "logl" else "$\\chi^2$"
    
        if confidenceLevel is not None:
            cri = self._get_confidence_region(confidenceLevel)
            qtys['confidenceIntervalScaleFctr'] = "%.3g" % cri.intervalScaling
            qtys['confidenceIntervalNumNonGaugeParams'] = "%d" % cri.nNonGaugeParams
        else:
            cri = None
            qtys['confidenceIntervalScaleFctr'] = "NOT-SET"
            qtys['confidenceIntervalNumNonGaugeParams'] = "NOT-SET"

        pdfInfo = [('Author','pyGSTi'), ('Title', title),
                   ('Keywords', 'GST'), ('pyGSTi Version',_version.__version__),
                   ('opt_long_tables', self.options.long_tables),
                   ('opt_table_class', self.options.table_class),
                   ('opt_template_path', self.options.template_path),
                   ('opt_latex_cmd', self.options.latex_cmd) ]
        for key,val in self.parameters.iteritems():
            pdfInfo.append( (key, val) )
        qtys['pdfinfo'] = _to_pdfinfo( pdfInfo )


        #Get figure directory for figure generation *and* as a 
        # scratch space for tables.
        D = report_base + "_files" #figure directory relative to reportDir
        if not _os.path.isdir( _os.path.join(report_dir,D)):
            _os.mkdir( _os.path.join(report_dir,D))

            
        # 1) get latex tables
        if verbosity > 0: 
            print "*** Generating tables ***"; _sys.stdout.flush()

        std_tables =('targetSpamTable','targetGatesTable',
                     'datasetOverviewTable','bestGatesetSpamTable',
                     'bestGatesetSpamParametersTable',
                     'bestGatesetGatesTable','bestGatesetChoiTable',
                     'bestGatesetDecompTable','bestGatesetRotnAxisTable',
                     'bestGatesetVsTargetTable','bestGatesetErrorGenTable')
        ls_and_germs_tables = ('fiducialListTable','prepStrListTable',
                               'effectStrListTable','germListTable',
                               'progressTable')

        tables_to_compute = std_tables
        tables_to_blank = []

        if self._LsAndGermInfoSet:
            tables_to_compute += ls_and_germs_tables
        else:
            tables_to_blank += ls_and_germs_tables

        for key in tables_to_compute:
            qtys[key] = self.tables.get(key, verbosity=v).render(
                'latex',longtables=self.options.long_tables, scratchDir=D)
            qtys["tt_"+key] = tooltiptex(".tables['%s']" % key)

        for key in tables_to_blank:
            qtys[key] = _generation.get_blank_table().render(
                'latex',longtables=self.options.long_tables)
            qtys["tt_"+key] = ""

    
        # 2) generate plots
        if verbosity > 0: 
            print "*** Generating plots ***"; _sys.stdout.flush()

        if _matplotlib.is_interactive():
            _matplotlib.pyplot.ioff()
            bWasInteractive = True
        else: bWasInteractive = False
    
        maxW,maxH = 4.0,3.0 #max width and height of graphic in latex presentation (in inches)
        maxHc = 2.5 #max height allowed for a figure with a caption (in inches)

        def incgr(figFilenm,W=None,H=None): #includegraphics "macro"
            if W is None: W = maxW
            if H is None: H = maxH
            return "\\includegraphics[width=%.2fin,height=%.2fin" % (W,H) + \
                ",keepaspectratio]{%s/%s}" % (D,figFilenm)

        def set_fig_qtys(figkey, figFilenm, W=None,H=None):
            fig = self.figures.get(figkey, verbosity=v)
            fig.save_to(_os.path.join(report_dir, D, figFilenm))
            qtys[figkey] = incgr(figFilenm,W,H)
            qtys['tt_' + figkey] = tooltiptex(".figures['%s']" % figkey)
            return fig

        #Chi2 or logl plots
        if self._LsAndGermInfoSet:
            baseStr_dict = self._getBaseStrDict()

            Ls = self.parameters['max length list']
            st = 1 if Ls[0] == 0 else 0 #start index: skip LGST col in box plots
            nPlots = (len(Ls[st:])-1)+1 if pixelPlotAppendix else 1

            if self.parameters['objective'] == "chi2":
                plotFnName,plotFnLatex = "Chi2", "$\chi^2$"
            elif self.parameters['objective'] == "logl":
                plotFnName,plotFnLatex = "LogL", "$\\log(\\mathcal{L})$"
            else: 
                raise ValueError("Invalid objective value: %s" 
                                 % self.parameters['objective'])
            
            if verbosity > 0: 
                print " -- %s plots (%d): " % (plotFnName, nPlots),; _sys.stdout.flush()

            if verbosity > 0: 
                print "1 ",; _sys.stdout.flush()
            fig = set_fig_qtys("bestEstimateColorBoxPlot",
                               "best%sBoxes.pdf" % plotFnName)
            maxX = fig.get_extra_info()['nUsedXs']
            maxY = fig.get_extra_info()['nUsedYs']

        else:
            for figkey in ["bestEstimateColorBoxPlot"]:
                qtys[figkey] = qtys["tt_"+figkey] = ""

        
        pixplots = ""
        if pixelPlotAppendix:
            for i in range(st,len(self.parameters['max length list'])-1):

                if verbosity > 0: 
                    print "%d " % (i-st+2),; _sys.stdout.flush()

                fig = self.figures.get("estimateForLIndex%dColorBoxPlot" % i,
                                       verbosity=v)
                fig.save_to( _os.path.join(report_dir, D,
                                           "L%d_%sBoxes.pdf" % (i,plotFnName)) )
                lx = fig.get_extra_info()['nUsedXs']
                ly = fig.get_extra_info()['nUsedYs']

                #scale figure size according to number of rows and columns+1
                # (+1 for labels ~ another col) relative to initial plot
                W = float(lx+1)/float(maxX+1) * maxW
                H = float(ly)  /float(maxY)   * maxH
            
                pixplots += "\n"
                pixplots += "\\begin{frame}\n"
                pixplots += "\\frametitle{Iteration %d ($L=%d$): %s values}\n" \
                    % (i, self.parameters['max length list'][i], plotFnLatex)
                pixplots += "\\begin{figure}\n"
                pixplots += "\\begin{center}\n"
                #pixplots += "\\adjustbox{max height=\\dimexpr\\textheight-5.5cm\\relax, max width=\\textwidth}{"
                pixplots += "\\includegraphics[width=%.2fin,height=%.2fin,keepaspectratio]{%s/L%d_%sBoxes.pdf}\n" % (W,H,D,i,plotFnName)
                #FUTURE: add caption and conditional tooltip string?
                pixplots += "\\end{center}\n"
                pixplots += "\\end{figure}\n"
                pixplots += "\\end{frame}\n"

        #Set template quantity (empty string if appendix disabled)
        qtys['intermediate_pixel_plot_slides'] = pixplots

    
        if verbosity > 0: 
            print ""; _sys.stdout.flush()
        
        if debugAidsAppendix:
            #Direct-GST and deviation
            if verbosity > 0: 
                print " -- Direct-X plots (2)",; _sys.stdout.flush()

            if verbosity > 0: 
                print " 1",; _sys.stdout.flush()        
            fig = set_fig_qtys("directLongSeqGSTColorBoxPlot",
                           "directLongSeqGST%sBoxes.pdf" % plotFnName,H=maxHc)

            if verbosity > 0: 
                print " 2",; _sys.stdout.flush()
            fig = set_fig_qtys("directLongSeqGSTDeviationColorBoxPlot",
                               "directLongSeqGSTDeviationBoxes.pdf",H=maxHc)

            if verbosity > 0: 
                print ""; _sys.stdout.flush()

    
            #Small eigenvalue error rate
            if verbosity > 0: 
                print " -- Error rate plots..."; _sys.stdout.flush()
            fig = set_fig_qtys("smallEigvalErrRateColorBoxPlot",
                               "smallEigvalErrRateBoxes.pdf",H=maxHc)

        else:
            for figkey in ["directLongSeqGSTColorBoxPlot",
                           "directLongSeqGSTDeviationColorBoxPlot",
                           "smallEigvalErrRateColorBoxPlot"]:
                qtys[figkey] = qtys["tt_"+figkey] = ""



        whackamoleplots = ""
        if whackamoleAppendix:    
            #Whack-a-mole plots for highest L of each length-1 germ
            highestL = self.parameters['max length list'][-1]
            allGateStrings = self.gatestring_lists['all']
            hammerWeight = 10.0
            len1Germs = [ g for g in self.gatestring_lists['germs']
                          if len(g) == 1 ]

            if verbosity > 0: 
                print " -- Whack-a-mole plots (%d): " % (2*len(len1Germs)),; _sys.stdout.flush()

            for i,germ in enumerate(len1Germs):
                if verbosity > 0: 
                    print "%d " % (i+1),; _sys.stdout.flush()

                fig = self.figures.get("whack%sMoleBoxes" % germ[0],verbosity=v)
                fig.save_to(_os.path.join(report_dir, D,"whack%sMoleBoxes.pdf"
                                          % germ[0]))
        
                whackamoleplots += "\n"
                whackamoleplots += "\\begin{frame}\n"
                whackamoleplots += "\\frametitle{Whack-a-%s-mole plot for $\mathrm{%s}^{%d}$}" % (plotFnLatex,germ[0],highestL)
                whackamoleplots += "\\begin{figure}\n"
                whackamoleplots += "\\begin{center}\n"
                #whackamoleplots += "\\adjustbox{max height=\\dimexpr\\textheight-5.5cm\\relax, max width=\\textwidth}{"
                whackamoleplots += "\\includegraphics[width=%.2fin,height=%.2fin,keepaspectratio]{%s/whack%sMoleBoxes.pdf}\n" % (maxW,maxH,D,germ[0])
                #FUTURE: add caption and conditional tooltip?
                whackamoleplots += "\\end{center}\n"
                whackamoleplots += "\\end{figure}\n"
                whackamoleplots += "\\end{frame}\n"
        
            for i,germ in enumerate(len1Germs):
                if verbosity > 0: 
                    print "%d " % (len(len1Germs)+i+1),; _sys.stdout.flush()
    
                fig = self.figures.get("whack%sMoleBoxesSummed" % germ[0],
                                       verbosity=v)
                fig.save_to(_os.path.join(report_dir, D,"whack%sMoleBoxesSummed.pdf" % germ[0]))

                whackamoleplots += "\n"
                whackamoleplots += "\\begin{frame}\n"
                whackamoleplots += "\\frametitle{Summed whack-a-%s-mole plot for $\mathrm{%s}^{%d}$}" % (plotFnLatex,germ[0],highestL)
                whackamoleplots += "\\begin{figure}\n"
                whackamoleplots += "\\begin{center}\n"
                #whackamoleplots += "\\adjustbox{max height=\\dimexpr\\textheight-5.5cm\\relax, max width=\\textwidth}{"
                whackamoleplots += "\\includegraphics[width=%.2fin,height=%.2fin,keepaspectratio]{%s/whack%sMoleBoxesSummed.pdf}\n" % (maxW,maxH,D,germ[0])
                #FUTURE: add caption and conditional tooltip?
                whackamoleplots += "\\end{center}\n"
                whackamoleplots += "\\end{figure}\n"
                whackamoleplots += "\\end{frame}\n"
    
            if verbosity > 0: 
                print ""; _sys.stdout.flush()
        
        #Set template quantity (empty string if appendix disabled)
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
            ret = _os.system( "%s %s %s" % 
                              (self.options.latex_cmd, 
                               _os.path.basename(mainTexFilename),
                               self.options.latex_postcmd) )
            if ret == 0:
                #We could check if the log file contains "Rerun" in it, but we'll just re-run all the time now
                if verbosity > 0: 
                    print "Initial output PDF %s successfully generated." % pdfFilename #mainTexFilename
                ret = _os.system( "%s %s %s" % 
                                  (self.options.latex_cmd,
                                   _os.path.basename(mainTexFilename),
                                   self.options.latex_postcmd) )
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
            print "Error trying to run pdflatex to generate output PDF %s. Is '%s' on your path?" % (pdfFilename,self.options.latex_cmd)
        finally: 
            _os.chdir(cwd)
    
        return
        



    def create_presentation_ppt(self, confidenceLevel=None, filename="auto", 
                            title="auto", datasetLabel="auto", suffix="",
                            debugAidsAppendix=False,
                            pixelPlotAppendix=False, whackamoleAppendix=False,
                            m=0, M=10, verbosity=0, pptTables=False, comm=None):
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

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.


        Returns
        -------
        None
        """

        assert(self._bEssentialResultsSet)
        self.confidence_level = confidenceLevel 
        self._comm = comm
        v = verbosity # shorthand

        #Currently, no tooltip option for presentations
        def tooltiptext(directive):
            return "" #tooltips disabled


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
        default_dir  = self.parameters['defaultDirectory']
        default_base = self.parameters['defaultBasename']

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
        best_gs = self.gatesets['final estimate']

        if not self._LsAndGermInfoSet: #cannot create appendices which depend on this structure
            debugAidsAppendix = False
            pixelPlotAppendix = False
            whackamoleAppendix = False
        
        qtys = {}
        qtys['title'] = title
        qtys['datasetLabel'] = datasetLabel
        qtys['confidenceLevel'] = "%g" % confidenceLevel \
            if confidenceLevel is not None else "NOT-SET"
        qtys['objective'] = "$\\log{\\mathcal{L}}$" \
            if self.parameters['objective'] == "logl" else "$\\chi^2$"
        qtys['gofObjective'] = "$2\\Delta\\log{\\mathcal{L}}$" \
            if self.parameters['objective'] == "logl" else "$\\chi^2$"
    
        if confidenceLevel is not None:
            cri = self._get_confidence_region(confidenceLevel)
            qtys['confidenceIntervalScaleFctr'] = "%.3g" % cri.intervalScaling
            qtys['confidenceIntervalNumNonGaugeParams'] = "%d" % cri.nNonGaugeParams
        else:
            cri = None
            qtys['confidenceIntervalScaleFctr'] = "NOT-SET"
            qtys['confidenceIntervalNumNonGaugeParams'] = "NOT-SET"

        #Get figure directory for figure generation *and* as a 
        # scratch space for tables.
        D = report_base + "_files" #figure directory relative to reportDir
        if not _os.path.isdir( _os.path.join(report_dir,D)):
            _os.mkdir( _os.path.join(report_dir,D))
            
        # 1) get ppt tables
        if verbosity > 0: 
            print "*** Generating tables ***"; _sys.stdout.flush()

        std_tables = ('targetSpamTable','targetGatesTable',
                      'datasetOverviewTable', 'bestGatesetSpamTable',
                      'bestGatesetSpamParametersTable','bestGatesetGatesTable',
                      'bestGatesetChoiTable', 'bestGatesetDecompTable',
                      'bestGatesetRotnAxisTable','bestGatesetVsTargetTable',
                      'bestGatesetErrorGenTable')

        ls_and_germs_tables = ('fiducialListTable','prepStrListTable',
                               'effectStrListTable','germListTable',
                               'progressTable')

        tables_to_compute = std_tables
        tables_to_blank = []

        if self._LsAndGermInfoSet:
            tables_to_compute += ls_and_germs_tables
        else:
            tables_to_blank += ls_and_germs_tables

        for key in tables_to_compute:
            qtys[key] = self.tables.get(key, verbosity=v)
            qtys["tt_"+key] = tooltiptext(".tables['%s']" % key)

        for key in tables_to_blank:
            qtys[key] = _generation.get_blank_table()
            qtys["tt_"+key] = ""
    

        # 2) generate plots
        if verbosity > 0: 
            print "*** Generating plots ***"; _sys.stdout.flush()

        if _matplotlib.is_interactive():
            _matplotlib.pyplot.ioff()
            bWasInteractive = True
        else: bWasInteractive = False
    
        fileDir = _os.path.join(report_dir, D)
        maxW,maxH = 4.0,3.0 #max width and height of graphic in latex presentation (in inches)

        def incgr(figFilenm,W=None,H=None): #includegraphics "macro"
            return "%s/%s" % (fileDir,figFilenm)

        def set_fig_qtys(figkey, figFilenm, W=None,H=None):
            fig = self.figures.get(figkey, verbosity=v)
            fig.save_to(_os.path.join(report_dir, D, figFilenm))
            qtys[figkey] = incgr(figFilenm,W,H)
            qtys['tt_' + figkey] = tooltiptext(".figures['%s']" % figkey)
            return fig


        #Chi2 or logl plots
        if self._LsAndGermInfoSet:
            baseStr_dict = self._getBaseStrDict()

            Ls = self.parameters['max length list']
            st = 1 if Ls[0] == 0 else 0 #start index: skip LGST col in box plots
            nPlots = (len(Ls[st:])-1)+1 if pixelPlotAppendix else 1

            if self.parameters['objective'] == "chi2":
                plotFnName,plotFnLatex = "Chi2", "$\chi^2$"
            elif self.parameters['objective'] == "logl":
                plotFnName,plotFnLatex = "LogL", "$\\log(\\mathcal{L})$"
            else: 
                raise ValueError("Invalid objective value: %s" \
                                     % self.parameters['objective'])
            
            if verbosity > 0: 
                print " -- %s plots (%d): " % (plotFnName, nPlots),; _sys.stdout.flush()

            if verbosity > 0: 
                print "1 ",; _sys.stdout.flush()
            fig = set_fig_qtys("bestEstimateColorBoxPlot",
                               "best%sBoxes.png" % plotFnName)
            maxX = fig.get_extra_info()['nUsedXs']
            maxY = fig.get_extra_info()['nUsedYs']

        else:
            for figkey in ["bestEstimateColorBoxPlot"]:
                qtys[figkey] = qtys["tt_"+figkey] = ""


        pixplots = []
        if pixelPlotAppendix:
            Ls = self.parameters['max length list']
            for i in range(st,len(Ls)-1):

                if verbosity > 0: 
                    print "%d " % (i-st+2),; _sys.stdout.flush()

                fig = self.figures.get("estimateForLIndex%dColorBoxPlot" % i,
                                       verbosity=v)
                fig.save_to( _os.path.join(report_dir, D,"L%d_%sBoxes.png" %
                                           (i,plotFnName)) )
                lx = fig.get_extra_info()['nUsedXs']
                ly = fig.get_extra_info()['nUsedYs']

                #scale figure size according to number of rows and columns+1
                # (+1 for labels ~ another col) relative to initial plot
                W = float(lx+1)/float(maxX+1) * maxW
                H = float(ly)  /float(maxY)   * maxH
            
                pixplots.append( _os.path.join(
                        report_dir, D, "L%d_%sBoxes.png" % (i,plotFnName)) )
                #FUTURE: Add tooltip caption info further down?

        #Set template quantity (empty array if appendix disabled)
        qtys['intermediate_pixel_plot_slides'] = pixplots
    
        if verbosity > 0: 
            print ""; _sys.stdout.flush()
        
        if debugAidsAppendix:
            #Direct-GST and deviation
            if verbosity > 0: 
                print " -- Direct-X plots (2)",; _sys.stdout.flush()

            if verbosity > 0: 
                print " 1",; _sys.stdout.flush()        
            fig = set_fig_qtys("directLongSeqGSTColorBoxPlot",
                           "directLongSeqGST%sBoxes.png" % plotFnName)

            if verbosity > 0: 
                print " 2",; _sys.stdout.flush()
            fig = set_fig_qtys("directLongSeqGSTDeviationColorBoxPlot",
                               "directLongSeqGSTDeviationBoxes.png")

            if verbosity > 0: 
                print ""; _sys.stdout.flush()
    
            #Small eigenvalue error rate
            if verbosity > 0: 
                print " -- Error rate plots..."; _sys.stdout.flush()
            fig = set_fig_qtys("smallEigvalErrRateColorBoxPlot",
                               "smallEigvalErrRateBoxes.png")

        else:
            for figkey in ["directLongSeqGSTColorBoxPlot",
                           "directLongSeqGSTDeviationColorBoxPlot",
                           "smallEigvalErrRateColorBoxPlot"]:
                qtys[figkey] = qtys["tt_"+figkey] = ""
                
    
        whackamoleplots = []
        if whackamoleAppendix:    
            #Whack-a-mole plots for highest L of each length-1 germ
            Ls = self.parameters['max length list']
            highestL = Ls[-1]; allGateStrings = self.gatestring_lists['all']
            hammerWeight = 10.0
            len1Germs = [ g for g in self.gatestring_lists['germs'] 
                          if len(g) == 1 ]

            if verbosity > 0: 
                print " -- Whack-a-mole plots (%d): " % (2*len(len1Germs)),
                _sys.stdout.flush()

            for i,germ in enumerate(len1Germs):
                if verbosity > 0: 
                    print "%d " % (i+1),; _sys.stdout.flush()

                fig = self.figures.get("whack%sMoleBoxes" % germ[0],verbosity=v)
                fig.save_to(_os.path.join(report_dir, D,"whack%sMoleBoxes.png"
                                          % germ[0]))
                whackamoleplots.append( _os.path.join(
                        report_dir, D, "whack%sMoleBoxes.png" % germ[0]) )
                #FUTURE: Add tooltip caption info further down?
        
            for i,germ in enumerate(len1Germs):
                if verbosity > 0: 
                    print "%d " % (len(len1Germs)+i+1),; _sys.stdout.flush()

                fig = self.figures.get("whack%sMoleBoxesSummed" % germ[0],
                                       verbosity=v)
                fig.save_to(_os.path.join(
                        report_dir, D, "whack%sMoleBoxesSummed.png" % germ[0]))
                whackamoleplots.append( _os.path.join(
                        report_dir, D,"whack%sMoleBoxesSummed.png" % germ[0]) )
                #FUTURE: Add tooltip caption info further down?
    
            if verbosity > 0: 
                print ""; _sys.stdout.flush()

        #Set template quantity (empty array if appendix disabled)
        qtys['whackamole_plot_slides'] = whackamoleplots
    
        if bWasInteractive:
            _matplotlib.pyplot.ion()

    
        # 3) create PPT file via python-pptx
        if verbosity > 0: 
            print "*** Assembling PPT file ***"; _sys.stdout.flush()

        mainPPTFilename = _os.path.join(report_dir, report_base + ".pptx")        
        templatePath = self.options.template_path \
            if (self.options.template_path is not None) else \
            _os.path.join( _os.path.dirname(_os.path.abspath(__file__)),
                           "templates")
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
            tabDict = qtys[key].render('ppt')
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
            latexTabStr = qtys[key].render('latex',
                                           longtables=self.options.long_tables,
                                           scratchDir=D)
            d = {'toLatex': latexTabStr }
            print "Latexing %s table..." % key; _sys.stdout.flush()
            outputFilename = _os.path.join(fileDir, "%s.tex" % key)
            self._merge_template(d, "standalone.tex", outputFilename)

            cwd = _os.getcwd()
            _os.chdir(fileDir)
            try:
                ret = _os.system("%s -shell-escape %s.tex %s" \
                                     % (self.options.latex_cmd, key,
                                        self.options.latex_postcmd) )
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
        templateDir =_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  "templates") if self.options.template_path \
                                  is None else self.options.template_path
        prs = Presentation( _os.path.join( templateDir, "GSTTemplate.pptx" ) )


        # title slide
        slide = add_slide(SLD_LAYOUT_TITLE, title)
        subtitle_shape = slide.placeholders[1]
        subtitle_shape.text = "Your GST results in Powerpoint!"

        # goodness of fit
        if self._LsAndGermInfoSet:
            slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "%s vs GST iteration" % plotFnName)
            #body_shape = slide.shapes.placeholders[1]; tf = body_shape.text_frame
            add_text_list(slide.shapes, 1, 2, 8, 2, ['Ns is the number of gate strings', 'Np is the number of parameters'], 15)
            drawTable(slide.shapes, 'progressTable', 1, 3, 8.5, 4, ptSize=10)
        
            slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "Detailed %s Analysis" % plotFnName)
            draw_pic(slide.shapes, qtys['bestEstimateColorBoxPlot'], 1, 1.5, 8, 5.5)

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

        if self._LsAndGermInfoSet:
            slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "Fiducial and Germ Gate Strings")
            drawTable(slide.shapes, 'fiducialListTable', 1, 1.5, 4, 3, ptSize=10)
            drawTable(slide.shapes, 'germListTable', 5.5, 1.5, 4, 5, ptSize=10)

        slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "Dataset Overview")
        drawTable(slide.shapes, 'datasetOverviewTable', 1, 2, 5, 4, ptSize=10)

        if debugAidsAppendix:
            #Debugging aids slides
            slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "Direct-GST")
            draw_pic(slide.shapes, qtys['directLongSeqGSTColorBoxPlot'], 1, 1.5, 8, 5.5)

            slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "Direct-GST Deviation")
            draw_pic(slide.shapes, qtys['directLongSeqGSTDeviationColorBoxPlot'], 1, 1.5, 8, 5.5)

            slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "Per-gate error rates")
            draw_pic(slide.shapes, qtys['smallEigvalErrRateColorBoxPlot'], 1, 1.5, 8, 5.5)

        if pixelPlotAppendix:
            Ls = self.parameters['max length list']
            for i,pixPlotPath in zip( range(st,len(Ls)-1), pixplots ):
                slide = add_slide(SLD_LAYOUT_TITLE_NO_CONTENT, "Iteration %d (L=%d): %s values" % (i,Ls[i],plotFnName))
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


    def create_general_report_pdf(self, confidenceLevel=None, filename="auto", 
                                  title="auto", datasetLabel="auto", suffix="",
                                  tips=False, verbosity=0, comm=None):
        """
        Create a "general" GST report.  This report is suited to display
        results for any number of qubits, and is detailed in the sense that
        it includes background and explanation text to help the user
        interpret the contained results.

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

        tips : boolean, optional
           If True, additional markup and tooltips are included in the produced
           PDF which indicate how tables and figures in the report correspond
           to members of this Results object.  These "tips" can be useful if
           you want to further manipulate the data contained in a table or
           figure.

        verbosity : int, optional
           How much detail to send to stdout.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        Returns
        -------
        None
        """
        assert(self._bEssentialResultsSet)
        self.confidence_level = confidenceLevel 
        self._comm = comm
          #set "current" level, used by ResultCache member dictionaries

        if self.parameters['objective'] != "logl":
            raise NotImplementedError("General reports are currently " +
                                      "only implemented for log-likelihood " +
                                      "objective function case.")

        if tips:
            def tooltiptex(directive):
                return '\\pdftooltip{{\\color{blue}\\texttt{%s}}\\quad}' \
                    % directive + '{Access this information in pyGSTi via' \
                    + '<ResultsObject>%s}' % directive


        else:
            def tooltiptex(directive):
                return "" #tooltips disabled


        #Get report output filename
        default_dir = self.parameters['defaultDirectory']
        default_base = self.parameters['defaultBasename']

        if filename != "auto":
            report_dir = _os.path.dirname(filename)
            report_base = _os.path.splitext( _os.path.basename(filename) )[0] \
                           + suffix
        else:
            cwd = _os.getcwd()
            report_dir  = default_dir  if (default_dir  is not None) else cwd
            report_base = default_base if (default_base is not None) \
                                       else "GSTReport"
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
        #  FUTURE?? determine what we need to compute & plot by reading
        #           through the template file?
        
        #Note: for now, we assume the best gateset corresponds to the last
        #      L-value
        best_gs = self.gatesets['final estimate']
        v = verbosity # shorthand

        if not self._LsAndGermInfoSet: #cannot create appendices 
            debugAidsAppendix = False  # which depend on this structure
            pixelPlotAppendix = False
            whackamoleAppendix = False
        
        qtys = {} # dictionary to store all latex strings
                  # to be inserted into report template
        qtys['title'] = title   
        qtys['datasetLabel'] = datasetLabel
        qtys['settoggles'] =  "\\toggle%s{confidences}\n" % \
            ("false" if confidenceLevel is None else "true")
        qtys['settoggles'] += "\\toggle%s{LsAndGermsSet}\n" % \
            ("true" if self._LsAndGermInfoSet else "false")
        #qtys['settoggles'] += "\\toggle%s{debuggingaidsappendix}\n" % \
        #    ("true" if debugAidsAppendix else "false")
        #qtys['settoggles'] += "\\toggle%s{gaugeoptappendix}\n" % \
        #    ("true" if gaugeOptAppendix else "false")
        #qtys['settoggles'] += "\\toggle%s{pixelplotsappendix}\n" % \
        #    ("true" if pixelPlotAppendix else "false")
        #qtys['settoggles'] += "\\toggle%s{whackamoleappendix}\n" % \
        #    ("true" if whackamoleAppendix else "false")
        qtys['confidenceLevel'] = "%g" % \
            confidenceLevel if confidenceLevel is not None else "NOT-SET"
        qtys['linlg_pcntle'] = self.parameters['linlogPercentile']

        if confidenceLevel is not None:
            cri = self._get_confidence_region(confidenceLevel)
            qtys['confidenceIntervalScaleFctr'] = \
                "%.3g" % cri.intervalScaling
            qtys['confidenceIntervalNumNonGaugeParams'] = \
                "%d" % cri.nNonGaugeParams
        else:
            cri = None
            qtys['confidenceIntervalScaleFctr'] = "NOT-SET"
            qtys['confidenceIntervalNumNonGaugeParams'] = "NOT-SET"

        pdfInfo = [('Author','pyGSTi'), ('Title', title),
                   ('Keywords', 'GST'), ('pyGSTi Version',_version.__version__),
                   ('opt_long_tables', self.options.long_tables),
                   ('opt_table_class', self.options.table_class),
                   ('opt_template_path', self.options.template_path),
                   ('opt_latex_cmd', self.options.latex_cmd) ]
        for key,val in self.parameters.iteritems():
            pdfInfo.append( (key, val) )
        qtys['pdfinfo'] = _to_pdfinfo( pdfInfo )

        #Get figure directory for figure generation *and* as a 
        # scratch space for tables.
        D = report_base + "_files" #figure directory relative to reportDir
        if not _os.path.isdir( _os.path.join(report_dir,D)):
            _os.mkdir( _os.path.join(report_dir,D))
 
        # 1) get latex tables
        if verbosity > 0: 
            print "*** Generating tables ***"; _sys.stdout.flush()

        
        std_tables = \
            ('targetSpamBriefTable', 'bestGatesetSpamBriefTable',
             'bestGatesetSpamParametersTable', 'bestGatesetVsTargetTable',
             'bestGatesetSpamVsTargetTable', 'bestGatesetGaugeOptParamsTable',
             'bestGatesetChoiEvalTable', 'datasetOverviewTable',
             'bestGatesetEvalTable', 'bestGatesetRelEvalTable',
             'targetGatesBoxTable', 'bestGatesetErrGenBoxTable')
             
#'bestGatesetDecompTable','bestGatesetRotnAxisTable',
#'bestGatesetClosestUnitaryTable',
        
        ls_and_germs_tables = ('fiducialListTable','prepStrListTable',
                               'effectStrListTable','germList2ColTable',
                               'progressTable')


        tables_to_compute = std_tables
        tables_to_blank = []

        if self._LsAndGermInfoSet:
            tables_to_compute += ls_and_germs_tables
        else:
            tables_to_blank += ls_and_germs_tables

        #Change to report directory so figure generation works correctly
        cwd = _os.getcwd()
        if len(report_dir) > 0: _os.chdir(report_dir)

        for key in tables_to_compute:
            qtys[key] = self.tables.get(key, verbosity=v).render(
                'latex',longtables=self.options.long_tables, scratchDir=D)
            qtys["tt_"+key] = tooltiptex(".tables['%s']" % key)

        _os.chdir(cwd) #change back to original directory

        for key in tables_to_blank:
            qtys[key] = _generation.get_blank_table().render(
                'latex',longtables=self.options.long_tables)
            qtys["tt_"+key] = ""

        #get appendix tables if needed
        #if gaugeOptAppendix: 
        #    goaTables = self._specials.get('gaugeOptAppendixTables',verbosity=v)
        #    qtys.update( { key : goaTables[key].render('latex') 
        #                   for key in goaTables }  )
        #    #TODO: tables[ref] and then tooltips?
        #
        #elif any((debugAidsAppendix, pixelPlotAppendix, whackamoleAppendix)):
        #    goaTables = self._specials.get('blankGaugeOptAppendixTables',
        #                      verbosity=v)   # fill keys with blank tables
        #    qtys.update( { key : goaTables[key].render('latex') 
        #                   for key in goaTables }  )  # for format substitution
        #    #TODO: tables[ref] and then tooltips?

    
        # 2) generate plots
        if verbosity > 0: 
            print "*** Generating plots ***"; _sys.stdout.flush()

        if _matplotlib.is_interactive():
            _matplotlib.pyplot.ioff()
            bWasInteractive = True
        else: bWasInteractive = False
    
        maxW,maxH = 6.5,9.0 #max width and height of graphic in latex document (in inches)

        def incgr(figFilenm,W=None,H=None): #includegraphics "macro"
            if W is None: W = maxW
            if H is None: H = maxH
            return "\\includegraphics[width=%.2fin,height=%.2fin" % (W,H) + \
                ",keepaspectratio]{%s/%s}" % (D,figFilenm)

        def set_fig_qtys(figkey, figFilenm, W=None,H=None):
            fig = self.figures.get(figkey, verbosity=v)
            fig.save_to(_os.path.join(report_dir, D, figFilenm))
            qtys[figkey] = incgr(figFilenm,W,H)
            qtys['tt_' + figkey] = tooltiptex(".figures['%s']" % figkey)
            return fig

        ## Gate/SPAM box tables for visualizing large matrices.
        ##  - these tables are "special" in that they contain figures, so 
        ##    there's no way to simply incorporate them into the figures or
        ##    tables member dictionaries.
        #def make_gateset_box_table(gsKey, tablekey, figPrefixes, figColWidths, figHeadings):
        #    gs = self.gatesets[gsKey]
        #    latex = "\\begin{tabular}[l]{| >{\\centering\\arraybackslash}m{0.75in} | %s |}\n\hline\n" % \
        #        " | ".join([ ">{\\centering\\arraybackslash}m{%.1fin}" % w for w in figColWidths ])
        #    latex += "Gate & %s \\\\ \hline\n" % " & ".join(figHeadings)
        #
        #    for gateLabel in gs.gates:
        #        latex += gateLabel
        #        for figprefix,colW in zip(figPrefixes,figColWidths):
        #            figkey = figprefix + gateLabel
        #            figFilenm = figkey + ".pdf"
        #            fig = self.figures.get(figkey, verbosity=v)
        #            fig.save_to(_os.path.join(report_dir, D, figFilenm))
        #            maxFigH = min(0.95*(maxH / len(gs.gates)),colW)
        #            sz = min(gs.gates[gateLabel].shape[0] * 0.15, maxFigH)
        #            latex += " & " + incgr(figFilenm,sz,sz)
        #        latex += "\\\\ \hline\n"
        #        
        #    latex += "\end{tabular}\n"
        #        
        #    qtys['tt_' + tablekey] = "" #tooltiptex(".tables['%s']" % tablekey)
        #    qtys[tablekey] = latex
        #
        #basisNm = _bt.basis_longname(self.gatesets['target'].get_basis_name(),
        #                             self.gatesets['target'].get_basis_dimension())
        #make_gateset_box_table('target', 'targetGatesBoxTable',
        #                       ("targetGateBoxes",), (3,), ("Matrix (%s basis)" % basisNm,) )
        #
        #basisNm = _bt.basis_longname(self.gatesets['final estimate'].get_basis_name(),
        #                             self.gatesets['final estimate'].get_basis_dimension())
        #make_gateset_box_table('final estimate', 'bestGatesetErrGenBoxTable',
        #                       ("bestGateErrGenBoxes","pauliProdHamiltonianDecompBoxes"), (3,1.5),
        #                       ("Error Generator (%s basis)" % basisNm, "Pauli-product projections") )

        #Chi2 or logl plots
        if self._LsAndGermInfoSet:
            Ls = self.parameters['max length list']
            st = 1 if Ls[0] == 0 else 0 #start index: skip LGST column in plots
            nPlots = 4 #(len(Ls[st:])-1)+2 if pixelPlotAppendix else 2

            if self.parameters['objective'] == "chi2":
                plotFnName,plotFnLatex = "Chi2", "$\chi^2$"
            elif self.parameters['objective'] == "logl":
                plotFnName,plotFnLatex = "LogL", "$\\log(\\mathcal{L})$"
            else: 
                raise ValueError("Invalid objective value: %s" 
                                 % self.parameters['objective'])
            
            if verbosity > 0: 
                print " -- %s plots (%d): " % (plotFnName, nPlots),
                _sys.stdout.flush()

            if verbosity > 0:
                print "1 ",; _sys.stdout.flush()

            w = min(len(self.gatestring_lists['prep fiducials']) * 0.3,maxW)
            h = min(len(self.gatestring_lists['effect fiducials']) * 0.3,maxH)
            fig = set_fig_qtys("colorBoxPlotKeyPlot",
                               "colorBoxPlotKey.png", w,h)

            if verbosity > 0:
                print "2 ",; _sys.stdout.flush()

            fig = set_fig_qtys("bestEstimateSummedColorBoxPlot",
                               "best%sBoxesSummed.png" % plotFnName,
                               maxW, maxH-1.0) # -1 for room for caption

            if verbosity > 0:
                print "3 ",; _sys.stdout.flush()


            figkey = "bestEstimateColorBoxPlotPages"
            figs = self._specials.get(figkey, verbosity=v)
            incgr_list = []
            for iFig,fig in enumerate(figs):
                figFilenm = "best%sBoxes_pg%d.png" % (plotFnName,iFig)
                fig.save_to(_os.path.join(report_dir, D, figFilenm))
                if iFig == 0:
                    maxX = fig.get_extra_info()['nUsedXs']
                    maxY = fig.get_extra_info()['nUsedYs']
                    incgr_list.append(incgr(figFilenm,maxW,maxH-1.25),)
                else:
                    lx = fig.get_extra_info()['nUsedXs']
                    ly = fig.get_extra_info()['nUsedYs']
        
                    #scale figure size according to number of rows and columns+1
                    # (+1 for labels ~ another col) relative to initial plot
                    W = float(lx+1)/float(maxX+1) * maxW 
                    H = float(ly)  /float(maxY)   * (maxH - 1.25) # -1 for caption
                    incgr_list.append(incgr(figFilenm,W,H))
            qtys[figkey] = "\\end{center}\\end{figure}\\begin{figure}\\begin{center}".join(
                           incgr_list)
            qtys['tt_' + figkey] = tooltiptex(".figures['%s']" % "bestEstimateColorBoxPlot")

            #fig = set_fig_qtys("bestEstimateColorBoxPlot",
            #                   "best%sBoxes.png" % plotFnName)
            #maxX = fig.get_extra_info()['nUsedXs']
            #maxY = fig.get_extra_info()['nUsedYs']

            #qtys["bestEstimateColorBoxPlot_hist"] = \
            #    incgr("best%sBoxes_hist.pdf" % plotFnName figFilenm)
            #    #no tooltip for histogram... - probably should make it 
            #    # it's own element of .figures dict

            #if verbosity > 0: 
            #    print "2 ",; _sys.stdout.flush()
            #fig = set_fig_qtys("invertedBestEstimateColorBoxPlot",
            #                   "best%sBoxes_inverted.pdf" % plotFnName)

            #Unused polar plots figure...
            #if verbosity > 0:
            #    print "4 ",; _sys.stdout.flush()
            #
            #qtys["bestEstimatePolarEvalPlots"] = ""
            #qtys["tt_bestEstimatePolarEvalPlots"] = ""
            #for gl in self.gatesets['final estimate'].gates:
            #    figkey = "bestEstimatePolar%sEvalPlot" % gl
            #    figFilenm = "best%sPolarEvals.png" % gl
            #    fig = self.figures.get(figkey, verbosity=v)
            #    fig.save_to(_os.path.join(report_dir, D, figFilenm))
            #    W = H = 2.5
            #    qtys["bestEstimatePolarEvalPlots"] += incgr(figFilenm,W,H) + "\n"
            #    qtys['tt_bestEstimatePolarEvalPlots'] += tooltiptex(".figures['%s']" % figkey)

        else:
            for figkey in ["colorBoxPlotKeyPlot", 
                           "bestEstimateColorBoxPlot"]:
                qtys[figkey] = qtys["tt_"+figkey] = ""
                # "invertedBestEstimateColorBoxPlot"

    
        pixplots = ""
        #if pixelPlotAppendix:
        #    Ls = self.parameters['max length list']
        #    for i in range(st,len(Ls)-1):
        #
        #        if verbosity > 0: 
        #            print "%d " % (i-st+3),; _sys.stdout.flush()
        #        fig = self.figures.get("estimateForLIndex%dColorBoxPlot" % i,
        #                               verbosity=v)
        #        fig.save_to( _os.path.join(report_dir, D,
        #                                   "L%d_%sBoxes.pdf" % (i,plotFnName)))
        #        lx = fig.get_extra_info()['nUsedXs']
        #        ly = fig.get_extra_info()['nUsedYs']
        #
        #        #scale figure size according to number of rows and columns+1
        #        # (+1 for labels ~ another col) relative to initial plot
        #        W = float(lx+1)/float(maxX+1) * maxW 
        #        H = float(ly)  /float(maxY)   * maxH 
        #    
        #        pixplots += "\n"
        #        pixplots += "\\begin{figure}\n"
        #        pixplots += "\\begin{center}\n"
        #        pixplots += "\\includegraphics[width=%.2fin,height=%.2fin," \
        #            % (W,H) + "keepaspectratio]{%s/L%d_%sBoxes.pdf}\n" \
        #            %(D,i,plotFnName)
        #        pixplots += \
        #            "\\caption{Box plot of iteration %d (L=%d) " % (i,Ls[i]) \
        #            + "gateset %s values.\label{L%dGateset%sBoxPlot}}\n" \
        #            % (plotFnLatex,i,plotFnName)
        #        #TODO: add conditional tooltip string to start of caption
        #        pixplots += "\\end{center}\n"
        #        pixplots += "\\end{figure}\n"

        #Set template quantity (empty string if appendix disabled)
        qtys['intermediate_pixel_plot_figures'] = pixplots

        if verbosity > 0: 
            print ""; _sys.stdout.flush()
        
        #if debugAidsAppendix:
        #    #DirectLGST and deviation
        #    if verbosity > 0: 
        #        print " -- Direct-X plots ",; _sys.stdout.flush()
        #        print "(2):"; _sys.stdout.flush()    
        #
        #    #if verbosity > 0: 
        #    #    print " ?",; _sys.stdout.flush()
        #    #fig = set_fig_qtys("directLGSTColorBoxPlot",
        #    #                   "directLGST%sBoxes.pdf" % plotFnName)
        #
        #    if verbosity > 0: 
        #        print " 1",; _sys.stdout.flush()        
        #    fig = set_fig_qtys("directLongSeqGSTColorBoxPlot",
        #                   "directLongSeqGST%sBoxes.pdf" % plotFnName)
        #
        #    #if verbosity > 0: 
        #    #    print " ?",; _sys.stdout.flush()        
        #    #fig = set_fig_qtys("directLGSTDeviationColorBoxPlot",
        #    #                   "directLGSTDeviationBoxes.pdf",W=4,H=5)
        #
        #    if verbosity > 0: 
        #        print " 2",; _sys.stdout.flush()
        #    fig = set_fig_qtys("directLongSeqGSTDeviationColorBoxPlot",
        #                       "directLongSeqGSTDeviationBoxes.pdf",W=4,H=5)
        #
        #    if verbosity > 0: 
        #        print ""; _sys.stdout.flush()
        #
        #
        #    #Small eigenvalue error rate
        #    if verbosity > 0: 
        #        print " -- Error rate plots..."; _sys.stdout.flush()
        #    fig = set_fig_qtys("smallEigvalErrRateColorBoxPlot",
        #                       "smallEigvalErrRateBoxes.pdf",W=4,H=5)
        #else:
        #    #UNUSED: "directLGSTColorBoxPlot", "directLGSTDeviationColorBoxPlot"
        #    for figkey in ["directLongSeqGSTColorBoxPlot",
        #                   "directLongSeqGSTDeviationColorBoxPlot",
        #                   "smallEigvalErrRateColorBoxPlot"]:
        #        qtys[figkey] = qtys["tt_"+figkey] = ""


        whackamoleplots = ""
#        if whackamoleAppendix:    
#            #Whack-a-mole plots for highest L of each length-1 germ
#            Ls = self.parameters['max length list']
#            highestL = Ls[-1]; allGateStrings = self.gatestring_lists['all']
#            hammerWeight = 10.0
#            len1Germs = [ g for g in self.gatestring_lists['germs'] 
#                          if len(g) == 1 ]
#
#            if verbosity > 0: 
#                print " -- Whack-a-mole plots (%d): " % (2*len(len1Germs)),
#                _sys.stdout.flush()
#
#            for i,germ in enumerate(len1Germs):
#                if verbosity > 0: 
#                    print "%d " % (i+1),; _sys.stdout.flush()
#
#                fig = self.figures.get("whack%sMoleBoxes" % germ[0],verbosity=v)
#                fig.save_to(_os.path.join(report_dir, D,"whack%sMoleBoxes.pdf"
#                                          % germ[0]))
#        
#                whackamoleplots += "\n"
#                whackamoleplots += "\\begin{figure}\n"
#                whackamoleplots += "\\begin{center}\n"
#                whackamoleplots += "\\includegraphics[width=%.2fin,height=%.2fin,keepaspectratio]{%s/whack%sMoleBoxes.pdf}\n" % (maxW,maxH,D,germ[0])
#                whackamoleplots += "\\caption{Whack-a-%s-mole box plot for $\mathrm{%s}^{%d}$." % (plotFnLatex,germ[0],highestL)
#                #TODO: add conditional tooltip string to start of caption
#                whackamoleplots += "  Hitting with hammer of weight %.1f.\label{Whack%sMoleBoxPlot}}\n" % (hammerWeight,germ[0])
#                whackamoleplots += "\\end{center}\n"
#                whackamoleplots += "\\end{figure}\n"
#        
#            for i,germ in enumerate(len1Germs):
#                if verbosity > 0: 
#                    print "%d " % (len(len1Germs)+i+1),; _sys.stdout.flush()
#
#                fig = self.figures.get("whack%sMoleBoxesSummed" % germ[0],
#                                       verbosity=v)
#                fig.save_to(_os.path.join(report_dir, D,"whack%sMoleBoxesSummed.pdf" % germ[0]))
#    
#                whackamoleplots += "\n"
#                whackamoleplots += "\\begin{figure}\n"
#                whackamoleplots += "\\begin{center}\n"
#                whackamoleplots += "\\includegraphics[width=4in,height=5in,keepaspectratio]{%s/whack%sMoleBoxesSummed.pdf}\n" % (D,germ[0])
#                whackamoleplots += "\\caption{Whack-a-%s-mole box plot for $\mathrm{%s}^{%d}$, summed over fiducial matrix." % (plotFnLatex,germ[0],highestL)
#                #TODO: add conditional tooltip string to start of caption
#                whackamoleplots += "  Hitting with hammer of weight %.1f.\label{Whack%sMoleBoxPlotSummed}}\n" % (hammerWeight,germ[0])
#                whackamoleplots += "\\end{center}\n"
#                whackamoleplots += "\\end{figure}\n"
#    
#            if verbosity > 0: 
#                print ""; _sys.stdout.flush()

        #Set template quantity (empty string if appendix disabled)
        qtys['whackamole_plot_figures'] = whackamoleplots
            
        if bWasInteractive:
            _matplotlib.pyplot.ion()
    

        # 3) populate template latex file => report latex file
        if verbosity > 0: 
            print "*** Merging into template file ***"; _sys.stdout.flush()
        
        mainTexFilename = _os.path.join(report_dir, report_base + ".tex")
        appendicesTexFilename = _os.path.join(report_dir, report_base + "_appendices.tex")
        pdfFilename = _os.path.join(report_dir, report_base + ".pdf")
    
        mainTemplate = "report_general_main.tex"
        #if self.parameters['objective'] == "chi2":    
        #    mainTemplate = "report_chi2_main.tex"
        #    appendicesTemplate = "report_chi2_appendices.tex"
        #elif self.parameters['objective'] == "logl":
        #    mainTemplate = "report_logL_main.tex"
        #    appendicesTemplate = "report_logL_appendices.tex"
        #else: 
        #    raise ValueError("Invalid objective value: %s" 
        #                     % self.parameters['objective'])
    
        #if any( (debugAidsAppendix, gaugeOptAppendix,
        #         pixelPlotAppendix, whackamoleAppendix) ):
        #    qtys['appendices'] = "\\input{%s}" % \
        #        _os.path.basename(appendicesTexFilename)
        #    self._merge_template(qtys, appendicesTemplate,
        #                         appendicesTexFilename)
        #else: qtys['appendices'] = ""
        self._merge_template(qtys, mainTemplate, mainTexFilename)
    
    
        # 4) compile report latex file into PDF
        if verbosity > 0: 
            print "Latex file(s) successfully generated.  Attempting to compile with pdflatex..."; _sys.stdout.flush()
        cwd = _os.getcwd()
        if len(report_dir) > 0:  
            _os.chdir(report_dir)
    
        try:
            ret = _os.system( "%s %s %s" % 
                              (self.options.latex_cmd,
                               _os.path.basename(mainTexFilename),
                               self.options.latex_postcmd) )
            if ret == 0:
                #We could check if the log file contains "Rerun" in it, 
                # but we'll just re-run all the time now
                if verbosity > 0: 
                    print "Initial output PDF %s successfully generated." \
                        % pdfFilename

                ret = _os.system( "%s %s %s" % 
                                  (self.options.latex_cmd,
                                   _os.path.basename(mainTexFilename),
                                   self.options.latex_postcmd) )
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
            print "Error trying to run pdflatex to generate output PDF %s. Is '%s' on your path?" % (pdfFilename,self.options.latex_cmd)
        finally: 
            _os.chdir(cwd)
    
        return


    #FUTURE?
    #def create_brief_html_page(self, tableClass):
    #    pass
    #
    #def create_full_html_page(self, tableClass):
    #    pass



class ResultOptions(object):
    """ Class encapsulating the display options of a Results instance """
    def __init__(self):
        self.long_tables = False
        self.table_class = "pygstiTbl"
        self.template_path = "."
        self.latex_cmd = "pdflatex"
        if _os.path.exists("/dev/null"):
            self.latex_postcmd = "-halt-on-error </dev/null >/dev/null"
        else:
            self.latex_postcmd = "" #no /dev/null, so probably not Unix,
                                    #so don't assume halt-on-error works either.

    def describe(self,prefix):
        s = ""
        s += prefix + ".long_tables    -- long latex tables?  %s\n" \
            % str(self.long_tables)
        s += prefix + ".table_class    -- HTML table class = %s\n" \
            % str(self.table_class)
        s += prefix + ".template_path  -- pyGSTi templates path = '%s'\n" \
            % str(self.template_path)
        s += prefix + ".latex_cmd      -- latex compiling command = '%s'\n" \
            % str(self.latex_cmd)
        s += prefix + ".latex_postcmd  -- latex compiling command postfix = '%s'\n" \
            % str(self.latex_postcmd)
        return s


    def __str__(self):
        s  = "Display options:\n"
        s += self.describe("  ")
        return s


def _to_pdfinfo(list_of_keyval_tuples):

    def sanitize(val):
        if type(val) in (list,tuple):
            sanitized_val = "[" + ", ".join([sanitize(el) 
                                             for el in val]) + "]"
        elif type(val) in (dict,_collections.OrderedDict):
            sanitized_val = "Dict[" + \
                ", ".join([ "%s: %s" % (sanitize(k),sanitize(v)) for k,v
                            in val.iteritems()]) + "]"
        else:
            sanitized_val = sanitize_str( str(val) )
        return sanitized_val

    def sanitize_str(s):
        ret = s.replace("^","")
        ret = ret.replace("(","[")
        ret = ret.replace(")","]")
        return ret

    def sanitize_key(s):
        #More stringent string replacement for keys
        ret = s.replace(" ","_")
        ret = ret.replace("^","")
        ret = ret.replace(",","_")
        ret = ret.replace("(","[")
        ret = ret.replace(")","]")
        return ret


    sanitized_list = []
    for key,val in list_of_keyval_tuples:
        sanitized_key = sanitize_key(key)
        sanitized_val = sanitize(val)
        sanitized_list.append( (sanitized_key, sanitized_val) )

    return ",\n".join( ["%s={%s}" % (key,val) for key,val in sanitized_list] )
