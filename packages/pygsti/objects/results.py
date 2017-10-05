from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines the Results class."""

import collections as _collections
import itertools   as _itertools
import warnings    as _warnings
import time        as _time
import numpy       as _np

from .. import tools as _tools
from .gatestringstructure import LsGermsStructure as _LsGermsStructure
from .estimate import Estimate as _Estimate

#A flag to enable fast-loading of old results files (should
# only be changed by experts)
_SHORTCUT_OLD_RESULTS_LOAD = False        

class Results(object):
    """
    Encapsulates a set of related GST estimates.

    A Results object is a container which associates a single `DataSet` and a
    structured set of gate sequences (usually the experiments contained in the
    data set) with a set of estimates.  Each estimate (`Estimate` object) contains
    gate sets as well as parameters used to generate those inputs.  Associated
    `ConfidenceRegion` objects, because they are associated with a set of gate
    sequences, are held in the `Results` object but are associated with estimates.

    Typically, each `Estimate` is related to the input & output of a single
    GST calculation performed by a high-level driver routine like
    :func:`do_long_sequence_gst`.
    """

    def __init__(self):
        """
        Initialize an empty Results object.
        """

        #Dictionaries of inputs & outputs
        self.dataset = None
        self.gatestring_lists = _collections.OrderedDict()
        self.gatestring_structs = _collections.OrderedDict()
        self.estimates = _collections.OrderedDict()


    def init_dataset(self, dataset):
        """
        Initialize the (single) dataset of this `Results` object.

        Parameters
        ----------
        dataset : DataSet
            The dataset used to construct the estimates found in this
            `Results` object.

        Returns
        -------
        None
        """
        if self.dataset is not None:
            _warnings.warn(("Re-initializing the dataset of a Results object!"
                            "  Usually you don't want to do this."))
        self.dataset = dataset

        
    def init_gatestrings(self, structsByIter):
        """
        Initialize the common set gate sequences used to form the 
        estimates of this Results object.

        There is one such set per GST iteration (if a non-iterative
        GST method was used, this is treated as a single iteration).

        Parameters
        ----------
        structsByIter : list
            The gate strings used at each iteration. Ideally, elements are
            `LsGermsStruct` objects, which contain the structure needed to 
            create color box plots in reports.  Elements may also be 
            unstructured lists of gate sequences (but this may limit
            the amount of data visualization one can perform later).

        Returns
        -------
        None
        """
        if len(self.gatestring_structs) > 0:
            _warnings.warn(("Re-initializing the gate sequences of a Results"
                            " object!  Usually you don't want to do this."))
        
        #Set gatestring structures
        self.gatestring_structs['iteration'] = []
        for gss in structsByIter:
            if isinstance(gss, _LsGermsStructure):
                self.gatestring_structs['iteration'].append(gss)
            elif isinstance(gss, list):
                unindexed_gss = _LsGermsStructure([],[],[],[],None)
                unindexed_gss.add_unindexed(gss)
                self.gatestring_structs['iteration'].append(unindexed_gss)
            else:
                raise ValueError("Unknown type of gate string specifier: %s"
                                 % str(type(gss)))
                
        self.gatestring_structs['final'] = \
                self.gatestring_structs['iteration'][-1]

        #Extract raw gatestring lists from structs
        self.gatestring_lists['iteration'] = \
                [ gss.allstrs for gss in self.gatestring_structs['iteration'] ]
        self.gatestring_lists['final'] = self.gatestring_lists['iteration'][-1]
        self.gatestring_lists['all'] = _tools.remove_duplicates(
            list(_itertools.chain(*self.gatestring_lists['iteration'])) )
        
        running_lst = []; delta_lsts = []
        for lst in self.gatestring_lists['iteration']:
            delta_lst = [ x for x in lst if (x not in running_lst) ]
            delta_lsts.append(delta_lst); running_lst.extend(delta_lst) 
        self.gatestring_lists['iteration delta'] = delta_lsts # *added* at each iteration

        #Set "Ls and germs" info: gives particular structure
        # to the gateStringLists used to obtain estimates
        finalStruct = self.gatestring_structs['final']
        self.gatestring_lists['prep fiducials'] = finalStruct.prepStrs
        self.gatestring_lists['effect fiducials'] = finalStruct.effectStrs
        self.gatestring_lists['germs'] = finalStruct.germs


    def add_estimates(self, results, estimatesToAdd=None):
        """
        Add some or all of the estimates from `results` to this `Results` object.

        Parameters
        ----------
        results : Results
            The object to import estimates from.  Note that this object must contain
            the same data set and gate sequence information as the importing object
            or an error is raised.

        estimatesToAdd : list, optional
            A list of estimate keys to import from `results`.  If None, then all
            the estimates contained in `results` are imported.

        Returns
        -------
        None
        """
        if self.dataset is None:
            raise ValueError(("The data set must be initialized"
                              "*before* adding estimates"))

        if 'iteration' not in self.gatestring_structs:
            raise ValueError(("Gate sequences must be initialized"
                              "*before* adding estimates"))

        assert(results.dataset is self.dataset), "DataSet inconsistency: cannot import estimates!"
        assert(len(self.gatestring_structs['iteration']) == len(results.gatestring_structs['iteration'])), \
            "Iteration count inconsistency: cannot import estimates!"

        for estimate_key in results.estimates:
            if estimatesToAdd is None or estimate_key in estimatesToAdd:
                if estimate_key in self.estimates:
                    _warnings.warn("Re-initializing the %s estimate" % estimate_key
                                   + " of this Results object!  Usually you don't"
                                   + " want to do this.")
                self.estimates[estimate_key] = results.estimates[estimate_key]

        
    def add_estimate(self, targetGateset, seedGateset, gatesetsByIter,
                     parameters, estimate_key='default'):
        """
        Add a set of `GateSet` estimates to this `Results` object.

        Parameters
        ----------
        targetGateset : GateSet
            The target gateset used when optimizing the objective.

        seedGateset : GateSet
            The initial gateset used to seed the iterative part
            of the objective optimization.  Typically this is
            obtained via LGST.

        gatesetsByIter : list of GateSets
            The estimated gateset at each GST iteration. Typically these are the
            estimated gate sets *before* any gauge optimization is performed.

        parameters : dict
            A dictionary of parameters associated with how this estimate
            was obtained.

        estimate_key : str, optional
            The key or label used to identify this estimate.

        Returns
        -------
        None
        """
        if self.dataset is None:
            raise ValueError(("The data set must be initialized"
                              "*before* adding estimates"))

        if 'iteration' not in self.gatestring_structs:
            raise ValueError(("Gate sequences must be initialized"
                              "*before* adding estimates"))

        assert(len(self.gatestring_structs['iteration']) == len(gatesetsByIter))

        if estimate_key in self.estimates:
            _warnings.warn("Re-initializing the %s estimate" % estimate_key
                           + " of this Results object!  Usually you don't"
                           + " want to do this.")

        self.estimates[estimate_key] = _Estimate(self, targetGateset, seedGateset,
                                                gatesetsByIter, parameters)

        #Set gate sequence related parameters inherited from Results
        self.estimates[estimate_key].parameters['max length list'] = \
                                        self.gatestring_structs['final'].Ls


    def copy(self):
        """ Creates a copy of this Results object. """
        #TODO: check whether this deep copies (if we want it to...) - I expect it doesn't currently
        cpy = Results()
        cpy.dataset = self.dataset.copy()
        cpy.gatestring_lists = self.gatestring_lists.copy()
        cpy.gatestring_structs = self.gatestring_structs.copy()
        cpy.estimates = self.estimates.copy()
        return cpy


    def __setstate__(self, stateDict):
        if '_bEssentialResultsSet' in stateDict:
            #Then we're unpickling an old-version Results object
            print("NOTE: you're loading a prior-version Results object from a pickle."
                  " This Results object will been updated, and while most data will"
                  " be transferred seamlessly, there may be some saved values which"
                  " are not imported. Please re-save (or re-pickle) this upgraded object"
                  " to avoid seeing this message, or re-run the analysis leading to "
                  " these results to create a new current-version Results object.")
            
            params = _collections.OrderedDict()
            goparams = _collections.OrderedDict()
            for k,v in stateDict['parameters'].items():
                if k != 'gaugeOptParams': params[k] = v
                elif isinstance(v,list) and len(v) == 1:
                    goparams['go0'] = v[0]
                else:
                    goparams['go0'] = v

            gstrStructs = _collections.OrderedDict()
            if _SHORTCUT_OLD_RESULTS_LOAD == False:
                from ..construction import make_lsgst_structs as _make_lsgst_structs
                try:
                    prepStrs = stateDict['gatestring_lists']['prep fiducials']
                    effectStrs = stateDict['gatestring_lists']['effect fiducials']
                    germs = stateDict['gatestring_lists']['germs']
                    aliases = stateDict['parameters'].get('gateLabelAliases',None)
                    fidPairs = stateDict['parameters'].get('fiducial pairs',None)
                    maxLengthList = stateDict['parameters']['max length list']
                    if maxLengthList[0] == 0:
                        maxLengthList = maxLengthList[1:] #Fine; includeLGST is always True below
                
                    structs = _make_lsgst_structs(stateDict['gatesets']['target'].gates.keys(),
                                                        prepStrs, effectStrs, germs, maxLengthList,
                                                        fidPairs, truncScheme="whole germ powers",
                                                        nest=True, keepFraction=1, keepSeed=None,
                                                        includeLGST=True, gateLabelAliases=aliases)
                except:
                    print("Warning: Ls & germs structure not found.  Loading unstructured Results.")
                    structs = []
                    for lst in stateDict['gatestring_lists']['iteration']:
                        unindexed_gss = _LsGermsStructure([],[],[],[],None)
                        unindexed_gss.add_unindexed(lst)
                        structs.append(unindexed_gss)
                        
                gstrStructs['iteration'] = structs
                gstrStructs['final'] = structs[-1]

            gstrLists = _collections.OrderedDict()
            for k,v in stateDict['gatestring_lists'].items():
                gstrLists[k] = v

            gatesets =  _collections.OrderedDict()
            for k,v in stateDict['gatesets'].items():
                if k == 'final estimate':
                    gatesets['go0'] = v
                elif k == 'iteration estimates pre gauge opt':
                    gatesets['iteration estimates'] = v
                else: gatesets[k] = v
            gatesets['final iteration estimate'] = gatesets['iteration estimates'][-1]

            estimate = _Estimate(self, gatesets['target'], gatesets['seed'],
                                gatesets['iteration estimates'], params)
            if 'go0' in gatesets:
                estimate.add_gaugeoptimized(goparams.get('go0',{}), gatesets['go0'])
                
            filteredDict = {
                'dataset': stateDict['dataset'],
                'gatestring_lists': gstrLists,
                'gatestring_structs': gstrStructs,
                'estimates': _collections.OrderedDict( [('default',estimate)] ),
            }
            self.__dict__.update(filteredDict)
        else:
            #unpickle normally
            self.__dict__.update(stateDict)
            for est in self.estimates.values():
                est.set_parent(self)
                

    def __str__(self):
        s  = "----------------------------------------------------------\n"
        s += "---------------- pyGSTi Results Object -------------------\n"
        s += "----------------------------------------------------------\n"
        s += "\n"
        s += "How to access my contents:\n\n"
        s += " .dataset    -- the DataSet used to generate these results\n\n"
        s += " .gatestring_lists   -- a dict of GateString lists w/keys:\n"
        s += " ---------------------------------------------------------\n"
        s += "  " + "\n  ".join(list(self.gatestring_lists.keys())) + "\n"
        s += "\n"
        s += " .gatestring_structs   -- a dict of GatestringStructures w/keys:\n"
        s += " ---------------------------------------------------------\n"
        s += "  " + "\n  ".join(list(self.gatestring_structs.keys())) + "\n"
        s += "\n"
        s += " .estimates   -- a dictionary of Estimate objects:\n"
        s += " ---------------------------------------------------------\n"
        s += "  " + "\n  ".join(list(self.estimates.keys())) + "\n"
        s += "\n"
        return s


    #OLD Methods for generating reports which have been removed - show alert
    # message directing users to new factory functions
    def create_full_report_pdf(self, confidenceLevel=None, filename="auto",
                               title="auto", datasetLabel="auto", suffix="",
                               debugAidsAppendix=False, gaugeOptAppendix=False,
                               pixelPlotAppendix=False, whackamoleAppendix=False,
                               pureDataAppendix=False,  m=0, M=10, tips=False,
                               verbosity=0, comm=None):
        _warnings.warn(
            ('create_full_report_pdf(...) has been removed from pyGSTi.\n'
             '  Starting in version 0.9.4, pyGSTi\'s PDF reports have been\n'
             '  replaced with (better) HTML ones. As a part of this change,\n'
             '  the functions that generate reports are now separate functions.\n'
             '  Please update this call with one to:\n'
             '  pygsti.report.create_general_report(...)\n'))

    def create_brief_report_pdf(self, confidenceLevel=None,
                                filename="auto", title="auto", datasetLabel="auto",
                                suffix="", m=0, M=10, tips=False, verbosity=0,
                                comm=None):
        _warnings.warn(
            ('create_brief_report_pdf(...) has been removed from pyGSTi.\n'
             '  Starting in version 0.9.4, pyGSTi\'s PDF reports have been\n'
             '  replaced with (better) HTML ones. As a part of this change,\n'
             '  the functions that generate reports are now separate functions.\n'
             '  Please update this call with one to:\n'
             '  pygsti.report.create_general_report(...)\n'))

    def create_presentation_pdf(self, confidenceLevel=None, filename="auto",
                                title="auto", datasetLabel="auto", suffix="",
                                debugAidsAppendix=False,
                                pixelPlotAppendix=False, whackamoleAppendix=False,
                                m=0, M=10, verbosity=0, comm=None):
        _warnings.warn(
            ('create_presentation_pdf(...) has been removed from pyGSTi.\n'
             '  Starting in version 0.9.4, pyGSTi\'s PDF reports have been\n'
             '  replaced with (better) HTML ones. As a part of this change,\n'
             '  Beamer presentations have been removed.  Please try using\n'
             '  pygsti.report.create_general_report(...)\n'))

    def create_presentation_ppt(self, confidenceLevel=None, filename="auto",
                            title="auto", datasetLabel="auto", suffix="",
                            debugAidsAppendix=False,
                            pixelPlotAppendix=False, whackamoleAppendix=False,
                            m=0, M=10, verbosity=0, pptTables=False, comm=None):
        _warnings.warn(
            ('create_presentation_ppt(...) has been removed from pyGSTi.\n'
             '  Starting in version 0.9.4, pyGSTi\'s PDF reports have been\n'
             '  replaced with (better) HTML ones. As a part of this change,\n'
             '  Powerpoint presentations have been removed.  Please try using\n'
             '  pygsti.report.create_general_report(...)\n'))

    def create_general_report_pdf(self, confidenceLevel=None, filename="auto",
                                  title="auto", datasetLabel="auto", suffix="",
                                  tips=False, verbosity=0, comm=None,
                                  showAppendix=False):
        _warnings.warn(
            ('create_general_report_pdf(...) has been removed from pyGSTi.\n'
             '  Starting in version 0.9.4, pyGSTi\'s PDF reports have been\n'
             '  replaced with (better) HTML ones. As a part of this change,\n'
             '  the functions that generate reports are now separate functions.\n'
             '  Please update this call with one to:\n'
             '  pygsti.report.create_general_report(...)\n'))


def enable_old_python_results_unpickling():
    
    #Define empty ResultCache class in resultcache module to enable loading old Results pickles
    import sys as _sys
    class dummy_ResultCache(object): pass
    class dummy_ResultOptions(object): pass
    class dummy_resultcache_module(object):
        def __init__(self):
            self.ResultCache = dummy_ResultCache
    _sys.modules[__name__].ResultOptions = dummy_ResultOptions
    _sys.modules['pygsti.report.resultcache'] = dummy_resultcache_module()
    _sys.modules['pygsti.report.results'] = _sys.modules[__name__]
