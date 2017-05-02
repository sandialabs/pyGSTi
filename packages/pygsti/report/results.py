from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines the Results class."""

import collections as _collections
import itertools   as _itertools

from ..             import construction         as _const
from ..             import objects              as _objs
from ..algorithms   import gaugeopt_to_target   as _gaugeopt_to_target
from ..tools        import listtools            as _lt

class Results(object):
    """
    Encapsulates a set of GST results.

    A Results object is a container which associates a single `DataSet` and a
    dictionary of input "parameters" with a set of output `GateSet` and 
    `ConfidenceRegion` objects.

    Typically, these quantities are related to the input & output of a single
    GST calculation performed by a high-level driver routine like
    :func:`do_long_sequence_gst`.  A `Results` can also, however, be used 
    generally as a way of associating some arbitrary set of parameters with
    a set of gate sets and/or confidence regions.
    """

    def __init__(self):
        """
        Initialize an empty Results object.
        """

        #Dictionaries of inputs & outputs
        self.dataset = None
        self.parameters = _collections.OrderedDict()
        self.goparameters = _collections.OrderedDict()
        self.gatestring_lists = _collections.OrderedDict()
        self.gatestring_structs = _collections.OrderedDict()
        self.gatesets = _collections.OrderedDict()
        self.confidence_regions = _collections.OrderedDict()
          #usually, setting a confidence_region key similar or the same
          # as a gatesets key is useful


    def init_Ls_and_germs(self, objective, dataset, targetGateset,
                          seedGateset, gatesetsByIter,
                          gateStringStructsByIter):
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

        dataset : DataSet
            The dataset used when optimizing the objective.

        targetGateset : GateSet
            The target gateset used when optimizing the objective.

        seedGateset : GateSet
            The initial gateset used to seed the iterative part
            of the objective optimization.  Typically this is
            obtained via LGST.

        gatesetsByIter : list of GateSets
            The estimated gateset at each GST iteration. Typically these are the
            estimated gate sets *before* any gauge optimization is performed.

        gateStringStructsByIter : list of lists or LsGermsStructs
            The gate string list or structure used at each iteration.
            If the gate sequences are unstructured, you can just pass
            lists of `GateString` objects instead (but this will limit
            the amount of data visualization you can perform later).

        Returns
        -------
        None
        """

        assert(len(gateStringStructsByIter) == len(gatesetsByIter))

        #Set gatesets
        self.gatesets['target'] = targetGateset
        self.gatesets['seed'] = seedGateset
        self.gatesets['iteration estimates'] = gatesetsByIter
        self.gatesets['final iteration estimate'] = gatesetsByIter[-1]

        #Set gatestring structures
        self.gatestring_structs['iteration'] = []
        for gss in gateStringStructsByIter:
            if isinstance(gss, _objs.LsGermsStructure):
                self.gatestring_structs['iteration'].append(gss)
            else: #assume gss is a raw list of GateStrings
                unindexed_gss = _objs.LsGermsStructure([],[],[],[],None)
                unindexed_gss.add_unindexed(gss)
                self.gatestring_structs['iteration'].append(unindexed_gss)
                
        self.gatestring_structs['final'] = \
                self.gatestring_structs['iteration'][-1]

        #Extract raw gatestring lists from structs
        self.gatestring_lists['iteration'] = \
                [ gss.allstrs for gss in self.gatestring_structs['iteration'] ]
        self.gatestring_lists['final'] = self.gatestring_lists['iteration'][-1]
        self.gatestring_lists['all'] = _lt.remove_duplicates(
            list(_itertools.chain(*self.gatestring_lists['iteration'])) )
        
        running_lst = []; delta_lsts = []
        for lst in self.gatestring_lists['iteration']:
            delta_lst = [ x for x in lst if (x not in running_lst) ]
            delta_lsts.append(delta_lst); running_lst.extend(delta_lst) 
        self.gatestring_lists['iteration delta'] = delta_lsts # *added* at each iteration

        self.dataset = dataset
        self.parameters['objective'] = objective

        #Set "Ls and germs" info: gives particular structure
        # to the gateStringLists used to obtain estimates
        finalStruct = self.gatestring_structs['final']
        self.gatestring_lists['prep fiducials'] = finalStruct.prepStrs
        self.gatestring_lists['effect fiducials'] = finalStruct.effectStrs
        self.gatestring_lists['germs'] = finalStruct.germs
        self.parameters['max length list'] = finalStruct.Ls
        


    def add_gaugeoptimized(self, goparams, gateset=None, label=None):
        """
        Adds a gauge-optimized GateSet (computing it if needed) to this object.

        Parameters
        ----------
        goparams : dict
            A dictionary of gauge-optimization parameters, typically arguments
            to :func:`gaugeopt_to_target`, specifying how the gauge optimization
            was (or should be) performed.  When `gateset` is `None` (and this
            function computes the gate set internally) the keys and values of
            this dictionary must correspond to allowed arguments of 
            :func:`gaugeopt_to_target`. By default, :func:`gaugeopt_to_target`'s
            first two arguments, the `GateSet` to optimize and the target,
            are taken to be `self.gatesets['final iteration estimate']` and 
            self.gatesets['target'].

        gateset : GateSet, optional
            The gauge-optimized gate set to store.  If None, then this gate set
            is computed by calling :func:`gaugeopt_to_target` with the contents
            of `goparams` as arguments as described above.

        label : str, optional
            A label for this gauge-optimized gate set, used as the key in
            this object's `gatesets` and `goparameters` member dictionaries.
            If None, then the next available "go<X>", where <X> is a 
            non-negative integer, is used as the label.

        Returns
        -------
        None
        """

        if label is None:
            i = 0
            while True:
                label = "go%d" % i; i += 1
                if (label not in self.goparameters) and \
                   (label not in self.gatesets): break

        if gateset is None:
            goparams = goparams.copy() #so we don't change the caller's dict
            
            if "gateset" not in goparams:
                if 'final iteration estimate' in self.gatesets:
                    goparams["gateset"] = self.gatesets['final iteration estimate']
                else: raise ValueError("Must supply 'gateset' in 'goparams' argument")
                
            if "targetGateset" not in goparams:
                if 'target' in self.gatesets:
                    goparams["targetGateset"] = self.gatesets['target']
                else: raise ValueError("Must supply 'targetGateset' in 'goparams' argument")

            gateset = _gaugeopt_to_target(**goparams)


        #sort te parameters by name for consistency
        ordered_goparams = _collections.OrderedDict( 
            [(k,goparams[k]) for k in sorted(list(goparams.keys()))])

        self.gatesets[label] = gateset
        self.goparameters[label] = ordered_goparams


    def copy(self):
        """ Creates a copy of this Results object. """
        cpy = Results()
        cpy.dataset = self.dataset.copy()
        cpy.parameters = self.parameters.copy()
        cpy.goparameters = self.goparameters.copy()
        cpy.gatestring_lists = self.gatestring_lists.copy()
        cpy.gatestring_structs = self.gatestring_structs.copy()
        cpy.gatesets = self.gatesets.copy()
        cpy.confidence_regions = self.confidence_regions.copy()
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
                else: goparams['go0'] = v

            gstrStructs = _collections.OrderedDict()
            #COMMENTED TO re-run gatestring reduction analysis - TODO: re un-comment these lines
            #try:
            #    prepStrs = stateDict['gatestring_lists']['prep fiducials']
            #    effectStrs = stateDict['gatestring_lists']['effect fiducials']
            #    germs = stateDict['gatestring_lists']['germs']
            #    aliases = stateDict['parameters'].get('gateLabelAliases',None)
            #    fidPairs = stateDict['parameters'].get('fiducial pairs',None)
            #    maxLengthList = stateDict['parameters']['max length list']
            #    if maxLengthList[0] == 0:
            #        maxLengthList = maxLengthList[1:] #Fine; includeLGST is always True below
            #
            #    structs = _const.make_lsgst_structs(stateDict['gatesets']['target'].gates.keys(),
            #                                        prepStrs, effectStrs, germs, maxLengthList,
            #                                        fidPairs, truncScheme="whole germ powers",
            #                                        nest=True, keepFraction=1, keepSeed=None,
            #                                        includeLGST=True, gateLabelAliases=aliases)
            #except:
            #    print("Warning: Ls & germs structure not found.  Loading unstructured Results.")
            #    structs = []
            #    for lst in stateDict['gatestring_lists']['iteration']:
            #        unindexed_gss = _objs.LsGermsStructure([],[],[],[],None)
            #        unindexed_gss.add_unindexed(lst)
            #        structs.append(unindexed_gss)
            #        
            #gstrStructs['iteration'] = structs
            #gstrStructs['final'] = structs[-1]

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
                
            filteredDict = {
                'dataset': stateDict['dataset'],
                'parameters': params,
                'goparameters': goparams,
                'gatestring_lists': gstrLists,
                'gatestring_structs': gstrStructs,
                'gatesets': gatesets,
                'confidence_regions' : _collections.OrderedDict() #don't convert
            }
            self.__dict__.update(filteredDict)
        else:
            #unpickle normally
            self.__dict__.update(stateDict)

    def __str__(self):
        s  = "----------------------------------------------------------\n"
        s += "---------------- pyGSTi Results Object -------------------\n"
        s += "----------------------------------------------------------\n"
        s += "\n"
        s += "How to access my contents:\n\n"
        s += " .dataset    -- the DataSet used to generate these results\n\n"
        s += " .gatesets   -- a dictionary of GateSet objects w/keys:\n"
        s += " ---------------------------------------------------------\n"
        s += "  " + "\n  ".join(list(self.gatesets.keys())) + "\n"
        s += "\n"
        s += " .gatestring_lists   -- a dict of GateString lists w/keys:\n"
        s += " ---------------------------------------------------------\n"
        s += "  " + "\n  ".join(list(self.gatestring_lists.keys())) + "\n"
        s += "\n"
        s += " .gatestring_structs   -- a dict of GatestringStructures w/keys:\n"
        s += " ---------------------------------------------------------\n"
        s += "  " + "\n  ".join(list(self.gatestring_structs.keys())) + "\n"
        s += "\n"
        s += " .parameters   -- a dictionary of simulation parameters:\n"
        s += " ---------------------------------------------------------\n"
        s += "  " + "\n  ".join(list(self.parameters.keys())) + "\n"
        s += "\n"
        s += " .goparameters   -- a dictionary of gauge-optimization parameter dictionaries:\n"
        s += " ---------------------------------------------------------\n"
        s += "  " + "\n  ".join(list(self.goparameters.keys())) + "\n"
        s += "\n"
        return s


    def get_confidence_region(self, gateset_key="go0", gatestrings_key="final",
                              confidenceLevel=95, label=None, forcecreate=False,
                              comm=None):
        """
        Get the ConfidenceRegion object associated with a given gate set and
        confidence level.

        If such a region already exists within this Result object's 
        `confidence_regions` dictionary (dictated by `label`) then it will be
        returned instead of creating a region (unless `forcecreate == True`).
        If a new region is constructed it will automatically be added to this
        object's `confidence_regions` dictionary with the key `label`.

        This function is essentailly a wrapper which maps values from the
        `parameters` dictionary of this object to the arguments of 
        :func:`logl_confidence_region` or :func:`chi2_confidence_region`
        based on the value of `parameters['objective']`.  Namely, the values of
        'objective', 'probClipInterval', 'minProbClip', 'minProbClipForWeighting',
        'radius', 'hessianProjection', 'memlimit', 'cptpPentaltyFactor', and 
        'distributeMethod' are used. If you find yourself having to set values
        in the `parameters` dictionary *just* to call this function, you should
        probably be calling one of the aforementioned functions directly instead
        of this one.

        Parameters
        ----------
        gateset_key : str, optional
            The key in `self.gatesets` of the `GateSet` to retrieve or create
            a `ConfidenceRegion` for.

        gatestrings_key : str, optional
            The key in `self.gatestring_structs` (attempted first) or 
            `self.gatestring_lists` (attempted second) specifying the list
            of gatestrings to create create a `ConfidenceRegion` for.

        confidenceLevel : int, optional
            The confidence level (between 0 and 100) for normal confidence
            regions or a *negative* integer (between -1 and -100) for 
            "non-Markovian error bar" regions.

        label : str, optional
            The label to give this confidence region.  If None, then
            `gateset_key + "." + gatestrings_key + "." + str(confidenceLevel)`
            is taken to be the label.

        forcecreate : bool, optional
            If True, then a new region will always be created, even if one
            already exists.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        Returns
        -------
        ConfidenceRegion or None
            returns None if `confidenceLevel` is None.
        """

        if confidenceLevel is None:
            return None

        crkey = label if (label is not None) else \
                gateset_key + "." + gatestrings_key + "." + str(confidenceLevel)
        gateset = self.gatesets[gateset_key]
        if gatestrings_key in self.gatestring_structs:
            gatestrings = self.gatestring_structs[gatestrings_key].allstrs
            aliases = self.gatestring_structs[gatestrings_key].aliases
        elif gatestrings_key in self.gatestring_lists:
            gatestrings = self.gatestring_lists[gatestrings_key]
            aliases = None
        else: raise ValueError("key '%s' not found in " % gatestrings_key +
                               "gatestring_structs or gatestring_lists")
        
        if forcecreate or (crkey not in self.confidence_regions):

            #Negative confidence levels ==> non-Markovian error bars
            if confidenceLevel < 0:
                actual_confidenceLevel = -confidenceLevel
                regionType = "non-markovian"
            else:
                actual_confidenceLevel = confidenceLevel
                regionType = "std"

            if self.parameters['objective'] == "logl":
                cr = _const.logl_confidence_region(
                    gateset, self.dataset, actual_confidenceLevel, gatestrings,
                    self.parameters.get('probClipInterval',(-1e6,1e6)),
                    self.parameters.get('minProbClip',1e-4),
                    self.parameters.get('radius',1e-4),
                    self.parameters.get('hessianProjection','optimal gate CIs'),
                    regionType, comm, self.parameters.get('memLimit',None),
                    self.parameters.get('cptpPenaltyFactor',0.0),
                    self.parameters.get('distributeMethod','deriv'),
                    aliases)
            elif self.parameters['objective'] == "chi2":
                cr = _const.chi2_confidence_region(
                    gateset, self.dataset, actual_confidenceLevel, gatestrings,
                    self.parameters.get('probClipInterval',(-1e6,1e6)),
                    self.parameters.get('minProbClipForWeighting',1e-4),
                    self.parameters.get('hessianProjection','optimal gate CIs'),
                    regionType, comm, self.parameters.get('memLimit',None),
                    aliases)
            else:
                raise ValueError("Invalid objective given in essential" +
                                 " info: %s" % self.parameters['objective'])

            self.confidence_regions[crkey] = cr
            
        return self.confidence_regions[crkey]


class ResultOptions(object):
    """ Unused.  Exists for sole purpose of loading old Results pickles """
    pass
