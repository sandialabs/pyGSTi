""" Defines the Estimate class."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy       as _np
import collections as _collections
import warnings    as _warnings
import copy        as _copy

from ..baseobjs import VerbosityPrinter as _VerbosityPrinter
from .. import tools as _tools
from ..tools import compattools as _compat
from .confidenceregionfactory import ConfidenceRegionFactory as _ConfidenceRegionFactory

#Class for holding confidence region factory keys
CRFkey = _collections.namedtuple('CRFkey', ['gateset','gatestring_list'])

class Estimate(object):
    """
    A class encapsulating the `GateSet` objects related to 
    a single GST estimate up-to-gauge freedoms. 

    Thus, this class holds the "iteration" `GateSet`s leading up to a
    final `GateSet`, and then different gauge optimizations of the final
    set.
    """
    
    def __init__(self, parent, targetGateset=None, seedGateset=None,
                 gatesetsByIter=None, parameters=None):
        """
        Initialize an empty Estimate object.

        Parameters
        ----------
        parent : Results
            The parent Results object containing the dataset and
            gate string structure used for this Estimate.

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
            A dictionary of parameters associated with how these gate sets
            were obtained.
        """
        self.parent = parent
        self.parameters = _collections.OrderedDict()
        self.goparameters = _collections.OrderedDict()
        self.gatesets = _collections.OrderedDict()
        self.confidence_region_factories = _collections.OrderedDict()

        #Set gatesets
        if targetGateset: self.gatesets['target'] = targetGateset
        if seedGateset: self.gatesets['seed'] = seedGateset
        if gatesetsByIter:
            self.gatesets['iteration estimates'] = gatesetsByIter
            self.gatesets['final iteration estimate'] = gatesetsByIter[-1]

        #Set parameters
        if isinstance(parameters, _collections.OrderedDict):
            self.parameters = parameters
        elif parameters is not None:
            for key in sorted(list(parameters.keys())):
                self.parameters[key] = parameters[key]

        #Meta info
        self.meta = {}

                
    def get_start_gateset(self, goparams):
        """
        Returns the starting gateset for the gauge optimization given by `goparams`.

        This has a particular (and perhaps singular) use for deciding whether
        the gauge-optimized gate set for one estimate can be simply copied to
        another estimate, without actually re-gauge-optimizing.

        Parameters
        ----------
        goparams : dict or list
            A dictionary of gauge-optimization parameters, just as in
            :func:`add_gaugeoptimized`.

        Returns
        -------
        GateSet
        """
        goparams_list = [goparams] if hasattr(goparams,'keys') else goparams
        return goparams_list[0].get('gateset',self.gatesets['final iteration estimate'])

                
    def add_gaugeoptimized(self, goparams, gateset=None, label=None, comm=None, verbosity=None):
        """
        Adds a gauge-optimized GateSet (computing it if needed) to this object.

        Parameters
        ----------
        goparams : dict or list
            A dictionary of gauge-optimization parameters, typically arguments
            to :func:`gaugeopt_to_target`, specifying how the gauge optimization
            was (or should be) performed.  When `gateset` is `None` (and this
            function computes the gate set internally) the keys and values of
            this dictionary must correspond to allowed arguments of 
            :func:`gaugeopt_to_target`. By default, :func:`gaugeopt_to_target`'s
            first two arguments, the `GateSet` to optimize and the target,
            are taken to be `self.gatesets['final iteration estimate']` and 
            self.gatesets['target'].  This argument can also be a *list* of
            such parameter dictionaries, which specifies a multi-stage gauge-
            optimization whereby the output of one stage is the input of the
            next.

        gateset : GateSet, optional
            The gauge-optimized gate set to store.  If None, then this gate set
            is computed by calling :func:`gaugeopt_to_target` with the contents
            of `goparams` as arguments as described above.

        label : str, optional
            A label for this gauge-optimized gate set, used as the key in
            this object's `gatesets` and `goparameters` member dictionaries.
            If None, then the next available "go<X>", where <X> is a 
            non-negative integer, is used as the label.

        comm : mpi4py.MPI.Comm, optional
            A default MPI communicator to use when one is not specified
            as the 'comm' element of/within `goparams`.

       verbosity : int, optional
            An integer specifying the level of detail printed to stdout
            during the calculations performed in this function.  If not
            None, this value will override any verbosity values set
            within `goparams`.

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
            
        goparams_list = [goparams] if hasattr(goparams,'keys') else goparams
        ordered_goparams = []
        last_gs = None


        #Create a printer based on specified or maximum goparams
        # verbosity and default or existing comm.
        printer_comm = comm
        for gop in goparams_list:
            if gop.get('comm',None) is not None:
                printer_comm = gop['comm']; break
        max_vb = verbosity if (verbosity is not None) else \
                 max( [ gop.get('verbosity',0) for gop in goparams_list ])
        printer = _VerbosityPrinter.build_printer(max_vb, printer_comm)
        printer.log("-- Adding Gauge Optimized (%s) --" % label)
        
        for i,gop in enumerate(goparams_list):
            
            if gateset is not None:
                last_gs = gateset #just use user-supplied result
            else:
                from ..algorithms import gaugeopt_to_target as _gaugeopt_to_target
                gop = gop.copy() #so we don't change the caller's dict

                printer.log("Stage %d:" % i, 2)
                if verbosity is not None:
                    gop['verbosity'] = printer-1 #use common printer

                if comm is not None and 'comm' not in gop: 
                    gop['comm'] = comm

                if last_gs:
                    gop["gateset"] = last_gs
                elif "gateset" not in gop:
                    if 'final iteration estimate' in self.gatesets:
                        gop["gateset"] = self.gatesets['final iteration estimate']
                    else: raise ValueError("Must supply 'gateset' in 'goparams' argument")
                    
                if "targetGateset" not in gop:
                    if 'target' in self.gatesets:
                        gop["targetGateset"] = self.gatesets['target']
                    else: raise ValueError("Must supply 'targetGateset' in 'goparams' argument")
    
                gop['returnAll'] = True
                _, gaugeGroupEl, last_gs = _gaugeopt_to_target(**gop)
                gop['_gaugeGroupEl'] = gaugeGroupEl # an output stored here for convenience

            #sort the parameters by name for consistency
            ordered_goparams.append( _collections.OrderedDict( 
                [(k,gop[k]) for k in sorted(list(gop.keys()))]) )

        assert(last_gs is not None)
        self.gatesets[label] = last_gs
        self.goparameters[label] = ordered_goparams if len(goparams_list) > 1 \
                                   else ordered_goparams[0]

        
    def add_confidence_region_factory(self,
                                      gateset_label='final iteration estimate',
                                      gatestrings_label='final'):
        """
        Creates a new confidence region factory.

        An instance of :class:`ConfidenceRegionFactory` serves to create
        confidence intervals and regions in reports and elsewhere.  This
        function creates such a factory, which is specific to a given
        `GateSet` (given by this object's `.gatesets[gateset_label]` ) and 
        gate string list (given by the parent `Results`'s 
        `.gatestring_lists[gastrings_label]` list).

        Parameters
        ----------
        gateset_label : str, optional
            The label of a `GateSet` held within this `Estimate`.

        gatestrings_label : str, optional
            The label of a gate string list within this estimate's parent
            `Results` object.

        Returns
        -------
        ConfidenceRegionFactory
            The newly created factory (also cached internally) and accessible
            via the :func:`get_confidence_region_factory` method.
        """
        ky = CRFkey(gateset_label, gatestrings_label)
        if ky in self.confidence_region_factories:
            _warnings.warn("Confidence region factory for %s already exists - overwriting!" % str(ky))
            
        newCRF = _ConfidenceRegionFactory(self, gateset_label, gatestrings_label)
        self.confidence_region_factories[ky] = newCRF
        return newCRF
                                                                                    

    def has_confidence_region_factory(self, gateset_label='final iteration estimate',
                                      gatestrings_label='final'):
        """
        Checks whether a confidence region factory for the given gate set
        and gate string list labels exists.

        Parameters
        ----------
        gateset_label : str, optional
            The label of a `GateSet` held within this `Estimate`.

        gatestrings_label : str, optional
            The label of a gate string list within this estimate's parent
            `Results` object.

        Returns
        -------
        bool
        """
        return bool( CRFkey(gateset_label, gatestrings_label) in self.confidence_region_factories)

    
    def get_confidence_region_factory(self, gateset_label='final iteration estimate',
                                      gatestrings_label='final', createIfNeeded=False):
        """
        Retrieves a confidence region factory for the given gate set
        and gate string list labels.  For more information about
        confidence region factories, see :func:`add_confidence_region_factory`.

        Parameters
        ----------
        gateset_label : str, optional
            The label of a `GateSet` held within this `Estimate`.

        gatestrings_label : str, optional
            The label of a gate string list within this estimate's parent
            `Results` object.

        createIfNeeded : bool, optional
            If True, a new confidence region factory will be created if none
            exists.  Otherwise a `KeyError` is raised when the requested 
            factory doesn't exist.

        Returns
        -------
        ConfidenceRegionFactory
        """
        ky = CRFkey(gateset_label, gatestrings_label)
        if ky in self.confidence_region_factories:
            return self.confidence_region_factories[ky]
        elif createIfNeeded:
            return self.add_confidence_region_factory(gateset_label, gatestrings_label)
        else:
            raise KeyError("No confidence region factory for key %s exists!" % str(ky))
        
    def gauge_propagate_confidence_region_factory(
            self, to_gateset_label, from_gateset_label='final iteration estimate',
            gatestrings_label = 'final', EPS=1e-3, verbosity=0):
        """
        Propagates an existing "reference" confidence region for a GateSet
        "G0" to a new confidence region for a gauge-equivalent gateset "G1".

        When successful, a new confidence region factory is created for the 
        `.gatesets[to_gateset_label]` `GateSet` and `gatestrings_label` gate
        string list from the existing factory for `.gatesets[from_gateset_label]`.

        Parameters
        ----------
        to_gateset_label : str
            The key into this `Estimate` object's `gatesets` and `goparameters`
            dictionaries that identifies the final gauge-optimized result to
            create a factory for.  This gauge optimization must have begun at
            "from" reference gateset, i.e., `gatesets[from_gateset_label]` must
            equal (by frobeinus distance) `goparameters[to_gateset_label]['gateset']`.

        from_gateset_label : str, optional
            The key into this `Estimate` object's `gatesets` dictionary
            that identifies the reference gate set.
        
        gatestrings_label : str, optional
            The key of the gate string list (within the parent `Results`'s
            `.gatestring_lists` dictionary) that identifies the gate string
            list used by the old (&new) confidence region factories.

        EPS : float, optional
            A small offset used for constructing finite-difference derivatives.
            Usually the default value is fine.

        verbosity : int, optional
            A non-negative integer indicating the amount of detail to print
            to stdout.

        Returns
        -------
        ConfidenceRegionFactory
            Note: this region is also stored internally and as such the return
            value of this function can often be ignored.
        """
        printer = _VerbosityPrinter.build_printer(verbosity)
        
        ref_gateset = self.gatesets[from_gateset_label]
        goparams = self.goparameters[to_gateset_label]
        start_gateset = goparams['gateset'].copy()
        final_gateset = self.gatesets[to_gateset_label].copy()

        goparams_list = [goparams] if hasattr(goparams,'keys') else goparams
        gaugeGroupEls = []
        for gop in goparams_list:
            assert('_gaugeGroupEl' in gop),"To propagate a confidence " + \
                "region, goparameters must contain the gauge-group-element as `_gaugeGroupEl`"
            gaugeGroupEls.append( goparams['_gaugeGroupEl'] )

        assert(start_gateset.frobeniusdist(ref_gateset) < 1e-6), \
            "Gauge-opt starting point must be the 'from' (reference) GateSet"
        
        crf = self.confidence_region_factories.get(
            CRFkey(from_gateset_label, gatestrings_label), None)
            
        assert(crf is not None), "Initial confidence region factory doesn't exist!"
        assert(crf.has_hessian()), "Initial factory must contain a computed Hessian!"
                            
        #Update hessian by TMx = d(diffs in current go'd gateset)/d(diffs in ref gateset)
        TMx = _np.empty( (final_gateset.num_params(), ref_gateset.num_params()), 'd' )
        v0, w0 = ref_gateset.to_vector(), final_gateset.to_vector()
        gs = ref_gateset.copy()

        printer.log(" *** Propagating Hessian from '%s' to '%s' ***" %
                    (from_gateset_label, to_gateset_label))

        with printer.progress_logging(1):
            for iCol in range(ref_gateset.num_params()):
               v = v0.copy(); v[iCol] += EPS # dv is along iCol-th direction 
               gs.from_vector(v)
               for gaugeGroupEl in gaugeGroupEls:
                   gs.transform(gaugeGroupEl)
               w = gs.to_vector()
               dw = (w - w0)/EPS
               TMx[:,iCol] = dw
               printer.show_progress(iCol, ref_gateset.num_params(), prefix='Column: ')
                 #,suffix = "; finite_diff = %g" % _np.linalg.norm(dw)

        #rank = _np.linalg.matrix_rank(TMx)
        #print("DEBUG: constructed TMx: rank = ", rank)
        
        # Hessian is gauge-transported via H -> TMx_inv^T * H * TMx_inv
        TMx_inv = _np.linalg.inv(TMx)
        new_hessian = _np.dot(TMx_inv.T, _np.dot(crf.hessian, TMx_inv))

        #Create a new confidence region based on the new hessian
        new_crf = _ConfidenceRegionFactory(self, to_gateset_label,
                                           gatestrings_label, new_hessian,
                                           crf.nonMarkRadiusSq)
        self.confidence_region_factories[CRFkey(to_gateset_label, gatestrings_label)] = new_crf
        printer.log("   Successfully transported Hessian and ConfidenceRegionFactory.")

        return new_crf


    def get_effective_dataset(self, return_subMxs=False):
        """
        Generate a `DataSet` containing the effective counts as dictated by
        the "weights" parameter, which specifies a dict of gate string weights.

        This function rescales the actual data contained in this Estimate's
        parent `Results` object according to the estimate's "weights" parameter.
        The scaled data set is returned, along with (optionall) a list-of-lists
        of matrices containing the scaling values which can be easily plotted
        via a `ColorBoxPlot`.

        Parameters
        ----------
        return_subMxs : boolean
            If true, also return a list-of-lists of matrices containing the
            scaling values, as described above.

        Returns
        -------
        ds : DataSet
            The "effective" (scaled) data set.

        subMxs : list-of-lists
            Only returned if `return_subMxs == True`.  Contains the
            scale values (see above).
        """
        p = self.parent
        gss = p.gatestring_structs['final'] #FUTURE: overrideable?
        weights = self.parameters.get("weights",None)
        
        if weights is not None:
            scaled_dataset = p.dataset.copy_nonstatic()
            nRows, nCols = gss.plaquette_rows_cols()
            
            subMxs = []
            for y in gss.used_yvals():
                subMxs.append( [] )
                for x in gss.used_xvals():
                    scalingMx = _np.nan * _np.ones( (nRows,nCols), 'd')
                    plaq = gss.get_plaquette(x,y).expand_aliases()
                    if len(plaq) > 0:
                        for i,j,gstr in plaq:
                            scalingMx[i,j] = weights.get(gstr,1.0)
                            if scalingMx[i,j] != 1.0:
                                scaled_dataset[gstr].scale(scalingMx[i,j])

                    #build up a subMxs list-of-lists as a plotting
                    # function does, so we can easily plot the scaling
                    # factors in a color box plot.
                    subMxs[-1].append(scalingMx)

            scaled_dataset.done_adding_data()
            if return_subMxs:
                return scaled_dataset, subMxs
            else: return scaled_dataset
            
        else: #no weights specified - just return original dataset (no scaling)
            
            if return_subMxs: #then need to create subMxs with all 1's
                subMxs = []
                for y in gss.used_yvals():
                    subMxs.append( [] )
                    for x in gss.used_xvals():
                        plaq = gss.get_plaquette(x,y)
                        scalingMx = _np.nan * _np.ones( (plaq.rows,plaq.cols), 'd')
                        for i,j,gstr in plaq:
                            scalingMx[i,j] = 1.0
                        subMxs[-1].append( scalingMx )
                return p.dataset, subMxs #copy dataset?
            else:
                return p.dataset

    def misfit_sigma(self, use_accurate_Np=False):
        """
        Returns the number of standard deviations (sigma) of model violation.

        Returns
        -------
        float
        """
        p = self.parent
        obj = self.parameters.get('objective',None)
        assert(obj in ('chi2','logl')),"Invalid objective!"

        gs = self.gatesets['final iteration estimate'] #FUTURE: overrideable?
        gss = p.gatestring_structs['final'] #FUTURE: overrideable?
        mpc = self.parameters.get('minProbClipForWeighting',1e-4)
        ds = self.get_effective_dataset()
        
        if obj == "chi2":
            fitQty = _tools.chi2( gs, ds, gss.allstrs,
                                  minProbClipForWeighting=mpc,
                                  gateLabelAliases=gss.aliases)
        elif obj == "logl":
            logL_upperbound = _tools.logl_max(gs, ds, gss.allstrs, gateLabelAliases=gss.aliases)
            logl = _tools.logl( gs, ds, gss.allstrs, gateLabelAliases=gss.aliases)
            fitQty = 2*(logL_upperbound - logl) # twoDeltaLogL

        ds_allstrs = _tools.find_replace_tuple_list(
            gss.allstrs, gss.aliases)
        Ns  = ds.get_degrees_of_freedom(ds_allstrs)  #number of independent parameters in dataset
        Np = gs.num_nongauge_params() if use_accurate_Np else gs.num_params()
        k = max(Ns-Np,1) #expected chi^2 or 2*(logL_ub-logl) mean
        if Ns <= Np: _warnings.warn("Max-model params (%d) <= gate set params (%d)!  Using k == 1." % (Ns,Np))
        return (fitQty-k)/_np.sqrt(2*k)

    
    def view(self, gaugeopt_keys, parent=None):
        """
        Creates a shallow copy of this Results object containing only the
        given gauge-optimization keys.

        Parameters
        ----------
        gaugeopt_keys : str or list, optional
            Either a single string-value gauge-optimization key or a list of
            such keys.  If `None`, then all gauge-optimization keys are 
            retained.

        parent : Results, optional
            The parent `Results` object of the view.  If `None`, then the
            current `Estimate`'s parent is used.

        Returns
        -------
        Estimate
        """
        if parent is None: parent = self.parent
        view = Estimate(parent)
        view.parameters = self.parameters
        view.gatesets = self.gatesets
        view.confidence_region_factories = self.confidence_region_factories
        
        if gaugeopt_keys is None:
            gaugeopt_keys = list(self.goparameters.keys())
        elif _compat.isstr(gaugeopt_keys):
            gaugeopt_keys = [gaugeopt_keys]
        for go_key in gaugeopt_keys:
            if go_key in self.goparameters:
                view.goparameters[go_key] = self.goparameters[go_key]

        return view
    

    def copy(self):
        """ Creates a copy of this Estimate object. """
        #TODO: check whether this deep copies (if we want it to...) - I expect it doesn't currently
        cpy = Estimate(self.parent)
        cpy.parameters = _copy.deepcopy(self.parameters)
        cpy.goparameters = _copy.deepcopy(self.goparameters)
        cpy.gatesets = self.gatesets.copy()
        cpy.confidence_region_factories = _copy.deepcopy(self.confidence_region_factories)
        cpy.meta = _copy.deepcopy(self.meta)
        return cpy

    
    def __str__(self):
        s  = "----------------------------------------------------------\n"
        s += "---------------- pyGSTi Estimate Object ------------------\n"
        s += "----------------------------------------------------------\n"
        s += "\n"
        s += "How to access my contents:\n\n"
        s += " .gatesets   -- a dictionary of GateSet objects w/keys:\n"
        s += " ---------------------------------------------------------\n"
        s += "  " + "\n  ".join(list(self.gatesets.keys())) + "\n"
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
    
    def __getstate__(self):
        #Don't pickle comms in goparameters
        to_pickle = self.__dict__.copy()
        to_pickle['goparameters'] = _collections.OrderedDict()
        for lbl,goparams in self.goparameters.items():            
            if hasattr(goparams,"keys"):
                if 'comm' in goparams:
                    goparams = goparams.copy()
                    goparams['comm'] = None
                to_pickle['goparameters'][lbl] = goparams
            else: #goparams is a list
                new_goparams = [] #new list
                for goparams_dict in goparams:
                    if 'comm' in goparams_dict:
                        goparams_dict = goparams_dict.copy()
                        goparams_dict['comm'] = None
                    new_goparams.append(goparams_dict)
                to_pickle['goparameters'][lbl] = new_goparams

        # don't pickle parent (will create circular reference)
        del to_pickle['parent'] 
        return  to_pickle

    def __setstate__(self, stateDict):
        #BACKWARDS COMPATIBILITY
        if 'confidence_regions' in stateDict: 
            del stateDict['confidence_regions']
            stateDict['confidence_region_factories'] = _collections.OrderedDict()
        if 'meta' not in stateDict: stateDict['meta'] = {}
            
        self.__dict__.update(stateDict)
        for crf in self.confidence_region_factories.values():
            crf.set_parent(self)
        self.parent = None # initialize to None upon unpickling

    def set_parent(self, parent):
        """
        Sets the parent Results object of this Estimate.
        """
        self.parent = parent
     

