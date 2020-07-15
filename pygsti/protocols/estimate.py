"""
Defines the Estimate class.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import collections as _collections
import warnings as _warnings
import copy as _copy

from ..objects.verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from pygsti import tools as _tools
from ..objects import objectivefns as _objfns
from ..objects.confidenceregionfactory import ConfidenceRegionFactory as _ConfidenceRegionFactory
from ..objects.circuit import Circuit as _Circuit
from ..objects.explicitmodel import ExplicitOpModel as _ExplicitOpModel
from ..objects.circuitlist import CircuitList as _CircuitList
from ..objects.circuitstructure import PlaquetteGridCircuitStructure as _PlaquetteGridCircuitStructure

#Class for holding confidence region factory keys
CRFkey = _collections.namedtuple('CRFkey', ['model', 'circuit_list'])


class Estimate(object):
    """
    A class encapsulating the `Model` objects related to a single GST estimate up-to-gauge freedoms.

    Thus, this class holds the "iteration" `Model`s leading up to a
    final `Model`, and then different gauge optimizations of the final
    set.

    Parameters
    ----------
    parent : Results
        The parent Results object containing the dataset and
        circuit structure used for this Estimate.

    models : dict, optional
        A dictionary of models to included in this estimate

    parameters : dict, optional
        A dictionary of parameters associated with how these models
        were obtained.
    """

    @classmethod
    def create_gst_estimate(cls, parent, target_model=None, seed_model=None,
                            models_by_iter=None, parameters=None):
        """
        Initialize an empty Estimate object.

        Parameters
        ----------
        parent : Results
            The parent Results object containing the dataset and
            circuit structure used for this Estimate.

        target_model : Model
            The target model used when optimizing the objective.

        seed_model : Model
            The initial model used to seed the iterative part
            of the objective optimization.  Typically this is
            obtained via LGST.

        models_by_iter : list of Models
            The estimated model at each GST iteration. Typically these are the
            estimated models *before* any gauge optimization is performed.

        parameters : dict
            A dictionary of parameters associated with how these models
            were obtained.

        Returns
        -------
        Estimate
        """
        models = {}
        if target_model: models['target'] = target_model
        if seed_model: models['seed'] = seed_model
        if models_by_iter:
            models['iteration estimates'] = models_by_iter
            models['final iteration estimate'] = models_by_iter[-1]
        return cls(parent, models, parameters)

    def __init__(self, parent, models=None, parameters=None):
        """
        Initialize an empty Estimate object.

        Parameters
        ----------
        parent : Results
            The parent Results object containing the dataset and
            circuit structure used for this Estimate.

        models : dict, optional
            A dictionary of models to included in this estimate

        parameters : dict, optional
            A dictionary of parameters associated with how these models
            were obtained.
        """
        self.parent = parent
        self.parameters = _collections.OrderedDict()
        self.goparameters = _collections.OrderedDict()
        self.models = _collections.OrderedDict()
        self.confidence_region_factories = _collections.OrderedDict()

        #Set models
        if models:
            self.models.update(models)

        #Set parameters
        if isinstance(parameters, _collections.OrderedDict):
            self.parameters = parameters
        elif parameters is not None:
            for key in sorted(list(parameters.keys())):
                self.parameters[key] = parameters[key]

        #Meta info
        self.meta = {}

    def get_start_model(self, goparams):
        """
        Returns the starting model for the gauge optimization given by `goparams`.

        This has a particular (and perhaps singular) use for deciding whether
        the gauge-optimized model for one estimate can be simply copied to
        another estimate, without actually re-gauge-optimizing.

        Parameters
        ----------
        goparams : dict or list
            A dictionary of gauge-optimization parameters, just as in
            :func:`add_gaugeoptimized`.

        Returns
        -------
        Model
        """
        goparams_list = [goparams] if hasattr(goparams, 'keys') else goparams
        return goparams_list[0].get('model', self.models['final iteration estimate'])

    def add_gaugeoptimized(self, goparams, model=None, label=None, comm=None, verbosity=None):
        """
        Adds a gauge-optimized Model (computing it if needed) to this object.

        Parameters
        ----------
        goparams : dict or list
            A dictionary of gauge-optimization parameters, typically arguments
            to :func:`gaugeopt_to_target`, specifying how the gauge optimization
            was (or should be) performed.  When `model` is `None` (and this
            function computes the model internally) the keys and values of
            this dictionary must correspond to allowed arguments of
            :func:`gaugeopt_to_target`. By default, :func:`gaugeopt_to_target`'s
            first two arguments, the `Model` to optimize and the target,
            are taken to be `self.models['final iteration estimate']` and
            self.models['target'].  This argument can also be a *list* of
            such parameter dictionaries, which specifies a multi-stage gauge-
            optimization whereby the output of one stage is the input of the
            next.

        model : Model, optional
            The gauge-optimized model to store.  If None, then this model
            is computed by calling :func:`gaugeopt_to_target` with the contents
            of `goparams` as arguments as described above.

        label : str, optional
            A label for this gauge-optimized model, used as the key in
            this object's `models` and `goparameters` member dictionaries.
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
                   (label not in self.models): break

        goparams_list = [goparams] if hasattr(goparams, 'keys') else goparams
        ordered_goparams = []
        last_gs = None

        #Create a printer based on specified or maximum goparams
        # verbosity and default or existing comm.
        printer_comm = comm
        for gop in goparams_list:
            if gop.get('comm', None) is not None:
                printer_comm = gop['comm']; break
        if verbosity is not None:
            max_vb = verbosity
        else:
            verbosities = [gop.get('verbosity', 0) for gop in goparams_list]
            max_vb = max([v.verbosity if isinstance(v, _VerbosityPrinter) else v for v in verbosities])
        printer = _VerbosityPrinter.create_printer(max_vb, printer_comm)
        printer.log("-- Adding Gauge Optimized (%s) --" % label)

        for i, gop in enumerate(goparams_list):

            if model is not None:
                last_gs = model  # just use user-supplied result
            else:
                from ..algorithms import gaugeopt_to_target as _gaugeopt_to_target
                gop = gop.copy()  # so we don't change the caller's dict
                if '_gaugeGroupEl' in gop: del gop['_gaugeGroupEl']

                printer.log("Stage %d:" % i, 2)
                if verbosity is not None:
                    gop['verbosity'] = printer - 1  # use common printer

                if comm is not None and 'comm' not in gop:
                    gop['comm'] = comm

                if last_gs:
                    gop["model"] = last_gs
                elif "model" not in gop:
                    if 'final iteration estimate' in self.models:
                        gop["model"] = self.models['final iteration estimate']
                    else: raise ValueError("Must supply 'model' in 'goparams' argument")

                if "target_model" not in gop:
                    if 'target' in self.models:
                        gop["target_model"] = self.models['target']
                    else: raise ValueError("Must supply 'target_model' in 'goparams' argument")

                if "maxiter" not in gop:
                    gop["maxiter"] = 100

                gop['return_all'] = True
                if isinstance(gop['model'], _ExplicitOpModel):
                    #only explicit models can be gauge optimized
                    _, gauge_group_el, last_gs = _gaugeopt_to_target(**gop)
                else:
                    #but still fill in results for other models (?)
                    gauge_group_el, last_gs = None, gop['model'].copy()

                gop['_gaugeGroupEl'] = gauge_group_el  # an output stored here for convenience

            #sort the parameters by name for consistency
            ordered_goparams.append(_collections.OrderedDict(
                [(k, gop[k]) for k in sorted(list(gop.keys()))]))

        assert(last_gs is not None)
        self.models[label] = last_gs
        self.goparameters[label] = ordered_goparams if len(goparams_list) > 1 \
            else ordered_goparams[0]

    def add_confidence_region_factory(self,
                                      model_label='final iteration estimate',
                                      circuits_label='final'):
        """
        Creates a new confidence region factory.

        An instance of :class:`ConfidenceRegionFactory` serves to create
        confidence intervals and regions in reports and elsewhere.  This
        function creates such a factory, which is specific to a given
        `Model` (given by this object's `.models[model_label]` ) and
        circuit list (given by the parent `Results`'s
        `.circuit_lists[gastrings_label]` list).

        Parameters
        ----------
        model_label : str, optional
            The label of a `Model` held within this `Estimate`.

        circuits_label : str, optional
            The label of a circuit list within this estimate's parent
            `Results` object.

        Returns
        -------
        ConfidenceRegionFactory
            The newly created factory (also cached internally) and accessible
            via the :func:`create_confidence_region_factory` method.
        """
        ky = CRFkey(model_label, circuits_label)
        if ky in self.confidence_region_factories:
            _warnings.warn("Confidence region factory for %s already exists - overwriting!" % str(ky))

        new_crf = _ConfidenceRegionFactory(self, model_label, circuits_label)
        self.confidence_region_factories[ky] = new_crf
        return new_crf

    def has_confidence_region_factory(self, model_label='final iteration estimate',
                                      circuits_label='final'):
        """
        Checks whether a confidence region factory for the given model and circuit list labels exists.

        Parameters
        ----------
        model_label : str, optional
            The label of a `Model` held within this `Estimate`.

        circuits_label : str, optional
            The label of a circuit list within this estimate's parent
            `Results` object.

        Returns
        -------
        bool
        """
        return bool(CRFkey(model_label, circuits_label) in self.confidence_region_factories)

    def create_confidence_region_factory(self, model_label='final iteration estimate',
                                         circuits_label='final', create_if_needed=False):
        """
        Retrieves a confidence region factory for the given model and circuit list labels.

        For more information about confidence region factories, see :func:`add_confidence_region_factory`.

        Parameters
        ----------
        model_label : str, optional
            The label of a `Model` held within this `Estimate`.

        circuits_label : str, optional
            The label of a circuit list within this estimate's parent
            `Results` object.

        create_if_needed : bool, optional
            If True, a new confidence region factory will be created if none
            exists.  Otherwise a `KeyError` is raised when the requested
            factory doesn't exist.

        Returns
        -------
        ConfidenceRegionFactory
        """
        ky = CRFkey(model_label, circuits_label)
        if ky in self.confidence_region_factories:
            return self.confidence_region_factories[ky]
        elif create_if_needed:
            return self.add_confidence_region_factory(model_label, circuits_label)
        else:
            raise KeyError("No confidence region factory for key %s exists!" % str(ky))

    def gauge_propagate_confidence_region_factory(
            self, to_model_label, from_model_label='final iteration estimate',
            circuits_label='final', eps=1e-3, verbosity=0):
        """
        Propagates a confidence region among gauge-equivalent models.

        More specifically, this function propagates an existing "reference"
        confidence region for a Model "G0" to a new confidence region for a
        gauge-equivalent model "G1".

        When successful, a new confidence region factory is created for the
        `.models[to_model_label]` `Model` and `circuits_label` gate
        string list from the existing factory for `.models[from_model_label]`.

        Parameters
        ----------
        to_model_label : str
            The key into this `Estimate` object's `models` and `goparameters`
            dictionaries that identifies the final gauge-optimized result to
            create a factory for.  This gauge optimization must have begun at
            "from" reference model, i.e., `models[from_model_label]` must
            equal (by frobeinus distance) `goparameters[to_model_label]['model']`.

        from_model_label : str, optional
            The key into this `Estimate` object's `models` dictionary
            that identifies the reference model.

        circuits_label : str, optional
            The key of the circuit list (within the parent `Results`'s
            `.circuit_lists` dictionary) that identifies the circuit
            list used by the old (&new) confidence region factories.

        eps : float, optional
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
        printer = _VerbosityPrinter.create_printer(verbosity)

        ref_model = self.models[from_model_label]
        goparams = self.goparameters[to_model_label]
        goparams_list = [goparams] if hasattr(goparams, 'keys') else goparams
        start_model = goparams_list[0]['model'].copy()
        final_model = self.models[to_model_label].copy()

        gauge_group_els = []
        for gop in goparams_list:
            assert('_gaugeGroupEl' in gop), "To propagate a confidence " + \
                "region, goparameters must contain the gauge-group-element as `_gaugeGroupEl`"
            gauge_group_els.append(gop['_gaugeGroupEl'])

        assert(start_model.frobeniusdist(ref_model) < 1e-6), \
            "Gauge-opt starting point must be the 'from' (reference) Model"

        crf = self.confidence_region_factories.get(
            CRFkey(from_model_label, circuits_label), None)

        assert(crf is not None), "Initial confidence region factory doesn't exist!"
        assert(crf.has_hessian), "Initial factory must contain a computed Hessian!"

        #Update hessian by TMx = d(diffs in current go'd model)/d(diffs in ref model)
        tmx = _np.empty((final_model.num_params, ref_model.num_params), 'd')
        v0, w0 = ref_model.to_vector(), final_model.to_vector()
        mdl = ref_model.copy()

        printer.log(" *** Propagating Hessian from '%s' to '%s' ***" %
                    (from_model_label, to_model_label))

        with printer.progress_logging(1):
            for icol in range(ref_model.num_params):
                v = v0.copy(); v[icol] += eps  # dv is along iCol-th direction
                mdl.from_vector(v)
                for gauge_group_el in gauge_group_els:
                    mdl.transform_inplace(gauge_group_el)
                w = mdl.to_vector()
                dw = (w - w0) / eps
                tmx[:, icol] = dw
                printer.show_progress(icol, ref_model.num_params, prefix='Column: ')
                #,suffix = "; finite_diff = %g" % _np.linalg.norm(dw)

        #rank = _np.linalg.matrix_rank(TMx)
        #print("DEBUG: constructed TMx: rank = ", rank)

        # Hessian is gauge-transported via H -> TMx_inv^T * H * TMx_inv
        tmx_inv = _np.linalg.inv(tmx)
        new_hessian = _np.dot(tmx_inv.T, _np.dot(crf.hessian, tmx_inv))

        #Create a new confidence region based on the new hessian
        new_crf = _ConfidenceRegionFactory(self, to_model_label,
                                           circuits_label, new_hessian,
                                           crf.nonMarkRadiusSq)
        self.confidence_region_factories[CRFkey(to_model_label, circuits_label)] = new_crf
        printer.log("   Successfully transported Hessian and ConfidenceRegionFactory.")

        return new_crf

    def create_effective_dataset(self, return_submxs=False):
        """
        Generate a `DataSet` containing the effective counts as dictated by the "weights" parameter.

        An estimate's `self.parameters['weights']` value specifies a dictionary
        of circuits weights, which modify (typically *reduce*) the counts given in
        its (parent's) data set.

        This function rescales the actual data contained in this Estimate's
        parent :class:`ModelEstimteResults` object according to the estimate's
        "weights" parameter.  The scaled data set is returned, along with
        (optionall) a list-of-lists of matrices containing the scaling values
        which can be easily plotted via a `ColorBoxPlot`.

        Parameters
        ----------
        return_submxs : boolean
            If true, also return a list-of-lists of matrices containing the
            scaling values, as described above.

        Returns
        -------
        ds : DataSet
            The "effective" (scaled) data set.
        subMxs : list-of-lists
            Only returned if `return_submxs == True`.  Contains the
            scale values (see above).
        """
        p = self.parent
        gss = _PlaquetteGridCircuitStructure.cast(p.circuit_lists['final'])  # FUTURE: overrideable?
        weights = self.parameters.get("weights", None)

        if weights is not None:
            scaled_dataset = p.dataset.copy_nonstatic()

            sub_mxs = []
            for y in gss.used_ys:
                sub_mxs.append([])
                for x in gss.used_xs:
                    plaq = gss.plaquette(x, y, empty_if_missing=True).expand_aliases()
                    scaling_mx = _np.nan * _np.ones((plaq.num_rows, plaq.num_cols), 'd')

                    if len(plaq) > 0:
                        for i, j, opstr in plaq:
                            scaling_mx[i, j] = weights.get(opstr, 1.0)
                            if scaling_mx[i, j] != 1.0:
                                scaled_dataset[opstr].scale_inplace(scaling_mx[i, j])

                    #build up a subMxs list-of-lists as a plotting
                    # function does, so we can easily plot the scaling
                    # factors in a color box plot.
                    sub_mxs[-1].append(scaling_mx)

            scaled_dataset.done_adding_data()
            if return_submxs:
                return scaled_dataset, sub_mxs
            else: return scaled_dataset

        else:  # no weights specified - just return original dataset (no scaling)

            if return_submxs:  # then need to create subMxs with all 1's
                sub_mxs = []
                for y in gss.used_ys:
                    sub_mxs.append([])
                    for x in gss.used_xs:
                        plaq = gss.get_plaquette(x, y, empty_if_missing=True)
                        scaling_mx = _np.nan * _np.ones((plaq.rows, plaq.cols), 'd')
                        for i, j, opstr in plaq:
                            scaling_mx[i, j] = 1.0
                        sub_mxs[-1].append(scaling_mx)
                return p.dataset, sub_mxs  # copy dataset?
            else:
                return p.dataset

    def misfit_sigma(self, use_accurate_np=False, comm=None):
        """
        Returns the number of standard deviations (sigma) of model violation.

        Parameters
        ----------
        use_accurate_np : bool, optional
            Whether to use the more accurate number of *non-gauge* parameters
            (but more expensive to compute), or just use the total number of
            model parameters.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        Returns
        -------
        float
        """
        p = self.parent

        mdl = self.models['final iteration estimate']  # FUTURE: overrideable?
        circuit_list = p.circuit_lists['final']  # FUTURE: overrideable?
        ds = self.create_effective_dataset()
        objfn_builder = self.parameters.get('final_objfn_builder', _objfns.PoissonPicDeltaLogLFunction.builder())
        objfn = objfn_builder.build(mdl, ds, circuit_list, {'comm': comm}, verbosity=0)
        fitqty = objfn.chi2k_distributed_qty(objfn.fn())
        aliases = circuit_list.op_label_aliases if isinstance(circuit_list, _CircuitList) else None

        ds_allstrs = _tools.apply_aliases_to_circuits(circuit_list, aliases)
        ds_dof = ds.degrees_of_freedom(ds_allstrs)  # number of independent parameters in dataset
        if hasattr(mdl, 'num_nongauge_params'):
            mdl_dof = mdl.num_nongauge_params() if use_accurate_np else mdl.num_params
        else:
            mdl_dof = mdl.num_params
        k = max(ds_dof - mdl_dof, 1)  # expected chi^2 or 2*(logL_ub-logl) mean
        if ds_dof <= mdl_dof: _warnings.warn("Max-model params (%d) <= model params (%d)!  Using k == 1."
                                             % (ds_dof, mdl_dof))
        return (fitqty - k) / _np.sqrt(2 * k)

    def view(self, gaugeopt_keys, parent=None):
        """
        Creates a shallow copy of this Results object containing only the given gauge-optimization keys.

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
        view.models = self.models
        view.confidence_region_factories = self.confidence_region_factories

        if gaugeopt_keys is None:
            gaugeopt_keys = list(self.goparameters.keys())
        elif isinstance(gaugeopt_keys, str):
            gaugeopt_keys = [gaugeopt_keys]
        for go_key in gaugeopt_keys:
            if go_key in self.goparameters:
                view.goparameters[go_key] = self.goparameters[go_key]

        return view

    def copy(self):
        """
        Creates a copy of this Estimate object.

        Returns
        -------
        Estimate
        """
        #TODO: check whether this deep copies (if we want it to...) - I expect it doesn't currently
        cpy = Estimate(self.parent)
        cpy.parameters = _copy.deepcopy(self.parameters)
        cpy.goparameters = _copy.deepcopy(self.goparameters)
        cpy.models = self.models.copy()
        cpy.confidence_region_factories = _copy.deepcopy(self.confidence_region_factories)
        cpy.meta = _copy.deepcopy(self.meta)
        return cpy

    def __str__(self):
        s = "----------------------------------------------------------\n"
        s += "---------------- pyGSTi Estimate Object ------------------\n"
        s += "----------------------------------------------------------\n"
        s += "\n"
        s += "How to access my contents:\n\n"
        s += " .models   -- a dictionary of Model objects w/keys:\n"
        s += " ---------------------------------------------------------\n"
        s += "  " + "\n  ".join(list(self.models.keys())) + "\n"
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
        for lbl, goparams in self.goparameters.items():
            if hasattr(goparams, "keys"):
                if 'comm' in goparams:
                    goparams = goparams.copy()
                    goparams['comm'] = None
                to_pickle['goparameters'][lbl] = goparams
            else:  # goparams is a list
                new_goparams = []  # new list
                for goparams_dict in goparams:
                    if 'comm' in goparams_dict:
                        goparams_dict = goparams_dict.copy()
                        goparams_dict['comm'] = None
                    new_goparams.append(goparams_dict)
                to_pickle['goparameters'][lbl] = new_goparams

        # don't pickle parent (will create circular reference)
        del to_pickle['parent']
        return to_pickle

    def __setstate__(self, state_dict):
        #BACKWARDS COMPATIBILITY
        if 'confidence_regions' in state_dict:
            del state_dict['confidence_regions']
            state_dict['confidence_region_factories'] = _collections.OrderedDict()
        if 'meta' not in state_dict: state_dict['meta'] = {}
        if 'gatesets' in state_dict:
            state_dict['models'] = state_dict['gatesets']
            del state_dict['gatesets']

        self.__dict__.update(state_dict)
        for crf in self.confidence_region_factories.values():
            crf.set_parent(self)
        self.parent = None  # initialize to None upon unpickling

    def set_parent(self, parent):
        """
        Sets the parent object of this estimate.

        This is used, for instance, to re-establish parent-child links
        after loading objects from disk.

        Parameters
        ----------
        parent : ModelEstimateResults
            This object's parent.

        Returns
        -------
        None
        """
        self.parent = parent
