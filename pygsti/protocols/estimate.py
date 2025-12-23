"""
Defines the Estimate class.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import collections as _collections
import copy as _copy
import warnings as _warnings
import pathlib as _pathlib

import numpy as _np

from pygsti import tools as _tools
from pygsti import io as _io
from pygsti.objectivefns.objectivefns import CachedObjectiveFunction as _CachedObjectiveFunction
from pygsti.objectivefns.objectivefns import ModelDatasetCircuitsStore as _ModelDatasetCircuitStore
from pygsti.protocols.confidenceregionfactory import ConfidenceRegionFactory as _ConfidenceRegionFactory
from pygsti.models.explicitmodel import ExplicitOpModel as _ExplicitOpModel
from pygsti.objectivefns import objectivefns as _objfns
from pygsti.circuits import CircuitList as _CircuitList, Circuit as _Circuit
from pygsti.circuits.circuitstructure import PlaquetteGridCircuitStructure as _PlaquetteGridCircuitStructure
from pygsti.baseobjs.verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from pygsti.baseobjs.mongoserializable import MongoSerializable as _MongoSerializable


#Class for holding confidence region factory keys
CRFkey = _collections.namedtuple('CRFkey', ['model', 'circuit_list'])


class Estimate(_MongoSerializable):
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
    collection_name = "pygsti_estimates"

    @classmethod
    def from_dir(cls, dirname, quick_load=False):
        """
        Initialize a new Protocol object from `dirname`.

        quick_load : bool, optional
            Setting this to True skips the loading of components that may take
            a long time to load.

        Parameters
        ----------
        dirname : str
            The directory name.

        quick_load : bool, optional
            Setting this to True skips the loading of components that may take
            a long time to load.

        Returns
        -------
        Protocol
        """
        ret = cls.__new__(cls)
        _MongoSerializable.__init__(ret)
        state = _io.load_meta_based_dir(_pathlib.Path(dirname), 'auxfile_types', quick_load=quick_load)
        ret.__dict__.update(state)
        for crf in ret.confidence_region_factories.values():
            crf.set_parent(ret)  # re-link confidence_region_factories
        if ret.circuit_weights is not None:
            from pygsti.circuits.circuitparser import parse_circuit
            cws : dict[_Circuit, float] = dict()
            for cstr, w in ret.circuit_weights.items():
                lbls = parse_circuit(cstr, True, True)[0]
                ckt = _Circuit(lbls)
                cws[ckt] = w
            ret.circuit_weights = cws
        return ret

    @classmethod
    def _create_obj_from_doc_and_mongodb(cls, doc, mongodb, quick_load=False):
        ret = cls.__new__(cls)
        _MongoSerializable.__init__(ret, doc.get('_id', None))
        ret.__dict__.update(_io.read_auxtree_from_mongodb_doc(mongodb, doc, 'auxfile_types', quick_load=quick_load))
        for crf in ret.confidence_region_factories.values():
            crf.set_parent(ret)  # re-link confidence_region_factories
        return ret

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
            for k, mdl in enumerate(models_by_iter):
                models['iteration %d estimate' % k] = mdl
            models['final iteration estimate'] = models_by_iter[-1]
        return cls(parent, models, parameters)

    def __init__(self, parent, models=None, parameters=None, extra_parameters=None):
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
        super().__init__()
        self.parent = parent
        #self.parameters = _collections.OrderedDict()
        #self.goparameters = _collections.OrderedDict()

        if parameters is None: parameters = {}
        self.circuit_weights = parameters.get('weights', None)
        self.protocol = parameters.get('protocol', None)
        self.profiler = parameters.get('profiler', None)
        self._final_mdc_store = parameters.get('final_mdc_store', None)
        self._final_objfn_cache = parameters.get('final_objfn_cache', None)
        self.final_objfn_builder = parameters.get('final_objfn_builder', _objfns.PoissonPicDeltaLogLFunction.builder())
        self._final_objfn = parameters.get('final_objfn', None)
        self._per_iter_mdc_store = parameters.get('per_iter_mdc_store', None)


        self.extra_parameters = extra_parameters if (extra_parameters is not None) else {}

        from .gst import GSTGaugeOptSuite as _GSTGaugeOptSuite
        self._gaugeopt_suite = _GSTGaugeOptSuite(gaugeopt_argument_dicts={})  # used for its serialization capabilities

        self.models = _collections.OrderedDict()
        self.num_iterations = 0
        self.confidence_region_factories = _collections.OrderedDict()

        #Set models
        if models:
            self.models.update(models)
            while ('iteration %d estimate' % self.num_iterations) in models:
                self.num_iterations += 1

        #Meta info
        self.meta = {}

        self.auxfile_types = {'parent': 'reset',
                              'models': 'dict:serialized-object',
                              'confidence_region_factories': 'fancykeydict:serialized-object',
                              'protocol': 'dir-serialized-object',
                              'profiler': 'reset',
                              '_final_mdc_store': 'reset',
                              '_final_objfn_cache': 'dir-serialized-object',
                              'final_objfn_builder': 'serialized-object',
                              '_final_objfn': 'reset',
                              '_gaugeopt_suite': 'serialized-object',
                              '_per_iter_mdc_store': 'reset'
                              }

    @property
    def parameters(self):
        #HACK for now, until we can remove references that access these parameters
        parameters = dict()
        parameters['protocol'] = self.protocol  # Estimates can hold sub-Protocols <=> sub-results
        parameters['profiler'] = self.profiler
        parameters['final_mdc_store'] = self._final_mdc_store
        parameters['final_objfn'] = self._final_objfn
        parameters['final_objfn_cache'] = self._final_objfn_cache
        parameters['final_objfn_builder'] = self.final_objfn_builder
        parameters['weights'] = self.circuit_weights
        parameters['per_iter_mdc_store'] = self._per_iter_mdc_store
        parameters.update(self.extra_parameters)
        #parameters['raw_objective_values']
        #parameters['model_test_values']
        return parameters

    @property
    def goparameters(self):
        #HACK for now, until external references are removed
        return self._gaugeopt_suite.gaugeopt_argument_dicts

    def write(self, dirname):
        """
        Write this protocol to a directory.

        Parameters
        ----------
        dirname : str
            The directory name to write.  This directory will be created
            if needed, and the files in an existing directory will be
            overwritten.

        Returns
        -------
        None
        """
        old_cw = self.circuit_weights
        if isinstance(old_cw, dict):
            new_cw : dict[str, float] = dict()
            for c, w in old_cw.items():
                if not isinstance(c, _Circuit):
                    raise ValueError()
                new_cw[c.str] = w
            self.circuit_weights = new_cw
        _io.write_obj_to_meta_based_dir(self, dirname, 'auxfile_types')
        self.circuit_weights = old_cw
        return

    def _add_auxiliary_write_ops_and_update_doc(self, doc, write_ops, mongodb, collection_name, overwrite_existing):
        _io.add_obj_auxtree_write_ops_and_update_doc(self, doc, write_ops, mongodb, collection_name,
                                                     'auxfile_types', overwrite_existing=overwrite_existing)

    @classmethod
    def _remove_from_mongodb(cls, mongodb, collection_name, doc_id, session, recursive):
        _io.remove_auxtree_from_mongodb(mongodb, collection_name, doc_id, 'auxfile_types', session,
                                        recursive=recursive)

    @classmethod
    def remove_from_mongodb(cls, mongodb_collection, doc_id, session=None):
        """
        Remove an Estimate from a MongoDB database.

        Returns
        -------
        bool
            `True` if the specified experiment design was removed, `False` if it didn't exist.
        """
        delcnt = _io.remove_auxtree_from_mongodb(mongodb_collection, doc_id, 'auxfile_types',
                                                 session=session)
        return bool(delcnt == 1)

    def retrieve_start_model(self, goparams):
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
        if goparams_list:
            return goparams_list[0].get('model', self.models['final iteration estimate'])
        else:
            return None

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
                if (label not in self._gaugeopt_suite.gaugeopt_argument_dicts) and \
                   (label not in self.models): break
        if hasattr(goparams, 'keys'):
            goparams_list = [goparams]
        elif goparams is None:
            goparams_list = []
            #since this will be empty much of the code/iteration below will
            #be skipped.
        else:
            goparams_list = goparams
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

        if model is not None:
            last_gs = model  # just use user-supplied result
            #sort the parameters by name for consistency
            for gop in goparams_list:
                ordered_goparams.append(_collections.OrderedDict(
                    [(k, gop[k]) for k in sorted(list(gop.keys()))]))
        else:
            for i, gop in enumerate(goparams_list):
                from ..algorithms import gaugeopt_to_target as _gaugeopt_to_target
                default_model = default_target_model = False
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
                        default_model = True
                    else: raise ValueError("Must supply 'model' in 'goparams' argument")

                if "target_model" not in gop:
                    if 'target' in self.models:
                        gop["target_model"] = self.models['target']
                        default_target_model = True
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

                #Don't store (and potentially serialize) model that we don't need to
                if default_model: del gop['model']
                if default_target_model: del gop['target_model']

                #sort the parameters by name for consistency
                ordered_goparams.append(_collections.OrderedDict(
                    [(k, gop[k]) for k in sorted(list(gop.keys()))]))

        assert(last_gs is not None)
        self.models[label] = last_gs

        if goparams_list: #only do this if goparams_list wasn't empty to begin with.
            #which would be the case except for the special case where the label is 'none'.
            self._gaugeopt_suite.gaugeopt_argument_dicts[label] = ordered_goparams \
                if len(goparams_list) > 1 else ordered_goparams[0]
        else:
            self._gaugeopt_suite.gaugeopt_argument_dicts[label] = None


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
        goparams = self._gaugeopt_suite.gaugeopt_argument_dicts[to_model_label]
        goparams_list = [goparams] if hasattr(goparams, 'keys') else goparams
        start_model = goparams_list[0]['model'].copy() if ('model' in goparams_list[0]) else ref_model.copy()
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
        weights = self.circuit_weights

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
                        plaq = gss.plaquette(x, y, empty_if_missing=True)
                        scaling_mx = _np.nan * _np.ones((plaq.num_rows, plaq.num_cols), 'd')
                        for i, j, opstr in plaq:
                            scaling_mx[i, j] = 1.0
                        sub_mxs[-1].append(scaling_mx)
                return p.dataset, sub_mxs  # copy dataset?
            else:
                return p.dataset

    def final_mdc_store(self, resource_alloc=None, array_types=('e', 'ep')):
        """
        The final (not intermediate) model-dataset-circuit storage object (MDC store) for this estimate.

        This object is created and cached as needed, and combined the final model, data set,
        and circuit list for this estimate.

        Parameters
        ----------
        resource_alloc : ResourceAllocation
            The resource allocation object used to create the MDC store.  This can just be left as
            `None` unless multiple processors are being utilized.  Note that this argument is only
            used when a MDC store needs to be created -- if this estimate has already created one
            then this argument is ignored.

        array_types : tuple
            A tuple of array types passed to the MDC store constructor (if a new MDC store needs
            to be created).  These affect how memory is allocated within the MDC store object and
            can enable (or disable) the use of certain MDC store functionality later on (e.g. the
            use of Jacobian or Hessian quantities).

        Returns
        -------
        ModelDatasetCircuitsStore
        """
        #Note: default array_types include 'ep' so, e.g. robust-stat re-optimization is possible.
        if self._final_mdc_store is None:
            assert(self.parent is not None), "Estimate must be linked with parent before objectivefn can be created"
            circuit_list = self.parent.circuit_lists['final']
            mdl = self.models['final iteration estimate']
            ds = self.parent.dataset
            self._final_mdc_store = _ModelDatasetCircuitStore(mdl, ds, circuit_list, resource_alloc,
                                                              array_types)
        return self._final_mdc_store

    def final_objective_fn(self, resource_alloc=None):
        """
        The final (not intermediate) objective function object for this estimate.

        This object is created and cached as needed, and is the evaluated (and sometimes
        optimized) objective function associated with this estimate.  Often this is a
        log-likelihood or chi-squared function, or a close variant.

        Parameters
        ----------
        resource_alloc : ResourceAllocation
            The resource allocation object used to create the MDC store underlying the objective function.
            This can just be left as `None` unless multiple processors are being utilized.  Note that this
            argument is only used when an underlying MDC store needs to be created -- if this estimate has
            already created a MDC store then this argument is ignored.

        Returns
        -------
        MDCObjectiveFunction
        """
        if self._final_objfn is None:
            mdc_store = self.final_mdc_store(resource_alloc)
            objfn = self.final_objfn_builder.build_from_store(mdc_store)
            self._final_objfn = objfn
        return self._final_objfn

    def final_objective_fn_cache(self, resource_alloc=None):
        """
        The final (not intermediate) *serializable* ("cached") objective function object for this estimate.

        This is an explicitly serializable version of the final objective function, useful because is often
        doesn't need be constructed.  To become serializable, however, the objective function is stripped of
        any MPI comm or multi-processor information (since this may be different between loading and saving).
        This makes the cached objective function convenient for fast calls/usages of the objective function.

        Parameters
        ----------
        resource_alloc : ResourceAllocation
            The resource allocation object used to create the MDC store underlying the objective function.
            This can just be left as `None` unless multiple processors are being utilized - and in this case
            the *cached* objective function doesn't even benefit from these processors (but calls to
            :meth:`final_objective_fn` will return an objective function setup for multiple processors).
            Note that this argument is only used when there is no existing cached objective function and
            an underlying MDC store needs to be created.

        Returns
        -------
        CachedObjectiveFunction
        """
        if self._final_objfn_cache is None:
            objfn = self.final_objective_fn(resource_alloc)
            self._final_objfn_cache = _CachedObjectiveFunction(objfn)
        return self._final_objfn_cache

    def misfit_sigma(self, resource_alloc=None):
        """
        Returns the number of standard deviations (sigma) of model violation.

        Parameters
        ----------
        resource_alloc : ResourceAllocation, optional
            What resources are available for this computation.

        Returns
        -------
        float
        """
        p = self.parent
        ds = self.create_effective_dataset()
        mdl = self.models['final iteration estimate']
        circuit_list = p.circuit_lists['final']

        if ds == self.parent.dataset:  # no effective ds => we can use cached quantities
            objfn_cache = self.final_objective_fn_cache(resource_alloc)  # creates cache if needed
            fitqty = objfn_cache.chi2k_distributed_fn
        else:
            objfn = self.final_objfn_builder.build(mdl, ds, circuit_list, resource_alloc, verbosity=0)
            fitqty = objfn.chi2k_distributed_qty(objfn.fn())

        aliases = circuit_list.op_label_aliases if isinstance(circuit_list, _CircuitList) else None

        ds_allstrs = _tools.apply_aliases_to_circuits(circuit_list, aliases)
        ds_dof = ds.degrees_of_freedom(ds_allstrs)  # number of independent parameters in dataset
        mdl_dof = mdl.num_modeltest_params

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

        view.circuit_weights = self.circuit_weights
        view.protocol = self.protocol
        view.profiler = self.profiler
        view._final_mdc_store = self._final_mdc_store
        view._final_objfn_cache = self._final_objfn_cache
        view.final_objfn_builder = self.final_objfn_builder
        view._final_objfn = self._final_objfn
        view.extra_parameters = self.extra_parameters

        view.models = self.models
        view.confidence_region_factories = self.confidence_region_factories

        goparameters = self._gaugeopt_suite.gaugeopt_argument_dicts
        if gaugeopt_keys is None:
            gaugeopt_keys = list(goparameters.keys())
        elif isinstance(gaugeopt_keys, str):
            gaugeopt_keys = [gaugeopt_keys]
        for go_key in gaugeopt_keys:
            if go_key in goparameters:
                view._gaugopt_suite.gaugeopt_argument_dicts[go_key] = goparameters[go_key]

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

        cpy.circuit_weights = _copy.deepcopy(self.circuit_weights)
        cpy.protocol = _copy.deepcopy(self.protocol)
        cpy.profiler = _copy.deepcopy(self.profiler)
        cpy._final_mdc_store = _copy.deepcopy(self._final_mdc_store)
        cpy._final_objfn_cache = _copy.deepcopy(self._final_objfn_cache)
        cpy.final_objfn_builder = _copy.deepcopy(self.final_objfn_builder)
        cpy._final_objfn = _copy.deepcopy(self._final_objfn)
        cpy.extra_parameters = _copy.deepcopy(self.extra_parameters)
        cpy.num_iterations = self.num_iterations

        cpy._gaugeopt_suite = _copy.deepcopy(self._gaugeopt_suite)
        cpy.models = self.models.copy()
        cpy.confidence_region_factories = _copy.deepcopy(self.confidence_region_factories)
        for crf in cpy.confidence_region_factories.values():
            crf.set_parent(cpy)  # because deepcopy above blanks out parent link
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
        #s += " .parameters   -- a dictionary of simulation parameters:\n"
        #s += " ---------------------------------------------------------\n"
        #s += "  " + "\n  ".join(list(self.parameters.keys())) + "\n"
        #s += "\n"
        s += " .goparameters   -- a dictionary of gauge-optimization parameter dictionaries:\n"
        s += " ---------------------------------------------------------\n"
        s += "  " + "\n  ".join(list(self.goparameters.keys())) + "\n"
        s += "\n"
        return s

    def __getstate__(self):
        to_pickle = self.__dict__.copy()

        # don't pickle MDC objective function or store objects b/c they might contain
        #  comm objects (in their layouts)
        del to_pickle['_final_mdc_store']
        del to_pickle['_final_objfn']
        del to_pickle['_final_objfn_cache']

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

        # reset MDC objective function and store objects
        state_dict['_final_mdc_store'] = None
        state_dict['_final_objfn'] = None
        state_dict['_final_objfn_cache'] = None

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
