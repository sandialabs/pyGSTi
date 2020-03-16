""" GST Protocol objects """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import time as _time
import os as _os
import numpy as _np
import pickle as _pickle
import collections as _collections
import warnings as _warnings
import itertools as _itertools
import copy as _copy
import scipy.optimize as _spo
from scipy.stats import chi2 as _chi2

from . import protocol as _proto
from .modeltest import ModelTest as _ModelTest
from .. import objects as _objs
from .. import algorithms as _alg
from .. import construction as _construction
from .. import io as _io
from .. import tools as _tools
from .. import optimize as _opt

from ..objects import wildcardbudget as _wild
from ..objects.profiler import DummyProfiler as _DummyProfiler
from ..objects import objectivefns as _objfns


#For results object:
from ..objects.estimate import Estimate as _Estimate
from ..objects.circuitstructure import LsGermsStructure as _LsGermsStructure
from ..objects.circuitstructure import LsGermsSerialStructure as _LsGermsSerialStructure
from ..objects.gaugegroup import TrivialGaugeGroup as _TrivialGaugeGroup
from ..objects.gaugegroup import TrivialGaugeGroupElement as _TrivialGaugeGroupElement


ROBUST_SUFFIX_LIST = [".robust", ".Robust", ".robust+", ".Robust+"]
DEFAULT_BAD_FIT_THRESHOLD = 2.0


class HasTargetModel(object):
    """ Adds to an experiment design a target model """

    def __init__(self, targetModelFilenameOrObj):
        self.target_model = _load_model(targetModelFilenameOrObj)
        self.auxfile_types['target_model'] = 'pickle'


class GateSetTomographyDesign(_proto.CircuitListsDesign, HasTargetModel):
    """ Minimal experiment design needed for GST """

    def __init__(self, targetModelFilenameOrObj, circuit_lists, all_circuits_needing_data=None,
                 qubit_labels=None, nested=False):
        super().__init__(circuit_lists, all_circuits_needing_data, qubit_labels, nested)
        HasTargetModel.__init__(self, targetModelFilenameOrObj)


class StructuredGSTDesign(GateSetTomographyDesign, _proto.CircuitStructuresDesign):
    """ GST experiment design where circuits are structured by length and germ (typically). """

    def __init__(self, targetModelFilenameOrObj, circuit_structs, qubit_labels=None,
                 nested=False):
        _proto.CircuitStructuresDesign.__init__(self, circuit_structs, qubit_labels, nested)
        HasTargetModel.__init__(self, targetModelFilenameOrObj)
        #Note: we *don't* need to init GateSetTomographyDesign here, only HasTargetModel,
        # GateSetTomographyDesign's non-target-model data is initialized by CircuitStructuresDesign.


class StandardGSTDesign(StructuredGSTDesign):
    """ Standard GST experiment design consisting of germ-powers sandwiched between fiducials. """

    def __init__(self, targetModelFilenameOrObj, prepStrsListOrFilename, effectStrsListOrFilename,
                 germsListOrFilename, maxLengths, germLengthLimits=None, fidPairs=None, keepFraction=1,
                 keepSeed=None, includeLGST=True, nest=True, sequenceRules=None, opLabelAliases=None,
                 dscheck=None, actionIfMissing="raise", qubit_labels=None, verbosity=0,
                 add_default_protocol=False):

        #Get/load fiducials and germs
        prep, meas, germs = _load_fiducials_and_germs(
            prepStrsListOrFilename,
            effectStrsListOrFilename,
            germsListOrFilename)
        self.prep_fiducials = prep
        self.meas_fiducials = meas
        self.germs = germs
        self.maxlengths = maxLengths
        self.germ_length_limits = germLengthLimits
        self.includeLGST = includeLGST
        self.aliases = opLabelAliases
        self.sequence_rules = sequenceRules

        #Hardcoded for now... - include so gets written when serialized
        self.truncation_method = "whole germ powers"
        self.nested = nest

        #FPR support
        self.fiducial_pairs = fidPairs
        self.fpr_keep_fraction = keepFraction
        self.fpr_keep_seed = keepSeed

        #TODO: add a line_labels arg to make_lsgst_structs and pass qubit_labels in?
        target_model = _load_model(targetModelFilenameOrObj)
        structs = _construction.make_lsgst_structs(
            target_model, self.prep_fiducials, self.meas_fiducials, self.germs,
            self.maxlengths, self.fiducial_pairs, self.truncation_method, self.nested,
            self.fpr_keep_fraction, self.fpr_keep_seed, self.includeLGST,
            self.aliases, self.sequence_rules, dscheck, actionIfMissing,
            self.germ_length_limits, verbosity)
        #FUTURE: add support for "advanced options" (probably not in __init__ though?):
        # truncScheme=advancedOptions.get('truncScheme', "whole germ powers")

        super().__init__(target_model, structs, qubit_labels, self.nested)
        self.auxfile_types['prep_fiducials'] = 'text-circuit-list'
        self.auxfile_types['meas_fiducials'] = 'text-circuit-list'
        self.auxfile_types['germs'] = 'text-circuit-list'
        self.auxfile_types['germ_length_limits'] = 'pickle'
        self.auxfile_types['fiducial_pairs'] = 'pickle'
        if add_default_protocol:
            self.add_default_protocol(StandardGST(name='StdGST'))


class GSTInitialModel(object):
    @classmethod
    def build(cls, obj):
        return obj if isinstance(obj, GSTInitialModel) else cls(obj)

    def __init__(self, model=None, starting_point=None, depolarize_start=0, randomize_start=0,
                 lgst_gaugeopt_tol=None, contractStartToCPTP=False):
        # Note: starting_point can be an initial model or string
        self.model = model
        if starting_point is None:
            self.starting_point = "target" if (model is None) else "User-supplied-Model"
        else:
            self.starting_point = starting_point

        self.lgst_gaugeopt_tol = lgst_gaugeopt_tol
        self.contract_start_to_cptp = contractStartToCPTP
        self.depolarize_start = depolarize_start
        self.randomize_start = randomize_start

    def get_model(self, target_model, gaugeopt_target, lgst_struct, dataset, qubit_labels, comm):
        #Get starting point (model), which is used to compute other quantities
        # Note: should compute on rank 0 and distribute?
        startingPt = self.starting_point
        if startingPt == "User-supplied-Model":
            mdl_start = self.model

        elif startingPt in ("LGST", "LGST-if-possible"):
            #lgst_advanced = advancedOptions.copy(); lgst_advanced.update({'estimateLabel': "LGST", 'onBadFit': []})
            mdl_start = self.model if (self.model is not None) else target_model
            lgst = LGST(mdl_start,
                        gaugeopt_suite={'lgst_gaugeopt': {'tol': self.lgst_gaugeopt_tol}},
                        gaugeopt_target=gaugeopt_target, badfit_options=None, name="LGST")

            try:  # see if LGST can be run on this data
                if lgst_struct:
                    lgst_data = _proto.ProtocolData(StructuredGSTDesign(mdl_start, [lgst_struct], qubit_labels),
                                                    dataset)
                    lgst.check_if_runnable(lgst_data)
                    startingPt = "LGST"
                else:
                    raise ValueError("Experiment design must contain circuit structures in order to run LGST")
            except ValueError as e:
                if startingPt == "LGST": raise e  # error if we *can't* run LGST

                #Fall back to target or custom model
                if self.model is not None:
                    startingPt = "User-supplied-Model"
                    mdl_start = self.model
                else:
                    startingPt = "target"
                    mdl_start = target_model

            if startingPt == "LGST":
                lgst_results = lgst.run(lgst_data)
                mdl_start = lgst_results.estimates['LGST'].models['lgst_gaugeopt']

        elif startingPt == "target":
            mdl_start = target_model
        else:
            raise ValueError("Invalid starting point: %s" % startingPt)

        #Post-processing mdl_start : done only on root proc in case there is any nondeterminism.
        if comm is None or comm.Get_rank() == 0:
            #Advanced Options can specify further manipulation of starting model
            if self.contract_start_to_cptp:
                mdl_start = _alg.contract(mdl_start, "CPTP")
                raise ValueError(
                    "'contractStartToCPTP' has been removed b/c it can change the parameterization of a model")
            if self.depolarize_start > 0:
                mdl_start = mdl_start.depolarize(op_noise=self.depolarize_start)
            if self.randomize_start > 0:
                v = mdl_start.to_vector()
                vrand = 2 * (_np.random.random(len(v)) - 0.5) * self.randomize_start
                mdl_start.from_vector(v + vrand)

            if comm is not None:  # broadcast starting model
                #OLD: comm.bcast(mdl_start, root=0)
                # just broadcast *vector* to avoid huge pickles (if cached calcs!)
                comm.bcast(mdl_start.to_vector(), root=0)
        else:
            #OLD: mdl_start = comm.bcast(None, root=0)
            v = comm.bcast(None, root=0)
            mdl_start.from_vector(v)

        return mdl_start


class GSTBadFitOptions(object):
    @classmethod
    def build(cls, obj):
        if isinstance(obj, GSTBadFitOptions):
            return obj
        else:  # assum obj is a dict of arguments
            return cls(**obj) if obj else cls()  # allow obj to be None => defaults

    def __init__(self, threshold=DEFAULT_BAD_FIT_THRESHOLD, actions=(),
                 wildcard_budget_includes_spam=True, wildcard_smart_init=True):
        self.threshold = threshold
        self.actions = actions  # e.g. ["wildcard", "Robust+"]; empty list => 'do nothing'
        self.wildcard_budget_includes_spam = wildcard_budget_includes_spam
        self.wildcard_smart_init = wildcard_smart_init


class GSTObjFnBuilders(object):
    @classmethod
    def build(cls, obj):
        if isinstance(obj, cls): return obj
        elif obj is None: return cls.init_simple()
        elif isinstance(obj, dict): return cls.init_simple(**obj)
        elif isinstance(obj, (list, tuple)): return cls(*obj)
        else: raise ValueError("Cannot build a GSTObjFnBuilders object from '%s'" % str(type(obj)))

    @classmethod
    def init_simple(cls, objective='logl', freqWeightedChi2=False, alwaysPerformMLE=False, onlyPerformMLE=False):
        chi2_builder = _objfns.ObjectiveFunctionBuilder.simple('chi2', freqWeightedChi2)
        mle_builder = _objfns.ObjectiveFunctionBuilder.simple('logl')

        if objective == "chi2":
            iteration_builders = [chi2_builder]
            final_builders = []

        elif objective == "logl":
            if alwaysPerformMLE:
                iteration_builders = [mle_builder] if onlyPerformMLE else [chi2_builder, mle_builder]
                final_builders = []
            else:
                iteration_builders = [chi2_builder]
                final_builders = [mle_builder]
        else:
            raise ValueError("Invalid objective: %s" % objective)
        return cls(iteration_builders, final_builders)

    def __init__(self, iteration_builders, final_builders=()):
        self.iteration_builders = iteration_builders
        self.final_builders = final_builders


class GateSetTomography(_proto.Protocol):
    """ The core gate set tomography protocol, which optimizes a parameterized model to (best) fit a data set."""

    def __init__(self, initial_model=None, gaugeopt_suite='stdgaugeopt',
                 gaugeopt_target=None, objfn_builders=None, optimizer=None,
                 badfit_options=None, verbosity=2, name=None):
        """
        TODO: docstring
        Note: initialModel can also be an InitialModel object
        """

        super().__init__(name)
        self.initial_model = GSTInitialModel.build(initial_model)
        self.gaugeopt_suite = gaugeopt_suite
        self.gaugeopt_target = gaugeopt_target
        self.badfit_options = GSTBadFitOptions.build(badfit_options)
        self.verbosity = verbosity

        if isinstance(optimizer, _opt.Optimizer):
            self.optimizer = optimizer
        else:
            if optimizer is None: optimizer = {}
            if 'first_fditer' not in optimizer:  # then add default first_fditer value
                mdl = self.initial_model.model
                optimizer['first_fditer'] = 0 if mdl and mdl.simtype in ("termorder", "termgap") else 1
            self.optimizer = _opt.CustomLMOptimizer.build(optimizer)

        objfn_builders = GSTObjFnBuilders.build(objfn_builders)
        self.iteration_builders = objfn_builders.iteration_builders
        self.final_builders = objfn_builders.final_builders

        self.auxfile_types['initial_model'] = 'pickle'
        self.auxfile_types['gaugeopt_suite'] = 'pickle'  # TODO - better later? - json?
        self.auxfile_types['gaugeopt_target'] = 'pickle'  # TODO - better later? - json?
        self.auxfile_types['iteration_builders'] = 'pickle'
        self.auxfile_types['final_builders'] = 'pickle'

        #Advanced options that could be changed by users who know what they're doing
        #self.estimate_label = estimate_label -- just use name?
        self.profile = 1
        self.record_output = True
        self.distribute_method = "default"
        self.oplabel_aliases = None
        self.circuit_weights = None
        self.unreliable_ops = ('Gcnot', 'Gcphase', 'Gms', 'Gcn', 'Gcx', 'Gcz')


    #TODO: Maybe make methods like this separate functions??
    #def run_using_germs_and_fiducials(self, dataset, target_model, prep_fiducials, meas_fiducials, germs, maxLengths):
    #    design = StandardGSTDesign(target_model, prep_fiducials, meas_fiducials, germs, maxLengths)
    #    return self.run(_proto.ProtocolData(design, dataset))
    #
    #def run_using_circuit_structures(self, target_model, circuit_structs, dataset):
    #    design = StructuredGSTDesign(target_model, circuit_structs)
    #    return self.run(_proto.ProtocolData(design, dataset))
    #
    #def run_using_circuit_lists(self, target_model, circuit_lists, dataset):
    #    design = GateSetTomographyDesign(target_model, circuit_lists)
    #    return self.run(_proto.ProtocolData(design, dataset))

    def run(self, data, memlimit=None, comm=None):

        tRef = _time.time()

        profile = self.profile
        if profile == 0: profiler = _DummyProfiler()
        elif profile == 1: profiler = _objs.Profiler(comm, False)
        elif profile == 2: profiler = _objs.Profiler(comm, True)
        else: raise ValueError("Invalid value for 'profile' argument (%s)" % profile)

        printer = _objs.VerbosityPrinter.build_printer(self.verbosity, comm)
        if self.record_output and not printer.is_recording():
            printer.start_recording()

        resource_alloc = _objfns.ResourceAllocation(comm, memlimit, profiler,
                                                    distributeMethod=self.distribute_method)

        try:  # take structs if available
            circuit_lists_or_structs = data.edesign.circuit_structs
            first_struct = data.edesign.circuit_structs[0]  # for LGST
            aliases = circuit_lists_or_structs[-1].aliases
        except:
            circuit_lists_or_structs = data.edesign.circuit_lists
            first_struct = None  # for LGST
            aliases = None
        ds = data.dataset

        if self.oplabel_aliases:  # override any other aliases with ones specifically given
            aliases = self.oplabel_aliases

        bulk_circuit_lists = [_objfns.BulkCircuitList(lst, aliases, self.circuit_weights)
                              for lst in circuit_lists_or_structs]

        tNxt = _time.time(); profiler.add_time('GST: loading', tRef); tRef = tNxt

        mdl_start = self.initial_model.get_model(data.edesign.target_model, self.gaugeopt_target,
                                                 first_struct, data.dataset, data.edesign.qubit_labels, comm)

        tNxt = _time.time(); profiler.add_time('GST: Prep Initial seed', tRef); tRef = tNxt

        #Run Long-sequence GST on data
        mdl_lsgst_list, optimums_list, final_cache = _alg.do_iterative_gst(
            ds, mdl_start, bulk_circuit_lists, self.optimizer,
            self.iteration_builders, self.final_builders,
            resource_alloc, printer)

        tNxt = _time.time(); profiler.add_time('GST: total iterative optimization', tRef); tRef = tNxt

        #set parameters
        parameters = _collections.OrderedDict()
        parameters['protocol'] = self  # Estimates can hold sub-Protocols <=> sub-results
        parameters['final_cache'] = final_cache  # ComputationCache associated w/final circuit list
        # Note: we associate 'final_cache' with the Estimate, which means we assume that *all*
        # of the models in the estimate can use same evaltree, have the same default prep/POVMs, etc.

        #TODO: add qtys abot fit from optimums_list

        ret = ModelEstimateResults(data, self)
        estimate = _Estimate.gst_init(ret, data.edesign.target_model, mdl_start, mdl_lsgst_list, parameters)
        ret.add_estimate(estimate, estimate_key=self.name)
        return _add_gaugeopt_and_badfit(ret, self.name, mdl_lsgst_list[-1], data.edesign.target_model,
                                        self.gaugeopt_suite, self.gaugeopt_target, self.unreliable_ops,
                                        self.badfit_options, self.final_builders[-1], self.optimizer,
                                        resource_alloc, printer)


class LinearGateSetTomography(_proto.Protocol):
    """ The linear gate set tomography protocol."""

    def __init__(self, target_model=None, gaugeopt_suite='stdgaugeopt', gaugeopt_target=None,
                 badfit_options=None, verbosity=2, name=None):
        super().__init__(name)
        self.target_model = target_model
        self.gaugeopt_suite = gaugeopt_suite
        self.gaugeopt_target = gaugeopt_target
        self.badfit_options = GSTBadFitOptions.build(badfit_options)        
        self.verbosity = verbosity

        #Advanced options that could be changed by users who know what they're doing
        self.profile = 1
        self.record_output = True
        self.oplabels = "default"
        self.oplabel_aliases = None
        self.unreliable_ops = ('Gcnot', 'Gcphase', 'Gms', 'Gcn', 'Gcx', 'Gcz')

    def check_if_runnable(self, data):
        """Raises a ValueError if LGST cannot be run on data"""
        edesign = data.edesign

        target_model = self.target_model if (self.target_model is not None) else edesign.target_model
        if isinstance(target_model, _objs.ExplicitOpModel):
            if not all([(isinstance(g, _objs.FullDenseOp)
                         or isinstance(g, _objs.TPDenseOp))
                        for g in target_model.operations.values()]):
                raise ValueError("LGST can only be applied to explicit models with dense operators")
        else:
            raise ValueError("LGST can only be applied to explicit models with dense operators")

        if not isinstance(edesign, _proto.CircuitStructuresDesign):
            raise ValueError("LGST must be given an experiment design with fiducials!")
        if len(edesign.circuit_structs) != 1:
            raise ValueError("There should only be one circuit structure in the input exp-design!")
        circuit_struct = edesign.circuit_structs[0]

        validStructTypes = (_objs.LsGermsStructure, _objs.LsGermsSerialStructure)
        if not isinstance(circuit_struct, validStructTypes):
            raise ValueError("Cannot run LGST: fiducials not specified in input experiment design!")

    def run(self, data, memlimit=None, comm=None):

        self.check_if_runnable(data)

        edesign = data.edesign
        target_model = self.target_model if (self.target_model is not None) else edesign.target_model
        circuit_struct = edesign.circuit_structs[0]

        profile = self.profile
        if profile == 0: profiler = _DummyProfiler()
        elif profile == 1: profiler = _objs.Profiler(comm, False)
        elif profile == 2: profiler = _objs.Profiler(comm, True)
        else: raise ValueError("Invalid value for 'profile' argument (%s)" % profile)

        printer = _objs.VerbosityPrinter.build_printer(self.verbosity, comm)
        if self.record_output and not printer.is_recording():
            printer.start_recording()

        resource_alloc = _objfns.ResourceAllocation(comm, memlimit, profiler,
                                                    distributeMethod="default")

        ds = data.dataset
        aliases = circuit_struct.aliases if self.oplabel_aliases is None else self.oplabel_aliases
        opLabels = self.oplabels if self.oplabels != "default" else \
            list(target_model.operations.keys()) + list(target_model.instruments.keys())

        # Note: this returns a model with the *same* parameterizations as target_model
        mdl_lgst = _alg.do_lgst(ds, circuit_struct.prepStrs, circuit_struct.effectStrs, target_model,
                                opLabels, svdTruncateTo=target_model.get_dimension(),
                                opLabelAliases=aliases, verbosity=printer)

        parameters = _collections.OrderedDict()
        parameters['protocol'] = self  # Estimates can hold sub-Protocols <=> sub-results
        #parameters['objective'] = 'lgst'

        ret = ModelEstimateResults(data, self)
        estimate = _Estimate(ret, {'target': target_model, 'lgst': mdl_lgst,
                                   'iteration estimates': [mdl_lgst],
                                   'final iteration estimate': mdl_lgst},
                             parameters)
        ret.add_estimate(estimate, estimate_key=self.name)
        return _add_gaugeopt_and_badfit(ret, self.name, mdl_lgst, data.edesign.target_model, self.gaugeopt_suite,
                                        self.gaugeopt_target, self.unreliable_ops, self.badfit_options,
                                        None, None, resource_alloc, printer)

#HERE's what we need to do:
#x continue upgrading this module: StandardGST (similar to others, but need "appendTo" workaround)
#x upgrade ModelTest
#x fix do_XXX driver functions in longsequence.py
# (maybe upgraded advancedOptions there to a class w/validation, etc)
# upgrade likelihoodfns.py and chi2.py to use objective funtions -- should be lots of consolidation, and maybe
# add hessian & non-poisson-pic logl to objective fns.
# fix report generation (changes to ModelEstimateResults)
# run/update tests - test out custom/new objective functions.
class StandardGST(_proto.Protocol):
    """The standard-practice GST protocol."""

    def __init__(self, modes="TP,CPTP,Target",
                 gaugeopt_suite='stdgaugeopt',
                 gaugeopt_target=None, modelsToTest=None,
                 objfn_builders=None, optimizer=None,
                 badfit_options=None, verbosity=2, name=None):
        
        super().__init__(name)
        self.modes = modes.split(',')
        self.models_to_test = modelsToTest
        self.gaugeopt_suite = gaugeopt_suite
        self.gaugeopt_target = gaugeopt_target
        self.objfn_builders = objfn_builders
        self.optimizer = optimizer
        self.badfit_options = badfit_options
        self.verbosity = verbosity

        self.auxfile_types['models_to_test'] = 'pickle'
        self.auxfile_types['gaugeopt_suite'] = 'pickle'
        self.auxfile_types['gaugeopt_target'] = 'pickle'
        self.auxfile_types['advancedOptions'] = 'pickle'
        self.auxfile_types['comm'] = 'reset'

        #Advanced options that could be changed by users who know what they're doing
        self.starting_point = {}  # a dict whose keys are modes

    #def run_using_germs_and_fiducials(self, dataset, target_model, prep_fiducials, meas_fiducials, germs, maxLengths):
    #    design = StandardGSTDesign(target_model, prep_fiducials, meas_fiducials, germs, maxLengths)
    #    data = _proto.ProtocolData(design, dataset)
    #    return self.run(data)

    def run(self, data, memlimit=None, comm=None):
        printer = _objs.VerbosityPrinter.build_printer(self.verbosity, comm)

        modes = self.modes
        modelsToTest = self.models_to_test
        if modelsToTest is None: modelsToTest = {}

        ret = ModelEstimateResults(data, self)
        with printer.progress_logging(1):
            for i, mode in enumerate(modes):
                printer.show_progress(i, len(modes), prefix='-- Std Practice: ', suffix=' (%s) --' % mode)

                if mode == "Target":
                    model_to_test = data.edesign.target_model.copy()  # no parameterization change
                    mdltest = _ModelTest(model_to_test, None, self.gaugeopt_suite, self.gaugeopt_target,
                                         None, self.badfit_options, verbosity=printer - 1, name=mode)
                    result = mdltest.run(data, memlimit, comm)
                    ret.add_estimates(result)

                elif mode in ('full', 'TP', 'CPTP', 'H+S', 'S', 'static'):  # mode is a parameterization
                    parameterization = mode  # for now, 1-1 correspondence
                    initial_model = data.edesign.target_model.copy()
                    initial_model.set_all_parameterizations(parameterization)
                    initial_model = GSTInitialModel(initial_model, self.starting_point.get(mode, None))

                    gst = GST(initial_model, self.gaugeopt_suite, self.gaugeopt_target, self.objfn_builders,
                              self.optimizer, self.badfit_options, verbosity=printer - 1, name=mode)
                    result = gst.run(data, memlimit, comm)
                    ret.add_estimates(result)

                elif mode in modelsToTest:
                    mdltest = _ModelTest(modelsToTest[mode], None, self.gaugeopt_suite, self.gaugeopt_target,
                                         None, self.badfit_options, verbosity=printer - 1, name=mode)
                    result = mdltest.run(data, memlimit, comm)
                    ret.add_estimates(result)
                else:
                    raise ValueError("Invalid item in 'modes' argument: %s" % mode)

        return ret


# ------------------ HELPER FUNCTIONS -----------------------------------

def gaugeopt_suite_to_dictionary(gaugeOptSuite, model, unreliableOps=(), verbosity=0):
    """
    Constructs a dictionary of gauge-optimization parameter dictionaries based
    on "gauge optimization suite" name(s).

    This is primarily a helper function for :func:`do_stdpractice_gst`, but can
    be useful in its own right for constructing the would-be gauge optimization
    dictionary used in :func:`do_stdpractice_gst` and modifying it slightly before
    before passing it in (`do_stdpractice_gst` will accept a raw dictionary too).

    Parameters
    ----------
    gaugeOptSuite : str or dict, optional
        Specifies which gauge optimizations to perform on each estimate.  An
        string (see below) specifies a built-in set of gauge optimizations,
        otherwise `gaugeOptSuite` should be a dictionary of gauge-optimization
        parameter dictionaries, as specified by the `gaugeOptParams` argument
        of :func:`do_long_sequence_gst`.  The key names of `gaugeOptSuite` then
        label the gauge optimizations within the resuling `Estimate` objects.
        The built-in gauge optmization suites are:

          - "single" : performs only a single "best guess" gauge optimization.
          - "varySpam" : varies spam weight and toggles SPAM penalty (0 or 1).
          - "varySpamWt" : varies spam weight but no SPAM penalty.
          - "varyValidSpamWt" : varies spam weight with SPAM penalty == 1.
          - "toggleValidSpam" : toggles spame penalty (0 or 1); fixed SPAM wt.
          - "unreliable2Q" : adds branch to a spam suite that weights 2Q gates less
          - "none" : no gauge optimizations are performed.

    model : Model
        A model which specifies the dimension (i.e. parameterization) of the
        gauge-optimization and the basis.  Typically the model that is optimized
        or the ideal model using the same parameterization and having the correct
        default-gauge-group as the model that is optimized.

    unreliableOps : tuple, optional
        A list of gate names considered to be less reliable, and which therefore
        should be weighted less in certain gauge optimization suites (those ending
        with "-unreliable2Q", e.g. "stdgaugeopt-unreliable2Q").

    verbosity : int
        The verbosity to attach to the various gauge optimization parameter
        dictionaries.

    Returns
    -------
    dict
        A dictionary whose keys are the labels of the different gauge
        optimizations to perform and whose values are the corresponding
        dictionaries of arguments to :func:`gaugeopt_to_target` (or lists
        of such dictionaries for a multi-stage gauge optimization).
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    if gaugeOptSuite is None:
        gaugeOptSuite = {}
    elif isinstance(gaugeOptSuite, str):
        gaugeOptSuite = {gaugeOptSuite: gaugeOptSuite}
    elif isinstance(gaugeOptSuite, tuple):
        gaugeOptSuite = {nm: nm for nm in gaugeOptSuite}

    assert(isinstance(gaugeOptSuite, dict)), \
        "Can't convert type '%s' to a gauge optimization suite dictionary!" % str(type(gaugeOptSuite))

    #Build ordered dict of gauge optimization parameters
    gaugeOptSuite_dict = _collections.OrderedDict()
    for lbl, goparams in gaugeOptSuite.items():
        if isinstance(goparams, str):
            _update_gaugeopt_dict_from_suitename(gaugeOptSuite_dict, lbl, goparams,
                                                 model, unreliableOps, printer)
        elif hasattr(goparams, 'keys'):
            gaugeOptSuite_dict[lbl] = goparams.copy()
            gaugeOptSuite_dict[lbl].update({'verbosity': printer})
        else:
            assert(isinstance(goparams, list)), "If not a dictionary, gauge opt params should be a list of dicts!"
            gaugeOptSuite_dict[lbl] = []
            for goparams_stage in goparams:
                dct = goparams_stage.copy()
                dct.update({'verbosity': printer})
                gaugeOptSuite_dict[lbl].append(dct)

    return gaugeOptSuite_dict


def _update_gaugeopt_dict_from_suitename(gaugeOptSuite_dict, rootLbl, suiteName, model, unreliableOps, printer):
    if suiteName in ("stdgaugeopt", "stdgaugeopt-unreliable2Q"):

        stages = []  # multi-stage gauge opt
        gg = model.default_gauge_group
        if isinstance(gg, _objs.TrivialGaugeGroup):
            if suiteName == "stdgaugeopt-unreliable2Q" and model.dim == 16:
                if any([gl in model.operations.keys() for gl in unreliableOps]):
                    gaugeOptSuite_dict[rootLbl] = {'verbosity': printer}
            else:
                #just do a single-stage "trivial" gauge opts using default group
                gaugeOptSuite_dict[rootLbl] = {'verbosity': printer}

        elif gg is not None:

            #Stage 1: plain vanilla gauge opt to get into "right ballpark"
            if gg.name in ("Full", "TP"):
                stages.append(
                    {
                        'itemWeights': {'gates': 1.0, 'spam': 1.0},
                        'verbosity': printer
                    })

            #Stage 2: unitary gauge opt that tries to nail down gates (at
            #         expense of spam if needed)
            stages.append(
                {
                    'itemWeights': {'gates': 1.0, 'spam': 0.0},
                    'gauge_group': _objs.UnitaryGaugeGroup(model.dim, model.basis),
                    'verbosity': printer
                })

            #Stage 3: spam gauge opt that fixes spam scaling at expense of
            #         non-unital parts of gates (but shouldn't affect these
            #         elements much since they should be small from Stage 2).
            s3gg = _objs.SpamGaugeGroup if (gg.name == "Full") else \
                _objs.TPSpamGaugeGroup
            stages.append(
                {
                    'itemWeights': {'gates': 0.0, 'spam': 1.0},
                    'spam_penalty_factor': 1.0,
                    'gauge_group': s3gg(model.dim),
                    'oob_check_interval': 1,
                    'verbosity': printer
                })

            if suiteName == "stdgaugeopt-unreliable2Q" and model.dim == 16:
                if any([gl in model.operations.keys() for gl in unreliableOps]):
                    stage2_item_weights = {'gates': 1, 'spam': 0.0}
                    for gl in unreliableOps:
                        if gl in model.operations.keys(): stage2_item_weights[gl] = 0.01
                    stages_2QUR = [stage.copy() for stage in stages]  # ~deep copy of stages
                    iStage2 = 1 if gg.name in ("Full", "TP") else 0
                    stages_2QUR[iStage2]['itemWeights'] = stage2_item_weights
                    gaugeOptSuite_dict[rootLbl] = stages_2QUR  # add additional gauge opt
                else:
                    _warnings.warn(("`unreliable2Q` was given as a gauge opt suite, but none of the"
                                    " gate names in 'unreliableOps', i.e., %s,"
                                    " are present in the target model.  Omitting 'single-2QUR' gauge opt.")
                                   % (", ".join(unreliableOps)))
            else:
                gaugeOptSuite_dict[rootLbl] = stages  # can be a list of stage dictionaries

    elif suiteName in ("varySpam", "varySpamWt", "varyValidSpamWt", "toggleValidSpam") or \
        suiteName in ("varySpam-unreliable2Q", "varySpamWt-unreliable2Q",
                      "varyValidSpamWt-unreliable2Q", "toggleValidSpam-unreliable2Q"):

        baseWts = {'gates': 1}
        if suiteName.endswith("unreliable2Q") and model.dim == 16:
            if any([gl in model.operations.keys() for gl in unreliableOps]):
                base = {'gates': 1}
                for gl in unreliableOps:
                    if gl in model.operations.keys(): base[gl] = 0.01
                baseWts = base

        if suiteName == "varySpam":
            vSpam_range = [0, 1]; spamWt_range = [1e-4, 1e-1]
        elif suiteName == "varySpamWt":
            vSpam_range = [0]; spamWt_range = [1e-4, 1e-1]
        elif suiteName == "varyValidSpamWt":
            vSpam_range = [1]; spamWt_range = [1e-4, 1e-1]
        elif suiteName == "toggleValidSpam":
            vSpam_range = [0, 1]; spamWt_range = [1e-3]

        if suiteName == rootLbl:  # then shorten the root name
            rootLbl = "2QUR-" if suiteName.endswith("unreliable2Q") else ""

        for vSpam in vSpam_range:
            for spamWt in spamWt_range:
                lbl = rootLbl + "Spam %g%s" % (spamWt, "+v" if vSpam else "")
                itemWeights = baseWts.copy()
                itemWeights['spam'] = spamWt
                gaugeOptSuite_dict[lbl] = {
                    'itemWeights': itemWeights,
                    'spam_penalty_factor': vSpam, 'verbosity': printer}

    elif suiteName == "unreliable2Q":
        raise ValueError(("unreliable2Q is no longer a separate 'suite'.  You should precede it with the suite name, "
                          "e.g. 'stdgaugeopt-unreliable2Q' or 'varySpam-unreliable2Q'"))
    elif suiteName == "none":
        pass  # add nothing
    else:
        raise ValueError("Unknown gauge-optimization suite '%s'" % suiteName)


def _load_model(modelFilenameOrObj):
    if isinstance(modelFilenameOrObj, str):
        return _io.load_model(modelFilenameOrObj)
    else:
        return modelFilenameOrObj  # assume a Model object


def _load_fiducials_and_germs(prepStrsListOrFilename,
                              effectStrsListOrFilename,
                              germsListOrFilename):

    if isinstance(prepStrsListOrFilename, str):
        prepStrs = _io.load_circuit_list(prepStrsListOrFilename)
    else: prepStrs = prepStrsListOrFilename

    if effectStrsListOrFilename is None:
        effectStrs = prepStrs  # use same strings for effectStrs if effectStrsListOrFilename is None
    else:
        if isinstance(effectStrsListOrFilename, str):
            effectStrs = _io.load_circuit_list(effectStrsListOrFilename)
        else: effectStrs = effectStrsListOrFilename

    #Get/load germs
    if isinstance(germsListOrFilename, str):
        germs = _io.load_circuit_list(germsListOrFilename)
    else: germs = germsListOrFilename

    return prepStrs, effectStrs, germs


def _load_dataset(dataFilenameOrSet, comm, verbosity):
    """Loads a DataSet from the dataFilenameOrSet argument of functions in this module."""
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    if isinstance(dataFilenameOrSet, str):
        if comm is None or comm.Get_rank() == 0:
            if _os.path.splitext(dataFilenameOrSet)[1] == ".pkl":
                with open(dataFilenameOrSet, 'rb') as pklfile:
                    ds = _pickle.load(pklfile)
            else:
                ds = _io.load_dataset(dataFilenameOrSet, True, "aggregate", printer)
            if comm is not None: comm.bcast(ds, root=0)
        else:
            ds = comm.bcast(None, root=0)
    else:
        ds = dataFilenameOrSet  # assume a Dataset object

    return ds


def _get_lsgst_lists(dschk, target_model, prepStrs, effectStrs, germs,
                     maxLengths, advancedOptions, verbosity):
    """
    Sequence construction logic, fatctored into this separate
    function because it's shared do_long_sequence_gst and
    do_model_evaluation.
    """
    if advancedOptions is None: advancedOptions = {}

    #Update: now always include LGST strings unless advanced options says otherwise
    #Get starting point (so we know whether to include LGST strings)
    #LGSTcompatibleOps = all([(isinstance(g,_objs.FullDenseOp) or
    #                            isinstance(g,_objs.TPDenseOp))
    #                           for g in target_model.operations.values()])
    #if  LGSTcompatibleOps:
    #    startingPt = advancedOptions.get('starting point',"LGST")
    #else:
    #    startingPt = advancedOptions.get('starting point',"target")

    #Construct operation sequences
    actionIfMissing = advancedOptions.get('missingDataAction', 'drop')
    opLabels = advancedOptions.get(
        'opLabels', list(target_model.get_primitive_op_labels()))
    lsgstLists = _construction.stdlists.make_lsgst_structs(
        opLabels, prepStrs, effectStrs, germs, maxLengths,
        truncScheme=advancedOptions.get('truncScheme', "whole germ powers"),
        nest=advancedOptions.get('nestedCircuitLists', True),
        includeLGST=advancedOptions.get('includeLGST', True),
        opLabelAliases=advancedOptions.get('opLabelAliases', None),
        sequenceRules=advancedOptions.get('stringManipRules', None),
        dscheck=dschk, actionIfMissing=actionIfMissing,
        germLengthLimits=advancedOptions.get('germLengthLimits', None),
        verbosity=verbosity)
    assert(len(maxLengths) == len(lsgstLists))

    return lsgstLists


def _add_gaugeopt_and_badfit(results, estlbl, model_to_gaugeopt, target_model, gaugeopt_suite, gaugeopt_target,
                             unreliable_ops, badfit_options, objfn_builder, optimizer, resource_alloc, printer):
    tRef = _time.time()
    comm = resource_alloc.comm
    profiler = resource_alloc.profiler

    #Do final gauge optimization to *final* iteration result only
    if gaugeopt_suite:
        gaugeopt_target = gaugeopt_target if gaugeopt_target else target_model
        add_gauge_opt(results, estlbl, gaugeopt_suite, gaugeopt_target,
                      model_to_gaugeopt, unreliable_ops, comm, printer - 1)
    profiler.add_time('%s: gauge optimization' % estlbl, tRef); tRef = _time.time()

    add_badfit_estimates(results, estlbl, badfit_options, objfn_builder, optimizer, resource_alloc, printer)
    profiler.add_time('%s: add badfit estimates' % estlbl, tRef); tRef = _time.time()

    #Add recorded info (even robust-related info) to the *base*
    #   estimate label's "stdout" meta information
    if printer.is_recording():
        results.estimates[estlbl].meta['stdout'] = printer.stop_recording()

    return results


#TODO REMOVE
def OLD_package_into_results(callerProtocol, data, target_model, mdl_start, lsgstLists,
                          parameters, mdl_lsgst_list, gaugeopt_suite, gaugeopt_target,
                          comm, memLimit, output_pkl, verbosity,
                          profiler, evaltree_cache=None):
    # advancedOptions, opt_args, 
    """
    Performs all of the post-optimization processing common to
    do_long_sequence_gst and do_model_evaluation.

    Creates a Results object to be returned from do_long_sequence_gst
    and do_model_evaluation (passed in as 'callerName').  Performs
    gauge optimization, and robust data scaling (with re-optimization
    if needed and opt_args is not None - i.e. only for
    do_long_sequence_gst).
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    tRef = _time.time()
    callerName = callerProtocol.name

    #ret = advancedOptions.get('appendTo', None)
    #if ret is None:
    ret = ModelEstimateResults(data, callerProtocol)
    #else:
    #    # a dummy object to check compatibility w/ret2
    #    dummy = ModelEstimateResults(data, callerProtocol)
    #    ret.add_estimates(dummy)  # does nothing, but will complain when appropriate

    #add estimate to Results
    profiler.add_time('%s: results initialization' % callerName, tRef); tRef = _time.time()

    #Do final gauge optimization to *final* iteration result only
    if gaugeopt_suite:
        if gaugeopt_target is None: gaugeopt_target = target_model
        add_gauge_opt(ret, estlbl, gaugeopt_suite, gaugeopt_target,
                      mdl_lsgst_list[-1], comm, advancedOptions, printer - 1)
        profiler.add_time('%s: gauge optimization' % callerName, tRef)

    #Perform extra analysis if a bad fit was obtained - do this *after* gauge-opt b/c it mimics gaugeopts
    badFitThreshold = advancedOptions.get('badFitThreshold', DEFAULT_BAD_FIT_THRESHOLD)
    onBadFit = advancedOptions.get('onBadFit', [])  # ["wildcard"]) #["Robust+"]) # empty list => 'do nothing'
    badfit_opts = advancedOptions.get('badFitOptions', {'wildcard_budget_includes_spam': True,
                                                        'wildcard_smart_init': True})
    add_badfit_estimates(ret, estlbl, onBadFit, badFitThreshold, badfit_opts, opt_args, evaltree_cache,
                         comm, memLimit, printer)
    profiler.add_time('%s: add badfit estimates' % callerName, tRef); tRef = _time.time()

    #Add recorded info (even robust-related info) to the *base*
    #   estimate label's "stdout" meta information
    if printer.is_recording():
        ret.estimates[estlbl].meta['stdout'] = printer.stop_recording()

    #Write results to a pickle file if desired
    if output_pkl and (comm is None or comm.Get_rank() == 0):
        if isinstance(output_pkl, str):
            with open(output_pkl, 'wb') as pklfile:
                _pickle.dump(ret, pklfile)
        else:
            _pickle.dump(ret, output_pkl)

    return ret


#def add_gauge_opt(estimate, gaugeOptParams, target_model, starting_model,
#                  comm=None, verbosity=0):

def add_gauge_opt(results, base_est_label, gaugeopt_suite, target_model, starting_model,
                  unreliableOps, comm=None, verbosity=0):
    """
    Add a gauge optimization to an estimate.
    TODO: docstring - more details
    - ** target_model should have default gauge group set **
    - note: give results and base_est_label instead of an estimate so that related (e.g. badfit) estimates
      can also be updated -- it this isn't needed, than could just take an estimate as input
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)

    #Get gauge optimization dictionary
    gaugeOptSuite_dict = gaugeopt_suite_to_dictionary(gaugeopt_suite, starting_model,
                                                      unreliableOps, printer - 1)

    if target_model is not None:
        assert(isinstance(target_model, _objs.Model)), "`gaugeOptTarget` must be None or a Model"
        for goparams in gaugeOptSuite_dict.values():
            goparams_list = [goparams] if hasattr(goparams, 'keys') else goparams
            for goparams_dict in goparams_list:
                if 'targetModel' in goparams_dict:
                    _warnings.warn(("`gaugeOptTarget` argument is overriding"
                                    "user-defined targetModel in gauge opt"
                                    "param dict(s)"))
                goparams_dict.update({'targetModel': target_model})

    #Gauge optimize to list of gauge optimization parameters
    for goLabel, goparams in gaugeOptSuite_dict.items():

        printer.log("-- Performing '%s' gauge optimization on %s estimate --" % (goLabel, base_est_label), 2)

        #Get starting model
        results.estimates[base_est_label].add_gaugeoptimized(goparams, None, goLabel, comm, printer - 3)
        gsStart = results.estimates[base_est_label].get_start_model(goparams)

        #Gauge optimize data-scaled estimate also
        for suffix in ROBUST_SUFFIX_LIST:
            robust_est_label = base_est_label + suffix
            if robust_est_label in results.estimates:
                gsStart_robust = results.estimates[robust_est_label].get_start_model(goparams)

                if gsStart_robust.frobeniusdist(gsStart) < 1e-8:
                    printer.log("-- Conveying '%s' gauge optimization from %s to %s estimate --" %
                                (goLabel, base_est_label, robust_est_label), 2)
                    params = results.estimates[base_est_label].goparameters[goLabel]  # no need to copy here
                    gsopt = results.estimates[base_est_label].models[goLabel].copy()
                    results.estimates[robust_est_label].add_gaugeoptimized(params, gsopt, goLabel, comm, printer - 3)
                else:
                    printer.log("-- Performing '%s' gauge optimization on %s estimate --" %
                                (goLabel, robust_est_label), 2)
                    results.estimates[robust_est_label].add_gaugeoptimized(goparams, None, goLabel, comm, printer - 3)


def add_badfit_estimates(results, base_estimate_label, badfit_options, objfn_builder, optimizer,
                         resource_alloc=None, verbosity=0):
    #estimate_types=('wildcard',), badFitThreshold=None, badfit_opts=None,
    """
    Add any and all "bad fit" estimates to `results`.
    TODO: docstring
    """

    if badfit_options is None:
        return  # nothing to do

    comm = resource_alloc.comm if resource_alloc else None
    memLimit = resource_alloc.memLimit if resource_alloc else None
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    base_estimate = results.estimates[base_estimate_label]

    if badfit_options.threshold is not None and \
       base_estimate.misfit_sigma(use_accurate_Np=True, comm=comm) <= badfit_options.threshold:
        return  # fit is good enough - no need to add any estimates

    circuit_list = results.circuit_lists['final']
    mdl = base_estimate.models['final iteration estimate']
    cache = base_estimate.parameters.get('final_cache', None)
    parameters = base_estimate.parameters
    ds = results.dataset

    assert(parameters.get('weights', None) is None), \
        "Cannot perform bad-fit scaling when weights are already given!"

    #objective = parameters.get('objective', 'logl')
    #validStructTypes = (_objs.LsGermsStructure, _objs.LsGermsSerialStructure)
    #rawLists = [l.allstrs if isinstance(l, validStructTypes) else l
    #            for l in lsgstLists]
    #circuitList = rawLists[-1]  # use final circuit list
    #mdl = mdl_lsgst_list[-1]    # and model

    for badfit_typ in badfit_options.actions:
        new_params = parameters.copy()
        new_final_model = None

        if badfit_typ in ("robust", "Robust", "robust+", "Robust+"):
            new_params['weights'] = get_robust_scaling(badfit_typ, mdl, ds, circuit_list,
                                                       parameters, cache, comm, memLimit)
            if badfit_typ in ("Robust", "Robust+") and (optimizer is not None):
                mdl_reopt = reoptimize_with_weights(mdl, ds, circuit_list, new_params['weights'],
                                                    objfn_builder, optimizer, resource_alloc, cache, printer - 1)
                new_final_model = mdl_reopt

        elif badfit_typ == "wildcard":
            try:
                badfit_opts = {'wildcard_budget_includes_spam':  badfit_options.wildcard_budget_includes_spam,
                               'wildcard_smart_init': badfit_options.wildcard_smart_init}
                unmodeled = get_wildcard_budget(mdl, ds, circuit_list, parameters, badfit_opts,
                                                cache, comm, memLimit, printer - 1)
                base_estimate.parameters['unmodeled_error'] = unmodeled
                # new_params['unmodeled_error'] = unmodeled  # OLD: when we created a new estimate (seems unneces
            except NotImplementedError as e:
                printer.warning("Failed to get wildcard budget - continuing anyway.  Error was:\n" + str(e))
                new_params['unmodeled_error'] = None
            except AssertionError as e:
                printer.warning("Failed to get wildcard budget - continuing anyway.  Error was:\n" + str(e))
                new_params['unmodeled_error'] = None
            continue  # no need to add a new estimate - we just update the base estimate

        elif badfit_typ == "do nothing":
            continue  # go to next on-bad-fit directive

        else:
            raise ValueError("Invalid on-bad-fit directive: %s" % badfit_typ)

        # In case we've computed an updated final model, Just keep (?) old estimates of all
        # prior iterations (or use "blank" sentinel once this is supported).
        mdl_lsgst_list = base_estimate.models['iteration estimates']
        mdl_start = base_estimate.models.get('seed', None)
        target_model = base_estimate.models.get('target', None)

        models_by_iter = mdl_lsgst_list[:] if (new_final_model is None) \
            else mdl_lsgst_list[0:-1] + [new_final_model]

        results.add_estimate(_Estimate.gst_init(results, target_model, mdl_start, models_by_iter, new_params),
                             base_estimate_label + "." + badfit_typ)

        #Add gauge optimizations to the new estimate
        for gokey, gaugeOptParams in base_estimate.goparameters.items():
            if new_final_model is not None:
                unreliableOps = ()  # pass this in?
                add_gauge_opt(results, base_estimate_label + '.' + badfit_typ, {gokey: gaugeOptParams},
                              target_model, new_final_model, unreliableOps, comm, printer - 1)
            else:
                # add same gauge-optimized result as above
                go_gs_final = base_estimate.models[gokey]
                results.estimates[base_estimate_label + '.' + badfit_typ].add_gaugeoptimized(
                    gaugeOptParams.copy(), go_gs_final, gokey, comm, printer - 1)


def _get_fit_qty(model, ds, circuitList, parameters, evaltree_cache, comm, memLimit):
    # Get by-sequence goodness of fit
    objective = parameters.get('objective', 'logl')

    if objective == "chi2":
        fitQty = _tools.chi2_terms(model, ds, circuitList,
                                   parameters.get('minProbClipForWeighting', 1e-4),
                                   parameters.get('probClipInterval', (-1e6, 1e6)),
                                   False, False, memLimit,
                                   parameters.get('opLabelAliases', None),
                                   evaltree_cache=evaltree_cache, comm=comm)
    else:  # "logl" or "lgst"
        maxLogL = _tools.logl_max_terms(model, ds, circuitList,
                                        opLabelAliases=parameters.get(
                                            'opLabelAliases', None),
                                        evaltree_cache=evaltree_cache)

        logL = _tools.logl_terms(model, ds, circuitList,
                                 parameters.get('minProbClip', 1e-4),
                                 parameters.get('probClipInterval', (-1e6, 1e6)),
                                 parameters.get('radius', 1e-4),
                                 opLabelAliases=parameters.get('opLabelAliases', None),
                                 evaltree_cache=evaltree_cache, comm=comm)
        fitQty = 2 * (maxLogL - logL)
    return fitQty


def get_robust_scaling(scale_typ, model, ds, circuitList, parameters, evaltree_cache, comm, memLimit):
    """
    Get the per-circuit data scaling ("weights") for a given type of robust-data-scaling.
    TODO: docstring - more details
    """

    fitQty = _get_fit_qty(model, ds, circuitList, parameters, evaltree_cache, comm, memLimit)
    #Note: fitQty[iCircuit] gives fit quantity for a single circuit, aggregated over outcomes.

    expected = (len(ds.get_outcome_labels()) - 1)  # == "k"
    dof_per_box = expected; nboxes = len(circuitList)
    pc = 0.05  # hardcoded (1 - confidence level) for now -- make into advanced option w/default

    circuitWeights = {}
    if scale_typ in ("robust", "Robust"):
        # Robust scaling V1: drastically scale down weights of especially bad sequences
        threshold = _np.ceil(_chi2.ppf(1 - pc / nboxes, dof_per_box))
        for i, opstr in enumerate(circuitList):
            if fitQty[i] > threshold:
                circuitWeights[opstr] = expected / fitQty[i]  # scaling factor

    elif scale_typ in ("robust+", "Robust+"):
        # Robust scaling V2: V1 + rescale to desired chi2 distribution without reordering
        threshold = _np.ceil(_chi2.ppf(1 - pc / nboxes, dof_per_box))
        scaled_fitQty = fitQty.copy()
        for i, opstr in enumerate(circuitList):
            if fitQty[i] > threshold:
                circuitWeights[opstr] = expected / fitQty[i]  # scaling factor
                scaled_fitQty[i] = expected  # (fitQty[i]*circuitWeights[opstr])

        N = len(fitQty)
        percentiles = [_chi2.ppf((i + 1) / (N + 1), dof_per_box) for i in range(N)]
        for iBin, i in enumerate(_np.argsort(scaled_fitQty)):
            opstr = circuitList[i]
            fit, expected = scaled_fitQty[i], percentiles[iBin]
            if fit > expected:
                if opstr in circuitWeights: circuitWeights[opstr] *= expected / fit
                else: circuitWeights[opstr] = expected / fit

    return circuitWeights


def get_wildcard_budget(model, ds, circuitsToUse, parameters, badfit_opts, evaltree_cache, comm, memLimit, verbosity):
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    fitQty = _get_fit_qty(model, ds, circuitsToUse, parameters, evaltree_cache, comm, memLimit)
    badfit_opts = badfit_opts or {}

    printer.log("******************* Adding Wildcard Budget **************************")

    # Approach: we create an objective function that, for a given Wvec, computes:
    # (amt_of_2DLogL over threshold) + (amt of "red-box": per-outcome 2DlogL over threshold) + eta*|Wvec|_1                                     # noqa
    # and minimize this for different eta (binary search) to find that largest eta for which the
    # first two terms is are zero.  This Wvec is our keeper.
    if evaltree_cache and 'evTree' in evaltree_cache:
        #use cache dictionary to speed multiple calls which use
        # the same model, operation sequences, comm, memlim, etc.
        evTree = evaltree_cache['evTree']
    else:
        # Note: simplify_circuits doesn't support aliased dataset (yet)
        dstree = ds if (parameters.get('opLabelAliases', None) is None) else None
        evTree, _, _, lookup, outcomes_lookup = \
            model.bulk_evaltree_from_resources(
                circuitsToUse, None, memLimit, "deriv", ['bulk_fill_probs'], dstree)

        #Fill cache dict if one was given
        if evaltree_cache is not None:
            evaltree_cache['evTree'] = evTree
            evaltree_cache['lookup'] = lookup
            evaltree_cache['outcomes_lookup'] = outcomes_lookup

    nDataParams = ds.get_degrees_of_freedom(circuitsToUse)  # number of independent parameters
    # in dataset (max. model # of params)
    nModelParams = model.num_params()  # just use total number of params
    percentile = 0.05; nBoxes = len(circuitsToUse)
    twoDeltaLogL_threshold = _chi2.ppf(1 - percentile, nDataParams - nModelParams)
    redbox_threshold = _chi2.ppf(1 - percentile / nBoxes, 1)
    eta = 10.0  # some default starting value - this *shouldn't* really matter
    #print("DB2: ",twoDeltaLogL_threshold,redbox_threshold)

    objective = parameters.get('objective', 'logl')
    assert(objective == "logl"), "Can only use wildcard scaling with 'logl' objective!"
    twoDeltaLogL_terms = fitQty
    twoDeltaLogL = sum(twoDeltaLogL_terms)

    budget = _wild.PrimitiveOpsWildcardBudget(model.get_primitive_op_labels() + model.get_primitive_instrument_labels(),
                                              add_spam=badfit_opts.get('wildcard_budget_includes_spam', True),
                                              start_budget=0.0)

    if twoDeltaLogL <= twoDeltaLogL_threshold \
       and sum(_np.clip(twoDeltaLogL_terms - redbox_threshold, 0, None)) < 1e-6:
        printer.log("No need to add budget!")
        Wvec = _np.zeros(len(budget.to_vector()), 'd')
    else:
        pci = parameters.get('probClipInterval', (-1e6, 1e6))
        min_p = parameters.get('minProbClip', 1e-4)
        a = parameters.get('radius', 1e-4)

        loglFn = _objfns.LogLFunction.simple_init(model, ds, circuitsToUse, min_p, pci, a,
                                                  poissonPicture=True, evaltree_cache=evaltree_cache,
                                                  comm=comm)
        sqrt_dlogl_elements = loglFn.fn(model.to_vector())  # must evaluate loglFn before using it to init loglWCFn
        loglWCFn = _objfns.LogLWildcardFunction(loglFn, model.to_vector(), budget)
        nCircuits = len(circuitsToUse)
        dlogl_terms = _np.empty(nCircuits, 'd')
        # b/c loglFn gives sqrt of terms (for use in leastsq optimizer)
        dlogl_elements = sqrt_dlogl_elements**2
        for i in range(nCircuits):
            dlogl_terms[i] = _np.sum(dlogl_elements[loglFn.lookup[i]], axis=0)
        #print("INITIAL 2DLogL (before any wildcard) = ", sum(2 * dlogl_terms), max(2 * dlogl_terms))
        #print("THRESHOLDS = ", twoDeltaLogL_threshold, redbox_threshold, nBoxes)

        def _wildcard_objective_firstTerms(Wv):
            dlogl_elements = loglWCFn.fn(Wv)**2  # b/c loglWCFn gives sqrt of terms (for use in leastsq optimizer)
            for i in range(nCircuits):
                dlogl_terms[i] = _np.sum(dlogl_elements[loglFn.lookup[i]], axis=0)

            twoDLogL_terms = 2 * dlogl_terms
            twoDLogL = sum(twoDLogL_terms)
            return max(0, twoDLogL - twoDeltaLogL_threshold) \
                + sum(_np.clip(twoDLogL_terms - redbox_threshold, 0, None))

        nIters = 0
        Wvec_init = budget.to_vector()

        # Optional: set initial wildcard budget by pushing on each Wvec component individually
        if badfit_opts.get('wildcard_smart_init', True):
            probe = Wvec_init.copy(); MULT = 2
            for i in range(len(Wvec_init)):
                #print("-------- Index ----------", i)
                Wv = Wvec_init.copy()
                #See how big Wv[i] needs to get before penalty stops decreasing
                last_penalty = 1e100; penalty = 0.9e100
                delta = 1e-6
                while penalty < last_penalty:
                    Wv[i] = delta
                    last_penalty = penalty
                    penalty = _wildcard_objective_firstTerms(Wv)
                    #print("  delta=%g  => penalty = %g" % (delta, penalty))
                    delta *= MULT
                probe[i] = delta / MULT**2
                #print(" ==> Probe[%d] = %g" % (i, probe[i]))

            probe /= len(Wvec_init)  # heuristic: set as new init point
            budget.from_vector(probe)
            Wvec_init = budget.to_vector()

        printer.log("INITIAL Wildcard budget = %s" % str(budget))

        # Find a value of eta that is small enough that the "first terms" are 0.
        while nIters < 10:
            printer.log("  Iter %d: trying eta = %g" % (nIters, eta))

            def _wildcard_objective(Wv):
                return _wildcard_objective_firstTerms(Wv) + eta * _np.linalg.norm(Wv, ord=1)

            #TODO REMOVE
            #import bpdb; bpdb.set_trace()
            #Wvec_init[:] = 0.0; print("TEST budget 0\n", _wildcard_objective(Wvec_init))
            #Wvec_init[:] = 1e-5; print("TEST budget 1e-5\n", _wildcard_objective(Wvec_init))
            #Wvec_init[:] = 0.1; print("TEST budget 0.1\n", _wildcard_objective(Wvec_init))
            #Wvec_init[:] = 1.0; print("TEST budget 1.0\n", _wildcard_objective(Wvec_init))

            if printer.verbosity > 1:
                printer.log(("NOTE: optimizing wildcard budget with verbose progress messages"
                             " - this *increases* the runtime significantly."), 2)

                def callbackF(Wv):
                    a, b = _wildcard_objective_firstTerms(Wv), eta * _np.linalg.norm(Wv, ord=1)
                    printer.log('wildcard: misfit + L1_reg = %.3g + %.3g = %.3g Wvec=%s' % (a, b, a + b, str(Wv)), 2)
            else:
                callbackF = None
            soln = _spo.minimize(_wildcard_objective, Wvec_init,
                                 method='Nelder-Mead', callback=callbackF, tol=1e-6)
            if not soln.success:
                _warnings.warn("Nelder-Mead optimization failed to converge!")
            Wvec = soln.x
            firstTerms = _wildcard_objective_firstTerms(Wvec)
            #printer.log("  Firstterms value = %g" % firstTerms)
            meets_conditions = bool(firstTerms < 1e-4)  # some zero-tolerance here
            if meets_conditions:  # try larger eta
                break
            else:  # nonzero objective => take Wvec as new starting point; try smaller eta
                Wvec_init = Wvec
                eta /= 10

            printer.log("  Trying eta = %g" % eta)
            nIters += 1
    #print("Wildcard budget found for Wvec = ",Wvec)
    #print("FINAL Wildcard budget = ", str(budget))
    budget.from_vector(Wvec)
    printer.log(str(budget))
    return budget


def reoptimize_with_weights(model, ds, circuit_list, circuit_weights, objfn_builder, optimizer,
                            resource_alloc, cache, verbosity):
    """
    TODO: docstring
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity)
    printer.log("--- Re-optimizing after robust data scaling ---")
    bulk_circuit_list = _objfns.BulkCircuitList(circuit_list, circuitWeights=circuit_weights)
    opt_result, mdl_reopt = _alg.do_gst_fit(ds, model, bulk_circuit_list, optimizer, objfn_builder,
                                            resource_alloc, cache, printer - 1)
    return mdl_reopt


class ModelEstimateResults(_proto.ProtocolResults):
    """
    A results object that holds model estimates.
    """
    #Note: adds functionality to bare ProtocolResults object but *doesn't*
    #add additional data storage - all is still within same members,
    #even if this is is exposed differently.

    @classmethod
    def from_dir(cls, dirname, name, preloaded_data=None):
        """
        Initialize a new ModelEstimateResults object from `dirname` / results / `name`.

        Parameters
        ----------
        dirname : str
            The *root* directory name (under which there is are 'edesign',
            'data', and 'results' subdirectories).

        name : str
            The sub-directory name of the particular results object to load
            (there can be multiple under a given root `dirname`).  This is the
            name of a subdirectory of `dirname` / results.

        preloaded_data : ProtocolData, optional
            In the case that the :class:`ProtocolData` object for `dirname`
            is already loaded, it can be passed in here.  Otherwise leave this
            as None and it will be loaded.

        Returns
        -------
        ModelEstimateResults
        """
        ret = super().from_dir(dirname, name, preloaded_data)  # loads members, but doesn't create parent "links"
        for est in ret.estimates.values():
            est.parent = ret  # link estimate to parent results object
        return ret

    def __init__(self, data, protocol_instance, init_circuits=True):
        """
        Initialize an empty Results object.
        TODO: docstring
        """
        super().__init__(data, protocol_instance)

        #Initialize some basic "results" by just exposing the circuit lists more directly
        circuit_lists = _collections.OrderedDict()

        if init_circuits:
            edesign = self.data.edesign
            if isinstance(edesign, _proto.CircuitStructuresDesign):
                circuit_lists['iteration'] = [_objfns.BulkCircuitList(cs) for cs in edesign.circuit_structs]

                #Set "Ls and germs" info: gives particular structure
                finalStruct = edesign.circuit_structs[-1]
                if isinstance(finalStruct, _LsGermsStructure):  # FUTURE: do something w/ a *LsGermsSerialStructure*
                    circuit_lists['prep fiducials'] = finalStruct.prepStrs
                    circuit_lists['meas fiducials'] = finalStruct.effectStrs
                    circuit_lists['germs'] = finalStruct.germs

            elif isinstance(edesign, _proto.CircuitListsDesign):
                circuit_lists['iteration'] = [_objfns.BulkCircuitList(cl) for cl in edesign.circuit_lists]

            else:
                #Single iteration
                circuit_lists['iteration'] = [_objfns.BulkCircuitList(edesign.all_circuits_needing_data)]

            circuit_lists['final'] = circuit_lists['iteration'][-1]

            #TODO REMOVE
            ##We currently expect to have these keys (in future have users check for them?)
            #if 'prep fiducials' not in circuit_lists: circuit_lists['prep fiducials'] = []
            #if 'meas fiducials' not in circuit_lists: circuit_lists['meas fiducials'] = []
            #if 'germs' not in circuit_lists: circuit_lists['germs'] = []

            #I think these are UNUSED - TODO REMOVE
            #circuit_lists['all'] = _tools.remove_duplicates(
            #    list(_itertools.chain(*circuit_lists['iteration']))) # USED?
            #running_set = set(); delta_lsts = []
            #for lst in circuit_lists['iteration']:
            #    delta_lst = [x for x in lst if (x not in running_set)]
            #    delta_lsts.append(delta_lst); running_set.update(delta_lst)
            #circuit_lists['iteration delta'] = delta_lsts  # *added* at each iteration

        self.circuit_lists = circuit_lists
        self.estimates = _collections.OrderedDict()

        #Punt on serialization of these qtys for now...
        self.auxfile_types['circuit_lists'] = 'pickle'
        self.auxfile_types['estimates'] = 'pickle'

    @property
    def dataset(self):
        return self.data.dataset

    def as_nameddict(self):
        #Just return estimates
        ret = _tools.NamedDict('Estimate', 'category')
        for k, v in self.estimates.items():
            ret[k] = v
        return ret

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

        if 'iteration' not in self.circuit_lists:
            raise ValueError(("Circuits must be initialized"
                              "*before* adding estimates"))

        assert(results.dataset is self.dataset), "DataSet inconsistency: cannot import estimates!"
        assert(len(self.circuit_lists['iteration']) == len(results.circuit_lists['iteration'])), \
            "Iteration count inconsistency: cannot import estimates!"

        for estimate_key in results.estimates:
            if estimatesToAdd is None or estimate_key in estimatesToAdd:
                if estimate_key in self.estimates:
                    _warnings.warn("Re-initializing the %s estimate" % estimate_key
                                   + " of this Results object!  Usually you don't"
                                   + " want to do this.")
                self.estimates[estimate_key] = results.estimates[estimate_key]

    def rename_estimate(self, old_name, new_name):
        """
        Rename an estimate in this Results object.  Ordering of estimates is
        not changed.

        Parameters
        ----------
        old_name : str
            The labels of the estimate to be renamed

        new_name : str
            The new name for the estimate.

        Returns
        -------
        None
        """
        if old_name not in self.estimates:
            raise KeyError("%s does not name an existing estimate" % old_name)

        ordered_keys = list(self.estimates.keys())
        self.estimates[new_name] = self.estimates[old_name]  # at end
        del self.estimates[old_name]
        keys_to_move = ordered_keys[ordered_keys.index(old_name) + 1:]  # everything after old_name
        for key in keys_to_move: self.estimates.move_to_end(key)

    def add_estimate(self, estimate, estimate_key='default'):
        """
        Add a set of `Model` estimates to this `Results` object.

        Parameters
        ----------
        estimate : Estimate
            The estimate to add.

        estimate_key : str, optional
            The key or label used to identify this estimate.

        Returns
        -------
        None
        """
        if self.dataset is None:
            raise ValueError(("The data set must be initialized"
                              "*before* adding estimates"))

        if 'iteration' not in self.circuit_lists:
            raise ValueError(("Circuits must be initialized"
                              "*before* adding estimates"))

        if 'iteration' in self.circuit_lists:
            modelsByIter = estimate.models.get('iteration estimates', [])
            la, lb = len(self.circuit_lists['iteration']), len(modelsByIter)
            assert(la == lb), "Number of iterations (%d) must equal %d!" % (lb, la)

        if estimate_key in self.estimates:
            _warnings.warn("Re-initializing the %s estimate" % estimate_key
                           + " of this Results object!  Usually you don't"
                           + " want to do this.")

        self.estimates[estimate_key] = estimate

        #TODO REMOVE
        ##Set gate sequence related parameters inherited from Results
        #self.estimates[estimate_key].parameters['max length list'] = \
        #    self.circuit_structs['final'].Ls

    def add_model_test(self, targetModel, themodel,
                       estimate_key='test', gauge_opt_keys="auto"):
        """
        Add a new model-test (i.e. non-optimized) estimate to this `Results` object.

        Parameters
        ----------
        targetModel : Model
            The target model used for comparison to the model.

        themodel : Model
            The "model" model whose fit to the data and distance from
            `targetModel` are assessed.

        estimate_key : str, optional
            The key or label used to identify this estimate.

        gauge_opt_keys : list, optional
            A list of gauge-optimization keys to add to the estimate.  All
            of these keys will correspond to trivial gauge optimizations,
            as the model model is assumed to be fixed and to have no
            gauge degrees of freedom.  The special value "auto" creates
            gauge-optimized estimates for all the gauge optimization labels
            currently in this `Results` object.

        Returns
        -------
        None
        """
        nIter = len(self.circuit_lists['iteration'])

        # base parameter values off of existing estimate parameters
        defaults = {'objective': 'logl', 'minProbClip': 1e-4, 'radius': 1e-4,
                    'minProbClipForWeighting': 1e-4, 'opLabelAliases': None,
                    'truncScheme': "whole germ powers"}
        for est in self.estimates.values():
            for ky in defaults:
                if ky in est.parameters: defaults[ky] = est.parameters[ky]

        #Construct a parameters dict, similar to do_model_test(...)
        parameters = _collections.OrderedDict()
        parameters['objective'] = defaults['objective']
        if parameters['objective'] == 'logl':
            parameters['minProbClip'] = defaults['minProbClip']
            parameters['radius'] = defaults['radius']
        elif parameters['objective'] == 'chi2':
            parameters['minProbClipForWeighting'] = defaults['minProbClipForWeighting']
        else:
            raise ValueError("Invalid objective: %s" % parameters['objective'])
        parameters['profiler'] = None
        parameters['opLabelAliases'] = defaults['opLabelAliases']
        parameters['weights'] = None  # Hardcoded

        #Set default gate group to trival group to mimic do_model_test (an to
        # be consistent with this function creating "gauge-optimized" models
        # by just copying the initial one).
        themodel = themodel.copy()
        themodel.default_gauge_group = _TrivialGaugeGroup(themodel.dim)

        self.add_estimate(targetModel, themodel, [themodel] * nIter,
                          parameters, estimate_key=estimate_key)

        #add gauge optimizations (always trivial)
        if gauge_opt_keys == "auto":
            gauge_opt_keys = []
            for est in self.estimates.values():
                for gokey in est.goparameters:
                    if gokey not in gauge_opt_keys:
                        gauge_opt_keys.append(gokey)

        est = self.estimates[estimate_key]
        for gokey in gauge_opt_keys:
            trivialEl = _TrivialGaugeGroupElement(themodel.dim)
            goparams = {'model': themodel,
                        'targetModel': targetModel,
                        '_gaugeGroupEl': trivialEl}
            est.add_gaugeoptimized(goparams, themodel, gokey)

    def view(self, estimate_keys, gaugeopt_keys=None):
        """
        Creates a shallow copy of this Results object containing only the
        given estimate and gauge-optimization keys.

        Parameters
        ----------
        estimate_keys : str or list
            Either a single string-value estimate key or a list of such keys.

        gaugeopt_keys : str or list, optional
            Either a single string-value gauge-optimization key or a list of
            such keys.  If `None`, then all gauge-optimization keys are
            retained.

        Returns
        -------
        Results
        """
        view = ModelEstimateResults(self.data, self.protocol, init_circuits=False)
        view.qtys['circuit_lists'] = self.circuit_lists

        if isinstance(estimate_keys, str):
            estimate_keys = [estimate_keys]
        for ky in estimate_keys:
            if ky in self.estimates:
                view.estimates[ky] = self.estimates[ky].view(gaugeopt_keys, view)

        return view

    def copy(self):
        """ Creates a copy of this Results object. """
        #TODO: check whether this deep copies (if we want it to...) - I expect it doesn't currently
        data = _proto.ProtocolData(self.data.edesign, self.data.dataset)
        cpy = ModelEstimateResults(data, self.protocol, init_circuits=False)
        cpy.circuit_lists = _copy.deepcopy(self.circuit_lists)
        for est_key, est in self.estimates.items():
            cpy.estimates[est_key] = est.copy()
        return cpy

    def __setstate__(self, stateDict):
        self.__dict__.update(stateDict)
        for est in self.estimates.values():
            est.set_parent(self)

    def __str__(self):
        s = "----------------------------------------------------------\n"
        s += "----------- pyGSTi ModelEstimateResults Object -----------\n"
        s += "----------------------------------------------------------\n"
        s += "\n"
        s += "How to access my contents:\n\n"
        s += " .dataset    -- the DataSet used to generate these results\n\n"
        s += " .circuit_lists   -- a dict of Circuit lists w/keys:\n"
        s += " ---------------------------------------------------------\n"
        s += "  " + "\n  ".join(list(self.circuit_lists.keys())) + "\n"
        s += "\n"
        s += " .estimates   -- a dictionary of Estimate objects:\n"
        s += " ---------------------------------------------------------\n"
        s += "  " + "\n  ".join(list(self.estimates.keys())) + "\n"
        s += "\n"
        return s


GSTDesign = GateSetTomographyDesign
GST = GateSetTomography
LGST = LinearGateSetTomography
