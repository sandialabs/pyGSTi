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
from . import support as _support
from .modeltest import ModelTest as _ModelTest
from .. import objects as _objs
from .. import algorithms as _alg
from .. import construction as _construction
from .. import io as _io
from .. import tools as _tools

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
    def __init__(self, targetModelFilenameOrObj):
        self.target_model = _load_model(targetModelFilenameOrObj)
        self.auxfile_types['target_model'] = 'pickle'


class GateSetTomographyInput(_proto.CircuitListsInput, HasTargetModel):
    """ Minimal Inputs needed for GST """
    def __init__(self, targetModelFilenameOrObj, circuit_lists, all_circuits_needing_data=None,
                 qubit_labels=None, nested=False):
        super().__init__(circuit_lists, all_circuits_needing_data, qubit_labels, nested)
        HasTargetModel.__init__(self, targetModelFilenameOrObj)


class StructuredGSTInput(_proto.CircuitStructuresInput, HasTargetModel):
    def __init__(self, targetModelFilenameOrObj, circuit_structs, qubit_labels=None,
                 nested=False):
        super().__init__(circuit_structs, qubit_labels, nested)
        HasTargetModel.__init__(self, targetModelFilenameOrObj)


class StandardGSTInput(StructuredGSTInput):
    def __init__(self, targetModelFilenameOrObj, prepStrsListOrFilename, effectStrsListOrFilename,
                 germsListOrFilename, maxLengths, germLengthLimits=None, fidPairs=None, keepFraction=1,
                 keepSeed=None, qubit_labels=None, verbosity=0, add_default_protocol=False):

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

        #Hardcoded for now... - include so gets written when serialized
        self.truncation_method = "whole germ powers"
        self.nested = True

        #FPR support
        self.fiducial_pairs = fidPairs
        self.fpr_keep_fraction = keepFraction
        self.fpr_keep_seed = keepSeed

        #TODO: add a line_labels arg to make_lsgst_structs and pass qubit_labels in?
        target_model = _load_model(targetModelFilenameOrObj)
        structs = _construction.make_lsgst_structs(
            target_model, self.prep_fiducials, self.meas_fiducials, self.germs,
            self.maxlengths, self.fiducial_pairs, self.truncation_method, self.nested,
            self.fpr_keep_fraction, self.fpr_keep_seed, germLengthLimits=self.germ_length_limits,
            verbosity=verbosity)

        super().__init__(target_model, structs, qubit_labels, self.nested)
        self.auxfile_types['prep_fiducials'] = 'text-circuit-list'
        self.auxfile_types['meas_fiducials'] = 'text-circuit-list'
        self.auxfile_types['germs'] = 'text-circuit-list'
        if add_default_protocol:
            self.add_default_protocol(StandardGST(name='StdGST'))


class GateSetTomography(_proto.Protocol):
    def __init__(self, initialModelFilenameOrObj=None, gaugeOptParams=None,
                 advancedOptions=None, comm=None, memLimit=None,
                 output_pkl=None, verbosity=2, name=None):

        #Note: *don't* specify default dictionary arguments, as this is dangerous
        # because they are mutable objects
        if advancedOptions is None: advancedOptions = {}
        if gaugeOptParams is None:
            gaugeOptParams = {'itemWeights': {'gates': 1.0, 'spam': 0.001}}

        super().__init__(name)
        self.initial_model = _load_model(initialModelFilenameOrObj) if initialModelFilenameOrObj else None
        self.gaugeOptParams = gaugeOptParams
        self.advancedOptions = advancedOptions
        self.comm = comm
        self.memLimit = memLimit
        self.output_pkl = output_pkl
        self.verbosity = verbosity

        self.auxfile_types['initial_model'] = 'pickle'
        self.auxfile_types['gaugeOptParams'] = 'pickle'  #TODO - better later? - json?
        self.auxfile_types['advancedOptions'] = 'pickle'  #TODO - better later? - json?
        self.auxfile_types['comm'] = 'reset'


    #TODO: Maybe make methods like this separate functions??
    def run_using_germs_and_fiducials(self, dataset, target_model, prep_fiducials, meas_fiducials, germs, maxLengths):
        inp = StandardGSTInput(target_model, prep_fiducials, meas_fiducials, germs, maxLengths)
        return self.run(_proto.ProtocolData(inp, dataset))

    def run_using_circuit_structures(self, target_model, circuit_structs, dataset):
        inp = StructuredGSTInput(target_model, circuit_structs)
        return self.run(_proto.ProtocolData(inp, dataset))

    def run_using_circuit_lists(self, target_model, circuit_lists, dataset):
        inp = GSTInput(target_model, circuit_lists)
        return self.run(_proto.ProtocolData(inp, dataset))

    def run(self, data):

        tRef = _time.time()
        comm = self.comm
        memLimit = self.memLimit
        advancedOptions = self.advancedOptions
        gaugeOptParams = self.gaugeOptParams

        profile = advancedOptions.get('profile', 1)

        if profile == 0: profiler = _DummyProfiler()
        elif profile == 1: profiler = _objs.Profiler(comm, False)
        elif profile == 2: profiler = _objs.Profiler(comm, True)
        else: raise ValueError("Invalid value for 'profile' argument (%s)" % profile)

        printer = _objs.VerbosityPrinter.build_printer(self.verbosity, comm)
        if advancedOptions.get('recordOutput', True) and not printer.is_recording():
            printer.start_recording()

        tNxt = _time.time()
        profiler.add_time('do_long_sequence_gst: loading', tRef); tRef = tNxt

        try:  # take structs if available
            lsgstLists = data.input.circuit_structs
        except:
            lsgstLists = data.input.circuit_lists
        ds = data.dataset

        #Get starting point (model), which is used to compute other quantities
        # Note: should compute on rank 0 and distribute?
        if self.initial_model is not None:  # then use user-supplied model
            mdl_start = self.initial_model
            startingPt = "User-supplied-Model"  # for profiler log below
        else:
            #Use the target model, with optional post-processing
            target_model = data.input.target_model
            if isinstance(target_model, _objs.ExplicitOpModel):
                LGSTcompatibleOps = all([(isinstance(g, _objs.FullDenseOp)
                                          or isinstance(g, _objs.TPDenseOp))
                                         for g in target_model.operations.values()])
            else:
                LGSTcompatibleOps = False

            #Get default target-starting-point postprocessing
            validStructTypes = (_objs.LsGermsStructure, _objs.LsGermsSerialStructure)
            if isinstance(lsgstLists[0], validStructTypes) and LGSTcompatibleOps:
                startingPt = advancedOptions.get('starting point', "LGST")
            else:
                startingPt = advancedOptions.get('starting point', "target")

            if startingPt == "LGST":
                assert(isinstance(lsgstLists[0], validStructTypes)), \
                    "Cannot run LGST: fiducials not specified!"
                opLabels = advancedOptions.get('opLabels',
                                               list(target_model.operations.keys())
                                               + list(target_model.instruments.keys()))
                mdl_start = _alg.do_lgst(ds, lsgstLists[0].prepStrs, lsgstLists[0].effectStrs, target_model,
                                         opLabels, svdTruncateTo=target_model.get_dimension(),
                                         opLabelAliases=lsgstLists[0].aliases,
                                         verbosity=printer)  # returns a model with the *same*
                # parameterizations as target_model

                #In LGST case, gauge optimimize starting point to the target
                # (historical; sometimes seems to help in practice, since it's gauge
                # optimizing to physical gates (e.g. something in CPTP)
                tol = gaugeOptParams.get('tol', 1e-8) if gaugeOptParams else 1e-8
                mdl_start = _alg.gaugeopt_to_target(mdl_start, target_model, tol=tol, comm=comm)
                #Note: use *default* gauge-opt params when optimizing

            elif startingPt == "target":
                mdl_start = target_model.copy()
            elif isinstance(startingPt, _objs.Model):  # TODO: REMOVE THIS OPTION since we have an initial-model arg now
                mdl_start = startingPt
                startingPt = "User-supplied-Model"  # for profiler log below
            else:
                raise ValueError("Invalid starting point: %s" % startingPt)

        tNxt = _time.time()
        profiler.add_time('do_long_sequence_gst: Starting Point (%s)' % startingPt, tRef); tRef = tNxt

        #Post-processing mdl_start : done only on root proc in case there is any nondeterminism.
        if comm is None or comm.Get_rank() == 0:
            #Advanced Options can specify further manipulation of starting model
            if advancedOptions.get('contractStartToCPTP', False):
                mdl_start = _alg.contract(mdl_start, "CPTP")
                raise ValueError("'contractStartToCPTP' has been removed b/c it can change the parameterization of a model")
            if advancedOptions.get('depolarizeStart', 0) > 0:
                mdl_start = mdl_start.depolarize(op_noise=advancedOptions.get('depolarizeStart', 0))
            if advancedOptions.get('randomizeStart', 0) > 0:
                v = mdl_start.to_vector()
                vrand = 2 * (_np.random.random(len(v)) - 0.5) * advancedOptions.get('randomizeStart', 0)
                mdl_start.from_vector(v + vrand)

            if comm is not None:  # broadcast starting model
                #OLD: comm.bcast(mdl_start, root=0)
                # just broadcast *vector* to avoid huge pickles (if cached calcs!)
                comm.bcast(mdl_start.to_vector(), root=0)
        else:
            #OLD: mdl_start = comm.bcast(None, root=0)
            v = comm.bcast(None, root=0)
            mdl_start.from_vector(v)

        tNxt = _time.time()
        profiler.add_time('do_long_sequence_gst: Prep Initial seed', tRef); tRef = tNxt

        # lsgstLists can hold either circuit lists or structures - get
        # just the lists for calling core gst routines (structure is used only
        # for LGST and post-analysis).
        validStructTypes = (_objs.LsGermsStructure, _objs.LsGermsSerialStructure)
        rawLists = [l.allstrs if isinstance(l, validStructTypes) else l
                    for l in lsgstLists]

        aliases = lsgstLists[-1].aliases if isinstance(
            lsgstLists[-1], validStructTypes) else None
        aliases = advancedOptions.get('opLabelAliases', aliases)

        #Run Long-sequence GST on data
        objective = advancedOptions.get('objective', 'logl')
        default_fditer = 0 if mdl_start.simtype in ("termorder", "termgap") else 1

        args = dict(
            dataset=ds,
            startModel=mdl_start,
            circuitSetsToUseInEstimation=rawLists,
            tol=advancedOptions.get('tolerance', 1e-6),
            extra_lm_opts=advancedOptions.get('extra_lm_opts', None),
            cptp_penalty_factor=advancedOptions.get('cptpPenaltyFactor', 0),
            spam_penalty_factor=advancedOptions.get('spamPenaltyFactor', 0),
            maxiter=advancedOptions.get('maxIterations', 100000),
            fditer=advancedOptions.get('fdIterations', default_fditer),
            probClipInterval=advancedOptions.get('probClipInterval', (-1e6, 1e6)),
            returnAll=True,
            circuitWeightsDict=advancedOptions.get('circuitWeights', None),
            opLabelAliases=aliases,
            verbosity=printer,
            memLimit=memLimit,
            profiler=profiler,
            comm=comm, distributeMethod=advancedOptions.get(
                'distributeMethod', "default"),
            check=advancedOptions.get('check', False),
            evaltree_cache={},
            time_dependent=advancedOptions.get('timeDependent', False))

        if objective == "chi2":
            args['useFreqWeightedChiSq'] = advancedOptions.get(
                'useFreqWeightedChiSq', False)
            args['minProbClipForWeighting'] = advancedOptions.get(
                'minProbClipForWeighting', 1e-4)
            args['check_jacobian'] = advancedOptions.get('check', False)
            mdl_lsgst_list = _alg.do_iterative_mc2gst(**args)

        elif objective == "logl":
            args['minProbClip'] = advancedOptions.get('minProbClip', 1e-4)
            args['radius'] = advancedOptions.get('radius', 1e-4)
            args['alwaysPerformMLE'] = advancedOptions.get('alwaysPerformMLE', False)
            args['onlyPerformMLE'] = advancedOptions.get('onlyPerformMLE', False)
            mdl_lsgst_list = _alg.do_iterative_mlgst(**args)

        elif objective == "lgst":
            assert(startingPt == "LGST"), "Can only set objective=\"lgst\" for parameterizations compatible with LGST"
            assert(len(lsgstLists) == 1), "Can only set objective=\"lgst\" with number if lists/max-lengths == 1"
            mdl_lsgst_list = [args['startModel']]
        else:
            raise ValueError("Invalid objective: %s" % objective)

        tNxt = _time.time()
        profiler.add_time('do_long_sequence_gst: total long-seq. opt.', tRef); tRef = tNxt

        #set parameters
        parameters = _collections.OrderedDict()
        parameters['objective'] = objective
        parameters['memLimit'] = memLimit
        parameters['starting point'] = startingPt
        parameters['profiler'] = profiler

        #from advanced options
        parameters['minProbClip'] = \
            advancedOptions.get('minProbClip', 1e-4)
        parameters['minProbClipForWeighting'] = \
            advancedOptions.get('minProbClipForWeighting', 1e-4)
        parameters['probClipInterval'] = \
            advancedOptions.get('probClipInterval', (-1e6, 1e6))
        parameters['radius'] = advancedOptions.get('radius', 1e-4)
        parameters['weights'] = advancedOptions.get('circuitWeights', None)
        parameters['cptpPenaltyFactor'] = advancedOptions.get('cptpPenaltyFactor', 0)
        parameters['spamPenaltyFactor'] = advancedOptions.get('spamPenaltyFactor', 0)
        parameters['distributeMethod'] = advancedOptions.get('distributeMethod', 'default')
        parameters['depolarizeStart'] = advancedOptions.get('depolarizeStart', 0)
        parameters['randomizeStart'] = advancedOptions.get('randomizeStart', 0)
        parameters['contractStartToCPTP'] = advancedOptions.get('contractStartToCPTP', False)
        parameters['tolerance'] = advancedOptions.get('tolerance', 1e-6)
        parameters['maxIterations'] = advancedOptions.get('maxIterations', 100000)
        parameters['useFreqWeightedChiSq'] = advancedOptions.get('useFreqWeightedChiSq', False)
        parameters['nestedCircuitLists'] = advancedOptions.get('nestedCircuitLists', True)
        parameters['profile'] = advancedOptions.get('profile', 1)
        parameters['check'] = advancedOptions.get('check', False)
        parameters['truncScheme'] = advancedOptions.get('truncScheme', "whole germ powers")
        parameters['opLabelAliases'] = advancedOptions.get('opLabelAliases', None)
        parameters['includeLGST'] = advancedOptions.get('includeLGST', True)

        return _package_into_results(self, data, data.input.target_model, mdl_start,
                                     lsgstLists, parameters, args, mdl_lsgst_list,
                                     gaugeOptParams, advancedOptions, comm, memLimit,
                                     self.output_pkl, printer, profiler, args['evaltree_cache'])


class StandardGST(_proto.Protocol):
    def __init__(self, modes="TP,CPTP,Target",
                 gaugeOptSuite=('single', 'unreliable2Q'),
                 gaugeOptTarget=None, modelsToTest=None, comm=None, memLimit=None,
                 advancedOptions=None, output_pkl=None, verbosity=2, name=None):

        #Note: *don't* specify default dictionary arguments, as this is dangerous
        # because they are mutable objects
        if advancedOptions is None: advancedOptions = {}

        super().__init__(name)
        self.modes = modes.split(',')
        self.models_to_test = modelsToTest
        self.gaugeOptSuite = gaugeOptSuite
        self.gaugeOptTarget = gaugeOptTarget
        self.advancedOptions = advancedOptions
        self.comm = comm
        self.memLimit = memLimit
        self.output_pkl = output_pkl
        self.verbosity = verbosity

        self.auxfile_types['models_to_test'] = 'pickle'
        self.auxfile_types['gaugeOptSuite'] = 'pickle'
        self.auxfile_types['gaugeOptTarget'] = 'pickle'
        self.auxfile_types['advancedOptions'] = 'pickle'
        self.auxfile_types['comm'] = 'reset'

    def run_using_germs_and_fiducials(self, dataset, target_model, prep_fiducials, meas_fiducials, germs, maxLengths):
        inp = StandardGSTInput(target_model, prep_fiducials, meas_fiducials, germs, maxLengths)
        data = _proto.ProtocolData(inp, dataset)
        return self.run(data)

    def run(self, data):
        printer = _objs.VerbosityPrinter.build_printer(self.verbosity, self.comm)

        modes = self.modes
        modelsToTest = self.models_to_test
        advancedOptions = self.advancedOptions.copy()  # ever None?? - if not, simplify below logic
        if modelsToTest is None: modelsToTest = {}

        #TODO REMOVE
        #inp = data.input
        #ds = data.dataset
        #target_model = inp.target_model
        #prepStrs = inp.prep_fiducials
        #effectStrs = inp.meas_fiducials
        #germs = inp.germs
        #maxLengths = inp.max_lengths

        ret = None
        with printer.progress_logging(1):
            for i, mode in enumerate(modes):
                printer.show_progress(i, len(modes), prefix='-- Std Practice: ', suffix=' (%s) --' % mode)

                #prepare advanced options dictionary
                advanced = advancedOptions.get('all', {})
                advanced.update(advancedOptions.get(mode, {}))

                if mode == "Target":
                    est_label = mode
                    model_to_test = data.input.target_model.copy()  # no parameterization change
                    advanced.update({'appendTo': ret, 'estimateLabel': est_label, 'onBadFit': []})
                    mdltest = _ModelTest(model_to_test, False, advanced, self.comm,
                                         self.memLimit, verbosity=printer - 1)
                    ret = mdltest.run(data)

                elif mode in ('full', 'TP', 'CPTP', 'H+S', 'S', 'static'):  # mode is a parameterization
                    est_label = parameterization = mode  # for now, 1-1 correspondence
                    initial_model = data.input.target_model.copy()
                    initial_model.set_all_parameterizations(parameterization)
                    advanced.update({'appendTo': ret, 'estimateLabel': est_label})

                    gst = GST(initial_model, False, advanced, self.comm, self.memLimit, verbosity=printer - 1)
                    ret = gst.run(data)

                elif mode in modelsToTest:
                    est_label = mode
                    advanced.update({'appendTo': ret, 'estimateLabel': est_label})
                    mdltest = _ModelTest(modelsToTest[mode], False, advanced,
                                         self.comm, self.memLimit, verbosity=printer - 1)
                    ret = mdltest.run(data)
                else:
                    raise ValueError("Invalid item in 'modes' argument: %s" % mode)

                #Get gauge optimization dictionary
                assert(not printer.is_recording()); printer.start_recording()
                gaugeOptSuite = self.gaugeOptSuite
                gaugeOptTarget = self.gaugeOptTarget
                target_with_dgg =  data.input.target_model.copy()  # with valid default gauge group, dictating gauge opt
                if mode in ('full', 'TP', 'CPTP', 'H+S', 'S', 'static'):  # mode = parameterization
                    target_with_dgg.set_all_parameterizations(mode)
                else:  # no gauge opt is done on everything else (model tests)
                    target_with_dgg.default_gauge_group = _objs.TrivialGaugeGroup(target_with_dgg.dim)
                gaugeOptSuite_dict = gaugeopt_suite_to_dictionary(gaugeOptSuite, target_with_dgg,
                                                                  advancedOptions, printer - 1)

                if gaugeOptTarget is not None:
                    assert(isinstance(gaugeOptTarget, _objs.Model)), "`gaugeOptTarget` must be None or a Model"
                    for goparams in gaugeOptSuite_dict.values():
                        goparams_list = [goparams] if hasattr(goparams, 'keys') else goparams
                        for goparams_dict in goparams_list:
                            if 'targetModel' in goparams_dict:
                                _warnings.warn(("`gaugeOptTarget` argument is overriding"
                                                "user-defined targetModel in gauge opt"
                                                "param dict(s)"))
                            goparams_dict.update({'targetModel': gaugeOptTarget})

                #Gauge optimize to list of gauge optimization parameters
                for goLabel, goparams in gaugeOptSuite_dict.items():

                    printer.log("-- Performing '%s' gauge optimization on %s estimate --" % (goLabel, est_label), 2)
                    gsStart = ret.estimates[est_label].get_start_model(goparams)
                    ret.estimates[est_label].add_gaugeoptimized(goparams, None, goLabel, self.comm, printer - 3)

                    #Gauge optimize data-scaled estimate also
                    for suffix in ROBUST_SUFFIX_LIST:
                        if est_label + suffix in ret.estimates:
                            gsStart_robust = ret.estimates[est_label + suffix].get_start_model(goparams)
                            if gsStart_robust.frobeniusdist(gsStart) < 1e-8:
                                printer.log("-- Conveying '%s' gauge optimization to %s estimate --" %
                                            (goLabel, est_label + suffix), 2)
                                params = ret.estimates[est_label].goparameters[goLabel]  # no need to copy here
                                gsopt = ret.estimates[est_label].models[goLabel].copy()
                                ret.estimates[est_label + suffix].add_gaugeoptimized(params, gsopt, goLabel, self.comm,
                                                                                     printer - 3)
                            else:
                                printer.log("-- Performing '%s' gauge optimization on %s estimate --" %
                                            (goLabel, est_label + suffix), 2)
                                ret.estimates[est_label + suffix].add_gaugeoptimized(goparams, None, goLabel, self.comm,
                                                                                     printer - 3)

                # Add gauge optimizations to end of any existing "stdout" meta info
                if 'stdout' in ret.estimates[est_label].meta:
                    ret.estimates[est_label].meta['stdout'].extend(printer.stop_recording())
                else:
                    ret.estimates[est_label].meta['stdout'] = printer.stop_recording()

        #Write results to a pickle file if desired
        if self.output_pkl and (self.comm is None or self.comm.Get_rank() == 0):
            if isinstance(self.output_pkl, str):
                with open(self.output_pkl, 'wb') as pklfile:
                    _pickle.dump(ret, pklfile)
            else:
                _pickle.dump(ret, self.output_pkl)

        ret.name = self.name
        return ret


# ------------------ HELPER FUNCTIONS -----------------------------------

def gaugeopt_suite_to_dictionary(gaugeOptSuite, target_model, advancedOptions=None, verbosity=0):
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

    target_model : Model
        A model which specifies the dimension (i.e. parameterization) of the
        gauge-optimization and the basis.  Usually this is set to the *ideal*
        `target model` for the model being gauge optimized.

    advancedOptions : dict, optional
        A dictionary of advanced options for internal use.

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

    #Build ordered dict of gauge optimization parameters
    if isinstance(gaugeOptSuite, dict):
        gaugeOptSuite_dict = _collections.OrderedDict()
        for lbl, goparams in gaugeOptSuite.items():
            if hasattr(goparams, 'keys'):
                gaugeOptSuite_dict[lbl] = goparams.copy()
                gaugeOptSuite_dict[lbl].update({'verbosity': printer})
            else:
                assert(isinstance(goparams, list)), "If not a dictionary, gauge opt params should be a list of dicts!"
                gaugeOptSuite_dict[lbl] = []
                for goparams_stage in goparams:
                    dct = goparams_stage.copy()
                    dct.update({'verbosity': printer})
                    gaugeOptSuite_dict[lbl].append(dct)

    else:
        gaugeOptSuite_dict = _collections.OrderedDict()
        if isinstance(gaugeOptSuite, str):
            gaugeOptSuites = [gaugeOptSuite]
        else:
            gaugeOptSuites = gaugeOptSuite[:]  # assumes gaugeOptSuite is a list/tuple of strs

        for suiteName in gaugeOptSuites:
            if suiteName == "single":

                stages = []  # multi-stage gauge opt
                gg = target_model.default_gauge_group
                if isinstance(gg, _objs.TrivialGaugeGroup):
                    #just do a single-stage "trivial" gauge opts using default group
                    gaugeOptSuite_dict['single'] = {'verbosity': printer}

                    if "unreliable2Q" in gaugeOptSuites and target_model.dim == 16:
                        if advancedOptions is not None:
                            # 'unreliableOps' can only be specified in 'all' options
                            advanced = advancedOptions.get('all', {})
                        else: advanced = {}
                        unreliableOps = advanced.get('unreliableOps', ['Gcnot', 'Gcphase', 'Gms', 'Gcn', 'Gcx', 'Gcz'])
                        if any([gl in target_model.operations.keys() for gl in unreliableOps]):
                            gaugeOptSuite_dict['single-2QUR'] = {'verbosity': printer}

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
                            'gauge_group': _objs.UnitaryGaugeGroup(target_model.dim, target_model.basis),
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
                            'gauge_group': s3gg(target_model.dim),
                            'verbosity': printer
                        })

                    gaugeOptSuite_dict['single'] = stages  # can be a list of stage dictionaries

                    if "unreliable2Q" in gaugeOptSuites and target_model.dim == 16:
                        if advancedOptions is not None:
                            # 'unreliableOps' can only be specified in 'all' options
                            advanced = advancedOptions.get('all', {})
                        else: advanced = {}
                        unreliableOps = advanced.get('unreliableOps', ['Gcnot', 'Gcphase', 'Gms', 'Gcn', 'Gcx', 'Gcz'])
                        if any([gl in target_model.operations.keys() for gl in unreliableOps]):
                            stage2_item_weights = {'gates': 1, 'spam': 0.0}
                            for gl in unreliableOps:
                                if gl in target_model.operations.keys(): stage2_item_weights[gl] = 0.01
                            stages_2QUR = [stage.copy() for stage in stages]  # ~deep copy of stages
                            iStage2 = 1 if gg.name in ("Full", "TP") else 0
                            stages_2QUR[iStage2]['itemWeights'] = stage2_item_weights
                            gaugeOptSuite_dict['single-2QUR'] = stages_2QUR  # add additional gauge opt
                        else:
                            _warnings.warn(("`unreliable2Q` was given as a gauge opt suite, but none of the"
                                            " gate names in advancedOptions['all']['unreliableOps'], i.e., %s,"
                                            " are present in the target model.  Omitting 'single-2QUR' gauge opt.")
                                           % (", ".join(unreliableOps)))

            elif suiteName in ("varySpam", "varySpamWt", "varyValidSpamWt", "toggleValidSpam"):

                itemWeights_bases = _collections.OrderedDict()
                itemWeights_bases[""] = {'gates': 1}

                if "unreliable2Q" in gaugeOptSuites and target_model.dim == 16:
                    if advancedOptions is not None:
                        # 'unreliableOps' can only be specified in 'all' options
                        advanced = advancedOptions.get('all', {})
                    else: advanced = {}
                    unreliableOps = advanced.get('unreliableOps', ['Gcnot', 'Gcphase', 'Gms', 'Gcn', 'Gcx', 'Gcz'])
                    if any([gl in target_model.operations.keys() for gl in unreliableOps]):
                        base = {'gates': 1}
                        for gl in unreliableOps:
                            if gl in target_model.operations.keys(): base[gl] = 0.01
                        itemWeights_bases["-2QUR"] = base

                if suiteName == "varySpam":
                    vSpam_range = [0, 1]; spamWt_range = [1e-4, 1e-1]
                elif suiteName == "varySpamWt":
                    vSpam_range = [0]; spamWt_range = [1e-4, 1e-1]
                elif suiteName == "varyValidSpamWt":
                    vSpam_range = [1]; spamWt_range = [1e-4, 1e-1]
                elif suiteName == "toggleValidSpam":
                    vSpam_range = [0, 1]; spamWt_range = [1e-3]

                for postfix, baseWts in itemWeights_bases.items():
                    for vSpam in vSpam_range:
                        for spamWt in spamWt_range:
                            lbl = "Spam %g%s%s" % (spamWt, "+v" if vSpam else "", postfix)
                            itemWeights = baseWts.copy()
                            itemWeights['spam'] = spamWt
                            gaugeOptSuite_dict[lbl] = {
                                'itemWeights': itemWeights,
                                'spam_penalty_factor': vSpam, 'verbosity': printer}

            elif suiteName == "unreliable2Q":
                assert(any([nm in ("single", "varySpam", "varySpamWt", "varyValidSpamWt", "toggleValidSpam")
                            for nm in gaugeOptSuite])), "'unreliable2Q' suite must be used with a spam or single suite."

            elif suiteName == "none":
                pass  # add nothing

            else:
                raise ValueError("Invalid `gaugeOptSuite` argument - unknown suite '%s'" % suiteName)

    return gaugeOptSuite_dict


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


def _package_into_results(callerProtocol, data, target_model, mdl_start, lsgstLists,
                          parameters, opt_args, mdl_lsgst_list, gaugeOptParams,
                          advancedOptions, comm, memLimit, output_pkl, verbosity,
                          profiler, evaltree_cache=None):
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
    if advancedOptions is None: advancedOptions = {}
    tRef = _time.time()
    callerName = callerProtocol.name

    ret = advancedOptions.get('appendTo', None)
    if ret is None:
        ret = ModelEstimateResults(data, callerProtocol)
    else:
        # a dummy object to check compatibility w/ret2
        dummy = ModelEstimateResults(data, callerProtocol)
        ret.add_estimates(dummy)  # does nothing, but will complain when appropriate

    #add estimate to Results
    estlbl = advancedOptions.get('estimateLabel', 'default')
    ret.add_estimate(target_model, mdl_start, mdl_lsgst_list, parameters, estlbl)
    profiler.add_time('%s: results initialization' % callerName, tRef); tRef = _time.time()

    #Do final gauge optimization to *final* iteration result only
    if gaugeOptParams:
        add_gauge_opt(ret.estimates[estlbl], gaugeOptParams, target_model,
                      mdl_lsgst_list[-1], comm, printer - 1)
        profiler.add_time('%s: gauge optimization' % callerName, tRef)

    #Perform extra analysis if a bad fit was obtained
    badFitThreshold = advancedOptions.get('badFitThreshold', DEFAULT_BAD_FIT_THRESHOLD)
    onBadFit = advancedOptions.get('onBadFit', [])  # ["wildcard"]) #["Robust+"]) # empty list => 'do nothing'
    add_badfit_estimates(ret, estlbl, onBadFit, badFitThreshold, opt_args, evaltree_cache, comm, memLimit, printer - 1)
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


def add_gauge_opt(estimate, gaugeOptParams, target_model, starting_model,
                  comm=None, verbosity=0):
    """
    Add a gauge optimization to an estimate.
    TODO: docstring - more details
    """
    gaugeOptParams = gaugeOptParams.copy()  # so we don't modify the caller's dict
    if '_gaugeGroupEl' in gaugeOptParams: del gaugeOptParams['_gaugeGroupEl']

    if "targetModel" not in gaugeOptParams:
        assert(target_model is not None), "No target model!  Cannot gauge optimize."
        gaugeOptParams["targetModel"] = target_model

    # somewhat redundant given add_gaugeoptimized behavior - but
    #  if not here won't do parallel gaugeopt_to_target below
    if "comm" not in gaugeOptParams:
        gaugeOptParams["comm"] = comm

    gaugeOptParams['returnAll'] = True  # so we get gaugeEl to save
    gaugeOptParams['model'] = starting_model

    if isinstance(gaugeOptParams['model'], _objs.ExplicitOpModel):
        #only explicit models can be gauge optimized
        _, gaugeEl, go_gs_final = _alg.gaugeopt_to_target(**gaugeOptParams)
    else:
        #but still fill in results for other models (?)
        gaugeEl, go_gs_final = None, gaugeOptParams['model'].copy()

    gaugeOptParams['_gaugeGroupEl'] = gaugeEl  # store gaugeopt el
    estimate.add_gaugeoptimized(gaugeOptParams, go_gs_final,
                                None, comm, verbosity)


def add_badfit_estimates(results, base_estimate_label="default", estimate_types=('wildcard',),
                         badFitThreshold=None, opt_args=None, evaltree_cache=None,
                         comm=None, memLimit=None, verbosity=0):
    """
    Add any and all "bad fit" estimates to `results`.
    TODO: docstring
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    base_estimate = results.estimates[base_estimate_label]
    lsgstLists = results.circuit_structs['iteration']
    mdl_lsgst_list = base_estimate.models['iteration estimates']
    mdl_start = base_estimate.models['seed']
    target_model = base_estimate.models.get('target', None)
    ds = results.dataset
    parameters = base_estimate.parameters
    if evaltree_cache is None: evaltree_cache = {}  # so tree gets cached

    if badFitThreshold is not None and \
       base_estimate.misfit_sigma(evaltree_cache=evaltree_cache, comm=comm) <= badFitThreshold:
        return  # fit is good enough - no need to add any estimates

    objective = parameters.get('objective', 'logl')
    validStructTypes = (_objs.LsGermsStructure, _objs.LsGermsSerialStructure)
    rawLists = [l.allstrs if isinstance(l, validStructTypes) else l
                for l in lsgstLists]
    circuitList = rawLists[-1]  # use final circuit list
    mdl = mdl_lsgst_list[-1]    # and model

    assert(parameters.get('weights', None) is None), \
        "Cannot perform bad-fit scaling when weights are already given!"

    for badfit_typ in estimate_types:
        new_params = parameters.copy()
        new_final_model = None

        if badfit_typ in ("robust", "Robust", "robust+", "Robust+"):
            new_params['weights'] = get_robust_scaling(badfit_typ, mdl, ds, circuitList,
                                                       parameters, evaltree_cache, comm, memLimit)
            if badfit_typ in ("Robust", "Robust+") and (opt_args is not None):
                mdl_reopt = reoptimize_with_weights(mdl, ds, circuitList, new_params['weights'],
                                                    objective, opt_args, printer - 1)
                new_final_model = mdl_reopt

        elif badfit_typ == "wildcard":
            new_params['unmodeled_error'] = get_wildcard_budget(mdl, ds, circuitList, parameters,
                                                                evaltree_cache, comm, memLimit, printer - 1)

        elif badfit_typ == "do nothing":
            continue  # go to next on-bad-fit directive

        else:
            raise ValueError("Invalid on-bad-fit directive: %s" % badfit_typ)

        # In case we've computed an updated final model, Just keep (?) old estimates of all
        # prior iterations (or use "blank" sentinel once this is supported).
        models_by_iter = mdl_lsgst_list[:] if (new_final_model is None) \
            else mdl_lsgst_list[0:-1] + [new_final_model]

        results.add_estimate(target_model, mdl_start, models_by_iter,
                             new_params, base_estimate_label + "." + badfit_typ)

        #Add gauge optimizations to the new estimate
        for gokey, gaugeOptParams in base_estimate.goparameters.items():
            if new_final_model is not None:
                add_gauge_opt(results.estimates[base_estimate_label + '.' + badfit_typ], gaugeOptParams,
                              target_model, new_final_model, comm, printer - 1)
            else:
                # add same gauge-optimized result as above
                go_gs_final = base_estimate.models[gokey]
                results.estimates[base_estimate_label + '.' + badfit_typ].add_gaugeoptimized(
                    gaugeOptParams.copy(), go_gs_final, None, comm, printer - 1)


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


def get_wildcard_budget(model, ds, circuitsToUse, parameters, evaltree_cache, comm, memLimit, verbosity):
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    fitQty = _get_fit_qty(model, ds, circuitsToUse, parameters, evaltree_cache, comm, memLimit)

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

    budget = _wild.PrimitiveOpsWildcardBudget(model.get_primitive_op_labels(), add_spam=True,
                                              start_budget=0.0)

    if twoDeltaLogL <= twoDeltaLogL_threshold \
       and sum(_np.clip(twoDeltaLogL_terms - redbox_threshold, 0, None)) < 1e-6:
        printer.log("No need to add budget!")
        Wvec = _np.zeros(len(model.get_primitive_op_labels()), 'd')
    else:
        pci = parameters.get('probClipInterval', (-1e6, 1e6))
        min_p = parameters.get('minProbClip', 1e-4)
        a = parameters.get('radius', 1e-4)

        loglFn = _objfns.LogLFunction.simple_init(model, ds, circuitsToUse, min_p, pci, a,
                                                  poissonPicture=True, evaltree_cache=evaltree_cache,
                                                  comm=comm)
        loglWCFn = _objfns.LogLWildcardFunction(loglFn, model.to_vector(), budget)
        nCircuits = len(circuitsToUse)
        dlogl_terms = _np.empty(nCircuits, 'd')
        # b/c loglFn gives sqrt of terms (for use in leastsq optimizer)
        dlogl_elements = loglFn.fn(model.to_vector())**2
        for i in range(nCircuits):
            dlogl_terms[i] = _np.sum(dlogl_elements[loglFn.lookup[i]], axis=0)
        print("INITIAL 2DLogL (before any wildcard) = ", sum(2 * dlogl_terms), max(2 * dlogl_terms))
        print("THRESHOLDS = ", twoDeltaLogL_threshold, redbox_threshold, nBoxes)

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
        print("INITIAL Wildcard budget = ", str(budget))

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
    budget.from_vector(Wvec)
    printer.log(str(budget))
    return budget


def reoptimize_with_weights(model, ds, circuitList, circuitWeights, objective, opt_args, verbosity):
    """
    TODO: docstring
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    #convert weights dict to an array for do_XXX methods below
    circuitWeightsArray = _np.ones(len(circuitList), 'd')
    gsindx = {opstr: i for i, opstr in enumerate(circuitList)}
    for opstr, weight in circuitWeights.items():
        circuitWeightsArray[gsindx[opstr]] = weight

    reopt_args = dict(dataset=ds,
                      startModel=model,
                      circuitsToUse=circuitList,
                      circuitWeights=circuitWeightsArray,
                      verbosity=printer - 1)
    for x in ('maxiter', 'tol', 'cptp_penalty_factor', 'spam_penalty_factor',
              'probClipInterval', 'check', 'opLabelAliases',
              'memLimit', 'comm', 'evaltree_cache', 'distributeMethod', 'profiler'):
        reopt_args[x] = opt_args[x]

    printer.log("--- Re-optimizing %s after robust data scaling ---" % objective)
    if objective == "chi2":
        reopt_args['useFreqWeightedChiSq'] = opt_args['useFreqWeightedChiSq']
        reopt_args['minProbClipForWeighting'] = opt_args['minProbClipForWeighting']
        reopt_args['check_jacobian'] = opt_args['check_jacobian']
        _, mdl_reopt = _alg.do_mc2gst(**reopt_args)

    elif objective == "logl":
        reopt_args['minProbClip'] = opt_args['minProbClip']
        reopt_args['radius'] = opt_args['radius']
        _, mdl_reopt = _alg.do_mlgst(**reopt_args)

    else: raise ValueError("Invalid objective '%s' for robust data scaling reopt" % objective)

    return mdl_reopt


class ModelEstimateResults(_proto.ProtocolResults):
    """
    TODO: docstring (better)
    Adds functionality to bare ProtocolResults object but *doesn't*
    add additional data storage - all is still within same members,
    even if this is is exposed differently.
    """

    def __init__(self, data, protocol_instance, init_circuits=True):
        """
        Initialize an empty Results object.
        TODO: docstring
        """
        super().__init__(data, protocol_instance)

        #Initialize some basic "results" by just exposing the circuit lists more directly
        circuit_lists = _collections.OrderedDict()
        circuit_structs = _collections.OrderedDict()

        if init_circuits:
            inp = self.data.input
            if isinstance(inp, _proto.CircuitStructuresInput):
                circuit_structs['iteration'] = inp.circuit_structs[:]
    
                #Set "Ls and germs" info: gives particular structure
                finalStruct = circuit_structs['final']
                if isinstance(finalStruct, _LsGermsStructure):  # FUTURE: do something sensible w/ LsGermsSerialStructure?
                    circuit_lists['prep fiducials'] = finalStruct.prepStrs
                    circuit_lists['effect fiducials'] = finalStruct.effectStrs
                    circuit_lists['germs'] = finalStruct.germs
    
            elif isinstance(inp, _proto.CircuitListsInput):
                circuit_structs['iteration'] = []
                for lst in inp.circuit_lists:
                    unindexed_gss = _LsGermsStructure([], [], [], [], None)
                    unindexed_gss.add_unindexed(lst)
                    circuit_structs['iteration'].append(unindexed_gss)
    
                #Needed?
                circuit_lists['prep fiducials'] = []
                circuit_lists['effect fiducials'] = []
                circuit_lists['germs'] = []
            else:
                #Single iteration
                lst = inp.all_circuits_needing_data
                unindexed_gss = _LsGermsStructure([], [], [], [], None)
                unindexed_gss.add_unindexed(lst)
                circuit_structs['iteration'] = [unindexed_gss]
    
                #Needed?
                circuit_lists['prep fiducials'] = []
                circuit_lists['effect fiducials'] = []
                circuit_lists['germs'] = []
                
    
            # Extract raw circuit lists from structs
            circuit_lists['iteration'] = \
                [gss.allstrs for gss in circuit_structs['iteration']]
            circuit_lists['final'] = circuit_lists['iteration'][-1]
            circuit_lists['all'] = _tools.remove_duplicates(
                list(_itertools.chain(*circuit_lists['iteration'])))

            if len(circuit_structs['iteration']) > 0:
                circuit_structs['final'] = circuit_structs['iteration'][-1]
    
            running_set = set(); delta_lsts = []
            for lst in circuit_lists['iteration']:
                delta_lst = [x for x in lst if (x not in running_set)]
                delta_lsts.append(delta_lst); running_set.update(delta_lst)
            circuit_lists['iteration delta'] = delta_lsts  # *added* at each iteration

        self.circuit_lists = circuit_lists
        self.circuit_structs = circuit_structs
        self.estimates = _collections.OrderedDict()

        #Punt on serialization of these qtys for now...
        self.auxfile_types['circuit_lists'] = 'pickle'
        self.auxfile_types['circuit_structs'] = 'pickle'
        self.auxfile_types['estimates'] = 'pickle'

    @property
    def dataset(self):
        return self.data.dataset

    def as_nameddict(self):
        #Just return estimates
        ret = _support.NamedDict('Estimate', 'category')
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

        if 'iteration' not in self.circuit_structs:
            raise ValueError(("Circuits must be initialized"
                              "*before* adding estimates"))

        assert(results.dataset is self.dataset), "DataSet inconsistency: cannot import estimates!"
        assert(len(self.circuit_structs['iteration']) == len(results.circuit_structs['iteration'])), \
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

    def add_estimate(self, targetModel, seedModel, modelsByIter,
                     parameters, estimate_key='default'):
        """
        Add a set of `Model` estimates to this `Results` object.

        Parameters
        ----------
        targetModel : Model
            The target model used when optimizing the objective.

        seedModel : Model
            The initial model used to seed the iterative part
            of the objective optimization.  Typically this is
            obtained via LGST.

        modelsByIter : list of Models
            The estimated model at each GST iteration. Typically these are the
            estimated models *before* any gauge optimization is performed.

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

        if 'iteration' not in self.circuit_structs:
            raise ValueError(("Circuits must be initialized"
                              "*before* adding estimates"))

        la, lb = len(self.circuit_structs['iteration']), len(modelsByIter)
        assert(la == lb), "Number of iterations (%d) must equal %d!" % (lb, la)

        if estimate_key in self.estimates:
            _warnings.warn("Re-initializing the %s estimate" % estimate_key
                           + " of this Results object!  Usually you don't"
                           + " want to do this.")

        self.estimates[estimate_key] = _Estimate(self, targetModel, seedModel,
                                                 modelsByIter, parameters)

        #Set gate sequence related parameters inherited from Results
        self.estimates[estimate_key].parameters['max length list'] = \
            self.circuit_structs['final'].Ls

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
        nIter = len(self.circuit_structs['iteration'])

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
        view.qyts['circuit_structs'] = self.circuit_structs

        if isinstance(estimate_keys, str):
            estimate_keys = [estimate_keys]
        for ky in estimate_keys:
            if ky in self.estimates:
                view.estimates[ky] = self.estimates[ky].view(gaugeopt_keys, view)

        return view

    def copy(self):
        """ Creates a copy of this Results object. """
        #TODO: check whether this deep copies (if we want it to...) - I expect it doesn't currently
        data = _proto.ProtocolData(self.input, self.dataset)
        cpy = ModelEstimateResults(data, self.protocol, init_circuits=False)
        cpy.dataset = self.dataset.copy()
        cpy.circuit_lists = _copy.deepcopy(self.circuit_lists)
        cpy.circuit_structs = _copy.deepcopy(self.circuit_structs)
        for est_key, est in self.estimates.items():
            cpy.estimates[est_key] = est.copy()
        return cpy

    def __setstate__(self, stateDict):
        self.__dict__.update(stateDict)
        for est in self.estimates.values():
            est.set_parent(self)

    def __str__(self):
        s = "----------------------------------------------------------\n"
        s += "---------------- pyGSTi Results Object -------------------\n"
        s += "----------------------------------------------------------\n"
        s += "\n"
        s += "How to access my contents:\n\n"
        s += " .dataset    -- the DataSet used to generate these results\n\n"
        s += " .circuit_lists   -- a dict of Circuit lists w/keys:\n"
        s += " ---------------------------------------------------------\n"
        s += "  " + "\n  ".join(list(self.circuit_lists.keys())) + "\n"
        s += "\n"
        s += " .circuit_structs   -- a dict of CircuitStructures w/keys:\n"
        s += " ---------------------------------------------------------\n"
        s += "  " + "\n  ".join(list(self.circuit_structs.keys())) + "\n"
        s += "\n"
        s += " .estimates   -- a dictionary of Estimate objects:\n"
        s += " ---------------------------------------------------------\n"
        s += "  " + "\n  ".join(list(self.estimates.keys())) + "\n"
        s += "\n"
        return s


GSTInput = GateSetTomographyInput
GST = GateSetTomography
